# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Competitive multi-agent interception task.

Two agents (pursuer & evader), each trained with its own actor + critic.
No shared value network — competitive objectives require independent critics.

Observation per agent: [B, 1, obs_dim]
  - Agent 0 ("pursuer"): evader rel heading (3) + pursuer lin vel (3)
    + pursuer rotation matrix flat (9) = 15
  - Agent 1 ("evader"): pursuer rel heading from evader view (3) + evader lin vel (3)
    + evader rotation matrix flat (9) = 15

Action per agent: [B, 1, 4] CTBR (body rates x3 + thrust x1), internally
converted to per-rotor motor commands via PIDRateController.

Reward per step (composed):
  - Pursuer: delta_distance + heading_alignment + terminal (+/- capture).
  - Evader: -(delta_distance) [negative of pursuer's] + misbehavior penalty
    on its own low-alt / NaN crashes. Terminal reward is negation of the
    pursuer's terminal event (capture → evader gets -R; evade_success → +R).

Curriculum learning for success_radius follows the same pattern as the
original Intercept task: slow exponential decay from init → max over a
global cross-episode step counter.
"""

import torch
import torch.distributions as D

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Composite, UnboundedContinuous

import omni_drones.utils.kit as kit_utils

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import (
    euler_to_quaternion,
    normalize,
    quaternion_to_rotation_matrix,
)


class InterceptCompetitiveMA(IsaacEnv):

    def __init__(self, cfg, headless):
        self.cfg = cfg
        task_cfg = cfg.task

        # ---- Curriculum knobs ------------------------------------------------
        self.success_radius_init = float(task_cfg.get("success_radius_init", 0.3))
        self.success_radius_max = float(task_cfg.get("success_radius_max", 0.1))
        self.success_radius_lr = float(task_cfg.get("success_radius_lr", 2e-6))
        self.success_radius_eval = float(task_cfg.get("success_radius_eval", 0.15))
        self.success_radius_global_step = 0
        self.success_radius = self.success_radius_init

        # ---- Reward knobs ----------------------------------------------------
        self.reset_thres = float(task_cfg.get("reset_thres", 15.0))
        self.reward_heading_alignment_weight = float(
            task_cfg.get("reward_heading_alignment_weight", 1.0)
        )
        self.reward_delta_distance_weight = float(
            task_cfg.get("reward_delta_distance_weight", 10.0)
        )
        self.reward_terminal_weight = float(
            task_cfg.get("reward_terminal_weight", 100.0)
        )

        # ---- Drone configs ---------------------------------------------------
        pursuer_cfg = task_cfg.pursuer
        evader_cfg = task_cfg.evader

        self.pursuer_model_name = pursuer_cfg.get("model", "Crazyflie")
        self.pursuer_controller_name = pursuer_cfg.get(
            "controller", "PIDRateController"
        )
        self.pursuer_target_speed = float(pursuer_cfg.get("target_speed", 10.0))
        self._pursuer_spawn_pos_range = pursuer_cfg.spawn_pos_range
        self._pursuer_spawn_rpy_range = pursuer_cfg.spawn_rpy_range

        self.evader_model_name = evader_cfg.get("model", "Crazyflie")
        # Evader is RL-controlled here; always use PIDRateController.
        self.evader_controller_name = "PIDRateController"

        self.evader_spawn_distance_range = list(
            evader_cfg.get("spawn_distance_range", [1.0, 7.0])
        )

        # ---- EMA factor for running stats ------------------------------------
        self.alpha = 0.8

        super().__init__(cfg, headless)

        self.pursuer.initialize()
        self.evader.initialize()

        # ---- Spawn distributions ---------------------------------------------
        self.pursuer_init_pos_dist = D.Uniform(
            torch.tensor(self._pursuer_spawn_pos_range.min, device=self.device),
            torch.tensor(self._pursuer_spawn_pos_range.max, device=self.device),
        )
        self.pursuer_init_rpy_dist = D.Uniform(
            torch.tensor(self._pursuer_spawn_rpy_range.min, device=self.device) * torch.pi,
            torch.tensor(self._pursuer_spawn_rpy_range.max, device=self.device) * torch.pi,
        )
        self.evader_spawn_distance_dist = D.Uniform(
            torch.tensor(self.evader_spawn_distance_range[0], device=self.device),
            torch.tensor(self.evader_spawn_distance_range[1], device=self.device),
        )

        # ---- Local-frame state buffers ---------------------------------------
        self.pursuer_local_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.pursuer_local_rot = torch.zeros(self.num_envs, 1, 4, device=self.device)
        self.evader_local_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.evader_local_rot = torch.zeros(self.num_envs, 1, 4, device=self.device)

        # Previous-step distance for delta-distance reward.
        self.prev_distance = torch.zeros(self.num_envs, 1, device=self.device)

        # Buffers for action smoothness tracking (both drones).
        self.pursuer_current_action = torch.zeros(
            self.num_envs, 1, 4, device=self.device
        )
        self.pursuer_prev_action = torch.zeros(
            self.num_envs, 1, 4, device=self.device
        )
        self.evader_current_action = torch.zeros(
            self.num_envs, 1, 4, device=self.device
        )
        self.evader_prev_action = torch.zeros(
            self.num_envs, 1, 4, device=self.device
        )

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _design_scene(self):
        self.pursuer, self.pursuer_controller = MultirotorBase.make(
            self.pursuer_model_name,
            self.pursuer_controller_name,
            device=str(self.device),
            name="pursuer",
        )
        self.evader, self.evader_controller = MultirotorBase.make(
            self.evader_model_name,
            self.evader_controller_name,
            device=str(self.device),
            name="evader",
        )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        )

        self.pursuer.spawn(translations=[(0.0, 0.0, 1.6)])
        self.evader.spawn(translations=[(5.0, 0.0, 2.0)])
        return ["/World/defaultGroundPlane"]

    # ------------------------------------------------------------------
    # Specs — two independent AgentSpecs (n=1 each)
    # ------------------------------------------------------------------
    def _set_specs(self):
        obs_dim = 15   # rel heading(3) + lin_vel(3) + rot_matrix_flat(9)
        act_dim = 4     # CTBR: 3 body rates + 1 thrust

        self.observation_spec = Composite({
            "pursuer": {
                "observation": UnboundedContinuous(
                    torch.Size([1, obs_dim]), device=self.device,
                ),
                "state": UnboundedContinuous(
                    torch.Size([1, obs_dim]), device=self.device,
                ),
            },
            "evader": {
                "observation": UnboundedContinuous(
                    torch.Size([1, obs_dim]), device=self.device,
                ),
                "state": UnboundedContinuous(
                    torch.Size([1, obs_dim]), device=self.device,
                ),
            },
        }).expand(self.num_envs).to(self.device)

        self.action_spec = Composite({
            "pursuer": {
                "action": UnboundedContinuous(
                    torch.Size([1, act_dim]), device=self.device,
                ),
            },
            "evader": {
                "action": UnboundedContinuous(
                    torch.Size([1, act_dim]), device=self.device,
                ),
            },
        }).expand(self.num_envs).to(self.device)

        self.reward_spec = Composite({
            "pursuer": {
                "reward": UnboundedContinuous(
                    torch.Size([1, 1]), device=self.device,
                ),
            },
            "evader": {
                "reward": UnboundedContinuous(
                    torch.Size([1, 1]), device=self.device,
                ),
            },
        }).expand(self.num_envs).to(self.device)

        # Two AgentSpecs — independent agent handling.
        self.agent_spec["pursuer"] = AgentSpec(
            "pursuer", n=1,
            observation_key=("pursuer", "observation"),
            action_key=("pursuer", "action"),
            reward_key=("pursuer", "reward"),
        )
        self.agent_spec["evader"] = AgentSpec(
            "evader", n=1,
            observation_key=("evader", "observation"),
            action_key=("evader", "action"),
            reward_key=("evader", "reward"),
        )

        def scalar():
            return UnboundedContinuous(torch.Size([1]), device=self.device)

        self.stats_spec = Composite({
            "pursuer_return": scalar(),
            "evader_return": scalar(),
            "episode_len": scalar(),
            "distance": scalar(),
            "capture_rate": scalar(),
            "evade_rate": scalar(),
            "pursuer_crash_rate": scalar(),
            "evader_crash_rate": scalar(),
            "reward_delta_distance_pursuer": scalar(),
            "reward_delta_distance_evader": scalar(),
            "reward_heading_alignment": scalar(),
            "reward_terminal_pursuer": scalar(),
            "reward_terminal_evader": scalar(),
            "success_radius": scalar(),
        }).expand(self.num_envs).to(self.device)

        self.info_spec = Composite({
            "pursuer_drone_state": UnboundedContinuous(
                torch.Size([1, 13]), device=self.device),
            "evader_drone_state": UnboundedContinuous(
                torch.Size([1, 13]), device=self.device),
            "pursuer_prev_action": UnboundedContinuous(
                torch.Size([1, act_dim]), device=self.device),
            "evader_prev_action": UnboundedContinuous(
                torch.Size([1, act_dim]), device=self.device),
        }).expand(self.num_envs).to(self.device)

        self.observation_spec["stats"] = self.stats_spec
        self.observation_spec["info"] = self.info_spec

        self.stats = self.stats_spec.zero()
        self.info = self.info_spec.zero()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor):
        n = len(env_ids)
        self.stats[env_ids] = 0.0

        self.pursuer._reset_idx(env_ids, self.training)
        self.evader._reset_idx(env_ids, self.training)

        # ---- Pursuer pose ----------------------------------------------------
        pursuer_pos = self.pursuer_init_pos_dist.sample(torch.Size([n, 1]))
        pursuer_rpy = self.pursuer_init_rpy_dist.sample(torch.Size([n, 1]))
        pursuer_rot = euler_to_quaternion(pursuer_rpy)

        # ---- Evader pose: random direction & distance from pursuer -----------
        pursuer_yaw = pursuer_rpy[..., 2]
        yaw_noise = torch.randn_like(pursuer_yaw) * 0.5
        spawn_yaw = pursuer_yaw + yaw_noise
        spawn_dir = normalize(torch.stack([
            torch.cos(spawn_yaw), torch.sin(spawn_yaw),
            torch.rand((n, 1), device=self.device),
        ], dim=-1))
        spawn_dir[..., 2] = spawn_dir[..., 2].abs()
        spawn_dir = normalize(spawn_dir)
        spawn_dist = self.evader_spawn_distance_dist.sample(torch.Size([n, 1, 1]))
        evader_pos = pursuer_pos + spawn_dir * spawn_dist

        evader_heading = normalize(torch.randn(n, 1, 3, device=self.device))
        evader_heading[..., 2] = 0.0
        evader_heading = normalize(evader_heading)
        evader_yaw = torch.atan2(evader_heading[..., 1], evader_heading[..., 0])
        evader_rot = euler_to_quaternion(torch.stack([
            torch.zeros_like(evader_yaw), torch.zeros_like(evader_yaw), evader_yaw,
        ], dim=-1))

        self.pursuer_local_pos[env_ids] = pursuer_pos
        self.pursuer_local_rot[env_ids] = pursuer_rot
        self.evader_local_pos[env_ids] = evader_pos
        self.evader_local_rot[env_ids] = evader_rot

        env_origins = self.envs_positions[env_ids].unsqueeze(1)
        self.pursuer.set_world_poses(
            env_origins + self.pursuer_local_pos[env_ids],
            self.pursuer_local_rot[env_ids], env_ids,
        )
        self.evader.set_world_poses(
            env_origins + self.evader_local_pos[env_ids],
            self.evader_local_rot[env_ids], env_ids,
        )

        # Zero action buffers.
        self.pursuer_prev_action[env_ids] = 0.0
        self.evader_prev_action[env_ids] = 0.0

        # Seed prev_distance so first-step delta is ~0.
        spawn_vec = evader_pos - pursuer_pos
        self.prev_distance[env_ids] = torch.norm(spawn_vec, dim=-1)

    # ------------------------------------------------------------------
    # Pre-sim: convert CTBR actions to motor commands via controllers
    # ------------------------------------------------------------------
    def _pre_sim_step(self, tensordict: TensorDictBase):
        pursuer_action = tensordict[("pursuer", "action")]   # [N, 1, 4]
        evader_action = tensordict[("evader", "action")]     # [N, 1, 4]

        self.pursuer_current_action = pursuer_action.clone()
        self.evader_current_action = evader_action.clone()

        # CTBR → motor for pursuer
        pursuer_motor = self._ctbr_to_motor(
            pursuer_action, self.pursuer, self.pursuer_controller,
            tensordict["done"].squeeze(-1),
            self.pursuer_prev_action,
        )
        self.pursuer_effort = self.pursuer.apply_action(pursuer_motor)

        # CTBR → motor for evader
        evader_motor = self._ctbr_to_motor(
            evader_action, self.evader, self.evader_controller,
            tensordict["done"].squeeze(-1),
            self.evader_prev_action,
        )
        self.evader.apply_action(evader_motor)

    def _ctbr_to_motor(
        self, action: torch.Tensor, drone, controller,
        done_mask: torch.Tensor, prev_action: torch.Tensor,
    ) -> torch.Tensor:
        """Convert CTBR action [-1, 1] to per-rotor motor commands."""
        action = torch.tanh(action).clamp(-1.0, 1.0)  # [N, 1, 4]
        target_rate, target_thrust = action.split([3, 1], dim=-1)

        ctbr = torch.cat([target_rate, target_thrust], dim=-1)
        prev_ctbr = prev_action.clone()

        # LPF smoothing (same convention as PIDRateController transform).
        lpf_coef = getattr(controller, "LPF_coef", 1.0)
        ctbr_smooth = lpf_coef * ctbr + (1.0 - lpf_coef) * prev_ctbr
        target_rate_s, target_thrust_s = ctbr_smooth.split([3, 1], dim=-1)

        # Scale to controller-native ranges.
        target_clip = getattr(controller, "target_clip", 1.0)
        min_thr = getattr(controller, "min_thrust_ratio", 0.0)
        max_thr = getattr(controller, "max_thrust_ratio", 1.0)
        target_rate_s = target_rate_s * 180.0 * target_clip
        target_thrust_s = torch.clamp(
            (target_thrust_s + 1) / 2, min=min_thr, max=max_thr
        ) * 2**16

        drone_state = drone.get_state()[..., :13]
        cmds, _ctbr_out = controller(
            drone_state,
            target_rate=target_rate_s,
            target_thrust=target_thrust_s,
            reset_pid=done_mask,
        )
        torch.nan_to_num_(cmds, 0.0)

        # Update prev_action buffer for next-step LPF.
        if "pursuer" in str(drone.name):
            self.pursuer_prev_action = ctbr.detach().clone()
        else:
            self.evader_prev_action = ctbr.detach().clone()

        return cmds.reshape(self.num_envs, 1, -1)

    # ------------------------------------------------------------------
    # Observations / state
    # ------------------------------------------------------------------
    def _compute_state_and_obs(self):
        pursuer_root = self.pursuer.get_state()[..., :13]   # [N, 1, 13]
        evader_root = self.evader.get_state()[..., :13]     # [N, 1, 13]

        p_pos = pursuer_root[..., :3]
        p_rot_q = pursuer_root[..., 3:7]
        p_vel_w = pursuer_root[..., 7:13]
        p_linvel = p_vel_w[..., :3]

        e_pos = evader_root[..., :3]
        e_rot_q = evader_root[..., 3:7]
        e_vel_w = evader_root[..., 7:13]
        e_linvel = e_vel_w[..., :3]

        p_rot_mat = quaternion_to_rotation_matrix(p_rot_q).reshape(self.num_envs, 1, 9)
        e_rot_mat = quaternion_to_rotation_matrix(e_rot_q).reshape(self.num_envs, 1, 9)

        # Relative headings.
        rel_e_from_p = normalize(e_pos - p_pos)   # evader from pursuer view
        rel_p_from_e = normalize(p_pos - e_pos)   # pursuer from evader view

        # Pursuer obs: rel heading (3) + lin_vel (3) + rot_mat (9) = 15
        pursuer_obs = torch.cat([rel_e_from_p, p_linvel, p_rot_mat], dim=-1)
        # Evader obs: rel heading (3) + lin_vel (3) + rot_mat (9) = 15
        evader_obs = torch.cat([rel_p_from_e, e_linvel, e_rot_mat], dim=-1)

        self.info["pursuer_drone_state"][:] = pursuer_root[..., :13]
        self.info["evader_drone_state"][:] = evader_root[..., :13]
        self.info["pursuer_prev_action"][:] = self.pursuer_prev_action
        self.info["evader_prev_action"][:] = self.evader_prev_action

        # Latch success before cloning stats (like original Intercept).
        distance = torch.norm(e_pos - p_pos, dim=-1)  # [N, 1]
        reached = (distance <= self.active_success_radius).float()
        self.stats["capture_rate"][:] = torch.maximum(
            self.stats["capture_rate"], reached
        )

        return TensorDict({
            "pursuer": {
                "observation": pursuer_obs,
                "state": pursuer_obs,
            },
            "evader": {
                "observation": evader_obs,
                "state": evader_obs,
            },
            "stats": self.stats.clone(),
            "info": self.info.clone(),
        }, self.batch_size)

    # ------------------------------------------------------------------
    # Reward / done
    # ------------------------------------------------------------------
    def _compute_reward_and_done(self):
        pursuer_root = self.pursuer.get_state()[..., :13]   # [N, 1, 13]
        evader_root = self.evader.get_state()[..., :13]     # [N, 1, 13]

        p_pos = pursuer_root[..., :3]
        e_pos = evader_root[..., :3]
        p_linvel = pursuer_root[..., 7:10]
        p_rot_mat = pursuer_root[..., 3:7]

        distance = torch.norm(e_pos - p_pos, dim=-1, keepdim=True)   # [N, 1, 1]
        heading = e_pos - p_pos                                      # LOA toward evader

        # ---- Pursuer dense shaping -----------------------------------------
        reward_dd_p = self._reward_delta_distance(distance)
        reward_ha = self._reward_heading_alignment(p_rot_mat, heading)
        reward_pursuer_dense = reward_dd_p + reward_ha

        # ---- Evader dense shaping: negative of pursuer's delta_distance -----
        reward_dd_e = -reward_dd_p  # adversarial mirror

        # ---- Terminal events ------------------------------------------------
        r_pos = e_pos - p_pos
        dist_2d = torch.norm(torch.stack([r_pos[..., 0], r_pos[..., 1]], dim=-1),
                             dim=-1, keepdim=True)  # [N, 1, 1]
        captured = (distance <= self.active_success_radius).reshape(self.num_envs, 1)

        p_misbehave = ((pursuer_root[..., 2:3].reshape(self.num_envs, 1) < 0.15)
                       | torch.isnan(pursuer_root).any(-1)).float()
        e_misbehave = ((evader_root[..., 2:3].reshape(self.num_envs, 1) < 0.15)
                       | torch.isnan(evader_root).any(-1)).float()
        too_far = (distance > self.reset_thres).reshape(self.num_envs, 1).float()

        R = self.reward_terminal_weight

        reward_term_pursuer = torch.where(
            captured, R * torch.ones_like(captured).float(),
            torch.zeros_like(captured).float(),
        )
        reward_term_evader = torch.where(
            captured, -R * torch.ones_like(captured).float(),
            torch.ones_like(captured).float() * 0.0,  # no bonus for surviving alone; only when pursued fails
        )

        # Terminal: misbehavior penalty (self-only) + too_far → pursuer loses.
        reward_term_pursuer = torch.where(
            p_misbehave > 0, -R * p_misbehave, reward_term_pursuer
        )
        reward_term_pursuer = torch.where(
            too_far > 0, -R * too_far, reward_term_pursuer
        )
        reward_term_evader = torch.where(
            e_misbehave > 0, -R * e_misbehave, reward_term_evader
        )
        # Evader wins if pursuer misbehaves or drones drift too far apart.
        evader_win = ((p_misbehave > 0).bool() | (too_far > 0).bool())
        reward_term_evader = torch.where(
            evader_win,
            R * (p_misbehave + too_far).clamp(max=1.0),
            reward_term_evader,
        )

        # Compose total rewards.
        reward_pursuer = reward_pursuer_dense + reward_term_pursuer.unsqueeze(-1)
        reward_evader = reward_dd_e + reward_term_evader.unsqueeze(-1)

        # ---- Done flags ----------------------------------------------------
        terminated = (captured | (p_misbehave > 0).bool()
                      | (e_misbehave > 0).bool() | (too_far > 0).bool()
                      ).reshape(self.num_envs, 1)
        truncated = (self.progress_buf >= self.max_episode_length - 1).reshape(
            self.num_envs, 1
        )
        done_mask = terminated | truncated

        # ---- Stats EMA -----------------------------------------------------
        self.stats["pursuer_return"] += reward_pursuer.reshape(self.num_envs, 1)
        self.stats["evader_return"] += reward_evader.reshape(self.num_envs, 1)
        self.stats["distance"].lerp_(distance.squeeze(-1), 1 - self.alpha)
        self.stats["reward_delta_distance_pursuer"].lerp_(
            reward_dd_p.squeeze(-1), 1 - self.alpha)
        self.stats["reward_delta_distance_evader"].lerp_(
            reward_dd_e.squeeze(-1), 1 - self.alpha)
        self.stats["reward_heading_alignment"].lerp_(
            reward_ha.squeeze(-1), 1 - self.alpha)
        self.stats["reward_terminal_pursuer"].lerp_(
            reward_term_pursuer, 1 - self.alpha)
        self.stats["reward_terminal_evader"].lerp_(
            reward_term_evader, 1 - self.alpha)
        self.stats["evade_rate"][:] = torch.maximum(
            self.stats["evade_rate"],
            ((too_far + p_misbehave).clamp(max=1.0)),
        )
        self.stats["pursuer_crash_rate"][:] = torch.maximum(
            self.stats["pursuer_crash_rate"], p_misbehave)
        self.stats["evader_crash_rate"][:] = torch.maximum(
            self.stats["evader_crash_rate"], e_misbehave)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["success_radius"][:] = self.active_success_radius

        # Advance curriculum once per simulator step.
        self.success_radius_global_step += 1
        self._update_success_radius()

        return TensorDict({
            "pursuer": {"reward": reward_pursuer.unsqueeze(-1)},
            "evader": {"reward": reward_evader.unsqueeze(-1)},
            "done": done_mask,
            "terminated": terminated,
            "truncated": truncated,
        }, self.batch_size)

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------
    @property
    def active_success_radius(self) -> float:
        if not self.training:
            return self.success_radius_eval
        return self.success_radius

    def _update_success_radius(self):
        radius = self.success_radius_init - self.success_radius_lr * float(
            self.success_radius_global_step
        )
        self.success_radius = max(radius, self.success_radius_max)

    # ------------------------------------------------------------------
    # Reward helpers (same signatures as original Intercept)
    # ------------------------------------------------------------------
    def _reward_delta_distance(self, distance: torch.Tensor) -> torch.Tensor:
        delta = self.prev_distance - distance.squeeze(-1)
        first_step_mask = (self.progress_buf == 0).unsqueeze(-1).float()
        reward = self.reward_delta_distance_weight * delta * (1.0 - first_step_mask)
        self.prev_distance = distance.squeeze(-1).detach().clone()
        return reward.unsqueeze(-1)

    def _reward_heading_alignment(self, rot_quat: torch.Tensor,
                                  heading: torch.Tensor) -> torch.Tensor:
        forward = quaternion_to_rotation_matrix(rot_quat)[..., 0]   # body-x axis
        heading_n = normalize(heading)
        cos_sim = (forward * heading_n).sum(dim=-1, keepdim=True)
        return self.reward_heading_alignment_weight * ((cos_sim + 1.0) / 2.0).clamp(0.0, 1.0)
