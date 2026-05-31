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

import torch
import torch.distributions as D

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Composite, UnboundedContinuous

import omni_drones.utils.kit as kit_utils

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import (
    euler_to_quaternion,
    normalize,
    quaternion_to_rotation_matrix,
)


class InterceptCompetitive(IsaacEnv):
    r"""
    Standalone two-agent competitive interception task (pursuer vs evader).

    Layout (single :class:`AgentSpec` of ``n=2`` so the existing multi-agent
    algorithms in ``omni_drones.learning`` can train it without changes):

    - ``("agents", "observation")``: ``[B, 2, obs_dim]`` (index 0 = pursuer,
      index 1 = evader). Each row is the agent-centric egocentric view, so a
      shared or non-shared actor sees symmetric features.
    - ``("agents", "observation_central")``: ``[B, 2, state_dim]``. The
      centralized critic input — the SAME joint state, broadcast across the
      agent dimension because the modern MAPPO/MATD3 critic indexes by agent.
    - ``("agents", "action")``: ``[B, 2, 4]``. Per-agent body-rate (3) +
      thrust (1) command in ``[-1, 1]``. The env converts each row to
      per-rotor commands internally using the corresponding
      ``RateController``.
    - ``("agents", "reward")``: ``[B, 2, 1]``. Asymmetric per-agent reward;
      see :meth:`_compute_reward_and_done`.

    Reward design (hybrid):

    - **Terminal events** are zero-sum: capture awards ``+R`` to the
      pursuer and ``-R`` to the evader; the evader reaching its goal (or
      the two drones drifting beyond ``reset_thres``) awards ``-R`` to the
      pursuer and ``+R`` to the evader.
    - **Per-step dense shaping** is asymmetric by default (each agent's
      own objective). Set ``task.zero_sum_dense=true`` to instead mirror
      the pursuer's dense reward into ``-reward`` for the evader.
    - **Crash penalty** is self-only (whichever drone crashes is
      penalised).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, cfg, headless):
        self.cfg = cfg
        task_cfg = cfg.task

        # ---- General / shared knobs --------------------------------------
        self.reset_thres = float(task_cfg.get("reset_thres", 15.0))
        self.time_encoding_dim = int(task_cfg.get("time_encoding_dim", 0))
        self.include_rel_opp_vel = bool(task_cfg.get("include_rel_opp_vel", False))
        self.include_pursuer_rel_goal_heading = bool(
            task_cfg.get("include_pursuer_rel_goal_heading", True)
        )
        self.alpha = 0.8  # EMA factor for running stats
        self.reward_approach_velocity_weight = float(
            task_cfg.get("reward_approach_velocity_weight", 1.0))

        # ---- Pursuer-side cfg -------------------------------------------
        pursuer_cfg = task_cfg.pursuer
        self.pursuer_model_name = pursuer_cfg.get("model", "Hummingbird")
        self.pursuer_controller_name = pursuer_cfg.get(
            "controller", "RateController")
        self.pursuer_target_speed = float(
            pursuer_cfg.get("target_speed", 15.0))
        self._pursuer_spawn_pos_range = pursuer_cfg.spawn_pos_range
        self._pursuer_spawn_rpy_range = pursuer_cfg.spawn_rpy_range

        # ---- Evader-side cfg --------------------------------------------
        # The evader is a learned RL agent here, so we ALWAYS use the
        # RateController regardless of what the task yaml says. Drone model
        # and spawn ranges are still honoured.
        evader_cfg = task_cfg.evader
        self.evader_model_name = evader_cfg.get("model", "Hummingbird")
        self.evader_controller_name = "RateController"
        self.evader_spawn_distance_range = evader_cfg.get(
            "spawn_distance_range", [4.0, 7.0])

        # ---- Competitive-only knobs --------------------------------------
        self.capture_radius = float(task_cfg.get("capture_radius", 0.5))
        self.goal_radius = float(task_cfg.get("goal_radius", 0.5))
        self.crash_alt = float(task_cfg.get("crash_alt", 0.15))
        self.crash_penalty = float(task_cfg.get("crash_penalty", 5.0))
        self.terminal_reward = float(task_cfg.get("terminal_reward", 10.0))
        self.zero_sum_dense = bool(task_cfg.get("zero_sum_dense", False))
        self.evader_goal_reward_scale = float(
            task_cfg.get("evader_goal_reward_scale", 0.8))
        self.evader_evade_bonus = float(
            task_cfg.get("evader_evade_bonus", 0.1))

        goal_range = task_cfg.get("goal_pos_range", None)
        if goal_range is None:
            self._goal_min = [-3.0, -3.0, 1.0]
            self._goal_max = [3.0, 3.0, 3.0]
        else:
            self._goal_min = list(goal_range["min"])
            self._goal_max = list(goal_range["max"])

        # IsaacEnv.__init__ runs _design_scene then _set_specs.
        super().__init__(cfg, headless)

        self.pursuer.initialize()
        self.evader.initialize()

        # ---- Spawn distributions ----------------------------------------
        self.pursuer_init_pos_dist = D.Uniform(
            torch.tensor(self._pursuer_spawn_pos_range.min, device=self.device),
            torch.tensor(self._pursuer_spawn_pos_range.max, device=self.device),
        )
        self.pursuer_init_rpy_dist = D.Uniform(
            torch.tensor(self._pursuer_spawn_rpy_range.min, device=self.device)
            * torch.pi,
            torch.tensor(self._pursuer_spawn_rpy_range.max, device=self.device)
            * torch.pi,
        )
        self.evader_spawn_distance_dist = D.Uniform(
            torch.tensor(self.evader_spawn_distance_range[0], device=self.device),
            torch.tensor(self.evader_spawn_distance_range[1], device=self.device),
        )
        self.goal_pos_dist = D.Uniform(
            torch.tensor(self._goal_min, device=self.device),
            torch.tensor(self._goal_max, device=self.device),
        )

        # ---- Local-frame state buffers ----------------------------------
        # ``local`` = relative to the per-env origin (envs_positions).
        self.pursuer_local_pos = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.pursuer_local_rot = torch.zeros(
            self.num_envs, 1, 4, device=self.device)
        self.pursuer_local_vel = torch.zeros(
            self.num_envs, 1, 6, device=self.device)
        self.evader_local_pos = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.evader_local_rot = torch.zeros(
            self.num_envs, 1, 4, device=self.device)
        self.evader_local_vel = torch.zeros(
            self.num_envs, 1, 6, device=self.device)
        self.goal_local_pos = torch.zeros(
            self.num_envs, 1, 3, device=self.device)

        # ---- Debug visualization ---------------------------------------
        self.goal_marker_size = float(task_cfg.get("goal_marker_size", 0.2))
        self.arrow_pos_offset = torch.tensor([0.0, 0.0, 0.15], device=self.device)
        self.pursuer_arrow_color = (1.0, 0.2, 0.2, 1.0)  # red
        self.evader_arrow_color = (0.2, 0.4, 1.0, 1.0)   # blue
        self.goal_marker_color = (0.2, 1.0, 0.2, 1.0)    # green

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _design_scene(self):
        """Create pursuer and evader drones plus a ground plane in each env."""
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
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        self.pursuer.spawn(translations=[(0.0, 0.0, 1.6)])
        self.evader.spawn(translations=[(5.0, 0.0, 2.0)])
        return ["/World/defaultGroundPlane"]

    # ------------------------------------------------------------------
    # Spec definition
    # ------------------------------------------------------------------
    def _set_specs(self):
        """Single AgentSpec (n=2) with stacked obs / action / reward / state."""
        agent_self_dim = 3 + 9 + 1            # lin_vel + R(3x3) + altitude = 13
        rel_dim = 3 + (3 if self.include_rel_opp_vel else 0) + 3  # rel_opp + [rel_opp_vel] + rel_goal
        time_dim = self.time_encoding_dim or 0

        per_agent_obs_dim = agent_self_dim + rel_dim + time_dim
        # Central state: pursuer self + evader self + goal_pos + time encoding.
        state_dim = agent_self_dim * 2 + 3 + time_dim
        action_dim = 4                        # 3 body rates + 1 thrust
        num_agents = 2

        self.observation_spec = Composite({
            "agents": {
                "observation": UnboundedContinuous(
                    torch.Size([num_agents, per_agent_obs_dim]),
                    device=self.device,
                ),
                "observation_central": UnboundedContinuous(
                    torch.Size([num_agents, state_dim]),
                    device=self.device,
                ),
            }
        }).expand(self.num_envs).to(self.device)

        self.action_spec = Composite({
            "agents": {
                "action": Bounded(
                    low=-1.0, high=1.0,
                    shape=torch.Size([num_agents, action_dim]),
                    device=self.device,
                ),
            }
        }).expand(self.num_envs).to(self.device)

        self.reward_spec = Composite({
            "agents": {
                "reward": UnboundedContinuous(
                    torch.Size([num_agents, 1]),
                    device=self.device,
                ),
            }
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["agents"] = AgentSpec(  # pyright: ignore[reportArgumentType]
            "agents",
            num_agents,
            observation_key=("agents", "observation"),  # pyright: ignore[reportArgumentType]
            action_key=("agents", "action"),  # pyright: ignore[reportArgumentType]
            reward_key=("agents", "reward"),  # pyright: ignore[reportArgumentType]
            state_key=("agents", "observation_central"),  # pyright: ignore[reportArgumentType]
        )

        # ---- Stats: scalar episode summaries (not per-agent) -------------
        def scalar(): return UnboundedContinuous(
            torch.Size([1]), device=self.device)
        self.stats_spec = Composite({
            "pursuer_return": scalar(),
            "evader_return": scalar(),
            "episode_len": scalar(),
            "distance_pursuer_evader": scalar(),
            "distance_evader_goal": scalar(),
            "capture_rate": scalar(),
            "goal_reach_rate": scalar(),
            "pursuer_crash_rate": scalar(),
            "evader_crash_rate": scalar(),
        }).expand(self.num_envs).to(self.device)

        self.info_spec = Composite({
            "pursuer_drone_state": UnboundedContinuous(
                torch.Size([1, 13]), device=self.device),
            "evader_drone_state": UnboundedContinuous(
                torch.Size([1, 13]), device=self.device),
        }).expand(self.num_envs).to(self.device)

        self.observation_spec["stats"] = self.stats_spec
        self.observation_spec["info"] = self.info_spec

        self.stats = self.stats_spec.zero()
        self.info = self.info_spec.zero()

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor):
        """Randomize both drones' poses + sample a fresh goal."""
        n = len(env_ids)

        # Zero stats for the new episode under our schema.
        self.stats[env_ids] = 0.0

        # Internal drone reset (e.g. randomization wrappers).
        self.pursuer._reset_idx(env_ids, self.training)
        self.evader._reset_idx(env_ids, self.training)

        # ---- Sample pursuer pose ----------------------------------------
        pursuer_pos = self.pursuer_init_pos_dist.sample(torch.Size([n, 1]))
        pursuer_rpy = self.pursuer_init_rpy_dist.sample(torch.Size([n, 1]))
        pursuer_rot = euler_to_quaternion(pursuer_rpy)

        # ---- Sample evader pose -----------------------------------------
        # Place the evader at a random direction & distance from the pursuer,
        # with a positive vertical component so it doesn't spawn under-ground.
        pursuer_yaw = pursuer_rpy[..., 2]
        yaw_noise = torch.randn_like(pursuer_yaw) * 0.5
        spawn_yaw = pursuer_yaw + yaw_noise
        spawn_direction = normalize(torch.stack([
            torch.cos(spawn_yaw),
            torch.sin(spawn_yaw),
            torch.rand((n, 1), device=self.device),
        ], dim=-1))
        spawn_direction[..., 2] = spawn_direction[..., 2].abs()
        spawn_direction = normalize(spawn_direction)
        spawn_distance = self.evader_spawn_distance_dist.sample(
            torch.Size([n, 1, 1]))
        evader_pos = pursuer_pos + spawn_direction * spawn_distance

        # Evader yaw points in a random horizontal direction.
        evader_heading = normalize(torch.randn(n, 1, 3, device=self.device))
        evader_heading[..., 2] = 0.0
        evader_heading = normalize(evader_heading)
        evader_yaw = torch.atan2(
            evader_heading[..., 1], evader_heading[..., 0])
        evader_rot = euler_to_quaternion(torch.stack([
            torch.zeros_like(evader_yaw),
            torch.zeros_like(evader_yaw),
            evader_yaw,
        ], dim=-1))

        # ---- Persist into local-frame buffers ---------------------------
        self.pursuer_local_pos[env_ids] = pursuer_pos
        self.pursuer_local_rot[env_ids] = pursuer_rot
        self.pursuer_local_vel[env_ids] = 0.0
        self.evader_local_pos[env_ids] = evader_pos
        self.evader_local_rot[env_ids] = evader_rot
        self.evader_local_vel[env_ids] = 0.0

        # ---- Push to simulator ------------------------------------------
        env_origins = self.envs_positions[env_ids].unsqueeze(1)
        self.pursuer.set_world_poses(
            env_origins + self.pursuer_local_pos[env_ids],
            self.pursuer_local_rot[env_ids],
            env_ids,
        )
        self.pursuer.set_velocities(self.pursuer_local_vel[env_ids], env_ids)
        self.evader.set_world_poses(
            env_origins + self.evader_local_pos[env_ids],
            self.evader_local_rot[env_ids],
            env_ids,
        )
        self.evader.set_velocities(self.evader_local_vel[env_ids], env_ids)

        # ---- Sample a fresh goal in env-local frame ---------------------
        self.goal_local_pos[env_ids] = self.goal_pos_dist.sample(
            torch.Size([n, 1]))

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------
    def _pre_sim_step(self, tensordict: TensorDictBase):
        """Split joint action into pursuer/evader and convert rate → motor."""
        actions = tensordict[("agents", "action")]    # [N, 2, 4]
        pursuer_action = actions[:, 0:1, :].contiguous()
        evader_action = actions[:, 1:2, :].contiguous()

        pursuer_motor = self._rate_action_to_motor(
            pursuer_action, self.pursuer, self.pursuer_controller)
        evader_motor = self._rate_action_to_motor(
            evader_action, self.evader, self.evader_controller)

        self.pursuer_effort = self.pursuer.apply_action(pursuer_motor)
        self.evader.apply_action(evader_motor)

    def _rate_action_to_motor(
        self,
        action: torch.Tensor,
        drone,
        controller,
    ) -> torch.Tensor:
        """Map a normalized rate+thrust action in [-1, 1] to per-rotor commands."""
        action = torch.nan_to_num(action, 0.0).clamp(-1.0, 1.0)
        action = action.reshape(self.num_envs, 1, -1)

        # Match the torchrl RateController transform convention: thrust is a
        # single scalar per drone scaled by the SUM of max motor thrusts.
        target_rate, target_thrust = action.split([3, 1], -1)
        max_thrust = controller.max_thrusts.sum(-1)
        target_rate = target_rate * torch.pi
        target_thrust = ((target_thrust + 1) / 2).clip(0.0) * max_thrust
        drone_state = drone.get_state()[..., :13]
        motor_cmds = controller(
            drone_state,
            target_rate=target_rate,
            target_thrust=target_thrust,
        )
        torch.nan_to_num_(motor_cmds, 0.0)
        return motor_cmds.reshape(self.num_envs, 1, -1)

    # ------------------------------------------------------------------
    # Observations / state
    # ------------------------------------------------------------------
    def _compute_state_and_obs(self):
        """Build stacked per-agent observations and the shared centralized state."""
        pursuer_state_full = self.pursuer.get_state()  # [N, 1, S]
        evader_state_full = self.evader.get_state()    # [N, 1, S]

        pursuer_pos = pursuer_state_full[..., :3]
        pursuer_rot_quat = pursuer_state_full[..., 3:7]
        pursuer_vel = pursuer_state_full[..., 7:13]
        pursuer_lin_vel = pursuer_vel[..., :3]
        pursuer_alt = pursuer_pos[..., 2:3]
        pursuer_rot = quaternion_to_rotation_matrix(pursuer_rot_quat).reshape(
            self.num_envs, 1, 9)

        evader_pos = evader_state_full[..., :3]
        evader_rot_quat = evader_state_full[..., 3:7]
        evader_vel = evader_state_full[..., 7:13]
        evader_lin_vel = evader_vel[..., :3]
        evader_alt = evader_pos[..., 2:3]
        evader_rot = quaternion_to_rotation_matrix(evader_rot_quat).reshape(
            self.num_envs, 1, 9)

        goal_world_pos = (
            self.envs_positions.unsqueeze(1) + self.goal_local_pos
        )

        rel_pursuer_to_evader = evader_pos - pursuer_pos
        rel_pursuer_to_evader_vel = evader_lin_vel - pursuer_lin_vel
        rel_pursuer_to_goal = goal_world_pos - pursuer_pos
        rel_evader_to_goal = goal_world_pos - evader_pos

        if not self.include_pursuer_rel_goal_heading:
            rel_pursuer_to_goal = torch.zeros_like(rel_pursuer_to_goal)

        pursuer_self = torch.cat(
            [pursuer_lin_vel, pursuer_rot, pursuer_alt], dim=-1)
        evader_self = torch.cat(
            [evader_lin_vel, evader_rot, evader_alt], dim=-1)

        # Pursuer egocentric obs: self + rel_pos(opp) + [rel_vel(opp)] + rel_pos(goal).
        # Evader  egocentric obs: self + rel_pos(opp) + [rel_vel(opp)] + rel_pos(goal).
        # (signs flip because each agent observes the opponent from its own perspective)
        pursuer_obs_parts = [
            pursuer_self,
            rel_pursuer_to_evader,
            rel_pursuer_to_goal,
        ]
        evader_obs_parts = [
            evader_self,
            -rel_pursuer_to_evader,
            rel_evader_to_goal,
        ]

        if self.include_rel_opp_vel:
            pursuer_obs_parts.insert(2, rel_pursuer_to_evader_vel)
            evader_obs_parts.insert(2, -rel_pursuer_to_evader_vel)

        state_parts = [pursuer_self, evader_self, self.goal_local_pos]

        if self.time_encoding_dim:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            time_enc = t.expand(-1, self.time_encoding_dim).unsqueeze(1)
            pursuer_obs_parts.append(time_enc)
            evader_obs_parts.append(time_enc)
            state_parts.append(time_enc)

        pursuer_obs = torch.cat(pursuer_obs_parts, dim=-1)   # [N, 1, obs_dim]
        evader_obs = torch.cat(evader_obs_parts, dim=-1)     # [N, 1, obs_dim]
        joint_state = torch.cat(state_parts, dim=-1)         # [N, 1, state_dim]

        agents_obs = torch.cat([pursuer_obs, evader_obs], dim=1)        # [N, 2, obs_dim]
        agents_central = joint_state.expand(-1, 2, -1).contiguous()     # [N, 2, state_dim]

        # Cache for the reward function (avoid recomputing get_state).
        self._cached_pursuer_pos = pursuer_pos
        self._cached_evader_pos = evader_pos
        self._cached_pursuer_vel = pursuer_lin_vel
        self._cached_evader_vel = evader_lin_vel
        self._cached_goal_world = goal_world_pos
        self._cached_pursuer_state13 = pursuer_state_full[..., :13]
        self._cached_evader_state13 = evader_state_full[..., :13]

        self.info["pursuer_drone_state"][:] = pursuer_state_full[..., :13]
        self.info["evader_drone_state"][:] = evader_state_full[..., :13]

        # self._render_debug_visuals(
        #     pursuer_pos,
        #     pursuer_lin_vel,
        #     evader_pos,
        #     evader_lin_vel,
        #     goal_world_pos,
        # )

        return TensorDict(
            {
                "agents": {
                    "observation": agents_obs,
                    "observation_central": agents_central,
                },
                "stats": self.stats.clone(),
                "info": self.info.clone(),
            },
            self.batch_size,
        )

    # ------------------------------------------------------------------
    # Rewards / termination
    # ------------------------------------------------------------------
    def _compute_reward_and_done(self):
        """Zero-sum terminals + per-agent dense shaping + self-only crashes."""
        pursuer_pos = self._cached_pursuer_pos      # [N, 1, 3]
        evader_pos = self._cached_evader_pos        # [N, 1, 3]
        pursuer_vel = self._cached_pursuer_vel      # [N, 1, 3]
        goal_world = self._cached_goal_world        # [N, 1, 3]

        dist_pe = torch.norm(
            evader_pos - pursuer_pos, dim=-1, keepdim=True)   # [N, 1, 1]
        dist_eg = torch.norm(
            goal_world - evader_pos, dim=-1, keepdim=True)    # [N, 1, 1]

        # ----- Dense per-step shaping ------------------------------------
        rel_pursuer_to_evader = evader_pos - pursuer_pos
        reward_pursuer_dense = self._reward_approach_velocity_to_evader(
            pursuer_vel, rel_pursuer_to_evader)               # [N, 1, 1]

        reward_evader_goal = torch.exp(
            -self.evader_goal_reward_scale * dist_eg.squeeze(-1))  # [N, 1]
        reward_evader_evade = self.evader_evade_bonus * torch.tanh(
            dist_pe.squeeze(-1))                                   # [N, 1]
        reward_evader_dense = (
            reward_evader_goal + reward_evader_evade
        ).unsqueeze(-1)                                            # [N, 1, 1]

        if self.zero_sum_dense:
            reward_evader_dense = -reward_pursuer_dense

        # ----- Terminal events -------------------------------------------
        dist_pe_2d = dist_pe.squeeze(-1)                       # [N, 1]
        dist_eg_2d = dist_eg.squeeze(-1)                       # [N, 1]
        captured = (dist_pe_2d <= self.capture_radius)
        evader_reached_goal = (
            dist_eg_2d <= self.goal_radius) & (~captured)

        pursuer_state13 = self._cached_pursuer_state13
        evader_state13 = self._cached_evader_state13

        pursuer_crashed = (
            (pursuer_state13[..., 2:3] < self.crash_alt)
            | torch.isnan(pursuer_state13).any(-1, keepdim=True)
        )                                                       # [N, 1, 1]
        evader_crashed = (
            (evader_state13[..., 2:3] < self.crash_alt)
            | torch.isnan(evader_state13).any(-1, keepdim=True)
        )
        too_far = (dist_pe_2d > self.reset_thres)               # [N, 1]

        zero = torch.zeros_like(reward_pursuer_dense)
        R = self.terminal_reward

        # Reshape boolean masks to [N, 1, 1] so torch.where broadcasts cleanly.
        captured3 = captured.unsqueeze(-1)
        reached3 = evader_reached_goal.unsqueeze(-1)
        too_far3 = too_far.unsqueeze(-1)

        terminal_pursuer = torch.where(
            captured3, torch.full_like(zero, +R),
            torch.where(
                reached3 | too_far3,
                torch.full_like(zero, -R),
                zero,
            ),
        )
        terminal_evader = torch.where(
            captured3, torch.full_like(zero, -R),
            torch.where(
                reached3 | too_far3,
                torch.full_like(zero, +R),
                zero,
            ),
        )

        # Self-only crash penalty.
        terminal_pursuer = torch.where(
            pursuer_crashed, torch.full_like(zero, -self.crash_penalty),
            terminal_pursuer)
        terminal_evader = torch.where(
            evader_crashed, torch.full_like(zero, -self.crash_penalty),
            terminal_evader)

        # reward_pursuer = reward_pursuer_dense + terminal_pursuer   # [N, 1, 1]
        # reward_evader = reward_evader_dense + terminal_evader      # [N, 1, 1]
        reward_pursuer = terminal_pursuer
        reward_evader = terminal_evader

        # Stack along the agent dim → [N, 2, 1].
        agents_reward = torch.cat([reward_pursuer, reward_evader], dim=1)

        # ----- Done flags ------------------------------------------------
        terminated = (
            captured
            | evader_reached_goal
            | too_far
            | pursuer_crashed.squeeze(-1)
            | evader_crashed.squeeze(-1)
        ).reshape(self.num_envs, 1)
        truncated = (
            self.progress_buf >= self.max_episode_length - 1
        ).reshape(self.num_envs, 1)
        done = terminated | truncated

        # ----- Stats -----------------------------------------------------
        self.stats["pursuer_return"] += reward_pursuer.reshape(self.num_envs, 1)
        self.stats["evader_return"] += reward_evader.reshape(self.num_envs, 1)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(-1)
        self.stats["distance_pursuer_evader"].lerp_(
            dist_pe_2d, 1 - self.alpha)
        self.stats["distance_evader_goal"].lerp_(
            dist_eg_2d, 1 - self.alpha)
        self.stats["capture_rate"][:] = torch.maximum(
            self.stats["capture_rate"], captured.float())
        self.stats["goal_reach_rate"][:] = torch.maximum(
            self.stats["goal_reach_rate"], evader_reached_goal.float())
        self.stats["pursuer_crash_rate"][:] = torch.maximum(
            self.stats["pursuer_crash_rate"],
            pursuer_crashed.float().reshape(self.num_envs, 1))
        self.stats["evader_crash_rate"][:] = torch.maximum(
            self.stats["evader_crash_rate"],
            evader_crashed.float().reshape(self.num_envs, 1))

        return TensorDict(
            {
                "agents": {"reward": agents_reward},
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------
    def _reward_approach_velocity_to_evader(
        self,
        pursuer_velocity: torch.Tensor,
        pursuer_to_evader_heading: torch.Tensor,
    ) -> torch.Tensor:
        """Reward forward progress toward the evader at the desired speed."""
        approach_speed = (pursuer_velocity * normalize(pursuer_to_evader_heading)).sum(
            dim=-1, keepdim=True
        )
        normalized_speed = approach_speed / self.pursuer_target_speed
        return self.reward_approach_velocity_weight * normalized_speed.clamp(-1.0, 1.0)

    # ------------------------------------------------------------------
    # Debug rendering helpers
    # ------------------------------------------------------------------
    def _render_debug_visuals(
        self,
        pursuer_pos: torch.Tensor,
        pursuer_lin_vel: torch.Tensor,
        evader_pos: torch.Tensor,
        evader_lin_vel: torch.Tensor,
        goal_world_pos: torch.Tensor,
    ):
        """Draw role-colored velocity arrows and goal markers for all envs."""
        if not self._should_render(0):
            return

        self.debug_draw.clear()

        self._draw_velocity_arrow(
            pursuer_pos,
            pursuer_lin_vel,
            self.pursuer_arrow_color,
        )
        self._draw_velocity_arrow(
            evader_pos,
            evader_lin_vel,
            self.evader_arrow_color,
        )
        self._draw_goal_marker(goal_world_pos)

    def _draw_velocity_arrow(
        self,
        pos: torch.Tensor,
        lin_vel: torch.Tensor,
        color,
    ):
        """Draw a velocity unit-vector arrow from a point above each drone."""
        direction = normalize(lin_vel)
        arrow_start = pos + self.arrow_pos_offset.view(1, 1, 3)
        self.debug_draw.vector(
            arrow_start.reshape(-1, 3),
            direction.reshape(-1, 3),
            size=3.0,
            color=color,
        )

    def _draw_goal_marker(self, goal_pos: torch.Tensor):
        """Draw a 3-axis cross marker at the goal point."""
        s = self.goal_marker_size
        center = goal_pos.reshape(-1, 3)

        x_start = center + torch.tensor([[-s, 0.0, 0.0]], device=self.device)
        y_start = center + torch.tensor([[0.0, -s, 0.0]], device=self.device)
        z_start = center + torch.tensor([[0.0, 0.0, -s]], device=self.device)

        x_vec = torch.tensor([[2 * s, 0.0, 0.0]], device=self.device).expand_as(x_start)
        y_vec = torch.tensor([[0.0, 2 * s, 0.0]], device=self.device).expand_as(y_start)
        z_vec = torch.tensor([[0.0, 0.0, 2 * s]], device=self.device).expand_as(z_start)

        self.debug_draw.vector(x_start, x_vec, size=4.0, color=self.goal_marker_color)
        self.debug_draw.vector(y_start, y_vec, size=4.0, color=self.goal_marker_color)
        self.debug_draw.vector(z_start, z_vec, size=4.0, color=self.goal_marker_color)
