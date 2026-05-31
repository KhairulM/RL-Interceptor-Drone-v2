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

import omni_drones.utils.kit as kit_utils

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Composite, UnboundedContinuous

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import (
    euler_to_quaternion,
    normalize,
    quaternion_to_rotation_matrix,
)


class Intercept(IsaacEnv):
    r"""
    A pursuit task where a Hummingbird drone chases an evader drone that moves in a straight line trajectory

    ## Observation

    - `evader_rel_hdg` (3): Relative heading of the evader from the pursuer.
    - `pursuer_lin_vel` (3): Pursuer's linear velocity.
    - `pursuer_rot` (9): Pursuer's orientation as a flattened rotation matrix.
    - `pursuer_alt` (1): Pursuer's  altitude
    - `evader_rel_lin_vel` (optional, 3): Evader linear velocity relative to the pursuer linear velocity.
    - `pursuer_pos` (optional, 3): Pursuer's position in world frame.
    - `pursuer_rot_vel` (optional, 3): Pursuer's angular velocity.
    - `time_encoding` (optional, 4): Sinusoidal encoding of the normalized episode time.

    ## Reward

        - `closing`: Distance-closing reward, ``exp(-reward_distance_scale * distance)``,
            approaching 1 as the drones get closer and 0 when they are far apart.
            (`_reward_distance_to_evader`)
        - `alignment`: Cosine-similarity reward for aligning the pursuer's velocity
            vector with the line of sight to the evader, mapped to ``[0, 1]``.
            (`_reward_align_velocity_to_heading`)
        - `approach_speed`: Reward for forward progress toward the evader at the
            desired speed; projects the pursuer's velocity onto the line of sight
            and normalizes by ``pursuer_target_speed``. Weighted by
            ``reward_approach_velocity_weight``.
            (`_reward_approach_velocity_to_evader`)
        - `time_to_intercept`: Negative-shaped reward derived from the estimated
            time-to-intercept given the current relative motion, clamped to
            ``[-1, 0]``.
            (`_reward_intercept_time`)
        - `action_smoothness`: ``exp(-||a_t - a_{t-1}||)`` weighted by
            ``reward_action_smoothness_weight``; gated to zero on the first two
            steps of an episode.
            (`_reward_action_smoothness`)
        - `heading_alignment`: Reward for the pursuer's forward (body-x) axis
            being aligned with the line of sight to the evader, weighted by
            ``reward_heading_alignment_weight``.
            (`_reward_heading_alignment`)
        - `delta_distance`: Step-over-step reduction in pursuer-to-evader distance
            (``prev_distance - distance``), weighted by
            ``reward_delta_distance_weight``. Positive when closing, negative when
            falling behind; zeroed on the first step of each episode.
            (`_reward_delta_distance`)
        - `terminal`: ``+1`` when the pursuer enters ``success_radius`` of the
            evader, ``-1`` when it misbehaves (drops below 0.15 m altitude, goes
            NaN, or exceeds ``reset_thres`` distance), ``0`` otherwise.

    The composed training reward (see `_compute_reward_and_done`) is currently
    ``reward_approach_velocity + reward_heading_alignment + reward_delta_distance``
    plus the terminal bonus/penalty. Other reward terms above are computed and
    logged in ``stats`` but not summed into the training signal; toggle them by
    editing the composition line.
    """

    def __init__(self, cfg, headless):
        """Initialize the interception task, cached configuration, and buffers."""
        self.cfg = cfg

        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.reset_thres = cfg.task.get("reset_thres", 15.0)
        self.success_radius = cfg.task.get("success_radius", 0.5)
        self.time_encoding_dim = cfg.task.get("time_encoding_dim", 0)

        self.reward_action_smoothness_weight = cfg.task.get(
            "reward_action_smoothness_weight", 0.0)
        self.reward_heading_alignment_weight = cfg.task.get(
            "reward_heading_alignment_weight", 0.0)
        self.reward_approach_velocity_weight = cfg.task.get(
            "reward_approach_velocity_weight", 1.0)
        self.reward_delta_distance_weight = cfg.task.get(
            "reward_delta_distance_weight", 0.0)

        self.pursuer_cfg = cfg.task.pursuer
        self.evader_cfg = cfg.task.evader

        self.pursuer_model_name = self.pursuer_cfg.get("model", "Hummingbird")
        self.pursuer_controller_name = self.pursuer_cfg.get(
            "controller", "RateController")
        self.pursuer_target_speed = self.pursuer_cfg.get("target_speed", 15.0)
        self.pursuer_use_ab_world_frame = self.pursuer_cfg.get(
            "use_ab_world_frame", False)
        self.pursuer_use_rot_speed = self.pursuer_cfg.get("use_rot_speed", False)

        self.evader_model_name = self.evader_cfg.get("model", "Hummingbird")
        self.evader_controller_name = self.evader_cfg.get(
            "controller", "LeePositionController"
        )
        self.evader_spawn_distance_range = self.evader_cfg.get(
            "spawn_distance_range",
            [4.0, 7.0],
        )
        self.evader_speed_range = self.evader_cfg.get(
            "speed_range",
            [2, 15],
        )
        self.evader_use_relative_velocity = self.evader_cfg.get(
            "use_relative_velocity", False)

        # Trajectory selection. Map enabled types to integer codes used by
        # _compute_evader_action (0 = linear, 1 = zigzag).
        self._traj_type_codes = {"linear": 0, "zigzag": 1}
        enabled = list(self.evader_cfg.get("trajectory_types", ["linear"]))
        unknown = [t for t in enabled if t not in self._traj_type_codes]
        if unknown:
            raise ValueError(
                f"Unknown evader trajectory_types: {unknown}. "
                f"Supported: {list(self._traj_type_codes)}")
        self.evader_enabled_traj_codes = [self._traj_type_codes[t] for t in enabled]

        zigzag_cfg = self.evader_cfg.get("zigzag", {})
        self.evader_zigzag_amp_range = list(
            zigzag_cfg.get("amplitude_range", [1.0, 3.0]))
        self.evader_zigzag_freq_range = list(
            zigzag_cfg.get("frequency_range", [0.3, 1.0]))

        super().__init__(cfg, headless)

        self.pursuer.initialize()
        self.evader.initialize()

        self.pursuer_init_pos_dist = D.Uniform(
            torch.tensor(self.pursuer_cfg.spawn_pos_range.min, device=self.device),
            torch.tensor(self.pursuer_cfg.spawn_pos_range.max, device=self.device),
        )
        self.pursuer_init_rpy_dist = D.Uniform(
            torch.tensor(self.pursuer_cfg.spawn_rpy_range.min, device=self.device) * torch.pi,
            torch.tensor(self.pursuer_cfg.spawn_rpy_range.max, device=self.device) * torch.pi,
        )
        self.evader_speed_dist = D.Uniform(
            torch.tensor(self.evader_speed_range[0], device=self.device),
            torch.tensor(self.evader_speed_range[1], device=self.device),
        )
        self.evader_spawn_distance_dist = D.Uniform(
            torch.tensor(
                self.evader_spawn_distance_range[0], device=self.device),
            torch.tensor(
                self.evader_spawn_distance_range[1], device=self.device),
        )
        self.evader_zigzag_amp_dist = D.Uniform(
            torch.tensor(self.evader_zigzag_amp_range[0], device=self.device),
            torch.tensor(self.evader_zigzag_amp_range[1], device=self.device),
        )
        self.evader_zigzag_freq_dist = D.Uniform(
            torch.tensor(self.evader_zigzag_freq_range[0], device=self.device),
            torch.tensor(self.evader_zigzag_freq_range[1], device=self.device),
        )
        self._evader_enabled_codes_tensor = torch.tensor(
            self.evader_enabled_traj_codes, device=self.device, dtype=torch.long)

        # Buffers for storing the local states of the pursuer and evader relative to their spawn positions, which are used for computing observations and resetting the drones
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

        self.evader_line_dir = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.evader_line_speed = torch.zeros(
            self.num_envs, 1, 1, device=self.device)
        # Per-env trajectory type code (0 = linear, 1 = zigzag) and zig-zag params.
        self.evader_traj_type = torch.zeros(
            self.num_envs, 1, device=self.device, dtype=torch.long)
        self.evader_zigzag_amp = torch.zeros(
            self.num_envs, 1, 1, device=self.device)
        self.evader_zigzag_freq = torch.zeros(
            self.num_envs, 1, 1, device=self.device)
        self.evader_zigzag_perp = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        # Previous-step pursuer-to-evader distance for the delta-distance reward.
        self.prev_distance = torch.zeros(
            self.num_envs, 1, device=self.device)

        self.alpha = 0.8

        # Observation components that require storing across steps
        self.evader_rel_hdg = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.evader_rel_lin_vel = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.pursuer_pos = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.pursuer_lin_vel = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.pursuer_rot_vel = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.pursuer_rot = torch.zeros(
            self.num_envs, 1, 9, device=self.device)

        # Buffers for action smoothness reward
        self.prev_action = torch.zeros(
            self.num_envs, 1, self.pursuer.action_spec.shape[-1],
            device=self.device,
        )
        self.action_error_order1 = torch.zeros(
            self.num_envs, 1, device=self.device)

    def _design_scene(self):
        """Create the pursuer, evader, and ground plane for each environment."""
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

        # Spawn drones at default positions; scene cloning handles replication across num_envs
        self.pursuer.spawn(translations=[(0.0, 0.0, 1.6)])
        self.evader.spawn(translations=[(5.0, 0.0, 2.0)])
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        """Define observation, action, reward, and stats specs for the task."""
        pursuer_state_dim = 3 + 9 + 1  # lin vel + rot matrix + altitude

        if self.pursuer_use_ab_world_frame:
            pursuer_state_dim += 3  # absolute position in world frame
        if self.pursuer_use_rot_speed:
            pursuer_state_dim += 3  # angular velocity

        evader_state_dim = 3  # relative heading

        if self.evader_use_relative_velocity:
            evader_state_dim += 3  # relative linear velocity

        obs_dim = pursuer_state_dim + evader_state_dim

        if self.time_encoding_dim:
            state_dim = obs_dim + self.time_encoding_dim
        else:
            state_dim = obs_dim

        self.observation_spec = Composite({
            "agents": {
                "observation":  UnboundedContinuous(torch.Size([1, obs_dim])),
                "state": UnboundedContinuous(torch.Size([1, state_dim])),
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = Composite({
            "agents": {
                # [num_motors = 4] for the pursuer, 0 for the evader since it's controlled by a scripted controller
                "action": self.pursuer.action_spec.unsqueeze(0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = Composite({
            "agents": {
                "reward": UnboundedContinuous(torch.Size([1, 1]))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )

        self.stats_spec = Composite({
            "return": UnboundedContinuous(torch.Size([1]), device=self.device),
            "episode_len": UnboundedContinuous(torch.Size([1]), device=self.device),
            "distance": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_closing": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_approach_speed": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_action_smoothness": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_heading_alignment": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_delta_distance": UnboundedContinuous(torch.Size([1]), device=self.device),
            "action_error_order1_mean": UnboundedContinuous(torch.Size([1]), device=self.device),
            "action_error_order1_max": UnboundedContinuous(torch.Size([1]), device=self.device),
            "approach_speed": UnboundedContinuous(torch.Size([1]), device=self.device),
            "success_rate": UnboundedContinuous(torch.Size([1]), device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.info_spec = Composite({
            "drone_state": UnboundedContinuous(torch.Size([1, 13]), device=self.device),
            # "prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
            # "policy_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
            # "prev_prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
        }).expand(self.num_envs).to(self.device)
        # self.info_spec = self.pursuer.info_spec.to(self.device)

        self.observation_spec["stats"] = self.stats_spec
        self.observation_spec["info"] = self.info_spec

        self.stats = self.stats_spec.zero()
        self.info = self.info_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset the requested environments and randomize both drones."""
        # Reset ALL stats (including success_rate) for the new episode. The
        # value from the previous episode has already been picked up by
        # EpisodeStats at its terminal step (we latch success_rate inside
        # _compute_state_and_obs, before the stats are cloned into the obs).
        self.stats[env_ids] = 0.0

        # Reset the drones
        self.pursuer._reset_idx(env_ids, self.training)
        self.evader._reset_idx(env_ids, self.training)

        pursuer_pos = self.pursuer_init_pos_dist.sample(
            torch.Size([len(env_ids), 1]))
        pursuer_rpy = self.pursuer_init_rpy_dist.sample(
            torch.Size([len(env_ids), 1]))
        pursuer_rot = euler_to_quaternion(pursuer_rpy)

        pursuer_yaw = pursuer_rpy[..., 2]
        yaw_noise = torch.randn_like(pursuer_yaw) * 0.5
        pursuer_yaw = pursuer_yaw + yaw_noise

        spawn_direction = normalize(torch.stack([
            torch.cos(pursuer_yaw),
            torch.sin(pursuer_yaw),
            torch.rand((len(env_ids), 1), device=self.device)
        ], dim=-1))
        spawn_direction[..., 2] = spawn_direction[..., 2].abs()
        spawn_direction = normalize(spawn_direction)

        spawn_distance = self.evader_spawn_distance_dist.sample(
            torch.Size([len(env_ids), 1, 1]))

        evader_pos = pursuer_pos + spawn_direction * spawn_distance  # [len(env_ids), 3]

        # Sample a horizontal direction; the trajectory keeps the spawn altitude.
        line_dir = normalize(torch.randn(
            len(env_ids), 1, 3, device=self.device))
        line_dir[..., 2] = line_dir[..., 2].abs()  # Make sure the evader moves upwards or horizontally, not downwards, to avoid collisions with the ground plane.
        # line_dir = normalize(line_dir)
        self.evader_line_dir[env_ids] = line_dir
        speed = self.evader_cfg.get("speed", None)
        if speed is None:
            line_speed = self.evader_speed_dist.sample(
                torch.Size([len(env_ids), 1, 1]))
        else:
            line_speed = torch.full(
                (len(env_ids), 1, 1), float(speed), device=self.device)
        self.evader_line_speed[env_ids] = line_speed

        # Sample trajectory type per reset env from the enabled list.
        n = len(env_ids)
        sel = torch.randint(
            0, len(self._evader_enabled_codes_tensor),
            (n, 1), device=self.device)
        self.evader_traj_type[env_ids] = self._evader_enabled_codes_tensor[sel.squeeze(-1)].unsqueeze(-1)

        # Sample zig-zag amplitude / frequency for the reset envs (used only
        # by envs whose traj type is zigzag).
        self.evader_zigzag_amp[env_ids] = self.evader_zigzag_amp_dist.sample(
            torch.Size([n, 1, 1]))
        self.evader_zigzag_freq[env_ids] = self.evader_zigzag_freq_dist.sample(
            torch.Size([n, 1, 1]))

        # Perpendicular axis for the lateral oscillation. Prefer horizontal
        # (cross with world up); fall back to cross with world x if line_dir
        # is nearly vertical.
        world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(line_dir)
        perp = torch.cross(line_dir, world_up, dim=-1)
        perp_norm = torch.norm(perp, dim=-1, keepdim=True)
        world_x = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand_as(line_dir)
        perp_fallback = torch.cross(line_dir, world_x, dim=-1)
        perp = torch.where(perp_norm < 1e-3, perp_fallback, perp)
        self.evader_zigzag_perp[env_ids] = normalize(perp)

        evader_heading = normalize(torch.randn(
            len(env_ids), 1, 3, device=self.device))
        evader_heading[..., 2] = 0.0
        evader_heading = normalize(evader_heading)
        evader_yaw = torch.atan2(
            evader_heading[..., 1], evader_heading[..., 0])
        evader_rot = euler_to_quaternion(torch.stack([
            torch.zeros_like(evader_yaw),
            torch.zeros_like(evader_yaw),
            evader_yaw,
        ], dim=-1))  # [len(env_ids), 1, 4]

        self.pursuer_local_pos[env_ids] = pursuer_pos
        self.pursuer_local_rot[env_ids] = pursuer_rot
        self.pursuer_local_vel[env_ids] = torch.zeros(
            len(env_ids), 1, 6, device=self.device)
        self.evader_local_pos[env_ids] = evader_pos
        self.evader_local_rot[env_ids] = evader_rot
        self.evader_local_vel[env_ids] = torch.zeros(
            len(env_ids), 1, 6, device=self.device)
        # self.evader_target_pos[env_ids] = evader_pos + \
        #     self.envs_positions[env_ids]
        # self.evader_target_yaw[env_ids] = evader_yaw.unsqueeze(-1)

        self.pursuer.set_world_poses(
            self.envs_positions[env_ids].unsqueeze(
                1) + self.pursuer_local_pos[env_ids],
            self.pursuer_local_rot[env_ids],
            env_ids,
        )
        self.pursuer.set_velocities(self.pursuer_local_vel[env_ids], env_ids)

        self.evader.set_world_poses(
            self.envs_positions[env_ids].unsqueeze(
                1) + self.evader_local_pos[env_ids],
            self.evader_local_rot[env_ids],
            env_ids,
        )
        self.evader.set_velocities(self.evader_local_vel[env_ids], env_ids)

        # Reset previous-action buffer used for the smoothness reward.
        self.prev_action[env_ids] = 0.0

        # Seed prev_distance with the actual spawn distance so the first-step
        # delta-distance reward contribution is ~0.
        spawn_distance_vec = torch.norm(
            self.evader_local_pos[env_ids] - self.pursuer_local_pos[env_ids],
            dim=-1,
        )  # [n, 1]
        self.prev_distance[env_ids] = spawn_distance_vec

    def _pre_sim_step(self, tensordict: TensorDictBase):
        """Apply the pursuer policy action and update the evader trajectory before stepping the simulation."""
        actions = self._format_action(tensordict[("agents", "action")])

        # First-order action error (norm of action delta). Matches the
        # quantity that PIDRateController publishes for Track.
        self.action_error_order1 = torch.norm(
            actions - self.prev_action, dim=-1
        )  # [num_envs, 1]
        self.prev_action = actions.clone()

        self.pursuer_effort = self.pursuer.apply_action(actions)

        evader_action = self._compute_evader_action(tensordict)
        evader_action = self._format_action(evader_action)
        self.evader.apply_action(evader_action)

    def _format_action(self, action: torch.Tensor) -> torch.Tensor:
        """Clamp an action tensor and reshape it to the batched controller format."""
        action = torch.nan_to_num(action, 0.0).clamp(-1.0, 1.0)
        return action.reshape(self.num_envs, 1, -1)

    def _compute_evader_action(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Compute the evader action based on the current trajectory mode and target."""
        evader_state = self.evader.get_state()[..., :13].squeeze(1)

        t = self.progress_buf.float() * self.cfg.sim.dt  # [num_envs]
        start = self.evader_local_pos.squeeze(1)  # [num_envs, 3]
        direction = self.evader_line_dir.squeeze(1)  # [num_envs, 3]
        speed = self.evader_line_speed.squeeze(1).squeeze(-1)  # [num_envs]
        displacement = direction * (speed * t).unsqueeze(-1)
        pos = start + displacement

        # Add zig-zag lateral offset for envs whose trajectory type is zigzag.
        perp = self.evader_zigzag_perp.squeeze(1)  # [num_envs, 3]
        amp = self.evader_zigzag_amp.squeeze(1).squeeze(-1)  # [num_envs]
        freq = self.evader_zigzag_freq.squeeze(1).squeeze(-1)  # [num_envs]
        lateral = perp * (amp * torch.sin(2 * torch.pi * freq * t)).unsqueeze(-1)
        is_zigzag = (self.evader_traj_type.squeeze(-1) == 1).unsqueeze(-1).float()
        pos = pos + lateral * is_zigzag

        yaw = torch.atan2(direction[..., 1], direction[..., 0])

        return self.evader_controller.compute(evader_state, target_pos=pos, target_yaw=yaw)

    def _compute_state_and_obs(self):
        """Build the observation tensor and diagnostic state payloads."""
        pursuer_root_state = self.pursuer.get_state()  # [num_envs, 1, state_dim]
        evader_root_state = self.evader.get_state()  # [num_envs, 1, state_dim]

        pursuer_pos = pursuer_root_state[..., :3]
        pursuer_alt = pursuer_root_state[..., 2]
        pursuer_rot_quat = pursuer_root_state[..., 3:7]
        pursuer_vel = pursuer_root_state[..., 7:13]

        evader_pos = evader_root_state[..., :3]
        evader_rot_quat = evader_root_state[..., 3:7]
        evader_vel = evader_root_state[..., 7:13]

        self.evader_rel_hdg = normalize(evader_pos - pursuer_pos)
        # TODO: Change computation to based on how the evader is seen from the pursuer camera frame instead of just the world frame
        self.evader_rel_lin_vel = evader_vel[..., :3] - pursuer_vel[..., :3]
        self.pursuer_pos = pursuer_pos
        self.pursuer_alt = pursuer_alt
        self.pursuer_lin_vel = pursuer_vel[..., :3]
        self.pursuer_rot_vel = pursuer_vel[..., 3:6]
        self.pursuer_rot = quaternion_to_rotation_matrix(pursuer_rot_quat).reshape(
            self.num_envs, 1, 9)

        obs = [
            self.evader_rel_hdg,
            self.pursuer_lin_vel,
            self.pursuer_rot,
        ]
        if self.pursuer_use_ab_world_frame:
            obs.append(self.pursuer_pos)
        else:
            obs.append(self.pursuer_pos[..., 2:3])  # insert altitude only

        if self.evader_use_relative_velocity:
            obs.append(self.evader_rel_lin_vel)

        if self.pursuer_use_rot_speed:
            obs.append(self.pursuer_rot_vel)

        state = obs.copy()

        if self.time_encoding_dim:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            state.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))

        obs = torch.cat(obs, dim=-1)
        state = torch.cat(state, dim=-1)

        self.info["drone_state"][:] = pursuer_root_state[..., :13]

        # Latch success here (BEFORE cloning self.stats into the obs) so that
        # the terminal-step snapshot consumed by EpisodeStats already contains
        # the success bit. IsaacEnv._step calls _compute_state_and_obs before
        # _compute_reward_and_done, so latching success inside the reward
        # function would be one step too late: the terminal step's `next.stats`
        # would always show 0.
        distance = torch.norm(evader_pos - pursuer_pos, dim=-1)  # [num_envs, 1]
        reached_target = (distance <= self.success_radius).float()  # [num_envs, 1]
        self.stats["success_rate"][:] = torch.maximum(
            self.stats["success_rate"], reached_target
        )

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "state": state,
                },
                "stats": self.stats.clone(),
                "info": self.info.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        """Compute reward terms and episode termination flags."""
        pursuer_pos, _ = self.pursuer.get_world_poses(True)
        evader_pos, _ = self.evader.get_world_poses(True)
        pursuer_vel = self.pursuer.get_velocities(True)
        pursuer_rot = self.pursuer.get_world_poses(True)[1]

        evader_velocity = self.evader.get_velocities(True)
        pursuer_pos = self._squeeze_batch(pursuer_pos)
        evader_pos = self._squeeze_batch(evader_pos)
        pursuer_vel = self._squeeze_batch(pursuer_vel)
        pursuer_rot = self._squeeze_batch(pursuer_rot)
        evader_velocity = self._squeeze_batch(evader_velocity)
        pursuer_state = torch.cat(
            [pursuer_pos, pursuer_rot, pursuer_vel], dim=-1)
        distance = torch.norm(evader_pos - pursuer_pos, dim=-1, keepdim=True)

        ######
        # Dense reward shaping
        ######
        reward_distance = self._reward_distance_to_evader(
            pursuer_pos, evader_pos)
        reward_alignment = self._reward_align_velocity_to_heading(
            pursuer_vel[..., :3], evader_pos - pursuer_pos
        )
        reward_approach_velocity = self._reward_approach_velocity_to_evader(
            pursuer_vel[..., :3], evader_pos - pursuer_pos
        )
        reward_time_to_intercept = self._reward_intercept_time(
            pursuer_pos, pursuer_vel, evader_pos, evader_velocity
        )
        reward_action_smoothness = self._reward_action_smoothness()
        reward_heading_alignment = self._reward_heading_alignment()

        reward_delta_distance = self._reward_delta_distance(distance)
        self.prev_distance = distance.detach().clone()

        # reward = 0.4 * reward_distance + 0.3 * reward_alignment + 0.3 * reward_approach_velocity
        # reward = (success_interception_reward + reward_time_to_intercept) / 2.0
        # reward = (reward_approach_velocity + reward_distance) / 2.0
        # reward = reward_approach_velocity + reward_action_smoothness + reward_heading_alignment
        # reward = (2.0 * reward - 1.0).clamp(-1.0, 1.0)
        reward = reward_delta_distance
        # reward = reward_approach_velocity + reward_heading_alignment + reward_delta_distance

        ######
        # Terminal reward
        ######
        reached_target = (distance <= self.success_radius).reshape(
            self.num_envs, 1)
        misbehave = (
            (pursuer_state[..., 2:3] < 0.15)
            | torch.isnan(pursuer_state).any(-1, keepdim=True)
            | (distance > self.reset_thres)
        ).reshape(self.num_envs, 1)

        terminated = misbehave | reached_target
        truncated = (self.progress_buf >= self.max_episode_length - 1)
        truncated = truncated.reshape(self.num_envs, 1)
        done_mask = terminated | truncated

        terminal_reward = torch.where(
            reached_target,
            torch.ones_like(reward),
            torch.where(misbehave, -torch.ones_like(reward),
                        torch.zeros_like(reward)),
        )

        reward += terminal_reward

        approach_speed = (pursuer_vel[..., :3] * normalize(evader_pos - pursuer_pos)).sum(
            dim=-1, keepdim=True
        )

        self.stats["distance"].lerp_(distance, 1 - self.alpha)
        self.stats["reward_closing"].lerp_(reward_distance, 1 - self.alpha)
        self.stats["reward_approach_speed"].lerp_(reward_approach_velocity, 1 - self.alpha)
        self.stats["reward_action_smoothness"].lerp_(
            reward_action_smoothness, 1 - self.alpha)
        self.stats["reward_heading_alignment"].lerp_(
            reward_heading_alignment, 1 - self.alpha)
        self.stats["reward_delta_distance"].lerp_(
            reward_delta_distance, 1 - self.alpha)
        self.stats["action_error_order1_mean"].lerp_(
            self.action_error_order1, 1 - self.alpha)
        self.stats["action_error_order1_max"].set_(
            torch.max(self.stats["action_error_order1_max"], self.action_error_order1)
        )
        self.stats["approach_speed"].lerp_(approach_speed, 1 - self.alpha)
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done": done_mask,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )

    def _squeeze_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        """Remove a singleton middle dimension from batched simulator tensors."""
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            return tensor.squeeze(1)
        return tensor

    def _reward_distance_to_evader(
        self,
        pursuer_pos: torch.Tensor,
        evader_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Reward the pursuer for reducing the distance to the evader."""
        distance = torch.norm(evader_pos - pursuer_pos, dim=-1, keepdim=True)
        return torch.exp(-self.reward_distance_scale * distance)

    def _reward_intercept_time(
        self,
        pursuer_pos: torch.Tensor,
        pursuer_vel: torch.Tensor,
        evader_pos: torch.Tensor,
        evader_vel: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate time-to-intercept from the current relative motion."""
        relative_pos = evader_pos - pursuer_pos
        relative_vel = evader_vel[..., :3] - pursuer_vel[..., :3]
        relative_speed = torch.norm(relative_vel, dim=-1, keepdim=True) + 1e-6
        time_to_intercept = (relative_pos * normalize(relative_vel)).sum(
            dim=-1, keepdim=True) / relative_speed
        return (-time_to_intercept / self.max_episode_length).clamp(-1.0, 0.0)

    def _reward_align_velocity_to_heading(
        self,
        pursuer_velocity: torch.Tensor,
        pursuer_to_evader_heading: torch.Tensor,
    ) -> torch.Tensor:
        """Reward the pursuer for moving in the direction of the evader."""
        pursuer_velocity_direction = normalize(pursuer_velocity)
        evader_heading_direction = normalize(pursuer_to_evader_heading)
        cosine_similarity = (pursuer_velocity_direction * evader_heading_direction).sum(
            dim=-1, keepdim=True
        )
        return ((cosine_similarity + 1.0) / 2.0).clamp(0.0, 1.0)

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
        reward = self.reward_approach_velocity_weight * normalized_speed.clamp(-1.0, 1.0)
        return reward

    def _reward_heading_alignment(self) -> torch.Tensor:
        """Reward the pursuer for having its forward direction aligned with the line of sight to the evader."""
        pursuer_forward = self.pursuer_rot[..., 0:3].reshape(self.num_envs, 3)
        relative_heading = self.evader_rel_hdg.reshape(self.num_envs, 3)

        heading_cos = (pursuer_forward * relative_heading).sum(dim=-1, keepdim=True)
        reward_heading_alignment = self.reward_heading_alignment_weight * (
            (heading_cos + 1.0) / 2.0
        ).clamp(0.0, 1.0)
        return reward_heading_alignment

    def _reward_action_smoothness(self) -> torch.Tensor:
        """Reward the pursuer for keeping consecutive actions close to each other.

        Uses ``exp(-||action_t - action_{t-1}||)`` weighted by
        ``reward_action_smoothness_weight``. Gated to zero on the first two
        steps of an episode, when ``prev_action`` is still the reset zero.
        """
        not_begin_flag = (self.progress_buf > 1).unsqueeze(1).float()
        return (
            self.reward_action_smoothness_weight
            * torch.exp(-self.action_error_order1)
            * not_begin_flag
        )

    def _reward_delta_distance(self, distance: torch.Tensor) -> torch.Tensor:
        """Reward step-over-step closing of the pursuer-to-evader distance.

        Positive when the current distance is smaller than the previous step
        (pursuer closing), negative when growing. Zeroed on the first step of
        each episode to avoid using a stale ``prev_distance``.
        """
        delta_distance = self.prev_distance - distance
        first_step_mask = (self.progress_buf == 0).unsqueeze(-1).float()
        return (
            self.reward_delta_distance_weight
            * delta_distance
            * (1.0 - first_step_mask)
        )
