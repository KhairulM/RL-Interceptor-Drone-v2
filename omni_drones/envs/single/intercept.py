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
from omni_drones.planners.velocity_obstacle import VelocityObstaclePlanner
from omni_drones.utils.torch import (
    euler_to_quaternion,
    normalize,
    quaternion_to_rotation_matrix,
    quat_rotate_inverse,
)


class Intercept(IsaacEnv):
    r"""Pursuit-evasion task: a pursuer drone chases a scripted evader drone.

    Observation: relative heading (3), body-frame lin vel (3), rotation matrix (9),
    altitude (1), optional evader rel lin vel (3), world pos (3), rot vel (3), time encoding.

    Active reward: ``delta_distance + precision + terminal``. Additional terms are
    computed and logged in ``stats`` but not summed into the training signal.
    """

    def __init__(self, cfg, headless):
        self.cfg = cfg

        self.success_radius_init = cfg.task.get("success_radius_init", 0.3)
        self.success_radius_max = cfg.task.get("success_radius_max", 0.1)
        self.success_radius_lr = cfg.task.get("success_radius_lr", 0.0005)
        self.success_radius_eval = cfg.task.get("success_radius_eval", 0.15)  # fixed for eval
        self.global_step = 0  # cross-episode curriculum step
        self.success_radius = self.success_radius_init
        self._update_success_radius()

        vo_cfg = cfg.task.evader.get("vo_curriculum", {})  # VO curriculum params
        vo_th = vo_cfg.get("time_horizon", {"init": 3.0, "max": 3.0, "lr": 0.0})
        vo_ir = vo_cfg.get("inflation_radius", {"init": 0.1, "max": 0.1, "lr": 0.0})
        vo_mv = vo_cfg.get("max_velocity", {"init": 10.0, "max": 10.0, "lr": 0.0})
        vo_nd = vo_cfg.get("n_directions", {"init": 64, "max": 64, "lr": 0.0})

        self.vo_time_horizon_init = float(vo_th["init"])
        self.vo_time_horizon_max = float(vo_th["max"])
        self.vo_time_horizon_lr = float(vo_th["lr"])

        self.vo_inflation_radius_init = float(vo_ir["init"])
        self.vo_inflation_radius_max = float(vo_ir["max"])
        self.vo_inflation_radius_lr = float(vo_ir["lr"])

        self.vo_max_velocity_init = float(vo_mv["init"])
        self.vo_max_velocity_max = float(vo_mv["max"])
        self.vo_max_velocity_lr = float(vo_mv["lr"])

        self.vo_n_directions_init = int(vo_nd["init"])
        self.vo_n_directions_max = int(vo_nd["max"])
        self.vo_n_directions_lr = float(vo_nd["lr"])

        # Current curriculum values.
        self.vo_time_horizon = self.vo_time_horizon_init
        self.vo_inflation_radius = self.vo_inflation_radius_init
        self.vo_max_velocity = self.vo_max_velocity_init
        self.vo_n_directions = self.vo_n_directions_init

        # Fixed eval-time values (hardest setting).
        self.vo_time_horizon_eval = float(max(vo_th["init"], vo_th["max"]))
        self.vo_inflation_radius_eval = float(max(vo_ir["init"], vo_ir["max"]))
        self.vo_max_velocity_eval = float(max(vo_mv["init"], vo_mv["max"]))
        self.vo_n_directions_eval = int(max(vo_nd["init"], vo_nd["max"]))

        # Last-applied n_directions to avoid rebuilding candidates every step.
        self._last_vo_n_directions = None

        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.reset_thres = cfg.task.get("reset_thres", 9.0)
        self.minimum_altitude = cfg.task.get("minimum_altitude", 0.15)

        self.time_encoding_dim = cfg.task.get("time_encoding_dim", 0)

        self.reward_action_smoothness_weight = cfg.task.get(
            "reward_action_smoothness_weight", 1.0)
        self.reward_heading_alignment_weight = cfg.task.get(
            "reward_heading_alignment_weight", 1.0)
        self.reward_approach_velocity_weight = cfg.task.get(
            "reward_approach_velocity_weight", 1.0)
        self.reward_delta_distance_weight = cfg.task.get(
            "reward_delta_distance_weight", 10.0)
        self.reward_precision_weight = cfg.task.get(
            "reward_precision_weight", 10.0)
        self.reward_precision_scale = cfg.task.get(
            "reward_precision_scale", 5.0)
        self.reward_terminal_weight = cfg.task.get(
            "reward_terminal_weight", 10.0)
        self.reward_action_norm_weight = cfg.task.get(
            "reward_action_norm_weight", 1.0)

        self.pursuer_cfg = cfg.task.pursuer
        self.evader_cfg = cfg.task.evader
        self.obs_cfg = cfg.task.observation

        self.pursuer_model_name = self.pursuer_cfg.get("model", "Hummingbird")
        self.pursuer_controller_name = self.pursuer_cfg.get(
            "controller", "RateController")
        self.pursuer_target_speed = self.pursuer_cfg.get("target_speed", 15.0)

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

        self.obs_use_world_frame_pos = self.obs_cfg.get("use_world_frame_pos", False)
        self.obs_include_rot_speed = self.obs_cfg.get("include_rot_speed", False)
        self.obs_include_noise = self.obs_cfg.get("include_noise", False)
        self.obs_include_evader_rel_lin_vel = self.obs_cfg.get(
            "include_evader_rel_lin_vel", False)

        # Per-component Gaussian noise stds (sensor-matched defaults).
        noise_cfg = self.obs_cfg.get("noise", {})
        self.obs_noise_rel_hdg_std = float(noise_cfg.get("rel_hdg_std", 0.02))      # ~1° angular
        self.obs_noise_lin_vel_std = float(noise_cfg.get("lin_vel_std", 0.05))     # ~5 cm/s (IMU)
        self.obs_noise_rot_std = float(noise_cfg.get("rot_std", 0.01))             # ~1° orientation
        self.obs_noise_alt_std = float(noise_cfg.get("alt_std", 0.02))             # ~2 cm barometer
        self.obs_noise_rel_lin_vel_std = float(noise_cfg.get("rel_lin_vel_std", 0.08))  # derived
        self.obs_noise_rot_vel_std = float(noise_cfg.get("rot_vel_std", 0.02))    # ~2°/s gyro

        self._traj_type_codes = {  # codes for _compute_evader_action
            "linear": 0,
            "zigzag": 1,
            "velocity_obstacle": 2,
            "random": 3,
            "hover": 4,
        }

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

        random_cfg = self.evader_cfg.get("random", {})
        self.evader_random_turn_interval_range = list(
            random_cfg.get("turn_interval_range", [20, 80]))
        self.evader_random_vertical_component_range = list(
            random_cfg.get("vertical_component_range", [-0.2, 0.2]))
        self.evader_random_target_lookahead = float(
            random_cfg.get("target_lookahead", 0.5))

        super().__init__(cfg, headless)

        self.pursuer.initialize()
        self.evader.initialize()

        # Domain randomization for the RL-controlled pursuer only.
        randomization = self.cfg.task.get("randomization", {})
        if "pursuer" in randomization:
            self.pursuer.setup_randomization(randomization["pursuer"])

        # VO planner for the "velocity_obstacle" evader trajectory type.
        self.vo_planner = VelocityObstaclePlanner(
            self.evader.params, max_n_directions=self.vo_n_directions_max).to(self.device)

        self._apply_vo_params_to_planner()  # apply initial curriculum values

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

        # Buffers for local drone states relative to spawn positions.
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
        # Per-env trajectory type code and zig-zag params.
        self.evader_traj_type = torch.zeros(
            self.num_envs, 1, device=self.device, dtype=torch.long)
        self.evader_zigzag_amp = torch.zeros(
            self.num_envs, 1, 1, device=self.device)
        self.evader_zigzag_freq = torch.zeros(
            self.num_envs, 1, 1, device=self.device)
        self.evader_zigzag_perp = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.evader_random_dir = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.evader_random_next_turn_step = torch.zeros(
            self.num_envs, 1, device=self.device, dtype=torch.long)
        self.prev_distance = torch.zeros(
            self.num_envs, 1, device=self.device)  # for delta-distance reward

        self.alpha = 0.8

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

        # Buffers for action rewards
        self.current_action = torch.zeros(
            self.num_envs, 1, self.pursuer.action_spec.shape[-1],
            device=self.device,
        )
        self.prev_action = torch.zeros(
            self.num_envs, 1, self.pursuer.action_spec.shape[-1],
            device=self.device,
        )
        self.action_error_order1 = torch.zeros(
            self.num_envs, 1, device=self.device)

    def _design_scene(self):
        """Create pursuer, evader, and ground plane."""
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

    def _set_specs(self):
        """Define observation, action, reward, and stats specs."""
        evader_state_dim = 3  # relative heading

        if self.obs_include_evader_rel_lin_vel:
            evader_state_dim += 3  # relative linear velocity

        pursuer_state_dim = 3 + 9 + 1  # lin vel + rot matrix + altitude

        if self.obs_use_world_frame_pos:
            pursuer_state_dim += 3  # absolute position in world frame
        if self.obs_include_rot_speed:
            pursuer_state_dim += 3  # angular velocity

        obs_dim = evader_state_dim + pursuer_state_dim

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
                "action": self.pursuer.action_spec.unsqueeze(0),  # pursuer motor commands
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
            "reward_alignment": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_approach_speed": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_time_to_intercept": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_action_smoothness": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_action_norm": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_heading_alignment": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_delta_distance": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_precision": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_terminal": UnboundedContinuous(torch.Size([1]), device=self.device),
            "action_error_order1_mean": UnboundedContinuous(torch.Size([1]), device=self.device),
            "action_error_order1_max": UnboundedContinuous(torch.Size([1]), device=self.device),
            "action_norm_mean": UnboundedContinuous(torch.Size([1]), device=self.device),
            "action_norm_max": UnboundedContinuous(torch.Size([1]), device=self.device),
            "approach_speed": UnboundedContinuous(torch.Size([1]), device=self.device),
            "success_radius": UnboundedContinuous(torch.Size([1]), device=self.device),
            "success_rate": UnboundedContinuous(torch.Size([1]), device=self.device),
            "vo_time_horizon": UnboundedContinuous(torch.Size([1]), device=self.device),  # VO curriculum
            "vo_inflation_radius": UnboundedContinuous(torch.Size([1]), device=self.device),
            "vo_max_velocity": UnboundedContinuous(torch.Size([1]), device=self.device),
            "vo_n_directions": UnboundedContinuous(torch.Size([1]), device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.info_spec = Composite({
            "drone_state": UnboundedContinuous(torch.Size([1, 13]), device=self.device),
            "prev_action": self.pursuer.action_spec.unsqueeze(0),
            "policy_action": self.pursuer.action_spec.unsqueeze(0),
            # "prev_prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
        }).expand(self.num_envs).to(self.device)

        self.observation_spec["stats"] = self.stats_spec
        self.observation_spec["info"] = self.info_spec

        self.stats = self.stats_spec.zero()
        self.info = self.info_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.stats[env_ids] = 0.0  # reset stats; terminal step already captured
        n = len(env_ids)

        self.pursuer._reset_idx(env_ids, self.training)
        self.evader._reset_idx(env_ids, self.training)

        pursuer_pos = self.pursuer_init_pos_dist.sample(torch.Size([n, 1]))
        pursuer_rpy = self.pursuer_init_rpy_dist.sample(torch.Size([n, 1]))
        pursuer_rot = euler_to_quaternion(pursuer_rpy)

        pursuer_yaw = pursuer_rpy[..., 2]
        yaw_noise = torch.randn_like(pursuer_yaw) * 0.5
        pursuer_yaw = pursuer_yaw + yaw_noise

        spawn_direction = normalize(torch.stack([
            torch.cos(pursuer_yaw),
            torch.sin(pursuer_yaw),
            torch.rand((n, 1), device=self.device)
        ], dim=-1))
        spawn_direction[..., 2] = spawn_direction[..., 2].abs()
        spawn_direction = normalize(spawn_direction)

        spawn_distance = self.evader_spawn_distance_dist.sample(torch.Size([n, 1, 1]))

        evader_pos = pursuer_pos + spawn_direction * spawn_distance  # [len(env_ids), 3]

        line_dir = normalize(torch.randn(n, 1, 3, device=self.device))
        line_dir[..., 2] = line_dir[..., 2].abs()  # keep evader aloft
        self.evader_line_dir[env_ids] = line_dir
        speed = self.evader_cfg.get("speed", None)
        if speed is None:
            line_speed = self.evader_speed_dist.sample(torch.Size([n, 1, 1]))
        else:
            line_speed = torch.full((n, 1, 1), float(speed), device=self.device)
        self.evader_line_speed[env_ids] = line_speed

        sel = torch.randint(0, len(self._evader_enabled_codes_tensor), (n, 1), device=self.device)
        self.evader_traj_type[env_ids] = self._evader_enabled_codes_tensor[sel.squeeze(-1)].unsqueeze(-1)

        self.evader_random_dir[env_ids] = self._sample_random_evader_direction(n)  # random trajectory state
        turn_steps = self._sample_random_turn_steps(n)
        self.evader_random_next_turn_step[env_ids] = (
            self.progress_buf[env_ids].unsqueeze(-1).long() + turn_steps
        )

        # Zig-zag params for reset envs.
        self.evader_zigzag_amp[env_ids] = self.evader_zigzag_amp_dist.sample(
            torch.Size([n, 1, 1]))
        self.evader_zigzag_freq[env_ids] = self.evader_zigzag_freq_dist.sample(
            torch.Size([n, 1, 1]))

        # Perpendicular axis for zig-zag oscillation; fall back if line_dir is vertical.
        world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(line_dir)
        perp = torch.cross(line_dir, world_up, dim=-1)
        perp_norm = torch.norm(perp, dim=-1, keepdim=True)
        world_x = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand_as(line_dir)
        perp_fallback = torch.cross(line_dir, world_x, dim=-1)
        perp = torch.where(perp_norm < 1e-3, perp_fallback, perp)
        self.evader_zigzag_perp[env_ids] = normalize(perp)

        evader_heading = normalize(torch.randn(n, 1, 3, device=self.device))
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
            n, 1, 6, device=self.device)
        self.evader_local_pos[env_ids] = evader_pos
        self.evader_local_rot[env_ids] = evader_rot
        self.evader_local_vel[env_ids] = torch.zeros(
            n, 1, 6, device=self.device)
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

        self.prev_action[env_ids] = 0.0  # reset for smoothness reward

        spawn_distance_vec = torch.norm(
            self.evader_local_pos[env_ids] - self.pursuer_local_pos[env_ids], dim=-1,
        )  # [n, 1]
        self.prev_distance[env_ids] = spawn_distance_vec  # seed so first-step delta ~0

    def _pre_sim_step(self, tensordict: TensorDictBase):
        """Apply the pursuer policy action and update the evader trajectory before stepping the simulation."""
        self.current_action = tensordict[("info", "policy_action")]
        self.prev_action = tensordict[("info", "prev_action")]
        self.action_error_order1 = tensordict[("stats", "action_error_order1")]  # these are set on the PIDRateController

        pursuer_action = self._format_action(tensordict[("agents", "action")])  # this action is the motor commands, not the policy action
        self.pursuer_effort = self.pursuer.apply_action(pursuer_action)

        evader_action = self._compute_evader_action(tensordict)  # this action is the motor commands, not ctbr or target position
        evader_action = self._format_action(evader_action)
        self.evader.apply_action(evader_action)

    def _format_action(self, action: torch.Tensor) -> torch.Tensor:
        """Reshape action tensor to the batched controller format."""
        return action.reshape(self.num_envs, 1, -1)

    def _sample_random_evader_direction(self, n: int) -> torch.Tensor:
        """Sample normalized random directions with bounded vertical component."""
        random_dir = normalize(torch.randn(n, 1, 3, device=self.device))
        random_dir[..., 2] = torch.empty(n, 1, device=self.device).uniform_(
            float(self.evader_random_vertical_component_range[0]),
            float(self.evader_random_vertical_component_range[1]),
        )
        return normalize(random_dir)

    def _sample_random_turn_steps(self, n: int) -> torch.Tensor:
        """Sample per-env turn intervals for the random trajectory mode."""
        min_turn, max_turn = self.evader_random_turn_interval_range
        min_turn = max(1, int(min_turn))
        max_turn = max(min_turn, int(max_turn))
        return torch.randint(
            min_turn,
            max_turn + 1,
            (n, 1),
            device=self.device,
            dtype=torch.long,
        )

    def _compute_evader_action(self, _tensordict: TensorDictBase) -> torch.Tensor:
        """Compute the evader action based on the current trajectory mode and target."""
        evader_state = self.evader.get_state()[..., :13].squeeze(1)
        traj_type = self.evader_traj_type.squeeze(-1)

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
        is_zigzag = (traj_type == self._traj_type_codes["zigzag"]).unsqueeze(-1).float()
        pos = pos + lateral * is_zigzag

        # VO override: replace open-loop target with planner output.
        is_vo = (traj_type == self._traj_type_codes["velocity_obstacle"])
        if bool(is_vo.any()):
            evader_full = self.evader.get_state()[..., :13]  # [num_envs, 1, 13]
            pursuer_full = self.pursuer.get_state()[..., :13]  # [num_envs, 1, 13]
            pref_velocity = self.evader_line_dir * self.evader_line_speed  # [num_envs, 1, 3]
            vo_out = self.vo_planner.plan(evader_full, pursuer_full, pref_velocity)
            # Planner output is in the same frame as its inputs (world frame
            # here); convert to env-local to match the linear/zigzag target.
            vo_pos_local = vo_out["desired_position"].squeeze(1) - self.envs_positions
            pos = torch.where(is_vo.unsqueeze(-1), vo_pos_local, pos)

        yaw = torch.atan2(direction[..., 1], direction[..., 0])

        # Random trajectory: piecewise-constant heading with periodic resampling.
        is_random = (traj_type == self._traj_type_codes["random"]).unsqueeze(-1)
        if bool(is_random.any()):
            should_turn = is_random & (
                self.progress_buf.unsqueeze(-1)
                >= self.evader_random_next_turn_step
            )
            if bool(should_turn.any()):
                turn_idx = should_turn.squeeze(-1).nonzero(as_tuple=False).squeeze(-1)
                num_turn = int(turn_idx.numel())

                self.evader_random_dir[turn_idx] = self._sample_random_evader_direction(num_turn)
                turn_steps = self._sample_random_turn_steps(num_turn)
                self.evader_random_next_turn_step[turn_idx] = (
                    self.progress_buf[turn_idx].unsqueeze(-1).long() + turn_steps
                )

            random_dir = self.evader_random_dir.squeeze(1)  # [num_envs, 3]
            random_speed = speed.unsqueeze(-1)  # [num_envs, 1]
            random_target = evader_state[..., :3] + (
                random_dir
                * random_speed
                * self.evader_random_target_lookahead
            )
            random_target[..., 2] = random_target[..., 2].clamp(min=0.3)
            random_yaw = torch.atan2(random_dir[..., 1], random_dir[..., 0])

            pos = torch.where(is_random, random_target, pos)
            yaw = torch.where(is_random.squeeze(-1), random_yaw, yaw)

        # Hover trajectory: evader stays in place.
        is_hover = (traj_type == self._traj_type_codes["hover"])
        if bool(is_hover.any()):
            target_pos = self.evader_local_pos.squeeze(1)
            target_yaw = torch.atan2(direction[..., 1], direction[..., 0])
            pos = torch.where(is_hover.unsqueeze(-1), target_pos, pos)
            yaw = torch.where(is_hover, target_yaw, yaw)

        return self.evader_controller.compute(evader_state, target_pos=pos, target_yaw=yaw)

    def _compute_state_and_obs(self):
        """Build the observation tensor and diagnostic state payloads."""
        pursuer_root_state = self.pursuer.get_state()  # [num_envs, 1, state_dim]
        evader_root_state = self.evader.get_state()  # [num_envs, 1, state_dim]

        pursuer_pos = pursuer_root_state[..., :3]
        pursuer_alt = pursuer_root_state[..., 2]
        pursuer_rot_quat = pursuer_root_state[..., 3:7]

        pursuer_vel_b = self.pursuer.vel_b  # [num_envs, 1, 6] body frame

        evader_pos = evader_root_state[..., :3]
        evader_rot_quat = evader_root_state[..., 3:7]
        evader_vel_w = evader_root_state[..., 7:13]  # world-frame velocity

        self.evader_rel_hdg = normalize(evader_pos - pursuer_pos)

        # Transform evader velocity to pursuer body frame so relative velocity
        # is also expressed in the pursuer's body frame for consistency.
        evader_vel_b = torch.cat([
            quat_rotate_inverse(pursuer_rot_quat, evader_vel_w[..., :3]),
            quat_rotate_inverse(pursuer_rot_quat, evader_vel_w[..., 3:6]),
        ], dim=-1)

        self.evader_rel_lin_vel = evader_vel_b[..., :3] - pursuer_vel_b[..., :3]
        self.pursuer_pos = pursuer_pos
        self.pursuer_alt = pursuer_alt
        self.pursuer_lin_vel = pursuer_vel_b[..., :3]
        self.pursuer_rot_vel = pursuer_vel_b[..., 3:6]
        self.pursuer_rot = quaternion_to_rotation_matrix(pursuer_rot_quat).reshape(
            self.num_envs, 1, 9)

        obs = [
            self.evader_rel_hdg,
            self.pursuer_lin_vel,
            self.pursuer_rot,
        ]
        if self.obs_use_world_frame_pos:
            obs.append(self.pursuer_pos)
        else:
            obs.append(self.pursuer_pos[..., 2:3])  # insert altitude only

        if self.obs_include_evader_rel_lin_vel:
            obs.append(self.evader_rel_lin_vel)

        if self.obs_include_rot_speed:
            obs.append(self.pursuer_rot_vel)

        # Apply per-component Gaussian noise BEFORE concatenation.
        if self.obs_include_noise:
            # Relative heading: add noise then re-normalize (stays on sphere).
            noisy_hdg = self.evader_rel_hdg + torch.randn_like(
                self.evader_rel_hdg) * self.obs_noise_rel_hdg_std
            obs[0] = normalize(noisy_hdg)

            # Body-frame linear velocity: additive Gaussian.
            noisy_lin_vel = self.pursuer_lin_vel + torch.randn_like(
                self.pursuer_lin_vel) * self.obs_noise_lin_vel_std
            obs[1] = noisy_lin_vel

            # Flattened rotation matrix: small additive perturbation.
            noisy_rot = self.pursuer_rot + torch.randn_like(
                self.pursuer_rot) * self.obs_noise_rot_std
            obs[2] = noisy_rot

            # Altitude or full position.
            noisy_alt = obs[3] + torch.randn_like(
                obs[3]) * self.obs_noise_alt_std
            obs[3] = noisy_alt

            # Optional: evader relative linear velocity.
            if self.obs_include_evader_rel_lin_vel:
                noisy_rel_vel = self.evader_rel_lin_vel + torch.randn_like(
                    self.evader_rel_lin_vel) * self.obs_noise_rel_lin_vel_std
                obs[4] = noisy_rel_vel

            # Optional: angular velocity (gyro noise).
            if self.obs_include_rot_speed:
                noisy_rot_vel = self.pursuer_rot_vel + torch.randn_like(
                    self.pursuer_rot_vel) * self.obs_noise_rot_vel_std
                obs_idx = 5 if self.obs_include_evader_rel_lin_vel else 4
                obs[obs_idx] = noisy_rot_vel

        state = obs.copy()

        if self.time_encoding_dim:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            state.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))

        obs = torch.cat(obs, dim=-1)
        state = torch.cat(state, dim=-1)

        self.info["drone_state"][:] = pursuer_root_state[..., :13]

        # Latch success before cloning self.stats into the obs. _step runs
        # _compute_state_and_obs before _compute_reward_and_done, so latching
        # in the reward fn would miss the terminal step's snapshot.
        distance = torch.norm(evader_pos - pursuer_pos, dim=-1)  # [num_envs, 1]
        reached_target = (distance <= self.active_success_radius).float()  # [num_envs, 1]
        self.stats["success_rate"][:] = torch.maximum(
            self.stats["success_rate"], reached_target
        )

        # Latch the terminal reward here for the same reason as success_rate:
        # it is nonzero only on the terminal step, whose snapshot is cloned
        # below (before _compute_reward_and_done runs).
        misbehave = (
            (pursuer_pos[..., 2:3].reshape(self.num_envs, 1) < 0.15)
            | torch.isnan(pursuer_root_state[..., :13]).any(-1)
            | (distance > self.reset_thres)
        ).float()  # [num_envs, 1]
        self.stats["reward_terminal"][:] = (
            self.reward_terminal_weight * reached_target
            - self.reward_terminal_weight * misbehave
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
        approach_speed = (pursuer_vel[..., :3] * normalize(evader_pos - pursuer_pos)).sum(
            dim=-1, keepdim=True
        )

        # Dense rewards.
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
        reward_precision = self._reward_precision(distance)
        self.prev_distance = distance.detach().clone()

        action_norm = torch.norm(self.current_action, dim=-1)
        reward_action_norm = self._reward_action_norm()

        reward = reward_delta_distance + reward_precision

        # Terminal rewards.
        reached_target = (distance <= self.active_success_radius).reshape(
            self.num_envs, 1)
        misbehave = (
            (pursuer_state[..., 2:3] < self.minimum_altitude)
            | torch.isnan(pursuer_state).any(-1, keepdim=True)
            | (distance > self.reset_thres)
        ).reshape(self.num_envs, 1)

        terminated = misbehave | reached_target
        truncated = (self.progress_buf >= self.max_episode_length - 1)
        truncated = truncated.reshape(self.num_envs, 1)
        done_mask = terminated | truncated

        terminal_success_reward = torch.where(
            reached_target,
            self.reward_terminal_weight * torch.ones_like(reward),
            torch.zeros_like(reward),
        )
        terminal_failure_reward = torch.where(
            misbehave,
            -self.reward_terminal_weight * torch.ones_like(reward),
            torch.zeros_like(reward),
        )
        terminal_reward = terminal_success_reward + terminal_failure_reward

        reward += terminal_reward

        self.stats["reward_closing"].lerp_(reward_distance, 1 - self.alpha)
        self.stats["reward_approach_speed"].lerp_(reward_approach_velocity, 1 - self.alpha)
        self.stats["reward_action_smoothness"].lerp_(
            reward_action_smoothness, 1 - self.alpha)
        self.stats["reward_action_norm"].lerp_(
            reward_action_norm, 1 - self.alpha)
        self.stats["reward_heading_alignment"].lerp_(
            reward_heading_alignment, 1 - self.alpha)
        self.stats["reward_delta_distance"].lerp_(
            reward_delta_distance, 1 - self.alpha)
        self.stats["reward_precision"].lerp_(
            reward_precision, 1 - self.alpha)
        self.stats["action_error_order1_mean"].lerp_(
            self.action_error_order1, 1 - self.alpha)
        self.stats["action_error_order1_max"].copy_(
            torch.max(self.stats["action_error_order1_max"], self.action_error_order1)
        )
        self.stats["action_norm_mean"].lerp_(action_norm, 1 - self.alpha)
        self.stats["action_norm_max"].copy_(
            torch.max(self.stats["action_norm_max"], action_norm)
        )
        self.stats["distance"].lerp_(distance, 1 - self.alpha)
        self.stats["approach_speed"].lerp_(approach_speed, 1 - self.alpha)
        self.stats["return"] += reward
        self.stats["success_radius"][:] = self.active_success_radius
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        # VO curriculum stats.
        self.stats["vo_time_horizon"][:] = self.vo_time_horizon
        self.stats["vo_inflation_radius"][:] = self.vo_inflation_radius
        self.stats["vo_max_velocity"][:] = self.vo_max_velocity
        self.stats["vo_n_directions"][:] = self.vo_n_directions

        # Advance global curriculum once per simulator step.
        self.global_step += 1
        self._update_success_radius()
        self._update_vo_curriculum()

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

    @property
    def active_success_radius(self) -> float:
        """Success radius in use: the fixed eval radius outside training,
        otherwise the current curriculum radius."""
        if not self.training:
            return self.success_radius_eval
        return self.success_radius

    def _update_success_radius(self):
        """Update a single global success radius from training-time progress."""
        radius = self.success_radius_init - self.success_radius_lr * float(
            self.global_step
        )
        self.success_radius = max(radius, self.success_radius_max)

    def _update_vo_curriculum(self):
        """Update VO planner parameters based on the global curriculum step.

        During training, each parameter increases linearly from its ``init``
        value toward its ``max`` bound, scaled by the respective ``lr`` (per
        global step).  During evaluation, fixed "hard" values are used so that
        eval is deterministic and challenging.
        """
        if not self.training:
            self.vo_time_horizon = self.vo_time_horizon_eval
            self.vo_inflation_radius = self.vo_inflation_radius_eval
            self.vo_max_velocity = self.vo_max_velocity_eval
            self.vo_n_directions = self.vo_n_directions_eval
        else:
            step = self.global_step
            self.vo_time_horizon = min(
                self.vo_time_horizon_init + self.vo_time_horizon_lr * float(step),
                self.vo_time_horizon_max)
            self.vo_inflation_radius = min(
                self.vo_inflation_radius_init + self.vo_inflation_radius_lr * float(step),
                self.vo_inflation_radius_max)
            self.vo_max_velocity = min(
                self.vo_max_velocity_init + self.vo_max_velocity_lr * float(step),
                self.vo_max_velocity_max)
            nd_val = self.vo_n_directions_init + int(self.vo_n_directions_lr * step)
            self.vo_n_directions = min(nd_val, self.vo_n_directions_max)

        self._apply_vo_params_to_planner()

    def _apply_vo_params_to_planner(self):
        drone_radius = self.evader.params["l"] + self.vo_inflation_radius  # base + inflation

        self.vo_planner.set_params(
            time_horizon=self.vo_time_horizon,
            drone_radius=drone_radius,
            max_velocity=self.vo_max_velocity,
        )

        if self.vo_n_directions != self._last_vo_n_directions:
            self.vo_planner._rebuild_candidates(self.vo_n_directions)
            self._last_vo_n_directions = self.vo_n_directions

    def _reward_distance_to_evader(
        self, pursuer_pos: torch.Tensor, evader_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Exponential decay on distance to evader."""
        distance = torch.norm(evader_pos - pursuer_pos, dim=-1, keepdim=True)
        return torch.exp(-self.reward_distance_scale * distance)

    def _reward_intercept_time(
        self,
        pursuer_pos: torch.Tensor, pursuer_vel: torch.Tensor,
        evader_pos: torch.Tensor, evader_vel: torch.Tensor,
    ) -> torch.Tensor:
        """Negative shaped reward from estimated time-to-intercept."""
        relative_pos = evader_pos - pursuer_pos
        relative_vel = evader_vel[..., :3] - pursuer_vel[..., :3]
        relative_speed = torch.norm(relative_vel, dim=-1, keepdim=True) + 1e-6
        time_to_intercept = (relative_pos * normalize(relative_vel)).sum(
            dim=-1, keepdim=True) / relative_speed
        return (-time_to_intercept / self.max_episode_length).clamp(-1.0, 0.0)

    def _reward_align_velocity_to_heading(
        self,
        pursuer_velocity: torch.Tensor, pursuer_to_evader_heading: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine similarity of velocity direction with heading to evader."""
        pursuer_velocity_direction = normalize(pursuer_velocity)
        evader_heading_direction = normalize(pursuer_to_evader_heading)
        cosine_similarity = (pursuer_velocity_direction * evader_heading_direction).sum(
            dim=-1, keepdim=True
        )
        return ((cosine_similarity + 1.0) / 2.0).clamp(0.0, 1.0)

    def _reward_approach_velocity_to_evader(
        self,
        pursuer_velocity: torch.Tensor, pursuer_to_evader_heading: torch.Tensor,
    ) -> torch.Tensor:
        """Forward progress toward evader at desired speed."""
        approach_speed = (pursuer_velocity * normalize(pursuer_to_evader_heading)).sum(
            dim=-1, keepdim=True
        )
        normalized_speed = approach_speed / self.pursuer_target_speed
        reward = self.reward_approach_velocity_weight * normalized_speed.clamp(-1.0, 1.0)
        return reward

    def _reward_heading_alignment(self) -> torch.Tensor:
        """Pursuer forward axis aligned with LOS to evader."""
        pursuer_forward = self.pursuer_rot[..., 0:3].reshape(self.num_envs, 3)
        relative_heading = self.evader_rel_hdg.reshape(self.num_envs, 3)

        heading_cos = (pursuer_forward * relative_heading).sum(dim=-1, keepdim=True)
        reward_heading_alignment = self.reward_heading_alignment_weight * (
            (heading_cos + 1.0) / 2.0
        ).clamp(0.0, 1.0)
        return reward_heading_alignment

    def _reward_action_smoothness(self) -> torch.Tensor:
        """exp(-||a_t - a_{t-1}||), gated to zero on first two steps."""
        not_begin_flag = (self.progress_buf > 1).unsqueeze(1).float()
        return (
            self.reward_action_smoothness_weight
            * torch.exp(-self.action_error_order1)
            * not_begin_flag
        )

    def _reward_action_norm(self) -> torch.Tensor:
        """Exponential decay on action magnitude, gated to zero on first two steps."""
        not_begin_flag = (self.progress_buf > 1).unsqueeze(1).float()
        return (
            self.reward_action_norm_weight
            * torch.exp(-torch.norm(self.current_action, dim=-1))
            * not_begin_flag
        )

    def _reward_precision(self, distance: torch.Tensor) -> torch.Tensor:
        """Potential-difference: rewards closing the gap, pays zero for orbiting."""
        potential = torch.exp(-self.reward_precision_scale * distance)
        prev_potential = torch.exp(
            -self.reward_precision_scale * self.prev_distance)
        first_step_mask = (self.progress_buf == 0).unsqueeze(-1).float()
        return (
            self.reward_precision_weight
            * (potential - prev_potential)
            * (1.0 - first_step_mask)
        )

    def _reward_delta_distance(self, distance: torch.Tensor) -> torch.Tensor:
        """Step-over-step distance reduction; zeroed on the first step."""
        delta_distance = self.prev_distance - distance
        first_step_mask = (self.progress_buf == 0).unsqueeze(-1).float()
        return (
            self.reward_delta_distance_weight
            * delta_distance
            * (1.0 - first_step_mask)
        )
