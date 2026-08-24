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


import math

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
    quat_rotate_inverse,
)


class Intercept(IsaacEnv):
    r"""Pursuit-evasion task: a pursuer drone chases an evader drone.

    Observation: altitude (1), rotation matrix (9), body-frame lin vel (3),
    body-frame rot vel (3), relative heading (3)
    Optional observation: world-frame position (3) in place of altitude,
    body-frame evader rel lin vel (3), previous action (4)

    State: same as observation, plus optional time encoding (time-to-go or step count)

    Active reward: ``delta_distance + precision + terminal``. Additional terms are
    computed and logged in ``stats`` but not summed into the training signal.
    """

    def __init__(self, cfg, headless):
        self.cfg = cfg

        self.success_radius_init = cfg.task.get("success_radius_init", 0.3)
        self.success_radius_end = cfg.task.get("success_radius_end", 0.1)
        self.success_radius_lr = cfg.task.get("success_radius_lr", 0.0005)
        self.success_radius_eval = cfg.task.get("success_radius_eval", 0.1)  # fixed for eval
        self.global_step = 0  # cross-episode curriculum step
        self.success_radius = self.success_radius_init
        self._update_success_radius()

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
        self.reward_terminal_precision_weight = cfg.task.get(
            "reward_terminal_precision_weight", 150.0)
        self.reward_terminal_precision_scale = cfg.task.get(
            "reward_terminal_precision_scale", 5.0)
        # +weight when the evader is inside the pursuer camera FOV, -weight when
        # it is outside (symmetric; magnitude tuned by this single weight).
        self.reward_fov_weight = cfg.task.get("reward_fov_weight", 1.0)

        self.pursuer_cfg = cfg.task.pursuer
        self.evader_cfg = cfg.task.evader
        self.obs_cfg = cfg.task.observation

        self.pursuer_model_name = self.pursuer_cfg.get("model", "Hummingbird")
        self.pursuer_controller_name = self.pursuer_cfg.get(
            "controller", "RateController")
        self.pursuer_target_speed = self.pursuer_cfg.get("target_speed", 3.0)

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
        self.pursuer_initial_velocity_cfg = self.pursuer_cfg.get(
            "initial_velocity", {}
        )

        # Optional forward-facing RGB camera on the pursuer (see _design_scene).
        self.camera_cfg = self.pursuer_cfg.get("camera", {})
        self.use_camera = bool(self.camera_cfg.get("enabled", False))
        self.pursuer_camera = None
        self.pursuer_camera_frames = None
        self.evader_initial_velocity_cfg = self.evader_cfg.get(
            "initial_velocity", {}
        )

        self.obs_include_noise = self.obs_cfg.get("include_noise", False)
        self.obs_include_previous_action = self.obs_cfg.get("include_previous_action", False)
        self.obs_use_world_frame_pos = self.obs_cfg.get("use_world_frame_pos", False)
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
            "hover": 0,
            "linear": 1,
            "random": 2,
            "frpn": 3,
            "apf": 4,
        }

        enabled = list(self.evader_cfg.get("trajectory_types", ["linear"]))
        unknown = [t for t in enabled if t not in self._traj_type_codes]
        if unknown:
            raise ValueError(
                f"Unknown evader trajectory_types: {unknown}. "
                f"Supported: {list(self._traj_type_codes)}")
        self.evader_enabled_traj_codes = [self._traj_type_codes[t] for t in enabled]

        random_cfg = self.evader_cfg.get("random", {})
        self.evader_random_turn_interval_range = list(
            random_cfg.get("turn_interval_range", [20, 80]))
        self.evader_random_vertical_component_range = list(
            random_cfg.get("vertical_component_range", [-0.2, 0.2]))
        self.evader_random_target_lookahead = float(
            random_cfg.get("target_lookahead", 0.5))

        frpn_cfg = self.evader_cfg.get("frpn", {})
        self.evader_frpn_gain = float(frpn_cfg.get("gain", 5.0))
        self.evader_frpn_position_blend = float(
            frpn_cfg.get("position_blend", 0.2))
        self.evader_frpn_min_relative_speed = float(
            frpn_cfg.get("min_relative_speed", 0.1))
        self.evader_frpn_max_time_to_go = float(
            frpn_cfg.get("max_time_to_go", 2.0))
        self.evader_frpn_target_lookahead = float(
            frpn_cfg.get("target_lookahead", 0.5))
        self.evader_frpn_max_speed = float(frpn_cfg.get("max_speed", 3.0))

        apf_cfg = self.evader_cfg.get("apf", {})
        self.evader_apf_pursuer_gain = float(apf_cfg.get("pursuer_gain", 1.0))
        self.evader_apf_pursuer_power = float(apf_cfg.get("pursuer_power", 3.0))
        self.evader_apf_containment_radius = float(
            apf_cfg.get("containment_radius", 3.5))
        self.evader_apf_containment_gain = float(
            apf_cfg.get("containment_gain", 1.0))
        self.evader_apf_target_lookahead = float(
            apf_cfg.get("target_lookahead", 0.5))
        self.evader_apf_max_speed = float(apf_cfg.get("max_speed", 3.0))

        super().__init__(cfg, headless)

        self.pursuer.initialize()
        self.evader.initialize()

        # The camera prim is spawned under the template env in _design_scene and
        # cloned across envs; attach render products now that the sim is reset.
        if self.use_camera:
            self.pursuer_camera.initialize(
                f"/World/envs/env_.*/{self.pursuer.name}_0/base_link/Camera"
            )

        # Precompute the camera frame (in the pursuer body frame) for the
        # geometric FOV reward. Derived purely from the mount config, so it is
        # available whether or not the render camera is enabled.
        self._setup_fov_geometry()

        # Domain randomization for the RL-controlled pursuer only.
        randomization = self.cfg.task.get("randomization", {})
        if "pursuer" in randomization:
            self.pursuer.setup_randomization(randomization["pursuer"])

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
        self.pursuer_initial_velocity_dist = D.Uniform(
            torch.tensor(
                self.pursuer_initial_velocity_cfg.get(
                    "min", [-0.5, -0.5, -0.5, -0.2, -0.2, -0.2]
                ),
                device=self.device,
            ),
            torch.tensor(
                self.pursuer_initial_velocity_cfg.get(
                    "max", [0.5, 0.5, 0.5, 0.2, 0.2, 0.2]
                ),
                device=self.device,
            ),
        )
        self.evader_initial_velocity_dist = D.Uniform(
            torch.tensor(
                self.evader_initial_velocity_cfg.get(
                    "min", [-0.5, -0.5, -0.5, -0.2, -0.2, -0.2]
                ),
                device=self.device,
            ),
            torch.tensor(
                self.evader_initial_velocity_cfg.get(
                    "max", [0.5, 0.5, 0.5, 0.2, 0.2, 0.2]
                ),
                device=self.device,
            ),
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

        # Per-env trajectory type code params
        self.evader_traj_type = torch.zeros(
            self.num_envs, 1, device=self.device, dtype=torch.long)
        self.evader_random_dir = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.evader_random_next_turn_step = torch.zeros(
            self.num_envs, 1, device=self.device, dtype=torch.long)
        self.prev_distance = torch.zeros(
            self.num_envs, 1, device=self.device)  # for delta-distance reward

        self.alpha = 0.8

        self.evader_rel_hdg = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.evader_rel_distance = torch.zeros(
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

        self.pursuer.spawn(translations=[(0.0, 0.0, 1.0)])
        self.evader.spawn(translations=[(5.0, 0.0, 1.0)])

        if self.use_camera:
            self._spawn_pursuer_camera()

        return ["/World/defaultGroundPlane"]

    def _spawn_pursuer_camera(self):
        """Mount a forward-facing RGB camera on the pursuer's base_link.

        The prim is created under the template env (env_0); the GridCloner
        duplicates it into every env during IsaacEnv setup.
        """
        from omni_drones.sensors.camera import Camera, PinholeCameraCfg

        resolution = tuple(self.camera_cfg.get("resolution", [320, 320]))
        fov = float(self.camera_cfg.get("fov", 87.0))
        fps = float(self.camera_cfg.get("fps", 30.0))
        translation = tuple(self.camera_cfg.get("translation", [0.028, 0.0, 0.014]))
        rotation = tuple(self.camera_cfg.get("rotation", [0.0, 0.0, 0.0]))

        # Convert FOV to focal length while keeping the default USD sensor width
        # (aperture): fov = 2*atan(aperture / (2*focal)). Resolution is square,
        # so USD derives an equal vertical aperture -> vertical FOV == fov too.
        horizontal_aperture = 20.955
        focal_length = horizontal_aperture / (
            2.0 * math.tan(math.radians(fov) / 2.0)
        )

        camera_cfg = PinholeCameraCfg(
            sensor_tick=1.0 / fps,
            resolution=resolution,
            data_types=["rgb"],
            usd_params=PinholeCameraCfg.UsdCameraCfg(
                focal_length=focal_length,
                horizontal_aperture=horizontal_aperture,
                clipping_range=(0.1, 1.0e5),
            ),
        )
        self.pursuer_camera = Camera(camera_cfg)

        # Aim direction: rotate the default forward (+x) body axis by the
        # configured [roll, pitch, yaw] and point the look-at target that way.
        # Note: Camera.spawn uses a world-up look-at, so pitch/yaw fully steer
        # the camera while roll (rotation about the view axis) is not applied.
        rotation_quat = euler_to_quaternion(
            torch.tensor([math.radians(a) for a in rotation])
        )
        forward = quaternion_to_rotation_matrix(rotation_quat)[..., 0]  # R @ [1,0,0]
        forward = forward.reshape(3).tolist()
        target = (
            translation[0] + forward[0],
            translation[1] + forward[1],
            translation[2] + forward[2],
        )
        self.pursuer_camera.spawn(
            [f"/World/envs/env_0/{self.pursuer.name}_0/base_link/Camera"],
            translations=[translation],
            targets=[target],
        )

    def _setup_fov_geometry(self):
        """Build the pursuer camera frame in the body frame for the FOV reward.

        The camera mount (offset + orientation) is fixed relative to the
        pursuer's base_link, so its optical axis and image right/up axes are
        constant in the body frame. The look-at spawn uses a world-up
        reference, so we reconstruct the same orthonormal frame here.
        """
        self.camera_fov = float(self.camera_cfg.get("fov", 87.0))
        self.camera_half_fov = math.radians(self.camera_fov / 2.0)

        translation = torch.tensor(
            self.camera_cfg.get("translation", [0.028, 0.0, 0.014]),
            device=self.device, dtype=torch.float,
        )
        rotation = self.camera_cfg.get("rotation", [0.0, 0.0, 0.0])
        rot_quat = euler_to_quaternion(
            torch.tensor(
                [math.radians(a) for a in rotation],
                device=self.device, dtype=torch.float,
            )
        )
        # Optical axis: the drone's forward (+x) body axis rotated by the mount.
        forward = normalize(quaternion_to_rotation_matrix(rot_quat)[..., 0].reshape(3))

        # Image right/up axes from a world-up look-at (fall back near the poles).
        up_ref = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        if torch.abs((forward * up_ref).sum()) > 0.999:
            up_ref = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        right = normalize(torch.cross(forward, up_ref, dim=-1))
        cam_up = torch.cross(right, forward, dim=-1)

        # Store as [1, 1, 3] so they broadcast over [num_envs, 1, 3] rel vectors.
        self.camera_offset_body = translation.reshape(1, 1, 3)
        self.camera_forward_body = forward.reshape(1, 1, 3)
        self.camera_right_body = right.reshape(1, 1, 3)
        self.camera_up_body = cam_up.reshape(1, 1, 3)

    def _set_specs(self):
        """Define observation, action, reward, and stats specs."""
        pursuer_state_dim = 9 + 3 + 3  # rot matrix + lin vel + rot vel

        if self.obs_use_world_frame_pos:
            pursuer_state_dim += 3  # absolute position in world frame
        else:
            pursuer_state_dim += 1  # altitude only

        evader_state_dim = 3  # relative heading

        if self.obs_include_evader_rel_lin_vel:
            evader_state_dim += 3  # relative evader linear velocity

        obs_dim = pursuer_state_dim + evader_state_dim

        if self.obs_include_previous_action:
            obs_dim += self.pursuer.action_spec.shape[-1]  # previous action

        if self.time_encoding_dim:
            state_dim = obs_dim + self.time_encoding_dim
        else:
            state_dim = obs_dim

        self.observation_spec = Composite({
            "agents": {
                "observation":  UnboundedContinuous(torch.Size([1, obs_dim])),
                "observation_h": UnboundedContinuous(torch.Size([1, obs_dim, 32])),
                "state": UnboundedContinuous(torch.Size([1, state_dim])),
                "intrinsics": self.pursuer.intrinsics_spec_flattened.unsqueeze(0)
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = Composite({
            "agents": {
                "action": self.pursuer.action_spec.unsqueeze(0),  # CTBR (1, 4)
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
            state_key=("agents", "state"),
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
            "reward_fov": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_terminal": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_terminal_precision": UnboundedContinuous(torch.Size([1]), device=self.device),
            "action_error_order1_mean": UnboundedContinuous(torch.Size([1]), device=self.device),
            "action_error_order1_max": UnboundedContinuous(torch.Size([1]), device=self.device),
            "action_norm_mean": UnboundedContinuous(torch.Size([1]), device=self.device),
            "action_norm_max": UnboundedContinuous(torch.Size([1]), device=self.device),
            "approach_speed": UnboundedContinuous(torch.Size([1]), device=self.device),
            "success_radius": UnboundedContinuous(torch.Size([1]), device=self.device),
            "success_rate": UnboundedContinuous(torch.Size([1]), device=self.device),
            "target_thrust": UnboundedContinuous(torch.Size([1]), device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.info_spec = Composite({
            "drone_state": UnboundedContinuous(torch.Size([1, 13]), device=self.device),
            "prev_action": self.pursuer.action_spec.unsqueeze(0),
            "policy_action": self.pursuer.action_spec.unsqueeze(0),
            # "prev_prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
        }).expand(self.num_envs).to(self.device)

        self.observation_spec["stats"] = self.stats_spec
        self.observation_spec["info"] = self.info_spec
        self.observation_h = self.observation_spec[("agents", "observation_h")].zero()

        self.stats = self.stats_spec.zero()
        self.info = self.info_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.stats[env_ids] = 0.0  # reset stats; terminal step already captured
        num_env = len(env_ids)

        self.pursuer._reset_idx(env_ids, self.training)
        self.evader._reset_idx(env_ids, self.training)

        # Pursuer spawn location
        pursuer_pos = self.pursuer_init_pos_dist.sample(torch.Size([num_env, 1]))
        pursuer_rpy = self.pursuer_init_rpy_dist.sample(torch.Size([num_env, 1]))
        pursuer_rot = euler_to_quaternion(pursuer_rpy)

        # Evader spawn location
        spawn_direction = normalize(torch.randn(num_env, 1, 3, device=self.device))
        spawn_distance = self.evader_spawn_distance_dist.sample(torch.Size([num_env, 1, 1]))
        evader_pos = pursuer_pos + spawn_direction * spawn_distance  # [len(env_ids), 3]
        evader_pos[..., 2] = torch.clamp(
            evader_pos[..., 2],
            min=self.minimum_altitude,
        )

        # Select a random trajectory type for each env from the enabled set.
        sel = torch.randint(0, len(self._evader_enabled_codes_tensor), (num_env, 1), device=self.device)
        self.evader_traj_type[env_ids] = self._evader_enabled_codes_tensor[sel.squeeze(-1)].unsqueeze(-1)

        # Linear trajectory
        line_dir = normalize(torch.randn(num_env, 1, 3, device=self.device))
        line_dir[..., 2] = line_dir[..., 2].abs()  # keep evader aloft
        self.evader_line_dir[env_ids] = line_dir
        speed = self.evader_cfg.get("speed", None)
        if speed is None:
            line_speed = self.evader_speed_dist.sample(torch.Size([num_env, 1, 1]))
        else:
            line_speed = torch.full((num_env, 1, 1), float(speed), device=self.device)
        self.evader_line_speed[env_ids] = line_speed

        # Random trajectory
        self.evader_random_dir[env_ids] = self._sample_random_evader_direction(num_env)  # random trajectory state
        turn_steps = self._sample_random_turn_steps(num_env)
        self.evader_random_next_turn_step[env_ids] = (
            self.progress_buf[env_ids].unsqueeze(-1).long() + turn_steps
        )

        evader_heading = normalize(torch.randn(num_env, 1, 3, device=self.device))
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
        if self.pursuer_initial_velocity_cfg.get("enabled", False):
            self.pursuer_local_vel[env_ids] = self.pursuer_initial_velocity_dist.sample(
                torch.Size([num_env, 1])
            )
        else:
            self.pursuer_local_vel[env_ids] = torch.zeros(
                num_env, 1, 6, device=self.device)

        self.evader_local_pos[env_ids] = evader_pos
        self.evader_local_rot[env_ids] = evader_rot
        if self.evader_initial_velocity_cfg.get("enabled", False):
            self.evader_local_vel[env_ids] = self.evader_initial_velocity_dist.sample(
                torch.Size([num_env, 1])
            )
        else:
            self.evader_local_vel[env_ids] = torch.zeros(
                num_env, 1, 6, device=self.device)

        self.pursuer.set_world_poses(
            self.envs_positions[env_ids].unsqueeze(
                1) + self.pursuer_local_pos[env_ids],
            self.pursuer_local_rot[env_ids],
            env_ids,
        )
        self.evader.set_world_poses(
            self.envs_positions[env_ids].unsqueeze(
                1) + self.evader_local_pos[env_ids],
            self.evader_local_rot[env_ids],
            env_ids,
        )
        self.pursuer.set_velocities(self.pursuer_local_vel[env_ids], env_ids)
        self.evader.set_velocities(self.evader_local_vel[env_ids], env_ids)

        self.prev_action[env_ids] = torch.zeros_like(self.prev_action[env_ids])  # reset for smoothness reward

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

    def _post_sim_step(self, tensordict: TensorDictBase):
        """Latch the latest RGB frames from the pursuer camera, if enabled.

        Camera annotators only refresh when the sim renders (i.e. not during
        headless training), so the frames are captured here for visualization
        and offline use rather than fed into the policy observation.
        """
        if self.use_camera:
            # TensorDict of shape [num_cameras] with an "rgb" entry per camera.
            self.pursuer_camera_frames = self.pursuer_camera.get_images()

    def get_pursuer_images(self):
        """Return the most recent pursuer camera frames (None if disabled)."""
        return self.pursuer_camera_frames

    def _format_action(self, action: torch.Tensor) -> torch.Tensor:
        """Reshape action tensor to the batched controller format."""
        return action.reshape(self.num_envs, 1, -1)

    def _compute_evader_action(self, _tensordict: TensorDictBase) -> torch.Tensor:
        """Compute the evader action based on the current trajectory mode and target."""
        evader_state = self.evader.get_state()[..., :13].squeeze(1)
        pursuer_state = self.pursuer.get_state()[..., :13].squeeze(1)
        # Keep trajectory code strictly per-env [num_envs] to avoid accidental
        # broadcasting (e.g. [N, 1] masks with [N] values -> [N, N]).
        traj_type = self.evader_traj_type.reshape(self.num_envs)

        t = self.progress_buf.float() * self.cfg.sim.dt  # [num_envs]
        start = self.evader_local_pos.squeeze(1)  # [num_envs, 3]
        direction = self.evader_line_dir.squeeze(1)  # [num_envs, 3]
        speed = self.evader_line_speed.squeeze(1).squeeze(-1)  # [num_envs]
        yaw = torch.atan2(direction[..., 1], direction[..., 0])
        pos = start + direction * (speed * t).unsqueeze(-1)

        # Linear trajectory is the default. The position is clamped above the
        # floor before any mode-specific overrides are applied.
        pos[..., 2] = torch.clamp(pos[..., 2], min=self.minimum_altitude + 0.1)

        is_random = traj_type == self._traj_type_codes["random"]
        is_hover = traj_type == self._traj_type_codes["hover"]
        is_frpn = traj_type == self._traj_type_codes["frpn"]
        is_apf = traj_type == self._traj_type_codes["apf"]

        # Random trajectory: piecewise-constant heading with periodic resampling.
        if bool(is_random.any()):
            should_turn = is_random & (
                self.progress_buf
                >= self.evader_random_next_turn_step.squeeze(-1)
            )
            if bool(should_turn.any()):
                turn_idx = should_turn.nonzero(as_tuple=False).squeeze(-1)
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
            random_target[..., 2] = random_target[..., 2].clamp(min=self.minimum_altitude + 0.1)
            random_yaw = torch.atan2(random_dir[..., 1], random_dir[..., 0])

            pos = torch.where(is_random.unsqueeze(-1), random_target, pos)
            yaw = torch.where(is_random, random_yaw, yaw)

        # FRPN and APF trajectories use the pursuer's current state to choose
        # a target velocity, then give the position controller a short-horizon
        # position target in that direction.
        if bool((is_frpn | is_apf).any()):
            rel_pos = evader_state[..., :3] - pursuer_state[..., :3]
            rel_vel = evader_state[..., 7:10] - pursuer_state[..., 7:10]
            rel_distance = torch.norm(rel_pos, dim=-1, keepdim=True).clamp_min(1e-3)
            away = rel_pos / rel_distance

            # Fast-response proportional navigation, expressed from the
            # evader's perspective so the commanded direction escapes the
            # pursuer while responding to relative velocity.
            rel_speed = torch.norm(rel_vel, dim=-1, keepdim=True).clamp_min(
                self.evader_frpn_min_relative_speed)
            time_to_go = (rel_distance / rel_speed).clamp(
                max=self.evader_frpn_max_time_to_go)
            pn_term = (rel_pos + rel_vel * time_to_go) / time_to_go.square().clamp_min(1e-3)
            frpn_velocity = self.evader_frpn_gain * (
                self.evader_frpn_position_blend * rel_pos
                + (1.0 - self.evader_frpn_position_blend) * pn_term
            )
            frpn_velocity = self._limit_velocity(
                frpn_velocity, self.evader_frpn_max_speed)
            frpn_target = evader_state[..., :3] + (
                frpn_velocity * self.evader_frpn_target_lookahead
            )

            # Artificial potential field: repel the pursuer while attracting
            # the evader back toward its spawn neighborhood past the configured
            # radius. This avoids immediately ending an episode at reset_thres.
            pursuer_force = (
                self.evader_apf_pursuer_gain
                * away
                / rel_distance.pow(self.evader_apf_pursuer_power)
            )
            displacement = evader_state[..., :3] - self.evader_local_pos.squeeze(1)
            displacement_norm = torch.norm(
                displacement, dim=-1, keepdim=True).clamp_min(1e-3)
            containment_excess = (
                displacement_norm - self.evader_apf_containment_radius
            ).clamp_min(0.0)
            containment_force = (
                -self.evader_apf_containment_gain
                * containment_excess
                * displacement
                / displacement_norm
            )
            apf_velocity = self._limit_velocity(
                pursuer_force + containment_force,
                self.evader_apf_max_speed,
            )
            apf_target = evader_state[..., :3] + (
                apf_velocity * self.evader_apf_target_lookahead
            )

            frpn_target[..., 2] = frpn_target[..., 2].clamp(
                min=self.minimum_altitude + 0.1)
            apf_target[..., 2] = apf_target[..., 2].clamp(
                min=self.minimum_altitude + 0.1)
            frpn_yaw = torch.atan2(frpn_velocity[..., 1], frpn_velocity[..., 0])
            apf_yaw = torch.atan2(apf_velocity[..., 1], apf_velocity[..., 0])

            pos = torch.where(is_frpn.unsqueeze(-1), frpn_target, pos)
            yaw = torch.where(is_frpn, frpn_yaw, yaw)
            pos = torch.where(is_apf.unsqueeze(-1), apf_target, pos)
            yaw = torch.where(is_apf, apf_yaw, yaw)

        # Hover trajectory: evader stays in place.
        if bool(is_hover.any()):
            target_pos = self.evader_local_pos.clone().squeeze(1)
            target_pos[..., 2] = torch.clamp(target_pos[..., 2], min=self.minimum_altitude + 0.1)
            target_yaw = torch.atan2(direction[..., 1], direction[..., 0])
            pos = torch.where(is_hover.unsqueeze(-1), target_pos, pos)
            yaw = torch.where(is_hover, target_yaw, yaw)

        return self.evader_controller.compute(evader_state, target_pos=pos, target_yaw=yaw)

    def _compute_state_and_obs(self):
        """Build the observation tensor and diagnostic state payloads."""
        # get_state() is what refreshes the .pos/.rot/.vel_b/.vel_w buffers read below.
        pursuer_root_state = self.pursuer.get_state()  # [num_envs, 1, state_dim]
        self.evader.get_state()

        pursuer_pos = self.pursuer.pos
        pursuer_alt = self.pursuer.pos[..., 2:3]
        pursuer_rot_quat = self.pursuer.rot
        pursuer_vel_b = self.pursuer.vel_b  # [num_envs, 1, 6] pursuer body frame
        pursuer_vel_w = self.pursuer.vel_w  # [num_envs, 1, 6] world frame

        evader_pos = self.evader.pos
        evader_alt = self.evader.pos[..., 2:3]
        evader_rot_quat = self.evader.rot
        evader_vel_b = self.evader.vel_b  # [num_envs, 1, 6] evader body frame
        evader_vel_w = self.evader.vel_w  # [num_envs, 1, 6] world frame

        distance = evader_pos - pursuer_pos
        rel_distance = quat_rotate_inverse(pursuer_rot_quat, distance)
        self.evader_rel_distance = rel_distance
        self.evader_rel_hdg = normalize(rel_distance)  # [num_envs, 1, 3]
        self.evader_rel_lin_vel = quat_rotate_inverse(
            pursuer_rot_quat, evader_vel_w[..., :3] - pursuer_vel_w[..., :3]
        )
        self.pursuer_pos = pursuer_pos
        self.pursuer_alt = pursuer_alt
        self.pursuer_lin_vel = pursuer_vel_b[..., :3]
        self.pursuer_rot_vel = pursuer_vel_b[..., 3:6]
        self.pursuer_rot = quaternion_to_rotation_matrix(pursuer_rot_quat).reshape(
            self.num_envs, 1, 9)

        obs = []

        if self.obs_use_world_frame_pos:
            obs.append(self.pursuer_pos)
        else:
            obs.append(self.pursuer_alt)

        obs += [
            self.pursuer_rot,
            self.pursuer_lin_vel,
            self.pursuer_rot_vel,
            self.evader_rel_hdg,
            # self.evader_rel_distance,
        ]

        # observation noise for default observation space
        if self.obs_include_noise:
            # Position/altitude: additive Gaussian.
            noise_pos = obs[0] + torch.randn_like(
                obs[0]) * self.obs_noise_alt_std
            obs[0] = noise_pos

            # Flattened rotation matrix: small additive perturbation.
            noisy_rot = self.pursuer_rot + torch.randn_like(
                self.pursuer_rot) * self.obs_noise_rot_std
            obs[1] = noisy_rot

            # Body-frame linear velocity: additive Gaussian.
            noisy_lin_vel = self.pursuer_lin_vel + torch.randn_like(
                self.pursuer_lin_vel) * self.obs_noise_lin_vel_std
            obs[2] = noisy_lin_vel

            # Body-frame angular velocity: additive Gaussian.
            noisy_rot_vel = self.pursuer_rot_vel + torch.randn_like(
                self.pursuer_rot_vel) * self.obs_noise_rot_vel_std
            obs[3] = noisy_rot_vel

            # Relative heading: add noise then re-normalize (stays on sphere).
            noisy_hdg = self.evader_rel_hdg + torch.randn_like(
                self.evader_rel_hdg) * self.obs_noise_rel_hdg_std
            obs[4] = normalize(noisy_hdg)

        if self.obs_include_evader_rel_lin_vel:
            if self.obs_include_noise:
                noisy_rel_vel = self.evader_rel_lin_vel + torch.randn_like(
                    self.evader_rel_lin_vel) * self.obs_noise_rel_lin_vel_std
                obs.append(noisy_rel_vel)
            else:
                obs.append(self.evader_rel_lin_vel)

        # Previous action
        if self.obs_include_previous_action:
            obs.append(self.current_action)

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
        failed_to_reach_target = (
            (pursuer_pos[..., 2:3].reshape(self.num_envs, 1) < self.minimum_altitude)
            | (distance > self.reset_thres)
        ).float()  # [num_envs, 1]
        self.stats["reward_terminal"][:] = (
            self.reward_terminal_weight * reached_target
            - self.reward_terminal_weight * failed_to_reach_target
        )
        self.stats["reward_terminal_precision"][:] = torch.where(
            reached_target.bool(),
            self._reward_terminal_precision(distance).reshape(self.num_envs, 1),
            torch.zeros_like(reached_target),
        )

        intrinsics_dict = self.pursuer.intrinsics
        intrinsics_flat = torch.cat([
            intrinsics_dict["mass"],
            intrinsics_dict["inertia"],
            intrinsics_dict["com"],
            intrinsics_dict["KF"],
            intrinsics_dict["KM"],
            intrinsics_dict["tau_up"],
            intrinsics_dict["tau_down"],
            intrinsics_dict["drag_coef"],
        ], dim=-1)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "state": state,
                    "intrinsics": intrinsics_flat,
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
        # reward_distance = self._reward_distance_to_evader(
        #     pursuer_pos, evader_pos)
        # reward_alignment = self._reward_align_velocity_to_heading(
        #     pursuer_vel[..., :3], evader_pos - pursuer_pos
        # )
        # reward_approach_velocity = self._reward_approach_velocity_to_evader(
        #     pursuer_vel[..., :3], evader_pos - pursuer_pos
        # )
        # reward_time_to_intercept = self._reward_intercept_time(
        #     pursuer_pos, pursuer_vel, evader_pos, evader_velocity
        # )
        # reward_heading_alignment = self._reward_heading_alignment()

        reward_delta_distance = self._reward_delta_distance(distance)
        reward_precision = self._reward_precision(distance)
        reward_action_smoothness = self._reward_action_smoothness()
        reward_fov = self._reward_fov()

        self.prev_distance = distance.detach().clone()

        action_norm = torch.norm(self.current_action, dim=-1)
        # reward_action_norm = self._reward_action_norm()

        # reward = reward_delta_distance + reward_precision + reward_action_smoothness + reward_fov
        reward = reward_delta_distance + reward_precision + reward_action_smoothness

        # Terminal rewards.
        reached_target = (distance <= self.active_success_radius).reshape(
            self.num_envs, 1)
        failed_to_reach_target = (
            (distance > self.reset_thres)
            | (pursuer_state[..., 2:3] < self.minimum_altitude)
        ).reshape(self.num_envs, 1)
        misbehave = torch.isnan(pursuer_state).any(-1, keepdim=True).reshape(self.num_envs, 1)

        terminated = misbehave | reached_target | failed_to_reach_target
        truncated = (self.progress_buf >= self.max_episode_length - 1)
        truncated = truncated.reshape(self.num_envs, 1)
        done_mask = terminated | truncated

        terminal_success_reward = torch.where(
            reached_target,
            self.reward_terminal_weight * torch.ones_like(reward),
            torch.zeros_like(reward),
        )
        terminal_failure_reward = torch.where(
            failed_to_reach_target,
            -self.reward_terminal_weight * torch.ones_like(reward),
            torch.zeros_like(reward),
        )
        terminal_precision_reward = torch.where(
            reached_target,
            self._reward_terminal_precision(distance).reshape_as(reward),
            torch.zeros_like(reward),
        )
        # terminal_reward = terminal_success_reward + terminal_failure_reward + terminal_precision_reward
        terminal_reward = terminal_success_reward + terminal_failure_reward

        reward += terminal_reward

        # self.stats["reward_closing"].lerp_(reward_distance, 1 - self.alpha)
        # self.stats["reward_approach_speed"].lerp_(reward_approach_velocity, 1 - self.alpha)
        self.stats["reward_action_smoothness"].lerp_(
            reward_action_smoothness, 1 - self.alpha)
        # self.stats["reward_action_norm"].lerp_(
        #     reward_action_norm, 1 - self.alpha)
        # self.stats["reward_heading_alignment"].lerp_(
        #     reward_heading_alignment, 1 - self.alpha)
        self.stats["reward_delta_distance"].lerp_(
            reward_delta_distance, 1 - self.alpha)
        self.stats["reward_precision"].lerp_(
            reward_precision, 1 - self.alpha)
        self.stats["reward_fov"].lerp_(
            reward_fov, 1 - self.alpha)
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
        self.stats["target_thrust"][:] = self.current_action[..., -1]

        # Advance global curriculum once per simulator step.
        self.global_step += 1
        self._update_success_radius()

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
        self.success_radius = max(radius, self.success_radius_end)

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

    def _reward_fov(self) -> torch.Tensor:
        """+weight inside the camera FOV; growing negative penalty outside.

        Purely geometric: projects the evader (relative to the camera) into the
        camera frame and checks the horizontal/vertical angles against the
        (symmetric) half-FOV. Inside the FOV the reward is the constant
        +weight. Outside, the penalty grows with the angular excess beyond the
        FOV boundary, from ~0 just outside the edge to -weight when the evader
        is directly behind the camera. Works headless, no rendered image
        required.
        """
        # Evader displacement relative to the camera, in the pursuer body frame.
        rel = self.evader_rel_distance - self.camera_offset_body  # [num_envs, 1, 3]

        depth = (rel * self.camera_forward_body).sum(-1)   # along optical axis
        horiz = (rel * self.camera_right_body).sum(-1)     # image right
        vert = (rel * self.camera_up_body).sum(-1)         # image up

        ang_h = torch.atan2(horiz, depth)
        ang_v = torch.atan2(vert, depth)
        in_view = (
            (depth > 0)
            & (ang_h.abs() <= self.camera_half_fov)
            & (ang_v.abs() <= self.camera_half_fov)
        )  # [num_envs, 1]

        # Angular excess beyond the FOV edge (0 when inside). When the evader
        # is behind the camera the atan2 angles are unreliable, so use the
        # maximum possible excess (directly behind the optical axis).
        max_excess = math.pi - self.camera_half_fov
        excess_h = (ang_h.abs() - self.camera_half_fov).clamp(min=0.0)
        excess_v = (ang_v.abs() - self.camera_half_fov).clamp(min=0.0)
        excess = torch.maximum(excess_h, excess_v)
        excess = torch.where(
            depth > 0, excess, torch.full_like(excess, max_excess)
        )
        penalty = excess / max_excess  # 0 at FOV boundary -> 1 directly behind

        return self.reward_fov_weight * torch.where(
            in_view, torch.ones_like(depth), -penalty
        )  # [num_envs, 1]

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
        """-||a_t - a_{t-1}||, gated to zero on first two steps."""
        not_begin_flag = (self.progress_buf > 1).unsqueeze(1).float()
        return (
            self.reward_action_smoothness_weight
            * -self.action_error_order1
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

    def _reward_terminal_precision(self, terminal_distance: torch.Tensor) -> torch.Tensor:
        """Terminal reward based on final distance to evader."""
        return self.reward_terminal_precision_weight * torch.exp(
            -self.reward_terminal_precision_scale * terminal_distance
        )

    def _sample_random_evader_direction(self, n: int) -> torch.Tensor:
        """Sample normalized random directions with bounded vertical component."""
        random_dir = normalize(torch.randn(n, 1, 3, device=self.device))
        random_dir[..., 2] = torch.empty(n, 1, device=self.device).uniform_(
            float(self.evader_random_vertical_component_range[0]),
            float(self.evader_random_vertical_component_range[1]),
        )
        return normalize(random_dir)

    @staticmethod
    def _limit_velocity(velocity: torch.Tensor, max_speed: float) -> torch.Tensor:
        """Cap vector magnitude while preserving direction."""
        speed = torch.norm(velocity, dim=-1, keepdim=True)
        return velocity * (max_speed / speed.clamp_min(max_speed))

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
