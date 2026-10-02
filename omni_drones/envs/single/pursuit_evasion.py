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

"""Population-based self-play pursuit-evasion task.

A single shared environment with a ``role`` flag: the RL agent is the ``role``
side (``pursuer`` or ``evader``) and the opponent is a frozen policy drawn from a
pool (RL checkpoints and/or scripted heuristics), assigned per-environment for
the whole run. Both drones share a symmetric observation layout so a frozen
policy trained as one role can be replayed as the opponent of the other.

The outer generational driver (``scripts/selfplay/train_selfplay.py``) alternates
which side trains, samples opponents from the policy database with a
temperature-scaled multi-metric softmax, and records matchup metrics
(win rate, time-to-capture, distance, motor effort) back into the database.
"""

import math

import torch
import torch.distributions as D
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Composite, UnboundedContinuous, Binary

import omni_drones.utils.kit as kit_utils
from omni_drones.controllers import ControllerBase
from omni_drones.controllers.intercept_baseline_common import GeometricCTBR
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.selfplay.opponents import FrozenPolicyBank
from omni_drones.utils.torch import (
    euler_to_quaternion,
    normalize,
    quaternion_to_rotation_matrix,
    quat_rotate_inverse,
)

PURSUER = "pursuer"
EVADER = "evader"


class PursuitEvasion(IsaacEnv):
    r"""1v1 pursuit-evasion with a frozen opponent pool.

    The ``role`` side is trained with RL; the opponent is stepped internally
    from frozen policies. Observation (per side, symmetric so a frozen policy
    replays as either role's opponent): own position (3), flattened rotation
    matrix (9), body-frame linear velocity (3), body-frame angular velocity (3),
    opponent relative distance in the agent body frame (3), and opponent
    relative linear velocity in the agent body frame (3).
    """

    def __init__(self, cfg, headless):
        self.cfg = cfg
        task_cfg = cfg.task
        self._headless = headless

        self.role = str(task_cfg.get("role", PURSUER)).lower()
        if self.role not in (PURSUER, EVADER):
            raise ValueError(f"role must be 'pursuer' or 'evader', got {self.role}")
        self.opp_role = EVADER if self.role == PURSUER else PURSUER

        # Arena (confined 3D box, env-local coordinates centred on env origin).
        arena = task_cfg.get("arena", {})
        self.arena_half_xy = float(arena.get("half_xy", 5.0))
        self.arena_z_min = float(arena.get("z_min", 0.5))
        self.arena_z_max = float(arena.get("z_max", 4.5))
        self.arena_visualize = bool(arena.get("visualize", False))

        # Capture-radius curriculum (shrinks the capture threshold during
        # training, like Intercept). Fixed at the eval radius outside training.
        self.success_radius_init = float(task_cfg.get("success_radius_init", 0.3))
        self.success_radius_end = float(task_cfg.get("success_radius_end", 0.1))
        self.success_radius_lr = float(task_cfg.get("success_radius_lr", 2e-6))
        self.success_radius_eval = float(task_cfg.get("success_radius_eval", 0.1))
        self.global_step = 0  # cross-episode curriculum step
        self.success_radius = self.success_radius_init
        self._update_success_radius()
        self.minimum_altitude = float(task_cfg.get("minimum_altitude", 0.1))

        # Reward weights (role picks the sign pattern).
        self.reward_approach_weight = float(task_cfg.get("reward_approach_weight", 10.0))
        self.reward_precision_weight = float(task_cfg.get("reward_precision_weight", 10.0))
        self.reward_precision_scale = float(task_cfg.get("reward_precision_scale", 5.0))
        self.reward_body_rate_weight = float(task_cfg.get("reward_body_rate_weight", 0.0005))
        # Penalty on the step-to-step action jerk (norm of the CTBR action delta).
        # Directly discourages the bang-bang command oscillation that tumbles the
        # drone on the real/CrazySim firmware rate loop.
        self.reward_action_smoothness_weight = float(
            task_cfg.get("reward_action_smoothness_weight", 0.0))
        # Penalty on the MAGNITUDE of the commanded body rate (|tanh action| of
        # the 3 rate channels, in [0, sqrt(3)]). This is the term the other two
        # miss: in the 50 Hz training sim the rate loop never fully realises a
        # saturated command within a step, so the actual-angular-velocity penalty
        # (reward_body_rate) barely fires, and a *sustained* saturated command is
        # ~free under the step-to-step jerk penalty (reward_action_smoothness).
        # The real / CrazySim firmware rate loop (~10x faster) tracks those crisp
        # saturated setpoints and tumbles. Penalising commanded magnitude teaches
        # the policy to chase with gentle rates that deploy cleanly.
        self.reward_cmd_rate_weight = float(
            task_cfg.get("reward_cmd_rate_weight", 0.0))
        # Single magnitude for every terminal outcome (capture / arena violation /
        # timeout), so no ending is a cheap escape from another.
        self.reward_terminal_weight = float(task_cfg.get(
            "reward_terminal_weight", task_cfg.get("reward_capture_weight", 100.0)))
        # Partial credit for *forcing* the opponent into the ground or out of the
        # arena, as a fraction of reward_terminal_weight. Strictly less than 1 so
        # an actual capture (pursuer) / surviving the episode (evader) stays the
        # best available outcome and the terminal bonus cannot be farmed by
        # standing off and waiting for the opponent to self-destruct.
        self.reward_forced_error_scale = float(
            task_cfg.get("reward_forced_error_scale", 0.25))
        self.reward_bounds_weight = float(task_cfg.get("reward_bounds_weight", 1.0))
        self.bounds_margin = float(task_cfg.get("bounds_margin", 1.0))
        self.reward_step = float(task_cfg.get("reward_step", 0.005))
        # Dense per-step distance penalty (pursuer only; matches Intercept's
        # _reward_distance_to_evader = reward_distance_weight * distance). With a
        # negative weight it penalizes being far from the evader. Disabled by
        # default via use_distance.
        self.use_distance = bool(task_cfg.get("use_distance", False))
        self.reward_distance_weight = float(task_cfg.get("reward_distance_weight", -0.1))

        self.obs_cfg = task_cfg.get("observation", {})
        self.include_obs_noise = bool(self.obs_cfg.get("include_noise", False))
        self.use_previous_action = bool(self.obs_cfg.get("use_previous_action", True))
        self.use_time_encoding = bool(self.obs_cfg.get("use_time_encoding", True))
        self.time_encoding_dim = int(self.obs_cfg.get("time_encoding_dim", 4))

        # Per-component Gaussian noise stds (sensor-matched defaults, from Intercept).
        noise_cfg = self.obs_cfg.get("noise", {})
        self.obs_noise_pos_std = float(noise_cfg.get("pos_std", 0.02))            # ~2 cm position
        self.obs_noise_rot_std = float(noise_cfg.get("rot_std", 0.01))            # ~1° orientation
        self.obs_noise_lin_vel_std = float(noise_cfg.get("lin_vel_std", 0.05))    # ~5 cm/s (IMU)
        self.obs_noise_rot_vel_std = float(noise_cfg.get("rot_vel_std", 0.02))    # ~2°/s gyro
        self.obs_noise_rel_dist_std = float(noise_cfg.get("rel_dist_std", 0.02))  # relative position
        self.obs_noise_rel_lin_vel_std = float(noise_cfg.get("rel_lin_vel_std", 0.08))  # relative vel

        # Drone models (both sides). PIDRateController for CTBR on both.
        self.pursuer_cfg = task_cfg.get("pursuer", {})
        self.evader_cfg = task_cfg.get("evader", {})
        self.pursuer_model_name = self.pursuer_cfg.get("model", "Crazyflie")
        self.evader_model_name = self.evader_cfg.get("model", "Crazyflie")

        # Heuristic opponent motion params.
        heur = task_cfg.get("heuristic", {})
        self.heur_circular_radius = float(heur.get("circular_radius", 1.5))
        self.heur_circular_omega = float(heur.get("circular_omega", 1.0))
        self.heur_speed = float(heur.get("speed", 2.0))
        self.heur_lookahead = float(heur.get("lookahead", 0.5))
        # Keep-out band the scripted evader respects: the wall-avoidance push
        # starts this far from a wall, and every heuristic target is clamped
        # into the arena shrunk by `heur_safe_margin`. Sized so the evader's
        # ~1.3 m/s flee speed (stopping distance ~0.15 m under the geometric
        # outer loop) can never carry it through a wall or into the floor.
        self.heur_wall_margin = float(heur.get("wall_margin", 1.5))
        self.heur_safe_margin = float(heur.get("safe_margin", 0.75))
        # Distance over which the flee step fades to zero as the drone
        # approaches the keep-in edge (arena_half - wall_margin). Sized so the
        # drone decelerates to a stop before the wall instead of overshooting it.
        self.heur_brake_zone = float(heur.get("brake_zone", 1.0))
        # Strength of the always-on inward push (as a fraction of the full flee
        # step), which curls the evader along the wall and returns it from the
        # edge; kept separate from the faded outward drive.
        self.heur_wall_gain = float(heur.get("wall_gain", 2.0))

        self.alpha = 0.8  # EMA for logged stats

        super().__init__(cfg, headless)

        self.pursuer.initialize()
        self.evader.initialize()

        # Domain randomization
        randomization = self.cfg.task.get("randomization", {})
        self.pursuer.setup_randomization(randomization)
        self.evader.setup_randomization(randomization)

        # Spawn distributions (env-local, inside the arena).
        self._build_spawn_dists()

        # State buffers.
        self.agent_local_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.opp_local_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Per-episode tie-break heading for the `flee` heuristic (used only when
        # the pursuer is almost directly above/below it, where the horizontal
        # flee direction is undefined).
        self.flee_fallback_dir = torch.zeros(self.num_envs, 1, 2, device=self.device)
        # "This env was just reset" latch, consumed by the opponent controllers.
        # The tensordict handed to _pre_sim_step is the *action* td and carries
        # no "done" key, so the opponent's rate-PID and the geometric outer
        # loop's altitude integrator both saw a constant False and never reset:
        # their integral state leaked straight across episode boundaries. The
        # env has to signal the reset itself.
        self._just_reset = torch.zeros(
            self.num_envs, 1, dtype=torch.bool, device=self.device)
        self.prev_distance = torch.zeros(self.num_envs, 1, device=self.device)
        self._distance = torch.zeros(self.num_envs, 1, device=self.device)
        self._captured = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        # A "crash" is hitting the ground (or a NaN state). Leaving the arena
        # through the walls/ceiling is NOT terminal anymore; only the dense
        # boundary penalty discourages it.
        self._agent_crash = torch.zeros(
            self.num_envs, 1, dtype=torch.bool, device=self.device)
        self._opp_crash = torch.zeros(
            self.num_envs, 1, dtype=torch.bool, device=self.device)
        # Leaving the arena (walls/ceiling): terminal loss for the leaver, but
        # NOT a win for the other drone.
        self._agent_oob = torch.zeros(
            self.num_envs, 1, dtype=torch.bool, device=self.device)
        self._opp_oob = torch.zeros(
            self.num_envs, 1, dtype=torch.bool, device=self.device)
        self._truncated = torch.zeros(
            self.num_envs, 1, dtype=torch.bool, device=self.device)
        self._terminated = torch.zeros(
            self.num_envs, 1, dtype=torch.bool, device=self.device)
        self._timeout = torch.zeros(
            self.num_envs, 1, dtype=torch.bool, device=self.device)
        self._win = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        self._loss = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        self._terminal_reward = torch.zeros(self.num_envs, 1, device=self.device)
        self.current_action = torch.zeros(
            self.num_envs, 1, self.agent_drone.action_spec.shape[-1], device=self.device)
        # The frozen opponent's own previous action (never the agent's).
        self.opp_prev_action = torch.zeros(
            self.num_envs, 1, self.agent_drone.action_spec.shape[-1], device=self.device)
        self.action_error_order1 = torch.zeros(self.num_envs, 1, device=self.device)
        self.agent_effort = torch.zeros(self.num_envs, 1, device=self.device)
        self._opp_obs = self.observation_spec[("agents", "observation")].zero()

        # Frozen opponent bank built from cfg (driver injects the pool/probs).
        self._build_opponent_bank(task_cfg)

        # Draw the confined arena boundary (visible only when not headless).
        self._init_arena_draw()

    # --------------------------------------------------------------- setup
    def _build_spawn_dists(self):
        margin = 0.5
        low = torch.tensor(
            [-self.arena_half_xy + margin, -self.arena_half_xy + margin,
             self.arena_z_min + margin],
            device=self.device)
        high = torch.tensor(
            [self.arena_half_xy - margin, self.arena_half_xy - margin,
             self.arena_z_max - margin],
            device=self.device)
        self.spawn_pos_dist = D.Uniform(low, high)
        rpy_min = torch.tensor([-0.1, -0.1, -1.0], device=self.device) * torch.pi
        rpy_max = torch.tensor([0.1, 0.1, 1.0], device=self.device) * torch.pi
        self.spawn_rpy_dist = D.Uniform(rpy_min, rpy_max)
        self.min_spawn_separation = float(
            self.cfg.task.get("min_spawn_separation", 2.0))

    def _build_opponent_bank(self, task_cfg):
        from omegaconf import OmegaConf

        opp_cfg = task_cfg.get("opponent", {})
        policies = opp_cfg.get("policies", None)
        if policies is not None:
            policies = OmegaConf.to_container(policies, resolve=True)
        if not policies:
            # Default single hover heuristic so the task runs standalone.
            policies = [{
                "id": f"{self.opp_role}:heuristic0",
                "kind": "heuristic",
                "checkpoint_path": None,
                "metadata": {"heuristic": "hover"},
            }]
        probs = opp_cfg.get("probs", None)
        if probs is not None:
            probs = OmegaConf.to_container(probs, resolve=True)
        if not probs:
            probs = {p["id"]: 1.0 / len(policies) for p in policies}

        # Frozen RL opponents are PPO policies; reuse this env's specs since the
        # observation layout is role-symmetric.
        algo_cfg = opp_cfg.get("algo_cfg", None)
        self.opponent_bank = FrozenPolicyBank(
            policies,
            self.observation_spec,
            self.action_spec,
            self.reward_spec,
            algo_cfg,
            self.device,
        )
        self.opponent_bank.assign(probs, self.num_envs)

    def _init_arena_draw(self):
        """Acquire the debug-draw interface and draw the arena box (non-headless)."""
        self.draw = None
        if self._headless or not self.arena_visualize:
            return
        try:
            from isaacsim.util.debug_draw import _debug_draw
            self.draw = _debug_draw.acquire_debug_draw_interface()
        except Exception:
            self.draw = None
            return
        self._draw_arena(torch.arange(self.num_envs, device=self.device))

    def _draw_arena(self, env_ids: torch.Tensor):
        if getattr(self, "draw", None) is None:
            return
        lo = [-self.arena_half_xy, -self.arena_half_xy, self.arena_z_min]
        hi = [self.arena_half_xy, self.arena_half_xy, self.arena_z_max]
        color = (0.1, 0.9, 0.2, 1.0)
        width = 2.0

        def _rect(z, off):
            c = [
                [lo[0] + off[0], lo[1] + off[1], z + off[2]],
                [hi[0] + off[0], lo[1] + off[1], z + off[2]],
                [hi[0] + off[0], hi[1] + off[1], z + off[2]],
                [lo[0] + off[0], hi[1] + off[1], z + off[2]],
                [lo[0] + off[0], lo[1] + off[1], z + off[2]],
            ]
            self.draw.draw_lines(c[:-1], c[1:], [color] * 4, [width] * 4)

        for env_id in env_ids.tolist():
            off = self.envs_positions[env_id].tolist()
            _rect(lo[2], off)   # floor
            _rect(hi[2], off)   # ceiling
            floor = [
                [lo[0] + off[0], lo[1] + off[1], lo[2] + off[2]],
                [hi[0] + off[0], lo[1] + off[1], lo[2] + off[2]],
                [hi[0] + off[0], hi[1] + off[1], lo[2] + off[2]],
                [lo[0] + off[0], hi[1] + off[1], lo[2] + off[2]],
            ]
            ceil = [[x, y, hi[2] + off[2]] for x, y, _ in floor]
            self.draw.draw_lines(floor, ceil, [color] * 4, [width] * 4)

    def _design_scene(self):
        self.pursuer, self.pursuer_controller = MultirotorBase.make(
            self.pursuer_model_name, "PIDRateController",
            device=str(self.device), name="pursuer",
        )
        self.evader, self.evader_controller = MultirotorBase.make(
            self.evader_model_name, "PIDRateController",
            device=str(self.device), name="evader",
        )
        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        )
        self.pursuer.spawn(translations=[(0.0, 0.0, 1.0)])
        self.evader.spawn(translations=[(1.0, 0.0, 1.0)])

        # Role -> (agent drone / controller, opponent drone / controllers).
        # Done here (before _set_specs) so the specs can read the agent drone.
        if self.role == PURSUER:
            self.agent_drone, self.opp_drone = self.pursuer, self.evader
            self.controller = self.pursuer_controller  # RL CTBR controller
            self.opp_rate_controller = self.evader_controller
        else:
            self.agent_drone, self.opp_drone = self.evader, self.pursuer
            self.controller = self.evader_controller
            self.opp_rate_controller = self.pursuer_controller
        # Heuristic opponents are flown through the proven CTBR + on-board rate
        # PID stack; LeePositionController is only marginally stable and sinks on
        # the motor-lagged Crazyflie.
        g = 9.81
        mass = float(self.opp_drone.params["mass"])
        per_rotor_max = float(self.opp_rate_controller.max_thrusts.reshape(-1)[0])
        hover_throttle = math.sqrt(mass * g / (4.0 * per_rotor_max))
        # max_tilt_deg is the dominant knob for scripted-opponent survival: at
        # 42 deg a quarter of the thrust vector points sideways (cos42=0.74), so
        # on the T/W~1.4 Crazyflie any reactive horizontal maneuver -- near a
        # wall OR dodging in open space -- dumped the lift needed to hold
        # altitude and the drone dived (diagnostic: crashes 45% at walls, 25%
        # interior, none a spawn transient). Capping tilt at 22 deg (cos=0.93)
        # keeps a comfortable vertical margin so the opponent cannot fly itself
        # into the ground; it also gentles horizontal accel, which curbs the
        # near-wall momentum that caused the wall overshoots.
        opp_max_tilt = float(self.cfg.task.get("heuristic", {}).get(
            "max_tilt_deg", 22.0))
        self.opp_ctbr = GeometricCTBR(
            mass=mass, g=g, hover_throttle=hover_throttle,
            target_clip=float(self.opp_rate_controller.target_clip),
            min_ratio=float(self.opp_rate_controller.min_thrust_ratio),
            max_ratio=float(self.opp_rate_controller.max_thrust_ratio),
            kp=8.0, kv=6.0, k_att=12.0, k_yaw=2.0, max_tilt_deg=opp_max_tilt,
        )

        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        # pos(3) + rot matrix(9) + body lin vel(3) + body ang vel(3)
        # + opponent rel distance(3) + opponent rel lin vel(3)
        # + previous action (4, optional)
        obs_dim = 3 + 9 + 3 + 3 + 3 + 3
        if self.use_previous_action:
            obs_dim += self.agent_drone.action_spec.shape[-1]

        state_dim = obs_dim + (self.time_encoding_dim if self.use_time_encoding else 0)

        self.observation_spec = Composite({
            "agents": {
                "observation": UnboundedContinuous(torch.Size([1, obs_dim])),
                "state": UnboundedContinuous(torch.Size([1, state_dim])),
                "intrinsics": self.agent_drone.intrinsics_spec_flattened.unsqueeze(0),
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = Composite({
            "agents": {
                "action": self.agent_drone.action_spec.unsqueeze(0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = Composite({
            "agents": {"reward": UnboundedContinuous(torch.Size([1, 1]))}
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            state_key=("agents", "state"),
            reward_key=("agents", "reward"),
        )

        self.stats_spec = Composite({
            "return": UnboundedContinuous(torch.Size([1]), device=self.device),
            "episode_len": UnboundedContinuous(torch.Size([1]), device=self.device),
            "distance": UnboundedContinuous(torch.Size([1]), device=self.device),
            "success_rate": UnboundedContinuous(torch.Size([1]), device=self.device),
            "capture": UnboundedContinuous(torch.Size([1]), device=self.device),
            "time_to_capture": UnboundedContinuous(torch.Size([1]), device=self.device),
            "motor_effort": UnboundedContinuous(torch.Size([1]), device=self.device),
            "crash": UnboundedContinuous(torch.Size([1]), device=self.device),
            "opp_crash": UnboundedContinuous(torch.Size([1]), device=self.device),
            "out_of_bounds": UnboundedContinuous(torch.Size([1]), device=self.device),
            "opp_out_of_bounds": UnboundedContinuous(torch.Size([1]), device=self.device),
            "win_rate": UnboundedContinuous(torch.Size([1]), device=self.device),
            "forced_error": UnboundedContinuous(torch.Size([1]), device=self.device),
            "timeout": UnboundedContinuous(torch.Size([1]), device=self.device),
            "capture_radius": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_approach": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_precision": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_distance": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_body_rate": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_action_smoothness": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_cmd_rate": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_terminal": UnboundedContinuous(torch.Size([1]), device=self.device),
        }).expand(self.num_envs).to(self.device)

        self.info_spec = Composite({
            "drone_state": UnboundedContinuous(torch.Size([1, 13]), device=self.device),
            "prev_action": self.agent_drone.action_spec.unsqueeze(0),
            "policy_action": self.agent_drone.action_spec.unsqueeze(0),
        }).expand(self.num_envs).to(self.device)

        self.observation_spec["stats"] = self.stats_spec
        self.observation_spec["info"] = self.info_spec
        self.stats = self.stats_spec.zero()
        self.info = self.info_spec.zero()

    # --------------------------------------------------------------- reset
    def _reset_idx(self, env_ids: torch.Tensor):
        self.stats[env_ids] = 0.0
        num_env = len(env_ids)
        self.pursuer._reset_idx(env_ids, self.training)
        self.evader._reset_idx(env_ids, self.training)

        agent_pos = self.spawn_pos_dist.sample(torch.Size([num_env, 1]))
        opp_pos = self.spawn_pos_dist.sample(torch.Size([num_env, 1]))
        # Push apart until at least min_spawn_separation.
        for _ in range(8):
            sep = torch.norm(opp_pos - agent_pos, dim=-1, keepdim=True)
            close = (sep < self.min_spawn_separation).squeeze(-1)
            if not bool(close.any()):
                break
            resample = self.spawn_pos_dist.sample(torch.Size([num_env, 1]))
            opp_pos = torch.where(close.unsqueeze(-1), resample, opp_pos)

        agent_rpy = self.spawn_rpy_dist.sample(torch.Size([num_env, 1]))
        opp_rpy = self.spawn_rpy_dist.sample(torch.Size([num_env, 1]))
        agent_rot = euler_to_quaternion(agent_rpy)
        opp_rot = euler_to_quaternion(opp_rpy)

        self.agent_local_pos[env_ids] = agent_pos
        self.opp_local_pos[env_ids] = opp_pos
        theta = torch.rand(num_env, 1, device=self.device) * (2 * torch.pi)
        self.flee_fallback_dir[env_ids] = torch.stack(
            [torch.cos(theta), torch.sin(theta)], dim=-1)

        self.agent_drone.set_world_poses(
            self.envs_positions[env_ids].unsqueeze(1) + agent_pos, agent_rot, env_ids)
        self.opp_drone.set_world_poses(
            self.envs_positions[env_ids].unsqueeze(1) + opp_pos, opp_rot, env_ids)
        zero_vel = torch.zeros(num_env, 1, 6, device=self.device)
        self.agent_drone.set_velocities(zero_vel, env_ids)
        self.opp_drone.set_velocities(zero_vel.clone(), env_ids)

        rel0 = opp_pos - agent_pos
        self.prev_distance[env_ids] = torch.norm(rel0, dim=-1)
        self.opp_prev_action[env_ids] = 0.0
        self._just_reset[env_ids] = True

    # ------------------------------------------------------------ sim step
    def _pre_sim_step(self, tensordict: TensorDictBase):
        # RL agent: action arrives already converted to motor commands.
        agent_action = tensordict[("agents", "action")].reshape(self.num_envs, 1, -1)
        self.agent_effort = self.agent_drone.apply_action(agent_action)
        self.current_action = tensordict[("info", "policy_action")]
        self.action_error_order1 = tensordict[("stats", "action_error_order1")]

        # Frozen opponent: combine RL (CTBR->motor) and heuristic (pos->motor).
        opp_cmds = self._compute_opponent_motor_commands(tensordict)
        self.opp_drone.apply_action(opp_cmds)

    def _compute_opponent_motor_commands(self, tensordict: TensorDictBase) -> torch.Tensor:
        opp_state = self.opp_drone.get_state()[..., :13]  # [N,1,13]
        # Envs that were reset since the last sim step. Latched by _reset_idx
        # because the action tensordict has no "done" key; cleared here so only
        # the first substep after a reset clears the controllers' integral state.
        done = self._just_reset.clone()
        td_done = tensordict.get("done", None)
        if td_done is not None:
            done = done | td_done.reshape(self.num_envs, 1).bool()
        self._just_reset[:] = False

        # Pre-tanh CTBR action [wx,wy,wz,T] per env: RL opponents from the bank,
        # heuristic opponents from the geometric outer loop.
        raw = torch.zeros(
            self.num_envs, 1, self.agent_drone.action_spec.shape[-1], device=self.device)

        rl_mask = self.opponent_bank.rl_mask
        if bool(rl_mask.any()):
            rl_raw = self.opponent_bank.rl_raw_actions(self._opp_obs)
            raw = torch.where(rl_mask.reshape(-1, 1, 1), rl_raw, raw)

        masks = self.opponent_bank.heuristic_masks()
        if masks:
            target_pos, target_yaw = self._heuristic_targets()
            # `done` must be forwarded: GeometricCTBR holds a per-env altitude
            # integrator (ki_z) to null the Crazyflie's steady-state height
            # droop, and it is reset on `done`. Without it the accumulator
            # carried up to +/-i_limit_z of stale error straight into the next
            # episode -- a large persistent thrust bias that pinned the scripted
            # opponent to the floor or pushed it through the ceiling, regardless
            # of what the heuristic geometry commanded.
            heur_raw = self.opp_ctbr.compute(
                opp_state.squeeze(1), target_pos.squeeze(1), None,
                target_yaw.squeeze(1), done=done,
            ).reshape(self.num_envs, 1, -1)
            any_heur = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for mask in masks.values():
                any_heur |= mask
            raw = torch.where(any_heur.reshape(-1, 1, 1), heur_raw, raw)

        # Single CTBR -> motor conversion through the on-board rate PID.
        action = torch.tanh(raw)
        self.opp_prev_action = action.detach().clone()
        target_rate, target_thrust = action.split([3, 1], dim=-1)
        ctrl = self.opp_rate_controller
        target_thrust = torch.clamp(
            (target_thrust + 1) / 2,
            min=ctrl.min_thrust_ratio, max=ctrl.max_thrust_ratio) * 2 ** 16
        target_rate = target_rate * 180.0 * ctrl.target_clip
        cmds, _ = ctrl(
            opp_state, target_rate=target_rate, target_thrust=target_thrust,
            reset_pid=done.expand(-1, opp_state.shape[1]))
        torch.nan_to_num_(cmds, 0.0)
        return cmds.reshape(self.num_envs, 1, -1)

    def _heuristic_targets(self):
        """Env-local target position and yaw for each heuristic type, blended by mask."""
        opp_pos = self.opp_drone.pos  # [N,1,3] env-local (refreshed in get_state)
        agent_pos = self.agent_drone.pos
        t = (self.progress_buf.float() * self.cfg.sim.dt).reshape(self.num_envs, 1, 1)

        # Default target: hold current position.
        target = opp_pos.clone()
        yaw = torch.zeros(self.num_envs, 1, device=self.device)

        masks = self.opponent_bank.heuristic_masks()

        if "hover" in masks:
            m = masks["hover"].reshape(-1, 1, 1)
            target = torch.where(m, self.opp_local_pos, target)

        if "circular" in masks:
            # Clamp the orbit centre inward so the whole circle stays in bounds.
            r = min(self.heur_circular_radius, self.arena_half_xy - 0.5)
            cx = self.opp_local_pos[..., 0:1].clamp(
                -self.arena_half_xy + r, self.arena_half_xy - r)
            cy = self.opp_local_pos[..., 1:2].clamp(
                -self.arena_half_xy + r, self.arena_half_xy - r)
            center = torch.cat([cx, cy, self.opp_local_pos[..., 2:3]], dim=-1)
            angle = self.heur_circular_omega * t
            offset = torch.cat([
                r * torch.cos(angle),
                r * torch.sin(angle),
                torch.zeros_like(angle),
            ], dim=-1)
            circ = center + offset
            m = masks["circular"].reshape(-1, 1, 1)
            target = torch.where(m, circ, target)
            circ_yaw = (angle.reshape(self.num_envs, 1) + torch.pi / 2)
            yaw = torch.where(masks["circular"].reshape(-1, 1), circ_yaw, yaw)

        if "pursue" in masks:
            m = masks["pursue"].reshape(-1, 1, 1)
            target = torch.where(m, agent_pos, target)
            to_agent = agent_pos - opp_pos
            p_yaw = torch.atan2(to_agent[..., 1], to_agent[..., 0])
            yaw = torch.where(masks["pursue"].reshape(-1, 1), p_yaw, yaw)

        if "flee" in masks:
            # Flee HORIZONTALLY away from the pursuer at a held altitude.
            #
            # The previous version fled along the full 3D `opp_pos - agent_pos`,
            # so whenever the pursuer was above the evader (half of all spawns,
            # both drones spawning uniformly in z) "flee" was a dive command and
            # the motor-lagged Crazyflie flew itself into the floor within a few
            # seconds. Restricting the flee direction to the xy-plane and
            # servoing z back to the spawn altitude makes the evader
            # self-correcting in altitude: it can never descend on purpose, and
            # any sink is actively climbed out of.
            rel_xy = (opp_pos - agent_pos)[..., :2]
            # Directly-overhead tie-break: as the horizontal separation vanishes
            # the flee direction is undefined, so blend in this episode's fixed
            # escape heading instead of letting the evader park under the pursuer.
            d_xy = torch.norm(rel_xy, dim=-1, keepdim=True)
            blend = 1.0 - (d_xy / 0.5).clamp(max=1.0)
            rel_xy = rel_xy + blend * self.flee_fallback_dir
            away_xy = torch.cat(
                [rel_xy, torch.zeros_like(opp_pos[..., 2:3])], dim=-1)
            away = normalize(away_xy)
            # Wall handling has two SEPARATE parts, because coupling them was
            # the bug in the first fix:
            #
            #  * OUTWARD flee drive -- faded to zero as the drone nears the
            #    keep-in edge. A target that recedes outward at full speed right
            #    up to the wall builds momentum a position push cannot brake: the
            #    drone overshot |xy|=arena_half (out of bounds) and the hard tilt
            #    needed to reverse dumped the lift the thrust-limited Crazyflie
            #    needs, so it simultaneously dived into the floor (diagnostic:
            #    49% crash / 45% wall-OOB, both at the walls, descent to -9 m/s).
            #    Fading the outward step lets the CTBR's velocity damping (kv)
            #    coast the drone to a stop inside the arena.
            #  * INWARD push -- ALWAYS active (never faded). This is what curls
            #    the evader along the wall and pulls it back from the edge. The
            #    first fix multiplied the push into the same step as the outward
            #    drive, so fading the step also killed the push and the drone
            #    simply drifted into the wall with nothing pulling it in.
            push = self._boundary_push(opp_pos, margin=self.heur_wall_margin)
            push_xy = torch.cat(
                [push[..., :2], torch.zeros_like(push[..., 2:3])], dim=-1)
            edge = self.arena_half_xy - self.heur_wall_margin
            dist_in = (edge - opp_pos[..., :2].abs().amax(dim=-1, keepdim=True))
            speed_scale = (dist_in / self.heur_brake_zone).clamp(0.0, 1.0)
            full_step = self.heur_speed * self.heur_lookahead
            outward = away * (full_step * speed_scale)
            inward = push_xy * (full_step * self.heur_wall_gain)
            flee = opp_pos + outward + inward
            # Hold the spawn altitude rather than tracking the (possibly sunk)
            # current one, so the evader always has a positive climb setpoint.
            flee = torch.cat([flee[..., :2], self.opp_local_pos[..., 2:3]], dim=-1)
            m = masks["flee"].reshape(-1, 1, 1)
            target = torch.where(m, flee, target)
            move = outward + inward
            f_yaw = torch.atan2(move[..., 1], move[..., 0])
            yaw = torch.where(masks["flee"].reshape(-1, 1), f_yaw, yaw)

        # Keep every heuristic target inside the arena shrunk by a safety band,
        # so tracking overshoot can never put the scripted opponent through a
        # wall or below the crash altitude.
        sm = self.heur_safe_margin
        lim = max(self.arena_half_xy - sm, 0.0)
        target[..., 0] = target[..., 0].clamp(-lim, lim)
        target[..., 1] = target[..., 1].clamp(-lim, lim)
        target[..., 2] = target[..., 2].clamp(
            self.arena_z_min + max(sm, 0.5), self.arena_z_max - sm)
        return target, yaw

    def _boundary_push(self, pos: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        """Inward push per axis that grows within `margin` of each arena wall."""
        margin = max(float(margin), 1e-6)
        half = self.arena_half_xy

        def axis(coord, lo, hi):
            near_lo = (lo + margin - coord).clamp(min=0.0) / margin  # push toward +
            near_hi = (coord - (hi - margin)).clamp(min=0.0) / margin  # push toward -
            return near_lo - near_hi

        px = axis(pos[..., 0:1], -half, half)
        py = axis(pos[..., 1:2], -half, half)
        pz = axis(pos[..., 2:3], self.arena_z_min, self.arena_z_max)
        return torch.cat([px, py, pz], dim=-1)

    # ------------------------------------------------------ obs and reward
    def _agent_observation(self, self_drone, other_drone, prev_action, obs_config):
        """Symmetric observation for ``self_drone``, in its own body frame."""
        self_pos = self_drone.pos
        self_rot_quat = self_drone.rot
        self_vel_b = self_drone.vel_b
        self_vel_w = self_drone.vel_w
        self_previous_action = prev_action
        other_pos = other_drone.pos
        other_vel_w = other_drone.vel_w

        rel_dist = quat_rotate_inverse(self_rot_quat, other_pos - self_pos)
        rel_hdg = normalize(rel_dist)
        rel_vel = quat_rotate_inverse(
            self_rot_quat, other_vel_w[..., :3] - self_vel_w[..., :3])
        rot_mat = quaternion_to_rotation_matrix(self_rot_quat).reshape(
            self.num_envs, 1, 9)

        self_pos_obs = self_pos
        lin_vel = self_vel_b[..., :3]
        ang_vel = self_vel_b[..., 3:6]

        # Additive Gaussian sensor noise (agent only; frozen opponents observe clean).
        if obs_config.get("include_noise", False):
            self_pos_obs = self_pos_obs + torch.randn_like(self_pos_obs) * self.obs_noise_pos_std
            rot_mat = rot_mat + torch.randn_like(rot_mat) * self.obs_noise_rot_std
            lin_vel = lin_vel + torch.randn_like(lin_vel) * self.obs_noise_lin_vel_std
            ang_vel = ang_vel + torch.randn_like(ang_vel) * self.obs_noise_rot_vel_std
            rel_dist = rel_dist + torch.randn_like(rel_dist) * self.obs_noise_rel_dist_std
            rel_hdg = normalize(rel_dist)
            rel_vel = rel_vel + torch.randn_like(rel_vel) * self.obs_noise_rel_lin_vel_std

        obs = [
            self_pos_obs,
            rot_mat,
            lin_vel,
            ang_vel,
            rel_dist if obs_config.get("use_rel_dist", True) else rel_hdg,
            rel_vel,
        ]

        if self.use_previous_action:
            obs.append(self_previous_action)

        return torch.cat(obs, dim=-1)

    def _compute_state_and_obs(self):
        self.agent_drone.get_state()
        self.opp_drone.get_state()

        obs = self._agent_observation(
            self.agent_drone, self.opp_drone, self.current_action, self.obs_cfg)
        # Opponent observation for the frozen policy on the next pre-sim step.
        self._opp_obs = self._agent_observation(
            self.opp_drone, self.agent_drone, self.opp_prev_action, self.obs_cfg)

        state = obs
        if self.use_time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            state = torch.cat(
                [obs, t.expand(-1, self.time_encoding_dim).unsqueeze(1)], dim=-1)

        drone_state = self.agent_drone.get_state()[..., :13]
        self.info["drone_state"][:] = drone_state

        # Capture and arena violations. Computed here (before
        # _compute_reward_and_done) so the terminal snapshot is latched into the
        # stats that get cloned below; otherwise the episode resets first and
        # every terminal stat logs as zero.
        agent_pos = self.agent_drone.pos
        opp_pos = self.opp_drone.pos
        distance = torch.norm(opp_pos - agent_pos, dim=-1)  # [N,1]
        self._distance = distance
        self._captured = distance <= self.active_capture_radius  # [N,1]
        # Two distinct terminal geometric events, both resolved the same way:
        #  * CRASH (ground, or NaN state): a full loss for whoever crashed.
        #  * LEAVE ARENA (through the walls/ceiling): a full loss for whoever left.
        # For the *other* drone each is partial credit only
        # (reward_forced_error_scale * W), never a full win. The dense boundary
        # penalty still discourages approaching a wall, for both drones.
        self._agent_crash = (
            self._crashed(agent_pos) | torch.isnan(agent_pos).any(-1))
        self._opp_crash = (
            self._crashed(opp_pos) | torch.isnan(opp_pos).any(-1))
        self._agent_oob = self._out_of_arena(agent_pos)
        self._opp_oob = self._out_of_arena(opp_pos)

        captured = self._captured.float()
        newly = ((captured > 0) & (self.stats["success_rate"] <= 0)).float()
        self.stats["success_rate"][:] = torch.maximum(
            self.stats["success_rate"], captured)
        self.stats["capture"][:] = self.stats["success_rate"]
        cap_time = self.progress_buf.unsqueeze(1).float() * self.cfg.sim.dt
        self.stats["time_to_capture"][:] = torch.where(
            newly > 0, cap_time, self.stats["time_to_capture"])
        self.stats["motor_effort"] += self.agent_effort
        self.stats["distance"].lerp_(distance, 1 - self.alpha)
        self.stats["crash"][:] = torch.maximum(
            self.stats["crash"], self._agent_crash.float())
        self.stats["opp_crash"][:] = torch.maximum(
            self.stats["opp_crash"], self._opp_crash.float())
        self.stats["out_of_bounds"][:] = torch.maximum(
            self.stats["out_of_bounds"], self._agent_oob.float())
        self.stats["opp_out_of_bounds"][:] = torch.maximum(
            self.stats["opp_out_of_bounds"], self._opp_oob.float())

        # Resolve the episode outcome here too, so the terminal snapshot is
        # latched before these stats are cloned into the observation below.
        self._truncated = (self.progress_buf >= self.max_episode_length - 1).reshape(
            self.num_envs, 1)
        # Any of these ends the episode. Leaving the arena terminates too, but
        # (unlike a crash) only as a loss for the leaver, not a win for the other.
        self._terminated = (
            self._captured | self._agent_crash | self._opp_crash
            | self._agent_oob | self._opp_oob)
        self._timeout = self._truncated & ~self._terminated
        if self.role == PURSUER:
            win = self._captured                      # only an actual capture wins
            loss = self._agent_crash | self._timeout  # crashed, or ran out of time
        else:
            win = self._timeout                       # survived the full episode
            loss = self._captured | self._agent_crash
        loss = loss | self._agent_oob
        self._win = win
        self._loss = loss & ~win
        # Driving the opponent into the ground or out of the arena is a real
        # achievement but NOT a full win: it pays reward_forced_error_scale * W.
        # At full parity the pursuer simply farmed it -- standing off and waiting
        # for a scripted evader to fall out of the sky gave the same +W as a
        # capture for a fraction of the effort, and the capture rate collapsed to
        # ~3% while 93% of "wins" were evader crashes. Partial credit keeps the
        # signal ("you forced the error") while leaving capture strictly the best
        # outcome. Only counted when the episode was not already resolved as a
        # win or a loss for the agent (e.g. both drones hitting the ground).
        self._forced_error = (
            (self._opp_crash | self._opp_oob) & ~self._win & ~self._loss)
        self._terminal_reward = self.reward_terminal_weight * (
            self._win.float() - self._loss.float()
            + self.reward_forced_error_scale * self._forced_error.float())
        self.stats["reward_terminal"][:] = self._terminal_reward
        self.stats["win_rate"][:] = torch.maximum(
            self.stats["win_rate"], self._win.float())
        self.stats["forced_error"][:] = torch.maximum(
            self.stats["forced_error"], self._forced_error.float())
        self.stats["timeout"][:] = torch.maximum(
            self.stats["timeout"], self._timeout.float())

        intrinsics = self.agent_drone.intrinsics
        intrinsics_flat = torch.cat([
            intrinsics["mass"], intrinsics["inertia"], intrinsics["com"],
            intrinsics["KF"], intrinsics["KM"], intrinsics["tau_up"],
            intrinsics["tau_down"], intrinsics["drag_coef"],
        ], dim=-1)

        return TensorDict({
            "agents": {
                "observation": obs,
                "state": state,
                "intrinsics": intrinsics_flat,
            },
            "stats": self.stats.clone(),
            "info": self.info.clone(),
        }, self.batch_size)

    def _compute_reward_and_done(self):
        agent_pos = self.agent_drone.pos
        agent_vel_b = self.agent_drone.vel_b

        # distance and capture were computed in _compute_state_and_obs.
        distance = self._distance
        delta = self.prev_distance - distance
        # Precision potential (telescopes to <= weight; strong pull near contact).
        precision = self.reward_precision_weight * (
            torch.exp(-self.reward_precision_scale * distance)
            - torch.exp(-self.reward_precision_scale * self.prev_distance)
        )
        first_step = (self.progress_buf == 0).reshape(self.num_envs, 1).float()
        precision = precision * (1.0 - first_step)
        self.prev_distance = distance.detach().clone()

        body_rate = agent_vel_b[..., 3:6]
        body_rate_pen = self.reward_body_rate_weight * torch.norm(
            body_rate, dim=-1)  # [N,1]

        # Action-jerk penalty: norm of the CTBR action delta this step (already
        # computed by the PIDRateController transform and latched in
        # _pre_sim_step). Bang-bang commands -> large jerk every step.
        smoothness_pen = (self.reward_action_smoothness_weight
                          * self.action_error_order1)  # [N,1]

        # Commanded body-rate magnitude (tanh action, rate channels only).
        cmd_rate = self.current_action[..., :3].reshape(self.num_envs, -1)
        cmd_rate_pen = self.reward_cmd_rate_weight * torch.norm(
            cmd_rate, dim=-1, keepdim=True)  # [N,1]

        # Outcome was resolved (and latched into stats) in _compute_state_and_obs.
        truncated = self._truncated
        terminated = self._terminated
        done_mask = terminated | truncated
        terminal = self._terminal_reward

        # Dense inward push so the boundary is learned before it is hit.
        bounds_pen = self.reward_bounds_weight * self._bounds_proximity(agent_pos)

        # Optional dense distance penalty (pursuer shaping; see __init__).
        if self.role == PURSUER and self.use_distance:
            reward_distance = self._reward_distance(distance)
        else:
            reward_distance = torch.zeros(self.num_envs, 1, device=self.device)

        if self.role == PURSUER:
            approach = self.reward_approach_weight * delta
            reward = (approach + precision + reward_distance - body_rate_pen
                      - smoothness_pen - cmd_rate_pen - bounds_pen + terminal)
            self.stats["reward_approach"].lerp_(approach, 1 - self.alpha)
            self.stats["reward_precision"].lerp_(precision, 1 - self.alpha)
        else:
            reward = (self.reward_step - body_rate_pen - smoothness_pen
                      - cmd_rate_pen - bounds_pen + terminal)
            self.stats["reward_approach"].lerp_(torch.zeros_like(reward), 1 - self.alpha)
            self.stats["reward_precision"].lerp_(torch.zeros_like(reward), 1 - self.alpha)

        self.stats["reward_distance"].lerp_(reward_distance, 1 - self.alpha)
        self.stats["reward_body_rate"].lerp_(-body_rate_pen, 1 - self.alpha)
        self.stats["reward_action_smoothness"].lerp_(-smoothness_pen, 1 - self.alpha)
        self.stats["reward_cmd_rate"].lerp_(-cmd_rate_pen, 1 - self.alpha)
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["capture_radius"][:] = self.active_capture_radius

        # Advance the global capture-radius curriculum once per simulator step.
        self.global_step += 1
        self._update_success_radius()

        return TensorDict({
            "agents": {"reward": reward.unsqueeze(-1)},
            "done": done_mask,
            "terminated": terminated,
            "truncated": truncated,
        }, self.batch_size)

    def _reward_distance(self, distance: torch.Tensor) -> torch.Tensor:
        """Negative reward scaled by distance to the opponent (pursuer shaping).

        Mirrors Intercept._reward_distance_to_evader: reward_distance_weight is
        negative, so being far from the evader is penalized.
        """
        return self.reward_distance_weight * distance

    @property
    def active_capture_radius(self) -> float:
        """Fixed eval radius outside training, else the current curriculum radius."""
        if not self.training:
            return self.success_radius_eval
        return self.success_radius

    def _update_success_radius(self):
        """Linearly shrink the capture radius from init toward end with global step."""
        radius = self.success_radius_init - self.success_radius_lr * float(
            self.global_step)
        self.success_radius = max(radius, self.success_radius_end)

    def _crashed(self, pos: torch.Tensor) -> torch.Tensor:
        """True where the drone has hit the ground (altitude below the crash
        threshold). This is the only terminal geometric event; the arena
        walls/ceiling are enforced softly via the boundary penalty, not by
        terminating the episode."""
        z = pos[..., 2]  # [N,1]
        return z < self.minimum_altitude

    def _out_of_arena(self, pos: torch.Tensor) -> torch.Tensor:
        """True where the drone has left the arena through a wall or the ceiling.
        The floor is intentionally excluded: hitting the ground is a *crash*
        (see `_crashed`), which is resolved differently (a win for the other
        drone), whereas leaving the arena is only a loss for the one who left."""
        x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]  # each [N,1]
        return (
            (x.abs() > self.arena_half_xy)
            | (y.abs() > self.arena_half_xy)
            | (z > self.arena_z_max)
        )

    def _out_of_bounds(self, pos: torch.Tensor) -> torch.Tensor:
        x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]  # each [N,1]
        return (
            (x.abs() > self.arena_half_xy)
            | (y.abs() > self.arena_half_xy)
            | (z < self.arena_z_min)
            | (z > self.arena_z_max)
        )

    def _bounds_proximity(self, pos: torch.Tensor) -> torch.Tensor:
        """0 deep inside the arena, ramping to 1 at each wall."""
        m = max(self.bounds_margin, 1e-6)
        half = self.arena_half_xy

        def axis(coord, lo, hi):
            return torch.maximum(
                (lo + m - coord).clamp(min=0.0), (coord - (hi - m)).clamp(min=0.0)
            ) / m

        px = axis(pos[..., 0], -half, half)
        py = axis(pos[..., 1], -half, half)
        pz = axis(pos[..., 2], self.arena_z_min, self.arena_z_max)
        return torch.maximum(torch.maximum(px, py), pz).clamp(0.0, 1.0)  # [N,1]
