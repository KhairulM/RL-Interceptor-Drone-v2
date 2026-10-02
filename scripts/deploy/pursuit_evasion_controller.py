"""PursuitEvasion policy controller built on the drone/mocap helpers.

The PursuitEvasion counterpart of
[intercept_controller_v2.py](intercept_controller_v2.py). It loads an artifact
produced by [export_pursuit_evasion_policy.py](export_pursuit_evasion_policy.py)
and flies it on a Crazyflie.

Because the PursuitEvasion observation is role-symmetric, one controller covers
both sides: the artifact's ``role`` decides which drone is "self". The opponent
can be driven three ways (``controller.opponent.mode``):

* ``policy``   - a second PursuitEvasion artifact flying the opposing role, so
                 both learned policies compete on hardware (the real self-play
                 deployment);
* ``cf``       - a real Crazyflie tracked by mocap but commanded elsewhere;
* ``scripted`` - hover / linear / circular / flee position setpoints, mirroring
                 the seed heuristics the population was trained against.

Run with:  python pursuit_evasion_controller.py [--config config_pursuit_evasion.yaml]
"""
from __future__ import annotations

import argparse
import importlib
import logging
import math
import os
import time
import warnings
from typing import Optional

import numpy as np
import torch

import pursuit_evasion_common as pec
from drone import CrazyflieDrone, ScriptedDrone, DronePosePublisher
from mocap import MocapReceiver, MocapTfPublisher

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)

warnings.filterwarnings(
    'ignore',
    message=r'Using legacy TYPE_HOVER_LEGACY\. Please update your crazyflie-firmware\.',
    category=DeprecationWarning,
    module=r'cflib\.crazyflie\.commander',
)
warnings.filterwarnings(
    "ignore",
    message="The supervisor subsystem requires CRTP protocol version 12",
)

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'config_pursuit_evasion.yaml'
)

SCRIPTED_MOTIONS = ('hover', 'linear', 'circular', 'flee')


def _read_yaml(path: str) -> dict:
    import yaml

    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def _load_policy(artifact_dir: str):
    """Load a TorchScript PursuitEvasion policy plus its metadata."""
    artifact_dir = os.path.abspath(os.path.expanduser(artifact_dir))
    ts_path, meta_path = pec.artifact_paths(artifact_dir)
    if not (os.path.isfile(ts_path) and os.path.isfile(meta_path)):
        raise FileNotFoundError(
            f'Missing artifact(s) under {artifact_dir}: expected '
            f'{pec.POLICY_TS_FILENAME} and {pec.METADATA_FILENAME}. '
            f'Run export_pursuit_evasion_policy.py first.'
        )
    metadata = pec.load_metadata(meta_path)
    # Deploy on the CPU even when a GPU is present: the actor is a tiny MLP, so
    # host<->device transfer would only add latency and jitter to the control
    # loop, and deterministic timing is what keeps the body-rate commands
    # matched to training.
    device = torch.device('cpu')
    policy = torch.jit.load(ts_path, map_location=device).eval()
    logger.info(
        '[pe] Loaded %s %s policy (obs_dim=%d) from %s',
        metadata.algo, metadata.role, metadata.obs.obs_dim, ts_path,
    )
    return metadata, policy, device


def _build_command(
    policy: torch.nn.Module,
    metadata: pec.PursuitEvasionMetadata,
    own: pec.DroneState,
    opponent: pec.DroneState,
    arena_origin: np.ndarray,
    previous_action: torch.Tensor,
    device: torch.device = torch.device('cpu'),
) -> 'tuple[torch.Tensor, pec.CTBRCommand, torch.Tensor]':
    """Run the policy and return the tanh-scaled action, the CTBR command, and
    the exact observation vector fed to the network (for debugging)."""
    def _t(value) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    # The env observes positions relative to the arena origin.
    obs = pec.build_observation(
        metadata.obs,
        self_pos=_t(own.pos - arena_origin),
        self_quat_wxyz=_t(own.quat_wxyz),
        self_lin_vel_world=_t(own.lin_vel),
        self_ang_vel_body=_t(own.ang_vel),
        opponent_pos=_t(opponent.pos - arena_origin),
        opponent_lin_vel_world=_t(opponent.lin_vel),
        previous_action=_t(previous_action),
    ).reshape(1, metadata.obs.obs_dim)

    with torch.no_grad():
        raw_action = policy(obs)
    return (
        torch.tanh(raw_action).squeeze(0),
        pec.decode_action_to_ctbr(raw_action, metadata.ctbr),
        obs.reshape(-1),
    )


def _quat_to_rpy_deg(quat_wxyz: np.ndarray) -> 'tuple[float, float, float]':
    """Roll/pitch/yaw in degrees from an ``(w, x, y, z)`` quaternion (for logs)."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(np.degrees([roll, pitch, yaw]))


def _log_obs_breakdown(
    metadata: pec.PursuitEvasionMetadata,
    own: pec.DroneState,
    opponent: pec.DroneState,
    arena_origin: np.ndarray,
    obs: torch.Tensor,
) -> None:
    """Dump the raw states and the named observation components.

    Use ``controller.debug_obs: true`` and hover both drones (opponent static)
    to check the deploy observation against what the sim env produces for the
    same physical state. The single most important line is ``rel_pos(body)``:
    if the two drones are physically metres apart but this reads ~0 (or a wrong
    offset), the two Crazyflies' position estimates are NOT in a shared world
    frame (e.g. each CrazySim EKF is relative to its own spawn), which makes the
    relative-position observation garbage and the policy chase a phantom.
    """
    o = obs.detach().cpu().numpy().reshape(-1)
    # Component offsets mirror pec.build_observation's concat order.
    self_pos = o[0:3]
    rot = o[3:12]
    lin_vel_b = o[12:15]
    ang_vel_b = o[15:18]
    rel_pos_b = o[18:21]
    rel_vel_b = o[21:24]
    prev_act = o[24:28] if metadata.obs.use_previous_action else np.zeros(4)

    own_rpy = _quat_to_rpy_deg(own.quat_wxyz)
    world_rel = (opponent.pos - arena_origin) - (own.pos - arena_origin)

    logger.info(
        '[pe:dbg] --- raw states (world frame) ---\n'
        '  own   pos=[%+.2f %+.2f %+.2f] rpy(deg)=[%+.0f %+.0f %+.0f] '
        'vel=[%+.2f %+.2f %+.2f] gyro(rad/s)=[%+.2f %+.2f %+.2f]\n'
        '  opp   pos=[%+.2f %+.2f %+.2f] vel=[%+.2f %+.2f %+.2f]\n'
        '  rel_pos(world)=[%+.2f %+.2f %+.2f] |rel|=%.2fm\n'
        '[pe:dbg] --- observation components (network input) ---\n'
        '  self_pos      =[%+.2f %+.2f %+.2f]\n'
        '  rot(diag)     =[%+.2f %+.2f %+.2f]  (all ~1 => level)\n'
        '  lin_vel(body) =[%+.2f %+.2f %+.2f]\n'
        '  ang_vel(body) =[%+.2f %+.2f %+.2f]\n'
        '  rel_pos(body) =[%+.2f %+.2f %+.2f]  <== should point at the opponent\n'
        '  rel_vel(body) =[%+.2f %+.2f %+.2f]\n'
        '  prev_action   =[%+.2f %+.2f %+.2f %+.2f]',
        own.pos[0], own.pos[1], own.pos[2], own_rpy[0], own_rpy[1], own_rpy[2],
        own.lin_vel[0], own.lin_vel[1], own.lin_vel[2],
        own.ang_vel[0], own.ang_vel[1], own.ang_vel[2],
        opponent.pos[0], opponent.pos[1], opponent.pos[2],
        opponent.lin_vel[0], opponent.lin_vel[1], opponent.lin_vel[2],
        world_rel[0], world_rel[1], world_rel[2], float(np.linalg.norm(world_rel)),
        self_pos[0], self_pos[1], self_pos[2],
        rot[0], rot[4], rot[8],
        lin_vel_b[0], lin_vel_b[1], lin_vel_b[2],
        ang_vel_b[0], ang_vel_b[1], ang_vel_b[2],
        rel_pos_b[0], rel_pos_b[1], rel_pos_b[2],
        rel_vel_b[0], rel_vel_b[1], rel_vel_b[2],
        prev_act[0], prev_act[1], prev_act[2], prev_act[3],
    )


class PursuitEvasionController:
    def __init__(self, args: argparse.Namespace) -> None:
        self.config_path = os.path.abspath(os.path.expanduser(args.config))
        config = _read_yaml(self.config_path)

        artifact_dir = config.get('artifact_dir')
        if not artifact_dir:
            raise ValueError(f"Config must set 'artifact_dir' in {self.config_path}.")
        self.metadata, self.policy, self.device = _load_policy(str(artifact_dir))

        self.role = self.metadata.role
        self.opponent_role = pec.opponent_role(self.role)

        controller_config = config.get('controller', {}) or {}
        if not isinstance(controller_config, dict):
            raise ValueError("Config section 'controller' must be a mapping.")

        self.state_timeout = float(controller_config.get('state_timeout', 0.5))
        self.min_altitude = float(controller_config.get('min_altitude', 0.15))
        self.takeoff_height = float(controller_config.get('takeoff_height', 1.0))
        self.takeoff_duration = float(controller_config.get('takeoff_duration', 3.0))
        self.log_commands = bool(controller_config.get('log_commands', True))
        # Verbose per-component observation dump (states + network input), used
        # to compare the deploy observation against the sim env for a matched
        # state. Throttled to debug_period_s so it does not flood the log.
        self.debug_obs = bool(controller_config.get('debug_obs', False))
        self.debug_period_s = float(controller_config.get('debug_period_s', 0.5))
        self._last_debug_time = 0.0
        self.publish_tf = bool(controller_config.get('publish_tf', False))
        self.mocap_world_frame = str(controller_config.get('mocap_world_frame', 'world'))
        self.arena_origin = np.asarray(
            controller_config.get('arena_origin', [0.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)
        # Stop once the drones are this close; mirrors the trained capture radius.
        self.stop_on_capture = bool(controller_config.get('stop_on_capture', True))
        self.capture_radius = float(
            controller_config.get('capture_radius', self.metadata.capture_radius))

        # Keep the vehicles inside the trained arena regardless of what the
        # policy commands; outside it the observation is out of distribution.
        self.enforce_arena = bool(controller_config.get('enforce_arena', True))
        self.arena_margin = float(controller_config.get('arena_margin', 0.3))

        self.control_dt = float(
            controller_config.get('control_dt', 0.0) or self.metadata.ctbr.dt
        )

        # Optional first-order low-pass on the commanded body rates. The policy
        # was trained with the on-board rate loop stepped at the 50 Hz control
        # period; the CrazySim / real Crazyflie firmware rate loop runs ~10x
        # faster and so tracks the raw body-rate setpoints far more crisply,
        # turning the policy's aggressive rate commands into the visible roll
        # oscillation. Feeding the fast firmware loop a smoothed setpoint
        # narrows the effective closed-loop rate bandwidth back toward what the
        # (unchanged, 50 Hz) training loop produced. ``command_rate_lpf_tau`` is
        # the filter time constant in seconds; 0 disables (raw passthrough).
        # A good starting point is ~1-2 control periods (0.02-0.05 s); increase
        # until the roll oscillation is gone, decrease if the chase gets sluggish.
        self.command_rate_lpf_tau = float(
            controller_config.get('command_rate_lpf_tau', 0.0))
        if self.command_rate_lpf_tau > 0.0:
            self._rate_lpf_alpha = 1.0 - math.exp(
                -self.control_dt / self.command_rate_lpf_tau)
        else:
            self._rate_lpf_alpha = 1.0
        # Per-drone filtered body-rate state (keyed 'own'/'opp').
        self._rate_filt: dict = {}

        opponent_config = controller_config.get('opponent', {}) or {}
        if not isinstance(opponent_config, dict):
            raise ValueError("Config key 'controller.opponent' must be a mapping.")
        self.opponent_mode = str(opponent_config.get('mode', 'scripted')).strip().lower()
        if self.opponent_mode not in ('policy', 'cf', 'scripted'):
            raise ValueError(
                "Config key 'controller.opponent.mode' must be one of "
                "['policy', 'cf', 'scripted']; got "
                f"'{self.opponent_mode}'."
            )

        self.opponent_metadata = None
        self.opponent_policy = None
        if self.opponent_mode == 'policy':
            opponent_artifact = opponent_config.get('artifact_dir')
            if not opponent_artifact:
                raise ValueError(
                    "Config key 'controller.opponent.artifact_dir' is required "
                    "when controller.opponent.mode == 'policy'."
                )
            self.opponent_metadata, self.opponent_policy, _ = _load_policy(
                str(opponent_artifact))
            if self.opponent_metadata.role != self.opponent_role:
                raise ValueError(
                    f"Opponent artifact declares role "
                    f"'{self.opponent_metadata.role}' but the agent is "
                    f"'{self.role}', so the opponent must be "
                    f"'{self.opponent_role}'."
                )

        motion = opponent_config.get('motion', {}) or {}
        if not isinstance(motion, dict):
            raise ValueError("Config key 'controller.opponent.motion' must be a mapping.")
        self.motion_type = str(motion.get('type', 'hover')).strip().lower()
        if self.motion_type not in SCRIPTED_MOTIONS:
            raise ValueError(
                f"Config key 'controller.opponent.motion.type' must be one of "
                f"{list(SCRIPTED_MOTIONS)}; got '{self.motion_type}'."
            )
        self.motion_yaw_deg = float(motion.get('yaw_deg', 0.0))
        self.motion_anchor = np.asarray(
            motion.get('anchor', [2.0, 0.0, self.takeoff_height]), dtype=np.float64
        ).reshape(3)
        self.motion_speed = float(motion.get('speed', 1.0))
        self.motion_direction = self._unit_vec3(
            motion.get('direction', [1.0, 0.0, 0.0]),
            'controller.opponent.motion.direction',
        )
        self.motion_radius = float(motion.get('radius', 1.5))
        self.motion_omega = float(motion.get('omega', 1.0))
        self.motion_lookahead = float(motion.get('lookahead', 0.5))

        self.self_config = pec.load_drone_config_from_yaml(self.config_path, self.role)
        self.opponent_drone_config = pec.load_drone_config_from_yaml(
            self.config_path, self.opponent_role)
        self.mocap_config = pec.load_mocap_config_from_yaml(self.config_path)

        self.drone_pose_pub = DronePosePublisher(world_frame=self.mocap_world_frame)
        self.own = CrazyflieDrone(self.self_config, pose_publisher=self.drone_pose_pub)
        if self.opponent_mode == 'scripted':
            self.opponent = ScriptedDrone(
                self.opponent_drone_config, pose_publisher=self.drone_pose_pub)
        else:
            self.opponent = CrazyflieDrone(
                self.opponent_drone_config, pose_publisher=self.drone_pose_pub)

        self.mocap_tf_publisher: Optional[MocapTfPublisher] = None
        self.mocap_receiver: Optional[MocapReceiver] = None

        self.previous_action = torch.zeros(4, dtype=torch.float32, device=self.device)
        self.opponent_previous_action = torch.zeros(
            4, dtype=torch.float32, device=self.device)
        self._motion_start_time = 0.0
        self._position_setpoint = self.motion_anchor.copy()

    @staticmethod
    def _unit_vec3(value, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size != 3:
            raise ValueError(f"Config key '{name}' must have exactly 3 elements.")
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-6:
            raise ValueError(f"Config key '{name}' must be nonzero.")
        return arr / norm

    # -- mocap / connection -------------------------------------------------
    def _setup_mocap(self) -> None:
        if not self.mocap_config.enabled:
            return
        if self.publish_tf:
            self.mocap_tf_publisher = MocapTfPublisher(
                world_frame=self.mocap_world_frame)
        self.mocap_receiver = MocapReceiver(
            self.mocap_config, tf_publisher=self.mocap_tf_publisher)

        rigid_ids = {
            'pursuer': self.mocap_config.pursuer_rigid_body_id,
            'evader': self.mocap_config.evader_rigid_body_id,
        }
        self.mocap_receiver.register(
            rigid_ids[self.role], self.own, frame_id=self.role)
        if self.opponent_mode != 'scripted':
            self.mocap_receiver.register(
                rigid_ids[self.opponent_role], self.opponent,
                frame_id=self.opponent_role)
        self.mocap_receiver.start()

    def _connect_and_setup(self) -> None:
        self.own.connect()
        self.own.setup()
        self.opponent.connect()
        self.opponent.setup()

    def _smooth_command(
        self, key: str, command: pec.CTBRCommand) -> pec.CTBRCommand:
        """Low-pass the commanded body rates (in place) to emulate the trained
        50 Hz rate-loop bandwidth on the faster firmware rate loop.

        No-op when ``command_rate_lpf_tau`` is 0 (``_rate_lpf_alpha`` == 1).
        The thrust command is deliberately left untouched: the filter targets
        the roll/pitch/yaw oscillation, and low-passing a setpoint does not
        change its steady-state value anyway.
        """
        if self._rate_lpf_alpha >= 1.0:
            return command
        rates = command.body_rate_deg
        prev = self._rate_filt.get(key)
        if prev is None:
            prev = rates.clone()
        filt = self._rate_lpf_alpha * rates + (1.0 - self._rate_lpf_alpha) * prev
        self._rate_filt[key] = filt.detach()
        command.body_rate_deg = filt
        return command

    def _arm_and_takeoff(self) -> None:
        self.own.arm()
        self.opponent.arm()
        self.own.takeoff(self.takeoff_height, duration=self.takeoff_duration)
        self.opponent.takeoff(self.takeoff_height, duration=self.takeoff_duration)

    def _initialize_motion_anchor(self) -> None:
        if self.opponent_mode == 'scripted':
            self.motion_anchor[2] = self.takeoff_height
            return
        deadline = time.time() + 2.0
        while time.time() < deadline:
            state = self.opponent.get_state()
            if state is not None:
                self.motion_anchor = state.pos.copy()
                break
            time.sleep(0.05)
        self.motion_anchor[2] = max(self.motion_anchor[2], self.takeoff_height)

    # -- scripted opponent motion ------------------------------------------
    def _scripted_setpoint(self, own_state: Optional[pec.DroneState]) -> np.ndarray:
        """Position setpoint for the scripted opponent, clamped to the arena."""
        elapsed = time.time() - self._motion_start_time
        anchor = self.motion_anchor

        if self.motion_type == 'hover':
            target = anchor.copy()
        elif self.motion_type == 'linear':
            target = anchor + self.motion_direction * (self.motion_speed * elapsed)
        elif self.motion_type == 'circular':
            angle = self.motion_omega * elapsed
            target = anchor + np.array([
                self.motion_radius * np.cos(angle),
                self.motion_radius * np.sin(angle),
                0.0,
            ])
        else:  # flee
            current = self._position_setpoint
            if own_state is None:
                target = current.copy()
            else:
                away = current - own_state.pos
                norm = float(np.linalg.norm(away))
                away = away / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
                target = current + away * (self.motion_speed * self.motion_lookahead)

        return self._clamp_to_arena(target)

    def _clamp_to_arena(self, position: np.ndarray) -> np.ndarray:
        """Clamp a world-frame setpoint into the trained arena volume."""
        half = self.metadata.arena_half_xy - self.arena_margin
        local = position - self.arena_origin
        local[0] = float(np.clip(local[0], -half, half))
        local[1] = float(np.clip(local[1], -half, half))
        local[2] = float(np.clip(
            local[2],
            self.metadata.arena_z_min + self.arena_margin,
            self.metadata.arena_z_max - self.arena_margin,
        ))
        return local + self.arena_origin

    def _outside_arena(self, position: np.ndarray) -> bool:
        local = position - self.arena_origin
        return bool(
            abs(local[0]) > self.metadata.arena_half_xy
            or abs(local[1]) > self.metadata.arena_half_xy
            or local[2] < self.metadata.arena_z_min
            or local[2] > self.metadata.arena_z_max
        )

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:
        cflib_crtp = importlib.import_module('cflib.crtp')
        cflib_crtp.init_drivers()

        try:
            self._connect_and_setup()
            self._setup_mocap()
            self._arm_and_takeoff()
            self._initialize_motion_anchor()
            self._motion_start_time = time.time()
            self._position_setpoint = self.motion_anchor.copy()

            logger.info(
                '[pe] Running %s policy at %.1f Hz vs %s (%s). Ctrl+C to stop.',
                self.role, 1.0 / self.control_dt, self.opponent_role,
                self.opponent_mode if self.opponent_mode != 'scripted'
                else f'scripted:{self.motion_type}',
            )
            self._rate_filt = {}
            if self._rate_lpf_alpha < 1.0:
                logger.info(
                    '[pe] Body-rate command low-pass ON (tau=%.3fs, alpha=%.2f '
                    'at %.1f Hz): smoothing the setpoint sent to the firmware '
                    'rate loop to match the trained 50 Hz rate response.',
                    self.command_rate_lpf_tau, self._rate_lpf_alpha,
                    1.0 / self.control_dt,
                )

            # Fixed-rate scheduling anchored to an absolute deadline: the policy
            # was trained at a fixed control period, and a plain sleep(dt) would
            # stretch it by the compute time and jitter, holding each body-rate
            # command longer than in training.
            next_tick = time.perf_counter()
            overrun_count = 0
            tick_count = 0
            while self.own.connected and self.opponent.connected:
                own_state = self.own.get_state()
                opponent_state = self.opponent.get_state()

                if self.opponent_mode == 'scripted':
                    self._position_setpoint = self._scripted_setpoint(own_state)
                    self.opponent.send_position_setpoint(
                        self._position_setpoint[0],
                        self._position_setpoint[1],
                        self._position_setpoint[2],
                        yaw_deg=self.motion_yaw_deg,
                    )

                if own_state is None or opponent_state is None:
                    time.sleep(self.control_dt)
                    continue

                now = time.time()
                if now - own_state.stamp > self.state_timeout:
                    logger.warning('[pe] %s state timed out; stopping.', self.role)
                    break
                if own_state.pos[2] < self.min_altitude:
                    logger.warning(
                        '[pe] %s below min altitude (%.2f m); stopping.',
                        self.role, own_state.pos[2],
                    )
                    break
                if self.enforce_arena and self._outside_arena(own_state.pos):
                    logger.warning(
                        '[pe] %s left the trained arena at [%.2f, %.2f, %.2f]; '
                        'stopping (observation would be out of distribution).',
                        self.role, *own_state.pos,
                    )
                    break

                distance = float(np.linalg.norm(opponent_state.pos - own_state.pos))
                if self.stop_on_capture and distance <= self.capture_radius:
                    logger.info(
                        '[pe] Capture radius reached (%.2f m <= %.2f m); stopping.',
                        distance, self.capture_radius,
                    )
                    break

                action, command, obs_vec = _build_command(
                    self.policy, self.metadata, own_state, opponent_state,
                    self.arena_origin, self.previous_action, self.device,
                )
                # previous_action feeds the observation, so store the raw
                # (unfiltered) policy action; the low-pass only shapes the
                # body-rate setpoint actually sent to the firmware.
                self.previous_action = action.detach()
                command = self._smooth_command('own', command)
                self.own.send_ctbr(command)

                if self.debug_obs and (
                        now - self._last_debug_time >= self.debug_period_s):
                    _log_obs_breakdown(
                        self.metadata, own_state, opponent_state,
                        self.arena_origin, obs_vec)
                    self._last_debug_time = now

                if self.opponent_mode == 'policy':
                    assert self.opponent_policy is not None
                    assert self.opponent_metadata is not None
                    opp_action, opp_command, _ = _build_command(
                        self.opponent_policy, self.opponent_metadata,
                        opponent_state, own_state, self.arena_origin,
                        self.opponent_previous_action, self.device,
                    )
                    self.opponent_previous_action = opp_action.detach()
                    opp_command = self._smooth_command('opp', opp_command)
                    self.opponent.send_ctbr(opp_command)

                if self.log_commands:
                    rates = command.body_rate_deg.detach().cpu().numpy().reshape(-1)
                    logger.info(
                        '[pe] alt=%.2fm dist=%.2fm rates(deg/s)=[%+.0f,%+.0f,%+.0f] '
                        'thrust_pwm=%.0f',
                        own_state.pos[2], distance,
                        rates[0], rates[1], rates[2],
                        float(command.thrust_pwm.item()),
                    )

                self.own.publish_pose()
                if self.opponent_mode != 'scripted':
                    self.opponent.publish_pose()

                next_tick += self.control_dt
                tick_count += 1
                slack = next_tick - time.perf_counter()
                if slack > 0.0:
                    time.sleep(slack)
                else:
                    overrun_count += 1
                    next_tick = time.perf_counter()
                    if overrun_count % 50 == 1:
                        logger.warning(
                            '[pe] Control loop overran its %.1f ms budget by '
                            '%.1f ms (%d/%d ticks late); the policy is running '
                            'slower than its trained %.1f Hz.',
                            self.control_dt * 1e3, -slack * 1e3,
                            overrun_count, tick_count, 1.0 / self.control_dt,
                        )
        except KeyboardInterrupt:
            logger.info('[pe] Stopping.')
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        for drone in (self.own, self.opponent):
            for action in ('land', 'disarm', 'disconnect'):
                try:
                    getattr(drone, action)()
                except Exception:
                    pass

        if self.mocap_receiver is not None:
            self.mocap_receiver.stop()
            self.mocap_receiver = None
        if self.mocap_tf_publisher is not None:
            self.mocap_tf_publisher.close()
            self.mocap_tf_publisher = None


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description='Run a PursuitEvasion policy using the drone/mocap helpers.'
    )
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG_PATH,
        help=f'Path to the YAML configuration file (default: {DEFAULT_CONFIG_PATH}).',
    )
    args = parser.parse_args(argv)
    PursuitEvasionController(argparse.Namespace(config=args.config)).run()


if __name__ == '__main__':
    main()
