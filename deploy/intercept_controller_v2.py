"""Concise Intercept policy controller built on the new drone/mocap helpers.

This version keeps the old policy execution flow but delegates Crazyflie link
management, state logging, CTBR dispatch, and mocap forwarding to
``drone.py`` and ``mocap.py``. It is intentionally demo-style: load the YAML
config, connect both drones, take them off first, and then run the policy loop.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import logging
import os
import time
import warnings
from typing import Optional

import numpy as np
import torch

import intercept_common as ic
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
    message="The supervisor subsystem requires CRTP protocol version 12"
)


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'config_v2.yaml'
)


def _read_yaml(path: str) -> dict:
    import yaml

    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def _load_artifact_dir(config_path: str) -> str:
    config = _read_yaml(config_path)
    artifact_dir = config.get('artifact_dir')
    if not artifact_dir:
        raise ValueError(f"Config must set 'artifact_dir' in {config_path}.")
    return str(artifact_dir)


def _load_controller_config(config_path: str) -> dict:
    config = _read_yaml(config_path)
    controller = config.get('controller', {}) or {}
    if not isinstance(controller, dict):
        raise ValueError("Config section 'controller' must be a mapping.")
    return controller


def _load_policy(artifact_dir: str):
    artifact_dir = os.path.abspath(os.path.expanduser(artifact_dir))
    ts_path, meta_path = ic.artifact_paths(artifact_dir)
    if not (os.path.isfile(ts_path) and os.path.isfile(meta_path)):
        raise FileNotFoundError(
            f'Missing artifact(s) under {artifact_dir}: expected '
            f'{ic.POLICY_TS_FILENAME} and {ic.METADATA_FILENAME}. '
            f'Run export_policy.py first.'
        )
    metadata = ic.load_metadata(meta_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    policy = torch.jit.load(ts_path, map_location=device).eval()
    logger.info('[intercept] Loaded %s policy (obs_dim=%d) from %s on %s',
                metadata.algo, metadata.obs.obs_dim, ts_path, device)
    return metadata, policy, device


def _build_command(policy: torch.nn.Module, metadata: ic.PolicyMetadata,
                   pursuer: ic.DroneState, evader: ic.DroneState,
                   device: torch.device) -> ic.CTBRCommand:
    obs = ic.build_observation(
        metadata.obs,
        pursuer_pos=torch.as_tensor(pursuer.pos, dtype=torch.float32, device=device),
        pursuer_quat_wxyz=torch.as_tensor(pursuer.quat_wxyz, dtype=torch.float32, device=device),
        pursuer_lin_vel_world=torch.as_tensor(pursuer.lin_vel, dtype=torch.float32, device=device),
        evader_pos=torch.as_tensor(evader.pos, dtype=torch.float32, device=device),
        pursuer_ang_vel_world=torch.as_tensor(pursuer.ang_vel, dtype=torch.float32, device=device),
        evader_lin_vel_world=torch.as_tensor(evader.lin_vel, dtype=torch.float32, device=device),
    ).reshape(1, metadata.obs.obs_dim)

    with torch.no_grad():
        raw_action = policy(obs)
    return ic.decode_action_to_ctbr(raw_action, metadata.ctbr)


class InterceptController:
    def __init__(self, args: argparse.Namespace) -> None:
        self.config_path = os.path.abspath(os.path.expanduser(args.config))
        self.artifact_dir = str(args.artifact_dir)
        self.metadata, self.policy, self.device = _load_policy(self.artifact_dir)
        self._start_time = time.time()

        self.pursuer_config = ic.load_drone_config_from_yaml(self.config_path, 'pursuer')
        self.evader_config = ic.load_drone_config_from_yaml(self.config_path, 'evader')
        self.mocap_config = ic.load_mocap_config_from_yaml(self.config_path)

        controller_config = _load_controller_config(self.config_path)

        self.evader_source = str(controller_config.get('evader_source', 'cf'))
        self.state_timeout = float(controller_config.get('state_timeout', 0.5))
        self.min_altitude = float(controller_config.get('min_altitude', 0.15))
        self.takeoff_height = float(controller_config.get('takeoff_height', 1.0))
        self.takeoff_duration = float(controller_config.get('takeoff_duration', 3.0))
        self.control_dt = float(
            controller_config.get('control_dt', 0.0) or (
                self.pursuer_config.control_dt
                if self.pursuer_config.control_dt > 0.0 else self.metadata.ctbr.dt
            )
        )
        self.log_commands = bool(controller_config.get('log_commands', False))
        self.publish_tf = bool(controller_config.get('publish_tf', False))
        self.mocap_world_frame = str(controller_config.get('mocap_world_frame', 'world'))

        evader_motion = controller_config.get('evader_motion', {}) or {}
        if not isinstance(evader_motion, dict):
            raise ValueError("Config key 'controller.evader_motion' must be a mapping.")

        self.evader_motion_type = str(evader_motion.get('type', 'hover')).strip().lower()
        self.evader_motion_yaw_deg = float(evader_motion.get('yaw_deg', 0.0))

        default_anchor = np.array([3.0, 0.0, self.takeoff_height], dtype=np.float64)
        self.evader_motion_anchor = self._vec3_from_config(
            evader_motion.get('anchor', default_anchor.tolist()),
            default=default_anchor,
            name='controller.evader_motion.anchor',
        )

        random_cfg = evader_motion.get('random', {}) or {}
        if not isinstance(random_cfg, dict):
            raise ValueError("Config key 'controller.evader_motion.random' must be a mapping.")
        self.evader_random_speed = float(random_cfg.get('speed', 0.25))
        self.evader_random_radius = float(random_cfg.get('radius_xy', 1.0))
        self.evader_random_update_interval = float(random_cfg.get('update_interval', 2.0))
        random_seed = random_cfg.get('seed', None)
        self._evader_rng = np.random.default_rng(random_seed)

        self.evader_trajectory_loop = bool(evader_motion.get('trajectory_loop', True))
        self.evader_trajectory_path = evader_motion.get('trajectory_file', '')
        self._evader_traj_times: Optional[np.ndarray] = None
        self._evader_traj_pos: Optional[np.ndarray] = None
        self._evader_traj_vel: Optional[np.ndarray] = None

        self._evader_motion_start_time = 0.0
        self._evader_random_pos = self.evader_motion_anchor.copy()
        self._evader_random_target = self.evader_motion_anchor.copy()
        self._evader_random_last_update = 0.0

        self._validate_evader_motion_config()
        self._load_evader_trajectory_if_needed()

        self.drone_pose_pub = DronePosePublisher(world_frame='world')
        self.pursuer = CrazyflieDrone(
            self.pursuer_config, pose_publisher=self.drone_pose_pub
        )
        if self.evader_source == 'scripted':
            self.evader = ScriptedDrone(
                self.evader_config, pose_publisher=self.drone_pose_pub
            )
        else:
            self.evader = CrazyflieDrone(
                self.evader_config, pose_publisher=self.drone_pose_pub
            )

        self.mocap_tf_publisher: Optional[MocapTfPublisher] = None
        self.mocap_receiver: Optional[MocapReceiver] = None

    @staticmethod
    def _vec3_from_config(value, default: np.ndarray, name: str) -> np.ndarray:
        if value is None:
            return default.copy()
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size != 3:
            raise ValueError(f"Config key '{name}' must have exactly 3 elements.")
        return arr.copy()

    def _validate_evader_motion_config(self) -> None:
        supported = {'hover', 'random', 'trajectory'}
        if self.evader_motion_type not in supported:
            raise ValueError(
                f"Config key 'controller.evader_motion.type' must be one of "
                f"{sorted(supported)}, got '{self.evader_motion_type}'."
            )
        if self.evader_random_speed < 0.0:
            raise ValueError("Config key 'controller.evader_motion.random.speed' must be >= 0.")
        if self.evader_random_radius < 0.0:
            raise ValueError("Config key 'controller.evader_motion.random.radius_xy' must be >= 0.")
        if self.evader_random_update_interval <= 0.0:
            raise ValueError(
                "Config key 'controller.evader_motion.random.update_interval' must be > 0."
            )

    def _load_evader_trajectory_if_needed(self) -> None:
        if self.evader_motion_type != 'trajectory':
            return
        if not self.evader_trajectory_path:
            raise ValueError(
                "Config key 'controller.evader_motion.trajectory_file' is required "
                "when evader motion type is 'trajectory'."
            )

        config_dir = os.path.dirname(self.config_path)
        traj_path = os.path.expanduser(str(self.evader_trajectory_path))
        if not os.path.isabs(traj_path):
            traj_path = os.path.join(config_dir, traj_path)
        traj_path = os.path.abspath(traj_path)

        if not os.path.isfile(traj_path):
            raise FileNotFoundError(f"Evader trajectory file not found: {traj_path}")

        times = []
        pos = []
        vel = []

        with open(traj_path, 'r', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Trajectory file {traj_path} has no CSV header.")
            required_pos = {'x', 'y', 'z'}
            if not required_pos.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"Trajectory file {traj_path} must include columns x,y,z."
                )

            if 't' in reader.fieldnames:
                time_key = 't'
            elif 't_rel' in reader.fieldnames:
                time_key = 't_rel'
            else:
                raise ValueError(
                    f"Trajectory file {traj_path} must include either t or t_rel."
                )

            has_vel = {'vx', 'vy', 'vz'}.issubset(set(reader.fieldnames))
            for row in reader:
                t = float(row[time_key])
                times.append(t)
                pos.append([float(row['x']), float(row['y']), float(row['z'])])
                if has_vel:
                    vel.append([float(row['vx']), float(row['vy']), float(row['vz'])])

        if len(times) < 2:
            raise ValueError(f"Trajectory file {traj_path} must have at least 2 rows.")

        t_arr = np.asarray(times, dtype=np.float64)
        p_arr = np.asarray(pos, dtype=np.float64)
        if np.any(np.diff(t_arr) <= 0.0):
            raise ValueError(f"Trajectory file {traj_path} must have strictly increasing time.")

        if len(vel) == len(times):
            v_arr = np.asarray(vel, dtype=np.float64)
        else:
            dt = np.diff(t_arr)
            dp = np.diff(p_arr, axis=0)
            segment_vel = dp / dt[:, None]
            v_arr = np.vstack([segment_vel[0], segment_vel])

        self._evader_traj_times = t_arr
        self._evader_traj_pos = p_arr
        self._evader_traj_vel = v_arr
        logger.info('[intercept] Loaded evader trajectory from %s (%d points).', traj_path, len(times))

    def _setup_mocap(self) -> None:
        if not self.mocap_config.enabled:
            return

        if self.publish_tf:
            self.mocap_tf_publisher = MocapTfPublisher(
                world_frame=self.mocap_world_frame
            )
        self.mocap_receiver = MocapReceiver(
            self.mocap_config, tf_publisher=self.mocap_tf_publisher
        )
        self.mocap_receiver.register(
            self.mocap_config.pursuer_rigid_body_id, self.pursuer, frame_id='pursuer'
        )
        if self.evader_source == 'cf':
            self.mocap_receiver.register(
                self.mocap_config.evader_rigid_body_id, self.evader, frame_id='evader'
            )

        self.mocap_receiver.start()

    def _connect_and_setup(self) -> None:
        self.pursuer.connect()
        self.pursuer.setup()
        self.evader.connect()
        self.evader.setup()

    def _arm_and_takeoff(self) -> None:
        self.pursuer.arm()
        self.evader.arm()
        self.pursuer.takeoff(self.takeoff_height, duration=self.takeoff_duration)
        self.evader.takeoff(self.takeoff_height, duration=self.takeoff_duration)

    def _initialize_evader_motion_anchor(self) -> None:
        if self.evader_source != 'cf':
            self.evader_motion_anchor[2] = self.takeoff_height
            return

        deadline = time.time() + 2.0
        while time.time() < deadline:
            state = self.evader.get_state()
            if state is not None:
                self.evader_motion_anchor = state.pos.copy()
                break
            time.sleep(0.05)

        self.evader_motion_anchor[2] = max(self.evader_motion_anchor[2], self.takeoff_height)

    def _interpolate_trajectory(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if self._evader_traj_times is None or self._evader_traj_pos is None:
            return self.evader_motion_anchor.copy(), np.zeros(3, dtype=np.float64)

        traj_times = self._evader_traj_times
        traj_pos = self._evader_traj_pos
        traj_vel = self._evader_traj_vel
        assert traj_vel is not None

        final_t = float(traj_times[-1])
        if self.evader_trajectory_loop and final_t > 0.0:
            t_query = t % final_t
        else:
            t_query = float(np.clip(t, traj_times[0], final_t))

        idx = int(np.searchsorted(traj_times, t_query, side='right'))
        idx = int(np.clip(idx, 1, len(traj_times) - 1))
        t0 = traj_times[idx - 1]
        t1 = traj_times[idx]
        alpha = float((t_query - t0) / max(t1 - t0, 1e-6))
        pos = (1.0 - alpha) * traj_pos[idx - 1] + alpha * traj_pos[idx]
        vel = (1.0 - alpha) * traj_vel[idx - 1] + alpha * traj_vel[idx]
        return pos, vel

    def _sample_random_target(self) -> np.ndarray:
        angle = float(self._evader_rng.uniform(0.0, 2.0 * np.pi))
        radius = float(self.evader_random_radius * np.sqrt(self._evader_rng.uniform(0.0, 1.0)))
        target_xy = self.evader_motion_anchor[:2] + radius * np.array(
            [np.cos(angle), np.sin(angle)], dtype=np.float64
        )
        return np.array([target_xy[0], target_xy[1], self.evader_motion_anchor[2]], dtype=np.float64)

    def _compute_evader_motion_state(self) -> ic.DroneState:
        now = time.time()
        t = now - self._evader_motion_start_time
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if self.evader_motion_type == 'hover':
            return ic.DroneState(
                pos=self.evader_motion_anchor.copy(),
                quat_wxyz=quat,
                lin_vel=np.zeros(3, dtype=np.float64),
                ang_vel=np.zeros(3, dtype=np.float64),
                stamp=now,
            )

        if self.evader_motion_type == 'trajectory':
            pos, vel = self._interpolate_trajectory(t)
            return ic.DroneState(
                pos=pos,
                quat_wxyz=quat,
                lin_vel=vel,
                ang_vel=np.zeros(3, dtype=np.float64),
                stamp=now,
            )

        dt = max(0.0, now - self._evader_random_last_update)
        to_target = self._evader_random_target - self._evader_random_pos
        dist = float(np.linalg.norm(to_target))
        if dist < 0.05 or (now - self._evader_random_last_update) >= self.evader_random_update_interval:
            self._evader_random_target = self._sample_random_target()
            to_target = self._evader_random_target - self._evader_random_pos
            dist = float(np.linalg.norm(to_target))

        vel = np.zeros(3, dtype=np.float64)
        if dist > 1e-6 and dt > 0.0:
            direction = to_target / dist
            step = min(dist, self.evader_random_speed * dt)
            self._evader_random_pos = self._evader_random_pos + direction * step
            vel = direction * self.evader_random_speed

        self._evader_random_last_update = now
        return ic.DroneState(
            pos=self._evader_random_pos.copy(),
            quat_wxyz=quat,
            lin_vel=vel,
            ang_vel=np.zeros(3, dtype=np.float64),
            stamp=now,
        )

    def _get_evader_state(self, commanded_state: ic.DroneState) -> Optional[ic.DroneState]:
        if self.evader_source == 'scripted':
            return commanded_state
        state = self.evader.get_state()
        return state if state is not None else commanded_state

    def run(self) -> None:
        cflib_crtp = importlib.import_module('cflib.crtp')
        cflib_crtp.init_drivers()

        try:
            self._connect_and_setup()
            self._setup_mocap()
            self._arm_and_takeoff()
            self._initialize_evader_motion_anchor()
            self._evader_motion_start_time = time.time()
            self._evader_random_pos = self.evader_motion_anchor.copy()
            self._evader_random_target = self.evader_motion_anchor.copy()
            self._evader_random_last_update = self._evader_motion_start_time

            logger.info(
                '[intercept] Running policy at %.1f Hz. '
                'Ctrl+C to stop. evader_motion=%s', 1.0 / self.control_dt, self.evader_motion_type
            )
            while self.pursuer.connected and self.evader.connected:
                commanded_evader_state = self._compute_evader_motion_state()
                self.evader.send_position_setpoint(
                    commanded_evader_state.pos[0],
                    commanded_evader_state.pos[1],
                    commanded_evader_state.pos[2],
                    yaw_deg=self.evader_motion_yaw_deg,
                )

                pursuer_state = self.pursuer.get_state()
                evader_state = self.evader.get_state()
                if pursuer_state is None or evader_state is None:
                    time.sleep(self.control_dt)
                    continue

                now = time.time()
                if now - pursuer_state.stamp > self.state_timeout:
                    logger.warning('[intercept] Pursuer state timed out; stopping.')
                    break
                if pursuer_state.pos[2] < self.min_altitude:
                    logger.warning(
                        '[intercept] Pursuer below min altitude (%.2f m); stopping.',
                        pursuer_state.pos[2]
                    )
                    break

                command = _build_command(
                    self.policy, self.metadata, pursuer_state, evader_state, self.device
                )
                self.pursuer.send_ctbr(command)

                if self.log_commands:
                    rates = command.body_rate_deg.detach().cpu().numpy().reshape(-1)
                    dist = float(np.linalg.norm(evader_state.pos - pursuer_state.pos))
                    logger.info(
                        '[intercept] alt=%.2fm dist=%.2fm '
                        'rates(deg/s)=[%+.0f,%+.0f,%+.0f] '
                        'thrust_pwm=%.0f',
                        pursuer_state.pos[2], dist,
                        rates[0], rates[1], rates[2],
                        float(command.thrust_pwm.item())
                    )

                self.pursuer.publish_pose()
                if self.evader_source == 'cf':
                    self.evader.publish_pose()
                time.sleep(self.control_dt)
        except KeyboardInterrupt:
            logger.info('[intercept] Stopping.')
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        for drone in (self.pursuer, self.evader):
            try:
                drone.land()
            except Exception:
                pass
            try:
                drone.disarm()
            except Exception:
                pass
            try:
                drone.disconnect()
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
        description='Run the Intercept policy using the drone/mocap helpers.'
    )
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG_PATH,
        help=f'Path to the YAML configuration file (default: {DEFAULT_CONFIG_PATH}).',
    )
    args = parser.parse_args(argv)

    artifact_dir = _load_artifact_dir(args.config)
    controller_args = argparse.Namespace(
        config=args.config,
        artifact_dir=artifact_dir,
    )
    controller = InterceptController(controller_args)
    controller.run()


if __name__ == '__main__':
    main()
