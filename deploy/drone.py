import logging
import threading
import time
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

from typing import Optional

import rclpy
from geometry_msgs.msg import TransformStamped, PoseStamped

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger

import intercept_common as ic

logger = logging.getLogger(__name__)

STAB_MODE_ROLL_PARAM = 'flightmode.stabModeRoll'
STAB_MODE_PITCH_PARAM = 'flightmode.stabModePitch'
STAB_ESTIMATOR_PARAM = 'stabilizer.estimator'
STAB_MODE_RATE = 0
STAB_MODE_ANGLE = 1
STAB_ESTIMATOR_DEFAULT = 1
STAB_ESTIMATOR_KALMAN = 2


class DronePosePublisher:
    def __init__(self, world_frame: str = 'world', node_name: str = 'intercept_drone_pose') -> None:
        self._world_frame = world_frame
        self._node_name = node_name
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self._node = rclpy.create_node(node_name)
        self._pub = self._node.create_publisher(PoseStamped, 'drone_pose', 10)
        self._pubs: dict[str, rclpy.publisher.Publisher] = {}
        self._connected = True

    def register(self, drone_id: str) -> None:
        if drone_id not in self._pubs:
            self._pubs[drone_id] = self._node.create_publisher(PoseStamped, f'drone_pose/{drone_id}', 10)

    def publish(self, drone_id: str, drone: ic.DroneState) -> None:
        if not self._connected or not rclpy.ok():
            return

        publisher = self._pubs.get(drone_id)
        if publisher is None:
            logger.warning('No publisher registered for drone_id: %s', drone_id)
            return

        msg = PoseStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._world_frame
        px, py, pz = drone.pos
        msg.pose.position.x = float(px)
        msg.pose.position.y = float(py)
        msg.pose.position.z = float(pz)
        qw, qx, qy, qz = drone.quat_wxyz
        msg.pose.orientation.x = float(qx)
        msg.pose.orientation.y = float(qy)
        msg.pose.orientation.z = float(qz)
        msg.pose.orientation.w = float(qw)
        publisher.publish(msg)


class StateBuffer:
    """Accumulates the latest state from several cflib log blocks.

    Log callbacks run on a background thread, so all reads/writes go through a
    lock and consumers take an immutable :class:`ic.DroneState` snapshot.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pos = np.zeros(3)
        self._vel = np.zeros(3)
        self._quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        self._ang_vel_body_rad = np.zeros(3)
        self._pos_stamp = 0.0
        self._att_stamp = 0.0

    def update_pos_vel(self, x, y, z, vx, vy, vz) -> None:
        with self._lock:
            self._pos = np.array([x, y, z], dtype=np.float64)
            self._vel = np.array([vx, vy, vz], dtype=np.float64)
            self._pos_stamp = time.time()

    def update_quat(self, qw, qx, qy, qz) -> None:
        with self._lock:
            self._quat_wxyz = np.array([qw, qx, qy, qz], dtype=np.float64)
            self._att_stamp = time.time()

    def update_gyro_deg(self, gx, gy, gz) -> None:
        with self._lock:
            self._ang_vel_body_rad = np.radians(
                np.array([gx, gy, gz], dtype=np.float64))

    def snapshot(self) -> Optional[ic.DroneState]:
        """Return the latest state, or ``None`` if no pose has arrived yet."""
        with self._lock:
            if self._pos_stamp == 0.0 or self._att_stamp == 0.0:
                return None
            rot = ic.quaternion_to_rotation_matrix_np(self._quat_wxyz)
            ang_vel_world = rot @ self._ang_vel_body_rad
            return ic.DroneState(
                pos=self._pos.copy(),
                quat_wxyz=self._quat_wxyz.copy(),
                lin_vel=self._vel.copy(),
                ang_vel=ang_vel_world,
                stamp=min(self._pos_stamp, self._att_stamp),
            )


class Drone(ABC):

    @abstractmethod
    def connect(self):
        raise NotImplementedError

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def disconnect(self):
        raise NotImplementedError

    @abstractmethod
    def arm(self):
        raise NotImplementedError

    @abstractmethod
    def disarm(self):
        raise NotImplementedError

    @abstractmethod
    def takeoff(self, height, duration: float = 3.0):
        raise NotImplementedError

    @abstractmethod
    def land(self):
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> Optional[ic.DroneState]:
        raise NotImplementedError

    @abstractmethod
    def set_param(self, group, name, value):
        raise NotImplementedError

    @abstractmethod
    def send_mocap_pose(self, pose: ic.MocapPose):
        raise NotImplementedError

    @abstractmethod
    def send_mocap_pos(self, position: tuple[float, float, float]):
        raise NotImplementedError

    @abstractmethod
    def send_ctbr(self, command: ic.CTBRCommand):
        raise NotImplementedError

    @abstractmethod
    def send_position_setpoint(self, x: float, y: float, z: float,
                               yaw_deg: float = 0.0):
        raise NotImplementedError


class CrazyflieDrone(Drone):
    def __init__(self, drone_config: ic.DroneConfig, pose_publisher: Optional[DronePosePublisher] = None):
        self.name = drone_config.name
        self.uri = drone_config.uri
        self.drone_config = drone_config

        self.max_thrust_pwm = drone_config.max_thrust_pwm
        self.rate_sign = np.array(drone_config.rate_sign)
        self.control_dt = drone_config.control_dt
        self.log_period_ms = max(10, int(drone_config.log_period_ms))

        self.cf = Crazyflie(rw_cache=drone_config.cache_dir)
        self.state = StateBuffer()
        self.connected = False
        self.armed = False
        self.pose_publisher = pose_publisher

        if self.pose_publisher is not None:
            self.pose_publisher.register(drone_id=self.name)

        self.mocap_orientation_aligned = threading.Event()

    def _relax_setpoint_priority(self, wait_s: float = 0.1) -> None:
        """Hand control back to the high-level commander after low-level setpoints."""
        try:
            self.cf.commander.send_notify_setpoint_stop()
        except Exception:
            logger.warning('[%s] Error relaxing commander priority for %s', self.name, self.uri)
            return
        if wait_s > 0.0:
            time.sleep(wait_s)

    def _require_connected(self, action: str) -> bool:
        """Return ``True`` if connected, otherwise log why ``action`` was skipped."""
        if not self.connected:
            logger.info('[%s] Cannot %s, not connected to %s', self.name, action, self.uri)
        return self.connected

    def _require_armed(self, action: str) -> bool:
        """Return ``True`` if armed, otherwise log why ``action`` was skipped."""
        if not self.armed:
            logger.info('[%s] Cannot %s, drone is not armed', self.name, action)
        return self.armed

    def connect(self):
        self.cf.connected.add_callback(self._on_connected)
        self.cf.connection_lost.add_callback(self._on_disconnected)
        self.cf.open_link(self.uri)

        while not self.connected:
            time.sleep(0.1)

    def setup(self):
        self._setup_params()
        self._setup_loggers()

    def disconnect(self):
        self.cf.close_link()
        deadline = time.time() + 1.0
        while self.connected and time.time() < deadline:
            time.sleep(0.05)
        if self.connected:
            logger.info('[%s] Disconnected from %s', self.name, self.uri)
            self.connected = False

    def arm(self):
        if not self._require_connected('arm'):
            return

        self.cf.supervisor.send_arming_request(True)
        time.sleep(1.0)

        logger.info('[%s] Arming request sent. Waiting for Mocap to align orientation...', self.name)

        if not self.mocap_orientation_aligned.wait(timeout=10.0):
            logger.warning('[%s] Mocap orientation not aligned. '
                           'Position estimate may be inaccurate.', self.name)

        if not self._is_estimator_converged(timeout=10.0):
            logger.warning('[%s] Kalman filter not converged. '
                           'Position estimate may be inaccurate.', self.name)

        # The first setpoint must be a zero-thrust one to unlock the commander.
        self.cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
        time.sleep(0.1)
        self._relax_setpoint_priority()

        self.armed = True
        logger.info('[%s] Armed. Commander unlocked. Ready to take off.', self.name)

    def disarm(self):
        if not self._require_connected('disarm'):
            return
        self.cf.supervisor.send_arming_request(False)
        time.sleep(1.0)

        self.armed = False
        logger.info('[%s] Disarmed. Commander locked. Motors stopped.', self.name)

    def takeoff(self, height, duration: float = 3.0):
        if not self._require_connected('takeoff') or not self._require_armed('takeoff'):
            return

        logger.info('[%s] Taking off to %.2f m over %.1f s...', self.name, height, duration)

        self._relax_setpoint_priority()
        self.cf.high_level_commander.takeoff(height, duration)
        time.sleep(duration + 1.0)  # wait for takeoff to complete

    def land(self, duration: float = 3.0):
        if not self._require_connected('land') or not self._require_armed('land'):
            return

        logger.info('[%s] Landing over %.1f s...', self.name, duration)

        self._relax_setpoint_priority()
        self.cf.high_level_commander.land(0.1, duration)
        time.sleep(duration + 1.0)  # wait for landing to complete

    def get_state(self) -> Optional[ic.DroneState]:
        return self.state.snapshot()

    def set_param(self, group, name, value):
        self.cf.param.set_value(f'{group}.{name}', value)

    def send_mocap_pose(self, pose: ic.MocapPose):
        px, py, pz = pose.position
        qx, qy, qz, qw = pose.quat_xyzw
        try:
            self.cf.extpos.send_extpose(px, py, pz, qx, qy, qz, qw)
        except Exception:  # pragma: no cover - link may be tearing down
            logger.error('[%s] Error sending mocap pose to %s', self.name, self.uri)

    def send_mocap_pos(self, position: tuple[float, float, float]):
        px, py, pz = position
        try:
            self.cf.extpos.send_extpos(px, py, pz)
        except Exception:  # pragma: no cover - link may be tearing down
            logger.error('[%s] Error sending mocap position to %s', self.name, self.uri)

    def send_ctbr(self, command: ic.CTBRCommand):
        rates = command.body_rate_deg.detach().cpu().numpy().reshape(-1)  # deg/s
        sign = self.rate_sign

        roll_rate = float(sign[0]) * float(rates[0])
        pitch_rate = float(sign[1]) * float(rates[1])
        yaw_rate = float(sign[2]) * float(rates[2])
        thrust = int(np.clip(
            float(command.thrust_pwm.detach().cpu().item()),
            0.0, self.max_thrust_pwm))
        self.cf.commander.send_setpoint(roll_rate, pitch_rate, yaw_rate, thrust)

    def send_position_setpoint(self, x: float, y: float, z: float,
                               yaw_deg: float = 0.0):
        self.cf.commander.send_position_setpoint(
            float(x), float(y), float(z), float(yaw_deg))

    def send_hover_setpoint(self, z: float):
        self.cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, float(z))

    def publish_pose(self):
        if self.pose_publisher is None:
            return
        state = self.get_state()
        if state is not None:
            self.pose_publisher.publish(drone_id=self.name, drone=state)

    def _on_connected(self, link_uri):
        logger.info('[%s] Connected to %s', self.name, link_uri)
        self.connected = True

    def _on_disconnected(self, link_uri, msg=None):
        logger.info('[%s] Disconnected from %s%s', self.name, link_uri,
                    f' ({msg})' if msg else '')
        self.connected = False
        self.armed = False

    def _setup_params(self):
        # Set the stabilizer mode to rate mode for roll and pitch
        self.cf.param.set_value(STAB_MODE_ROLL_PARAM, STAB_MODE_RATE)
        self.cf.param.set_value(STAB_MODE_PITCH_PARAM, STAB_MODE_RATE)
        # Set the stabilizer estimator to Kalman filter
        self.cf.param.set_value(STAB_ESTIMATOR_PARAM, STAB_ESTIMATOR_KALMAN)

    def _setup_loggers(self):
        self._start_log(
            'pos_vel',
            ('stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z',
             'stateEstimate.vx', 'stateEstimate.vy', 'stateEstimate.vz'),
            ('float', 'float', 'float', 'float', 'float', 'float'),
            self._pos_vel_cb,
        )
        self._start_log(
            'attitude',
            ('stateEstimate.qw', 'stateEstimate.qx',
             'stateEstimate.qy', 'stateEstimate.qz'),
            ('float', 'float', 'float', 'float'),
            self._att_cb,
        )
        if self.drone_config.need_rot_speed:
            self._start_log(
                'gyro',
                ('gyro.x', 'gyro.y', 'gyro.z'),
                ('float', 'float', 'float'),
                self._gyro_cb,
            )

    def _start_log(self, name, variables, types, callback):
        log_config = LogConfig(name=name, period_in_ms=self.log_period_ms)
        for var, type in zip(variables, types):
            log_config.add_variable(var, type)
        log_config.data_received_cb.add_callback(callback)
        self.cf.log.add_config(log_config)
        log_config.start()
        return log_config

    def _pos_vel_cb(self, timestamp, data, logconf_name):
        self.state.update_pos_vel(
            data['stateEstimate.x'],
            data['stateEstimate.y'],
            data['stateEstimate.z'],
            data['stateEstimate.vx'],
            data['stateEstimate.vy'],
            data['stateEstimate.vz'],
        )

    def _att_cb(self, timestamp, data, logconf_name):
        self.state.update_quat(
            data['stateEstimate.qw'],
            data['stateEstimate.qx'],
            data['stateEstimate.qy'],
            data['stateEstimate.qz'],
        )

    def _gyro_cb(self, timestamp, data, logconf_name):
        self.state.update_gyro_deg(
            data['gyro.x'],
            data['gyro.y'],
            data['gyro.z'],
        )

    def _is_estimator_converged(self, threshold=0.001, timeout=10.0) -> bool:
        """Wait for the Kalman filter to converge on a position estimate."""
        log_conf = LogConfig(name='Kalman Variance', period_in_ms=100)
        log_conf.add_variable('kalman.varPX', 'float')
        log_conf.add_variable('kalman.varPY', 'float')
        log_conf.add_variable('kalman.varPZ', 'float')
        var_hist = {'x': [1000]*10, 'y': [1000]*10, 'z': [1000]*10}
        deadline = time.time() + timeout
        with SyncLogger(self.cf, log_conf) as sync_logger:
            for _, data, _ in sync_logger:
                px = data['kalman.varPX']
                py = data['kalman.varPY']
                pz = data['kalman.varPZ']

                for k, v in zip('xyz', (px, py, pz)):
                    var_hist[k] = var_hist[k][1:] + [v]

                # Log latest Kalman variances (verbose level - noisy).
                logger.debug(
                    '[%s] Kalman var: px=%.6f py=%.6f pz=%.6f',
                    self.name, px, py, pz,
                )

                if all(max(h) - min(h) < threshold for h in var_hist.values()):
                    return True
                if time.time() > deadline:
                    return False
            return False


class ScriptedDrone(Drone):
    """A dummy drone that implements the Drone ABC without any cflib dependency.

    Used when ``evader_source == "scripted"`` so the controller can treat the
    evader as a virtual proxy whose state is fully driven by simulated motion
    (hover, random-walk, or pre-recorded trajectory) rather than a real radio
    link to hardware.
    """

    def __init__(self, drone_config: ic.DroneConfig, pose_publisher=None) -> None:
        self.name = drone_config.name
        self.drone_config = drone_config
        self.control_dt = drone_config.control_dt
        self.connected = True  # "always connected" – no real link
        self.armed = False
        self.pose_publisher = pose_publisher

        if self.pose_publisher is not None:
            self.pose_publisher.register(drone_id=self.name)

        # Internal simulated state (starts at origin, identity quat).
        self._state_buffer = StateBuffer()

    # -- connection lifecycle ------------------------------------------------

    def connect(self) -> None:
        """No-op – scripted drone is always "connected"."""
        self.connected = True
        logger.info('[%s] Scripted drone ready (no real link)', self.name)

    def setup(self) -> None:
        """No-op – nothing to configure on a virtual drone."""
        pass

    def disconnect(self) -> None:
        """Clean-up; simply mark as disconnected."""
        self.connected = False
        self.armed = False
        logger.info('[%s] Scripted drone stopped', self.name)

    # -- arming --------------------------------------------------------------

    def arm(self) -> None:
        self.armed = True
        logger.info('[%s] Armed (scripted)', self.name)

    def disarm(self) -> None:
        self.armed = False
        logger.info('[%s] Disarmed (scripted)', self.name)

    # -- high-level commands -------------------------------------------------

    def takeoff(self, height, duration: float = 3.0) -> None:
        """Pre-fill the state buffer so ``get_state()`` returns something valid."""
        logger.info('[%s] Scripted takeoff to %.2f m (no real flight)', self.name, height)
        # Seed an initial position at (0, 0, height) so the rest of the
        # pipeline sees a plausible starting state.
        self._state_buffer.update_pos_vel(0.0, 0.0, height, 0.0, 0.0, 0.0)
        self._state_buffer.update_quat(1.0, 0.0, 0.0, 0.0)
        self._state_buffer.update_gyro_deg(0.0, 0.0, 0.0)

    def land(self, duration: float = 3.0) -> None:
        logger.info('[%s] Scripted landing (no-op)', self.name)

    # -- state ---------------------------------------------------------------

    def get_state(self) -> Optional[ic.DroneState]:
        return self._state_buffer.snapshot()

    # -- low-level commands (no-ops – scripted evader doesn't fly) -----------

    def set_param(self, group, name, value) -> None:
        pass

    def send_mocap_pose(self, pose: ic.MocapPose) -> None:
        """Update internal state from mocap so get_state() reflects reality."""
        px, py, pz = pose.position
        qw, qx, qy, qz = pose.quat_wxyz  # use the wxyz property directly
        self._state_buffer.update_pos_vel(px, py, pz, 0.0, 0.0, 0.0)
        self._state_buffer.update_quat(qw, qx, qy, qz)

    def send_mocap_pos(self, position: tuple[float, float, float]) -> None:
        """Update internal state from mocap position only."""
        px, py, pz = position
        self._state_buffer.update_pos_vel(px, py, pz, 0.0, 0.0, 0.0)

    def send_ctbr(self, command: ic.CTBRCommand) -> None:
        """No-op – scripted drone doesn't accept control commands."""
        pass

    def send_position_setpoint(self, x: float, y: float, z: float,
                               yaw_deg: float = 0.0) -> None:
        """Update internal state to reflect the commanded position."""
        self._state_buffer.update_pos_vel(
            x, y, z, 0.0, 0.0, 0.0
        )

    def send_hover_setpoint(self, z: float) -> None:
        """Update internal state z to reflect the commanded hover altitude."""
        current = self._state_buffer._pos.copy() if self._state_buffer._pos_stamp > 0 else np.zeros(3)
        current[2] = z
        self._state_buffer.update_pos_vel(
            current[0], current[1], current[2], 0.0, 0.0, 0.0
        )

    def publish_pose(self) -> None:
        if self.pose_publisher is None:
            return
        state = self.get_state()
        if state is not None:
            self.pose_publisher.publish(drone_id=self.name, drone=state)
