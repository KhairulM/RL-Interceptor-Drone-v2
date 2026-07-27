import socket
import threading
import time
from typing import Optional

import rclpy
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

from NatNetClient import NatNetClient
import intercept_common as ic
from drone import Drone


def _detect_local_ip_for_server(server_ip: str) -> str:
    """Return the local interface IP that routes to ``server_ip``.

    Uses a connect-less UDP socket so no packets are sent; the kernel simply
    resolves the outbound interface. Falls back to ``0.0.0.0`` (bind-any) if the
    route cannot be determined.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((server_ip, 1))
        return sock.getsockname()[0]
    except OSError:
        return '0.0.0.0'
    finally:
        sock.close()


class MocapTfPublisher:
    """Optional ROS 2 TF publisher for mocap poses.

    Only instantiated when TF publishing is requested, so the rest of the
    controller keeps its pure-``cflib`` dependency footprint. Initialises
    ``rclpy`` if the caller has not done so already and tears it down on
    :meth:`close`.
    """

    def __init__(self, world_frame: str = 'world',
                 node_name: str = 'intercept_mocap_tf') -> None:
        self._world_frame = world_frame
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self._node = rclpy.create_node(node_name)
        self._pub = self._node.create_publisher(TFMessage, 'tf', 10)
        self._connected = True

    def publish(self, child_frame_id: str, pose: ic.MocapPose) -> None:
        if not self._connected or not rclpy.ok():
            return

        transform = TransformStamped()
        transform.header.stamp = self._node.get_clock().now().to_msg()
        transform.header.frame_id = self._world_frame
        transform.child_frame_id = child_frame_id
        px, py, pz = pose.position
        transform.transform.translation.x = float(px)
        transform.transform.translation.y = float(py)
        transform.transform.translation.z = float(pz)
        qx, qy, qz, qw = pose.quat_xyzw
        transform.transform.rotation.x = float(qx)
        transform.transform.rotation.y = float(qy)
        transform.transform.rotation.z = float(qz)
        transform.transform.rotation.w = float(qw)
        self._pub.publish(TFMessage(transforms=[transform]))

    def close(self) -> None:
        self._connected = False
        try:
            self._node.destroy_node()
        except Exception:  # pragma: no cover - best-effort teardown
            pass
        if self._owns_rclpy:
            try:
                rclpy.shutdown()
            except Exception:  # pragma: no cover - best-effort teardown
                pass


class MocapReceiver:
    """Stream OptiTrack rigid-body poses and forward them to Drone.

    The receiver owns a background NatNet thread (started by :meth:`start`).
    Each rigid-body frame is transformed into the ROS FLU convention via
    :func:`intercept_common.transform_mocap_pose` and pushed to the Drone
    registered under the matching streaming id through ``extpos.send_extpos``.
    Position are optionally re-published on a ROS 2 TF tree. Frames for
    unregistered ids are ignored.
    """

    def __init__(self, cfg: ic.MocapConfig,
                 tf_publisher: Optional[MocapTfPublisher] = None) -> None:
        self._cfg = cfg
        self._tf_publisher = tf_publisher
        self._targets: dict[int, tuple[Drone, str]] = {}
        self._marker_targets: dict[int, tuple[Drone, str]] = {}
        self._last_send_stamp: dict[int, float] = {}
        self._first_send_stamp: dict[int, float] = {}
        self._lock = threading.Lock()
        self._client = None
        self._connected = False
        self._min_dt = 1.0 / self._cfg.mocap_send_rate_hz
        self._orientation_align_time = self._cfg.orientation_align_time

    def register(self, rigid_body_id: int, drone: Drone,
                 frame_id: Optional[str] = None) -> None:
        """Route frames for ``rigid_body_id`` to ``drone`` (and TF ``frame_id``)."""
        rb_id = int(rigid_body_id)
        with self._lock:
            self._targets[rb_id] = (drone, frame_id or f'cf_{rb_id}')

    def register_marker(self, marker_id: int, drone: Drone,
                        frame_id: Optional[str] = None) -> None:
        """Route labelled-marker frames for ``marker_id`` to ``drone``."""
        m_id = int(marker_id)
        with self._lock:
            self._marker_targets[m_id] = (drone, frame_id or f'marker_{m_id}')

    def start(self) -> None:
        self._connected = True
        client = NatNetClient()
        client.serverIPAddress = self._cfg.server_ip
        client.localIPAddress = (
            self._cfg.local_ip
            or _detect_local_ip_for_server(self._cfg.server_ip))
        client.multicastAddress = self._cfg.multicast_address
        client.commandPort = self._cfg.command_port
        client.dataPort = self._cfg.data_port
        client.rigidBodyListener = self._on_rigid_body
        # client.markerListener = self._on_labeled_marker
        # client.unlabeledMarkerListener = self._on_unlabeled_markers
        self._client = client
        client.run()

    def _on_rigid_body(self, rigid_body_id, position, rotation,
                       tracking_valid) -> None:
        if not self._connected:
            return
        
        if not tracking_valid:
            return

        rb_id = int(rigid_body_id)
        now = time.monotonic()

        # Frequency throttling: only send at the configured rate (if > 0.0 Hz).
        with self._lock:
            target = self._targets.get(rb_id)
            if target is None:
                return
            if self._cfg.mocap_send_rate_hz > 0.0:
                last_stamp = self._last_send_stamp.get(rb_id, 0.0)
                if (now - last_stamp) < self._min_dt:
                    return
                self._last_send_stamp[rb_id] = now
        
        drone, frame_id = target
        
        # Send the first pose with orientation to align the EKF, then send only position for the rest of the time.
        first_send_stamp = self._first_send_stamp.get(rb_id)
        if first_send_stamp is None:
            self._first_send_stamp[rb_id] = now
            first_send_stamp = now

        pose = ic.transform_mocap_pose(
            self._cfg, rb_id, position, rotation, tracking_valid)
        
        try:
            if (now - first_send_stamp) < self._orientation_align_time:
                drone.send_mocap_pose(pose)
            else:
                if not drone.mocap_orientation_aligned.is_set():
                    drone.mocap_orientation_aligned.set()
                    
                drone.send_mocap_pos(pose.position)
        except Exception:  # pragma: no cover - link may be tearing down
            return

        if self._tf_publisher is not None:
            self._tf_publisher.publish(frame_id, pose)

    def _on_labeled_marker(self, marker_id, position, size, occluded) -> None:
        """Handle labelled marker callback from NatNet."""
        if not self._connected or occluded:
            return
        m_id = int(marker_id)
        now = time.monotonic()
        with self._lock:
            target = self._marker_targets.get(m_id)
            if target is None:
                return
            if self._cfg.mocap_send_rate_hz > 0.0:
                if (now - self._last_send_stamp.get(m_id, 0.0)) < self._min_dt:
                    return
                self._last_send_stamp[m_id] = now
        drone, frame_id = target

        pose = ic.transform_mocap_pose(
            self._cfg, m_id, position, (1.0, 0.0, 0.0, 0.0), True)
        if not pose.tracking_valid:
            return

        try:
            drone.send_mocap_pos(pose.position)
        except Exception:  # pragma: no cover - link may be tearing down
            return

        if self._tf_publisher is not None:
            self._tf_publisher.publish(frame_id, pose)

    def _on_unlabeled_markers(self, positions: list) -> None:
        """Handle unlabeled markers callback from NatNet.

        Sends each unlabeled marker position to the drone(s) registered for
        it (use :meth:`register_marker` with any integer id). Since unlabeled
        markers have no stable IDs, they will only reach this callback if you
        explicitly call ``markerListener`` on the client — by default this
        forwards positions for all targets currently in ``_marker_targets``.
        """
        if not self._connected or not positions:
            return
        now = time.monotonic()

        # For unlabeled markers there's no ID to match, so we broadcast each
        # position only if there is exactly one marker-target registered (as a
        # convenience for single-marker tracking setups).
        if len(self._marker_targets) != 1:
            return
        m_id = next(iter(self._marker_targets))
        if self._cfg.mocap_send_rate_hz > 0.0:
            if (now - self._last_send_stamp.get(m_id, 0.0)) < self._min_dt:
                return
            self._last_send_stamp[m_id] = now

        drone, frame_id = self._marker_targets[m_id]
        # Use the first position in the list (single-marker use case).
        pos = positions[0]
        pose = ic.transform_mocap_pose(
            self._cfg, m_id, pos, (1.0, 0.0, 0.0, 0.0), True)
        if not pose.tracking_valid:
            return

        try:
            drone.send_mocap_pos(pose.position)
        except Exception:  # pragma: no cover - link may be tearing down
            return

        if self._tf_publisher is not None:
            self._tf_publisher.publish(frame_id, pose)

    def stop(self) -> None:
        """Best-effort teardown of the NatNet sockets (threads are daemons)."""
        self._connected = False
        client = self._client
        self._client = None
        if client is None:
            return
        client.rigidBodyListener = None
        client.markerListener = None
        client.unlabeledMarkerListener = None
        for sock_attr in ('dataSocket', 'commandSocket'):
            sock = getattr(client, sock_attr, None)
            if sock is not None:
                try:
                    sock.close()
                except Exception:  # pragma: no cover - best-effort teardown
                    pass
