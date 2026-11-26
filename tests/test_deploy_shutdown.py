"""PursuitEvasion controller shutdown (SAFETY_PLAN.md bug B1 and the fault responses), with ROS / cflib stubbed."""

import sys
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

DEPLOY = Path(__file__).resolve().parents[1] / "scripts/deploy"


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)
    return sys.modules[name]


class _Any:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return _Any()

    def __call__(self, *a, **k):
        return _Any()


for name in ("rclpy", "geometry_msgs", "geometry_msgs.msg", "tf2_msgs", "tf2_msgs.msg", "cflib", "cflib.crazyflie",
             "cflib.crazyflie.log", "cflib.crazyflie.syncLogger", "cflib.crtp", "cflib.crtp.crtpstack", "NatNetClient"):
    _stub(name, ok=lambda: True, init=lambda: None, create_node=_Any, TransformStamped=_Any, PoseStamped=_Any,
          TFMessage=_Any, Crazyflie=_Any, LogConfig=_Any, SyncLogger=_Any, NatNetClient=_Any, publisher=_Any(),
          CRTPPacket=_Any, CRTPPort=_Any)
sys.path.insert(0, str(DEPLOY))
import pursuit_evasion_controller as pec_ctrl  # noqa: E402
from safety import Level, SafetyConfig, Verdict  # noqa: E402


class FakeDrone(pec_ctrl.CrazyflieDrone):
    def __init__(self, name, log):
        self.name, self.log, self.armed, self.connected = name, log, True, True

    def land_async(self, duration=3.0):
        self.log.append(("land", self.name, time.time()))
        return True

    def arm(self):
        return {"estimator_converged": True}

    def takeoff_async(self, height, duration=3.0):
        self.log.append(("takeoff", self.name, time.time()))
        return True

    def emergency_stop(self):
        self.log.append(("estop", self.name, time.time()))
        self.armed = False

    def send_position_setpoint(self, x, y, z, yaw_deg=0.0):
        self.log.append(("hold", self.name, time.time()))

    def get_state(self):
        return SimpleNamespace(pos=np.array([0.0, 0.0, 1.0]), quat_wxyz=np.array([1.0, 0, 0, 0]),
                               lin_vel=np.zeros(3))

    def send_velocity_world_setpoint(self, vx, vy, vz, yaw_rate_deg=0.0):
        self.log.append(("vel", self.name, (vx, vy, vz)))

    def go_to_async(self, x, y, z, yaw_rad, duration):
        self.log.append(("goto", self.name, (x, y, z, duration)))
        return True

    def disarm(self):
        self.log.append(("disarm", self.name, time.time()))

    def disconnect(self):
        pass


def controller(fault=None):
    c = pec_ctrl.PursuitEvasionController.__new__(pec_ctrl.PursuitEvasionController)
    log = []
    c.own, c.opponent = FakeDrone("pursuer", log), FakeDrone("evader", log)
    c.role, c.opponent_role = "pursuer", "evader"
    c._fault, c.land_duration, c.control_dt = fault, 0.2, 0.02
    c.land_speed = 10.0                                   # keeps every landing at land_duration here
    c.brake_decel = 2.0
    c.land_settle_timeout = 0.2
    c.safety_cfg = SafetyConfig(hold_s=0.1)
    c.cfg = SimpleNamespace(episode=pec_ctrl.EpisodeConfig())
    c.arena_origin = np.zeros(3)
    c.opponent_flies = True
    c._outcome = None
    c.mocap_receiver = c.mocap_tf_publisher = None
    return c, log


class ShutdownTest(unittest.TestCase):

    def test_both_landings_start_together(self):          # B1: was land(own) -> 4 s -> land(opponent)
        c, log = controller()
        c.shutdown()
        lands = [t for kind, _, t in log if kind == "land"]
        self.assertEqual(len(lands), 2)
        self.assertLess(abs(lands[1] - lands[0]), 0.05)
        self.assertGreater(min(t for k, _, t in log if k == "disarm"), max(lands) + 0.2)   # disarm after landing

    def test_emergency_stops_only_the_faulty_drone(self):
        v = Verdict()
        v.raise_to(Level.EMERGENCY, "tilt", "pursuer")
        c, log = controller(v)
        c.shutdown()
        self.assertIn(("estop", "pursuer"), [(k, n) for k, n, _ in log])
        self.assertEqual([n for k, n, _ in log if k == "land"], ["evader"])

    def test_hold_then_land_both(self):
        v = Verdict()
        v.raise_to(Level.HOLD, "stale")
        c, log = controller(v)
        c.shutdown()
        kinds = [k for k, _, _ in log]
        self.assertIn("hold", kinds)
        self.assertLess(kinds.index("hold"), kinds.index("land"))
        self.assertEqual(sorted(n for k, n, _ in log if k == "land"), ["evader", "pursuer"])

    def test_below_min_altitude_stops_only_affected_drones(self):
        for roles in (("pursuer",), ("evader",), ("pursuer", "evader")):
            with self.subTest(roles=roles):
                c, log = controller()
                for d in (c.own, c.opponent):
                    if d.name in roles:
                        d.get_state = lambda: SimpleNamespace(pos=np.array([0.0, 0.0, 0.12]),
                                                              lin_vel=np.array([2.0, 0.0, -1.0]))
                c.shutdown()
                self.assertEqual([n for k, n, _ in log if k == "estop"], list(roles))
                for role in roles:
                    self.assertFalse(any(k in ("vel", "hold", "goto", "land") and n == role for k, n, _ in log))
                self.assertEqual([n for k, n, _ in log if k == "land"],
                                 [r for r in ("pursuer", "evader") if r not in roles])

    def test_episode_floor_latches_both_roles_before_shutdown(self):
        c, log = controller()
        states = [SimpleNamespace(pos=np.array([0.0, 0.0, 0.12]), stamp=10.0) for _ in range(2)]
        self.assertTrue(c._episode_over(*states, 10.0))
        self.assertEqual((c._fault.level, c._fault.emergency_roles),
                         (Level.EMERGENCY, ["pursuer", "evader"]))
        c.shutdown()
        self.assertEqual([n for k, n, _ in log if k == "estop"], ["pursuer", "evader"])
        self.assertFalse(any(k in ("goto", "land", "vel", "hold") for k, _, _ in log))

    def test_shutdown_floor_uses_safety_threshold_relative_to_arena(self):
        c, log = controller()
        c.cfg.episode.min_altitude = 0.05
        c.safety_cfg.min_altitude = 0.2
        c.arena_origin[2] = 1.0
        c.own.get_state = lambda: SimpleNamespace(pos=np.array([0.0, 0.0, 1.12]))
        c.opponent.get_state = lambda: SimpleNamespace(pos=np.array([0.0, 0.0, 1.5]),
                                                      quat_wxyz=np.array([1.0, 0, 0, 0]), lin_vel=np.zeros(3))
        c.shutdown()
        self.assertEqual([n for k, n, _ in log if k == "estop"], ["pursuer"])
        self.assertEqual([n for k, n, _ in log if k == "land"], ["evader"])

    def test_episode_floor_ignores_virtual_opponent(self):
        c, _ = controller()
        c.opponent_flies = False
        c.cfg.episode.stop_on_evader_exit = c.cfg.episode.stop_on_timeout = c.cfg.episode.stop_on_capture = False
        own = SimpleNamespace(pos=np.array([0.0, 0.0, 1.0]), stamp=10.0)
        opponent = SimpleNamespace(pos=np.array([1.0, 0.0, 0.05]), stamp=10.0)
        self.assertFalse(c._episode_over(own, opponent, 10.0))
        self.assertIsNone(c._fault)


class LandingDurationTest(unittest.TestCase):

    def test_landing_time_grows_with_height(self):          # fixed 3 s from 3 m hit the floor at 2.5 m/s (CrazySim)
        c, log = controller()
        c.land_duration, c.land_speed = 3.0, 0.5
        high = FakeDrone("pursuer", log)
        high.get_state = lambda: SimpleNamespace(pos=np.array([0.0, 0.0, 3.1]))
        self.assertAlmostEqual(c._landing_duration(high), 5.6)       # (3.1 - 0.3) / 0.5: descent to the hover point
        self.assertAlmostEqual(c._landing_duration(c.own), 3.0)      # 1 m: the minimum applies
        nan = FakeDrone("evader", log)
        nan.get_state = lambda: SimpleNamespace(pos=np.array([0.0, 0.0, np.nan]))
        self.assertAlmostEqual(c._landing_duration(nan), 3.0)

class BrakeTest(unittest.TestCase):

    def test_moving_drone_is_ramped_to_rest_before_landing(self):    # CrazySim: landing from 1.5-3 m/s flipped drones
        c, log = controller()
        c.brake_decel = 20.0                                          # 2 m/s -> 0.1 s ramp
        c.own.get_state = lambda: SimpleNamespace(pos=np.array([0.0, 0.0, 1.0]), quat_wxyz=np.array([1.0, 0, 0, 0]),
                                                  lin_vel=np.array([2.0, 0.0, 0.0]))
        c.shutdown()
        vel = [v for k, n, v in log if k == "vel" and n == "pursuer"]
        self.assertGreaterEqual(len(vel), 3)
        self.assertLessEqual(vel[0][0], 2.0)
        self.assertEqual(vel[-1], (0.0, 0.0, 0.0))                    # ends at rest
        self.assertTrue(all(a[0] >= b[0] for a, b in zip(vel, vel[1:])))   # monotone ramp
        kinds = [(k, n) for k, n, _ in log]
        self.assertLess(max(i for i, x in enumerate(kinds) if x == ("vel", "pursuer")), kinds.index(("land", "pursuer")))
        self.assertNotIn(("vel", "evader"), kinds)                    # the evader was at rest: no braking

class ShutdownLandingTest(unittest.TestCase):

    def test_floor_is_checked_throughout_shutdown_after_capture(self):
        for phase in ("vel", "hold", "goto", "land"):
            for role in ("pursuer", "evader"):
                with self.subTest(phase=phase, role=role):
                    fault = Verdict() if phase == "hold" else None
                    if fault is not None:
                        fault.raise_to(Level.HOLD, "stale")
                    c, log = controller(fault)
                    c._outcome = "capture"
                    clock = [10.0]
                    low = [False]
                    crossed_at = []
                    drone = c.own if role == "pursuer" else c.opponent

                    def state():
                        return SimpleNamespace(pos=np.array([0.0, 0.0, 0.05 if low[0] else 1.0]),
                                               quat_wxyz=np.array([1.0, 0, 0, 0]),
                                               lin_vel=np.array([2.0 if phase == "vel" else 0.0, 0, 0]))

                    def sleep(duration):
                        clock[0] += duration
                        if any(k == phase and n == role for k, n, _ in log):
                            low[0] = True
                            if not crossed_at:
                                crossed_at.append(clock[0])

                    drone.get_state = state
                    with patch.object(pec_ctrl.time, "time", side_effect=lambda: clock[0]), \
                            patch.object(pec_ctrl.time, "sleep", side_effect=sleep):
                        c.shutdown()
                    stops = [i for i, (k, n, _) in enumerate(log) if k == "estop" and n == role]
                    self.assertEqual(len(stops), 1)
                    self.assertLessEqual(log[stops[0]][2] - crossed_at[0], c.control_dt + 1e-9)
                    self.assertFalse(any(k in ("vel", "hold", "goto", "land") and n == role
                                         for k, n, _ in log[stops[0] + 1:]))
                    self.assertEqual(c._outcome, "capture")
                    self.assertFalse(any(k == "estop" and n != role for k, n, _ in log))

    def test_descends_with_go_to_then_lands_from_the_hover_point(self):
        # A finished firmware landing cuts the motors wherever the drone is (planner.c), so the long descent is a
        # go_to to LAND_HOVER_Z, which keeps holding, and only the last LAND_HOVER_Z metres are a landing.
        c, log = controller()
        c.shutdown()
        for name in ("pursuer", "evader"):
            kinds = [k for k, n, _ in log if n == name]
            self.assertLess(kinds.index("goto"), kinds.index("land"))
            goto = next(v for k, n, v in log if k == "goto" and n == name)
            self.assertAlmostEqual(goto[2], pec_ctrl.LAND_HOVER_Z)

class TakeoffTest(unittest.TestCase):

    def test_both_takeoffs_start_together(self):          # was takeoff(own) -> 4 s -> takeoff(opponent)
        c, log = controller()
        c.opponent_flies = True
        c.mocap_config = SimpleNamespace(enabled=False)
        c.safety_cfg = SafetyConfig(enabled=False)
        c.cfg = SimpleNamespace(flight=SimpleNamespace(takeoff_height=1.0, takeoff_duration=0.1))
        start = time.time()
        c._arm_and_takeoff()
        takeoffs = [(n, t) for kind, n, t in log if kind == "takeoff"]
        self.assertEqual([n for n, _ in takeoffs], ["pursuer", "evader"])
        self.assertLess(abs(takeoffs[1][1] - takeoffs[0][1]), 0.05)
        self.assertGreaterEqual(time.time() - start, 1.1)  # waits once for the take-off to finish


class CppmSetpointTest(unittest.TestCase):
    """The cppmEmu packet must give the firmware the setpoint the legacy send_setpoint gave, without its thrust cap."""

    @staticmethod
    def firmware_decode(data):
        """crtp_commander_generic.c cppmEmuDecoder (rate mode) -> (attitudeRate roll, pitch, yaw, thrust)."""
        import struct
        kind, hdr, r, p, y, t, aux0 = struct.unpack('<BBHHHHH', data)
        assert (kind, hdr & 0x0F, aux0 < 1500) == (3, 1, True)
        unit = lambda ch, mid, rng: (ch - mid) / rng  # noqa: E731
        return (-unit(r, 1500, 500) * 720.0, -unit(p, 1500, 500) * 720.0, -unit(y, 1500, 500) * 400.0,
                max(0.0, unit(t, 1000, 1000) * 65535.0))

    def test_matches_legacy_setpoint_without_thrust_cap(self):
        import drone
        for roll, pitch, yaw, thrust in ((0, 0, 0, 0), (40, -25, 90, 30000), (-180, 180, -180, 65535), (12.3, -7.7, 3.3, 62000)):
            r, p, y, t = self.firmware_decode(drone.cppm_rate_setpoint(roll, pitch, yaw, thrust))
            # legacy path (send_setpoint + crtp_commander_rpyt.c, rate mode): roll, -pitch, -yaw, min(thrust, 60000)
            self.assertAlmostEqual(r, roll, delta=0.8)
            self.assertAlmostEqual(p, -pitch, delta=0.8)
            self.assertAlmostEqual(y, -yaw, delta=0.5)
            self.assertAlmostEqual(t, thrust, delta=40.0)                 # full range: 65535 stays 65535



class EstimatorHeadingTest(unittest.TestCase):
    """arm() starts the Kalman filter at the mocap pose and heading (only the position is streamed afterwards)."""

    def drone(self, mocap_yaw_deg, estimator_yaw_deg):
        import drone
        d = drone.CrazyflieDrone.__new__(drone.CrazyflieDrone)
        d.name, self.params = "pursuer", []
        d.cf = SimpleNamespace(param=SimpleNamespace(set_value=lambda k, v: self.params.append((k, v))))
        h = np.radians(mocap_yaw_deg) / 2
        d.mocap_pose = SimpleNamespace(position=(0.5, -1.0, 0.02), quat_xyzw=(0.0, 0.0, np.sin(h), np.cos(h)))
        e = np.radians(estimator_yaw_deg) / 2
        d.get_state = lambda: SimpleNamespace(quat_wxyz=np.array([np.cos(e), 0.0, 0.0, np.sin(e)]))
        return d

    def test_reset_sets_initial_pose_then_resets(self):
        d = self.drone(120.0, 0.0)
        yaw = d._reset_estimator_to_mocap()
        self.assertAlmostEqual(np.degrees(yaw), 120.0)
        names = [k for k, _ in self.params]
        self.assertEqual(names, ["kalman.initialX", "kalman.initialY", "kalman.initialZ", "kalman.initialYaw",
                                 "kalman.resetEstimation", "kalman.resetEstimation"])
        self.assertAlmostEqual(dict(self.params[:4])["kalman.initialYaw"], np.radians(120.0))   # firmware: radians
        self.assertEqual([v for k, v in self.params[4:]], [1, 0])

    def test_heading_error_wraps(self):
        self.assertAlmostEqual(self.drone(175.0, -175.0)._heading_error_deg(np.radians(175.0)), 10.0)
        self.assertIsNone(self.drone(0.0, 0.0)._heading_error_deg(None))

    def test_preflight_refuses_a_wrong_heading(self):
        c, log = controller()
        c.opponent_flies = False
        c.mocap_config = SimpleNamespace(enabled=True)
        c.safety_cfg = SafetyConfig(enabled=False, strict_preflight=True)
        c.cfg = SimpleNamespace(flight=SimpleNamespace(takeoff_height=1.0, takeoff_duration=0.1))
        c.own.arm = lambda: {"mocap_aligned": True, "estimator_converged": True, "heading_error_deg": 35.0}
        with self.assertRaisesRegex(RuntimeError, "heading"):
            c._arm_and_takeoff()
        self.assertNotIn("takeoff", [k for k, _, _ in log])


class VelocitySetpointTest(unittest.TestCase):
    """drone.CrazyflieDrone.send_velocity_world_setpoint builds cflib's packet itself (no DeprecationWarning)."""

    def packet(self, protocol):
        import struct
        import warnings
        import drone
        sent = []
        d = drone.CrazyflieDrone.__new__(drone.CrazyflieDrone)
        d.cf = SimpleNamespace(platform=SimpleNamespace(get_protocol_version=lambda: protocol),
                               send_packet=sent.append)
        from unittest import mock
        with warnings.catch_warnings(), \
                mock.patch.object(drone, "CRTPPacket", SimpleNamespace), \
                mock.patch.object(drone, "CRTPPort", SimpleNamespace(COMMANDER_GENERIC=7)):
            warnings.simplefilter("error")                     # any warning fails the test
            d.send_velocity_world_setpoint(0.5, -1.0, 0.25, 30.0)
        return struct.unpack("<Bffff", sent[0].data)

    def test_legacy_firmware_gets_type_1_with_negated_yaw_rate(self):
        self.assertEqual(self.packet(7), (1, 0.5, -1.0, 0.25, -30.0))     # as cflib's legacy branch

    def test_new_firmware_gets_type_8(self):
        self.assertEqual(self.packet(9), (8, 0.5, -1.0, 0.25, 30.0))

if __name__ == "__main__":
    unittest.main()
