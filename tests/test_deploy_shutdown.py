"""PursuitEvasion controller shutdown (SAFETY_PLAN.md bug B1 and the fault responses), with ROS / cflib stubbed."""

import sys
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

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
        return SimpleNamespace(pos=np.array([0.0, 0.0, 1.0]), quat_wxyz=np.array([1.0, 0, 0, 0]))

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
    c.safety_cfg = SafetyConfig(hold_s=0.1)
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


if __name__ == "__main__":
    unittest.main()
