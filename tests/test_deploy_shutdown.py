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
             "cflib.crazyflie.log", "cflib.crazyflie.syncLogger", "NatNetClient"):
    _stub(name, ok=lambda: True, init=lambda: None, create_node=_Any, TransformStamped=_Any, PoseStamped=_Any,
          TFMessage=_Any, Crazyflie=_Any, LogConfig=_Any, SyncLogger=_Any, NatNetClient=_Any, publisher=_Any())
sys.path.insert(0, str(DEPLOY))
import pursuit_evasion_controller as pec_ctrl  # noqa: E402
from safety import Level, SafetyConfig, Verdict  # noqa: E402


class FakeDrone(pec_ctrl.CrazyflieDrone):
    def __init__(self, name, log):
        self.name, self.log, self.armed, self.connected = name, log, True, True

    def land_async(self, duration=3.0):
        self.log.append(("land", self.name, time.time()))
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


if __name__ == "__main__":
    unittest.main()
