"""Deploy safety layer (scripts/deploy/safety.py; SAFETY_PLAN.md)."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

_P = Path(__file__).resolve().parents[1] / "scripts/deploy/safety.py"
_spec = importlib.util.spec_from_file_location("deploy_safety", _P)
sf = importlib.util.module_from_spec(_spec)
sys.modules["deploy_safety"] = sf          # dataclasses look the module up
_spec.loader.exec_module(sf)
L = sf.Level
ARENA = sf.Arena(origin=np.zeros(3), half_xy=2.0, z_min=0.1, z_max=3.0)


def st(pos=(0, 0, 1), quat=(1, 0, 0, 0), vel=(0, 0, 0), ang=(0, 0, 0), stamp=10.0):
    return SimpleNamespace(pos=np.array(pos, float), quat_wxyz=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(ang, float), stamp=stamp)


def roll_quat(deg):
    h = math.radians(deg) / 2
    return (math.cos(h), math.sin(h), 0, 0)


class MonitorTest(unittest.TestCase):

    def mon(self, **kw):
        return sf.SafetyMonitor(sf.SafetyConfig(**kw), ARENA)

    def test_nominal_is_ok(self):
        v = self.mon().check({"pursuer": st(), "evader": st(pos=(1, 0, 1))}, now=10.05)
        self.assertEqual(v.level, L.OK, v.reasons)

    def test_stale_opponent_holds_then_lands(self):        # bug B2: the opponent's staleness is checked too
        m = self.mon()
        self.assertEqual(m.check({"pursuer": st(stamp=10.2), "evader": st(pos=(1, 0, 1))}, now=10.25).level, L.HOLD)
        self.assertEqual(m.check({"pursuer": st(stamp=10.5), "evader": st(pos=(1, 0, 1))}, now=10.55).level, L.LAND)

    def test_mocap_stale_and_rejects_and_swap(self):       # bug B3
        m = self.mon()
        ok = {"pursuer": st(), "evader": st(pos=(1, 0, 1))}
        self.assertEqual(m.check(ok, 10.0, mocap={"pursuer": {"age": 0.3}}).level, L.HOLD)
        self.assertEqual(self.mon().check(ok, 10.0, mocap={"pursuer": {"age": 0.6}}).level, L.LAND)
        self.assertEqual(self.mon().check(ok, 10.0, mocap={"evader": {"age": 0.0, "rejects": 5}}).level, L.HOLD)
        self.assertEqual(self.mon().check(ok, 10.0, mocap={"evader": {"age": 0.0, "swap": True}}).level, L.LAND)

    def test_tilt_and_body_rate_are_emergencies(self):
        v = self.mon().check({"pursuer": st(quat=roll_quat(70)), "evader": st(pos=(1, 0, 1))}, 10.0)
        self.assertEqual((v.level, v.emergency_roles), (L.EMERGENCY, ["pursuer"]))
        v = self.mon().check({"pursuer": st(ang=(0, 0, math.radians(800))), "evader": st(pos=(1, 0, 1))}, 10.0)
        self.assertEqual(v.level, L.EMERGENCY)
        self.assertAlmostEqual(sf.tilt_deg(roll_quat(30)), 30.0, places=6)

    def test_nonfinite_state_and_action(self):
        self.assertEqual(self.mon().check({"pursuer": st(pos=(np.nan, 0, 1))}, 10.0).level, L.EMERGENCY)
        self.assertEqual(self.mon().check({"pursuer": st()}, 10.0, action_finite=False).level, L.HOLD)

    def test_geofence_both_drones_and_predictive(self):
        self.assertEqual(self.mon().check({"pursuer": st(pos=(2.4, 0, 1))}, 10.0).level, L.LAND)   # pursuer fenced
        self.assertEqual(self.mon().check({"pursuer": st(pos=(2.2, 0, 1))}, 10.0).level, L.OK)    # inside margin
        self.assertEqual(self.mon().check({"pursuer": st(pos=(2.2, 0, 1), vel=(2, 0, 0))}, 10.0).level, L.HOLD)
        self.assertEqual(self.mon().check({"pursuer": st(pos=(0, 0, 3.4))}, 10.0).level, L.LAND)   # ceiling

    def test_floor_and_fast_descent(self):
        self.assertEqual(self.mon().check({"pursuer": st(pos=(0, 0, 0.05))}, 10.0).level, L.LAND)
        self.assertEqual(self.mon().check({"pursuer": st(pos=(0, 0, 0.4), vel=(0, 0, -3))}, 10.0).level, L.HOLD)

    def test_separation_optional(self):
        s = {"pursuer": st(), "evader": st(pos=(0.15, 0, 1))}
        self.assertEqual(self.mon().check(s, 10.0).level, L.OK)
        self.assertEqual(self.mon(min_separation=0.2).check(s, 10.0).level, L.HOLD)

    def test_latch_never_decreases(self):
        m = self.mon()
        m.check({"pursuer": st(pos=(0, 0, 0.05))}, 10.0)
        self.assertEqual(m.check({"pursuer": st()}, 10.05).level, L.LAND)

    def test_preflight(self):
        m = self.mon()
        ok = {"pursuer": st(pos=(-1, 0, 0)), "evader": st(pos=(1, 0, 0))}
        self.assertEqual(m.preflight(ok, 10.0), [])
        self.assertEqual(m.preflight({"pursuer": st(pos=(-1, 0, 0)), "evader": st(pos=(-0.8, 0, 0))}, 10.0), [])  # no separation check
        self.assertTrue(m.preflight({"pursuer": st(pos=(3, 0, 0)), "evader": st(pos=(1, 0, 0))}, 10.0))
        self.assertTrue(m.preflight({"pursuer": None, "evader": st(pos=(1, 0, 0))}, 10.0))

    def test_config_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            sf.SafetyConfig.from_dict({"geofence_margn": 0.3})


class MocapGateTest(unittest.TestCase):

    def test_jump_rejected_and_counted(self):
        g = sf.MocapGate(sf.SafetyConfig())
        self.assertTrue(g.accept(31, (0, 0, 1), 0.0))
        self.assertTrue(g.accept(31, (0.05, 0, 1), 0.0167))            # 3 m/s: plausible
        self.assertFalse(g.accept(31, (1.0, 0, 1), 0.0333))            # 0.95 m in 17 ms: jump
        self.assertEqual(g.health(31, 0.0333)["rejects"], 1)
        self.assertTrue(g.accept(31, (0.1, 0, 1), 0.05))               # back on track, counter reset
        self.assertEqual(g.health(31, 0.05)["rejects"], 0)

    def test_swap_detected(self):
        g = sf.MocapGate(sf.SafetyConfig())
        g.accept(31, (0, 0, 1), 0.0)
        g.accept(32, (1, 0, 1), 0.0)
        self.assertFalse(g.accept(31, (1.0, 0, 1), 0.5))               # body 31 now reads body 32's place
        self.assertTrue(g.health(31, 0.5)["swap"])

    def test_age(self):
        g = sf.MocapGate(sf.SafetyConfig())
        self.assertEqual(g.health(31, 1.0)["age"], float("inf"))
        g.accept(31, (0, 0, 1), 1.0)
        self.assertAlmostEqual(g.health(31, 1.3)["age"], 0.3)


if __name__ == "__main__":
    unittest.main()
