"""F6b design checks that need no Isaac run: the arm config and the pre-registered pilot rule."""

from pathlib import Path
import sys
import unittest

_THESIS = Path(__file__).resolve().parents[1] / "scripts/experiments/thesis"
sys.path[:0] = [str(_THESIS), str(_THESIS / "exps")]
import f6b_div_sp as f6b  # noqa: E402


def _row(J, timeout, d, bonus):
    return {"J_vs_training_pursuers": J, "timeout": timeout, "pursuer_crash": 0.0, "div_distance_end": d,
            "div_bonus_end": bonus}


class F6bDesignTests(unittest.TestCase):

    def test_arm_is_f3_hedge_plus_action_mode_term(self):
        cfg = f6b.arm_cfg(0.1, 0.07)
        self.assertEqual({k: v for k, v in cfg.items() if not k.startswith("diversity")}, f6b.SAMPLERS["hedge"])
        self.assertEqual((cfg["diversity_roles"], cfg["diversity_beta"], cfg["diversity_mode"], cfg["diversity_tau"],
                          cfg["diversity_centre"]), ("[evader]", 0.1, "action", 0.07, True))
        self.assertEqual(f6b.name(1), "thesis-F6b-hedge_divb-s1")
        self.assertEqual(f6b.PILOT_FROM, ("thesis-F3-hedge-both-s0", 5))
        self.assertEqual(f6b.PILOT_BETAS, [0.03, 0.1, 0.3])
        self.assertIn("diversity.measure=true", f6b._div_overrides(["a.pt"], 0.0, 1.0, measure=True))

    def test_pilot_rule(self):
        tau = 0.1
        rows = {0.0: _row(0.50, 0.15, 0.1, 0.6),
                0.03: _row(0.52, 0.20, 0.13, 0.7),      # kept and active
                0.1: _row(0.40, 0.24, 0.20, 0.85),      # kept and active (largest)
                0.3: _row(0.45, 0.30, 0.40, 0.95)}      # timeout +0.15: F6's failure mode -> dropped
        self.assertEqual(f6b.choose(rows, tau, (0.50, 0.15)), (0.1, [0.03, 0.1], [0.03, 0.1]))
        rows[0.1] = _row(0.40, 0.24, 0.11, 0.65)        # kept but not moved (< 1.25 tau)
        self.assertEqual(f6b.choose(rows, tau, (0.50, 0.15))[0], 0.03)
        rows[0.03] = _row(0.60, 0.15, 0.2, 0.8)         # J +0.10: dropped; 0.1 kept, not active -> largest kept
        self.assertEqual(f6b.choose(rows, tau, (0.50, 0.15))[0], 0.1)
        rows[0.1] = _row(0.60, 0.15, 0.2, 0.8)
        self.assertIsNone(f6b.choose(rows, tau, (0.50, 0.15))[0])      # none kept -> F6b stops
        rows[0.03] = _row(0.30, 0.15, 0.2, 0.8)
        self.assertEqual(f6b.choose(rows, tau, (0.50, 0.15))[0], 0.03)
        self.assertIsNone(f6b.choose(rows, tau, (0.18, 0.15))[0])      # stricter reference (F3 stage 5) drops it

    def test_tau_is_robust_to_a_late_spike(self):
        import tempfile, os
        vals = [0.07] * 60 + [0.3, 0.7, 1.1, 1.0, 0.8, 0.5, 0.3, 0.2]      # v1: mean of the last 10 = 0.52
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "log.txt")
            with open(log, "w") as f:
                f.write("".join(f"div_distance: {v}\n" for v in vals))
            self.assertAlmostEqual(f6b._tau(log), 0.07)


if __name__ == "__main__":
    unittest.main()
