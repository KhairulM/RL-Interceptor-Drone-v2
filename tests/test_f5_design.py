from pathlib import Path
import sys
import unittest

_THESIS = Path(__file__).resolve().parents[1] / "scripts/experiments/thesis"
sys.path[:0] = [str(_THESIS), str(_THESIS / "exps")]
import f5_acc_sp as f5  # noqa: E402


class F5DesignTests(unittest.TestCase):
    def test_new_pursuers_use_a2b_rls_estimator_and_evader_mirrors_it(self):
        pov = f5.rls_overrides()
        self.assertIn("task.adapt.enabled=true", pov)
        self.assertNotIn(f5.FREEZE, pov)                      # trained with RLS on (B1)
        ev = f5._roles()["evader"]
        self.assertIn("task.adapt.opp_side=true", ev)        # frozen pursuers keep their estimator
        self.assertNotIn("task.adapt.enabled=true", ev)      # the evader itself stays plain

    def test_rls_frozen_only_at_eval_and_only_for_f5_pursuers(self):
        pov = f5.rls_overrides()
        c = f5.conditions("uniform_rls", pov)
        self.assertEqual(set(c), {"clean", "noisy", "clean_frozen", "noisy_frozen"})
        self.assertIn(f5.FREEZE, c["noisy_frozen"])
        self.assertIn(f5.NOISY, c["noisy_frozen"])
        self.assertEqual(set(f5.conditions("hedge_plain", [])), {"clean", "noisy"})

    def test_pursuer_roster_pairs_two_seeds_and_reuses_f3_exploiters(self):
        P = f5.pursuers()
        arms = [a for a, *_ in P]
        for a in ("uniform_rls", "qd_dpp_rls", "uniform_plain", "hedge_plain", "pfsp_plain", "qd_dpp_plain"):
            self.assertEqual(arms.count(a), 2, a)
        self.assertEqual(arms.count("uniform_fd"), 1)
        for a, _, c, _, ex in P:
            if a.endswith("_plain"):
                self.assertTrue(ex.endswith(f"F3_sampler_sp/ckpt/exploiter_{c}.pt"), ex)


if __name__ == "__main__":
    unittest.main()
