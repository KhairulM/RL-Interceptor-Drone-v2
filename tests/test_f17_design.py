from pathlib import Path
import sys
import unittest
from unittest.mock import patch

_THESIS = Path(__file__).resolve().parents[1] / "scripts/experiments/thesis"
sys.path[:0] = [str(_THESIS), str(_THESIS / "exps")]
import f5_acc_sp as f5  # noqa: E402
import f8_roncero_sp as f8  # noqa: E402
import f14_generalization_sp as f14  # noqa: E402
import f15_switch_baseline as f15  # noqa: E402
import f17_roncero_acc as f17  # noqa: E402


def _ev(c, s):
    return {"id": f"{c}/evader:s{s}", "kind": "rl", "checkpoint_path": f"{c}_{s}.pt", "metadata": {}}


class F17DesignTests(unittest.TestCase):
    def test_campaigns_match_f8_except_pursuer_estimator(self):
        self.assertEqual(f17.SEEDS, (0, 100, 200))
        self.assertEqual(len(set(f17.CAMPS)), 3)
        for seed in f17.SEEDS:
            cfg = f17.cfg(seed)
            self.assertEqual((cfg["seed"], cfg["n_stages"], cfg["frames_per_stage"]), (seed, 26, 50_000_000))
            self.assertEqual({k: v for k, v in cfg.items() if k != "seed"},
                             {k: v for k, v in f8.arm_cfg(0).items() if k != "seed"})
        roles = f14._roles("rls")
        self.assertIn("task.adapt.enabled=true", roles["pursuer"])
        self.assertNotIn(f5.FREEZE, roles["pursuer"])
        self.assertIn("task.adapt.opp_side=true", roles["evader"])

    def test_exploiter_seeds_are_new(self):
        used = {970, 971, 972, 960, 961, 962} | {1100 + s for s in f14.SEEDS} | {1200 + s for s in f14.SEEDS}
        new = {f17.OWN_EXPLOITER_SEED + s for s in f17.SEEDS} | {f17.THESIS_EXPLOITER_SEED + s for s in f17.SEEDS}
        self.assertEqual(len(new), 6)
        self.assertFalse(new & used)

    def test_eval_jobs_cross_bank_excludes_own_campaign(self):
        exploiters = []

        def fake_exploiter(path, pid, c, ck, rules, seed, opponent_extra=None):
            exploiters.append((rules, seed, opponent_extra))
            return [{"id": path}]

        with patch.object(f14, "_ck", return_value="p.pt"), \
                patch.object(f14, "_evaders", return_value=[_ev(c, 1) for c in f17.CAMPS]), \
                patch.object(f14, "_exploiter", side_effect=fake_exploiter), \
                patch.object(f17.f8d, "_finals", return_value=[]):
            jobs = f17._eval_jobs()
        self.assertEqual(len(jobs), 21)
        for c, group, opps, _, extra in jobs:
            self.assertEqual(f5.FREEZE in extra, group.startswith("frozen_"))
            if group.endswith("cross"):
                self.assertEqual({p["id"].split("/")[0] for p in opps}, set(f17.CAMPS) - {c})
        self.assertEqual(sum(r == f8.RULES for r, _, _ in exploiters), 3)
        self.assertTrue(all("task.adapt.opp_side=true" in o for _, _, o in exploiters))

    def test_switch_jobs_reuse_f15_design_for_live_and_frozen(self):
        def fake_pair(seed, direction):
            a, b = _ev("f8", seed), _ev("f10", seed)
            return (a, b) if direction == "f8_to_f10" else (b, a)

        with patch.object(f14, "_ck", return_value="p.pt"), patch.object(f14, "_switch_pair", side_effect=fake_pair):
            jobs = f17._switch_jobs()
            f8_jobs = f15._baseline_jobs()
        self.assertEqual(len(jobs), 24)
        self.assertEqual(len({j["path"] for j in jobs}), 24)
        for j in jobs:
            ref = next(b for b in f8_jobs if b["group"] == f"f8_s{j['seed']}_" + j["path"].split(f"switch_{j['model']}_")[1][:-5])
            self.assertEqual((j["switch"], j["rng_seed"], j["policies"]), (ref["switch"], ref["rng_seed"], ref["policies"]))
            self.assertEqual(f5.FREEZE in j["extra"], j["model"] == "f17_frozen")

    def test_contrast_rule(self):
        self.assertEqual(f17._contrast([0.06, 0.04, 0.07])["verdict"], "notable_improvement")
        self.assertEqual(f17._contrast([0.2, -0.01, 0.1])["verdict"], "inconclusive")
        self.assertEqual(f17._contrast([-0.06, -0.06, -0.06])["verdict"], "notable_degradation")


if __name__ == "__main__":
    unittest.main()
