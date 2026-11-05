from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest

import torch

_THESIS = Path(__file__).resolve().parents[1] / "scripts/experiments/thesis"
sys.path[:0] = [str(_THESIS), str(_THESIS / "exps")]
import f5_acc_sp as f5  # noqa: E402
from a2_nfacc import A2A  # noqa: E402


def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


class F5DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The retrained estimator (amendment 1) does not exist before F5 runs: stand in with A2a's selection files.
        cls.tmp = tempfile.mkdtemp()
        for f in ("horizon_choice.json", "tuned_rls.json"):
            shutil.copy(os.path.join(A2A, f), cls.tmp)
        cls._est, f5.EST = f5.EST, cls.tmp

    @classmethod
    def tearDownClass(cls):
        f5.EST = cls._est
        shutil.rmtree(cls.tmp)

    def test_new_pursuers_use_a2b_rls_estimator_and_evader_mirrors_it(self):
        pov = f5.rls_overrides()
        self.assertIn("task.adapt.enabled=true", pov)
        self.assertNotIn(f5.FREEZE, pov)                      # trained with RLS on (B1)
        ev = f5._roles()["evader"]
        self.assertIn("task.adapt.opp_side=true", ev)        # frozen pursuers keep their estimator
        self.assertNotIn("task.adapt.enabled=true", ev)      # the evader itself stays plain

    def test_rls_overrides_use_the_retrained_estimator(self):
        model = [o for o in f5.rls_overrides() if o.startswith("task.adapt.model_path=")]
        self.assertEqual(len(model), 1)
        self.assertTrue(model[0].split("=", 1)[1].startswith(f5.EST + os.sep), model)

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

    def test_paused_exp2_leaves_only_exp1_pursuers(self):
        self.assertEqual(f5.PAUSED_ARMS, {"qd_dpp_rls"})
        arms = [a for a, *_ in f5.active_pursuers()]
        self.assertEqual(sorted(arms), sorted(["uniform_rls"] * 2 + ["uniform_plain"] * 2 + ["uniform_fd"]))
        keys = [(a, ca, b, cb) for a, ca, b, cb in f5.CONTRASTS if {a, b} <= f5.EXP1_ARMS]
        self.assertIn(("uniform_rls", "clean", "uniform_plain", "clean"), keys)
        self.assertIn(("uniform_rls", "noisy", "uniform_rls", "noisy_frozen"), keys)

    def test_estimator_evaders_share_no_checkpoint_with_evaluated_ones(self):
        plain, acc = f5.independent_evaders()
        self.assertEqual((len(plain), len(acc)), (7, 4))      # F4 seed-2 stage-1 evader is shared by nf and ema
        mine = {_md5(e["checkpoint_path"]) for e in plain + acc}
        self.assertEqual(len(mine), 11)
        evaluated = {_md5(e["checkpoint_path"]) for a, _, c, _, _ in f5.pursuers() if a not in f5.NEW
                     and a != "uniform_fd" for e in f5._policies(c, "evader")}
        split = f5.load_split()
        pool = {_md5(e["checkpoint_path"]) for e in split["train"] + f5.test_set(split)
                if e.get("checkpoint_path") and os.path.isfile(e["checkpoint_path"])}
        self.assertFalse(mine & evaluated)
        self.assertFalse(mine & pool)

    def test_merge_traj_truncates_to_the_shortest_recording(self):
        def rec(T1, M, opp):
            return {"z": torch.randn(T1, M, 9), "own_pos": torch.randn(T1, M, 3), "done": torch.zeros(T1 - 1, M,
                    dtype=torch.bool), "envs": torch.arange(M), "opponent": [opp] * M, "step_dt": 0.02}
        a, b = rec(50, 3, "a"), rec(40, 2, "b")
        paths = [os.path.join(self.tmp, f"{k}.pt") for k in "ab"]
        torch.save(a, paths[0])
        torch.save(b, paths[1])
        m = torch.load(f5.merge_traj(paths, os.path.join(self.tmp, "m.pt")))
        self.assertEqual(tuple(m["z"].shape), (40, 5, 9))
        self.assertEqual(tuple(m["done"].shape), (39, 5))
        self.assertEqual(m["opponent"], ["a"] * 3 + ["b"] * 2)
        self.assertTrue(torch.equal(m["z"][:, 3:], b["z"]))
        self.assertTrue(torch.equal(m["z"][:, :3], a["z"][:40]))

    def test_campaign_metrics_exclude_stage1_evaders(self):
        c = "camp"
        prefix = os.path.join(self.tmp, "ev")
        rates = {"camp/evader:s1": 0.0, "camp/evader:s3": 0.6, "other/evader:s1": 0.0, "other/evader:s3": 0.8,
                 "other/evader:s5": 1.0}
        for suffix, m in (("_train.json", {"t": 0.5}), ("_test.json", {"h": 0.4}), ("_campaign_evaders.json", rates)):
            with open(prefix + suffix, "w") as f:
                json.dump({"matchups": {k: {"neutralization_rate": v} for k, v in m.items()}}, f)
        split = {"train": [{"id": "t"}], "heldout_rl": [{"id": "h"}], "heldout_scripted": []}
        r = f5._metrics(prefix, c, split, [], {}, "neutralization_rate")
        self.assertAlmostEqual(r["own_campaign_evaders"], 0.6)
        self.assertAlmostEqual(r["other_campaign_evaders"], 0.9)


if __name__ == "__main__":
    unittest.main()
