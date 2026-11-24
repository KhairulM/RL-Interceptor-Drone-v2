"""Thesis F16 driver design: banks and weights, contrasts, coverage LP, pilot rule, mixtures, bootstrap isolation."""

import itertools
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts/experiments/thesis"))
sys.path.insert(0, os.path.join(ROOT, "scripts/experiments/thesis/exps"))
import common  # noqa: E402
import f16_roncero_population as f16  # noqa: E402


class BankTests(unittest.TestCase):
    def test_cross_bank_excludes_own_seed_and_bootstrap(self):
        for s in f16.SEEDS:
            ids = f16.cross_ids(s)
            self.assertEqual(len(ids), len(f16.ARMS) * 2 * 12)
            self.assertFalse(any(e["campaign"].endswith(f"-s{s}") for e in ids))
            self.assertEqual(sorted({e["stage"] for e in ids}), list(range(3, 26, 2)))
            self.assertEqual({e["arm"] for e in ids}, set(f16.ARMS))

    def test_cross_weights_arms_then_campaigns_then_stages(self):
        ids = [{"id": f"{a}{c}{st}", "arm": a, "campaign": f"{a}{c}", "stage": st}
               for a, n_c, n_st in (("x", 1, 2), ("y", 2, 3)) for c in range(n_c) for st in range(n_st)]
        w = f16.cross_weights(ids)
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertAlmostEqual(sum(v for k, v in w.items() if k.startswith("x")), 0.5)
        self.assertAlmostEqual(w["x00"], 0.25)
        self.assertAlmostEqual(w["y10"], 0.5 / 2 / 3)

    def test_attack_ids_seeds(self):
        a = f16.attack_ids(100)
        self.assertEqual(len(a), 2 * len(f16.ARMS))
        self.assertEqual(sorted({x["seed"] for x in a}), [2700, 3700])


class AnalysisTests(unittest.TestCase):
    def test_contrast_rule(self):
        self.assertEqual(f16.contrast({0: .1, 1: .06, 2: .01}, {0: 0, 1: 0, 2: 0})["decision"], "notable gain")
        self.assertEqual(f16.contrast({0: .2, 1: .06, 2: -.01}, {0: 0, 1: 0, 2: 0})["decision"], "inconclusive")
        self.assertEqual(f16.contrast({0: -.1, 1: -.06, 2: -.01}, {0: 0, 1: 0, 2: 0})["decision"], "notable decrease")
        self.assertEqual(f16.contrast({0: .04, 1: .04, 2: .04}, {0: 0, 1: 0, 2: 0})["decision"], "inconclusive")

    def test_maximin_matches_grid(self):
        rng = np.random.default_rng(0)
        J = rng.uniform(size=(3, 4))
        v, x = f16.maximin(J)
        grid = max(min(np.array(w) @ J) for w in itertools.product(np.linspace(0, 1, 101), repeat=3)
                   if abs(sum(w) - 1) < 1e-9)
        self.assertGreaterEqual(v, grid - 1e-9)
        self.assertLess(v - grid, 0.02)
        self.assertAlmostEqual(x.sum(), 1.0)
        self.assertAlmostEqual(min(x @ J), v, places=6)

    def test_mixture_is_exact_expectation(self):
        per = {0: {"a": 0.2, "b": 1.0}, 2: {"a": 0.6, "b": 0.0}}
        m = f16.mix_opp(per, {0: 0.25, 2: 0.75})
        self.assertAlmostEqual(m["a"], 0.5)
        self.assertAlmostEqual(m["b"], 0.25)
        self.assertAlmostEqual(f16.summarise({"a": 1.0, "b": 0.0, "c": 5.0}, ["a", "b"], {"a": 3, "b": 1}), 0.75)

    def test_arms_after_psd_drop(self):
        self.assertEqual(f16.ARMS, ("hedge", "qd_dpp", "psro"))
        self.assertEqual(f16.CONTRASTS, (("qd_dpp", "hedge"), ("psro", "hedge")))
        self.assertEqual(dict(x.split("=", 1) for x in f16.psro_cli("c", "psro", 0, 0.0))["hull.lam"], "0.0")

    def test_expected_f8_cli_matches_registered_campaign(self):
        cli = f16.expected_f8_cli(200)
        self.assertEqual(cli["frames_per_stage"], "50000000")
        self.assertEqual(cli["pursuer_sampler"], "hedge")
        self.assertEqual(cli["task_overrides"], "[task.reward_mode=roncero,algo.gae_cut_at_truncation=true]")
        qd = dict(x.split("=", 1) for x in f16.qd_cli("c", 100))
        self.assertEqual((qd["pursuer_sampler"], qd["evader_sampler"], qd["seed"]), ("qd_dpp", "qd_dpp", "100"))
        for k in ("frames_per_stage", "hedge_gamma", "uniform_mix", "warm_start", "task_overrides"):
            self.assertEqual(qd.get(k), {**cli, "hedge_gamma": None}.get(k) if k == "hedge_gamma" else cli[k], k)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.old = common.SELFPLAY_OUT
        common.SELFPLAY_OUT = self.root
        src = os.path.join(self.root, "src")
        for d in ("db", "checkpoints", "stages"):
            os.makedirs(os.path.join(src, d))
        pools = {"pursuer": [{"id": "pursuer:heuristic0", "kind": "heuristic", "checkpoint_path": None,
                              "metadata": {"heuristic": "apf"}}],
                 "evader": [{"id": "evader:heuristic0", "kind": "heuristic", "checkpoint_path": None,
                             "metadata": {"heuristic": "hover"}}]}
        mt = {}
        for st in range(4):
            role = "pursuer" if st % 2 == 0 else "evader"
            ck = os.path.join(src, "checkpoints", f"{role}_stage{st}.pt")
            torch.save({"w": torch.full((3,), float(st))}, ck)
            pid = f"{role}:s{st + 1}"
            pools[role].append({"id": pid, "kind": "rl", "checkpoint_path": ck, "metadata": {"stage": st}})
            opps = ["evader:heuristic0"] if st == 0 else ["pursuer:heuristic0"] if st == 1 else \
                [p["id"] for p in pools["evader" if role == "pursuer" else "pursuer"] if p["kind"] == "rl"]
            ev = {}
            for o in opps:
                key = f"{pid}|{o}" if role == "pursuer" else f"{o}|{pid}"
                mt[key] = ev[o] = {"win_rate": 0.1 * st, "episodes": 10 + st}
            for part in ("train_opp", "train_out", "eval_opp"):
                json.dump({}, open(os.path.join(src, "stages", f"stage{st}_{role}_{part}.json"), "w"))
            json.dump({"matchups": ev}, open(os.path.join(src, "stages", f"stage{st}_{role}_eval_out.json"), "w"))
        for r in pools:
            json.dump({"role": r, "policies": pools[r]}, open(os.path.join(src, "db", f"{r}_pool.json"), "w"))
        json.dump({"matchups": mt}, open(os.path.join(src, "db", "matchups.json"), "w"))

    def tearDown(self):
        common.SELFPLAY_OUT = self.old
        shutil.rmtree(self.root)

    def test_clone_contains_only_stage_0_and_1(self):
        b = f16.clone_bootstrap("src", "dst")
        out = os.path.join(self.root, "dst")
        self.assertEqual(sorted(os.listdir(os.path.join(out, "checkpoints"))),
                         ["evader_stage1.pt", "pursuer_stage0.pt"])
        for r, st in (("pursuer", 0), ("evader", 1)):
            pol = json.load(open(os.path.join(out, "db", f"{r}_pool.json")))["policies"]
            rl = [p for p in pol if p["kind"] == "rl"]
            self.assertEqual([p["metadata"]["stage"] for p in rl], [st])
            self.assertTrue(rl[0]["checkpoint_path"].startswith(out))
        mt = json.load(open(os.path.join(out, "db", "matchups.json")))["matchups"]
        self.assertEqual(sorted(mt), ["pursuer:heuristic0|evader:s2", "pursuer:s1|evader:heuristic0"])
        self.assertEqual(sorted(os.listdir(os.path.join(out, "stages")))[0], "stage0_pursuer_eval_opp.json")
        self.assertFalse(any("stage2" in f or "stage3" in f for f in os.listdir(os.path.join(out, "stages"))))
        self.assertEqual(f16.clone_bootstrap("src", "dst"), b)               # idempotent
        torch.save({"w": torch.ones(3) * 9}, os.path.join(out, "checkpoints", "pursuer_stage0.pt"))
        with self.assertRaises(RuntimeError):                               # corrupt state is detected
            f16.clone_bootstrap("src", "dst")


if __name__ == "__main__":
    unittest.main()
