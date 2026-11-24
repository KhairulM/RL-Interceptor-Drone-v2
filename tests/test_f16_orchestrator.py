"""Thesis F16 orchestration gates: PSRO / PSD generation schedule, frozen meta-strategies, payoff completion,
deterministic resume without duplicated policies or payoff records (fake Isaac worker)."""

import importlib.util
import json
import os
import shutil
import tempfile
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location("f16_psro_t", os.path.join(ROOT, "scripts/selfplay/f16_psro.py"))
f16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f16)


def _kv(overrides):
    out = {}
    for o in overrides:
        k, _, v = o.partition("=")
        out[k] = v
    return out


class FakeWorker:
    """Writes checkpoints / eval outputs; payoffs come from a fixed function of the two stages."""

    def __init__(self, fail_at=None, n=2100):
        self.calls, self.fail_at, self.n = [], fail_at, n

    def __call__(self, overrides):
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            self.calls.append(("fail", None))
            raise RuntimeError("simulated worker crash")
        kv = _kv(overrides)
        self.calls.append((kv["mode"], kv))
        if kv["mode"] == "train":
            torch.save({"stage": int(kv["stage"])}, kv["stage_ckpt"])
            with open(kv["stage_out"], "w") as f:
                json.dump({"checkpoint_path": kv["stage_ckpt"]}, f)
            return
        spec = json.load(open(kv["opponent_spec_path"]))
        stage = lambda path: int(path.split("stage")[-1].split(".")[0])
        a = stage(kv["eval_policy"])
        role = kv["task.role"]
        m = {}
        for p in spec["policies"]:
            o = stage(p["checkpoint_path"])
            ps, es = (a, o) if role == "pursuer" else (o, a)
            up = 10.0 * ((ps * 7 + es * 3) % 5 - 2) / 3
            ue = 10.0 * ((ps * 2 + es * 5) % 7 - 3) / 4
            m[p["id"]] = {"payoff": {"n": self.n, "mean_pursuer": up, "std_pursuer": 1.0, "mean_evader": ue,
                                     "std_evader": 1.0},
                          "win_rate": 0.5, "avg_time_to_capture": 1.0, "avg_min_distance": 0.2,
                          "avg_motor_effort": 0.1, "pursuer_crash_rate": 0.0, "outcomes": {"capture": 0.5},
                          "neutralization_rate": 0.5, "episodes": 1500}
        with open(kv["stage_out"], "w") as f:
            json.dump({"matchups": m}, f)


def _bootstrap(root, name):
    out = os.path.join(root, name)
    db = f16.PolicyDatabase(os.path.join(out, "db"))
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    for h in ("pure_pursuit", "apf"):
        db.add_policy("pursuer", None, kind="heuristic", metadata={"heuristic": h})
    for h in ("hover", "flee"):
        db.add_policy("evader", None, kind="heuristic", metadata={"heuristic": h})
    for role, st in (("pursuer", 0), ("evader", 1)):
        path = os.path.join(out, "checkpoints", f"{role}_stage{st}.pt")
        torch.save({"stage": st}, path)
        db.add_policy(role, path, kind="rl", metadata={"stage": st, "sha256": f16._ps.sha256_file(path)})


def _cfg(root, name, arm="psd", gens=3):
    lam = 0.1 if arm == "psd" else 0.0
    return f16.load_cfg([f"run_name={name}", f"out_root={root}", f"arm={arm}", f"generations={gens}",
                         "payoff_min_episodes=2048", "payoff_num_envs=4096", f"hull.lam={lam}",
                         "task_overrides=[task.reward_mode=roncero]"])


def _state(root, name):
    out = os.path.join(root, name)
    recs = [{k: v for k, v in r.items() if k != "time"}
            for r in f16._ps.PayoffStore(os.path.join(out, "payoffs")).records()]
    metas = {}
    for fn in sorted(os.listdir(os.path.join(out, "meta"))):
        m = json.load(open(os.path.join(out, "meta", fn)))
        metas[fn] = {k: m[k] for k in ("x", "y", "row_ids", "col_ids")}
    db = json.load(open(os.path.join(out, "db", "pursuer_pool.json"))), \
        json.load(open(os.path.join(out, "db", "evader_pool.json")))
    return recs, metas, [p["id"] for p in db[0]["policies"]], [p["id"] for p in db[1]["policies"]]


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_schedule_payoffs_and_frozen_meta(self):
        _bootstrap(self.root, "c")
        w = FakeWorker()
        final = f16.Campaign(_cfg(self.root, "c"), w).run()
        db = f16.PolicyDatabase(os.path.join(self.root, "c", "db"))
        self.assertEqual([f16._ts._policy_stage(p) for p in f16.rl_policies(db, "pursuer")], [0, 2, 4, 6])
        self.assertEqual([f16._ts._policy_stage(p) for p in f16.rl_policies(db, "evader")], [1, 3, 5, 7])
        self.assertEqual(len(final["row_ids"]), 4)
        trains = [kv for mode, kv in w.calls if mode == "train"]
        evals = [kv for mode, kv in w.calls if mode == "eval"]
        self.assertEqual([int(kv["stage"]) for kv in trains], [2, 3, 4, 5, 6, 7])
        self.assertEqual(len(evals), 1 + 2 * 3)           # 1x1 bootstrap cell, then one row + one column per gen
        for kv in evals:
            self.assertEqual(kv["eval_stochastic"], "true")
            self.assertEqual(kv["task.payoff_returns"], "true")
            self.assertEqual(kv["task.reward_mode"], "roncero")
        # each oracle: opponents = support of the frozen pre-generation meta-strategy, per-episode resampling;
        # PSD hull = this role's pre-generation policies only.
        for kv in trains:
            st, role = int(kv["stage"]), kv["task.role"]
            g = (st - (st % 2)) // 2
            meta = json.load(open(os.path.join(self.root, "c", "meta", f"gen{g}.json")))
            spec = json.load(open(kv["opponent_spec_path"]))
            probs = meta["y"] if role == "pursuer" else meta["x"]
            self.assertEqual(spec["probs"], {k: v for k, v in probs.items() if v > 0})
            self.assertIn("resample", spec)
            self.assertNotIn("stochastic", spec)             # oracle opponents act deterministically (as F8)
            refs = kv["hull_kl.ckpts"].strip("[]").split(",")
            self.assertEqual([int(r.split("stage")[-1].split(".")[0]) for r in refs],
                             list(range(0 if role == "pursuer" else 1, st, 2)))
            self.assertTrue(kv["init_policy"].endswith(f"{role}_stage{st - 2}.pt"))
        store = f16._ps.PayoffStore(os.path.join(self.root, "c", "payoffs"))
        UP, UE, N = store.matrices(final["row_ids"], final["col_ids"], 2048)
        self.assertTrue((N >= 2048).all())

    def test_resume_after_crashes_is_identical(self):
        _bootstrap(self.root, "ref")
        f16.Campaign(_cfg(self.root, "ref"), FakeWorker(n=1500)).run()     # 1500 < 2048: every cell needs a top-up
        _bootstrap(self.root, "res")
        for fail_at in (3, 1, 5, 2, 4):                      # crash at different points, then resume each time
            try:
                f16.Campaign(_cfg(self.root, "res"), FakeWorker(fail_at=fail_at, n=1500)).run()
            except RuntimeError:
                pass
        f16.Campaign(_cfg(self.root, "res"), FakeWorker(n=1500)).run()
        a, b = _state(self.root, "ref"), _state(self.root, "res")
        strip = lambda recs: sorted(json.dumps({k: v for k, v in r.items() if k not in ("agent_sha256",
                                    "opponent_sha256")}, sort_keys=True) for r in recs)
        self.assertEqual(strip(a[0]), strip(b[0]))
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[2:], b[2:])

    def test_guards(self):
        _bootstrap(self.root, "g")
        with self.assertRaises(ValueError):
            f16.Campaign(f16.load_cfg(["run_name=g", f"out_root={self.root}", "arm=psro", "hull.lam=0.1"]))
        with self.assertRaises(ValueError):
            f16.Campaign(f16.load_cfg(["run_name=g", f"out_root={self.root}", "arm=psd", "hull.lam=0"]))
        f16.Campaign(_cfg(self.root, "g", gens=1), FakeWorker()).run()
        meta = os.path.join(self.root, "g", "meta", "gen1.json")
        m = json.load(open(meta))
        m["row_ids"] = ["x"]
        json.dump(m, open(meta, "w"))
        with self.assertRaises(RuntimeError):
            f16.Campaign(_cfg(self.root, "g", gens=1), FakeWorker()).meta("gen1", ["y"], ["z"])
        nb = os.path.join(self.root, "empty")
        f16.PolicyDatabase(os.path.join(nb, "db"))
        with self.assertRaises(RuntimeError):
            f16.Campaign(_cfg(self.root, "empty"), FakeWorker()).run()

    def test_cutoff_and_job_order(self):
        self.assertEqual(f16.cutoff_steps(4096, 13, 2048, 1500), 7 * 1500)
        self.assertEqual(f16.cutoff_steps(4096, 1, 2048, 1500), 1500)
        miss = [("P4", "E1"), ("P4", "E3"), ("P4", "E5"), ("P0", "E5"), ("P2", "E5")]
        self.assertEqual(f16.next_payoff_job(miss), ("pursuer", "P4", ["E1", "E3", "E5"]))
        self.assertEqual(f16.next_payoff_job([("P0", "E5"), ("P2", "E5")]), ("evader", "E5", ["P0", "P2"]))
        self.assertIsNone(f16.next_payoff_job([]))


if __name__ == "__main__":
    unittest.main()
