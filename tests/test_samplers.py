import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sm = _load("thesis_samplers", "omni_drones/utils/selfplay/samplers.py")
policy_db = _load("thesis_policy_db", "omni_drones/utils/selfplay/policy_db.py")


def _metrics(win, ttc=1.0, dist=0.1, eff=500.0, crash=0.1):
    return {"win_rate": win, "avg_time_to_capture": ttc, "avg_min_distance": dist,
            "avg_motor_effort": eff, "pursuer_crash_rate": crash}


class SamplerTests(unittest.TestCase):
    def test_softmax_matches_policy_db(self):
        with tempfile.TemporaryDirectory() as root:
            db = policy_db.PolicyDatabase(root)
            ref = db.add_policy(policy_db.PURSUER, "ref.pt")
            opps = [db.add_policy(policy_db.EVADER, f"e{i}.pt") for i in range(4)]
            raw = {}
            for i, o in enumerate(opps):
                m = _metrics(0.2 * i, ttc=1 + i, dist=0.05 * i, eff=400 + 10 * i, crash=0.05 * i)
                db.record_matchup(ref, o, m)
                raw[o] = m
            w = {"win_rate": 1.0, "avg_time_to_capture": 0.8, "avg_min_distance": 0.5,
                 "avg_motor_effort": 0.3, "pursuer_crash_rate": 0.0}
            expected = db.compute_sampling_probs(policy_db.PURSUER, temperature=0.9, metric_weights=w,
                                                 kind="rl", uniform_mix=0.5)
            got = sm.compute_probs("softmax", raw, "pursuer", 0.9, 0.5, w)
            for o in opps:
                self.assertAlmostEqual(got[o], expected[o])

    def test_dpp_duplicate_closed_form(self):
        for m in (1, 2, 5, 20):
            q = np.ones(m + 1)
            S = np.zeros((m + 1, m + 1))
            S[:m, :m] = 1.0          # m exact copies
            S[m, m] = 1.0            # one isolated opponent
            k = sm.dpp_marginals(q, S, scale=1.0)
            self.assertAlmostEqual(k[:m].sum(), m / (1 + m), places=9)
            self.assertAlmostEqual(k[m], 0.5, places=9)

    def test_qd_dpp_keeps_hardness_order_when_dissimilar(self):
        raw = {f"e{i}": _metrics(1.0 - 0.2 * i) for i in range(5)}
        desc = {f"e{i}": [100.0 * i, 0.0] for i in range(5)}     # far apart -> S ~ I
        p = sm.compute_probs("qd_dpp", raw, "pursuer", 0.5, 0.0, {"win_rate": 1.0}, desc)
        order = sorted(p, key=p.get)
        self.assertEqual(order, [f"e{i}" for i in range(5)])      # lower win rate = harder = more mass
        self.assertAlmostEqual(sum(p.values()), 1.0)

    def test_duplicates_grow_linearly_under_softmax_but_bounded_under_dpp(self):
        def shares(m):
            raw = {f"o{i}": _metrics(0.5) for i in range(4)}
            desc = {f"o{i}": [10.0 * i] for i in range(4)}
            for c in range(m):
                raw[f"dup{c}"] = _metrics(0.5)
                desc[f"dup{c}"] = [0.0]                            # copies of o0
            soft = sm.compute_probs("softmax", raw, "pursuer", 1.0, 0.0, {"win_rate": 1.0})
            dpp = sm.compute_probs("qd_dpp", raw, "pursuer", 1.0, 0.0, {"win_rate": 1.0}, desc)
            cluster = ["o0"] + [f"dup{c}" for c in range(m)]
            return sum(soft[k] for k in cluster), sum(dpp[k] for k in cluster)
        s0, d0 = shares(0)
        s5, d5 = shares(5)
        s10, d10 = shares(10)
        s20, d20 = shares(20)
        self.assertAlmostEqual(s0, 0.25)
        self.assertAlmostEqual(s5, 6 / 9)                          # 6 of 9 equal-score opponents
        self.assertLess(d5, s5)
        self.assertGreater(s20 - s10, 0.05)                        # softmax keeps growing
        self.assertLess(d20 - d10, 0.05)                           # DPP saturates
        self.assertLess(d20, 0.6)

    def test_uniform_floor(self):
        raw = {f"e{i}": _metrics(0.1 * i) for i in range(4)}
        desc = {f"e{i}": [float(i)] for i in range(4)}
        for name in ("softmax", "qd_dpp"):
            p = sm.compute_probs(name, raw, "pursuer", 0.1, 0.5, None, desc)
            self.assertAlmostEqual(sum(p.values()), 1.0)
            self.assertTrue(all(v >= 0.5 / 4 - 1e-12 for v in p.values()))

    def test_opponent_metrics(self):
        m = sm.opponent_metrics([1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0.4, 0, 0, 0.6],
                                [0.1, 0.2, 0.3, 0.4], [1, 2, 3, 4], max_time=30.0)
        self.assertAlmostEqual(m["win_rate"], 0.5)
        self.assertAlmostEqual(m["avg_time_to_capture"], 0.5)
        self.assertAlmostEqual(m["pursuer_crash_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()


def test_hedge_matches_roncero_eq12():
    c = {"a": 0.9, "b": 0.5, "c": 0.1, "d": 0.0}
    p = sm.hedge_probs(c, gamma=0.7, eta=2.0)
    tot = sum(v ** 2 for v in c.values())
    for k, v in c.items():
        assert abs(p[k] - (0.7 / 4 + 0.3 * v ** 2 / tot)) < 1e-6
    u = sm.hedge_probs({"a": 0.0, "b": 0.0}, gamma=0.7, eta=2.0)
    assert abs(u["a"] - 0.5) < 1e-9


def test_qd_dpp_with_hedge_quality_keeps_hedge_order_when_dissimilar():
    c = {"a": 0.9, "b": 0.5, "c": 0.1}
    desc = {"a": [0.0, 0.0], "b": [100.0, 0.0], "c": [0.0, 100.0]}   # far apart -> S ~ I
    p = sm.qd_dpp_probs(sm.hedge_scores(c, 2.0), desc, 1.0, 0.7, sigma=1.0)
    assert p["a"] > p["b"] > p["c"]


def test_hedge_gamma_zero_is_pfsp_squared():
    c = {"a": 0.9, "b": 0.5, "c": 0.0}
    p = sm.hedge_probs(c, gamma=0.0, eta=2.0)
    assert abs(p["a"] - 0.81 / 1.06) < 1e-6 and abs(p["b"] - 0.25 / 1.06) < 1e-6 and p["c"] < 1e-9
