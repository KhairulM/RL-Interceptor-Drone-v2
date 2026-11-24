"""Thesis F16 algorithm gates (docs/research_plan.md F16 D): meta-game solver, payoff store, episode-reset opponent
sampling, own-policy-hull KL objective (exact small-tree gradients), lambda-zero equality with PSRO."""

import copy
import importlib.util
import itertools
import json
import math
import os
import tempfile
import unittest

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ms = _load("omni_drones/utils/selfplay/meta_solver.py", "f16_meta_solver")
ps = _load("omni_drones/utils/selfplay/payoff_store.py", "f16_payoff_store")
acc = _load("omni_drones/utils/selfplay/eval_accumulator.py", "f16_eval_acc")
from omni_drones.learning.ppo import hull_kl as hk  # noqa: E402
from omni_drones.learning.ppo.ppo import PPOConfig, PPOPolicy  # noqa: E402
from omni_drones.utils.selfplay.opponents import FrozenPolicyBank  # noqa: E402


class SolverTests(unittest.TestCase):
    def ids(self, k, p):
        return [f"{p}{i}" for i in range(k)]

    def test_matching_pennies_and_rps(self):
        A = np.array([[1, -1], [-1, 1.]])
        r = ms.solve(A, -A, ["p0", "p1"], ["e0", "e1"])
        np.testing.assert_allclose(list(r["x"].values()), [.5, .5])
        np.testing.assert_allclose(list(r["y"].values()), [.5, .5])
        R = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0.]])
        r = ms.solve(R, -R, self.ids(3, "p"), self.ids(3, "e"))
        np.testing.assert_allclose(list(r["x"].values()), [1 / 3] * 3, atol=1e-9)
        self.assertLessEqual(max(r["regret_row"], r["regret_col"]), 1e-9)

    def test_general_sum_candidates_are_equilibria_and_tie_break(self):
        A, B = np.array([[3, 0], [0, 2.]]), np.array([[2, 0], [0, 3.]])        # battle of the sexes
        r = ms.solve(A, B, ["o", "f"], ["O", "F"])
        found = {(tuple(c["x"]), tuple(c["y"])) for c in r["candidates"] if c["verified"]}
        self.assertEqual(found, {((1.0, 0.0), (1.0, 0.0)), ((0.0, 1.0), (0.0, 1.0))})
        self.assertEqual(r["x"], {"o": 0.0, "f": 1.0})                       # equal entropy: smallest support ids
        self.assertEqual(r["n_verified_distinct"], 2)

    def test_random_games_match_support_enumeration(self):
        rng = np.random.default_rng(3)
        for m, n in [(1, 1), (2, 3), (3, 3), (4, 4), (5, 4)]:
            for _ in range(5):
                A, B = rng.normal(size=(m, n)), rng.normal(size=(m, n))
                r = ms.solve(A, B, self.ids(m, "p"), self.ids(n, "e"))
                x, y = np.array(list(r["x"].values())), np.array(list(r["y"].values()))
                self.assertLessEqual(max(ms.regrets(A, B, x, y)), 1e-6)
                se = ms.support_enumeration(A, B)
                self.assertTrue(any(np.allclose(x, a, atol=1e-6) and np.allclose(y, b, atol=1e-6) for a, b in se))
                ver = [c for c in r["candidates"] if c["verified"]]
                self.assertAlmostEqual(r["joint_entropy"], max(c["entropy"] for c in ver))

    def test_scale_and_shift_invariance_and_determinism(self):
        rng = np.random.default_rng(5)
        A, B = rng.normal(size=(6, 5)), rng.normal(size=(6, 5))
        r1 = ms.solve(A, B, self.ids(6, "p"), self.ids(5, "e"))
        r2 = ms.solve(3 * A + 7, 0.5 * B - 2, self.ids(6, "p"), self.ids(5, "e"))
        r3 = ms.solve(A, B, self.ids(6, "p"), self.ids(5, "e"))
        np.testing.assert_allclose(list(r1["x"].values()), list(r2["x"].values()), atol=1e-7)
        self.assertEqual(json.dumps(r1), json.dumps(r3))

    def test_malformed_and_failure(self):
        with self.assertRaises(ValueError):
            ms.solve([[1, np.nan]], [[0, 0]], ["p"], ["a", "b"])
        with self.assertRaises(ValueError):
            ms.solve([[1, 2]], [[0, 0, 1]], ["p"], ["a", "b"])
        with self.assertRaises(ValueError):
            ms.solve([[1], [2]], [[0], [0]], ["p", "p"], ["a"])
        orig = ms._lemke_howson
        try:
            def boom(*a, **k):
                raise ms.NoEquilibrium("forced")
            ms._lemke_howson = boom
            with self.assertRaises(ms.NoEquilibrium):
                ms.solve([[1.0]], [[0.0]], ["p"], ["e"])
        finally:
            ms._lemke_howson = orig


class PayoffStoreTests(unittest.TestCase):
    def rec(self, run, p, e, n, mp, sp, me, se):
        return {"run_id": run, "pursuer": p, "evader": e, "n": n, "mean_pursuer": mp, "std_pursuer": sp,
                "mean_evader": me, "std_evader": se, "events": {"capture": 0.5}}

    def test_append_pool_missing_and_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            st = ps.PayoffStore(d)
            st.append([self.rec("r1", "P0", "E1", 1000, 1.0, 0.5, -2.0, 1.0)])
            with self.assertRaises(ValueError):
                st.append([self.rec("r1", "P0", "E1", 10, 0, 0, 0, 0)])
            with self.assertRaises(ps.MissingCells):
                st.matrices(["P0"], ["E1"], 2048)
            st.append([self.rec("r2", "P0", "E1", 3000, 3.0, 0.5, -4.0, 1.0)])
            c = st.cell("P0", "E1")
            self.assertEqual(c["n"], 4000)
            self.assertAlmostEqual(c["mean_pursuer"], 2.5)
            self.assertAlmostEqual(c["std_pursuer"], math.sqrt(0.25 + (1000 * 1.5 ** 2 + 3000 * 0.5 ** 2) / 4000))
            UP, UE, N = st.matrices(["P0"], ["E1"], 2048)
            self.assertAlmostEqual(UP[0, 0], 0.25)
            self.assertAlmostEqual(UE[0, 0], -0.35)
            self.assertEqual(len(st.records()), 2)
            with self.assertRaises(ValueError):
                st.append([self.rec("r3", "P0", "E1", 5, float("nan"), 0, 0, 0)])

    def test_records_orientation(self):
        m = {"P3": {"payoff": {"n": 5, "mean_pursuer": 1.0, "std_pursuer": 0.1, "mean_evader": -1.0,
                               "std_evader": 0.2}, "outcomes": {}, "neutralization_rate": 0.4}}
        r = ps.records_from_matchups(m, "E7", "evader", "run", 3, {"cfg": "x"})[0]
        self.assertEqual((r["pursuer"], r["evader"], r["mean_evader"]), ("P3", "E7", -1.0))

    def test_accumulator_payoff_block(self):
        ep = {"env": torch.tensor([0, 1, 2]), "capture": torch.tensor([1., 0, 0]), "crash": torch.zeros(3),
              "opp_crash": torch.zeros(3), "out_of_bounds": torch.zeros(3), "opp_out_of_bounds": torch.zeros(3),
              "timeout": torch.tensor([0., 1, 1]), "episode_len": torch.ones(3), "min_distance": torch.ones(3),
              "motor_effort": torch.ones(3), "feat_mean": torch.zeros(3, 9), "feat_std": torch.zeros(3, 9),
              "payoff_return_pursuer": torch.tensor([3., 1, 2]), "payoff_return_evader": torch.tensor([-1., 0, 1])}
        out = acc.summarize_matchups(ep, {"E": torch.tensor([0, 1, 2])}, "pursuer", 0.02, 1500)["E"]["payoff"]
        self.assertEqual(out["n"], 3)
        self.assertAlmostEqual(out["mean_pursuer"], 2.0)
        self.assertAlmostEqual(out["std_evader"], math.sqrt(2 / 3))


class _Pol:
    def act(self, td):
        return td[("agents", "observation")][..., :2]


def _bank(ids, seed=0, kinds=None):
    bank = object.__new__(FrozenPolicyBank)
    bank.device = torch.device("cpu")
    bank.switch = None
    bank.resample = {"seed": seed}
    bank.stochastic = False
    bank.resample_counts = {}
    bank.policies = [{"id": i, "kind": (kinds or {}).get(i, "rl")} for i in ids]
    bank.ids = list(ids)
    bank.kind = {p["id"]: p["kind"] for p in bank.policies}
    bank.heuristic = {i: None for i in ids}
    bank.heuristic_names = []
    bank._rl = {i: _Pol() for i in ids}
    bank.action_dim = 2
    return bank


class ResampleTests(unittest.TestCase):
    def test_frequencies_reset_only_and_determinism(self):
        probs = {"a": 0.2, "b": 0.0, "c": 0.5, "d": 0.3}
        bank = _bank(list(probs))
        bank.assign(probs, 64)
        before = bank.env_policy.clone()
        for _ in range(5):                       # no reset -> identity held
            bank.rl_raw_actions(torch.randn(64, 1, 3))
        self.assertTrue(torch.equal(before, bank.env_policy))
        for _ in range(2000):
            bank.on_reset(torch.arange(64))
        n = sum(bank.resample_counts.values())
        self.assertEqual(n, 2000 * 64)
        self.assertNotIn("b", bank.resample_counts)
        for k in ("a", "c", "d"):
            self.assertAlmostEqual(bank.resample_counts[k] / n, probs[k], delta=0.005)
        part = torch.tensor([3, 7])
        keep = bank.env_policy.clone()
        bank.on_reset(part)
        mask = torch.ones(64, dtype=torch.bool)
        mask[part] = False
        self.assertTrue(torch.equal(keep[mask], bank.env_policy[mask]))
        flat = torch.cat([v for v in bank.assignment.values()]).sort().values
        self.assertTrue(torch.equal(flat, torch.arange(64)))
        b1, b2 = _bank(list(probs), seed=9), _bank(list(probs), seed=9)
        b1.assign(probs, 32)
        b2.assign(probs, 32)
        for _ in range(10):
            b1.on_reset(torch.arange(32))
            b2.on_reset(torch.arange(32))
        self.assertTrue(torch.equal(b1.env_policy, b2.env_policy))

    def test_rejects_heuristics_and_switch(self):
        bank = _bank(["a", "h"], kinds={"h": "heuristic"})
        with self.assertRaises(ValueError):
            bank.assign({"a": 0.5, "h": 0.5}, 8)


# ----------------------------------------------------------------------------- hull KL: exact small tree
def _softmax(x):
    return torch.log_softmax(x, -1)


class HullTreeTests(unittest.TestCase):
    """Two steps, two actions; s1 fixed, s2 = a1. Policies are tabular logits [3 states, 2 actions]."""

    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.refs = [torch.randn(3, 2, generator=g) for _ in range(3)]
        self.alpha = torch.tensor([0.5, 0.3, 0.2])
        self.theta = torch.randn(3, 2, generator=g, dtype=torch.float64).requires_grad_(True)

    def traj_terms(self, theta, a1, a2):
        s = [0, 1 + a1]
        acts = [a1, a2]
        lp = torch.stack([_softmax(theta[s[t]].float())[acts[t]] for t in range(2)])          # [T]
        lpk = torch.stack([torch.stack([_softmax(r[s[t]])[acts[t]] for r in self.refs]) for t in range(2)])  # [T, K]
        return s, acts, lp, lpk

    def exact_kl(self, theta):
        tot = 0.0
        for a1, a2 in itertools.product(range(2), range(2)):
            _, _, lp, lpk = self.traj_terms(theta, a1, a2)
            logP = lp.sum()
            logQ = torch.logsumexp(torch.log(self.alpha) + lpk.sum(0), 0)
            tot = tot + logP.exp() * (logP - logQ)
        return tot

    def test_chain_rule_and_gradient_decomposition(self):
        log_alpha = torch.log(self.alpha)
        est = torch.zeros_like(self.theta)
        for a1, a2 in itertools.product(range(2), range(2)):
            s, acts, lp, lpk = self.traj_terms(self.theta, a1, a2)
            hist, _ = hk.history_logl(lpk.detach().unsqueeze(0), torch.tensor([[True, False]]),
                                      torch.zeros(1, 3))
            logw = hk.mixture_logw(log_alpha, hist)[0]                                       # [T, K]
            # chain rule: sum_t log q(a_t | h_t) = log sum_k alpha_k prod_t pi_k(a_t)
            seq = hk.mixture_logprob(logw, lpk.detach()).sum()
            self.assertAlmostEqual(float(seq), float(torch.logsumexp(log_alpha + lpk.sum(0), 0)), places=5)
            # per-step exact KL(pi(.|s_t) || q(.|h_t))
            kls = []
            for t in range(2):
                logpi = _softmax(self.theta[s[t]])
                comp = torch.stack([_softmax(r[s[t]]) for r in self.refs], -1).double()     # [A, K]
                logq = hk.mixture_logprob(logw[t].double(), comp)                           # [A]
                kls.append((logpi.exp() * (logpi - logq)).sum())
            P = lp.sum().exp().detach().double()
            score = [torch.autograd.grad(_softmax(self.theta[s[t]])[acts[t]], self.theta, retain_graph=True)[0]
                     for t in range(2)]
            ret = [kls[0].detach() + kls[1].detach(), kls[1].detach()]                      # reward-to-go
            direct = torch.autograd.grad(kls[0] + kls[1], self.theta)[0]
            est = est + P * (score[0] * ret[0] + score[1] * ret[1] + direct)
        D = self.exact_kl(self.theta.double())
        grad = torch.autograd.grad(D, self.theta)[0]
        torch.testing.assert_close(est, grad.double(), atol=1e-5, rtol=1e-4)
        # finite differences of the exact objective
        h, fd = 1e-4, torch.zeros_like(self.theta)
        with torch.no_grad():
            for i in range(3):
                for j in range(2):
                    tp, tm = self.theta.clone(), self.theta.clone()
                    tp[i, j] += h
                    tm[i, j] -= h
                    fd[i, j] = (self.exact_kl(tp) - self.exact_kl(tm)) / (2 * h)
        torch.testing.assert_close(fd, grad.double(), atol=1e-3, rtol=1e-3)

    def test_history_carry_equals_single_batch(self):
        g = torch.Generator().manual_seed(1)
        lpk = torch.randn(4, 10, 3, generator=g)
        init = torch.rand(4, 10, generator=g) < 0.2
        init[:, 0] = True
        full, _ = hk.history_logl(lpk, init, torch.zeros(4, 3))
        h1, c = hk.history_logl(lpk[:, :6], init[:, :6], torch.zeros(4, 3))
        h2, _ = hk.history_logl(lpk[:, 6:], init[:, 6:], c)
        torch.testing.assert_close(full, torch.cat([h1, h2], 1))
        self.assertTrue(torch.all(full[init] == 0))


class GaussianKLTests(unittest.TestCase):
    def test_single_component_mc_matches_analytic_and_pathwise_grad(self):
        g = torch.Generator().manual_seed(2)
        loc = torch.randn(5, 4, generator=g, dtype=torch.float64).requires_grad_(True)
        scale = (0.3 + torch.rand(5, 4, generator=g, dtype=torch.float64)).requires_grad_(True)
        mu = torch.randn(5, 1, 4, generator=g, dtype=torch.float64)
        sig = 0.5 + torch.rand(1, 4, generator=g, dtype=torch.float64)
        eps = torch.randn(200000, 5, 4, generator=g, dtype=torch.float64)
        mc = hk.mc_kl(loc, scale, torch.zeros(5, 1, dtype=torch.float64), mu, sig, eps)
        an = torch.distributions.kl_divergence(torch.distributions.Normal(loc, scale),
                                               torch.distributions.Normal(mu[:, 0], sig)).sum(-1)
        torch.testing.assert_close(mc, an, atol=2e-2, rtol=2e-2)
        g_mc = torch.autograd.grad(mc.sum(), (loc, scale))
        g_an = torch.autograd.grad(an.sum(), (loc, scale))
        for a, b in zip(g_mc, g_an):
            torch.testing.assert_close(a, b, atol=3e-2, rtol=3e-2)

    def test_mixture_kl_nonnegative_and_zero_at_vertex(self):
        g = torch.Generator().manual_seed(3)
        loc = torch.randn(6, 4, generator=g)
        scale = torch.full((6, 4), 0.7)
        mu = torch.stack([loc, loc + 1.0], 1)
        sig = torch.full((2, 4), 0.7)
        eps = torch.randn(4000, 6, 4, generator=g)
        at_vertex = hk.mc_kl(loc, scale, torch.log(torch.tensor([[1.0, 1e-30]])).expand(6, 2), mu, sig, eps)
        self.assertLess(float(at_vertex.abs().max()), 1e-4)
        mixed = hk.mc_kl(loc, scale, torch.log(torch.tensor([[0.5, 0.5]])).expand(6, 2), mu, sig, eps)
        self.assertGreater(float(mixed.min()), 0.0)

    def test_candidates_and_selection(self):
        al = hk.candidate_alphas(3, 16, 0)
        self.assertEqual(al.shape, (19, 3))
        np.testing.assert_allclose(al.sum(1), 1)
        np.testing.assert_array_equal(al, hk.candidate_alphas(3, 16, 0))
        # data generated by component 1 -> the pure vertex 1 has the smallest log-ratio
        g = torch.Generator().manual_seed(4)
        logp_k = torch.randn(500, 3, generator=g) - torch.tensor([3.0, 0.0, 3.0])
        logp_cur = logp_k[:, 1].clone()
        la = torch.log(torch.as_tensor(np.where(al > 0, al, 1e-300), dtype=torch.float32))
        c, _ = hk.select_candidate(la, torch.zeros(500, 3), logp_k, logp_cur)
        self.assertEqual(c, 1)


OBS, ACT = 6, 4


def _policy(seed=0):
    obs = Composite({"agents": Composite({"observation": Unbounded((1, OBS))}, shape=())}, shape=())
    act = Composite({"agents": Composite({"action": Unbounded((1, ACT))}, shape=())}, shape=())
    rew = Composite({"agents": Composite({"reward": Unbounded((1, 1))}, shape=())}, shape=())
    torch.manual_seed(seed)
    return PPOPolicy(PPOConfig(), obs, act, rew, device="cpu")


def _rollout(policy, E=8, T=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(E, T, 1, OBS, generator=g)
    td = TensorDict({"agents": TensorDict({"observation": obs}, [E, T])}, [E, T])
    with torch.no_grad():
        policy(td)
    init = torch.zeros(E, T, 1, dtype=torch.bool)
    init[:, 0] = True
    init[:, 9] = True
    done = torch.zeros(E, T, 1, dtype=torch.bool)
    done[:, 8] = True
    nxt = TensorDict({"agents": TensorDict({"observation": torch.randn(E, T, 1, OBS, generator=g),
                                            "reward": torch.randn(E, T, 1, 1, generator=g)}, [E, T]),
                      "terminated": done.clone(), "done": done.clone()}, [E, T])
    td.set("next", nxt)
    td.set("is_init", init)
    return td


class PPOHullTests(unittest.TestCase):
    def ckpts(self, d):
        paths = []
        for k in range(2):
            p = os.path.join(d, f"ref{k}.pt")
            torch.save(_policy(seed=10 + k).state_dict(), p)
            paths.append(p)
        return paths

    def test_lambda_zero_equals_psro(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self.ckpts(d)
            a, b = _policy(), _policy()
            b.set_hull_kl(paths, 0.0, seed=3)
            td = _rollout(a)
            torch.manual_seed(5)
            a.train_op(td.clone())
            torch.manual_seed(5)
            info = b.train_op(td.clone())
            for (k, x), (_, y) in zip(a.state_dict().items(), b.state_dict().items()):
                torch.testing.assert_close(x, y, msg=k)
            self.assertEqual(info["hull_intrinsic"], 0.0)

    def test_positive_lambda_finite_nonzero_kl_gradient(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self.ckpts(d)
            p = _policy()
            p.set_hull_kl(paths, 0.5, seed=3)
            info = p.train_op(_rollout(p))
            self.assertTrue(all(math.isfinite(v) for v in info.values() if isinstance(v, float)))
            self.assertGreater(info["hull_kl"], 0.0)
            self.assertGreater(info["hull_kl_grad_norm"], 0.0)
            self.assertGreater(info["hull_intrinsic"], 0.0)
            info2 = p.train_op(_rollout(p, seed=1))
            self.assertIn("hull_heldout_gap_16_vs_64", info2)
            for ref in p.hull.refs:                  # old policies stay frozen
                self.assertFalse(any(q.requires_grad for q in ref.parameters()))


if __name__ == "__main__":
    unittest.main()
