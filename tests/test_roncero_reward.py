"""Thesis F8: Sanchez Roncero et al. 2026 reward / endings (roncero_reward.py) and the GAE truncation option."""

import importlib.util
import math
from pathlib import Path
import unittest

import torch

_P = Path(__file__).resolve().parents[1] / "omni_drones/envs/single/roncero_reward.py"
_spec = importlib.util.spec_from_file_location("roncero_reward", _P)      # no Isaac import (package __init__)
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)

B = lambda *v: torch.tensor(v, dtype=torch.bool).reshape(-1, 1)
F = lambda *v: torch.tensor(v, dtype=torch.float32).reshape(-1, 1)


class RonceroRewardTest(unittest.TestCase):

    def test_coefficients_are_the_papers(self):
        self.assertEqual(rr.DEFAULTS, {"k_br": 0.001, "k_sm": 0.002, "k_b": 10.0, "k_c": 10.0, "k_app": 0.25,
                                       "k_s": 0.0025, "v_safe": 2.0, "alpha_close": 3.0})

    def test_endings(self):
        # rows: capture | own crash | opponent crash | own oob | opponent oob | timeout only | capture at the time limit
        cap, ac, oc, ao, oo, tl = (B(*x) for x in zip(
            (1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0),
            (0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 1)))
        term, trunc, timeout = rr.outcome(cap, ac, oc, ao, oo, tl)
        self.assertEqual(term.flatten().tolist(), [True, False, False, False, False, False, True])
        self.assertEqual(trunc.flatten().tolist(), [False, True, True, True, True, True, False])
        self.assertEqual(timeout.flatten().tolist(), [False, False, False, False, False, True, False])

    def test_closing_speed(self):
        p = torch.tensor([[0., 0., 1.]]); e = torch.tensor([[2., 0., 1.]])
        self.assertAlmostEqual(rr.closing_speed(p, torch.tensor([[3., 0., 0.]]), e, torch.zeros(1, 3)).item(), 3.0)
        self.assertAlmostEqual(rr.closing_speed(p, torch.zeros(1, 3), e, torch.tensor([[1., 0., 0.]])).item(), -1.0)

    def test_pursuer_reward(self):
        z3 = torch.zeros(2, 3)
        r = rr.reward(True, B(1, 1), F(0.0, 0.0), z3, F(0, 0), B(0, 0), F(1.0, 2.5))
        self.assertAlmostEqual(r[0].item(), 10.0)                                 # v_close <= v_safe: full bonus
        self.assertAlmostEqual(r[1].item(), 10.0 * math.exp(-3.0 * 0.5), places=5)
        r = rr.reward(True, B(0), F(0.4), torch.tensor([[1., 2., 2.]]), F(0.5), B(1), F(0.0))
        self.assertAlmostEqual(r.item(), 0.25 * 0.4 - 0.001 * 9 - 0.002 * 0.5 - 10.0, places=6)

    def test_evader_reward_ignores_pursuer_crash(self):
        z3 = torch.zeros(3, 3)
        r = rr.reward(False, B(0, 1, 0), F(0, 0, 0), z3, F(0, 0, 0), B(0, 0, 1), F(0, 0, 0))
        self.assertEqual([round(x, 6) for x in r.flatten().tolist()], [0.0025, round(0.0025 - 10, 6), round(0.0025 - 10, 6)])


class GaeTruncationTest(unittest.TestCase):

    def test_default_unchanged_and_truncation_cuts_recursion_only(self):
        from omni_drones.learning.ppo.common import GAE
        g = GAE(0.9, 0.5)
        T = 4
        rew = torch.tensor([1., 0., 2., 0.]).reshape(1, T, 1)
        val = torch.tensor([0.5, 0.4, 0.3, 0.2]).reshape(1, T, 1)
        nxt = torch.tensor([0.4, 0.7, 0.2, 0.1]).reshape(1, T, 1)     # step 1 ends by truncation: nxt[1] = V(final obs)
        term = torch.zeros(1, T, 1, dtype=torch.bool)
        done = term.clone(); done[0, 1] = True
        a0, _ = g(rew, term, val, nxt)
        a1, _ = g(rew, term, val, nxt, None)
        self.assertTrue(torch.equal(a0, a1))                           # done=None: original behaviour
        a2, _ = g(rew, term, val, nxt, done)
        d = rew + 0.9 * nxt - val                                      # bootstrap kept everywhere (no termination)
        self.assertAlmostEqual(a2[0, 1, 0].item(), d[0, 1, 0].item(), places=6)        # no leak from step 2
        self.assertAlmostEqual(a2[0, 0, 0].item(), d[0, 0, 0].item() + 0.45 * d[0, 1, 0].item(), places=6)
        self.assertNotAlmostEqual(a0[0, 1, 0].item(), a2[0, 1, 0].item(), places=4)     # the old path leaked

    def test_ppo_config_flag_default_off(self):
        from omni_drones.learning.ppo.ppo import PPOConfig
        self.assertFalse(PPOConfig().gae_cut_at_truncation)


if __name__ == "__main__":
    unittest.main()
