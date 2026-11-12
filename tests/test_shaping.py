"""Thesis v2 (2026-11-11): potential-based pursuer shaping (omni_drones/envs/single/shaping.py)."""

import importlib.util
from pathlib import Path
import unittest

import torch

_P = Path(__file__).resolve().parents[1] / "omni_drones/envs/single/shaping.py"
_spec = importlib.util.spec_from_file_location("shaping", _P)      # no Isaac import (package __init__)
sh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sh)

W_APP, W_PREC, C, G = 2.5, 5.0, 5.0, 0.99


class ShapingTest(unittest.TestCase):

    def test_plain_differences_unchanged(self):
        prev, d = torch.tensor([[2.0], [0.5]]), torch.tensor([[1.5], [0.3]])
        a, p = sh.shaping(prev, d, 7.0, 10.0, C)
        self.assertTrue(torch.allclose(a, 7.0 * (prev - d)))
        self.assertTrue(torch.allclose(p, 10.0 * (torch.exp(-C * d) - torch.exp(-C * prev))))

    def _episode(self, ds, captured_at_end):
        """Discounted shaped return of a distance trajectory d_0 .. d_T (T steps)."""
        ret, disc = 0.0, 1.0
        for t in range(1, len(ds)):
            term = torch.tensor([[captured_at_end and t == len(ds) - 1]])
            a, p = sh.shaping(torch.tensor([[ds[t - 1]]]), torch.tensor([[ds[t]]]), W_APP, W_PREC, C, G, term)
            ret += disc * (a + p).item()
            disc *= G
        return ret

    def test_potential_telescopes_to_minus_phi0_at_capture(self):
        phi = lambda d: -W_APP * d + W_PREC * torch.exp(torch.tensor(-C * d)).item()
        for ds in ([3.0, 2.0, 1.0, 0.4, 0.1], [3.0, 3.5, 2.5, 2.6, 0.9, 0.2, 0.08]):
            self.assertAlmostEqual(self._episode(ds, True), -phi(ds[0]), places=5)   # path-independent

    def test_truncation_keeps_phi_of_last_state(self):
        phi = lambda d: -W_APP * d + W_PREC * torch.exp(torch.tensor(-C * d)).item()
        ds = [3.0, 2.0, 1.2]
        expect = -phi(ds[0]) + G ** (len(ds) - 1) * phi(ds[-1])
        self.assertAlmostEqual(self._episode(ds, False), expect, places=5)

    def test_hidden_term_of_plain_differences(self):
        # plain - potential = (1 - gamma) Phi(s') on a non-terminal step
        prev, d, term = torch.tensor([[2.0]]), torch.tensor([[1.6]]), torch.tensor([[False]])
        a0, p0 = sh.shaping(prev, d, W_APP, W_PREC, C)
        a1, p1 = sh.shaping(prev, d, W_APP, W_PREC, C, G, term)
        phi_next = -W_APP * 1.6 + W_PREC * torch.exp(torch.tensor(-C * 1.6))
        self.assertAlmostEqual(((a0 + p0) - (a1 + p1)).item(), ((1 - G) * phi_next).item(), places=6)


if __name__ == "__main__":
    unittest.main()
