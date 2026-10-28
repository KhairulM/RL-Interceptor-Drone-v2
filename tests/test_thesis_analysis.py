import importlib.util
from pathlib import Path
import unittest

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "scripts/experiments/thesis/analysis.py"
_SPEC = importlib.util.spec_from_file_location("thesis_analysis", _PATH)
an = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(an)


class AnalysisTests(unittest.TestCase):
    def test_transitive_payoff_has_r2_one_and_no_reversals(self):
        a = np.array([0.0, 1.0, 2.0])[:, None]
        b = np.array([0.5, 1.5, -1.0, 0.0])[None, :]
        P = 1 / (1 + np.exp(-(a - b)))
        self.assertAlmostEqual(an.additive_fit(P)["r2"], 1.0, places=6)
        self.assertEqual(an.rank_reversals(P, 0.05)["reversal_fraction"], 0.0)

    def test_cyclic_payoff_has_reversals(self):
        P = np.array([[0.9, 0.1], [0.1, 0.9]])
        self.assertAlmostEqual(an.rank_reversals(P, 0.1)["reversal_fraction"], 1.0)
        self.assertLess(an.additive_fit(P)["r2"], 0.5)

    def test_vendi_score_extremes(self):
        self.assertAlmostEqual(an.vendi_score(np.eye(5)), 5.0, places=6)
        self.assertAlmostEqual(an.vendi_score(np.ones((5, 5))), 1.0, places=6)

    def test_kmedoids_and_silhouette_on_two_blobs(self):
        rng = np.random.default_rng(0)
        X = np.concatenate([rng.normal(0, 0.1, (10, 2)), rng.normal(5, 0.1, (10, 2))])
        D = an.pairwise_dist(X)
        labels, med, _ = an.kmedoids(D, 2)
        self.assertEqual(len(set(labels[:10])), 1)
        self.assertEqual(len(set(labels[10:])), 1)
        self.assertNotEqual(labels[0], labels[10])
        self.assertGreater(an.silhouette(D, labels), 0.9)

    def test_iqm_and_bootstrap(self):
        self.assertAlmostEqual(an.iqm([0, 1, 2, 3, 100, -100, 1, 2]), np.mean([1, 1, 2, 2]))
        m, lo, hi = an.bootstrap_ci([1.0] * 10)
        self.assertEqual((m, lo, hi), (1.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
