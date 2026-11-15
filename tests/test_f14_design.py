import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

_THESIS = Path(__file__).resolve().parents[1] / "scripts/experiments/thesis"
sys.path[:0] = [str(_THESIS), str(_THESIS / "exps")]
import f14_generalization_sp as f14  # noqa: E402


class F14DesignTests(unittest.TestCase):
    def test_frozen_rls_score_reads_frozen_own_evaluation(self):
        paths = []
        heldout = "heldout"

        def fake_rates(path, key, radius):
            name = os.path.basename(path)
            paths.append(name)
            if name.endswith("stageC_frozen_strong.json"):
                return {f"{heldout}/evader:test": 0.8}
            if name.endswith("stageC_frozen_cross.json"):
                return {"camp/evader:test": 0.6}
            if name.endswith("stageC_frozen_own.json"):
                return {"camp/evader:test": 0.75}
            return {}

        with patch.object(f14, "rates_at", side_effect=fake_rates), \
                patch.object(f14, "_group_cross", return_value=0.6), \
                patch.object(f14, "_mean", return_value=0.8):
            score = f14._score_stagec("camp", "rls", 0.10, frozen=True)

        self.assertAlmostEqual(score["own_j"], 0.75)
        self.assertTrue(any(path.endswith("stageC_frozen_own.json") for path in paths))
        self.assertFalse(any(path.endswith("stageC_frozen_frozen_own.json") for path in paths))


if __name__ == "__main__":
    unittest.main()
