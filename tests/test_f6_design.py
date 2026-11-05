"""F6 design checks that need no Isaac run: the frozen-space evader Vendi reproduces F3, and the arm config."""

from pathlib import Path
import json
import os
import sys
import tempfile
import unittest

_THESIS = Path(__file__).resolve().parents[1] / "scripts/experiments/thesis"
sys.path[:0] = [str(_THESIS), str(_THESIS / "exps")]
import f6_div_sp as f6  # noqa: E402


class F6DesignTests(unittest.TestCase):

    def test_arm_is_f3_hedge_plus_evader_diversity(self):
        cfg = f6.arm_cfg(0.1)
        self.assertEqual({k: v for k, v in cfg.items() if not k.startswith("diversity")}, f6.SAMPLERS["hedge"])
        self.assertEqual(f6.name(0), "thesis-F6-hedge_div-s0")
        self.assertEqual(f6.PILOT_FROM, ("thesis-F3-hedge-both-s0", 5))
        self.assertEqual(cfg["diversity_roles"], "[evader]")
        self.assertEqual((cfg["diversity_beta"], cfg["diversity_cap"]), (0.1, 2.0))
        self.assertEqual(f6._div_overrides(["a.pt"], 0.0), [])

    def test_frozen_space_vendi_reproduces_f3_and_projects_copies_identically(self):
        f3 = f6._f3_camps()
        probes = [c for _, _, c in f3]
        src = "thesis-F3-hedge-both-s1"
        fake = f6.name(0)
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "evals"))
            for c in probes:       # the F6 "evaders" are copies of src's evaders, as the probes saw them in F3c
                m = json.load(open(os.path.join(f6.F3C, "evals", f"{c}_final_campaign_evaders.json")))["matchups"]
                cp = {k.replace(src + "/", fake + "/"): v for k, v in m.items() if k.startswith(src + "/")}
                with open(os.path.join(tmp, "evals", f"probe_{c}_vs_f6_evaders.json"), "w") as f:
                    json.dump({"matchups": cp}, f)
            old, f6.F6 = f6.F6, tmp
            try:
                v = f6.evader_vendi(f3, [fake])
            finally:
                f6.F6 = old
        ref = json.load(open(os.path.join(f6.F3C, "results.json")))["per_campaign"]
        for s, k, c in f3:
            self.assertAlmostEqual(v[c]["vendi"], ref[f"{s}/{k}"]["evader_vendi"], places=9)
        self.assertAlmostEqual(v[fake]["vendi"], v[src]["vendi"], places=9)


if __name__ == "__main__":
    unittest.main()
