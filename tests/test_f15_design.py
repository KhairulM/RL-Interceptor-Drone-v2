from pathlib import Path
import sys
import unittest
from unittest.mock import patch

_THESIS = Path(__file__).resolve().parents[1] / "scripts/experiments/thesis"
sys.path[:0] = [str(_THESIS), str(_THESIS / "exps")]
import f14_generalization_sp as f14  # noqa: E402
import f15_switch_baseline as f15  # noqa: E402


class F15DesignTests(unittest.TestCase):
    def test_baseline_has_two_switches_and_two_unique_fixed_controls_per_seed(self):
        def fake_pair(seed, direction):
            return (
                {"id": f"f8-{seed}/evader:s25", "kind": "rl"},
                {"id": f"f10-{seed}/evader:s11", "kind": "rl"},
            )

        with patch.object(f14, "F8_CAMPS", ("c0", "c100", "c200")), \
                patch.object(f14, "_ck", return_value="pursuer.pt"), \
                patch.object(f14, "_switch_pair", side_effect=fake_pair):
            jobs = f15._baseline_jobs()

        self.assertEqual(len(jobs), 12)
        for seed in f15.SEEDS:
            seed_jobs = [job for job in jobs if job["seed"] == seed]
            self.assertEqual(sum(job["kind"] == "switch" for job in seed_jobs), 2)
            self.assertEqual(sum(job["kind"] == "fixed" for job in seed_jobs), 2)
            self.assertTrue(all(job["checkpoint"] == "pursuer.pt" for job in seed_jobs))
            self.assertEqual({job["rng_seed"] for job in seed_jobs}, {seed + 1400})

    def test_reuse_deduplicates_fixed_controls_across_directions(self):
        generated = []
        for model in ("rls", "no_rls", "rls_frozen"):
            for seed in f15.SEEDS:
                for direction in ("f8_to_f10", "f10_to_f8"):
                    policies = [
                        {"id": f"f8/evader:s{seed}"},
                        {"id": f"f10/evader:s{seed}"},
                    ]
                    switch = {
                        "first_id": policies[0]["id"],
                        "second_id": policies[1]["id"],
                        "times_s": [0.5, 1.0, 1.5],
                        "group_id": f"switch:{direction}:randomized",
                        "seed": seed + 401,
                    }
                    generated.append((f"{model}-{seed}-{direction}-switch.json", "camp", model, seed,
                                      direction, [0.5, 1.0, 1.5], policies, "ckpt", [], switch, "switch"))
                    for policy in policies:
                        generated.append((f"{model}-{seed}-{direction}-{policy['id']}.json", "camp", model, seed,
                                          direction, None, [policy], "ckpt", [], None, policy["id"]))

        with patch.object(f14, "_selected_rules", return_value="v2d"), \
                patch.object(f14, "_stagee_campaigns", return_value={"rls": (), "no_rls": ()}), \
                patch.object(f14, "_adaptation_jobs", return_value=generated):
            jobs = f15._reused_f14_jobs()

        self.assertEqual(len(jobs), 36)
        self.assertEqual(sum(job["kind"] == "switch" for job in jobs), 18)
        self.assertEqual(sum(job["kind"] == "fixed" for job in jobs), 18)


if __name__ == "__main__":
    unittest.main()
