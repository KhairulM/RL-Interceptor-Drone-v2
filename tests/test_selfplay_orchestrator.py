import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "selfplay" / "train_selfplay.py"
_SPEC = importlib.util.spec_from_file_location("selfplay_orchestrator", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
orchestrator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(orchestrator)


class SelfplayOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.cfg = OmegaConf.load(_SCRIPT.with_suffix(".yaml"))
        self.cfg.run_name = "test"
        self.cfg.n_stages = 6
        self.cfg.fresh_start = True
        self.calls = []
        self.fail_eval = False

    def run_campaign(self):
        with (
            patch.object(orchestrator, "_THIS_DIR", self.temp_dir.name),
            patch.object(orchestrator.OmegaConf, "load", return_value=self.cfg),
            patch.object(
                orchestrator.OmegaConf, "from_cli",
                return_value=OmegaConf.create({}),
            ),
            patch.object(orchestrator, "_run_worker", side_effect=self.worker),
        ):
            orchestrator.main()

    def database(self):
        return orchestrator.PolicyDatabase(
            str(Path(self.temp_dir.name) / "outputs" / "test" / "db")
        )

    def worker(self, cfg, overrides):
        args = dict(value.split("=", 1) for value in overrides)
        with open(args["opponent_spec_path"]) as f:
            spec = json.load(f)
        self.assertNotIn("eval_heuristic", args)
        self.calls.append((args, spec))
        if args["mode"] == "train":
            result = {"checkpoint_path": args["stage_ckpt"]}
        else:
            if self.fail_eval:
                raise RuntimeError("interrupted evaluation")
            result = {
                "matchups": {
                    policy["id"]: {
                        key: (index + 1) / (len(spec["policies"]) + 1)
                        for key in orchestrator._policy_db.METRIC_KEYS
                    }
                    for index, policy in enumerate(spec["policies"])
                }
            }
        with open(args["stage_out"], "w") as f:
            json.dump(result, f)

    def assert_pool(self, spec, kind):
        self.assertTrue(spec["policies"])
        self.assertEqual({p["kind"] for p in spec["policies"]}, {kind})
        self.assertEqual(
            {p["id"] for p in spec["policies"]}, set(spec["probs"])
        )
        self.assertAlmostEqual(sum(spec["probs"].values()), 1.0)

    def test_only_first_generation_uses_heuristics_for_either_start_role(self):
        for start_role in (orchestrator.PURSUER, orchestrator.EVADER):
            with self.subTest(start_role=start_role):
                self.cfg.start_role = start_role
                self.calls.clear()
                self.run_campaign()

                self.assertEqual(len(self.calls), 12)
                for stage in range(6):
                    train_args, train_spec = self.calls[stage * 2]
                    eval_args, eval_spec = self.calls[stage * 2 + 1]
                    kind = "heuristic" if stage < 2 else "rl"
                    self.assertEqual(int(train_args["stage"]), stage)
                    self.assertEqual(train_args["task.role"], eval_args["task.role"])
                    self.assert_pool(train_spec, kind)
                    self.assert_pool(eval_spec, kind)
                    self.assertEqual(
                        set(train_spec["probs"]), set(eval_spec["probs"])
                    )
                    self.assertTrue(all(
                        prob == 1.0 / len(eval_spec["probs"])
                        for prob in eval_spec["probs"].values()
                    ))
                    if stage < 2:
                        self.assertEqual(train_spec["probs"], eval_spec["probs"])
                    if stage >= 5:
                        self.assertGreater(
                            len(set(train_spec["probs"].values())), 1
                        )

                db = self.database()
                for role in (orchestrator.PURSUER, orchestrator.EVADER):
                    self.assertTrue(any(
                        p["kind"] == "heuristic" for p in db.list_policies(role)
                    ))
                self.assertEqual(
                    orchestrator._resume_point(db), (6, start_role, None)
                )

    def test_evader_hedge_upweights_pursuers_that_neutralise_the_reference_evader(self):
        db = self.database()
        P, E = orchestrator.PURSUER, orchestrator.EVADER
        hard = db.add_policy(P, "p0.pt", metadata={"stage": 0})
        easy = db.add_policy(P, "p2.pt", metadata={"stage": 2})
        ref = db.add_policy(E, "e1.pt", metadata={"stage": 1})
        metrics = {k: 0.5 for k in orchestrator._policy_db.METRIC_KEYS}
        db.record_matchup(hard, ref, dict(metrics, neutralization_rate=0.9))
        db.record_matchup(easy, ref, dict(metrics, neutralization_rate=0.1))
        self.cfg.evader_sampler = "hedge"
        self.cfg.hedge_gamma = 0.0                      # pfsp: p ~ c^2, c = J(pursuer, reference evader)
        probs = orchestrator._opponent_probs(db, E, ref, self.cfg)
        self.assertAlmostEqual(probs[hard], 0.81 / 0.82)
        self.assertAlmostEqual(probs[easy], 0.01 / 0.82)

    def test_two_sided_qd_dpp_samples_both_roles_from_behaviour(self):
        self.cfg.pursuer_sampler = self.cfg.evader_sampler = "qd_dpp"
        nf = 9

        def worker(cfg, overrides):
            args = dict(value.split("=", 1) for value in overrides)
            with open(args["opponent_spec_path"]) as f:
                spec = json.load(f)
            self.calls.append((args, spec))
            if args["mode"] == "train":
                result = {"checkpoint_path": args["stage_ckpt"]}
            else:
                result = {"matchups": {}}
                for i, p in enumerate(spec["policies"]):
                    m = {k: 0.5 for k in orchestrator._policy_db.METRIC_KEYS}
                    m["neutralization_rate"] = (i + 1) / (len(spec["policies"]) + 1)
                    beh = {"mean": [float(i)] * nf, "std": [1.0] * nf}
                    m["behaviour"] = m["behaviour_pursuer"] = beh
                    result["matchups"][p["id"]] = m
            with open(args["stage_out"], "w") as f:
                json.dump(result, f)

        original = orchestrator._opponent_probs
        with (patch.object(self, "worker", side_effect=worker),
              patch.object(orchestrator, "_opponent_probs", side_effect=original) as spy):
            self.run_campaign()
        self.assertEqual({c.args[1] for c in spy.call_args_list}, {orchestrator.PURSUER, orchestrator.EVADER})
        trains = [(a, s) for a, s in self.calls if a["mode"] == "train"]
        for args, spec in trains:
            if int(args["stage"]) >= 3:
                self.assert_pool(spec, "rl")
                self.assertGreater(len(set(spec["probs"].values())), 1)
        self.assertRaises(ValueError, orchestrator._opponent_probs, self.database(), orchestrator.EVADER,
                          self.database().list_policies(orchestrator.EVADER)[-1]["id"],
                          OmegaConf.merge(self.cfg, {"evader_sampler": "bogus"}))

    def test_diversity_term_only_for_listed_role_vs_earlier_same_role_checkpoints(self):
        self.cfg.diversity_roles = [orchestrator.EVADER]
        self.cfg.diversity_beta = 0.1
        self.cfg.diversity_cap = 2.0
        self.run_campaign()
        train = [args for args, _ in self.calls if args["mode"] == "train"]
        evader_ckpts = []
        for args in train:
            if args["task.role"] == orchestrator.PURSUER or not evader_ckpts:
                self.assertNotIn("diversity.ckpts", args)
            else:
                self.assertEqual(args["diversity.ckpts"], "[" + ",".join(evader_ckpts) + "]")
                self.assertEqual(float(args["diversity.beta"]), 0.1)
                self.assertEqual(float(args["diversity.cap"]), 2.0)
                self.assertEqual((args["diversity.mode"], float(args["diversity.tau"]), args["diversity.centre"]),
                                 ("kl", 1.0, "false"))                  # F6 defaults
            if args["task.role"] == orchestrator.EVADER:
                evader_ckpts.append(args["stage_ckpt"])
        self.assertEqual(len(evader_ckpts), 3)

    def test_diversity_action_mode_overrides(self):                     # thesis F6b
        self.cfg.diversity_roles = [orchestrator.EVADER]
        self.cfg.diversity_beta = 0.1
        self.cfg.diversity_mode = "action"
        self.cfg.diversity_tau = 0.07
        self.cfg.diversity_centre = True
        self.run_campaign()
        div = [args for args, _ in self.calls if args["mode"] == "train" and "diversity.ckpts" in args]
        self.assertTrue(div)
        for args in div:
            self.assertEqual((args["diversity.mode"], float(args["diversity.tau"]), args["diversity.centre"]),
                             ("action", 0.07, "true"))

    def test_no_diversity_term_by_default(self):
        self.run_campaign()
        self.assertFalse(any("diversity.ckpts" in args for args, _ in self.calls))

    def test_max_total_stages_caps_a_resumed_campaign(self):
        self.cfg.n_stages = 3
        self.run_campaign()                     # fresh: stages 0-2
        self.cfg.fresh_start = False
        self.cfg.max_total_stages = 5
        self.calls.clear()
        self.run_campaign()                     # resume at 3: without the cap it would run 3-5
        trained = [int(a["stage"]) for a, _ in self.calls if a["mode"] == "train"]
        self.assertEqual(trained, [3, 4])
        self.assertEqual(orchestrator._resume_point(self.database())[0], 5)

    def test_resuming_first_generation_still_uses_heuristics(self):
        for start_role in (orchestrator.PURSUER, orchestrator.EVADER):
            with self.subTest(start_role=start_role):
                self.cfg.start_role = start_role
                self.cfg.fresh_start = True
                self.cfg.n_stages = 1
                self.fail_eval = False
                self.run_campaign()
                self.cfg.fresh_start = False
                self.fail_eval = True
                with self.assertRaisesRegex(RuntimeError, "interrupted evaluation"):
                    self.run_campaign()
                db = self.database()
                role = orchestrator._other(start_role)
                policy = next(
                    p for p in db.list_policies(role) if p["kind"] == "rl"
                )
                self.assertEqual(
                    orchestrator._resume_point(db), (1, role, policy)
                )

                self.cfg.fresh_start = False
                self.fail_eval = False
                self.calls.clear()
                self.run_campaign()

                self.assertEqual(len(self.calls), 2)
                for _, spec in self.calls:
                    self.assert_pool(spec, "heuristic")
                db = self.database()
                self.assertEqual(
                    [p["id"] for p in db.list_policies(role)
                     if p["kind"] == "rl"],
                    [policy["id"]],
                )
                self.assertEqual(
                    orchestrator._resume_point(db),
                    (2, start_role, None),
                )

    def test_uniform_mixture_is_forwarded_to_sampling(self):
        self.cfg.uniform_mix = 0.5
        original = orchestrator.PolicyDatabase.compute_sampling_probs
        with patch.object(
            orchestrator.PolicyDatabase, "compute_sampling_probs",
            autospec=True, side_effect=original,
        ) as sampling:
            self.run_campaign()
        self.assertEqual(sampling.call_count, 4)
        for call in sampling.call_args_list:
            self.assertEqual(call.kwargs["uniform_mix"], 0.5)
        for args, spec in self.calls:
            if args["mode"] == "train":
                floor = 0.5 / len(spec["probs"])
                self.assertTrue(all(prob >= floor for prob in spec["probs"].values()))

    def test_invalid_uniform_mixture_fails_before_launching_workers(self):
        self.cfg.uniform_mix = 1.1
        with self.assertRaisesRegex(ValueError, "uniform_mix"):
            self.run_campaign()
        self.assertEqual(self.calls, [])

    def test_resuming_later_generation_does_not_require_scripted_matchups(self):
        self.run_campaign()
        self.cfg.fresh_start = False
        self.cfg.n_stages = 1
        self.calls.clear()

        self.run_campaign()

        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[0][0]["stage"], "6")
        for _, spec in self.calls:
            self.assert_pool(spec, "rl")
        self.assertEqual(
            orchestrator._resume_point(self.database()),
            (7, orchestrator.EVADER, None),
        )

    def test_empty_rl_pool_fails_instead_of_falling_back_to_heuristics(self):
        db = self.database()
        orchestrator._seed_pool(db, orchestrator.PURSUER, ["apf"])
        orchestrator._seed_pool(db, orchestrator.EVADER, ["flee"])
        db.add_policy(
            orchestrator.PURSUER, "pursuer_stage0.pt", metadata={"stage": 0}
        )
        db.add_policy(
            orchestrator.PURSUER, "pursuer_stage2.pt", metadata={"stage": 2}
        )
        self.cfg.fresh_start = False
        self.cfg.n_stages = 2
        with self.assertRaisesRegex(ValueError, "no rl opponents"):
            self.run_campaign()

    def test_resuming_later_generation_preserves_sampling_reference(self):
        self.run_campaign()
        self.cfg.fresh_start = False
        self.cfg.n_stages = 1
        self.fail_eval = True
        self.calls.clear()
        with self.assertRaisesRegex(RuntimeError, "interrupted evaluation"):
            self.run_campaign()
        original_train_spec = self.calls[0][1]
        db = self.database()
        stage, role, policy = orchestrator._resume_point(db)
        self.assertEqual((stage, role), (6, orchestrator.PURSUER))
        self.assertIsNotNone(policy)

        self.fail_eval = False
        self.calls.clear()
        self.run_campaign()

        self.assertEqual(self.calls[0][1], original_train_spec)
        for _, spec in self.calls:
            self.assert_pool(spec, "rl")
        self.assertEqual(
            len([p for p in self.database().list_policies(role)
                 if p["kind"] == "rl"]),
            4,
        )
        self.assertEqual(
            orchestrator._resume_point(self.database()),
            (7, orchestrator.EVADER, None),
        )


if __name__ == "__main__":
    unittest.main()
