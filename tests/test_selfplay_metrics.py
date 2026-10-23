import ast
from contextlib import nullcontext
import glob
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict


_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = _ROOT / "omni_drones/utils/selfplay/policy_db.py"
_SPEC = importlib.util.spec_from_file_location("selfplay_metrics_db", _DB_PATH)
assert _SPEC is not None and _SPEC.loader is not None
policy_db = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(policy_db)


class SelfplayMetricTests(unittest.TestCase):
    def test_uniform_mixture_matches_formula_and_probability_floor(self):
        for role in (policy_db.PURSUER, policy_db.EVADER):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as root:
                db = policy_db.PolicyDatabase(root)
                other = policy_db.EVADER if role == policy_db.PURSUER else policy_db.PURSUER
                reference = db.add_policy(role, "reference.pt")
                opponents = [db.add_policy(other, f"opponent{i}.pt") for i in range(3)]
                db.add_policy(other, None, kind="heuristic")
                for opponent, win_rate in zip(opponents, [0.0, 0.5, 1.0]):
                    metrics = dict.fromkeys(policy_db.METRIC_KEYS, 0.0)
                    metrics["win_rate"] = win_rate
                    pair = (reference, opponent) if role == policy_db.PURSUER else (opponent, reference)
                    db.record_matchup(*pair, metrics)
                prioritized = db.compute_sampling_probs(role, temperature=0.01, kind="rl")
                for mix in (0.0, 0.5, 1.0):
                    probs = db.compute_sampling_probs(
                        role, temperature=0.01, kind="rl", uniform_mix=mix,
                    )
                    self.assertEqual(set(probs), set(opponents))
                    self.assertAlmostEqual(sum(probs.values()), 1.0)
                    for opponent in opponents:
                        self.assertAlmostEqual(
                            probs[opponent], (1 - mix) * prioritized[opponent] + mix / 3,
                        )
                        self.assertGreaterEqual(probs[opponent], mix / 3)

    def test_uniform_mixture_validation_and_cold_start(self):
        with tempfile.TemporaryDirectory() as root:
            db = policy_db.PolicyDatabase(root)
            for mix in (-0.1, 1.1, math.nan, math.inf):
                with self.subTest(mix=mix), self.assertRaisesRegex(ValueError, "uniform_mix"):
                    db.compute_sampling_probs(policy_db.PURSUER, uniform_mix=mix)
            self.assertEqual(db.compute_sampling_probs(policy_db.PURSUER, uniform_mix=0.5), {})
            first = db.add_policy(policy_db.EVADER, "first.pt")
            self.assertEqual(
                db.compute_sampling_probs(policy_db.PURSUER, uniform_mix=0.5), {first: 1.0},
            )
            second = db.add_policy(policy_db.EVADER, "second.pt")
            self.assertEqual(
                db.compute_sampling_probs(policy_db.PURSUER, uniform_mix=0.5),
                {first: 0.5, second: 0.5},
            )

    def test_pursuer_crash_rate_orientation_for_both_roles(self):
        for role in (policy_db.PURSUER, policy_db.EVADER):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as root:
                db = policy_db.PolicyDatabase(root)
                other = policy_db.EVADER if role == policy_db.PURSUER else policy_db.PURSUER
                reference = db.add_policy(role, "reference.pt")
                opponents = [db.add_policy(other, f"opponent{i}.pt") for i in range(2)]
                for opponent, crash_rate in zip(opponents, [0.1, 0.9]):
                    metrics = dict.fromkeys(policy_db.METRIC_KEYS, 0.0)
                    metrics["pursuer_crash_rate"] = crash_rate
                    pair = (reference, opponent) if role == policy_db.PURSUER else (opponent, reference)
                    db.record_matchup(*pair, metrics)
                weights = dict.fromkeys(policy_db.METRIC_KEYS, 0.0)
                weights["pursuer_crash_rate"] = 1.0
                probs = db.compute_sampling_probs(role, metric_weights=weights)
                harder, easier = opponents[::-1] if role == policy_db.PURSUER else opponents
                self.assertAlmostEqual(probs[harder] / probs[easier], math.e)

    def test_historical_crash_rates_use_pursuer_outcomes_not_agent_errors(self):
        with tempfile.TemporaryDirectory() as root:
            records = {}
            for role in (policy_db.PURSUER, policy_db.EVADER):
                records[f"pursuer|{role}"] = {
                    "avg_min_distance": 0.1, "agent_error_rate": 0.9,
                    "eval_role": role, "outcomes": {"pursuer_crash": 0.2},
                }
            records["pursuer|unknown"] = {
                "avg_min_distance": 0.1, "agent_error_rate": 0.9,
            }
            records["pursuer|new"] = {
                "avg_min_distance": 0.1, "pursuer_crash_rate": 0.3,
                "outcomes": {"pursuer_crash": 0.8},
            }
            path = Path(root) / "matchups.json"
            path.write_text(json.dumps({"matchups": records}))
            with self.assertLogs(level="WARNING") as logs:
                db = policy_db.PolicyDatabase(root)
            self.assertTrue(any("pursuer_crash_rate" in line for line in logs.output))
            for role in (policy_db.PURSUER, policy_db.EVADER):
                record = db.get_matchup("pursuer", role)
                self.assertEqual(record["pursuer_crash_rate"], 0.2)
                self.assertNotIn("agent_error_rate", record)
            self.assertNotIn("pursuer_crash_rate", db.get_matchup("pursuer", "unknown"))
            self.assertEqual(db.get_matchup("pursuer", "new")["pursuer_crash_rate"], 0.3)
            raw = {key: db.get_matchup("pursuer", key) for key in ("pursuer", "unknown", "new")}
            self.assertEqual(policy_db._normalise_metrics(raw)["unknown"]["pursuer_crash_rate"], 0.5)
            db.save()
            saved = json.loads(path.read_text())["matchups"]
            self.assertEqual(saved["pursuer|evader"]["pursuer_crash_rate"], 0.2)

    def test_legacy_metric_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            db = policy_db.PolicyDatabase(root)
            with self.assertRaisesRegex(ValueError, "pursuer_crash_rate"):
                db.compute_sampling_probs(
                    policy_db.PURSUER, metric_weights={"agent_error_rate": 1.0},
                )

    def test_report_reads_new_and_historical_pursuer_crash_rates(self):
        path = _ROOT / "scripts/experiments/esn_adapt/report.py"
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "load")
        with tempfile.TemporaryDirectory() as root:
            scope = {"ROOT": root, "SETS": ["T0"], "glob": glob, "json": json, "os": os}
            exec(compile(ast.Module([function], type_ignores=[]), str(path), "exec"), scope)
            output = Path(root) / "test" / "method" / "seed0" / "eval_T0.json"
            output.parent.mkdir(parents=True)
            records = {
                "new": {"pursuer_crash_rate": 0.2},
                "legacy": {"agent_error_rate": 0.9, "outcomes": {"pursuer_crash": 0.3}},
            }
            output.write_text(json.dumps({"matchups": records}))
            data, _ = scope["load"]("test")
            matchups = data["method"]["T0"]["seed0"]
            self.assertEqual(matchups["new"]["pursuer_crash_rate"], 0.2)
            self.assertEqual(matchups["legacy"]["pursuer_crash_rate"], 0.3)
            records["unknown"] = {"agent_error_rate": 0.9}
            output.write_text(json.dumps({"matchups": records}))
            with self.assertRaisesRegex(ValueError, "re-evaluate"):
                scope["load"]("test")

    def test_sampling_distance_orientation_for_both_roles(self):
        for role in (policy_db.PURSUER, policy_db.EVADER):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as root:
                db = policy_db.PolicyDatabase(root)
                other = policy_db.EVADER if role == policy_db.PURSUER else policy_db.PURSUER
                reference = db.add_policy(role, "reference.pt")
                opponents = [db.add_policy(other, f"opponent{i}.pt") for i in range(2)]
                for opponent, distance in zip(opponents, [0.1, 2.0]):
                    metrics = dict.fromkeys(policy_db.METRIC_KEYS, 0.0)
                    metrics["avg_min_distance"] = distance
                    pair = (reference, opponent) if role == policy_db.PURSUER else (opponent, reference)
                    db.record_matchup(*pair, metrics)
                weights = dict.fromkeys(policy_db.METRIC_KEYS, 0.0)
                weights["avg_min_distance"] = 0.5
                probs = db.compute_sampling_probs(role, metric_weights=weights)
                harder, easier = opponents[::-1] if role == policy_db.PURSUER else opponents
                self.assertAlmostEqual(probs[harder] / probs[easier], math.exp(0.5))
                self.assertAlmostEqual(sum(probs.values()), 1.0)
                weights["avg_min_distance"] = 0.0
                self.assertEqual(
                    db.compute_sampling_probs(role, metric_weights=weights),
                    dict.fromkeys(opponents, 0.5),
                )

    def test_legacy_distance_is_not_reinterpreted_as_minimum(self):
        with tempfile.TemporaryDirectory() as root:
            db = policy_db.PolicyDatabase(root)
            metrics = dict.fromkeys(policy_db.METRIC_KEYS, 0.0)
            for evader, distance in [("old", 99.0), ("near", 0.1), ("far", 2.0)]:
                db.record_matchup("pursuer", evader, {**metrics, "avg_min_distance": distance})
            legacy = db.get_matchup("pursuer", "old")
            legacy["avg_distance"] = legacy.pop("avg_min_distance")
            db.save()
            with self.assertLogs(level="WARNING") as logs:
                loaded = policy_db.PolicyDatabase(root)
            self.assertIn("cannot be converted", logs.output[0])
            raw = {key: loaded.get_matchup("pursuer", key) for key in ("old", "near", "far")}
            normed = policy_db._normalise_metrics(raw)
            self.assertEqual(normed["old"]["avg_min_distance"], 0.5)
            self.assertEqual(normed["near"]["avg_min_distance"], 0.0)
            self.assertEqual(normed["far"]["avg_min_distance"], 1.0)

    def test_episode_minimum_initialization_updates_and_partial_reset(self):
        # Execute the actual tracker statements without importing the Isaac stack.
        path = _ROOT / "omni_drones/envs/single/pursuit_evasion.py"
        tree = ast.parse(path.read_text())
        statements = {
            node.name: [
                child for child in ast.walk(node)
                if isinstance(child, ast.Assign)
                and ast.unparse(child.targets[0]).startswith("self.stats['min_distance']")
            ]
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name in ("_reset_idx", "_compute_state_and_obs")
        }
        reset = compile(ast.Module(statements["_reset_idx"], type_ignores=[]), str(path), "exec")
        update = compile(
            ast.Module(statements["_compute_state_and_obs"], type_ignores=[]), str(path), "exec"
        )
        self.assertEqual(len(statements["_reset_idx"]), 1)
        self.assertEqual(len(statements["_compute_state_and_obs"]), 1)
        state = SimpleNamespace(
            stats={"min_distance": torch.zeros(2, 1)},
            prev_distance=torch.tensor([[2.0], [3.0]]),
        )
        scope = {"self": state, "torch": torch, "env_ids": torch.tensor([0, 1])}
        exec(reset, scope)
        for distances in ([[1.0], [4.0]], [[1.5], [0.1]]):
            scope["distance"] = torch.tensor(distances)
            exec(update, scope)
        torch.testing.assert_close(state.stats["min_distance"], torch.tensor([[1.0], [0.1]]))
        state.prev_distance[0] = 5.0
        scope["env_ids"] = torch.tensor([0])
        exec(reset, scope)
        torch.testing.assert_close(state.stats["min_distance"], torch.tensor([[5.0], [0.1]]))

    def test_evaluation_averages_first_episode_minima_per_opponent(self):
        path = _ROOT / "scripts/selfplay/run_stage.py"
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "_run_eval")
        write = Mock()
        scope = {
            "torch": torch,
            "_load_state_dict": Mock(),
            "_write_stage_out": write,
            "set_exploration_type": lambda _: nullcontext(),
            "ExplorationType": SimpleNamespace(MODE=None),
        }
        exec(compile(ast.Module([function], type_ignores=[]), str(path), "exec"), scope)
        # Later episodes have smaller minima and must not enter this matchup.
        minimum = torch.tensor([[[2.0], [1.0], [0.0]],
                                [[4.0], [3.0], [2.0]],
                                [[8.0], [7.0], [0.0]]])
        done = torch.tensor([[[False], [True], [True]],
                             [[False], [False], [True]],
                             [[False], [True], [True]]])
        stats = {key: torch.zeros(3, 3, 1)
                 for key in ("capture", "time_to_capture", "motor_effort")}
        stats.update(min_distance=minimum, distance=torch.full((3, 3, 1), 99.0))
        stats["crash"] = torch.tensor([[[0.0], [1.0], [0.0]],
                                      [[0.0], [0.0], [0.0]],
                                      [[0.0], [0.0], [1.0]]])
        stats["opp_crash"] = torch.tensor([[[0.0], [0.0], [1.0]],
                                          [[0.0], [0.0], [0.0]],
                                          [[0.0], [1.0], [0.0]]])
        stats["out_of_bounds"] = torch.ones(3, 3, 1)
        stats["opp_out_of_bounds"] = torch.ones(3, 3, 1)
        stats["timeout"] = torch.zeros(3, 3, 1)
        trajs = TensorDict(
            {("next", "done"): done,
             ("next", "stats"): TensorDict(stats, batch_size=[3, 3])},
            batch_size=[3, 3],
        )
        base = SimpleNamespace(
            eval=Mock(), device="cpu", max_episode_length=3,
            opponent_bank=SimpleNamespace(assignment={
                "first": torch.tensor([0, 1]), "second": torch.tensor([2]),
            }),
        )
        env = Mock()
        env.rollout.return_value = trajs
        cfg = OmegaConf.create({
            "seed": 0, "eval_policy": "policy.pt",
            "sim": {"dt": 0.1}, "task": {"role": "pursuer"},
        })
        for role in (policy_db.PURSUER, policy_db.EVADER):
            with self.subTest(role=role):
                cfg.task.role = role
                scope["_run_eval"](cfg, env, base, Mock())
                matchups = write.call_args.args[1]["matchups"]
                self.assertAlmostEqual(matchups["first"]["avg_min_distance"], 1.5)
                self.assertAlmostEqual(matchups["second"]["avg_min_distance"], 7.0)
                self.assertNotIn("avg_distance", matchups["first"])
                self.assertEqual(matchups["first"]["episodes"], 2)
                expected = [0.5, 0.0] if role == policy_db.PURSUER else [0.0, 1.0]
                for opponent, crash_rate in zip(("first", "second"), expected):
                    self.assertEqual(matchups[opponent]["pursuer_crash_rate"], crash_rate)
                    self.assertEqual(
                        matchups[opponent]["outcomes"]["pursuer_crash"], crash_rate,
                    )
                    self.assertNotIn("agent_error_rate", matchups[opponent])
                    self.assertNotIn("agent_terminal_win_rate", matchups[opponent])

    def test_evaluation_win_rate_uses_decisive_pursuer_outcomes(self):
        path = _ROOT / "scripts/selfplay/run_stage.py"
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "_run_eval")
        write = Mock()
        scope = {
            "torch": torch,
            "_load_state_dict": Mock(),
            "_write_stage_out": write,
            "set_exploration_type": lambda _: nullcontext(),
            "ExplorationType": SimpleNamespace(MODE=None),
        }
        exec(compile(ast.Module([function], type_ignores=[]), str(path), "exec"), scope)

        def column(values):
            return torch.tensor(values, dtype=torch.float32).reshape(-1, 1, 1)

        # Episodes: capture+pursuer crash (win), timeout (loss), pursuer crash
        # (loss), evader crash (neutral), evader OOB (neutral) | evader crash.
        capture = column([1, 0, 0, 0, 0, 0])
        timeout = column([0, 1, 0, 0, 0, 0])
        pursuer_crash = column([1, 0, 1, 0, 0, 0])
        evader_crash = column([0, 0, 0, 1, 0, 1])
        evader_oob = column([0, 0, 0, 0, 1, 0])
        base = SimpleNamespace(
            eval=Mock(), device="cpu", max_episode_length=1,
            opponent_bank=SimpleNamespace(assignment={
                "mixed": torch.arange(5), "neutral": torch.tensor([5]),
            }),
        )
        cfg = OmegaConf.create({
            "seed": 0, "eval_policy": "policy.pt",
            "sim": {"dt": 0.1}, "task": {"role": "pursuer"},
        })
        for role in (policy_db.PURSUER, policy_db.EVADER):
            with self.subTest(role=role):
                is_pursuer = role == policy_db.PURSUER
                stats = {key: torch.zeros(6, 1, 1)
                         for key in ("time_to_capture", "motor_effort", "min_distance")}
                stats.update(
                    capture=capture, timeout=timeout,
                    crash=pursuer_crash if is_pursuer else evader_crash,
                    opp_crash=evader_crash if is_pursuer else pursuer_crash,
                    out_of_bounds=torch.zeros(6, 1, 1) if is_pursuer else evader_oob,
                    opp_out_of_bounds=evader_oob if is_pursuer else torch.zeros(6, 1, 1),
                )
                env = Mock()
                env.rollout.return_value = TensorDict(
                    {("next", "done"): torch.ones(6, 1, 1, dtype=torch.bool),
                     ("next", "stats"): TensorDict(stats, batch_size=[6, 1])},
                    batch_size=[6, 1],
                )
                cfg.task.role = role
                scope["_run_eval"](cfg, env, base, Mock())
                matchups = write.call_args.args[1]["matchups"]
                mixed = matchups["mixed"]
                self.assertAlmostEqual(mixed["win_rate"], 1 / 3)
                self.assertEqual(mixed["decisive_episodes"], 3)
                self.assertEqual(mixed["episodes"], 5)
                self.assertAlmostEqual(mixed["capture_rate"], 0.2)
                self.assertAlmostEqual(mixed["outcomes"]["capture"], 0.2)
                neutral = matchups["neutral"]
                self.assertIsNone(neutral["win_rate"])
                self.assertEqual(neutral["decisive_episodes"], 0)
                self.assertEqual(neutral["capture_rate"], 0.0)

    def test_undefined_win_rate_is_recorded_and_sampled_as_neutral(self):
        with tempfile.TemporaryDirectory() as root:
            db = policy_db.PolicyDatabase(root)
            reference = db.add_policy(policy_db.PURSUER, "reference.pt")
            evaders = [db.add_policy(policy_db.EVADER, f"e{i}.pt") for i in range(3)]
            for evader, win_rate in zip(evaders, [0.0, None, 1.0]):
                metrics = dict.fromkeys(policy_db.METRIC_KEYS, 0.0)
                metrics["win_rate"] = win_rate
                db.record_matchup(reference, evader, metrics)
            self.assertIsNone(db.get_matchup(reference, evaders[1])["win_rate"])
            reloaded = policy_db.PolicyDatabase(root)
            self.assertIsNone(reloaded.get_matchup(reference, evaders[1])["win_rate"])
            probs = reloaded.compute_sampling_probs(
                policy_db.PURSUER, metric_weights={m: 0.0 for m in policy_db.METRIC_KEYS}
                | {"win_rate": 1.0},
            )
            # Hardest (0.0) > undefined (neutral 0.5) > easiest (1.0).
            self.assertGreater(probs[evaders[0]], probs[evaders[1]])
            self.assertGreater(probs[evaders[1]], probs[evaders[2]])


if __name__ == "__main__":
    unittest.main()
