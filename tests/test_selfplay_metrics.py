import ast
from contextlib import nullcontext
import importlib.util
import math
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


if __name__ == "__main__":
    unittest.main()
