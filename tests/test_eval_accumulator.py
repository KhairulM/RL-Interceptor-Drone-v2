import importlib.util
from pathlib import Path
import unittest

import torch

_PATH = Path(__file__).resolve().parents[1] / "omni_drones/utils/selfplay/eval_accumulator.py"
_SPEC = importlib.util.spec_from_file_location("eval_accumulator", _PATH)
ea = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ea)

NF = len(ea.BEHAVIOUR_FEATURES)


def _stats(n, **vals):
    s = {k: torch.zeros(n, 1) for k in ea.EPISODE_STATS}
    for k, v in vals.items():
        s[k] = torch.as_tensor(v, dtype=torch.float32).reshape(n, 1)
    return s


def _simulate(lengths_per_env, cutoff, total_steps, outcome="capture"):
    """Env i runs episodes of fixed length lengths_per_env[i]; returns counted episodes."""
    n = len(lengths_per_env)
    acc = ea.EpisodeAccumulator(n, cutoff, "cpu")
    progress = torch.zeros(n)
    for t in range(total_steps):
        progress += 1
        done = torch.tensor([progress[i].item() >= lengths_per_env[i] for i in range(n)])
        stats = _stats(n, episode_len=progress.clone(), **{outcome: done.float()})
        feats = torch.full((n, NF), float(t))
        acc.step(t, done, stats, feats)
        progress[done] = 0
        if acc.all_counted_done():
            break
    return acc.episodes()


class EvalAccumulatorTests(unittest.TestCase):
    def test_cutoff_one_counts_first_episode_only(self):
        ep = _simulate([3, 5, 7], cutoff=1, total_steps=50)
        self.assertEqual(sorted(ep["env"].tolist()), [0, 1, 2])
        self.assertTrue((ep["episode_index"] == 0).all())

    def test_counts_every_episode_started_before_cutoff(self):
        # env0 (len 4) starts episodes at 0, 4, 8 -> 3 counted with cutoff 10; env1 (len 10) starts at 0 only.
        ep = _simulate([4, 10], cutoff=10, total_steps=40)
        self.assertEqual((ep["env"] == 0).sum().item(), 3)
        self.assertEqual((ep["env"] == 1).sum().item(), 1)
        self.assertTrue((ep["episode_len"][ep["env"] == 0] == 4).all())

    def test_features_exclude_terminal_step_and_reset_per_episode(self):
        ep = _simulate([3], cutoff=4, total_steps=20)
        # episode 0: steps 0..2, done at 2 -> features of steps 0,1 (mean 0.5); episode 1: steps 3..5 -> 3,4
        means = ep["feat_mean"][:, 0].tolist()
        self.assertAlmostEqual(means[0], 0.5)
        self.assertAlmostEqual(means[1], 3.5)
        self.assertAlmostEqual(ep["feat_std"][0, 0].item(), 0.5, places=5)

    def test_summarize_neutralisation_censoring_and_roles(self):
        episodes = {
            "env": torch.tensor([0, 0, 1, 1]),
            "episode_index": torch.tensor([0, 1, 0, 1]),
            "capture": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "opp_crash": torch.tensor([0.0, 1.0, 0.0, 0.0]),       # evader crash (agent = pursuer)
            "opp_out_of_bounds": torch.tensor([0.0, 0.0, 0.0, 0.0]),
            "crash": torch.tensor([0.0, 0.0, 1.0, 0.0]),           # pursuer crash
            "out_of_bounds": torch.tensor([0.0, 0.0, 0.0, 0.0]),
            "timeout": torch.tensor([0.0, 0.0, 0.0, 1.0]),
            "episode_len": torch.tensor([10.0, 20.0, 30.0, 100.0]),
            "min_distance": torch.zeros(4), "motor_effort": torch.zeros(4),
            "feat_mean": torch.zeros(4, NF), "feat_std": torch.zeros(4, NF),
        }
        out = ea.summarize_matchups(episodes, {"opp": torch.tensor([0, 1])}, "pursuer",
                                    step_dt=0.02, max_episode_steps=100)["opp"]
        self.assertEqual(out["episodes"], 4)
        self.assertAlmostEqual(out["capture_rate"], 0.25)
        self.assertAlmostEqual(out["neutralization_rate"], 0.5)
        self.assertAlmostEqual(out["win_rate"], 1 / 3)               # 1 capture vs crash + timeout
        self.assertAlmostEqual(out["avg_time_to_capture"], 0.2)      # 10 steps x 0.02 s
        # censored: 0.2, 0.4, 2.0, 2.0
        self.assertAlmostEqual(out["time_to_neutralize_censored"], (0.2 + 0.4 + 2.0 + 2.0) / 4)
        # Evaluated as the evader: "crash" is now the evader's own crash.
        out_e = ea.summarize_matchups(episodes, {"opp": torch.tensor([0, 1])}, "evader",
                                      step_dt=0.02, max_episode_steps=100)["opp"]
        self.assertAlmostEqual(out_e["outcomes"]["evader_crash"], 0.25)
        self.assertAlmostEqual(out_e["outcomes"]["pursuer_crash"], 0.25)

    def test_behaviour_features_straight_flight_away(self):
        e_pos = torch.tensor([[1.0, 0.0, 1.5]])
        e_vel = torch.tensor([[2.0, 0.0, 0.0]])
        f = ea.behaviour_step_features(e_pos, e_vel, torch.tensor([[0.0, 0.0, 1.5]]), e_vel.clone(), 0.02)[0]
        names = ea.BEHAVIOUR_FEATURES
        self.assertAlmostEqual(f[names.index("evader_speed")].item(), 2.0)
        self.assertAlmostEqual(f[names.index("evader_accel")].item(), 0.0)
        self.assertAlmostEqual(f[names.index("evader_turn_rate")].item(), 0.0)
        self.assertAlmostEqual(f[names.index("flee_cos")].item(), 1.0, places=5)
        self.assertAlmostEqual(f[names.index("distance")].item(), 1.0, places=5)
        self.assertAlmostEqual(f[names.index("evader_altitude")].item(), 1.5)

    def test_behaviour_turn_rate_and_lateral_accel(self):
        prev = torch.tensor([[1.0, 0.0, 0.0]])
        now = torch.tensor([[0.0, 1.0, 0.0]])
        f = ea.behaviour_step_features(torch.zeros(1, 3), now, torch.tensor([[-1.0, 0.0, 0.0]]), prev, 0.5)[0]
        names = ea.BEHAVIOUR_FEATURES
        self.assertAlmostEqual(f[names.index("evader_turn_rate")].item(), (torch.pi / 2) / 0.5, places=5)
        # acc = (-2, 2, 0); velocity along y -> lateral component = 2
        self.assertAlmostEqual(f[names.index("evader_lat_accel")].item(), 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
