"""Thesis F6: policy-space diversity term in PPOPolicy (set_diversity / _diversity_distance / _update)."""

import os
import tempfile
import unittest

import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from omni_drones.learning.ppo.ppo import PPOConfig, PPOPolicy

OBS, ACT = 6, 4


def _policy():
    obs = Composite({"agents": Composite({"observation": Unbounded((1, OBS))}, shape=())}, shape=())
    act = Composite({"agents": Composite({"action": Unbounded((1, ACT))}, shape=())}, shape=())
    rew = Composite({"agents": Composite({"reward": Unbounded((1, 1))}, shape=())}, shape=())
    torch.manual_seed(0)
    return PPOPolicy(PPOConfig(), obs, act, rew, device="cpu")


def _save(policy, path):
    torch.save(policy.state_dict(), path)
    return path


def _batch(n=64):
    return TensorDict({"agents": TensorDict({"observation": torch.randn(n, 1, OBS)}, [n])}, [n])


class DiversityTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _loc(self, policy, td):
        return policy.actor.get_dist(td.clone()).mean

    def test_identical_reference_gives_zero(self):
        p = _policy()
        p.set_diversity([_save(p, os.path.join(self.tmp.name, "a.pt"))], beta=1.0, cap=2.0)
        td = _batch()
        self.assertTrue(torch.allclose(p._diversity_distance(td, self._loc(p, td)), torch.zeros(64, 1), atol=1e-6))

    def test_closed_form_and_min_over_references(self):
        p = _policy()
        ref = _policy()
        with torch.no_grad():
            ref.actor.module[0].module[1].actor_mean.bias.add_(0.5)     # mean shift 0.5 in every action dim
            ref.actor.module[0].module[1].actor_std.fill_(0.0)          # sigma_j = 1
        far = _policy()
        with torch.no_grad():
            far.actor.module[0].module[1].actor_mean.bias.add_(3.0)
        p.set_diversity([_save(ref, os.path.join(self.tmp.name, "r.pt")),
                         _save(far, os.path.join(self.tmp.name, "f.pt"))], beta=1.0, cap=2.0)
        td = _batch()
        d = p._diversity_distance(td, self._loc(p, td))
        self.assertTrue(torch.allclose(d, torch.full((64, 1), 0.5 * ACT * 0.25), atol=1e-5))   # nearest = ref

    def test_learner_std_does_not_change_distance(self):
        p = _policy()
        ref = _policy()
        with torch.no_grad():
            ref.actor.module[0].module[1].actor_mean.bias.add_(0.5)
        p.set_diversity([_save(ref, os.path.join(self.tmp.name, "r.pt"))], beta=1.0, cap=2.0)
        td = _batch()
        d0 = p._diversity_distance(td, self._loc(p, td))
        with torch.no_grad():
            p.actor.module[0].module[1].actor_std.fill_(-3.0)
        self.assertTrue(torch.allclose(d0, p._diversity_distance(td, self._loc(p, td))))

    def _train_batch(self, p, n=256):
        td = _batch(n)
        with torch.no_grad():
            p.actor(td)
        td.set("adv", torch.zeros(n, 1, 1))
        td.set("ret", torch.zeros(n, 1, 1))
        td.set("state_value", torch.zeros(n, 1, 1))
        return td

    def test_update_moves_policy_away_and_cap_stops_it(self):
        p = _policy()
        ref = _policy()
        with torch.no_grad():
            ref.actor.module[0].module[1].actor_mean.bias.add_(0.05)
        p.set_diversity([_save(ref, os.path.join(self.tmp.name, "r.pt"))], beta=1.0, cap=0.5)
        td = self._train_batch(p)
        before = p._diversity_distance(td, self._loc(p, td)).mean()
        for _ in range(20):
            info = p._update(td.clone())
        after = p._diversity_distance(td, self._loc(p, td)).mean()
        self.assertGreater(after, before)
        self.assertIn("div_distance", info.keys())
        # beyond the cap the term has no gradient: zero advantages + capped distance -> actor mean unchanged by it
        p2 = _policy()
        far = _policy()
        with torch.no_grad():
            far.actor.module[0].module[1].actor_mean.bias.add_(10.0)
        p2.set_diversity([_save(far, os.path.join(self.tmp.name, "f.pt"))], beta=1.0, cap=0.5)
        td2 = self._train_batch(p2)
        p2.entropy_coef = 0.0
        w0 = p2.actor.module[0].module[1].actor_mean.weight.clone()
        p2._update(td2)
        self.assertTrue(torch.allclose(w0, p2.actor.module[0].module[1].actor_mean.weight))

    def _rollout(self, p, E=4, T=8):
        obs = torch.randn(E, T, 1, OBS)
        td = TensorDict({
            "agents": TensorDict({"observation": obs}, [E, T]),
            "next": TensorDict({"agents": TensorDict({"observation": torch.randn(E, T, 1, OBS),
                                                      "reward": torch.zeros(E, T, 1, 1)}, [E, T]),
                                "terminated": torch.zeros(E, T, 1, dtype=torch.bool)}, [E, T]),
        }, [E, T])
        with torch.no_grad():
            p(td)
        return td

    def test_intrinsic_reward_is_capped_distance_and_reaches_gae(self):
        p = _policy()
        ref = _policy()
        with torch.no_grad():
            ref.actor.module[0].module[1].actor_mean.bias.add_(0.5)
            ref.actor.module[0].module[1].actor_std.fill_(0.0)
        p.set_diversity([_save(ref, os.path.join(self.tmp.name, "r.pt"))], beta=0.3, cap=2.0)
        td = self._rollout(p)
        r = p._intrinsic_reward(td)
        self.assertEqual(r.shape, td["next", "agents", "reward"].shape)
        self.assertTrue(torch.allclose(r, torch.full_like(r, 0.3 * 0.5 * ACT * 0.25), atol=1e-5))
        p.set_diversity([_save(ref, os.path.join(self.tmp.name, "r.pt"))], beta=0.3, cap=0.1)
        self.assertTrue(torch.allclose(p._intrinsic_reward(td), torch.full_like(r, 0.3 * 0.1), atol=1e-6))
        seen = {}
        forward = p.gae.forward

        def spy(rewards, *a):
            seen["r"] = rewards.clone()
            return forward(rewards, *a)
        p.gae.forward = spy
        info = p.train_op(td.clone())
        self.assertTrue(torch.allclose(seen["r"], torch.full_like(r, 0.3 * 0.1), atol=1e-6))   # task reward is 0
        self.assertAlmostEqual(info["div_reward"], 0.03, places=5)
        self.assertTrue(torch.all(td["next", "agents", "reward"] == 0))      # stored task reward untouched

    def test_no_diversity_by_default(self):
        p = _policy()
        info = p._update(self._train_batch(p))
        self.assertNotIn("div_loss", info.keys())


if __name__ == "__main__":
    unittest.main()
