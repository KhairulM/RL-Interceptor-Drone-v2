import unittest

import torch
from tensordict import TensorDict
from torch import nn

from omni_drones.learning.ppo.ppo import PPOPolicy
from omni_drones.utils.selfplay.opponents import FrozenPolicyBank


class _ObservationPolicy:
    def __init__(self, scale):
        self.scale = scale
        self.batch_sizes = []

    def act(self, tensordict):
        obs = tensordict[("agents", "observation")]
        self.batch_sizes.append(obs.shape[0])
        return obs[..., :2] * self.scale


class _ActionActor:
    def __call__(self, tensordict):
        obs = tensordict[("agents", "observation")]
        tensordict.set(("agents", "action"), obs[..., :2] * 2)
        return tensordict


class _UnexpectedCritic:
    def __call__(self, tensordict):
        raise AssertionError("actor-only inference must not call the critic")


class FrozenPolicyInferenceTests(unittest.TestCase):
    def test_rl_policies_only_process_assigned_rows(self):
        bank = object.__new__(FrozenPolicyBank)
        bank.device = torch.device("cpu")
        bank.num_envs = 5
        bank.action_dim = 2
        first = _ObservationPolicy(1)
        second = _ObservationPolicy(10)
        unused = _ObservationPolicy(100)
        bank._rl = {"first": first, "second": second, "unused": unused}
        bank.assignment = {
            "first": torch.tensor([0, 3]),
            "second": torch.tensor([1]),
        }
        obs = torch.arange(15, dtype=torch.float32).reshape(5, 1, 3)

        actions = bank.rl_raw_actions(obs)

        expected = torch.zeros(5, 1, 2)
        expected[[0, 3]] = obs[[0, 3], ..., :2]
        expected[[1]] = obs[[1], ..., :2] * 10
        torch.testing.assert_close(actions, expected)
        self.assertEqual(first.batch_sizes, [2])
        self.assertEqual(second.batch_sizes, [1])
        self.assertEqual(unused.batch_sizes, [])

    def test_ppo_actor_only_method_skips_critic(self):
        policy = PPOPolicy.__new__(PPOPolicy)
        nn.Module.__init__(policy)
        policy.actor = _ActionActor()
        policy.critic = _UnexpectedCritic()
        obs = torch.arange(12, dtype=torch.float32).reshape(4, 1, 3)
        td = TensorDict(
            {("agents", "observation"): obs},
            batch_size=[4],
        )
        expected = policy.actor(
            TensorDict(
                {("agents", "observation"): obs.clone()},
                batch_size=[4],
            )
        )[("agents", "action")]

        actions = policy.act(td)

        torch.testing.assert_close(actions, expected)
        self.assertFalse(actions.requires_grad)


if __name__ == "__main__":
    unittest.main()
