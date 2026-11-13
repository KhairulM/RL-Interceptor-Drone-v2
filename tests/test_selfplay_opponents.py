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
        bank.switch = None
        bank.ids = ["first", "second", "unused"]
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

    def test_rl_opponent_switches_within_episode_and_resets(self):
        bank = object.__new__(FrozenPolicyBank)
        bank.device = torch.device("cpu")
        bank.ids = ["first", "second"]
        bank.kind = {"first": "rl", "second": "rl"}
        bank.switch = {"first_id": "first", "second_id": "second", "times_s": [0.5, 1.0, 1.5],
                       "group_id": "switch:first>second", "seed": 13}
        bank.num_envs = 4
        bank.action_dim = 2
        bank._rl = {"first": _ObservationPolicy(1), "second": _ObservationPolicy(10)}

        bank.assign({}, 4)
        self.assertEqual(bank.assignment["switch:first>second"].tolist(), [0, 1, 2, 3])
        progress = torch.floor(bank.switch_steps_s / 0.02).long() - 1
        bank.update_switch(progress, 0.02)
        self.assertEqual(bank.env_policy.tolist(), [0, 0, 0, 0])
        bank.update_switch(progress + 1, 0.02)
        self.assertEqual(bank.env_policy.tolist(), [1, 1, 1, 1])
        self.assertTrue(bank.rl_mask.all())
        obs = torch.arange(12, dtype=torch.float32).reshape(4, 1, 3)
        actions = bank.rl_raw_actions(obs)
        torch.testing.assert_close(actions[3], obs[3, ..., :2] * 10)

        bank.update_switch(torch.zeros(4, dtype=torch.long), 0.02)
        self.assertEqual(bank.env_policy.tolist(), [0, 0, 0, 0])

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
