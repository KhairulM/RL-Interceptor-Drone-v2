# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from .modules.networks import MLP, ENCODERS_MAP
from .common import make_encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.nn import TensorDictModule
from tensordict import TensorDict, TensorDictBase

from torchrl.data import (
    TensorSpec,
    Bounded,
    Unbounded,
    Composite,
    TensorDictReplayBuffer
)
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.objectives.utils import hold_out_net

import copy
from tqdm import tqdm
from .common import soft_update


class MATD3Policy(object):
    """MADDPG / MATD3 trainer for the modern nested-key env layout.

    Expects the env to expose:
        observation_spec[("agents", "observation")]          -> [n, obs_dim]
        observation_spec[("agents", "observation_central")]  -> [n, state_dim]
        action_spec      [("agents", "action")]              -> [n, act_dim]
        reward_spec      [("agents", "reward")]              -> [n, 1]
    """

    def __init__(self,
                 cfg,
                 observation_spec: Composite,
                 action_spec: TensorSpec,
                 reward_spec: TensorSpec,
                 device: str = "cuda",
                 ) -> None:
        self.cfg = cfg
        self.device = device

        self.gradient_steps = int(cfg.gradient_steps)
        self.batch_size = int(cfg.batch_size)
        self.buffer_size = int(cfg.buffer_size)

        self.target_noise = self.cfg.target_noise
        self.policy_noise = self.cfg.policy_noise
        self.noise_clip = self.cfg.noise_clip

        # Modern nested keys used by MAPPO and the multi-agent envs.
        self.obs_name = ("agents", "observation")
        self.act_name = ("agents", "action")
        self.state_name = ("agents", "observation_central")
        self.reward_name = ("agents", "reward")

        # Per-agent specs (shape [n, dim]).
        self._obs_spec = observation_spec[self.obs_name]
        self._action_spec = action_spec[self.act_name]
        try:
            self._state_spec = observation_spec[self.state_name]
            self._has_state = True
        except (KeyError, IndexError):
            self._state_spec = self._obs_spec
            self._has_state = False

        self.num_agents, self.action_dim = self._action_spec.shape[-2:]

        train_agent_indices = self.cfg.get("train_agent_indices", None)
        if train_agent_indices is None:
            self.train_agent_indices = list(range(self.num_agents))
        else:
            self.train_agent_indices = sorted(
                {int(i) for i in train_agent_indices if 0 <= int(i) < self.num_agents}
            )
            if not self.train_agent_indices:
                raise ValueError(
                    "cfg.train_agent_indices is empty or out of range for MATD3Policy"
                )

        self.make_model()

        self.replay_buffer = TensorDictReplayBuffer(
            batch_size=self.batch_size,
            storage=LazyTensorStorage(max_size=self.buffer_size, device="cpu"),
            sampler=RandomSampler(),
        )

    def make_model(self):

        self.policy_in_keys = [self.obs_name]
        self.policy_out_keys = [self.act_name]

        def create_actor():
            encoder = make_encoder(self.cfg.actor, self._obs_spec)
            return TensorDictModule(
                nn.Sequential(
                    encoder,
                    nn.ELU(),
                    nn.Linear(encoder.output_shape.numel(), self.action_dim),
                    nn.Tanh()
                ),
                in_keys=self.policy_in_keys,
                out_keys=self.policy_out_keys
            ).to(self.device)

        if self.cfg.share_actor:
            self.actor = create_actor()
            self.actor_target = copy.deepcopy(self.actor)
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor.lr)
            self._shared_actor = True
        else:
            self.actors = nn.ModuleList([create_actor() for _ in range(self.num_agents)])
            self.actors_target = nn.ModuleList([copy.deepcopy(actor) for actor in self.actors])
            self.actor = self.actors[0]
            self.actor_opt = torch.optim.Adam(self.actors.parameters(), lr=self.cfg.actor.lr)
            self._shared_actor = False

        critic_state_spec = self._state_spec if self._has_state else self._obs_spec
        self.value_in_keys = [self.state_name if self._has_state else self.obs_name, self.act_name]
        self.value_out_keys = [("agents", "q")]

        self.critic = Critic(
            self.cfg.critic,
            self.num_agents,
            critic_state_spec,
            self._action_spec
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic.lr)
        self.critic_loss_fn = {"mse": F.mse_loss, "smooth_l1": F.smooth_l1_loss}[self.cfg.critic_loss]

    def __call__(self, *args, deterministic: bool = False, **kwargs):
        """Support both TensorDict and TensorDictModule positional call styles."""
        if len(args) == 1 and isinstance(args[0], TensorDictBase):
            tensordict = args[0]
            actor_output = self._call_actor(tensordict)
            action = actor_output[self.act_name]  # [..., n, action_dim]
            if not deterministic and self.policy_noise > 0:
                action_noise = (
                    torch.randn_like(action)
                    .mul_(self.policy_noise)
                    .clamp_(-self.noise_clip, self.noise_clip)
                )
                action = action.add(action_noise)
            action = action.clamp(-1, 1)
            tensordict.set(self.act_name, action)
            return tensordict

        if not args:
            raise TypeError("MATD3Policy.__call__ expected at least one positional argument")

        # TorchRL may wrap callable policies in a TensorDictModule for rollout
        # and pass the root observation fields as positional args.
        first = args[0]
        if isinstance(first, TensorDictBase):
            obs = first.get("observation")
            if obs is None:
                raise KeyError("Expected 'observation' in the first positional TensorDict")
        elif torch.is_tensor(first):
            obs = first
        else:
            raise TypeError(f"Unsupported MATD3Policy input type: {type(first)}")

        obs_td = TensorDict({self.obs_name: obs}, batch_size=obs.shape[:-2])
        action = self._call_actor(obs_td)[self.act_name]
        if not deterministic and self.policy_noise > 0:
            action_noise = (
                torch.randn_like(action)
                .mul_(self.policy_noise)
                .clamp_(-self.noise_clip, self.noise_clip)
            )
            action = action.add(action_noise)
        return action.clamp(-1, 1)

    def _call_actor(self, tensordict: TensorDict, target: bool = False):
        obs = tensordict[self.obs_name]
        batch_size = tensordict.batch_size

        if self._shared_actor:
            actor_module = self.actor_target if target else self.actor
            actor_input = TensorDict({self.obs_name: obs}, batch_size=batch_size)
            action = actor_module(actor_input)[self.act_name]
            return TensorDict({self.act_name: action}, batch_size=batch_size)

        modules = self.actors_target if target else self.actors
        actions = []
        for i, actor_module in enumerate(modules):
            actor_input = TensorDict(
                {self.obs_name: obs[:, i:i+1, :]},
                batch_size=batch_size,
            )
            actions.append(actor_module(actor_input)[self.act_name])
        action = torch.cat(actions, dim=1)
        return TensorDict({self.act_name: action}, batch_size=batch_size)

    def train_op(self, data: TensorDict):
        self.replay_buffer.extend(data.to("cpu").reshape(-1))

        if len(self.replay_buffer) < self.cfg.buffer_size:
            print(f"{len(self.replay_buffer)} < {self.cfg.buffer_size}")
            return {}

        infos_critic = []
        infos_actor = []

        with tqdm(range(1, self.gradient_steps+1)) as t:
            for gradient_step in t:

                transition = self.replay_buffer.sample(self.batch_size).to(self.device)

                state = transition[self.state_name]
                actions_taken = transition[self.act_name]

                reward = transition[("next", *self.reward_name)]
                # done is [B, 1] (scalar per env). Broadcast to [B, n, 1] so it
                # matches the per-agent next-Q tensor produced by the critic.
                next_dones = transition[("next", "done")].float().unsqueeze(-1)
                if next_dones.dim() < reward.dim():
                    next_dones = next_dones.unsqueeze(-2).expand_as(reward)
                next_state = transition[("next", *self.state_name)]

                with torch.no_grad():
                    next_action: torch.Tensor = self._call_actor(
                        transition["next"], target=True
                    )[self.act_name]

                    if self.target_noise > 0:  # target smoothing
                        action_noise = (
                            next_action
                            .clone()
                            .normal_(0, self.target_noise)
                            .clamp_(-self.noise_clip, self.noise_clip)
                        )
                        next_action = torch.clamp(next_action + action_noise, -1, 1)

                    next_qs = self.critic_target(next_state, next_action)
                    next_q = torch.min(next_qs, dim=-1, keepdim=True).values
                    target_q = (reward + self.cfg.gamma * (1 - next_dones) * next_q).detach().squeeze(-1)
                    assert not torch.isinf(target_q).any()
                    assert not torch.isnan(target_q).any()

                qs = self.critic(state, actions_taken)
                critic_loss = sum(self.critic_loss_fn(q, target_q) for q in qs.unbind(-1))
                self.critic_opt.zero_grad()
                critic_loss.backward()
                critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_opt.step()
                infos_critic.append(TensorDict({
                    "critic_loss": critic_loss,
                    "critic_grad_norm": critic_grad_norm,
                    "q_taken": qs.mean()
                }, []))

                if (gradient_step + 1) % self.cfg.actor_delay == 0:

                    with hold_out_net(self.critic):

                        actor_output = self._call_actor(transition)
                        actions_new = actor_output[self.act_name]

                        actor_losses = []
                        for a in self.train_agent_indices:
                            actions = actions_taken.clone()
                            actions[..., a, :] = actions_new[..., a, :]
                            qs = self.critic(state, actions)
                            q = torch.min(qs, dim=-1, keepdim=True).values
                            actor_losses.append(- q.mean())

                        actor_loss = torch.stack(actor_losses).sum()
                        self.actor_opt.zero_grad()
                        actor_loss.backward()
                        actor_grad_norm = nn.utils.clip_grad_norm_(
                            self.actor_opt.param_groups[0]["params"], self.cfg.max_grad_norm
                        )
                        self.actor_opt.step()

                        infos_actor.append(TensorDict({
                            "actor_loss": actor_loss,
                            "actor_grad_norm": actor_grad_norm,
                        }, []))

                    with torch.no_grad():
                        if self._shared_actor:
                            soft_update(self.actor_target, self.actor, self.cfg.tau)
                        else:
                            for target_actor, actor in zip(self.actors_target, self.actors):
                                soft_update(target_actor, actor, self.cfg.tau)
                        soft_update(self.critic_target, self.critic, self.cfg.tau)

                t.set_postfix({"critic_loss": critic_loss.item()})

        infos = {**torch.stack(infos_actor), **torch.stack(infos_critic)}
        infos = {k: torch.mean(v).item() for k, v in infos.items()}
        return infos


class Critic(nn.Module):
    def __init__(self,
                 cfg,
                 num_agents: int,
                 state_spec: TensorSpec,
                 action_spec: Bounded,
                 num_critics: int = 2,
                 ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_agents = num_agents
        self.act_space = action_spec
        self.state_spec = state_spec
        self.num_critics = num_critics

        self.critics = nn.ModuleList([
            self._make_critic() for _ in range(self.num_critics)
        ])

    def _make_critic(self):
        if isinstance(self.state_spec, (Bounded, Unbounded)):
            action_dim = self.act_space.shape[-1]
            state_dim = self.state_spec.shape[-1]
            num_units = [
                action_dim * self.num_agents + state_dim,
                *self.cfg["hidden_units"]
            ]
            base = MLP(num_units)
        elif isinstance(self.state_spec, Composite):
            encoder_cls = ENCODERS_MAP[self.cfg.attn_encoder]
            base = encoder_cls(Composite(self.state_spec))
        else:
            raise NotImplementedError

        v_out = nn.Linear(base.output_shape.numel(), self.num_agents)
        return nn.Sequential(base, v_out)

    def forward(self, state: torch.Tensor, actions: torch.Tensor):
        """
        Args:
            state: (batch_size, state_dim)
            actions: (batch_size, num_agents, action_dim)
        """
        # Some envs broadcast the same centralized state for each agent
        # (shape [B, num_agents, state_dim]). The critic MLP is built from
        # state_dim, so collapse that duplicated axis before flattening.
        if state.ndim >= 3 and state.shape[-2] == self.num_agents:
            state = state[..., 0, :]
        state = state.flatten(1)
        actions = actions.flatten(1)
        x = torch.cat([state, actions], dim=-1)
        return torch.stack([critic(x) for critic in self.critics], dim=-1)


def soft_update_params(target: TensorDict, source: TensorDict, tau: float):
    ...
