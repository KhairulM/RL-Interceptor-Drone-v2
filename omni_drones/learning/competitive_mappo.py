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

"""Competitive MAPPO with per-agent independent actors + critics.

Each agent (pursuer, evader) maintains its own actor network and its own
critic/value network — no shared value table, because adversarial objectives
produce conflicting gradients under a common critic.

Architecture mirrors the proven TorchRL pattern in ``mappo_new.py``:
    TensorDictModule(MLP + Actor → loc, scale)
      → ProbabilisticActor(loc, scale → action, log_prob)

Expected spec keys (set by InterceptCompetitiveMA):
    observation_spec["pursuer"]["observation"]  -> [N, 1, obs_dim]
    observation_spec["evader"]["observation"]   -> [N, 1, obs_dim]
    action_spec["pursuer"]["action"]            -> [N, 1, act_dim]
    action_spec["evader"]["action"]             -> [N, 1, act_dim]
    reward_spec["pursuer"]["reward"]            -> [N, 1, 1]
    reward_spec["evader"]["reward"]             -> [N, 1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, make_functional, TensorDictParams
from torch.func import vmap
from torchrl.modules import ProbabilisticActor
from torchrl.data import Composite

from omni_drones.learning.ppo.common import GAE, make_mlp
from omni_drones.learning.mappo_new import Actor as PPOActor
from omni_drones.learning.modules.distributions import IndependentNormal


def init_(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, 0.01)
        nn.init.constant_(module.bias, 0.)


class CompetitiveMAPPO:
    """Two-agent competitive MAPPO with independent actors + critics.

    Agent names are keyed as ``"pursuer"`` and ``"evader"``.  Each has its own
    actor (policy) and critic (value).  No parameters are shared between them.
    """

    AGENT_NAMES = ["pursuer", "evader"]

    def __init__(self, cfg, observation_spec: Composite,
                 action_spec: Composite, reward_spec: Composite, device):
        self.cfg = cfg
        self.device = device

        # ---- PPO knobs -----------------------------------------------------
        self.clip_param = float(cfg.get("clip_param", 0.1))
        self.ppo_epochs = int(cfg.get("ppo_epochs", 4))
        self.num_minibatches = int(cfg.get("num_minibatches", 8))
        self.entropy_coef = float(cfg.get("entropy_coef", 0.001))

        # ---- GAE knobs -----------------------------------------------------
        self.gae_gamma = float(cfg.get("gamma", 0.99))
        self.gae_lambda = float(cfg.get("gae_lambda", 0.95))
        self.normalize_advantages = bool(cfg.get("normalize_advantages", True))

        # ---- GAE module ----------------------------------------------------
        self.gae = GAE(self.gae_gamma, self.gae_lambda).to(self.device)

        # ---- Build per-agent networks --------------------------------------
        self.actors = {}       # ProbabilisticActor (samples action + log_prob)
        self.critics = {}      # TensorDictModule (state_value)
        self.actor_opts = {}
        self.critic_opts = {}

        actor_lr = float(cfg.actor.get("lr", 3e-4))
        critic_lr = float(cfg.critic.get("lr", 3e-4))
        actor_hidden = list(cfg.actor.get("hidden_units", [256, 256]))
        critic_hidden = list(cfg.critic.get("hidden_units", [256, 256]))

        for agent_name in self.AGENT_NAMES:
            obs_key = (agent_name, "observation")
            act_key = (agent_name, "action")

            act_dim = action_spec[act_key].shape[-1]

            # --- Actor: TensorDictModule(MLP → loc/scale) → ProbabilisticActor
            actor_module = TensorDictModule(
                nn.Sequential(
                    make_mlp(actor_hidden, nn.Mish),
                    PPOActor(act_dim),
                ),
                in_keys=[obs_key],
                out_keys=["loc", "scale"],   # generic keys for IndependentNormal kwarg compat
            ).to(self.device)

            # Warm up LazyLinear weights with a dummy forward.
            fake_obs = torch.zeros(2, 15, device=self.device)
            actor_module(TensorDict({obs_key: fake_obs}, [2]))
            init_(actor_module.module)

            self.actors[agent_name] = ProbabilisticActor(
                module=actor_module,
                in_keys=["loc", "scale"],     # must match out_keys above
                out_keys=[act_key],
                distribution_class=IndependentNormal,
                return_log_prob=True,
                log_prob_key=f"sample_log_prob_{agent_name}",
            ).to(self.device)

            # --- Critic: obs → MLP → scalar value (per-agent, no sharing)
            critic = TensorDictModule(
                nn.Sequential(
                    make_mlp(critic_hidden, nn.Mish),
                    nn.LazyLinear(1),
                ),
                in_keys=[obs_key],
                out_keys=[f"{agent_name}_state_value"],
            ).to(self.device)
            critic(TensorDict({obs_key: fake_obs}, [2]))
            init_(critic.module)

            self.critics[agent_name] = critic
            self.actor_opts[agent_name] = torch.optim.Adam(
                self.actors[agent_name].parameters(), lr=actor_lr)
            self.critic_opts[agent_name] = torch.optim.Adam(
                critic.parameters(), lr=critic_lr)

    # ------------------------------------------------------------------
    # Forward (rollout / eval)
    # ------------------------------------------------------------------
    def __call__(self, tensordict: TensorDict):
        for agent_name in self.AGENT_NAMES:
            self.actors[agent_name](tensordict)
            self.critics[agent_name](tensordict)
        return tensordict

    # ------------------------------------------------------------------
    # Training step — independent GAE + PPO per agent
    # ------------------------------------------------------------------
    def train_op(self, tensordict: TensorDict):
        next_td = tensordict["next"]
        infos = {}

        for agent_name in self.AGENT_NAMES:
            reward_key = (agent_name, "reward")

            # Value at final frame (no grad).
            with torch.no_grad():
                next_val = self.critics[agent_name](next_td)

            rewards = _ensure_NT(tensordict[("next", *reward_key)])
            dones = _ensure_NT(tensordict[("next", "terminated")])
            values = _ensure_NT(tensordict[f"{agent_name}_state_value"])
            next_values = _ensure_NT(next_val[f"{agent_name}_state_value"])

            if rewards.shape[-1] != 1:
                rewards = rewards.sum(-1, keepdim=True)

            # Broadcast done mask so shape is [N, T, 1].
            while dones.ndim < rewards.ndim:
                dones = dones.unsqueeze(-1)

            adv, ret = self.gae(rewards, dones.float(), values, next_values)

            adv_mean = adv.mean()
            adv_std = adv.std()
            if self.normalize_advantages and adv_std > 1e-7:
                adv = (adv - adv_mean) / adv_std

            tensordict.set(f"{agent_name}_adv", adv)
            tensordict.set(f"{agent_name}_ret", ret)

            # ---- PPO epochs ------------------------------------------------
            epoch_infos = []
            for _ in range(self.ppo_epochs):
                for mb in self._make_minibatches(tensordict, agent_name):
                    epoch_infos.append(self._update_agent(mb, agent_name))

            # Aggregate info across epochs: stack values per key
            all_keys = list(epoch_infos[0].keys())
            aggregated = {k: torch.stack([epoch[k] for epoch in epoch_infos])
                          for k in all_keys}
            agent_info = {f"{agent_name}/{k}": v.mean().item() for k, v in aggregated.items()}
            agent_info[f"{agent_name}/adv_mean"] = adv_mean.item()
            agent_info[f"{agent_name}/adv_std"] = adv_std.item()
            infos.update(agent_info)

        return infos

    # ------------------------------------------------------------------
    # Per-agent PPO update
    # ------------------------------------------------------------------
    def _update_agent(self, minibatch: TensorDict, agent_name: str):
        act_key = (agent_name, "action")
        log_prob_key = f"sample_log_prob_{agent_name}"

        # Current log-prob from actor distribution.
        dist = self.actors[agent_name].get_dist(minibatch)
        log_probs = dist.log_prob(minibatch[act_key])
        entropy = dist.entropy().mean()

        old_log_probs = minibatch[log_prob_key]
        adv = minibatch[f"{agent_name}_adv"].unsqueeze(-1)

        ratio = torch.exp(log_probs - old_log_probs).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        entropy_loss = -self.entropy_coef * entropy

        # --- Critic --------------------------------------------------------
        b_values = minibatch[f"{agent_name}_state_value"]
        b_returns = minibatch[f"{agent_name}_ret"]
        new_values = self.critics[agent_name](minibatch)[f"{agent_name}_state_value"]

        vloss_clip = F.smooth_l1_loss(
            b_returns, b_values + (new_values - b_values).clamp(
                -self.clip_param, self.clip_param))
        vloss_orig = F.smooth_l1_loss(b_returns, new_values)
        value_loss = torch.max(vloss_orig, vloss_clip)

        total_loss = policy_loss + entropy_loss + value_loss

        self.actor_opts[agent_name].zero_grad()
        self.critic_opts[agent_name].zero_grad()
        total_loss.backward()

        actor_gnorm = nn.utils.clip_grad_norm_(
            self.actors[agent_name].parameters(), 5.0)
        critic_gnorm = nn.utils.clip_grad_norm_(
            self.critics[agent_name].parameters(), 5.0)
        self.actor_opts[agent_name].step()
        self.critic_opts[agent_name].step()

        explained_var = 1.0 - F.mse_loss(new_values, b_returns) / \
            (b_returns.var() + 1e-8)

        return {
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach(),
            "actor_grad_norm": torch.tensor(float(actor_gnorm)),
            "critic_grad_norm": torch.tensor(float(critic_gnorm)),
            "explained_var": explained_var.detach(),
        }

    # ------------------------------------------------------------------
    # Minibatch generation
    # ------------------------------------------------------------------
    def _make_minibatches(self, tensordict: TensorDict, agent_name: str):
        keys = [
            (agent_name, "observation"), (agent_name, "action"),
            f"sample_log_prob_{agent_name}",
            f"{agent_name}_adv", f"{agent_name}_ret",
            f"{agent_name}_state_value",
        ]
        mb = tensordict.select(*keys, strict=False)
        flat = mb.reshape(-1)
        total = flat.shape[0]
        n_mb = min(self.num_minibatches, max(1, total))
        perm = torch.randperm((total // n_mb) * n_mb, device=self.device)
        for indices in perm.reshape(n_mb, -1):
            yield flat[indices]

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def state_dict(self):
        sd = {}
        for k in self.AGENT_NAMES:
            sd[f"actor_{k}"] = self.actors[k].state_dict()
            sd[f"critic_{k}"] = self.critics[k].state_dict()
        return sd

    def load_state_dict(self, sd):
        for k in self.AGENT_NAMES:
            self.actors[k].load_state_dict(sd[f"actor_{k}"])
            self.critics[k].load_state_dict(sd[f"critic_{k}"])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _ensure_NT(x: torch.Tensor) -> torch.Tensor:
    """Ensure shape is [N_envs, T_steps, …] from [T, N, …]."""
    if x.ndim < 2:
        return x
    # Heuristic: the collector returns [T, N, …] where T = train_every.
    # Transpose to [N, T, …] for GAE which expects env-first layout.
    train_every = int(getattr(x, "_train_every", 0))
    if train_every > 0 and x.shape[0] == train_every:
        return x.transpose(0, 1).contiguous()
    return x
