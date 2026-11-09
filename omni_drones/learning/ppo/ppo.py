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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchrl.data import Composite, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union
import einops

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal
from .common import GAE


@dataclass
class PPOConfig:
    name: str = "ppo"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16

    clip_param: float = 0.1
    learning_rate: float = 5e-4
    gamma: float = 0.99

    # whether to use privileged information
    priv_actor: bool = False
    priv_critic: bool = False

    checkpoint_path: Union[str, None] = None

    # Thesis F8: stop the GAE recursion at truncations too (bootstrap still only cut by `terminated`). Default off:
    # earlier runs cut the recursion only at terminations.
    gae_cut_at_truncation: bool = False


cs = ConfigStore.instance()
cs.store("ppo", node=PPOConfig, group="algo")
cs.store("ppo_priv", node=PPOConfig(priv_actor=True, priv_critic=True), group="algo")
cs.store("ppo_priv_critic", node=PPOConfig(priv_critic=True), group="algo")


def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale


class PPOPolicy(TensorDictModuleBase):

    def __init__(
            self,
            cfg: PPOConfig,
            observation_spec: Composite,
            action_spec: Composite,
            reward_spec: TensorSpec,
            device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.entropy_coef = 0.001
        self.clip_param = 0.1
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.n_agents, self.action_dim = action_spec[("agents", "action")].shape[-2:]
        self.gae = GAE(0.99, 0.95)

        fake_input = observation_spec.zero()

        if self.cfg.priv_actor:
            intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]
            actor_module = TensorDictSequential(
                TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["feature"]),
                TensorDictModule(
                    nn.Sequential(nn.LayerNorm(intrinsics_dim), make_mlp([64, 64])),
                    [("agents", "intrinsics")], ["context"]
                ),
                CatTensors(["feature", "context"], "feature"),
                TensorDictModule(
                    nn.Sequential(make_mlp([256, 256]), Actor(self.action_dim)),
                    ["feature"], ["loc", "scale"]
                )
            )
        else:
            actor_module = TensorDictModule(
                nn.Sequential(make_mlp([256, 256, 256]), Actor(self.action_dim)),
                [("agents", "observation")], ["loc", "scale"]
            )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True,
            log_prob_key="sample_log_prob"
        ).to(self.device)

        if self.cfg.priv_critic:
            intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]
            self.critic = TensorDictSequential(
                TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["feature"]),
                TensorDictModule(
                    nn.Sequential(nn.LayerNorm(intrinsics_dim), make_mlp([64, 64])),
                    [("agents", "intrinsics")], ["context"]
                ),
                CatTensors(["feature", "context"], "feature"),
                TensorDictModule(
                    nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(1)),
                    ["feature"], ["state_value"]
                )
            ).to(self.device)
        else:
            self.critic = TensorDictModule(
                nn.Sequential(make_mlp([256, 256, 256]), nn.LazyLinear(1)),
                [("agents", "observation")], ["state_value"]
            ).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)

        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)

            self.actor.apply(init_)
            self.critic.apply(init_)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
        self.value_norm = ValueNorm1(reward_spec[("agents", "reward")].shape[-2:]).to(self.device)

    def __call__(self, tensordict: TensorDict):
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude("loc", "scale", "feature", inplace=True)
        return tensordict

    @torch.no_grad()
    def act(self, tensordict: TensorDict) -> torch.Tensor:
        self.actor(tensordict)
        return tensordict[("agents", "action")]

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict[("next", "agents", "reward")]
        div_reward = None
        if getattr(self, "div_refs", None):
            div_reward = self._intrinsic_reward(tensordict)
            rewards = rewards + div_reward
        dones = einops.repeat(
            tensordict[("next", "terminated")],
            "t e 1 -> t e a 1",
            a=self.n_agents
        )
        values = tensordict["state_value"]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        ends = None
        if getattr(self.cfg, "gae_cut_at_truncation", False):
            ends = einops.repeat(tensordict[("next", "done")], "t e 1 -> t e a 1", a=self.n_agents)
        adv, ret = self.gae(rewards, dones, values, next_values, ends)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))

        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        out = {k: v.item() for k, v in infos.items()}
        if div_reward is not None:
            out["div_reward"] = div_reward.mean().item()
        return out

    def _update(self, tensordict: TensorDict):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * torch.mean(entropy)

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        loss = policy_loss + entropy_loss + value_loss
        div_info = {}
        if getattr(self, "div_refs", None):
            div = self._diversity_distance(tensordict, dist.mean)
            bonus = self._div_bonus(div)
            div_loss = - self.div_beta * bonus.mean()
            loss = loss + div_loss
            div_info = {"div_loss": div_loss.detach(), "div_distance": div.detach().mean()}
            if self.div_mode == "action":
                div_info.update(div_bonus=bonus.detach().mean(), div_sat=(bonus.detach() > 0.9).float().mean())
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        self.actor_opt.step()
        self.critic_opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var,
            **div_info,
        }, [])

    def set_diversity(self, ckpt_paths, beta: float, cap: float, mode: str = "kl", tau: float = 1.0,
                      centre: bool = False):
        """Thesis F6: PSD-PSRO's policy-space diversity regulariser (Yao et al. 2023, Eq. 14 / 17), adapted to
        continuous actions. Both parts of its gradient: -beta * mean_s min(min_j d_j(s), cap) in the loss (direct KL
        gradient) and the same quantity per step as an intrinsic reward (_intrinsic_reward). d_j(s) = 0.5 * sum_k
        ((mu(s) - mu_j(s)) / sigma_j)_k^2 is the KL between the learner's and frozen policy j's action Gaussians with
        the learner's std replaced by sigma_j (mean-only KL: the learner cannot raise it by changing its own std).
        The j are earlier same-role policies loaded from ``ckpt_paths`` (min over them, not over their convex hull).

        Thesis F6b (mode="action", F6 saturated and acted as a survival bonus): d_j(s) = RMS over action dims of
        mu(s) - mu_j(s) (action units, no division by sigma_j), bonus = 1 - exp(-d / tau) instead of min(d, cap) (no
        cap); centre=True subtracts the batch mean from the intrinsic reward, so it is zero-mean (no per-step bonus
        for staying alive). beta = 0 measures d without changing training."""
        import copy
        assert mode in ("kl", "action"), mode
        self.div_beta, self.div_cap = float(beta), float(cap)
        self.div_mode, self.div_tau, self.div_centre = mode, float(tau), bool(centre)
        self.div_refs = []
        for path in ckpt_paths:
            state = torch.load(path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            ref = copy.deepcopy(self.actor)
            ref.load_state_dict({k[len("actor."):]: v for k, v in state.items() if k.startswith("actor.")})
            ref.requires_grad_(False)
            self.div_refs.append(ref)

    @torch.no_grad()
    def _intrinsic_reward(self, tensordict: TensorDict) -> torch.Tensor:
        """PSD-PSRO's intrinsic reward (Yao et al. 2023, Eq. 17 / Alg. 1): beta * min(min_j d_j(s_t), cap) per rollout
        step, added to the task reward before GAE, so the policy also seeks states where it differs. Shape of the
        reward ([..., agents, 1])."""
        loc = self.actor.get_dist(tensordict.select(*self.actor.in_keys).clone(False)).mean
        r = self.div_beta * self._div_bonus(self._diversity_distance(tensordict, loc))
        if self.div_centre:
            r = r - r.mean()
        return r.unsqueeze(-1)

    def _div_bonus(self, d: torch.Tensor) -> torch.Tensor:
        """F6: min(d, cap); F6b (mode="action"): 1 - exp(-d / tau)."""
        if self.div_mode == "action":
            return 1.0 - torch.exp(-d / self.div_tau)
        return d.clamp(max=self.div_cap)

    def _diversity_distance(self, tensordict: TensorDict, loc: torch.Tensor) -> torch.Tensor:
        """min_j d_j(s) per sample and agent (see set_diversity)."""
        d = []
        for ref in self.div_refs:
            with torch.no_grad():
                ref_dist = ref.get_dist(tensordict.select(*ref.in_keys).clone(False))
            if self.div_mode == "action":
                d.append((((loc - ref_dist.mean) ** 2).mean(-1) + 1e-8).sqrt())
            else:
                d.append((0.5 * ((loc - ref_dist.mean) / ref_dist.base_dist.scale) ** 2).sum(-1))
        return torch.stack(d).min(0).values


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]
