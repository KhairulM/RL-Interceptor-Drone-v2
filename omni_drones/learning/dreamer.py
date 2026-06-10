# MIT License
#
# Copyright (c) 2026 Botian Xu, Tsinghua University
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
"""DreamerV3 implementation for single-agent continuous-control tasks.

This is a from-scratch port of the DreamerV3 algorithm (Hafner et al., 2023)
that follows the conventions used by the other algorithms in
``omni_drones.learning``: a single ``Policy`` class with the signature
``(cfg, observation_spec, action_spec, reward_spec, device)``, a ``__call__``
that consumes/returns a ``TensorDict`` rollout step, and a ``train_op`` that
extends a sequence replay buffer and runs ``cfg.gradient_steps`` updates.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from tensordict import TensorDict
from torchrl.data import Composite, TensorSpec
from torchrl.envs.utils import ExplorationType, exploration_type

from .common import MyBuffer, soft_update
from .modules.networks import MLP


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


class TwoHotSymlog:
    """Twohot distribution over symlog-spaced bins for reward / value heads."""

    def __init__(self, logits: torch.Tensor, low: float = -20., high: float = 20.):
        self.logits = logits
        self.num_bins = logits.shape[-1]
        self.bins = torch.linspace(low, high, self.num_bins, device=logits.device)
        self.log_probs = F.log_softmax(logits, dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        probs = self.log_probs.exp()
        expected = (probs * self.bins).sum(-1, keepdim=True)
        return symexp(expected)

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        # target has a trailing singleton dim; convert to symlog-space scalar.
        x = symlog(target).squeeze(-1).clamp(self.bins[0], self.bins[-1])
        below = (self.bins.unsqueeze(0) <= x.unsqueeze(-1)).sum(-1) - 1
        below = below.clamp(0, self.num_bins - 2)
        above = below + 1
        bin_low = self.bins[below]
        bin_high = self.bins[above]
        weight_high = ((x - bin_low) / (bin_high - bin_low + 1e-8)).clamp(0., 1.)
        weight_low = 1. - weight_high
        twohot = torch.zeros_like(self.logits)
        twohot.scatter_add_(-1, below.unsqueeze(-1), weight_low.unsqueeze(-1))
        twohot.scatter_add_(-1, above.unsqueeze(-1), weight_high.unsqueeze(-1))
        return (twohot * self.log_probs).sum(-1)


def categorical_kl(post_logits: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
    """KL(post || prior) per stoch dim. Inputs [..., stoch, disc] -> [..., stoch].

    Returning the KL per latent (instead of summing over the stoch dim) lets
    DreamerV3's free-bits clamp be applied per latent before summation, which
    is the convention used in the paper / reference impl.
    """
    post_logp = F.log_softmax(post_logits, dim=-1)
    prior_logp = F.log_softmax(prior_logits, dim=-1)
    post = post_logp.exp()
    return (post * (post_logp - prior_logp)).sum(-1)


class EMAReturnNorm:
    """EMA of the 5th..95th percentile spread of returns (DreamerV3 §4.4)."""

    def __init__(self, decay: float = 0.99, eps: float = 1.):
        self.decay = decay
        self.eps = eps
        self.scale = None  # type: torch.Tensor | None

    def update(self, returns: torch.Tensor) -> torch.Tensor:
        flat = returns.detach().flatten()
        if flat.numel() < 2:
            return torch.tensor(1., device=flat.device)
        low = torch.quantile(flat, 0.05)
        high = torch.quantile(flat, 0.95)
        spread = (high - low).clamp(min=1e-6)
        if self.scale is None:
            self.scale = spread.clone()
        else:
            self.scale.mul_(self.decay).add_(spread, alpha=1. - self.decay)
        return torch.maximum(self.scale, torch.tensor(self.eps, device=self.scale.device))


# ---------------------------------------------------------------------------
# World model (RSSM with discrete categorical latents)
# ---------------------------------------------------------------------------

class RSSM(nn.Module):

    def __init__(self, cfg, obs_dim: int, action_dim: int, num_bins: int = 255):
        super().__init__()
        wm = cfg.world_model
        self.deter_dim = int(wm.deter_dim)
        self.stoch = int(wm.stoch_dim)
        self.disc = int(wm.discrete_dim)
        self.stoch_flat = self.stoch * self.disc
        self.action_dim = action_dim
        self.num_bins = num_bins

        # Encoder: obs -> embed
        enc_units = [obs_dim] + list(cfg.encoder.hidden_units)
        norm = nn.LayerNorm if cfg.encoder.get("layer_norm", True) else None
        embed_dim = int(cfg.encoder.proj_units[-1])
        self.encoder = nn.Sequential(
            nn.LayerNorm(obs_dim),
            MLP(enc_units, norm),
            nn.Linear(enc_units[-1], embed_dim),
        )
        self.embed_dim = embed_dim

        # Decoder: (h, z) -> obs_recon
        dec_units = [self.deter_dim + self.stoch_flat] + list(cfg.decoder.hidden_units)
        self.decoder = nn.Sequential(
            MLP(dec_units, nn.LayerNorm),
            nn.Linear(dec_units[-1], obs_dim),
        )

        # Sequence model: (z, a) -> GRU input; GRU updates deter h.
        pre_units = [self.stoch_flat + action_dim] + list(wm.hidden_units)
        self.pre_gru = MLP(pre_units, nn.LayerNorm)
        self.gru = nn.GRUCell(pre_units[-1], self.deter_dim)

        # Prior: h -> stoch logits
        prior_units = [self.deter_dim] + list(wm.prior_hidden_units)
        self.prior_head = nn.Sequential(
            MLP(prior_units, nn.LayerNorm),
            nn.Linear(prior_units[-1], self.stoch_flat),
        )
        # Posterior: (h, embed) -> stoch logits
        post_units = [self.deter_dim + embed_dim] + list(wm.post_hidden_units)
        self.post_head = nn.Sequential(
            MLP(post_units, nn.LayerNorm),
            nn.Linear(post_units[-1], self.stoch_flat),
        )

        # Reward head (twohot symlog)
        rw_units = [self.deter_dim + self.stoch_flat] + list(cfg.reward_pred.hidden_units)
        self.reward_head = nn.Sequential(
            MLP(rw_units, nn.LayerNorm),
            nn.Linear(rw_units[-1], num_bins),
        )
        # Continue head (Bernoulli on 1 - terminated)
        ct_units = [self.deter_dim + self.stoch_flat] + list(cfg.discount_pred.hidden_units)
        self.cont_head = nn.Sequential(
            MLP(ct_units, nn.LayerNorm),
            nn.Linear(ct_units[-1], 1),
        )

    # ---- latent helpers -------------------------------------------------

    def initial(self, batch_shape, device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(*batch_shape, self.deter_dim, device=device)
        z = torch.zeros(*batch_shape, self.stoch_flat, device=device)
        return h, z

    def _unimix(self, logits: torch.Tensor, mix: float = 0.01) -> torch.Tensor:
        # logits: [..., stoch*disc] -> [..., stoch, disc] with uniform mixture
        logits = logits.unflatten(-1, (self.stoch, self.disc))
        probs = F.softmax(logits, dim=-1)
        probs = (1. - mix) * probs + mix / self.disc
        return torch.log(probs)

    def _sample_stoch(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [..., stoch, disc] (unimixed)
        dist = D.OneHotCategoricalStraightThrough(logits=logits)
        sample = dist.rsample()  # [..., stoch, disc] one-hot, ST gradient
        return sample.flatten(-2)

    # ---- single step ----------------------------------------------------

    def img_step(self, prev_h, prev_z, prev_a):
        x = self.pre_gru(torch.cat([prev_z, prev_a], dim=-1))
        h = self.gru(x, prev_h)
        prior_logits = self._unimix(self.prior_head(h))
        z = self._sample_stoch(prior_logits)
        return h, z, prior_logits

    def obs_step(self, prev_h, prev_z, prev_a, embed):
        x = self.pre_gru(torch.cat([prev_z, prev_a], dim=-1))
        h = self.gru(x, prev_h)
        prior_logits = self._unimix(self.prior_head(h))
        post_logits = self._unimix(self.post_head(torch.cat([h, embed], dim=-1)))
        z = self._sample_stoch(post_logits)
        return h, z, prior_logits, post_logits

    # ---- rollouts -------------------------------------------------------

    def observe(self, embeds, actions, is_first):
        """Run posterior rollout over a batch of trajectories.

        embeds:   [T, B, embed_dim]
        actions:  [T, B, action_dim]  — action that led INTO state t (a_{t-1})
        is_first: [T, B, 1]           — episode boundary at step t
        """
        T, B = embeds.shape[:2]
        h, z = self.initial((B,), embeds.device)
        prev_a = torch.zeros(B, self.action_dim, device=embeds.device)
        hs, zs, priors, posts = [], [], [], []
        for t in range(T):
            mask = (1. - is_first[t].float())
            h = h * mask
            z = z * mask
            a = prev_a * mask
            h, z, prior_logits, post_logits = self.obs_step(h, z, a, embeds[t])
            hs.append(h)
            zs.append(z)
            priors.append(prior_logits)
            posts.append(post_logits)
            prev_a = actions[t]
        return (
            torch.stack(hs), torch.stack(zs),
            torch.stack(priors), torch.stack(posts),
        )

    def imagine(self, h0, z0, actor, horizon: int):
        """Roll the actor under the dynamics prior for ``horizon`` steps."""
        hs, zs, actions, entropies = [h0], [z0], [], []
        h, z = h0, z0
        for _ in range(horizon):
            dist = actor.dist(h, z)
            a = dist.rsample()
            actions.append(a)
            entropies.append(dist.entropy())
            h, z, _ = self.img_step(h, z, a)
            hs.append(h)
            zs.append(z)
        return (
            torch.stack(hs),       # [H+1, B, ...]
            torch.stack(zs),
            torch.stack(actions),  # [H,   B, A]
            torch.stack(entropies),
        )


# ---------------------------------------------------------------------------
# Actor / Critic
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Continuous-action actor producing a tanh-squashed Normal."""

    def __init__(self, cfg, deter_dim: int, stoch_flat: int, action_dim: int):
        super().__init__()
        units = [deter_dim + stoch_flat] + list(cfg.hidden_units)
        self.trunk = MLP(units, nn.LayerNorm)
        self.loc = nn.Linear(units[-1], action_dim)
        self.scale = nn.Linear(units[-1], action_dim)
        self.action_dim = action_dim

    def _features(self, h, z):
        return self.trunk(torch.cat([h, z], dim=-1))

    def dist(self, h, z) -> D.Independent:
        f = self._features(h, z)
        # tanh-bounded mean, softplus-bounded std with floor & ceiling.
        loc = torch.tanh(self.loc(f))
        scale = 0.1 + 0.9 * torch.sigmoid(self.scale(f))
        return D.Independent(D.Normal(loc, scale), 1)

    def mode(self, h, z) -> torch.Tensor:
        return torch.tanh(self.loc(self._features(h, z)))


class Critic(nn.Module):
    """Value head emitting twohot-symlog logits."""

    def __init__(self, cfg, deter_dim: int, stoch_flat: int, num_bins: int = 255):
        super().__init__()
        units = [deter_dim + stoch_flat] + list(cfg.hidden_units)
        self.net = nn.Sequential(
            MLP(units, nn.LayerNorm),
            nn.Linear(units[-1], num_bins),
        )
        self.num_bins = num_bins

    def forward(self, h, z) -> torch.Tensor:
        return self.net(torch.cat([h, z], dim=-1))

    def value(self, h, z) -> torch.Tensor:
        return TwoHotSymlog(self.forward(h, z)).mean


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class DreamerPolicy:
    """DreamerV3 policy following the `omni_drones.learning` conventions.

    Assumptions:
    * Single-agent task: ``("agents", "observation")`` has shape
      ``[..., 1, obs_dim]`` and ``("agents", "action")`` has shape
      ``[..., 1, action_dim]``. The agent dim is squeezed internally.
    * Continuous actions (Normal actor). Discrete-action and pixel branches
      are intentionally out of scope.
    """

    obs_name = ("agents", "observation")
    act_name = ("agents", "action")
    reward_name = ("next", "agents", "reward")
    term_name = ("next", "terminated")

    def __init__(self, cfg, observation_spec: Composite, action_spec: Composite,
                 reward_spec: TensorSpec, device):
        self.cfg = cfg
        self.device = device

        self.n_agents, self.action_dim = action_spec[self.act_name].shape[-2:]
        obs_shape = observation_spec[self.obs_name].shape
        assert obs_shape[-2] == 1, (
            "DreamerPolicy currently supports single-agent tasks "
            f"(got {obs_shape[-2]} agents)."
        )
        self.obs_dim = obs_shape[-1]

        # --- modules
        self.world_model = RSSM(cfg, self.obs_dim, self.action_dim).to(device)
        self.actor = Actor(
            cfg.actor, self.world_model.deter_dim,
            self.world_model.stoch_flat, self.action_dim,
        ).to(device)
        self.critic = Critic(
            cfg.critic, self.world_model.deter_dim, self.world_model.stoch_flat,
        ).to(device)
        self.critic_slow = copy.deepcopy(self.critic).to(device).requires_grad_(False)

        # --- optimisers
        self.wm_opt = torch.optim.Adam(self.world_model.parameters(),
                                       lr=float(cfg.world_model.lr), eps=1e-8)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(),
                                          lr=float(cfg.actor.lr), eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(),
                                           lr=float(cfg.critic.lr), eps=1e-5)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.kl_free = float(cfg.world_model.kl_free)
        self.discount = float(cfg.discount)
        self.return_lambda = float(cfg.return_lambda)
        self.imag_horizon = int(cfg.imag_horizon)
        self.entropy_coef = float(cfg.actor_entropy_coeff)
        self.slow_tau = float(cfg.get("slow_critic_tau", 0.02))

        # --- replay
        self.buffer = MyBuffer(int(cfg.buffer_size), device=device)
        self.batch_size = int(cfg.batch_size)
        self.batch_length = int(cfg.batch_length)
        self.gradient_steps = int(cfg.gradient_steps)

        self.ret_norm = EMAReturnNorm()

        # --- recurrent state buffers for rollout (filled lazily on first call)
        self._h = None
        self._z = None
        self._prev_action = None

        if cfg.get("checkpoint_path", None):
            state = torch.load(cfg.checkpoint_path, map_location=device)
            self.load_state_dict(state, strict=False)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _ensure_state(self, batch_size: int):
        if self._h is None or self._h.shape[0] != batch_size:
            self._h, self._z = self.world_model.initial((batch_size,), self.device)
            self._prev_action = torch.zeros(batch_size, self.action_dim,
                                            device=self.device)

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        obs = tensordict[self.obs_name].squeeze(-2)        # [E, obs_dim]
        batch = obs.shape[0]
        self._ensure_state(batch)

        is_init = tensordict.get("is_init", None)
        if is_init is not None:
            mask = (1. - is_init.reshape(batch, 1).float())
            self._h = self._h * mask
            self._z = self._z * mask
            self._prev_action = self._prev_action * mask

        with torch.no_grad():
            embed = self.world_model.encoder(obs)
            h, z, _, _ = self.world_model.obs_step(
                self._h, self._z, self._prev_action, embed,
            )
            if exploration_type() == ExplorationType.MODE:
                action = self.actor.mode(h, z)
            else:
                action = self.actor.dist(h, z).sample()
            action = action.clamp(-1., 1.)

        self._h, self._z = h, z
        self._prev_action = action
        tensordict.set(self.act_name, action.unsqueeze(-2))
        return tensordict

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_op(self, data: TensorDict) -> dict:
        self.buffer.extend(data)
        if len(self.buffer) <= self.batch_length:
            return {}

        metrics: defaultdict = defaultdict(list)

        for _ in range(self.gradient_steps):
            batch = self.buffer.sample(self.batch_size, self.batch_length)
            info = self._update(batch)
            for k, v in info.items():
                metrics[k].append(v)

        return {k: torch.stack(v).mean().item() for k, v in metrics.items()}

    # ------------------------------------------------------------------

    def _update(self, batch: TensorDict) -> dict:
        # batch shape: [B, T, ...]. Move to time-major and drop the singleton
        # agent dim so we can vectorise inside the RSSM.
        obs = batch[self.obs_name].squeeze(-2).transpose(0, 1)        # [T, B, O]
        action = batch[self.act_name].squeeze(-2).transpose(0, 1)     # [T, B, A]
        reward = batch[self.reward_name].squeeze(-2).transpose(0, 1)  # [T, B, 1]
        terminated = batch[self.term_name].transpose(0, 1).float()    # [T, B, 1]
        is_first = batch.get("is_init", None)
        if is_first is None:
            is_first = torch.zeros_like(terminated)
        else:
            is_first = is_first.transpose(0, 1).reshape(*terminated.shape).float()
        cont = 1. - terminated

        T, B = obs.shape[:2]

        # ----- World model loss ---------------------------------------
        embeds = self.world_model.encoder(obs)
        hs, zs, prior_logits, post_logits = self.world_model.observe(
            embeds, action, is_first,
        )
        feats = torch.cat([hs, zs], dim=-1)
        recon = self.world_model.decoder(feats)
        recon_loss = F.mse_loss(recon, obs)

        reward_logits = self.world_model.reward_head(feats)
        reward_dist = TwoHotSymlog(reward_logits)
        reward_loss = -reward_dist.log_prob(reward).mean()

        cont_logits = self.world_model.cont_head(feats)
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont)

        # KL with dynamics/representation balancing (DreamerV3 §3.3).
        # Free bits are applied PER LATENT before summing across the stoch
        # dim, otherwise a global clamp at `kl_free` collapses all signal.
        kl_dyn_per = categorical_kl(post_logits.detach(), prior_logits)
        kl_rep_per = categorical_kl(post_logits, prior_logits.detach())
        free = torch.tensor(self.kl_free, device=self.device)
        kl_dyn = torch.maximum(kl_dyn_per, free).sum(-1).mean()
        kl_rep = torch.maximum(kl_rep_per, free).sum(-1).mean()
        kl_loss = 0.5 * kl_dyn + 0.1 * kl_rep

        wm_loss = recon_loss + reward_loss + cont_loss + kl_loss

        self.wm_opt.zero_grad(set_to_none=True)
        wm_loss.backward()
        wm_gn = nn.utils.clip_grad_norm_(self.world_model.parameters(),
                                         self.max_grad_norm)
        self.wm_opt.step()

        # ----- Imagination from every posterior state -----------------
        # Detach posterior latents so actor/critic gradients do not flow
        # back into the world model.
        start_h = hs.detach().reshape(T * B, -1)
        start_z = zs.detach().reshape(T * B, -1)

        img_hs, img_zs, img_actions, img_entropies = self.world_model.imagine(
            start_h, start_z, self.actor, self.imag_horizon,
        )  # img_hs: [H+1, T*B, ...]; img_actions: [H, T*B, A]
        img_feats = torch.cat([img_hs, img_zs], dim=-1)

        # Predict rewards / continues / values along the imagined trajectory.
        # The reward and continue heads remain DIFFERENTIABLE w.r.t. the
        # imagined latents (which were produced by the actor via reparameter-
        # ised sampling). The slow critic provides the value bootstrap and
        # is detached. World-model parameters do not move here because their
        # optimiser was already stepped above; we still freeze them via the
        # reward/cont heads' gradient flow by zero-ing wm grads after the
        # actor / critic backward passes (Adam state already advanced, so
        # the only thing to avoid is mutating the parameters again).
        reward_logits_img = self.world_model.reward_head(img_feats)
        img_rewards = TwoHotSymlog(reward_logits_img).mean.squeeze(-1)   # [H+1, T*B]
        cont_logits_img = self.world_model.cont_head(img_feats)
        img_conts = torch.sigmoid(cont_logits_img).squeeze(-1)            # [H+1, T*B]
        with torch.no_grad():
            img_values = TwoHotSymlog(
                self.critic_slow(img_hs, img_zs)
            ).mean.squeeze(-1)                                            # [H+1, T*B]

        # Discount factors: gamma * predicted continue. Detached because they
        # only define the temporal weighting of the actor objective.
        discount = (self.discount * img_conts).detach()                  # [H+1, T*B]
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], dim=0),
            dim=0,
        )

        # ---- λ-returns. Differentiable in img_rewards (and therefore in
        # the actor through the imagined trajectory). Values are detached.
        returns = []
        last = img_values[-1]
        for t in reversed(range(self.imag_horizon)):
            r = img_rewards[t + 1]
            g = discount[t + 1]
            v_next = img_values[t + 1]
            last = r + g * ((1. - self.return_lambda) * v_next
                            + self.return_lambda * last)
            returns.insert(0, last)
        returns = torch.stack(returns)                  # [H, T*B]

        # ----- Actor update (dynamics gradient) -----------------------
        scale = self.ret_norm.update(returns)
        adv = returns / scale
        actor_weights = weights[:-1]
        actor_loss = -(actor_weights * (adv + self.entropy_coef * img_entropies)).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        # Clear any wm grads accumulated by the actor backward through the
        # reward / cont heads. The wm optimiser has already stepped this
        # iteration, so we must not step it again.
        self.wm_opt.zero_grad(set_to_none=True)
        actor_loss.backward(retain_graph=False)
        actor_gn = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                            self.max_grad_norm)
        self.actor_opt.step()
        # Drop wm grads picked up by the actor backward.
        self.wm_opt.zero_grad(set_to_none=True)

        # ----- Critic update ------------------------------------------
        critic_feats_h = img_hs[:-1].detach()
        critic_feats_z = img_zs[:-1].detach()
        critic_logits = self.critic(critic_feats_h, critic_feats_z)
        critic_dist = TwoHotSymlog(critic_logits)
        critic_loss = -critic_dist.log_prob(
            returns.detach().unsqueeze(-1)
        ).mean()

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gn = nn.utils.clip_grad_norm_(self.critic.parameters(),
                                             self.max_grad_norm)
        self.critic_opt.step()

        soft_update(self.critic_slow, self.critic, self.slow_tau)

        return {
            "wm_loss":      wm_loss.detach(),
            "recon_loss":   recon_loss.detach(),
            "reward_loss":  reward_loss.detach(),
            "cont_loss":    cont_loss.detach(),
            "kl_loss":      kl_loss.detach(),
            "wm_grad_norm": wm_gn.detach(),
            "actor_loss":   actor_loss.detach(),
            "actor_entropy": img_entropies.mean().detach(),
            "actor_grad_norm": actor_gn.detach(),
            "critic_loss":  critic_loss.detach(),
            "critic_grad_norm": critic_gn.detach(),
            "return_mean":  returns.mean().detach(),
            "return_scale": scale.detach(),
            "imag_reward_mean": img_rewards.mean().detach(),
        }

    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_slow": self.critic_slow.state_dict(),
        }

    def load_state_dict(self, state: dict, strict: bool = True):
        self.world_model.load_state_dict(state["world_model"], strict=strict)
        self.actor.load_state_dict(state["actor"], strict=strict)
        self.critic.load_state_dict(state["critic"], strict=strict)
        if "critic_slow" in state:
            self.critic_slow.load_state_dict(state["critic_slow"], strict=strict)

    def train(self):
        self.world_model.train()
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.world_model.eval()
        self.actor.eval()
        self.critic.eval()
