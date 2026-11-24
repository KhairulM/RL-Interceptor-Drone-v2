"""Thesis F16: PSD-PSRO's own-policy-hull distance for PPO (Yao et al. 2023, Eq. 14-17; docs/research_plan.md F16 C.4-6).

The new policy pi maximises  u(pi, sigma_opp) + lam * min_{q in Hull(old policies)} KL(P_pi || P_q)  where P_. is the
distribution of the learner's own action sequence (environment and opponent terms cancel). A hull member q with
mixture weights alpha is realised behaviourally as a sequence-form mixture: at step t its conditional is
    q(a | h_t) = sum_k w_k(h_t) pi_k(a | s_t),   log w_k(h_t) = log alpha_k + sum_{t' < t} log pi_k(a_t' | s_t') - lse,
reset at every episode start, so sum_t log q(a_t | h_t) = log sum_k alpha_k prod_t pi_k(a_t | s_t) (chain rule). The
trajectory KL then equals E_pi[sum_t KL(pi(.|s_t) || q(.|h_t))], and its gradient has two parts, both used here:
  * score-function part: the per-step KL is added to the reward before GAE (intrinsic reward lam * KL_t);
  * direct part: lam * mean_t KL_t(theta) in the loss, differentiated pathwise (reparameterised Monte Carlo).
The hull minimum is approximated over a fixed candidate set (all pure vertices + Dirichlet(1) interior mixtures from a
fixed seed); each PPO iteration picks the candidate with the smallest batch-averaged sampled log-ratio (the
trajectory-averaged distance). KL is computed between the raw (pre-tanh) Gaussian action distributions: the
environment applies the same bijection (tanh; the thrust clamp to [0, 1] of (tanh + 1) / 2 is inactive for the
Crazyflie limits 0 / 1) to every policy, and KL is invariant under a shared bijection.

The pure functions take generic per-component log-probabilities so tests can check them on enumerable discrete trees.
"""

import math
from typing import List, Optional

import numpy as np
import torch


# ------------------------------------------------------------------ pure functions (tested on discrete trees)
def history_logl(logp_k: torch.Tensor, is_init: torch.Tensor, carry: torch.Tensor):
    """Per-step cumulative own-action log-likelihoods of every component BEFORE step t.

    logp_k [E, T, K]: log pi_k(a_t | s_t) of the action actually taken; is_init [E, T] bool: step t starts an episode;
    carry [E, K]: the running sums at the end of the previous batch. Returns (hist [E, T, K], new carry [E, K])."""
    E, T, K = logp_k.shape
    hist = torch.empty_like(logp_k)
    L = carry.clone()
    for t in range(T):
        L = torch.where(is_init[:, t].unsqueeze(-1), torch.zeros_like(L), L)
        hist[:, t] = L
        L = L + logp_k[:, t]
    return hist, L


def mixture_logw(log_alpha: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
    """Normalised log posterior weights log w_k(h_t) = log alpha_k + hist_k - logsumexp (log_alpha may hold -inf)."""
    z = log_alpha + hist
    return z - torch.logsumexp(z, dim=-1, keepdim=True)


def mixture_logprob(logw: torch.Tensor, comp_logp: torch.Tensor) -> torch.Tensor:
    """log q(a | h) = logsumexp_k(log w_k + log pi_k(a | s)); comp_logp broadcasts against logw over [..., K]."""
    return torch.logsumexp(logw + comp_logp, dim=-1)


def gauss_logp(a: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Diagonal Gaussian log density summed over the last (action) dim."""
    return (-0.5 * ((a - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)).sum(-1)


def mc_kl(loc, scale, logw, mu_k, sigma_k, eps):
    """Monte Carlo KL(N(loc, scale) || sum_k w_k N(mu_k, sigma_k)) per state, reparameterised (differentiable in
    loc / scale). loc, scale [B, A]; logw [B, K]; mu_k [B, K, A]; sigma_k [K, A]; eps [M, B, A] ~ N(0, 1)."""
    a = loc.unsqueeze(0) + scale.unsqueeze(0) * eps                          # [M, B, A]
    logp = gauss_logp(a, loc.unsqueeze(0), scale.unsqueeze(0))               # [M, B]
    comp = gauss_logp(a.unsqueeze(-2), mu_k.unsqueeze(0), sigma_k)           # [M, B, K]
    return (logp - mixture_logprob(logw.unsqueeze(0), comp)).mean(0)         # [B]


def candidate_alphas(K: int, n_interior: int, seed: int) -> np.ndarray:
    """[K + n_interior, K]: the K pure vertices, then n_interior Dirichlet(1) draws from a fixed seed."""
    rng = np.random.default_rng(int(seed))
    return np.concatenate([np.eye(K), rng.dirichlet(np.ones(K), size=n_interior)], 0) if n_interior else np.eye(K)


def select_candidate(log_alphas: torch.Tensor, hist: torch.Tensor, logp_k: torch.Tensor, logp_cur: torch.Tensor):
    """Index of the candidate with the smallest batch-mean sampled log-ratio log pi(a_t|s_t) - log q_c(a_t|h_t), and
    the per-candidate means. log_alphas [C, K]; hist / logp_k [N, K]; logp_cur [N]."""
    scores = []
    for la in log_alphas:
        logq = mixture_logprob(mixture_logw(la, hist), logp_k)
        scores.append((logp_cur - logq).mean())
    scores = torch.stack(scores)
    return int(torch.argmin(scores)), scores


# ------------------------------------------------------------------ PPO-side state
class HullKL:
    """Own-policy-hull distance term for one PPO learner (one role, one oracle)."""

    def __init__(self, ref_actors: List, lam: float, n_interior: int = 16, n_check: int = 64, seed: int = 0,
                 mc_samples: int = 8, device="cpu"):
        if not ref_actors:
            raise ValueError("PSD needs at least one earlier same-role policy")
        self.refs = ref_actors
        self.lam = float(lam)
        self.K = len(ref_actors)
        self.mc = int(mc_samples)
        self.device = device
        f = lambda a: torch.as_tensor(np.log(np.where(a > 0, a, 1.0)) + np.where(a > 0, 0.0, -np.inf),
                                      dtype=torch.float32, device=device)
        self.alphas = candidate_alphas(self.K, n_interior, seed)
        self.check_alphas = candidate_alphas(self.K, n_check, seed + 1)
        self.log_alphas, self.log_check = f(self.alphas), f(self.check_alphas)
        self.gen = torch.Generator(device=device).manual_seed(int(seed) + 7)   # MC noise: keeps the global RNG intact
        self.carry: Optional[torch.Tensor] = None
        self._prev = None
        self.sigma_k: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _refs_on(self, td):
        mus, sigmas, logps = [], [], []
        action = td[("agents", "action")]
        for ref in self.refs:
            dist = ref.get_dist(td.select(*ref.in_keys).clone(False))
            mus.append(dist.mean)
            sigmas.append(dist.base_dist.scale)
            logps.append(dist.log_prob(action))
        mu = torch.stack(mus, -2)                       # [E, T, 1, K, A]
        sig = torch.stack(sigmas, -2)
        sigma_k = sig.reshape(-1, self.K, sig.shape[-1])[0]
        if not torch.allclose(sig, sigma_k.expand_as(sig)):
            raise ValueError("hull KL assumes state-independent action std")
        return mu, sigma_k, torch.stack(logps, -1)     # logp [E, T, 1, K]

    @torch.no_grad()
    def prepare(self, td, actor):
        """Before GAE: choose the hull candidate on this batch, store per-step mixture weights / component means for
        the minibatch updates, and return the intrinsic reward lam * KL_t [E, T, 1, 1] plus diagnostics."""
        E, T = td.batch_size[:2]
        if td[("agents", "action")].shape[-2] != 1:
            raise ValueError("hull KL supports one agent")
        mu, sigma_k, logp_k = self._refs_on(td)
        self.sigma_k = sigma_k
        mu, logp_k = mu[:, :, 0], logp_k[:, :, 0]                                   # [E, T, K, A], [E, T, K]
        if not (torch.isfinite(mu).all() and torch.isfinite(logp_k).all()):
            raise FloatingPointError("non-finite reference policy outputs")
        is_init = td.get("is_init").reshape(E, T).bool()
        if self.carry is None:
            if not bool(is_init[:, 0].all()):
                raise ValueError("hull KL must start at episode boundaries")
            self.carry = torch.zeros(E, self.K, device=logp_k.device)
        hist, self.carry = history_logl(logp_k, is_init, self.carry)
        dist = actor.get_dist(td.select(*actor.in_keys).clone(False))
        logp_cur = dist.log_prob(td[("agents", "action")])[:, :, 0]              # [E, T]
        flat = lambda x: x.reshape(E * T, *x.shape[2:])
        info = {}
        if self._prev is not None:                                              # held-out check of the 16 vs 64 set
            i16, i64 = self._prev
            s16 = select_candidate(self.log_alphas[i16:i16 + 1], flat(hist), flat(logp_k), flat(logp_cur))[1][0]
            s64 = select_candidate(self.log_check[i64:i64 + 1], flat(hist), flat(logp_k), flat(logp_cur))[1][0]
            info["hull_heldout_gap_16_vs_64"] = float(s16 - s64)
        c, scores = select_candidate(self.log_alphas, flat(hist), flat(logp_k), flat(logp_cur))
        c64, scores64 = select_candidate(self.log_check, flat(hist), flat(logp_k), flat(logp_cur))
        self._prev = (c, c64)
        logw = mixture_logw(self.log_alphas[c], hist)                           # [E, T, K]
        loc, scale = dist.mean[:, :, 0], dist.base_dist.scale[:, :, 0]          # [E, T, A]
        eps = torch.randn((self.mc,) + tuple(loc.reshape(E * T, -1).shape), generator=self.gen,
                          device=loc.device)
        kl = mc_kl(flat(loc), flat(scale), flat(logw), flat(mu), sigma_k, eps).reshape(E, T)
        if not torch.isfinite(kl).all():
            raise FloatingPointError("non-finite hull KL")
        td.set("hull_logw", logw)
        td.set("hull_mu", mu)
        info.update({"hull_candidate": c, "hull_alpha": self.alphas[c].tolist(), "hull_kl": float(kl.mean()),
                     "hull_logratio_min": float(scores[c]), "hull_logratio_min64": float(scores64[c64]),
                     "hull_intrinsic": float(self.lam * kl.mean()), "hull_weight_entropy":
                     float(-(logw.exp() * logw.clamp_min(-1e30)).nan_to_num().sum(-1).mean())})
        return (self.lam * kl).reshape(E, T, 1, 1), info

    def direct_kl(self, td, dist):
        """Differentiable per-sample KL_t(theta) on a minibatch (pathwise Monte Carlo)."""
        loc, scale = dist.mean[:, 0], dist.base_dist.scale[:, 0]                 # [B, A]
        eps = torch.randn((self.mc,) + tuple(loc.shape), generator=self.gen, device=loc.device)
        return mc_kl(loc, scale, td["hull_logw"], td["hull_mu"], self.sigma_k, eps)
