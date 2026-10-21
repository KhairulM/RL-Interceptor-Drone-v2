"""Batched Echo State Network with online RLS readout (ESN-OA, arXiv:2602.06326).

Readout modes:
  per_env       every env starts each episode at W_out=0 and adapts with its own RLS (ESN-OA).
  shared        one readout fitted centrally on data pooled from all envs; no per-env RLS.
  shared_adapt  each episode starts from the shared readout and adapts with per-env RLS (ESN-OA-PT).

The reservoir state x stays per env in every mode: it is that env's trajectory memory.

Torch-only on purpose: the deploy scripts load this file by path to avoid
importing the ``omni_drones`` package (which pulls in Isaac Sim).
"""

import os
from typing import Any, Dict, Optional, Union

import torch

READOUT_MODES = ("per_env", "shared", "shared_adapt")


def readout_sidecar_path(ckpt_path: str) -> str:
    """Where the shared readout is saved next to a policy checkpoint."""
    root, _ = os.path.splitext(ckpt_path)
    return root + "_esn.pt"


def save_readout_sidecar(env, ckpt_path: str) -> Optional[str]:
    """Save ``env.esn``'s shared readout next to ``ckpt_path`` (no-op for per_env / no ESN)."""
    esn = getattr(env, "esn", None)
    if esn is None or esn.readout == "per_env":
        return None
    path = readout_sidecar_path(ckpt_path)
    torch.save(esn.readout_state(), path)
    return path


class BatchedESN:
    """Fixed random reservoir shared by all envs, per-env reservoir state, and a
    readout that is per env (RLS, Eqs. 5-8), shared (central ridge fit), or both."""

    def __init__(
        self,
        num_envs: int,
        input_dim: int,
        output_dim: int,
        reservoir_size: int = 500,
        spectral_radius: float = 0.9,
        leak_rate: float = 0.3,
        input_scale: float = 0.1,
        forgetting_factor: float = 0.99,
        delta: float = 100.0,
        p_trace_max: float = 1.0e6,
        use_bias: bool = True,
        seed: int = 0,
        readout: str = "per_env",
        fit_interval: int = 50,
        fit_forgetting: float = 0.999,
        ridge: float = 1.0e-4,
        device: Union[str, torch.device] = "cpu",
        weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if readout not in READOUT_MODES:
            raise ValueError(f"readout must be one of {READOUT_MODES}, got {readout!r}")
        self.num_envs = int(num_envs)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.reservoir_size = int(reservoir_size)
        self.spectral_radius = float(spectral_radius)
        self.leak_rate = float(leak_rate)
        self.input_scale = float(input_scale)
        self.forgetting_factor = float(forgetting_factor)
        self.delta = float(delta)
        self.p_trace_max = float(p_trace_max)
        self.use_bias = bool(use_bias)
        self.seed = int(seed)
        self.readout = readout
        self.fit_interval = int(fit_interval)
        self.fit_forgetting = float(fit_forgetting)
        self.ridge = float(ridge)
        self.device = torch.device(device)
        self.adapt = readout != "shared"

        if weights is None:
            weights = self._generate_weights()
        self.W_in = weights["W_in"].to(self.device, torch.float32)
        self.W_res = weights["W_res"].to(self.device, torch.float32)
        assert self.W_in.shape == (self.reservoir_size, self.input_dim)
        assert self.W_res.shape == (self.reservoir_size, self.reservoir_size)

        self.feat_dim = self.reservoir_size + (1 if self.use_bias else 0)
        n, f = self.num_envs, self.feat_dim
        self.x = torch.zeros(n, self.reservoir_size, device=self.device)
        self.feat = torch.zeros(n, f, device=self.device)
        self.W_out = torch.zeros(n, self.output_dim, f, device=self.device)
        # Readout each env is reset to (zeros for per_env; shared/opponent readout otherwise).
        self.W_init = torch.zeros(n, self.output_dim, f, device=self.device)
        self.P = None
        if self.adapt:
            self.P = torch.zeros(n, f, f, device=self.device)
            self.P.diagonal(dim1=-2, dim2=-1).fill_(self.delta)
        self.pred = torch.zeros(n, self.output_dim, device=self.device)
        self.err = torch.zeros(n, device=self.device)
        self.has_prev = torch.zeros(n, dtype=torch.bool, device=self.device)

        # Central fit: exponentially-forgotten normal equations R W^T = Z^T, in float64.
        self.W_shared = torch.zeros(self.output_dim, f, device=self.device)
        self.R = torch.zeros(f, f, dtype=torch.float64, device=self.device)
        self.Z = torch.zeros(self.output_dim, f, dtype=torch.float64, device=self.device)
        self.fit_steps = 0

    def _generate_weights(self) -> Dict[str, torch.Tensor]:
        # CPU generator so the same seed gives the same reservoir on any machine.
        g = torch.Generator().manual_seed(self.seed)
        nx = self.reservoir_size
        W_in = (torch.rand(nx, self.input_dim, generator=g) * 2 - 1) * self.input_scale
        W_res = torch.rand(nx, nx, generator=g) * 2 - 1
        rho = torch.linalg.eigvals(W_res.double()).abs().max().item()
        W_res = W_res * (self.spectral_radius / rho)
        return {"W_in": W_in, "W_res": W_res}

    @classmethod
    def from_export(cls, data: Dict[str, Any], num_envs: int = 1, device="cpu") -> "BatchedESN":
        esn = cls(
            num_envs=num_envs,
            weights={"W_in": data["W_in"], "W_res": data["W_res"]},
            device=device,
            **data["hparams"],
        )
        if data.get("W_shared") is not None:
            esn.set_shared_readout(data["W_shared"])
        return esn

    def export(self) -> Dict[str, Any]:
        return {
            "W_in": self.W_in.detach().cpu(),
            "W_res": self.W_res.detach().cpu(),
            "W_shared": self.W_shared.detach().cpu() if self.readout != "per_env" else None,
            "hparams": {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "reservoir_size": self.reservoir_size,
                "spectral_radius": self.spectral_radius,
                "leak_rate": self.leak_rate,
                "input_scale": self.input_scale,
                "forgetting_factor": self.forgetting_factor,
                "delta": self.delta,
                "p_trace_max": self.p_trace_max,
                "use_bias": self.use_bias,
                "seed": self.seed,
                "readout": self.readout,
                "fit_interval": self.fit_interval,
                "fit_forgetting": self.fit_forgetting,
                "ridge": self.ridge,
            },
        }

    def readout_state(self) -> Dict[str, Any]:
        """Central-fit state saved next to a checkpoint (warm start / export)."""
        return {
            "W_shared": self.W_shared.detach().cpu(),
            "R": self.R.detach().cpu(),
            "Z": self.Z.detach().cpu(),
            "fit_steps": self.fit_steps,
        }

    def load_readout_state(self, state: Dict[str, Any]):
        self.R.copy_(state["R"].to(self.device))
        self.Z.copy_(state["Z"].to(self.device))
        self.fit_steps = int(state.get("fit_steps", 0))
        self.set_shared_readout(state["W_shared"])

    def set_shared_readout(self, W: torch.Tensor, env_ids: Optional[torch.Tensor] = None):
        """Make ``W`` the reset readout of ``env_ids`` (all envs if None)."""
        W = W.to(self.device, torch.float32)
        if env_ids is None:
            self.W_shared.copy_(W)
            self.W_init[:] = W
            if not self.adapt:
                self.W_out[:] = W
        else:
            self.W_init[env_ids] = W
            if not self.adapt:
                self.W_out[env_ids] = W

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() == 0:
            return
        self.x[env_ids] = 0.0
        self.feat[env_ids] = 0.0
        self.W_out[env_ids] = self.W_init[env_ids]
        if self.P is not None:
            P = torch.zeros(env_ids.numel(), self.feat_dim, self.feat_dim, device=self.device)
            P.diagonal(dim1=-2, dim2=-1).fill_(self.delta)
            self.P[env_ids] = P
        self.pred[env_ids] = 0.0
        self.err[env_ids] = 0.0
        self.has_prev[env_ids] = False

    @torch.no_grad()
    def step(
        self,
        u: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        fit: bool = False,
    ) -> torch.Tensor:
        """Advance envs in ``mask``; returns the next-step prediction for all envs.

        u: [N, input_dim] current input. target: [N, output_dim] values observed
        now, i.e. the ground truth for the prediction made on the previous step.
        fit: also accumulate these samples into the central (shared) readout fit.
        """
        if mask is None or bool(mask.all()):
            rows = slice(None)
        else:
            rows = mask.nonzero(as_tuple=False).squeeze(-1)
            if rows.numel() == 0:
                return self.pred

        full = isinstance(rows, slice)
        feat = self.feat[rows]
        upd = self.has_prev[rows]
        u = u[rows].to(torch.float32)
        target = target[rows].to(torch.float32)
        e = target - self.pred[rows]
        err = torch.where(upd, e.norm(dim=-1), torch.zeros_like(e[..., 0]))

        if fit and bool(upd.any()):
            self._accumulate(feat[upd], target[upd])
        W_out = self.W_out if full else self.W_out[rows]

        if self.adapt:
            P = self.P if full else self.P[rows]
            # RLS update of the previous prediction (envs without one get k=0, lambda=1).
            Px = torch.bmm(P, feat.unsqueeze(-1)).squeeze(-1)
            lam = torch.where(upd, self.forgetting_factor, 1.0).to(Px.dtype)
            k = Px / (lam + (feat * Px).sum(-1)).unsqueeze(-1) * upd.unsqueeze(-1)
            W_out.add_(e.unsqueeze(-1) * k.unsqueeze(1))
            P.baddbmm_(k.unsqueeze(-1), Px.unsqueeze(1), alpha=-1.0)
            P.div_(lam.view(-1, 1, 1))
            # Cap P growth in unexcited directions (1/lambda wind-up).
            trace = P.diagonal(dim1=-2, dim2=-1).sum(-1)
            P.mul_((self.p_trace_max / trace.clamp(min=1e-12)).clamp(max=1.0).view(-1, 1, 1))
            if not full:
                self.P[rows] = P

        x = self.x[rows]
        x = (1.0 - self.leak_rate) * x + self.leak_rate * torch.tanh(
            u @ self.W_in.T + x @ self.W_res.T
        )
        if self.use_bias:
            feat = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        else:
            feat = x
        pred = torch.bmm(W_out, feat.unsqueeze(-1)).squeeze(-1)

        if not full:
            self.W_out[rows] = W_out
        self.x[rows] = x
        self.feat[rows] = feat
        self.pred[rows] = pred
        self.err[rows] = err
        self.has_prev[rows] = True

        bad = ~(torch.isfinite(self.pred).all(-1) & torch.isfinite(self.W_out).flatten(1).all(-1))
        if bool(bad.any()):
            self.reset(bad.nonzero(as_tuple=False).squeeze(-1))
        return self.pred

    def _accumulate(self, feat: torch.Tensor, target: torch.Tensor):
        self.R.mul_(self.fit_forgetting).add_((feat.T @ feat).double())
        self.Z.mul_(self.fit_forgetting).add_((target.T @ feat).double())
        self.fit_steps += 1
        if self.fit_steps % self.fit_interval == 0:
            self.solve()

    def solve(self):
        """Ridge solve of the pooled normal equations; becomes every env's reset readout."""
        reg = self.ridge * self.R.diagonal().mean().clamp(min=1e-12)
        A = self.R + reg * torch.eye(self.feat_dim, dtype=self.R.dtype, device=self.device)
        W = torch.linalg.solve(A, self.Z.T).T
        if bool(torch.isfinite(W).all()):
            self.set_shared_readout(W.float())
