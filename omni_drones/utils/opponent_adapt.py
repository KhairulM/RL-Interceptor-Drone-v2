"""Online opponent adaptation for the pursuer (thesis experiment A1): horizon-H opponent predictor with an online
recursive-least-squares (RLS) last layer, in the spirit of Neural-Fly (O'Connell et al., Sci. Robotics 2022).

Per step each env gets z_t = [rel_pos, rel_vel, own_vel] (world frame, rel = evader - pursuer). The predictor
outputs the evader's relative displacement over the next H steps, y_hat_t = theta phi(h_t), where h_t is the last
``history`` z's. When the true displacement of a past prediction becomes known (H steps later), theta is updated by
RLS with forgetting. Feature kinds (only the features differ; the RLS core is shared):

  nf     learned basis phi (MLP trained offline with per-evader heads; theta starts at the learned mean head)
  esn    random leaky reservoir driven by z_t (echo state network); theta starts at 0
  raw    the normalised history itself (linear RLS on raw inputs); theta starts at 0
  stack  no predictor: outputs the normalised history (frame-stack control)

``target``: "disp" = relative displacement over H (above); "dvel" = change of the evader's own velocity
(rel_vel + own_vel) over H, i.e. H dt x its mean acceleration (thesis A2: learned acceleration estimator).

``freeze`` disables the RLS update (theta stays at its initial value). The covariance trace is capped at
``p_max_ratio`` x its initial value (bounded-covariance RLS), which prevents 1/lambda wind-up in directions the
opponent does not excite. ``reset_on_episode=False`` keeps theta and P
across episodes (across-episode adaptation); the history and pending predictions are always cleared at a reset.

Pure torch; also used offline on recorded trajectories (A1a). Tests load it by path.
"""

from typing import Optional

import torch
import torch.nn as nn

Z_DIM = 9


def target_signal(z: torch.Tensor, target: str) -> torch.Tensor:
    """Quantity whose change over H steps is predicted: rel_pos (disp) or the evader's velocity (dvel)."""
    return z[..., :3] if target == "disp" else z[..., 3:6] + z[..., 6:9]


class BasisNet(nn.Module):
    """phi: flattened normalised history -> d features (a bias feature is appended outside)."""

    def __init__(self, in_dim: int, d: int = 32, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, d), nn.Tanh())

    def forward(self, x):
        return self.net(x)


class OpponentAdapter:
    def __init__(self, num_envs: int, kind: str = "nf", history: int = 10, horizon: int = 15,
                 forgetting: float = 0.98, delta: float = 10.0, model_path: Optional[str] = None,
                 reservoir_size: int = 128, freeze: bool = False, reset_on_episode: bool = True,
                 p_max_ratio: float = 1.0, target: str = "disp", seed: int = 0, device="cpu"):
        if kind not in ("nf", "esn", "raw", "stack"):
            raise ValueError(f"unknown adapter kind {kind!r}")
        if target not in ("disp", "dvel"):
            raise ValueError(f"unknown adapter target {target!r}")
        self.target = target
        self.n, self.kind, self.K, self.H = num_envs, kind, int(history), int(horizon)
        self.lam, self.delta, self.freeze, self.reset_on_episode = float(forgetting), float(delta), freeze, reset_on_episode
        self.device = device
        self.z_mean = torch.zeros(Z_DIM, device=device)
        self.z_std = torch.ones(Z_DIM, device=device)
        self.phi = None
        theta0 = None
        if model_path:
            m = torch.load(model_path, map_location=device)
            self.z_mean, self.z_std = m["z_mean"].to(device), m["z_std"].to(device)
            if kind == "nf":
                self.phi = BasisNet(m["in_dim"], m["d"], m["hidden"]).to(device)
                self.phi.load_state_dict(m["phi"])
                self.phi.eval()
                theta0 = m["theta0"].to(device)
                if m["horizon"] != self.H or m["history"] != self.K or m.get("target", "disp") != target:
                    raise ValueError("adapter model was trained for a different horizon/history/target")
        elif kind == "nf":
            raise ValueError("kind=nf needs model_path")
        if kind == "esn":
            g = torch.Generator().manual_seed(seed)
            W = torch.randn(reservoir_size, reservoir_size, generator=g)
            W *= 0.9 / torch.linalg.eigvals(W).abs().max()
            self.W_res = W.to(device)
            self.W_in = (torch.rand(reservoir_size, Z_DIM, generator=g) * 2 - 1).to(device) * 0.5
            self.leak = 0.3
            self.x = torch.zeros(num_envs, reservoir_size, device=device)
        feat_dim = {"nf": (self.phi.net[-2].out_features if self.phi is not None else 0),
                    "esn": reservoir_size, "raw": self.K * Z_DIM, "stack": 0}[kind] + 1
        self.F = feat_dim
        self.p_trace_max = float(p_max_ratio) * self.delta * feat_dim
        self.out_dim = self.K * Z_DIM if kind == "stack" else 3
        self.hist = torch.zeros(num_envs, self.K, Z_DIM, device=device)
        self.t = torch.zeros(num_envs, dtype=torch.long, device=device)    # steps in the current episode
        if kind != "stack":
            self.theta0 = theta0 if theta0 is not None else torch.zeros(3, self.F, device=device)
            self.theta = self.theta0.expand(num_envs, 3, self.F).clone()
            self.P = torch.eye(self.F, device=device).expand(num_envs, self.F, self.F).clone() * self.delta
            self.feat_buf = torch.zeros(num_envs, self.H + 1, self.F, device=device)
            self.pos_buf = torch.zeros(num_envs, self.H + 1, 3, device=device)
            self.pred_buf = torch.zeros(num_envs, self.H + 1, 3, device=device)
        self.pred = torch.zeros(num_envs, self.out_dim, device=device)
        self.err = torch.zeros(num_envs, device=device)          # |y - y_hat| of the prediction resolved this step
        self.err_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # ---------------------------------------------------------------- state
    def reset(self, env_ids: torch.Tensor):
        self.hist[env_ids] = 0.0
        self.t[env_ids] = 0
        self.pred[env_ids] = 0.0
        self.err_valid[env_ids] = False
        if self.kind == "esn":
            self.x[env_ids] = 0.0
        if self.kind != "stack" and self.reset_on_episode:
            self.theta[env_ids] = self.theta0
            self.P[env_ids] = torch.eye(self.F, device=self.device) * self.delta

    def reset_adaptation(self, env_ids: torch.Tensor):
        """Forget the adapted readout even when reset_on_episode=False (new opponent)."""
        if self.kind != "stack":
            self.theta[env_ids] = self.theta0
            self.P[env_ids] = torch.eye(self.F, device=self.device) * self.delta

    # ------------------------------------------------------------- features
    @torch.no_grad()
    def _features(self, rows, zn):
        if self.kind == "nf":
            f = self.phi(self.hist[rows].flatten(1))
        elif self.kind == "esn":
            x = self.x[rows]
            x = (1 - self.leak) * x + self.leak * torch.tanh(zn @ self.W_in.T + x @ self.W_res.T)
            self.x[rows] = x
            f = x
        else:
            f = self.hist[rows].flatten(1)
        return torch.cat([f, torch.ones_like(f[:, :1])], dim=-1)

    # ----------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None,
             tgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Advance envs in ``mask`` with z [N, 9]; returns the current output for all envs.

        ``tgt`` [N, 3] overrides the signal whose change is the target (used when it is not part of z).
        """
        rows = torch.arange(self.n, device=self.device) if mask is None else mask.nonzero().squeeze(-1)
        if rows.numel() == 0:
            return self.pred
        zn = (z[rows] - self.z_mean) / self.z_std
        self.hist[rows] = torch.cat([self.hist[rows, 1:], zn.unsqueeze(1)], dim=1)
        t = self.t[rows]
        self.err_valid[rows] = False
        if self.kind == "stack":
            self.pred[rows] = self.hist[rows].flatten(1)
            self.t[rows] = t + 1
            return self.pred

        f = self._features(rows, zn)
        rel_pos = tgt[rows] if tgt is not None else target_signal(z[rows], self.target)
        slot = t % (self.H + 1)
        old = (t - self.H) % (self.H + 1)
        ready = t >= self.H
        if bool(ready.any()):
            r = rows[ready]
            f_old = self.feat_buf[r, old[ready]]
            y = rel_pos[ready] - self.pos_buf[r, old[ready]]
            e = y - torch.bmm(self.theta[r], f_old.unsqueeze(-1)).squeeze(-1)
            self.err[r] = (y - self.pred_buf[r, old[ready]]).norm(dim=-1)
            self.err_valid[r] = True
            if not self.freeze:
                P = self.P[r]
                Pf = torch.bmm(P, f_old.unsqueeze(-1)).squeeze(-1)
                k = Pf / (self.lam + (f_old * Pf).sum(-1, keepdim=True))
                self.theta[r] = self.theta[r] + e.unsqueeze(-1) * k.unsqueeze(1)
                P = (P - k.unsqueeze(-1) * Pf.unsqueeze(1)) / self.lam
                tr = P.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=1e-12)
                P = P * (self.p_trace_max / tr).clamp(max=1.0).view(-1, 1, 1)
                self.P[r] = P
        pred = torch.bmm(self.theta[rows], f.unsqueeze(-1)).squeeze(-1)
        self.feat_buf[rows, slot] = f
        self.pos_buf[rows, slot] = rel_pos
        self.pred_buf[rows, slot] = pred
        self.pred[rows] = pred
        self.t[rows] = t + 1
        bad = ~torch.isfinite(self.theta).flatten(1).all(-1)
        if bool(bad.any()):
            self.reset_adaptation(bad.nonzero().squeeze(-1))
        return self.pred
