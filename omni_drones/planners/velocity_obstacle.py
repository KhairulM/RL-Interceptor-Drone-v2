import math
import os.path as osp

import torch
import torch.nn as nn
import yaml
from tensordict import TensorDict

from .planner import PlannerBase


def _fibonacci_sphere(n: int) -> torch.Tensor:
    """Deterministic, ~uniform points on the unit sphere."""
    i = torch.arange(n, dtype=torch.float32) + 0.5
    phi = torch.acos(1.0 - 2.0 * i / n)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    theta = golden * i
    sin_phi = torch.sin(phi)
    return torch.stack(
        [sin_phi * torch.cos(theta), sin_phi * torch.sin(theta), torch.cos(phi)],
        dim=-1,
    )


class VelocityObstaclePlanner(PlannerBase):
    """
    A 3D velocity-obstacle planner.

    Given the ego state, the states of N other agents, and a preferred velocity
    toward the goal, returns a collision-free desired velocity and a short
    lookahead desired position. Uses plain VO (no reciprocity assumption) with
    a discrete candidate set on the 3D sphere, selecting the valid candidate
    closest to ``pref_velocity`` in L2.

    Inputs:
        * root_state: tensor of shape (..., 1, 13) -- pos, quat, lin vel, ang vel.
        * others_state: tensor of shape (..., N, 13).
        * pref_velocity: tensor of shape (..., 1, 3).

    Outputs (TensorDict with batch_size == leading batch dims):
        * desired_position: tensor of shape (..., 1, 3).
        * desired_velocity: tensor of shape (..., 1, 3).
    """

    def __init__(self, uav_params):
        super().__init__()
        self.uav_params = uav_params

        planner_param_path = osp.join(
            osp.dirname(__file__), "cfg", "velocity_obstacle.yaml"
        )
        with open(planner_param_path, "r") as f:
            planner_params = yaml.safe_load(f)

        self.collision_radius = nn.Parameter(
            torch.as_tensor(planner_params["collision_radius"]).float()
        )
        self.time_horizon = nn.Parameter(
            torch.as_tensor(planner_params["time_horizon"]).float()
        )
        self.max_velocity = nn.Parameter(
            torch.as_tensor(planner_params["max_velocity"]).float()
        )
        self.drone_radius = nn.Parameter(
            torch.as_tensor(uav_params["l"] + planner_params["inflation_radius"]).float()
        )
        self.lookahead_dt = nn.Parameter(
            torch.as_tensor(planner_params["lookahead_dt"]).float()
        )

        n_dirs = int(planner_params["n_directions"])
        n_speeds = int(planner_params["n_speeds"])
        max_v = float(planner_params["max_velocity"])
        dirs = _fibonacci_sphere(n_dirs)                                   # (n_dirs, 3)
        speeds = torch.linspace(max_v / n_speeds, max_v, n_speeds)         # (n_speeds,)
        cands = dirs.unsqueeze(0) * speeds.view(-1, 1, 1)                  # (n_speeds, n_dirs, 3)
        cands = cands.reshape(-1, 3)
        # Index 0 reserved as the zero-velocity fallback.
        cands = torch.cat([torch.zeros(1, 3), cands], dim=0)               # (K, 3)
        self.register_buffer("candidates", cands)

    def plan(
        self,
        root_state: torch.Tensor,
        others_state: torch.Tensor,
        pref_velocity: torch.Tensor,
    ) -> TensorDict:
        batch_shape = root_state.shape[:-2]
        device = root_state.device

        ego_pos = root_state[..., :1, 0:3]                  # (*B, 1, 3)
        others_pos = others_state[..., 0:3]                 # (*B, N, 3)
        others_vel = others_state[..., 7:10]                # (*B, N, 3)

        p_rel = others_pos - ego_pos                        # (*B, N, 3)

        # Combined collision radius (other agents share the ego drone_radius).
        R = 2.0 * self.drone_radius
        R2 = R * R

        cands = self.candidates                             # (K, 3)
        K = cands.shape[0]
        expand_ones = (1,) * len(batch_shape)

        v_cand = cands.view(*expand_ones, K, 1, 3)          # (1.., K, 1, 3)
        v_other_b = others_vel.unsqueeze(-3)                # (*B, 1, N, 3)
        p_rel_b = p_rel.unsqueeze(-3)                       # (*B, 1, N, 3)

        # Plain VO: collision iff the line w*t passes within R of p_rel for
        # some t in (0, time_horizon]. With a=w.w, b=-w.p_rel, c=|p_rel|^2-R^2,
        # min-dist^2 along the (clamped) ray is a*t*^2 + 2b*t* + c at
        # t* = clip(b/a, 0, T). Already-overlapping pairs (c<0) are blocked too.
        w = v_cand - v_other_b                              # (*B, K, N, 3)
        a = (w * w).sum(dim=-1)                             # (*B, K, N)
        b = -(w * p_rel_b).sum(dim=-1)                      # (*B, K, N)
        c = (p_rel_b * p_rel_b).sum(dim=-1) - R2            # (*B, 1, N) -> broadcasts

        eps = 1e-8
        t_star = torch.clamp(b / (a + eps), min=0.0, max=float(self.time_horizon))
        min_dist_sq = a * t_star * t_star + 2.0 * b * t_star + c
        collide = (min_dist_sq <= 0.0) | (c < 0.0)          # (*B, K, N)
        invalid = collide.any(dim=-1)                       # (*B, K)

        pref = pref_velocity[..., :1, :].squeeze(-2).unsqueeze(-2)  # (*B, 1, 3)
        diff = cands.view(*expand_ones, K, 3) - pref                # (*B, K, 3)
        cost = (diff * diff).sum(dim=-1)                            # (*B, K)
        cost_masked = cost.masked_fill(invalid, float("inf"))

        best = torch.argmin(cost_masked, dim=-1, keepdim=True)      # (*B, 1)
        all_blocked = invalid.all(dim=-1, keepdim=True)             # (*B, 1)
        best = torch.where(all_blocked, torch.zeros_like(best), best)

        cand_b = cands.view(*expand_ones, K, 3).expand(*batch_shape, K, 3)
        gather_idx = best.unsqueeze(-1).expand(*batch_shape, 1, 3)
        desired_velocity = torch.gather(cand_b, dim=-2, index=gather_idx)  # (*B, 1, 3)

        desired_position = ego_pos + desired_velocity * self.lookahead_dt

        return TensorDict(
            {
                "desired_position": desired_position,
                "desired_velocity": desired_velocity,
            },
            batch_size=batch_shape,
            device=device,
        )
