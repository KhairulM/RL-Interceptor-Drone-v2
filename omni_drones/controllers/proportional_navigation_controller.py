from __future__ import annotations

import os.path as osp

import torch
import yaml

from .intercept_baseline_common import GuidanceBaseline


class ProportionalNavigationController(GuidanceBaseline):
    """Proportional navigation, i.e. steer to the constant-bearing collision
    course.

    PN nulls the line-of-sight rotation rate, which for a constant-velocity
    evader is equivalent to flying straight at the point where pursuer and
    evader paths intersect. That intercept point is found in closed form by
    solving, for the time-to-go ``t``,

        || (evader_pos - pos) + evader_vel * t || = pursuer_speed * t,

    a quadratic in ``t`` whose smallest positive root is the collision
    time-to-go. Aiming at ``evader_pos + evader_vel * t`` places the pursuer on
    a constant-bearing course. This is exact for head-on, crossing and tail-chase
    geometries alike (a fixed lead-time heuristic is not: it over-leads slow
    crossers and points backwards on a closing head-on shot).

    When no real collision course exists (the evader is as fast as or faster than
    the pursuer, so the quadratic has no positive root) it falls back to a
    range/speed time-to-go so the pursuer still leads sensibly.
    """

    def __init__(self, ctbr, task_cfg, dt: float, config_name: str = "pn_intercept"):
        cfg_path = osp.join(osp.dirname(__file__), "cfg", f"{config_name}.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        super().__init__(ctbr)
        self.dt = float(dt)
        # pursuer_speed [m/s]: pursuer speed capability used to solve the
        # collision triangle. Should match the pursuer's achievable cruise speed.
        self.pursuer_speed = float(
            cfg.get("pursuer_speed", cfg.get("closing_speed", task_cfg.pursuer.get("target_speed", 4.0)))
        )
        # nav_gain: multiplier on the collision time-to-go lead. 1.0 is the exact
        # constant-bearing course; >1 leads more aggressively.
        self.nav_gain = float(cfg.get("nav_gain", 1.0))
        # max_tgo [s]: safety cap on the time-to-go lead (bounds extrapolation of
        # a maneuvering evader).
        self.max_tgo = float(cfg.get("max_tgo", 3.0))

    def guidance(self, pos, vel, evader_pos, evader_vel, done):
        rel = evader_pos - pos
        rng = rel.norm(dim=-1, keepdim=True).clamp_min(1e-3)

        # Solve || rel + evader_vel * t || = pursuer_speed * t  ->  a t^2 + b t + c = 0.
        a = (evader_vel * evader_vel).sum(dim=-1, keepdim=True) - self.pursuer_speed ** 2
        b = 2.0 * (rel * evader_vel).sum(dim=-1, keepdim=True)
        c = (rel * rel).sum(dim=-1, keepdim=True)
        disc = b * b - 4.0 * a * c
        sqrt_disc = disc.clamp_min(0.0).sqrt()

        # Guard the a~0 case (evader speed == pursuer speed) to avoid div-by-zero.
        a_safe = torch.where(a.abs() < 1e-6, torch.full_like(a, -1e-6), a)
        t1 = (-b - sqrt_disc) / (2.0 * a_safe)
        t2 = (-b + sqrt_disc) / (2.0 * a_safe)
        big = torch.full_like(t1, 1e9)
        t1p = torch.where(t1 > 1e-4, t1, big)
        t2p = torch.where(t2 > 1e-4, t2, big)
        t_root = torch.minimum(t1p, t2p)  # smallest strictly-positive root

        # Fall back to a range/speed estimate when there is no valid collision root.
        fallback = rng / max(self.pursuer_speed, 1e-3)
        valid = (disc >= 0.0) & (t_root < 1e9)
        t_go = torch.where(valid, t_root, fallback).clamp(max=self.max_tgo)

        lead = evader_pos + evader_vel * (self.nav_gain * t_go)
        return lead, evader_vel
