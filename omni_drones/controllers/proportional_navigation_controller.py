from __future__ import annotations

import os.path as osp

import yaml

from .intercept_baseline_common import GuidanceBaseline


class ProportionalNavigationController(GuidanceBaseline):
    """Proportional navigation: aim at the constant-bearing intercept point
    (evader position led by its velocity over the estimated time-to-go)."""

    def __init__(self, ctbr, task_cfg, dt: float, config_name: str = "pn_intercept"):
        cfg_path = osp.join(osp.dirname(__file__), "cfg", f"{config_name}.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        super().__init__(ctbr)
        self.dt = float(dt)
        # closing_speed [m/s]: floor on estimated closing speed for time-to-go.
        self.closing_speed = float(cfg.get("closing_speed", task_cfg.pursuer.get("target_speed", 4.0)))
        # nav_gain: multiplier on the lead time (constant-bearing intercept).
        self.nav_gain = float(cfg.get("nav_gain", 1.0))
        # max_tgo [s]: cap on the estimated time-to-go used for the lead.
        self.max_tgo = float(cfg.get("max_tgo", 3.0))

    def guidance(self, pos, vel, evader_pos, evader_vel, done):
        rel = evader_pos - pos
        rng = rel.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        los = rel / rng
        closing = (-(evader_vel - vel) * los).sum(dim=-1, keepdim=True).clamp_min(self.closing_speed)
        t_go = (rng / closing).clamp(max=self.max_tgo)
        lead = evader_pos + evader_vel * (self.nav_gain * t_go)
        return lead, evader_vel
