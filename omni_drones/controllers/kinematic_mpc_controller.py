from __future__ import annotations

import os.path as osp

import yaml

from .intercept_baseline_common import GuidanceBaseline


class KinematicMPCController(GuidanceBaseline):
    """Kinematic MPC: constant-velocity receding-horizon intercept. Predicts
    the evader position over the estimated time-to-go and tracks it, matching
    the evader velocity at intercept."""

    def __init__(self, ctbr, task_cfg, dt: float, config_name: str = "kinematic_mpc_intercept"):
        cfg_path = osp.join(osp.dirname(__file__), "cfg", f"{config_name}.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        super().__init__(ctbr)
        self.dt = float(dt)
        # speed_target [m/s]: assumed pursuer closing speed for time-to-go.
        self.speed_target = float(cfg.get("speed_target", task_cfg.pursuer.get("target_speed", 4.0)))
        # max_tgo [s]: cap on the prediction horizon (time-to-go).
        self.max_tgo = float(cfg.get("max_tgo", 3.0))

    def guidance(self, pos, vel, evader_pos, evader_vel, done):
        rel = evader_pos - pos
        rng = rel.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        t_go = (rng / self.speed_target).clamp(max=self.max_tgo)
        predicted = evader_pos + evader_vel * t_go
        return predicted, evader_vel
