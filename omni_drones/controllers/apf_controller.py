from __future__ import annotations

import os.path as osp

import torch
import yaml

from .intercept_baseline_common import GuidanceBaseline


class APFController(GuidanceBaseline):
    """Artificial Potential Field (APF) pursuer.

    Steers along the negative gradient of an attractive potential centred on the
    evader. With no obstacles the field reduces to a single attractive well:

        F_att = attract_gain * (evader_pos - pos)

    which is the gradient of the quadratic potential ``0.5*k*||evader_pos-pos||^2``.
    Far from the evader this saturates to a constant-magnitude approach
    (``max_speed``); close in it ramps down linearly, giving the smooth
    deceleration characteristic of a potential well. A feed-forward of the
    evader velocity lets the field track a moving target instead of always
    lagging behind it.

    The attractive velocity ``v_des`` is handed to the shared
    :class:`GeometricCTBR` as the velocity feed-forward, together with the goal
    (the evader position) as the position target. Aiming the position loop at the
    goal itself -- rather than a look-ahead point projected from ``v_des`` -- is
    what lets the pursuer *damp onto* a stationary target instead of overshooting
    and orbiting it: near the goal both the position error and ``v_des`` vanish,
    so the outer loop simply arrests the remaining velocity. (It also lets the
    CTBR's altitude integral null the hover offset, since the position target is
    now the true goal altitude.)
    """

    def __init__(self, ctbr, task_cfg, config_name: str = "apf_intercept"):
        cfg_path = osp.join(osp.dirname(__file__), "cfg", f"{config_name}.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        super().__init__(ctbr)
        # attract_gain [1/s]: strength of the attractive well (velocity per metre
        # of range) in the near field.
        self.attract_gain = float(cfg.get("attract_gain", 2.0))
        # max_speed [m/s]: cap on the attractive-field velocity magnitude.
        self.max_speed = float(cfg.get("max_speed", task_cfg.pursuer.get("target_speed", 4.0)))
        # decel [m/s^2]: the pursuer's usable horizontal deceleration. The
        # approach speed is capped at sqrt(2*decel*range) so the drone never
        # travels faster than it can brake from before reaching the target --
        # without this the thrust-limited Crazyflie (only ~5 m/s^2 of braking at
        # the tilt cap) overshoots max_speed straight past a stationary target and
        # orbits it. Keep it at or below the true capability.
        self.decel = float(cfg.get("decel", 4.0))
        # vel_feedforward: add the evader velocity so the field tracks a mover.
        self.vel_feedforward = bool(cfg.get("vel_feedforward", True))

    def guidance(self, pos, vel, evader_pos, evader_vel, done):
        rel = evader_pos - pos
        rng = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        los = rel / rng
        # Attractive-field speed: the quadratic well's linear ramp near the goal,
        # but never above the braking-limited speed sqrt(2*decel*range) nor the
        # global max_speed. This is what lets the pursuer arrest onto the target.
        v_mag = torch.minimum(self.attract_gain * rng, (2.0 * self.decel * rng).sqrt())
        v_mag = v_mag.clamp(max=self.max_speed)
        v_des = v_mag * los
        if self.vel_feedforward:
            v_des = v_des + evader_vel
        # APF is a velocity-field law, so the horizontal channel must track the
        # field velocity, not a position. Aiming the horizontal position target
        # at the drone's *own* xy makes the outer loop pure velocity control
        # there (a_h = kv*(v_des - v)), so the braking-limited speed profile is
        # actually followed and the pursuer arrests onto the target instead of
        # the position P-term saturating and flinging it past. The vertical
        # target stays the goal altitude so altitude-hold and its integral work.
        target_pos = torch.cat([pos[..., 0:2], evader_pos[..., 2:3]], dim=-1)
        return target_pos, v_des
