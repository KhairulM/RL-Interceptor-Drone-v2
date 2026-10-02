"""Classical (non-learned) intercept controllers for the real Crazyflie.

This is the hardware twin of the guidance baselines evaluated in
``scripts/evaluate.py`` (``omni_drones/controllers/{pure_pursuit,
proportional_navigation,apf,nonlinear_mpc}_controller.py``): pure pursuit,
proportional navigation, artificial potential field and sampling-based
nonlinear MPC (MPPI). Each one emits exactly the same pre-tanh CTBR action the
RL policy emits, which ``intercept_common.decode_action_to_ctbr`` turns into
the body-rate + collective-thrust setpoint the firmware's rate PID tracks -- so
the baseline and the policy fly through an identical command path and the
sim-to-real comparison stays apples-to-apples.

Everything around the controller (link management, takeoff, evader motion
scripting, mocap forwarding, the fixed-rate 50 Hz loop and the altitude/staleness
safety cutouts) is inherited from
:class:`intercept_controller_v2.InterceptController`; only the two hooks
``_setup_policy`` and ``_compute_action`` are replaced.

The guidance laws are re-implemented here rather than imported from
``omni_drones``: the deployment venv (Crazyswarm2, Python 3.10) must not import
the Isaac-coupled training package. They are line-for-line equivalents of the
sim versions, batched over a single drone, with two deliberate differences
noted inline (body-frame angular velocity, and the real drone's hover throttle
coming from config instead of the URDF rotor model).

Run with::

    python traditional_controller.py --method pn [--config config_v2.yaml]
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import time
from typing import Optional

import numpy as np
import torch

import intercept_common as ic
from intercept_controller_v2 import (
    DEFAULT_CONFIG_PATH,
    InterceptController,
    _read_yaml,
)

logger = logging.getLogger(__name__)

METHODS = ('pure_pursuit', 'pn', 'apf', 'nonlinear_mpc')


# ---------------------------------------------------------------------------
# Shared outer loop: position/velocity target -> pre-tanh CTBR action
# ---------------------------------------------------------------------------
class GeometricCTBR:
    """SE(3)-style outer loop, mirroring ``intercept_baseline_common.GeometricCTBR``.

    Produces the pre-tanh ``[body_rate(3), thrust(1)]`` action so the proven
    on-board rate PID -- the same stack the RL policy drives -- handles the
    low-level stabilization. A full ``LeePositionController`` is only marginally
    stable on the motor-lagged Crazyflie, so it is deliberately not used.

    Tunables (``traditional.ctbr`` in the YAML):
        kp, kv       : outer position / velocity P gains.
        k_att        : attitude (thrust-vector) alignment gain -> body rates.
        k_yaw        : yaw-alignment gain.
        max_tilt_deg : clamp on the commanded tilt (limits altitude loss).
        ki_z         : integral on the altitude channel (nulls hover-throttle
                       miscalibration, which on hardware is the difference
                       between hovering at the target and 0.2 m under it).
    """

    def __init__(
        self,
        g: float,
        hover_throttle: float,
        target_clip: float,
        min_ratio: float,
        max_ratio: float,
        dt: float,
        kp: float = 8.0,
        kv: float = 7.0,
        k_att: float = 6.0,
        k_yaw: float = 1.0,
        max_tilt_deg: float = 28.0,
        ki_z: float = 10.0,
        i_limit_z: float = 1.0,
        i_speed_gate: float = 1.5,
    ):
        self.g = float(g)
        self.hover = float(hover_throttle)
        self.target_clip = float(target_clip)
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)
        self.dt = float(dt)
        self.kp = float(kp)
        self.kv = float(kv)
        self.k_att = float(k_att)
        self.k_yaw = float(k_yaw)
        self.max_tilt = math.radians(float(max_tilt_deg))
        self.ki_z = float(ki_z)
        self.i_limit_z = float(i_limit_z)
        # The integral only accumulates while the drone is slow (settling onto
        # the target), which is exactly when the steady-state offset shows. It
        # stays frozen during a fast approach so it cannot wind up on the APF
        # baseline's leading velocity waypoint.
        self.i_speed_gate = float(i_speed_gate)
        self.integ_z: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.integ_z = None

    def compute(self, drone_state: torch.Tensor, target_pos: torch.Tensor,
                target_vel: Optional[torch.Tensor],
                target_yaw: torch.Tensor) -> torch.Tensor:
        pos = drone_state[..., 0:3]
        quat = drone_state[..., 3:7]
        vel = drone_state[..., 7:10]
        R = ic.quaternion_to_rotation_matrix(quat)  # world_from_body [..., 3, 3]

        if target_vel is None:
            target_vel = torch.zeros_like(vel)

        z_err = target_pos[..., 2:3] - pos[..., 2:3]  # + when drone is below target
        if self.integ_z is None or self.integ_z.shape != z_err.shape:
            self.integ_z = torch.zeros_like(z_err)
        gate = (vel.norm(dim=-1, keepdim=True) < self.i_speed_gate).to(z_err.dtype)
        self.integ_z = (self.integ_z + z_err * self.dt * gate).clamp(
            -self.i_limit_z, self.i_limit_z
        )

        # Altitude-priority thrust allocation. The Crazyflie's thrust-to-weight
        # is only ~1.4, so a fixed tilt limit lets the horizontal chase steal
        # the lift needed to hold altitude and the drone sinks into the floor
        # just short of the target. Reserve the vertical specific thrust first,
        # then spend whatever is left on horizontal acceleration.
        a_max = self.g * (self.max_ratio / self.hover) ** 2

        az = (-self.kp * (pos[..., 2:3] - target_pos[..., 2:3])
              - self.kv * (vel[..., 2:3] - target_vel[..., 2:3])
              + self.ki_z * self.integ_z + self.g)
        az = az.clamp(0.3 * self.g, 0.98 * a_max)

        headroom = (a_max ** 2 - az ** 2).clamp_min(0.0).sqrt()
        max_horiz = torch.minimum(headroom, math.tan(self.max_tilt) * az)
        horiz = (-self.kp * (pos[..., 0:2] - target_pos[..., 0:2])
                 - self.kv * (vel[..., 0:2] - target_vel[..., 0:2]))
        hn = horiz.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        horiz = horiz * (max_horiz / hn).clamp(max=1.0)
        a_des = torch.cat([horiz, az], dim=-1)

        b3 = R[..., :, 2]  # current body-z in world
        # Collective thrust to realise |a_des| along the (near-vertical) thrust
        # axis; thrust ~ throttle^2.
        f_acc = a_des.norm(dim=-1, keepdim=True).clamp_min(0.1 * self.g)
        throttle = self.hover * torch.sqrt((f_acc / self.g).clamp(0.04, 4.0))

        # Body rates that align body-z with the desired thrust direction.
        b3_des = a_des / a_des.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        e_R = torch.cross(b3, b3_des, dim=-1)  # world-frame rotation error
        omega_world = self.k_att * e_R
        omega_body = torch.einsum('...ji,...j->...i', R, omega_world)  # R^T @ omega

        yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
        yaw_err = torch.atan2(torch.sin(target_yaw - yaw), torch.cos(target_yaw - yaw))
        wz = self.k_yaw * yaw_err

        body_rate = torch.stack([omega_body[..., 0], omega_body[..., 1], wz], dim=-1)
        rate_deg = body_rate * (180.0 / math.pi)
        return pack_ctbr_action(
            rate_deg, throttle, self.target_clip, self.min_ratio, self.max_ratio
        )


def pack_ctbr_action(rate_deg: torch.Tensor, throttle: torch.Tensor,
                     target_clip: float, min_ratio: float,
                     max_ratio: float) -> torch.Tensor:
    """Invert ``decode_action_to_ctbr`` so ``tanh(action)`` reproduces the command.

    ``target_rate = tanh(a) * 180 * target_clip`` and
    ``target_thrust_ratio = (tanh(a) + 1) / 2``.
    """
    a_rate = torch.atanh((rate_deg / (180.0 * target_clip)).clamp(-0.995, 0.995))
    ratio = throttle.clamp(min_ratio + 1e-3, max_ratio - 1e-3)
    a_thrust = torch.atanh((2.0 * ratio - 1.0).clamp(-0.995, 0.995))
    return torch.cat([a_rate, a_thrust], dim=-1)


class GuidanceBaseline:
    """Classical guidance law: relative state -> ``(target_pos, target_vel)``.

    Subclasses override :meth:`guidance`; the shared :class:`GeometricCTBR`
    turns the result into the CTBR action. These laws use the full relative
    state (range, bearing and evader velocity), which PN and MPC require.
    """

    def __init__(self, ctbr: GeometricCTBR):
        self.ctbr = ctbr

    def reset(self) -> None:
        self.ctbr.reset()

    def guidance(self, pos, vel, evader_pos, evader_vel):
        return evader_pos, None

    def __call__(self, drone_state: torch.Tensor, evader_pos: torch.Tensor,
                 evader_vel: torch.Tensor) -> torch.Tensor:
        pos = drone_state[..., 0:3]
        vel = drone_state[..., 7:10]
        target_pos, target_vel = self.guidance(pos, vel, evader_pos, evader_vel)
        aim = target_pos - pos
        target_yaw = torch.atan2(aim[..., 1], aim[..., 0])
        return self.ctbr.compute(drone_state, target_pos, target_vel, target_yaw)


# ---------------------------------------------------------------------------
# Guidance laws
# ---------------------------------------------------------------------------
class PurePursuitController(GuidanceBaseline):
    """Pure pursuit: steer straight at the evader's current position."""

    def __init__(self, ctbr: GeometricCTBR, cfg: dict):
        super().__init__(ctbr)


class ProportionalNavigationController(GuidanceBaseline):
    """Proportional navigation, i.e. the constant-bearing collision course.

    PN nulls the line-of-sight rotation rate, which for a constant-velocity
    evader means flying straight at the point where the two paths intersect.
    That point follows in closed form by solving, for the time-to-go ``t``,

        || (evader_pos - pos) + evader_vel * t || = pursuer_speed * t,

    a quadratic whose smallest positive root is the collision time-to-go.
    Aiming at ``evader_pos + evader_vel * t`` is exact for head-on, crossing and
    tail-chase geometries alike (a fixed lead time is not: it over-leads slow
    crossers and points backwards on a closing head-on shot). When no real
    collision course exists (the evader is at least as fast as the pursuer) it
    falls back to a range/speed time-to-go so the lead still points sensibly.
    """

    def __init__(self, ctbr: GeometricCTBR, cfg: dict):
        super().__init__(ctbr)
        # pursuer_speed [m/s]: speed capability used to solve the collision
        # triangle. Match the pursuer's achievable cruise speed.
        self.pursuer_speed = float(cfg.get('pursuer_speed', 2.0))
        # nav_gain: multiplier on the collision time-to-go lead. 1.0 is the
        # exact constant-bearing course; >1 leads more aggressively.
        self.nav_gain = float(cfg.get('nav_gain', 1.0))
        # max_tgo [s]: cap on the lead (bounds extrapolation of a manoeuvring
        # evader).
        self.max_tgo = float(cfg.get('max_tgo', 3.0))

    def guidance(self, pos, vel, evader_pos, evader_vel):
        rel = evader_pos - pos
        rng = rel.norm(dim=-1, keepdim=True).clamp_min(1e-3)

        # Solve || rel + evader_vel * t || = pursuer_speed * t -> a t^2 + b t + c = 0.
        a = (evader_vel * evader_vel).sum(dim=-1, keepdim=True) - self.pursuer_speed ** 2
        b = 2.0 * (rel * evader_vel).sum(dim=-1, keepdim=True)
        c = (rel * rel).sum(dim=-1, keepdim=True)
        disc = b * b - 4.0 * a * c
        sqrt_disc = disc.clamp_min(0.0).sqrt()

        # Guard a~0 (evader speed == pursuer speed) to avoid a div-by-zero.
        a_safe = torch.where(a.abs() < 1e-6, torch.full_like(a, -1e-6), a)
        t1 = (-b - sqrt_disc) / (2.0 * a_safe)
        t2 = (-b + sqrt_disc) / (2.0 * a_safe)
        big = torch.full_like(t1, 1e9)
        t1p = torch.where(t1 > 1e-4, t1, big)
        t2p = torch.where(t2 > 1e-4, t2, big)
        t_root = torch.minimum(t1p, t2p)  # smallest strictly-positive root

        fallback = rng / max(self.pursuer_speed, 1e-3)
        valid = (disc >= 0.0) & (t_root < 1e9)
        t_go = torch.where(valid, t_root, fallback).clamp(max=self.max_tgo)

        lead = evader_pos + evader_vel * (self.nav_gain * t_go)
        return lead, evader_vel


class APFController(GuidanceBaseline):
    """Artificial potential field: steer down the gradient of an attractive well.

    With no obstacles the field is a single attractive term
    ``F_att = attract_gain * (evader_pos - pos)``, the gradient of
    ``0.5*k*||evader_pos - pos||^2``. Far away it saturates at ``max_speed``;
    close in it ramps down, giving the smooth deceleration characteristic of a
    potential well. A feed-forward of the evader velocity lets the field track a
    mover instead of lagging behind it.

    The horizontal position target is the drone's *own* xy, which makes the
    horizontal channel pure velocity control (``a_h = kv * (v_des - v)``) so the
    braking-limited speed profile is actually followed and the pursuer arrests
    onto the target rather than being flung past it by a saturating position
    P-term. The vertical target stays the goal altitude so altitude-hold and its
    integral keep working.
    """

    def __init__(self, ctbr: GeometricCTBR, cfg: dict):
        super().__init__(ctbr)
        # attract_gain [1/s]: strength of the well (velocity per metre of range).
        self.attract_gain = float(cfg.get('attract_gain', 2.0))
        # max_speed [m/s]: cap on the attractive-field velocity magnitude.
        self.max_speed = float(cfg.get('max_speed', 2.0))
        # decel [m/s^2]: usable horizontal braking. The approach speed is capped
        # at sqrt(2*decel*range) so the drone never flies faster than it can
        # brake from -- without this the thrust-limited Crazyflie (~5 m/s^2 at
        # the tilt cap) sails straight past a stationary target and orbits it.
        self.decel = float(cfg.get('decel', 2.0))
        # vel_feedforward: add the evader velocity so the field tracks a mover.
        self.vel_feedforward = bool(cfg.get('vel_feedforward', True))

    def guidance(self, pos, vel, evader_pos, evader_vel):
        rel = evader_pos - pos
        rng = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        los = rel / rng
        v_mag = torch.minimum(self.attract_gain * rng, (2.0 * self.decel * rng).sqrt())
        v_mag = v_mag.clamp(max=self.max_speed)
        v_des = v_mag * los
        if self.vel_feedforward:
            v_des = v_des + evader_vel
        target_pos = torch.cat([pos[..., 0:2], evader_pos[..., 2:3]], dim=-1)
        return target_pos, v_des


# ---------------------------------------------------------------------------
# Sampling-based nonlinear MPC (MPPI)
# ---------------------------------------------------------------------------
def _skew(w: torch.Tensor) -> torch.Tensor:
    """Batched skew-symmetric matrix from a ``[..., 3]`` vector."""
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    zero = torch.zeros_like(wx)
    row0 = torch.stack([zero, -wz, wy], dim=-1)
    row1 = torch.stack([wz, zero, -wx], dim=-1)
    row2 = torch.stack([-wy, wx, zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _orthonormalize(R: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt re-orthonormalization of batched ``[..., 3, 3]`` matrices."""
    b1 = R[..., :, 0]
    b2 = R[..., :, 1]
    b1 = b1 / b1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    b2 = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


class NonlinearMPCController:
    """Sampling-based nonlinear MPC (MPPI) over the full 6-DOF quadrotor model.

    Unlike the kinematic MPC (a point mass that only picks an intercept point),
    this rolls out the rigid-body quadrotor dynamics for many sampled CTBR input
    sequences, scores them against the predicted evader trajectory, and returns
    the importance-weighted optimal first input:

        p_dot = v,  v_dot = thrust_accel * (R e3) + g,  R_dot = R * skew(omega)

    The first input is emitted as the same pre-tanh CTBR action the RL policy
    uses, so the on-board rate PID performs the low-level tracking.

    Deployment note: unlike training, this solves on the CPU inside the 50 Hz
    loop, so ``num_samples`` / ``horizon`` default well below the sim values and
    the per-solve time is logged. If the loop starts overrunning (the inherited
    loop warns), cut ``num_samples`` first.
    """

    def __init__(self, ctbr_params: dict, cfg: dict, dt: float):
        self.g = float(ctbr_params['g'])
        self.hover = float(ctbr_params['hover_throttle'])
        self.target_clip = float(ctbr_params['target_clip'])
        self.min_ratio = float(ctbr_params['min_ratio'])
        self.max_ratio = float(ctbr_params['max_ratio'])

        # The plan is discretised coarser than the control loop for a longer
        # lookahead: horizon * step_mult * dt seconds.
        self.step_mult = int(cfg.get('step_mult', 2))
        self.mpc_dt = float(dt) * self.step_mult
        self.horizon = int(cfg.get('horizon', 20))
        self.num_samples = int(cfg.get('num_samples', 256))
        # temperature: lower -> greedier averaging of the sampled rollouts.
        self.lambda_ = float(cfg.get('temperature', 0.05))
        self.gamma = float(cfg.get('discount', 0.99))

        # Sampling noise std for the body-rate [rad/s] and thrust-accel [m/s^2].
        # Kept tight so samples stay near the (coherent) warm-started nominal
        # instead of exploring the saturated envelope, which makes the command
        # chatter.
        self.noise_rate = float(cfg.get('noise_rate', 0.5))
        self.noise_thrust = float(cfg.get('noise_thrust', 1.0))

        # First-order actuator lag modelled inside the rollout. The pursuer runs
        # a rate PID over motor-lagged rotors (tau_up/down ~0.08-0.17 s), so
        # body rate and thrust cannot change instantly. A lag-free rollout plans
        # violent reorientations the real drone overshoots (tilting past 60 deg,
        # losing lift, hitting the floor). alpha = mpc_dt / tau, clamped to
        # (0, 1]; alpha = 1 recovers no lag.
        self.rate_tau = float(cfg.get('rate_tau', 0.10))
        self.thrust_tau = float(cfg.get('thrust_tau', 0.12))
        self.alpha_rate = min(max(self.mpc_dt / max(self.rate_tau, 1e-3), 0.0), 1.0)
        self.alpha_thrust = min(max(self.mpc_dt / max(self.thrust_tau, 1e-3), 0.0), 1.0)
        # Unlike in sim (re-solved every 0.01 s sim step against a 0.02 s plan
        # step), here the controller is called exactly once per control period,
        # so shifting the warm start by one plan step per call is correct
        # whenever step_mult == 1. With step_mult > 1 the plan advances slower
        # than real time, so the default is off.
        self.warm_shift = bool(cfg.get('warm_shift', self.step_mult == 1))
        # Moving-average window (in horizon steps) low-passing the sampling
        # noise; >1 enables smooth-MPPI, 1 gives independent per-step noise.
        self.noise_smooth = int(cfg.get('noise_smooth', 8))

        # Cost weights. On a thrust-to-weight ~1.4 airframe a plain distance
        # cost drives the drone to dive at the target and hit the floor, so
        # altitude keeping is weighted heavily; w_rate + w_smooth keep the
        # body-rate command gentle and coherent.
        self.w_dist = float(cfg.get('w_dist', 2.0))
        self.w_term = float(cfg.get('w_terminal', 10.0))
        self.w_rate = float(cfg.get('w_rate', 0.3))
        # Penalty on step-to-step change of the commanded body rate: without it
        # the sampler picks a fresh near-saturated rate every step, chattering
        # the attitude and bleeding altitude.
        self.w_smooth = float(cfg.get('w_smooth', 2.0))
        self.w_tilt = float(cfg.get('w_tilt', 3.0))
        self.w_ground = float(cfg.get('w_ground', 150.0))
        self.min_altitude = float(cfg.get('min_altitude', 0.4))
        # Stiff barrier on the tilt *angle*: body rates are bounded but they
        # integrate into arbitrarily large tilts, and past ~30 deg this airframe
        # cannot hold altitude at full thrust, so the rollout would happily plan
        # a lift-losing dive. The lag model above makes the rollout tilt track
        # the real tilt, so this bounds the real one.
        self.tilt_max = math.radians(float(cfg.get('tilt_max', 32.0)))
        self.cos_tilt_max = math.cos(self.tilt_max)
        self.w_tilt_barrier = float(cfg.get('w_tilt_barrier', 3000.0))
        # Penalise sitting below the target's altitude anywhere in the rollout:
        # the pursuer must not trade height away to close horizontally.
        self.w_alt = float(cfg.get('w_alt', 8.0))
        # Reward for line-of-sight closing speed. A short-horizon terminal
        # distance cost is myopic for far targets (over ~0.8 s a hard bank
        # barely changes the end distance, so the sampler stays timid);
        # rewarding approach speed makes MPPI commit to the chase. It is capped
        # at both max_closing and sqrt(2*closing_decel*range), so in the last
        # half-metre the plan brakes onto the target instead of orbiting it.
        self.w_closing = float(cfg.get('w_closing', 30.0))
        self.max_closing = float(cfg.get('max_closing', 6.0))
        self.closing_decel = float(cfg.get('closing_decel', 50.0))

        # Input bounds from the CTBR envelope. The thrust-accel floor sits near
        # hover (not near zero) so no sample can command a lift-collapsing
        # throttle while banked.
        self.w_max = math.pi * self.target_clip  # 180*target_clip deg/s
        self.ta_min = float(cfg.get('ta_min_frac', 0.6)) * self.g
        self.ta_max = self.g * (self.max_ratio / self.hover) ** 2

        self._nominal: Optional[torch.Tensor] = None  # [1, H, 4]

    def reset(self) -> None:
        self._nominal = None

    def _hover_nominal(self, device) -> torch.Tensor:
        nominal = torch.zeros(1, self.horizon, 4, device=device)
        nominal[..., 3] = self.g  # hover thrust accel
        return nominal

    def __call__(self, drone_state: torch.Tensor, evader_pos: torch.Tensor,
                 evader_vel: torch.Tensor) -> torch.Tensor:
        device = drone_state.device
        n = drone_state.shape[0]
        K, H = self.num_samples, self.horizon

        pos = drone_state[..., 0:3]
        quat = drone_state[..., 3:7]
        vel = drone_state[..., 7:10]
        R0 = ic.quaternion_to_rotation_matrix(quat)  # [N, 3, 3]

        if self._nominal is None or self._nominal.shape[0] != n:
            self._nominal = self._hover_nominal(device).expand(n, H, 4).contiguous()

        g_vec = torch.tensor([0.0, 0.0, -self.g], device=device)

        # Sample K control sequences around the warm-started nominal.
        noise = torch.randn(n, K, H, 4, device=device)
        if self.noise_smooth > 1:
            # Smooth-MPPI: low-pass the noise along the horizon so each sampled
            # control is a coherent manoeuvre rather than chatter.
            w = self.noise_smooth
            pad = w // 2
            flat = noise.permute(0, 1, 3, 2).reshape(n * K * 4, 1, H)
            kernel = torch.ones(1, 1, w, device=device) / w
            flat = torch.nn.functional.conv1d(flat, kernel, padding=pad)[..., :H]
            noise = flat.reshape(n, K, 4, H).permute(0, 1, 3, 2).contiguous()
            noise = noise * math.sqrt(w)  # restore variance lost to averaging
        noise[..., 0:3] *= self.noise_rate
        noise[..., 3] *= self.noise_thrust
        controls = self._nominal.unsqueeze(1) + noise  # [N, K, H, 4]
        controls[..., 0:3] = controls[..., 0:3].clamp(-self.w_max, self.w_max)
        controls[..., 3] = controls[..., 3].clamp(self.ta_min, self.ta_max)

        # Batched rollout of the full quadrotor dynamics.
        p = pos.unsqueeze(1).expand(n, K, 3).contiguous()
        v = vel.unsqueeze(1).expand(n, K, 3).contiguous()
        R = R0.unsqueeze(1).expand(n, K, 3, 3).contiguous()
        eye = torch.eye(3, device=device).expand(n, K, 3, 3)

        # Effective (lagged) inputs start from what the drone is actually doing.
        # ``DroneState.ang_vel`` is already body-frame (the firmware's gyro),
        # unlike the sim's world-frame drone_state slice, so no rotation here.
        omega0 = drone_state[..., 10:13]
        omega_eff = omega0.unsqueeze(1).expand(n, K, 3).contiguous()
        ta_eff = torch.full((n, K, 1), self.g, device=device)
        prev_omega_cmd = omega0.unsqueeze(1).expand(n, K, 3)

        cost = torch.zeros(n, K, device=device)
        dist2 = torch.zeros(n, K, device=device)
        for h in range(H):
            omega_cmd = controls[:, :, h, 0:3]
            ta_cmd = controls[:, :, h, 3:4]
            if self.w_smooth > 0.0:
                cost = cost + self.w_smooth * ((omega_cmd - prev_omega_cmd) ** 2).sum(dim=-1)
                prev_omega_cmd = omega_cmd
            # First-order lag: inputs approach the command, they do not jump.
            omega_eff = omega_eff + (omega_cmd - omega_eff) * self.alpha_rate
            ta_eff = ta_eff + (ta_cmd - ta_eff) * self.alpha_thrust
            omega = omega_eff
            b3 = R[..., :, 2]  # [N, K, 3]

            acc = ta_eff * b3 + g_vec
            v = v + acc * self.mpc_dt
            p = p + v * self.mpc_dt
            R = _orthonormalize(torch.matmul(R, eye + _skew(omega) * self.mpc_dt))

            t = (h + 1) * self.mpc_dt
            ev_pred = (evader_pos + evader_vel * t).unsqueeze(1)  # [N, 1, 3]
            rel = ev_pred - p
            dist2 = (rel ** 2).sum(dim=-1)  # [N, K]

            disc = self.gamma ** h
            cost = cost + disc * self.w_dist * dist2
            if self.w_closing > 0.0:
                rng = dist2.clamp_min(1e-4).sqrt()
                closing = (v * rel).sum(dim=-1) / rng  # >0 when approaching
                cap = (2.0 * self.closing_decel * rng).sqrt().clamp(max=self.max_closing)
                cost = cost - disc * self.w_closing * torch.minimum(closing, cap)
            cost = cost + self.w_tilt * (1.0 - b3[..., 2]).clamp_min(0.0)
            cost = cost + self.w_tilt_barrier * (self.cos_tilt_max - b3[..., 2]).clamp_min(0.0)
            cost = cost + self.w_rate * (omega ** 2).sum(dim=-1)
            cost = cost + self.w_ground * (self.min_altitude - p[..., 2]).clamp_min(0.0)
            if self.w_alt > 0.0:
                cost = cost + self.w_alt * (ev_pred[..., 2] - p[..., 2]).clamp_min(0.0)

        cost = cost + self.w_term * dist2  # terminal distance

        # MPPI importance weights and nominal update.
        beta = cost.min(dim=1, keepdim=True).values
        weights = torch.exp(-(cost - beta) / self.lambda_)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        self._nominal = (weights.unsqueeze(-1).unsqueeze(-1) * controls).sum(dim=1)

        u0 = self._nominal[:, 0, :]  # [N, 4]

        if self.warm_shift:
            self._nominal = torch.roll(self._nominal, shifts=-1, dims=1)
            self._nominal[:, -1, :] = self._nominal[:, -2, :]

        rate_deg = u0[..., 0:3] * (180.0 / math.pi)
        throttle = self.hover * torch.sqrt((u0[..., 3:4] / self.g).clamp(0.04, 4.0))
        return pack_ctbr_action(
            rate_deg, throttle, self.target_clip, self.min_ratio, self.max_ratio
        )


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
def _load_traditional_config(config_path: str) -> dict:
    config = _read_yaml(config_path)
    section = config.get('traditional', {}) or {}
    if not isinstance(section, dict):
        raise ValueError("Config section 'traditional' must be a mapping.")
    return section


def _subsection(section: dict, key: str) -> dict:
    value = section.get(key, {}) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config key 'traditional.{key}' must be a mapping.")
    return value


class TraditionalController(InterceptController):
    """Fly a classical guidance law through the policy controller's flight loop."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.method = str(args.method).strip().lower()
        if self.method not in METHODS:
            raise ValueError(
                f"Unknown method '{self.method}'. Choose one of {list(METHODS)}."
            )
        super().__init__(args)
        self._solve_time_sum = 0.0
        self._solve_time_max = 0.0
        self._solve_count = 0

    # -- policy hook ---------------------------------------------------------
    def _setup_policy(self) -> None:
        self.device = torch.device('cpu')
        self.policy = None
        section = _load_traditional_config(self.config_path)
        ctbr_cfg = _subsection(section, 'ctbr')

        # CTBR envelope. Default to the exported policy's metadata when an
        # artifact is available so the baseline and the policy share the exact
        # same command scaling (that is what makes the comparison fair); fall
        # back to the Crazyflie defaults otherwise.
        ctbr = ic.CTBRConfig()
        artifact_metadata = self._try_load_artifact_metadata()
        if artifact_metadata is not None:
            ctbr = artifact_metadata.ctbr
        self.metadata = ic.PolicyMetadata(
            artifact_version=ic.ARTIFACT_VERSION,
            algo=self.method,
            ctbr=ic.CTBRConfig(
                target_clip=float(ctbr_cfg.get('target_clip', ctbr.target_clip)),
                min_thrust_ratio=float(
                    ctbr_cfg.get('min_thrust_ratio', ctbr.min_thrust_ratio)
                ),
                max_thrust_ratio=float(
                    ctbr_cfg.get('max_thrust_ratio', ctbr.max_thrust_ratio)
                ),
                dt=float(ctbr_cfg.get('dt', ctbr.dt)),
            ),
        )
        self._traditional_config = section
        self._ctbr_config = ctbr_cfg

    def _try_load_artifact_metadata(self) -> Optional[ic.PolicyMetadata]:
        artifact_dir = os.path.abspath(os.path.expanduser(self.artifact_dir or ''))
        _, meta_path = ic.artifact_paths(artifact_dir)
        if not os.path.isfile(meta_path):
            return None
        try:
            return ic.load_metadata(meta_path)
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.warning(
                '[traditional] Ignoring unreadable policy metadata at %s (%s); '
                'using the configured CTBR envelope instead.', meta_path, exc
            )
            return None

    def _build_controller(self) -> None:
        """Instantiate the guidance law. Called once ``control_dt`` is known."""
        section = self._traditional_config
        ctbr_cfg = self._ctbr_config
        cfg = self.metadata.ctbr

        # hover_throttle is the single calibration that matters on hardware: the
        # normalised collective thrust (thrust_pwm / 65536) at which this drone
        # holds altitude. The default is the sim value for the Crazyflie rotor
        # model, sqrt(m*g / (4*max_thrust_per_rotor)) with m=0.034 kg and
        # 0.12 N/rotor, i.e. a thrust-to-weight of ~1.44. Measure it on the real
        # airframe (hover PWM / 65536) and override it -- too low and the drone
        # climbs away, too high and the altitude integral has to fight a
        # permanent deficit.
        hover_throttle = float(ctbr_cfg.get('hover_throttle', 0.8336))
        gravity = float(ctbr_cfg.get('gravity', 9.81))
        self.ctbr_params = {
            'g': gravity,
            'hover_throttle': hover_throttle,
            'target_clip': cfg.target_clip,
            'min_ratio': cfg.min_thrust_ratio,
            'max_ratio': cfg.max_thrust_ratio,
        }

        if self.method == 'nonlinear_mpc':
            self.controller = NonlinearMPCController(
                self.ctbr_params, _subsection(section, 'nonlinear_mpc'),
                dt=self.control_dt,
            )
        else:
            # Gains tuned for the thrust-limited Crazyflie: a deliberately
            # gentle attitude loop (k_att, tilt) so the low-authority,
            # motor-lagged airframe does not tumble, and a well-damped position
            # loop (kv >= kp) for an overshoot-free approach.
            geometric = GeometricCTBR(
                g=gravity,
                hover_throttle=hover_throttle,
                target_clip=cfg.target_clip,
                min_ratio=cfg.min_thrust_ratio,
                max_ratio=cfg.max_thrust_ratio,
                dt=self.control_dt,
                kp=float(ctbr_cfg.get('kp', 8.0)),
                kv=float(ctbr_cfg.get('kv', 7.0)),
                k_att=float(ctbr_cfg.get('k_att', 6.0)),
                k_yaw=float(ctbr_cfg.get('k_yaw', 1.0)),
                max_tilt_deg=float(ctbr_cfg.get('max_tilt_deg', 28.0)),
                ki_z=float(ctbr_cfg.get('ki_z', 10.0)),
                i_limit_z=float(ctbr_cfg.get('i_limit_z', 1.0)),
                i_speed_gate=float(ctbr_cfg.get('i_speed_gate', 1.5)),
            )
            factories = {
                'pure_pursuit': PurePursuitController,
                'pn': ProportionalNavigationController,
                'apf': APFController,
            }
            self.controller = factories[self.method](
                geometric, _subsection(section, self.method)
            )

        self.controller.reset()
        logger.info(
            '[traditional] Using %s guidance at %.1f Hz '
            '(hover_throttle=%.3f, target_clip=%.2f, thrust_ratio=[%.2f, %.2f]).',
            self.method, 1.0 / self.control_dt, hover_throttle,
            cfg.target_clip, cfg.min_thrust_ratio, cfg.max_thrust_ratio,
        )

    # -- per-step hook -------------------------------------------------------
    def _drone_state_tensor(self, state: ic.DroneState) -> torch.Tensor:
        """Pack a :class:`ic.DroneState` into the sim's ``drone_state`` layout.

        ``[pos(3), quat_wxyz(4), lin_vel_world(3), ang_vel_body(3)]`` -- the
        slices 0:3 / 3:7 / 7:10 / 10:13 the guidance laws index.
        """
        return torch.as_tensor(
            np.concatenate([state.pos, state.quat_wxyz, state.lin_vel, state.ang_vel]),
            dtype=torch.float32, device=self.device,
        ).reshape(1, 13)

    def _compute_action(self, pursuer_state: ic.DroneState,
                        evader_state: ic.DroneState) -> ic.CTBRCommand:
        drone_state = self._drone_state_tensor(pursuer_state)
        evader_pos = torch.as_tensor(
            evader_state.pos, dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        evader_vel = torch.as_tensor(
            evader_state.lin_vel, dtype=torch.float32, device=self.device
        ).reshape(1, 3)

        solve_start = time.perf_counter()
        with torch.no_grad():
            raw_action = self.controller(drone_state, evader_pos, evader_vel)
        self._record_solve_time(time.perf_counter() - solve_start)

        self.previous_action = torch.tanh(raw_action).squeeze(0).detach()
        return ic.decode_action_to_ctbr(raw_action, self.metadata.ctbr)

    def _record_solve_time(self, elapsed: float) -> None:
        """Track guidance solve time; MPPI is the one law that can blow the budget."""
        self._solve_time_sum += elapsed
        self._solve_time_max = max(self._solve_time_max, elapsed)
        self._solve_count += 1
        if self._solve_count % 250 == 0:
            logger.info(
                '[traditional] %s solve time: mean %.2f ms, max %.2f ms '
                '(budget %.1f ms).',
                self.method, 1e3 * self._solve_time_sum / self._solve_count,
                1e3 * self._solve_time_max, 1e3 * self.control_dt,
            )

    def run(self) -> None:
        # control_dt is resolved by the base __init__, so the guidance law (whose
        # integrator / MPC discretisation depends on it) is built just before
        # the loop starts rather than inside _setup_policy.
        self._build_controller()
        super().run()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description='Run a classical intercept baseline (pure pursuit, PN, APF '
                    'or nonlinear MPC) on the Crazyflie.'
    )
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG_PATH,
        help=f'Path to the YAML configuration file (default: {DEFAULT_CONFIG_PATH}).',
    )
    parser.add_argument(
        '--method',
        default=None,
        choices=list(METHODS),
        help="Guidance law to fly (default: the config's traditional.method, "
             "else pure_pursuit).",
    )
    args = parser.parse_args(argv)

    config_path = os.path.abspath(os.path.expanduser(args.config))
    section = _load_traditional_config(config_path)
    method = args.method or str(section.get('method', 'pure_pursuit'))

    # The artifact is optional here: it is only read to inherit the policy's
    # CTBR envelope so both controllers command through the same scaling.
    controller_args = argparse.Namespace(
        config=config_path,
        artifact_dir=str(_read_yaml(config_path).get('artifact_dir', '') or ''),
        method=method,
    )
    TraditionalController(controller_args).run()


if __name__ == '__main__':
    main()
