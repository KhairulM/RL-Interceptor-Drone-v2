from __future__ import annotations

import math

import torch

from omni_drones.utils.torch import quaternion_to_rotation_matrix


class GeometricCTBR:
    """SE(3)-style outer loop: a position/velocity target -> CTBR action.

    Produces the pre-tanh ``[body_rate(3), thrust(1)]`` action consumed by the
    task's ``PIDrate`` action transform, so the *proven* on-board rate PID
    (``PIDRateController``, the same stack the RL policy uses) handles low-level
    stabilization. The plain ``LeePositionController`` is only marginally stable
    on the perturbed, motor-lagged Crazyflie pursuer, so it is not used here.

    Tunables (safe to adjust):
        kp, kv       : outer position / velocity P gains.
        k_att        : attitude (thrust-vector) alignment gain -> body rates.
        k_yaw        : yaw-alignment gain.
        max_tilt_deg : clamp on the commanded tilt (limits altitude loss).
    """

    def __init__(
        self,
        mass: float,
        g: float,
        hover_throttle: float,
        target_clip: float,
        min_ratio: float,
        max_ratio: float,
        kp: float = 6.0,
        kv: float = 4.0,
        k_att: float = 10.0,
        k_yaw: float = 2.0,
        max_tilt_deg: float = 35.0,
        ki_z: float = 1.5,
        i_limit_z: float = 1.5,
        i_speed_gate: float = 1.5,
    ):
        self.mass = float(mass)
        self.g = float(g)
        self.hover = float(hover_throttle)
        self.target_clip = float(target_clip)
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)
        self.kp = float(kp)
        self.kv = float(kv)
        self.k_att = float(k_att)
        self.k_yaw = float(k_yaw)
        self.max_tilt = math.radians(float(max_tilt_deg))
        # Integral action on the altitude channel. The pursuer's true
        # thrust-to-weight is randomised per episode (t2w_scale), so the single
        # nominal hover throttle is wrong by a fixed amount and a pure P/D
        # altitude loop settles ~0.2 m below the target (error = deficit/kp) --
        # enough to sit just outside the 0.1 m success radius forever. The
        # integral nulls that steady-state offset; i_limit_z is the anti-windup
        # clamp on the accumulated error [m*s].
        self.ki_z = float(ki_z)
        self.i_limit_z = float(i_limit_z)
        # The integral only accumulates while the drone is slow (settling onto the
        # target), which is exactly when the steady-state offset shows. During a
        # fast approach it stays frozen -- this keeps it from winding up on the
        # velocity-command waypoint of the APF baseline (whose target_pos leads
        # the drone rather than marking the goal) on moving targets.
        self.i_speed_gate = float(i_speed_gate)
        self.integ_z = None  # lazily-sized accumulator, reset per episode

    def reset(self, batch_size: int, device) -> None:
        # Sized lazily on the first compute() to match the drone-state layout.
        self.integ_z = None

    def compute(self, drone_state, target_pos, target_vel, target_yaw, done=None):
        pos = drone_state[..., 0:3]
        quat = drone_state[..., 3:7]
        vel = drone_state[..., 7:10]
        R = quaternion_to_rotation_matrix(quat)  # world_from_body [..., 3, 3]

        if target_vel is None:
            target_vel = torch.zeros_like(vel)

        # Vertical integral term: accumulate the altitude error and reset it for
        # any environment that has just been reset (done), with anti-windup.
        z_err = target_pos[..., 2:3] - pos[..., 2:3]  # + when drone is below target
        if self.integ_z is None or self.integ_z.shape != z_err.shape:
            self.integ_z = torch.zeros_like(z_err)
        if done is not None:
            done_any = done.reshape(z_err.shape[0], -1).any(dim=-1)  # [N]
            keep = (~done_any).to(z_err.dtype).reshape(z_err.shape[0], *([1] * (z_err.ndim - 1)))
            self.integ_z = self.integ_z * keep
        gate = (vel.norm(dim=-1, keepdim=True) < self.i_speed_gate).to(z_err.dtype)
        self.integ_z = (self.integ_z + z_err * 0.01 * gate).clamp(-self.i_limit_z, self.i_limit_z)

        # Altitude-priority thrust allocation. On a thrust-limited quad (the
        # Crazyflie pursuer has a thrust-to-weight ratio of only ~1.4) a fixed
        # tilt limit lets the horizontal chase steal the lift needed to hold
        # altitude, so the drone sinks to the ground a fraction of a metre below
        # a target it has otherwise reached. Instead we first reserve the
        # *vertical* specific thrust required to track the target's altitude,
        # then spend only the remaining thrust envelope on horizontal accel.
        # a_max is the max specific thrust: throttle=max_ratio -> f=g*(max_ratio/hover)^2.
        a_max = self.g * (self.max_ratio / self.hover) ** 2

        # Vertical channel gets first claim on the thrust budget (with gravity
        # compensation), kept strictly positive and just under the envelope so a
        # little headroom is always left for attitude control.
        az = (-self.kp * (pos[..., 2:3] - target_pos[..., 2:3])
              - self.kv * (vel[..., 2:3] - target_vel[..., 2:3])
              + self.ki_z * self.integ_z + self.g)
        az = az.clamp(0.3 * self.g, 0.98 * a_max)

        # Horizontal accel is limited by whatever thrust is left over, not by a
        # fixed tilt: h_max = sqrt(a_max^2 - az^2). This also honours the tilt
        # cap so altitude-hold near hover still behaves gently.
        headroom = (a_max ** 2 - az ** 2).clamp_min(0.0).sqrt()
        max_horiz = torch.minimum(headroom, math.tan(self.max_tilt) * az)
        horiz = -self.kp * (pos[..., 0:2] - target_pos[..., 0:2]) \
            - self.kv * (vel[..., 0:2] - target_vel[..., 0:2])
        hn = horiz.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        horiz = horiz * (max_horiz / hn).clamp(max=1.0)
        a_des = torch.cat([horiz, az], dim=-1)

        b3 = R[..., :, 2]  # current body-z in world
        # Collective thrust to realise |a_des| along the (near-vertical) thrust
        # axis; thrust ~ throttle^2. Use the magnitude so the reserved vertical
        # accel is delivered even as the body tilts to produce the horizontal.
        f_acc = a_des.norm(dim=-1, keepdim=True).clamp_min(0.1 * self.g)
        throttle = self.hover * torch.sqrt((f_acc / self.g).clamp(0.04, 4.0))

        # Body rates to align body-z with the desired thrust direction.
        b3_des = a_des / a_des.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        e_R = torch.cross(b3, b3_des, dim=-1)  # world-frame rotation error
        omega_world = self.k_att * e_R
        omega_body = torch.einsum("...ji,...j->...i", R, omega_world)  # R^T @ omega

        yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
        yaw_err = torch.atan2(torch.sin(target_yaw - yaw), torch.cos(target_yaw - yaw))
        wz = self.k_yaw * yaw_err

        body_rate = torch.stack([omega_body[..., 0], omega_body[..., 1], wz], dim=-1)
        rate_deg = body_rate * (180.0 / math.pi)

        # Invert the PIDrate transform scaling so tanh(action) reproduces the
        # desired rate/thrust: target_rate = tanh(a)*180*target_clip,
        # target_thrust_ratio = (tanh(a)+1)/2.
        a_rate = torch.atanh((rate_deg / (180.0 * self.target_clip)).clamp(-0.995, 0.995))
        ratio = throttle.clamp(self.min_ratio + 1e-3, self.max_ratio - 1e-3)
        a_thrust = torch.atanh((2.0 * ratio - 1.0).clamp(-0.995, 0.995))
        return torch.cat([a_rate, a_thrust], dim=-1)


class GuidanceBaseline:
    """Classical guidance baseline.

    Subclasses override `guidance` to return a ``(target_pos, target_vel)`` for
    the pursuer; the shared :class:`GeometricCTBR` turns it into a CTBR action.

    Note: these guidance laws use the full relative state (range + bearing +
    evader velocity), which PN/MPC require. The RL policy's observation is
    configurable (bearing-only, or range via ``use_evader_rel_dist``); both run
    on the same env and flight pipeline, so the comparison stays fair.
    """

    def __init__(self, ctbr: GeometricCTBR):
        self.ctbr = ctbr

    def reset(self, batch_size: int, device) -> None:
        self.ctbr.reset(batch_size, device)

    def guidance(self, pos, vel, evader_pos, evader_vel, done):
        # returns (target_pos, target_vel|None); default chases the evader.
        return evader_pos, None

    def __call__(self, drone_state, evader_pos, evader_vel, done):
        pos = drone_state[..., 0:3]
        vel = drone_state[..., 7:10]
        target_pos, target_vel = self.guidance(pos, vel, evader_pos, evader_vel, done)
        aim = target_pos - pos
        target_yaw = torch.atan2(aim[..., 1], aim[..., 0])
        return self.ctbr.compute(drone_state, target_pos, target_vel, target_yaw, done=done)
