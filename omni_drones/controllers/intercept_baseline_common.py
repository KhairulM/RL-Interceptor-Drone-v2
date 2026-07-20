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

    def compute(self, drone_state, target_pos, target_vel, target_yaw):
        pos = drone_state[..., 0:3]
        quat = drone_state[..., 3:7]
        vel = drone_state[..., 7:10]
        R = quaternion_to_rotation_matrix(quat)  # world_from_body [..., 3, 3]

        e3 = torch.tensor([0.0, 0.0, 1.0], device=drone_state.device)
        if target_vel is None:
            target_vel = torch.zeros_like(vel)

        # Desired specific thrust (world), with gravity compensation.
        a_des = -self.kp * (pos - target_pos) - self.kv * (vel - target_vel) + self.g * e3

        # Clamp the horizontal component so the commanded tilt stays bounded
        # (a saturated tilt is what makes a position controller lose altitude).
        az = a_des[..., 2:3].clamp_min(0.5 * self.g)
        horiz = a_des[..., 0:2]
        max_horiz = math.tan(self.max_tilt) * az
        hn = horiz.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        horiz = horiz * (max_horiz / hn).clamp(max=1.0)
        a_des = torch.cat([horiz, az], dim=-1)

        b3 = R[..., :, 2]  # current body-z in world
        # Collective thrust from accel projected on body-z; thrust ~ throttle^2.
        f_acc = (a_des * b3).sum(dim=-1, keepdim=True).clamp_min(0.1 * self.g)
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
    evader velocity), which PN/MPC require -- unlike the RL policy, which is
    trained on bearing-only observations.
    """

    def __init__(self, ctbr: GeometricCTBR):
        self.ctbr = ctbr

    def reset(self, batch_size: int, device) -> None:
        return None

    def guidance(self, pos, vel, evader_pos, evader_vel, done):
        # returns (target_pos, target_vel|None); default chases the evader.
        return evader_pos, None

    def __call__(self, drone_state, evader_pos, evader_vel, done):
        pos = drone_state[..., 0:3]
        vel = drone_state[..., 7:10]
        target_pos, target_vel = self.guidance(pos, vel, evader_pos, evader_vel, done)
        aim = target_pos - pos
        target_yaw = torch.atan2(aim[..., 1], aim[..., 0])
        return self.ctbr.compute(drone_state, target_pos, target_vel, target_yaw)
