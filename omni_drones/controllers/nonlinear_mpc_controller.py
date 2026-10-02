from __future__ import annotations

import math
import os.path as osp

import torch
import yaml

from omni_drones.utils.torch import quaternion_to_rotation_matrix


def _skew(w: torch.Tensor) -> torch.Tensor:
    """Batched skew-symmetric matrix from a [..., 3] vector."""
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    zero = torch.zeros_like(wx)
    row0 = torch.stack([zero, -wz, wy], dim=-1)
    row1 = torch.stack([wz, zero, -wx], dim=-1)
    row2 = torch.stack([-wy, wx, zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _orthonormalize(R: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt re-orthonormalization of batched [..., 3, 3] matrices."""
    b1 = R[..., :, 0]
    b2 = R[..., :, 1]
    b1 = b1 / b1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    b2 = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


class NonlinearMPCController:
    """Sampling-based nonlinear MPC (MPPI) over the full 6-DOF quadrotor model.

    Unlike the kinematic MPC (which reasons about a point mass and only picks an
    intercept point), this controller rolls out the full rigid-body quadrotor
    dynamics for many sampled collective-thrust + body-rate (CTBR) input
    sequences, scores them against the predicted evader trajectory, and returns
    the importance-weighted optimal first input. The dynamics model driven here
    is the standard CTBR quadrotor model:

        p_dot = v
        v_dot = thrust_accel * (R e3) + g_vec
        R_dot = R * skew(omega)

    with inputs ``omega`` (body rates) and ``thrust_accel`` (collective thrust /
    mass). The commanded first input is emitted as the same pre-tanh CTBR action
    the RL policy uses, so the proven on-board rate PID performs the low-level
    tracking.

    Tunables live in ``cfg/nonlinear_mpc_intercept.yaml`` (horizon, samples,
    temperature, sampling noise, cost weights).
    """

    def __init__(
        self,
        ctbr_params,
        task_cfg,
        dt: float,
        config_name: str = "nonlinear_mpc_intercept",
    ):
        cfg_path = osp.join(osp.dirname(__file__), "cfg", f"{config_name}.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        self.g = float(ctbr_params["g"])
        self.hover = float(ctbr_params["hover_throttle"])
        self.target_clip = float(ctbr_params["target_clip"])
        self.min_ratio = float(ctbr_params["min_ratio"])
        self.max_ratio = float(ctbr_params["max_ratio"])

        # MPC horizon uses a coarser step than the sim for a longer lookahead.
        self.step_mult = int(cfg.get("step_mult", 4))
        self.mpc_dt = float(dt) * self.step_mult
        self.horizon = int(cfg.get("horizon", 12))
        self.num_samples = int(cfg.get("num_samples", 128))
        # temperature: lower -> greedier averaging of the sampled rollouts.
        self.lambda_ = float(cfg.get("temperature", 0.2))
        self.gamma = float(cfg.get("discount", 0.98))

        # Sampling noise std for the body-rate [rad/s] and thrust-accel [m/s^2].
        self.noise_rate = float(cfg.get("noise_rate", 1.2))
        self.noise_thrust = float(cfg.get("noise_thrust", 2.0))

        # First-order actuator lag modelled inside the rollout. The pursuer runs
        # a rate PID over motor-lagged rotors (tau_up/down ~0.08-0.17 s), so body
        # rate and thrust cannot change instantly. A lag-free rollout would plan
        # violent reorientations the real drone overshoots (tilting to 60-100 deg
        # and losing lift -> ground crash). Filtering the commanded inputs toward
        # the state with these time constants makes the planned trajectories
        # executable, so the tilt barrier actually bounds the *real* tilt.
        # alpha = mpc_dt / tau, clamped to (0, 1]; alpha=1 recovers no lag.
        self.rate_tau = float(cfg.get("rate_tau", 0.10))
        self.thrust_tau = float(cfg.get("thrust_tau", 0.12))
        self.warm_shift = bool(cfg.get("warm_shift", False))
        # Moving-average window (in horizon steps) used to low-pass the sampling
        # noise; >1 enables smooth-MPPI. 1 disables it (independent per-step noise).
        self.noise_smooth = int(cfg.get("noise_smooth", 1))
        self.alpha_rate = min(max(self.mpc_dt / max(self.rate_tau, 1e-3), 0.0), 1.0)
        self.alpha_thrust = min(max(self.mpc_dt / max(self.thrust_tau, 1e-3), 0.0), 1.0)

        # Cost weights.
        self.w_dist = float(cfg.get("w_dist", 1.0))
        self.w_term = float(cfg.get("w_terminal", 6.0))
        self.w_rate = float(cfg.get("w_rate", 0.01))
        # Penalty on step-to-step change of the commanded body rate. Without it
        # the sampler picks a fresh near-saturated rate every step, producing a
        # chattering command that swings the attitude and bleeds altitude; this
        # term makes MPPI prefer a coherent, smooth reorientation.
        self.w_smooth = float(cfg.get("w_smooth", 0.0))
        self.w_tilt = float(cfg.get("w_tilt", 2.0))
        self.w_ground = float(cfg.get("w_ground", 50.0))
        self.min_altitude = float(cfg.get("min_altitude", 0.2))
        # Stiff barrier on the tilt *angle*. Body rates alone are bounded, but
        # they integrate into arbitrarily large tilts, and on a thrust-limited
        # airframe (T/W ~1.4) any tilt past ~30 deg cannot hold altitude at full
        # thrust, so the rollout happily plans a lift-losing dive. This quadratic
        # barrier past cos(tilt_max) keeps the sampled trajectories inside the
        # envelope where the (lag-free) rollout still matches the real drone.
        self.tilt_max = math.radians(float(cfg.get("tilt_max", 30.0)))
        self.cos_tilt_max = math.cos(self.tilt_max)
        self.w_tilt_barrier = float(cfg.get("w_tilt_barrier", 200.0))

        # Cost on staying below the target's altitude: the pursuer must not trade
        # its height away to close horizontally (that is exactly what craters it),
        # so descending below the evader's height is penalised throughout the
        # rollout, not only near the ground.
        self.w_alt = float(cfg.get("w_alt", 0.0))

        # Reward for closing velocity along the line of sight. A short-horizon
        # terminal-distance cost is myopic for far targets (over ~0.6 s a hard
        # bank barely changes the end distance, so the sampler stays timid);
        # rewarding approach speed makes MPPI commit to accelerating toward the
        # evader. Capped at max_closing so it does not chase unbounded speed.
        self.w_closing = float(cfg.get("w_closing", 0.0))
        self.max_closing = float(cfg.get("max_closing", 6.0))
        # Brake-limited closing: the rewarded closing speed is also capped at
        # sqrt(2*decel*range) so that near the target the reward for approaching
        # fast vanishes and the plan decelerates onto it instead of overshooting
        # and orbiting (the thrust-limited drone cannot brake from full speed in
        # the last half-metre). Matches the APF baseline's stopping profile.
        self.closing_decel = float(cfg.get("closing_decel", 3.0))

        # Input bounds derived from the CTBR envelope. The thrust-accel floor is
        # kept near hover (not near zero) so no sampled input can command a
        # lift-collapsing throttle while the drone is banked.
        self.w_max = math.pi * self.target_clip  # 180*target_clip deg/s
        self.ta_min = float(cfg.get("ta_min_frac", 0.1)) * self.g
        self.ta_max = self.g * (self.max_ratio / self.hover) ** 2

        self._nominal = None  # [N, H, 4]: (wx, wy, wz, thrust_accel)

    def reset(self, batch_size: int, device) -> None:
        self._nominal = None

    def _hover_nominal(self, n: int, device) -> torch.Tensor:
        nominal = torch.zeros(n, self.horizon, 4, device=device)
        nominal[..., 3] = self.g  # hover thrust accel
        return nominal

    def __call__(self, drone_state, evader_pos, evader_vel, done):
        state = drone_state.squeeze(-2) if drone_state.ndim == 3 else drone_state
        ev_pos = evader_pos.squeeze(-2) if evader_pos.ndim == 3 else evader_pos
        ev_vel = evader_vel.squeeze(-2) if evader_vel.ndim == 3 else evader_vel

        device = state.device
        n = state.shape[0]
        K, H = self.num_samples, self.horizon

        pos = state[..., 0:3]
        quat = state[..., 3:7]
        vel = state[..., 7:10]
        R0 = quaternion_to_rotation_matrix(quat)  # [N, 3, 3]

        if self._nominal is None or self._nominal.shape[0] != n:
            self._nominal = self._hover_nominal(n, device)
        if done is not None:
            done_mask = done.reshape(n, -1).any(dim=-1)
            if bool(done_mask.any()):
                self._nominal[done_mask] = self._hover_nominal(int(done_mask.sum()), device)

        g_vec = torch.tensor([0.0, 0.0, -self.g], device=device)

        # Sample K control sequences around the warm-started nominal.
        noise = torch.randn(n, K, H, 4, device=device)
        # Temporally smooth the noise along the horizon (smooth-MPPI): independent
        # per-step noise yields a chattering, near-saturated command that swings
        # the attitude; a moving-average low-pass makes every sampled control
        # sequence smooth, so the averaged command is a coherent maneuver.
        if self.noise_smooth > 1:
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

        # Effective (lagged) inputs, initialised from the current state so the
        # first planned step starts from what the drone is actually doing:
        # body-rate from the measured angular velocity (world -> body), thrust
        # accel from a hover estimate.
        ang_w = state[..., 10:13]
        omega0 = torch.einsum("...ji,...j->...i", R0, ang_w)  # body-frame rate
        omega_eff = omega0.unsqueeze(1).expand(n, K, 3).contiguous()
        ta_eff = torch.full((n, K, 1), self.g, device=device)
        prev_omega_cmd = omega0.unsqueeze(1).expand(n, K, 3)  # for smoothness cost

        cost = torch.zeros(n, K, device=device)
        for h in range(H):
            omega_cmd = controls[:, :, h, 0:3]
            ta_cmd = controls[:, :, h, 3:4]
            if self.w_smooth > 0.0:
                cost = cost + self.w_smooth * ((omega_cmd - prev_omega_cmd) ** 2).sum(dim=-1)
                prev_omega_cmd = omega_cmd
            # First-order lag: inputs approach the command, they do not jump to it.
            omega_eff = omega_eff + (omega_cmd - omega_eff) * self.alpha_rate
            ta_eff = ta_eff + (ta_cmd - ta_eff) * self.alpha_thrust
            omega = omega_eff
            b3 = R[..., :, 2]  # [N, K, 3]

            acc = ta_eff * b3 + g_vec
            v = v + acc * self.mpc_dt
            p = p + v * self.mpc_dt
            R = _orthonormalize(torch.matmul(R, eye + _skew(omega) * self.mpc_dt))

            t = (h + 1) * self.mpc_dt
            ev_pred = (ev_pos + ev_vel * t).unsqueeze(1)  # [N, 1, 3]
            rel = ev_pred - p  # [N, K, 3]
            dist2 = (rel ** 2).sum(dim=-1)  # [N, K]

            disc = self.gamma ** h
            cost = cost + disc * self.w_dist * dist2
            if self.w_closing > 0.0:
                rng = dist2.clamp_min(1e-4).sqrt()
                closing = (v * rel).sum(dim=-1) / rng  # >0 when approaching
                cap = (2.0 * self.closing_decel * rng).sqrt().clamp(max=self.max_closing)
                cost = cost - disc * self.w_closing * torch.minimum(closing, cap)
            cost = cost + self.w_tilt * (1.0 - b3[..., 2]).clamp_min(0.0)
            # Near-hard barrier once the tilt exceeds tilt_max: a steep linear
            # ramp whose gain dwarfs the distance reward, so no sampled
            # trajectory profits from tilting past the envelope (mirrors the hard
            # tilt clamp the CTBR baselines rely on). The lag model above makes
            # the rollout tilt track the real tilt, so this bounds it in reality.
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
        self._nominal = (weights.unsqueeze(-1).unsqueeze(-1) * controls).sum(dim=1)  # [N, H, 4]

        u0 = self._nominal[:, 0, :]  # [N, 4]

        # Warm start. The controller is re-solved every sim step (0.01 s) but the
        # plan is discretised at mpc_dt (0.02 s), so shifting a full step per call
        # would advance the nominal at 2x real time and desync it, turning the
        # command into chatter. By default we keep the aligned nominal and let
        # each call refine it; enable warm_shift only if called once per mpc_dt.
        if self.warm_shift:
            self._nominal = torch.roll(self._nominal, shifts=-1, dims=1)
            self._nominal[:, -1, :] = self._nominal[:, -2, :]

        return self._action_from_input(u0).unsqueeze(-2)  # [N, 1, 4]

    def _action_from_input(self, u0: torch.Tensor) -> torch.Tensor:
        omega = u0[..., 0:3]
        thrust_accel = u0[..., 3:4]

        rate_deg = omega * (180.0 / math.pi)
        a_rate = torch.atanh((rate_deg / (180.0 * self.target_clip)).clamp(-0.995, 0.995))

        throttle = self.hover * torch.sqrt((thrust_accel / self.g).clamp(0.04, 4.0))
        ratio = throttle.clamp(self.min_ratio + 1e-3, self.max_ratio - 1e-3)
        a_thrust = torch.atanh((2.0 * ratio - 1.0).clamp(-0.995, 0.995))
        return torch.cat([a_rate, a_thrust], dim=-1)
