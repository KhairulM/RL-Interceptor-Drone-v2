"""Safety layer for the PursuitEvasion hardware controller (SAFETY_PLAN.md, part A).

Pure numpy (no cflib / ROS / torch) so every detector is unit-tested (tests/test_deploy_safety.py).

* :class:`SafetyMonitor` checks both drones every control tick and returns one :class:`Verdict` with a graded level:
  OK < WARN < HOLD < LAND < EMERGENCY. Levels latch (a run never de-escalates) and HOLD escalates to LAND when its
  cause persists.
* :class:`MocapGate` checks each OptiTrack rigid-body frame before it is forwarded to a Crazyflie's estimator: it drops
  position jumps and suspected rigid-body swaps and records per-body health (time since the last accepted frame,
  consecutive rejections).
"""
from __future__ import annotations

import dataclasses
import enum
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


class Level(enum.IntEnum):
    OK = 0
    WARN = 1
    HOLD = 2        # stop the policy, hold position, then land
    LAND = 3        # land every drone (onboard estimator still trusted)
    EMERGENCY = 4   # motors off on the affected drone(s)


@dataclass
class Verdict:
    level: Level = Level.OK
    reasons: List[str] = field(default_factory=list)
    emergency_roles: List[str] = field(default_factory=list)   # drones to stop at EMERGENCY (others land)

    def raise_to(self, level: Level, reason: str, role: Optional[str] = None) -> None:
        if level == Level.EMERGENCY and role is not None and role not in self.emergency_roles:
            self.emergency_roles.append(role)
        if level > self.level:
            self.level = level
        self.reasons.append(f'{level.name}: {reason}')


@dataclass
class SafetyConfig:
    enabled: bool = True
    # Hardware geofence (arena + margin) for BOTH drones; separate from the training rule that only the evader is
    # confined. Predictive check: position + velocity * stop_lookahead_s.
    geofence_margin: float = 0.3
    stop_lookahead_s: float = 0.3
    min_altitude: float = 0.10
    max_descent_speed: float = 2.5      # m/s, checked below descent_check_alt
    descent_check_alt: float = 0.6
    max_tilt_deg: float = 60.0
    max_body_rate_deg: float = 720.0
    max_speed: float = 6.0
    min_separation: float = 0.0         # m; 0 = off (rely on the capture stop)
    telemetry_timeout: float = 0.1      # s without state packets -> HOLD
    telemetry_land_after: float = 0.3   # s -> LAND
    mocap_timeout: float = 0.2          # s without an accepted mocap frame -> HOLD
    mocap_land_after: float = 0.5       # s -> LAND
    mocap_max_rejects: int = 5          # consecutive rejected mocap frames -> HOLD
    hold_s: float = 1.0                 # HOLD duration before landing
    strict_preflight: bool = True       # refuse to take off without estimator convergence / mocap alignment
    # MocapGate
    mocap_max_speed: float = 5.0        # m/s used for the jump gate
    mocap_jump_tol: float = 0.05        # m
    mocap_swap_tol: float = 0.05        # m
    # Fault injection for simulator tests: {kind: none|stale_opponent|nan_action|position_jump|tilt, at_s: float}
    inject: Dict = field(default_factory=lambda: {'kind': 'none', 'at_s': 5.0})

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> 'SafetyConfig':
        d = dict(d or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"Unknown 'safety' config keys: {unknown}")
        return cls(**d)


@dataclass
class Arena:
    origin: np.ndarray
    half_xy: float
    z_min: float
    z_max: float


def tilt_deg(quat_wxyz) -> float:
    """Angle between the body z axis and world up, from a (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    n = w * w + x * x + y * y + z * z
    if n <= 0.0:
        return float('nan')
    cos_t = 1.0 - 2.0 * (x * x + y * y) / n
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_t))))


def _finite(*arrays) -> bool:
    return all(np.all(np.isfinite(np.asarray(a, dtype=np.float64))) for a in arrays)


class SafetyMonitor:
    """Per-tick checks over both drones; the returned level never decreases within a run."""

    def __init__(self, cfg: SafetyConfig, arena: Arena) -> None:
        self.cfg = cfg
        self.arena = arena
        self.latched = Verdict()
        self._hold_since: Dict[str, float] = {}

    def _fence(self, local: np.ndarray) -> bool:
        a, m = self.arena, self.cfg.geofence_margin
        return bool(abs(local[0]) > a.half_xy + m or abs(local[1]) > a.half_xy + m or local[2] > a.z_max + m)

    def check_drone(self, v: Verdict, role: str, state, now: float, mocap_age: Optional[float] = None,
                    mocap_rejects: int = 0, mocap_swap: bool = False) -> None:
        c = self.cfg
        if state is None:
            v.raise_to(Level.LAND, f'{role}: no state')
            return
        if not _finite(state.pos, state.quat_wxyz, state.lin_vel, state.ang_vel):
            v.raise_to(Level.EMERGENCY, f'{role}: non-finite state', role)
            return
        age = now - float(state.stamp)
        if age > c.telemetry_land_after:
            v.raise_to(Level.LAND, f'{role}: state stale {age:.2f} s')
        elif age > c.telemetry_timeout:
            v.raise_to(Level.HOLD, f'{role}: state stale {age:.2f} s')
        if mocap_age is not None:
            if mocap_age > c.mocap_land_after:
                v.raise_to(Level.LAND, f'{role}: mocap stale {mocap_age:.2f} s')
            elif mocap_age > c.mocap_timeout:
                v.raise_to(Level.HOLD, f'{role}: mocap stale {mocap_age:.2f} s')
        if mocap_swap:
            v.raise_to(Level.LAND, f'{role}: mocap rigid-body swap suspected')
        if mocap_rejects >= c.mocap_max_rejects:
            v.raise_to(Level.HOLD, f'{role}: {mocap_rejects} mocap frames rejected in a row')
        tilt = tilt_deg(state.quat_wxyz)
        if tilt > c.max_tilt_deg:
            v.raise_to(Level.EMERGENCY, f'{role}: tilt {tilt:.0f} deg', role)
        rate = math.degrees(float(np.linalg.norm(state.ang_vel)))
        if rate > c.max_body_rate_deg:
            v.raise_to(Level.EMERGENCY, f'{role}: body rate {rate:.0f} deg/s', role)
        speed = float(np.linalg.norm(state.lin_vel))
        if speed > c.max_speed:
            v.raise_to(Level.LAND, f'{role}: speed {speed:.1f} m/s')
        local = np.asarray(state.pos, dtype=np.float64) - self.arena.origin
        if self._fence(local):
            v.raise_to(Level.LAND, f'{role}: outside the geofence at {np.round(local, 2).tolist()}')
        elif self._fence(local + np.asarray(state.lin_vel) * c.stop_lookahead_s):
            v.raise_to(Level.HOLD, f'{role}: about to leave the geofence')
        if local[2] < c.descent_check_alt and state.lin_vel[2] < -c.max_descent_speed:
            # HOLD, not motors off: a position setpoint makes the firmware brake with full thrust, while cutting the
            # motors at this height only makes the impact harder (CrazySim test 2026-11-12). If the drone still hits
            # the floor and flips, the tilt rule stops the motors.
            v.raise_to(Level.HOLD, f'{role}: falling {state.lin_vel[2]:.1f} m/s at {local[2]:.2f} m')
        if local[2] < c.min_altitude:
            v.raise_to(Level.LAND, f'{role}: below min altitude ({local[2]:.2f} m)')

    def check(self, states: Dict[str, object], now: float, action_finite: bool = True,
              mocap: Optional[Dict[str, dict]] = None) -> Verdict:
        """states: role -> DroneState (None if missing); mocap: role -> {'age', 'rejects', 'swap'} for tracked
        drones. Returns the latched verdict (never lower than any earlier one)."""
        v = Verdict()
        if not self.cfg.enabled:
            return v
        for role, st in states.items():
            m = (mocap or {}).get(role) or {}
            self.check_drone(v, role, st, now, m.get('age'), int(m.get('rejects', 0)), bool(m.get('swap', False)))
        if not action_finite:
            v.raise_to(Level.HOLD, 'non-finite policy action')
        if self.cfg.min_separation > 0.0 and all(s is not None for s in states.values()) and len(states) == 2:
            a, b = (np.asarray(s.pos) for s in states.values())
            d = float(np.linalg.norm(a - b))
            if d < self.cfg.min_separation:
                v.raise_to(Level.HOLD, f'separation {d:.2f} m < {self.cfg.min_separation:.2f} m')
        # latch: keep the worst level seen
        if v.level > self.latched.level:
            self.latched.level = v.level
        for r in v.reasons:
            if r not in self.latched.reasons:
                self.latched.reasons.append(r)
        for r in v.emergency_roles:
            if r not in self.latched.emergency_roles:
                self.latched.emergency_roles.append(r)
        return self.latched

    def preflight(self, states: Dict[str, object], now: float) -> List[str]:
        """Problems that must block take-off (empty list = OK). Drones must be on the ground (z near 0)."""
        problems = []
        for role, st in states.items():
            if st is None:
                problems.append(f'{role}: no state')
                continue
            if not _finite(st.pos, st.quat_wxyz, st.lin_vel):
                problems.append(f'{role}: non-finite state')
                continue
            if now - float(st.stamp) > self.cfg.telemetry_land_after:
                problems.append(f'{role}: state stale')
            local = np.asarray(st.pos) - self.arena.origin
            if abs(local[0]) > self.arena.half_xy or abs(local[1]) > self.arena.half_xy:
                problems.append(f'{role}: outside the arena at {np.round(local, 2).tolist()}')
            if tilt_deg(st.quat_wxyz) > 20.0:
                problems.append(f'{role}: not level ({tilt_deg(st.quat_wxyz):.0f} deg)')
        return problems


class MocapGate:
    """Per-rigid-body plausibility gate for mocap frames (jumps, swaps) plus health bookkeeping."""

    def __init__(self, cfg: SafetyConfig) -> None:
        self.cfg = cfg
        self._last_pos: Dict[int, np.ndarray] = {}
        self._last_t: Dict[int, float] = {}
        self._rejects: Dict[int, int] = {}
        self._rejected_total: Dict[int, int] = {}
        self._swap: Dict[int, bool] = {}

    def accept(self, rb_id: int, pos, now: float) -> bool:
        p = np.asarray(pos, dtype=np.float64)
        if not np.all(np.isfinite(p)):
            return self._reject(rb_id)
        last = self._last_pos.get(rb_id)
        if last is not None:
            dt = max(now - self._last_t[rb_id], 1e-3)
            own_jump = float(np.linalg.norm(p - last))
            for other, op in self._last_pos.items():
                if other != rb_id and float(np.linalg.norm(p - op)) < self.cfg.mocap_swap_tol \
                        and own_jump > 2 * self.cfg.mocap_swap_tol:
                    self._swap[rb_id] = True
                    return self._reject(rb_id)
            if own_jump > self.cfg.mocap_max_speed * dt + self.cfg.mocap_jump_tol:
                return self._reject(rb_id)
        self._last_pos[rb_id], self._last_t[rb_id] = p, now
        self._rejects[rb_id] = 0
        return True

    def _reject(self, rb_id: int) -> bool:
        self._rejects[rb_id] = self._rejects.get(rb_id, 0) + 1
        self._rejected_total[rb_id] = self._rejected_total.get(rb_id, 0) + 1
        return False

    def health(self, rb_id: int, now: float) -> dict:
        t = self._last_t.get(rb_id)
        return {'age': (now - t) if t is not None else float('inf'), 'rejects': self._rejects.get(rb_id, 0),
                'swap': self._swap.get(rb_id, False), 'rejected_total': self._rejected_total.get(rb_id, 0)}
