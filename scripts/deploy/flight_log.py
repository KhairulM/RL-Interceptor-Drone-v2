"""Per-flight CSV log for pursuit_evasion_controller.py (config: logging.csv_dir).

One CSV per flight, one row per control tick, plus a JSON sidecar with the run's metadata (config, artifacts, outcome).
Rows stream to ``<stem>.csv.partial`` during the flight, so a crash of the controller keeps the data; ``finish()``
writes ``<stem>.csv`` with the outcome filled into every row and removes the partial file.

Frames and sources: positions are the drones' state estimates in the mocap world frame (arena centre = config
arena.origin); velocities are the estimators' world-frame velocities (stateEstimate.v*); attitudes are the estimators'
roll, pitch, yaw in degrees (stateEstimate.q*, ZYX Euler angles, world frame), the attitude the policies see.
"""
from __future__ import annotations

import csv
import json
import os
import time
from typing import Optional

import numpy as np

# How a flight ended. pursuer_oob = the safety fence stopped the pursuer (training lets it leave the arena).
OUTCOMES = ('capture', 'evader_crash', 'evader_oob', 'pursuer_crash', 'pursuer_oob', 'timeout',
            'safety_stop', 'state_timeout', 'link_lost', 'operator_abort', 'error')

COLUMNS = (
    'wall_time', 't', 'outcome',
    'pursuer_x', 'pursuer_y', 'pursuer_z', 'pursuer_vx', 'pursuer_vy', 'pursuer_vz', 'pursuer_speed',
    'pursuer_roll', 'pursuer_pitch', 'pursuer_yaw',
    'approach_velocity', 'distance',
    'evader_x', 'evader_y', 'evader_z', 'evader_vx', 'evader_vy', 'evader_vz', 'evader_speed',
    'evader_roll', 'evader_pitch', 'evader_yaw',
    'evader_policy',
)


def approach_velocity(p_pursuer, v_pursuer, p_evader, v_evader) -> float:
    """Closing speed along the line of sight (m/s, positive = closing): -d|p_e - p_p|/dt (Roncero Eq. 10's v_close)."""
    los = np.asarray(p_evader, float) - np.asarray(p_pursuer, float)
    u = los / max(float(np.linalg.norm(los)), 1e-6)
    return float(-np.dot(u, np.asarray(v_evader, float) - np.asarray(v_pursuer, float)))


def euler_deg(quat_wxyz) -> 'tuple[float, float, float]':
    """(roll, pitch, yaw) in degrees, ZYX convention, of a (w, x, y, z) quaternion."""
    w, x, y, z = np.asarray(quat_wxyz, float)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(float(a) for a in np.degrees((roll, pitch, yaw)))


def artifact_label(artifact_dir: str) -> str:
    """'artifacts/thesis_F14_rls_s200/evader_stage23' -> 'thesis_F14_rls_s200-evader_stage23'."""
    parts = os.path.normpath(artifact_dir).split(os.sep)
    return '-'.join(parts[-2:])


def file_stem(start_wall_time: float, pursuer: str, evader: str) -> str:
    """Sortable, filesystem-safe name: YYYY-MM-DD_HH-MM-SS-<pursuer>-vs-<evader>."""
    stamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(start_wall_time))
    safe = lambda s: ''.join(c if c.isalnum() or c in '._@-' else '_' for c in s)  # noqa: E731
    return f'{stamp}-{safe(pursuer)}-vs-{safe(evader)}'


def outcome_from_safety(reasons) -> 'tuple[str, str]':
    """Map a safety verdict's reasons ('LEVEL: role: text') to (outcome, detail) using the first reason."""
    detail = ' | '.join(reasons)
    first = reasons[0] if reasons else ''
    parts = [p.strip() for p in first.split(':', 2)]
    role, text = (parts[1], parts[2]) if len(parts) == 3 else ('', first)
    if role in ('pursuer', 'evader'):
        if 'geofence' in text:
            return f'{role}_oob', detail
        if any(k in text for k in ('tilt', 'body rate', 'below min altitude', 'falling')):
            return f'{role}_crash', detail
    return 'safety_stop', detail


class FlightLog:
    def __init__(self, directory: str, pursuer: str, evader: str, start_wall_time: float, metadata: dict) -> None:
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(os.path.abspath(directory), file_stem(start_wall_time, pursuer, evader) + '.csv')
        self._partial = self.path + '.partial'
        self._start = start_wall_time
        self._meta = dict(metadata, pursuer=pursuer, evader=evader, start_wall_time=start_wall_time)
        self._file = open(self._partial, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)
        self.rows = 0

    def row(self, wall_time: float, pursuer_state, evader_state, evader_policy: str) -> None:
        pp, pv = np.asarray(pursuer_state.pos, float), np.asarray(pursuer_state.lin_vel, float)
        ep, ev = np.asarray(evader_state.pos, float), np.asarray(evader_state.lin_vel, float)
        self._writer.writerow([
            f'{wall_time:.4f}', f'{wall_time - self._start:.4f}', '',
            *(f'{x:.4f}' for x in pp), *(f'{x:.4f}' for x in pv), f'{np.linalg.norm(pv):.4f}',
            *(f'{a:.2f}' for a in euler_deg(pursuer_state.quat_wxyz)),
            f'{approach_velocity(pp, pv, ep, ev):.4f}', f'{np.linalg.norm(ep - pp):.4f}',
            *(f'{x:.4f}' for x in ep), *(f'{x:.4f}' for x in ev), f'{np.linalg.norm(ev):.4f}',
            *(f'{a:.2f}' for a in euler_deg(evader_state.quat_wxyz)),
            evader_policy,
        ])
        self._file.flush()
        self.rows += 1

    def finish(self, outcome: str, detail: str = '', extra: Optional[dict] = None) -> str:
        """Write <stem>.csv with the outcome in every row and the JSON sidecar; returns the CSV path."""
        assert outcome in OUTCOMES, outcome
        self._file.close()
        with open(self._partial, newline='') as src, open(self.path, 'w', newline='') as dst:
            reader, writer = csv.reader(src), csv.writer(dst)
            writer.writerow(next(reader))
            col = COLUMNS.index('outcome')
            for r in reader:
                r[col] = outcome
                writer.writerow(r)
        os.remove(self._partial)
        meta = dict(self._meta, outcome=outcome, outcome_detail=detail, rows=self.rows,
                    end_wall_time=time.time(), **(extra or {}))
        with open(self.path[:-4] + '.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        return self.path
