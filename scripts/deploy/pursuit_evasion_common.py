# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the conditions in the LICENSE
# file at the repository root.
"""Shared, dependency-light building blocks for deploying a PursuitEvasion policy.

Like :mod:`intercept_common`, this module depends **only** on ``torch``, ``numpy``
and the standard library, so it can be imported both inside the Isaac Sim
training environment (used by :mod:`export_pursuit_evasion_policy`) and inside
the Crazyswarm2 environment (used by :mod:`pursuit_evasion_controller`).

It must **not** import ``omni_drones``, ``torchrl``, ``isaacsim`` or ``rclpy``.

The task-specific piece that has to match training bit-for-bit is
:func:`build_observation`, which reproduces
``PursuitEvasion._agent_observation`` in
``omni_drones/envs/single/pursuit_evasion.py``.

Everything else (CTBR decoding, drone/mocap config loading, quaternion helpers)
is imported from :mod:`intercept_common`: PursuitEvasion uses the *identical*
4-D CTBR action space and the same ``PIDRateController`` action transform, so
re-using that code keeps the two deployment paths from drifting apart.

Note on frames: the PursuitEvasion observation carries the agent's *env-local*
position, i.e. its position relative to the arena origin. On hardware there is
a single arena, so the mocap world origin must coincide with the arena centre
(or be offset via ``arena_origin`` in the controller config).
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Optional

import torch

# Shared primitives. PursuitEvasion's action space and hardware plumbing are
# identical to Intercept's, so these are imported rather than duplicated.
from intercept_common import (  # noqa: F401  (re-exported for controllers)
    CTBRCommand,
    CTBRConfig,
    DroneConfig,
    DroneState,
    MocapConfig,
    MocapPose,
    artifact_paths,
    decode_action_to_ctbr,
    load_drone_config_from_yaml,
    load_mocap_config_from_yaml,
    normalize,
    quat_rotate_inverse,
    quaternion_to_rotation_matrix,
    METADATA_FILENAME,
    POLICY_TS_FILENAME,
)

# Format identifier written into ``metadata.json``. Separate from the Intercept
# artifact version because the observation layout is a different contract.
# v1: pos(3) rot(9) body lin vel(3) body ang vel(3) rel hdg(3) rel vel(3) [act(4)].
#     "rel hdg" is the unit heading to the opponent (normalize(rel_dist)), matching
#     PursuitEvasion._agent_observation, which observes rel_hdg and NOT raw rel_dist.
PE_ARTIFACT_VERSION = 1

ROLES = ('pursuer', 'evader')


# ---------------------------------------------------------------------------
# Configuration payloads (serialised into metadata.json alongside the policy)
# ---------------------------------------------------------------------------
@dataclass
class PursuitEvasionObsConfig:
    """Observation layout for the PursuitEvasion task.

    The layout is **role-symmetric**: a pursuer and an evader policy consume the
    exact same vector, only differing in which drone is "self" and which is the
    opponent. That is what lets one frozen policy be replayed as either side's
    opponent during self-play, and it lets a single controller fly either role.

    Component order is fixed by ``PursuitEvasion._agent_observation``:
    position, rotation matrix, body linear velocity, body angular velocity,
    opponent relative heading (unit vector, body frame), opponent relative
    velocity (body frame), [previous action].
    """

    use_previous_action: bool = True
    obs_dim: int = 28
    action_dim: int = 4

    def expected_obs_dim(self) -> int:
        """Recompute the observation dimension from the layout flags."""
        # pos(3) + rot matrix(9) + body lin vel(3) + body ang vel(3)
        # + opponent relative position(3) + opponent relative velocity(3)
        result = 3 + 9 + 3 + 3 + 3 + 3
        if self.use_previous_action:
            result += self.action_dim
        return result


@dataclass
class PursuitEvasionMetadata:
    """Everything the controller needs besides the TorchScript weights."""

    artifact_version: int
    algo: str
    role: str
    obs: PursuitEvasionObsConfig = field(default_factory=PursuitEvasionObsConfig)
    ctbr: CTBRConfig = field(default_factory=CTBRConfig)
    sim_dt: float = 0.02
    # Arena the policy was trained in; the controller uses it for safety limits.
    arena_half_xy: float = 5.0
    arena_z_min: float = 0.5
    arena_z_max: float = 4.5
    capture_radius: float = 0.1
    # Free-form provenance (checkpoint path, task name, opponent pool, ...).
    notes: dict = field(default_factory=dict)

    # -- (de)serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            'artifact_version': self.artifact_version,
            'algo': self.algo,
            'role': self.role,
            'obs': dataclasses.asdict(self.obs),
            'ctbr': dataclasses.asdict(self.ctbr),
            'sim_dt': self.sim_dt,
            'arena_half_xy': self.arena_half_xy,
            'arena_z_min': self.arena_z_min,
            'arena_z_max': self.arena_z_max,
            'capture_radius': self.capture_radius,
            'notes': self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PursuitEvasionMetadata':
        return cls(
            artifact_version=int(data['artifact_version']),
            algo=str(data['algo']),
            role=str(data['role']),
            obs=PursuitEvasionObsConfig(**data['obs']),
            ctbr=CTBRConfig(**data['ctbr']),
            sim_dt=float(data.get('sim_dt', 0.02)),
            arena_half_xy=float(data.get('arena_half_xy', 5.0)),
            arena_z_min=float(data.get('arena_z_min', 0.5)),
            arena_z_max=float(data.get('arena_z_max', 4.5)),
            capture_radius=float(data.get('capture_radius', 0.1)),
            notes=dict(data.get('notes', {})),
        )


def save_metadata(metadata: PursuitEvasionMetadata, path: str) -> None:
    """Write ``metadata`` to ``path`` as pretty-printed JSON."""
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(metadata.to_dict(), handle, indent=2, sort_keys=True)


def load_metadata(path: str) -> PursuitEvasionMetadata:
    """Read :class:`PursuitEvasionMetadata`, validating version and role."""
    with open(path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    version = int(data.get('artifact_version', -1))
    if version != PE_ARTIFACT_VERSION:
        raise ValueError(
            f'Incompatible PursuitEvasion artifact version {version} '
            f'(expected {PE_ARTIFACT_VERSION}). Re-run '
            f'export_pursuit_evasion_policy.py.'
        )
    role = str(data.get('role', '')).lower()
    if role not in ROLES:
        raise ValueError(
            f"Artifact declares role '{role}'; expected one of {list(ROLES)}."
        )
    data['role'] = role
    return PursuitEvasionMetadata.from_dict(data)


# ---------------------------------------------------------------------------
# Observation construction
# ---------------------------------------------------------------------------
def build_observation(
    cfg: PursuitEvasionObsConfig,
    self_pos: torch.Tensor,
    self_quat_wxyz: torch.Tensor,
    self_lin_vel_world: torch.Tensor,
    self_ang_vel_body: torch.Tensor,
    opponent_pos: torch.Tensor,
    opponent_lin_vel_world: torch.Tensor,
    previous_action: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Assemble the PursuitEvasion observation vector from raw states.

    Reproduces ``PursuitEvasion._agent_observation``. All tensors have a
    trailing feature dimension and may carry arbitrary leading batch
    dimensions. The quaternion follows the Isaac ``(w, x, y, z)`` convention.

    Unlike :func:`intercept_common.build_observation`, both linear velocities
    are taken in the **world** frame and rotated here, because the env computes
    the relative velocity as ``R^T (v_opponent - v_self)``. Angular velocity is
    expected already in the body frame (as ``drone.get_state()`` reports it).

    Args:
        cfg: Observation layout flags (must match the trained policy).
        self_pos: ``[..., 3]`` agent position relative to the arena origin.
        self_quat_wxyz: ``[..., 4]`` agent orientation as ``(w, x, y, z)``.
        self_lin_vel_world: ``[..., 3]`` agent linear velocity (world frame).
        self_ang_vel_body: ``[..., 3]`` agent angular velocity (body frame).
        opponent_pos: ``[..., 3]`` opponent position relative to the arena origin.
        opponent_lin_vel_world: ``[..., 3]`` opponent linear velocity (world frame).
        previous_action: ``[..., 4]`` agent's own previous action in [-1, 1],
            required only when ``cfg.use_previous_action`` is True.

    Returns:
        ``[..., obs_dim]`` observation tensor.
    """
    rot = quaternion_to_rotation_matrix(self_quat_wxyz)
    rot = rot.reshape(*rot.shape[:-2], 9)  # (9)

    # Own linear velocity in the body frame: R^T v_self.
    lin_vel_body = quat_rotate_inverse(self_quat_wxyz, self_lin_vel_world)
    # Opponent relative position / velocity, both in the agent body frame.
    rel_pos_body = quat_rotate_inverse(self_quat_wxyz, opponent_pos - self_pos)
    # The env observes only the *heading* to the opponent (a unit vector), not
    # the raw relative position: PursuitEvasion._agent_observation uses
    # ``rel_hdg = normalize(rel_dist)`` and leaves ``rel_dist`` commented out.
    # Feeding the un-normalised vector here would put the observation far out of
    # distribution (its magnitude is the metre-scale distance, not 1).
    rel_hdg_body = normalize(rel_pos_body)
    rel_vel_body = quat_rotate_inverse(
        self_quat_wxyz, opponent_lin_vel_world - self_lin_vel_world)

    components = [
        self_pos,           # (3)
        rot,                # (9)
        lin_vel_body,       # (3)
        self_ang_vel_body,  # (3)
        rel_pos_body,
        # rel_hdg_body,       # (3)
        rel_vel_body,       # (3)
    ]

    if cfg.use_previous_action:
        if previous_action is None:
            raise ValueError(
                'use_previous_action=True requires previous_action.'
            )
        components.append(previous_action)  # (4)

    obs = torch.cat(components, dim=-1)
    if obs.shape[-1] != cfg.obs_dim:
        raise ValueError(
            f'Assembled observation has dim {obs.shape[-1]} but metadata '
            f'declares obs_dim={cfg.obs_dim}. Check the ObsConfig flags.'
        )
    return obs


def opponent_role(role: str) -> str:
    """Return the opposing role name."""
    role = str(role).lower()
    if role not in ROLES:
        raise ValueError(f"Unknown role '{role}'; expected one of {list(ROLES)}.")
    return 'evader' if role == 'pursuer' else 'pursuer'
