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
import importlib.util
import json
import os
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
    rotation_obs,
    rotation_obs_dim,
    esn_augment,
    load_esn,
    ESN_FILENAME,
    METADATA_FILENAME,
    POLICY_TS_FILENAME,
)

# Format identifier written into ``metadata.json``. Separate from the Intercept
# artifact version because the observation layout is a different contract.
# v1: pos(3) rot(9|4|3) body lin vel(3) body ang vel(3) rel dist|hdg(3) rel vel(3)
#     [act(4)] [esn(6)]. The opponent slot is the raw body-frame relative position
#     when obs.use_rel_dist (env default) else its unit heading.
# v2: v1 + optional [opponent adapter(3)] (learned acceleration estimate + RLS)
#     and the top-level ``adapter`` block. v1 artifacts still load unchanged.
PE_ARTIFACT_VERSION = 2
SUPPORTED_ARTIFACT_VERSIONS = (1, 2)

# Learned basis phi + input normalisation of the opponent adapter, copied into
# the artifact folder by the exporter.
ADAPTER_BASIS_FILENAME = 'adapter_basis.pt'
# Loaded by file path (like the ESN): importing the omni_drones package would
# pull in Isaac Sim. Using the training module keeps the RLS bit-identical.
_ADAPTER_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'omni_drones', 'utils', 'opponent_adapt.py',
)

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

    Component order is fixed by ``PursuitEvasion._agent_observation`` and
    ``PursuitEvasion._compute_state_and_obs``: position, rotation (matrix /
    quat / log), body linear velocity, body angular velocity, opponent relative
    position or heading (body frame), opponent relative velocity (body frame),
    [previous action], [ESN prediction], [opponent adapter output].
    """

    use_previous_action: bool = True
    # Opponent slot: raw body-frame relative position (True) or its unit heading (False).
    use_rel_dist: bool = True
    # Attitude slot: 'matrix' (vec(R), 9-D), 'quat' (w >= 0, 4-D) or 'log' (log(R), 3-D).
    rot_repr: str = 'matrix'
    # ESN-OA: 6-D prediction (rel distance + rel lin vel) appended after the base obs.
    use_esn: bool = False
    esn_dim: int = 6
    # Opponent adapter (metadata.adapter): 3-D body-frame estimate appended last.
    use_adapter: bool = False
    adapter_dim: int = 3
    obs_dim: int = 28
    action_dim: int = 4

    def base_obs_dim(self) -> int:
        """Observation dimension before the ESN / adapter outputs are appended."""
        return (self.obs_dim - (self.esn_dim if self.use_esn else 0)
                - (self.adapter_dim if self.use_adapter else 0))

    def expected_obs_dim(self) -> int:
        """Recompute the observation dimension from the layout flags."""
        # pos(3) + rotation(9/4/3) + body lin vel(3) + body ang vel(3)
        # + opponent relative position(3) + opponent relative velocity(3)
        result = 3 + rotation_obs_dim(self.rot_repr) + 3 + 3 + 3 + 3
        if self.use_previous_action:
            result += self.action_dim
        if self.use_esn:
            result += self.esn_dim
        if self.use_adapter:
            result += self.adapter_dim
        return result


@dataclass
class OpponentAdapterConfig:
    """Learned opponent-acceleration estimator with an online RLS readout.

    Mirrors ``task.adapt`` in ``cfg/task/PursuitEvasion.yaml`` (thesis A2 / F5 /
    F14); the estimator itself is ``OpponentAdapter`` in
    ``omni_drones/utils/opponent_adapt.py``. Only the configuration the thesis
    pursuers were trained with is deployable: a learned basis (``kind='nf'``)
    fed with the true relative velocity (``z_mode='vel'``).
    """

    kind: str = 'nf'
    # 'dvel': predicts the evader's velocity change over `horizon` steps, fed to
    # the policy as 0.1 x its mean acceleration; 'disp': relative displacement.
    target: str = 'dvel'
    z_mode: str = 'vel'
    history: int = 10          # input window, control steps
    horizon: int = 10          # prediction horizon, control steps
    forgetting: float = 0.999  # RLS lambda
    delta: float = 0.01        # RLS initial covariance scale
    p_max_ratio: float = 1.0   # covariance-trace cap relative to its initial value
    freeze: bool = False       # True: no online RLS update
    reset_on_episode: bool = True

    def validate(self) -> None:
        if self.kind != 'nf' or self.z_mode != 'vel' or self.target not in ('dvel', 'disp'):
            raise ValueError(
                f'Unsupported opponent adapter (kind={self.kind!r}, z_mode={self.z_mode!r}, '
                f'target={self.target!r}); deploy supports kind=nf, z_mode=vel, target=dvel|disp.'
            )


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
    # Training episode length in control steps (timeout = evader win); 0 = unknown.
    max_episode_length: int = 0
    # Free-form provenance (checkpoint path, task name, opponent pool, ...).
    notes: dict = field(default_factory=dict)
    # Opponent adapter settings when obs.use_adapter (else None).
    adapter: Optional[OpponentAdapterConfig] = None

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
            'max_episode_length': self.max_episode_length,
            'notes': self.notes,
            'adapter': dataclasses.asdict(self.adapter) if self.adapter is not None else None,
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
            max_episode_length=int(data.get('max_episode_length', 0)),
            notes=dict(data.get('notes', {})),
            adapter=(OpponentAdapterConfig(**data['adapter'])
                     if data.get('adapter') else None),
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
    if version not in SUPPORTED_ARTIFACT_VERSIONS:
        raise ValueError(
            f'Incompatible PursuitEvasion artifact version {version} '
            f'(expected one of {SUPPORTED_ARTIFACT_VERSIONS}). Re-run '
            f'export_pursuit_evasion_policy.py.'
        )
    role = str(data.get('role', '')).lower()
    if role not in ROLES:
        raise ValueError(
            f"Artifact declares role '{role}'; expected one of {list(ROLES)}."
        )
    data['role'] = role
    metadata = PursuitEvasionMetadata.from_dict(data)
    if metadata.obs.use_adapter:
        if metadata.adapter is None:
            raise ValueError('obs.use_adapter is set but metadata has no adapter block.')
        if metadata.role != 'pursuer':
            raise ValueError('The opponent adapter is defined for the pursuer only.')
        metadata.adapter.validate()
    return metadata


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
    rot = rotation_obs(self_quat_wxyz, cfg.rot_repr)  # (9), (4) or (3)

    # Own linear velocity in the body frame: R^T v_self.
    lin_vel_body = quat_rotate_inverse(self_quat_wxyz, self_lin_vel_world)
    # Opponent relative position / velocity, both in the agent body frame.
    rel_pos_body = quat_rotate_inverse(self_quat_wxyz, opponent_pos - self_pos)
    rel_vel_body = quat_rotate_inverse(
        self_quat_wxyz, opponent_lin_vel_world - self_lin_vel_world)

    components = [
        self_pos,           # (3)
        rot,                # (9), (4) or (3)
        lin_vel_body,       # (3)
        self_ang_vel_body,  # (3)
        rel_pos_body if cfg.use_rel_dist else normalize(rel_pos_body),  # (3)
        rel_vel_body,       # (3)
    ]

    if cfg.use_previous_action:
        if previous_action is None:
            raise ValueError(
                'use_previous_action=True requires previous_action.'
            )
        components.append(previous_action)  # (4)

    obs = torch.cat(components, dim=-1)
    if obs.shape[-1] != cfg.base_obs_dim():
        raise ValueError(
            f'Assembled observation has dim {obs.shape[-1]} but metadata '
            f'declares base obs_dim={cfg.base_obs_dim()}. Check the ObsConfig flags.'
        )
    return obs


def build_esn_target(
    self_pos: torch.Tensor,
    self_quat_wxyz: torch.Tensor,
    self_lin_vel_world: torch.Tensor,
    opponent_pos: torch.Tensor,
    opponent_lin_vel_world: torch.Tensor,
) -> torch.Tensor:
    """ESN RLS target: body-frame opponent rel distance (3) + rel lin vel (3),
    matching ``PursuitEvasion._agent_observation``."""
    rel_pos_body = quat_rotate_inverse(self_quat_wxyz, opponent_pos - self_pos)
    rel_vel_body = quat_rotate_inverse(
        self_quat_wxyz, opponent_lin_vel_world - self_lin_vel_world)
    return torch.cat([rel_pos_body, rel_vel_body], dim=-1)


# ---------------------------------------------------------------------------
# Opponent adapter (learned acceleration estimate + online RLS)
# ---------------------------------------------------------------------------
def load_opponent_adapter(artifact_dir: str, cfg: OpponentAdapterConfig,
                          device: torch.device = torch.device('cpu')):
    """Fresh single-env ``OpponentAdapter`` (readout at its learned mean, P = delta*I)."""
    cfg.validate()
    spec = importlib.util.spec_from_file_location('_omni_drones_opponent_adapt', _ADAPTER_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OpponentAdapter(
        1, kind=cfg.kind, history=cfg.history, horizon=cfg.horizon,
        forgetting=cfg.forgetting, delta=cfg.delta,
        model_path=os.path.join(artifact_dir, ADAPTER_BASIS_FILENAME),
        freeze=cfg.freeze, reset_on_episode=cfg.reset_on_episode,
        p_max_ratio=cfg.p_max_ratio, target=cfg.target, device=device,
    )


def build_adapter_input(
    pursuer_pos: torch.Tensor,
    pursuer_lin_vel_world: torch.Tensor,
    evader_pos: torch.Tensor,
    evader_lin_vel_world: torch.Tensor,
) -> torch.Tensor:
    """Adapter input z = [rel_pos, rel_vel, pursuer_vel] (world frame, rel = evader - pursuer).

    Reproduces ``PursuitEvasion.opponent_z(mode='vel')``.
    """
    return torch.cat([
        evader_pos - pursuer_pos,
        evader_lin_vel_world - pursuer_lin_vel_world,
        pursuer_lin_vel_world,
    ], dim=-1)


def adapter_step(adapter, cfg: OpponentAdapterConfig, z: torch.Tensor,
                 self_quat_wxyz: torch.Tensor, control_dt: float) -> torch.Tensor:
    """Advance the adapter by one control step; returns its 3-D body-frame observation slot.

    Reproduces ``PursuitEvasion._adapt_features``: for ``target='dvel'`` the
    predicted velocity change over the horizon becomes 0.1 x the predicted mean
    acceleration. Call exactly once per control tick, as in training (the RLS
    labels are ``horizon`` ticks delayed).
    """
    out = adapter.step(z.reshape(1, -1)).reshape(-1)
    if cfg.target == 'dvel':
        out = out * (0.1 / (cfg.horizon * control_dt))
    return quat_rotate_inverse(self_quat_wxyz, out)


def opponent_role(role: str) -> str:
    """Return the opposing role name."""
    role = str(role).lower()
    if role not in ROLES:
        raise ValueError(f"Unknown role '{role}'; expected one of {list(ROLES)}.")
    return 'evader' if role == 'pursuer' else 'pursuer'
