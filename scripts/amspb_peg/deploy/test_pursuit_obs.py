"""Unit test for the AMSPB Pursuit deploy observation builder.

Validates that :class:`PursuitObservationBuilder` (in ``intercept_controller``)
reproduces ``Pursuit._compute_state_and_obs`` from
``omni_drones/envs/single/pursuit.py`` bit-for-bit, with special attention to
the reference-frame conversions that differ between training and deployment:

* Pursuit observes **world**-frame linear velocity and closing velocity
  (``drone.vel_w``), while ``DroneState.lin_vel`` (Crazyflie kalman.statePX/PY/PZ)
  is **body** frame -> the builder must rotate body->world.
* Pursuit's ``body_rates`` is the body-frame angular velocity, which the gyro in
  ``DroneState.ang_vel`` already provides -> no extra rotation.

Runs with only ``torch``/``numpy`` (no Isaac Sim / ROS)::

    python -m scripts.amspb_peg.deploy.test_pursuit_obs
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import scripts.deploy.intercept_common as ic  # noqa: E402
from scripts.amspb_peg.deploy.intercept_controller import (  # noqa: E402
    PursuitObservationBuilder,
)

# Constants matching train_pursuer.yaml (arena_size=2.0).
K_P = torch.tensor([6.0, 6.0, 3.0])
K_V = torch.tensor([15.0, 15.0, 5.0])
K_OMEGA = torch.tensor([15.0, 15.0, 5.0])
K_RP = K_P * 2
ARENA_CENTER = torch.tensor([0.0, 0.0, 3.5])
MAX_LEN = 600

_CFG = dict(
    time_encoding=True, time_encoding_dim=1,
    include_distances=True, include_closing_velocity=True,
    include_heading_to_target=False, include_last_action=False,
    include_effort=False, use_body_rates=True,
    prev_traj_steps=0, prev_evader_traj_steps=0, prev_evader_steps_fix=4,
    max_episode_length=MAX_LEN,
    k_p=K_P.tolist(), k_v=K_V.tolist(), k_omega=K_OMEGA.tolist(),
    k_rp=K_RP.tolist(), arena_center=ARENA_CENTER.tolist(),
)


def _R(q: torch.Tensor) -> torch.Tensor:
    """Body->world rotation matrix for a (w,x,y,z) quaternion."""
    return ic.quaternion_to_rotation_matrix(q.view(1, 4)).reshape(3, 3)


def _training_obs(step, p_pos, p_quat, p_lin_w, p_ang_w, e_pos, e_lin_w):
    """Faithful reimplementation of Pursuit._compute_state_and_obs for _CFG."""
    Rp = _R(p_quat)
    error_pos = e_pos - p_pos
    return torch.cat([
        torch.tensor([step / MAX_LEN]),               # time encoding
        error_pos / K_RP,                             # relative position
        (error_pos / K_RP).norm().reshape(1),         # distance
        -(p_lin_w - e_lin_w) / K_V,                   # closing velocity (world)
        (((p_pos - ARENA_CENTER) / K_P)[2]).reshape(1),  # z coordinate
        Rp.reshape(9),                                # rotation matrix
        p_lin_w / K_V,                                # linear velocity (world)
        (Rp.T @ p_ang_w) / K_OMEGA,                   # body rates
    ])


def test_matches_training_observation():
    torch.manual_seed(0)
    max_err = 0.0
    for trial in range(200):
        p_pos = torch.randn(3) * torch.tensor([4.0, 4.0, 1.0]) + ARENA_CENTER
        e_pos = torch.randn(3) * torch.tensor([4.0, 4.0, 1.0]) + ARENA_CENTER
        p_quat = torch.randn(4); p_quat = p_quat / p_quat.norm()
        e_quat = torch.randn(4); e_quat = e_quat / e_quat.norm()
        p_lin_w = torch.randn(3) * 2.0
        p_ang_w = torch.randn(3) * 1.0
        e_lin_w = torch.randn(3) * 2.0

        obs_train = _training_obs(trial, p_pos, p_quat, p_lin_w, p_ang_w, e_pos, e_lin_w)

        # DroneState as the real runtime provides it: body-frame velocities.
        Rp, Re = _R(p_quat), _R(e_quat)
        pursuer = ic.DroneState(
            pos=p_pos.numpy(), quat_wxyz=p_quat.numpy(),
            lin_vel=(Rp.T @ p_lin_w).numpy(), ang_vel=(Rp.T @ p_ang_w).numpy(),
            stamp=0.0)
        evader = ic.DroneState(
            pos=e_pos.numpy(), quat_wxyz=e_quat.numpy(),
            lin_vel=(Re.T @ e_lin_w).numpy(), ang_vel=np.zeros(3), stamp=0.0)

        builder = PursuitObservationBuilder(_CFG, torch.device("cpu"))
        builder._step = trial
        obs_deploy = builder.build(pursuer, evader, previous_action=None, obs_dim=24).reshape(-1)

        assert obs_deploy.numel() == 24, obs_deploy.numel()
        max_err = max(max_err, (obs_train - obs_deploy).abs().max().item())

    assert max_err < 1e-5, f"deploy/training observation mismatch: {max_err:.3e}"
    return max_err


if __name__ == "__main__":
    err = test_matches_training_observation()
    print(f"PASS: deploy observation matches training (max abs error {err:.3e}).")
