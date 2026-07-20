# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# See the LICENSE file at the repository root for full terms.

"""Unit tests for the deployment core (``intercept_common``).

These tests validate that the deployment-side observation construction and
action decoding match the reference implementations in ``omni_drones`` and are
therefore safe to run in any environment that has ``torch`` (no Isaac Sim / ROS
required)::

    python deploy/test_intercept_common.py
"""

import intercept_common as ic
import math
import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _reference_rotation_matrix(quat):
    """Independent w,x,y,z -> R implementation for cross-checking."""
    w, x, y, z = quat.tolist()
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def test_rotation_matrix_matches_reference():
    quat = torch.tensor([0.9239, 0.0, 0.3827, 0.0])  # ~45 deg about y
    quat = quat / quat.norm()
    got = ic.quaternion_to_rotation_matrix(quat)
    exp = _reference_rotation_matrix(quat)
    assert torch.allclose(got, exp, atol=1e-5), (got, exp)


def test_identity_quat_gives_identity_rotation():
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
    R = ic.quaternion_to_rotation_matrix(quat)
    assert torch.allclose(R, torch.eye(3), atol=1e-6)


def test_observation_dim_default_layout():
    cfg = ic.ObsConfig()  # obs_dim == 16
    assert cfg.expected_obs_dim() == 16
    obs = ic.build_observation(
        cfg,
        pursuer_pos=torch.tensor([0.0, 0.0, 1.6]),
        pursuer_quat_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        pursuer_lin_vel_world=torch.tensor([0.1, 0.2, 0.3]),
        evader_pos=torch.tensor([3.0, 0.0, 1.6]),
    )
    assert obs.shape == (16,)
    # evader_rel_hdg points from pursuer to evader -> +x.
    assert torch.allclose(obs[:3], torch.tensor([1.0, 0.0, 0.0]), atol=1e-4)
    # pursuer_lin_vel component.
    assert torch.allclose(obs[3:6], torch.tensor([0.1, 0.2, 0.3]), atol=1e-6)
    # rotation matrix (identity) flattened.
    assert torch.allclose(obs[6:15], torch.eye(3).reshape(9), atol=1e-6)
    # altitude.
    assert math.isclose(obs[15].item(), 1.6, rel_tol=1e-5)


def test_observation_optional_components():
    cfg = ic.ObsConfig(
        use_ab_world_frame=True,
        use_rot_speed=True,
        use_relative_velocity=True,
    )
    cfg.obs_dim = cfg.expected_obs_dim()
    assert cfg.obs_dim == 3 + 3 + 9 + 3 + 3 + 3  # 24
    obs = ic.build_observation(
        cfg,
        pursuer_pos=torch.tensor([1.0, 2.0, 3.0]),
        pursuer_quat_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        pursuer_lin_vel_world=torch.tensor([0.0, 0.0, 0.0]),
        evader_pos=torch.tensor([1.0, 2.0, 5.0]),
        pursuer_ang_vel_world=torch.tensor([0.5, 0.6, 0.7]),
        evader_lin_vel_world=torch.tensor([1.0, 0.0, 0.0]),
    )
    assert obs.shape == (24,)
    # world-frame position block.
    assert torch.allclose(obs[15:18], torch.tensor([1.0, 2.0, 3.0]), atol=1e-6)


def test_decode_action_hover_and_ranges():
    cfg = ic.CTBRConfig(target_clip=1.0, min_thrust_ratio=0.0, max_thrust_ratio=1.0)
    # Zero raw action -> tanh(0)=0 -> zero body rates, mid thrust (0.5).
    cmd = ic.decode_action_to_ctbr(torch.zeros(1, 4), cfg)
    assert torch.allclose(cmd.body_rate_deg, torch.zeros(1, 3), atol=1e-6)
    assert math.isclose(cmd.thrust_ratio.item(), 0.5, rel_tol=1e-5)
    assert math.isclose(cmd.thrust_pwm.item(), 0.5 * 2 ** 16, rel_tol=1e-5)

    # Large positive raw -> saturates: rate ~ +180 deg/s, thrust ratio ~ 1.0.
    cmd = ic.decode_action_to_ctbr(torch.tensor([[10.0, 10.0, 10.0, 10.0]]), cfg)
    assert torch.all(cmd.body_rate_deg > 179.0)
    assert cmd.thrust_ratio.item() > 0.999


def test_decode_respects_thrust_clamp_and_target_clip():
    cfg = ic.CTBRConfig(target_clip=0.5, min_thrust_ratio=0.1, max_thrust_ratio=0.8)
    cmd = ic.decode_action_to_ctbr(torch.tensor([[10.0, 0.0, 0.0, -10.0]]), cfg)
    # target_clip halves the rate scale (180 * 0.5 = 90).
    assert cmd.body_rate_deg[0, 0].item() > 89.0
    assert cmd.body_rate_deg[0, 0].item() <= 90.0 + 1e-3
    # thrust clamped to min_thrust_ratio.
    assert math.isclose(cmd.thrust_ratio.item(), 0.1, rel_tol=1e-5)


def test_metadata_roundtrip(tmp_path=None):
    import tempfile
    meta = ic.PolicyMetadata(
        artifact_version=ic.ARTIFACT_VERSION,
        algo="ppo",
        obs=ic.ObsConfig(),
        ctbr=ic.CTBRConfig(target_clip=1.0),
        sim_dt=0.02,
        notes={"task": "Intercept"},
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "metadata.json")
        ic.save_metadata(meta, path)
        loaded = ic.load_metadata(path)
    assert loaded.algo == "ppo"
    assert loaded.obs.obs_dim == 16
    assert loaded.ctbr.target_clip == 1.0
    assert loaded.notes["task"] == "Intercept"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as err:
            failures += 1
            print(f"FAIL {test.__name__}: {err}")
        except Exception as err:  # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(err).__name__}: {err}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
