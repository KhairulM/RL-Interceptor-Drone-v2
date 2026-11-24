"""PursuitEvasion deploy with the opponent adapter (learned acceleration estimate + RLS, thesis F14), and the
controller config, with ROS / cflib stubbed.

The adapter parity tests replay recorded F18 trajectories of the deployed pursuer (F14 RLS seed 200) and compare the
deploy path (one fresh single-env adapter per episode, as one controller run) with the env's batched adapter and its
feature formula (PursuitEvasion._adapt_features). They are skipped when the artifact or the recordings are missing.
"""

import argparse
import os
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "scripts/deploy"
ARTIFACT = DEPLOY / "artifacts/thesis_F14_rls_s200"
TRAJ = REPO / "experiments/thesis/F18_trajectories/traj/T2_rls_s200_cross.pt"
PE_CONFIGS = ["config_pursuit_evasion", "config_sim_pvp", "config_sim_pvp_lpf", "config_sim_pvp_v4",
              "config_sim_pursuer_hover", "config_sim_pursuer_vs_hover"]


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)
    return sys.modules[name]


class _Any:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return _Any()

    def __call__(self, *a, **k):
        return _Any()


for name in ("rclpy", "geometry_msgs", "geometry_msgs.msg", "tf2_msgs", "tf2_msgs.msg", "cflib", "cflib.crazyflie",
             "cflib.crazyflie.log", "cflib.crazyflie.syncLogger", "NatNetClient"):
    _stub(name, ok=lambda: True, init=lambda: None, create_node=_Any, TransformStamped=_Any, PoseStamped=_Any,
          TFMessage=_Any, Crazyflie=_Any, LogConfig=_Any, SyncLogger=_Any, NatNetClient=_Any, publisher=_Any())
sys.path.insert(0, str(DEPLOY))
import pursuit_evasion_common as pec  # noqa: E402
import pursuit_evasion_controller as pe_ctrl  # noqa: E402

HAVE_ARTIFACT = (ARTIFACT / "pursuer" / pec.METADATA_FILENAME).is_file()


def _rotmat(q):
    """[N, 4] (w, x, y, z) -> [N, 3, 3], independent of the quaternion helpers under test."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


@unittest.skipUnless(HAVE_ARTIFACT and TRAJ.is_file(), "F14 RLS artifact or F18 trajectories missing")
class AdapterParityTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.meta = pec.load_metadata(str(ARTIFACT / "pursuer" / pec.METADATA_FILENAME))
        d = torch.load(TRAJ, map_location="cpu")
        n, steps = 6, 400
        cls.z, cls.own_pos, cls.done = d["z"][:steps, :n], d["own_pos"][:steps, :n], d["done"][:steps - 1, :n]
        cls.dt = float(d["step_dt"])

    def test_metadata(self):
        a = self.meta.adapter
        self.assertEqual((a.kind, a.target, a.z_mode, a.history, a.horizon), ("nf", "dvel", "vel", 10, 10))
        self.assertEqual((a.delta, a.forgetting, a.freeze), (0.01, 0.999, False))
        self.assertEqual(self.meta.obs.obs_dim, 31)
        self.assertEqual(self.meta.obs.base_obs_dim(), 28)
        self.assertAlmostEqual(self.meta.ctbr.dt, self.dt)

    def test_adapter_input_matches_env(self):
        z, p_pos = self.z, self.own_pos
        p_vel = z[..., 6:9]
        e_pos, e_vel = p_pos + z[..., 0:3], z[..., 3:6] + p_vel
        torch.testing.assert_close(pec.build_adapter_input(p_pos, p_vel, e_pos, e_vel), z, rtol=0, atol=1e-6)

    def test_single_env_adapter_matches_batched_env_replay(self):
        a = self.meta.adapter
        T, N = self.z.shape[:2]
        quats = torch.nn.functional.normalize(torch.randn(T, N, 4, generator=torch.Generator().manual_seed(0)), dim=-1)

        # Env: one batched adapter constructed like PursuitEvasion.__init__, reset per env on done, then
        # out * 0.1 / (H * dt * substeps) rotated into the body frame.
        ref_cls = type(pec.load_opponent_adapter(str(ARTIFACT / "pursuer"), a))
        ref_adapter = ref_cls(
            N, kind=a.kind, history=a.history, horizon=a.horizon, forgetting=a.forgetting, delta=a.delta,
            model_path=str(ARTIFACT / "pursuer" / pec.ADAPTER_BASIS_FILENAME), freeze=a.freeze,
            reset_on_episode=a.reset_on_episode, p_max_ratio=a.p_max_ratio, target=a.target)
        ref = []
        for t in range(T):
            if t > 0 and self.done[t - 1].any():
                ref_adapter.reset(self.done[t - 1].nonzero().squeeze(-1))
            out = ref_adapter.step(self.z[t]) * (0.1 / (a.horizon * self.dt))
            ref.append(torch.einsum("nji,nj->ni", _rotmat(quats[t]), out))
        ref = torch.stack(ref)

        # Deploy: a fresh single-env adapter per episode (one controller run per episode).
        dep = torch.zeros_like(ref)
        for i in range(N):
            adapter = None
            for t in range(T):
                if adapter is None or (t > 0 and self.done[t - 1, i]):
                    adapter = pec.load_opponent_adapter(str(ARTIFACT / "pursuer"), a)
                dep[t, i] = pec.adapter_step(adapter, a, self.z[t, i], quats[t, i], self.meta.ctbr.dt)

        torch.testing.assert_close(dep, ref, rtol=1e-4, atol=1e-5)
        self.assertGreater(float(ref[a.horizon + 1:].abs().mean()), 1e-3)   # the estimate is not trivially zero
        self.assertFalse(torch.allclose(adapter.theta, adapter.theta0))      # and the RLS readout adapted online


def _state(pos, vel, quat=(1.0, 0.0, 0.0, 0.0)):
    return pec.DroneState(pos=np.asarray(pos, float), quat_wxyz=np.asarray(quat, float),
                          lin_vel=np.asarray(vel, float), ang_vel=np.zeros(3), stamp=0.0)


@unittest.skipUnless(HAVE_ARTIFACT, "F14 RLS artifact missing")
class PolicyAgentTest(unittest.TestCase):

    def test_pursuer_observation_and_actions(self):
        agent = pe_ctrl.PolicyAgent(str(ARTIFACT / "pursuer"))
        evader = pe_ctrl.PolicyAgent(str(ARTIFACT / "evader"))
        self.assertIsNotNone(agent.adapter)
        self.assertIsNone(evader.adapter)
        origin = np.zeros(3)
        for k in range(30):
            p = _state([0.1 * k * 0.02, 0.0, 1.0], [0.1, 0.0, 0.0])
            e = _state([1.0, 0.5 * np.sin(0.2 * k), 1.0], [0.0, 0.1 * np.cos(0.2 * k), 0.0])
            prev = agent.previous_action
            action, command, obs = agent.act(p, e, origin)
            e_action, _, e_obs = evader.act(e, p, origin)
            self.assertEqual(obs.shape, (31,))
            self.assertEqual(e_obs.shape, (28,))
            self.assertTrue(torch.isfinite(action).all() and torch.isfinite(e_action).all())
            torch.testing.assert_close(obs[24:28], prev)          # previous action, then the adapter slot
            agent.previous_action = action
        self.assertGreater(float(obs[28:].abs().sum()), 0.0)


class ConfigTest(unittest.TestCase):

    def _yaml(self, name):
        return pe_ctrl._read_yaml(str(DEPLOY / f"{name}.yaml"))

    def test_all_pe_configs_parse(self):
        for name in PE_CONFIGS:
            with self.subTest(name):
                cfg = pe_ctrl.ControllerConfig.from_yaml_dict(self._yaml(name), name)
                self.assertTrue(cfg.policy.artifact_dir)
                pec.load_drone_config_from_yaml(str(DEPLOY / f"{name}.yaml"), "pursuer")
                pec.load_mocap_config_from_yaml(str(DEPLOY / f"{name}.yaml"))

    def test_defaults_when_sections_are_omitted(self):
        cfg = pe_ctrl.ControllerConfig.from_yaml_dict({"policy": {"artifact_dir": "x"}}, "t")
        self.assertEqual(cfg.opponent.mode, "scripted")
        self.assertEqual(cfg.episode.min_altitude, 0.15)
        self.assertTrue(cfg.episode.stop_on_timeout)
        self.assertFalse(cfg.logging.publish_tf)
        self.assertEqual(cfg.flight.land_duration, 3.0)
        self.assertIsNone(cfg.opponent.scripted.anchor)

    def test_rejects_old_layout_and_typos(self):
        with self.assertRaisesRegex(ValueError, "old layout"):
            pe_ctrl.ControllerConfig.from_yaml_dict({"artifact_dir": "x", "controller": {}}, "t")
        with self.assertRaisesRegex(ValueError, "min_altitud"):
            pe_ctrl.ControllerConfig.from_yaml_dict(
                {"policy": {"artifact_dir": "x"}, "episode": {"min_altitud": 0.2}}, "t")
        with self.assertRaisesRegex(ValueError, "opponent.artifact_dir"):
            pe_ctrl.ControllerConfig.from_yaml_dict(
                {"policy": {"artifact_dir": "x"}, "opponent": {"mode": "policy"}}, "t")
        with self.assertRaisesRegex(ValueError, "scripted.type"):
            pe_ctrl.ControllerConfig.from_yaml_dict(
                {"policy": {"artifact_dir": "x"}, "opponent": {"scripted": {"type": "zigzag"}}}, "t")

    @unittest.skipUnless(HAVE_ARTIFACT, "F14 RLS artifact missing")
    def test_controller_builds_from_main_config(self):
        cwd = os.getcwd()
        os.chdir(DEPLOY)
        try:
            c = pe_ctrl.PursuitEvasionController(argparse.Namespace(config="config_pursuit_evasion.yaml"))
        finally:
            os.chdir(cwd)
        self.assertEqual((c.role, c.opponent_role, c.opponent_mode), ("pursuer", "evader", "policy"))
        self.assertIsNotNone(c.agent.adapter)
        self.assertEqual(c.opponent_agent.role, "evader")
        self.assertAlmostEqual(c.control_dt, 0.02)
        self.assertAlmostEqual(c.capture_radius, c.metadata.capture_radius)


if __name__ == "__main__":
    unittest.main()
