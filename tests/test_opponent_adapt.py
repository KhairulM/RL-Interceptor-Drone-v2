import importlib.util
from pathlib import Path
import unittest

import torch

_PATH = Path(__file__).resolve().parents[1] / "omni_drones/utils/opponent_adapt.py"
_SPEC = importlib.util.spec_from_file_location("opponent_adapt", _PATH)
oa = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(oa)

DT = 0.02


def _z(t, v, p0):
    """Constant relative velocity v [N,3]: rel_pos = p0 + v t dt; z = [rel_pos, v, 0]."""
    return torch.cat([p0 + v * t * DT, v, torch.zeros_like(v)], dim=-1)


class OpponentAdapterTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.v = torch.randn(4, 3, generator=g)
        self.p0 = torch.randn(4, 3, generator=g)

    def test_horizon_alignment_with_frozen_zero_readout(self):
        H = 5
        ad = oa.OpponentAdapter(4, kind="raw", history=3, horizon=H, freeze=True)
        for t in range(H):
            ad.step(_z(t, self.v, self.p0))
            self.assertFalse(ad.err_valid.any())
        ad.step(_z(H, self.v, self.p0))
        self.assertTrue(ad.err_valid.all())
        torch.testing.assert_close(ad.err, (self.v * H * DT).norm(dim=-1), rtol=1e-5, atol=1e-6)
        self.assertTrue((ad.pred == 0).all())

    def test_raw_rls_learns_linear_displacement(self):
        H = 5
        ad = oa.OpponentAdapter(4, kind="raw", history=3, horizon=H, forgetting=0.99, delta=100.0)
        for t in range(200):
            ad.step(_z(t, self.v, self.p0))
        torch.testing.assert_close(ad.pred, self.v * H * DT, rtol=1e-2, atol=1e-3)
        self.assertLess(ad.err.max().item(), 1e-3)

    def test_esn_runs_and_reduces_error(self):
        ad = oa.OpponentAdapter(4, kind="esn", history=3, horizon=5, reservoir_size=32)
        errs = []
        for t in range(300):
            ad.step(_z(t, self.v, self.p0))
            if ad.err_valid.all():
                errs.append(ad.err.mean().item())
        self.assertLess(errs[-1], 0.5 * errs[0])

    def test_reset_semantics(self):
        for keep in (False, True):
            ad = oa.OpponentAdapter(4, kind="raw", history=3, horizon=5, reset_on_episode=not keep)
            for t in range(50):
                ad.step(_z(t, self.v, self.p0))
            theta = ad.theta.clone()
            ad.reset(torch.tensor([0, 1]))
            self.assertTrue((ad.hist[:2] == 0).all())
            self.assertTrue((ad.t[:2] == 0).all())
            if keep:
                torch.testing.assert_close(ad.theta, theta)
            else:
                self.assertTrue((ad.theta[:2] == 0).all())
                torch.testing.assert_close(ad.theta[2:], theta[2:])

    def test_covariance_trace_is_bounded(self):
        ad = oa.OpponentAdapter(4, kind="raw", history=3, horizon=2, forgetting=0.9, delta=1.0, p_max_ratio=1.0)
        z = _z(0, torch.zeros(4, 3), self.p0)            # stationary target: no excitation -> wind-up without the cap
        for t in range(300):
            ad.step(z)
        tr = ad.P.diagonal(dim1=-2, dim2=-1).sum(-1)
        self.assertTrue((tr <= ad.F * 1.0 + 1e-3).all())

    def test_dvel_target_learns_constant_acceleration(self):
        # Evader accelerates at a (per env); pursuer still: rel_vel = v0 + a t dt, own_vel = 0 -> dvel over H = a H dt.
        H, a = 5, torch.randn(4, 3)
        ad = oa.OpponentAdapter(4, kind="raw", history=3, horizon=H, target="dvel", forgetting=0.99, delta=100.0)
        for t in range(300):
            v = self.v + a * t * DT
            ad.step(torch.cat([self.p0, v, torch.zeros_like(v)], dim=-1))
        torch.testing.assert_close(ad.pred, a * H * DT, rtol=1e-2, atol=1e-3)

    def test_mask_only_advances_selected_envs(self):
        ad = oa.OpponentAdapter(4, kind="raw", history=3, horizon=2)
        ad.step(_z(0, self.v, self.p0), mask=torch.tensor([True, False, True, False]))
        self.assertEqual(ad.t.tolist(), [1, 0, 1, 0])

    def test_stack_outputs_history(self):
        ad = oa.OpponentAdapter(4, kind="stack", history=2)
        ad.step(_z(0, self.v, self.p0))
        out = ad.step(_z(1, self.v, self.p0))
        self.assertEqual(out.shape, (4, 2 * oa.Z_DIM))
        torch.testing.assert_close(out[:, oa.Z_DIM:], _z(1, self.v, self.p0))


if __name__ == "__main__":
    unittest.main()


def test_fd_ema_velocity_estimator_exact_for_constant_velocity():
    # Evader at p_e = p0 + v t dt, pursuer still at the origin: z_pos = [rel_pos, own_pos, own_vel].
    g = torch.Generator().manual_seed(1)
    v, p0 = torch.randn(4, 3, generator=g), torch.randn(4, 3, generator=g)
    est = oa.EvaderVelEstimator(4, "fd_ema", DT, ema_alpha=0.3)
    zero = torch.zeros(4, 3)
    out0 = est.step(torch.cat([p0, zero, zero], -1)).clone()
    torch.testing.assert_close(out0, zero)
    for t in range(1, 20):
        out = est.step(torch.cat([p0 + v * t * DT, zero, zero], -1))
        torch.testing.assert_close(out, v, rtol=1e-4, atol=1e-4)
    est.reset(torch.tensor([0, 2]))
    out = est.step(torch.cat([p0, zero, zero], -1), mask=torch.tensor([True, False, True, False]))
    torch.testing.assert_close(out[[0, 2]], torch.zeros(2, 3))
    torch.testing.assert_close(out[[1, 3]], v[[1, 3]], rtol=1e-4, atol=1e-4)


def test_nf_velocity_target_is_frozen_readout_of_history(tmp_path):
    K, d, hidden = 3, 4, 8
    phi = oa.BasisNet(K * oa.Z_DIM, d, hidden)
    theta0 = torch.randn(3, d + 1)
    path = tmp_path / "vel.pt"
    torch.save({"phi": phi.state_dict(), "theta0": theta0, "z_mean": torch.zeros(9), "z_std": torch.ones(9),
                "in_dim": K * oa.Z_DIM, "d": d, "hidden": hidden, "horizon": 0, "history": K, "target": "vel"}, path)
    est = oa.EvaderVelEstimator(2, "nf", DT, model_path=str(path), history=K)
    zs = [torch.randn(2, 9) for _ in range(4)]
    for z in zs:
        out = est.step(z)
    f = phi(torch.stack(zs[-K:], 1).flatten(1))
    f = torch.cat([f, torch.ones_like(f[:, :1])], -1)
    torch.testing.assert_close(out, f @ theta0.T)
    try:
        oa.OpponentAdapter(2, kind="nf", history=K, horizon=0, model_path=str(path), target="vel")
        raise AssertionError("target=vel without freeze must fail")
    except ValueError:
        pass
