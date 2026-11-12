"""Fake Isaac backend for running PursuitEvasion without the simulator: stub modules + toy deterministic physics.

install(root) puts the stubs into sys.modules; load_env(path, name) loads an env source file under a unique name.
"""
import importlib.util
import sys
import types

import torch
from torchrl.data import UnboundedContinuous

from omni_drones.utils.torch import normalize, quat_rotate_inverse


# ----------------------------------------------------------------------------- drones / controllers
class FakeDrone:
    def __init__(self, name, n, origins):
        self.name, self.n = name, n
        self.params = {"mass": 0.0321}
        self.action_spec = UnboundedContinuous(torch.Size([4]))
        self.intrinsics_spec_flattened = UnboundedContinuous(torch.Size([12]))
        self.origins = origins                       # set by the env after _design_scene
        self._pos = torch.zeros(n, 1, 3)
        self._rot = torch.tensor([1.0, 0, 0, 0]).repeat(n, 1, 1)
        self._vel = torch.zeros(n, 1, 6)
        self._cmd = torch.zeros(n, 1, 4)
        self.pos = torch.zeros(n, 1, 3)
        self.rot = torch.zeros(n, 1, 4)
        self.vel = self.vel_w = torch.zeros(n, 1, 6)
        self.vel_b = torch.zeros(n, 1, 6)
        self.mass_scale = torch.ones(n, 1, 1)
        g = torch.Generator().manual_seed(sum(map(ord, name)))
        self.intrinsics = {k: torch.rand(n, 1, d, generator=g) for k, d in
                           (("mass", 1), ("inertia", 3), ("com", 3), ("KF", 1), ("KM", 1), ("tau_up", 1),
                            ("tau_down", 1), ("drag_coef", 1))}

    def initialize(self): pass
    def setup_randomization(self, r): pass
    def spawn(self, translations): pass

    def _reset_idx(self, env_ids, training):
        self.mass_scale[env_ids] = 0.95 + 0.1 * torch.rand(len(env_ids), 1, 1)     # consumes RNG like the real DR

    def set_world_poses(self, pos, rot, env_ids):
        self._pos[env_ids] = pos
        self._rot[env_ids] = rot

    def set_velocities(self, vel, env_ids):
        self._vel[env_ids] = vel

    def get_state(self):
        self.pos[:] = self._pos - self.origins.unsqueeze(1)
        self.rot[:] = self._rot
        self.vel_w[:] = self._vel
        self.vel_b[:] = torch.cat([quat_rotate_inverse(self._rot, self._vel[..., :3]),
                                   quat_rotate_inverse(self._rot, self._vel[..., 3:])], -1)
        return torch.cat([self.pos, self.rot, self.vel, self._cmd[..., :2]], -1)

    def apply_action(self, cmds):
        self._cmd = cmds.clone()
        return cmds.abs().sum(-1)

    def sim_step(self, dt):
        c = torch.nan_to_num(self._cmd, 0.0).clamp(-3, 3)
        acc = torch.cat([1.5 * c[..., 0:1], 1.5 * c[..., 1:2], 3.0 * c[..., 3:4] - 1.0], -1) / self.mass_scale
        self._vel[..., :3] += acc * dt
        self._vel[..., 3:] = 0.9 * self._vel[..., 3:] + 0.5 * c[..., :3]
        self._pos += self._vel[..., :3] * dt
        w = self._vel[..., 3:]
        dq = torch.cat([torch.ones_like(w[..., :1]), 0.5 * dt * w], -1)
        r = self._rot
        q = torch.stack([
            r[..., 0] * dq[..., 0] - (r[..., 1:] * dq[..., 1:]).sum(-1),
            r[..., 0] * dq[..., 1] + dq[..., 0] * r[..., 1] + r[..., 2] * dq[..., 3] - r[..., 3] * dq[..., 2],
            r[..., 0] * dq[..., 2] + dq[..., 0] * r[..., 2] + r[..., 3] * dq[..., 1] - r[..., 1] * dq[..., 3],
            r[..., 0] * dq[..., 3] + dq[..., 0] * r[..., 3] + r[..., 1] * dq[..., 2] - r[..., 2] * dq[..., 1]], -1)
        self._rot = normalize(q)


class FakeRateController:
    def __init__(self, n):
        self.max_thrusts = torch.full((4,), 0.15)
        self.target_clip, self.min_thrust_ratio, self.max_thrust_ratio, self.LPF_coef = 0.5, 0.0, 0.9, 0.5
        self.i = torch.zeros(n, 1, 3)

    def __call__(self, state, target_rate, target_thrust, reset_pid):
        self.i = torch.where(reset_pid.unsqueeze(-1), torch.zeros_like(self.i), self.i) + 1e-4 * target_rate
        return torch.cat([0.002 * target_rate + self.i, target_thrust / 2 ** 16], -1), None


class FakeGeometricCTBR:
    def __init__(self, **kw):
        self.kw = kw
        self.integ = None

    def compute(self, state, target_pos, target_vel, target_yaw, done):
        e = target_pos - state[:, :3]
        if self.integ is None:
            self.integ = torch.zeros(state.shape[0], 1)
        self.integ = torch.where(done.reshape(-1, 1), torch.zeros_like(self.integ), self.integ + e[:, 2:3])
        return torch.cat([0.4 * e[:, 0:1] + 0.1 * (target_vel[:, 0:1] - state[:, 7:8]),
                          0.4 * e[:, 1:2] + 0.1 * (target_vel[:, 1:2] - state[:, 8:9]),
                          0.2 * torch.sin(target_yaw).reshape(-1, 1),
                          0.6 * e[:, 2:3] + 0.01 * self.integ + 0.1 * self.kw["max_tilt_deg"] / 22], -1)


class FakeAPF:
    def __init__(self, ctbr, task_cfg): pass

    def guidance(self, pos, vel, evader_pos, evader_vel, done):
        return evader_pos + 0.3 * evader_vel, 0.5 * evader_vel


class FakeBank:
    def __init__(self, policies, obs_spec, action_spec, reward_spec, algo_cfg, device):
        self.policies = list(policies)
        self.ids = [p["id"] for p in self.policies]
        self.obs_dim = obs_spec[("agents", "observation")].shape[-1]
        self.W = torch.randn(self.obs_dim, 4, generator=torch.Generator().manual_seed(7)) * 0.2

    def assign(self, probs, n):
        self.env_policy = torch.arange(n) % len(self.policies)
        self.assignment = {pid: (self.env_policy == i).nonzero().squeeze(-1) for i, pid in enumerate(self.ids)}
        self._masks = {}
        for i, p in enumerate(self.policies):
            if p["kind"] == "heuristic":
                h = p["metadata"]["heuristic"]
                m = self.env_policy == i
                self._masks[h] = self._masks[h] | m if h in self._masks else m
        self._rl = torch.tensor([self.policies[i]["kind"] == "rl" for i in self.env_policy.tolist()])

    def heuristic_masks(self):
        return dict(self._masks)

    @property
    def rl_mask(self):
        return self._rl

    def rl_raw_actions(self, obs):
        return torch.tanh(obs @ self.W) * 2


# ----------------------------------------------------------------------------- env base
class AgentSpec:
    def __init__(self, *a, **kw): self.a, self.kw = a, kw


class FakeIsaacEnv:
    REGISTRY = {}

    def __init__(self, cfg, headless):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.num_envs = cfg.env.num_envs
        self.max_episode_length = cfg.env.max_episode_length
        self.substeps = cfg.sim.substeps
        self.dt = cfg.sim.dt
        self.training = True
        self.batch_size = torch.Size([self.num_envs])
        self.envs_positions = torch.arange(self.num_envs).float()[:, None] * torch.tensor([12.0, 0.0, 0.0])
        self.agent_spec = {}
        self._design_scene()
        for d in (self.pursuer, self.evader):
            d.origins = self.envs_positions
        self.progress_buf = torch.zeros(self.num_envs)
        self._set_specs()

    def reset(self, env_ids=None):
        env_ids = torch.arange(self.num_envs) if env_ids is None else env_ids
        self._reset_idx(env_ids)
        self.progress_buf[env_ids] = 0.0
        return self._compute_state_and_obs()

    def step(self, td):
        for _ in range(self.substeps):
            self._pre_sim_step(td)
            for d in (self.pursuer, self.evader):
                d.sim_step(self.dt)
        self.progress_buf += 1
        out = self._compute_state_and_obs()
        out.update(self._compute_reward_and_done())
        return out


def _multirotor_make(model, controller, device, name):
    n = _multirotor_make.num_envs
    return FakeDrone(name, n, torch.zeros(n, 3)), FakeRateController(n)


def install(num_envs):
    _multirotor_make.num_envs = num_envs

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        m.__path__ = []
        sys.modules[name] = m
        return m

    import omni_drones.utils  # noqa: F401  (real, Isaac-free)
    mod("omni_drones.envs")
    mod("omni_drones.envs.isaac_env", IsaacEnv=FakeIsaacEnv, AgentSpec=AgentSpec)
    mod("omni_drones.robots")
    mod("omni_drones.robots.drone", MultirotorBase=types.SimpleNamespace(make=_multirotor_make))
    mod("omni_drones.utils.kit", create_ground_plane=lambda *a, **k: None)
    mod("omni_drones.controllers", ControllerBase=object)
    mod("omni_drones.controllers.apf_controller", APFController=FakeAPF)
    mod("omni_drones.controllers.intercept_baseline_common", GeometricCTBR=FakeGeometricCTBR)
    mod("omni_drones.utils.selfplay")
    mod("omni_drones.utils.selfplay.opponents", FrozenPolicyBank=FakeBank)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m
