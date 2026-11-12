"""Old (backup) vs refactored PursuitEvasion on the fake backend: every output tensor must be bit-identical.

usage: PYTHONPATH=<checkout with the refactor> python compare.py [scenario ...]  (set WT below to that checkout)
"""
import json
import os
import sys

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fake_backend as fb  # noqa: E402

WT = "/home/user/Projects/RLInterceptorDrone-refactor"
BACKUP = "/home/user/Projects/RLInterceptorDrone/backups/pursuit_evasion_2026-11-12_pre_refactor"
N, T = 48, 260

fb.install(N)
import types  # noqa: E402
single = types.ModuleType("omni_drones.envs.single")
single.__path__ = []
sys.modules["omni_drones.envs.single"] = single
single.roncero_reward = fb.load_module(f"{BACKUP}/roncero_reward.py", "omni_drones.envs.single.roncero_reward")
single.shaping = fb.load_module(f"{BACKUP}/shaping.py", "omni_drones.envs.single.shaping")
OLD = fb.load_module(f"{BACKUP}/pursuit_evasion.py", "pe_old").PursuitEvasion
NEW = fb.load_module(f"{WT}/omni_drones/envs/single/pursuit_evasion.py", "pe_new").PursuitEvasion

H = lambda name, i: {"id": f"h{i}", "kind": "heuristic", "checkpoint_path": None, "metadata": {"heuristic": name}}
RL = lambda i: {"id": f"rl{i}", "kind": "rl", "checkpoint_path": None, "metadata": {}}
EVADERS = [H("hover", 0), H("circular", 1), H("flee", 2), H("weave", 3), H("zigzag", 4), H("fast_flee", 5),
           H("circular>flee@1.0", 6), RL(7)]
PURSUERS = [H("pure_pursuit", 0), H("apf", 1), H("pursue", 2), H("hover>apf@0.8", 3), RL(4)]
DESC = os.path.join(HERE, "desc.json")

BIG_R = {"success_radius_init": 0.5, "success_radius_end": 0.5, "success_radius_eval": 0.5}
SCENARIOS = {
    "pursuer_thesis": dict(role="pursuer", pol=EVADERS, o={**BIG_R, "observation": {"include_noise": True},
                           "reward_floor_weight": 2.0, "use_distance": True, "shaping_gamma": 0.99}),
    "pursuer_thesis_plain": dict(role="pursuer", pol=EVADERS, o={**BIG_R}),
    "evader_thesis_ablate": dict(role="evader", pol=PURSUERS, o={**BIG_R, "observation": {"include_noise": True},
                                 "ablation": {"truncate_failures": True, "no_timeout_evader": True,
                                              "closing_discount": True}}),
    "pursuer_ablate": dict(role="pursuer", pol=EVADERS, o={**BIG_R, "ablation": {"no_timeout": True,
                           "closing_discount": True, "truncate_failures": True}}),
    "pursuer_roncero": dict(role="pursuer", pol=EVADERS, o={**BIG_R, "reward_mode": "roncero",
                            "observation": {"include_noise": True}}),
    "evader_roncero": dict(role="evader", pol=PURSUERS, o={**BIG_R, "reward_mode": "roncero"}),
    "pursuer_oracle_vel": dict(role="pursuer", pol=EVADERS, o={**BIG_R, "observation": {"include_noise": True},
                               "oracle": {"opp_accel": True, "accel_ema": 0.2, "opp_command": True,
                                          "descriptor_path": DESC}}),
    "pursuer_oracle_pos": dict(role="pursuer", pol=EVADERS, o={**BIG_R, "observation": {"include_noise": True},
                               "oracle": {"opp_accel": True, "accel_source": "pos", "accel_ema": 0.3}}),
    "pursuer_oracle_pos_noema": dict(role="pursuer", pol=EVADERS, o={**BIG_R, "oracle": {
                                     "opp_accel": True, "accel_source": "pos"}}),
    "pursuer_adapt_esn_velest": dict(role="pursuer", pol=EVADERS, o={**BIG_R, "observation": {
                                     "include_noise": True, "rel_vel_source": "fd_ema"},
                                     "adapt": {"enabled": True, "kind": "esn"}}),
    "pursuer_adapt_pos_dvel": dict(role="pursuer", pol=EVADERS, o={**BIG_R, "observation": {
                                   "include_noise": True, "rel_vel_source": "fd_ema"},
                                   "adapt": {"enabled": True, "kind": "raw", "z_mode": "pos", "freeze": True,
                                             "target": "dvel"}}),
    "pursuer_adapt_stack": dict(role="pursuer", pol=EVADERS, o={**BIG_R, "observation": {"include_noise": True},
                                "adapt": {"enabled": True, "kind": "stack", "history": 4}}),
    "evader_oppside": dict(role="evader", pol=PURSUERS, o={**BIG_R, "observation": {
                           "include_noise": True, "opp_rel_vel_source": "fd_ema"},
                           "adapt": {"opp_side": True, "kind": "esn", "target": "dvel"},
                           "oracle": {"opp_side_accel": True}}),
    "esn_both": dict(role="pursuer", pol=EVADERS, o={**BIG_R, "observation": {"include_noise": True,
                     "rot_repr": "quat", "use_rel_dist": False},
                     "esn": {"enabled": True, "roles": ["pursuer", "evader"], "readout": "per_env",
                             "reservoir_size": 40}}),
    "esn_evader_shared": dict(role="evader", pol=[H("apf", 0), H("pure_pursuit", 1)], o={**BIG_R, "esn": {
                              "enabled": True, "roles": ["evader"], "readout": "shared", "reservoir_size": 40,
                              "fit_interval": 20}}),
    "positions_only": dict(role="pursuer", pol=EVADERS, o={**BIG_R, "observation": {
                           "agent_rel_vel": False, "rot_repr": "log", "use_previous_action": False,
                           "use_time_encoding": False}}),
    "evader_opp_drop": dict(role="evader", pol=PURSUERS, o={**BIG_R, "observation": {"opp_drop_rel_vel": True},
                            "reward_bounds_weight": 3.0}),
    "small_radius": dict(role="pursuer", pol=EVADERS, o={}),
    "pursuer_falls": dict(role="pursuer", pol=EVADERS, thrust=0.15, o={**BIG_R, "reward_floor_weight": 1.0}),
    "evader_falls": dict(role="evader", pol=PURSUERS, thrust=0.15, o={**BIG_R}),
    "pursuer_falls_roncero": dict(role="pursuer", pol=EVADERS, thrust=0.15, o={**BIG_R, "reward_mode": "roncero"}),
    "evader_falls_v2d": dict(role="evader", pol=PURSUERS, thrust=0.15, o={**BIG_R, "ablation": {
                             "truncate_failures": True, "no_timeout_evader": True}}),
}


def make_cfg(sc):
    task = OmegaConf.load(f"{WT}/cfg/task/PursuitEvasion.yaml")
    task.pop("defaults")
    task = OmegaConf.merge(task, {"role": sc["role"], "env": {"num_envs": N, "max_episode_length": 120},
                                  "opponent": {"policies": sc["pol"]}, "arena": {"visualize": False}}, sc["o"])
    r = task.success_radius_eval        # the old env still reads the removed curriculum keys: make them constant
    task = OmegaConf.merge(task, {"success_radius_init": r, "success_radius_end": r, "success_radius_lr": 0.0})
    return OmegaConf.create({"task": task, "env": task.env, "sim": {"dt": 0.01, "substeps": 2}})


def run(cls, cfg, thrust=0.4):
    torch.manual_seed(0)
    env = cls(cfg, headless=True)
    torch.manual_seed(1)
    outs = [env.reset()]
    g = torch.Generator().manual_seed(2)
    for t in range(T):
        td = TensorDict({
            ("agents", "action"): torch.randn(N, 1, 4, generator=g) * 0.5 + torch.tensor([0, 0, 0, thrust]),
            ("info", "policy_action"): torch.tanh(torch.randn(N, 1, 4, generator=g)),
            ("stats", "action_error_order1"): torch.rand(N, 1, generator=g),
        }, [N])
        if t % 7 == 3:
            td.set("done", torch.rand(N, 1, generator=g) < 0.05)
        out = env.step(td)
        outs.append(out)
        done = out["done"].reshape(-1)
        if done.any():
            outs.append(env.reset(done.nonzero().squeeze(-1)))
    outs.append(TensorDict({"rng": torch.rand(3)}, []))
    return outs


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())


def compare(name):
    cfg = make_cfg(SCENARIOS[name])
    th = SCENARIOS[name].get("thrust", 0.4)
    old, new = run(OLD, cfg, th), run(NEW, cfg, th)
    assert len(old) == len(new), (len(old), len(new))
    events = {"capture": 0, "crash": 0, "opp_crash": 0, "opp_out_of_bounds": 0, "out_of_bounds": 0, "timeout": 0}
    for i, (a, b) in enumerate(zip(old, new)):
        ka, kb = set(a.keys(True, True)), set(b.keys(True, True))
        assert ka == kb, (i, ka ^ kb)
        for k in sorted(ka, key=str):
            assert same(a[k], b[k]), f"{name}: output {i} key {k} differs; max |d| = " \
                f"{(a[k].float() - b[k].float()).abs().nan_to_num(1e9).max().item() if a[k].shape == b[k].shape else 'shape'}"
        if "done" in ka:
            d = a["done"].reshape(-1)
            for e in events:
                events[e] += int((a["stats", e].reshape(-1)[d] > 0.5).sum())
    return len(old), events


if os.environ.get("MUTATE"):          # sanity check: a 1-ulp-scale change must be detected
    from omni_drones.utils.pursuit_evasion import arena as _ar
    _orig = _ar.Arena.wall_proximity
    _ar.Arena.wall_proximity = lambda self, pos: _orig(self, pos) * (1 + 1e-7)

if __name__ == "__main__":
    with open(DESC, "w") as f:
        json.dump({f"h{i}": [0.1 * i, -0.2 * i, 1.0] for i in range(8)} | {"rl7": [9.0, 8.0, 7.0]}, f)
    names = sys.argv[1:] or list(SCENARIOS)
    bad = 0
    for name in names:
        try:
            n, ev = compare(name)
            print(f"OK   {name:28s} {n} outputs  ended: {ev}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            import traceback
            print(f"FAIL {name:28s} {type(e).__name__}: {e}")
            traceback.print_exc(limit=4)
    print("ALL IDENTICAL" if not bad else f"{bad} scenario(s) differ")
    sys.exit(1 if bad else 0)
