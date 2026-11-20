"""F18: recorded trajectories and offline estimator replay (docs/research_plan.md, F18_trajectories; registered
2026-11-18 13:33 before any F18 job; eval only, no training, existing checkpoints and exploiters).

T1 (figures, seed 0): RLS (F14 Stage E, stage 24, live RLS) and F8 (stage 24) vs the F14/F15 switching pair (both
directions, randomized switch) and vs their existing lure (thesis-rule) exploiters.
T2 (replay, seeds 0/100/200): RLS vs strong-16, vs its F14 Stage E cross pool, and vs each switching evader alone.
Analysis: replay OpponentAdapter on the recorded z (live and frozen): prediction error per opponent set and the
relative readout drift ||theta_t - theta_bar||_F / ||theta_bar||_F.
"""

import json
import os

import numpy as np
import torch

import common
from common import OUT, evaluate, log_progress, save_json
import f5_acc_sp as f5
import f8_roncero_sp as f8
import f8d_heldout as f8d
import f14_generalization_sp as f14

F18 = os.path.join(OUT, "F18_trajectories")
SEEDS = f14.SEEDS
REC_ENVS = 512
F8_SEED0 = f14.F8_CAMPS[0]


def _rls_camp(seed):
    return f14._stagee_campaigns(f14._selected_rules())["rls"][SEEDS.index(seed)]


def _paths(name):
    return (os.path.join(F18, "evals", f"{name}.json"), os.path.join(F18, "traj", f"{name}.pt"))


def _record(name, opponents, ck, extra, seed=0, switch=None, rec_envs=REC_ENVS, num_envs=4096):
    out, traj = _paths(name)
    if os.path.isfile(out) and os.path.isfile(traj):
        return
    evaluate(out, opponents, policy_ckpt=ck, extra=list(extra or []) +
             [f"eval_traj_out={traj}", f"eval_traj_envs={rec_envs}"], seed=seed, switch=switch,
             num_envs=num_envs, save_episodes=switch is not None or name.startswith("T1_"))   # per-episode outcomes/switch steps for the figures
    if not os.path.isfile(traj):
        raise RuntimeError(f"F18: no trajectory file for {name}")


def _switch_spec(seed, direction):
    first, second = f14._switch_pair(seed, direction)
    return [first, second], {"first_id": first["id"], "second_id": second["id"], "times_s": [0.5, 1.0, 1.5],
                             "group_id": f"switch:{direction}:randomized", "seed": seed + 401}


def _cross_pool(c):
    _, _, _, jobs = f14._stagee_eval_jobs()
    return next(opps for camp, group, opps, *_ in jobs if camp == c and group == "stageE_cross")


def jobs():
    rls_extra = f5.rls_overrides()
    strong = [p for p in f8d._finals() if p["id"].split("/")[0] in f8d.COMMON]
    out = []
    # T1: figures (seed 0)
    c0 = _rls_camp(0)
    ck_rls0, ck_f8 = f14._ck(c0, 24), f14._ck(F8_SEED0, 24)
    for direction in ("f8_to_f10", "f10_to_f8"):
        opps, sw = _switch_spec(0, direction)
        out.append((f"T1_rls_s0_switch_{direction}", opps, ck_rls0, rls_extra, 1400, sw))
        out.append((f"T1_f8_s0_switch_{direction}", opps, ck_f8, None, 1400, sw))
    lure_rls = os.path.join(f14.F14, "ckpt", f"exploiter_thesis_{c0}.pt")
    lure_f8 = os.path.join(f8.F8, "ckpt", f"exploiter_thesis_{F8_SEED0}.pt")
    out.append(("T1_rls_s0_lure", [{"id": "f18/evader:lure_rls_s0", "kind": "rl", "checkpoint_path": lure_rls,
                                    "metadata": {}}], ck_rls0, rls_extra, 0, None))
    out.append(("T1_f8_s0_lure", [{"id": "f18/evader:lure_f8_s0", "kind": "rl", "checkpoint_path": lure_f8,
                                   "metadata": {}}], ck_f8, None, 0, None))
    # T2: replay sets (all seeds, RLS)
    for seed in SEEDS:
        c = _rls_camp(seed)
        ck = f14._ck(c, 24)
        out.append((f"T2_rls_s{seed}_strong", strong, ck, rls_extra, 0, None))
        out.append((f"T2_rls_s{seed}_cross", _cross_pool(c), ck, rls_extra, 0, None))
        first, second = f14._switch_pair(seed, "f8_to_f10")
        out.append((f"T2_rls_s{seed}_fixed_f8ev", [first], ck, rls_extra, 1400 + seed, None))
        out.append((f"T2_rls_s{seed}_fixed_earlier", [second], ck, rls_extra, 1400 + seed, None))
    for name, opps, ck, extra, _, _ in out:
        for o in opps:
            if o.get("checkpoint_path") and not os.path.isfile(o["checkpoint_path"]):
                raise FileNotFoundError(f"F18 {name}: missing opponent checkpoint {o['checkpoint_path']}")
    return out


# ------------------------------------------------------------------ offline replay
def _adapter_module():
    import importlib.util
    path = os.path.join(common.REPO, "omni_drones", "utils", "opponent_adapt.py")
    spec = importlib.util.spec_from_file_location("opponent_adapt", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rls_params():
    ov = dict(o.split("=", 1) for o in f5.rls_overrides())
    return {"history": int(ov["task.adapt.history"]), "horizon": int(ov["task.adapt.horizon"]),
            "delta": float(ov["task.adapt.delta"]), "forgetting": float(ov["task.adapt.forgetting"]),
            "model_path": ov["task.adapt.model_path"], "target": ov["task.adapt.target"]}


def replay(traj_path, freeze):
    """Replay the estimator on a recorded file. Returns per-env arrays of errors (list of floats per step) and drift."""
    oa = _adapter_module()
    p = _rls_params()
    data = torch.load(traj_path, map_location="cpu", weights_only=False)
    z, done = data["z"].float(), data["done"].bool()           # z [T1, M, 9]; done [T, M]
    T1, M, _ = z.shape
    ad = oa.OpponentAdapter(M, kind="nf", history=p["history"], horizon=p["horizon"], forgetting=p["forgetting"],
                            delta=p["delta"], model_path=p["model_path"], freeze=freeze, target=p["target"])
    theta_bar = ad.theta0.clone()
    tb_norm = float(theta_bar.norm())
    dt = float(data["step_dt"])
    errs = [[] for _ in range(M)]
    drift = {1.0: [], 2.0: [], "end": []}
    t_ep = torch.zeros(M, dtype=torch.long)
    for k in range(T1):
        if k > 0:
            ended = done[k - 1]
            if bool(ended.any()):
                idx = ended.nonzero().squeeze(-1)
                for i in idx.tolist():
                    drift["end"].append(float((ad.theta[i] - theta_bar).norm()) / tb_norm)
                ad.reset(idx)
                t_ep[idx] = 0
        ad.step(z[k])
        t_ep += 1
        for i in ad.err_valid.nonzero().squeeze(-1).tolist():
            errs[i].append(float(ad.err[i]))
        for sec in (1.0, 2.0):
            hit = (t_ep == int(round(sec / dt))).nonzero().squeeze(-1)
            for i in hit.tolist():
                drift[sec].append(float((ad.theta[i] - theta_bar).norm()) / tb_norm)
    return errs, drift, data["opponent"]


def analyse():
    res = {}
    for name in sorted(os.listdir(os.path.join(F18, "traj"))):
        if not name.endswith(".pt") or not name.startswith("T2_"):
            continue
        path = os.path.join(F18, "traj", name)
        row = {}
        for mode, freeze in (("live", False), ("frozen", True)):
            errs, drift, _ = replay(path, freeze)
            flat = [e for env in errs for e in env]
            row[f"err_{mode}"] = float(np.mean(flat)) if flat else None
            row[f"n_{mode}"] = len(flat)
            if mode == "live":
                row["drift_median"] = {str(k): (float(np.median(v)) if v else None) for k, v in drift.items()}
        res[name.removesuffix(".pt")] = row
    save_json(os.path.join(F18, "results.json"), res)
    lines = ["# F18 offline estimator replay", "",
             "Mean |y - y_hat| (m/s) of the estimator replayed on recorded trajectories of the stage-24 RLS pursuer "
             "(live RLS reproduces the pursuer's own estimate; frozen = theta held at theta_bar on the same "
             "trajectories). Drift = median ||theta_t - theta_bar||_F / ||theta_bar||_F.", "",
             "| Recording | Error live | Error frozen | Live vs frozen | Drift 1 s | Drift 2 s | Drift at end |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for k, r in res.items():
        rel = (r["err_live"] - r["err_frozen"]) / r["err_frozen"] if r["err_live"] and r["err_frozen"] else None
        d = r["drift_median"]
        cell = lambda x: f"{x:.3f}" if x is not None else "-"
        lines.append(f"| {k} | {cell(r['err_live'])} | {cell(r['err_frozen'])} | {cell(rel)} | {cell(d['1.0'])} | "
                     f"{cell(d['2.0'])} | {cell(d['end'])} |")
    with open(os.path.join(F18, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log_progress("F18 analysis done")
    return res


def smoke():
    """Gate: one small recording whose live replay error is finite."""
    name = "smoke_rls_s0_strong"
    out, traj = _paths(name)
    if not os.path.isfile(traj):
        c = _rls_camp(0)
        strong = [p for p in f8d._finals() if p["id"].split("/")[0] in f8d.COMMON][:2]
        _record(name, strong, f14._ck(c, 24), f5.rls_overrides(), rec_envs=64, num_envs=256)
    errs, drift, _ = replay(traj, False)
    flat = [e for env in errs for e in env]
    if not flat or not np.isfinite(np.mean(flat)):
        raise RuntimeError("F18 smoke: replay error not finite")
    log_progress(f"F18 smoke passed: replay error {np.mean(flat):.3f} m/s over {len(flat)} steps")


def run_f18(args):
    os.makedirs(os.path.join(F18, "traj"), exist_ok=True)
    smoke()
    if args.smoke:
        return
    for name, opps, ck, extra, seed, sw in jobs():
        _record(name, opps, ck, extra, seed=seed, switch=sw)
        log_progress(f"F18 recorded {name}")
    analyse()
