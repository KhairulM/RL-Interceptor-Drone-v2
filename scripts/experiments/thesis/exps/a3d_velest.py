"""A3d: learned estimate of the evader's velocity (and acceleration) from noisy positions (user choice 10-29, after
A3b/A3c; redesign of F4). docs/research_plan.md, pre-registered 2026-10-29 before any A3d result.

Offline. A3a trajectories (X3 generalist seed 0, float32). Input window z_pos = [rel_pos, own_pos, own_vel] (K = 10,
no evader velocity), target = the evader's current world velocity. Learned frozen basis (per-evader heads, shared
head used at run time), trained on clean ("clean") or noise-augmented ("aug") inputs. Baselines: first difference of
the evader position estimate, raw and EMA-filtered (alpha tuned on noisy val); reference: the measured velocity of
A3c (true relative velocity + rel_lin_vel_std noise). Pre-registered: the policy uses "aug" if its noisy-val error is
lower than "clean"'s and its clean-val error <= 1.1x "clean"'s (A3a rule); the EMA arm uses the tuned alpha.
Closed loop. Train pool, uniform, 150M frames, noise-free training (as A3b), 3 seeds. The estimated relative velocity
replaces the true one in the observation (`observation.rel_vel_source`). V_ema: tuned EMA finite difference. V_nf:
learned velocity + the A3a learned acceleration estimate (as P_nf), sharing one noisy measurement per step. Eval
clean and noisy on train + test. References: M0 clean/noisy (A3c), P_none (A3b).
"""

import json
import os

import numpy as np
import torch

from a_adapt import HISTORY, STEP_DT, _J, _adapt_module, test_set
from a3_posonly import (A3A, A3B, A3C, NOISE_STD, NOISY, add_noise, basis_predict, episode_ids, fit_basis, load_traj,
                        windows)
from common import OUT, evaluate, log_progress, save_json, train
from x3_gap import EXP as X3, GEN_FRAMES, load_split

A3D = os.path.join(OUT, "A3d_velest")
EMA_GRID = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
REL_VEL_STD = 0.08          # observation.noise.rel_lin_vel_std (A3c's measured velocity)


def fd_ema_velocity(zpos, ep, alpha):
    """EvaderVelEstimator("fd_ema") run over a recorded batch [T1, M, 9]; returns [T1, M, 3]."""
    oa = _adapt_module()
    T1, M = zpos.shape[:2]
    est = oa.EvaderVelEstimator(M, "fd_ema", STEP_DT, ema_alpha=alpha)
    out = torch.zeros(T1, M, 3)
    for t in range(T1):
        if t > 0:
            new = (ep[t] != ep[t - 1]).nonzero().squeeze(-1)
            if new.numel():
                est.reset(new)
        out[t] = est.step(zpos[t])
    return out


def offline():
    path = os.path.join(A3D, "choice.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    tr_z, tr_v, _, tr_done, tr_opp = load_traj(os.path.join(A3A, "rollouts", "traj_train.pt"))
    te_z, te_v, _, te_done, _ = load_traj(os.path.join(A3A, "rollouts", "traj_test.pt"))
    tr_ep, te_ep = episode_ids(tr_done), episode_ids(te_done)
    M = tr_z.shape[1]
    val_cols = torch.randperm(M, generator=torch.Generator().manual_seed(0))[: M // 5]   # same split as A3a
    is_val = torch.zeros(M, dtype=torch.bool)
    is_val[val_cols] = True
    mean, std = tr_z.reshape(-1, 9).mean(0), tr_z.reshape(-1, 9).std(0).clamp(min=1e-3)
    names = sorted(set(tr_opp))
    col_head = torch.tensor([names.index(o) for o in tr_opp])
    noisy = {"train": add_noise(tr_z, 1), "test": add_noise(te_z, 2)}
    X, _, valid = windows(tr_z, tr_v, tr_ep, 0, HISTORY, mean, std)
    Xn, _, _ = windows(noisy["train"], tr_v, tr_ep, 0, HISTORY, mean, std)
    Xt, _, vt = windows(te_z, te_v, te_ep, 0, HISTORY, mean, std)
    Xtn, _, _ = windows(noisy["test"], te_v, te_ep, 0, HISTORY, mean, std)
    val = valid & is_val.expand_as(valid)
    heads = col_head.expand_as(valid)[valid]
    vmask = is_val.expand_as(valid)[valid]
    models = {b: fit_basis(Xin[valid], tr_v[valid], heads, vmask, 0, mean, std,
                           os.path.join(A3D, "models", f"vel_nf_{b}.pt"), target="vel")
              for b, Xin in (("clean", X), ("aug", Xn))}
    res = {}
    g = torch.Generator().manual_seed(3)
    for part, (Xc, Xnn, Y, ok, zc, zn, ep) in {
            "val": (X, Xn, tr_v, val, tr_z, noisy["train"], tr_ep),
            "test": (Xt, Xtn, te_v, vt, te_z, noisy["test"], te_ep)}.items():
        for cond, Xin, z in (("clean", Xc, zc), ("noisy", Xnn, zn)):
            r = lambda pred: float((Y - pred)[ok].norm(dim=-1).mean())
            res[f"{part}/{cond}/zero"] = r(torch.zeros_like(Y))
            res[f"{part}/{cond}/FD_raw"] = r(fd_ema_velocity(z, ep, 1.0))
            for b, m in models.items():
                pred = torch.zeros_like(Y)
                pred[ok] = basis_predict(m, Xin[ok])
                res[f"{part}/{cond}/NF_{b}"] = r(pred)
        res[f"{part}/noisy/measured_vel"] = float((torch.randn(Y[ok].shape, generator=g) * REL_VEL_STD).norm(dim=-1).mean())
    grid = {str(a): float((tr_v - fd_ema_velocity(noisy["train"], tr_ep, a))[val].norm(dim=-1).mean()) for a in EMA_GRID}
    alpha = float(min(grid, key=grid.get))
    for part, (Y, ok, z, ep) in {"val": (tr_v, val, noisy["train"], tr_ep), "test": (te_v, vt, noisy["test"], te_ep)}.items():
        res[f"{part}/noisy/FD_ema"] = float((Y - fd_ema_velocity(z, ep, alpha))[ok].norm(dim=-1).mean())
        res[f"{part}/clean/FD_ema"] = float((Y - fd_ema_velocity(te_z if part == "test" else tr_z, ep, alpha))[ok]
                                            .norm(dim=-1).mean())
    use_aug = (res["val/noisy/NF_aug"] < res["val/noisy/NF_clean"]) and (
        res["val/clean/NF_aug"] <= 1.1 * res["val/clean/NF_clean"])
    choice = {"basis": "aug" if use_aug else "clean", "ema_alpha": alpha, "ema_grid_val_noisy": grid,
              "model_path": os.path.join(A3D, "models", f"vel_nf_{'aug' if use_aug else 'clean'}.pt")}
    save_json(os.path.join(A3D, "pred_err.json"), res)
    keys = ["zero", "FD_raw", "FD_ema", "NF_clean", "NF_aug"]
    lines = ["# A3d_velest offline summary", "",
             "Mean error |v - v_hat| (m/s) of the evader's current world velocity from a 10-step window of "
             "[rel_pos, own_pos, own_vel] (no evader velocity). noisy = sensor-noise model on the inputs (0.02 m, 0.02 m, "
             f"0.05 m/s). FD = first difference of the evader position estimate; FD_ema alpha = {alpha} (tuned on noisy val). "
             f"measured_vel = error of A3c's noisy measured velocity (rel_lin_vel_std {REL_VEL_STD}).", "",
             "| set | inputs | " + " | ".join(keys) + " | measured_vel |", "|---" * (len(keys) + 3) + "|"]
    for part in ("val", "test"):
        for cond in ("clean", "noisy"):
            mv = f"{res[f'{part}/noisy/measured_vel']:.4f}" if cond == "noisy" else "-"
            lines.append(f"| {part} | {cond} | " + " | ".join(f"{res[f'{part}/{cond}/{k}']:.4f}" for k in keys) + f" | {mv} |")
    lines += ["", f"Pre-registered choice: {choice}"]
    os.makedirs(A3D, exist_ok=True)
    with open(os.path.join(A3D, "summary_offline.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    save_json(path, choice)
    log_progress(f"A3d offline: {choice['basis']} basis, EMA alpha {alpha}; noisy test NF_aug "
                 f"{res['test/noisy/NF_aug']:.3f} vs FD_ema {res['test/noisy/FD_ema']:.3f} m/s")
    return choice


def methods(choice=None):
    """{arm: pursuer overrides}; also used by F4 (with noise) and its frozen-opponent mirror."""
    c = choice or offline()
    with open(os.path.join(A3A, "choice.json")) as f:
        a = json.load(f)
    acc = ["task.adapt.enabled=true", "task.adapt.kind=nf", "task.adapt.z_mode=pos", "task.adapt.target=dvel",
           "task.adapt.freeze=true", f"task.adapt.horizon={a['H_star']}", f"task.adapt.history={HISTORY}",
           f"task.adapt.model_path={a['model_path']}"]
    return {"V_ema": ["task.observation.rel_vel_source=fd_ema", f"task.observation.rel_vel_ema={c['ema_alpha']}"],
            "V_nf": ["task.observation.rel_vel_source=nf", f"task.observation.rel_vel_model={c['model_path']}",
                     f"task.observation.rel_vel_history={HISTORY}"] + acc}


def run_a3d(args):
    choice = offline()
    if args.smoke:
        return
    split = load_split()
    test = test_set(split)
    for s in range(3):
        for name, ov in methods(choice).items():
            d = os.path.join(A3D, name)
            ck = train(os.path.join(d, "ckpt", f"seed{s}.pt"), split["train"], GEN_FRAMES, seed=s, extra=ov)
            log_progress(f"A3d {name} seed{s}: {ck or 'run by another lane / failed'}")
            if not ck:
                continue
            for cond, extra in (("clean", []), ("noisy", NOISY)):
                for part, opps in (("train", split["train"]), ("test", test)):
                    evaluate(os.path.join(d, "evals", f"seed{s}_{cond}_{part}.json"), opps, policy_ckpt=ck,
                             extra=ov + extra)
    analyse(split, test)


def analyse(split, test):
    import analysis as an
    groups = {"train": [e["id"] for e in split["train"]], "heldout_rl": [e["id"] for e in split["heldout_rl"]],
              "heldout_scripted": [e["id"] for e in split["heldout_scripted"]],
              "exploiter": [e["id"] for e in test if e["id"].startswith("exploit/")]}

    def files(name, s, cond):
        if name == "M0":
            if cond == "clean":
                return [os.path.join(X3, "evals", f"generalist{s}_{p}.json") for p in ("train", "test")]
            return [os.path.join(A3C, "M0", f"seed{s}_noisy_{p}.json") for p in ("train", "test")]
        root = A3B if name.startswith("P_") else A3D
        return [os.path.join(root, name, "evals", f"seed{s}_{cond}_{p}.json") for p in ("train", "test")]

    per = {}
    for name in ("M0", "P_none", "V_ema", "V_nf"):
        for cond in ("clean", "noisy"):
            rows = []
            for s in range(3):
                J = {}
                for f in files(name, s, cond):
                    J.update(_J(f))
                if J and all(i in J for i in groups["train"]) and all(i in J for i in groups["heldout_rl"]):
                    rows.append({g: float(np.mean([J[i] for i in ids if i in J])) for g, ids in groups.items()})
                else:
                    rows.append(None)
            per[f"{name}/{cond}"] = rows
    agg = {k: {g: an.bootstrap_ci([r[g] for r in v if r]) for g in groups} | {"n": sum(r is not None for r in v)}
           for k, v in per.items() if any(v)}
    pairs = [("V_nf/noisy", "V_ema/noisy"), ("V_nf/clean", "V_ema/clean"), ("V_nf/noisy", "P_none/noisy"),
             ("V_ema/noisy", "P_none/noisy"), ("V_nf/noisy", "M0/noisy"), ("V_ema/noisy", "M0/noisy")]
    diffs = {}
    for a, b in pairs:
        ok = [i for i in range(3) if per.get(a, [None] * 3)[i] and per.get(b, [None] * 3)[i]]
        if ok:
            diffs[f"{a} - {b}"] = {g: an.bootstrap_ci([per[a][i][g] - per[b][i][g] for i in ok]) for g in groups} | {"n": len(ok)}
    save_json(os.path.join(A3D, "results.json"), {"per_seed": per, "aggregate": agg, "paired": diffs})
    fmt = lambda t, sign="": f"{t[0]:{sign}.3f} [{t[1]:{sign}.3f}, {t[2]:{sign}.3f}]"
    lines = ["# A3d_velest summary", "", "Mean J [95 % bootstrap CI over seeds 0-2]. Trained noise-free; clean / noisy "
             "eval. V_* observe an estimated evader velocity (no true velocity); M0 observes the true (noisy) one; P_none "
             "none.", "", "| policy / eval | n | " + " | ".join(groups) + " |", "|---" * (len(groups) + 2) + "|"]
    for k, r in agg.items():
        lines.append(f"| {k} | {r['n']} | " + " | ".join(fmt(r[g]) for g in groups) + " |")
    lines += ["", "Seed-paired differences:", "", "| comparison | n | " + " | ".join(groups) + " |",
              "|---" * (len(groups) + 2) + "|"]
    for k, r in diffs.items():
        lines.append(f"| {k} | {r['n']} | " + " | ".join(fmt(r[g], "+") for g in groups) + " |")
    with open(os.path.join(A3D, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
