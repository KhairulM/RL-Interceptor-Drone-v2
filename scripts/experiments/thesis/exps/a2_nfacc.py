"""A2a_nfacc / A2b_nfacc: learned estimator of the evader's acceleration (thesis RQ2, F1 option B).

Same learned basis + RLS machinery as A1, new target ("dvel"): the change of the evader's own velocity over the next
H steps, i.e. H dt times its mean acceleration. The policy receives it as an acceleration, scaled exactly like the
finite-difference oracle O2a (x0.1, agent body frame).

A2a (offline, A1a trajectories): H in {1, 5, 10}. Baselines: zero (constant velocity) and finite-difference
persistence (last-step acceleration held for H steps, the offline analogue of O2a). RLS (delta, forgetting) tuned on
train-pool validation envs.
Pre-registered (10-27, before running): H* = horizon with the largest relative error reduction of NF RLS (S1) over
finite-difference persistence on the train-pool validation envs; ties -> 5.
A2b: pursuers with NF-acc frozen and NF-acc + RLS (3 seeds, train pool, uniform, 150M frames, noise-free).
"""

import json
import os

import numpy as np
import torch

from a_adapt import (A1A, HISTORY, STEP_DT, _J, _adapt_module, replay, summarise, test_set, train_basis, tune,
                     val_columns)
from common import OUT, evaluate, log_progress, save_json, train
from x3_gap import EXP as X3, GEN_FRAMES, load_split

A2A = os.path.join(OUT, "A2a_nfacc")
A2B = os.path.join(OUT, "A2b_nfacc")
HORIZONS = [1, 5, 10]
ACC_PREFIX = ["task.adapt.enabled=true", "task.adapt.kind=nf", "task.adapt.target=dvel",
              f"task.adapt.history={HISTORY}"]


def baseline_errors(traj_path, H, cols=None):
    """Errors of the zero (constant-velocity) and finite-difference-persistence predictions of dvel over H."""
    oa = _adapt_module()
    data = torch.load(traj_path)
    z, done = data["z"].float(), data["done"]
    if cols is not None:
        z, done = z[:, cols], done[:, cols]
    T1 = z.shape[0]
    ep = torch.zeros(T1, z.shape[1], dtype=torch.long)
    ep[1:] = torch.cumsum(done.long(), 0)
    v = oa.target_signal(z, "dvel")
    t = torch.arange(T1)
    src, prev = t - H, t - H - 1                     # prediction made at src, using the step before it
    ok = (prev >= 0).unsqueeze(1) & (ep[src.clamp(min=0)] == ep) & (ep[prev.clamp(min=0)] == ep)
    y = v - v[src.clamp(min=0)]
    fd = (v[src.clamp(min=0)] - v[prev.clamp(min=0)]) * H
    return {"zero": (y.norm(dim=-1), ok), "FD_persistence": ((y - fd).norm(dim=-1), ok)}


def run_a2a(args):
    paths = {p: os.path.join(A1A, "rollouts", f"traj_{p}.pt") for p in ("train", "test")}
    for p in paths.values():
        if not os.path.isfile(p):
            raise RuntimeError("A1a trajectories missing; run A1a first")
    cols = val_columns(paths["train"])
    results, tuned = {}, {}
    for H in HORIZONS:
        model_path = os.path.join(A2A, "models", f"nf_dvel_H{H}.pt")
        model = train_basis(paths["train"], H, model_path, target="dvel")
        log_progress(f"A2a basis dvel H={H}: val shared mse {model['train_log'][-1]['val_shared_mse']:.5f}")
        kw = dict(kind="nf", model_path=model_path, target="dvel")
        best, grid = tune(paths["train"], H, kw)
        tuned[str(H)] = {"best": best, "grid": grid}
        save_json(os.path.join(A2A, "tuned_rls.json"), tuned)
        res = {}
        for part, c in (("val", cols), ("test", None)):
            p = paths["train"] if part == "val" else paths["test"]
            for name, (e, ok) in baseline_errors(p, H, c).items():
                res[f"{part}/{name}"] = {"mean": float(e[ok].mean()), "n": int(ok.sum())}
            for name, extra in (("NF_frozen", dict(freeze=True)), ("NF_RLS_S1", best),
                                ("NF_RLS_S2", {**best, "reset_on_episode": False})):
                res[f"{part}/{name}"] = summarise(*replay(p, H, cols=c, **kw, **extra))
        results[str(H)] = res
        save_json(os.path.join(A2A, "pred_err.json"), results)
    gains = {H: 1 - results[str(H)]["val/NF_RLS_S1"]["mean"] / results[str(H)]["val/FD_persistence"]["mean"]
             for H in HORIZONS}
    best_gain = max(gains.values())
    h_star = 5 if abs(gains[5] - best_gain) < 1e-9 else max(gains, key=gains.get)
    save_json(os.path.join(A2A, "horizon_choice.json"), {"relative_gain_vs_FD_on_val": gains, "H_star": h_star})
    names = ["zero", "FD_persistence", "NF_frozen", "NF_RLS_S1", "NF_RLS_S2"]
    lines = ["# A2a_nfacc summary", "",
             "Mean error |y - y_hat| (m/s) of the evader's velocity change over H steps (= H dt x mean acceleration).",
             "val = 20 % train-pool envs not used to train the basis; test = held-out RL, scripted, exploiters.", "",
             "| H | set | " + " | ".join(names) + " |", "|---" * (len(names) + 2) + "|"]
    for H, res in results.items():
        for part in ("val", "test"):
            lines.append(f"| {H} | {part} | " + " | ".join(f"{res[f'{part}/{n}']['mean']:.4f}" for n in names) + " |")
    lines += ["", "Relative error reduction of NF_RLS_S1 vs FD persistence (val): " +
              ", ".join(f"H={h}: {g:+.3f}" for h, g in gains.items()), "", f"**H* = {h_star}** (pre-registered rule).",
              "", "Tuned RLS: " + ", ".join(f"H={h}: {v['best']}" for h, v in tuned.items())]
    with open(os.path.join(A2A, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def a2b_methods():
    with open(os.path.join(A2A, "horizon_choice.json")) as f:
        H = json.load(f)["H_star"]
    with open(os.path.join(A2A, "tuned_rls.json")) as f:
        best = json.load(f)[str(H)]["best"]
    base = ACC_PREFIX + [f"task.adapt.horizon={H}", f"task.adapt.model_path={os.path.join(A2A, 'models', f'nf_dvel_H{H}.pt')}"]
    return {"NFacc_frozen": base + ["task.adapt.freeze=true"],
            "NFacc_OA": base + [f"task.adapt.delta={best['delta']}", f"task.adapt.forgetting={best['forgetting']}"]}


def run_a2b(args):
    split = load_split()
    test = test_set(split)
    frames = 2_000_000 if args.smoke else GEN_FRAMES
    for name, ov in a2b_methods().items():
        for s in range(args.seeds):
            ck = train(os.path.join(A2B, name, "ckpt", f"seed{s}.pt"), split["train"], frames, seed=s, extra=ov)
            log_progress(f"A2b {name} seed{s}: {ck}")
            if not ck:
                continue
            evaluate(os.path.join(A2B, name, "evals", f"seed{s}_train.json"), split["train"], policy_ckpt=ck, extra=ov)
            evaluate(os.path.join(A2B, name, "evals", f"seed{s}_test_S1.json"), test, policy_ckpt=ck, extra=ov)
            if name == "NFacc_OA":
                evaluate(os.path.join(A2B, name, "evals", f"seed{s}_test_S2.json"), test, policy_ckpt=ck,
                         extra=ov + ["task.adapt.reset_on_episode=false"])
    analyse_a2b(split, test, args.seeds)


def analyse_a2b(split, test, n_seeds):
    import analysis as an
    groups = {"train": [e["id"] for e in split["train"]], "heldout_rl": [e["id"] for e in split["heldout_rl"]],
              "heldout_scripted": [e["id"] for e in split["heldout_scripted"]],
              "exploiter": [e["id"] for e in test if e["id"].startswith("exploit/")]}
    x4 = os.path.join(OUT, "X4_oracles", "evals")
    files = {
        "M0": lambda s: [os.path.join(X3, "evals", f"generalist{s}_{p}.json") for p in ("train", "test")],
        "O2a_fd_accel": lambda s: ([os.path.join(x4, f"O2a_accel_seed{s}_{p}.json") for p in ("train", "test", "exploit")]
                                   if os.path.isfile(os.path.join(x4, f"O2a_accel_seed{s}_train.json")) else
                                   [os.path.join(OUT, "F1_final", "SU_acc", "evals", f"seed{s}_{p}.json")
                                    for p in ("train", "test")]),
        "NF_OA_disp (A1b)": lambda s: [os.path.join(A1A.replace("A1a_basis", "A1b_policy"), "NF_OA", "evals",
                                                     f"seed{s}_{p}.json") for p in ("train", "test_S1")],
        "NFacc_frozen": lambda s: [os.path.join(A2B, "NFacc_frozen", "evals", f"seed{s}_{p}.json") for p in ("train", "test_S1")],
        "NFacc_OA/S1": lambda s: [os.path.join(A2B, "NFacc_OA", "evals", f"seed{s}_{p}.json") for p in ("train", "test_S1")],
        "NFacc_OA/S2": lambda s: [os.path.join(A2B, "NFacc_OA", "evals", f"seed{s}_{p}.json") for p in ("train", "test_S2")],
    }
    per, agg = {}, {}
    for m, fs in files.items():
        rows = []
        for s in range(n_seeds):
            J = {}
            for f in fs(s):
                J.update(_J(f))
            if J and all(i in J for i in groups["train"]):
                rows.append({g: float(np.mean([J[i] for i in ids if i in J])) for g, ids in groups.items()
                             if any(i in J for i in ids)})
        per[m] = rows
        agg[m] = {g: dict(zip(("mean", "lo", "hi"), an.bootstrap_ci([r[g] for r in rows if g in r]))) for g in groups}
    save_json(os.path.join(A2B, "results.json"), {"per_seed": per, "aggregate": agg})
    lines = ["# A2b_nfacc summary", "", "Mean J (95 % bootstrap CI over seeds), noise-free.", "",
             "| method | " + " | ".join(groups) + " |", "|---" * (len(groups) + 1) + "|"]
    for m, a in agg.items():
        lines.append(f"| {m} (n={len(per[m])}) | " + " | ".join(
            f"{a[g]['mean']:.3f} [{a[g]['lo']:.3f}, {a[g]['hi']:.3f}]" for g in groups) + " |")
    with open(os.path.join(A2B, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
