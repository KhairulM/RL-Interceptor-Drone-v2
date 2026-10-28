"""A1a_basis (offline opponent-prediction study) and A1b_policy (pursuers with an online opponent adapter).

A1a
  1. Record trajectories of the X3 generalist (seed 0) against the train pool and against the test set.
  2. Train the learned basis phi (BasisNet) on train-pool trajectories: per-evader linear heads + a shared head
     (theta0), target = evader relative displacement over H steps, H in {5, 15, 25} (0.1 / 0.3 / 0.5 s).
  3. Replay the identical OpponentAdapter code over held-out trajectories: CV extrapolation, raw RLS, ESN RLS,
     NF frozen, NF RLS (reset per episode, S1), NF RLS and raw RLS kept across episodes (S2).
  Pre-registered rule for A1b: H* = horizon with the largest relative error reduction of NF-RLS (S1) over CV on
  held-out trajectories; ties -> 15.
A1b
  Pursuers on the train pool (uniform, 150M frames, 3 seeds): M0h (frame stack), ESN, NF-frozen, NF-OA.
  M0 = X3 generalist and ESO-style acceleration input = X4 O2a are reused. Eval: S1 (reset per episode) and S2
  (adapter kept across the episodes of one env, i.e. against the same evader).
"""

import json
import os

import numpy as np
import torch

from common import OUT, REPO, evaluate, log_progress, save_json, train
from x3_gap import EXP as X3, GEN_FRAMES, generalist_ckpt, load_split

A1A = os.path.join(OUT, "A1a_basis")
A1B = os.path.join(OUT, "A1b_policy")
HORIZONS = [5, 15, 25]
HISTORY = 10
STEP_DT = 0.02


def _adapt_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("opponent_adapt", os.path.join(REPO, "omni_drones/utils/opponent_adapt.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_set(split):
    with open(os.path.join(X3, "exploiters.json")) as f:
        exploiters = json.load(f)
    return split["heldout_rl"] + split["heldout_scripted"] + exploiters


def record(split, smoke=False):
    out = {}
    for part, opps in (("train", split["train"]), ("test", test_set(split))):
        path = os.path.join(A1A, "rollouts", f"traj_{part}.pt")
        if not os.path.isfile(path):
            evaluate(os.path.join(A1A, "evals", f"record_{part}.json"), opps, policy_ckpt=generalist_ckpt(0),
                     extra=[f"eval_traj_out={path}", f"eval_traj_envs={256 if smoke else 1024}"], save_episodes=False)
        out[part] = path
    return out


# ------------------------------------------------------------------ dataset
def windows(z, done, H, K, z_mean, z_std, target="disp"):
    """z [T1, M, 9] (float), done [T, M] -> X [n, K*9] (normalised history, zero-padded), Y [n, 3], col [n], valid mask."""
    T1, M, _ = z.shape
    ep = torch.zeros(T1, M, dtype=torch.long)
    ep[1:] = torch.cumsum(done.long(), 0)
    zn = (z - z_mean) / z_std
    Xs = []
    for j in range(K - 1, -1, -1):           # oldest ... newest
        idx = torch.arange(T1) - j
        ok = (idx >= 0).unsqueeze(1) & (ep[idx.clamp(min=0)] == ep)
        Xs.append(torch.where(ok.unsqueeze(-1), zn[idx.clamp(min=0)], torch.zeros_like(zn)))
    X = torch.stack(Xs, dim=2)               # [T1, M, K, 9]
    tgt = torch.arange(T1) + H
    valid = (tgt < T1).unsqueeze(1) & (ep[tgt.clamp(max=T1 - 1)] == ep)
    oa = _adapt_module()
    sig = oa.target_signal(z, target)
    Y = sig[tgt.clamp(max=T1 - 1)] - sig
    col = torch.arange(M).expand(T1, M)
    return X[valid].flatten(1), Y[valid], col[valid]


def train_basis(traj_path, H, out_path, d=32, hidden=128, epochs=8, max_samples=1_500_000, seed=0, target="disp"):
    if os.path.isfile(out_path):
        return torch.load(out_path)
    oa = _adapt_module()
    torch.manual_seed(seed)
    data = torch.load(traj_path)
    z, done = data["z"].float(), data["done"]
    flat = z.reshape(-1, 9)
    z_mean, z_std = flat.mean(0), flat.std(0).clamp(min=1e-3)
    X, Y, col = windows(z, done, H, HISTORY, z_mean, z_std, target)
    opp_names = sorted(set(data["opponent"]))
    opp_of_col = torch.tensor([opp_names.index(o) for o in data["opponent"]])
    head_idx = opp_of_col[col]
    # Hold out 20 % of envs (columns) for validation.
    g = torch.Generator().manual_seed(seed)
    val_cols = torch.randperm(z.shape[1], generator=g)[: z.shape[1] // 5]
    is_val = torch.isin(col, val_cols)
    tr = (~is_val).nonzero().squeeze(-1)
    if len(tr) > max_samples:
        tr = tr[torch.randperm(len(tr), generator=g)[:max_samples]]
    va = is_val.nonzero().squeeze(-1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    phi = oa.BasisNet(X.shape[1], d, hidden).to(dev)
    heads = torch.nn.Parameter(torch.zeros(len(opp_names), 3, d + 1, device=dev))
    shared = torch.nn.Parameter(torch.zeros(3, d + 1, device=dev))
    opt = torch.optim.Adam(list(phi.parameters()) + [heads, shared], lr=1e-3)
    Xd, Yd, Hd = X.to(dev), Y.to(dev), head_idx.to(dev)

    def loss_on(idx):
        f = phi(Xd[idx])
        f = torch.cat([f, torch.ones_like(f[:, :1])], -1)
        y_task = torch.einsum("nij,nj->ni", heads[Hd[idx]], f)
        y_sh = f @ shared.T
        return ((y_task - Yd[idx]) ** 2).sum(-1).mean(), ((y_sh - Yd[idx]) ** 2).sum(-1).mean()

    log = []
    for ep in range(epochs):
        perm = tr[torch.randperm(len(tr))]
        for b in range(0, len(perm), 4096):
            lt, ls = loss_on(perm[b:b + 4096].to(dev))
            opt.zero_grad()
            (lt + ls).backward()
            opt.step()
        with torch.no_grad():
            vt, vs = loss_on(va[:200_000].to(dev))
        log.append({"epoch": ep, "val_task_mse": vt.item(), "val_shared_mse": vs.item()})
    model = {"phi": {k: v.cpu() for k, v in phi.state_dict().items()}, "theta0": shared.detach().cpu(),
             "z_mean": z_mean, "z_std": z_std, "in_dim": X.shape[1], "d": d, "hidden": hidden,
             "horizon": H, "history": HISTORY, "train_log": log, "opponents": opp_names, "target": target}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model, out_path)
    return model


def norm_only(traj_path, out_path):
    """Input normalisation for the esn/raw/stack kinds (same statistics as the NF basis)."""
    if not os.path.isfile(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        z = torch.load(traj_path)["z"].float().reshape(-1, 9)
        torch.save({"z_mean": z.mean(0), "z_std": z.std(0).clamp(min=1e-3)}, out_path)
    return out_path


# ------------------------------------------------------------- offline replay
def val_columns(traj_path, seed=0):
    """The 20 % of recorded envs held out from basis training (same draw as train_basis)."""
    M = torch.load(traj_path)["z"].shape[1]
    return torch.randperm(M, generator=torch.Generator().manual_seed(seed))[: M // 5]


@torch.no_grad()
def replay(traj_path, H, cols=None, **kw):
    """Run OpponentAdapter over recorded trajectories. Returns per-step err [T1, M], valid, t-in-episode, episode idx."""
    oa = _adapt_module()
    data = torch.load(traj_path)
    z, done = data["z"].float(), data["done"]
    if cols is not None:
        z, done = z[:, cols], done[:, cols]
        data = {**data, "opponent": [data["opponent"][i] for i in cols.tolist()]}
    T1, M, _ = z.shape
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ad = oa.OpponentAdapter(M, horizon=H, history=HISTORY, device=dev, **kw)
    err = torch.zeros(T1, M)
    valid = torch.zeros(T1, M, dtype=torch.bool)
    tin = torch.zeros(T1, M, dtype=torch.long)
    epi = torch.zeros(T1, M, dtype=torch.long)
    for k in range(T1):
        if k > 0 and bool(done[k - 1].any()):
            ad.reset(done[k - 1].nonzero().squeeze(-1).to(dev))
        tin[k] = ad.t.cpu()
        ad.step(z[k].to(dev))
        err[k], valid[k] = ad.err.cpu(), ad.err_valid.cpu()
    epi[1:] = torch.cumsum(done.long(), 0)
    return err, valid, tin, epi, data["opponent"]


def cv_errors(traj_path, H):
    data = torch.load(traj_path)
    z, done = data["z"].float(), data["done"]
    T1 = z.shape[0]
    ep = torch.zeros(T1, z.shape[1], dtype=torch.long)
    ep[1:] = torch.cumsum(done.long(), 0)
    src = torch.arange(T1) - H
    ok = (src >= 0).unsqueeze(1) & (ep[src.clamp(min=0)] == ep)
    zs = z[src.clamp(min=0)]
    y = z[:, :, :3] - zs[:, :, :3]
    y_hat = zs[:, :, 3:6] * H * STEP_DT
    return (y - y_hat).norm(dim=-1), ok


def summarise(err, valid, tin, epi, opponents):
    out = {"mean": float(err[valid].mean()), "n": int(valid.sum())}
    bins = [(0, 25), (25, 50), (50, 100), (100, 10_000)]
    out["by_time_in_episode"] = {f"{a}-{b}": float(err[valid & (tin >= a) & (tin < b)].mean())
                                 for a, b in bins if bool((valid & (tin >= a) & (tin < b)).any())}
    out["by_episode_index"] = {str(i): float(err[valid & (epi == i)].mean()) for i in range(0, 12)
                               if bool((valid & (epi == i)).any())}
    per_opp = {}
    for j, o in enumerate(opponents):
        per_opp.setdefault(o, []).append(j)
    out["by_opponent"] = {o: float(err[:, cols][valid[:, cols]].mean()) for o, cols in per_opp.items()
                          if bool(valid[:, cols].any())}
    return out


DELTAS = [0.01, 0.1, 1.0, 10.0]
LAMBDAS = [0.98, 0.995, 0.999]


def tune(train_path, H, base_kw):
    """Grid over (delta, forgetting) on the validation envs of the train-pool trajectories (S1)."""
    cols = val_columns(train_path)
    grid = {}
    for d in DELTAS:
        for lam in LAMBDAS:
            err, valid, *_ = replay(train_path, H, cols=cols, delta=d, forgetting=lam, **base_kw)
            grid[f"{d}/{lam}"] = float(err[valid].mean())
    best = min(grid, key=grid.get)
    d, lam = best.split("/")
    return {"delta": float(d), "forgetting": float(lam)}, grid


def run_a1a(args):
    split = load_split()
    paths = record(split, args.smoke)
    norm = norm_only(paths["train"], os.path.join(A1A, "models", "norm.pt"))
    results, tuned = {}, {}
    for H in HORIZONS:
        model_path = os.path.join(A1A, "models", f"nf_H{H}.pt")
        model = train_basis(paths["train"], H, model_path)
        log_progress(f"A1a basis H={H}: final val shared mse {model['train_log'][-1]['val_shared_mse']:.5f}")
        tuned[str(H)] = {}
        for kind, kw in (("raw", dict(kind="raw", model_path=norm)), ("esn", dict(kind="esn", model_path=norm)),
                         ("nf", dict(kind="nf", model_path=model_path))):
            best, grid = tune(paths["train"], H, kw)
            tuned[str(H)][kind] = {"best": best, "grid": grid}
            log_progress(f"A1a tune H={H} {kind}: {best} (val err {min(grid.values()):.4f})")
        save_json(os.path.join(A1A, "tuned_rls.json"), tuned)
        hp = {k: v["best"] for k, v in tuned[str(H)].items()}
        res = {}
        for part in ("train", "test"):
            p = paths[part]
            e, ok = cv_errors(p, H)
            res[f"{part}/CV"] = {"mean": float(e[ok].mean()), "n": int(ok.sum())}
            methods = {
                "raw_RLS_S1": dict(kind="raw", model_path=norm, **hp["raw"]),
                "ESN_RLS_S1": dict(kind="esn", model_path=norm, **hp["esn"]),
                "NF_frozen": dict(kind="nf", model_path=model_path, freeze=True),
                "NF_RLS_S1": dict(kind="nf", model_path=model_path, **hp["nf"]),
                "NF_RLS_S2": dict(kind="nf", model_path=model_path, reset_on_episode=False, **hp["nf"]),
                "raw_RLS_S2": dict(kind="raw", model_path=norm, reset_on_episode=False, **hp["raw"]),
            }
            for name, kw in methods.items():
                res[f"{part}/{name}"] = summarise(*replay(p, H, **kw))
        results[str(H)] = res
        save_json(os.path.join(A1A, "pred_err.json"), results)
    # Pre-registered horizon choice.
    gains = {H: 1 - results[str(H)]["test/NF_RLS_S1"]["mean"] / results[str(H)]["test/CV"]["mean"] for H in HORIZONS}
    best = max(gains.values())
    h_star = 15 if abs(gains[15] - best) < 1e-9 else max(gains, key=gains.get)
    save_json(os.path.join(A1A, "horizon_choice.json"), {"relative_gain_vs_CV": gains, "H_star": h_star})
    _a1a_summary(results, gains, h_star)


def _a1a_summary(results, gains, h_star):
    lines = ["# A1a_basis summary", "",
             "Mean prediction error |y - y_hat| (m) of the evader's relative displacement over H steps.", ""]
    names = ["CV", "raw_RLS_S1", "ESN_RLS_S1", "NF_frozen", "NF_RLS_S1", "NF_RLS_S2", "raw_RLS_S2"]
    lines += ["| H (steps) | set | " + " | ".join(names) + " |", "|---" * (len(names) + 2) + "|"]
    for H, res in results.items():
        for part in ("train", "test"):
            lines.append(f"| {H} | {part} | " + " | ".join(
                f"{res[f'{part}/{n}']['mean']:.4f}" if f"{part}/{n}" in res else "-" for n in names) + " |")
    lines += ["", "Relative error reduction of NF_RLS_S1 vs CV on the test set: " +
              ", ".join(f"H={h}: {g:+.3f}" for h, g in gains.items()), "", f"**H* = {h_star}** (pre-registered rule).",
              "", "Error by time in episode (test, H*):", ""]
    r = results[str(h_star)]
    for n in ("NF_frozen", "NF_RLS_S1", "ESN_RLS_S1", "raw_RLS_S1"):
        lines.append(f"- {n}: " + ", ".join(f"{k}: {v:.4f}" for k, v in r[f"test/{n}"]["by_time_in_episode"].items()))
    lines += ["", "Error by episode index within an env (test, H*; S2 keeps the readout across episodes):", ""]
    for n in ("NF_RLS_S1", "NF_RLS_S2", "raw_RLS_S1", "raw_RLS_S2"):
        lines.append(f"- {n}: " + ", ".join(f"{k}: {v:.4f}" for k, v in r[f"test/{n}"]["by_episode_index"].items()))
    with open(os.path.join(A1A, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ A1b
def a1b_methods():
    with open(os.path.join(A1A, "horizon_choice.json")) as f:
        H = json.load(f)["H_star"]
    norm = os.path.join(A1A, "models", "norm.pt")
    nf = os.path.join(A1A, "models", f"nf_H{H}.pt")
    with open(os.path.join(A1A, "tuned_rls.json")) as f:
        hp = {k: v["best"] for k, v in json.load(f)[str(H)].items()}
    rls = lambda k: [f"task.adapt.delta={hp[k]['delta']}", f"task.adapt.forgetting={hp[k]['forgetting']}"]
    base = ["task.adapt.enabled=true", f"task.adapt.horizon={H}", f"task.adapt.history={HISTORY}"]
    return {
        "M0h_stack": base + ["task.adapt.kind=stack", f"task.adapt.model_path={norm}"],
        "ESN": base + ["task.adapt.kind=esn", f"task.adapt.model_path={norm}"] + rls("esn"),
        "NF_frozen": base + ["task.adapt.kind=nf", f"task.adapt.model_path={nf}", "task.adapt.freeze=true"],
        "NF_OA": base + ["task.adapt.kind=nf", f"task.adapt.model_path={nf}"] + rls("nf"),
    }


def run_a1b(args):
    split = load_split()
    test = test_set(split)
    frames = 2_000_000 if args.smoke else GEN_FRAMES
    for name, ov in a1b_methods().items():
        for s in range(args.seeds):
            ck = train(os.path.join(A1B, name, "ckpt", f"seed{s}.pt"), split["train"], frames, seed=s, extra=ov)
            log_progress(f"A1b {name} seed{s}: {ck}")
            if not ck:
                continue
            evaluate(os.path.join(A1B, name, "evals", f"seed{s}_train.json"), split["train"], policy_ckpt=ck, extra=ov)
            evaluate(os.path.join(A1B, name, "evals", f"seed{s}_test_S1.json"), test, policy_ckpt=ck, extra=ov)
            if name not in ("M0h_stack", "NF_frozen"):
                evaluate(os.path.join(A1B, name, "evals", f"seed{s}_test_S2.json"), test, policy_ckpt=ck,
                         extra=ov + ["task.adapt.reset_on_episode=false"])
    analyse_a1b(split, test, args.seeds)


def _J(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return {k: v["neutralization_rate"] for k, v in json.load(f)["matchups"].items()}


def _J_by_episode(path, max_idx=10):
    ep_path = os.path.splitext(path)[0] + "_episodes.pt"
    if not os.path.isfile(ep_path):
        return None
    d = torch.load(ep_path)
    e = d["episodes"]
    neut = (e["capture"] > 0.5) | (e["opp_crash"] > 0.5) | (e["opp_out_of_bounds"] > 0.5)
    return [float(neut[e["episode_index"] == i].float().mean()) for i in range(max_idx)
            if bool((e["episode_index"] == i).any())]


def analyse_a1b(split, test, n_seeds):
    import analysis as an
    with open(os.path.join(X3, "gap.json")) as f:
        gap = {r["evader"]: r for r in json.load(f)["rows"]}
    gap_ids = [i for i, r in gap.items() if (r.get("delta") or 0) >= 0.15]
    groups = {"train": [e["id"] for e in split["train"]], "heldout_rl": [e["id"] for e in split["heldout_rl"]],
              "heldout_scripted": [e["id"] for e in split["heldout_scripted"]],
              "exploiter": [e["id"] for e in test if e["id"].startswith("exploit/")], "gap_evaders": gap_ids}
    paths = {"M0": lambda s, part: os.path.join(X3, "evals", f"generalist{s}_{'train' if part == 'train' else 'test'}.json"),
             "ESO_O2a": lambda s, part: os.path.join(OUT, "X4_oracles", "evals",
                                                     f"O2a_accel_seed{s}_{'train' if part == 'train' else 'test'}.json")}
    for name in a1b_methods():
        paths[name] = (lambda n: lambda s, part: os.path.join(A1B, n, "evals", f"seed{s}_{'train' if part == 'train' else part}.json"))(name)
    res = {}
    for m, p in paths.items():
        for setting in ("S1", "S2"):
            if setting == "S2" and m in ("M0", "ESO_O2a", "M0h_stack", "NF_frozen"):
                continue
            rows = []
            for s in range(n_seeds):
                part = "test" if m in ("M0", "ESO_O2a") else f"test_{setting}"
                J = {**_J(p(s, "train")), **_J(p(s, part))}
                if m == "ESO_O2a":
                    J.update(_J(os.path.join(OUT, "X4_oracles", "evals", f"O2a_accel_seed{s}_exploit.json")))
                if not J:
                    continue
                row = {g: float(np.mean([J[i] for i in ids if i in J])) for g, ids in groups.items() if any(i in J for i in ids)}
                gc = [(J[i] - gap[i]["J_generalist_seed0"]) / gap[i]["delta"] for i in gap_ids if i in J]
                if gc:
                    row["gap_closed"] = float(np.mean(gc))
                row["J_by_episode_index"] = _J_by_episode(p(s, part))
                rows.append(row)
            if rows:
                res[f"{m}/{setting}"] = rows
    agg = {}
    for k, rows in res.items():
        agg[k] = {}
        for g in list(groups) + ["gap_closed"]:
            v = [r[g] for r in rows if r.get(g) is not None]
            if v:
                mean, lo, hi = an.bootstrap_ci(v)
                agg[k][g] = {"mean": mean, "ci": [lo, hi], "per_seed": v}
    save_json(os.path.join(A1B, "results.json"), {"per_seed": res, "aggregate": agg, "groups": groups})
    cols = list(groups) + ["gap_closed"]
    lines = ["# A1b_policy summary", "", "Mean neutralisation J (95 % bootstrap CI over seeds). S1: adapter reset each "
             "episode; S2: adapter kept across the episodes of an env (same evader).", "",
             "| method/setting | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for k, a in agg.items():
        lines.append(f"| {k} | " + " | ".join(
            f"{a[g]['mean']:.3f} [{a[g]['ci'][0]:.3f}, {a[g]['ci'][1]:.3f}]" if g in a else "-" for g in cols) + " |")
    with open(os.path.join(A1B, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
