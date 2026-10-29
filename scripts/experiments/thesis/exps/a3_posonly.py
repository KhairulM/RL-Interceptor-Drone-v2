"""A3: positions-only setting (no evader velocity in the pursuer observation) and sensor-noise evaluation.

A3a (offline). Trajectories of the X3 generalist (seed 0), re-recorded with the pursuer's own position.
  Estimator input window z_pos = [rel_pos, own_pos, own_vel] (no evader velocity); target = true change of the
  evader's velocity over H steps; frozen basis (online RLS would need the unobservable evader velocity as target).
  Bases: trained on clean inputs ("clean") and on noise-augmented inputs ("aug"; clean targets).
  Baselines: zero; finite-difference persistence from positions: a(t) = (p(t) - 2 p(t-1) + p(t-2)) / dt^2 with
  p = own_pos + rel_pos, raw and EMA-filtered (alpha tuned on noisy val).
  Pre-registered (10-28): H* = H in {5, 10} with the largest error reduction of the clean frozen basis vs raw FD
  persistence on clean val (ties -> 10). The policy uses the "aug" basis if its noisy-val error is lower than the
  clean basis's and its clean-val error is <= 1.1x the clean basis's; else "clean".
A3b. Pursuers without evader velocity (observation.agent_rel_vel=false), train pool, uniform, 150M frames,
  noise-free training, 3 seeds: P_none, P_fd (oracle.accel_source=pos), P_nf (adapt z_mode=pos, frozen).
  Eval on train + test, noise-free and with sensor noise (P_fd also with the tuned EMA).
A3c. Noise reference: existing velocity-based policies evaluated with sensor noise (eval only).
"""

import json
import os

import numpy as np
import torch

from a_adapt import HISTORY, STEP_DT, _J, _adapt_module, test_set
from common import OUT, evaluate, log_progress, save_json, train
from x3_gap import EXP as X3, GEN_FRAMES, generalist_ckpt, load_split

A3A = os.path.join(OUT, "A3a_posonly")
A3B = os.path.join(OUT, "A3b_posonly")
A3C = os.path.join(OUT, "A3c_noise_ref")
HORIZONS = [5, 10]
NOISE_STD = torch.tensor([0.02] * 3 + [0.02] * 3 + [0.05] * 3)   # rel_dist_std, pos_std, lin_vel_std (cfg defaults)
EMA_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
NOISY = ["task.observation.include_noise=true"]
POSONLY = ["task.observation.agent_rel_vel=false"]


# ------------------------------------------------------------------ data
def record(split, smoke=False):
    paths = {}
    for part, opps in (("train", split["train"]), ("test", test_set(split))):
        path = os.path.join(A3A, "rollouts", f"traj_{part}.pt")
        if not os.path.isfile(path):
            evaluate(os.path.join(A3A, "evals", f"record_{part}.json"), opps, policy_ckpt=generalist_ckpt(0),
                     extra=[f"eval_traj_out={path}", f"eval_traj_envs={256 if smoke else 1024}"], save_episodes=False)
        paths[part] = path
    return paths


def load_traj(path):
    d = torch.load(path)
    z = d["z"].float()
    own_pos = d["own_pos"].float()
    zpos = torch.cat([z[..., 0:3], own_pos, z[..., 6:9]], dim=-1)       # [T1, M, 9]
    v_e = z[..., 3:6] + z[..., 6:9]                                      # true evader velocity
    p_e = own_pos + z[..., 0:3]                                          # evader position
    return zpos, v_e, p_e, d["done"], d["opponent"]


def episode_ids(done):
    T1 = done.shape[0] + 1
    ep = torch.zeros(T1, done.shape[1], dtype=torch.long)
    ep[1:] = torch.cumsum(done.long(), 0)
    return ep


def add_noise(zpos, seed=1):
    g = torch.Generator().manual_seed(seed)
    return zpos + torch.randn(zpos.shape, generator=g) * NOISE_STD


def windows(zin, v_e, ep, H, K, mean, std):
    """X [T1, M, K*9] normalised zero-padded history, Y [T1, M, 3] = v_e(t+H) - v_e(t), valid [T1, M]."""
    T1 = zin.shape[0]
    zn = (zin - mean) / std
    xs = []
    for j in range(K - 1, -1, -1):
        idx = torch.arange(T1) - j
        ok = (idx >= 0).unsqueeze(1) & (ep[idx.clamp(min=0)] == ep)
        xs.append(torch.where(ok.unsqueeze(-1), zn[idx.clamp(min=0)], torch.zeros_like(zn)))
    X = torch.stack(xs, dim=2).flatten(2)
    tgt = torch.arange(T1) + H
    valid = (tgt < T1).unsqueeze(1) & (ep[tgt.clamp(max=T1 - 1)] == ep)
    Y = v_e[tgt.clamp(max=T1 - 1)] - v_e
    return X, Y, valid


def fd_predictions(p_e, ep, H, alpha=None):
    """Finite-difference persistence prediction of the velocity change over H, from positions. valid needs 3 samples."""
    T1 = p_e.shape[0]
    a = torch.zeros_like(p_e)
    a[2:] = (p_e[2:] - 2 * p_e[1:-1] + p_e[:-2]) / STEP_DT ** 2
    same = torch.zeros(T1, p_e.shape[1], dtype=torch.bool)
    same[2:] = (ep[2:] == ep[1:-1]) & (ep[1:-1] == ep[:-2])
    a = torch.where(same.unsqueeze(-1), a, torch.zeros_like(a))
    if alpha is not None and alpha < 1.0:
        f = torch.zeros_like(a)
        for t in range(T1):
            prev = f[t - 1] if t > 0 else torch.zeros_like(a[0])
            new_ep = (ep[t] != ep[t - 1]) if t > 0 else torch.ones(p_e.shape[1], dtype=torch.bool)
            prev = torch.where(new_ep.unsqueeze(-1), torch.zeros_like(prev), prev)
            f[t] = torch.where(same[t].unsqueeze(-1), prev + alpha * (a[t] - prev), prev)
        a = f
    return a * H * STEP_DT, same


# ------------------------------------------------------------------ basis
def fit_basis(X, Y, heads, val_mask, H, mean, std, out_path, d=32, hidden=128, epochs=8, max_samples=1_500_000, seed=0,
              target="dvel"):
    if os.path.isfile(out_path):
        return torch.load(out_path)
    oa = _adapt_module()
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    tr = (~val_mask).nonzero().squeeze(-1)
    if len(tr) > max_samples:
        tr = tr[torch.randperm(len(tr), generator=g)[:max_samples]]
    va = val_mask.nonzero().squeeze(-1)[:200_000]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    phi = oa.BasisNet(X.shape[1], d, hidden).to(dev)
    n_heads = int(heads.max()) + 1
    W = torch.nn.Parameter(torch.zeros(n_heads, 3, d + 1, device=dev))
    W0 = torch.nn.Parameter(torch.zeros(3, d + 1, device=dev))
    opt = torch.optim.Adam(list(phi.parameters()) + [W, W0], lr=1e-3)
    Xd, Yd, Hd = X.to(dev), Y.to(dev), heads.to(dev)

    def losses(idx):
        f = phi(Xd[idx])
        f = torch.cat([f, torch.ones_like(f[:, :1])], -1)
        lt = ((torch.einsum("nij,nj->ni", W[Hd[idx]], f) - Yd[idx]) ** 2).sum(-1).mean()
        ls = ((f @ W0.T - Yd[idx]) ** 2).sum(-1).mean()
        return lt, ls

    log = []
    for ep in range(epochs):
        perm = tr[torch.randperm(len(tr))]
        for b in range(0, len(perm), 4096):
            lt, ls = losses(perm[b:b + 4096].to(dev))
            opt.zero_grad()
            (lt + ls).backward()
            opt.step()
        with torch.no_grad():
            vt, vs = losses(va.to(dev))
        log.append({"epoch": ep, "val_task_mse": vt.item(), "val_shared_mse": vs.item()})
    model = {"phi": {k: v.cpu() for k, v in phi.state_dict().items()}, "theta0": W0.detach().cpu(),
             "z_mean": mean, "z_std": std, "in_dim": X.shape[1], "d": d, "hidden": hidden, "horizon": H,
             "history": HISTORY, "target": target, "z_mode": "pos", "train_log": log}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model, out_path)
    return model


@torch.no_grad()
def basis_predict(model, X):
    oa = _adapt_module()
    phi = oa.BasisNet(model["in_dim"], model["d"], model["hidden"])
    phi.load_state_dict(model["phi"])
    out = []
    for b in range(0, X.shape[0], 200_000):
        f = phi(X[b:b + 200_000])
        out.append(torch.cat([f, torch.ones_like(f[:, :1])], -1) @ model["theta0"].T)
    return torch.cat(out)


def run_a3a(args):
    split = load_split()
    paths = record(split, args.smoke)
    tr_z, tr_v, tr_p, tr_done, tr_opp = load_traj(paths["train"])
    te_z, te_v, te_p, te_done, _ = load_traj(paths["test"])
    tr_ep, te_ep = episode_ids(tr_done), episode_ids(te_done)
    M = tr_z.shape[1]
    val_cols = torch.randperm(M, generator=torch.Generator().manual_seed(0))[: M // 5]
    mean, std = tr_z.reshape(-1, 9).mean(0), tr_z.reshape(-1, 9).std(0).clamp(min=1e-3)
    names = sorted(set(tr_opp))
    col_head = torch.tensor([names.index(o) for o in tr_opp])
    noisy = {"train": add_noise(tr_z, 1), "test": add_noise(te_z, 2)}
    # Noisy evader position estimate = noisy own_pos + noisy rel_pos.
    p_noisy = {k: z[..., 3:6] + z[..., 0:3] for k, z in noisy.items()}
    res, choice = {}, {}
    for H in HORIZONS:
        r = {}
        X, Y, valid = windows(tr_z, tr_v, tr_ep, H, HISTORY, mean, std)
        Xn, _, _ = windows(noisy["train"], tr_v, tr_ep, H, HISTORY, mean, std)
        is_val = torch.zeros(M, dtype=torch.bool)
        is_val[val_cols] = True
        is_val = is_val.expand_as(valid)
        sel = valid
        heads = col_head.expand_as(valid)[sel]
        models = {
            "clean": fit_basis(X[sel], Y[sel], heads, is_val[sel], H, mean, std,
                               os.path.join(A3A, "models", f"pos_nf_H{H}_clean.pt")),
            "aug": fit_basis(Xn[sel], Y[sel], heads, is_val[sel], H, mean, std,
                             os.path.join(A3A, "models", f"pos_nf_H{H}_aug.pt")),
        }
        Xt, Yt, vt = windows(te_z, te_v, te_ep, H, HISTORY, mean, std)
        Xtn, _, _ = windows(noisy["test"], te_v, te_ep, H, HISTORY, mean, std)
        sets = {"val": (X, Xn, Y, valid & is_val, tr_p, p_noisy["train"], tr_ep),
                "test": (Xt, Xtn, Yt, vt, te_p, p_noisy["test"], te_ep)}
        for part, (Xc, Xnn, Yc, ok, pc, pn, ep) in sets.items():
            for cond, Xin, p in (("clean", Xc, pc), ("noisy", Xnn, pn)):
                fd, fd_ok = fd_predictions(p, ep, H)
                m = ok & fd_ok
                r[f"{part}/{cond}/zero"] = float(Yc[m].norm(dim=-1).mean())
                r[f"{part}/{cond}/FD"] = float((Yc - fd)[m].norm(dim=-1).mean())
                for bname, model in models.items():
                    pred = basis_predict(model, Xin[m])
                    r[f"{part}/{cond}/NF_{bname}"] = float((Yc[m] - pred).norm(dim=-1).mean())
        # EMA alpha for FD on noisy val.
        grid = {}
        X_, Y_, ok_ = sets["val"][0], sets["val"][2], sets["val"][3]
        for alpha in EMA_GRID:
            fd, fd_ok = fd_predictions(p_noisy["train"], tr_ep, H, alpha)
            m = ok_ & fd_ok
            grid[str(alpha)] = float((Y_ - fd)[m].norm(dim=-1).mean())
        best_alpha = float(min(grid, key=grid.get))
        fd, fd_ok = fd_predictions(p_noisy["test"], te_ep, H, best_alpha)
        m = vt & fd_ok
        r["test/noisy/FD_ema"] = float((Yt - fd)[m].norm(dim=-1).mean())
        r["val/noisy/FD_ema"] = grid[str(best_alpha)]
        r["ema_grid_val_noisy"] = grid
        r["ema_alpha"] = best_alpha
        res[str(H)] = r
        log_progress(f"A3a H={H}: " + ", ".join(f"{k}={v:.4f}" for k, v in r.items() if isinstance(v, float)))
        save_json(os.path.join(A3A, "pred_err.json"), res)
    gains = {H: 1 - res[str(H)]["val/clean/NF_clean"] / res[str(H)]["val/clean/FD"] for H in HORIZONS}
    best = max(gains.values())
    h_star = 10 if abs(gains[10] - best) < 1e-9 else max(gains, key=gains.get)
    r = res[str(h_star)]
    use_aug = (r["val/noisy/NF_aug"] < r["val/noisy/NF_clean"]) and (r["val/clean/NF_aug"] <= 1.1 * r["val/clean/NF_clean"])
    choice = {"gain_vs_FD_clean_val": gains, "H_star": h_star, "basis": "aug" if use_aug else "clean",
              "model_path": os.path.join(A3A, "models", f"pos_nf_H{h_star}_{'aug' if use_aug else 'clean'}.pt"),
              "fd_ema_alpha": r["ema_alpha"]}
    save_json(os.path.join(A3A, "choice.json"), choice)
    keys = ["zero", "FD", "NF_clean", "NF_aug"]
    lines = ["# A3a_posonly summary", "",
             "Mean error |y - y_hat| (m/s) of the evader's velocity change over H steps, estimator inputs WITHOUT evader "
             "velocity. FD = second difference of the evader position estimate, held for H steps. noisy = sensor-noise "
             "model on all inputs (rel_pos 0.02 m, own_pos 0.02 m, own_vel 0.05 m/s).", "",
             "| H | set | inputs | " + " | ".join(keys) + " | FD_ema |", "|---" * (len(keys) + 4) + "|"]
    for H, rr in res.items():
        for part in ("val", "test"):
            for cond in ("clean", "noisy"):
                ema = f"{rr[f'{part}/noisy/FD_ema']:.4f}" if cond == "noisy" else "-"
                lines.append(f"| {H} | {part} | {cond} | " + " | ".join(f"{rr[f'{part}/{cond}/{k}']:.4f}" for k in keys)
                             + f" | {ema} |")
    lines += ["", f"Pre-registered choice: {choice}"]
    with open(os.path.join(A3A, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ policies
def a3b_methods():
    with open(os.path.join(A3A, "choice.json")) as f:
        c = json.load(f)
    nf = ["task.adapt.enabled=true", "task.adapt.kind=nf", "task.adapt.z_mode=pos", "task.adapt.target=dvel",
          "task.adapt.freeze=true", f"task.adapt.horizon={c['H_star']}", f"task.adapt.history={HISTORY}",
          f"task.adapt.model_path={c['model_path']}"]
    fd = ["task.oracle.opp_accel=true", "task.oracle.accel_source=pos"]
    return {"P_none": POSONLY, "P_fd": POSONLY + fd, "P_nf": POSONLY + nf}, c


def run_a3b(args):
    split = load_split()
    test = test_set(split)
    methods, c = a3b_methods()
    frames = 2_000_000 if args.smoke else GEN_FRAMES
    for s in range(args.seeds):
        for name, ov in methods.items():
            d = os.path.join(A3B, name)
            ck = train(os.path.join(d, "ckpt", f"seed{s}.pt"), split["train"], frames, seed=s, extra=ov)
            log_progress(f"A3b {name} seed{s}: {ck}")
            if not ck:
                continue
            for cond, extra in (("clean", []), ("noisy", NOISY)):
                for part, opps in (("train", split["train"]), ("test", test)):
                    evaluate(os.path.join(d, "evals", f"seed{s}_{cond}_{part}.json"), opps, policy_ckpt=ck,
                             extra=ov + extra)
            if name == "P_fd":
                for part, opps in (("train", split["train"]), ("test", test)):
                    evaluate(os.path.join(d, "evals", f"seed{s}_noisyema_{part}.json"), opps, policy_ckpt=ck,
                             extra=ov + NOISY + [f"task.oracle.accel_ema={c['fd_ema_alpha']}"])
    analyse(split, test, args.seeds)


def _ref_policies():
    """Velocity-based policies from earlier experiments and their eval overrides."""
    out = {}
    for s in range(5):
        out[("M0", s)] = (generalist_ckpt(s), [])
        x4 = os.path.join(OUT, "X4_oracles", "ckpt", f"O2a_accel_seed{s}.pt")
        f1 = os.path.join(OUT, "F1_final", "SU_acc", "ckpt", f"seed{s}.pt")
        out[("FD_vel_acc", s)] = (x4 if os.path.isfile(x4) else f1, ["task.oracle.opp_accel=true"])
    from a2_nfacc import a2b_methods, A2B
    for name, ov in a2b_methods().items():
        for s in range(5):
            out[(name, s)] = (os.path.join(A2B, name, "ckpt", f"seed{s}.pt"), ov)
    return out


def run_a3c(args):
    split = load_split()
    test = test_set(split)
    for (name, s), (ck, ov) in sorted(_ref_policies().items()):
        if not os.path.isfile(ck):
            continue
        for part, opps in (("train", split["train"]), ("test", test)):
            evaluate(os.path.join(A3C, name, f"seed{s}_noisy_{part}.json"), opps, policy_ckpt=ck, extra=ov + NOISY)
        log_progress(f"A3c {name} seed{s} noisy evals done")
    analyse(split, test, 5)


def analyse(split, test, n_seeds):
    import analysis as an
    groups = {"train": [e["id"] for e in split["train"]], "heldout_rl": [e["id"] for e in split["heldout_rl"]],
              "heldout_scripted": [e["id"] for e in split["heldout_scripted"]],
              "exploiter": [e["id"] for e in test if e["id"].startswith("exploit/")]}
    from a2_nfacc import A2B
    x4 = os.path.join(OUT, "X4_oracles", "evals")

    def clean_files(name, s):
        if name == "M0":
            return [os.path.join(X3, "evals", f"generalist{s}_{p}.json") for p in ("train", "test")]
        if name == "FD_vel_acc":
            if os.path.isfile(os.path.join(x4, f"O2a_accel_seed{s}_train.json")):
                return [os.path.join(x4, f"O2a_accel_seed{s}_{p}.json") for p in ("train", "test", "exploit")]
            return [os.path.join(OUT, "F1_final", "SU_acc", "evals", f"seed{s}_{p}.json") for p in ("train", "test")]
        if name.startswith("NFacc"):
            return [os.path.join(A2B, name, "evals", f"seed{s}_{p}.json") for p in ("train", "test_S1")]
        return [os.path.join(A3B, name, "evals", f"seed{s}_clean_{p}.json") for p in ("train", "test")]

    def noisy_files(name, s, tag="noisy"):
        if name in ("M0", "FD_vel_acc", "NFacc_frozen", "NFacc_OA"):
            return [os.path.join(A3C, name, f"seed{s}_noisy_{p}.json") for p in ("train", "test")]
        return [os.path.join(A3B, name, "evals", f"seed{s}_{tag}_{p}.json") for p in ("train", "test")]

    rows = {}
    for name in ("M0", "FD_vel_acc", "NFacc_frozen", "NFacc_OA", "P_none", "P_fd", "P_nf"):
        for cond in ("clean", "noisy", "noisyema"):
            if cond == "noisyema" and name != "P_fd":
                continue
            per = []
            for s in range(n_seeds):
                J = {}
                for f in (clean_files(name, s) if cond == "clean" else noisy_files(name, s, cond)):
                    J.update(_J(f))
                if J and all(i in J for i in groups["train"]):
                    per.append({g: float(np.mean([J[i] for i in ids if i in J])) for g, ids in groups.items()
                                if any(i in J for i in ids)})
            if per:
                rows[f"{name}/{cond}"] = {g: dict(zip(("mean", "lo", "hi"), an.bootstrap_ci([r[g] for r in per if g in r])))
                                          for g in groups}
                rows[f"{name}/{cond}"]["n"] = len(per)
    os.makedirs(A3B, exist_ok=True)
    save_json(os.path.join(A3B, "results.json"), rows)
    lines = ["# A3b / A3c summary", "", "Mean J (95 % bootstrap CI over seeds). clean = noise-free eval; noisy = "
             "observation.include_noise=true; noisyema = noisy + EMA-filtered acceleration. All policies trained "
             "noise-free. P_* policies observe no evader velocity.", "",
             "| policy / eval | n | " + " | ".join(groups) + " |", "|---" * (len(groups) + 2) + "|"]
    for k, r in rows.items():
        lines.append(f"| {k} | {r['n']} | " + " | ".join(
            f"{r[g]['mean']:.3f} [{r[g]['lo']:.3f}, {r[g]['hi']:.3f}]" if g in r else "-" for g in groups) + " |")
    with open(os.path.join(A3B, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
