"""A3f: spectral-norm regularised estimator, offline (user request 11-03 after Pan et al. RA-L 2026; plan, pre-registered
before any A3f run).

Same data, split, noise draws, windows, masks and error metric as A3a (acc: evader velocity change over H = 10 steps)
and A3d (vel: current evader velocity), positions-only inputs. The A3a/A3d MLP is retrained with
loss + beta * sum_l ||W_l||^2 over the basis' linear layers and the shared head:
  spec  ||W||_2 = largest singular value (Pan et al. 2026, after Neural Lander);
  frob  Frobenius norm (plain weight decay), the control.
beta in BETAS, chosen per (target, basis, penalty) on val in the basis' own training condition (clean basis: clean val;
aug basis: noisy val). Reference: the unregularised A3a / A3d models, re-evaluated here (consistency check vs their
pred_err.json). Lipschitz bound = prod_l ||W_l||_2 (ELU / tanh are 1-Lipschitz), in normalised-input units.
"""

import json
import os

import torch

from a3_posonly import (A3A, add_noise, basis_predict, episode_ids, fd_predictions, fit_basis, load_traj, weight_mats,
                        windows)
from a3d_velest import A3D
from a_adapt import HISTORY, _adapt_module
from common import OUT, log_progress, save_json

A3F = os.path.join(OUT, "A3f_specnorm")
H_ACC = 10
BETAS = [1e-4, 1e-3, 1e-2]
REGS = ["spec", "frob"]
BASES = ["clean", "aug"]
REF = {"acc": {b: os.path.join(A3A, "models", f"pos_nf_H{H_ACC}_{b}.pt") for b in BASES},
       "vel": {b: os.path.join(A3D, "models", f"vel_nf_{b}.pt") for b in BASES}}


def lipschitz_bound(model):
    oa = _adapt_module()
    phi = oa.BasisNet(model["in_dim"], model["d"], model["hidden"])
    phi.load_state_dict(model["phi"])
    with torch.no_grad():
        return float(torch.prod(torch.stack([torch.linalg.matrix_norm(M, ord=2)
                                             for M in weight_mats(phi, model["theta0"])])))


def run_a3f(args):
    root = os.path.join(A3F, "smoke") if args.smoke else A3F
    tr_z, tr_v, tr_p, tr_done, tr_opp = load_traj(os.path.join(A3A, "rollouts", "traj_train.pt"))
    te_z, te_v, te_p, te_done, _ = load_traj(os.path.join(A3A, "rollouts", "traj_test.pt"))
    M = tr_z.shape[1]
    val_cols = torch.randperm(M, generator=torch.Generator().manual_seed(0))[: M // 5]   # same split as A3a / A3d
    mean, std = tr_z.reshape(-1, 9).mean(0), tr_z.reshape(-1, 9).std(0).clamp(min=1e-3)
    names = sorted(set(tr_opp))
    col_head = torch.tensor([names.index(o) for o in tr_opp])
    noisy = {"train": add_noise(tr_z, 1), "test": add_noise(te_z, 2)}                 # same noise draws
    is_val = torch.zeros(M, dtype=torch.bool)
    is_val[val_cols] = True
    if args.smoke:                                       # tiny: 64 columns, 1 epoch
        cols = torch.arange(64)
        tr_z, tr_v, tr_p, tr_done, col_head, is_val = (tr_z[:, cols], tr_v[:, cols], tr_p[:, cols], tr_done[:, cols],
                                                       col_head[cols], is_val[cols])
        te_z, te_v, te_p, te_done = te_z[:, cols], te_v[:, cols], te_p[:, cols], te_done[:, cols]
        noisy = {"train": noisy["train"][:, cols], "test": noisy["test"][:, cols]}
    tr_ep, te_ep = episode_ids(tr_done), episode_ids(te_done)
    p_noisy = {k: z[..., 3:6] + z[..., 0:3] for k, z in noisy.items()}
    epochs = 1 if args.smoke else 8
    res, lips, choice = {}, {}, {}
    for target in ("acc", "vel"):
        H = H_ACC if target == "acc" else 0
        X, Ya, valid = windows(tr_z, tr_v, tr_ep, H, HISTORY, mean, std)
        Xn, _, _ = windows(noisy["train"], tr_v, tr_ep, H, HISTORY, mean, std)
        Xt, Yta, vt = windows(te_z, te_v, te_ep, H, HISTORY, mean, std)
        Xtn, _, _ = windows(noisy["test"], te_v, te_ep, H, HISTORY, mean, std)
        Y, Yt = (Ya, Yta) if target == "acc" else (tr_v, te_v)
        vmask = is_val.expand_as(valid)
        heads = col_head.expand_as(valid)[valid]
        # Evaluation masks exactly as A3a (acc: & finite-difference mask of the inputs' position estimate) / A3d (vel).
        sets = {"val": (X, Xn, Y, valid & vmask, tr_p, p_noisy["train"], tr_ep),
                "test": (Xt, Xtn, Yt, vt, te_p, p_noisy["test"], te_ep)}
        masks = {}
        for part, (_, _, _, ok, pc, pn, ep) in sets.items():
            for cond, p in (("clean", pc), ("noisy", pn)):
                masks[(part, cond)] = (ok & fd_predictions(p, ep, H)[1]) if target == "acc" else ok

        def evaluate_model(name, model):
            for part, (Xc, Xnn, Yc, _, _, _, _) in sets.items():
                for cond, Xin in (("clean", Xc), ("noisy", Xnn)):
                    m = masks[(part, cond)]
                    res[f"{target}/{part}/{cond}/{name}"] = float((Yc[m] - basis_predict(model, Xin[m])).norm(dim=-1).mean())
            lips[f"{target}/{name}"] = lipschitz_bound(model)

        for b in BASES:
            ref = torch.load(REF[target][b]) if not args.smoke else fit_basis(
                (X if b == "clean" else Xn)[valid], Y[valid], heads, vmask[valid], H, mean, std,
                os.path.join(root, "models", f"{target}_{b}_none.pt"), epochs=epochs,
                target="dvel" if target == "acc" else "vel")
            evaluate_model(f"MLP_{b}", ref)
            sel_cond = "clean" if b == "clean" else "noisy"
            for reg in REGS:
                grid = {}
                for beta in BETAS:
                    name = f"{reg}_{b}_b{beta:g}"
                    model = fit_basis((X if b == "clean" else Xn)[valid], Y[valid], heads, vmask[valid], H, mean, std,
                                      os.path.join(root, "models", f"{target}_{name}.pt"), epochs=epochs,
                                      target="dvel" if target == "acc" else "vel", reg=reg, beta=beta)
                    evaluate_model(name, model)
                    grid[beta] = res[f"{target}/val/{sel_cond}/{name}"]
                best = min(grid, key=grid.get)
                choice[f"{target}/{reg}_{b}"] = {"beta": best, "val_" + sel_cond: grid}
                for k in [k for k in list(res) if k.startswith(f"{target}/") and k.endswith(f"/{reg}_{b}_b{best:g}")]:
                    res[k.rsplit("/", 1)[0] + f"/{reg}_{b}"] = res[k]
                lips[f"{target}/{reg}_{b}"] = lips[f"{target}/{reg}_{b}_b{best:g}"]
                log_progress(f"A3f {target} {reg}_{b}: beta* {best:g}, noisy test "
                             f"{res[f'{target}/test/noisy/{reg}_{b}']:.4f} vs MLP {res[f'{target}/test/noisy/MLP_{b}']:.4f}")
                save_json(os.path.join(root, "pred_err.json"), {"errors": res, "lipschitz": lips, "choice": choice})
        del X, Xn, Xt, Xtn
    if args.smoke:
        log_progress(f"A3f smoke ok: {len(res)} errors")
        return
    # Consistency check: re-evaluated unregularised models vs the stored A3a / A3d numbers.
    with open(os.path.join(A3A, "pred_err.json")) as f:
        a3a = json.load(f)[str(H_ACC)]
    with open(os.path.join(A3D, "pred_err.json")) as f:
        a3d = json.load(f)
    check = {f"{t}/{part}/{cond}/MLP_{b}": res[f"{t}/{part}/{cond}/MLP_{b}"] - src[f"{part}/{cond}/NF_{b}"]
             for t, src in (("acc", a3a), ("vel", a3d)) for part in ("val", "test") for cond in ("clean", "noisy")
             for b in BASES}
    save_json(os.path.join(A3F, "pred_err.json"), {"errors": res, "lipschitz": lips, "choice": choice,
                                                   "consistency_vs_a3a_a3d": check})
    cols = [f"{n}_{b}" for b in BASES for n in ("MLP", "spec", "frob")]
    lines = ["# A3f_specnorm summary", "",
             "Mean error (m/s): acc = evader velocity change over 0.2 s (A3a), vel = current evader velocity (A3d); "
             "positions-only inputs; noisy = A3a sensor-noise model; val = train-pool evaders (20 % of envs), test = "
             "held-out test-set evaders. MLP = unregularised A3a / A3d model; spec / frob = + beta * sum ||W||^2 with "
             "the spectral / Frobenius norm, beta* chosen on val in the basis' training condition (clean basis: clean "
             "val; aug basis: noisy val).", "",
             "| target | set | inputs | " + " | ".join(cols) + " |", "|---" * (len(cols) + 3) + "|"]
    for t in ("acc", "vel"):
        for part in ("val", "test"):
            for cond in ("clean", "noisy"):
                lines.append(f"| {t} | {part} | {cond} | " + " | ".join(f"{res[f'{t}/{part}/{cond}/{c}']:.4f}"
                                                                     for c in cols) + " |")
    lines += ["", "Lipschitz bound prod ||W_l||_2 (normalised inputs):", "",
              "| target | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for t in ("acc", "vel"):
        lines.append(f"| {t} | " + " | ".join(f"{lips[f'{t}/{c}']:.1f}" for c in cols) + " |")
    lines += ["", "beta* (val grid in choice of pred_err.json): " + ", ".join(f"{k} {v['beta']:g}" for k, v in choice.items()),
              "", f"Consistency (re-evaluated MLP - stored A3a / A3d, should be ~0): max |diff| "
              f"{max(abs(v) for v in check.values()):.2e}"]
    with open(os.path.join(A3F, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log_progress("A3f done")
