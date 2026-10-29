"""A3e: estimator architectures, offline (user request 10-29 after A3d; docs/research_plan.md, pre-registered before
any A3e result).

Is the 10-step MLP window (A3a / A3d) the right estimator, or does a recurrent network (memory over the whole episode)
or a classical Kalman filter estimate the evader's state better from noisy positions? Same data, split, noise draws,
targets, masks and error metric as A3a (H = 10 velocity change, i.e. 0.2 s-ahead acceleration) and A3d (current
velocity); MLP / FD / FD_ema numbers are reused from their pred_err.json.
  GRU (hidden 96) / LSTM (hidden 80): ~30k parameters like the MLP (32k); input z_pos per step (normalised as the MLP),
    state reset at episode start, linear output; trained on clean or noise-augmented inputs; Adam 1e-3, truncated
    BPTT 100 steps, batch 64 columns, 8 epochs, final epoch kept.
  KF: per-axis constant-acceleration Kalman filter on the evader-position measurement own_pos + rel_pos,
    R = 0.02^2 + 0.02^2 m^2, white-jerk process noise q tuned on noisy val per target.
"""

import json
import os

import torch
import torch.nn as nn

from a_adapt import STEP_DT
from a3_posonly import A3A, add_noise, episode_ids, fd_predictions, load_traj
from a3d_velest import A3D
from common import OUT, log_progress, save_json

A3E = os.path.join(OUT, "A3e_estimators")
H_ACC = 10
Q_GRID = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
R_POS = 0.02 ** 2 + 0.02 ** 2
RNN = {"GRU": 96, "LSTM": 80}
EPOCHS, TBPTT, BATCH, LR = 8, 100, 64, 1e-3


class RNNEst(nn.Module):
    def __init__(self, kind, hidden):
        super().__init__()
        self.kind, self.hidden = kind, hidden
        self.cell = (nn.GRUCell if kind == "GRU" else nn.LSTMCell)(9, hidden)
        self.out = nn.Linear(hidden, 3)

    def init(self, n, dev):
        h = torch.zeros(n, self.hidden, device=dev)
        return h if self.kind == "GRU" else (h, h.clone())

    def forward(self, x, keep, state):
        """x [T, B, 9], keep [T, B] (0 at an episode start) -> y [T, B, 3], last state."""
        ys = []
        for t in range(x.shape[0]):
            k = keep[t].unsqueeze(-1)
            state = state * k if self.kind == "GRU" else (state[0] * k, state[1] * k)
            state = self.cell(x[t], state)
            ys.append(self.out(state if self.kind == "GRU" else state[0]))
        return torch.stack(ys), state


def _detach(state):
    return state.detach() if torch.is_tensor(state) else tuple(s.detach() for s in state)


@torch.no_grad()
def rnn_predict(m, Zn, keep, dev):
    out = []
    for b in range(0, Zn.shape[1], 1024):
        x, k = Zn[:, b:b + 1024].to(dev), keep[:, b:b + 1024].to(dev)
        y, _ = m(x, k, m.init(x.shape[1], dev))
        out.append(y.cpu())
    return torch.cat(out, 1)


def fit_rnn(kind, Zn, Y, w, keep, train_cols, val_cols, path, epochs=EPOCHS, seed=0):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = RNNEst(kind, RNN[kind]).to(dev)
    if os.path.isfile(path):
        m.load_state_dict(torch.load(path)["state"])
        return m
    torch.manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    T1, log = Zn.shape[0], []
    for epoch in range(epochs):
        perm = train_cols[torch.randperm(len(train_cols))]
        for b in range(0, len(perm), BATCH):
            c = perm[b:b + BATCH]
            x, y, wc, k = Zn[:, c].to(dev), Y[:, c].to(dev), w[:, c].float().to(dev), keep[:, c].to(dev)
            state = m.init(len(c), dev)
            for t0 in range(0, T1, TBPTT):
                pred, state = m(x[t0:t0 + TBPTT], k[t0:t0 + TBPTT], state)
                state = _detach(state)
                ww = wc[t0:t0 + TBPTT]
                loss = (((pred - y[t0:t0 + TBPTT]) ** 2).sum(-1) * ww).sum() / ww.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
        pv = rnn_predict(m, Zn[:, val_cols], keep[:, val_cols], dev)
        wv = w[:, val_cols]
        log.append({"epoch": epoch, "val_err": float((Y[:, val_cols] - pv)[wv].norm(dim=-1).mean())})
        log_progress(f"A3e {os.path.basename(path)} epoch {epoch}: val err {log[-1]['val_err']:.4f}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state": m.state_dict(), "kind": kind, "hidden": RNN[kind], "train_log": log}, path)
    return m


def kalman(p_meas, ep, q, r=R_POS, dt=STEP_DT):
    """Constant-acceleration Kalman filter per axis; p_meas [T1, M, 3] -> filtered v, a [T1, M, 3]."""
    T1, M = p_meas.shape[:2]
    F = torch.tensor([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1]], dtype=torch.float64)
    Q = q * torch.tensor([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6], [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                          [dt ** 3 / 6, dt ** 2 / 2, dt]], dtype=torch.float64)
    P0 = torch.diag(torch.tensor([r, 4.0, 100.0], dtype=torch.float64))
    x = torch.zeros(M, 3, 3, dtype=torch.float64)            # [M, axis, (p, v, a)]
    P = P0.expand(M, 3, 3).clone()                           # identical for the three axes
    v_out, a_out = torch.zeros(T1, M, 3), torch.zeros(T1, M, 3)
    for t in range(T1):
        z = p_meas[t].double()
        x = x @ F.T
        P = F @ P @ F.T + Q
        S = P[:, 0, 0] + r
        K = P[:, :, 0] / S.unsqueeze(-1)                      # [M, 3]
        x = x + (z - x[:, :, 0]).unsqueeze(-1) * K.unsqueeze(1)
        P = P - K.unsqueeze(-1) * P[:, 0, :].unsqueeze(1)
        new = torch.ones(M, dtype=torch.bool) if t == 0 else ep[t] != ep[t - 1]
        if bool(new.any()):
            x[new] = 0.0
            x[new, :, 0] = z[new]
            P[new] = P0
        v_out[t], a_out[t] = x[:, :, 1].float(), x[:, :, 2].float()
    return v_out, a_out


def run_a3e(args):
    tr_z, tr_v, _, tr_done, _ = load_traj(os.path.join(A3A, "rollouts", "traj_train.pt"))
    te_z, te_v, _, te_done, _ = load_traj(os.path.join(A3A, "rollouts", "traj_test.pt"))
    tr_ep, te_ep = episode_ids(tr_done), episode_ids(te_done)
    M = tr_z.shape[1]
    val_cols = torch.randperm(M, generator=torch.Generator().manual_seed(0))[: M // 5]   # same split as A3a / A3d
    is_val = torch.zeros(M, dtype=torch.bool)
    is_val[val_cols] = True
    train_cols = (~is_val).nonzero().squeeze(-1)
    if args.smoke:
        train_cols, val_cols = train_cols[:16], val_cols[:16]
    mean, std = tr_z.reshape(-1, 9).mean(0), tr_z.reshape(-1, 9).std(0).clamp(min=1e-3)
    data = {}
    for part, (z, v, ep, seed) in {"train": (tr_z, tr_v, tr_ep, 1), "test": (te_z, te_v, te_ep, 2)}.items():
        T1 = z.shape[0]
        tgt = torch.arange(T1) + H_ACC
        _, fd_ok = fd_predictions(z[..., 3:6] + z[..., 0:3], ep, H_ACC)
        acc_ok = (tgt < T1).unsqueeze(1) & (ep[tgt.clamp(max=T1 - 1)] == ep) & fd_ok        # A3a mask
        keep = torch.zeros(T1, z.shape[1])
        keep[1:] = (ep[1:] == ep[:-1]).float()
        zn = add_noise(z, seed)
        data[part] = {"z": {"clean": z, "noisy": zn}, "ep": ep, "keep": keep,
                      "Y": {"vel": v, "acc": v[tgt.clamp(max=T1 - 1)] - v},
                      "ok": {"vel": torch.ones(T1, z.shape[1], dtype=torch.bool), "acc": acc_ok}}
    sets = {"val": ("train", is_val), "test": ("test", None)}
    res, q_star = {}, {}
    # Kalman filter: q tuned on noisy val per target.
    for target in ("vel", "acc"):
        d = data["train"]
        p = d["z"]["noisy"][..., 3:6] + d["z"]["noisy"][..., 0:3]
        ok = d["ok"][target] & is_val
        grid = {}
        for q in Q_GRID:
            v, a = kalman(p, d["ep"], q)
            est = v if target == "vel" else a * H_ACC * STEP_DT
            grid[str(q)] = float((d["Y"][target] - est)[ok].norm(dim=-1).mean())
        q_star[target] = {"q": float(min(grid, key=grid.get)), "grid_val_noisy": grid}
    for part, (src, colmask) in sets.items():
        d = data[src]
        for cond in ("clean", "noisy"):
            p = d["z"][cond][..., 3:6] + d["z"][cond][..., 0:3]
            for target in ("vel", "acc"):
                v, a = kalman(p, d["ep"], q_star[target]["q"])
                est = v if target == "vel" else a * H_ACC * STEP_DT
                ok = d["ok"][target] & (colmask if colmask is not None else True)
                res[f"{target}/{part}/{cond}/KF"] = float((d["Y"][target] - est)[ok].norm(dim=-1).mean())
                res[f"{target}/{part}/{cond}/zero"] = float(d["Y"][target][ok].norm(dim=-1).mean())
    # Recurrent estimators.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = data["train"]
    for target in ("vel", "acc"):
        for kind in RNN:
            for basis in ("clean", "aug"):
                Zn = (tr["z"]["clean" if basis == "clean" else "noisy"] - mean) / std
                path = os.path.join(A3E, "smoke" if args.smoke else "", "models", f"{target}_{kind}_{basis}.pt")
                m = fit_rnn(kind, Zn, tr["Y"][target], tr["ok"][target], tr["keep"], train_cols, val_cols, path,
                            epochs=1 if args.smoke else EPOCHS)
                for part, (src, colmask) in sets.items():
                    d = data[src]
                    for cond in ("clean", "noisy"):
                        pred = rnn_predict(m, (d["z"][cond] - mean) / std, d["keep"], dev)
                        ok = d["ok"][target] & (colmask if colmask is not None else True)
                        res[f"{target}/{part}/{cond}/{kind}_{basis}"] = float(
                            (d["Y"][target] - pred)[ok].norm(dim=-1).mean())
    if args.smoke:
        log_progress(f"A3e smoke ok: {len(res)} errors, q* {q_star['vel']['q']} / {q_star['acc']['q']}")
        return
    # Reused numbers (same data, split, noise draws and masks): MLP = NF basis, FD, FD_ema.
    with open(os.path.join(A3D, "pred_err.json")) as f:
        a3d = json.load(f)
    with open(os.path.join(A3A, "pred_err.json")) as f:
        a3a = json.load(f)[str(H_ACC)]
    for part in ("val", "test"):
        for cond in ("clean", "noisy"):
            for k, kk in (("MLP_clean", "NF_clean"), ("MLP_aug", "NF_aug"), ("FD", "FD_raw"), ("FD_ema", "FD_ema")):
                res[f"vel/{part}/{cond}/{k}"] = a3d[f"{part}/{cond}/{kk}"]
            for k, kk in (("MLP_clean", "NF_clean"), ("MLP_aug", "NF_aug"), ("FD", "FD")):
                res[f"acc/{part}/{cond}/{k}"] = a3a[f"{part}/{cond}/{kk}"]
            if cond == "noisy":
                res[f"acc/{part}/noisy/FD_ema"] = a3a[f"{part}/noisy/FD_ema"]
            # Consistency check: the zero baseline must match the reused experiments (same masks).
            for target, ref in (("vel", a3d), ("acc", a3a)):
                if abs(res[f"{target}/{part}/{cond}/zero"] - ref[f"{part}/{cond}/zero"]) > 1e-3:
                    raise RuntimeError(f"A3e mask mismatch {target}/{part}/{cond}")
    save_json(os.path.join(A3E, "pred_err.json"), {"errors": res, "kalman_q": q_star})
    keys = ["zero", "FD", "FD_ema", "KF", "MLP_clean", "MLP_aug", "GRU_clean", "GRU_aug", "LSTM_clean", "LSTM_aug"]
    lines = ["# A3e_estimators summary", "",
             "Mean error (m/s) from inputs without evader velocity; noisy = sensor-noise model (0.02 m, 0.02 m, "
             "0.05 m/s). MLP = 10-step window (A3a / A3d), FD / FD_ema / MLP reused from A3a / A3d. "
             f"KF q* (tuned on noisy val): vel {q_star['vel']['q']:g}, acc {q_star['acc']['q']:g}.", ""]
    for target, title in (("vel", "Evader velocity (A3d target)"),
                          ("acc", f"Evader velocity change over {H_ACC} steps (A3a target)")):
        lines += [f"## {title}", "", "| set | inputs | " + " | ".join(keys) + " |", "|---" * (len(keys) + 2) + "|"]
        for part in ("val", "test"):
            for cond in ("clean", "noisy"):
                lines.append(f"| {part} | {cond} | " + " | ".join(
                    f"{res[f'{target}/{part}/{cond}/{k}']:.4f}" if f"{target}/{part}/{cond}/{k}" in res else "-"
                    for k in keys) + " |")
        lines.append("")
    with open(os.path.join(A3E, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    best = lambda t: min((k for k in keys if k not in ("zero", "FD")), key=lambda k: res[f"{t}/test/noisy/{k}"])
    log_progress(f"A3e done: noisy test best vel {best('vel')} {res['vel/test/noisy/' + best('vel')]:.3f}, "
                 f"best acc {best('acc')} {res['acc/test/noisy/' + best('acc')]:.3f} m/s")
