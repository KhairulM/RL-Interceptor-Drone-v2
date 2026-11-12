"""F12: does the learned acceleration estimator still help under the thesis v2 rules? (docs/research_plan.md,
F12_acc_v2_sp; pre-registered 2026-11-12 before any F12 run; user request, queued after F11.)

Campaigns thesis-F12-acc-s{0,100} = F11 v2d (rules and pilot settings from F11, Hedge both roles, 26 x 50M, same
seeds) with F5's pursuer input: the learned acceleration estimate with online RLS (A2b NFacc_OA, F5 estimator), frozen
pursuers in the evader worker running the same estimator on clean inputs (F5 `_roles`). Paired by seed with
thesis-F11v2d-s{0,100}. Gate (stage 0 only) as F11. F12c: final pursuers (stage 24, estimator on) vs train pool, test
set, the default held-out set (v10 + F8d's 16 strong finals), F3 campaigns' evaders, the F11 and F12 campaigns'
evaders, and two fresh exploiters (thesis v2d rules; old thesis rules); RLS-frozen ablation on the held-out sets.
"""

import json
import os
import time

import numpy as np

import common
from a_adapt import test_set
from common import OUT, evaluate, log_progress, save_json, train
from d_sampling import EXPLOIT_FRAMES
from f2_final import _done, run_campaign
from f3_sampler_sp import _policies
from f3_sampler_sp import name as f3_name
from f4_estimator_sp import _mirror
import f5_acc_sp as f5
import f8_roncero_sp as f8
import f8d_heldout as f8d
import f11_v2_sp as f11
from rescore import rates_at
from x3_gap import load_split

F12 = os.path.join(OUT, "F12_acc_v2_sp")
SEEDS = f11.SEEDS
RADII = (0.10, 0.15)
F3_ARMS = ("uniform", "hedge", "qd_dpp", "pfsp")


def name(k, gate=False):
    return "thesis-F12-acc-gate" if gate else f"thesis-F12-acc-s{SEEDS[k]}"


def rules():
    return f11.rules(**f11.pilot_decision()["settings"])


def pov():
    return f5.rls_overrides()


def _scripted_capture(c, stage):
    ck = next(p for p in f8.all_stages(c, "pursuer") if int(p["metadata"]["stage"]) == stage)["checkpoint_path"]
    out = os.path.join(F12, "evals", f"{c}_P{stage}_scripted.json")
    if not os.path.isfile(out):
        evaluate(out, [common.heuristic(h, "evader") for h in f11.SCRIPTED], policy_ckpt=ck, extra=pov())
    while not os.path.isfile(out) and os.path.isfile(out + ".lock"):
        time.sleep(20)
    m = json.load(open(out))["matchups"]
    return float(np.mean([v["capture_rate"] for v in m.values()]))


def run_f12(args):
    roles = f5._roles()
    gate = name(0, gate=True)
    run_campaign(gate, rules(), f11.cfg(SEEDS[0], 1), role_overrides=roles)
    f11._wait(gate)
    cap = _scripted_capture(gate, 0)
    log_progress(f"F12 gate: stage-0 pursuer capture vs scripted {cap:.3f}")
    if cap < 0.80:
        raise RuntimeError(f"F12 gate failed: capture vs scripted {cap:.3f} < 0.80; campaigns not started")
    if args.smoke:
        return
    for k in range(len(SEEDS)):
        run_campaign(name(k), rules(), f11.cfg(SEEDS[k], f11.STAGES), role_overrides=roles)


def _src(c, g, root=F12):
    return os.path.join(root, "evals", f"{c}_final_{g}.json")


GROUPS = ("train", "test", "heldout_v2", "f3_evaders", "camp_evaders", "vs_exploiter_v2", "vs_exploiter_old",
          "test_frozen", "heldout_v2_frozen")


def _exploiter(c, tag, extra, seed):
    path = os.path.join(F12, "ckpt", f"exploiter_{tag}_{c}.pt")
    ck = f8.all_stages(c, "pursuer")[-1]["checkpoint_path"]
    train(path, [{"id": f"f12/pursuer:{c}", "kind": "rl", "checkpoint_path": ck, "metadata": {}}], EXPLOIT_FRAMES,
          role="evader", seed=seed, extra=list(extra))
    return ([{"id": f"f12/evader:exploit_{tag}_{c}", "kind": "rl", "checkpoint_path": path, "metadata": {}}]
            if os.path.isfile(path) else [])


def run_f12c(args):
    for k in range(len(SEEDS)):
        f11._wait(name(k))
    split = load_split()
    test = test_set(split)
    held = f8d.heldout_v2(split)
    f3_ev = [e for s in F3_ARMS for k in range(2) for e in _policies(f3_name(s, k), "evader")]
    camp_ev = [e for k in range(len(SEEDS)) for c in (f11.name(k), name(k)) for e in f8.all_stages(c, "evader")]
    p, mir = pov(), _mirror(pov())
    for i in range(len(SEEDS)):
        c = name(i)
        ck = f8.all_stages(c, "pursuer")[-1]["checkpoint_path"]
        jobs = [("train", split["train"], p), ("test", test, p), ("heldout_v2", held, p), ("f3_evaders", f3_ev, p),
                ("camp_evaders", camp_ev, p),
                ("vs_exploiter_v2", _exploiter(c, "v2", rules() + mir, 880 + i), p),
                ("vs_exploiter_old", _exploiter(c, "old", f11.OLD_RULES_EXPLOITER + mir, 890 + i), p),
                ("test_frozen", test, p + [f5.FREEZE]), ("heldout_v2_frozen", held, p + [f5.FREEZE])]
        for g, opps, extra in jobs:
            if opps and not os.path.isfile(_src(c, g)):
                evaluate(_src(c, g), opps, policy_ckpt=ck, extra=extra)
        log_progress(f"F12c {c} done")
    # the paired F11 pursuers need the default held-out set too (plain observation)
    for k in range(len(SEEDS)):
        c = f11.name(k)
        out = _src(c, "heldout_v2")
        if not os.path.isfile(out):
            evaluate(out, held, policy_ckpt=f8.all_stages(c, "pursuer")[-1]["checkpoint_path"])
    need = [_src(name(k), g) for k in range(len(SEEDS)) for g in GROUPS] + \
           [_src(f11.name(k), "heldout_v2") for k in range(len(SEEDS))]
    missing = [x for x in need if not os.path.isfile(x)]
    if missing:
        log_progress(f"F12c: {len(missing)} eval files missing on this lane, analysis left to the other lane")
        return
    analyse_f12(split, test)


def _one(c, root, radius, split, test, labels, held_file, test_file, exploit=("vs_exploiter_v2", "vs_exploiter_old"),
         train_file="train"):
    rate = lambda f, key="neutralization_rate": rates_at(f, key, radius)
    J = {**rate(_src(c, train_file, root)), **rate(_src(c, test_file, root))}
    H = rate(held_file)
    ids = lambda opps: [J[e["id"]] for e in opps if e["id"] in J]
    strong = [v for e, v in H.items() if e.split("/")[0] in f8d.COMMON]
    r = {"train": float(np.mean(ids(split["train"]))), "heldout_v10": float(np.mean(ids(split["heldout_rl"]))),
         "heldout_v2": float(np.mean(list(H.values()))), "strong_finals_mean": float(np.mean(strong)),
         "strong_finals_min": float(min(strong)),
         "heldout_scripted": float(np.mean(ids(split["heldout_scripted"]))),
         "x3_exploiters": float(np.mean([J[e["id"]] for e in test if e["id"].startswith("exploit/") and e["id"] in J]))}
    by_c = {}
    for e in split["heldout_rl"] + split["heldout_scripted"]:
        if e["id"] in J and e["id"] in labels:
            by_c.setdefault(labels[e["id"]], []).append(J[e["id"]])
    r["heldout_worst_cluster"] = float(min(np.mean(v) for v in by_c.values()))
    for g in exploit:
        f = _src(c, g, root)
        if os.path.isfile(f):
            tag = g.split("_")[-1]
            r[f"exploitability_{tag}"] = 1 - float(next(iter(rate(f).values())))
            r[f"exploiter_{tag}_pursuer_crash"] = float(next(iter(rate(f, "pursuer_crash").values())))
    return r


def analyse_f12(split, test):
    with open(os.path.join(OUT, "X2_descriptors", "clusters.json")) as f:
        labels = json.load(f)["labels"]
    res = {}
    for rad in RADII:
        k_ = f"r{rad:.2f}"
        res[k_] = {}
        for k in range(len(SEEDS)):
            a, b = name(k), f11.name(k)
            res[k_][f"acc/{k}"] = _one(a, F12, rad, split, test, labels, _src(a, "heldout_v2"), "test")
            res[k_][f"acc_frozen/{k}"] = _one(a, F12, rad, split, test, labels, _src(a, "heldout_v2_frozen"),
                                              "test_frozen", exploit=())
            res[k_][f"plain/{k}"] = _one(b, f11.F11, rad, split, test, labels, _src(b, "heldout_v2"), "test")
    metrics = ["heldout_v2", "heldout_v10", "strong_finals_mean", "strong_finals_min", "heldout_worst_cluster", "train",
               "heldout_scripted", "x3_exploiters", "exploitability_v2", "exploiter_v2_pursuer_crash",
               "exploitability_old", "exploiter_old_pursuer_crash"]
    diffs = {k_: {f"acc - plain /{k}": {m: v[f"acc/{k}"][m] - v[f"plain/{k}"][m] for m in metrics
                                         if m in v[f"acc/{k}"] and m in v[f"plain/{k}"]} for k in range(len(SEEDS))}
             for k_, v in res.items()}
    save_json(os.path.join(F12, "results.json"), {"metrics": res, "differences": diffs})
    cols = [f"{a}/{k}" for k in range(len(SEEDS)) for a in ("acc", "acc_frozen", "plain")]
    lines = ["# F12_acc_v2_sp summary", "",
             "Final pursuers (stage 24) under thesis v2d: acc = learned acceleration estimate + online RLS (F5's "
             "NFacc_OA), acc_frozen = the same pursuer with RLS frozen at eval, plain = F11 v2d (same seed). heldout_v2 "
             "= v10 held-out RL + F8d's 16 strong finals (the default held-out set from 11-12). Evals clean under the "
             "thesis rules; exploitability_v2 / _old = 1 - J vs a fresh 100M-frame exploiter (thesis v2d / old thesis "
             "rules). 2 seeds, descriptive.", ""]
    for k_, v in res.items():
        dcols = list(diffs[k_])
        lines += [f"## Capture radius {k_[1:]} m", "", "| metric | " + " | ".join(cols + dcols) + " |",
                  "|---" * (1 + len(cols) + len(dcols)) + "|"]
        for m in metrics:
            cells = [f"{v[c][m]:.3f}" if m in v[c] else "-" for c in cols]
            cells += [f"{diffs[k_][d][m]:+.3f}" if m in diffs[k_][d] else "-" for d in dcols]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    with open(os.path.join(F12, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log_progress("F12c analysis done")
