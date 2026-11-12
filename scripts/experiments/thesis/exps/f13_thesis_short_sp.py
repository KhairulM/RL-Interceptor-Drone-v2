"""F13: the original thesis rules with short stages (docs/research_plan.md, F13_thesis_short_sp; pre-registered
2026-11-13 01:16 before any F13 run; user decision after F11).

Campaigns thesis-F13-short-s{100,200} = F9 control exactly (thesis rules, GAE fix, Hedge both roles, 13 x 50M); seed 0
is thesis-F9-control-s0. F13c: final pursuers (stage 12) vs train pool, test set, F3 evaders, F8d's finals, the F13 and
F11 campaigns' evaders, a thesis-rules exploiter (seed 0: F9c's) and a v2d-rules exploiter; cross-play and stage
control for F11 (eval only); decision rule: does v2d (F11) beat this setup on >= 3 of 4 measures?
"""

import json
import os
import time

import numpy as np

from a_adapt import test_set
from common import OUT, evaluate, log_progress, save_json, train
from d_sampling import EXPLOIT_FRAMES
from f2_final import _done, run_campaign
from f3_sampler_sp import SAMPLERS, _policies
from f3_sampler_sp import name as f3_name
import f8_roncero_sp as f8
import f8d_heldout as f8d
import f9_ablation_sp as f9
import f11_v2_sp as f11
from rescore import rates_at
from x3_gap import load_split

F13 = os.path.join(OUT, "F13_thesis_short_sp")
SEEDS = (0, 100, 200)
NEW_SEEDS = (100, 200)
FRAMES, STAGES = 50_000_000, 13
COMMON = ["algo.gae_cut_at_truncation=true"]
RADII = (0.10, 0.15)
F3_ARMS = ("uniform", "hedge", "qd_dpp", "pfsp")
THESIS_EXPLOITER = f11.OLD_RULES_EXPLOITER
THESIS_EXPLOITER_SEED = {100: 900, 200: 901}
V2D_EXPLOITER_SEED = {0: 910, 100: 911, 200: 912}
F11_SEEDS = f11.SEEDS
DECISION = ("heldout_rl", "strong_finals_mean", "exploitability_thesis", "cross_play")
LOWER_IS_BETTER = {"exploitability_thesis"}


def name(seed):
    return f9.name("control") if seed == 0 else f"thesis-F13-short-s{seed}"


def cfg(seed):
    return dict(SAMPLERS["hedge"], seed=seed, frames_per_stage=FRAMES, n_stages=STAGES, max_total_stages=STAGES)


def run_f13(args):
    if args.smoke:
        raise SystemExit("F13 has no smoke run (same code path as F9 control, pre-registered)")
    for seed in NEW_SEEDS:                                  # lanes skip a campaign the other lane holds
        run_campaign(name(seed), COMMON, cfg(seed))


# ---- F13c
def _src(c, g):
    return os.path.join(F13, "evals", f"{c}_{g}.json")


def _final(c):
    return f8.all_stages(c, "pursuer")[-1]["checkpoint_path"]


def _stage_ckpt(c, stage):
    return next(p for p in f8.all_stages(c, "pursuer") if int(p["metadata"]["stage"]) == stage)["checkpoint_path"]


def _exploiter(c, seed, tag):
    if tag == "thesis" and seed == 0:
        path = os.path.join(f9.F9, "ckpt", f"exploiter_thesis_{c}.pt")         # F9c's, same recipe
    else:
        path = os.path.join(F13, "ckpt", f"exploiter_{tag}_{c}.pt")
        extra = THESIS_EXPLOITER if tag == "thesis" else f11.rules(**f11.pilot_decision()["settings"])
        s = THESIS_EXPLOITER_SEED[seed] if tag == "thesis" else V2D_EXPLOITER_SEED[seed]
        train(path, [{"id": f"f13/pursuer:{c}", "kind": "rl", "checkpoint_path": _final(c), "metadata": {}}],
              EXPLOIT_FRAMES, role="evader", seed=s, extra=list(extra))
    return ([{"id": f"f13/evader:exploit_{tag}_{c}", "kind": "rl", "checkpoint_path": path, "metadata": {}}]
            if os.path.isfile(path) else [])


GROUPS = ("final_train", "final_test", "final_f3_evaders", "final_finals", "final_f13_evaders", "final_f11_evaders",
          "final_vs_exploiter_thesis", "final_vs_exploiter_v2d")
F11_GROUPS = ("final_f13_evaders", "P12_test", "P12_finals", "P12_f13_evaders")


def _jobs(split, test):
    """[(campaign, group, opponents, checkpoint)] of every F13c evaluation."""
    f3_ev = [e for s in F3_ARMS for k in range(2) for e in _policies(f3_name(s, k), "evader")]
    f13_ev = [e for s in SEEDS for e in f8.all_stages(name(s), "evader")]
    f11_ev = [e for k in range(len(F11_SEEDS)) for e in f8.all_stages(f11.name(k), "evader")]
    jobs = []
    for seed in SEEDS:
        c, ck = name(seed), _final(name(seed))
        jobs += [(c, "final_train", split["train"], ck), (c, "final_test", test, ck), (c, "final_f3_evaders", f3_ev, ck),
                 (c, "final_finals", f8d._finals(), ck), (c, "final_f13_evaders", f13_ev, ck),
                 (c, "final_f11_evaders", f11_ev, ck),
                 (c, "final_vs_exploiter_thesis", lambda c=c, s=seed: _exploiter(c, s, "thesis"), ck),
                 (c, "final_vs_exploiter_v2d", lambda c=c, s=seed: _exploiter(c, s, "v2d"), ck)]
    for k in range(len(F11_SEEDS)):
        c = f11.name(k)
        p12 = _stage_ckpt(c, 12)
        jobs += [(c, "final_f13_evaders", f13_ev, _final(c)), (c, "P12_test", test, p12),
                 (c, "P12_finals", f8d._finals(), p12), (c, "P12_f13_evaders", f13_ev, p12)]
    return jobs


def run_f13c(args):
    for seed in SEEDS:
        d = _done(name(seed))
        while not os.path.isfile(d):
            if not os.path.isfile(d + ".lock"):
                raise RuntimeError(f"F13c: campaign {name(seed)} is neither finished nor running")
            time.sleep(60)
    split = load_split()
    test = test_set(split)
    for c, g, opps, ck in _jobs(split, test):
        if os.path.isfile(_src(c, g)):
            continue
        opps = opps() if callable(opps) else opps
        if opps:
            evaluate(_src(c, g), opps, policy_ckpt=ck)
    missing = [_src(c, g) for c, g, _, _ in _jobs(split, test) if not os.path.isfile(_src(c, g))]
    if missing:                                 # the other lane is still producing them and analyses after
        log_progress(f"F13c: {len(missing)} eval files missing on this lane, analysis left to the other lane")
        return
    analyse_f13(split, test)


# ---- analysis
def _metrics(c, radius, split, test, labels, own_prefix):
    """F13 pursuer metrics from this experiment's eval files (prefix final_)."""
    rate = lambda g, key="neutralization_rate": rates_at(_src(c, g), key, radius)
    no_s1 = lambda d: {k: v for k, v in d.items() if not k.endswith(":s1")}
    camp = lambda e: e.split("/")[0]
    J = {**rate("final_train"), **rate("final_test")}
    ids = lambda opps: [J[e["id"]] for e in opps if e["id"] in J]
    F, J13, J11 = rate("final_finals"), no_s1(rate("final_f13_evaders")), no_s1(rate("final_f11_evaders"))
    common_f = [e for e in F if camp(e) in f8d.COMMON]
    r = {"train": ids(split["train"]), "heldout_rl": ids(split["heldout_rl"]),
         "heldout_scripted": ids(split["heldout_scripted"]),
         "x3_exploiters": [J[e["id"]] for e in test if e["id"].startswith("exploit/") and e["id"] in J],
         "other_f3_evaders": list(no_s1(rate("final_f3_evaders")).values()),
         "strong_finals_mean": [F[e] for e in common_f],
         "vs_f8_finals": [F[e] for e in F if camp(e) in f8d.F8C],
         "vs_f3_hedge_finals": [F[e] for e in F if camp(e) in f8d.HEDGE],
         "other_f13_evaders": [v for k, v in J13.items() if not k.startswith(own_prefix)],
         "own_campaign_evaders": [v for k, v in J13.items() if k.startswith(own_prefix)],
         "cross_play": list(J11.values())}
    r = {m: float(np.mean(v)) for m, v in r.items() if v}
    r["strong_finals_min"] = float(min(F[e] for e in common_f))
    by_c = {}
    for e in split["heldout_rl"] + split["heldout_scripted"]:
        if e["id"] in J and e["id"] in labels:
            by_c.setdefault(labels[e["id"]], []).append(J[e["id"]])
    r["heldout_worst_cluster"] = float(min(np.mean(v) for v in by_c.values()))
    pc = rate("final_test", "pursuer_crash")
    r["heldout_scripted_pursuer_crash"] = float(np.mean([pc[e["id"]] for e in split["heldout_scripted"] if e["id"] in pc]))
    for tag in ("thesis", "v2d"):
        g = f"final_vs_exploiter_{tag}"
        r[f"exploitability_{tag}"] = 1 - float(next(iter(rate(g).values())))
        for key in ("pursuer_crash", "timeout"):
            r[f"exploiter_{tag}_{key}"] = float(next(iter(rate(g, key).values())))
    return r


def _f11_metrics(k, radius, split, test):
    """F11 final pursuer: F11c's metrics (exploitability_old = the same thesis-rules recipe) + cross-play vs F13;
    F11 stage-12 pursuer: held-out RL, strong finals, J vs F13 evaders."""
    c = f11.name(k)
    m = json.load(open(os.path.join(f11.F11, "results.json")))["metrics"][f"r{radius:.2f}"][f"v2/{k}"]
    no_s1 = lambda d: {kk: v for kk, v in d.items() if not kk.endswith(":s1")}
    rate = lambda g: rates_at(_src(c, g), "neutralization_rate", radius)
    camp = lambda e: e.split("/")[0]
    final = dict(m, exploitability_thesis=m["exploitability_old"],
                 exploiter_thesis_pursuer_crash=m["exploiter_old_pursuer_crash"],
                 cross_play=float(np.mean(list(no_s1(rate("final_f13_evaders")).values()))))
    J, F = rate("P12_test"), rate("P12_finals")
    p12 = {"heldout_rl": float(np.mean([J[e["id"]] for e in split["heldout_rl"] if e["id"] in J])),
           "strong_finals_mean": float(np.mean([F[e] for e in F if camp(e) in f8d.COMMON])),
           "cross_play": float(np.mean(list(no_s1(rate("P12_f13_evaders")).values())))}
    return final, p12


METRICS = ["heldout_rl", "heldout_worst_cluster", "strong_finals_mean", "strong_finals_min", "vs_f8_finals",
           "vs_f3_hedge_finals", "other_f3_evaders", "other_f13_evaders", "own_campaign_evaders", "cross_play", "train",
           "heldout_scripted", "heldout_scripted_pursuer_crash", "x3_exploiters", "exploitability_thesis",
           "exploiter_thesis_pursuer_crash", "exploiter_thesis_timeout", "exploitability_v2d",
           "exploiter_v2d_pursuer_crash", "exploiter_v2d_timeout"]


def decision(res):
    """Pre-registered rule at 0.10 m: v2d beats F13's setup if its seed mean is better on >= 3 of DECISION."""
    r = res["r0.10"]
    mean = lambda labs, m: float(np.mean([r[a][m] for a in labs]))
    t13, t11 = [f"f13/{s}" for s in SEEDS], [f"f11/{k}" for k in range(len(F11_SEEDS))]
    out = {}
    for m in DECISION:
        a, b = mean(t11, m), mean(t13, m)
        out[m] = {"v2d": a, "f13": b, "v2d_better": bool(a < b if m in LOWER_IS_BETTER else a > b)}
    wins = sum(v["v2d_better"] for v in out.values())
    return {"measures": out, "v2d_wins": wins, "v2d_beats_f13": wins >= 3,
            "action": "resume F12" if wins >= 3 else "re-run F12 on F13's setup"}


def analyse_f13(split, test):
    from f5_outcomes import stage_table
    with open(os.path.join(OUT, "X2_descriptors", "clusters.json")) as f:
        labels = json.load(f)["labels"]
    res, p12 = {}, {}
    for rad in RADII:
        k = f"r{rad:.2f}"
        res[k] = {f"f13/{s}": _metrics(name(s), rad, split, test, labels, name(s) + "/") for s in SEEDS}
        p12[k] = {}
        for i in range(len(F11_SEEDS)):
            res[k][f"f11/{i}"], p12[k][f"f11/{i}"] = _f11_metrics(i, rad, split, test)
    lures = {name(s): f8.lure_rates(name(s)) for s in SEEDS}
    dec = decision(res)
    save_json(os.path.join(F13, "results.json"), {"metrics": res, "f11_stage12": p12, "lure_rates": lures,
                                                  "measure_a": {c: f11.measure_a(l) for c, l in lures.items()},
                                                  "decision": dec})
    labs = list(res["r0.10"])
    lines = ["# F13_thesis_short_sp summary", "",
             "Final pursuers (stage 12) of thesis-F9-control-s0 / thesis-F13-short-s{100,200} (thesis rules, GAE fix, Hedge "
             "both roles, 13 x 50M) and F11's v2d final pursuers (stage 24). Evals clean under the thesis rules; "
             "exploitability_thesis = 1 - J vs a 100M-frame thesis-rules exploiter (F11: exploitability_old, same "
             "recipe); _v2d vs a v2d-rules exploiter (F11: F11c's v2 exploiter). cross_play = J vs the other rule set's "
             "campaign evaders. 0.10 m primary.", "",
             f"**Decision (pre-registered)**: v2d better on {dec['v2d_wins']} of 4 -> **{dec['action']}**", "",
             "| measure | v2d mean | F13 mean | v2d better |", "|---|---|---|---|"]
    lines += [f"| {m} | {v['v2d']:.3f} | {v['f13']:.3f} | {v['v2d_better']} |" for m, v in dec["measures"].items()]
    lines += ["", "In-campaign lure rate (a) (mean over evader stages 3-11): " +
              ", ".join(f"{c} {f11.measure_a(l):.2f}" for c, l in lures.items()), ""]
    for k, v in res.items():
        lines += [f"## Capture radius {k[1:]} m", "", "| metric | " + " | ".join(labs) + " |",
                  "|---" * (len(labs) + 1) + "|"]
        f11m = lambda a, m: m in v[a]
        lines += [f"| {m} | " + " | ".join(f"{v[a][m]:.3f}" if f11m(a, m) else "-" for a in labs) + " |"
                  for m in METRICS]
        lines += ["", "F11 stage-12 pursuers (same stage count as F13): " +
                  "; ".join(f"{a}: " + ", ".join(f"{m} {x:.3f}" for m, x in d.items()) for a, d in p12[k].items()), ""]
    lines += ["## In-campaign lure rates", "", "| campaign | evader stage | mean | min - max | n pursuers |",
              "|---|---|---|---|---|"]
    for c, lr in lures.items():
        for s, d in sorted(lr.items()):
            if d:
                x = list(d.values())
                lines.append(f"| {c} | {s} | {np.mean(x):.2f} | {min(x):.2f} - {max(x):.2f} | {len(x)} |")
    lines += ["", "## Inside the campaigns", ""]
    for s in NEW_SEEDS:
        lines += stage_table(name(s), STAGES)
    with open(os.path.join(F13, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log_progress(f"F13c analysis done: {dec['action']}")
