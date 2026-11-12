"""F11: self-play under the thesis v2 rules (docs/research_plan.md, F11_v2_sp; pre-registered 2026-11-11; amendment 2
after the failed v2a pilot: plain-difference shaping; amendment 3 after the failed v2b gate: weights 7 / 10, user choice A; amendment 4 after the failed v2c gate: the pursuer
keeps its -100 timeout, only the evader's timeout bonus is removed).

Thesis v2 (v2d) = the thesis rules with: Roncero-style endings (crashes and the evader leaving the arena are
truncations, `task.ablation.truncate_failures`; the pursuer may still leave freely), no timeout bonus for the evader (the pursuer keeps -100 at timeout;
`task.ablation.no_timeout_evader`), no closing-speed factor, plain-difference shaping with the original weights (approach 7 x
delta d, precision 10 x delta exp(-5 d)), smoothness 0.5 and command rate 0.2 unchanged, GAE fix.
Gate `thesis-F11v2d-gate` (stage 0 only): its pursuer must capture the 4 scripted training evaders >= 0.80, else stop.
Pilot `thesis-F11v2d-pilot` (13 x 50M) fixes k_app / w_prec (`pilot_decision`); then `thesis-F11v2d-s{0,100}`
(26 x 50M, Hedge both roles) and F11c (final pursuers, stage 24).
"""

import json
import os
import re
import time

import numpy as np

import common
from a_adapt import test_set
from common import OUT, evaluate, log_progress, save_json, train
from d_sampling import EXPLOIT_FRAMES
from f2_final import _done, run_campaign
from f3_sampler_sp import SAMPLERS, _policies
from f3_sampler_sp import name as f3_name
import f8_roncero_sp as f8
import f8d_heldout as f8d
from rescore import rates_at
from x3_gap import load_split

F11 = os.path.join(OUT, "F11_v2_sp")
SEEDS = (0, 100)
FRAMES, STAGES, PILOT_STAGES = 50_000_000, 26, 13
K_APP, W_PREC, W_SMOOTH, W_CMD = 7.0, 10.0, 0.5, 0.2          # the pilot may double k_app / w_prec (pilot_decision)
F3_ARMS = ("uniform", "hedge", "qd_dpp", "pfsp")
RADII = (0.10, 0.15)
OLD_RULES_EXPLOITER = ["algo.gae_cut_at_truncation=false"]   # thesis rules exploiter, same recipe as F3c / F9c


def rules(k_app=K_APP, w_prec=W_PREC, w_smooth=W_SMOOTH, w_cmd=W_CMD):
    return ["task.ablation.truncate_failures=true", "task.ablation.no_timeout_evader=true",
            f"task.reward_approach_weight={k_app}", f"task.reward_precision_weight={w_prec}",
            f"task.reward_action_smoothness_weight={w_smooth}", f"task.reward_cmd_rate_weight={w_cmd}",
            "algo.gae_cut_at_truncation=true"]


def name(k, pilot=False, gate=False):
    """v2d names (failed: v2a pilot thesis-F11p-s0, v2b gate thesis-F11b-gate, v2c gate thesis-F11v2c-gate)."""
    return "thesis-F11v2d-gate" if gate else ("thesis-F11v2d-pilot" if pilot else f"thesis-F11v2d-s{SEEDS[k]}")


SCRIPTED = ("hover", "circular", "flee", "weave")             # the campaigns' seed heuristics


def cfg(seed, n_stages):
    return dict(SAMPLERS["hedge"], seed=seed, frames_per_stage=FRAMES, n_stages=n_stages, max_total_stages=n_stages)


def _wait(c):
    d = _done(c)
    while not os.path.isfile(d):
        if not os.path.isfile(d + ".lock"):
            raise RuntimeError(f"F11: campaign {c} is neither finished nor running")
        time.sleep(60)


# ---- pilot: per-episode reward-term sizes of the final pursuer stage and its J vs scripted evaders
def _stage_stats(c, stage, role="pursuer"):
    """Mean of the logged episode stats over the second half of the training iterations of one stage, from the
    campaign log (worker stdout)."""
    lines = open(os.path.join(common.SELFPLAY_OUT, f"{c}_log.txt"), errors="ignore").read().split("\n")
    # the worker's own lines (the parent's "[worker]" prints are block-buffered and land out of order)
    i1 = max(i for i, l in enumerate(lines) if "saved checkpoint to" in l and f"/{role}_stage{stage}.pt" in l)
    i0 = max(i for i in range(i1) if "Passing the following args to the base kit application" in lines[i])
    rows = {}
    for l in lines[i0:i1]:
        m = re.match(r"train/stats\.([a-z_]+): (-?[0-9.eE+-]+)", l.strip())
        if m:
            rows.setdefault(m.group(1), []).append(float(m.group(2)))
    return {k: float(np.mean(v[len(v) // 2:])) for k, v in rows.items()}


def scripted_capture(c, stage):
    """Mean capture rate of a campaign's pursuer stage vs the 4 scripted training evaders (explicit eval)."""
    ck = next(p for p in f8.all_stages(c, "pursuer") if int(p["metadata"]["stage"]) == stage)["checkpoint_path"]
    out = os.path.join(F11, "evals", f"{c}_P{stage}_scripted.json")
    if not os.path.isfile(out):
        evaluate(out, [common.heuristic(h, "evader") for h in SCRIPTED], policy_ckpt=ck)
    while not os.path.isfile(out) and os.path.isfile(out + ".lock"):     # the other lane is running this eval
        time.sleep(20)
    m = json.load(open(out))["matchups"]
    return float(np.mean([v["capture_rate"] for v in m.values()])), float(np.mean([v["neutralization_rate"]
                                                                                    for v in m.values()]))


def pilot_decision():
    """Pre-registered rule (research plan F11, amendment 2): if the final pilot pursuer's (stage 12) J vs the 4
    scripted training evaders (explicit eval) < 0.80, double k_app and w_prec; else keep them. Applied once. The
    episode sums (`sum_shaping`, `sum_regularisation`) are recorded, not used."""
    path = os.path.join(F11, "pilot_decision_v2d.json")
    if os.path.isfile(path):
        return json.load(open(path))
    c, s = name(0, pilot=True), 2 * (PILOT_STAGES // 2)
    st = _stage_stats(c, s)
    cap, j_scripted = scripted_capture(c, s)
    k_app, w_prec = (K_APP * 2, W_PREC * 2) if j_scripted < 0.80 else (K_APP, W_PREC)
    out = {"stats": st, "j_scripted": j_scripted, "capture_scripted": cap,
           "settings": {"k_app": k_app, "w_prec": w_prec, "w_smooth": W_SMOOTH, "w_cmd": W_CMD}}
    save_json(path, out)
    log_progress(f"F11 pilot decision: {out['settings']} (J scripted {j_scripted:.3f}, capture {cap:.3f})")
    return out


def run_f11(args):
    gate = name(0, gate=True)
    run_campaign(gate, rules(), cfg(SEEDS[0], 1))
    _wait(gate)
    cap, _ = scripted_capture(gate, 0)
    log_progress(f"F11 gate: stage-0 pursuer capture vs scripted {cap:.3f}")
    if cap < 0.80:
        raise RuntimeError(f"F11 gate failed: capture vs scripted {cap:.3f} < 0.80; pilot and campaigns not started")
    if args.smoke:
        return
    pilot = name(0, pilot=True)
    run_campaign(pilot, rules(), cfg(SEEDS[0], PILOT_STAGES))
    _wait(pilot)
    rr = rules(**pilot_decision()["settings"])
    for k in range(len(SEEDS)):
        run_campaign(name(k), rr, cfg(SEEDS[k], STAGES))


# ---- F11c
def _src(c, g):
    return os.path.join(F11, "evals", f"{c}_final_{g}.json")


def _exploiter(c, tag, extra, seed):
    path = os.path.join(F11, "ckpt", f"exploiter_{tag}_{c}.pt")
    ck = f8.all_stages(c, "pursuer")[-1]["checkpoint_path"]
    train(path, [{"id": f"f11/pursuer:{c}", "kind": "rl", "checkpoint_path": ck, "metadata": {}}], EXPLOIT_FRAMES,
          role="evader", seed=seed, extra=list(extra))
    return ([{"id": f"f11/evader:exploit_{tag}_{c}", "kind": "rl", "checkpoint_path": path, "metadata": {}}]
            if os.path.isfile(path) else [])


GROUPS = ("train", "test", "f3_evaders", "finals", "f11_evaders", "vs_exploiter_v2", "vs_exploiter_old")


def run_f11c(args):
    for k in range(len(SEEDS)):
        _wait(name(k))
    rr = rules(**pilot_decision()["settings"])
    split = load_split()
    test = test_set(split)
    f3_ev = [e for s in F3_ARMS for k in range(2) for e in _policies(f3_name(s, k), "evader")]
    f11_ev = [e for k in range(len(SEEDS)) for e in f8.all_stages(name(k), "evader")]
    for i in range(len(SEEDS)):
        c = name(i)
        ck = f8.all_stages(c, "pursuer")[-1]["checkpoint_path"]
        jobs = [("train", split["train"], None), ("test", test, None), ("f3_evaders", f3_ev, None),
                ("finals", f8d._finals(), None), ("f11_evaders", f11_ev, None),
                ("vs_exploiter_v2", _exploiter(c, "v2", rr, 860 + i), None),
                ("vs_exploiter_old", _exploiter(c, "old", OLD_RULES_EXPLOITER, 870 + i), None)]
        for g, opps, extra in jobs:
            if opps and not os.path.isfile(_src(c, g)):
                evaluate(_src(c, g), opps, policy_ckpt=ck, extra=extra)
        log_progress(f"F11c {c} done")
    missing = [_src(name(k), g) for k in range(len(SEEDS)) for g in GROUPS if not os.path.isfile(_src(name(k), g))]
    if missing:
        log_progress(f"F11c: {len(missing)} eval files missing on this lane, analysis left to the other lane")
        return
    analyse_f11(split, test)


def measure_a(lr):
    """(a) as F9 / F10: mean over evader stages 3-11 of each stage's mean in-campaign lure rate."""
    return float(np.mean([np.mean(list(d.values())) for s, d in lr.items() if 3 <= s <= 11 and d]))


def _metrics(c, radius, split, test, labels):
    rate = lambda g, key="neutralization_rate": rates_at(_src(c, g), key, radius)
    J = {**rate("train"), **rate("test")}
    no_s1 = lambda d: {k: v for k, v in d.items() if not k.endswith(":s1")}
    J3, J11, F = no_s1(rate("f3_evaders")), no_s1(rate("f11_evaders")), rate("finals")
    camp = lambda e: e.split("/")[0]
    ids = lambda opps: [J[e["id"]] for e in opps if e["id"] in J]
    common_f = [e for e in F if camp(e) in f8d.COMMON]
    r = {"train": ids(split["train"]), "heldout_rl": ids(split["heldout_rl"]),
         "heldout_scripted": ids(split["heldout_scripted"]),
         "x3_exploiters": [J[e["id"]] for e in test if e["id"].startswith("exploit/") and e["id"] in J],
         "other_f3_evaders": list(J3.values()),
         "strong_finals_mean": [F[e] for e in common_f],
         "vs_f8_finals": [F[e] for e in F if camp(e) in f8d.F8C],
         "vs_f3_hedge_finals": [F[e] for e in F if camp(e) in f8d.HEDGE],
         "other_f11_evaders": [v for k, v in J11.items() if not k.startswith(c + "/")],
         "own_campaign_evaders": [v for k, v in J11.items() if k.startswith(c + "/")]}
    r = {m: float(np.mean(v)) for m, v in r.items() if v}
    r["strong_finals_min"] = float(min(F[e] for e in common_f))
    by_c = {}
    for e in split["heldout_rl"] + split["heldout_scripted"]:
        if e["id"] in J and e["id"] in labels:
            by_c.setdefault(labels[e["id"]], []).append(J[e["id"]])
    r["heldout_worst_cluster"] = float(min(np.mean(v) for v in by_c.values()))
    pc = rate("test", "pursuer_crash")
    r["heldout_scripted_pursuer_crash"] = float(np.mean([pc[e["id"]] for e in split["heldout_scripted"] if e["id"] in pc]))
    for tag in ("v2", "old"):
        g = f"vs_exploiter_{tag}"
        r[f"exploitability_{tag}"] = 1 - float(next(iter(rate(g).values())))
        for key in ("pursuer_crash", "timeout"):
            r[f"exploiter_{tag}_{key}"] = float(next(iter(rate(g, key).values())))
    return r


METRICS = ["heldout_rl", "heldout_worst_cluster", "strong_finals_mean", "strong_finals_min", "vs_f8_finals",
           "vs_f3_hedge_finals", "other_f3_evaders", "other_f11_evaders", "own_campaign_evaders", "train",
           "heldout_scripted", "heldout_scripted_pursuer_crash", "x3_exploiters", "exploitability_v2",
           "exploiter_v2_pursuer_crash", "exploiter_v2_timeout", "exploitability_old", "exploiter_old_pursuer_crash",
           "exploiter_old_timeout"]


def analyse_f11(split, test):
    from f5_outcomes import stage_table
    with open(os.path.join(OUT, "X2_descriptors", "clusters.json")) as f:
        labels = json.load(f)["labels"]
    labs = [f"v2/{k}" for k in range(len(SEEDS))]
    res = {f"r{rad:.2f}": {f"v2/{k}": _metrics(name(k), rad, split, test, labels) for k in range(len(SEEDS))}
           for rad in RADII}
    lures = {name(k): f8.lure_rates(name(k)) for k in range(len(SEEDS))}
    dec = pilot_decision()
    save_json(os.path.join(F11, "results.json"), {"metrics": res, "lure_rates": lures, "pilot": dec,
                                                  "measure_a": {c: measure_a(l) for c, l in lures.items()}})
    lines = ["# F11_v2_sp summary", "",
             "Final pursuers (stage 24) of thesis-F11v2d-s{0,100}: thesis v2 rules (truncation endings, no evader timeout bonus, "
             "plain-difference shaping 7 / 10, no closing-speed factor, GAE fix), Hedge both roles, 26 x 50M. Settings "
             f"from the pilot: {dec['settings']} (J vs scripted {dec['j_scripted']:.3f}). Evals clean under the "
             "thesis rules; exploitability_v2 / _old = 1 - J vs a fresh 100M-frame exploiter trained under thesis v2 / "
             "the old thesis rules (luring pays). strong_finals = F8d's 16 final evaders. 2 seeds, descriptive.", "",
             "Pre-registered (a), in-campaign lure rate (mean over evader stages 3-11): " +
             ", ".join(f"{c} {measure_a(l):.2f}" for c, l in lures.items()), ""]
    for k, v in res.items():
        lines += [f"## Capture radius {k[1:]} m", "", "| metric | " + " | ".join(labs) + " |",
                  "|---" * (len(labs) + 1) + "|"]
        lines += [f"| {m} | " + " | ".join(f"{v[a][m]:.3f}" if m in v[a] else "-" for a in labs) + " |" for m in METRICS]
        lines.append("")
    lines += ["## In-campaign lure rates", "", "| campaign | evader stage | mean | min - max | n pursuers |",
              "|---|---|---|---|---|"]
    for c, lr in lures.items():
        for s, d in sorted(lr.items()):
            if d:
                x = list(d.values())
                lines.append(f"| {c} | {s} | {np.mean(x):.2f} | {min(x):.2f} - {max(x):.2f} | {len(x)} |")
    lines += ["", "## Inside the campaigns", ""]
    for k in range(len(SEEDS)):
        lines += stage_table(name(k), STAGES)
    with open(os.path.join(F11, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log_progress("F11c analysis done")
