"""F9: which part of Roncero et al.'s rules removes the floor-lure exploit? One change at a time on top of the thesis
rules (docs/research_plan.md, F9_ablation_sp; pre-registered 2026-11-10 before any F9 run; user request after F8).

Four campaigns, seed 0, Hedge on both roles, 13 stages x 50M frames, all with `algo.gae_cut_at_truncation=true`:
control = thesis rules; no_timeout = the timeout gives 0 to both roles; beta = the pursuer's capture reward x Roncero's
closing-speed factor; endings = crashes / the evader leaving the arena are truncations instead of terminations
(`task.ablation.*`, roncero_reward.ablate). F9c: each final pursuer (stage 12) under the thesis rules vs train pool,
test set, F3 campaigns' evaders, the F9 campaigns' evaders, and a fresh 100M-frame thesis-rules exploiter (where luring
pays). Every J at 0.10 m (primary) and 0.15 m (re-score).
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
from f3_sampler_sp import SAMPLERS, _policies
from f3_sampler_sp import name as f3_name
import f8_roncero_sp as f8
from rescore import rates_at
from x3_gap import load_split

F9 = os.path.join(OUT, "F9_ablation_sp")
COMMON = ["algo.gae_cut_at_truncation=true"]
ARMS = {"control": [], "no_timeout": ["task.ablation.no_timeout=true"],
        "beta": ["task.ablation.closing_discount=true"], "endings": ["task.ablation.truncate_failures=true"]}
SEED, FRAMES, STAGES = 0, 50_000_000, 13
RADII = (0.10, 0.15)
F3_ARMS = ("uniform", "hedge", "qd_dpp", "pfsp")
METRICS = ["heldout_rl", "heldout_worst_cluster", "other_f3_evaders", "other_f9_evaders", "own_campaign_evaders",
           "train", "heldout_scripted", "heldout_scripted_pursuer_crash", "x3_exploiters",
           "exploitability_thesis", "exploiter_thesis_pursuer_crash", "exploiter_thesis_timeout"]


def name(arm, smoke=False):
    return "thesis-F9-smoke" if smoke else f"thesis-F9-{arm}-s{SEED}"


def arm_cfg():
    return dict(SAMPLERS["hedge"], seed=SEED, frames_per_stage=FRAMES, n_stages=STAGES, max_total_stages=STAGES)


def run_f9(args):
    smoke = name("", True)
    run_campaign(smoke, COMMON + [o for ov in ARMS.values() for o in ov], dict(SAMPLERS["hedge"]), smoke_stages=2)
    d = _done(smoke)
    while not os.path.isfile(d) and os.path.isfile(d + ".lock"):     # the other lane may be running it
        time.sleep(60)
    if not os.path.isfile(d):
        raise RuntimeError("F9 smoke did not pass; campaigns not started")
    if args.smoke:
        return
    for arm, ov in ARMS.items():                                       # priority order; lanes skip locked arms
        run_campaign(name(arm), COMMON + ov, arm_cfg())


def _src(c, g):
    return os.path.join(F9, "evals", f"{c}_final_{g}.json")


def run_f9c(args):
    for arm in ARMS:
        d = _done(name(arm))
        while not os.path.isfile(d):
            if not os.path.isfile(d + ".lock"):
                raise RuntimeError(f"F9c: campaign {name(arm)} is neither finished nor running")
            time.sleep(60)
    split = load_split()
    test = test_set(split)
    f3_ev = [e for s in F3_ARMS for k in range(2) for e in _policies(f3_name(s, k), "evader")]
    f9_ev = [e for a in ARMS for e in f8.all_stages(name(a), "evader")]
    for i, arm in enumerate(ARMS):
        c = name(arm)
        ck = f8.all_stages(c, "pursuer")[-1]["checkpoint_path"]
        ex_path = os.path.join(F9, "ckpt", f"exploiter_thesis_{c}.pt")
        target = {"id": f"f9/pursuer:{c}", "kind": "rl", "checkpoint_path": ck, "metadata": {}}
        train(ex_path, [target], EXPLOIT_FRAMES, role="evader", seed=840 + i)
        ex = ([{"id": f"f9/evader:exploit_thesis_{c}", "kind": "rl", "checkpoint_path": ex_path, "metadata": {}}]
              if os.path.isfile(ex_path) else [])
        for g, opps in (("train", split["train"]), ("test", test), ("f3_evaders", f3_ev), ("f9_evaders", f9_ev),
                        ("vs_exploiter_thesis", ex)):
            if opps and not os.path.isfile(_src(c, g)):
                evaluate(_src(c, g), opps, policy_ckpt=ck)
        log_progress(f"F9c {c} done")
    need = [_src(name(a), g) for a in ARMS for g in ("train", "test", "f3_evaders", "f9_evaders", "vs_exploiter_thesis")]
    missing = [x for x in need if not os.path.isfile(x)]
    if missing:                                 # the other lane is still producing them and analyses after
        log_progress(f"F9c: {len(missing)} eval files missing on this lane, analysis left to the other lane")
        return
    analyse_f9(split, test)


def _metrics(c, radius, split, test, labels):
    rate = lambda g, key="neutralization_rate": rates_at(_src(c, g), key, radius)
    J = {**rate("train"), **rate("test")}
    no_s1 = lambda d: {k: v for k, v in d.items() if not k.endswith(":s1")}
    J3, J9 = no_s1(rate("f3_evaders")), no_s1(rate("f9_evaders"))
    ids = lambda opps: [J[e["id"]] for e in opps if e["id"] in J]
    r = {"train": ids(split["train"]), "heldout_rl": ids(split["heldout_rl"]),
         "heldout_scripted": ids(split["heldout_scripted"]),
         "x3_exploiters": [J[e["id"]] for e in test if e["id"].startswith("exploit/") and e["id"] in J],
         "other_f3_evaders": list(J3.values()),
         "other_f9_evaders": [v for k, v in J9.items() if not k.startswith(c + "/")],
         "own_campaign_evaders": [v for k, v in J9.items() if k.startswith(c + "/")]}
    r = {m: float(np.mean(v)) for m, v in r.items() if v}
    by_c = {}
    for e in split["heldout_rl"] + split["heldout_scripted"]:
        if e["id"] in J and e["id"] in labels:
            by_c.setdefault(labels[e["id"]], []).append(J[e["id"]])
    if by_c:
        r["heldout_worst_cluster"] = float(min(np.mean(v) for v in by_c.values()))
    pc = rate("test", "pursuer_crash")
    if pc:
        r["heldout_scripted_pursuer_crash"] = float(np.mean([pc[e["id"]] for e in split["heldout_scripted"]
                                                              if e["id"] in pc]))
    ex = rate("vs_exploiter_thesis")
    if ex:
        r["exploitability_thesis"] = 1 - float(next(iter(ex.values())))
        r["exploiter_thesis_pursuer_crash"] = float(next(iter(rate("vs_exploiter_thesis", "pursuer_crash").values())))
        r["exploiter_thesis_timeout"] = float(next(iter(rate("vs_exploiter_thesis", "timeout").values())))
    return r


def analyse_f9(split, test):
    from f5_outcomes import stage_table
    with open(os.path.join(OUT, "X2_descriptors", "clusters.json")) as f:
        labels = json.load(f)["labels"]
    arms = list(ARMS)
    res = {f"r{rad:.2f}": {a: _metrics(name(a), rad, split, test, labels) for a in arms} for rad in RADII}
    diffs = {k: {f"{a} - control": {m: v[a][m] - v["control"][m] for m in METRICS if m in v[a] and m in v["control"]}
                 for a in arms[1:]} for k, v in res.items()}
    lures = {a: f8.lure_rates(name(a)) for a in arms}
    ref = json.load(open(os.path.join(f8.F8, "results.json")))["metrics"]
    save_json(os.path.join(F9, "results.json"), {"metrics": res, "differences": diffs, "lure_rates": lures})
    refs = [("F8 roncero s0", "roncero/0"), ("F3 hedge s0", "hedge/0")]
    lines = ["# F9_ablation_sp summary", "",
             "Final pursuers (stage 12) of four campaigns, seed 0, Hedge both roles, 13 stages x 50M, GAE truncation fix: "
             "control = thesis rules; no_timeout = timeout gives 0 to both roles; beta = pursuer capture reward x "
             "Roncero's closing-speed factor; endings = crashes / evader leaving are truncations. All evals clean "
             "under the thesis rules; exploitability_thesis = 1 - J vs a fresh exploiter trained under the thesis "
             "rules (luring pays). other_f3 / other_f9 / own evaders exclude stage-1 evaders. Reference columns: F8 "
             "(26 x 50M, all of Roncero's rules) and F3 hedge (9 x 300M, thesis rules), seed 0. 1 seed, descriptive.",
             ""]
    for k, v in res.items():
        pairs = list(diffs[k])
        lines += [f"## Capture radius {k[1:]} m", "", "| metric | " + " | ".join(arms) + " | " +
                  " | ".join(pairs) + " | " + " | ".join(r for r, _ in refs) + " |",
                  "|---" * (1 + len(arms) + len(pairs) + len(refs)) + "|"]
        for m in METRICS:
            cells = [f"{v[a][m]:.3f}" if m in v[a] else "-" for a in arms]
            cells += [f"{diffs[k][p][m]:+.3f}" if m in diffs[k][p] else "-" for p in pairs]
            rk = ref.get(k, {})
            cells += [f"{rk[lab][m]:.3f}" if lab in rk and m in rk[lab] else "-" for _, lab in refs]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## In-campaign lure rates (pursuer crash caused by each evader stage in the stage >= 2 pursuers)", "",
              "| arm | evader stage | mean | min - max | n pursuers |", "|---|---|---|---|---|"]
    for a, lr in lures.items():
        for s, d in sorted(lr.items()):
            if d:
                x = list(d.values())
                lines.append(f"| {a} | {s} | {np.mean(x):.2f} | {min(x):.2f} - {max(x):.2f} | {len(x)} |")
    lines += ["", "Pre-registered measure (a): in-campaign lure rate, mean over evader stages 3-11 of each stage's mean over "
              "the stage >= 2 pursuers (lure present if >= 0.50): " +
              ", ".join(f"{a} {np.mean([np.mean(list(d.values())) for s, d in lr.items() if 3 <= s <= 11 and d]):.2f}"
                        for a, lr in lures.items()), ""]
    lines += ["", "## Inside the campaigns (each stage's eval vs its training opponents)", ""]
    for a in arms:
        lines += stage_table(name(a), STAGES)
    with open(os.path.join(F9, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log_progress("F9c analysis done")
