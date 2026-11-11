"""F8d: a stronger held-out test and a learning curve for the F8 pursuers (docs/research_plan.md, F8d_heldout;
pre-registered 2026-11-11 before any F8d eval; user request: v10 no longer separates the strong pursuers). Eval only.

Pursuers: F8 (thesis-F8b-roncero-s{0,100}) stages 8, 16, 24 and the F3 hedge-both s0 / s1 final pursuers (stage 8).
Opponents, all clean under the thesis rules: (1) the test set (v10 held-out RL, held-out scripted, X3 exploiters;
existing F8c / F3c evals reused for F8 stage 24 and F3 hedge); (2) the final evader of 20 self-play campaigns: a
common set of 16 that none of these pursuers trained with (F2-plain, F3 uniform / qd_dpp / pfsp, F6, F6b, F7, F9) plus
the F3 hedge and F8 finals (cross terms, own campaign excluded). Every J at 0.10 m and 0.15 m.
"""

import json
import os

import numpy as np

from a_adapt import test_set
from common import OUT, evaluate, log_progress, save_json
from f3_sampler_sp import name as f3_name
import f8_roncero_sp as f8
from rescore import rates_at
from x3_gap import load_split

F8D = os.path.join(OUT, "F8d_heldout")
COMMON = ["thesis-F2-plain", "thesis-F3-uniform-s1", "thesis-F3-qd_dpp-both-s0", "thesis-F3-qd_dpp-both-s1",
          "thesis-F3-pfsp-both-s0", "thesis-F3-pfsp-both-s1", "thesis-F6-hedge_div-s0", "thesis-F6-hedge_div-s1",
          "thesis-F6b-hedge_divb-s0", "thesis-F6b-hedge_divb-s1", "thesis-F7-r15-s0", "thesis-F7-floor_r15-s0",
          "thesis-F9-control-s0", "thesis-F9-no_timeout-s0", "thesis-F9-beta-s0", "thesis-F9-endings-s0"]
HEDGE = [f3_name("hedge", k) for k in range(2)]
F8C = [f8.name(k) for k in range(len(f8.SEEDS))]
STAGES = (8, 16, 24)
RADII = (0.10, 0.15)


def pursuers():
    """[(label, campaign, stage)]."""
    return ([(f"roncero/{k}/P{s}", c, s) for k, c in enumerate(F8C) for s in STAGES] +
            [(f"hedge/{k}", c, 8) for k, c in enumerate(HEDGE)])


def _ck(c, stage):
    return next(p for p in f8.all_stages(c, "pursuer") if int(p["metadata"]["stage"]) == stage)["checkpoint_path"]


def _finals():
    return [f8.all_stages(c, "evader")[-1] for c in COMMON + HEDGE + F8C]


def _src(c, s, g):
    if g == "test" and s == f8.all_stages(c, "pursuer")[-1]["metadata"]["stage"]:
        return f8._src(c, "test")          # existing F8c (stage 24) / F3c (hedge final) evals
    return os.path.join(F8D, "evals", f"{c}_P{s}_{g}.json")


def run_f8d(args):
    split = load_split()
    test = test_set(split)
    finals = _finals()
    for _, c, s in pursuers():
        for g, opps in (("test", test), ("finals", finals)):
            if not os.path.isfile(_src(c, s, g)):
                evaluate(_src(c, s, g), opps, policy_ckpt=_ck(c, s))
        log_progress(f"F8d {c} P{s} done")
    missing = [_src(c, s, g) for _, c, s in pursuers() for g in ("test", "finals") if not os.path.isfile(_src(c, s, g))]
    if missing:
        raise RuntimeError(f"F8d: evals missing {missing}")
    analyse_f8d(split, test)


def _metrics(c, s, radius, split, test, labels):
    J = rates_at(_src(c, s, "test"), "neutralization_rate", radius)
    F = rates_at(_src(c, s, "finals"), "neutralization_rate", radius)
    PC = rates_at(_src(c, s, "finals"), "pursuer_crash", radius)
    camp = lambda e: e.split("/")[0]
    common = [e for e in F if camp(e) in COMMON]
    ids = lambda opps: [J[e["id"]] for e in opps if e["id"] in J]
    r = {"heldout_rl": float(np.mean(ids(split["heldout_rl"]))),
         "x3_exploiters": float(np.mean([J[e["id"]] for e in test if e["id"].startswith("exploit/") and e["id"] in J])),
         "strong_finals_mean": float(np.mean([F[e] for e in common])),
         "strong_finals_min": float(min(F[e] for e in common)),
         "strong_finals_pursuer_crash": float(np.mean([PC[e] for e in common])),
         "vs_f3_hedge_finals": float(np.mean([F[e] for e in F if camp(e) in HEDGE and camp(e) != c])),
         "vs_f8_finals": float(np.mean([F[e] for e in F if camp(e) in F8C and camp(e) != c]))}
    by_c = {}
    for e in split["heldout_rl"] + split["heldout_scripted"]:
        if e["id"] in J and e["id"] in labels:
            by_c.setdefault(labels[e["id"]], []).append(J[e["id"]])
    r["heldout_worst_cluster"] = float(min(np.mean(v) for v in by_c.values()))
    r["per_final"] = {e: float(F[e]) for e in F}
    return r


def analyse_f8d(split, test):
    with open(os.path.join(OUT, "X2_descriptors", "clusters.json")) as f:
        labels = json.load(f)["labels"]
    res = {f"r{rad:.2f}": {lab: _metrics(c, s, rad, split, test, labels) for lab, c, s in pursuers()} for rad in RADII}
    save_json(os.path.join(F8D, "results.json"), res)
    labs = [lab for lab, _, _ in pursuers()]
    rows = ["heldout_rl", "heldout_worst_cluster", "x3_exploiters", "strong_finals_mean", "strong_finals_min",
            "strong_finals_pursuer_crash", "vs_f3_hedge_finals", "vs_f8_finals"]
    lines = ["# F8d_heldout summary", "",
             "J (pursuer neutralisation rate) of F8 pursuer stages 8 / 16 / 24 (13 x 50M schedule: 250M / 450M / 650M "
             "pursuer frames) and the F3 hedge final pursuers (stage 8, 1.5B pursuer frames), clean under the thesis "
             "rules. strong_finals = the final evader of 16 campaigns none of them trained with (" + ", ".join(COMMON) +
             "); vs_f3_hedge_finals / vs_f8_finals exclude the pursuer's own campaign. 2 seeds, descriptive.", ""]
    for k, v in res.items():
        lines += [f"## Capture radius {k[1:]} m", "", "| metric | " + " | ".join(labs) + " |", "|---" * (len(labs) + 1) + "|"]
        lines += [f"| {m} | " + " | ".join(f"{v[a][m]:.3f}" for a in labs) + " |" for m in rows]
        lines += ["", "Per strong final evader (rows) x pursuer (columns):", "",
                  "| final evader | " + " | ".join(labs) + " |", "|---" * (len(labs) + 1) + "|"]
        evs = [e for e in v[labs[0]]["per_final"] if e.split("/")[0] in COMMON]
        lines += [f"| {e} | " + " | ".join(f"{v[a]['per_final'][e]:.2f}" for a in labs) + " |" for e in evs]
        lines.append("")
    with open(os.path.join(F8D, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log_progress("F8d analysis done")
