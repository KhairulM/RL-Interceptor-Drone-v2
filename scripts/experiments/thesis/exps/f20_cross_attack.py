"""F20: swap the F14 single-rules (thesis-rule) exploiters between F8 and RLS-26 within each seed
(docs/research_plan.md, F20_cross_attack; registered 2026-11-19 17:11 before any F20 job; eval only)."""

import json
import os

from common import OUT, evaluate, log_progress, save_json
import f5_acc_sp as f5
import f14_generalization_sp as f14

F20 = os.path.join(OUT, "F20_cross_attack")
SEEDS = f14.SEEDS
F14_EVALS = os.path.join(f14.F14, "evals")


def _camps(seed):
    return f14.F8_CAMPS[SEEDS.index(seed)], f14._stagee_campaigns("v2d")["rls"][SEEDS.index(seed)]


def _attacker(own_opp_json):
    """The exact attacker policy F14 evaluated (from its recorded opponent spec)."""
    d = json.load(open(own_opp_json))
    pols = d["policies"] if isinstance(d, dict) else d
    if len(pols) != 1 or not os.path.isfile(pols[0]["checkpoint_path"]):
        raise RuntimeError(f"F20: unexpected attacker spec {own_opp_json}")
    return pols[0]


def jobs():
    out = []
    for seed in SEEDS:
        f8c, rlsc = _camps(seed)
        a_f8 = _attacker(os.path.join(F14_EVALS, f"{f8c}_thesis_exploiter_opp.json"))
        a_rls = _attacker(os.path.join(F14_EVALS, f"{rlsc}_stageE_thesis_opp.json"))
        out.append((f"f8_s{seed}_vs_rls_attacker", f14._ck(f8c, 24), None, a_rls, seed))
        out.append((f"rls_s{seed}_vs_f8_attacker", f14._ck(rlsc, 24), f5.rls_overrides(), a_f8, seed))
    return out


def _row(path):
    m = next(iter(json.load(open(path))["matchups"].values()))
    o = m["outcomes"]
    return {"J": m["neutralization_rate"], "capture": o["capture"], "pursuer_crash": o["pursuer_crash"],
            "timeout": o["timeout"], "evader_crash": o["evader_crash"], "evader_oob": o["evader_oob"],
            "episodes": m["episodes"]}


def analyse():
    res = {}
    for seed in SEEDS:
        f8c, rlsc = _camps(seed)
        res[str(seed)] = {
            "f8_vs_own_attacker": _row(os.path.join(F14_EVALS, f"{f8c}_thesis_exploiter.json")),
            "f8_vs_rls_attacker": _row(os.path.join(F20, "evals", f"f8_s{seed}_vs_rls_attacker.json")),
            "rls_vs_own_attacker": _row(os.path.join(F14_EVALS, f"{rlsc}_stageE_thesis.json")),
            "rls_vs_f8_attacker": _row(os.path.join(F20, "evals", f"rls_s{seed}_vs_f8_attacker.json")),
        }
    save_json(os.path.join(F20, "results.json"), res)
    lines = ["# F20 cross-attack (single-rules exploiters swapped within seed; 0.10 m)", "",
             "| Seed | Matchup | J | Capture | Pursuer crash | Timeout | Evader crash | Evader OOB | Episodes |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s, d in res.items():
        for k, r in d.items():
            lines.append(f"| {s} | {k} | {r['J']:.3f} | {r['capture']:.3f} | {r['pursuer_crash']:.3f} | "
                         f"{r['timeout']:.3f} | {r['evader_crash']:.3f} | {r['evader_oob']:.3f} | {r['episodes']} |")
    with open(os.path.join(F20, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log_progress("F20 analysis done")
    return res


def run_f20(args):
    os.makedirs(os.path.join(F20, "evals"), exist_ok=True)
    for name, ck, extra, attacker, seed in jobs():
        path = os.path.join(F20, "evals", f"{name}.json")
        if not os.path.isfile(path):
            evaluate(path, [attacker], policy_ckpt=ck, extra=extra)
        log_progress(f"F20 eval {name}")
    analyse()
