"""Post-hoc (2026-11-09, user request; not pre-registered, no new runs): transitivity of the pursuit-evasion game
in the F3 self-play campaigns, per sampler, with X1's measures (analysis.additive_fit: R^2 of logit J ~ a_i - b_j;
analysis.rank_reversals, delta 0.1).

1. Within each campaign: its RL pursuer stages 2, 4, 6, 8 x RL evader stages 1, 3, 5, 7 (pursuer stage 0 is left out:
   evader stage 1 trains vs scripted pursuers only, so P0 vs E1 was never evaluated). Every pair is in
   the in-campaign stage evals (each stage is evaluated against all earlier opponents): P_i vs E_j from pursuer i's
   eval if i > j, else from evader j's eval. Training radius 0.10 m only (no per-episode files).
2. Across campaigns: the 8 F3 final pursuers x the RL evaders of each sampler's 2 campaigns (stage-1 evaders excluded,
   they are identical across same-seed campaigns), from the F3c campaign-evader evals, at 0.10 m and 0.15 m (rescore).
Usage: env -u PYTHONPATH .venv/bin/python scripts/experiments/thesis/exps/f3_transitivity.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analysis as an  # noqa: E402
import common  # noqa: E402
from f3_sampler_sp import F3C, SAMPLERS  # noqa: E402
from f3_sampler_sp import name as f3_name  # noqa: E402
from rescore import rates_at  # noqa: E402

P_STAGES, E_STAGES = (2, 4, 6, 8), (1, 3, 5, 7)


def _stage_of(c, role):
    pool = json.load(open(os.path.join(common.SELFPLAY_OUT, c, "db", f"{role}_pool.json")))["policies"]
    return {p["id"]: int(os.path.basename(p["checkpoint_path"]).split("stage")[1].split(".")[0])
            for p in pool if p.get("checkpoint_path")}


def within_matrix(c, key="neutralization_rate"):
    ps, es = _stage_of(c, "pursuer"), _stage_of(c, "evader")
    M = np.full((len(P_STAGES), len(E_STAGES)), np.nan)
    for s in range(9):
        role = "pursuer" if s % 2 == 0 else "evader"
        m = json.load(open(os.path.join(common.SELFPLAY_OUT, c, "stages", f"stage{s}_{role}_eval_out.json")))["matchups"]
        if role == "pursuer" and s not in P_STAGES:
            continue
        opp = es if role == "pursuer" else ps
        for k, v in m.items():
            if k not in opp or (role == "evader" and opp[k] not in P_STAGES):
                continue                                  # scripted opponents / pursuer stage 0
            val = v[key] if key in v else v["outcomes"][key]
            i, j = (P_STAGES.index(s), E_STAGES.index(opp[k])) if role == "pursuer" else \
                   (P_STAGES.index(opp[k]), E_STAGES.index(s))
            M[i, j] = val
    return M


def stats(M):
    return {"r2": an.additive_fit(M)["r2"], "reversal_fraction": an.rank_reversals(M, 0.1)["reversal_fraction"],
            "shape": list(M.shape)}


def main():
    camps = {s: [f3_name(s, k) for k in range(2)] for s in SAMPLERS}
    out = {"within": {}, "across": {}}
    for s, cs in camps.items():
        out["within"][s] = {}
        for c in cs:
            M, C = within_matrix(c), within_matrix(c, "capture_rate")
            assert not np.isnan(M).any(), (c, M)
            out["within"][s][c] = {"J": stats(M), "capture": stats(C), "matrix_J": M.round(3).tolist()}
    finals = [c for cs in camps.values() for c in cs]
    for rad in (0.10, 0.15):
        rows = {c: rates_at(os.path.join(F3C, "evals", f"{c}_final_campaign_evaders.json"), "neutralization_rate", rad)
                for c in finals}
        ev = sorted(set.intersection(*(set(r) for r in rows.values())))
        ev = [e for e in ev if not e.endswith(":s1") and e.split("/")[0] in finals]
        res = {}
        for s, cs in list(camps.items()) + [("all", finals)]:
            cols = [e for e in ev if e.split("/")[0] in cs]
            M = np.array([[rows[c][e] for e in cols] for c in finals])
            res[s] = stats(M)
        out["across"][f"r{rad:.2f}"] = res
    common.save_json(os.path.join(F3C, "transitivity.json"), out)
    lines = ["# F3 transitivity per sampler (post-hoc, 2026-11-09)", "",
             "X1's measures: additive-logit R^2 (one strength per player) and rank-reversal fraction (delta 0.1).", "",
             "## Within each campaign: pursuer stages 2-8 x RL evader stages 1-7 (in-campaign evals, 0.10 m)", "",
             "| sampler | campaign | R^2 (J) | reversals (J) | R^2 (capture) | reversals (capture) |", "|---|---|---|---|---|---|"]
    for s, d in out["within"].items():
        for c, r in d.items():
            lines.append(f"| {s} | {c} | {r['J']['r2']:.3f} | {r['J']['reversal_fraction']:.3f} | "
                         f"{r['capture']['r2']:.3f} | {r['capture']['reversal_fraction']:.3f} |")
    lines += ["", "## Across campaigns: 8 F3 final pursuers x each sampler's RL evaders (stage 1 excluded; F3c evals)", "",
              "| evader set | n evaders | R^2 0.10 m | reversals 0.10 m | R^2 0.15 m | reversals 0.15 m |",
              "|---|---|---|---|---|---|"]
    for s in list(camps) + ["all"]:
        a, b = out["across"]["r0.10"][s], out["across"]["r0.15"][s]
        lines.append(f"| {s} | {a['shape'][1]} | {a['r2']:.3f} | {a['reversal_fraction']:.3f} | {b['r2']:.3f} | "
                     f"{b['reversal_fraction']:.3f} |")
    lines += ["", "## Within-campaign J matrices (rows pursuer stages 2-8, columns evader stages 1-7)", ""]
    for s, d in out["within"].items():
        for c, r in d.items():
            lines += [f"**{c}**", "", "| P \\ E | E1 | E3 | E5 | E7 |", "|---|---|---|---|---|"]
            lines += [f"| P{p} | " + " | ".join(f"{x:.2f}" for x in row) + " |"
                      for p, row in zip(P_STAGES, r["matrix_J"])] + [""]
    with open(os.path.join(F3C, "transitivity.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
