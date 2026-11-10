"""Post-hoc (2026-11-10, user request; not pre-registered, no new runs): is the F8 game (Roncero's rules) transitive,
and how diverse are its evaders, compared with the F3 Hedge campaigns (thesis rules)? Question behind it: would
diversity-aware training (QD-DPP / PSD-PSRO style) have room to help under Roncero's rules?

1. Within each F8 campaign (in-campaign stage evals, Roncero's rules, 0.10 m): pursuer stages 2-24 x evader stages
   1-25, and the F3-matched block P2-8 x E1-7. Same measures as f3_transitivity (additive-logit R^2, rank-reversal
   fraction, delta 0.1), plus how often a new stage beats every older opponent.
2. Across campaigns (F8c / F3c evals, thesis rules, 0.10 m and 0.15 m): 4 probe pursuers (F3 hedge s0 / s1 final and
   F8 s0 / s1 stage 8: F8c evaluated stage 8, not the final stage 24, see f8_roncero_sp.all_stages) x the RL evaders of the 8 F3 campaigns and the 2 F8 campaigns (stage 1 excluded). F8c evaluated
   only F8 evader stages 1-7 (f3_sampler_sp._policies caps stages at 8), so stages 9-25 are not in this part.
3. Evader Vendi:
   a. Shared space: F3's descriptor construction (J and censored time vs each probe, mean / std behaviour features;
      one RBF kernel, median bandwidth) with the 4 probes above, over all 32 F3 + 8 F8 RL evaders (stages 1-7).
      Not F3's frozen 8-probe space (the other 6 F3 pursuers never played the F8 evaders), so values are compared
      within this space only.
   b. Within-campaign space: each evader described by J, censored time and behaviour vs the campaign's own pursuer
      stages (in-campaign evals). F8 all 13 evaders vs P2-24; F8 and F3 hedge evaders 1-7 vs P2-8 (matched).
Usage: env -u PYTHONPATH .venv/bin/python scripts/experiments/thesis/exps/f8_transitivity.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analysis as an  # noqa: E402
import common  # noqa: E402
from common import MAX_EPISODE_STEPS, STEP_DT  # noqa: E402
from f3_sampler_sp import F3C, SAMPLERS  # noqa: E402
from f3_sampler_sp import name as f3_name  # noqa: E402
import f8_roncero_sp as f8  # noqa: E402
from rescore import rates_at  # noqa: E402

F8_CAMPS = [f8.name(k) for k in range(len(f8.SEEDS))]
F3_HEDGE = [f3_name("hedge", k) for k in range(2)]
F3_CAMPS = [f3_name(s, k) for s in SAMPLERS for k in range(2)]
MAX_T = MAX_EPISODE_STEPS * STEP_DT


def _stage_of(c, role):
    pool = json.load(open(os.path.join(common.SELFPLAY_OUT, c, "db", f"{role}_pool.json")))["policies"]
    return {p["id"]: int(os.path.basename(p["checkpoint_path"]).split("stage")[1].split(".")[0])
            for p in pool if p.get("checkpoint_path")}


def in_campaign_records(c, n_stages):
    """{(pursuer stage, evader stage): matchup record} from the in-campaign stage evals. P_i vs E_j comes from
    pursuer i's eval if i > j, else from evader j's eval (each stage is evaluated against all earlier opponents)."""
    ps, es = _stage_of(c, "pursuer"), _stage_of(c, "evader")
    out = {}
    for s in range(n_stages):
        role = "pursuer" if s % 2 == 0 else "evader"
        m = json.load(open(os.path.join(common.SELFPLAY_OUT, c, "stages", f"stage{s}_{role}_eval_out.json")))["matchups"]
        opp = es if role == "pursuer" else ps
        for k, v in m.items():
            if k in opp:                                  # RL opponents only
                out[(s, opp[k]) if role == "pursuer" else (opp[k], s)] = v
    return out


def _val(rec, key):
    return rec[key] if key in rec else rec["outcomes"][key]


def matrix(recs, p_st, e_st, key="neutralization_rate"):
    M = np.array([[_val(recs[(p, e)], key) if (p, e) in recs else np.nan for e in e_st] for p in p_st])
    assert not np.isnan(M).any(), "missing in-campaign matchups"
    return M


def stats(M):
    return {"r2": an.additive_fit(M)["r2"], "reversal_fraction": an.rank_reversals(M, 0.1)["reversal_fraction"],
            "shape": list(M.shape)}


def staircase(recs, p_st, e_st):
    """Fraction of stages that beat every older opponent: pursuer J >= 0.5 vs all older evaders, evader J < 0.5 vs all
    older pursuers; and the newest pursuer's worst J vs any older evader."""
    pb = [all(_val(recs[(p, e)], "neutralization_rate") >= 0.5 for e in e_st if e < p) for p in p_st]
    eb = [all(_val(recs[(p, e)], "neutralization_rate") < 0.5 for p in p_st if p < e) for e in e_st if e > min(p_st)]
    last = max(p_st)
    return {"pursuers_beat_all_older": float(np.mean(pb)), "evaders_beat_all_older": float(np.mean(eb)),
            "newest_pursuer_min_vs_older": float(min(_val(recs[(last, e)], "neutralization_rate")
                                                     for e in e_st if e < last))}


def _descr(rows, evs, probes):
    """F3 _evader_vendi descriptor blocks: [J, censored time / T] per probe; behaviour mean / std averaged over probes."""
    resp = np.array([[v for p in probes for v in (rows[p][e]["neutralization_rate"],
                                                  rows[p][e]["time_to_neutralize_censored"] / MAX_T)] for e in evs])
    traj = np.array([np.concatenate([np.mean([rows[p][e]["behaviour"]["mean"] for p in probes], 0),
                                     np.mean([rows[p][e]["behaviour"]["std"] for p in probes], 0)]) for e in evs])
    return an.combine_blocks([resp, traj])


def vendi_of(X, groups):
    S, _ = an.rbf_kernel(an.pairwise_dist(X))
    return {g: {"n": len(ix), "vendi": an.vendi_score(S[np.ix_(ix, ix)])} for g, ix in groups.items() if ix}


def _load(path):
    return json.load(open(path))["matchups"]


def across(out):
    """Part 2 and 3a: probe pursuers x F3 + F8 RL evaders, thesis-rules evals."""
    files = {}
    for c in F3_HEDGE:
        files[c] = [os.path.join(F3C, "evals", f"{c}_final_campaign_evaders.json"),
                    os.path.join(f8.F8, "evals", f"{c}_final_f8_evaders.json")]
    for c in F8_CAMPS:
        files[c] = [os.path.join(f8.F8, "evals", f"{c}_final_{g}.json") for g in ("f3_evaders", "f8_evaders")]
    probes = list(files)
    rows = {p: {k: v for f in fs for k, v in _load(f).items()} for p, fs in files.items()}
    evs = sorted(set.intersection(*(set(r) for r in rows.values())))
    evs = [e for e in evs if e.split("/")[0] in F3_CAMPS + F8_CAMPS]
    groups = {"F8 evaders": [e for e in evs if e.split("/")[0] in F8_CAMPS],
              "F3 all samplers": [e for e in evs if e.split("/")[0] in F3_CAMPS]}
    out["across"] = {}
    for rad in (0.10, 0.15):
        J = {p: {} for p in probes}
        for p, fs in files.items():
            for f in fs:
                J[p].update(rates_at(f, "neutralization_rate", rad))
        res = {}
        for g, cols in list(groups.items()) + [("all", evs)]:
            cols = [e for e in cols if not e.endswith(":s1")]
            res[g] = stats(np.array([[J[p][e] for e in cols] for p in probes]))
        out["across"][f"r{rad:.2f}"] = res
    X = _descr(rows, evs, probes)
    vg = {c: [i for i, e in enumerate(evs) if e.startswith(c + "/")] for c in F3_CAMPS + F8_CAMPS}
    out["vendi_shared"] = vendi_of(X, vg)
    f3v = json.load(open(os.path.join(F3C, "results.json")))["evader_vendi"]
    out["vendi_shared_f3_official"] = {c: f3v[c]["vendi"] for c in F3_CAMPS if c in f3v}
    out["across_probes"], out["across_n_evaders"] = probes, len(evs)


def within(out):
    out["within"], out["vendi_within"] = {}, {}
    sets = [(c, 26, list(range(2, 26, 2)), list(range(1, 26, 2))) for c in F8_CAMPS] + \
           [(c, 9, [2, 4, 6, 8], [1, 3, 5, 7]) for c in F3_HEDGE]
    for c, n, p_full, e_full in sets:
        recs = in_campaign_records(c, n)
        r = {}
        for lab, ps, es in (("full", p_full, e_full), ("matched P2-8 x E1-7", [2, 4, 6, 8], [1, 3, 5, 7])):
            if lab == "full" and n == 9:
                continue
            M = matrix(recs, ps, es)
            r[lab] = {"J": stats(M), "capture": stats(matrix(recs, ps, es, "capture_rate")),
                      **staircase(recs, ps, es), "matrix_J": M.round(3).tolist(), "p": ps, "e": es}
        out["within"][c] = r
        v = {}
        for lab, ps, es in (("all evaders vs P2-24", p_full, e_full), ("E1-7 vs P2-8", [2, 4, 6, 8], [1, 3, 5, 7])):
            if lab.startswith("all") and n == 9:
                continue
            rows = {p: {e: recs[(p, e)] for e in es} for p in ps}
            X = _descr(rows, es, ps)
            groups = {lab: list(range(len(es)))}
            if len(es) > 4:
                groups.update({"first 4 (E1-7)": list(range(4)), "last 4 (E19-25)": list(range(len(es) - 4, len(es)))})
            v[lab] = vendi_of(X, groups)
        out["vendi_within"][c] = v


def report(out):
    L = ["# F8 transitivity and evader diversity (post-hoc, 2026-11-10; no new runs)", "",
         "X1's measures: additive-logit R^2 (one strength per player) and rank-reversal fraction (delta 0.1).", "",
         "## 1. Within each campaign (in-campaign evals, training rules, 0.10 m)", "",
         "| campaign | block | R^2 (J) | reversals (J) | R^2 (capture) | reversals (capture) | pursuers beat all older | "
         "evaders beat all older | newest pursuer min J vs older |", "|---|---|---|---|---|---|---|---|---|"]
    for c, r in out["within"].items():
        for lab, d in r.items():
            L.append(f"| {c} | {lab} {d['J']['shape'][0]}x{d['J']['shape'][1]} | {d['J']['r2']:.3f} | "
                     f"{d['J']['reversal_fraction']:.3f} | {d['capture']['r2']:.3f} | "
                     f"{d['capture']['reversal_fraction']:.3f} | {d['pursuers_beat_all_older']:.2f} | "
                     f"{d['evaders_beat_all_older']:.2f} | {d['newest_pursuer_min_vs_older']:.2f} |")
    L += ["", "## 2. Across campaigns (F3c / F8c evals, thesis rules; probes: " + ", ".join(out["across_probes"]) +
          " (F3 final pursuers; F8 pursuer stage 8, which F8c evaluated instead of stage 24); stage-1 evaders "
          "excluded; F8 evaders are stages 3-7 only)", "",
          "| evader set | n | R^2 0.10 m | reversals 0.10 m | R^2 0.15 m | reversals 0.15 m |", "|---|---|---|---|---|---|"]
    for g in out["across"]["r0.10"]:
        a, b = out["across"]["r0.10"][g], out["across"]["r0.15"][g]
        L.append(f"| {g} | {a['shape'][1]} | {a['r2']:.3f} | {a['reversal_fraction']:.3f} | {b['r2']:.3f} | "
                 f"{b['reversal_fraction']:.3f} |")
    L += ["", "## 3a. Evader Vendi in a shared 4-probe space (stages 1-7, 4 evaders per campaign, max 4)", "",
          "| campaign | Vendi (this space) | F3 official (8 probes) |", "|---|---|---|"]
    for c, d in out["vendi_shared"].items():
        o = out["vendi_shared_f3_official"].get(c)
        L.append(f"| {c} | {d['vendi']:.3f} | {o:.3f} |" if o is not None else f"| {c} | {d['vendi']:.3f} | - |")
    L += ["", "## 3b. Evader Vendi within each campaign (descriptors vs the campaign's own pursuers)", "",
          "| campaign | descriptor space | evader set | n | Vendi |", "|---|---|---|---|---|"]
    for c, v in out["vendi_within"].items():
        for space, gs in v.items():
            for g, d in gs.items():
                L.append(f"| {c} | {space} | {g} | {d['n']} | {d['vendi']:.3f} |")
    L += ["", "## Within-campaign J matrices, full blocks (rows pursuer stages, columns evader stages)", ""]
    for c, r in out["within"].items():
        d = r.get("full") or r["matched P2-8 x E1-7"]
        L += [f"**{c}**", "", "| P \\ E | " + " | ".join(f"E{e}" for e in d["e"]) + " |", "|---" * (len(d["e"]) + 1) + "|"]
        L += [f"| P{p} | " + " | ".join(f"{x:.2f}" for x in row) + " |" for p, row in zip(d["p"], d["matrix_J"])] + [""]
    return L


def main():
    out = {}
    within(out)
    across(out)
    common.save_json(os.path.join(f8.F8, "transitivity.json"), out)
    lines = report(out)
    with open(os.path.join(f8.F8, "transitivity.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
