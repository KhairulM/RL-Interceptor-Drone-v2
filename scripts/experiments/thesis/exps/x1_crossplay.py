"""X1_crossplay: payoff matrix of many pursuers vs all valid evaders; transitivity analysis.

Pursuers: last 4 RL stages of every valid campaign + scripted pure_pursuit and apf.
Evaders: every RL evader of every valid campaign + 10 scripted evaders (6 base + 4 switch forms at @0.5).
Each pursuer is evaluated once against all evaders (envs split uniformly; multi-episode estimator).
"""

import json
import os

import numpy as np

import analysis as an
from common import CAMPAIGNS, OUT, evaluate, heuristic, log_progress, rl_policies, save_json

EXP = os.path.join(OUT, "X1_crossplay")
SCRIPTED_EVADERS = ["hover", "circular", "flee", "weave", "zigzag", "fast_flee",
                    "flee>circular@0.5", "circular>hover@0.5", "hover>flee@0.5", "flee>weave@0.5"]
SCRIPTED_PURSUERS = ["pure_pursuit", "apf"]
N_LAST_PURSUERS = 4


def valid_campaigns():
    path = os.path.join(OUT, "X0_replay", "replay_check.json")
    if not os.path.isfile(path):
        raise RuntimeError("run X0 first")
    with open(path) as f:
        rows = json.load(f)
    ok = {}
    for r in rows:
        ok.setdefault(r["campaign"], []).append(bool(r.get("ok")))
    return [t for t in CAMPAIGNS if t in ok and sum(ok[t]) >= 2]


def agents_and_evaders(smoke=False):
    tags = valid_campaigns()
    pursuers = []
    for t in tags:
        pursuers += rl_policies(t, "pursuer")[-N_LAST_PURSUERS:]
    pursuers += [heuristic(h, "pursuer") for h in SCRIPTED_PURSUERS]
    evaders = []
    for t in tags:
        evaders += rl_policies(t, "evader")
    evaders += [heuristic(h, "evader") for h in SCRIPTED_EVADERS]
    if smoke:
        pursuers, evaders = pursuers[:2], evaders[:4]
    return tags, pursuers, evaders


def fname(pid):
    return pid.replace("/", "_").replace(":", "_").replace(">", "-").replace("@", "at")


def run(args):
    tags, pursuers, evaders = agents_and_evaders(args.smoke)
    log_progress(f"X1 campaigns={tags} pursuers={len(pursuers)} evaders={len(evaders)}")
    save_json(os.path.join(EXP, "roster.json"), {"campaigns": tags, "pursuers": pursuers, "evaders": evaders})
    results = {}
    for p in pursuers:
        out = os.path.join(EXP, "evals", fname(p["id"]) + ".json")
        if p["kind"] == "rl":
            res = evaluate(out, evaders, policy_ckpt=p["checkpoint_path"])
        else:
            res = evaluate(out, evaders, heuristic_agent=p["metadata"]["heuristic"])
        if res is not None:
            results[p["id"]] = res["matchups"]
        log_progress(f"X1 done {p['id']} ({len(results)}/{len(pursuers)})")
    analyse(pursuers, evaders, results)


def matrices(pursuers, evaders, results):
    keys = ["neutralization_rate", "capture_rate", "time_to_neutralize_censored", "pursuer_crash_rate"]
    M = {k: np.full((len(pursuers), len(evaders)), np.nan) for k in keys}
    E = np.zeros((len(pursuers), len(evaders)), dtype=int)
    for i, p in enumerate(pursuers):
        for j, e in enumerate(evaders):
            rec = results.get(p["id"], {}).get(e["id"])
            if rec is None:
                continue
            for k in keys:
                M[k][i, j] = rec[k]
            E[i, j] = rec["episodes"]
    return M, E


def analyse(pursuers, evaders, results):
    pursuers = [p for p in pursuers if p["id"] in results]
    M, E = matrices(pursuers, evaders, results)
    np.savez(os.path.join(EXP, "payoff.npz"), pursuers=[p["id"] for p in pursuers],
             evaders=[e["id"] for e in evaders], episodes=E, **M)
    P = M["neutralization_rate"]
    rl_rows = [i for i, p in enumerate(pursuers) if p["kind"] == "rl"]
    trans = {
        "all": {**an.additive_fit(P), **an.rank_reversals(P, 0.1)},
        "rl_pursuers_only": {**an.additive_fit(P[rl_rows]), **an.rank_reversals(P[rl_rows], 0.1)},
        "capture_all": {**an.additive_fit(M["capture_rate"]), **an.rank_reversals(M["capture_rate"], 0.1)},
    }
    save_json(os.path.join(EXP, "transitivity.json"), trans)
    _plot(pursuers, evaders, P, "neutralisation rate", os.path.join(EXP, "fig_payoff_neut.png"))
    _plot(pursuers, evaders, M["capture_rate"], "capture rate", os.path.join(EXP, "fig_payoff_capture.png"))
    _summary(pursuers, evaders, M, E, trans)


def _plot(pursuers, evaders, P, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(0.32 * len(evaders) + 3, 0.32 * len(pursuers) + 2))
    im = ax.imshow(P, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(evaders)), [e["id"] for e in evaders], rotation=90, fontsize=7)
    ax.set_yticks(range(len(pursuers)), [p["id"] for p in pursuers], fontsize=7)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if not np.isnan(P[i, j]):
                ax.text(j, i, f"{P[i, j]:.2f}"[1:] if P[i, j] < 1 else "1", ha="center", va="center",
                        fontsize=5, color="w" if P[i, j] < 0.5 else "k")
    fig.colorbar(im, ax=ax, label=title)
    ax.set_title(f"X1 cross-play: pursuer {title} (rows) vs evader (columns)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _summary(pursuers, evaders, M, E, trans):
    P = M["neutralization_rate"]
    lines = ["# X1_crossplay summary", "",
             f"{len(pursuers)} pursuers x {len(evaders)} evaders; episodes per cell: "
             f"min {E.min()}, median {int(np.median(E))}.", "",
             "## Transitivity", "",
             "| subset | additive-logit R² | rank-reversal fraction (delta 0.1) |", "|---|---|---|"]
    for k, v in trans.items():
        lines.append(f"| {k} | {v['r2']:.3f} | {v['reversal_fraction']:.4f} |")
    lines += ["", "## Pursuers (mean / min neutralisation over evaders)", "",
              "| pursuer | mean neut | min neut | hardest evader | mean capture | mean pursuer crash |",
              "|---|---|---|---|---|---|"]
    for i, p in enumerate(pursuers):
        j = int(np.nanargmin(P[i]))
        lines.append(f"| {p['id']} | {np.nanmean(P[i]):.3f} | {P[i, j]:.3f} | {evaders[j]['id']} | "
                     f"{np.nanmean(M['capture_rate'][i]):.3f} | {np.nanmean(M['pursuer_crash_rate'][i]):.3f} |")
    lines += ["", "## Evaders (mean neutralisation over pursuers; lower = harder)", "",
              "| evader | mean neut | max neut | min neut |", "|---|---|---|---|"]
    order = np.argsort(np.nanmean(P, 0))
    for j in order:
        lines.append(f"| {evaders[j]['id']} | {np.nanmean(P[:, j]):.3f} | {np.nanmax(P[:, j]):.3f} | "
                     f"{np.nanmin(P[:, j]):.3f} |")
    lines += ["", "Figures: fig_payoff_neut.png, fig_payoff_capture.png. Raw: payoff.npz, evals/*.json."]
    with open(os.path.join(EXP, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
