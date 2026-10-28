"""X2_descriptors: behaviour descriptors of every evader, pool redundancy, clusters, and the train/held-out split.

Descriptor of evader j = [response block | trajectory block], each block z-scored and weighted equally:
  response   = neutralisation rate and censored time-to-neutralise (/30 s) against every X1 pursuer;
  trajectory = mean and std of the 9 per-step behaviour features, averaged over the X1 pursuers.
"""

import json
import os

import numpy as np

import analysis as an
from common import OUT, MAX_EPISODE_STEPS, STEP_DT, save_json, log_progress

EXP = os.path.join(OUT, "X2_descriptors")
X1 = os.path.join(OUT, "X1_crossplay")
HELD_OUT_CAMPAIGN = "v10"
TRAIN_HEUR = ["hover", "circular", "flee", "weave"]


def load_x1():
    with open(os.path.join(X1, "roster.json")) as f:
        roster = json.load(f)
    data = np.load(os.path.join(X1, "payoff.npz"), allow_pickle=True)
    pursuer_ids = list(data["pursuers"])
    results = {}
    from x1_crossplay import fname
    for pid in pursuer_ids:
        with open(os.path.join(X1, "evals", fname(pid) + ".json")) as f:
            results[pid] = json.load(f)["matchups"]
    return roster, pursuer_ids, results


def descriptors(evaders, pursuer_ids, results):
    max_t = MAX_EPISODE_STEPS * STEP_DT
    resp, traj = [], []
    for e in evaders:
        r, tm, ts = [], [], []
        for pid in pursuer_ids:
            rec = results[pid][e["id"]]
            r += [rec["neutralization_rate"], rec["time_to_neutralize_censored"] / max_t]
            tm.append(rec["behaviour"]["mean"])
            ts.append(rec["behaviour"]["std"])
        resp.append(r)
        traj.append(np.concatenate([np.mean(tm, 0), np.mean(ts, 0)]))
    resp, traj = np.array(resp), np.array(traj)
    return resp, traj, an.combine_blocks([resp, traj])


def campaign_of(e):
    return (e.get("metadata") or {}).get("campaign", "scripted")


def run(args):
    roster, pursuer_ids, results = load_x1()
    evaders = roster["evaders"]
    resp, traj, X = descriptors(evaders, pursuer_ids, results)
    D = an.pairwise_dist(X)
    S, sigma = an.rbf_kernel(D)
    ids = [e["id"] for e in evaders]
    camp = np.array([campaign_of(e) for e in evaders])

    groups = {"all": np.arange(len(evaders)), "rl": np.where(camp != "scripted")[0],
              **{c: np.where(camp == c)[0] for c in sorted(set(camp))}}
    vendi = {g: {"n": int(len(ix)), "vendi": an.vendi_score(S[np.ix_(ix, ix)])} for g, ix in groups.items()}

    # Redundancy: distance between consecutive stages of a campaign vs. between campaigns (RL only).
    consec, within, across = [], [], []
    rl = groups["rl"]
    for a in rl:
        for b in rl:
            if a >= b:
                continue
            (within if camp[a] == camp[b] else across).append(D[a, b])
    for c in set(camp) - {"scripted"}:
        ix = sorted(np.where(camp == c)[0], key=lambda i: evaders[i]["metadata"]["stage"])
        consec += [D[ix[k], ix[k + 1]] for k in range(len(ix) - 1)]
    iu = np.triu_indices(len(D), 1)
    redundancy = {"sigma": sigma, "median_all": float(np.median(D[iu])),
                  "mean_consecutive_stage": float(np.mean(consec)), "mean_within_campaign": float(np.mean(within)),
                  "mean_across_campaign": float(np.mean(across)),
                  "nearest_neighbour_mean": float(np.mean(np.sort(D, 1)[:, 1]))}

    sil = {}
    best = None
    for k in range(2, min(9, len(evaders))):
        labels, med, _ = an.kmedoids(D, k)
        s = an.silhouette(D, labels)
        sil[k] = s
        if best is None or s > best[0]:
            best = (s, k, labels, med)
    _, k_best, labels, med = best
    clusters = {"k": int(k_best), "silhouette": sil,
                "labels": {ids[i]: int(labels[i]) for i in range(len(ids))},
                "medoids": [ids[m] for m in med],
                "members": {int(c): [ids[i] for i in np.where(labels == c)[0]] for c in range(k_best)}}

    split = make_split(evaders)
    save_json(os.path.join(EXP, "descriptors.json"), {
        "ids": ids, "descriptor": X.tolist(), "response": resp.tolist(), "trajectory": traj.tolist(),
        "probe_pursuers": pursuer_ids, "sigma": sigma})
    save_json(os.path.join(EXP, "diversity.json"), {"vendi": vendi, "redundancy": redundancy})
    save_json(os.path.join(EXP, "clusters.json"), clusters)
    save_json(os.path.join(EXP, "split.json"), split)
    _plot(X, D, camp, labels, ids)
    _summary(vendi, redundancy, clusters, split)
    log_progress(f"X2 k={k_best} vendi_all={vendi['all']['vendi']:.2f} vendi_rl={vendi['rl']['vendi']:.2f}")


def make_split(evaders):
    """Pre-registered split (docs/research_plan.md §5)."""
    rl = [e for e in evaders if e["kind"] == "rl"]
    camps = {campaign_of(e) for e in rl}
    if HELD_OUT_CAMPAIGN in camps:
        held_rl = [e for e in rl if campaign_of(e) == HELD_OUT_CAMPAIGN]
        train_rl = [e for e in rl if campaign_of(e) != HELD_OUT_CAMPAIGN]
        rule = f"held-out campaign {HELD_OUT_CAMPAIGN}"
    else:
        v9 = sorted([e for e in rl if campaign_of(e) == "v9"], key=lambda e: e["metadata"]["stage"])
        held_rl = v9[-5:]
        train_rl = [e for e in rl if e not in held_rl]
        rule = "fallback: last 5 stages of v9 held out"
    heur = [e for e in evaders if e["kind"] == "heuristic"]
    train_h = [e for e in heur if e["metadata"]["heuristic"] in TRAIN_HEUR]
    held_h = [e for e in heur if e["metadata"]["heuristic"] not in TRAIN_HEUR]
    return {"rule": rule, "train": train_rl + train_h, "heldout_rl": held_rl, "heldout_scripted": held_h}


def _plot(X, D, camp, labels, ids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Y = an.classical_mds(D, 2)
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = "osD^v<>P*X"
    cols = {c: f"C{i}" for i, c in enumerate(sorted(set(camp)))}
    for i in range(len(ids)):
        ax.scatter(Y[i, 0], Y[i, 1], c=cols[camp[i]], marker=markers[labels[i] % len(markers)], s=60)
        ax.annotate(ids[i].split("/")[-1], Y[i], fontsize=6, xytext=(3, 3), textcoords="offset points")
    for c, col in cols.items():
        ax.scatter([], [], c=col, label=c)
    ax.legend(title="campaign (colour); marker = cluster")
    ax.set_title("X2: evader behaviour descriptors (classical MDS)")
    fig.tight_layout()
    fig.savefig(os.path.join(EXP, "fig_mds.png"), dpi=150)
    plt.close(fig)


def _summary(vendi, red, clusters, split):
    lines = ["# X2_descriptors summary", "", "## Diversity (Vendi score = effective number of distinct evaders)", "",
             "| group | n | Vendi |", "|---|---|---|"]
    for g, v in vendi.items():
        lines.append(f"| {g} | {v['n']} | {v['vendi']:.2f} |")
    lines += ["", "## Redundancy (descriptor distances; RBF sigma = median distance)", ""]
    lines += [f"- {k}: {v:.3f}" for k, v in red.items()]
    lines += ["", f"## Clusters: k-medoids, best k = {clusters['k']} by silhouette", "",
              "| k | silhouette |", "|---|---|"]
    lines += [f"| {k} | {s:.3f} |" for k, s in clusters["silhouette"].items()]
    lines += ["", "| cluster | medoid | members |", "|---|---|---|"]
    for c, mem in clusters["members"].items():
        lines.append(f"| {c} | {clusters['medoids'][c]} | {', '.join(mem)} |")
    lines += ["", f"## Split ({split['rule']})", "",
              f"- train ({len(split['train'])}): {', '.join(e['id'] for e in split['train'])}",
              f"- held-out RL ({len(split['heldout_rl'])}): {', '.join(e['id'] for e in split['heldout_rl'])}",
              f"- held-out scripted ({len(split['heldout_scripted'])}): "
              f"{', '.join(e['id'] for e in split['heldout_scripted'])}", "", "Figure: fig_mds.png"]
    with open(os.path.join(EXP, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
