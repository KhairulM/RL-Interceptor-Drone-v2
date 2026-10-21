"""Aggregate eval JSONs into figures + summary.md.  python report.py --tag smoke|p0"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../experiments/esn_adapt"))
SETS = ["T0", "T1", "T2", "T3"]


def load(tag):
    """data[method][set][seed] = {opponent: record}"""
    data, cost = {}, {}
    for d in sorted(glob.glob(f"{ROOT}/{tag}/*/seed*")):
        m, s = d.split("/")[-2:]
        for T in SETS:
            f = f"{d}/eval_{T}.json"
            if os.path.exists(f):
                data.setdefault(m, {}).setdefault(T, {})[s] = json.load(open(f))["matchups"]
        if os.path.exists(f"{d}/state.json"):
            cost.setdefault(m, []).append(json.load(open(f"{d}/state.json")).get("train_seconds", np.nan))
    return data, cost


def seed_vals(runs, key):
    return np.array([np.mean([r[key] for r in opp.values() if r.get(key) is not None]) for opp in runs.values()])


def ci(v, n=2000):
    if len(v) < 2:
        return float(v.mean()), 0.0
    b = np.random.default_rng(0).choice(v, (n, len(v))).mean(1)
    return float(v.mean()), float((np.percentile(b, 97.5) - np.percentile(b, 2.5)) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p0")
    tag = ap.parse_args().tag
    data, cost = load(tag)
    out = f"{ROOT}/report_{tag}"
    os.makedirs(out, exist_ok=True)
    methods = list(data)
    if not methods:
        raise SystemExit("no results")

    lines = [f"# ESN adaptation report ({tag})", "", "Capture rate, mean +/- 95% bootstrap CI over seeds (equal weight per evader).", "",
             "| method | " + " | ".join(SETS) + " | train s |", "|---|" + "---|" * (len(SETS) + 1)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        row = []
        for j, T in enumerate(SETS):
            if T not in data[m]:
                row.append("-")
                continue
            mu, h = ci(seed_vals(data[m][T], "win_rate"))
            axes[0].bar(j + i * w, mu, w, yerr=h, label=m if j == 0 else None, color=f"C{i}")
            row.append(f"{mu:.3f} +/- {h:.3f}")
            if T in ("T0", "T1", "T2", "T3"):
                tt = seed_vals(data[m][T], "avg_time_to_capture")
                axes[1].bar(j + i * w, np.nanmean(tt), w, color=f"C{i}")
        lines.append(f"| {m} | " + " | ".join(row) + f" | {np.nanmean(cost.get(m, [np.nan])):.0f} |")
    for ax, t in zip(axes, ["Capture rate", "Mean time-to-capture (s)"]):
        ax.set_xticks(np.arange(len(SETS)) + 0.4 - w / 2)
        ax.set_xticklabels(SETS)
        ax.set_title(t)
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{out}/bars.png", dpi=130)
    plt.close()

    # Per-evader heatmap (seed-averaged capture rate) and own-error rate.
    opps = [(T, o) for T in SETS for o in next(iter(next(iter(data.values())).get(T, {}).values()), {})]
    for key, name in (("win_rate", "capture"), ("agent_error_rate", "own_error")):
        M = np.full((len(methods), len(opps)), np.nan)
        for i, m in enumerate(methods):
            for k, (T, o) in enumerate(opps):
                v = [r[o][key] for r in data[m].get(T, {}).values() if o in r]
                if v:
                    M[i, k] = np.mean(v)
        plt.figure(figsize=(1 + 0.6 * len(opps), 1 + 0.5 * len(methods)))
        plt.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        plt.xticks(range(len(opps)), [f"{T}:{o.split(':')[-1]}" for T, o in opps], rotation=70, ha="right", fontsize=7)
        plt.yticks(range(len(methods)), methods, fontsize=7)
        plt.colorbar()
        plt.title(name)
        plt.tight_layout()
        plt.savefig(f"{out}/heatmap_{name}.png", dpi=130)
        plt.close()

    # Paired difference vs the first method (M0) on held-out sets.
    base = methods[0]
    lines += ["", f"Capture-rate difference vs {base} (mean over seeds, held-out T1+T2+T3):", ""]
    for m in methods[1:]:
        d = [seed_vals(data[m][T], "win_rate").mean() - seed_vals(data[base][T], "win_rate").mean()
             for T in ("T1", "T2", "T3") if T in data[m] and T in data[base]]
        lines.append(f"- {m}: {np.mean(d):+.3f}" if d else f"- {m}: n/a")
    lines += ["", "Figures: bars.png, heatmap_capture.png, heatmap_own_error.png",
              "Not yet produced (need logging not implemented): learning curves, adaptation-latency curves, trajectories/GIFs."]
    open(f"{out}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
