# MIT License — see repository root.
"""Visualise the pursuer-vs-evader win-rate matrix from a self-play DB.

Reads a run's matchups.json + pool files and builds a pursuer-generation x
evader-generation matrix of the pursuer capture rate (win_rate). Saves a local
PNG heatmap and, with --wandb, logs it to Weights & Biases as (1) an image
panel, (2) a wandb.Table, and (3) a native heatmap custom chart built from that
table so it is interactive in the UI.

Usage:
    python scripts/selfplay/plot_winrate_matrix.py \
        --db scripts/selfplay/outputs/selfplay_v2/db \
        [--wandb] [--project "RL Pursuit Evasion"] [--entity khairul-makirin-team] \
        [--run-name winrate-matrix-v2] [--metric win_rate]
"""
import argparse, json, os
import numpy as np


def _gen_key(pid: str):
    # order: heuristics first (heuristic0,1,2), then s2,s3,... by number
    tag = pid.split(":", 1)[1]
    if tag.startswith("heuristic"):
        return (0, int(tag[len("heuristic"):]))
    if tag.startswith("s"):
        return (1, int(tag[1:]))
    return (2, tag)


def load_matrix(db, metric="win_rate"):
    M = json.load(open(f"{db}/matchups.json"))["matchups"]
    purs = [p["id"] for p in json.load(open(f"{db}/pursuer_pool.json"))["policies"]]
    evad = [p["id"] for p in json.load(open(f"{db}/evader_pool.json"))["policies"]]
    purs = sorted(purs, key=_gen_key)
    evad = sorted(evad, key=_gen_key)
    mat = np.full((len(purs), len(evad)), np.nan)
    eps = np.zeros((len(purs), len(evad)))
    for i, p in enumerate(purs):
        for j, e in enumerate(evad):
            rec = M.get(f"{p}|{e}")
            if rec is not None and metric in rec:
                mat[i, j] = float(rec[metric])
                eps[i, j] = int(rec.get("episodes", 0))
    return purs, evad, mat, eps


def short(label):  # "pursuer:s3" -> "s3", "evader:heuristic2" -> "h2(flee)"
    tag = label.split(":", 1)[1]
    names = {"heuristic0": "h0", "heuristic1": "h1", "heuristic2": "h2"}
    hint = {"pursuer": {"heuristic0": "pursue", "heuristic1": "hover"},
            "evader": {"heuristic0": "hover", "heuristic1": "circ", "heuristic2": "flee"}}
    role = label.split(":", 1)[0]
    if tag in names:
        h = hint.get(role, {}).get(tag, "")
        return f"{names[tag]}({h})" if h else names[tag]
    return tag


def save_png(purs, evad, mat, eps, out_png, metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.1 * len(evad) + 3, 0.9 * len(purs) + 2))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(evad))); ax.set_xticklabels([short(e) for e in evad], rotation=45, ha="right")
    ax.set_yticks(range(len(purs))); ax.set_yticklabels([short(p) for p in purs])
    ax.set_xlabel("evader generation"); ax.set_ylabel("pursuer generation")
    ax.set_title(f"Pursuer capture rate ({metric})  —  pursuer vs evader")
    for i in range(len(purs)):
        for j in range(len(evad)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] < 0.6 else "black", fontsize=8)
            else:
                ax.text(j, i, "-", ha="center", va="center", color="gray", fontsize=8)
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--metric", default="win_rate")
    ap.add_argument("--out", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--project", default="RL Pursuit Evasion")
    ap.add_argument("--entity", default="khairul-makirin-team")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    purs, evad, mat, eps = load_matrix(args.db, args.metric)
    out_png = args.out or os.path.join(os.path.dirname(args.db.rstrip("/")), "winrate_matrix.png")
    save_png(purs, evad, mat, eps, out_png, args.metric)
    print(f"[plot] saved {out_png}  ({len(purs)}x{len(evad)}, "
          f"{int(np.isfinite(mat).sum())} cells filled)")

    if args.wandb:
        import wandb
        run = wandb.init(project=args.project, entity=args.entity,
                         name=args.run_name or f"winrate-matrix-{os.path.basename(os.path.dirname(args.db.rstrip('/')))}",
                         job_type="analysis")
        # 1) image panel
        wandb.log({f"winrate_matrix/{args.metric}": wandb.Image(out_png)})
        # 2) long-form table (pursuer, evader, value, episodes)
        tbl = wandb.Table(columns=["pursuer", "evader", args.metric, "episodes"])
        for i, p in enumerate(purs):
            for j, e in enumerate(evad):
                if np.isfinite(mat[i, j]):
                    tbl.add_data(short(p), short(e), float(mat[i, j]), int(eps[i, j]))
        wandb.log({"winrate_table": tbl})
        # 3) native interactive heatmap from the table
        try:
            hm = wandb.plot.HeatMap(x_labels=[short(e) for e in evad],
                                    y_labels=[short(p) for p in purs],
                                    matrix_values=np.nan_to_num(mat, nan=0.0).tolist(),
                                    show_text=True)
            wandb.log({"winrate_heatmap": hm})
        except Exception as ex:
            print(f"[plot] wandb.plot.HeatMap unavailable ({ex}); image+table logged.")
        run.finish()
        print("[plot] logged to wandb.")


if __name__ == "__main__":
    main()
