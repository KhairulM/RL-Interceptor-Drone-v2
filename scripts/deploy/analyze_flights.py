"""Summarise and plot the per-flight logs written by flight_log.py.

    python scripts/deploy/analyze_flights.py scripts/deploy/logs/flights --date 2026-11-26 \
        --out scripts/deploy/logs/analysis/2026-11-26

Operator review in <log_dir>/annotations.json (optional) adds reviewed_outcome / excluded / review_note columns;
the raw logs are never modified.

Writes summary.csv (one row per flight), and one PNG per flight: top view in arena coordinates, altitude,
pursuer-evader distance, and tilt over time. Positions are converted to arena coordinates with the flight's
config arena.origin; the arena box drawn is the pursuer policy's training arena (artifact metadata.json).
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def tilt_deg(roll, pitch):
    """Angle between body z and world z (deg) from ZYX roll/pitch in degrees."""
    return np.degrees(np.arccos(np.clip(np.cos(np.radians(roll)) * np.cos(np.radians(pitch)), -1.0, 1.0)))


def arena_of(meta):
    path = os.path.join(HERE, meta["config"]["policy"]["artifact_dir"], "metadata.json")
    if not os.path.isfile(path):
        return None
    m = json.load(open(path))
    return m["arena_half_xy"], m["arena_z_min"], m["arena_z_max"]


def analyse(csv_path, annotations):
    meta = json.load(open(csv_path[:-4] + ".json"))
    note = annotations.get(os.path.basename(csv_path)[:19], {})
    d = pd.read_csv(csv_path)
    origin = np.asarray(meta["arena_origin"], float)
    for role in ("pursuer", "evader"):
        for i, ax in enumerate("xyz"):
            d[f"{role}_l{ax}"] = d[f"{role}_{ax}"] - origin[i]
        d[f"{role}_tilt"] = tilt_deg(d[f"{role}_roll"], d[f"{role}_pitch"])
    policies = list(dict.fromkeys(d.evader_policy))
    switch_t = float(d.t[d.evader_policy != policies[0]].iloc[0]) if len(policies) > 1 else None
    i_min = int(d.distance.idxmin())
    row = {
        "flight": os.path.basename(csv_path)[:19],
        "scenario": os.path.splitext(os.path.basename(meta["config_path"]))[0],
        "pursuer": meta["pursuer"],
        "evader": meta["evader"],
        "outcome": meta["outcome"],
        "detail": meta["outcome_detail"],
        "reviewed_outcome": note.get("reviewed_outcome", meta["outcome"]),
        "excluded": bool(note.get("exclude", False)),
        "review_note": note.get("note", ""),
        "logged_s": round(float(d.t.iloc[-1]), 2),
        "ticks": len(d),
        "max_tick_gap_s": round(float(np.diff(d.t).max()), 3) if len(d) > 1 else None,
        "switch_t": switch_t,
        "policy_reached_after_switch": len(policies) > 1,
        "d0": round(float(d.distance.iloc[0]), 2),
        "d_min": round(float(d.distance.min()), 3),
        "t_d_min": round(float(d.t[i_min]), 2),
        "d_end": round(float(d.distance.iloc[-1]), 3),
        "mean_closing_mps": round(float(d.approach_velocity.mean()), 2),
        "max_closing_mps": round(float(d.approach_velocity.max()), 2),
    }
    for role, tag in (("pursuer", "p"), ("evader", "e")):
        row.update({
            f"{tag}_max_speed": round(float(d[f"{role}_speed"].max()), 2),
            f"{tag}_max_tilt": round(float(d[f"{role}_tilt"].max()), 1),
            f"{tag}_t_tilt30": (round(float(d.t[d[f"{role}_tilt"] > 30].iloc[0]), 2)
                                if (d[f"{role}_tilt"] > 30).any() else None),
            f"{tag}_z_min": round(float(d[f"{role}_lz"].min()), 2),
            f"{tag}_z_max": round(float(d[f"{role}_lz"].max()), 2),
            f"{tag}_max_abs_xy": round(float(np.maximum(d[f"{role}_lx"].abs(), d[f"{role}_ly"].abs()).max()), 2),
        })
    return row, d, meta


def plot(d, meta, row, out_png):
    arena = arena_of(meta)
    cap = float(meta["capture_radius"])
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    colors = {"pursuer": "tab:red", "evader": "tab:blue"}
    ax = axs[0, 0]
    if arena:
        h = arena[0]
        ax.plot([-h, h, h, -h, -h], [-h, -h, h, h, -h], "k--", lw=1, label="training arena")
    for role, c in colors.items():
        ax.plot(d[f"{role}_lx"], d[f"{role}_ly"], color=c, label=role)
        ax.plot(d[f"{role}_lx"].iloc[0], d[f"{role}_ly"].iloc[0], "o", color=c)
        ax.plot(d[f"{role}_lx"].iloc[-1], d[f"{role}_ly"].iloc[-1], "x", color=c, ms=10)
    ax.set(xlabel="x (m, arena)", ylabel="y (m, arena)", title="Top view (o start, x end)", aspect="equal")
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    for role, c in colors.items():
        ax.plot(d.t, d[f"{role}_lz"], color=c, label=role)
    if arena:
        ax.axhline(arena[1], color="k", ls="--", lw=1)
    ax.set(xlabel="t (s)", ylabel="z (m)", title="Altitude")
    ax = axs[1, 0]
    ax.plot(d.t, d.distance, color="k", label="distance")
    ax.axhline(cap, color="g", ls="--", label=f"capture {cap} m")
    ax2 = ax.twinx()
    ax2.plot(d.t, d.approach_velocity, color="tab:purple", alpha=0.6)
    ax2.set_ylabel("closing speed (m/s)", color="tab:purple")
    ax.set(xlabel="t (s)", ylabel="distance (m)", title="Distance / closing speed")
    ax.legend(fontsize=8, loc="upper right")
    ax = axs[1, 1]
    for role, c in colors.items():
        ax.plot(d.t, d[f"{role}_tilt"], color=c, label=f"{role} tilt")
        ax.plot(d.t, d[f"{role}_speed"] * 10, color=c, ls=":", label=f"{role} speed x10")
    ax.set(xlabel="t (s)", ylabel="deg  |  m/s x10", title="Tilt and speed")
    ax.legend(fontsize=8)
    for a in (axs[0, 1], axs[1, 0], axs[1, 1]):
        if row["switch_t"] is not None:
            a.axvline(row["switch_t"], color="orange", lw=1)
    reviewed = ("" if row["reviewed_outcome"] == row["outcome"] else f"   [reviewed: {row['reviewed_outcome']}]") + \
        ("   [EXCLUDED]" if row["excluded"] else "")
    fig.suptitle(f"{row['flight']}  {row['scenario']}\nP: {row['pursuer']}   E: {row['evader']}\n"
                 f"{row['outcome']}: {row['detail'][:120]}{reviewed}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--date", default="", help="only flights whose file name starts with this (YYYY-MM-DD)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ann_path = os.path.join(args.log_dir, "annotations.json")   # operator review: {flight: {reviewed_outcome, exclude, note}}
    annotations = json.load(open(ann_path)) if os.path.isfile(ann_path) else {}
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(args.log_dir, f"{args.date}*.csv"))):
        row, d, meta = analyse(csv_path, annotations)
        plot(d, meta, row, os.path.join(args.out, row["flight"] + ".png"))
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
