"""F1_final: 2x2 factorial {uniform, QD-DPP sampler} x {plain obs, opponent-acceleration input}, 5 seeds, noise-free.

Cells (existing runs reused where the configuration is identical):
  SU_plain  uniform + plain obs          = X3 generalists
  QD_plain  QD-DPP + plain obs           = D1 QD
  SU_acc    uniform + acceleration input = X4 O2a (seeds 0-2) + new seeds here
  QD_acc    QD-DPP + acceleration input  = new here
A cell with the learned acceleration estimator (A2) is added only if A2 beats the finite-difference input.
"""

import json
import os

import numpy as np

from common import OUT, evaluate, log_progress, save_json, train
from d_sampling import D1, sampler_overrides, test_sets, write_descriptors
from x3_gap import EXP as X3, GEN_FRAMES

F1 = os.path.join(OUT, "F1_final")
X4 = os.path.join(OUT, "X4_oracles")
ACC = ["task.oracle.opp_accel=true"]


def run(args):
    split, test = test_sets()
    pool = split["train"]
    desc = write_descriptors(os.path.join(F1, "descriptors_train.json"), pool)
    frames = 2_000_000 if args.smoke else GEN_FRAMES
    for s in range(args.seeds):
        jobs = []
        if not os.path.isfile(os.path.join(X4, "ckpt", f"O2a_accel_seed{s}.pt")):
            jobs.append(("SU_acc", ACC))
        jobs.append(("QD_acc", ACC + sampler_overrides("QD", desc, os.path.join(F1, "QD_acc", f"refresh_seed{s}.jsonl"))))
        for cell, ov in jobs:
            d = os.path.join(F1, cell)
            ck = train(os.path.join(d, "ckpt", f"seed{s}.pt"), pool, frames, seed=s, extra=ov)
            log_progress(f"F1 {cell} seed{s}: {ck}")
            if ck:
                evaluate(os.path.join(d, "evals", f"seed{s}_train.json"), pool, policy_ckpt=ck, extra=ACC)
                evaluate(os.path.join(d, "evals", f"seed{s}_test.json"), test, policy_ckpt=ck, extra=ACC)
    analyse(split, test, args.seeds)


def _J(*paths):
    out = {}
    for p in paths:
        if os.path.isfile(p):
            with open(p) as f:
                out.update({k: v["neutralization_rate"] for k, v in json.load(f)["matchups"].items()})
    return out


def cell_J(cell, s):
    if cell == "SU_plain":
        return _J(*(os.path.join(X3, "evals", f"generalist{s}_{p}.json") for p in ("train", "test")))
    if cell == "QD_plain":
        return _J(*(os.path.join(D1, "QD", "evals", f"seed{s}_{p}.json") for p in ("train", "test")))
    if cell == "SU_acc" and os.path.isfile(os.path.join(X4, "evals", f"O2a_accel_seed{s}_train.json")):
        return _J(*(os.path.join(X4, "evals", f"O2a_accel_seed{s}_{p}.json") for p in ("train", "test", "exploit")))
    return _J(*(os.path.join(F1, cell, "evals", f"seed{s}_{p}.json") for p in ("train", "test")))


def analyse(split, test, n_seeds):
    import analysis as an
    groups = {"train": [e["id"] for e in split["train"]], "heldout_rl": [e["id"] for e in split["heldout_rl"]],
              "heldout_scripted": [e["id"] for e in split["heldout_scripted"]],
              "exploiter": [e["id"] for e in test if e["id"].startswith("exploit/")]}
    cells = ["SU_plain", "QD_plain", "SU_acc", "QD_acc"]
    per = {c: {} for c in cells}
    for c in cells:
        for s in range(n_seeds):
            J = cell_J(c, s)
            if J and all(i in J for i in groups["train"]):
                per[c][s] = {g: float(np.mean([J[i] for i in ids if i in J])) for g, ids in groups.items()
                             if any(i in J for i in ids)}
    agg = {c: {g: dict(zip(("mean", "lo", "hi"), an.bootstrap_ci([r[g] for r in per[c].values() if g in r])))
               for g in groups} for c in cells if per[c]}
    # Seed-paired effects (only seeds present in all four cells).
    seeds = [s for s in range(n_seeds) if all(s in per[c] for c in cells)]
    effects = {}
    for g in groups:
        v = {c: np.array([per[c][s][g] for s in seeds]) for c in cells}
        eff = {"sampler (QD - SU)": (v["QD_plain"] + v["QD_acc"] - v["SU_plain"] - v["SU_acc"]) / 2,
               "input (acc - plain)": (v["SU_acc"] + v["QD_acc"] - v["SU_plain"] - v["QD_plain"]) / 2,
               "interaction": (v["QD_acc"] - v["SU_acc"]) - (v["QD_plain"] - v["SU_plain"]),
               "QD vs SU, plain": v["QD_plain"] - v["SU_plain"], "QD vs SU, acc": v["QD_acc"] - v["SU_acc"]}
        effects[g] = {k: {"per_seed": x.tolist(), **dict(zip(("mean", "lo", "hi"), an.bootstrap_ci(x)))}
                      for k, x in eff.items()}
    save_json(os.path.join(F1, "results.json"), {"per_seed": per, "aggregate": agg, "effects": effects,
                                                 "paired_seeds": seeds})
    lines = ["# F1_final summary", "", f"Seeds per cell: { {c: sorted(per[c]) for c in cells} }", "",
             "Mean J (95 % bootstrap CI over seeds).", "",
             "| cell | " + " | ".join(groups) + " |", "|---" * (len(groups) + 1) + "|"]
    for c in cells:
        if c in agg:
            lines.append(f"| {c} | " + " | ".join(
                f"{agg[c][g]['mean']:.3f} [{agg[c][g]['lo']:.3f}, {agg[c][g]['hi']:.3f}]" for g in groups) + " |")
    lines += ["", f"Seed-paired effects over seeds {seeds} (mean [95 % CI]):", "",
              "| effect | " + " | ".join(groups) + " |", "|---" * (len(groups) + 1) + "|"]
    for k in next(iter(effects.values())):
        lines.append(f"| {k} | " + " | ".join(
            f"{effects[g][k]['mean']:+.3f} [{effects[g][k]['lo']:+.3f}, {effects[g][k]['hi']:+.3f}]" for g in groups) + " |")
    with open(os.path.join(F1, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log_progress(f"F1 analysed, paired seeds {seeds}")
