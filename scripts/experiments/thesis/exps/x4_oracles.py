"""X4_oracles: is privileged opponent information worth anything?

Pursuers trained exactly like the X3 generalist (train split, uniform, 150M frames, seeds 0..2) but with one oracle
input appended to the observation:
  O1  descriptor of the assigned opponent (top-8 PCA of the X2 behaviour descriptor);
  O2a opponent acceleration (finite difference; something an online estimator could recover);
  O2b opponent's latest CTBR command ("intent"; upper bound).
O3 (future position) from the plan is dropped: A1a's offline prediction study covers the horizon question.
"""

import json
import os

import numpy as np

from common import OUT, evaluate, log_progress, save_json, train
from x3_gap import EXP as X3, GEN_FRAMES, load_split

EXP = os.path.join(OUT, "X4_oracles")
DESC_PATH = os.path.join(EXP, "oracle_descriptors.json")
ORACLES = {
    "O1_descriptor": [f"task.oracle.descriptor_path={DESC_PATH}"],
    "O2a_accel": ["task.oracle.opp_accel=true"],
    "O2b_command": ["task.oracle.opp_command=true"],
}


def write_oracle_descriptors(k=8):
    with open(os.path.join(OUT, "X2_descriptors", "descriptors.json")) as f:
        d = json.load(f)
    X = np.array(d["descriptor"])
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:k].T
    Z = Z / Z.std(0).clip(1e-6)
    table = {i: z.tolist() for i, z in zip(d["ids"], Z)}
    save_json(DESC_PATH, table)
    return {"explained_variance": (S[:k] ** 2 / (S ** 2).sum()).tolist()}


def run(args):
    split = load_split()
    with open(os.path.join(X3, "exploiters.json")) as f:
        exploiters = json.load(f)
    pca = write_oracle_descriptors()
    save_json(os.path.join(EXP, "oracle_descriptor_pca.json"), pca)
    frames = 2_000_000 if args.smoke else GEN_FRAMES
    test = split["heldout_rl"] + split["heldout_scripted"]
    for name, ov in ORACLES.items():
        for s in range(args.seeds):
            ck = train(os.path.join(EXP, "ckpt", f"{name}_seed{s}.pt"), split["train"], frames, seed=s, extra=ov)
            log_progress(f"X4 {name} seed{s}: {ck}")
            if not ck:
                continue
            evaluate(os.path.join(EXP, "evals", f"{name}_seed{s}_train.json"), split["train"], policy_ckpt=ck, extra=ov)
            evaluate(os.path.join(EXP, "evals", f"{name}_seed{s}_test.json"), test, policy_ckpt=ck, extra=ov)
            if name != "O1_descriptor" and exploiters:
                evaluate(os.path.join(EXP, "evals", f"{name}_seed{s}_exploit.json"), exploiters,
                         policy_ckpt=ck, extra=ov)
    analyse(split, exploiters, args.seeds)


def _J(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return {k: v["neutralization_rate"] for k, v in json.load(f)["matchups"].items()}


def analyse(split, exploiters, n_seeds):
    with open(os.path.join(X3, "gap.json")) as f:
        gap = {r["evader"]: r for r in json.load(f)["rows"]}
    methods = {"M0_generalist": lambda s, part: os.path.join(X3, "evals", f"generalist{s}_{'train' if part == 'train' else 'test'}.json")}
    for name in ORACLES:
        methods[name] = (lambda n: lambda s, part: os.path.join(EXP, "evals", f"{n}_seed{s}_{part}.json"))(name)
    groups = {"train": [e["id"] for e in split["train"]], "heldout_rl": [e["id"] for e in split["heldout_rl"]],
              "heldout_scripted": [e["id"] for e in split["heldout_scripted"]],
              "exploiter": [e["id"] for e in exploiters]}
    table = {}
    for m, path in methods.items():
        per_seed = []
        for s in range(n_seeds):
            J = {}
            for part in ("train", "test", "exploit"):
                if m == "M0_generalist" and part == "exploit":
                    J.update(_J(path(s, "test")))
                else:
                    J.update(_J(path(s, part)))
            per_seed.append(J)
        table[m] = per_seed
    summary = {}
    for m, per_seed in table.items():
        summary[m] = {}
        for g, ids in groups.items():
            vals = [np.mean([J[i] for i in ids if i in J]) for J in per_seed if any(i in J for i in ids)]
            summary[m][g] = {"mean": float(np.mean(vals)) if vals else None, "per_seed": [float(v) for v in vals]}
        gc = []
        for eid, r in gap.items():
            if (r.get("delta") or 0) < 0.15:
                continue
            j_m = [J[eid] for J in per_seed if eid in J]
            j_g = [J[eid] for J in table["M0_generalist"] if eid in J]
            if j_m and j_g:
                gc.append((np.mean(j_m) - np.mean(j_g)) / r["delta"])
        summary[m]["gap_closed_mean"] = float(np.mean(gc)) if gc else None
        summary[m]["gap_closed_n"] = len(gc)
    save_json(os.path.join(EXP, "oracle.json"), {"summary": summary, "per_seed": table})
    lines = ["# X4_oracles summary", "", "Mean neutralisation J over evaders in each group (mean over seeds; per-seed in brackets).",
             "", "| method | train | heldout_rl | heldout_scripted | exploiter | gap closed (n) |", "|---|---|---|---|---|---|"]
    for m, s in summary.items():
        cell = lambda g: (f"{s[g]['mean']:.3f} [{', '.join(f'{v:.2f}' for v in s[g]['per_seed'])}]"
                          if s[g]["mean"] is not None else "-")
        gc = f"{s['gap_closed_mean']:.2f} ({s['gap_closed_n']})" if s["gap_closed_mean"] is not None else "-"
        lines.append(f"| {m} | {cell('train')} | {cell('heldout_rl')} | {cell('heldout_scripted')} | {cell('exploiter')} | {gc} |")
    with open(os.path.join(EXP, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
