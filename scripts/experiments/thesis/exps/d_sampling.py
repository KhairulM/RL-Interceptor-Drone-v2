"""D1_sampler, D1b_duplicates, D2_exploitability: diversity-aware opponent sampling.

All pursuers: train split (X2), 150M frames, opponent probabilities refreshed every 25M frames from the episodes
played (run_stage `refresh.*`). Samplers:
  S-U   uniform (identical to the X3 generalist, reused)
  S-W   hardness softmax with win rate only (PFSP / Hedge-style)
  S-M   hardness softmax with the multi-metric weights of v11 (current method)
  QD    quality-diversity DPP over X2 behaviour descriptors, same quality as S-M
D1b: train pool + 5 extra copies of the hardest train RL evader; S-M vs QD.
D2: a fresh exploiter evader (100M frames, from scratch) per D1 pursuer; exploitability = 1 - J(pursuer, exploiter).
"""

import json
import os

import numpy as np

from common import OUT, evaluate, log_progress, save_json, train
from x3_gap import EXP as X3, GEN_FRAMES, generalist_ckpt, load_split

D1 = os.path.join(OUT, "D1_sampler")
D1B = os.path.join(OUT, "D1b_duplicates")
D2 = os.path.join(OUT, "D2_exploitability")
REFRESH_FRAMES = 25_000_000
EXPLOIT_FRAMES = 100_000_000
N_DUP = 5

V11_WEIGHTS = {"win_rate": 1.0, "avg_time_to_capture": 0.8, "avg_min_distance": 0.5,
               "avg_motor_effort": 0.3, "pursuer_crash_rate": 0.0}
WIN_ONLY = {"win_rate": 1.0, "avg_time_to_capture": 0.0, "avg_min_distance": 0.0,
            "avg_motor_effort": 0.0, "pursuer_crash_rate": 0.0}


def _weights(w):
    return [f"refresh.metric_weights.{k}={v}" for k, v in w.items()]


def sampler_overrides(name, desc_path, log_path, uniform_mix=0.5):
    base = [f"refresh.every_frames={REFRESH_FRAMES}", "refresh.temperature=0.9", f"refresh.uniform_mix={uniform_mix}",
            f"refresh.log_path={log_path}"]
    if name == "S-W":
        return base + ["refresh.sampler=softmax"] + _weights(WIN_ONLY)
    if name == "S-M":
        return base + ["refresh.sampler=softmax"] + _weights(V11_WEIGHTS)
    if name == "QD":
        return base + ["refresh.sampler=qd_dpp", f"refresh.descriptors_path={desc_path}"] + _weights(V11_WEIGHTS)
    raise ValueError(name)


def write_descriptors(path, pool):
    with open(os.path.join(OUT, "X2_descriptors", "descriptors.json")) as f:
        d = json.load(f)
    table = dict(zip(d["ids"], d["descriptor"]))
    out = {}
    for e in pool:
        src = (e.get("metadata") or {}).get("duplicate_of", e["id"])
        out[e["id"]] = table[src]
    save_json(path, out)
    return path


def test_sets():
    split = load_split()
    with open(os.path.join(X3, "exploiters.json")) as f:
        exploiters = json.load(f)
    return split, split["heldout_rl"] + split["heldout_scripted"] + exploiters


def _train_and_eval(root, name, seed, pool, test, frames, desc_path, uniform_mix=0.5):
    d = os.path.join(root, name)
    ck = os.path.join(d, "ckpt", f"seed{seed}.pt")
    ck = train(ck, pool, frames, seed=seed,
               extra=sampler_overrides(name, desc_path, os.path.join(d, f"refresh_seed{seed}.jsonl"), uniform_mix))
    log_progress(f"{os.path.basename(root)} {name} seed{seed}: {ck}")
    if ck:
        evaluate(os.path.join(d, "evals", f"seed{seed}_train.json"), pool, policy_ckpt=ck)
        evaluate(os.path.join(d, "evals", f"seed{seed}_test.json"), test, policy_ckpt=ck)
    return ck


def run_d1(args):
    split, test = test_sets()
    pool = split["train"]
    desc = write_descriptors(os.path.join(D1, "descriptors_train.json"), pool)
    frames = 2_000_000 if args.smoke else GEN_FRAMES
    for seed in range(args.seeds):
        for name in ("S-W", "S-M", "QD"):
            _train_and_eval(D1, name, seed, pool, test, frames, desc)
    analyse(D1, ["S-U", "S-W", "S-M", "QD"], args.seeds, split, test)


def hardest_train_rl(split):
    with open(os.path.join(X3, "evals", "generalist0_train.json")) as f:
        m = json.load(f)["matchups"]
    rl = [e for e in split["train"] if e["kind"] == "rl"]
    return min(rl, key=lambda e: m[e["id"]]["neutralization_rate"])


def run_d1b(args):
    split, test = test_sets()
    hard = hardest_train_rl(split)
    dups = [{**hard, "id": f"dup{k}/{hard['id']}", "metadata": {**hard["metadata"], "duplicate_of": hard["id"]}}
            for k in range(N_DUP)]
    pool = split["train"] + dups
    save_json(os.path.join(D1B, "duplicated.json"), {"evader": hard["id"], "copies": N_DUP})
    desc = write_descriptors(os.path.join(D1B, "descriptors_dup.json"), pool)
    frames = 2_000_000 if args.smoke else GEN_FRAMES
    for seed in range(args.seeds):
        for name in ("S-M", "QD"):
            _train_and_eval(D1B, name, seed, pool, test, frames, desc)
    analyse(D1B, ["S-M", "QD"], args.seeds, split, test, exclude_train={hard["id"]} | {e["id"] for e in dups})


D1C = os.path.join(OUT, "D1c_nofloor")


def run_d1c(args):
    """Exploratory follow-up to D1b (designed after seeing D1b): no uniform floor (uniform_mix = 0), normal and
    duplicated pool, S-M vs QD-DPP."""
    split, test = test_sets()
    hard = hardest_train_rl(split)
    dups = [{**hard, "id": f"dup{k}/{hard['id']}", "metadata": {**hard["metadata"], "duplicate_of": hard["id"]}}
            for k in range(N_DUP)]
    frames = 2_000_000 if args.smoke else GEN_FRAMES
    pools = {"nodup": split["train"], "dup": split["train"] + dups}
    for seed in range(args.seeds):
        for tag, pool in pools.items():
            desc = write_descriptors(os.path.join(D1C, tag, "descriptors.json"), pool)
            for name in ("S-M", "QD"):
                _train_and_eval(os.path.join(D1C, tag), name, seed, pool, test, frames, desc, uniform_mix=0.0)
    for tag in pools:
        analyse(os.path.join(D1C, tag), ["S-M", "QD"], args.seeds, split, test,
                exclude_train={hard["id"]} | {e["id"] for e in dups})


def run_d2(args):
    frames = 2_000_000 if args.smoke else EXPLOIT_FRAMES
    rows = []
    for name in ("S-U", "S-W", "S-M", "QD"):
        for seed in range(args.seeds):
            ck = generalist_ckpt(seed) if name == "S-U" else os.path.join(D1, name, "ckpt", f"seed{seed}.pt")
            if not os.path.isfile(ck):
                continue
            target = {"id": f"d2/pursuer:{name}_s{seed}", "kind": "rl", "checkpoint_path": ck, "metadata": {}}
            ex_ck = os.path.join(D2, "ckpt", f"exploiter_{name}_seed{seed}.pt")
            ex_ck = train(ex_ck, [target], frames, role="evader", seed=200 + seed)
            if not ex_ck:
                continue
            ex = {"id": f"d2/evader:exploit_{name}_s{seed}", "kind": "rl", "checkpoint_path": ex_ck, "metadata": {}}
            res = evaluate(os.path.join(D2, "evals", f"{name}_seed{seed}.json"), [ex], policy_ckpt=ck)
            if res:
                m = res["matchups"][ex["id"]]
                rows.append({"sampler": name, "seed": seed, "J_vs_exploiter": m["neutralization_rate"],
                             "capture_vs_exploiter": m["capture_rate"], "exploitability": 1 - m["neutralization_rate"],
                             "pursuer_crash": m["outcomes"]["pursuer_crash"], "timeout": m["outcomes"]["timeout"]})
            log_progress(f"D2 {name} seed{seed}: {rows[-1] if rows else None}")
    save_json(os.path.join(D2, "exploit.json"), rows)
    lines = ["# D2_exploitability summary", "", "| sampler | exploitability per seed | mean |", "|---|---|---|"]
    for name in ("S-U", "S-W", "S-M", "QD"):
        v = [r["exploitability"] for r in rows if r["sampler"] == name]
        if v:
            lines.append(f"| {name} | {', '.join(f'{x:.3f}' for x in v)} | {np.mean(v):.3f} |")
    with open(os.path.join(D2, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _J(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return {k: v["neutralization_rate"] for k, v in json.load(f)["matchups"].items()}


def _paths(root, name, seed, part):
    if name == "S-U":
        return os.path.join(X3, "evals", f"generalist{seed}_{part}.json")
    return os.path.join(root, name, "evals", f"seed{seed}_{part}.json")


def analyse(root, names, n_seeds, split, test, exclude_train=frozenset()):
    import analysis as an
    with open(os.path.join(OUT, "X2_descriptors", "clusters.json")) as f:
        labels = json.load(f)["labels"]
    groups = {"train": [e["id"] for e in split["train"] if e["id"] not in exclude_train],
              "heldout_rl": [e["id"] for e in split["heldout_rl"]],
              "heldout_scripted": [e["id"] for e in split["heldout_scripted"]],
              "exploiter": [e["id"] for e in test if e["id"].startswith("exploit/")]}
    held = groups["heldout_rl"] + groups["heldout_scripted"]
    res = {}
    for name in names:
        per_seed = []
        for s in range(n_seeds):
            J = {**_J(_paths(root, name, s, "train")), **_J(_paths(root, name, s, "test"))}
            if not J:
                continue
            row = {g: float(np.mean([J[i] for i in ids if i in J])) for g, ids in groups.items()
                   if any(i in J for i in ids)}
            # Worst cluster on held-out evaders (clusters from X2; exploiters have no cluster).
            by_c = {}
            for i in held:
                if i in J and i in labels:
                    by_c.setdefault(labels[i], []).append(J[i])
            if by_c:
                row["heldout_worst_cluster"] = float(min(np.mean(v) for v in by_c.values()))
            row["heldout_worst_evader"] = float(min(J[i] for i in held if i in J)) if any(i in J for i in held) else None
            per_seed.append(row)
        res[name] = per_seed
    keys = ["train", "heldout_rl", "heldout_scripted", "exploiter", "heldout_worst_cluster", "heldout_worst_evader"]
    agg = {}
    for name, rows in res.items():
        agg[name] = {}
        for k in keys:
            v = [r[k] for r in rows if r.get(k) is not None]
            if v:
                m, lo, hi = an.bootstrap_ci(v)
                agg[name][k] = {"mean": m, "ci": [lo, hi], "per_seed": v}
    save_json(os.path.join(root, "results.json"), {"per_seed": res, "aggregate": agg, "groups": groups})
    lines = [f"# {os.path.basename(root)} summary", "",
             "Mean neutralisation J (95 % bootstrap CI over seeds; per-seed values).", "",
             "| sampler | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for name in names:
        cells = []
        for k in keys:
            a = agg.get(name, {}).get(k)
            cells.append(f"{a['mean']:.3f} [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}]" if a else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
