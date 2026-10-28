"""X3_gap: is there an adaptation gap? Generalist vs specialists on held-out evaders.

1. Generalist pi_G: pursuer trained from scratch on the X2 train split (uniform opponents), 150M frames, seeds 0..2.
   (Reused as M0 / uniform-sampler baseline by later experiments.)
2. Exploiters: evaders trained from scratch against frozen pi_G seed 0 (100M frames, 3 seeds).
3. Specialist per test evader E: pi_G seed 0 fine-tuned 20M frames against E alone.
Gap Delta_E = J(specialist_E, E) - J(pi_G, E), J = neutralisation rate.
"""

import json
import os

import numpy as np

from common import OUT, evaluate, log_progress, save_json, train

EXP = os.path.join(OUT, "X3_gap")
SPLIT = os.path.join(OUT, "X2_descriptors", "split.json")
GEN_FRAMES = 150_000_000
EXPLOIT_FRAMES = 100_000_000
SPEC_FRAMES = 20_000_000


def load_split():
    with open(SPLIT) as f:
        return json.load(f)


def generalist_ckpt(seed):
    return os.path.join(EXP, "ckpt", f"generalist_seed{seed}.pt")


def train_generalists(split, seeds, frames=GEN_FRAMES):
    out = []
    for s in seeds:
        ck = train(generalist_ckpt(s), split["train"], frames, role="pursuer", seed=s)
        log_progress(f"X3 generalist seed{s}: {ck}")
        out.append(ck)
    return out


def exploiter_policy(i):
    return {"id": f"exploit/evader:g0e{i}", "kind": "rl", "checkpoint_path": os.path.join(EXP, "ckpt", f"exploiter{i}.pt"),
            "metadata": {"exploits": "generalist_seed0", "seed": i}}


def run(args):
    split = load_split()
    seeds = list(range(args.seeds))
    gen_frames = 2_000_000 if args.smoke else GEN_FRAMES
    gens = train_generalists(split, seeds, gen_frames)
    if gens[0] is None:
        raise RuntimeError("generalist seed 0 failed")

    g0 = {"id": "x3/pursuer:generalist0", "kind": "rl", "checkpoint_path": gens[0], "metadata": {}}
    exploiters = []
    for i in range(3):
        pol = exploiter_policy(i)
        ck = train(pol["checkpoint_path"], [g0], 2_000_000 if args.smoke else EXPLOIT_FRAMES, role="evader", seed=100 + i)
        log_progress(f"X3 exploiter {i}: {ck}")
        if ck:
            exploiters.append(pol)
    save_json(os.path.join(EXP, "exploiters.json"), exploiters)

    test = split["heldout_rl"] + split["heldout_scripted"] + exploiters
    # Generalists vs everything (train + test).
    for s, ck in zip(seeds, gens):
        if ck:
            evaluate(os.path.join(EXP, "evals", f"generalist{s}_train.json"), split["train"], policy_ckpt=ck)
            evaluate(os.path.join(EXP, "evals", f"generalist{s}_test.json"), test, policy_ckpt=ck)

    # Specialists (fine-tuned from generalist seed 0) per test evader.
    for e in test:
        tag = e["id"].replace("/", "_").replace(":", "_").replace(">", "-").replace("@", "at")
        ck = train(os.path.join(EXP, "ckpt", f"spec_{tag}.pt"), [e], 2_000_000 if args.smoke else SPEC_FRAMES,
                   role="pursuer", init_policy=gens[0], seed=0)
        if ck:
            evaluate(os.path.join(EXP, "evals", f"spec_{tag}.json"), [e], policy_ckpt=ck)
        log_progress(f"X3 specialist {e['id']}: {ck}")
    analyse(split, test, seeds)


def _load(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)["matchups"]


def analyse(split, test, seeds):
    g = [_load(os.path.join(EXP, "evals", f"generalist{s}_test.json")) for s in seeds]
    gt = [_load(os.path.join(EXP, "evals", f"generalist{s}_train.json")) for s in seeds]
    rows = []
    for e in test:
        tag = e["id"].replace("/", "_").replace(":", "_").replace(">", "-").replace("@", "at")
        spec = _load(os.path.join(EXP, "evals", f"spec_{tag}.json")).get(e["id"])
        j_g0 = g[0].get(e["id"], {}).get("neutralization_rate")
        row = {"evader": e["id"], "group": _group(e, split),
               "J_generalist_seed0": j_g0,
               "J_generalist_seeds": [x.get(e["id"], {}).get("neutralization_rate") for x in g],
               "capture_generalist_seed0": g[0].get(e["id"], {}).get("capture_rate"),
               "J_specialist": spec["neutralization_rate"] if spec else None,
               "capture_specialist": spec["capture_rate"] if spec else None}
        if spec and j_g0 is not None:
            row["delta"] = spec["neutralization_rate"] - j_g0
        rows.append(row)
    train_j = [[m["neutralization_rate"] for m in x.values()] for x in gt if x]
    res = {"rows": rows, "generalist_train_mean_J": [float(np.mean(v)) for v in train_j]}
    save_json(os.path.join(EXP, "gap.json"), res)
    lines = ["# X3_gap summary", "",
             f"Generalist mean J on train pool per seed: {', '.join(f'{v:.3f}' for v in res['generalist_train_mean_J'])}",
             "", "| evader | group | J gen s0 | J gen seeds | J specialist | Delta |", "|---|---|---|---|---|---|"]
    for r in rows:
        js = ", ".join(f"{v:.2f}" if v is not None else "-" for v in r["J_generalist_seeds"])
        f = lambda v: f"{v:.3f}" if v is not None else "-"
        lines.append(f"| {r['evader']} | {r['group']} | {f(r['J_generalist_seed0'])} | {js} | "
                     f"{f(r['J_specialist'])} | {f(r.get('delta'))} |")
    big = [r for r in rows if (r.get("delta") or 0) >= 0.15]
    lines += ["", f"Evaders with Delta >= 0.15: {len(big)} ({', '.join(r['evader'] for r in big)})"]
    with open(os.path.join(EXP, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _group(e, split):
    ids = lambda key: {x["id"] for x in split[key]}
    if e["id"] in ids("heldout_rl"):
        return "heldout_rl"
    if e["id"] in ids("heldout_scripted"):
        return "heldout_scripted"
    return "exploiter"
