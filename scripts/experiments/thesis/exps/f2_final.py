"""F2: final runs (docs/research_plan.md, Phase F, pre-registered 10-28 19:30).

F2a  Exploitability of the acceleration-input pursuers (F1 "SU + acc", seeds 0-4): one fresh plain-obs exploiter each,
     paired with D2 "S-U" (plain obs, same seeds, same exploiter seeds).
F2b  Two self-play campaigns, identical except the input: acc (both roles observe the opponent's finite-difference
     acceleration, frozen opponents too) vs plain. Uniform sampling, 9 stages x 300M frames, 1 seed per arm.
F2c  Cross-play of all campaign pursuers vs all campaign evaders, final pursuers vs the thesis train pool and test
     set, and one fresh plain-obs exploiter per final pursuer.
"""

import json
import os
import subprocess
import time

import numpy as np

import common
from a_adapt import test_set
from common import OUT, REPO, _acquire_isaac_slot, _acquire_lock, _clean_env, evaluate, log_progress, save_json, train
from d_sampling import D2, EXPLOIT_FRAMES
from x3_gap import generalist_ckpt, load_split

F2A = os.path.join(OUT, "F2a_exploit_acc")
F2C = os.path.join(OUT, "F2c_final_eval")
ACC = ["task.oracle.opp_accel=true"]           # the agent observes its opponent's acceleration
OPP_ACC = ["task.oracle.opp_side_accel=true"]  # frozen RL opponents observe theirs (they were trained with it)
ARMS = {"acc": ACC + OPP_ACC, "plain": []}
CAMPAIGN = {
    "n_stages": 9, "frames_per_stage": 300_000_000, "uniform_mix": 1.0, "seed": 0, "start_role": "pursuer",
    "warm_start": "true", "wandb_mode": "disabled",
    "seed_heuristics.pursuer": "[pure_pursuit,apf]", "seed_heuristics.evader": "[hover,circular,flee,weave]",
}


def acc_ckpt(s):
    """F1 'SU + acc' pursuer of seed s (seeds 0-2 are the X4 O2a runs)."""
    x4 = os.path.join(OUT, "X4_oracles", "ckpt", f"O2a_accel_seed{s}.pt")
    return x4 if os.path.isfile(x4) else os.path.join(OUT, "F1_final", "SU_acc", "ckpt", f"seed{s}.pt")


def _match(res, opp_id):
    m = res["matchups"][opp_id]
    return {"J": m["neutralization_rate"], "capture": m["capture_rate"],
            "pursuer_crash": m["outcomes"]["pursuer_crash"], "timeout": m["outcomes"]["timeout"]}


# ------------------------------------------------------------------ F2a
def run_f2a(args):
    root = os.path.join(F2A, "smoke") if args.smoke else F2A
    frames = 2_000_000 if args.smoke else EXPLOIT_FRAMES
    rows = []
    for s in range(1 if args.smoke else 5):
        ck = acc_ckpt(s)
        target = {"id": f"f2a/pursuer:SU_acc_s{s}", "kind": "rl", "checkpoint_path": ck, "metadata": {}}
        ex_ck = train(os.path.join(root, "ckpt", f"exploiter_SU_acc_seed{s}.pt"), [target], frames, role="evader",
                      seed=200 + s, extra=OPP_ACC)
        if not ex_ck:
            if args.smoke:
                raise RuntimeError("F2a smoke: exploiter training failed")
            continue
        ex = {"id": f"f2a/evader:exploit_SU_acc_s{s}", "kind": "rl", "checkpoint_path": ex_ck, "metadata": {}}
        res = evaluate(os.path.join(root, "evals", f"SU_acc_seed{s}.json"), [ex], policy_ckpt=ck, extra=ACC)
        if res:
            rows.append({"seed": s, **_match(res, ex["id"])})
        elif args.smoke:
            raise RuntimeError("F2a smoke: eval failed")
        log_progress(f"F2a seed{s}: {rows[-1] if rows else None}")
    if not args.smoke:
        analyse_f2a(rows)


def analyse_f2a(rows):
    import analysis as an
    with open(os.path.join(D2, "exploit.json")) as f:
        d2 = {r["seed"]: r for r in json.load(f) if r["sampler"] == "S-U"}
    acc = {r["seed"]: 1 - r["J"] for r in rows}
    seeds = sorted(set(acc) & set(d2))
    diff = [acc[s] - d2[s]["exploitability"] for s in seeds]
    out = {"rows": rows, "acc_exploitability": acc, "plain_exploitability": {s: d2[s]["exploitability"] for s in seeds},
           "acc_mean_ci": an.bootstrap_ci(list(acc.values())) if acc else None,
           "paired_diff_acc_minus_plain": an.bootstrap_ci(diff) if diff else None}
    save_json(os.path.join(F2A, "exploit.json"), out)
    lines = ["# F2a_exploit_acc summary", "",
             "Exploitability = 1 - J of the pursuer against a fresh exploiter trained against it (plain-obs evader, "
             "100M frames). acc = F1 'SU + acc' pursuers; plain = D2 'S-U' (same seeds and exploiter seeds).", "",
             "| seed | plain exploitability | acc exploitability | acc capture | acc timeout |", "|---|---|---|---|---|"]
    for r in rows:
        s = r["seed"]
        pl = f"{d2[s]['exploitability']:.3f}" if s in d2 else "-"
        lines.append(f"| {s} | {pl} | {1 - r['J']:.3f} | {r['capture']:.3f} | {r['timeout']:.3f} |")
    if diff:
        m, lo, hi = out["paired_diff_acc_minus_plain"]
        lines += ["", f"acc mean {out['acc_mean_ci'][0]:.3f} [{out['acc_mean_ci'][1]:.3f}, {out['acc_mean_ci'][2]:.3f}]; "
                  f"paired acc - plain {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (n = {len(diff)})"]
    with open(os.path.join(F2A, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ F2b
def campaign_name(arm, smoke=False):
    return f"thesis-F2-{'smoke-' if smoke else ''}{arm}"


def campaign_done(arm, smoke=False):
    return _done(campaign_name(arm, smoke))


def _done(name):
    return os.path.join(common.SELFPLAY_OUT, name, "DONE")


def run_campaign(name, task_overrides, extra_cfg=None, smoke_stages=None, role_overrides=None):
    """Run (or resume) one self-play campaign while holding one Isaac slot. Skips finished campaigns and campaigns
    another lane is running. smoke_stages: tiny smoke version that must produce RL policies of both roles."""
    done = _done(name)
    if os.path.isfile(done):
        return
    lock = done + ".lock"
    if not _acquire_lock(lock):
        log_progress(f"[skip: campaign {name} run by another lane]")
        return
    try:
        cfg = dict(CAMPAIGN, run_name=name, **(extra_cfg or {}))
        if smoke_stages:
            cfg.update(n_stages=smoke_stages, frames_per_stage=2_000_000, num_envs=256)
        cli = [f"{k}={v}" for k, v in cfg.items()]
        cli.append("task_overrides=[" + ",".join(task_overrides) + "]")
        cli.append("eval_extra_overrides=[eval_cutoff_steps=1500]")   # behaviour stats + multi-episode estimator
        for role, ov in (role_overrides or {}).items():
            cli.append(f"role_task_overrides.{role}=[" + ",".join(ov) + "]")
        os.makedirs(common.SELFPLAY_OUT, exist_ok=True)
        slot = _acquire_isaac_slot()     # the campaign runs one Isaac worker at a time: hold one slot throughout
        try:
            log_progress(f"campaign {name} start: {' '.join(cli)}")
            with open(os.path.join(common.SELFPLAY_OUT, name + "_log.txt"), "a") as f:
                rc = subprocess.run([common.PYTHON, os.path.join(REPO, "scripts/selfplay/train_selfplay.py")] + cli,
                                    cwd=os.path.join(REPO, "scripts/selfplay"), stdout=f, stderr=subprocess.STDOUT,
                                    env=_clean_env()).returncode
        finally:
            if os.path.isfile(slot):
                os.remove(slot)
        log_progress(f"campaign {name} exit {rc}")
        if rc != 0:
            if smoke_stages:
                raise RuntimeError(f"smoke campaign {name}: exit {rc}")
            return
        if smoke_stages:
            common.CAMPAIGNS[name] = name
            n = (len(common.rl_policies(name, "pursuer")), len(common.rl_policies(name, "evader")))
            if min(n) < 1:
                raise RuntimeError(f"smoke campaign {name}: expected RL policies of both roles, got {n}")
        open(done, "w").close()
    finally:
        if os.path.isfile(lock):
            os.remove(lock)


def run_f2b(args):
    for arm in ARMS:
        run_campaign(campaign_name(arm, args.smoke), ARMS[arm], smoke_stages=2 if args.smoke else None)


# ------------------------------------------------------------------ F2c
def campaign_policies(arm, role, smoke=False):
    tag = f"F2{arm}"
    common.CAMPAIGNS[tag] = campaign_name(arm, smoke)
    return common.rl_policies(tag, role)


def run_f2c(args):
    while not all(os.path.isfile(campaign_done(a, args.smoke)) for a in ARMS):
        time.sleep(60)
    root = os.path.join(F2C, "smoke") if args.smoke else F2C
    split = load_split()
    groups = {"train": split["train"], "test": test_set(split)}
    pursuers = {a: campaign_policies(a, "pursuer", args.smoke) for a in ARMS}
    evaders = {a: campaign_policies(a, "evader", args.smoke) for a in ARMS}
    # Cross-play: every campaign pursuer vs every campaign evader (one eval job per opponent campaign, since the
    # frozen opponents' observation layout depends on their campaign).
    for pa in ARMS:
        for p in pursuers[pa]:
            for ea in ARMS:
                evaluate(os.path.join(root, "evals", f"{pa}_P{p['metadata']['stage']}_vs_{ea}.json"), evaders[ea],
                         policy_ckpt=p["checkpoint_path"], extra=(ACC if pa == "acc" else []) + (OPP_ACC if ea == "acc" else []))
    # Final pursuers vs the thesis pools and a fresh exploiter.
    frames = 2_000_000 if args.smoke else EXPLOIT_FRAMES
    for pa in ARMS:
        p = pursuers[pa][-1]
        for g, opps in groups.items():
            evaluate(os.path.join(root, "evals", f"{pa}_final_{g}.json"), opps, policy_ckpt=p["checkpoint_path"],
                     extra=ACC if pa == "acc" else [])
        target = {"id": f"f2c/pursuer:{pa}_final", "kind": "rl", "checkpoint_path": p["checkpoint_path"], "metadata": {}}
        ex_ck = train(os.path.join(root, "ckpt", f"exploiter_{pa}_final.pt"), [target], frames, role="evader", seed=400,
                      extra=OPP_ACC if pa == "acc" else [])
        if ex_ck:
            ex = {"id": f"f2c/evader:exploit_{pa}_final", "kind": "rl", "checkpoint_path": ex_ck, "metadata": {}}
            evaluate(os.path.join(root, "evals", f"{pa}_final_vs_exploiter.json"), [ex], policy_ckpt=p["checkpoint_path"],
                     extra=ACC if pa == "acc" else [])
    analyse_f2c(root, pursuers, evaders, split, groups["test"])


def _load(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)["matchups"]


def analyse_f2c(root, pursuers, evaders, split, test):
    P = [(a, p) for a in ARMS for p in pursuers[a]]
    E = [(a, e) for a in ARMS for e in evaders[a]]
    M = np.full((len(P), len(E)), np.nan)
    for i, (pa, p) in enumerate(P):
        for j, (ea, e) in enumerate(E):
            m = _load(os.path.join(root, "evals", f"{pa}_P{p['metadata']['stage']}_vs_{ea}.json")).get(e["id"])
            if m:
                M[i, j] = m["neutralization_rate"]
    plabels = [f"{a} P{p['metadata']['stage']}" for a, p in P]
    elabels = [f"{a} E{e['metadata']['stage']}" for a, e in E]
    out = {"pursuers": plabels, "evaders": elabels, "J": M.tolist(), "final": {}}
    idx = {"train": [e["id"] for e in split["train"]], "heldout_rl": [e["id"] for e in split["heldout_rl"]],
           "heldout_scripted": [e["id"] for e in split["heldout_scripted"]],
           "x3_exploiters": [e["id"] for e in test if e["id"].startswith("exploit/")]}
    for pa in ARMS:
        i = max(k for k, (a, _) in enumerate(P) if a == pa)
        row = {f"vs_{ea}_campaign_evaders": float(np.nanmean([M[i, j] for j, (a, _) in enumerate(E) if a == ea]))
               for ea in ARMS}
        J = {k: v["neutralization_rate"] for g in ("train", "test") for k, v in _load(
            os.path.join(root, "evals", f"{pa}_final_{g}.json")).items()}
        for g, ids in idx.items():
            v = [J[i_] for i_ in ids if i_ in J]
            if v:
                row[g] = float(np.mean(v))
                row[g + "_min"] = float(np.min(v))
        ex = _load(os.path.join(root, "evals", f"{pa}_final_vs_exploiter.json"))
        if ex:
            m = next(iter(ex.values()))
            row["exploitability"] = 1 - m["neutralization_rate"]
            row["capture_vs_exploiter"] = m["capture_rate"]
        out["final"][pa] = row
    save_json(os.path.join(root, "results.json"), out)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(1 + 0.5 * len(E), 1 + 0.45 * len(P)))
        im = ax.imshow(M, vmin=0, vmax=1, cmap="RdBu")
        ax.set_xticks(range(len(E)), elabels, rotation=60, ha="right")
        ax.set_yticks(range(len(P)), plabels)
        ax.set_xlabel("evader (campaign, stage)")
        ax.set_ylabel("pursuer (campaign, stage)")
        fig.colorbar(im, label="J (pursuer neutralisation rate)")
        fig.tight_layout()
        fig.savefig(os.path.join(root, "fig_crossplay.png"), dpi=150)
    except Exception as exc:  # the figure is optional
        print("[F2c figure failed]", exc)
    keys = sorted({k for r in out["final"].values() for k in r})
    lines = ["# F2c_final_eval summary", "", "1 seed per campaign: descriptive only.", "",
             "Final pursuer of each campaign (J unless noted):", "", "| metric | " + " | ".join(ARMS) + " |",
             "|---" * (len(ARMS) + 1) + "|"]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(f"{out['final'][a].get(k, float('nan')):.3f}" for a in ARMS) + " |")
    lines += ["", "Cross-play J (rows pursuers, columns evaders):", "", "| | " + " | ".join(elabels) + " |",
              "|---" * (len(E) + 1) + "|"]
    for i, lab in enumerate(plabels):
        lines.append(f"| {lab} | " + " | ".join("-" if np.isnan(x) else f"{x:.2f}" for x in M[i]) + " |")
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
