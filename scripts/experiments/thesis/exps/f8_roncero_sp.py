"""F8: self-play under Sanchez Roncero et al. 2026's reward and episode endings, with Hedge on both roles
(docs/research_plan.md, F8_roncero_sp; pre-registered 2026-11-09 before any F8 run).

Question: do evaders still beat self-play pursuers by luring them into the floor when the game uses Roncero's rules
(Eqs. 8-11: no reward for the opponent crashing, crash / out-of-arena / timeout are truncations, capture bonus scaled
down at high closing speed)? Our drone, observations, arena and 0.10 m capture radius; Hedge (gamma 0.7, eta 2) on
both roles; 26 stages x 50M frames = Roncero's 13 generations x 50M per agent (one generation trains both roles; user,
11-09 amendment 1); campaign seeds 0 / 100.
F8c: final pursuers of both F8 campaigns, evaluated under the thesis rules (as every other thesis pursuer) vs train
pool, test set, the F3 campaigns' evaders and the F8 evaders, plus two fresh exploiters each (100M frames): one under
the thesis rules (luring pays) and one under Roncero's rules (his game). The F3 hedge-both s0 / s1 final pursuers get a
Roncero-rules exploiter and an eval vs the F8 evaders too. Every J at 0.10 m (primary) and 0.15 m (re-score).
"""

import json
import os
import time

import numpy as np

import common
from a_adapt import test_set
from common import OUT, evaluate, log_progress, save_json, train
from d_sampling import EXPLOIT_FRAMES
from f2_final import _done, run_campaign
from f3_sampler_sp import F3C, SAMPLERS, _policies
from f3_sampler_sp import name as f3_name
from rescore import rates_at
from x3_gap import load_split

F8 = os.path.join(OUT, "F8_roncero_sp")
RULES = ["task.reward_mode=roncero", "algo.gae_cut_at_truncation=true"]
SEEDS = (0, 100)
FRAMES = 50_000_000
STAGES = 26                     # 13 generations (pursuer + evader each), as Roncero's K = 13
RADII = (0.10, 0.15)
F3_ARMS = ("uniform", "hedge", "qd_dpp", "pfsp")
METRICS = ["heldout_rl", "heldout_worst_cluster", "other_f3_evaders", "other_f8_evaders", "own_campaign_evaders",
           "train", "heldout_scripted", "heldout_scripted_pursuer_crash", "x3_exploiters",
           "exploitability_thesis", "exploiter_thesis_pursuer_crash",
           "exploitability_roncero", "exploiter_roncero_pursuer_crash", "exploiter_roncero_timeout"]


def name(k, smoke=False):
    return "thesis-F8-smoke-roncero" if smoke else f"thesis-F8-roncero-s{SEEDS[k]}"


def arm_cfg(k):
    return dict(SAMPLERS["hedge"], seed=SEEDS[k], frames_per_stage=FRAMES, n_stages=STAGES, max_total_stages=STAGES)


def run_f8(args):
    smoke = name(0, True)
    run_campaign(smoke, RULES, dict(SAMPLERS["hedge"]), smoke_stages=2)
    d = _done(smoke)
    while not os.path.isfile(d) and os.path.isfile(d + ".lock"):     # the other lane may be running it
        time.sleep(60)
    if not os.path.isfile(d):
        raise RuntimeError("F8 smoke did not pass; campaigns not started")
    if args.smoke:
        return
    for k in range(len(SEEDS)):
        run_campaign(name(k), RULES, arm_cfg(k))


def _f3_hedge():
    return [f3_name("hedge", k) for k in range(2)]


def pursuers():
    """[(label, campaign, kind)]: kind f8 = new pursuer (all evals), f3 = F3 hedge pursuer (Roncero exploiter + F8
    evaders only; the rest reused from F3c)."""
    return ([(f"roncero/{k}", name(k), "f8") for k in range(len(SEEDS))] +
            [(f"hedge/{k}", c, "f3") for k, c in enumerate(_f3_hedge())])


def _src(c, g):
    p = os.path.join(F8, "evals", f"{c}_final_{g}.json")
    if c in _f3_hedge() and g in ("train", "test", "f3_evaders", "vs_exploiter_thesis"):
        g3 = {"f3_evaders": "campaign_evaders", "vs_exploiter_thesis": "vs_exploiter"}.get(g, g)
        return os.path.join(F3C, "evals", f"{c}_final_{g3}.json")
    return p


def _exploiter(c, rules, i):
    tag = "roncero" if rules else "thesis"
    path = os.path.join(F8, "ckpt", f"exploiter_{tag}_{c}.pt")
    ck = _policies(c, "pursuer")[-1]["checkpoint_path"]
    target = {"id": f"f8/pursuer:{c}", "kind": "rl", "checkpoint_path": ck, "metadata": {}}
    train(path, [target], EXPLOIT_FRAMES, role="evader", seed=(830 if rules else 820) + i, extra=list(rules))
    return ([{"id": f"f8/evader:exploit_{tag}_{c}", "kind": "rl", "checkpoint_path": path, "metadata": {}}]
            if os.path.isfile(path) else [])


def run_f8c(args):
    for k in range(len(SEEDS)):
        d = _done(name(k))
        while not os.path.isfile(d):
            if not os.path.isfile(d + ".lock"):
                raise RuntimeError(f"F8c: campaign {name(k)} is neither finished nor running")
            time.sleep(60)
    split = load_split()
    test = test_set(split)
    f3_ev = [e for s in F3_ARMS for k in range(2) for e in _policies(f3_name(s, k), "evader")]
    f8_ev = [e for k in range(len(SEEDS)) for e in _policies(name(k), "evader")]
    for i, (lab, c, kind) in enumerate(pursuers()):
        ck = _policies(c, "pursuer")[-1]["checkpoint_path"]
        jobs = [("f8_evaders", f8_ev, None), ("vs_exploiter_roncero", _exploiter(c, RULES, i), RULES)]
        if kind == "f8":
            jobs = [("train", split["train"], None), ("test", test, None), ("f3_evaders", f3_ev, None),
                    ("vs_exploiter_thesis", _exploiter(c, [], i), None)] + jobs
        for g, opps, extra in jobs:
            if opps and not os.path.isfile(_src(c, g)):
                evaluate(_src(c, g), opps, policy_ckpt=ck, extra=list(extra) if extra else None)
        log_progress(f"F8c {c} done")
    groups = {"f8": ("train", "test", "f3_evaders", "f8_evaders", "vs_exploiter_thesis", "vs_exploiter_roncero"),
              "f3": ("f8_evaders", "vs_exploiter_roncero")}
    missing = [_src(c, g) for _, c, kind in pursuers() for g in groups[kind] if not os.path.isfile(_src(c, g))]
    if missing:                                 # the other lane is still producing them and analyses after
        log_progress(f"F8c: {len(missing)} eval files missing on this lane, analysis left to the other lane")
        return
    analyse_f8(split, test)


def _metrics(c, radius, split, test, labels):
    rate = lambda g, key="neutralization_rate": rates_at(_src(c, g), key, radius)
    J = {**rate("train"), **rate("test")}
    no_s1 = lambda d: {k: v for k, v in d.items() if not k.endswith(":s1")}
    J3, J8 = no_s1(rate("f3_evaders")), no_s1(rate("f8_evaders"))
    ids = lambda opps: [J[e["id"]] for e in opps if e["id"] in J]
    r = {"train": ids(split["train"]), "heldout_rl": ids(split["heldout_rl"]),
         "heldout_scripted": ids(split["heldout_scripted"]),
         "x3_exploiters": [J[e["id"]] for e in test if e["id"].startswith("exploit/") and e["id"] in J],
         "other_f3_evaders": [v for k, v in J3.items() if not k.startswith(c + "/")],
         "other_f8_evaders": [v for k, v in J8.items() if not k.startswith(c + "/")],
         "own_campaign_evaders": [v for k, v in {**J3, **J8}.items() if k.startswith(c + "/")]}
    r = {m: float(np.mean(v)) for m, v in r.items() if v}
    by_c = {}
    for e in split["heldout_rl"] + split["heldout_scripted"]:
        if e["id"] in J and e["id"] in labels:
            by_c.setdefault(labels[e["id"]], []).append(J[e["id"]])
    if by_c:
        r["heldout_worst_cluster"] = float(min(np.mean(v) for v in by_c.values()))
    pc = rate("test", "pursuer_crash")
    if pc:
        r["heldout_scripted_pursuer_crash"] = float(np.mean([pc[e["id"]] for e in split["heldout_scripted"]
                                                              if e["id"] in pc]))
    for tag in ("thesis", "roncero"):
        g = f"vs_exploiter_{tag}"
        ex = rate(g)
        if ex:
            r[f"exploitability_{tag}"] = 1 - float(next(iter(ex.values())))
            r[f"exploiter_{tag}_pursuer_crash"] = float(next(iter(rate(g, "pursuer_crash").values())))
            if tag == "roncero":
                r["exploiter_roncero_timeout"] = float(next(iter(rate(g, "timeout").values())))
    return r


def lure_rates(c):
    """In-campaign: pursuer crash rate that every evader stage from 3 on causes in the stage >= 2 RL pursuers (its
    own training opponents), from the stage evals (training radius, no per-episode files)."""
    pool = {p["id"]: os.path.basename(p["checkpoint_path"] or "")
            for p in json.load(open(os.path.join(common.SELFPLAY_OUT, c, "db", "pursuer_pool.json")))["policies"]}
    out = {}
    stages = [s for s in range(3, 2 * STAGES, 2)
              if os.path.isfile(os.path.join(common.SELFPLAY_OUT, c, "stages", f"stage{s}_evader_eval_out.json"))]
    for s in stages:
        m = json.load(open(os.path.join(common.SELFPLAY_OUT, c, "stages", f"stage{s}_evader_eval_out.json")))["matchups"]
        out[s] = {pool[k].split("_")[1].split(".")[0]: v["outcomes"]["pursuer_crash"] for k, v in m.items()
                  if pool.get(k, "").startswith("pursuer_stage") and int(pool[k].split("stage")[1].split(".")[0]) >= 2}
    return out


def analyse_f8(split, test):
    from f5_outcomes import stage_table
    with open(os.path.join(OUT, "X2_descriptors", "clusters.json")) as f:
        labels = json.load(f)["labels"]
    labs = [lab for lab, _, _ in pursuers()]
    res = {f"r{rad:.2f}": {lab: _metrics(c, rad, split, test, labels) for lab, c, _ in pursuers()} for rad in RADII}
    lures = {c: lure_rates(c) for _, c, _ in pursuers()}
    diffs = {k: {f"roncero/{i} - hedge/{i}": {m: v[f"roncero/{i}"][m] - v[f"hedge/{i}"][m] for m in METRICS
                                               if m in v[f"roncero/{i}"] and m in v[f"hedge/{i}"]}
                 for i in range(len(SEEDS))} for k, v in res.items()}
    save_json(os.path.join(F8, "results.json"), {"metrics": res, "differences": diffs, "lure_rates": lures})
    lines = ["# F8_roncero_sp summary", "",
             "Final pursuers (stage 8). roncero/k = F8 campaign seed k (Roncero's reward and endings, Hedge both roles, "
             "9 x 50M); hedge/k = F3 hedge-both seed k (thesis rules, Hedge both roles, 9 x 300M). All evals clean "
             "under the thesis rules except vs_exploiter_roncero (Roncero's rules). exploitability_<rules> = 1 - J vs a "
             "fresh exploiter trained under those rules; for hedge/k the thesis-rules exploiter is F3c's. other_f3 / "
             "other_f8 / own evaders exclude stage-1 evaders. J at 0.15 m is the exact re-score (eval-only). 2 seeds, "
             "descriptive; F8 = 26 stages x 50M (13 generations), F3 = 9 stages x 300M.", ""]
    for k, v in res.items():
        pairs = list(diffs[k])
        lines += [f"## Capture radius {k[1:]} m", "", "| metric | " + " | ".join(labs) + " | " + " | ".join(pairs)
                  + " |", "|---" * (1 + len(labs) + len(pairs)) + "|"]
        for m in METRICS:
            cells = [f"{v[a][m]:.3f}" if m in v[a] else "-" for a in labs]
            cells += [f"{diffs[k][p][m]:+.3f}" if m in diffs[k][p] else "-" for p in pairs]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## In-campaign lure rates (pursuer crash caused by each evader stage in the stage >= 2 pursuers)", "",
              "Stages 3 / 5 / 7 are comparable across F8 and F3 (F3 has 9 stages, F8 26).", "",
              "| campaign | evader stage | mean | min - max | n pursuers |", "|---|---|---|---|---|"]
    for c, lr in lures.items():
        for s, d in sorted(lr.items()):
            if d:
                v = list(d.values())
                lines.append(f"| {c} | {s} | {np.mean(v):.2f} | {min(v):.2f} - {max(v):.2f} | {len(v)} |")
    lines += ["", "## Inside the F8 campaigns (each stage's eval vs its training opponents, Roncero's rules)", ""]
    for k in range(len(SEEDS)):
        lines += stage_table(name(k), STAGES)
    with open(os.path.join(F8, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log_progress("F8c analysis done")
