"""F4: learned opponent-acceleration estimator in full self-play, realistic sensing (docs/research_plan.md, Phase F,
pre-registered 10-28 21:30).

The pursuer does not observe the evader's velocity (positions only) and trains and is evaluated with sensor noise
(observation.include_noise). Arms: nf = + the A3a learned acceleration estimator (frozen, noise-augmented basis,
H = 10); none = no estimator. The evader observes the full clean state (as in the reference papers). Uniform
sampling, 9 stages x 300M frames, 3 seeds per arm (campaign seed 0 / 100 / 200).
F4c: final pursuer of each campaign vs the thesis train pool / test set / all F4 campaign evaders (noisy, as
trained), vs the test set without noise, and vs one fresh exploiter.
"""

import json
import os
import time

import numpy as np

import common
from a_adapt import HISTORY, test_set
from common import OUT, evaluate, log_progress, save_json, train
from d_sampling import EXPLOIT_FRAMES
from f2_final import _done, run_campaign
from x3_gap import load_split

F4C = os.path.join(OUT, "F4_estimator_sp")
SEEDS = [0, 100, 200]
NOISY = "task.observation.include_noise=true"


def _adapt_common():
    with open(os.path.join(OUT, "A3a_posonly", "choice.json")) as f:
        c = json.load(f)
    return ["task.adapt.kind=nf", "task.adapt.z_mode=pos", "task.adapt.target=dvel", "task.adapt.freeze=true",
            f"task.adapt.horizon={c['H_star']}", f"task.adapt.history={HISTORY}", f"task.adapt.model_path={c['model_path']}"]


def arms():
    """{arm: (pursuer-worker overrides, evader-worker overrides)}; the evader's frozen pursuers see the same inputs."""
    a = _adapt_common()
    return {
        "nf": (["task.observation.agent_rel_vel=false", NOISY, "task.adapt.enabled=true"] + a,
               ["task.observation.opp_drop_rel_vel=true", "task.adapt.opp_side=true"] + a),
        "none": (["task.observation.agent_rel_vel=false", NOISY],
                 ["task.observation.opp_drop_rel_vel=true"]),
    }


def name(arm, k, smoke=False):
    return f"thesis-F4-{'smoke-' if smoke else ''}{arm}-s{k}"


def run_f4(args):
    A = arms()
    if args.smoke:
        p, e = A["nf"]
        run_campaign(name("nf", 0, True), [], {}, smoke_stages=3, role_overrides={"pursuer": p, "evader": e})
        return
    for k in range(len(SEEDS)):
        for arm, (p, e) in A.items():
            run_campaign(name(arm, k), [], {"seed": SEEDS[k]}, role_overrides={"pursuer": p, "evader": e})


def _policies(campaign, role):
    common.CAMPAIGNS[campaign] = campaign
    return common.rl_policies(campaign, role)


def run_f4c(args):
    A = arms()
    camps = [(a, k, name(a, k)) for k in range(len(SEEDS)) for a in A]
    while not all(os.path.isfile(_done(c)) for _, _, c in camps):
        time.sleep(60)
    split = load_split()
    test = test_set(split)
    all_ev = [e for _, _, c in camps for e in _policies(c, "evader")]
    for i, (a, k, c) in enumerate(camps):
        pov, eov = A[a]
        p = _policies(c, "pursuer")[-1]
        ck = p["checkpoint_path"]
        for g, opps in (("train", split["train"]), ("test", test), ("campaign_evaders", all_ev)):
            evaluate(os.path.join(F4C, "evals", f"{c}_final_{g}.json"), opps, policy_ckpt=ck, extra=pov)
        evaluate(os.path.join(F4C, "evals", f"{c}_final_test_clean.json"), test, policy_ckpt=ck,
                 extra=[o for o in pov if o != NOISY])
        target = {"id": f"f4/pursuer:{c}", "kind": "rl", "checkpoint_path": ck, "metadata": {}}
        ex_ck = train(os.path.join(F4C, "ckpt", f"exploiter_{c}.pt"), [target], EXPLOIT_FRAMES, role="evader",
                      seed=600 + i, extra=eov)
        if ex_ck:
            ex = {"id": f"f4/evader:exploit_{c}", "kind": "rl", "checkpoint_path": ex_ck, "metadata": {}}
            evaluate(os.path.join(F4C, "evals", f"{c}_final_vs_exploiter.json"), [ex], policy_ckpt=ck, extra=pov)
        log_progress(f"F4c {c} done")
    analyse_f4(camps, split, test)


def _J(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return {k: v["neutralization_rate"] for k, v in json.load(f)["matchups"].items()}


def analyse_f4(camps, split, test):
    import analysis as an
    rows = {}
    for a, k, c in camps:
        J = {**_J(os.path.join(F4C, "evals", f"{c}_final_train.json")),
             **_J(os.path.join(F4C, "evals", f"{c}_final_test.json"))}
        Jclean = _J(os.path.join(F4C, "evals", f"{c}_final_test_clean.json"))
        Jc = _J(os.path.join(F4C, "evals", f"{c}_final_campaign_evaders.json"))
        if not J or not Jc:
            continue
        mean = lambda d, ids: float(np.mean([d[i] for i in ids if i in d]))
        ids = {"train": [e["id"] for e in split["train"]], "heldout_rl": [e["id"] for e in split["heldout_rl"]],
               "heldout_scripted": [e["id"] for e in split["heldout_scripted"]],
               "x3_exploiters": [e["id"] for e in test if e["id"].startswith("exploit/")]}
        r = {g: mean(J, v) for g, v in ids.items()}
        if Jclean:
            r["heldout_rl_clean"] = mean(Jclean, ids["heldout_rl"])
            r["x3_exploiters_clean"] = mean(Jclean, ids["x3_exploiters"])
        r["other_campaign_evaders"] = float(np.mean([v for kk, v in Jc.items() if not kk.startswith(c + "/")]))
        r["own_campaign_evaders"] = float(np.mean([v for kk, v in Jc.items() if kk.startswith(c + "/")]))
        ex = _J(os.path.join(F4C, "evals", f"{c}_final_vs_exploiter.json"))
        if ex:
            r["exploitability"] = 1 - next(iter(ex.values()))
        rows[(a, k)] = r
    metrics = sorted({m for r in rows.values() for m in r})
    agg = {a: {m: dict(zip(("mean", "lo", "hi"), an.bootstrap_ci([rows[(a, k)][m] for k in range(len(SEEDS))
                                                                    if (a, k) in rows and m in rows[(a, k)]])))
               for m in metrics if any((a, k) in rows and m in rows[(a, k)] for k in range(len(SEEDS)))}
           for a in ("nf", "none")}
    diff = {}
    for m in metrics:
        d = [rows[("nf", k)][m] - rows[("none", k)][m] for k in range(len(SEEDS))
             if ("nf", k) in rows and ("none", k) in rows and m in rows[("nf", k)] and m in rows[("none", k)]]
        if d:
            diff[m] = dict(zip(("mean", "lo", "hi"), an.bootstrap_ci(d)), per_seed=d)
    save_json(os.path.join(F4C, "results.json"), {"per_campaign": {f"{a}/{k}": r for (a, k), r in rows.items()},
                                                  "aggregate": agg, "paired_nf_minus_none": diff})
    lines = ["# F4_estimator_sp summary", "", "Pursuer without evader velocity, trained and evaluated with sensor "
             "noise (except *_clean). nf = + learned acceleration estimator. Mean over 3 campaign seeds [95 % CI].",
             "", "| metric | none | nf | nf - none (paired) |", "|---|---|---|---|"]
    f = lambda x: f"{x['mean']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]"
    for m in metrics:
        lines.append(f"| {m} | {f(agg['none'][m]) if m in agg['none'] else '-'} | "
                     f"{f(agg['nf'][m]) if m in agg['nf'] else '-'} | "
                     + (f"{diff[m]['mean']:+.3f} [{diff[m]['lo']:+.3f}, {diff[m]['hi']:+.3f}]" if m in diff else "-") + " |")
    with open(os.path.join(F4C, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
