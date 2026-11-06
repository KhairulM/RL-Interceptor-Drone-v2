"""F7: pursuer floor penalty + 0.15 m capture radius in self-play (docs/research_plan.md, pre-registered 11-06 19:30).

Two campaigns, F3 uniform protocol (plain obs, uniform sampler on both roles, 9 x 300M, seed 0), differing from
thesis-F2-plain (the baseline, uniform_plain seed 0) only in: floor_r15 = 0.15 m capture radius in training and
in-campaign evals + pursuer floor penalty (reward_floor_weight 1.0); r15 = the 0.15 m radius only.
F7c: final pursuers of F2-plain, r15 and floor_r15, clean, at the standard 0.10 m eval radius with per-episode files,
vs train pool, test set, the F5c campaign evaders, both F7 campaigns' evaders, and a fresh exploiter per pursuer trained
under the 0.15 m rules. Every J is reported at 0.15 m (primary; exact re-score, rescore.py) and 0.10 m.
"""

import json
import os
import time

import numpy as np

from a_adapt import test_set
from common import OUT, evaluate, log_progress, save_json, train
from d_sampling import EXPLOIT_FRAMES
from f2_final import _done, run_campaign
from f3_sampler_sp import SAMPLERS, _policies
import f5_acc_sp as f5
from rescore import rates_at
from x3_gap import load_split

F7 = os.path.join(OUT, "F7_floor_sp")
R15 = ["task.success_radius_init=0.15", "task.success_radius_end=0.15", "task.success_radius_eval=0.15"]
FLOOR = ["task.reward_floor_weight=1.0"]
ARMS = {"floor_r15": R15 + FLOOR, "r15": R15}
SEED = 0
BASE = "thesis-F2-plain"
RADII = (0.15, 0.10)
GROUPS = ("train", "test", "campaign_evaders", "campaign_evaders_s300", "f7_evaders", "vs_exploiter_r15")
METRICS = ["exploitability", "exploiter_pursuer_crash", "heldout_rl", "heldout_worst_cluster", "other_campaign_evaders",
           "other_f7_evaders", "own_campaign_evaders", "train", "heldout_scripted", "heldout_scripted_pursuer_crash",
           "x3_exploiters"]
CONTRASTS = [("floor_r15", "r15"), ("r15", "plain"), ("floor_r15", "plain")]


def name(arm, smoke=False):
    return f"thesis-F7-smoke-{arm}" if smoke else f"thesis-F7-{arm}-s{SEED}"


def pursuers():
    """[(arm, campaign, exploiter checkpoint)]; the F2-plain exploiter is new (0.15 m rules)."""
    return [(a, c, os.path.join(F7, "ckpt", f"exploiter_r15_{c}.pt"))
            for a, c in [("plain", BASE)] + [(a, name(a)) for a in ARMS]]


def run_f7(args):
    smoke = name("floor_r15", True)
    run_campaign(smoke, ARMS["floor_r15"], dict(SAMPLERS["uniform"]), smoke_stages=2)
    d = _done(smoke)
    while not os.path.isfile(d) and os.path.isfile(d + ".lock"):     # the other lane may be running it
        time.sleep(60)
    if not os.path.isfile(d):
        raise RuntimeError("F7 smoke did not pass; campaigns not started")
    if args.smoke:
        return
    for arm, ov in ARMS.items():
        run_campaign(name(arm), ov, dict(SAMPLERS["uniform"], seed=SEED))


def _src(c, g):
    """Eval file of pursuer c vs group g: F7's own, else F5c's (F2-plain's train / test / campaign evals are reused)."""
    p = os.path.join(F7, "evals", f"{c}_clean_{g}.json")
    if g in ("train", "test", "campaign_evaders", "campaign_evaders_s300") and not os.path.isfile(p):
        q = os.path.join(f5.F5C, "evals", f"{c}_clean_{g}.json")
        if os.path.isfile(q):
            return q
    return p


def run_f7c(args):
    for arm in ARMS:
        d = _done(name(arm))
        while not os.path.isfile(d):
            if not os.path.isfile(d + ".lock"):
                raise RuntimeError(f"F7c: campaign {name(arm)} is neither finished nor running")
            time.sleep(60)
    split = load_split()
    test = test_set(split)
    s3 = f5.seed3_campaigns()
    all_ev = [e for a, _, c, _, _ in f5.pursuers() if a != "uniform_fd" and a not in f5.PAUSED_ARMS and c not in s3
              for e in _policies(c, "evader")]                       # the F5c campaign-evader sets
    ev_s300 = [e for c in s3 for e in _policies(c, "evader")]
    f7_ev = [e for a in ARMS for e in _policies(name(a), "evader")]
    for i, (a, c, ex_path) in enumerate(pursuers()):
        ck = _policies(c, "pursuer")[-1]["checkpoint_path"]
        target = {"id": f"f7/pursuer:{c}", "kind": "rl", "checkpoint_path": ck, "metadata": {}}
        train(ex_path, [target], EXPLOIT_FRAMES, role="evader", seed=800 + i, extra=list(R15))
        ex = ([{"id": f"f7/evader:exploit_r15_{c}", "kind": "rl", "checkpoint_path": ex_path, "metadata": {}}]
              if os.path.isfile(ex_path) else [])
        for g, opps in (("train", split["train"]), ("test", test), ("campaign_evaders", all_ev),
                        ("campaign_evaders_s300", ev_s300), ("f7_evaders", f7_ev), ("vs_exploiter_r15", ex)):
            if opps and not os.path.isfile(_src(c, g)):
                evaluate(_src(c, g), opps, policy_ckpt=ck)
        log_progress(f"F7c {c} done")
    missing = [_src(c, g) for _, c, _ in pursuers() for g in GROUPS if not os.path.isfile(_src(c, g))]
    if missing:                                 # the other lane is still producing them and analyses after
        log_progress(f"F7c: {len(missing)} eval files missing on this lane, analysis left to the other lane")
        return
    analyse_f7(split, test)


def _metrics(c, radius, split, test, labels):
    rate = lambda g, key="neutralization_rate": rates_at(_src(c, g), key, radius)
    J = {**rate("train"), **rate("test")}
    Jc = {k: v for k, v in {**rate("campaign_evaders"), **rate("campaign_evaders_s300")}.items() if not k.endswith(":s1")}
    Jf = {k: v for k, v in rate("f7_evaders").items() if not k.endswith(":s1")}
    ids = lambda opps: [J[e["id"]] for e in opps if e["id"] in J]
    own = [v for k, v in {**Jc, **Jf}.items() if k.startswith(c + "/")]
    r = {"train": ids(split["train"]), "heldout_rl": ids(split["heldout_rl"]),
         "heldout_scripted": ids(split["heldout_scripted"]),
         "x3_exploiters": [J[e["id"]] for e in test if e["id"].startswith("exploit/") and e["id"] in J],
         "other_campaign_evaders": [v for k, v in Jc.items() if not k.startswith(c + "/")],
         "other_f7_evaders": [v for k, v in Jf.items() if not k.startswith(c + "/")], "own_campaign_evaders": own}
    r = {m: float(np.mean(v)) for m, v in r.items() if v}
    by_c = {}
    for e in split["heldout_rl"] + split["heldout_scripted"]:
        if e["id"] in J and e["id"] in labels:
            by_c.setdefault(labels[e["id"]], []).append(J[e["id"]])
    r["heldout_worst_cluster"] = float(min(np.mean(v) for v in by_c.values()))
    pc = rate("test", "pursuer_crash")
    r["heldout_scripted_pursuer_crash"] = float(np.mean([pc[e["id"]] for e in split["heldout_scripted"] if e["id"] in pc]))
    ex, exc = rate("vs_exploiter_r15"), rate("vs_exploiter_r15", "pursuer_crash")
    if ex:
        r["exploitability"] = 1 - float(next(iter(ex.values())))
        r["exploiter_pursuer_crash"] = float(next(iter(exc.values())))
    return r


def analyse_f7(split, test):
    from f5_outcomes import stage_table
    with open(os.path.join(OUT, "X2_descriptors", "clusters.json")) as f:
        labels = json.load(f)["labels"]
    res = {f"r{rad:.2f}": {a: _metrics(c, rad, split, test, labels) for a, c, _ in pursuers()} for rad in RADII}
    diffs = {k: {f"{x} - {y}": {m: v[x][m] - v[y][m] for m in METRICS if m in v[x] and m in v[y]} for x, y in CONTRASTS}
             for k, v in res.items()}
    save_json(os.path.join(F7, "results.json"), {"metrics": res, "differences": diffs})
    arms = [a for a, _, _ in pursuers()]
    lines = ["# F7_floor_sp summary", "",
             "Final pursuers (stage 8), clean, 1 campaign each (seed 0; descriptive). plain = thesis-F2-plain (0.10 m, no "
             "floor penalty); r15 = 0.15 m capture radius in training; floor_r15 = 0.15 m + pursuer floor penalty. "
             "Evaluated at 0.10 m with per-episode files; J at 0.15 m is the exact re-score (`rescore.py`). "
             "exploitability = 1 - J vs a fresh exploiter trained under the 0.15 m rules; *_pursuer_crash = pursuer crash "
             "rate. other / own campaign and F7 evaders exclude stage-1 evaders.", ""]
    for k, v in res.items():
        lines += [f"## Capture radius {k[1:]} m", "", "| metric | " + " | ".join(arms) + " | " +
                  " | ".join(f"{x} - {y}" for x, y in CONTRASTS) + " |", "|---" * (1 + len(arms) + len(CONTRASTS)) + "|"]
        for m in METRICS:
            cells = [f"{v[a][m]:.3f}" if m in v[a] else "-" for a in arms]
            cells += [f"{diffs[k][f'{x} - {y}'][m]:+.3f}" if m in diffs[k][f"{x} - {y}"] else "-" for x, y in CONTRASTS]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Inside the campaigns (each stage's eval vs its training opponents; capture radius 0.15 m)", ""]
    for a in ARMS:
        lines += stage_table(name(a))
    with open(os.path.join(F7, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log_progress("F7c analysis done")
