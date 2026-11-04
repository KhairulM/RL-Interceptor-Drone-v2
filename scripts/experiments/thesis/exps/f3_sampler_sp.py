"""F3: opponent samplers in full self-play (docs/research_plan.md, Phase F, pre-registered 10-28 20:30).

Campaigns as in F2b "plain" (plain obs, 9 stages x 300M). The opponent sampler changes: uniform | hedge (Roncero et
al. 2026 eq. 12, gamma 0.7, eta 2) | qd_dpp (Hedge quality x DPP diversity over behaviour descriptors) | pfsp (hedge
with gamma 0: p ~ c^2, no uniform share; the PFSP core of AgilePE / Gavin).
Amendment 3 (11-03, user, before any F3 result): the sampler is used by BOTH roles, as in the reference papers
("-both" campaigns). The pursuer-only campaigns of amendments 1-2 (thesis-F3-{hedge,qd_dpp,pfsp}-s*) are kept on disk
but not analysed. Seed 0 of every arm first; seed-1 campaigns start only before S1_START_DEADLINE (11-05 freeze).
uniform is the same on both sides already, so uniform seed 0 (F2b "plain") and seed 1 are reused.
Amendment 4 (11-04 14:45, user, before any F3c result): 2 seeds for every arm (qd_dpp-both-s1, pfsp-both-s1 added);
S1_START_DEADLINE moved to 11-04 20:00 and F3c waits for all 8 campaigns (F3c then ends ~11-05 morning).
F3c: final pursuer of every campaign vs the thesis test set, the train pool and all campaigns' RL evaders, plus one
fresh exploiter each; seed-paired comparison of the samplers.
"""

import datetime
import json
import math
import os
import time

import numpy as np

import common
from a_adapt import test_set
from common import MAX_EPISODE_STEPS, OUT, STEP_DT, evaluate, log_progress, save_json, train
from d_sampling import EXPLOIT_FRAMES
from f2_final import _done, run_campaign
from x3_gap import load_split

F3C = os.path.join(OUT, "F3_sampler_sp")
SAMPLERS = {"uniform": {},
            "hedge": {"pursuer_sampler": "hedge", "evader_sampler": "hedge"},
            "qd_dpp": {"pursuer_sampler": "qd_dpp", "evader_sampler": "qd_dpp"},
            "pfsp": {"pursuer_sampler": "hedge", "evader_sampler": "hedge", "hedge_gamma": 0.0}}
SEEDS = [0, 100]
S1_START_DEADLINE = datetime.datetime(2026, 11, 4, 20, 0)   # amendment 4: a campaign takes ~12.7 h, F3c ends ~11-05 AM


def name(sampler, k):
    if sampler == "uniform":
        return "thesis-F2-plain" if k == 0 else f"thesis-F3-uniform-s{k}"
    return f"thesis-F3-{sampler}-both-s{k}"


def _running(c):
    """True while another driver holds the campaign lock with a live PID."""
    lock = _done(c) + ".lock"
    try:
        pid = int(open(lock).read().strip() or 0)
    except (OSError, ValueError):
        return False
    return bool(pid) and common._pid_alive(pid)


def run_f3(args):
    if args.smoke:
        n = "thesis-F3-smoke-qd_dpp-both"
        run_campaign(n, [], dict(SAMPLERS["qd_dpp"]), smoke_stages=5)
        for stage, role in ((3, "evader"), (4, "pursuer")):     # first stages with 2 RL opponents per role
            with open(os.path.join(common.SELFPLAY_OUT, n, "stages", f"stage{stage}_{role}_train_opp.json")) as f:
                spec = json.load(f)
            probs = spec["probs"]
            if len(probs) != 2 or abs(sum(probs.values()) - 1) > 1e-6 or any(
                    p["kind"] != "rl" for p in spec["policies"]):
                raise RuntimeError(f"F3 smoke: unexpected stage-{stage} {role} probs {probs}")
            log_progress(f"F3 smoke ok: stage-{stage} {role} probs {probs}")
        return
    for k in range(len(SEEDS)):                 # seed-major order keeps partial results balanced across samplers
        for s, extra in SAMPLERS.items():
            c = name(s, k)
            if k > 0 and not os.path.isfile(_done(c)) and datetime.datetime.now() > S1_START_DEADLINE:
                log_progress(f"[skip: {c} not started, past the seed-1 start deadline {S1_START_DEADLINE}]")
                continue
            run_campaign(c, [], dict(extra, seed=SEEDS[k]))


# The pre-registered protocol is 9 stages (0-8). Campaigns resumed after the 11-01 power loss ran 9 *further* stages
# (qd_dpp-s0 to stage 15, pfsp-s0 to stage 12); stages 0-8 are unaffected by later stages, so the analysis uses them only.
MAX_STAGE = 8


def _policies(campaign, role):
    common.CAMPAIGNS[campaign] = campaign
    return [p for p in common.rl_policies(campaign, role) if int(p["metadata"]["stage"]) <= MAX_STAGE]


def run_f3c(args):
    # Amendment 4: both seeds of every arm are required.
    allc = [(s, k, name(s, k)) for k in range(len(SEEDS)) for s in SAMPLERS]
    while not all(os.path.isfile(_done(c)) for _, _, c in allc) or any(_running(c) for _, _, c in allc):
        time.sleep(60)
    camps = [x for x in allc if os.path.isfile(_done(x[2]))]
    log_progress(f"F3c campaigns: {[c for _, _, c in camps]}")
    split = load_split()
    test = test_set(split)
    all_ev = [e for _, _, c in camps for e in _policies(c, "evader")]
    for i, (s, k, c) in enumerate(camps):
        p = _policies(c, "pursuer")[-1]
        for g, opps in (("train", split["train"]), ("test", test), ("campaign_evaders", all_ev)):
            evaluate(os.path.join(F3C, "evals", f"{c}_final_{g}.json"), opps, policy_ckpt=p["checkpoint_path"])
        target = {"id": f"f3/pursuer:{c}", "kind": "rl", "checkpoint_path": p["checkpoint_path"], "metadata": {}}
        ex_ck = train(os.path.join(F3C, "ckpt", f"exploiter_{c}.pt"), [target], EXPLOIT_FRAMES, role="evader",
                      seed=500 + i)
        if ex_ck:
            ex = {"id": f"f3/evader:exploit_{c}", "kind": "rl", "checkpoint_path": ex_ck, "metadata": {}}
            evaluate(os.path.join(F3C, "evals", f"{c}_final_vs_exploiter.json"), [ex], policy_ckpt=p["checkpoint_path"])
        log_progress(f"F3c {c} done")
    analyse_f3(camps, split, test)


def _J(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return {k: v["neutralization_rate"] for k, v in json.load(f)["matchups"].items()}


def _sampling_stats(c):
    """Effective number of opponents exp(H) and max probability of every RL-opponent pursuer stage."""
    d = os.path.join(common.SELFPLAY_OUT, c, "stages")
    out = []
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if fn.endswith("_pursuer_train_opp.json") and int(fn.split("_")[0][len("stage"):]) <= MAX_STAGE:
            with open(os.path.join(d, fn)) as f:
                spec = json.load(f)
            if any(p["kind"] == "rl" for p in spec["policies"]):
                pr = np.array(list(spec["probs"].values()))
                out.append({"n": len(pr), "eff": float(math.exp(-(pr * np.log(pr)).sum())), "max": float(pr.max())})
    return out


def _evader_vendi(camps):
    """Vendi score (effective number of distinct behaviours) of each campaign's RL evaders. One shared descriptor
    space as in X2: J and censored time-to-neutralise vs every campaign's final pursuer (the common probe set of F3c)
    plus mean / std of the behaviour features over those games; one RBF kernel over all F3 evaders."""
    import analysis as an
    recs = {}
    for _, _, c in camps:
        path = os.path.join(F3C, "evals", f"{c}_final_campaign_evaders.json")
        if not os.path.isfile(path):
            return {}
        with open(path) as f:
            recs[c] = json.load(f)["matchups"]
    probes = list(recs)
    evs = sorted(set.intersection(*(set(m) for m in recs.values())))
    max_t = MAX_EPISODE_STEPS * STEP_DT
    resp = [[v for c in probes for v in (recs[c][e]["neutralization_rate"],
                                         recs[c][e]["time_to_neutralize_censored"] / max_t)] for e in evs]
    traj = [np.concatenate([np.mean([recs[c][e]["behaviour"]["mean"] for c in probes], 0),
                            np.mean([recs[c][e]["behaviour"]["std"] for c in probes], 0)]) for e in evs]
    S, _ = an.rbf_kernel(an.pairwise_dist(an.combine_blocks([np.array(resp), np.array(traj)])))
    out = {}
    for c in probes:
        ix = [i for i, e in enumerate(evs) if e.startswith(c + "/")]
        if ix:
            out[c] = {"n": len(ix), "vendi": an.vendi_score(S[np.ix_(ix, ix)])}
    return out


def analyse_f3(camps, split, test):
    import analysis as an
    vendi = _evader_vendi(camps)
    with open(os.path.join(OUT, "X2_descriptors", "clusters.json")) as f:
        labels = json.load(f)["labels"]
    rows = {}
    for s, k, c in camps:
        J = {**_J(os.path.join(F3C, "evals", f"{c}_final_train.json")),
             **_J(os.path.join(F3C, "evals", f"{c}_final_test.json"))}
        Jc = _J(os.path.join(F3C, "evals", f"{c}_final_campaign_evaders.json"))
        if not J or not Jc:
            continue
        r = {"train": np.mean([J[e["id"]] for e in split["train"] if e["id"] in J]),
             "heldout_rl": np.mean([J[e["id"]] for e in split["heldout_rl"] if e["id"] in J]),
             "heldout_scripted": np.mean([J[e["id"]] for e in split["heldout_scripted"] if e["id"] in J]),
             "x3_exploiters": np.mean([J[e["id"]] for e in test if e["id"].startswith("exploit/") and e["id"] in J]),
             "other_campaign_evaders": np.mean([v for kk, v in Jc.items() if not kk.startswith(c + "/")]),
             "own_campaign_evaders": np.mean([v for kk, v in Jc.items() if kk.startswith(c + "/")])}
        by_c = {}
        for e in split["heldout_rl"] + split["heldout_scripted"]:
            if e["id"] in J and e["id"] in labels:
                by_c.setdefault(labels[e["id"]], []).append(J[e["id"]])
        r["heldout_worst_cluster"] = min(np.mean(v) for v in by_c.values())
        ex = _J(os.path.join(F3C, "evals", f"{c}_final_vs_exploiter.json"))
        if ex:
            r["exploitability"] = 1 - next(iter(ex.values()))
        st = _sampling_stats(c)
        if c in vendi:
            r["evader_vendi"] = vendi[c]["vendi"]
        if st:
            r["sampling_eff_frac"] = float(np.mean([x["eff"] / x["n"] for x in st]))
            r["sampling_max_prob"] = float(np.mean([x["max"] for x in st]))
        rows[(s, k)] = {m: float(v) for m, v in r.items()}
    metrics = ["train", "heldout_rl", "heldout_worst_cluster", "heldout_scripted", "x3_exploiters",
               "other_campaign_evaders", "own_campaign_evaders", "exploitability", "evader_vendi", "sampling_eff_frac",
               "sampling_max_prob"]
    n_diff = 9                                   # paired differences for the first n_diff metrics
    agg, diffs = {}, {}
    for s in SAMPLERS:
        agg[s] = {}
        for m in metrics:
            v = [rows[(s, k)][m] for k in range(len(SEEDS)) if (s, k) in rows and m in rows[(s, k)]]
            if v:
                agg[s][m] = {"mean": float(np.mean(v)), "ci": list(an.bootstrap_ci(v)[1:]), "per_seed": v}
    for a, b in (("qd_dpp", "hedge"), ("qd_dpp", "uniform"), ("hedge", "uniform"), ("pfsp", "uniform"),
                 ("qd_dpp", "pfsp")):
        diffs[f"{a}-{b}"] = {}
        for m in metrics[:n_diff]:
            d = [rows[(a, k)][m] - rows[(b, k)][m] for k in range(len(SEEDS))
                 if (a, k) in rows and (b, k) in rows and m in rows[(a, k)] and m in rows[(b, k)]]
            if d:
                diffs[f"{a}-{b}"][m] = {"mean": float(np.mean(d)), "ci": list(an.bootstrap_ci(d)[1:]), "per_seed": d}
    save_json(os.path.join(F3C, "results.json"), {"per_campaign": {f"{s}/{k}": r for (s, k), r in rows.items()},
                                                  "aggregate": agg, "paired": diffs, "evader_vendi": vendi})
    lines = ["# F3_sampler_sp summary", "", "Samplers used by both roles (amendment 3). Final pursuer of each "
             "campaign; mean J over the finished campaign seeds (1-2, see per_seed in results.json) [95 % "
             "bootstrap CI; with 2 seeds the CI is only the range of the two values; 1 seed: no CI]. exploitability = 1 - J vs a fresh exploiter. evader_vendi = "
             "effective number of distinct RL evaders a campaign produced (max = its number of RL evaders).", "",
             "| sampler | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
    for s in SAMPLERS:
        lines.append(f"| {s} | " + " | ".join(
            f"{agg[s][m]['mean']:.3f} [{agg[s][m]['ci'][0]:.3f}, {agg[s][m]['ci'][1]:.3f}]" if m in agg[s] else "-"
            for m in metrics) + " |")
    lines += ["", "Seed-paired differences:", "", "| pair | " + " | ".join(metrics[:n_diff]) + " |",
              "|---" * (n_diff + 1) + "|"]
    for pair, dd in diffs.items():
        lines.append(f"| {pair} | " + " | ".join(
            f"{dd[m]['mean']:+.3f} [{dd[m]['ci'][0]:+.3f}, {dd[m]['ci'][1]:+.3f}]" if m in dd else "-"
            for m in metrics[:n_diff]) + " |")
    with open(os.path.join(F3C, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
