"""F6: Hedge self-play with PSD-PSRO's policy-space diversity regulariser for the evaders (docs/research_plan.md,
Phase F, F6_div_sp; amendment 1, 11-05: Hedge instead of QD-DPP, so that hedge_div - hedge isolates the term).

Campaigns as F3 hedge (plain obs, 9 stages x 300M, Hedge on both roles, seeds 0 / 100). New: every evader after the
first RL evader maximises beta * min(min_j d_j, cap) as a loss term and a per-step intrinsic reward, d_j = mean-only KL
to earlier RL evader j of the same campaign (PPOPolicy.set_diversity). cap = 2 (fixed); beta from a pilot (pilot()).
F6c: the F3c evaluation of each final pursuer (train pool, test set, campaign evaders, fresh exploiter), and the 8 F3
final pursuers vs the F6 evaders (evader Vendi in F3's descriptor space, F3 normalisation and bandwidth frozen).
"""

import json
import os
import re
import time

import numpy as np

import common
from a_adapt import test_set
from common import MAX_EPISODE_STEPS, OUT, STEP_DT, evaluate, log_progress, save_json, train
from d_sampling import EXPLOIT_FRAMES
from f2_final import _done, run_campaign
from f3_sampler_sp import F3C, SAMPLERS, SEEDS, _J, _outcome_metrics, _policies
from f3_sampler_sp import name as f3_name
from x3_gap import load_split

F6 = os.path.join(OUT, "F6_div_sp")
CAP = 2.0
PILOT_BETAS = [0.0, 0.1, 1.0]
PILOT_FRAMES = 100_000_000
PILOT_FROM = ("thesis-F3-hedge-both-s0", 5)         # evader stage 5 of this campaign: 2 earlier RL evaders (s1, s3)
PILOT_J_TOL = 0.05


def name(k, smoke=False):
    return "thesis-F6-smoke-hedge_div" if smoke else f"thesis-F6-hedge_div-s{k}"


def arm_cfg(beta):
    return dict(SAMPLERS["hedge"], diversity_roles="[evader]", diversity_beta=beta, diversity_cap=CAP)


def _div_overrides(ckpts, beta):
    return [f"diversity.ckpts=[{','.join(ckpts)}]", f"diversity.beta={beta}", f"diversity.cap={CAP}"] if beta > 0 else []


def _last_div_distance(log):
    vals = re.findall(r"^div_distance: ([-0-9.eE+]+)", open(log).read(), re.M) if os.path.isfile(log) else []
    return float(np.mean([float(v) for v in vals[-10:]])) if vals else None


def pilot():
    """Pre-registered beta rule (training-time data only): from the F3 hedge-both-s0 state at evader stage 5
    (same opponents and probabilities, warm start from its stage-3 evader, earlier evaders s1 and s3), train an evader
    for 100M frames per beta in PILOT_BETAS and evaluate it vs its training pursuers. Choose the largest beta with
    J(beta) <= J(0) + 0.05 and final mean min-distance >= cap / 2; if none reaches cap / 2, the largest beta that meets
    the J condition. Returns beta."""
    choice = os.path.join(F6, "pilot", "choice.json")
    if os.path.isfile(choice):
        return json.load(open(choice))["beta"]
    c, stage = PILOT_FROM
    d = os.path.join(common.SELFPLAY_OUT, c)
    spec = json.load(open(os.path.join(d, "stages", f"stage{stage}_evader_train_opp.json")))
    refs = [os.path.join(d, "checkpoints", f"evader_stage{s}.pt") for s in range(1, stage, 2)]
    rows = {}
    for beta in PILOT_BETAS:
        ck = os.path.join(F6, "pilot", f"beta{beta}.pt")
        if train(ck, spec["policies"], PILOT_FRAMES, role="evader", probs=spec["probs"], init_policy=refs[-1],
                 seed=900, extra=_div_overrides(refs, beta)) is None:
            raise RuntimeError(f"F6 pilot beta={beta} failed")
        r = evaluate(os.path.join(F6, "pilot", f"beta{beta}_eval.json"), spec["policies"], role="evader",
                     policy_ckpt=ck)
        J = float(np.mean([m["neutralization_rate"] for m in r["matchups"].values()]))
        rows[beta] = {"J_vs_training_pursuers": J,
                      "div_distance_end": _last_div_distance(os.path.splitext(ck)[0] + "_log.txt")}
    j0 = rows[0.0]["J_vs_training_pursuers"]
    ok = [b for b in PILOT_BETAS if b > 0 and rows[b]["J_vs_training_pursuers"] <= j0 + PILOT_J_TOL]
    active = [b for b in ok if (rows[b]["div_distance_end"] or 0.0) >= CAP / 2]
    beta = max(active) if active else (max(ok) if ok else None)
    if beta is None:
        save_json(choice + ".failed", {"rows": rows})
        raise RuntimeError(f"F6 pilot: no beta keeps J within {PILOT_J_TOL} of beta=0: {rows}")
    save_json(choice, {"beta": beta, "cap": CAP, "rows": {str(k): v for k, v in rows.items()},
                       "rule": "largest beta with J <= J0 + 0.05 and div_distance_end >= cap/2, else largest with J ok"})
    log_progress(f"F6 pilot: beta={beta} ({rows})")
    return beta


def _wait_other_lane(done):
    """Block while another lane holds ``done``'s lock (it is producing ``done``)."""
    while not os.path.isfile(done) and os.path.isfile(done + ".lock"):
        time.sleep(60)


def _pilot_once():
    """pilot() on one lane; a second lane waits for its choice."""
    choice = os.path.join(F6, "pilot", "choice.json")
    lock = choice + ".lock"
    if not os.path.isfile(choice) and common._acquire_lock(lock):
        try:
            return pilot()
        finally:
            if os.path.isfile(lock):
                os.remove(lock)
    _wait_other_lane(choice)
    if not os.path.isfile(choice):
        raise RuntimeError("F6 pilot failed on the other lane")
    return json.load(open(choice))["beta"]


def run_f6(args):
    sm = name(0, smoke=True)
    run_campaign(sm, [], arm_cfg(1.0), smoke_stages=5)
    _wait_other_lane(_done(sm))
    if not os.path.isfile(_done(sm)):
        raise RuntimeError("F6 smoke campaign did not finish")
    log = open(os.path.join(common.SELFPLAY_OUT, sm + "_log.txt")).read()
    if log.count("diversity term: beta=") < 1 or "div_distance:" not in log or "div_reward:" not in log:
        raise RuntimeError("F6 smoke: the diversity term was never active")
    if args.smoke:
        return
    beta = _pilot_once()
    for k in range(len(SEEDS)):
        run_campaign(name(k), [], dict(arm_cfg(beta), seed=SEEDS[k]))


def _f3_camps():
    return [(s, k, f3_name(s, k)) for k in range(len(SEEDS)) for s in SAMPLERS]


def run_f6c(args, camps=None, root=None, arm="hedge_div", exploit_seed=600):
    """F6c; F6b reuses it with its own campaigns / root / arm label / exploiter seeds."""
    root = root or F6
    camps = camps or [name(k) for k in range(len(SEEDS))]
    while not all(os.path.isfile(_done(c)) for c in camps):
        time.sleep(60)
    split = load_split()
    test = test_set(split)
    f3 = _f3_camps()
    f3_ev = [e for _, _, c in f3 for e in _policies(c, "evader")]
    f6_ev = [e for c in camps for e in _policies(c, "evader")]
    for i, c in enumerate(camps):
        p = _policies(c, "pursuer")[-1]
        for g, opps in (("train", split["train"]), ("test", test), ("campaign_evaders", f3_ev + f6_ev)):
            evaluate(os.path.join(root, "evals", f"{c}_final_{g}.json"), opps, policy_ckpt=p["checkpoint_path"])
        target = {"id": f"f6/pursuer:{c}", "kind": "rl", "checkpoint_path": p["checkpoint_path"], "metadata": {}}
        ex_ck = train(os.path.join(root, "ckpt", f"exploiter_{c}.pt"), [target], EXPLOIT_FRAMES, role="evader",
                      seed=exploit_seed + i)
        if ex_ck:
            ex = {"id": f"f6/evader:exploit_{c}", "kind": "rl", "checkpoint_path": ex_ck, "metadata": {}}
            evaluate(os.path.join(root, "evals", f"{c}_final_vs_exploiter.json"), [ex], policy_ckpt=p["checkpoint_path"])
        log_progress(f"F6c {c} done")
    for _, _, c in f3:                                  # probes for the F6 evaders' descriptors
        evaluate(os.path.join(root, "evals", f"probe_{c}_vs_f6_evaders.json"), f6_ev,
                 policy_ckpt=_policies(c, "pursuer")[-1]["checkpoint_path"])
    need = [os.path.join(root, "evals", f"{c}_final_{g}.json") for c in camps
            for g in ("train", "test", "campaign_evaders", "vs_exploiter")]
    need += [os.path.join(root, "evals", f"probe_{c}_vs_f6_evaders.json") for _, _, c in f3]
    missing = [x for x in need if not os.path.isfile(x)]
    if missing:                                   # the other lane is still producing them and analyses after
        log_progress(f"F6c: {len(missing)} eval files missing on this lane, analysis left to the other lane")
        return
    analyse_f6(camps, split, test, root=root, arm=arm)


def _descriptors(recs, probes, evs):
    max_t = MAX_EPISODE_STEPS * STEP_DT
    resp = np.array([[v for c in probes for v in (recs[c][e]["neutralization_rate"],
                                                  recs[c][e]["time_to_neutralize_censored"] / max_t)] for e in evs])
    traj = np.array([np.concatenate([np.mean([recs[c][e]["behaviour"]["mean"] for c in probes], 0),
                                     np.mean([recs[c][e]["behaviour"]["std"] for c in probes], 0)]) for e in evs])
    return resp, traj


def evader_vendi(f3, f6_camps, root=None):
    """Vendi per campaign in F3's descriptor space: z-score parameters and RBF bandwidth from the 32 F3 evaders only
    (so the F3 values are reproduced exactly), F6 evaders projected into that space."""
    import analysis as an
    probes = [c for _, _, c in f3]
    recs = {c: json.load(open(os.path.join(F3C, "evals", f"{c}_final_campaign_evaders.json")))["matchups"]
            for c in probes}
    f3_evs = sorted(set.intersection(*(set(m) for m in recs.values())))
    f6_recs = {c: json.load(open(os.path.join(root or F6, "evals", f"probe_{c}_vs_f6_evaders.json")))["matchups"]
               for c in probes}
    f6_evs = sorted(set.intersection(*(set(m) for m in f6_recs.values())))
    blocks3 = _descriptors(recs, probes, f3_evs)
    blocks6 = _descriptors(f6_recs, probes, f6_evs)
    X3, X6 = [], []
    for B3, B6 in zip(blocks3, blocks6):
        mu, sd = B3.mean(0), B3.std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        X3.append((B3 - mu) / sd / np.sqrt(B3.shape[1]))
        X6.append((B6 - mu) / sd / np.sqrt(B3.shape[1]))
    X3, X6 = np.concatenate(X3, 1), np.concatenate(X6, 1)
    _, sigma = an.rbf_kernel(an.pairwise_dist(X3))
    X = np.concatenate([X3, X6])
    S, _ = an.rbf_kernel(an.pairwise_dist(X), sigma)
    evs = f3_evs + f6_evs
    out = {}
    for c in probes + list(f6_camps):
        ix = [i for i, e in enumerate(evs) if e.startswith(c + "/")]
        if ix:
            out[c] = {"n": len(ix), "vendi": an.vendi_score(S[np.ix_(ix, ix)])}
    return out


def analyse_f6(camps, split, test, root=None, arm="hedge_div"):
    import analysis as an
    root = root or F6
    with open(os.path.join(OUT, "X2_descriptors", "clusters.json")) as f:
        labels = json.load(f)["labels"]
    f3 = _f3_camps()
    f3_names = {c for _, _, c in f3}
    vendi = evader_vendi(f3, camps, root)
    f3res = json.load(open(os.path.join(F3C, "results.json")))["per_campaign"]
    rows = {}
    for k, c in enumerate(camps):
        r = _outcome_metrics(c, split, test, labels, "neutralization_rate", root=root)
        if r is None:
            continue
        r.update({f"{m}_cap": v for m, v in _outcome_metrics(c, split, test, labels, "capture_rate", root=root).items()})
        for key, m in (("neutralization_rate", "other_campaign_evaders"), ("capture_rate", "other_campaign_evaders_cap")):
            Jc = _J(os.path.join(root, "evals", f"{c}_final_campaign_evaders.json"), key)
            r[m] = np.mean([v for kk, v in Jc.items() if not kk.endswith(":s1") and kk.split("/")[0] in f3_names])
        r["evader_vendi"] = vendi[c]["vendi"]
        rows[(arm, k)] = {m: float(v) for m, v in r.items()}
    for s, k, c in f3:
        rows[(s, k)] = dict(f3res[f"{s}/{k}"])
        rows[(s, k)]["evader_vendi_check"] = vendi[c]["vendi"]   # must equal F3's evader_vendi
    metrics = ["train", "heldout_rl", "heldout_worst_cluster", "heldout_scripted", "x3_exploiters",
               "other_campaign_evaders", "own_campaign_evaders", "exploitability", "evader_vendi"]
    cap = [f"{m}_cap" for m in metrics[:8]]
    arms = [arm] + list(SAMPLERS)
    agg, diffs = {}, {}
    for s in arms:
        agg[s] = {}
        for m in metrics + cap:
            v = [rows[(s, k)][m] for k in range(len(SEEDS)) if (s, k) in rows and m in rows[(s, k)]]
            if v:
                agg[s][m] = {"mean": float(np.mean(v)), "ci": list(an.bootstrap_ci(v)[1:]), "per_seed": v}
    for b in ("hedge", "qd_dpp", "uniform", "pfsp"):
        diffs[f"{arm}-{b}"] = {}
        for m in metrics + cap:
            d = [rows[(arm, k)][m] - rows[(b, k)][m] for k in range(len(SEEDS))
                 if (arm, k) in rows and (b, k) in rows and m in rows[(arm, k)] and m in rows[(b, k)]]
            if d:
                diffs[f"{arm}-{b}"][m] = {"mean": float(np.mean(d)), "ci": list(an.bootstrap_ci(d)[1:]),
                                                "per_seed": d}
    save_json(os.path.join(root, "results.json"), {"per_campaign": {f"{s}/{k}": r for (s, k), r in rows.items()},
                                                "aggregate": agg, "paired": diffs, "evader_vendi": vendi})
    lines = [f"# {os.path.basename(root)} summary", "", f"{arm} = F3 hedge + PSD-PSRO diversity regulariser for evaders"
             f"{' (F6b form)' if arm != 'hedge_div' else ''}. F3 arms "
             f"copied from F3_sampler_sp/results.json. Mean over 2 seeds [range]. other_campaign_evaders: for {arm} "
             "the F3 evaders (stage-1 excluded); for F3 arms as in F3. Descriptive (2 seeds).", "",
             "| arm | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
    for s in arms:
        lines.append(f"| {s} | " + " | ".join(
            f"{agg[s][m]['mean']:.3f} [{agg[s][m]['ci'][0]:.3f}, {agg[s][m]['ci'][1]:.3f}]" if m in agg[s] else "-"
            for m in metrics) + " |")
    lines += ["", "Seed-paired differences:", "", "| pair | " + " | ".join(metrics) + " |",
              "|---" * (len(metrics) + 1) + "|"]
    for pair, dd in diffs.items():
        lines.append(f"| {pair} | " + " | ".join(
            f"{dd[m]['mean']:+.3f} [{dd[m]['ci'][0]:+.3f}, {dd[m]['ci'][1]:+.3f}]" if m in dd else "-"
            for m in metrics) + " |")
    lines += ["", "Vendi check (F3 values recomputed in the frozen space): " + ", ".join(
        f"{c}: {vendi[c]['vendi']:.3f}" for _, _, c in f3)]
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
