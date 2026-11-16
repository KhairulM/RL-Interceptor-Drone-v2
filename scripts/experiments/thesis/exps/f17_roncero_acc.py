"""F17: F8 (Roncero-rule, Hedge) self-play with the learned acceleration estimator + RLS on the pursuer."""

import json
import os
import time

import common
from common import OUT, evaluate, log_progress
from f2_final import _done, run_campaign
import f5_acc_sp as f5
import f8_roncero_sp as f8
import f8d_heldout as f8d
import f11_v2_sp as f11
import f14_generalization_sp as f14
import f15_switch_baseline as f15
from rescore import rates_at

F17 = os.path.join(OUT, "F17_roncero_acc")
SEEDS = (0, 100, 200)
CAMPS = tuple(f"thesis-F17-roncero-acc-rls-s{s}" for s in SEEDS)
SMOKE = "thesis-F17-smoke-roncero-acc-rls"
FINAL_STAGE = 24
OWN_EXPLOITER_SEED = 2000
THESIS_EXPLOITER_SEED = 2500
NOTABLE = 0.05


def cfg(seed):
    return dict(f8.SAMPLERS["hedge"], seed=seed, frames_per_stage=f8.FRAMES, n_stages=f8.STAGES,
                max_total_stages=f8.STAGES)


def _out(c, group):
    return os.path.join(F17, "evals", f"{c}_{group}.json")


def _wait(path, what):
    while not os.path.isfile(path):
        if not os.path.isfile(path + ".lock"):
            raise RuntimeError(f"F17: {what} failed: {path}")
        time.sleep(30)


def _smoke():
    run_campaign(SMOKE, list(f8.RULES), dict(f8.SAMPLERS["hedge"]), smoke_stages=2, role_overrides=f14._roles("rls"))
    _wait(_done(SMOKE), "smoke campaign")
    path = os.path.join(F17, "smoke", "eval.json")
    if not os.path.isfile(path):
        evaluate(path, f8.all_stages(SMOKE, "evader"), policy_ckpt=f14._ck(SMOKE, 0), num_envs=256, seed=399,
                 cutoff=300, extra=f5.rls_overrides())
    _wait(path, "smoke estimator evaluation")
    matchups = json.load(open(path))["matchups"]
    if not matchups or any(m["episodes"] < 1 for m in matchups.values()):
        raise RuntimeError("F17 smoke evaluation produced no episodes")


def run_f17(args):
    _smoke()
    log_progress("F17 smoke passed (both roles trained; estimator-on pursuer evaluates)")
    if args.smoke:
        return
    for seed, c in zip(SEEDS, CAMPS):
        run_campaign(c, list(f8.RULES), cfg(seed), role_overrides=f14._roles("rls"))


def _eval_jobs():
    """(campaign, group, opponents, checkpoint, extra). Exploiters are trained here (blocking, lock-shared)."""
    strong = [p for p in f8d._finals() if p["id"].split("/")[0] in f8d.COMMON]
    live = f5.rls_overrides()
    frozen = f5.rls_overrides() + [f5.FREEZE]
    jobs = []
    for seed, c in zip(SEEDS, CAMPS):
        ck = f14._ck(c, FINAL_STAGE)
        cross = [p for p in f14._evaders(CAMPS) if p["id"].split("/")[0] != c]
        pid = f"f17/pursuer:{c}"
        opp = f5._mirror(f5.rls_overrides())
        own = f14._exploiter(os.path.join(F17, "ckpt", f"exploiter_roncero_{c}.pt"), pid, c, ck, f8.RULES,
                             OWN_EXPLOITER_SEED + seed, opponent_extra=opp)
        thesis = f14._exploiter(os.path.join(F17, "ckpt", f"exploiter_thesis_{c}.pt"), pid, c, ck,
                                f11.OLD_RULES_EXPLOITER, THESIS_EXPLOITER_SEED + seed, opponent_extra=opp)
        jobs += [(c, "strong", strong, ck, live), (c, "cross", cross, ck, live),
                 (c, "own_exploiter", own, ck, live), (c, "thesis_exploiter", thesis, ck, live),
                 (c, "frozen_strong", strong, ck, frozen), (c, "frozen_cross", cross, ck, frozen),
                 (c, "frozen_own_exploiter", own, ck, frozen)]
    return jobs


def _switch_jobs():
    """F15's F8 switch/fixed-control design, for the F17 pursuer with live and frozen-at-evaluation RLS."""
    jobs = []
    for model, extra in (("f17", f5.rls_overrides()), ("f17_frozen", f5.rls_overrides() + [f5.FREEZE])):
        for seed, c in zip(SEEDS, CAMPS):
            for job in f15._baseline_jobs():
                if job["seed"] != seed:
                    continue
                job = dict(job, model=model, checkpoint=f14._ck(c, FINAL_STAGE), extra=extra,
                           path=_out(c, f"switch_{model}_{job['group'].removeprefix(f'f8_s{seed}_')}"))
                jobs.append(job)
    return jobs


def _score(c, radius, frozen=False):
    p = "frozen_" if frozen else ""
    rate = lambda g, key="neutralization_rate": rates_at(_out(c, p + g), key, radius)
    one = lambda g, key="neutralization_rate": float(next(iter(rate(g, key).values())))
    strong = f14._mean([v for k, v in rate("strong").items() if k.split("/")[0] in f8d.COMMON])
    cross = f14._group_cross(rate("cross"), c)
    r = {"strong_j": strong, "cross_j": cross, "g1": 0.5 * (strong + cross), "own_j": one("own_exploiter"),
         "own_exploitability": 1.0 - one("own_exploiter")}
    if not frozen:
        r.update(thesis_lure_j=one("thesis_exploiter"), thesis_lure_exploitability=1.0 - one("thesis_exploiter"),
                 own_pursuer_oob=one("own_exploiter", "pursuer_oob"),
                 thesis_pursuer_oob=one("thesis_exploiter", "pursuer_oob"))
    return r


def _contrast(diffs):
    mean = f14._mean(diffs)
    signs = {"all_positive": all(d > 0 for d in diffs), "all_negative": all(d < 0 for d in diffs)}
    verdict = ("notable_improvement" if mean >= NOTABLE and signs["all_positive"] else
               "notable_degradation" if mean <= -NOTABLE and signs["all_negative"] else "inconclusive")
    return {"per_seed": dict(zip(map(str, SEEDS), diffs)), "mean": mean, **signs, "verdict": verdict}


def _switch_summary(jobs):
    f15_res = json.load(open(os.path.join(f15.F15, "results.json")))
    keys = ("post_switch_neutralization", "post_switch_capture", "full_episode_j")
    by_seed = {}
    for model in ("f17", "f17_frozen"):
        for seed in SEEDS:
            recs = [r for j in jobs if j["model"] == model and j["seed"] == seed and j["kind"] == "switch"
                    for r in f15._switch_records(j, model)]
            fixed = [f15._fixed_record(j, j["path"]) for j in jobs
                     if j["model"] == model and j["seed"] == seed and j["kind"] == "fixed"]
            if len(recs) != 6 or len(fixed) != 2:
                raise RuntimeError(f"F17 switch cells incomplete for {model} seed {seed}")
            by_seed.setdefault(model, {})[str(seed)] = dict(
                {k: f14._mean([r[k] for r in recs]) for k in keys},
                fixed_j_r010=f14._mean([r["fixed_j_r010"] for r in fixed]))
    f8_fixed = {str(s): f14._mean([r["fixed_j_r010"] for r in f15_res["records"]
                                  if r["model"] == "f8" and r["seed"] == s and r["kind"] == "fixed"]) for s in SEEDS}
    f8_switch = f15_res["switch_means_by_seed"]["f8"]
    contrasts = {k: _contrast([by_seed["f17"][str(s)][k] - f8_switch[str(s)][k] for s in SEEDS]) for k in keys}
    contrasts["fixed_j_r010"] = _contrast([by_seed["f17"][str(s)]["fixed_j_r010"] - f8_fixed[str(s)] for s in SEEDS])
    contrasts["live_minus_frozen_post_switch_j"] = _contrast(
        [by_seed["f17"][str(s)]["post_switch_neutralization"] - by_seed["f17_frozen"][str(s)]["post_switch_neutralization"]
         for s in SEEDS])
    return {"by_seed": by_seed, "f8_switch_by_seed": f8_switch, "f8_fixed_by_seed": f8_fixed,
            "contrasts_vs_f8": contrasts}


def analyse(switch_jobs):
    baseline = json.load(open(os.path.join(f14.F14, "baseline_results.json")))
    out = {"campaigns": dict(zip(map(str, SEEDS), CAMPS)), "f8_campaigns": dict(zip(map(str, SEEDS), f14.F8_CAMPS)),
           "decision_rule": f"notable if mean paired F17-F8 >= {NOTABLE} and all three seeds share its sign"}
    for radius in f14.RADII:
        rk = f"r{radius:.2f}"
        f8m = baseline["metrics"][rk]["f8"]
        f17 = {str(s): dict(_score(c, radius), frozen_at_eval=_score(c, radius, frozen=True))
               for s, c in zip(SEEDS, CAMPS)}
        f8s = {str(s): f8m[c] for s, c in zip(SEEDS, f14.F8_CAMPS)}
        diff = lambda k: _contrast([f17[str(s)][k] - f8s[str(s)][k] for s in SEEDS])
        out[rk] = {"f17": f17, "f8": f8s,
                   "contrasts_vs_f8": {"g1": diff("g1"), "g2_own_j": diff("own_j"), "strong_j": diff("strong_j"),
                                       "cross_j": diff("cross_j"), "thesis_lure_j": diff("thesis_lure_j")},
                   "live_minus_frozen_g1": _contrast([f17[str(s)]["g1"] - f17[str(s)]["frozen_at_eval"]["g1"]
                                                      for s in SEEDS])}
    out["switch"] = _switch_summary(switch_jobs)
    common.save_json(os.path.join(F17, "results.json"), out)
    _write_summary(out)
    log_progress("F17 evaluations and analysis complete")


def _write_summary(out):
    r = out["r0.10"]
    lines = ["# F17: F8 rules + learned acceleration estimator/RLS on the pursuer", "",
             f"Paired per-seed F17 - F8 (stage-24 pursuers, 0.10 m). {out['decision_rule']}.", "",
             "| Metric | s0 | s100 | s200 | Mean | Verdict |", "|---|---:|---:|---:|---:|---|"]
    rows = [(f"{k} (0.10 m)", r["contrasts_vs_f8"][k]) for k in r["contrasts_vs_f8"]]
    rows += [(f"{k} (0.15 m)", out["r0.15"]["contrasts_vs_f8"][k]) for k in ("g1", "g2_own_j")]
    rows += [("live - frozen G1 (0.10 m)", r["live_minus_frozen_g1"])]
    rows += [(f"switch {k}", v) for k, v in out["switch"]["contrasts_vs_f8"].items()]
    for label, c in rows:
        vals = " | ".join(f"{c['per_seed'][str(s)]:+.3f}" for s in SEEDS)
        lines.append(f"| {label} | {vals} | {c['mean']:+.3f} | {c['verdict']} |")
    lines += ["", "| Seed | F17 G1 | F8 G1 | F17 own J | F8 own J | F17 thesis-lure J | F8 thesis-lure J |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for s in SEEDS:
        a, b = r["f17"][str(s)], r["f8"][str(s)]
        lines.append(f"| {s} | {a['g1']:.3f} | {b['g1']:.3f} | {a['own_j']:.3f} | {b['own_j']:.3f} | "
                     f"{a['thesis_lure_j']:.3f} | {b['thesis_lure_j']:.3f} |")
    lines += ["", "Caveat: cross-campaign J uses each family's own other-seed evaders (F14 style), so the F17 and F8 "
              "cross components face different evader banks. The estimator was fitted on F5 thesis-rule data."]
    with open(os.path.join(F17, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def run_f17c(args):
    if args.smoke:
        raise SystemExit("F17c has no smoke mode; run --exp F17 --smoke")
    for c in CAMPS:
        f14._wait_campaign(c)
    jobs = _eval_jobs()
    for c, group, opponents, ck, extra in jobs:
        if not os.path.isfile(_out(c, group)):
            evaluate(_out(c, group), opponents, policy_ckpt=ck, extra=extra)
    switch_jobs = _switch_jobs()
    for job in switch_jobs:
        if not os.path.isfile(job["path"]):
            evaluate(job["path"], job["policies"], policy_ckpt=job["checkpoint"], seed=job["rng_seed"],
                     extra=job["extra"], switch=job["switch"])
    missing = [_out(c, g) for c, g, *_ in jobs if not os.path.isfile(_out(c, g))]
    missing += [j["path"] for j in switch_jobs if not os.path.isfile(j["path"])]
    if missing:
        log_progress(f"F17c: {len(missing)} evaluation files missing (another lane?); analysis deferred")
        return
    analyse(switch_jobs)
