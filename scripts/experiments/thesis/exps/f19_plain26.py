"""F19: plain (no-estimator) pursuer at 26 stages under the thesis v2d rules (docs/research_plan.md, F19_plain26;
registered 2026-11-18 16:23 before any F19 job).

Seeds 0 / 100 reuse the 26-stage F11v2d campaigns and their F11c stage-24 exploiters. Seed 200 continues the 13-stage
F14 short campaign to 26 stages with F14 Stage E's verified-clone procedure, then trains its two exploiters.
Evaluations follow F14 (strong-16, Stage E cross pool, own / lure exploiters) and F15 (switch protocol).

F19_PART selects a lane: "evals" (seeds 0 / 100 only), "s200" (continuation + seed 200), default "all".
"""

import json
import os
import statistics as st

import torch

import common
from common import OUT, evaluate, log_progress, save_json, train
from d_sampling import EXPLOIT_FRAMES
from f2_final import run_campaign
import f8_roncero_sp as f8
import f8d_heldout as f8d
import f11_v2_sp as f11
import f14_generalization_sp as f14
import f15_switch_baseline as f15

F19 = os.path.join(OUT, "F19_plain26")
SEEDS = f14.SEEDS
SOURCE_S200 = f14.F11_NEW                         # thesis-F14-v2d-short-s200 (13 stages)
NEW_S200 = "thesis-F19-v2d-plain-26-s200"
CAMPS = {0: f11.name(0), 100: f11.name(1), 200: NEW_S200}


def _out(c, group):
    return os.path.join(F19, "evals", f"{c}_{group}.json")


def _continue_s200():
    f14._prepare_stagee_campaign(SOURCE_S200, NEW_S200, "plain", 200)
    cfg = f14._campaign_cfg("v2d", 200)
    cfg.update(n_stages=26, max_total_stages=26, frames_per_stage=50_000_000)
    if cfg["seed"] != 200 or cfg["n_stages"] != 26:
        raise RuntimeError("F19 continuation config verification failed")
    run_campaign(NEW_S200, f14._campaign_rules("v2d"), cfg)
    f14._wait_campaign(NEW_S200)


def _exploiters(seed, c, ck):
    if seed in (0, 100):
        own = os.path.join(f11.F11, "ckpt", f"exploiter_v2_{c}.pt")
        lure = os.path.join(f11.F11, "ckpt", f"exploiter_old_{c}.pt")
    else:
        own = os.path.join(F19, "ckpt", f"exploiter_own_{c}.pt")
        lure = os.path.join(F19, "ckpt", f"exploiter_lure_{c}.pt")
        target = [{"id": f"f19/pursuer:{c}", "kind": "rl", "checkpoint_path": ck, "metadata": {}}]
        if not os.path.isfile(own):
            train(own, target, EXPLOIT_FRAMES, role="evader", seed=862, extra=f14._campaign_rules("v2d"))
        if not os.path.isfile(lure):
            train(lure, target, EXPLOIT_FRAMES, role="evader", seed=872, extra=list(f11.OLD_RULES_EXPLOITER))
    for p in (own, lure):
        if not os.path.isfile(p):
            raise RuntimeError(f"F19: missing exploiter {p}")
    mk = lambda tag, p: [{"id": f"f19/evader:{tag}_{c}", "kind": "rl", "checkpoint_path": p, "metadata": {}}]
    return mk("own", own), mk("lure", lure)


def _cross_pool(c):
    """Exactly the F14 Stage E pool (all evaders of every v2d campaign evaluated there) minus campaign ``c``."""
    stagec, stagee = f14._arm_campaigns("v2d"), f14._stagee_campaigns("v2d")
    all_camps = tuple(stagec["plain"]) + tuple(stagec["rls"]) + tuple(stagee["rls"]) + tuple(stagec["no_rls"]) \
        + tuple(stagee["no_rls"])
    return [p for p in f14._evaders(all_camps) if p["id"].split("/")[0] != c]


def jobs_for(seed):
    c = CAMPS[seed]
    ck = f14._ck(c, 24)
    strong = [p for p in f8d._finals() if p["id"].split("/")[0] in f8d.COMMON]
    own, lure = _exploiters(seed, c, ck)
    out = [(_out(c, "strong"), strong, ck, 0, None), (_out(c, "cross"), _cross_pool(c), ck, 0, None),
           (_out(c, "own"), own, ck, 0, None), (_out(c, "lure"), lure, ck, 0, None)]
    for direction in ("f8_to_f10", "f10_to_f8"):
        first, second = f14._switch_pair(seed, direction)
        sw = {"first_id": first["id"], "second_id": second["id"], "times_s": [0.5, 1.0, 1.5],
              "group_id": f"switch:{direction}:randomized", "seed": seed + 401}
        out.append((_out(c, f"switch_{direction}_randomized"), [first, second], ck, seed + 1400, sw))
    first, second = f14._switch_pair(seed, "f8_to_f10")
    for pol in (first, second):
        out.append((_out(c, f"fixed_{pol['id'].split('/')[0]}"), [pol], ck, seed + 1400, None))
    return out


def _run_jobs(seed):
    for path, opps, ck, rng, sw in jobs_for(seed):
        if not os.path.isfile(path):
            evaluate(path, opps, policy_ckpt=ck, seed=rng, switch=sw)
        log_progress(f"F19 eval {os.path.basename(path)}")


# ------------------------------------------------------------------ analysis
def _j(path, key="neutralization_rate"):
    m = json.load(open(path))["matchups"]
    return m


def _campaign_balanced(path, self_c):
    by = {}
    for pid, v in json.load(open(path))["matchups"].items():
        camp = pid.split("/")[0]
        if camp != self_c:
            by.setdefault(camp, []).append(v["neutralization_rate"])
    return st.mean(st.mean(v) for v in by.values())


def _switch_seed(seed, c):
    rows = []
    for direction in ("f8_to_f10", "f10_to_f8"):
        job = {"path": _out(c, f"switch_{direction}_randomized"), "seed": seed, "direction": direction}
        rows.extend(f15._switch_records(job, "plain26"))
    keys = ("post_switch_neutralization", "post_switch_capture", "full_episode_j",
            "failure_penalized_post_switch_capture_time_s")
    res = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    res["alive_at_switch"] = sum(r["alive_fraction_at_switch"] for r in rows) / 2.0
    fixed = []
    first, second = f14._switch_pair(seed, "f8_to_f10")
    for pol in (first, second):
        m = next(iter(json.load(open(_out(c, f"fixed_{pol['id'].split('/')[0]}")))["matchups"].values()))
        fixed.append(m)
    res["fixed_capture"] = st.mean(m["capture_rate"] for m in fixed)
    res["fixed_j"] = st.mean(m["neutralization_rate"] for m in fixed)
    return res


def analyse():
    strong_ids = None
    per = {}
    for seed in SEEDS:
        c = CAMPS[seed]
        sm = json.load(open(_out(c, "strong")))["matchups"]
        strong = st.mean(v["neutralization_rate"] for k, v in sm.items() if k.split("/")[0] in f8d.COMMON)
        cross = _campaign_balanced(_out(c, "cross"), c)
        own = next(iter(json.load(open(_out(c, "own")))["matchups"].values()))
        lure = next(iter(json.load(open(_out(c, "lure")))["matchups"].values()))
        per[str(seed)] = {"campaign": c, "strong_j": strong, "cross_j": cross, "g1": 0.5 * (strong + cross),
                          "own_exploitability": 1.0 - own["neutralization_rate"],
                          "lure_j": lure["neutralization_rate"], "lure_pursuer_crash": lure["outcomes"]["pursuer_crash"],
                          **_switch_seed(seed, c)}
    E = json.load(open(os.path.join(f14.F14, "stageE_results.json")))["metrics_r0.10"]["rls"]
    base = json.load(open(os.path.join(f14.F14, "baseline_results.json")))["metrics"]["r0.10"]["f8"]
    f15r = json.load(open(os.path.join(OUT, "F15_switch_baseline", "results.json")))["switch_means_by_seed"]
    cmp = {}
    for seed in SEEDS:
        s = str(seed)
        rls = E[s]
        f8s = base[f14.F8_CAMPS[SEEDS.index(seed)]]
        p = per[s]
        cmp[s] = {
            "plain_minus_rls": {"strong_j": p["strong_j"] - rls["strong_j"], "cross_j": p["cross_j"] - rls["cross_j"],
                                "g1": p["g1"] - rls["g1"],
                                "own_exploitability": p["own_exploitability"] - rls["own_exploitability"],
                                "post_switch_capture": p["post_switch_capture"] - f15r["rls"][s]["post_switch_capture"],
                                "post_switch_j": p["post_switch_neutralization"] - f15r["rls"][s]["post_switch_neutralization"],
                                "fp_time_s": p["failure_penalized_post_switch_capture_time_s"]
                                - f15r["rls"][s]["failure_penalized_post_switch_capture_time_s"]},
            "plain_minus_f8": {"g1": p["g1"] - f8s["g1"], "strong_j": p["strong_j"] - f8s["strong_j"],
                               "own_exploitability": p["own_exploitability"] - f8s["own_exploitability"],
                               "lure_j": p["lure_j"] - f8s["thesis_lure_j"],
                               "post_switch_capture": p["post_switch_capture"] - f15r["f8"][s]["post_switch_capture"],
                               "post_switch_j": p["post_switch_neutralization"] - f15r["f8"][s]["post_switch_neutralization"],
                               "fp_time_s": p["failure_penalized_post_switch_capture_time_s"]
                               - f15r["f8"][s]["failure_penalized_post_switch_capture_time_s"]}}
    def verdict(vals):
        m = sum(vals) / len(vals)
        same = all(v > 0 for v in vals) or all(v < 0 for v in vals)
        return "notable" if abs(m) >= 0.05 and same else "inconclusive"
    summary = {}
    for side in ("plain_minus_rls", "plain_minus_f8"):
        summary[side] = {}
        for k in cmp["0"][side]:
            vals = [cmp[str(s)][side][k] for s in SEEDS]
            summary[side][k] = {"per_seed": vals, "mean": sum(vals) / 3, "verdict": verdict(vals)}
    res = {"plain26": per, "paired": summary}
    save_json(os.path.join(F19, "results.json"), res)
    lines = ["# F19 plain-26 vs RLS-26 and F8 (0.10 m)", "", "| Contrast | Metric | Seed 0 | Seed 100 | Seed 200 | Mean | Verdict |",
             "|---|---|---:|---:|---:|---:|---|"]
    for side, d in summary.items():
        for k, v in d.items():
            lines.append(f"| {side} | {k} | " + " | ".join(f"{x:+.3f}" for x in v["per_seed"]) +
                         f" | {v['mean']:+.3f} | {v['verdict']} |")
    lines += ["", "| Seed | Strong J | Cross J | G1 | Own expl. | Lure J | Lure crash | Post-switch capture | Post-switch J | FP time (s) | Fixed capture |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, p in per.items():
        lines.append(f"| {s} | {p['strong_j']:.3f} | {p['cross_j']:.3f} | {p['g1']:.3f} | {p['own_exploitability']:.3f} | "
                     f"{p['lure_j']:.3f} | {p['lure_pursuer_crash']:.3f} | {p['post_switch_capture']:.3f} | "
                     f"{p['post_switch_neutralization']:.3f} | {p['failure_penalized_post_switch_capture_time_s']:.2f} | "
                     f"{p['fixed_capture']:.3f} |")
    with open(os.path.join(F19, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log_progress("F19 analysis done")
    return res


def run_f19(args):
    part = os.environ.get("F19_PART", "all")
    os.makedirs(os.path.join(F19, "ckpt"), exist_ok=True)
    common.CAMPAIGNS[NEW_S200] = NEW_S200
    if part in ("evals", "all"):
        for seed in (0, 100):
            _run_jobs(seed)
    if part in ("s200", "all"):
        _continue_s200()
        _run_jobs(200)
    try:
        done = all(os.path.isfile(p) for seed in SEEDS for p, *_ in jobs_for(seed))
    except Exception as e:          # seed 200 not trained yet in the "evals" lane
        log_progress(f"F19: analysis deferred ({e})")
        return
    if done:
        analyse()
