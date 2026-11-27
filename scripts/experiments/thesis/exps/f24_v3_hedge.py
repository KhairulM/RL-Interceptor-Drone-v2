"""F24: F23 (v3 + learned acceleration estimator + online RLS) restarted with Hedge opponent sampling on both roles
and a 0.3 m boundary-penalty margin (research plan F24). Same seeds, stages, RLS settings and evaluation as F23's v3
part; F23's historical v2d re-evaluation is not repeated."""

import json
import os

import common
from common import OUT, evaluate, log_progress, train
from d_sampling import EXPLOIT_FRAMES
from f2_final import run_campaign
import f5_acc_sp as f5
import f8_roncero_sp as f8
import f8d_heldout as f8d
import f11_v2_sp as f11
import f14_generalization_sp as f14
import f15_switch_baseline as f15
import f21_rls_ablation as f21
import f23_v3_rls as f23

F24 = os.path.join(OUT, "F24_v3_hedge")
TASK = "PursuitEvasionV3b"          # PursuitEvasionV3 + bounds_margin 0.3
SEEDS = (0, 100, 200)
CAMPAIGNS = tuple(f"thesis-F24-v3b-hedge-acc-rls-s{s}" for s in SEEDS)
SMOKE = "thesis-F24-smoke-v3b-hedge"
V3_RULES = f23.V3_RULES
ROLE_OVERRIDES = f21.roles("baseline")
RLS_OVERRIDES = f5.rls_overrides()
STAGES = 26


def _cfg(seed, stages):
    return dict(f11.cfg(seed, stages), task=TASK)   # f11.cfg: Hedge on both roles, 50M frames/stage


def _ck(campaign, stage=24):
    return f14._ck(campaign, stage)


def _wait(path, what):
    if not os.path.isfile(path):
        raise RuntimeError(f"F24 {what} did not complete: {path}")


def _smoke():
    run_campaign(SMOKE, V3_RULES, _cfg(0, 2), smoke_stages=2, role_overrides=ROLE_OVERRIDES)
    f14._wait_campaign(SMOKE)
    common.CAMPAIGNS[SMOKE] = SMOKE
    for role in ("pursuer", "evader"):
        if not common.rl_policies(SMOKE, role):
            raise RuntimeError(f"F24 smoke produced no {role} policy")
    path = os.path.join(F24, "smoke", "v3b_rls_live.json")
    result = evaluate(path, common.rl_policies(SMOKE, "evader"), policy_ckpt=f14._ck(SMOKE, 0), num_envs=256,
                      cutoff=300, extra=RLS_OVERRIDES, task_config=TASK)
    if result is None or not result.get("matchups"):
        raise RuntimeError("F24 smoke RLS evaluation failed")
    log_progress("F24 v3b two-role training and live-RLS evaluation smoke passed")


def run_f24(args):
    if args.seeds != len(SEEDS):
        raise ValueError("F24 uses the registered seeds 0/100/200")
    os.makedirs(F24, exist_ok=True)
    _smoke()
    if args.smoke:
        return
    # Each lane takes every campaign no other lane holds, then waits for all (a second lane can join at any time).
    for seed, campaign in zip(SEEDS, CAMPAIGNS):
        run_campaign(campaign, V3_RULES, _cfg(seed, STAGES), role_overrides=ROLE_OVERRIDES)
    for campaign in CAMPAIGNS:
        f14._wait_campaign(campaign)
    log_progress("F24 campaigns complete: " + ", ".join(CAMPAIGNS))


def _exploiter(target_campaign, target_ck, tag, rules, task_config, seed, opponent_extra):
    path = os.path.join(F24, "ckpt", f"{target_campaign}_{tag}.pt")
    ckpt = train(path, [{"id": f"f24/pursuer:{target_campaign}", "kind": "rl", "checkpoint_path": target_ck,
                         "metadata": {}}],
                 EXPLOIT_FRAMES, role="evader", seed=seed, extra=list(rules) + list(opponent_extra),
                 task_config=task_config)
    if ckpt is None:
        _wait(path, f"{tag} exploiter")
        ckpt = path
    return [{"id": f"f24/evader:{target_campaign}_{tag}", "kind": "rl", "checkpoint_path": ckpt, "metadata": {}}]


def _eval(path, opponents, checkpoint, seed, switch=None, freeze=False, capture_radius=0.10, cutoff=1500):
    extra = [o for o in RLS_OVERRIDES if not o.startswith("task.adapt.freeze=")]
    extra.append(f"task.adapt.freeze={str(freeze).lower()}")
    extra.append(f"task.success_radius_eval={capture_radius}")
    result = evaluate(path, opponents, policy_ckpt=checkpoint, num_envs=4096, seed=seed, cutoff=cutoff, extra=extra,
                      switch=switch, task_config=TASK)
    _wait(path, "evaluation")
    if result is None:
        with open(path) as stream:
            result = json.load(stream)
    if not result.get("matchups"):
        raise RuntimeError(f"F24 evaluation has no matchups: {path}")
    return result


def _eval_pool(path, opponents, checkpoint, seed, capture_radius=0.10):
    results, batches = {}, []
    for i in range(0, len(opponents), 2):
        suffix = "_r015" if capture_radius != 0.10 else ""
        batch_path = f"{os.path.splitext(path)[0]}{suffix}_batch{i // 2:02d}.json"
        result = _eval(batch_path, opponents[i:i + 2], checkpoint, seed, capture_radius=capture_radius, cutoff=750)
        for opponent, metrics in result["matchups"].items():
            if metrics["episodes"] < 2048:
                raise RuntimeError(f"F24 evaluation has only {metrics['episodes']} episodes for {opponent}")
            results[opponent] = metrics
        batches.append(batch_path)
    if not results:
        raise RuntimeError(f"F24 evaluation pool is empty: {path}")
    return {"eval": {"batches": batches, "capture_radius": capture_radius}, "matchups": results}


def _target_jobs(campaign, seed, checkpoint, opponents):
    out = {}
    for group, policies in opponents.items():
        path = os.path.join(F24, "evals", f"v3b_s{seed}_{group}.json")
        out[group] = _eval_pool(path, policies, checkpoint, seed + 1000)
        out[f"{group}_r015"] = _eval_pool(path, policies, checkpoint, seed + 1000, capture_radius=0.15)
    mirror = f5._mirror(RLS_OVERRIDES)
    own = _exploiter(campaign, checkpoint, "v3b_own", V3_RULES, TASK, 12000 + seed, mirror)
    lure = _exploiter(campaign, checkpoint, "v3b_lure", f11.OLD_RULES_EXPLOITER, "PursuitEvasion", 12500 + seed,
                      mirror)
    for name, opp in (("own_exploiter", own), ("thesis_lure_exploiter", lure)):
        base = os.path.join(F24, "evals", f"v3b_s{seed}_{name}")
        out[name] = _eval(base + ".json", opp, checkpoint, seed + 1000)
        out[f"{name}_r015"] = _eval(base + "_r015.json", opp, checkpoint, seed + 1000, capture_radius=0.15)
    return out


def _switch_evals(seed, checkpoint):
    for job in f15._baseline_jobs():
        if job["seed"] != seed:
            continue
        path = os.path.join(F24, "evals", f"v3b_s{seed}_{job['group']}.json")
        _eval(path, job["policies"], checkpoint, job["rng_seed"], job["switch"])
        _eval(os.path.splitext(path)[0] + "_r015.json", job["policies"], checkpoint, job["rng_seed"], job["switch"],
              capture_radius=0.15)
        if job["kind"] == "switch":
            frozen = os.path.join(F24, "evals", f"v3b_s{seed}_{job['group']}_rls_frozen.json")
            _eval(frozen, job["policies"], checkpoint, job["rng_seed"], job["switch"], freeze=True)
            _eval(os.path.splitext(frozen)[0] + "_r015.json", job["policies"], checkpoint, job["rng_seed"],
                  job["switch"], freeze=True, capture_radius=0.15)


def run_f24c(args):
    if args.smoke:
        raise ValueError("Run F24 --smoke for the two-role gate.")
    if args.seeds != len(SEEDS):
        raise ValueError("F24c uses the registered seeds 0/100/200")
    for campaign in CAMPAIGNS:
        f14._wait_campaign(campaign)
    strong = [p for p in f8d._finals() if p["id"].split("/")[0] in f8d.COMMON]
    all_evaders = {c: f8.all_stages(c, "evader") for c in CAMPAIGNS}
    result = {}
    for seed, campaign in zip(SEEDS, CAMPAIGNS):
        checkpoint = _ck(campaign)
        cross = [p for c in CAMPAIGNS if c != campaign for p in all_evaders[c]]
        result[str(seed)] = _target_jobs(campaign, seed, checkpoint, {"strong": strong, "cross": cross})
        _switch_evals(seed, checkpoint)
    f23._summarize(os.path.join(F24, "results.json"), result)
    log_progress("F24 full evaluation jobs complete; results.json written")
