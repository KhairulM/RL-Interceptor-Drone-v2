"""F23: v3 self-play with the learned acceleration estimator and online RLS."""

import json
import os
import statistics

import common
from common import OUT, evaluate, log_progress, train
from d_sampling import EXPLOIT_FRAMES
from f2_final import _done, run_campaign
import f5_acc_sp as f5
import f8_roncero_sp as f8
import f8d_heldout as f8d
import f11_v2_sp as f11
import f14_generalization_sp as f14
import f15_switch_baseline as f15
import f21_rls_ablation as f21

F23 = os.path.join(OUT, "F23_v3_rls")
TASK = "PursuitEvasionV3"
SEEDS = (0, 100, 200)
CAMPAIGNS = tuple(f"thesis-F23-v3-acc-rls-s{s}" for s in SEEDS)
SMOKE = "thesis-F23-smoke-v3-acc-rls"
V3_RULES = ["algo.gae_cut_at_truncation=true"]
ROLE_OVERRIDES = f21.roles("baseline")
RLS_OVERRIDES = f5.rls_overrides()
STAGES = 26


def _cfg(seed, stages):
    return {
        "task": TASK,
        "seed": seed,
        "n_stages": stages,
        "max_total_stages": stages,
        "frames_per_stage": 50_000_000,
    }


def _ck(campaign, stage=24):
    return f14._ck(campaign, stage)


def _wait(path, what):
    if not os.path.isfile(path):
        raise RuntimeError(f"F23 {what} did not complete: {path}")


def _smoke():
    run_campaign(SMOKE, V3_RULES, _cfg(0, 2), smoke_stages=2, role_overrides=ROLE_OVERRIDES)
    _wait(_done(SMOKE), "two-role smoke campaign")
    common.CAMPAIGNS[SMOKE] = SMOKE
    for role in ("pursuer", "evader"):
        policies = common.rl_policies(SMOKE, role)
        if not policies:
            raise RuntimeError(f"F23 smoke produced no {role} policy")
    result = evaluate(
        os.path.join(F23, "smoke", "v3_rls_live.json"),
        common.rl_policies(SMOKE, "evader"),
        policy_ckpt=f14._ck(SMOKE, 0),
        num_envs=256,
        cutoff=300,
        extra=RLS_OVERRIDES,
        task_config=TASK,
    )
    if result is None or not result.get("matchups"):
        raise RuntimeError("F23 smoke RLS evaluation failed")
    if not all("capture_rate" in row for row in result["matchups"].values()):
        raise RuntimeError("F23 smoke RLS evaluation has incomplete matchup outputs")
    log_progress("F23 v3 two-role training and live-RLS evaluation smoke passed")


def run_f23(args):
    if args.seeds != len(SEEDS):
        raise ValueError("F23 uses the registered seeds 0/100/200")
    os.makedirs(F23, exist_ok=True)
    _smoke()
    if args.smoke:
        return
    for seed, campaign in zip(SEEDS, CAMPAIGNS):
        run_campaign(campaign, V3_RULES, _cfg(seed, STAGES), role_overrides=ROLE_OVERRIDES)
        _wait(_done(campaign), f"seed {seed} campaign")
        log_progress(f"F23 campaign complete: {campaign}")


def _exploiter(target_campaign, target_ck, tag, rules, task_config, seed, opponent_extra):
    path = os.path.join(F23, "ckpt", f"{target_campaign}_{tag}.pt")
    ckpt = train(
        path,
        [{"id": f"f23/pursuer:{target_campaign}", "kind": "rl", "checkpoint_path": target_ck,
          "metadata": {}}],
        EXPLOIT_FRAMES,
        role="evader",
        seed=seed,
        extra=list(rules) + list(opponent_extra),
        task_config=task_config,
    )
    if ckpt is None:
        _wait(path, f"{tag} exploiter")
        ckpt = path
    return [{"id": f"f23/evader:{target_campaign}_{tag}", "kind": "rl",
             "checkpoint_path": ckpt, "metadata": {}}]


def _eval(path, opponents, checkpoint, seed, switch=None, freeze=False, capture_radius=0.10, cutoff=1500):
    extra = [o for o in RLS_OVERRIDES if not o.startswith("task.adapt.freeze=")]
    extra.append(f"task.adapt.freeze={str(freeze).lower()}")
    extra.append(f"task.success_radius_eval={capture_radius}")
    result = evaluate(
        path,
        opponents,
        policy_ckpt=checkpoint,
        num_envs=4096,
        seed=seed,
        cutoff=cutoff,
        extra=extra,
        switch=switch,
        task_config=TASK,
    )
    _wait(path, "evaluation")
    if result is None:
        with open(path) as stream:
            result = json.load(stream)
    if not result.get("matchups"):
        raise RuntimeError(f"F23 evaluation has no matchups: {path}")
    return result


def _eval_pool(path, opponents, checkpoint, seed, capture_radius=0.10):
    results = {}
    batches = []
    for batch_index in range(0, len(opponents), 2):
        batch = opponents[batch_index:batch_index + 2]
        suffix = "_r015" if capture_radius != 0.10 else ""   # else the 0.15 m pass reuses the 0.10 m batch files
        batch_path = f"{os.path.splitext(path)[0]}{suffix}_batch{batch_index // 2:02d}.json"
        result = _eval(batch_path, batch, checkpoint, seed, capture_radius=capture_radius, cutoff=750)
        for opponent, metrics in result["matchups"].items():
            if metrics["episodes"] < 2048:
                raise RuntimeError(
                    f"F23 evaluation has only {metrics['episodes']} episodes for {opponent}; expected >=2048")
            results[opponent] = metrics
        batches.append(batch_path)
    if not results:
        raise RuntimeError(f"F23 evaluation pool is empty: {path}")
    return {"eval": {"batches": batches, "capture_radius": capture_radius}, "matchups": results}


def _target_jobs(campaign, seed, checkpoint, opponents, historical=False):
    prefix = "historical_v2d" if historical else "v3"
    out = {}
    for group, policies in opponents.items():
        path = os.path.join(F23, "evals", f"{prefix}_s{seed}_{group}.json")
        out[group] = _eval_pool(path, policies, checkpoint, seed + 1000)
        out[f"{group}_r015"] = _eval_pool(path, policies, checkpoint, seed + 1000, capture_radius=0.15)
    own_rules = f14._campaign_rules("v2d") if historical else V3_RULES
    own_task = "PursuitEvasion" if historical else TASK
    own_extra = f5._mirror(RLS_OVERRIDES)
    own = _exploiter(campaign, checkpoint, f"{prefix}_own", own_rules, own_task,
                      12000 + seed, own_extra)
    lure = _exploiter(campaign, checkpoint, f"{prefix}_lure", f11.OLD_RULES_EXPLOITER,
                       "PursuitEvasion", 12500 + seed, own_extra)
    out["own_exploiter"] = _eval(
        os.path.join(F23, "evals", f"{prefix}_s{seed}_own_exploiter.json"), own,
        checkpoint, seed + 1000)
    out["own_exploiter_r015"] = _eval(
        os.path.join(F23, "evals", f"{prefix}_s{seed}_own_exploiter_r015.json"), own,
        checkpoint, seed + 1000, capture_radius=0.15)
    out["thesis_lure_exploiter"] = _eval(
        os.path.join(F23, "evals", f"{prefix}_s{seed}_thesis_lure.json"), lure,
        checkpoint, seed + 1000)
    out["thesis_lure_exploiter_r015"] = _eval(
        os.path.join(F23, "evals", f"{prefix}_s{seed}_thesis_lure_r015.json"), lure,
        checkpoint, seed + 1000, capture_radius=0.15)
    return out


def _switch_evals(campaign, seed, checkpoint, historical=False):
    prefix = "historical_v2d" if historical else "v3"
    for job in f15._baseline_jobs():
        if job["seed"] != seed:
            continue
        path = os.path.join(F23, "evals", f"{prefix}_s{seed}_{job['group']}.json")
        _eval(path, job["policies"], checkpoint, job["rng_seed"], job["switch"])
        _eval(os.path.splitext(path)[0] + "_r015.json", job["policies"], checkpoint,
              job["rng_seed"], job["switch"], capture_radius=0.15)
        if job["kind"] == "switch":
            frozen_path = os.path.join(F23, "evals", f"{prefix}_s{seed}_{job['group']}_rls_frozen.json")
            _eval(frozen_path, job["policies"], checkpoint, job["rng_seed"], job["switch"], freeze=True)
            _eval(os.path.splitext(frozen_path)[0] + "_r015.json", job["policies"], checkpoint,
                  job["rng_seed"], job["switch"], freeze=True, capture_radius=0.15)


def _summarize(path, rows):
    output = {"evaluations": rows, "metrics": {}}
    for family, result in rows.items():
        metrics = {}
        for group, data in result.items():
            matchups = data["matchups"]
            metrics[group] = {
                opponent: {
                    key: matchup.get(key, matchup.get("outcomes", {}).get(key))
                    for key in ("neutralization_rate", "capture_rate", "timeout",
                                "pursuer_crash", "pursuer_oob", "evader_crash", "evader_oob")
                } | {"episodes": matchup.get("episodes"),
                     "outcomes": matchup.get("outcomes"),
                     "outcome_counts": matchup.get("outcome_counts"),
                     "reward_components": matchup.get("reward_components"),
                     "reward_components_by_capture": matchup.get("reward_components_by_capture")}
                for opponent, matchup in matchups.items()
            }
        output["metrics"][family] = metrics
    common.save_json(path, output)


def run_f23c(args):
    if args.smoke:
        raise ValueError("Run F23 --smoke for the two-role gate.")
    if args.seeds != len(SEEDS):
        raise ValueError("F23c uses the registered seeds 0/100/200")
    for campaign in CAMPAIGNS:
        _wait(_done(campaign), "campaign")
    strong = [p for p in f8d._finals() if p["id"].split("/")[0] in f8d.COMMON]
    all_evaders = {c: f8.all_stages(c, "evader") for c in CAMPAIGNS}
    result = {}
    for seed, campaign in zip(SEEDS, CAMPAIGNS):
        checkpoint = _ck(campaign)
        cross = [p for c in CAMPAIGNS if c != campaign for p in all_evaders[c]]
        result[str(seed)] = _target_jobs(
            campaign, seed, checkpoint, {"strong": strong, "cross": cross})
        _switch_evals(campaign, seed, checkpoint)

    historical_campaigns = f14._stagee_campaigns("v2d")["rls"]
    for seed, campaign in zip(SEEDS, historical_campaigns):
        f14._wait_campaign(campaign)
        checkpoint = _ck(campaign)
        result[f"historical_{seed}"] = _target_jobs(
            campaign, seed, checkpoint,
            {"strong": strong, "cross": [p for c in CAMPAIGNS if c != CAMPAIGNS[SEEDS.index(seed)]
                                          for p in all_evaders[c]]},
            historical=True)
        _switch_evals(campaign, seed, checkpoint, historical=True)
    _summarize(os.path.join(F23, "results.json"), result)
    log_progress("F23 full evaluation jobs complete; results.json written")
