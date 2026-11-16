"""F15: add the missing F8 pursuer to F14's learned-policy switch evaluation."""

import hashlib
import json
import os

import torch

import common
from common import OUT, evaluate, log_progress
import f14_generalization_sp as f14
import f8_roncero_sp as f8

F15 = os.path.join(OUT, "F15_switch_baseline")
SEEDS = (0, 100, 200)


def _out(group):
    return os.path.join(F15, "evals", f"{group}.json")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _baseline_jobs():
    jobs = []
    for seed in SEEDS:
        campaign = f14.F8_CAMPS[SEEDS.index(seed)]
        checkpoint = f14._ck(campaign, 24)
        for direction in ("f8_to_f10", "f10_to_f8"):
            first, second = f14._switch_pair(seed, direction)
            jobs.append({
                "group": f"f8_s{seed}_{direction}_randomized",
                "seed": seed,
                "direction": direction,
                "kind": "switch",
                "policies": [first, second],
                "checkpoint": checkpoint,
                "rng_seed": seed + 1400,
                "switch": {
                    "first_id": first["id"],
                    "second_id": second["id"],
                    "times_s": [0.5, 1.0, 1.5],
                    "group_id": f"switch:{direction}:randomized",
                    "seed": seed + 401,
                },
            })
        first, second = f14._switch_pair(seed, "f8_to_f10")
        for policy in (first, second):
            jobs.append({
                "group": f"f8_s{seed}_fixed_{policy['id'].split('/')[0]}",
                "seed": seed,
                "direction": "both",
                "kind": "fixed",
                "policies": [policy],
                "checkpoint": checkpoint,
                "rng_seed": seed + 1400,
                "switch": None,
            })
    return jobs


def _reused_f14_jobs():
    campaigns = f14._stagee_campaigns(f14._selected_rules())
    generated = f14._adaptation_jobs(campaigns, stage=24, group_prefix="stageE_adapt")
    selected = {}
    for job in generated:
        path, campaign, model, seed, direction, switch_times, policies, checkpoint, extra, switch, group = job
        if model == "rls_frozen":
            model_key = "rls_frozen"
        else:
            model_key = model
        condition = (model_key, seed, direction if switch_times else "fixed", "switch" if switch_times else policies[0]["id"])
        if condition in selected:
            continue
        selected[condition] = {
            "path": path,
            "campaign": campaign,
            "model": model_key,
            "seed": seed,
            "direction": direction,
            "kind": "switch" if switch_times else "fixed",
            "policies": policies,
            "checkpoint": checkpoint,
            "extra": extra,
            "switch": switch,
            "group": group,
        }
    if len(selected) != 36:
        raise RuntimeError(f"F15 expected 36 unique reusable F14 cells, found {len(selected)}")
    return list(selected.values())


def _verify_saved_eval(job):
    path = job["path"]
    episode_path = os.path.splitext(path)[0] + "_episodes.pt"
    spec_path = os.path.splitext(path)[0] + "_opp.json"
    missing = [p for p in (path, episode_path, spec_path) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"F15 cannot reuse incomplete F14 evaluation: {missing}")
    with open(path) as stream:
        data = json.load(stream)
    if data["eval"]["num_envs"] != 4096:
        raise RuntimeError(f"F15 F14 reuse has unexpected num_envs: {path}")
    episodes = torch.load(episode_path, map_location="cpu", weights_only=False)["episodes"]
    count = len(episodes["episode_len"])
    if count < 2048:
        raise RuntimeError(f"F15 F14 reuse has only {count} episodes: {path}")
    if job["kind"] == "switch":
        if "switch_step" not in episodes or data["eval"].get("switch") != job["switch"]:
            raise RuntimeError(f"F15 F14 switch metadata mismatch: {path}")
    elif len(data["matchups"]) != 1:
        raise RuntimeError(f"F15 fixed-control evaluation is not a single matchup: {path}")
    return {
        "json": path,
        "episodes": episode_path,
        "opponent_spec": spec_path,
        "json_sha256": _sha256(path),
        "opponent_spec_sha256": _sha256(spec_path),
        "episodes_bytes": os.path.getsize(episode_path),
        "episode_count": count,
    }


def _smoke():
    f8_job = _baseline_jobs()[0]
    smoke_jobs = [("f8", f8_job, None)]
    campaigns = f14._stagee_campaigns(f14._selected_rules())
    estimator_jobs = f14._adaptation_jobs(campaigns, stage=24, group_prefix="stageE_adapt")
    for model in ("rls", "rls_frozen", "no_rls"):
        job = next(item for item in estimator_jobs
                   if item[2] == model and item[3] == 0 and item[4] == "f8_to_f10" and item[5] is not None)
        smoke_jobs.append((model, {
            "policies": job[6],
            "checkpoint": job[7],
            "switch": job[9],
        }, job[8]))

    for model, job, extra in smoke_jobs:
        group_id = f"f15:{model}-switch-smoke"
        path = os.path.join(F15, f"smoke_{model}.json")
        switch = {**job["switch"], "times_s": [0.1], "group_id": group_id, "seed": 399}
        if not os.path.isfile(path):
            evaluate(path, job["policies"], policy_ckpt=job["checkpoint"], num_envs=256, seed=399, cutoff=300,
                     extra=extra, switch=switch)
        if not os.path.isfile(path):
            raise RuntimeError(f"F15 {model} switch smoke evaluation failed")
        with open(path) as stream:
            data = json.load(stream)
        episode_path = os.path.splitext(path)[0] + "_episodes.pt"
        if not os.path.isfile(episode_path):
            raise FileNotFoundError(f"F15 switch smoke episode file missing: {episode_path}")
        episodes = torch.load(episode_path, map_location="cpu", weights_only=False)["episodes"]
        if data["eval"].get("switch", {}).get("group_id") != group_id:
            raise RuntimeError(f"F15 {model} smoke did not record its switch metadata")
        if len(episodes.get("episode_len", [])) == 0 or "switch_step" not in episodes:
            raise RuntimeError(f"F15 {model} smoke produced no valid switched episodes")


def _run_baseline_jobs(jobs):
    for job in jobs:
        path = _out(job["group"])
        if os.path.isfile(path):
            continue
        evaluate(path, job["policies"], policy_ckpt=job["checkpoint"], seed=job["rng_seed"],
                 switch=job["switch"])
    missing = [_out(job["group"]) for job in jobs if not os.path.isfile(_out(job["group"]))]
    if missing:
        raise RuntimeError(f"F15 baseline evaluations incomplete: {missing}")


def _fixed_record(job, source):
    with open(source) as stream:
        data = json.load(stream)
    matchup = next(iter(data["matchups"].values()))
    episodes_path = os.path.splitext(source)[0] + "_episodes.pt"
    episodes = torch.load(episodes_path, map_location="cpu", weights_only=False)["episodes"]
    return {
        "model": "f8",
        "seed": job["seed"],
        "direction": job["direction"],
        "kind": "fixed",
        "opponent": job["policies"][0]["id"],
        "episodes": len(episodes["episode_len"]),
        "fixed_j_r010": float(matchup["neutralization_rate"]),
        "fixed_j_r015": float(f14.rates_at(source, "neutralization_rate", 0.15)[job["policies"][0]["id"]]),
        **{f"fixed_{key}": value for key, value in matchup.get("outcomes", {}).items()},
    }


def _switch_records(job, model, provenance=None):
    path = job["path"] if "path" in job else _out(job["group"])
    with open(path) as stream:
        data = json.load(stream)
    matchup_id, matchup = next(iter(data["matchups"].items()))
    episode_path = os.path.splitext(path)[0] + "_episodes.pt"
    episodes = torch.load(episode_path, map_location="cpu", weights_only=False)["episodes"]
    total = int(matchup["episodes"])
    radius_015_j = f14.rates_at(path, "neutralization_rate", 0.15)[matchup_id]
    metrics = f14._episode_switch_metrics(path)
    records = []
    for metric in metrics:
        step = int(round(metric["switch_seconds"] / data["eval"]["step_dt"]))
        alive = (episodes["switch_step"] == step) & (episodes["episode_len"] > step)
        capture = episodes["capture"][alive] > 0.5
        wait = (episodes["episode_len"][alive].float() - float(step)).clamp_min(0) * data["eval"]["step_dt"]
        remaining_horizon = max(0, common.MAX_EPISODE_STEPS - step) * data["eval"]["step_dt"]
        penalized = torch.where(capture, wait, torch.full_like(wait, remaining_horizon))
        record = {
            "model": model,
            "seed": job["seed"],
            "direction": job["direction"],
            "kind": "switch",
            **metric,
            "episodes_total": total,
            "alive_fraction_at_switch": metric["episodes_alive_at_switch"] / total,
            "failure_penalized_post_switch_capture_time_s": float(penalized.mean()),
            "full_episode_j_r015": float(radius_015_j),
        }
        if provenance is not None:
            record["source"] = provenance
        records.append(record)
    return records


def _records(jobs, reused):
    records = []
    for job in reused:
        provenance = _verify_saved_eval(job)
        if job["kind"] == "switch":
            records.extend(_switch_records(job, job["model"], provenance))
        else:
            record = _fixed_record(job, job["path"])
            record["model"] = job["model"]
            record["source"] = provenance
            records.append(record)

    f8_sources = {}
    for job in jobs:
        path = _out(job["group"])
        f8_sources[job["group"]] = {
            "json": path,
            "json_sha256": _sha256(path),
            "opponent_spec": os.path.splitext(path)[0] + "_opp.json",
            "opponent_spec_sha256": _sha256(os.path.splitext(path)[0] + "_opp.json"),
            "checkpoint": job["checkpoint"],
            "checkpoint_sha256": _sha256(job["checkpoint"]),
            "episode_count": int(next(iter(json.load(open(path))["matchups"].values()))["episodes"]),
        }
        if job["kind"] == "switch":
            records.extend(_switch_records(job, "f8", f8_sources[job["group"]]))
        else:
            records.append(_fixed_record(job, path))
    return records, f8_sources


def _summarize(records):
    switch_records = [r for r in records if r["kind"] == "switch"]
    by_seed = {}
    for model in ("f8", "rls", "rls_frozen", "no_rls"):
        by_seed[model] = {}
        for seed in SEEDS:
            selected = [r for r in switch_records if r["model"] == model and r["seed"] == seed]
            if len(selected) != 6:
                raise RuntimeError(f"F15 expected 6 switch-time cells for {model} seed {seed}, found {len(selected)}")
            capture_counts = [r["post_switch_capture"] * r["episodes_alive_at_switch"] for r in selected]
            captured = sum(capture_counts)
            by_seed[model][str(seed)] = {
                key: sum(r[key] for r in selected) / len(selected)
                for key in ("post_switch_neutralization", "post_switch_capture", "full_episode_j",
                            "full_episode_j_r015", "alive_fraction_at_switch",
                            "failure_penalized_post_switch_capture_time_s")
            }
            by_seed[model][str(seed)]["conditional_capture_time_s"] = (
                sum(r["mean_seconds_after_switch_to_capture"] * count
                    for r, count in zip(selected, capture_counts) if r["mean_seconds_after_switch_to_capture"] is not None)
                / captured
                if captured else None
            )
    ending_rates = {}
    for model in ("f8", "rls", "rls_frozen", "no_rls"):
        ending_rates[model] = {}
        for key in ("full_capture", "full_pursuer_crash", "full_evader_crash", "full_evader_oob",
                    "full_timeout", "full_pursuer_oob"):
            seed_means = []
            for seed in SEEDS:
                # Full-episode outcomes repeat for each switch-time bin in the same evaluation.
                distinct = {}
                for row in switch_records:
                    if row["model"] == model and row["seed"] == seed:
                        distinct[row["direction"]] = row[key]
                seed_means.append(sum(distinct.values()) / len(distinct))
            ending_rates[model][key.removeprefix("full_")] = sum(seed_means) / len(seed_means)
    fixed_means = {}
    for model in ("f8", "rls", "rls_frozen", "no_rls"):
        fixed_means[model] = {}
        for metric in ("fixed_j_r010", "fixed_j_r015", "fixed_capture"):
            seed_means = []
            for seed in SEEDS:
                selected = [r for r in records if r["model"] == model and r["seed"] == seed and r["kind"] == "fixed"]
                if len(selected) != 2:
                    raise RuntimeError(f"F15 expected two unique fixed controls for {model} seed {seed}")
                seed_means.append(sum(r[metric] for r in selected) / len(selected))
            fixed_means[model][metric] = sum(seed_means) / len(seed_means)
    return by_seed, ending_rates, fixed_means


def _write_summary(results, summary_path):
    means = {}
    for model, seeds in results["switch_means_by_seed"].items():
        means[model] = {
            key: sum(row[key] for row in seeds.values()) / len(seeds)
            for key in ("post_switch_neutralization", "post_switch_capture", "full_episode_j",
                        "full_episode_j_r015", "alive_fraction_at_switch",
                        "failure_penalized_post_switch_capture_time_s", "conditional_capture_time_s")
        }
    contrasts = {}
    for model in ("f8", "rls_frozen", "no_rls"):
        contrasts[model] = {
            key: [results["switch_means_by_seed"][model][str(seed)][key] -
                  results["switch_means_by_seed"]["rls"][str(seed)][key] for seed in SEEDS]
            for key in ("post_switch_neutralization", "post_switch_capture", "full_episode_j")
        }
    results["switch_means"] = means
    results["paired_contrasts_vs_rls"] = contrasts
    rows = ["# F15: F8 pursuer on learned-policy switches", "",
            "Descriptive means over three training seeds. Post-switch outcomes condition on episodes alive at the recorded switch; do not interpret them as same-state counterfactuals. Post-switch J is scored at 0.10 m; 0.15 m is a full-episode exact re-score only because it can change which episodes would have survived to switch.",
            "| Arm | Post-switch J | Capture | Full J (0.10 m) | Full J (0.15 m) | Alive at switch | Capture time (s) | Failure-penalized capture time (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in ("f8", "rls", "rls_frozen", "no_rls"):
        row = means[model]
        rows.append(f"| {model} | {row['post_switch_neutralization']:.3f} | {row['post_switch_capture']:.3f} | "
                    f"{row['full_episode_j']:.3f} | {row['full_episode_j_r015']:.3f} | "
                    f"{row['alive_fraction_at_switch']:.3f} | "
                    f"{row['conditional_capture_time_s']:.3f} | "
                    f"{row['failure_penalized_post_switch_capture_time_s']:.3f} |")
    rows += ["", "Fixed-policy controls (mean of seed-balanced F8 and F10 fixed-policy jobs):",
             "| Arm | J (0.10 m) | J (0.15 m) | Capture (0.10 m) |",
             "|---|---:|---:|---:|"]
    for model, row in results["fixed_control_means"].items():
        rows.append(f"| {model} | {row['fixed_j_r010']:.3f} | {row['fixed_j_r015']:.3f} | {row['fixed_capture']:.3f} |")
    rows += ["", "Paired per-seed differences versus RLS (control minus RLS), seeds 0/100/200:",
             "| Control | Post-switch J | Capture | Full-episode J |", "|---|---|---|---|"]
    for model in ("f8", "rls_frozen", "no_rls"):
        c = contrasts[model]
        rows.append(f"| {model} | {c['post_switch_neutralization']} | {c['post_switch_capture']} | "
                    f"{c['full_episode_j']} |")
    rows += ["", "Full-episode ending rates across randomized-switch conditions (flags can overlap):",
             "| Arm | Capture | Pursuer crash | Evader crash | Evader OOB | Timeout | Pursuer OOB |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for model, row in results["full_ending_means"].items():
        rows.append(f"| {model} | {row['capture']:.3f} | {row['pursuer_crash']:.3f} | "
                    f"{row['evader_crash']:.3f} | {row['evader_oob']:.3f} | "
                    f"{row['timeout']:.3f} | {row['pursuer_oob']:.3f} |")
    rows += ["", "Seed-level switch means:", "| Arm | Seed | Post-switch J | Capture | Full J | Alive at switch |",
             "|---|---:|---:|---:|---:|---:|"]
    for model, seeds in results["switch_means_by_seed"].items():
        for seed, row in seeds.items():
            rows.append(f"| {model} | {seed} | {row['post_switch_neutralization']:.3f} | "
                        f"{row['post_switch_capture']:.3f} | {row['full_episode_j']:.3f} | "
                        f"{row['alive_fraction_at_switch']:.3f} |")
    with open(summary_path, "w") as stream:
        stream.write("\n".join(rows) + "\n")
    return results


def run_f15(args):
    if args.smoke:
        _smoke()
        return
    log_progress("F15: verifying F14 reuse and running/reusing F8 switch evaluations")
    if f14._selected_rules() != "v2d":
        raise RuntimeError("F15 reuse manifest expects the completed F14 v2d Stage E campaigns")
    reused = _reused_f14_jobs()
    reuse_manifest = []
    for job in reused:
        data = _verify_saved_eval(job)
        data.update({
            "model": job["model"],
            "seed": job["seed"],
            "direction": job["direction"],
            "kind": job["kind"],
            "checkpoint": job["checkpoint"],
            "checkpoint_sha256": _sha256(job["checkpoint"]),
            "extra_overrides": job["extra"],
        })
        reuse_manifest.append(data)
    jobs = _baseline_jobs()
    _smoke()
    _run_baseline_jobs(jobs)
    records, f8_sources = _records(jobs, reused)
    switch_means_by_seed, full_ending_means, fixed_control_means = _summarize(records)
    results = {
        "seeds": list(SEEDS),
        "records": records,
        "switch_means_by_seed": switch_means_by_seed,
        "full_ending_means": full_ending_means,
        "fixed_control_means": fixed_control_means,
        "f14_reused_cells": reuse_manifest,
        "f8_evaluation_sources": f8_sources,
    }
    common.save_json(os.path.join(F15, "manifest.json"), {
        "experiment": "F15_switch_baseline",
        "seeds": list(SEEDS),
        "rules_for_f8_pursuer": list(f8.RULES),
        "evaluation": {
            "num_envs": 4096,
            "seed_rule": "target seed + 1400",
            "switch_seed_rule": "target seed + 401",
            "switch_times_s": [0.5, 1.0, 1.5],
            "radii_m": [0.10, 0.15],
            "primary_post_switch_radius_m": 0.10,
        },
        "f14_reused_cells": reuse_manifest,
        "new_f8_evaluation_jobs": jobs,
        "new_f8_evaluation_sources": f8_sources,
    })
    results = _write_summary(results, os.path.join(F15, "summary.md"))
    common.save_json(os.path.join(F15, "results.json"), results)
    log_progress("F15 switch baseline evaluations and analysis complete")
