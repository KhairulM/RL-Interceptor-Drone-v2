"""F21: RLS timescale/covariance ablation in full v2d self-play (research plan F21)."""

import json
import math
import os
import re
import statistics as st
from typing import TypedDict

import torch

import common
from common import OUT, evaluate, log_progress, save_json
from f2_final import _done, run_campaign
import f5_acc_sp as f5
import f8_roncero_sp as f8
import f8d_heldout as f8d
import f11_v2_sp as f11
import f14_generalization_sp as f14
import f15_switch_baseline as f15
from f17_roncero_acc import _wait
import f18_trajectories as f18
import f19_plain26 as f19
from rescore import rates_at

F21 = os.path.join(OUT, "F21_rls_ablation")
SEEDS = (0, 100, 200)
PARAMS = {
    "baseline": (0.999, 0.01, 1.0),
    "fast": (0.98, 0.01, 1.0),
    "cap": (0.999, 0.01, 10.0),
    "fast_cap": (0.98, 0.01, 10.0),
    "fast_cap_gain": (0.98, 0.1, 10.0),
}
NEW = tuple(a for a in PARAMS if a != "baseline")
ARMS = tuple(PARAMS) + ("frozen_train",)
CONTRASTS = (("fast", "baseline"), ("cap", "baseline"), ("fast_cap", "fast"),
             ("fast_cap", "cap"), ("fast_cap_gain", "fast_cap"))
WINDOWS = ((0.0, 0.5), (0.5, 1.5), (1.5, 3.0))


class SwitchJob(TypedDict):
    seed: int
    model: str
    kind: str
    direction: str
    extra: list[str]
    checkpoint: str
    policies: list[dict]
    switch: dict | None
    rng_seed: int
    path: str


def campaign(arm, seed):
    if arm in ("baseline", "frozen_train"):
        key = "rls" if arm == "baseline" else "no_rls"
        return f14._stagee_campaigns("v2d")[key][SEEDS.index(seed)]
    return f"thesis-F21-v2d-{arm}-s{seed}"


def overrides(arm, freeze=False):
    lam, delta, cap = PARAMS["baseline" if arm == "frozen_train" else arm]
    changed = {"task.adapt.forgetting": lam, "task.adapt.delta": delta, "task.adapt.p_max_ratio": cap,
               "task.adapt.reset_on_episode": "true", "task.adapt.freeze": str(freeze or arm == "frozen_train").lower()}
    return [o for o in f5.rls_overrides() if o.split("=", 1)[0] not in changed] + [
        f"{k}={v}" for k, v in changed.items()]


def roles(arm):
    pov = overrides(arm)
    return {"pursuer": pov, "evader": f5._mirror(pov)}


def specs(arms=NEW):
    return [(a, s) for a in arms for s in SEEDS]


def lane_specs(arms=NEW):
    lane = os.environ.get("F21_LANE", "all")
    if lane not in ("all", "0", "1"):
        raise ValueError(f"F21_LANE must be all, 0 or 1, got {lane!r}")
    return [item for item in specs(arms) if lane == "all" or specs(ARMS).index(item) % 2 == int(lane)]


def _out(arm, seed, group):
    return os.path.join(F21, "evals", f"{arm}_s{seed}_{group}.json")


def common_cross():
    return f14._evaders(tuple(f19.CAMPS[s] for s in SEEDS))


def audit():
    """Pin reused checkpoint lineage and the unchanged training contract before launching."""
    sources = {}
    driver = open(os.path.join(OUT, "driver_log.txt")).read().splitlines()
    rules = f14._campaign_rules("v2d")
    for arm, seed in specs(("baseline", "frozen_train")):
        c = campaign(arm, seed)
        if not os.path.isfile(_done(c)):
            raise RuntimeError(f"F21 baseline campaign incomplete: {c}")
        ck = f14._ck(c, 24)
        manifest = os.path.join(common.SELFPLAY_OUT, c, "stageE_resume.json")
        resume = json.load(open(manifest))
        if resume["seed"] != seed or resume["continuation"] != {
                "n_stages": 26, "max_total_stages": 26, "frames_per_stage": 50_000_000}:
            raise RuntimeError(f"F21 incompatible baseline continuation: {manifest}")
        source_ck = resume["stage12_checkpoint"]
        if f15._sha256(source_ck) != resume["stage12_checkpoint_sha256"]:
            raise RuntimeError(f"F21 baseline lineage hash mismatch: {source_ck}")
        history = {}
        for lineage in (resume["source_campaign"], c):
            commands = [line.split(" start: ", 1)[1] for line in driver if f"campaign {lineage} start:" in line]
            if not commands:
                raise RuntimeError(f"F21 missing historic baseline config: {lineage}")
            command = commands[-1]
            expected_roles = f14._roles("rls" if arm == "baseline" else "no_rls")
            for role, expected in expected_roles.items():
                match = re.search(rf"role_task_overrides\.{role}=\[([^]]*)\]", command)
                if match is None or match.group(1).split(",") != expected:
                    raise RuntimeError(f"F21 historic adapter mismatch: {lineage} {role}")
            if not all(rule in command for rule in rules) or not all(
                    token in command.split() for token in ("frames_per_stage=50000000", f"seed={seed}",
                                                           "pursuer_sampler=hedge", "evader_sampler=hedge")):
                raise RuntimeError(f"F21 historic training config mismatch: {lineage}")
            history[lineage] = command
        sources[c] = {"checkpoint": ck, "sha256": f15._sha256(ck),
                      "resume_manifest": resume, "resume_manifest_sha256": f15._sha256(manifest),
                      "historic_commands": history}
    params = f18._rls_params()
    if (params["forgetting"], params["delta"], params["horizon"], params["history"]) != (0.999, 0.01, 10, 10):
        raise RuntimeError(f"F21 reused estimator parameters changed: {params}")
    bank = common_cross()
    if len(bank) != 39 or len({p["id"] for p in bank}) != 39:
        raise RuntimeError("F21 common cross bank must have 39 distinct plain-26 evaders")
    provenance = {
        "parameters": PARAMS, "training_config": f11.cfg(0, 26), "rules": f14._campaign_rules("v2d"),
        "role_overrides": {a: roles(a) for a in PARAMS}, "sources": sources,
        "estimator_sha256": f15._sha256(params["model_path"]),
        "common_cross": [{"id": p["id"], "checkpoint_sha256": f15._sha256(p["checkpoint_path"])} for p in bank],
    }
    path = os.path.join(F21, "audit.json")
    if os.path.isfile(path) and json.load(open(path)) != json.loads(json.dumps(provenance)):
        raise RuntimeError("F21 inputs changed since audit; refusing incompatible resume")
    if not os.path.isfile(path):
        save_json(path, provenance)


def _smoke(arms=NEW):
    for arm in arms:
        c = f"thesis-F21-smoke-{arm}"
        run_campaign(c, f14._campaign_rules("v2d"), f11.cfg(0, 2), smoke_stages=2, role_overrides=roles(arm))
        f14._wait_campaign(c)
        for freeze in (False, True):
            path = os.path.join(F21, "smoke", f"{arm}_{'frozen' if freeze else 'live'}.json")
            evaluate(path, f8.all_stages(c, "evader"), policy_ckpt=f14._ck(c, 0), num_envs=256, cutoff=300,
                     extra=overrides(arm, freeze))
            _wait(path, "F21 smoke evaluation")
            matchups = json.load(open(path))["matchups"]
            if not matchups or any(m["episodes"] < 1 for m in matchups.values()):
                raise RuntimeError(f"F21 smoke has no episodes: {path}")
    log_progress("F21 all four arm smokes passed (both roles, live/frozen evaluations)")


def run_f21(args):
    if args.seeds != 3:
        raise ValueError("F21 requires the registered seeds 0/100/200")
    audit()
    _smoke()
    if args.smoke:
        return
    for arm, seed in lane_specs():
        c = campaign(arm, seed)
        run_campaign(c, f14._campaign_rules("v2d"), f11.cfg(seed, 26), role_overrides=roles(arm))
        f14._wait_campaign(c)


def _exploiters(arm, seed):
    c, ck = campaign(arm, seed), f14._ck(campaign(arm, seed), 24)
    i = NEW.index(arm)
    base = 10000 + 1000 * i + seed
    return {
        tag: f14._exploiter(os.path.join(F21, "ckpt", f"{arm}_s{seed}_{tag}.pt"), f"f21/pursuer:{c}",
                           c, ck, rules, rng, opponent_extra=f5._mirror(overrides(arm)))
        for tag, rules, rng in (("own", f14._campaign_rules("v2d"), base),
                               ("lure", f11.OLD_RULES_EXPLOITER, base + 500))
    }


def switch_jobs(arm, seed, freeze=False) -> list[SwitchJob]:
    model = arm + ("_frozen" if freeze else "")
    jobs: list[SwitchJob] = []
    for j in f15._baseline_jobs():
        if j["seed"] == seed:
            jobs.append({
                "seed": seed, "model": model, "kind": j["kind"], "direction": j["direction"],
                "extra": overrides(arm, freeze), "checkpoint": f14._ck(campaign(arm, seed), 24),
                "policies": j["policies"], "switch": j["switch"], "rng_seed": seed + 1400,
                "path": _out(arm, seed, f"{model}_{j['group'].removeprefix(f'f8_s{seed}_')}"),
            })
    return jobs


def _reused_switch(arm, seed, freeze=False):
    model = "no_rls" if arm == "frozen_train" else ("rls_frozen" if freeze else "rls")
    return [j for j in f15._reused_f14_jobs() if j["model"] == model and j["seed"] == seed]


def _reused_general(arm, seed, group, freeze=False):
    prefix = "stageE_" + ("frozen_" if freeze else "")
    return f14._out(campaign(arm, seed), prefix + ("thesis" if group == "lure" else group))


def _run_eval(path, arm, seed, policies, freeze=False, switch=None, rng=0, record=False):
    extra = overrides(arm, freeze)
    if record:
        extra += [f"eval_traj_out={os.path.splitext(path)[0]}_traj.pt", "eval_traj_envs=128"]
    evaluate(path, policies, policy_ckpt=f14._ck(campaign(arm, seed), 24), extra=extra, seed=rng, switch=switch)
    _wait(path, "F21 evaluation")
    if record and not os.path.isfile(os.path.splitext(path)[0] + "_traj.pt"):
        raise RuntimeError(f"F21 missing diagnostic recording: {path}")


def _evaluate_arm(arm, seed):
    cross = common_cross()
    strong = [p for p in f8d._finals() if p["id"].split("/")[0] in f8d.COMMON]
    if arm in NEW:
        probes = {"strong": strong, **_exploiters(arm, seed)}
        for group, opps in probes.items():
            _run_eval(_out(arm, seed, group), arm, seed, opps)
        _run_eval(_out(arm, seed, "frozen_strong"), arm, seed, strong, freeze=True)
    else:
        for group in ("strong", "own", "lure"):
            path = _reused_general(arm, seed, group)
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
    _run_eval(_out(arm, seed, "cross"), arm, seed, cross)
    if arm != "frozen_train":
        _run_eval(_out(arm, seed, "frozen_cross"), arm, seed, cross, freeze=True)
    for freeze in ((False, True) if arm != "frozen_train" else (False,)):
        if arm not in NEW:
            for j in _reused_switch(arm, seed, freeze):
                f15._verify_saved_eval(j)
        else:
            for j in switch_jobs(arm, seed, freeze):
                _run_eval(j["path"], arm, seed, j["policies"], freeze, j["switch"], j["rng_seed"],
                          record=not freeze and j["kind"] == "switch")
    if arm == "baseline":
        for j in switch_jobs(arm, seed):
            if j["kind"] == "switch":
                _run_eval(j["path"], arm, seed, j["policies"], switch=j["switch"], rng=j["rng_seed"], record=True)
    if arm != "frozen_train":
        diagnostics = {}
        for j in switch_jobs(arm, seed):
            if j["kind"] == "switch":
                diagnostics[j["direction"]] = replay_switch(j["path"], arm)
        save_json(_out(arm, seed, "diagnostics"), diagnostics)
    save_json(_out(arm, seed, "DONE"), {"arm": arm, "seed": seed})


def replay_switch(path, arm):
    data = torch.load(os.path.splitext(path)[0] + "_traj.pt", map_location="cpu", weights_only=False)
    episodes = torch.load(os.path.splitext(path)[0] + "_episodes.pt", map_location="cpu",
                          weights_only=False)["episodes"]
    steps = []
    for env in data["envs"]:
        values = episodes["switch_step"][episodes["env"] == env].unique()
        if len(values) != 1:
            raise RuntimeError(f"F21 cannot align recorded env {env}: {path}")
        steps.append(int(values[0]))
    return replay_arrays(data["z"], data["done"], torch.tensor(steps), float(data["step_dt"]), arm)


def error_window_mask(age, switch_steps, horizon, lo, hi, dt):
    origin = (age - horizon - switch_steps).float() * dt
    return (age >= horizon) & (origin >= lo) & (origin < hi)


def replay_arrays(z, done, switch_steps, dt, arm):
    oa, params = f18._adapter_module(), f18._rls_params()
    lam, delta, cap = PARAMS[arm]
    n = z.shape[1]
    result = {}
    for freeze in (False, True):
        ad = oa.OpponentAdapter(n, kind="nf", history=params["history"], horizon=params["horizon"],
                               model_path=params["model_path"], forgetting=lam, delta=delta,
                               p_max_ratio=cap, target="dvel", freeze=freeze)
        age = torch.zeros(n, dtype=torch.long)
        theta_switch = ad.theta.clone()
        sums, counts = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.long)
        snapshots = {sec: {"drift_initial": [], "drift_switch": [], "trace_ratio": []} for sec in (0.5, 1.5, 3.0)}
        for k in range(len(z)):
            if k:
                ended = done[k - 1].nonzero().squeeze(-1)
                ad.reset(ended)
                age[ended] = 0
                theta_switch[ended] = ad.theta0
            at_switch = age == switch_steps
            theta_switch[at_switch] = ad.theta[at_switch]
            ad.step(z[k])
            if not torch.isfinite(ad.err).all() or not torch.isfinite(ad.theta).all():
                raise RuntimeError("F21 nonfinite estimator replay")
            # z at a reset is already post-reset; pending targets were cleared above.
            for b, (lo, hi) in enumerate(WINDOWS):
                valid = ad.err_valid & error_window_mask(age, switch_steps, ad.H, lo, hi, dt)
                sums[b] += ad.err[valid].double().sum()
                counts[b] += valid.sum()
            for sec, snapshot in snapshots.items():
                hit = age == switch_steps + round(sec / dt)
                if hit.any():
                    norm = ad.theta0.norm().clamp_min(1e-12)
                    snapshot["drift_initial"].extend(((ad.theta[hit] - ad.theta0).flatten(1).norm(dim=1) / norm).tolist())
                    snapshot["drift_switch"].extend(((ad.theta[hit] - theta_switch[hit]).flatten(1).norm(dim=1) / norm).tolist())
                    snapshot["trace_ratio"].extend((ad.P[hit].diagonal(dim1=-2, dim2=-1).sum(-1) / ad.p_trace_max).tolist())
            age += 1
        result["frozen" if freeze else "live"] = {
            "error_windows": [{"start_s": lo, "end_s": hi, "count": int(counts[i]),
                               "mean_mps": float(sums[i] / counts[i]) if counts[i] else None}
                              for i, (lo, hi) in enumerate(WINDOWS)],
            "snapshots": {str(sec): {"count": len(v["drift_initial"]),
                                     **{key + "_median": st.median(vals) if vals else None for key, vals in v.items()},
                                     "cap_occupancy": st.mean(x >= 0.999 for x in v["trace_ratio"]) if v["trace_ratio"] else None}
                          for sec, v in snapshots.items()},
        }
    return result


def _switch_score(arm, seed, freeze=False):
    jobs = switch_jobs(arm, seed, freeze) if arm in NEW else _reused_switch(arm, seed, freeze)
    rows = [r for j in jobs if j["kind"] == "switch" for r in f15._switch_records(j, arm)]
    if len(rows) != 6:
        raise RuntimeError(f"F21 expected six switch cells: {arm} s{seed}")
    keys = ("post_switch_capture", "post_switch_neutralization", "alive_fraction_at_switch",
            "full_episode_j", "failure_penalized_post_switch_capture_time_s")
    fixed = [json.load(open(j["path"]))["matchups"] for j in jobs if j["kind"] == "fixed"]
    return {**{k: st.mean(r[k] for r in rows) for k in keys},
            "fixed_capture": st.mean(next(iter(m.values()))["capture_rate"] for m in fixed)}


def score(arm, seed, radius):
    def path(group, freeze=False):
        if group == "cross" or arm in NEW:
            return _out(arm, seed, ("frozen_" if freeze else "") + group)
        return _reused_general(arm, seed, group, freeze)

    def mean_rate(group, freeze=False, key="neutralization_rate"):
        values = rates_at(path(group, freeze), key, radius)
        return f14._group_cross(values, "") if group == "cross" else st.mean(values.values())

    strong, cross = mean_rate("strong"), mean_rate("cross")
    out = {"strong_j": strong, "cross_j": cross, "g1": (strong + cross) / 2,
           "own_j": mean_rate("own"), "lure_j": mean_rate("lure"), **_switch_score(arm, seed)}
    for group in ("own", "lure"):
        out[group + "_exploitability"] = 1 - out[group + "_j"]
        for key in ("capture_rate", "pursuer_crash", "pursuer_oob", "evader_crash", "evader_oob", "timeout"):
            out[group + "_" + key] = mean_rate(group, key=key)
    if arm != "frozen_train":
        out["frozen_g1"] = (mean_rate("strong", True) + mean_rate("cross", True)) / 2
        frozen = _switch_score(arm, seed, True)
        for key in ("post_switch_capture", "post_switch_neutralization"):
            out["live_minus_frozen_" + key] = out[key] - frozen[key]
    return out


def contrast(values):
    mean = st.mean(values)
    return {"per_seed": dict(zip(map(str, SEEDS), values)), "mean": mean,
            "verdict": ("notable_increase" if mean >= 0.05 and all(x > 0 for x in values) else
                        "notable_decrease" if mean <= -0.05 and all(x < 0 for x in values) else "inconclusive")}


def analyse():
    output = {"parameters": PARAMS, "nominal_memory_s": {a: -0.02 / math.log(p[0]) for a, p in PARAMS.items()},
              "audit": json.load(open(os.path.join(F21, "audit.json")))}
    metrics = ("g1", "own_j", "lure_j", "post_switch_capture", "post_switch_neutralization",
               "live_minus_frozen_post_switch_capture")
    for radius in (0.10, 0.15):
        scores = {a: {str(s): score(a, s, radius) for s in SEEDS} for a in ARMS}
        contrasts = {f"{a}_minus_{b}": {k: contrast([scores[a][str(s)][k] - scores[b][str(s)][k] for s in SEEDS])
                                      for k in metrics} for a, b in CONTRASTS}
        interaction = {k: contrast([scores["fast_cap"][str(s)][k] - scores["fast"][str(s)][k]
                                   - scores["cap"][str(s)][k] + scores["baseline"][str(s)][k] for s in SEEDS])
                       for k in metrics}
        vs_baseline = {a: {k: contrast([scores[a][str(s)][k] - scores["baseline"][str(s)][k] for s in SEEDS])
                           for k in metrics if k in scores[a][str(SEEDS[0])]} for a in ARMS if a != "baseline"}
        output[f"r{radius:.2f}"] = {"scores": scores, "contrasts": contrasts,
                                  "contrasts_vs_baseline": vs_baseline, "lambda_cap_interaction": interaction}
    output["diagnostics"] = {a: {str(s): json.load(open(_out(a, s, "diagnostics"))) for s in SEEDS} for a in PARAMS}
    primary = output["r0.10"]["contrasts"]
    drifts = {}
    for arm in ("baseline", "fast_cap"):
        drifts[arm] = {}
        for seed in SEEDS:
            values = [row["live"]["snapshots"]["0.5"]["drift_switch_median"]
                      for row in output["diagnostics"][arm][str(seed)].values()]
            drifts[arm][str(seed)] = st.mean(values) if all(v is not None for v in values) else None
    drift_pairs = [(drifts["fast_cap"][str(s)], drifts["baseline"][str(s)]) for s in SEEDS]
    output["predictions"] = {
        "P1": primary["fast_minus_baseline"]["post_switch_capture"]["verdict"] != "notable_increase",
        "P2": all(a > b for a, b in drift_pairs) if all(a is not None and b is not None for a, b in drift_pairs) else None,
        "P3": contrast([output["r0.10"]["scores"]["fast_cap"][str(s)]["post_switch_capture"]
                        - output["r0.10"]["scores"]["baseline"][str(s)]["post_switch_capture"]
                        for s in SEEDS])["verdict"] == "notable_increase",
        "P4": all(output["r0.10"]["scores"]["fast_cap"][str(s)]["live_minus_frozen_post_switch_capture"]
                  > output["r0.10"]["scores"]["baseline"][str(s)]["live_minus_frozen_post_switch_capture"] for s in SEEDS),
    }
    save_json(os.path.join(F21, "results.json"), output)
    lines = ["# F21 RLS self-play ablation", "", "Descriptive, three paired training seeds. No best-arm selection.",
             "Switch endpoints below remain at 0.10 m; only generalization/exploiter endpoints are re-scored.",
             "", "| Contrast | Metric | s0 | s100 | s200 | Mean | Decision |", "|---|---|---|---|---|---|---|"]
    for pair, row in primary.items():
        for key, c in row.items():
            lines.append(f"| {pair} | {key} | " + " | ".join(f"{c['per_seed'][str(s)]:+.3f}" for s in SEEDS)
                         + f" | {c['mean']:+.3f} | {c['verdict']} |")
    lines += ["", f"Predictions (held): {output['predictions']}"]
    with open(os.path.join(F21, "summary.md"), "w") as stream:
        stream.write("\n".join(lines) + "\n")
    log_progress("F21 analysis complete; update RESULTS/PROGRESS before declaring the study complete")


def run_f21c(args):
    if args.smoke:
        raise ValueError("Use F21 --smoke for the two-role gates")
    audit()
    for arm, seed in lane_specs(ARMS):
        f14._wait_campaign(campaign(arm, seed))
        if not os.path.isfile(_out(arm, seed, "DONE")):
            _evaluate_arm(arm, seed)
    if all(os.path.isfile(_out(a, s, "DONE")) for a, s in specs(ARMS)):
        f5._once(os.path.join(F21, "ANALYSIS_DONE"), analyse, "F21 analysis")
    else:
        log_progress("F21 lane evaluations complete; combined analysis waits for the other lane")
