"""F14: generalization, exploitability, and within-episode RLS adaptation study."""

import json
import os
import time

import torch

import numpy as np

import common
from common import OUT, evaluate, log_progress, train
from d_sampling import EXPLOIT_FRAMES
from f2_final import _done, run_campaign
import f5_acc_sp as f5
import f8_roncero_sp as f8
import f8d_heldout as f8d
import f10_gae_sp as f10
import f11_v2_sp as f11
import f13_thesis_short_sp as f13
from rescore import rates_at

F14 = os.path.join(OUT, "F14_generalization_sp")
F8_NEW = "thesis-F8b-roncero-s200"
F11_NEW = "thesis-F14-v2d-short-s200"
F11_GATE = "thesis-F14-v2d-gate-s200"
SEEDS = (0, 100, 200)
F8_CAMPS = (f8.name(0), f8.name(1), F8_NEW)
F10_CAMPS = tuple(f10.seed_name(s) for s in SEEDS)
F11_CAMPS = (f11.name(0), f11.name(1), F11_NEW)
RADII = (0.10, 0.15)


def _wait_file(path, what):
    while not os.path.isfile(path):
        time.sleep(60)
    log_progress(f"F14: prerequisite ready: {what}")


def _wait_campaign(c):
    done = _done(c)
    while not os.path.isfile(done):
        if not os.path.isfile(done + ".lock"):
            raise RuntimeError(f"F14: campaign {c} is neither finished nor running")
        time.sleep(60)


def _ck(c, stage):
    pool = f8.all_stages(c, "pursuer")
    return next(p["checkpoint_path"] for p in pool if int(p["metadata"]["stage"]) == stage)


def _evaders(camps):
    return [p for c in camps for p in f8.all_stages(c, "evader")]


def run_f14a(args):
    if args.smoke:
        raise SystemExit("F14 campaigns have no smoke mode; use the registered configurations")
    _wait_file(os.path.join(f10.F10, "seeds_results.json"), "F10sc analysis")
    f8_cfg = dict(f8.SAMPLERS["hedge"], seed=200, frames_per_stage=f8.FRAMES, n_stages=f8.STAGES,
                  max_total_stages=f8.STAGES)
    f11_cfg = f11.cfg(200, 13)
    f11_rules = f11.rules(**f11.pilot_decision()["settings"])
    run_campaign(F8_NEW, list(f8.RULES), f8_cfg)
    run_campaign(F11_GATE, list(f11_rules), f11.cfg(200, 1))
    _wait_campaign(F11_GATE)
    gate_path = os.path.join(F14, "evals", f"{F11_GATE}_scripted.json")
    if not os.path.isfile(gate_path):
        evaluate(gate_path, [common.heuristic(h, "evader") for h in f11.SCRIPTED], policy_ckpt=_ck(F11_GATE, 0))
    while not os.path.isfile(gate_path):
        if not os.path.isfile(gate_path + ".lock"):
            raise RuntimeError(f"F14 gate evaluation missing: {gate_path}")
        time.sleep(20)
    gate = json.load(open(gate_path))["matchups"]
    capture = float(np.mean([m["capture_rate"] for m in gate.values()]))
    if capture < 0.80:
        raise RuntimeError(f"F14 F11 seed-200 gate failed: scripted capture {capture:.3f} < 0.80")
    run_campaign(F11_NEW, list(f11_rules), f11_cfg)
    for c in (F8_NEW, F11_NEW):
        _wait_campaign(c)
    log_progress("F14 Stage A campaigns complete: F8 s200 and F11 v2d short s200")


def _exploiter(path, pid, c, checkpoint, rules, seed):
    target = [{"id": pid, "kind": "rl", "checkpoint_path": checkpoint, "metadata": {}}]
    if not os.path.isfile(path):
        train(path, target, EXPLOIT_FRAMES, role="evader", seed=seed, extra=list(rules))
    while not os.path.isfile(path):
        if not os.path.isfile(path + ".lock"):
            raise RuntimeError(f"F14 exploiter training failed: {path}")
        time.sleep(20)
    return [{"id": f"f14/evader:{os.path.basename(path).removesuffix('.pt')}", "kind": "rl",
             "checkpoint_path": path, "metadata": {}}]


def _baseline_exploiters(c, family, i, checkpoint):
    if family == "f8":
        own_path = os.path.join(f8.F8, "ckpt", f"exploiter_roncero_{c}.pt")
        thesis_path = os.path.join(f8.F8, "ckpt", f"exploiter_thesis_{c}.pt")
        own = _exploiter(own_path, f"f8/pursuer:{c}", c, checkpoint, f8.RULES, 970 + i)
        thesis = _exploiter(thesis_path, f"f8/pursuer:{c}", c, checkpoint, f11.OLD_RULES_EXPLOITER, 960 + i)
        return own, thesis
    if family == "f10":
        path = os.path.join(f10.F10, "ckpt", f"exploiter_thesis_{c}.pt")
        own = _exploiter(path, f"f10/pursuer:{c}", c, checkpoint, f10.NOFIX, f10.EXPLOITER_SEED.get(SEEDS[i], 850))
        return own, own
    own_rules = f11.rules(**f11.pilot_decision()["settings"])
    own = _exploiter(os.path.join(F14, "ckpt", f"exploiter_v2d_{c}.pt"), f"f14/pursuer:{c}", c, checkpoint,
                     own_rules, 940 + i)
    thesis = _exploiter(os.path.join(F14, "ckpt", f"exploiter_thesis_{c}.pt"), f"f14/pursuer:{c}", c, checkpoint,
                        f11.OLD_RULES_EXPLOITER, 950 + i)
    return own, thesis


def _out(c, group):
    return os.path.join(F14, "evals", f"{c}_{group}.json")


def _baseline_jobs():
    strong = [p for p in f8d._finals() if p["id"].split("/")[0] in f8d.COMMON]
    configs = [("f8", F8_CAMPS, 24), ("f10", F10_CAMPS, 12), ("f11", F11_CAMPS, 12)]
    jobs = []
    for family, camps, stage in configs:
        for i, c in enumerate(camps):
            ck = _ck(c, stage)
            if family == "f8":
                xpool = [p for p in _evaders(camps) if p["id"].split("/")[0] != c]
            elif family == "f10":
                xpool = [p for p in _evaders(camps) if p["id"].split("/")[0] != c]
            else:
                xpool = [p for p in _evaders(camps) if p["id"].split("/")[0] != c]
            own, thesis = _baseline_exploiters(c, family, i, ck)
            jobs += [(c, "strong", strong, ck, None), (c, "cross", xpool, ck, None),
                     (c, "own_exploiter", own, ck, None), (c, "thesis_exploiter", thesis, ck, None)]
    return jobs


def run_f14ac(args):
    for c in (F8_NEW, F11_NEW):
        _wait_campaign(c)
    _wait_file(os.path.join(f10.F10, "seeds_results.json"), "F10sc analysis")
    for c, group, opponents, ck, extra in _baseline_jobs():
        path = _out(c, group)
        if not os.path.isfile(path) and opponents:
            evaluate(path, opponents, policy_ckpt=ck, extra=extra)
    missing = [_out(c, g) for c, g, opps, _, _ in _baseline_jobs() if opps and not os.path.isfile(_out(c, g))]
    if missing:
        log_progress(f"F14 Stage A: {len(missing)} evaluation files missing; analysis deferred")
        return
    analyse_baseline()


def _mean(vals):
    if not vals:
        raise ValueError("empty F14 metric group")
    return float(np.mean(vals))


def _group_cross(rates, self_campaign):
    by_camp = {}
    for pid, value in rates.items():
        camp = pid.split("/")[0]
        if camp != self_campaign:
            by_camp.setdefault(camp, []).append(value)
    if not by_camp:
        raise ValueError(f"no cross-seed opponents for {self_campaign}")
    return _mean([_mean(v) for v in by_camp.values()])


def _score(c, radius):
    rate = lambda g, key="neutralization_rate": rates_at(_out(c, g), key, radius)
    strong = rate("strong")
    strong_values = [v for k, v in strong.items() if k.split("/")[0] in f8d.COMMON]
    cross = rate("cross")
    own = rate("own_exploiter")
    thesis = rate("thesis_exploiter")
    oob_own = rate("own_exploiter", "pursuer_oob")
    oob_thesis = rate("thesis_exploiter", "pursuer_oob")
    return {"strong_j": _mean(strong_values), "cross_j": _group_cross(cross, c),
            "g1": 0.5 * (_mean(strong_values) + _group_cross(cross, c)),
            "own_j": float(next(iter(own.values()))), "thesis_lure_j": float(next(iter(thesis.values()))),
            "own_exploitability": 1.0 - float(next(iter(own.values()))),
            "thesis_lure_exploitability": 1.0 - float(next(iter(thesis.values()))),
            "own_pursuer_oob": float(next(iter(oob_own.values()))),
            "thesis_pursuer_oob": float(next(iter(oob_thesis.values())))}


def analyse_baseline():
    groups = {"f8": F8_CAMPS, "f10": F10_CAMPS, "f11": F11_CAMPS}
    result = {f"r{r:.2f}": {family: {c: _score(c, r) for c in camps}
                            for family, camps in groups.items()} for r in RADII}
    r = result["r0.10"]
    means = {family: {k: _mean([v[k] for v in scores.values()])
                      for k in next(iter(scores.values()))} for family, scores in r.items()}
    gaps = {}
    for family in ("f10", "f11"):
        gaps[family] = {"g1": means[family]["g1"] - means["f8"]["g1"],
                        "g1_strong": means[family]["strong_j"] - means["f8"]["strong_j"],
                        "g1_cross": means[family]["cross_j"] - means["f8"]["cross_j"],
                        "g2": means[family]["own_j"] - means["f8"]["own_j"]}
        gaps[family]["sum"] = gaps[family]["g1"] + gaps[family]["g2"]
    choice = "thesis" if gaps["f10"]["sum"] > gaps["f11"]["sum"] else "v2d"
    out = {"metrics": result, "means_r0.10": means, "gaps_r0.10": gaps, "selected_rules": choice,
           "selection_note": "Per-seed and both gaps reported before equal-weight G1+G2; tie defaults to v2d."}
    common.save_json(os.path.join(F14, "baseline_results.json"), out)
    lines = ["# F14 Stage A: 13-stage rule-selection baseline", "",
             "All evaluations use common thesis episode-ending semantics. Pursuer OOB is not terminal or an evader win; it is reported separately.",
             "G1 = equal-weight mean of strong-set J and campaign-balanced cross-seed/campaign J. G2 = own-rules exploiter J minus F8 own-rules exploiter J.",
             "Both gaps are shown before their sum; positive is better than F8. Three seeds per family; descriptive seed means/ranges.", "",
             "## Per-seed scores at 0.10 m", "", "| Family | Campaign | Strong J | Cross J | G1 | Own-rules J | Own exploitability | Thesis-lure J | Pursuer OOB (own) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for family, scores in r.items():
        for c, v in scores.items():
            lines.append(f"| {family} | {c} | {v['strong_j']:.3f} | {v['cross_j']:.3f} | {v['g1']:.3f} | {v['own_j']:.3f} | {v['own_exploitability']:.3f} | {v['thesis_lure_j']:.3f} | {v['own_pursuer_oob']:.3f} |")
    lines += ["", "## Means and gaps vs F8 (0.10 m)", "", "| Candidate | G1 strong component gap | G1 cross-seed component gap | G1 total gap | G2 own-rules exploitability gap | G1+G2 |", "|---|---:|---:|---:|---:|---:|"]
    for family in ("f10", "f11"):
        g = gaps[family]
        lines.append(f"| {family} | {g['g1_strong']:+.3f} | {g['g1_cross']:+.3f} | {g['g1']:+.3f} | {g['g2']:+.3f} | {g['sum']:+.3f} |")
    lines += ["", f"**Selected rule set:** {choice} (larger equal-weight sum; tie defaults to v2d).", ""]
    with open(os.path.join(F14, "baseline_summary.md"), "w") as f:
        f.write("\n".join(lines))
    common.save_json(os.path.join(F14, "selected_rules.json"), {"rules": choice, "gaps": gaps})
    log_progress(f"F14 Stage A decision: {choice}; gaps {gaps}")


def _selected_rules():
    path = os.path.join(F14, "selected_rules.json")
    _wait_file(path, "Stage A rule selection")
    return json.load(open(path))["rules"]


def _f14_name(rules, arm, seed):
    return f"thesis-F14-{rules}-{arm}-s{seed}"


def _arm_campaigns(choice):
    if choice == "thesis":
        plain = F10_CAMPS
        rls = tuple(_f14_name("thesis", "acc-rls", s) for s in SEEDS)
        frozen = tuple(_f14_name("thesis", "acc-frozen", s) for s in SEEDS)
    else:
        plain = F11_CAMPS
        rls = (f12.name(0), f12.name(1), _f14_name("v2d", "acc-rls", 200))
        frozen = tuple(_f14_name("v2d", "acc-frozen", s) for s in SEEDS)
    return {"plain": plain, "rls": rls, "no_rls": frozen}


def _campaign_cfg(choice, seed):
    if choice == "v2d":
        return f11.cfg(seed, 13)
    return dict(f10.f9.SAMPLERS["hedge"], seed=seed, frames_per_stage=50_000_000, n_stages=13,
                max_total_stages=13)


def _campaign_rules(choice):
    if choice == "v2d":
        return list(f11.rules(**f11.pilot_decision()["settings"]))
    return list(f10.NOFIX)


def _roles(arm):
    pov = f5.rls_overrides()
    if arm == "rls":
        return f5._roles()
    return {"pursuer": pov + [f5.FREEZE], "evader": f5._mirror(pov) + [f5.FREEZE]}


def run_f14c(args):
    if args.smoke:
        raise SystemExit("F14 estimator runs use the registered 13-stage budget")
    choice = _selected_rules()
    camps = _arm_campaigns(choice)
    rules = _campaign_rules(choice)
    for arm in ("rls", "no_rls"):
        for seed, c in zip(SEEDS, camps[arm]):
            if choice == "v2d" and arm == "rls" and seed in (0, 100):
                c_cfg = f11.cfg(seed, 13)
            else:
                c_cfg = _campaign_cfg(choice, seed)
            overrides = list(rules)
            if choice == "thesis":
                overrides.append("algo.gae_cut_at_truncation=false")
            run_campaign(c, overrides, c_cfg, role_overrides=_roles(arm))
    for c in camps["rls"] + camps["no_rls"]:
        _wait_campaign(c)
    common.save_json(os.path.join(F14, "arm_campaigns.json"), {"rules": choice, **camps})
    log_progress(f"F14 Stage C campaigns complete: {choice}, RLS and no-RLS arms")


def _arm_specs():
    choice = _selected_rules()
    camps = _arm_campaigns(choice)
    output = []
    for arm in ("plain", "rls", "no_rls"):
        for seed, c in zip(SEEDS, camps[arm]):
            output.append((arm, seed, c))
    return choice, camps, output


def _stagec_eval_jobs():
    choice, camps, specs = _arm_specs()
    strong = [p for p in f8d._finals() if p["id"].split("/")[0] in f8d.COMMON]
    all_camps = tuple(c for arm in ("plain", "rls", "no_rls") for c in camps[arm])
    cross_by_camp = {c: [p for p in _evaders(all_camps) if p["id"].split("/")[0] != c] for _, _, c in specs}
    rules = _campaign_rules(choice)
    jobs = []
    for arm, seed, c in specs:
        ck = _ck(c, 12)
        extra = None if arm == "plain" else f5.rls_overrides() + ([f5.FREEZE] if arm == "no_rls" else [])
        if arm == "plain":
            # Reuse the Stage A strong-set and exploiter evaluations; cross-play is expanded to include the estimator campaigns.
            jobs.append((c, "stageC_cross", cross_by_camp[c], ck, extra, None))
            continue
        own_rules = rules
        own_path = os.path.join(F14, "ckpt", f"exploiter_own_{c}.pt")
        own = _exploiter(own_path, f"f14/pursuer:{c}", c, ck, own_rules, 1100 + seed)
        thesis = own if choice == "thesis" else _exploiter(
            os.path.join(F14, "ckpt", f"exploiter_thesis_{c}.pt"), f"f14/pursuer:{c}", c, ck,
            f11.OLD_RULES_EXPLOITER, 1200 + seed)
        jobs += [(c, "stageC_strong", strong, ck, extra, None),
                 (c, "stageC_cross", cross_by_camp[c], ck, extra, None),
                 (c, "stageC_own", own, ck, extra, None)]
        if choice == "v2d":
            jobs.append((c, "stageC_thesis", thesis, ck, extra, None))
        if arm == "rls":
            frozen_extra = f5.rls_overrides() + [f5.FREEZE]
            jobs += [(c, "stageC_frozen_strong", strong, ck, frozen_extra, None),
                     (c, "stageC_frozen_cross", cross_by_camp[c], ck, frozen_extra, None)]
    return choice, camps, specs, jobs


def run_f14cc(args):
    _wait_file(os.path.join(F14, "selected_rules.json"), "Stage A rule selection")
    choice = _selected_rules()
    camps = _arm_campaigns(choice)
    for c in camps["rls"] + camps["no_rls"]:
        _wait_campaign(c)
    choice, camps, specs, jobs = _stagec_eval_jobs()
    for c, group, opponents, ck, extra, switch in jobs:
        path = _out(c, group)
        if not os.path.isfile(path) and opponents:
            evaluate(path, opponents, policy_ckpt=ck, extra=extra, switch=switch)
    missing = [_out(c, g) for c, g, opps, _, _, _ in jobs if opps and not os.path.isfile(_out(c, g))]
    if missing:
        log_progress(f"F14 Stage C: {len(missing)} evaluation files missing; analysis deferred")
        return
    analyse_stage_c(choice, camps, specs)


def _score_stagec(c, arm, radius, frozen=False):
    if arm == "plain":
        base = _score(c, radius)
        base["cross_j"] = _group_cross(rates_at(_out(c, "stageC_cross"), "neutralization_rate", radius), c)
        base["g1"] = 0.5 * (base["strong_j"] + base["cross_j"])
        return base
    prefix = "stageC_frozen_" if frozen else "stageC_"
    rate = lambda g, key="neutralization_rate": rates_at(_out(c, prefix + g), key, radius)
    strong = rate("strong")
    strong_vals = [v for k, v in strong.items() if k.split("/")[0] in f8d.COMMON]
    cross = rate("cross")
    r = {"strong_j": _mean(strong_vals), "cross_j": _group_cross(cross, c),
         "g1": 0.5 * (_mean(strong_vals) + _group_cross(cross, c))}
    if not frozen:
        own = rate("own")
        r.update(own_j=float(next(iter(own.values()))),
                 own_exploitability=1 - float(next(iter(own.values()))))
    return r


def analyse_stage_c(choice, camps, specs):
    baseline = json.load(open(os.path.join(F14, "baseline_results.json")))
    f8_scores = baseline["metrics"]["r0.10"]["f8"]
    f8_g1 = _mean([v["g1"] for v in f8_scores.values()])
    f8_own = _mean([v["own_j"] for v in f8_scores.values()])
    result = {}
    for arm, seed, c in specs:
        score = _score_stagec(c, arm, 0.10)
        score.update(g1_gap=score["g1"] - f8_g1, g2_gap=score["own_j"] - f8_own)
        if arm == "rls":
            frozen = _score_stagec(c, arm, 0.10, frozen=True)
            score["frozen_at_eval"] = frozen
        result.setdefault(arm, {})[str(seed)] = score
    g1_mean = _mean([v["g1_gap"] for v in result["rls"].values()])
    g2_mean = _mean([v["g2_gap"] for v in result["rls"].values()])
    rls_mean = _mean([v["g1"] for v in result["rls"].values()])
    no_rls_mean = _mean([v["g1"] for v in result["no_rls"].values()])
    extend = g1_mean <= 0 or g2_mean <= 0
    out = {"rules": choice, "metrics_r0.10": result, "f8_g1": f8_g1, "f8_own_j": f8_own,
           "rls_g1_gap_mean": g1_mean, "rls_g2_gap_mean": g2_mean,
           "rls_minus_no_rls_g1": rls_mean - no_rls_mean, "extend_to_26": extend}
    common.save_json(os.path.join(F14, "stageC_results.json"), out)
    lines = ["# F14 Stage C: selected-rule 13-stage estimator comparison", "",
             f"Selected rules: {choice}. At 0.10 m; G1 = 0.5 x strong-set J + 0.5 x campaign-balanced cross-seed J; G2 = own-rules exploiter J minus F8 own-rules exploiter J.",
             "| Arm | Seed | Strong J | Cross J | G1 gap vs F8 | G2 gap vs F8 | Own exploitability |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ("plain", "rls", "no_rls"):
        for seed, v in result[arm].items():
            lines.append(f"| {arm} | {seed} | {v['strong_j']:.3f} | {v['cross_j']:.3f} | {v['g1_gap']:+.3f} | {v['g2_gap']:+.3f} | {v['own_exploitability']:.3f} |")
    lines += ["", f"Mean RLS G1 gap: {g1_mean:+.3f}; mean RLS G2 gap: {g2_mean:+.3f}; RLS - no-RLS G1: {rls_mean - no_rls_mean:+.3f}.",
              f"26-stage continuation required: {extend} (trigger: either mean gap is non-positive).", ""]
    with open(os.path.join(F14, "stageC_summary.md"), "w") as f:
        f.write("\n".join(lines))
    log_progress(f"F14 Stage C analysis done; extend_to_26={extend}")


def _switch_pair(seed, direction):
    index = SEEDS.index(seed)
    other = (index + 1) % len(SEEDS)
    f8camp = F8_CAMPS[other]
    f10camp = F10_CAMPS[other]
    f8_ev = f8.all_stages(f8camp, "evader")[-1]
    f10_ev = f8.all_stages(f10camp, "evader")[-1]
    return (f8_ev, f10_ev) if direction == "f8_to_f10" else (f10_ev, f8_ev)


def _adaptation_jobs():
    choice = _selected_rules()
    camps = _arm_campaigns(choice)
    jobs = []
    for model in ("rls", "no_rls", "rls_frozen"):
        arm = "rls" if model == "rls_frozen" else model
        for seed, c in zip(SEEDS, camps[arm]):
            ck = _ck(c, 12)
            extra = f5.rls_overrides() if arm == "rls" else f5.rls_overrides() + [f5.FREEZE]
            if model == "rls_frozen":
                extra = f5.rls_overrides() + [f5.FREEZE]
            for direction in ("f8_to_f10", "f10_to_f8"):
                first, second = _switch_pair(seed, direction)
                for switch_s in (0.5, 1.0, 1.5):
                    gid = f"switch:{direction}:{switch_s:.1f}s"
                    path = _out(c, f"adapt_{model}_{direction}_{switch_s:.1f}s")
                    switch = {"first_id": first["id"], "second_id": second["id"], "times_s": [switch_s],
                              "group_id": gid, "seed": seed + 401}
                    jobs.append((path, c, model, seed, direction, switch_s, [first, second], ck, extra, switch, gid))
                for policy in (first, second):
                    path = _out(c, f"adapt_{model}_{direction}_fixed_{policy['id'].split('/')[0]}")
                    jobs.append((path, c, model, seed, direction, None, [policy], ck, extra, None, policy["id"]))
    return jobs


def _smoke_switch():
    choice = _selected_rules()
    camps = _arm_campaigns(choice)
    c = camps["rls"][0]
    first, second = _switch_pair(0, "f8_to_f10")
    path = os.path.join(F14, "switch", "smoke.json")
    if not os.path.isfile(path):
        evaluate(path, [first, second], policy_ckpt=_ck(c, 12), num_envs=256, seed=399, cutoff=300,
                 extra=f5.rls_overrides(),
                 switch={"first_id": first["id"], "second_id": second["id"], "times_s": [0.1],
                         "group_id": "switch:smoke", "seed": 399})
    while not os.path.isfile(path):
        if not os.path.isfile(path + ".lock"):
            raise RuntimeError("F14 learned-policy switch smoke evaluation failed")
        time.sleep(20)
    data = json.load(open(path))
    episodes = torch.load(os.path.splitext(path)[0] + "_episodes.pt", map_location="cpu", weights_only=False)["episodes"]
    if data["eval"].get("switch", {}).get("group_id") != "switch:smoke" or "switch_step" not in episodes:
        raise RuntimeError("F14 switch smoke did not record switching metadata")
    if len(episodes.get("episode_len", [])) == 0:
        raise RuntimeError("F14 switch smoke produced no episodes")


def _episode_switch_metrics(path):
    import torch
    data = json.load(open(path))
    episode_path = os.path.splitext(path)[0] + "_episodes.pt"
    episodes = torch.load(episode_path, map_location="cpu", weights_only=False)["episodes"]
    alive = episodes["episode_len"] > episodes["switch_step"]
    if not bool(alive.any()):
        raise RuntimeError(f"no episodes survived to switch in {path}")
    e = {k: v[alive].float() for k, v in episodes.items() if k not in ("env", "episode_index", "feat_mean", "feat_std", "switch_step")}
    j = (e["capture"] > 0.5) | (e["opp_crash"] > 0.5) | (e["opp_out_of_bounds"] > 0.5)
    captured = e["capture"] > 0.5
    wait = (e["episode_len"] - episodes["switch_step"][alive].float()).clamp_min(0) * data["eval"]["step_dt"]
    return {"episodes_alive_at_switch": int(alive.sum()), "post_switch_neutralization": float(j.float().mean()),
            "post_switch_capture": float(captured.float().mean()),
            "mean_seconds_after_switch_to_capture": float(wait[captured].mean()) if bool(captured.any()) else None,
            "full_episode_j": float(next(iter(data["matchups"].values()))["neutralization_rate"])}


def run_f14d(args):
    _wait_file(os.path.join(F14, "stageC_results.json"), "Stage C analyses")
    _smoke_switch()
    jobs = _adaptation_jobs()
    for path, c, model, seed, direction, switch_s, policies, ck, extra, switch, gid in jobs:
        if not os.path.isfile(path):
            evaluate(path, policies, policy_ckpt=ck, extra=extra, seed=seed + 1400,
                     switch=switch)
    missing = [path for path, *_ in jobs if not os.path.isfile(path)]
    if missing:
        log_progress(f"F14 Stage D: {len(missing)} switch/control evals missing; analysis deferred")
        return
    records = []
    for path, c, model, seed, direction, switch_s, _, _, _, _, gid in jobs:
        result = _episode_switch_metrics(path) if switch_s is not None else {
            "fixed_policy_j": float(next(iter(json.load(open(path))["matchups"].values()))["neutralization_rate"])}
        records.append({"campaign": c, "model": model, "seed": seed, "direction": direction,
                        "switch_seconds": switch_s, "opponent": gid, **result})
    common.save_json(os.path.join(F14, "adaptation_results.json"), {"records": records})
    lines = ["# F14 Stage D: within-episode learned-evader switch", "",
             "Post-switch outcomes are conditional on episodes whose duration exceeds their recorded switch step. Pursuer OOB is not an ending; report it as an outcome, not an evader win.",
             "| Model | Seed | Direction | Switch (s) | Alive at switch | Post-switch J | Post-switch capture | Post-switch capture time (s) |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in records:
        if r["switch_seconds"] is not None:
            t = r["mean_seconds_after_switch_to_capture"]
            time_cell = f"{t:.3f}" if t is not None else "-"
            lines.append(f"| {r['model']} | {r['seed']} | {r['direction']} | {r['switch_seconds']:.1f} | {r['episodes_alive_at_switch']} | {r['post_switch_neutralization']:.3f} | {r['post_switch_capture']:.3f} | {time_cell} |")
    with open(os.path.join(F14, "adaptation_summary.md"), "w") as f:
        f.write("\n".join(lines))
    log_progress("F14 Stage D adaptation analysis done")
