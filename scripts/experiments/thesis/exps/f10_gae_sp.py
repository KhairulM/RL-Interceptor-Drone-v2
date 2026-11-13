"""F10: why is F9's control pursuer (thesis rules, 13 x 50M, GAE truncation fix) hardly lurable when F3's hedge pursuer
(thesis rules, 9 x 300M, no fix) is? One campaign = F9 control without the GAE fix (docs/research_plan.md,
F10_gae_sp; pre-registered 2026-11-11 before any F10 run; user chose this single cell).

Campaign thesis-F10-nofix-s0: seed 0, Hedge both roles, 13 stages x 50M, thesis rules, `algo.gae_cut_at_truncation`
off (the default, as F2-F7). F10c: its final pursuer (stage 12) under the thesis rules vs train pool, test set, F3
campaigns' evaders, the F9 campaigns' + its own evaders, and a fresh 100M-frame thesis-rules exploiter (seed 850).
Same measures and metrics as F9c (f9_ablation_sp._metrics); every J at 0.10 m and 0.15 m.
Amendment 1 (11-13): seeds 100 / 200 (F10s / F10sc, below), paired by seed with F13.
"""

import json
import os
import time

import numpy as np

from a_adapt import test_set
from common import OUT, evaluate, log_progress, save_json, train
from d_sampling import EXPLOIT_FRAMES
from f2_final import _done, run_campaign
from f3_sampler_sp import _policies
from f3_sampler_sp import name as f3_name
import f8_roncero_sp as f8
import f9_ablation_sp as f9
from x3_gap import load_split

F10 = os.path.join(OUT, "F10_gae_sp")
NAME = "thesis-F10-nofix-s0"
GROUPS = ("train", "test", "f3_evaders", "f9_evaders", "vs_exploiter_thesis")


def run_f10(args):
    if args.smoke:          # no new code path: the thesis rules without the flag are the F2-F7 default
        return
    run_campaign(NAME, [], f9.arm_cfg())


def run_f10c(args):
    d = _done(NAME)
    while not os.path.isfile(d):
        if not os.path.isfile(d + ".lock"):
            raise RuntimeError(f"F10c: campaign {NAME} is neither finished nor running")
        time.sleep(60)
    split = load_split()
    test = test_set(split)
    f3_ev = [e for s in f9.F3_ARMS for k in range(2) for e in _policies(f3_name(s, k), "evader")]
    f9_ev = [e for c in [f9.name(a) for a in f9.ARMS] + [NAME] for e in f8.all_stages(c, "evader")]
    ck = f8.all_stages(NAME, "pursuer")[-1]["checkpoint_path"]
    ex_path = os.path.join(F10, "ckpt", f"exploiter_thesis_{NAME}.pt")
    train(ex_path, [{"id": f"f10/pursuer:{NAME}", "kind": "rl", "checkpoint_path": ck, "metadata": {}}],
          EXPLOIT_FRAMES, role="evader", seed=850)
    ex = ([{"id": f"f10/evader:exploit_thesis_{NAME}", "kind": "rl", "checkpoint_path": ex_path, "metadata": {}}]
          if os.path.isfile(ex_path) else [])
    for g, opps in zip(GROUPS, (split["train"], test, f3_ev, f9_ev, ex)):
        if opps and not os.path.isfile(f9._src(NAME, g, F10)):
            evaluate(f9._src(NAME, g, F10), opps, policy_ckpt=ck)
    missing = [g for g in GROUPS if not os.path.isfile(f9._src(NAME, g, F10))]
    if missing:
        raise RuntimeError(f"F10c: evals missing {missing}")
    analyse_f10(split, test)


def measure_a(lr):
    """Pre-registered (a), as F9: mean over evader stages 3-11 of each stage's mean lure rate."""
    return float(np.mean([np.mean(list(d.values())) for s, d in lr.items() if 3 <= s <= 11 and d]))


def analyse_f10(split, test):
    with open(os.path.join(OUT, "X2_descriptors", "clusters.json")) as f:
        labels = json.load(f)["labels"]
    res = {f"r{rad:.2f}": _metrics_pair(rad, split, test, labels) for rad in f9.RADII}
    lr = f8.lure_rates(NAME)
    ref_lr = f8.lure_rates(f9.name("control"))
    save_json(os.path.join(F10, "results.json"), {"metrics": res, "lure_rates": lr,
                                                  "measure_a": {"nofix": measure_a(lr), "control": measure_a(ref_lr)}})
    f3 = json.load(open(os.path.join(f8.F8, "results.json")))["metrics"]
    cols = ["F10 no fix", "F9 control (fix)", "no fix - fix", "F3 hedge s0 (9 x 300M, no fix)"]
    lines = ["# F10_gae_sp summary", "",
             "Final pursuer (stage 12) of thesis-F10-nofix-s0 (thesis rules, Hedge both roles, 13 x 50M, seed 0, no GAE "
             "truncation fix) vs F9 control (the same with the fix) and F3 hedge s0. All evals clean under the thesis rules; "
             "exploitability_thesis = 1 - J vs a fresh 100M-frame thesis-rules exploiter. 1 seed, descriptive.", "",
             f"Pre-registered (a), in-campaign lure rate (mean over evader stages 3-11): no fix {measure_a(lr):.2f}, "
             f"control {measure_a(ref_lr):.2f}. (b), pursuer crash vs the exploiter: see the table.", ""]
    for k, v in res.items():
        lines += [f"## Capture radius {k[1:]} m", "", "| metric | " + " | ".join(cols) + " |", "|---" * 5 + "|"]
        for m in f9.METRICS:
            a, b = v["nofix"].get(m), v["control"].get(m)
            c = f3.get(k, {}).get("hedge/0", {}).get({"other_f9_evaders": "_"}.get(m, m))
            cells = [f"{x:.3f}" if x is not None else "-" for x in (a, b)]
            cells += [f"{a - b:+.3f}" if a is not None and b is not None else "-", f"{c:.3f}" if c is not None else "-"]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## In-campaign lure rates (pursuer crash caused by each evader stage in the stage >= 2 pursuers)", "",
              "| campaign | evader stage | mean | min - max | n pursuers |", "|---|---|---|---|---|"]
    for c, l in ((NAME, lr), (f9.name("control"), ref_lr)):
        for s, d in sorted(l.items()):
            if d:
                x = list(d.values())
                lines.append(f"| {c} | {s} | {np.mean(x):.2f} | {min(x):.2f} - {max(x):.2f} | {len(x)} |")
    from f5_outcomes import stage_table
    lines += ["", "## Inside the campaign (each stage's eval vs its training opponents)", ""] + stage_table(NAME, f9.STAGES)
    with open(os.path.join(F10, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log_progress("F10c analysis done")


def _metrics_pair(rad, split, test, labels):
    return {"nofix": f9._metrics(NAME, rad, split, test, labels, root=F10),
            "control": f9._metrics(f9.name("control"), rad, split, test, labels)}


# ---- amendment 1 (11-13): seeds 100 / 200 (F10s / F10sc), paired by seed with F13 (same with the GAE fix)
NEW_SEEDS = (100, 200)
SEEDS = (0,) + NEW_SEEDS
NOFIX = ["algo.gae_cut_at_truncation=false"]           # explicit: F10s is not in run.LEGACY_GAE
EXPLOITER_SEED = {100: 920, 200: 921}
SEED_GROUPS = GROUPS + ("finals", "f11_evaders", "f10_evaders", "f8_evaders", "f8_evaders_roncero")


def seed_name(seed):
    return NAME if seed == 0 else f"thesis-F10-nofix-s{seed}"


def run_f10s(args):
    if args.smoke:
        raise SystemExit("F10s has no smoke run (same code path as F10 / F13)")
    import f13_thesis_short_sp as f13
    for seed in NEW_SEEDS:                                  # lanes skip a campaign the other lane holds
        run_campaign(seed_name(seed), NOFIX, f13.cfg(seed))


def _exploiter(c, seed):
    if seed == 0:
        path = os.path.join(F10, "ckpt", f"exploiter_thesis_{NAME}.pt")        # F10c's
    else:
        import f11_v2_sp as f11
        path = os.path.join(F10, "ckpt", f"exploiter_thesis_{c}.pt")
        train(path, [{"id": f"f10/pursuer:{c}", "kind": "rl", "checkpoint_path": _final(c), "metadata": {}}],
              EXPLOIT_FRAMES, role="evader", seed=EXPLOITER_SEED[seed], extra=list(f11.OLD_RULES_EXPLOITER))
    return ([{"id": f"f10/evader:exploit_thesis_{c}", "kind": "rl", "checkpoint_path": path, "metadata": {}}]
            if os.path.isfile(path) else [])


def _final(c):
    return f8.all_stages(c, "pursuer")[-1]["checkpoint_path"]


def _seed_jobs(split, test):
    """[(campaign, group, opponents, checkpoint, extra)] of every F10sc eval (existing s0 files are skipped)."""
    import f8d_heldout as f8d
    import f11_v2_sp as f11
    f3_ev = [e for s in f9.F3_ARMS for k in range(2) for e in _policies(f3_name(s, k), "evader")]
    f11_ev = [e for k in range(len(f11.SEEDS)) for e in f8.all_stages(f11.name(k), "evader")]
    f8_ev = [e for k in range(len(f8.SEEDS)) for e in f8.all_stages(f8.name(k), "evader")]
    jobs = []
    for seed in SEEDS:
        c = seed_name(seed)
        ck = _final(c)
        f9_ev = [e for x in [f9.name(a) for a in f9.ARMS] + [c] for e in f8.all_stages(x, "evader")]
        f10_ev = [e for s in SEEDS if s != seed for e in f8.all_stages(seed_name(s), "evader")]
        jobs += [(c, "train", split["train"], ck, None), (c, "test", test, ck, None), (c, "f3_evaders", f3_ev, ck, None),
                 (c, "f9_evaders", f9_ev, ck, None), (c, "vs_exploiter_thesis", lambda c=c, s=seed: _exploiter(c, s), ck, None),
                 (c, "finals", f8d._finals(), ck, None), (c, "f11_evaders", f11_ev, ck, None),
                 (c, "f10_evaders", f10_ev, ck, None), (c, "f8_evaders", f8_ev, ck, None),
                 (c, "f8_evaders_roncero", f8_ev, ck, list(f8.RULES))]
    return jobs


def run_f10sc(args):
    for seed in SEEDS:
        d = _done(seed_name(seed))
        while not os.path.isfile(d):
            if not os.path.isfile(d + ".lock"):
                raise RuntimeError(f"F10sc: campaign {seed_name(seed)} is neither finished nor running")
            time.sleep(60)
    split = load_split()
    test = test_set(split)
    jobs = _seed_jobs(split, test)
    for c, g, opps, ck, extra in jobs:
        if os.path.isfile(f9._src(c, g, F10)):
            continue
        opps = opps() if callable(opps) else opps
        if opps:
            evaluate(f9._src(c, g, F10), opps, policy_ckpt=ck, extra=extra)
    missing = [f9._src(c, g, F10) for c, g, _, _, _ in jobs if not os.path.isfile(f9._src(c, g, F10))]
    if missing:                                 # the other lane is still producing them and analyses after
        log_progress(f"F10sc: {len(missing)} eval files missing on this lane, analysis left to the other lane")
        return
    analyse_f10s(split, test)


def _seed_metrics(c, rad, split, test, labels):
    import f8d_heldout as f8d
    from rescore import rates_at
    r = f9._metrics(c, rad, split, test, labels, root=F10)
    rate = lambda g, key="neutralization_rate", t=False: rates_at(f9._src(c, g, F10), key, rad, pursuer_oob_terminal=t)
    no_s1 = lambda d: [v for k, v in d.items() if not k.endswith(":s1")]
    camp = lambda e: e.split("/")[0]
    F = rate("finals")
    strong = [F[e] for e in F if camp(e) in f8d.COMMON]
    r.update(strong_finals_mean=float(np.mean(strong)), strong_finals_min=float(min(strong)),
             vs_f8_finals=float(np.mean([F[e] for e in F if camp(e) in f8d.F8C])),
             vs_f3_hedge_finals=float(np.mean([F[e] for e in F if camp(e) in f8d.HEDGE])),
             cross_play=float(np.mean(no_s1(rate("f11_evaders")))),
             other_f10_evaders=float(np.mean(no_s1(rate("f10_evaders")))),
             f8_evaders_thesis=float(np.mean(no_s1(rate("f8_evaders")))),
             f8_evaders_roncero=float(np.mean(no_s1(rate("f8_evaders_roncero")))),
             f8_evaders_roncero_pursuer_oob=float(np.mean(no_s1(rate("f8_evaders_roncero", "pursuer_oob", True)))))
    return r


def analyse_f10s(split, test):
    import f13_thesis_short_sp as f13
    with open(os.path.join(OUT, "X2_descriptors", "clusters.json")) as f:
        labels = json.load(f)["labels"]
    res = {f"r{rad:.2f}": {f"f10/{s}": _seed_metrics(seed_name(s), rad, split, test, labels) for s in SEEDS}
           for rad in f9.RADII}
    lure = {f"f10/{s}": measure_a(f8.lure_rates(seed_name(s))) for s in SEEDS}
    lure.update({f"f13/{s}": measure_a(f8.lure_rates(f13.name(s))) for s in SEEDS})
    f13m = json.load(open(os.path.join(f13.F13, "results.json")))["metrics"]
    keys = ("heldout_rl", "strong_finals_mean", "exploitability_thesis", "exploiter_thesis_pursuer_crash", "cross_play")
    diff = {k: float(np.mean([res["r0.10"][f"f10/{s}"][k] - f13m["r0.10"][f"f13/{s}"][k] for s in SEEDS])) for k in keys}
    n = {s: res["r0.10"][f"f10/{s}"] for s in NEW_SEEDS}
    pred = {"1_exploiter_crash_lt_0.50": all(n[s]["exploiter_thesis_pursuer_crash"] < .50 for s in NEW_SEEDS),
            "2_strong_finals_ge_0.80": all(n[s]["strong_finals_mean"] >= .80 for s in NEW_SEEDS),
            "3_heldout_rl_ge_0.90": all(n[s]["heldout_rl"] >= .90 for s in NEW_SEEDS),
            "4_roncero_f8_evaders_below_f8": all(n[s]["f8_evaders_roncero"] < .824 for s in NEW_SEEDS),
            "5_gae_fix_small": abs(diff["exploitability_thesis"]) < .15 and abs(diff["strong_finals_mean"]) < .10}
    save_json(os.path.join(F10, "seeds_results.json"),
              {"metrics": res, "measure_a": lure, "f10_minus_f13_mean_r0.10": diff, "predictions": pred})
    lines = ["# F10 seeds (amendment 1) summary", "",
             "Final pursuers (stage 12) of thesis-F10-nofix-s{0,100,200} (thesis rules, no GAE fix, Hedge both roles, "
             "13 x 50M) vs F13 / F9 control (same with the fix, paired by seed). Evals clean under the thesis rules except "
             "f8_evaders_roncero (Roncero's rules). 3 seeds, descriptive.", ""]
    cols = [f"F10 s{s}" for s in SEEDS] + ["F10 mean"] + [f"F13 s{s}" for s in SEEDS] + ["F13 mean"]
    for k, v in res.items():
        lines += [f"## Capture radius {k[1:]} m", "", "| metric | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for m in v["f10/0"]:
            a = [v[f"f10/{s}"].get(m) for s in SEEDS]
            b = [f13m.get(k, {}).get(f"f13/{s}", {}).get(m) for s in SEEDS]
            cell = lambda x: f"{x:.3f}" if x is not None else "-"
            mean = lambda xs: cell(float(np.mean(xs))) if all(x is not None for x in xs) else "-"
            lines.append(f"| {m} | " + " | ".join([cell(x) for x in a] + [mean(a)] + [cell(x) for x in b] + [mean(b)]) + " |")
        lines.append("")
    lines += ["## In-campaign lure rate (a), mean over evader stages 3-11", ""]
    lines += [f"- {k}: {x:.2f}" for k, x in lure.items()]
    lines += ["", "## F10 - F13 (mean over seed pairs, 0.10 m)", ""] + [f"- {k}: {x:+.3f}" for k, x in diff.items()]
    lines += ["", "## Pre-registered predictions (new seeds 100 / 200)", ""] + [f"- {k}: {'held' if x else 'did not hold'}"
                                                                               for k, x in pred.items()]
    with open(os.path.join(F10, "seeds_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log_progress("F10sc analysis done")
