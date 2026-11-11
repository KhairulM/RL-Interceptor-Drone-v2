"""F10: why is F9's control pursuer (thesis rules, 13 x 50M, GAE truncation fix) hardly lurable when F3's hedge pursuer
(thesis rules, 9 x 300M, no fix) is? One campaign = F9 control without the GAE fix (docs/research_plan.md,
F10_gae_sp; pre-registered 2026-11-11 before any F10 run; user chose this single cell).

Campaign thesis-F10-nofix-s0: seed 0, Hedge both roles, 13 stages x 50M, thesis rules, `algo.gae_cut_at_truncation`
off (the default, as F2-F7). F10c: its final pursuer (stage 12) under the thesis rules vs train pool, test set, F3
campaigns' evaders, the F9 campaigns' + its own evaders, and a fresh 100M-frame thesis-rules exploiter (seed 850).
Same measures and metrics as F9c (f9_ablation_sp._metrics); every J at 0.10 m and 0.15 m.
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
