"""F6b: F6 (Hedge self-play + evader diversity term) with the term redesigned (docs/research_plan.md, Phase F,
F6b_div_sp; pre-registered 2026-11-08 before any F6b run).

F6's term saturated (mean-only KL divided by sigma ~0.03, always >> cap 2) and acted as a +2 per-step survival bonus,
so its evaders stopped luring. F6b changes only the term (PPOPolicy.set_diversity mode="action"):
d_j(s) = RMS over action dims of mu(s) - mu_j(s) (action units), bonus = 1 - exp(-d / tau), the intrinsic reward
centred on its batch mean (zero-mean), plus the same bonus in the loss. Everything else as F6 (Hedge on both roles,
plain obs, 9 stages x 300M, seeds 0 / 100). tau and beta come from a pilot (pilot()); F6bc = F6c (run_f6c reused).
"""

import json
import os
import re
import time

import numpy as np

import common
from common import evaluate, log_progress, save_json, train
from f2_final import _done, run_campaign
from f3_sampler_sp import SAMPLERS, SEEDS
from f6_div_sp import F6, PILOT_FRAMES, PILOT_FROM, _wait_other_lane, run_f6c

F6B = os.path.join(common.OUT, "F6b_div_sp")
ARM = "hedge_divb"
PILOT_BETAS = [0.03, 0.1, 0.3]
PILOT_J_TOL = 0.05          # J(beta) <= J(0) + 0.05 (the evader still beats its training pursuers)
PILOT_TIMEOUT_TOL = 0.10    # mean timeout rate(beta) <= that of beta = 0 + 0.10 (it wins the same way: F6's failure)
PILOT_MOVE = 1.25           # active: final mean d >= 1.25 tau ...
PILOT_SAT = 0.9             # ... and final mean bonus <= 0.9 (not saturated)
SMOKE_BETA, SMOKE_TAU = 0.1, 0.05
EXPLOIT_SEED = 650          # F6c used 600 + i


def name(k, smoke=False):
    return "thesis-F6b-smoke-hedge_divb" if smoke else f"thesis-F6b-hedge_divb-s{k}"


def arm_cfg(beta, tau):
    return dict(SAMPLERS["hedge"], diversity_roles="[evader]", diversity_beta=beta, diversity_cap=2.0,
                diversity_mode="action", diversity_tau=tau, diversity_centre=True)


def _div_overrides(ckpts, beta, tau, measure=False):
    return [f"diversity.ckpts=[{','.join(ckpts)}]", f"diversity.beta={beta}", "diversity.mode=action",
            f"diversity.tau={tau}", "diversity.centre=true"] + (["diversity.measure=true"] if measure else [])


def _tau(log):
    """Amendment 1 (11-09): median d over the last quarter of the beta = 0 run's updates (v1 took the mean of the last
    10, which a short instability at the end of that run raised to 0.65; the run's d was 0.065-0.09 before it)."""
    vals = [float(v) for v in re.findall(r"^div_distance: ([-0-9.eE+]+)", open(log).read(), re.M)]
    return float(np.median(vals[-(len(vals) // 4):])) if len(vals) >= 4 else None


def _f3_reference():
    """Amendment 1: F3 hedge-both-s0's own stage-5 evader (no term, 300M frames) vs the same training pursuers, from
    its in-campaign stage eval: (mean J, mean timeout)."""
    c, stage = PILOT_FROM
    m = json.load(open(os.path.join(common.SELFPLAY_OUT, c, "stages", f"stage{stage}_evader_eval_out.json")))["matchups"]
    return (float(np.mean([x["neutralization_rate"] for x in m.values()])),
            float(np.mean([x["outcomes"]["timeout"] for x in m.values()])))


def _last(log, key, n=10):
    vals = re.findall(rf"^{key}: ([-0-9.eE+]+)", open(log).read(), re.M) if os.path.isfile(log) else []
    return float(np.mean([float(v) for v in vals[-n:]])) if vals else None


def _pilot_setup():
    c, stage = PILOT_FROM
    d = os.path.join(common.SELFPLAY_OUT, c)
    spec = json.load(open(os.path.join(d, "stages", f"stage{stage}_evader_train_opp.json")))
    refs = [os.path.join(d, "checkpoints", f"evader_stage{s}.pt") for s in range(1, stage, 2)]
    return spec, refs


def _pilot_arm(tag, spec, refs, extra):
    """Train + evaluate one pilot evader (skips work another lane holds); returns its row or None."""
    ck = os.path.join(F6B, "pilot", f"{tag}.pt")
    if train(ck, spec["policies"], PILOT_FRAMES, role="evader", probs=spec["probs"], init_policy=refs[-1],
             seed=900, extra=extra) is None:
        return None
    r = evaluate(os.path.join(F6B, "pilot", f"{tag}_eval.json"), spec["policies"], role="evader", policy_ckpt=ck)
    if r is None:
        return None
    m = list(r["matchups"].values())
    log = os.path.join(F6B, "pilot", f"{tag}_log.txt")
    return {"J_vs_training_pursuers": float(np.mean([x["neutralization_rate"] for x in m])),
            "timeout": float(np.mean([x["outcomes"]["timeout"] for x in m])),
            "pursuer_crash": float(np.mean([x["outcomes"]["pursuer_crash"] for x in m])),
            "div_distance_end": _last(log, "div_distance"), "div_bonus_end": _last(log, "div_bonus")}


def _wait_for(path, what, hours=6):
    """Wait for a pilot file another lane is producing (a crashed lane: re-run the lane command to resume)."""
    t0 = time.time()
    while not os.path.isfile(path):
        if time.time() - t0 > hours * 3600:
            raise RuntimeError(f"F6b pilot: {what} not produced within {hours} h: {path}")
        time.sleep(60)


def choose(rows, tau, ref):
    """The pre-registered selection (see pilot()); ref = (J, timeout) reference (amendment 1: the stricter of the
    beta = 0 run and F3's own stage-5 evader). Returns (beta or None, kept, active)."""
    kept = [b for b in PILOT_BETAS
            if rows[b]["J_vs_training_pursuers"] <= ref[0] + PILOT_J_TOL
            and rows[b]["timeout"] <= ref[1] + PILOT_TIMEOUT_TOL]
    active = [b for b in kept if (rows[b]["div_distance_end"] or 0) >= PILOT_MOVE * tau
              and (rows[b]["div_bonus_end"] if rows[b]["div_bonus_end"] is not None else 1.0) <= PILOT_SAT]
    return (max(active) if active else (max(kept) if kept else None)), kept, active


def pilot():
    """Pre-registered rule (training-time data only), from the F3 hedge-both-s0 state at evader stage 5 (same
    opponents and probabilities, warm start from its stage-3 evader, references s1 / s3, seed 900, 100M frames):
    1. beta = 0 with the term measured only: tau = its median d over the last quarter of its updates (the action
       change one evader stage makes without the term; amendment 1, was the mean of the last 10).
    2. beta in PILOT_BETAS with that tau. Reference (amendment 1) = the stricter (lower) J and timeout of the beta = 0
       run and of F3's own stage-5 evader. Keep the betas with J <= J_ref + 0.05 and mean timeout <= timeout_ref + 0.10;
       choose the largest of them that is active (final mean d >= 1.25 tau and final mean bonus <= 0.9), else the
       largest kept one. None kept: F6b stops (choice.json.failed, no campaigns). Returns (beta, tau)."""
    choice = os.path.join(F6B, "pilot", "choice.json")
    if os.path.isfile(choice):
        c = json.load(open(choice))
        return c["beta"], c["tau"]
    if os.path.isfile(choice + ".failed"):
        raise RuntimeError("F6b pilot failed earlier (choice.json.failed): no campaigns")
    spec, refs = _pilot_setup()
    rows = {}
    r0 = _pilot_arm("beta0", spec, refs, _div_overrides(refs, 0.0, 1.0, measure=True))
    if r0 is None:
        _wait_for(os.path.join(F6B, "pilot", "beta0_eval.json"), "beta = 0 run")
        r0 = _pilot_arm("beta0", spec, refs, [])
    rows[0.0] = r0
    tau = _tau(os.path.join(F6B, "pilot", "beta0_log.txt"))
    f3_ref = _f3_reference()
    ref = (min(r0["J_vs_training_pursuers"], f3_ref[0]), min(r0["timeout"], f3_ref[1]))
    if not tau or tau <= 0:
        raise RuntimeError(f"F6b pilot: no div_distance logged for beta = 0 ({r0})")
    for beta in PILOT_BETAS:                     # two lanes split these: train() skips runs the other lane holds
        _pilot_arm(f"beta{beta}", spec, refs, _div_overrides(refs, beta, tau))
    for beta in PILOT_BETAS:
        _wait_for(os.path.join(F6B, "pilot", f"beta{beta}_eval.json"), f"beta = {beta} run")
        rows[beta] = _pilot_arm(f"beta{beta}", spec, refs, _div_overrides(refs, beta, tau))
    beta, kept, active = choose(rows, tau, ref)
    out = {"beta": beta, "tau": tau, "reference": {"J": ref[0], "timeout": ref[1], "f3_stage5": f3_ref},
           "rows": {str(k): v for k, v in rows.items()}, "kept": kept, "active": active,
           "rule": "amendment 1: tau = median beta0 d over the last quarter; reference = stricter of beta0 and F3's "
                   "stage-5 evader; keep J <= J_ref + 0.05 and timeout <= timeout_ref + 0.10; largest active "
                   "(d >= 1.25 tau, bonus <= 0.9), else largest kept; none kept -> stop"}
    if beta is None:
        save_json(choice + ".failed", out)
        log_progress(f"F6b pilot: no beta kept, F6b stops ({rows})")
        raise RuntimeError("F6b pilot: no beta kept")
    save_json(choice, out)
    log_progress(f"F6b pilot: beta={beta} tau={tau:.4f} ({rows})")
    return beta, tau


def run_f6b(args):
    sm = name(0, smoke=True)
    run_campaign(sm, [], arm_cfg(SMOKE_BETA, SMOKE_TAU), smoke_stages=5)
    _wait_other_lane(_done(sm))
    if not os.path.isfile(_done(sm)):
        raise RuntimeError("F6b smoke campaign did not finish")
    log = open(os.path.join(common.SELFPLAY_OUT, sm + "_log.txt")).read()
    if "mode=action" not in log or "div_bonus:" not in log or "div_reward:" not in log:
        raise RuntimeError("F6b smoke: the action-mode diversity term was never active")
    if args.smoke:
        return
    beta, tau = pilot()
    for k in range(len(SEEDS)):
        run_campaign(name(k), [], dict(arm_cfg(beta, tau), seed=SEEDS[k]))


def run_f6bc(args):
    """F6c for the F6b campaigns, then the F6b - F6 (hedge_divb - hedge_div) differences."""
    camps = [name(k) for k in range(len(SEEDS))]
    run_f6c(args, camps=camps, root=F6B, arm=ARM, exploit_seed=EXPLOIT_SEED)
    if os.path.isfile(os.path.join(F6B, "results.json")):
        compare_f6()


def compare_f6():
    """Seed-paired hedge_divb - hedge_div from both results.json files (appended to summary.md)."""
    b = json.load(open(os.path.join(F6B, "results.json")))["per_campaign"]
    a = json.load(open(os.path.join(F6, "results.json")))["per_campaign"]
    ms = [m for m in b.get(f"{ARM}/0", {}) if m in a.get("hedge_div/0", {})]
    diffs = {m: [b[f"{ARM}/{k}"][m] - a[f"hedge_div/{k}"][m] for k in range(len(SEEDS))
                 if f"{ARM}/{k}" in b and f"hedge_div/{k}" in a] for m in ms}
    save_json(os.path.join(F6B, "vs_f6.json"), diffs)
    lines = ["", f"{ARM} - hedge_div (F6), seed-paired (mean [per seed]):", "", "| metric | difference |", "|---|---|"]
    lines += [f"| {m} | {np.mean(d):+.3f} [{', '.join(f'{x:+.3f}' for x in d)}] |" for m, d in diffs.items() if d]
    with open(os.path.join(F6B, "summary.md"), "a") as f:
        f.write("\n".join(lines) + "\n")
