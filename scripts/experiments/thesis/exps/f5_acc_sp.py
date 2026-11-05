"""F5: learned acceleration estimate + RLS in full self-play (docs/research_plan.md, Phase F, pre-registered 11-04 17:40).

New campaigns (F3 protocol, clean training, seeds 0 / 100): the pursuer observes the A2b learned acceleration estimate
with online RLS (A2b NFacc_OA) on top of the plain observation; the evader keeps the plain observation; frozen
pursuers the evader meets run the same estimator + RLS on clean inputs. Arms: uniform_rls (Exp 1) and qd_dpp_rls
(Exp 2, QD-DPP on both roles as F3 amendment 3). Reused: F2-plain / F3 (plain obs) and F2-acc (finite difference).
F5c: every final pursuer vs the train pool, the test set and all RL evaders of the plain-evader campaigns, clean and
noisy, plus its exploiter; F5 pursuers also with RLS frozen at eval (RLS ablation B1).
Amendment 1 (11-04 21:11): the estimator is retrained (A2a rules) on trajectories of a new generalist vs the train pool
+ self-play evaders F5c never evaluates (F4 seed 2, F2-acc); F5c campaign-evader metrics exclude stage-1 evaders.
"""

import hashlib
import json
import os
import time

import numpy as np
import torch

import common
from a2_nfacc import a2b_methods, fit_acc_estimator
from a_adapt import test_set
from common import OUT, evaluate, log_progress, save_json, train
from d_sampling import EXPLOIT_FRAMES
from f2_final import ACC, OPP_ACC, _done, run_campaign
from f3_sampler_sp import F3C, SAMPLERS, _policies
from f3_sampler_sp import name as f3_name
from f4_estimator_sp import NOISY, _mirror
from x3_gap import GEN_FRAMES, load_split

F5C = os.path.join(OUT, "F5_acc_sp")
EST = os.path.join(F5C, "estimator")          # amendment 1: retrained estimator
GEN_SEED = 0
INDEP_PLAIN = ["thesis-F4-nf-s2", "thesis-F4-ema-s2"]   # plain-obs evaders, seed 200: no F5c evaluation evader
INDEP_ACC = "thesis-F2-acc"                              # evaders observe the pursuer's acceleration
REC_ENVS = 1024
SEEDS = [0, 100]
FREEZE = "task.adapt.freeze=true"
NEW = {"uniform_rls": SAMPLERS["uniform"], "qd_dpp_rls": SAMPLERS["qd_dpp"]}
SMOKE_STAGES = {"uniform_rls": 3, "qd_dpp_rls": 5}      # qd_dpp: first stages with 2 RL opponents per role
PRIMARY = ["heldout_rl", "heldout_worst_cluster", "other_campaign_evaders", "exploitability"]
SECONDARY = ["train", "heldout_scripted", "x3_exploiters", "own_campaign_evaders"]
CONTRASTS = [("uniform_rls", "clean", "uniform_plain", "clean"), ("uniform_rls", "noisy", "uniform_plain", "noisy"),
             ("uniform_fd", "clean", "uniform_plain", "clean"), ("uniform_fd", "noisy", "uniform_plain", "noisy")] + [
    ("qd_dpp_rls", c, b, c) for c in ("clean", "noisy")
    for b in ("uniform_plain", "hedge_plain", "pfsp_plain", "qd_dpp_plain", "uniform_rls")] + [
    (a, c, a, c + "_frozen") for a in NEW for c in ("clean", "noisy")]


# 11-05 (user): Exp 2 (qd_dpp_rls) is paused until QD-DPP is revised (F6). Meanwhile F5c evaluates and analyses Exp 1
# only (uniform_rls vs uniform_plain / uniform_fd, RLS ablation of uniform_rls); hedge / pfsp / qd_dpp_plain are only
# needed by Exp 2 contrasts. Campaign evaders = those of the non-paused plain-evader campaigns; when Exp 2 resumes, the
# Exp 1 pursuers get an extra eval vs the new evaders (`*_campaign_evaders_add.json`, merged by _metrics).
PAUSED_ARMS = {"qd_dpp_rls"}
EXP1_ARMS = {"uniform_rls", "uniform_plain", "uniform_fd"}


def active_pursuers():
    P = pursuers()
    return [p for p in P if p[0] in EXP1_ARMS] if PAUSED_ARMS else P


def rls_overrides():
    return a2b_methods(EST)["NFacc_OA"]


def name(arm, k, smoke=False):
    return f"thesis-F5-smoke-{arm}" if smoke else f"thesis-F5-{arm}-s{k}"


def pursuers():
    """[(arm, k, campaign, pursuer overrides, exploiter checkpoint)] of every final pursuer F5c evaluates."""
    out = []
    for k in range(len(SEEDS)):
        for arm in NEW:
            c = name(arm, k)
            out.append((arm, k, c, rls_overrides(), os.path.join(F5C, "ckpt", f"exploiter_{c}.pt")))
        for s in ("uniform", "hedge", "pfsp", "qd_dpp"):
            c = f3_name(s, k)
            out.append((f"{s}_plain", k, c, [], os.path.join(F3C, "ckpt", f"exploiter_{c}.pt")))
    out.append(("uniform_fd", 0, "thesis-F2-acc", list(ACC),
                os.path.join(OUT, "F2c_final_eval", "ckpt", "exploiter_acc_final.pt")))
    return out


def conditions(arm, pov):
    out = {"clean": pov, "noisy": [NOISY] + pov}
    if arm in NEW:
        out.update(clean_frozen=pov + [FREEZE], noisy_frozen=[NOISY] + pov + [FREEZE])
    return out


def _roles():
    pov = rls_overrides()
    return {"pursuer": pov, "evader": _mirror(pov)}


def _once(done, fn, what):
    """Run fn on one lane; the other lane waits for its DONE marker."""
    if os.path.isfile(done):
        return
    if not common._acquire_lock(done + ".lock"):
        while not os.path.isfile(done) and os.path.isfile(done + ".lock"):
            time.sleep(60)
        if not os.path.isfile(done):
            raise RuntimeError(f"{what} failed on the other lane")
        return
    try:
        fn()
        open(done, "w").close()
    finally:
        if os.path.isfile(done + ".lock"):
            os.remove(done + ".lock")


def _smoke_evals():
    """One short eval per F5c override set, a tiny exploiter, and the reused pursuers under noise (one lane; the other
    waits for it)."""
    root = os.path.join(F5C, "smoke")
    _once(os.path.join(root, "DONE"), lambda: _smoke_eval_jobs(root), "F5 smoke evals")


def _smoke_eval_jobs(root):
    c = name("uniform_rls", 0, True)
    ck = _policies(c, "pursuer")[-1]["checkpoint_path"]
    evs = _policies(c, "evader")
    pov = rls_overrides()
    target = {"id": f"f5/pursuer:{c}", "kind": "rl", "checkpoint_path": ck, "metadata": {}}
    ex_ck = train(os.path.join(root, "ckpt", "exploiter.pt"), [target], 2_000_000, role="evader", num_envs=256,
                  extra=_mirror(pov))
    jobs = [(f"{cond}", ck, ov, evs) for cond, ov in conditions("uniform_rls", pov).items()]
    jobs.append(("noisy_frozen_vs_exploiter", ck, conditions("uniform_rls", pov)["noisy_frozen"],
                 [{"id": "f5/evader:smoke_exploit", "kind": "rl", "checkpoint_path": ex_ck, "metadata": {}}]))
    for arm, camp in (("uniform_fd", "thesis-F2-acc"), ("qd_dpp_plain", f3_name("qd_dpp", 0))):
        jobs.append((f"{arm}_noisy", _policies(camp, "pursuer")[-1]["checkpoint_path"],
                     [NOISY] + (list(ACC) if arm == "uniform_fd" else []), evs))
    for tag, p_ck, ov, opps in jobs:
        r = evaluate(os.path.join(root, "evals", f"{tag}.json"), opps, policy_ckpt=p_ck, num_envs=256, cutoff=100,
                     extra=ov)
        if r is None or not r.get("matchups") or any("capture_rate" not in m for m in r["matchups"].values()):
            raise RuntimeError(f"F5 smoke eval {tag} failed: {r}")
    log_progress(f"F5 smoke evals ok ({len(jobs)})")


def independent_evaders():
    """(plain-obs evaders, acceleration-input evaders) for the estimator: RL evaders of self-play campaigns that F5c
    never evaluates and that share no checkpoint with one it does; duplicates (same file) dropped."""
    seen, plain = set(), []
    for c in INDEP_PLAIN:
        for e in _policies(c, "evader"):
            h = hashlib.md5(open(e["checkpoint_path"], "rb").read()).hexdigest()
            if h not in seen:
                seen.add(h)
                plain.append(e)
    return plain, _policies(INDEP_ACC, "evader")


def merge_traj(paths, out):
    """Concatenate recorded trajectories along the env axis, truncated to the shortest recording."""
    ds = [torch.load(p) for p in paths]
    T1 = min(d["z"].shape[0] for d in ds)
    torch.save({"z": torch.cat([d["z"][:T1] for d in ds], 1), "own_pos": torch.cat([d["own_pos"][:T1] for d in ds], 1),
                "done": torch.cat([d["done"][:T1 - 1] for d in ds], 1), "envs": torch.cat([d["envs"] for d in ds]),
                "opponent": [o for d in ds for o in d["opponent"]], "step_dt": ds[0]["step_dt"],
                "lengths": [d["z"].shape[0] for d in ds]}, out)
    return out


def _retrain_jobs():
    split = load_split()
    plain, acc = independent_evaders()
    gen = train(os.path.join(EST, "ckpt", "generalist.pt"), split["train"] + plain, GEN_FRAMES, role="pursuer",
                seed=GEN_SEED)
    if not gen:
        raise RuntimeError("F5 estimator generalist failed")
    roll = os.path.join(EST, "rollouts")
    n_main = len(split["train"]) + len(plain)
    jobs = [("train_plain", split["train"] + plain, REC_ENVS, []),
            ("train_acc", acc, round(REC_ENVS * len(acc) / n_main), OPP_ACC),   # same envs per opponent
            ("test", test_set(split), REC_ENVS, [])]
    for part, opps, n, ov in jobs:
        path = os.path.join(roll, f"traj_{part}.pt")
        if not os.path.isfile(path):
            evaluate(os.path.join(EST, "evals", f"record_{part}.json"), opps, policy_ckpt=gen, save_episodes=False,
                     extra=[f"eval_traj_out={path}", f"eval_traj_envs={n}"] + ov)
        if not os.path.isfile(path):
            raise RuntimeError(f"F5 estimator recording {part} failed")
    train_path = merge_traj([os.path.join(roll, f"traj_{p}.pt") for p in ("train_plain", "train_acc")],
                            os.path.join(roll, "traj_train.pt"))
    fit_acc_estimator({"train": train_path, "test": os.path.join(roll, "traj_test.pt")}, EST, "F5 estimator (A2a rules)")
    log_progress(f"F5 estimator retrained: {rls_overrides()}")


def retrain_estimator():
    _once(os.path.join(EST, "DONE"), _retrain_jobs, "F5 estimator retraining")


def run_f5(args):
    retrain_estimator()
    roles = _roles()
    for arm, n in SMOKE_STAGES.items():
        run_campaign(name(arm, 0, True), [], dict(NEW[arm]), smoke_stages=n, role_overrides=roles)
    for arm in SMOKE_STAGES:                 # the other lane may still be running one of the smokes
        d = _done(name(arm, 0, True))
        while not os.path.isfile(d) and os.path.isfile(d + ".lock"):
            time.sleep(60)
        if not os.path.isfile(d):
            raise RuntimeError(f"F5 smoke {arm} did not pass; campaigns not started")
    _smoke_evals()
    if args.smoke:
        return
    for k in range(len(SEEDS)):
        for arm in NEW:
            run_campaign(name(arm, k), [], dict(NEW[arm], seed=SEEDS[k]), role_overrides=roles)


def run_f5c(args):
    P = active_pursuers()
    while not (all(os.path.isfile(_done(c)) for a, _, c, _, _ in P if a in NEW)
               and all(os.path.isfile(ex) for a, _, _, _, ex in P if a not in NEW)):
        time.sleep(60)
    split = load_split()
    test = test_set(split)
    all_ev = [e for a, _, c, _, _ in pursuers() if a != "uniform_fd" and a not in PAUSED_ARMS
              for e in _policies(c, "evader")]
    for i, (a, k, c, pov, ex_path) in enumerate(P):
        ck = _policies(c, "pursuer")[-1]["checkpoint_path"]
        if a in NEW:
            target = {"id": f"f5/pursuer:{c}", "kind": "rl", "checkpoint_path": ck, "metadata": {}}
            train(ex_path, [target], EXPLOIT_FRAMES, role="evader", seed=700 + i, extra=_mirror(pov))
        ex = ([{"id": f"f5/evader:exploit_{c}", "kind": "rl", "checkpoint_path": ex_path, "metadata": {}}]
              if os.path.isfile(ex_path) else [])
        for cond, ov in conditions(a, pov).items():
            for g, opps in (("train", split["train"]), ("test", test), ("campaign_evaders", all_ev),
                            ("vs_exploiter", ex)):
                if opps:
                    evaluate(os.path.join(F5C, "evals", f"{c}_{cond}_{g}.json"), opps, policy_ckpt=ck, extra=ov)
        log_progress(f"F5c {c} done")
    need = [os.path.join(F5C, "evals", f"{c}_{cond}_{g}.json") for a, _, c, pov, ex in P for cond in conditions(a, pov)
            for g in ("train", "test", "campaign_evaders") + (("vs_exploiter",) if os.path.isfile(ex) else ())]
    missing = [x for x in need if not os.path.isfile(x)]
    if missing:                                   # the other lane is still producing them and analyses after
        log_progress(f"F5c: {len(missing)} eval files missing on this lane, analysis left to the other lane")
        return
    analyse_f5(P, split, test, suffix="_exp1" if PAUSED_ARMS else "")


def _rates(path, key):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return {k: v[key] for k, v in json.load(f)["matchups"].items()}


def _metrics(prefix, c, split, test, labels, key):
    J = {**_rates(prefix + "_train.json", key), **_rates(prefix + "_test.json", key)}
    Jc = {kk: v for kk, v in {**_rates(prefix + "_campaign_evaders.json", key),
                              **_rates(prefix + "_campaign_evaders_add.json", key)}.items()
          if not kk.endswith(":s1")}                 # amendment 1: stage-1 evaders are shared by same-seed campaigns
    if not J or not Jc:
        return None
    ids = lambda opps: [J[e["id"]] for e in opps if e["id"] in J]
    r = {"train": ids(split["train"]), "heldout_rl": ids(split["heldout_rl"]),
         "heldout_scripted": ids(split["heldout_scripted"]),
         "x3_exploiters": [J[e["id"]] for e in test if e["id"].startswith("exploit/") and e["id"] in J],
         "other_campaign_evaders": [v for kk, v in Jc.items() if not kk.startswith(c + "/")],
         "own_campaign_evaders": [v for kk, v in Jc.items() if kk.startswith(c + "/")]}
    r = {m: float(np.mean(v)) for m, v in r.items() if v}
    by_c = {}
    for e in split["heldout_rl"] + split["heldout_scripted"]:
        if e["id"] in J and e["id"] in labels:
            by_c.setdefault(labels[e["id"]], []).append(J[e["id"]])
    if by_c:
        r["heldout_worst_cluster"] = float(min(np.mean(v) for v in by_c.values()))
    ex = _rates(prefix + "_vs_exploiter.json", key)
    if ex:
        r["exploitability"] = 1 - float(next(iter(ex.values())))
    return r


def analyse_f5(P, split, test, suffix=""):
    import analysis as an
    with open(os.path.join(OUT, "X2_descriptors", "clusters.json")) as f:
        labels = json.load(f)["labels"]
    metrics = PRIMARY + SECONDARY
    rows = {}                                    # (arm, cond, J|cap) -> {k: metrics}
    for a, k, c, pov, _ in P:
        for cond in conditions(a, pov):
            for tag, key in (("J", "neutralization_rate"), ("cap", "capture_rate")):
                r = _metrics(os.path.join(F5C, "evals", f"{c}_{cond}"), c, split, test, labels, key)
                if r:
                    rows.setdefault((a, cond, tag), {})[k] = r
    agg = {}
    for (a, cond, tag), per in rows.items():
        for m in metrics:
            v = [per[k][m] for k in sorted(per) if m in per[k]]
            if v:
                agg.setdefault(f"{tag}/{cond}/{a}", {})[m] = {"mean": float(np.mean(v)), "ci": list(an.bootstrap_ci(v)[1:]),
                                                         "per_seed": v}
    diffs = {}
    for a, ca, b, cb in CONTRASTS:
        for tag in ("J", "cap"):
            ra, rb = rows.get((a, ca, tag), {}), rows.get((b, cb, tag), {})
            for m in metrics:
                d = [ra[k][m] - rb[k][m] for k in sorted(ra) if k in rb and m in ra[k] and m in rb[k]]
                if d:
                    diffs.setdefault(f"{tag}/{a}:{ca} - {b}:{cb}", {})[m] = {
                        "mean": float(np.mean(d)), "ci": list(an.bootstrap_ci(d)[1:]), "per_seed": d}
    save_json(os.path.join(F5C, f"results{suffix}.json"), {"aggregate": agg, "paired": diffs})
    arms = list(dict.fromkeys(a for a, *_ in P))
    fmt = lambda x, sign="": f"{x['mean']:{sign}.3f} [{x['ci'][0]:{sign}.3f}, {x['ci'][1]:{sign}.3f}]"
    lines = ["# F5_acc_sp summary" + (" (Exp 1 only: Exp 2 / qd_dpp_rls paused 11-05, user)" if suffix else ""), "",
             "Final pursuer (stage 8) of each campaign; mean over the campaign seeds (2; uniform_fd 1) [95 % bootstrap CI; "
             "with 2 seeds the CI is the range of the two values]. *_rls = learned acceleration estimate + RLS (F5); "
             "*_plain = plain observation (F2 / F3); uniform_fd = finite-difference acceleration (F2-acc). noisy = A3c "
             "sensor noise at eval; *_frozen = F5 pursuer evaluated with RLS frozen. exploitability = 1 - J (or capture) vs "
             "a fresh exploiter. J = capture + evader crash + evader out of bounds; cap = capture only. other / own campaign evaders exclude "
             "stage-1 evaders (amendment 1). Estimator retrained (amendment 1): `estimator/summary.md`.", ""]
    for tag, title in (("J", "J"), ("cap", "Capture only")):
        for cond in ("clean", "noisy", "clean_frozen", "noisy_frozen"):
            present = [a for a in arms if f"{tag}/{cond}/{a}" in agg]
            if not present:
                continue
            lines += [f"## {title}, {cond}", "", "| arm | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
            for a in present:
                g = agg[f"{tag}/{cond}/{a}"]
                lines.append(f"| {a} | " + " | ".join(fmt(g[m]) if m in g else "-" for m in metrics) + " |")
            lines.append("")
        lines += [f"## {title}, seed-paired differences", "", "| contrast | " + " | ".join(metrics) + " |",
                  "|---" * (len(metrics) + 1) + "|"]
        for key, dd in diffs.items():
            if key.startswith(tag + "/"):
                lines.append(f"| {key[len(tag) + 1:]} | " + " | ".join(fmt(dd[m], "+") if m in dd else "-"
                                                                        for m in metrics) + " |")
        lines.append("")
    with open(os.path.join(F5C, f"summary{suffix}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
