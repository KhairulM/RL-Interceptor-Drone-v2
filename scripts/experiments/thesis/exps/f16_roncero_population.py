"""F16_roncero_population: Hedge (F8b, reused) vs QD-DPP vs PSRO under Roncero's rules, both roles, three paired seeds
(docs/research_plan.md#f16; design 2026-11-16, implementation decisions 2026-11-22). PSD-PSRO was dropped by the user
on 2026-11-23 after its full-mechanism KL objective diverged in the smoke (the implementation stays in f16_psro.py /
hull_kl.py, unused here); with it went the pilot, whose only purpose was choosing PSD's lambda.

F16  (training): F8b audit + manifest -> bootstrap clones -> Isaac smoke of every arm -> frozen evaluation manifest ->
     six main campaigns.
F16c (evaluation): deployment mixtures -> 18 attackers (final pursuer / population mixture targets) -> every stored
     pursuer vs the common banks under thesis and Roncero endings -> locked analysis (results.json, summary.md).
Both entries are resumable and lane-safe: every job takes its own lock, and gates wait for the other lane.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import time

import numpy as np

import common
from common import OUT, REPO, _acquire_isaac_slot, _acquire_lock, _clean_env, log_progress, save_json
from f2_final import CAMPAIGN, _done
import f8_roncero_sp as f8
import f8d_heldout as f8d
from f3_sampler_sp import SAMPLERS
from rescore import rates_at
from x3_gap import load_split


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_fp = _load("scripts/selfplay/f16_psro.py", "f16d_psro")
_ts, _ps = _fp._ts, _fp._ps

F16 = os.path.join(OUT, "F16_roncero_population")
SEEDS = (0, 100, 200)
ARMS = ("hedge", "qd_dpp", "psro")
NEW_ARMS = ("qd_dpp", "psro")
RULES = list(f8.RULES)                          # task.reward_mode=roncero, algo.gae_cut_at_truncation=true
FRAMES = 50_000_000
STAGES = 26
GENERATIONS = 12
MIN_EPISODES = 2048
ATTACK_FRAMES = 100_000_000
RADII = (0.10, 0.15)
CONVENTIONS = ("thesis", "roncero")
EVAL_SEED = {"thesis": 16_000, "roncero": 17_000, "deterministic": 18_000}
ATTACK_SEED = {"final": 2600, "mixture": 3600}
ATTACK_SHARE = 0.5                              # fraction of eval envs given to the attack bank
NOTABLE = 0.05
SMOKE_FRAMES, SMOKE_ENVS, SMOKE_EPISODES = 2_000_000, 256, 256
F8_CLI = dict(CAMPAIGN, **SAMPLERS["hedge"], frames_per_stage=FRAMES, n_stages=STAGES, max_total_stages=STAGES)
# Registered primary contrast: QD-DPP - Hedge. PSRO - Hedge is descriptive (PSD, the registered attribution partner of
# PSRO, was dropped).
CONTRASTS = (("qd_dpp", "hedge"), ("psro", "hedge"))


def name(arm, s):
    return f"thesis-F8b-roncero-s{s}" if arm == "hedge" else f"thesis-F16-{arm}-s{s}"


def smoke_name(arm):
    return f"thesis-F16-smoke-{arm}"


def _out(c):
    return os.path.join(common.SELFPLAY_OUT, c)


def _wait(path, what, poll=60):
    while not os.path.isfile(path):
        if not os.path.isfile(path + ".lock"):
            raise RuntimeError(f"F16: {what} failed or not started: {path}")
        time.sleep(poll)


def _locked(path, fn):
    """fn() under path.lock unless path exists; False when another lane holds the lock."""
    if os.path.isfile(path):
        return True
    if not _acquire_lock(path + ".lock"):
        return False
    try:
        if not os.path.isfile(path):
            fn()
        return True
    finally:
        if os.path.isfile(path + ".lock"):
            os.remove(path + ".lock")


def _ensure(path, fn, what):
    if not _locked(path, fn):
        _wait(path, what)
    if not os.path.isfile(path):
        raise RuntimeError(f"F16: {what} produced no {path}")


# ======================================================================================== audit
AUDIT_FILES = ["omni_drones/envs/single/pursuit_evasion.py", "omni_drones/envs/single/roncero_reward.py",
               "omni_drones/learning/ppo/ppo.py", "omni_drones/learning/ppo/common.py",
               "omni_drones/utils/selfplay/opponents.py", "omni_drones/utils/selfplay/samplers.py",
               "omni_drones/utils/torchrl/transforms.py", "scripts/selfplay/run_stage.py",
               "scripts/selfplay/train_selfplay.py", "cfg/task/PursuitEvasion.yaml"]


def _git(*args):
    return subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True).stdout.strip()


def _cli_from_log(c):
    with open(os.path.join(OUT, "driver_log.txt")) as f:
        lines = [l for l in f if f"campaign {c} start:" in l]
    if len(lines) != 1:
        raise RuntimeError(f"F16 audit: expected one start line for {c}, found {len(lines)}")
    return lines[0][:19], dict(kv.split("=", 1) for kv in lines[0].split("start: ", 1)[1].split())


def expected_f8_cli(s):
    cli = {k: str(v) for k, v in F8_CLI.items()}
    cli.update(seed=str(s), run_name=name("hedge", s), warm_start="true",
               task_overrides="[" + ",".join(RULES) + "]", eval_extra_overrides="[eval_cutoff_steps=1500]")
    return cli


def audit_campaign(s):
    c, problems, stages = name("hedge", s), [], {}
    out = _out(c)
    if not os.path.isfile(_done(c)):
        problems.append("no DONE marker")
    ts, cli = _cli_from_log(c)
    exp = expected_f8_cli(s)
    diff = {k: [cli.get(k), v] for k, v in exp.items() if cli.get(k) != v}
    diff.update({k: [v, None] for k, v in cli.items() if k not in exp})
    if diff:
        problems.append(f"CLI differs: {diff}")
    for role in ("pursuer", "evader"):
        pool = json.load(open(os.path.join(out, "db", f"{role}_pool.json")))["policies"]
        rl = [p for p in pool if p["kind"] == "rl"]
        got = [int(p["metadata"]["stage"]) for p in rl]
        want = list(range(0 if role == "pursuer" else 1, STAGES, 2))
        if got != want:
            problems.append(f"{role} stages {got} != {want}")
        heur = [p["metadata"]["heuristic"] for p in pool if p["kind"] == "heuristic"]
        if heur != CAMPAIGN[f"seed_heuristics.{role}"].strip("[]").split(","):
            problems.append(f"{role} heuristic seeds {heur}")
        for p in rl:
            if not os.path.isfile(p["checkpoint_path"]):
                problems.append(f"missing {p['checkpoint_path']}")
                continue
            stages[f"{role}:{p['metadata']['stage']}"] = {"id": p["id"], "path": p["checkpoint_path"],
                                                          "sha256": _ps.sha256_file(p["checkpoint_path"])}
    for st in range(STAGES):
        role = "pursuer" if st % 2 == 0 else "evader"
        for part in ("train_opp", "train_out", "eval_opp", "eval_out"):
            if not os.path.isfile(os.path.join(out, "stages", f"stage{st}_{role}_{part}.json")):
                problems.append(f"missing stage{st}_{role}_{part}.json")
    commit = _git("rev-list", "-1", f"--before={ts}", "HEAD")
    changes = _git("log", "--format=%h %ad %s", "--date=iso", f"{commit}..HEAD", "--", *AUDIT_FILES).splitlines()
    diffstat = _git("diff", "--stat", commit, "--", *AUDIT_FILES).splitlines()
    return {"campaign": c, "start": ts, "cli": cli, "start_commit": commit, "stages": stages,
            "code_changes_since_start": changes, "working_tree_diff_vs_start": diffstat,
            "f16_code_note": "F16 additions (payoff returns, episode-reset resampling, stochastic opponents/agent, "
                             "hull KL, sampler diagnostics) are flag-gated and off by default", "problems": problems,
            "warm_start": "module weights only (actor, critic, value norm); fresh Adam optimiser each stage",
            "excluded": "withdrawn thesis-F8-roncero-* (pre k_sm fix) campaigns are not used"}


def audit():
    """Immutable F8b audit manifest (config / stage list / hashes / provenance); a later mismatch stops F16."""
    path = os.path.join(F16, "manifest.json")

    def make():
        rec = {"seeds": list(SEEDS), "rules": RULES, "frames_per_stage": FRAMES, "stages": STAGES,
               "campaigns": {str(s): audit_campaign(s) for s in SEEDS}, "frozen": time.strftime("%F %T")}
        bad = {s: a["problems"] for s, a in rec["campaigns"].items() if a["problems"]}
        if bad:
            save_json(os.path.join(F16, "audit_failed.json"), rec)
            raise RuntimeError(f"F16 audit: F8b pairing stopped, mismatches {bad}")
        save_json(path, rec)
        log_progress("F16 audit manifest frozen")

    _ensure(path, make, "F8b audit")
    man = json.load(open(path))
    for s in SEEDS:
        for k, v in man["campaigns"][str(s)]["stages"].items():
            if _ps.sha256_file(v["path"]) != v["sha256"]:
                raise RuntimeError(f"F16 audit: {name('hedge', s)} {k} changed since the manifest was frozen")
    return man


# ======================================================================================== bootstrap clones
def _clone(src, dest):
    out, sout = _out(dest), _out(src)
    if os.path.exists(os.path.join(out, "db")):
        raise RuntimeError(f"F16: {out} has a DB but no bootstrap marker")
    for d in ("checkpoints", "stages", "db"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    hashes, ids = {}, {}
    for role, st in (("pursuer", 0), ("evader", 1)):
        pool = json.load(open(os.path.join(sout, "db", f"{role}_pool.json")))
        entries = []
        for p in pool["policies"]:
            if p["kind"] == "heuristic":
                entries.append(p)
            elif int(p["metadata"]["stage"]) == st:
                dst = os.path.join(out, "checkpoints", f"{role}_stage{st}.pt")
                shutil.copy2(p["checkpoint_path"], dst)
                h = _ps.sha256_file(p["checkpoint_path"])
                if _ps.sha256_file(dst) != h:
                    raise RuntimeError("bootstrap copy hash mismatch")
                entries.append(dict(p, checkpoint_path=dst, metadata=dict(p["metadata"], sha256=h, source=src)))
                hashes[f"{role}_stage{st}"], ids[role] = h, p["id"]
        if len([p for p in entries if p["kind"] == "rl"]) != 1:
            raise RuntimeError(f"F16 bootstrap: {src} has no unique {role} stage {st}")
        save_json(os.path.join(out, "db", f"{role}_pool.json"), dict(pool, policies=entries))
    # Matchups: exactly the records written by the stage-0 pursuer and stage-1 evader evaluations.
    src_mt = json.load(open(os.path.join(sout, "db", "matchups.json")))["matchups"]
    mt = {}
    for st, role in ((0, "pursuer"), (1, "evader")):
        ev = json.load(open(os.path.join(sout, "stages", f"stage{st}_{role}_eval_out.json")))["matchups"]
        for opp in ev:
            key = f"{ids[role]}|{opp}" if role == "pursuer" else f"{opp}|{ids[role]}"
            if key not in src_mt:
                raise RuntimeError(f"F16 bootstrap: matchup {key} missing from {src}")
            mt[key] = src_mt[key]
        for part in ("train_opp", "train_out", "eval_opp", "eval_out"):
            shutil.copy2(os.path.join(sout, "stages", f"stage{st}_{role}_{part}.json"),
                         os.path.join(out, "stages", f"stage{st}_{role}_{part}.json"))
    save_json(os.path.join(out, "db", "matchups.json"), {"matchups": mt})
    save_json(os.path.join(out, "BOOTSTRAP.json"), {"source": src, "hashes": hashes, "ids": ids,
                                                    "matchups": sorted(mt), "time": time.strftime("%F %T")})


def verify_bootstrap(dest):
    out = _out(dest)
    b = json.load(open(os.path.join(out, "BOOTSTRAP.json")))
    for key, h in b["hashes"].items():
        if _ps.sha256_file(os.path.join(out, "checkpoints", key + ".pt")) != h:
            raise RuntimeError(f"F16 bootstrap {dest}: {key} hash mismatch (corrupt state)")
    src_pools = {r: json.load(open(os.path.join(_out(b["source"]), "db", f"{r}_pool.json")))["policies"]
                 for r in ("pursuer", "evader")}
    for r, st in (("pursuer", 0), ("evader", 1)):
        src = next(p for p in src_pools[r] if p["kind"] == "rl" and int(p["metadata"]["stage"]) == st)
        if _ps.sha256_file(src["checkpoint_path"]) != b["hashes"][f"{r}_stage{st}"]:
            raise RuntimeError(f"F16 bootstrap {dest}: source {r} stage {st} changed")
    return b


def clone_bootstrap(src, dest):
    marker = os.path.join(_out(dest), "BOOTSTRAP.json")
    _ensure(marker, lambda: _clone(src, dest), f"bootstrap clone {dest}")
    return verify_bootstrap(dest)


# ======================================================================================== campaigns
def qd_cli(c, s, frames=FRAMES, max_stages=STAGES, num_envs=None, sampler="qd_dpp"):
    cfg = dict(F8_CLI, **SAMPLERS[sampler], run_name=c, seed=s, frames_per_stage=frames, n_stages=max_stages,
               max_total_stages=max_stages)
    if sampler == "qd_dpp":
        cfg["sampler_diagnostics"] = "true"
    if num_envs:
        cfg["num_envs"] = num_envs
    return ([f"{k}={v}" for k, v in cfg.items()] + ["task_overrides=[" + ",".join(RULES) + "]",
                                                    "eval_extra_overrides=[eval_cutoff_steps=1500]"])


def psro_cli(c, arm, s, lam, gens=GENERATIONS, frames=FRAMES, extra=(), min_episodes=MIN_EPISODES):
    return [f"run_name={c}", f"arm={arm}", f"seed={s}", f"generations={gens}", f"frames_per_stage={frames}",
            f"hull.lam={float(lam)}", f"payoff_min_episodes={min_episodes}",
            "task_overrides=[" + ",".join(RULES) + "]"] + list(extra)


def _campaign(c, script, cli, check=None):
    """Run (or resume) one campaign holding one Isaac slot; DONE only after exit 0 and the optional check."""
    done = _done(c)
    if os.path.isfile(done):
        return True
    lock = done + ".lock"
    if not _acquire_lock(lock):
        log_progress(f"[skip: campaign {c} run by another lane]")
        return False
    try:
        slot = _acquire_isaac_slot()
        try:
            log_progress(f"campaign {c} start: {script} {' '.join(cli)}")
            with open(os.path.join(common.SELFPLAY_OUT, c + "_log.txt"), "a") as f:
                rc = subprocess.run([common.PYTHON, os.path.join(REPO, "scripts/selfplay", script)] + cli,
                                    cwd=os.path.join(REPO, "scripts/selfplay"), stdout=f, stderr=subprocess.STDOUT,
                                    env=_clean_env()).returncode
        finally:
            if os.path.isfile(slot):
                os.remove(slot)
        log_progress(f"campaign {c} exit {rc}")
        if rc != 0:
            raise RuntimeError(f"F16 campaign {c} exited {rc} (resumable; see {c}_log.txt)")
        if check:
            check()
        open(done, "w").close()
        return True
    finally:
        if os.path.isfile(lock):
            os.remove(lock)


def run_arm_campaign(arm, s):
    c = name(arm, s)
    clone_bootstrap(name("hedge", s), c)
    if arm == "qd_dpp":
        return _campaign(c, "train_selfplay.py", qd_cli(c, s))
    return _campaign(c, "f16_psro.py", psro_cli(c, arm, s, 0.0))


# ======================================================================================== Isaac jobs
def _spec(path, policies, probs, **extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_json(path, dict({"policies": policies, "probs": probs}, **extra))
    return path


def _run(out_json, args, base):
    rc = common.run_stage(args, base + "_log.txt")
    if rc != 0 or not os.path.isfile(out_json):
        raise RuntimeError(f"F16 job failed rc={rc}: {out_json}")


def payoff_eval(out_json, agent_ckpt, role, opponents, seed, num_envs=4096, min_episodes=MIN_EPISODES):
    """Both roles' discounted Roncero returns of every (agent, opponent) cell, stochastic actors (meta-game)."""
    def job():
        base = out_json[:-len(".json")]
        spec = _spec(base + "_opp.json", opponents, {o["id"]: 1 / len(opponents) for o in opponents},
                     stochastic=True)
        cutoff = _fp.cutoff_steps(num_envs, len(opponents), min_episodes, common.MAX_EPISODE_STEPS)
        _run(out_json, common.common_overrides(role, num_envs, seed) + [
            "mode=eval", f"opponent_spec_path={spec}", f"stage_out={out_json}", f"eval_cutoff_steps={cutoff}",
            f"eval_policy={agent_ckpt}", "eval_stochastic=true", "task.payoff_returns=true",
            f"eval_episodes_out={base}_episodes.pt"] + RULES, base)
    _ensure(out_json, job, "payoff evaluation")
    return json.load(open(out_json))["matchups"]


def weighted_eval(out_json, ckpt, opponents, probs, seed, extra, num_envs=4096, cutoff=common.EVAL_CUTOFF):
    def job():
        base = out_json[:-len(".json")]
        spec = _spec(base + "_opp.json", opponents, probs)
        _run(out_json, common.common_overrides("pursuer", num_envs, seed) + [
            "mode=eval", f"opponent_spec_path={spec}", f"stage_out={out_json}", f"eval_cutoff_steps={cutoff}",
            f"eval_policy={ckpt}", f"eval_episodes_out={base}_episodes.pt"] + list(extra), base)
    return _locked(out_json, job)


def train_attacker(path, pursuers, probs, seed, resample, num_envs=4096, frames=ATTACK_FRAMES):
    def job():
        base = path[:-len(".pt")]
        extra = {"stochastic": True}
        if resample:
            extra["resample"] = {"seed": seed}
        spec = _spec(base + "_opp.json", pursuers, probs, **extra)
        _run(path, common.common_overrides("evader", num_envs, seed) + [
            "mode=train", f"total_frames={frames}", f"opponent_spec_path={spec}", f"stage_ckpt={path}",
            f"stage_out={base}_out.json"] + RULES, base)
    return _locked(path, job)


# ======================================================================================== populations
def rl_members(c, role="pursuer"):
    common.CAMPAIGNS[c] = c
    return common.rl_policies(c, role)


def _db(c):
    return _ts.PolicyDatabase(os.path.join(_out(c), "db"))


def deployment_mixture(arm, s):
    """{pursuer stage: probability}. PSRO / PSD: the frozen final Nash pursuer mixture. Hedge / QD-DPP: the arm's own
    sampler over its RL pursuers with the final own evader as reference (within-campaign data only)."""
    c = name(arm, s)
    db = _db(c)
    stage = {p["id"]: _ts._policy_stage(p) for p in db.list_policies("pursuer") if p["kind"] == "rl"}
    if arm == "psro":
        x = json.load(open(os.path.join(_out(c), "meta", "final.json")))["x"]
    else:
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"evader_sampler": "hedge" if arm == "hedge" else "qd_dpp",
                                "hedge_gamma": 0.7, "hedge_eta": 2.0})
        final_e = [p for p in db.list_policies("evader") if p["kind"] == "rl"][-1]
        x = _ts._opponent_probs(db, "evader", final_e["id"], cfg)
    mix = {str(stage[k]): float(v) for k, v in x.items() if v > 0}
    if not mix or abs(sum(mix.values()) - 1) > 1e-6 or not all(np.isfinite(list(mix.values()))):
        raise RuntimeError(f"F16: invalid deployment mixture for {c}: {x}")
    return mix


def mixtures():
    path = os.path.join(F16, "mixtures.json")
    _ensure(path, lambda: save_json(path, {f"{a}/{s}": deployment_mixture(a, s) for a in ARMS for s in SEEDS}),
            "deployment mixtures")
    return json.load(open(path))


# ======================================================================================== evaluation manifest + banks
def _policy(p):
    return {"id": p["id"], "kind": "rl", "checkpoint_path": p["checkpoint_path"], "metadata": p.get("metadata", {})}


def cross_ids(s):
    return [{"id": f"{name(a, o)}/evader:s{st}", "arm": a, "campaign": name(a, o), "stage": st}
            for a in ARMS for o in SEEDS if o != s for st in range(3, STAGES, 2)]


def cross_weights(entries):
    """Arms equal, then campaigns within an arm, then stages within a campaign."""
    arms = sorted({e["arm"] for e in entries})
    w = {}
    for a in arms:
        camps = sorted({e["campaign"] for e in entries if e["arm"] == a})
        for c in camps:
            st = [e for e in entries if e["campaign"] == c]
            for e in st:
                w[e["id"]] = w.get(e["id"], 0.0) + 1 / len(arms) / len(camps) / len(st)
    return w


def attack_ids(s):
    return [{"id": f"f16/evader:attacker-{a}-{t}-s{s}", "arm": a, "target": t, "seed": ATTACK_SEED[t] + s,
             "path": os.path.join(F16, "ckpt", f"attacker_{a}_{t}_s{s}.pt")} for a in ARMS for t in ATTACK_SEED]


def eval_manifest():
    """Frozen before the main campaigns: every bank, deterministic ids of not-yet-trained opponents, weights, seeds."""
    path = os.path.join(F16, "eval_manifest.json")

    def make():
        split = load_split()
        strong = [_policy(f8.all_stages(c, "evader")[-1]) for c in f8d.COMMON]
        v10 = [_policy(p) for p in split["heldout_rl"]]
        for p in strong + v10:
            p["sha256"] = _ps.sha256_file(p["checkpoint_path"])
        save_json(path, {
            "strong": strong, "v10": v10,
            "cross": {str(s): cross_ids(s) for s in SEEDS},
            "cross_weights": {str(s): cross_weights(cross_ids(s)) for s in SEEDS},
            "attack": {str(s): attack_ids(s) for s in SEEDS},
            "seeds": {"eval": EVAL_SEED, "attack": ATTACK_SEED},
            "allocation": f"attack bank {ATTACK_SHARE:.0%} of 4096 envs (equal per attacker), rest equal over "
                          "strong + v10 + cross; every episode starting in the first 1500 steps counted",
            "execution": "evaluated pursuer stochastic (primary) or mean action (deterministic secondary, finals vs "
                         "strong bank); bank opponents act deterministically as in every earlier thesis eval",
            "radii": list(RADII), "conventions": list(CONVENTIONS), "frozen": time.strftime("%F %T")})
        log_progress("F16 evaluation manifest frozen")

    _ensure(path, make, "evaluation manifest")
    return json.load(open(path))


def bank(s, man):
    """[(policy, prob)] for target seed s: strong + v10 + cross + attack (dedupe identical checkpoints)."""
    pols, seen = [], {}
    cross = []
    for e in man["cross"][str(s)]:
        p = next(q for q in rl_members(e["campaign"], "evader") if q["metadata"]["stage"] == e["stage"])
        cross.append(dict(_policy(p), id=e["id"]))
    attack = [{"id": a["id"], "kind": "rl", "checkpoint_path": a["path"], "metadata": {}}
              for a in man["attack"][str(s)]]
    aliases = {}
    for p in man["strong"] + man["v10"] + cross + attack:
        h = p.get("sha256") or _ps.sha256_file(p["checkpoint_path"])
        if h in seen:
            aliases[p["id"]] = seen[h]
            continue
        seen[h] = p["id"]
        pols.append({k: p[k] for k in ("id", "kind", "checkpoint_path", "metadata")})
    att = {a["id"] for a in attack}
    n_att, n_rest = sum(p["id"] in att for p in pols), sum(p["id"] not in att for p in pols)
    probs = {p["id"]: ATTACK_SHARE / n_att if p["id"] in att else (1 - ATTACK_SHARE) / n_rest for p in pols}
    return pols, probs, aliases


def member_eval(arm, s, member, conv, man, deterministic=False):
    c = name(arm, s)
    st = member["metadata"]["stage"]
    if deterministic:
        out = os.path.join(F16, "evals", f"{c}_P{st}_deterministic.json")
        pols = [{k: p[k] for k in ("id", "kind", "checkpoint_path", "metadata")} for p in man["strong"]]
        return weighted_eval(out, member["checkpoint_path"], pols, {p["id"]: 1 / len(pols) for p in pols},
                             EVAL_SEED["deterministic"] + s, [])
    out = os.path.join(F16, "evals", f"{c}_P{st}_{conv}.json")
    if os.path.isfile(out):
        return True
    pols, probs, _ = bank(s, man)
    extra = ["eval_stochastic=true"] + (RULES if conv == "roncero" else [])
    return weighted_eval(out, member["checkpoint_path"], pols, probs, EVAL_SEED[conv] + s, extra)


# ======================================================================================== smoke
def run_smoke():
    marker = os.path.join(F16, "smoke", "SMOKE_DONE.json")
    if os.path.isfile(marker):
        return
    src = name("hedge", 0)
    for arm in NEW_ARMS:
        clone_bootstrap(src, smoke_name(arm))
    small = [f"num_envs={SMOKE_ENVS}", f"payoff_num_envs={SMOKE_ENVS}"]
    _campaign(smoke_name("qd_dpp"), "train_selfplay.py",
              qd_cli(smoke_name("qd_dpp"), 0, frames=SMOKE_FRAMES, max_stages=4, num_envs=SMOKE_ENVS))
    _campaign(smoke_name("psro"), "f16_psro.py", psro_cli(smoke_name("psro"), "psro", 0, 0.0, gens=1,
                                                          frames=SMOKE_FRAMES, extra=small, min_episodes=SMOKE_EPISODES))
    for arm in NEW_ARMS:
        _wait(_done(smoke_name(arm)), f"smoke campaign {arm}")

    def checks():
        res = {}
        # payoff orientation: the same cell measured with either role as the evaluated agent
        c = smoke_name("psro")
        P0, E1 = rl_members(c, "pursuer")[0], rl_members(c, "evader")[0]
        d = os.path.join(F16, "smoke", "orientation")
        as_p = payoff_eval(os.path.join(d, "as_pursuer.json"), P0["checkpoint_path"], "pursuer", [_policy(E1)], 901,
                           num_envs=1024, min_episodes=1024)[E1["id"]]["payoff"]
        as_e = payoff_eval(os.path.join(d, "as_evader.json"), E1["checkpoint_path"], "evader", [_policy(P0)], 902,
                           num_envs=1024, min_episodes=1024)[P0["id"]]["payoff"]
        res["orientation"] = {}
        for role in ("pursuer", "evader"):
            se = np.hypot(as_p[f"std_{role}"] / np.sqrt(as_p["n"]), as_e[f"std_{role}"] / np.sqrt(as_e["n"]))
            diff = as_p[f"mean_{role}"] - as_e[f"mean_{role}"]
            res["orientation"][role] = {"as_pursuer": as_p[f"mean_{role}"], "as_evader": as_e[f"mean_{role}"],
                                        "diff": diff, "se": float(se), "ok": bool(abs(diff) <= 4 * se)}
        # QD-DPP sampler diagnostics, PSD activity, frozen meta-strategies
        diag = [json.loads(l) for l in open(os.path.join(_out(smoke_name("qd_dpp")), "stages",
                                                         "sampler_diagnostics.jsonl"))]
        res["qd_dpp"] = [{"stage": r["stage"], "role": r["role"], "tv": r.get("tv_from_hedge")} for r in diag]
        m = json.load(open(os.path.join(_out(smoke_name("psro")), "meta", "final.json")))
        res["psro_final_meta"] = {"x": m["x"], "y": m["y"], "regrets": [m["regret_row"], m["regret_col"]]}
        # evaluation pipeline: weighted stochastic member eval + mixture attacker with per-episode resampling
        ps = rl_members(smoke_name("psro"), "pursuer")
        man_strong = [_policy(f8.all_stages(c2, "evader")[-1]) for c2 in f8d.COMMON[:2]]
        ev = os.path.join(F16, "smoke", "member_eval.json")
        weighted_eval(ev, ps[-1]["checkpoint_path"], man_strong, {p["id"]: 0.5 for p in man_strong}, 903,
                      ["eval_stochastic=true"], num_envs=SMOKE_ENVS, cutoff=300)
        att = os.path.join(F16, "smoke", "attacker.pt")
        train_attacker(att, [_policy(p) for p in ps], {p["id"]: 1 / len(ps) for p in ps}, 904, True,
                       num_envs=SMOKE_ENVS, frames=SMOKE_FRAMES)
        counts = json.load(open(att[:-3] + "_out.json"))["opponent_episode_counts"]
        res["attacker_resample_counts"] = counts
        ok = (all(v["ok"] for v in res["orientation"].values()) and all(np.isfinite(r["tv"]) for r in res["qd_dpp"]
              if r["tv"] is not None) and len(res["qd_dpp"]) >= 2 and all(v > 0 for v in counts.values()) and
              json.load(open(ev))["eval"]["stochastic_agent"])
        res["passed"] = bool(ok)
        save_json(os.path.join(F16, "smoke", "checks.json"), res)
        if not ok:
            raise RuntimeError(f"F16 smoke gate failed: see {F16}/smoke/checks.json")
        save_json(marker, dict(res, time=time.strftime("%F %T")))
        log_progress("F16 smoke gates passed")

    _ensure(marker, checks, "smoke gates")


# ======================================================================================== entry points
def run_f16(args):
    os.makedirs(F16, exist_ok=True)
    audit()
    run_smoke()
    if args.smoke:
        return
    eval_manifest()
    for s in SEEDS:
        for arm in ("psro", "qd_dpp"):
            run_arm_campaign(arm, s)
    log_progress("F16: this lane has no unclaimed main campaign left")


def _all_done():
    for arm in NEW_ARMS:
        for s in SEEDS:
            _wait(_done(name(arm, s)), f"main campaign {name(arm, s)}", poll=120)


def run_f16c(args):
    _all_done()
    man = eval_manifest()
    mix = mixtures()
    for s in SEEDS:                                       # 24 attackers, paired seeds 2600 + s / 3600 + s
        for arm in ARMS:
            members = {m["metadata"]["stage"]: m for m in rl_members(name(arm, s))}
            a = {x["target"]: x for x in man["attack"][str(s)] if x["arm"] == arm}
            train_attacker(a["final"]["path"], [_policy(members[STAGES - 2])], {members[STAGES - 2]["id"]: 1.0},
                           a["final"]["seed"], False)
            m = mix[f"{arm}/{s}"]
            pols = [_policy(members[int(st)]) for st in m]
            train_attacker(a["mixture"]["path"], pols, {members[int(st)]["id"]: p for st, p in m.items()},
                           a["mixture"]["seed"], True)
    for s in SEEDS:
        for a in man["attack"][str(s)]:
            _wait(a["path"], f"attacker {a['id']}", poll=120)
    jobs = []
    for s in SEEDS:
        for arm in ARMS:
            ms = rl_members(name(arm, s))
            final = ms[-1]
            jobs += [(arm, s, final, conv, False) for conv in CONVENTIONS] + [(arm, s, final, "thesis", True)]
            jobs += [(arm, s, m, conv, False) for m in ms[:-1] for conv in CONVENTIONS]
    for arm, s, m, conv, det in jobs:
        member_eval(arm, s, m, conv, man, deterministic=det)
    missing = [p for p in eval_paths() if not os.path.isfile(p)]
    if missing:
        log_progress(f"F16c: {len(missing)} evals still running on the other lane; analysis left to it")
        return
    _locked(os.path.join(F16, "ANALYSIS_DONE"), analyse)


def eval_paths():
    out = []
    for s in SEEDS:
        for arm in ARMS:
            ms = rl_members(name(arm, s))
            for m in ms:
                st = m["metadata"]["stage"]
                out += [os.path.join(F16, "evals", f"{name(arm, s)}_P{st}_{conv}.json") for conv in CONVENTIONS]
            out.append(os.path.join(F16, "evals", f"{name(arm, s)}_P{ms[-1]['metadata']['stage']}_deterministic.json"))
    return out


# ======================================================================================== analysis
def contrast(a, b):
    """Per-seed a - b, mean, and the registered descriptive decision (+/-0.05 mean, same sign on all seeds)."""
    d = {s: a[s] - b[s] for s in a if s in b}
    mean = float(np.mean(list(d.values())))
    if mean >= NOTABLE and all(v > 0 for v in d.values()):
        dec = "notable gain"
    elif mean <= -NOTABLE and all(v < 0 for v in d.values()):
        dec = "notable decrease"
    else:
        dec = "inconclusive"
    return {"per_seed": d, "mean": mean, "decision": dec}


def maximin(J):
    """max_x min_a sum_i x_i J[i, a] over the simplex (finite zero-sum coverage LP). Returns (value, x)."""
    from scipy.optimize import linprog
    J = np.asarray(J, float)
    n, k = J.shape
    c = np.zeros(n + 1)
    c[-1] = -1.0                                            # maximise v
    A = np.hstack([-J.T, np.ones((k, 1))])                  # v - sum_i x_i J[i, a] <= 0
    res = linprog(c, A_ub=A, b_ub=np.zeros(k), A_eq=np.hstack([np.ones((1, n)), np.zeros((1, 1))]), b_eq=[1.0],
                  bounds=[(0, None)] * n + [(None, None)], method="highs")
    if not res.success:
        raise RuntimeError(f"coverage LP failed: {res.message}")
    return float(-res.fun), res.x[:n]


def summarise(per_opp, ids, weights=None):
    v = [(per_opp[i], (weights or {}).get(i, 1.0)) for i in ids if i in per_opp]
    if not v:
        return None
    x, w = np.array(v).T
    return float((x * w).sum() / w.sum())


def mix_opp(per_member, x):
    """Exact per-opponent value of the episode-sampled mixture: sum_i x_i value_i (identical opponent sets)."""
    keys = set.intersection(*(set(per_member[st]) for st in x))
    return {k: float(sum(p * per_member[st][k] for st, p in x.items())) for k in keys}


def _rates(path, key, radius, conv):
    return rates_at(path, key, radius, pursuer_oob_terminal=(conv == "roncero"))


def analyse():
    man = eval_manifest()
    mix = mixtures()
    labels = json.load(open(os.path.join(OUT, "X2_descriptors", "clusters.json")))["labels"]
    strong = [p["id"] for p in man["strong"]]
    v10 = [p["id"] for p in man["v10"]]
    res = {}
    aliases = {s: bank(s, man)[2] for s in SEEDS}      # deduplicated identical checkpoints share one estimate
    for radius in RADII:
        for conv in CONVENTIONS:
            key = f"r{radius:.2f}/{conv}"
            res[key] = {}
            for arm in ARMS:
                for s in SEEDS:
                    c = name(arm, s)
                    cw = man["cross_weights"][str(s)]
                    hedge_cross = [e["id"] for e in man["cross"][str(s)] if e["arm"] == "hedge"]
                    att = [a["id"] for a in man["attack"][str(s)]]
                    per = {}
                    for m in rl_members(c):
                        st = m["metadata"]["stage"]
                        p = os.path.join(F16, "evals", f"{c}_P{st}_{conv}.json")
                        per[st] = {k: _rates(p, k, radius, conv) for k in
                                   ("neutralization_rate", "capture", "timeout", "pursuer_crash", "pursuer_oob")}
                        for d in per[st].values():
                            d.update({a: d[k] for a, k in aliases[s].items() if k in d})
                    def endpoints(J):
                        by_c = {}
                        for i in v10:
                            if i in J["neutralization_rate"] and i in labels:
                                by_c.setdefault(labels[i], []).append(J["neutralization_rate"][i])
                        n = J["neutralization_rate"]
                        sj, cj = summarise(n, strong), summarise(n, list(cw), cw)
                        out = {"G1": 0.5 * sj + 0.5 * cj, "strong": sj, "cross": cj, "v10": summarise(n, v10),
                               "default_test": summarise(n, v10 + strong), "f8_anchor": summarise(n, hedge_cross),
                               "worst_cluster": float(min(np.mean(v) for v in by_c.values())) if by_c else None,
                               "attack": {a: n[a] for a in att if a in n}}
                        for k in ("capture", "timeout", "pursuer_crash", "pursuer_oob"):
                            out[f"strong_{k}"] = summarise(J[k], strong)
                        out["attack_worst"] = min(out["attack"].values()) if out["attack"] else None
                        return out
                    final_st = max(per)
                    x = {int(k): v for k, v in mix[f"{arm}/{s}"].items()}
                    single = endpoints(per[final_st])
                    mixed = endpoints({k: mix_opp({st: per[st][k] for st in per}, x) for k in per[final_st]})
                    own = {t: f"f16/evader:attacker-{arm}-{t}-s{s}" for t in ATTACK_SEED}
                    single["own_attacker"] = single["attack"].get(own["final"])
                    mixed["own_attacker"] = mixed["attack"].get(own["mixture"])
                    stages = sorted(per)
                    Jm = np.array([[per[st]["neutralization_rate"][a] for a in att] for st in stages])
                    v, xs = maximin(Jm)
                    res[key][f"{arm}/{s}"] = {"final": single, "mixture": mixed, "mixture_weights": x,
                                              "coverage_exploitability": 1 - v,
                                              "coverage_mix": dict(zip(map(str, stages), xs.tolist()))}
    det = {}
    for arm in ARMS:
        for s in SEEDS:
            c = name(arm, s)
            st = rl_members(c)[-1]["metadata"]["stage"]
            p = os.path.join(F16, "evals", f"{c}_P{st}_deterministic.json")
            det[f"{arm}/{s}"] = {f"r{r:.2f}": summarise(rates_at(p, "neutralization_rate", r), strong) for r in RADII}
    contrasts = {}
    for key, r in res.items():
        contrasts[key] = {}
        for a, b in CONTRASTS:
            g = lambda arm, f: {s: f(r[f"{arm}/{s}"]) for s in SEEDS}
            contrasts[key][f"{a}-{b}"] = {
                "final_G1": contrast(g(a, lambda v: v["final"]["G1"]), g(b, lambda v: v["final"]["G1"])),
                "final_strong": contrast(g(a, lambda v: v["final"]["strong"]), g(b, lambda v: v["final"]["strong"])),
                "final_cross": contrast(g(a, lambda v: v["final"]["cross"]), g(b, lambda v: v["final"]["cross"])),
                "mixture_G1": contrast(g(a, lambda v: v["mixture"]["G1"]), g(b, lambda v: v["mixture"]["G1"])),
                "mixture_attack_worst": contrast(g(a, lambda v: v["mixture"]["attack_worst"]),
                                                 g(b, lambda v: v["mixture"]["attack_worst"])),
                "final_attack_worst": contrast(g(a, lambda v: v["final"]["attack_worst"]),
                                               g(b, lambda v: v["final"]["attack_worst"]))}
    out = {"results": res, "contrasts": contrasts, "deterministic_strong": det, "training": training_diagnostics(),
           "hypotheses": hypotheses(contrasts, res)}
    save_json(os.path.join(F16, "results.json"), out)
    write_summary(out)
    open(os.path.join(F16, "ANALYSIS_DONE"), "w").close()
    log_progress("F16 analysis complete; update RESULTS/PROGRESS before declaring the study complete")


def training_diagnostics():
    d = {}
    for s in SEEDS:
        c = name("qd_dpp", s)
        rows = [json.loads(l) for l in open(os.path.join(_out(c), "stages", "sampler_diagnostics.jsonl"))]
        tv = {r: [x["tv_from_hedge"] for x in rows if x["role"] == r and x.get("tv_from_hedge") is not None]
              for r in ("pursuer", "evader")}
        d[f"qd_dpp/{s}"] = {"mean_tv": {r: float(np.mean(v)) if v else None for r, v in tv.items()},
                            "weak_intervention": all(v and np.mean(v) < 0.02 for v in tv.values())}
        for arm in ("psro",):
            c = name(arm, s)
            metas = []
            for g in list(range(1, GENERATIONS + 1)) + ["final"]:
                m = json.load(open(os.path.join(_out(c), "meta", f"gen{g}.json" if g != "final" else "final.json")))
                metas.append({"tag": m["tag"], "support_pursuer": sum(v > 0 for v in m["x"].values()),
                              "support_evader": sum(v > 0 for v in m["y"].values()), "entropy": m["joint_entropy"],
                              "regrets": [m["regret_row"], m["regret_col"]], "verified": m["n_verified_distinct"]})
            d[f"{arm}/{s}"] = {"meta": metas}
    return d


def hypotheses(contrasts, res):
    key = "r0.10/thesis"
    tv = training_diagnostics()
    return {"QD-DPP stays close to Hedge": {
                "final_G1_qd_minus_hedge": contrasts[key]["qd_dpp-hedge"]["final_G1"],
                "weak_intervention_all_seeds": all(tv[f"qd_dpp/{s}"]["weak_intervention"] for s in SEEDS)},
            "QD-DPP does not reliably exceed Hedge by +0.05 final G1": {
                "decision": contrasts[key]["qd_dpp-hedge"]["final_G1"]["decision"]},
            "PSD hypotheses": "not tested (PSD dropped after the smoke, 2026-11-23)"}


def write_summary(out):
    L = ["# F16_roncero_population summary", "",
         "Hedge = F8b (reused training, fresh common evaluation); QD-DPP / PSRO = new campaigns from the same F8b "
         "stage-0/1 bootstrap (PSD dropped after its smoke). Three paired seeds, descriptive. G1 = 0.5 strong-bank J + 0.5 cross-campaign J "
         "(other seeds, arms / campaigns / stages weighted equally). Mixtures: exact expectation of one pursuer sampled "
         "per episode. Evaluated pursuer stochastic; deterministic strong-bank J in the last table. 0.15 m values are "
         "re-scores of the same episodes (upper bounds; all policies trained at 0.10 m).", ""]
    for key, r in out["results"].items():
        L += [f"## {key}", "", "| arm/seed | final G1 | strong | cross | v10 | F8 anchor | worst cluster | own att. | "
              "bank worst | mix G1 | mix own att. | mix bank worst | coverage exploitability |",
              "|---" * 13 + "|"]
        f = lambda v: "-" if v is None else f"{v:.3f}"
        for k, v in r.items():
            a, m = v["final"], v["mixture"]
            L.append(f"| {k} | {f(a['G1'])} | {f(a['strong'])} | {f(a['cross'])} | {f(a['v10'])} | {f(a['f8_anchor'])}"
                     f" | {f(a['worst_cluster'])} | {f(a['own_attacker'])} | {f(a['attack_worst'])} | {f(m['G1'])} | "
                     f"{f(m['own_attacker'])} | {f(m['attack_worst'])} | {f(v['coverage_exploitability'])} |")
        L += ["", "| contrast | metric | " + " | ".join(f"s{s}" for s in SEEDS) + " | mean | decision |",
              "|---" * (4 + len(SEEDS)) + "|"]
        for cname, cm in out["contrasts"][key].items():
            for metric, c in cm.items():
                L.append(f"| {cname} | {metric} | " + " | ".join(f"{c['per_seed'][s]:+.3f}" for s in SEEDS) +
                         f" | {c['mean']:+.3f} | {c['decision']} |")
        L.append("")
    L += ["## Deterministic-actor final pursuer vs the strong bank (secondary)", "", "| arm/seed | 0.10 m | 0.15 m |",
          "|---|---|---|"]
    L += [f"| {k} | {v['r0.10']:.3f} | {v['r0.15']:.3f} |" for k, v in out["deterministic_strong"].items()]
    with open(os.path.join(F16, "summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
