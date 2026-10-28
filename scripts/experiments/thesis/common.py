"""Shared helpers for the thesis experiments (docs/research_plan.md).

Every Isaac job is a subprocess of scripts/selfplay/run_stage.py (one simulation per process). Jobs are skipped when
their output JSON already exists, so re-running an experiment resumes it.
"""

import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
OUT = os.path.join(REPO, "experiments/thesis")
RUN_STAGE = os.path.join(REPO, "scripts/selfplay/run_stage.py")
SELFPLAY_OUT = os.path.join(REPO, "scripts/selfplay/outputs")
PYTHON = os.path.join(REPO, ".venv/bin/python")

# Campaign tag -> output dir name under scripts/selfplay/outputs.
CAMPAIGNS = {
    "v6": "selfplay_v6_fix_reward_and_bounds",
    "v9": "v9-min-distance",
    "v10": "v10-without-warm-start",
    "v11": "v11-uniform-mix",
}

# One env step = sim.dt (0.01) x substeps (2).
STEP_DT = 0.02
MAX_EPISODE_STEPS = 1500
EVAL_CUTOFF = 1500        # count every episode that starts in the first 30 s of the eval rollout


def campaign_dir(tag: str) -> str:
    return os.path.join(SELFPLAY_OUT, CAMPAIGNS[tag])


def campaign_db(tag: str):
    """(pursuer pool, evader pool, matchups) as raw JSON."""
    db = os.path.join(campaign_dir(tag), "db")
    with open(os.path.join(db, "pursuer_pool.json")) as f:
        pursuers = json.load(f)["policies"]
    with open(os.path.join(db, "evader_pool.json")) as f:
        evaders = json.load(f)["policies"]
    with open(os.path.join(db, "matchups.json")) as f:
        matchups = json.load(f).get("matchups", {})
    return pursuers, evaders, matchups


def rl_policies(tag: str, role: str) -> List[dict]:
    """RL checkpoints of a campaign for ``role`` with globally unique ids ``<tag>/<role>:s<stage>``."""
    pursuers, evaders, _ = campaign_db(tag)
    pool = pursuers if role == "pursuer" else evaders
    out = []
    for p in pool:
        if p["kind"] != "rl" or not p.get("checkpoint_path") or not os.path.isfile(p["checkpoint_path"]):
            continue
        stage = (p.get("metadata") or {}).get("stage")
        out.append({"id": f"{tag}/{role}:s{stage}", "kind": "rl", "checkpoint_path": p["checkpoint_path"],
                    "metadata": {"campaign": tag, "stage": stage, "db_id": p["id"]}})
    return sorted(out, key=lambda p: p["metadata"]["stage"])


def heuristic(name: str, role: str = "evader") -> dict:
    return {"id": f"heur/{role}:{name}", "kind": "heuristic", "checkpoint_path": None,
            "metadata": {"heuristic": name}}


def write_spec(path: str, policies: List[dict], probs: Optional[Dict[str, float]] = None) -> str:
    if probs is None:
        probs = {p["id"]: 1.0 / len(policies) for p in policies}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"policies": policies, "probs": probs}, f, indent=2)
    return path


def _clean_env() -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)   # ROS humble (py3.10) paths break the py3.11 venv
    return env


# Each Isaac process needs ~17.5 GB of host RAM (62 GB machine): at most 2 may run at once, across all drivers.
# A third concurrent process caused repeated OOM kills on 10-28 15:40-16:14.
MAX_ISAAC = 2
SLOT_DIR = os.path.join(OUT, ".isaac_slots")


def _acquire_isaac_slot() -> str:
    """Block until one of MAX_ISAAC slot files is free; returns its path (holds this driver's PID for now)."""
    waited = False
    while True:
        for k in range(MAX_ISAAC):
            slot = os.path.join(SLOT_DIR, f"slot{k}.lock")
            if _acquire_lock(slot):
                if waited:
                    log_progress(f"[isaac slot {k} acquired]")
                return slot
        if not waited:
            log_progress(f"[waiting for a free Isaac slot (max {MAX_ISAAC})]")
            waited = True
        time.sleep(30)


def run_stage(args: List[str], log: str, retries: int = 2) -> int:
    """Run one run_stage.py job (blocking). Returns the exit code of the last attempt."""
    os.makedirs(os.path.dirname(log), exist_ok=True)
    cmd = [PYTHON, RUN_STAGE] + args
    rc = 1
    for attempt in range(retries + 1):
        slot = _acquire_isaac_slot()
        try:
            t0 = time.time()
            with open(log, "a") as f:
                f.write(f"\n### attempt {attempt} {time.strftime('%F %T')}\n{' '.join(cmd)}\n")
                f.flush()
                proc = subprocess.Popen(cmd, cwd=os.path.dirname(RUN_STAGE), stdout=f, stderr=subprocess.STDOUT,
                                        env=_clean_env())
                with open(slot, "w") as sf:     # the slot is held while the Isaac process itself is alive
                    sf.write(str(proc.pid))
                rc = proc.wait()
                f.write(f"### exit {rc} after {time.time() - t0:.0f} s\n")
        finally:
            if os.path.isfile(slot):
                os.remove(slot)
        if rc == 0:
            break
    return rc


def common_overrides(role: str, num_envs: int, seed: int) -> List[str]:
    return ["task=PursuitEvasion", f"task.role={role}", f"task.env.num_envs={num_envs}", "headless=true",
            f"seed={seed}", "wandb.mode=disabled"]


def evaluate(out_json: str, opponents: List[dict], role: str = "pursuer", policy_ckpt: Optional[str] = None,
             heuristic_agent: Optional[str] = None, num_envs: int = 4096, seed: int = 0,
             cutoff: int = EVAL_CUTOFF, extra: Optional[List[str]] = None, save_episodes: bool = True) -> Optional[dict]:
    """Evaluate one agent (checkpoint or heuristic) against ``opponents`` (envs split uniformly).

    Returns None if the eval failed or another lane is running it (lock).
    """
    if os.path.isfile(out_json):
        with open(out_json) as f:
            return json.load(f)
    lock = out_json + ".lock"
    if not _acquire_lock(lock):
        log_progress(f"[skip: eval locked by another lane] {out_json}")
        return None
    try:
        return _evaluate(out_json, opponents, role, policy_ckpt, heuristic_agent, num_envs, seed, cutoff, extra,
                         save_episodes)
    finally:
        if os.path.isfile(lock):
            os.remove(lock)


def _evaluate(out_json, opponents, role, policy_ckpt, heuristic_agent, num_envs, seed, cutoff, extra, save_episodes):
    if os.path.isfile(out_json):
        with open(out_json) as f:
            return json.load(f)
    base = os.path.splitext(out_json)[0]
    spec = write_spec(base + "_opp.json", opponents)
    args = common_overrides(role, num_envs, seed) + [
        "mode=eval", f"opponent_spec_path={spec}", f"stage_out={out_json}", f"eval_cutoff_steps={cutoff}"]
    if save_episodes:
        args.append(f"eval_episodes_out={base}_episodes.pt")
    if policy_ckpt:
        args.append(f"eval_policy={policy_ckpt}")
    elif heuristic_agent:
        args.append(f"eval_heuristic={heuristic_agent}")
    else:
        raise ValueError("need policy_ckpt or heuristic_agent")
    args += list(extra or [])
    rc = run_stage(args, base + "_log.txt")
    if rc != 0 or not os.path.isfile(out_json):
        print(f"[eval FAILED rc={rc}] {out_json}", flush=True)
        return None
    with open(out_json) as f:
        return json.load(f)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_lock(lock: str) -> bool:
    """Atomic per-run lock so two lanes can share one experiment. Stale locks (dead PID) are taken over."""
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            pid = int(open(lock).read().strip() or 0)
        except (OSError, ValueError):
            pid = 0
        if pid and _pid_alive(pid):
            return False
        os.remove(lock)
        return _acquire_lock(lock)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def train(ckpt: str, opponents: List[dict], frames: int, role: str = "pursuer", probs: Optional[Dict[str, float]] = None,
          init_policy: Optional[str] = None, num_envs: int = 4096, seed: int = 0,
          extra: Optional[List[str]] = None) -> Optional[str]:
    """Train ``role`` with PPO against a fixed opponent pool; returns the checkpoint path.

    Returns None if the run failed or another lane holds its lock (it is training it now).
    """
    if os.path.isfile(ckpt):
        return ckpt
    lock = ckpt + ".lock"
    if not _acquire_lock(lock):
        log_progress(f"[skip: run locked by another lane] {ckpt}")
        return None
    try:
        return _train(ckpt, opponents, frames, role, probs, init_policy, num_envs, seed, extra)
    finally:
        if os.path.isfile(lock):
            os.remove(lock)


def _train(ckpt, opponents, frames, role, probs, init_policy, num_envs, seed, extra):
    if os.path.isfile(ckpt):
        return ckpt
    base = os.path.splitext(ckpt)[0]
    spec = write_spec(base + "_opp.json", opponents, probs)
    args = common_overrides(role, num_envs, seed) + [
        "mode=train", f"total_frames={frames}", f"opponent_spec_path={spec}", f"stage_ckpt={ckpt}",
        f"stage_out={base}_out.json"]
    if init_policy:
        args.append(f"init_policy={init_policy}")
    args += list(extra or [])
    rc = run_stage(args, base + "_log.txt")
    if rc != 0 or not os.path.isfile(ckpt):
        print(f"[train FAILED rc={rc}] {ckpt}", flush=True)
        return None
    return ckpt


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def log_progress(msg: str) -> None:
    """Append a timestamped line to experiments/thesis/driver_log.txt (machine log, complements PROGRESS.md)."""
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "driver_log.txt"), "a") as f:
        f.write(f"{time.strftime('%F %T')} {msg}\n")
    print(msg, flush=True)
