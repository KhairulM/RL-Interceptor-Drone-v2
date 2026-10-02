# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Generational self-play orchestrator for pursuit-evasion.

Runs entirely outside Isaac Sim: it manages the JSON policy database and, each
stage, alternates which side trains, samples a frozen opponent distribution from
the database (temperature-scaled multi-metric softmax over win rate,
time-to-capture, distance, and motor effort), then spawns two isolated worker
processes (``run_stage.py``): one to train the new policy against the pool and
one to evaluate it against every opponent to fill in the matchup metrics.

Usage:
    python scripts/selfplay/train_selfplay.py [key=value ...]
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time

from omegaconf import OmegaConf

# Load the pure-python policy database by file path so the orchestrator stays
# lightweight (importing the omni_drones package would pull in the Isaac stack).
import importlib.util

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_DB_PATH = os.path.join(
    _REPO_ROOT, "omni_drones", "utils", "selfplay", "policy_db.py")
_spec = importlib.util.spec_from_file_location("selfplay_policy_db", _DB_PATH)
_policy_db = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_policy_db)
PURSUER, EVADER, PolicyDatabase = (
    _policy_db.PURSUER, _policy_db.EVADER, _policy_db.PolicyDatabase,
)

_RUN_STAGE = os.path.join(_THIS_DIR, "run_stage.py")


def _other(role):
    return EVADER if role == PURSUER else PURSUER


def _policy_stage(policy):
    """Return the persisted stage index for an RL policy, if available."""
    metadata = policy.get("metadata") or {}
    if metadata.get("stage") is not None:
        return int(metadata["stage"])
    checkpoint = policy.get("checkpoint_path") or ""
    match = re.search(r"(?:pursuer|evader)_stage(\d+)\.pt$", checkpoint)
    return int(match.group(1)) if match else None


def _resume_point(db):
    """Find the next stage, role, and optional incomplete stage to run."""
    rl_policies = []
    for role in (PURSUER, EVADER):
        for policy in db.list_policies(role):
            if policy.get("kind") == "rl" and _policy_stage(policy) is not None:
                rl_policies.append((int(_policy_stage(policy)), role, policy))
    if not rl_policies:
        return 0, PURSUER, None

    stage, role, policy = max(rl_policies, key=lambda item: item[0])
    opponent_role = _other(role)
    expected_opponents = db.policy_ids(opponent_role)
    complete = all(
        (db.get_matchup(policy["id"], opponent_id) is not None)
        if role == PURSUER
        else (db.get_matchup(opponent_id, policy["id"]) is not None)
        for opponent_id in expected_opponents
    )
    if complete:
        return stage + 1, opponent_role, None
    return stage, role, policy


def _abspath(path):
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def _seed_pool(db, role, heuristics):
    if db.policy_ids(role):
        return
    for name in heuristics:
        db.add_policy(role, None, kind="heuristic", metadata={"heuristic": name})
        print(f"[seed] {role} heuristic '{name}'")


def _pool_to_worker_spec(db, role, probs):
    """Serialise a pool (+ probs) into the worker's opponent spec format."""
    policies = []
    for p in db.list_policies(role):
        policies.append({
            "id": p["id"],
            "kind": p["kind"],
            "checkpoint_path": p.get("checkpoint_path"),
            "metadata": p.get("metadata", {}),
        })
    return {"policies": policies, "probs": probs}


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _run_worker(cfg, overrides):
    python = cfg.get("python", None) or sys.executable
    cmd = [python, _RUN_STAGE] + overrides
    # Retry on worker crashes (Isaac Sim occasionally SIGSEGVs at startup in its
    # telemetry/tasking plugins); each worker is a fresh process so a re-launch
    # almost always succeeds and a single crash won't abort the campaign.
    retries = int(cfg.get("worker_retries", 2))
    for attempt in range(retries + 1):
        print("[worker]", " ".join(cmd))
        # Run from the worker's own dir so run_stage.yaml's relative hydra
        # searchpath (file://../../cfg) resolves; all passed paths are absolute.
        result = subprocess.run(cmd, cwd=_THIS_DIR)
        if result.returncode == 0:
            return
        print(f"[worker] exited with code {result.returncode} "
              f"(attempt {attempt + 1}/{retries + 1})")
        if attempt < retries:
            time.sleep(10)  # let GPU/driver resources settle before relaunch
    raise RuntimeError(
        f"worker failed after {retries + 1} attempts (last code "
        f"{result.returncode}): {' '.join(cmd)}")


def main():
    base = OmegaConf.load(os.path.join(_THIS_DIR, "train_selfplay.yaml"))
    cfg = OmegaConf.merge(base, OmegaConf.from_cli())

    db_dir = _abspath(cfg.db_dir)
    ckpt_dir = _abspath(cfg.ckpt_dir)
    work_dir = _abspath(cfg.work_dir)

    # Fresh start: wipe the persisted population so heuristics are re-seeded and
    # a brand-new generational campaign begins (opt-in; default keeps/extends it).
    if bool(cfg.get("fresh_start", False)):
        for d in (db_dir, ckpt_dir, work_dir):
            if os.path.isdir(d):
                shutil.rmtree(d)
        print("[fresh_start] cleared DB, checkpoints, and stage artifacts")

    for d in (db_dir, ckpt_dir, work_dir):
        os.makedirs(d, exist_ok=True)

    db = PolicyDatabase(db_dir)
    _seed_pool(db, PURSUER, list(cfg.seed_heuristics.pursuer))
    _seed_pool(db, EVADER, list(cfg.seed_heuristics.evader))

    metric_weights = OmegaConf.to_container(cfg.metric_weights, resolve=True)
    temperature = float(cfg.temperature)

    if bool(cfg.get("fresh_start", False)):
        stage = 0
        role = str(cfg.start_role).lower()
        incomplete_policy = None
    else:
        stage, role, incomplete_policy = _resume_point(db)
        print(
            f"[resume] starting stage {stage} ({role}); "
            f"incomplete_policy={incomplete_policy['id'] if incomplete_policy else 'none'}"
        )

    for stage_offset in range(int(cfg.n_stages)):
        current_stage = stage + stage_offset
        opp_role = _other(role)
        print(f"\n===== stage {current_stage}: train {role} vs {opp_role} pool =====")

        # 1) Opponent sampling distribution from the database.
        probs = db.compute_sampling_probs(
            role, temperature=temperature, metric_weights=metric_weights)
        print(f"[sample] opponent probs: {probs}")

        train_spec = _pool_to_worker_spec(db, opp_role, probs)
        train_spec_path = os.path.join(
            work_dir, f"stage{current_stage}_{role}_train_opp.json")
        _write_json(train_spec_path, train_spec)

        # 2) Train a new policy for `role` against the frozen pool.
        stage_ckpt = os.path.join(ckpt_dir, f"{role}_stage{current_stage}.pt")
        train_out = os.path.join(
            work_dir, f"stage{current_stage}_{role}_train_out.json")
        overrides = [
            "mode=train",
            f"task={cfg.task}",
            f"algo={cfg.algo}",
            f"task.role={role}",
            f"task.env.num_envs={int(cfg.num_envs)}",
            f"total_frames={int(cfg.frames_per_stage)}",
            f"headless={bool(cfg.headless)}",
            f"seed={int(cfg.seed) + current_stage}",
            f"stage={current_stage}",
            f"wandb.mode={cfg.get('wandb_mode', 'online')}",
            f"opponent_spec_path={train_spec_path}",
            f"stage_ckpt={stage_ckpt}",
            f"stage_out={train_out}",
        ]
        if bool(cfg.warm_start):
            latest = db.latest_policy_id(role)
            latest_entry = db.get_policy(role, latest) if latest else None
            if latest_entry and latest_entry["kind"] == "rl":
                overrides.append(f"init_policy={latest_entry['checkpoint_path']}")
        _run_worker(cfg, overrides)

        with open(train_out, "r") as f:
            ckpt_path = json.load(f)["checkpoint_path"]
        if incomplete_policy is not None and current_stage == stage:
            new_id = incomplete_policy["id"]
            incomplete_policy = None
            print(f"[db] resumed existing {new_id} -> {ckpt_path}")
        else:
            new_id = db.add_policy(
                role, ckpt_path, kind="rl", metadata={"stage": current_stage})
        print(f"[db] added {new_id} -> {ckpt_path}")

        # 3) Evaluate the new policy against every opponent (uniform assignment).
        opp_ids = db.policy_ids(opp_role)
        uniform = {pid: 1.0 / len(opp_ids) for pid in opp_ids}
        eval_spec = _pool_to_worker_spec(db, opp_role, uniform)
        eval_spec_path = os.path.join(
            work_dir, f"stage{current_stage}_{role}_eval_opp.json")
        _write_json(eval_spec_path, eval_spec)

        eval_out = os.path.join(
            work_dir, f"stage{current_stage}_{role}_eval_out.json")
        eval_overrides = [
            "mode=eval",
            f"task={cfg.task}",
            f"algo={cfg.algo}",
            f"task.role={role}",
            f"task.env.num_envs={max(int(cfg.num_envs), len(opp_ids) * 8)}",
            f"headless={bool(cfg.headless)}",
            f"seed={int(cfg.seed) + current_stage}",
            f"opponent_spec_path={eval_spec_path}",
            f"eval_policy={ckpt_path}",
            f"stage_out={eval_out}",
        ]
        _run_worker(cfg, eval_overrides)

        with open(eval_out, "r") as f:
            matchups = json.load(f)["matchups"]
        for opp_id, metrics in matchups.items():
            if role == PURSUER:
                db.record_matchup(new_id, opp_id, metrics)
            else:
                db.record_matchup(opp_id, new_id, metrics)
        print(f"[db] recorded {len(matchups)} matchups for {new_id}")

        role = opp_role  # alternate sides

    print("\n[done] self-play finished. DB at:", db_dir)


if __name__ == "__main__":
    main()
