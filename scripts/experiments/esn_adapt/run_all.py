"""Resumable driver for the ESN adaptation experiment (P0 results summarised in docs/research_plan.md).

    python scripts/experiments/esn_adapt/run_all.py --phase smoke
    python scripts/experiments/esn_adapt/run_all.py --phase p0 --frames 100000000 --seeds 3

Runs sequentially (one Isaac process at a time). Finished runs have a DONE marker and are skipped.
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
ROOT = os.path.join(REPO, "experiments/esn_adapt")
RUN_STAGE = os.path.join(REPO, "scripts/selfplay/run_stage.py")
EVADER_CKPT = os.path.join(REPO, "scripts/selfplay/outputs/selfplay_v6_fix_reward_and_bounds/checkpoints")

# name -> hydra overrides (pursuer side)
METHODS = {
    "M0_ppo": ["algo=ppo", "task.esn.enabled=false"],
    "M1_esn_shared": ["algo=ppo", "task.esn.enabled=true", "task.esn.readout=shared"],
    "M2_esn_per_env": ["algo=ppo", "task.esn.enabled=true", "task.esn.readout=per_env"],
    "M3_esn_shared_adapt": ["algo=ppo", "task.esn.enabled=true", "task.esn.readout=shared_adapt"],
    "M4_gru": ["algo=ppo_gru", "task.esn.enabled=false"],
}


def heur(name):
    return {"id": f"evader:{name}", "kind": "heuristic", "checkpoint_path": None, "metadata": {"heuristic": name}}


def rl(stage):
    return {"id": f"evader:stage{stage}", "kind": "rl", "checkpoint_path": f"{EVADER_CKPT}/evader_stage{stage}.pt", "metadata": {}}


# Train pool and held-out test sets (T4 dynamics shift needs env support; not yet implemented).
POOLS = {
    "train": [heur(h) for h in ("hover", "circular", "pursue", "flee")] + [rl(3), rl(7)],
    "T0": [heur(h) for h in ("hover", "circular", "pursue", "flee")] + [rl(3), rl(7)],
    "T1": [heur(h) for h in ("weave", "zigzag", "fast_flee")],
    "T2": [rl(9), rl(11)],
    "T3": [heur(h) for h in ("flee>circular@2.0", "circular>hover@2.0", "hover>flee@2.0", "flee>weave@2.0")],
}


def write_pools():
    os.makedirs(f"{ROOT}/pools", exist_ok=True)
    for name, pols in POOLS.items():
        p = 1.0 / len(pols)
        with open(f"{ROOT}/pools/{name}.json", "w") as f:
            json.dump({"policies": pols, "probs": {x["id"]: p for x in pols}}, f, indent=2)


def sh(cmd, log):
    with open(log, "a") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        return subprocess.run(cmd, cwd=os.path.join(REPO, "scripts/selfplay"), stdout=f, stderr=subprocess.STDOUT).returncode


def run_one(method, seed, frames, num_envs, tag, retries=2):
    d = f"{ROOT}/{tag}/{method}/seed{seed}"
    os.makedirs(d, exist_ok=True)
    if os.path.exists(f"{d}/DONE"):
        return True
    log = f"{d}/log.txt"
    common = ["task=PursuitEvasion", "task.role=pursuer", f"task.env.num_envs={num_envs}", "headless=true",
              f"seed={seed}", "wandb.mode=disabled", "task.observation.include_noise=false"] + METHODS[method]
    ckpt = f"{d}/policy.pt"
    state = json.load(open(f"{d}/state.json")) if os.path.exists(f"{d}/state.json") else {}
    t0 = time.time()
    if not os.path.exists(ckpt):
        cmd = [sys.executable, RUN_STAGE, "mode=train", f"total_frames={frames}", f"stage_ckpt={ckpt}",
               f"opponent_spec_path={ROOT}/pools/train.json", f"stage_out={d}/train_out.json"] + common
        for _ in range(retries + 1):
            if sh(cmd, log) == 0 and os.path.exists(ckpt):
                break
        else:
            json.dump({"status": "failed_train"}, open(f"{d}/state.json", "w"))
            return False
        state["train_seconds"] = time.time() - t0
    for T in ("T0", "T1", "T2", "T3"):
        out = f"{d}/eval_{T}.json"
        if os.path.exists(out):
            continue
        # Recurrent rollouts store hidden states for every step and OOM at 4096 envs.
        eval_envs = [f"task.env.num_envs={min(num_envs, 1024)}"] if "gru" in method else []
        cmd = [sys.executable, RUN_STAGE, "mode=eval", f"eval_policy={ckpt}", f"opponent_spec_path={ROOT}/pools/{T}.json",
               f"stage_out={out}"] + common + eval_envs
        for _ in range(retries + 1):
            if sh(cmd, log) == 0 and os.path.exists(out):
                break
        else:
            state["status"] = f"failed_eval_{T}"
            json.dump(state, open(f"{d}/state.json", "w"))
            return False
    state["status"] = "done"
    json.dump(state, open(f"{d}/state.json", "w"))
    open(f"{d}/DONE", "w").close()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "p0"], required=True)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    a = ap.parse_args()
    write_pools()
    smoke = a.phase == "smoke"
    frames = a.frames or (2_000_000 if smoke else 100_000_000)
    tag, seeds = ("smoke", 1) if smoke else ("p0", a.seeds)
    table = {}
    for m in a.methods:
        for s in range(seeds):
            table[f"{m}/seed{s}"] = run_one(m, s, frames, a.num_envs, tag)
            print(m, s, "ok" if table[f"{m}/seed{s}"] else "FAILED", flush=True)
    json.dump(table, open(f"{ROOT}/{tag}_status.json", "w"), indent=2)


if __name__ == "__main__":
    main()
