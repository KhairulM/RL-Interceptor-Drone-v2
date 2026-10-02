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

"""Single self-play stage worker (one Isaac Sim process).

Isolating each stage in its own process avoids Isaac's one-simulation-per-process
constraint and the RobotBase duplicate-name issues that arise from rebuilding
envs in a loop. The orchestrator (train_selfplay.py) spawns this for every
train/eval stage and communicates via JSON files.

Modes:
  train  Train ``task.role`` (PPO) against the injected frozen opponent pool for
         ``total_frames`` and save a checkpoint to ``stage_ckpt``.
  eval   Load ``eval_policy`` as the ``task.role`` agent, assign every opponent in
         the pool uniformly across envs, roll out one episode batch, and write
         per-opponent matchup metrics (win rate, time-to-capture, distance,
         motor effort) to ``stage_out``.
"""

import json
import logging
import os

import hydra
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from omni_drones import init_simulation_app
# Import at module top so the algo ConfigStore nodes (e.g. "ppo") are registered
# before Hydra composes the config, mirroring scripts/train.py.
from omni_drones.learning import ALGOS  # noqa: F401

try:
    import wandb
except ModuleNotFoundError:
    wandb = None
try:
    from omni_drones.utils.wandb import init_wandb
except ModuleNotFoundError:
    init_wandb = None


class _NoOpRun:
    def log(self, *args, **kwargs):
        return None

    def finish(self, *args, **kwargs):
        return None


def _load_opponent_spec(cfg):
    """Merge the opponent pool JSON into cfg.task.opponent (for RL loading too)."""
    path = cfg.get("opponent_spec_path", None)
    if not path:
        return
    with open(path, "r") as f:
        spec = json.load(f)
    cfg.task.opponent.policies = spec["policies"]
    cfg.task.opponent.probs = spec.get("probs", None)
    # Frozen RL opponents are PPO policies; hand them this run's algo config.
    cfg.task.opponent.algo_cfg = cfg.algo


def _build_transformed_env(cfg, base_env):
    from omni_drones.utils.torchrl.transforms import PIDRateController

    transforms = [InitTracker()]
    action_transform = cfg.task.get("action_transform", None)
    if action_transform is not None and action_transform.lower() == "pidrate":
        controller = getattr(base_env, "controller", None)
        if controller is None:
            raise RuntimeError("PursuitEvasion requires base_env.controller for pidrate")
        transforms.append(PIDRateController(controller.to(base_env.device)))
    elif action_transform is not None:
        raise NotImplementedError(f"unsupported action_transform: {action_transform}")
    return TransformedEnv(base_env, Compose(*transforms))


def _make_policy(cfg, env, base_env):
    from omni_drones.learning import ALGOS

    policy = ALGOS["ppo"](
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device,
    )
    return policy


def _load_state_dict(policy, path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        policy.load_state_dict(state)
    except RuntimeError:
        policy.load_state_dict(state, strict=False)


def _run_train(cfg, env, base_env, policy):
    from omni_drones.utils.torchrl import EpisodeStats
    from omni_drones.utils.torchrl.collector import Collector

    if cfg.get("init_policy", None):
        _load_state_dict(policy, cfg.init_policy, base_env.device)
        logging.info(f"warm-started from {cfg.init_policy}")

    use_wandb = (
        wandb is not None and init_wandb is not None
        and str(cfg.wandb.mode).lower() != "disabled"
    )
    run = init_wandb(cfg) if use_wandb else _NoOpRun()

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = int(cfg.total_frames) // frames_per_batch * frames_per_batch

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = Collector(
        env, policy=policy,
        frames_per_batch=frames_per_batch, total_frames=total_frames,
        device=cfg.sim.device, return_same_td=True, trust_policy=True,
    )

    env.train()
    base_env.train()
    pbar = tqdm(collector, total=total_frames // frames_per_batch)
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())
        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k):
                    torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)
        info.update(policy.train_op(data.to_tensordict()))
        run.log(info)
        # Console: show only episode stats (the "train/" keys); the learner
        # metrics (policy_loss, value_loss, grad norms, ...) still go to run.log.
        stats_only = {k: v for k, v in info.items()
                      if isinstance(v, float) and k.startswith("train/")}
        if stats_only:
            print(OmegaConf.to_yaml(stats_only), end="")
        pbar.set_postfix({"rollout_fps": collector._fps,
                          "frames": collector._frames})

    ckpt_path = os.path.abspath(cfg.stage_ckpt)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(policy.state_dict(), ckpt_path)
    logging.info(f"saved checkpoint to {ckpt_path}")
    run.finish()

    _write_stage_out(cfg, {"checkpoint_path": ckpt_path})


@torch.no_grad()
def _run_eval(cfg, env, base_env, policy):
    _load_state_dict(policy, cfg.eval_policy, base_env.device)
    policy.eval()

    base_env.eval()
    env.eval()
    env.set_seed(int(cfg.seed))

    with set_exploration_type(ExplorationType.MODE):
        trajs = env.rollout(
            max_steps=base_env.max_episode_length,
            policy=policy,
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False,
        )

    done = trajs.get(("next", "done"))
    first_done = torch.argmax(done.long(), dim=1).cpu()

    def take_first(tensor):
        idx = first_done.reshape(first_done.shape + (1,) * (tensor.ndim - 2))
        return torch.take_along_dim(tensor, idx, dim=1).reshape(-1)

    stats = {k: take_first(v).float() for k, v in trajs[("next", "stats")].cpu().items()}
    capture = stats["capture"]
    ttc = stats["time_to_capture"]
    distance = stats["distance"]
    effort = stats["motor_effort"]

    dt = float(cfg.sim.dt)
    max_time = base_env.max_episode_length * dt

    # How each episode actually ended, from the evaluated agent's point of view.
    # Recorded per matchup so a zero capture rate can be attributed: an evader
    # that crashes or flies out of the arena ends the episode without ever being
    # intercepted, which is invisible in the four sampling metrics alone.
    role = str(cfg.task.role)
    agent_is_pursuer = role == "pursuer"
    outcome_src = {
        "pursuer_crash": "crash" if agent_is_pursuer else "opp_crash",
        "evader_crash": "opp_crash" if agent_is_pursuer else "crash",
        "pursuer_oob": "out_of_bounds" if agent_is_pursuer else "opp_out_of_bounds",
        "evader_oob": "opp_out_of_bounds" if agent_is_pursuer else "out_of_bounds",
        "timeout": "timeout",
    }
    ep_len = stats.get("episode_len")
    terminal_win = stats.get("win_rate")
    forced_error = stats.get("forced_error")

    matchups = {}
    for opp_id, idxs in base_env.opponent_bank.assignment.items():
        idxs = idxs.cpu()
        n = int(idxs.numel())
        if n == 0:
            continue
        cap = capture[idxs]
        win_rate = float(cap.mean().item())
        captured_mask = cap > 0.5
        if bool(captured_mask.any()):
            avg_ttc = float(ttc[idxs][captured_mask].mean().item())
        else:
            avg_ttc = max_time
        # Fraction of episodes where the EVALUATED AGENT (this role) crashed to
        # the ground or left the arena. stats["crash"]/["out_of_bounds"] are
        # always the agent's (opponent errors live under opp_*), so this is a
        # role-agnostic "how hard did this opponent push the agent" signal that
        # feeds opponent sampling (harder opponent -> higher agent error rate).
        agent_error_rate = 0.0
        if "crash" in stats and "out_of_bounds" in stats:
            err = (stats["crash"][idxs] > 0.5) | (stats["out_of_bounds"][idxs] > 0.5)
            agent_error_rate = float(err.float().mean().item())
        record = {
            # `win_rate` is the pursuer capture rate: the sampling metric.
            "win_rate": win_rate,
            "avg_time_to_capture": avg_ttc,
            "avg_distance": float(distance[idxs].mean().item()),
            "avg_motor_effort": float(effort[idxs].mean().item()),
            "agent_error_rate": agent_error_rate,
            "episodes": n,
            "eval_role": role,
        }
        # Termination breakdown (fractions of episodes; not mutually exclusive
        # only in the degenerate case of two events on the same step).
        outcomes = {"capture": win_rate}
        for name, key in outcome_src.items():
            if key in stats:
                outcomes[name] = float((stats[key][idxs] > 0.5).float().mean().item())
        record["outcomes"] = outcomes
        # Terminal reward actually earned by the evaluated agent, so a gap
        # between "wins" and "captures" shows up directly in the table.
        if terminal_win is not None:
            record["agent_terminal_win_rate"] = float(terminal_win[idxs].mean().item())
        if forced_error is not None:
            # Episodes the agent only got partial credit for: it forced the
            # opponent into the ground / out of the arena without finishing it.
            record["agent_forced_error_rate"] = float(
                forced_error[idxs].mean().item())
            outcomes["forced_error"] = record["agent_forced_error_rate"]
        if ep_len is not None:
            record["avg_episode_len"] = float(ep_len[idxs].mean().item())
        matchups[opp_id] = record

    _write_stage_out(cfg, {"matchups": matchups})


def _write_stage_out(cfg, payload):
    out = cfg.get("stage_out", None)
    if not out:
        return
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    logging.info(f"wrote stage output to {out}")


@hydra.main(version_base=None, config_path=".", config_name="run_stage")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    _load_opponent_spec(cfg)

    init_simulation_app(cfg)

    from omni_drones.envs.isaac_env import IsaacEnv

    base_env = IsaacEnv.REGISTRY[cfg.task.name](cfg, headless=cfg.headless)
    env = _build_transformed_env(cfg, base_env)
    env.set_seed(int(cfg.seed))
    policy = _make_policy(cfg, env, base_env)

    mode = str(cfg.mode).lower()
    if mode == "train":
        _run_train(cfg, env, base_env, policy)
    elif mode == "eval":
        _run_eval(cfg, env, base_env, policy)
    else:
        raise ValueError(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
