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
         per-opponent matchup metrics (win rate, time-to-capture, minimum distance,
         motor effort, pursuer crash rate) to ``stage_out``.
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
from omni_drones.utils.esn import save_readout_sidecar
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
    if cfg.algo.name == "ppo":
        cfg.task.opponent.algo_cfg = cfg.algo
    else:
        from omni_drones.learning.ppo.ppo import PPOConfig
        cfg.task.opponent.algo_cfg = OmegaConf.structured(PPOConfig)


def _select_esn_readout(cfg):
    """Train: warm-start the shared ESN readout from init_policy. Eval: use eval_policy's."""
    from omni_drones.utils.esn import readout_sidecar_path

    esn_cfg = cfg.task.get("esn", None)
    if esn_cfg is None or not esn_cfg.get("enabled", False) or esn_cfg.get("readout_path", None):
        return
    ckpt = cfg.get("eval_policy" if str(cfg.mode).lower() == "eval" else "init_policy", None)
    if ckpt and os.path.isfile(readout_sidecar_path(ckpt)):
        esn_cfg.readout_path = readout_sidecar_path(ckpt)


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

    policy = ALGOS[cfg.algo.name.lower()](
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


class _OpponentRefresher:
    """Re-samples the frozen-opponent assignment from the episodes played since the last refresh (thesis D1).

    Episodes in flight at a refresh finish against the newly assigned opponent and are credited to it.
    """

    def __init__(self, rcfg, base_env, max_time):
        from omni_drones.utils.selfplay import samplers

        self.samplers = samplers
        self.every = int(rcfg.every_frames)
        self.sampler = str(rcfg.sampler)
        self.temperature = float(rcfg.temperature)
        self.uniform_mix = float(rcfg.uniform_mix)
        self.weights = dict(rcfg.metric_weights) if rcfg.get("metric_weights") else None
        self.descriptors = None
        if rcfg.get("descriptors_path"):
            with open(rcfg.descriptors_path) as f:
                self.descriptors = json.load(f)
        self.log_path = rcfg.get("log_path", None)
        self.bank = base_env.opponent_bank
        self.role = base_env.role
        self.num_envs = base_env.num_envs
        self.max_time = max_time
        self.frames = 0
        self.since = 0
        self._chunks = []
        if getattr(base_env, "oracle_desc_table", None) is not None:
            raise ValueError("opponent refresh is incompatible with the per-env oracle descriptor")

    def add(self, data, frames):
        self.frames += frames
        self.since += frames
        done = data.get(("next", "done")).reshape(self.num_envs, -1)
        env_i, t_i = done.nonzero(as_tuple=True)
        if env_i.numel():
            stats = data.get(("next", "stats"))
            g = lambda k: stats.get(k).reshape(self.num_envs, -1)[env_i, t_i].float()
            crash_key = "crash" if self.role == "pursuer" else "opp_crash"
            self._chunks.append({
                "opp": self.bank.env_policy[env_i].clone(),
                "capture": g("capture") > 0.5, "pursuer_crash": g(crash_key) > 0.5, "timeout": g("timeout") > 0.5,
                "ttc": g("time_to_capture"), "min_distance": g("min_distance"), "effort": g("motor_effort")})
        if self.since >= self.every:
            self.refresh()

    def refresh(self):
        self.since = 0
        if not self._chunks:
            return
        cat = {k: torch.cat([c[k] for c in self._chunks]).cpu().numpy() for k in self._chunks[0]}
        self._chunks = []
        raw, counts = {}, {}
        for i, pid in enumerate(self.bank.ids):
            sel = cat["opp"] == i
            counts[pid] = int(sel.sum())
            raw[pid] = None if not sel.any() else self.samplers.opponent_metrics(
                cat["capture"][sel], cat["pursuer_crash"][sel], cat["timeout"][sel], cat["ttc"][sel],
                cat["min_distance"][sel], cat["effort"][sel], self.max_time)
        probs = self.samplers.compute_probs(
            self.sampler, raw, self.role, self.temperature, self.uniform_mix, self.weights, self.descriptors)
        self.bank.assign(probs, self.num_envs)
        logging.info(f"opponent refresh at {self.frames} frames: {probs}")
        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"frames": self.frames, "probs": probs, "episodes": counts,
                                    "metrics": raw}) + "\n")


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
    refresher = None
    rcfg = cfg.get("refresh", None)
    if rcfg is not None and int(rcfg.get("every_frames", 0)) > 0:
        refresher = _OpponentRefresher(
            rcfg, base_env, max_time=base_env.max_episode_length * float(cfg.sim.dt))
    pbar = tqdm(collector, total=total_frames // frames_per_batch)
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())
        if refresher is not None:
            refresher.add(data, frames_per_batch)
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
    save_readout_sidecar(base_env, ckpt_path)
    logging.info(f"saved checkpoint to {ckpt_path}")
    run.finish()

    _write_stage_out(cfg, {"checkpoint_path": ckpt_path})


@torch.no_grad()
def _run_eval(cfg, env, base_env, policy):
    heuristic = cfg.get("eval_heuristic", None)
    if heuristic:
        def policy(td):
            td.set(("agents", "action"),
                   base_env.agent_heuristic_raw_action(heuristic, td.get("is_init")))
            return td
    else:
        _load_state_dict(policy, cfg.eval_policy, base_env.device)
        policy.eval()

    base_env.eval()
    env.eval()
    env.set_seed(int(cfg.seed))

    if cfg.get("eval_cutoff_steps", None):
        _run_eval_multi(cfg, env, base_env, policy)
        return

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
    min_distance = stats["min_distance"]
    effort = stats["motor_effort"]

    dt = float(cfg.sim.dt)
    max_time = base_env.max_episode_length * dt

    # How each episode actually ended, from the evaluated agent's point of view.
    # Recorded per matchup so a zero capture rate can be attributed: an evader
    # that crashes or flies out of the arena ends the episode without ever being
    # intercepted, which is invisible in the aggregate sampling metrics alone.
    role = str(cfg.task.role)
    agent_is_pursuer = role == "pursuer"
    outcome_src = {
        "pursuer_crash": "crash" if agent_is_pursuer else "opp_crash",
        "evader_crash": "opp_crash" if agent_is_pursuer else "crash",
        "pursuer_oob": "out_of_bounds" if agent_is_pursuer else "opp_out_of_bounds",
        "evader_oob": "opp_out_of_bounds" if agent_is_pursuer else "out_of_bounds",
        "timeout": "timeout",
    }
    pursuer_crash = stats[outcome_src["pursuer_crash"]]
    timeout = stats["timeout"]
    ep_len = stats.get("episode_len")

    matchups = {}
    for opp_id, idxs in base_env.opponent_bank.assignment.items():
        idxs = idxs.cpu()
        n = int(idxs.numel())
        if n == 0:
            continue
        cap = capture[idxs]
        capture_rate = float(cap.mean().item())
        captured_mask = cap > 0.5
        # Pursuer win rate over decisive episodes only. Capture takes precedence
        # over a simultaneous pursuer crash; evader crash/OOB alone is neutral.
        pursuer_loss = ~captured_mask & (
            (pursuer_crash[idxs] > 0.5) | (timeout[idxs] > 0.5))
        wins = int(captured_mask.sum().item())
        decisive = wins + int(pursuer_loss.sum().item())
        win_rate = wins / decisive if decisive > 0 else None
        if bool(captured_mask.any()):
            avg_ttc = float(ttc[idxs][captured_mask].mean().item())
        else:
            avg_ttc = max_time
        pursuer_crash_rate = float((pursuer_crash[idxs] > 0.5).float().mean().item())
        record = {
            # `win_rate` is the pursuer's decisive-outcome win rate (the
            # sampling metric); None when every episode ended neutrally.
            "win_rate": win_rate,
            "decisive_episodes": decisive,
            "capture_rate": capture_rate,
            "avg_time_to_capture": avg_ttc,
            "avg_min_distance": float(min_distance[idxs].mean().item()),
            "avg_motor_effort": float(effort[idxs].mean().item()),
            "pursuer_crash_rate": pursuer_crash_rate,
            "episodes": n,
            "eval_role": role,
        }
        # Termination breakdown (fractions of episodes; not mutually exclusive
        # only in the degenerate case of two events on the same step).
        outcomes = {"capture": capture_rate}
        for name, key in outcome_src.items():
            if key in stats:
                outcomes[name] = float((stats[key][idxs] > 0.5).float().mean().item())
        record["outcomes"] = outcomes
        if ep_len is not None:
            record["avg_episode_len"] = float(ep_len[idxs].mean().item())
        matchups[opp_id] = record

    _write_stage_out(cfg, {"matchups": matchups})


@torch.no_grad()
def _run_eval_multi(cfg, env, base_env, policy):
    """Count every episode that starts before ``eval_cutoff_steps`` (thesis evals), plus behaviour features."""
    from omni_drones.utils.selfplay.eval_accumulator import (
        EpisodeAccumulator, behaviour_step_features, summarize_matchups)

    cutoff = int(cfg.eval_cutoff_steps)
    max_len = int(base_env.max_episode_length)
    step_dt = float(cfg.sim.dt) * int(cfg.sim.substeps)
    acc = EpisodeAccumulator(base_env.num_envs, cutoff, base_env.device)
    pursuer, evader = base_env.pursuer, base_env.evader

    td = env.reset()
    prev_vel = evader.vel_w[:, 0, :3].clone()
    traj_out = cfg.get("eval_traj_out", None)
    if traj_out:
        # float32: float16 positions (~2 mm steps) ruin second differences / dt^2 (A3a run 1).
        # Env -> opponent assignment is contiguous, so record evenly spaced envs to cover every opponent.
        rec = torch.linspace(0, base_env.num_envs - 1, int(cfg.get("eval_traj_envs", 512)),
                             device=base_env.device).long().unique()
        z_rec, d_rec = [base_env.opponent_z()[rec].float().cpu()], []
        pos_rec = [pursuer.pos[rec, 0].float().cpu()]
    steps = 0
    with set_exploration_type(ExplorationType.MODE):
        for t in range(cutoff + max_len):
            td = policy(td)
            td, td_next = env.step_and_maybe_reset(td)
            stats = {k: v for k, v in td.get(("next", "stats")).items()}
            e_vel = evader.vel_w[:, 0, :3]
            feats = behaviour_step_features(
                evader.pos[:, 0], e_vel, pursuer.pos[:, 0], prev_vel, step_dt)
            acc.step(t, td.get(("next", "done")), stats, feats)
            if traj_out:
                z_rec.append(base_env.opponent_z()[rec].float().cpu())
                pos_rec.append(pursuer.pos[rec, 0].float().cpu())
                d_rec.append(td.get(("next", "done")).reshape(-1)[rec].cpu())
            prev_vel = e_vel.clone()
            td = td_next
            steps = t + 1
            if acc.all_counted_done():
                break

    if traj_out:
        env_policy = base_env.opponent_bank.env_policy[rec].cpu()
        os.makedirs(os.path.dirname(os.path.abspath(traj_out)), exist_ok=True)
        # z[k] is the state after step k-1 (post-reset for envs done at step k-1); z[0] is the initial state.
        torch.save({"z": torch.stack(z_rec), "own_pos": torch.stack(pos_rec), "done": torch.stack(d_rec),
                    "envs": rec.cpu(),
                    "opponent": [base_env.opponent_bank.ids[i] for i in env_policy.tolist()],
                    "step_dt": step_dt}, traj_out)

    episodes = acc.episodes()
    assignment = base_env.opponent_bank.assignment
    matchups = summarize_matchups(episodes, assignment, str(cfg.task.role), step_dt, max_len)
    ep_out = cfg.get("eval_episodes_out", None)
    if ep_out:
        os.makedirs(os.path.dirname(os.path.abspath(ep_out)), exist_ok=True)
        torch.save({"episodes": episodes,
                    "assignment": {k: v.cpu() for k, v in assignment.items()}}, ep_out)
    _write_stage_out(cfg, {
        "matchups": matchups,
        "eval": {"cutoff_steps": cutoff, "steps_run": steps, "step_dt": step_dt,
                 "num_envs": int(base_env.num_envs),
                 "episodes": int(episodes["env"].numel()) if episodes else 0},
    })


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
    _select_esn_readout(cfg)

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
