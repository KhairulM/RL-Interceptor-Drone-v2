#!/usr/bin/env python
# MIT License
#
# Training script for competitive multi-agent intercept (pursuer vs evader).
# Both drones are RL-controlled with independent actors + critics.

import logging
import os
import time

import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from omni_drones import init_simulation_app
from torchrl.envs import TransformedEnv, Compose
from torchrl.envs.utils import ExplorationType
from omni_drones.utils.torchrl.collector import Collector
from omni_drones.learning import ALGOS
from setproctitle import setproctitle

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

try:
    from omni_drones.utils.wandb import init_wandb
except ModuleNotFoundError:
    init_wandb = None


class _NoOpRun:
    def __init__(self):
        self.name = "omnidrones-train-competitive"
        self.dir = os.getcwd()

    def log(self, *args, **kwargs):
        return None

    def log_artifact(self, *args, **kwargs):
        return None


def _make_episode_stats(env):
    """Collect stats keys from the env and track per-episode summaries."""
    stats_keys = [
        k for k in env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]

    _stats_buffer = {}
    for k in stats_keys:
        _stats_buffer[k] = []

    def add(tensordict):
        next_td = tensordict["next"]
        done = next_td.get("done")
        if done.any():
            done = done.squeeze(-1)
            mask = done.cpu().numpy()
            for k in stats_keys:
                val = next_td[k].cpu()
                _stats_buffer[k].append(val[mask])

    def pop():
        result = {}
        for k in stats_keys:
            if _stats_buffer[k]:
                stacked = torch.cat(_stats_buffer[k], dim=0)
                result[("stats", k[1] if isinstance(k, tuple) else k)] = (
                    stacked.float().mean().item())
                _stats_buffer[k].clear()
        return result

    return add, pop


@hydra.main(version_base=None, config_path="../cfg", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    simulation_app = init_simulation_app(cfg)
    use_wandb = (wandb is not None) and (init_wandb is not None)
    if use_wandb:
        run = init_wandb(cfg)
    else:
        logging.warning("W&B not available. Training without W&B.")
        run = _NoOpRun()

    setproctitle(run.name or "omnidrones-train-competitive")
    print(OmegaConf.to_yaml(cfg))

    # ---- Load environment ----------------------------------------------------
    from omni_drones.envs.isaac_env import IsaacEnv

    env_name = cfg.task.name
    base_env = IsaacEnv.REGISTRY[env_name](cfg, headless=cfg.headless)

    # No transforms needed — CTBR-to-motor conversion is handled inside the
    # env via PID controllers.  The policy outputs [N, 1, 4] CTBR per agent.
    env = base_env.train()
    env.set_seed(cfg.seed)

    # ---- Load algorithm ------------------------------------------------------
    algo_name = "competitive_mappo"
    policy = ALGOS[algo_name](
        cfg.algo,
        base_env.observation_spec,
        base_env.action_spec,
        base_env.reward_spec,
        device=base_env.device,
    )

    # ---- Training bookkeeping ------------------------------------------------
    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", 50)
    save_interval = cfg.get("save_interval", 100)

    stats_add, stats_pop = _make_episode_stats(base_env)

    collector = Collector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
        trust_policy=True,
    )

    # ---- Evaluation routine --------------------------------------------------
    @torch.no_grad()
    def evaluate():
        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(42)

        try:
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        except Exception as e:
            logging.warning(f"Evaluation rollout failed: {e}")
            trajs = None

        base_env.enable_render(not cfg.headless)
        env.reset()
        env.train()
        base_env.train()

        if trajs is None:
            return {}

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first(tensor):
            idx = first_done.reshape(
                first_done.shape + (1,) * (tensor.ndim - 2)
            )
            return torch.take_along_dim(tensor, idx, dim=1).reshape(-1)

        traj_stats = {
            k: take_first(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/" + (".".join(k) if isinstance(k, tuple) else k): (
                torch.mean(v.float()).item())
            for k, v in traj_stats.items()
        }
        return info

    # ---- Training loop -------------------------------------------------------
    pbar_total = total_frames // frames_per_batch if total_frames > 0 else max_iters
    if pbar_total <= 0:
        pbar_total = max_iters if max_iters > 0 else 1_000

    pbar = tqdm(collector, total=pbar_total)
    env.train()

    for i, data in enumerate(pbar):
        info = {
            "env_frames": collector._frames,
            "rollout_fps": collector._fps,
        }

        # Aggregate episode stats.
        stats_add(data.to_tensordict())
        completed_stats = stats_pop()
        if completed_stats:
            info.update({
                "train/" + (".".join(k) if isinstance(k, tuple) else k): v
                for k, v in completed_stats.items()
            })

        # PPO update.
        policy_info = policy.train_op(data.to_tensordict())
        info.update(policy_info)

        pbar.set_postfix({"frame": collector._frames})

        # Evaluate.
        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Evaluating at {collector._frames} frames.")
            info.update(evaluate())

        # Save checkpoint.
        if save_interval > 0 and i % save_interval == 0:
            try:
                ckpt_path = os.path.join(
                    run.dir, f"checkpoint_{collector._frames}.pt"
                )
                torch.save(policy.state_dict(), ckpt_path)
                logging.info(f"Saved checkpoint to {ckpt_path}")
            except AttributeError:
                logging.warning("Policy has no state_dict().")

        # Log.
        run.log(info, step=collector._frames)


if __name__ == "__main__":
    main()
