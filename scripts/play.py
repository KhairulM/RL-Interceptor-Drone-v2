import datetime
import logging
import os
import re

import hydra
import torch

from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import Collector
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    RateController,
    PIDRateController,
)
from omni_drones.utils.torchrl import EpisodeStats
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


FILE_PATH = os.path.dirname(__file__)


class _NoOpRun:
    def __init__(self):
        self.name = "omnidrones-play"

    def log(self, *args, **kwargs):
        return None


def _resolve_existing_path(path: str, expect: str):
    """Resolve path robustly across cwd/script-dir/repo-root.

    Args:
        path: user-provided path (absolute or relative)
        expect: "file" or "dir"
    """
    if not path:
        return None

    expanded = os.path.expanduser(str(path))
    script_dir = FILE_PATH
    repo_root = os.path.dirname(FILE_PATH)
    cwd = os.getcwd()

    candidates = []
    if os.path.isabs(expanded):
        candidates.append(expanded)
    else:
        candidates.extend([
            os.path.join(cwd, expanded),
            os.path.join(script_dir, expanded),
            os.path.join(repo_root, expanded),
        ])

        prefix = f"scripts{os.sep}"
        if expanded.startswith(prefix):
            trimmed = expanded[len(prefix):]
            candidates.extend([
                os.path.join(cwd, trimmed),
                os.path.join(script_dir, trimmed),
                os.path.join(repo_root, trimmed),
            ])

    # preserve order while deduplicating
    seen = set()
    unique_candidates = []
    for c in candidates:
        n = os.path.normpath(c)
        if n not in seen:
            seen.add(n)
            unique_candidates.append(n)

    for candidate in unique_candidates:
        if expect == "dir" and os.path.isdir(candidate):
            return candidate
        if expect == "file" and os.path.isfile(candidate):
            return candidate

    return None


def _infer_wandb_run_id_from_run_dir(run_dir: str):
    """Infer W&B run id from a run directory named run-YYYYMMDD_HHMMSS-<id>."""
    if not run_dir:
        return None

    normalized = os.path.normpath(os.path.expanduser(str(run_dir)))
    run_folder = os.path.basename(normalized)
    if not run_folder.startswith("run-"):
        return None

    match = re.match(r"^run-\d{8}_\d{6}-([A-Za-z0-9]+)$", run_folder)
    if match:
        return match.group(1)

    maybe_id = run_folder.split("-")[-1]
    if maybe_id and maybe_id != "run":
        return maybe_id
    return None


def _init_wandb_for_play(cfg):
    """Initialize W&B for a play/eval rollout.

    If `wandb.run_id` is set, resume that existing run and append a `play/`
    panel to it WITHOUT touching its name, group, config, or other metadata.
    Otherwise, start a fresh dedicated play run.
    """
    assert wandb is not None
    wandb_cfg = cfg.wandb

    run_id = wandb_cfg.get("run_id", None)
    if run_id is not None:
        # Resume the original run; do not pass name/group/config so existing
        # values are left untouched.
        return wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            mode=wandb_cfg.mode,
            id=run_id,
            resume="must",
        )

    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    run_name = f"{wandb_cfg.run_name}/{time_str}"
    return wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        mode=wandb_cfg.mode,
        group=wandb_cfg.group,
        name=run_name,
        job_type="play",
        tags=wandb_cfg.tags,
    )


@hydra.main(config_path=FILE_PATH, config_name="play", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)

    wandb_run_dir = cfg.get("wandb_run_dir", None)
    if wandb_run_dir:
        resolved_wandb_run_dir = _resolve_existing_path(wandb_run_dir, expect="dir")
        if resolved_wandb_run_dir is None:
            raise FileNotFoundError(
                f"wandb_run_dir not found: {wandb_run_dir}. "
                f"Current working directory is {os.getcwd()}."
            )

        wandb_run_dir = resolved_wandb_run_dir
        if cfg.get("checkpoint", None) is None:
            cfg.checkpoint = os.path.join(
                wandb_run_dir, "files", "checkpoint_final.pt")
            logging.info(
                "Using checkpoint inferred from wandb_run_dir: %s",
                cfg.checkpoint,
            )

        # Resume the original run so the play metrics show up as a panel there.
        inferred_run_id = _infer_wandb_run_id_from_run_dir(wandb_run_dir)
        if inferred_run_id and cfg.wandb.get("run_id", None) is None:
            cfg.wandb.run_id = inferred_run_id
            logging.info(
                "Resuming W&B run id '%s' to append play metrics.",
                inferred_run_id,
            )

    use_wandb = wandb is not None
    if use_wandb:
        run = _init_wandb_for_play(cfg)
        # Give play metrics their own x-axis pinned at step 0 so each play run
        # overwrites the same point (a single value) instead of appending to the
        # global training step. Old play history (logged without `play/step`)
        # will not appear on this axis.
        if hasattr(run, "define_metric"):
            assert wandb is not None
            wandb.define_metric("play/step", hidden=True)
            wandb.define_metric("play/*", step_metric="play/step")
    else:
        logging.warning(
            "W&B is not available. Proceeding without W&B logging.")
        run = _NoOpRun()

    setproctitle(run.name or f"{cfg.task.name}-play")
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms: list = [InitTracker()]

    # a CompositeSpec is by deafault processed by a entity-based encoder
    # ravel it to use a MLP encoder instead
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(
            base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(
            base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)

    # if cfg.task.get("history", False):
    #     # transforms.append(History([("info", "drone_state"), ("info", "prev_action")]))
    #     transforms.append(History([("agents", "observation")]))

    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        action_transform = action_transform.lower()
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("rate"):
            # some environments (e.g. Intercept) expose the pursuer controller
            # as `pursuer_controller` rather than `controller`.
            controller = getattr(base_env, "controller", None)
            if controller is None:
                controller = getattr(base_env, "pursuer_controller", None)

            if controller is None:
                raise RuntimeError(
                    "Rate action transform requires a controller in the task config."
                )

            transform = RateController(controller.to(base_env.device))
            transforms.append(transform)
        elif action_transform == "pidrate":
            controller = getattr(base_env, "controller", None)
            if controller is None:
                controller = getattr(base_env, "pursuer_controller", None)

            if controller is None:
                raise RuntimeError(
                    "PIDRate action transform requires a controller in the task config."
                )
            transform = PIDRateController(controller)
            # transforms.append(TanhTransform)
            transforms.append(transform)
        else:
            raise NotImplementedError(
                f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    algo_name = cfg.algo.name.lower()
    try:
        if algo_name in ["sac", "td3"]:
            policy = ALGOS[algo_name](
                cfg.algo,
                env.agent_spec["drone"],
                device=base_env.device
            )
        else:
            policy = ALGOS[algo_name](
                cfg.algo,
                env.observation_spec,
                env.action_spec,
                env.reward_spec,
                device=base_env.device
            )
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")

    # Optionally load a trained checkpoint produced by train.py.
    ckpt_path = cfg.get("checkpoint", None)
    if ckpt_path:
        resolved_ckpt_path = _resolve_existing_path(ckpt_path, expect="file")
        if resolved_ckpt_path is None:
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                "If using wandb_run_dir, ensure files/checkpoint_final.pt exists under that run folder."
            )
        ckpt_path = resolved_ckpt_path
        try:
            state = torch.load(ckpt_path, map_location=base_env.device)
            if isinstance(state, dict):
                if "state_dict" in state:
                    state = state["state_dict"]
                elif "model_state_dict" in state:
                    state = state["model_state_dict"]
                elif "policy_state_dict" in state:
                    state = state["policy_state_dict"]
            policy.load_state_dict(state)
            logging.info(f"Loaded checkpoint from {ckpt_path}")
        except RuntimeError as err:
            logging.warning(
                "Strict checkpoint load failed (%s). Retrying with strict=False.",
                str(err),
            )
            state = torch.load(ckpt_path, map_location=base_env.device)
            if isinstance(state, dict):
                if "state_dict" in state:
                    state = state["state_dict"]
                elif "model_state_dict" in state:
                    state = state["model_state_dict"]
                elif "policy_state_dict" in state:
                    state = state["policy_state_dict"]
            missing, unexpected = policy.load_state_dict(state, strict=False)
            logging.warning(
                "Loaded with strict=False. Missing keys: %d, Unexpected keys: %d",
                len(missing),
                len(unexpected),
            )
        except AttributeError:
            logging.warning(
                f"Policy {policy} does not implement `.load_state_dict()`; "
                f"ignoring checkpoint {ckpt_path}"
            )
    else:
        logging.info(
            "No checkpoint specified; running with randomly initialized policy. "
            "Pass `wandb_run_dir=/path/to/wandb/run-...` or "
            "`checkpoint=/path/to/checkpoint.pt` to load trained weights."
        )

    frames_per_batch = env.num_envs * int(cfg.get("frames_per_batch", 32))
    total_frames = int(cfg.get("total_frames", -1))
    max_iters = int(cfg.get("max_iters", -1))

    if total_frames <= 0 and max_iters <= 0:
        logging.warning(
            "Both total_frames and max_iters are unset/non-positive; defaulting max_iters=1 for a finite play run."
        )
        max_iters = 1

    collector_total_frames = (
        total_frames
        if total_frames > 0
        else frames_per_batch * max(max_iters, 1)
    )

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = Collector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=collector_total_frames,
        device=cfg.sim.device,
        return_same_td=True,
        trust_policy=True,
    )

    pbar_total = None
    if total_frames > 0:
        pbar_total = total_frames // frames_per_batch
    elif max_iters > 0:
        pbar_total = max_iters

    pbar = tqdm(collector, total=pbar_total)
    base_env.eval()
    env.eval()
    if hasattr(policy, "eval"):
        policy.eval()

    running_sum = {}
    running_count = {}

    with set_exploration_type(ExplorationType.MODE):
        for i, data in enumerate(pbar):
            info = {}
            episode_stats.add(data.to_tensordict())

            if len(episode_stats) >= base_env.num_envs:
                stats = {
                    "play/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                    for k, v in episode_stats.pop().items(True, True)
                }
                info.update(stats)

            # Keep performance as a running average instead of per-step logs.
            info["play/rollout_fps"] = float(collector._fps)

            for k, v in info.items():
                if not isinstance(v, (int, float)):
                    continue
                running_sum[k] = running_sum.get(k, 0.0) + float(v)
                running_count[k] = running_count.get(k, 0) + 1

            pbar.set_postfix({
                "rollout_fps": collector._fps,
                "frames": collector._frames,
            })

            if max_iters > 0 and i >= max_iters - 1:
                break

    avg_info = {
        f"play/avg.{k[5:]}": running_sum[k] / max(running_count[k], 1)
        for k in running_sum.keys()
    }
    play_info = {
        k: v for k, v in avg_info.items()
        if isinstance(k, str) and k.startswith("play/")
    }
    if play_info:
        # Pin every play run to play/step=0 so the panels show a single value.
        run.log({"play/step": 0, **play_info})
        print(OmegaConf.to_yaml(
            {k: v for k, v in play_info.items() if isinstance(v, float)}))

    if use_wandb:
        assert wandb is not None
        wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()
