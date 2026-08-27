# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# See the LICENSE file at the repository root for full terms.

"""Export a trained Intercept policy to a deployment artifact.

This script runs inside the **Isaac Sim** training environment (Python 3.11
``.venv``). It builds the task/policy exactly the way ``scripts/play.py`` does
-- so it is agnostic to which RL algorithm produced the checkpoint (ppo, mappo,
happo, sac, td3, ...) -- loads the checkpoint, extracts the *deterministic*
actor, and serialises it to a self-contained TorchScript module plus a
``metadata.json`` describing the observation layout and CTBR decoding
parameters.

The resulting artifact depends only on ``torch`` and can therefore be loaded by
the Crazyswarm2 controller ([intercept_controller.py](intercept_controller.py))
running in the separate Python 3.10 ``.venv-crazyswarm`` environment, without
pulling in Isaac Sim or torchrl.

Usage (from the repository root, with the Isaac ``.venv`` active)::

    python deploy/export_policy.py \\
        task=Intercept algo=ppo headless=true \\
        checkpoint=scripts/outputs/<date>/<time>/checkpoint_final.pt \\
        export_dir=deploy/artifacts/intercept_ppo

``export_dir`` defaults to ``deploy/artifacts/<task>_<algo>`` when omitted.
"""

import logging
import os
import sys

import hydra
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

# Make the sibling ``intercept_common`` importable regardless of CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import intercept_common as ic  # noqa: E402

from omni_drones import init_simulation_app  # noqa: E402

# Importing the learning package registers every algorithm's structured config
# (e.g. ``cs.store("ppo", ...)``) with Hydra's ConfigStore. This MUST happen
# before ``@hydra.main`` composes the config, otherwise CLI overrides such as
# ``algo=ppo`` cannot be resolved. It does not require Isaac Sim.
from omni_drones.learning import ALGOS  # noqa: E402

_SCRIPTS_DIR = os.path.split(_THIS_DIR)[0]  # parent of deploy/
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)


# ---------------------------------------------------------------------------
# Deterministic-actor extraction
# ---------------------------------------------------------------------------
class _GaussianDetActor(torch.nn.Module):
    """Wraps a Gaussian actor feed-forward net to output its mean (mode)."""

    def __init__(self, feed_forward: torch.nn.Module):
        super().__init__()
        self.feed_forward = feed_forward

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.feed_forward(obs)
        # The upstream ``Actor`` returns ``(loc, scale)``; the deterministic
        # action (mode of an ``IndependentNormal``) is ``loc``.
        return out[0] if isinstance(out, (tuple, list)) else out


class _ForwardDetActor(torch.nn.Module):
    """Wraps a deterministic (SAC/TD3-style) actor net."""

    def __init__(self, net: torch.nn.Module, pass_deterministic: bool):
        super().__init__()
        self.net = net
        self.pass_deterministic = pass_deterministic

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.net(obs, deterministic=True) if self.pass_deterministic else self.net(obs)
        # SAC returns ``(action, log_prob)``; TD3 returns the action tensor.
        return out[0] if isinstance(out, (tuple, list)) else out


def _find_gaussian_feed_forward(actor: torch.nn.Module) -> torch.nn.Module:
    """Locate the plain-tensor sub-network that outputs ``loc`` in a
    ``ProbabilisticActor`` (ppo/mappo/happo family)."""
    from tensordict.nn import TensorDictModuleBase

    # Reject vmapped ensemble actors: their parameters live in a functional
    # TensorDict, so a naive extraction would capture the wrong weights.
    for module in actor.modules():
        if type(module).__name__ == "EnsembleModule":
            raise NotImplementedError(
                "The checkpoint uses a per-agent EnsembleModule actor "
                "(share_actor=false). Re-train/export with share_actor=true, "
                "or extend export_policy.py to de-functionalise the ensemble."
            )

    candidates = []
    for module in actor.modules():
        out_keys = getattr(module, "out_keys", None)
        if not out_keys:
            continue
        leaf_names = [k if isinstance(k, str) else k[-1] for k in out_keys]
        if "loc" not in leaf_names:
            continue
        inner = getattr(module, "module", None)
        if isinstance(inner, torch.nn.Module) and not isinstance(inner, TensorDictModuleBase):
            candidates.append(inner)

    if not candidates:
        raise RuntimeError(
            "Could not locate the actor sub-network producing 'loc'. "
            "The policy structure may have changed; update export_policy.py."
        )
    # The innermost matching module is the feed-forward net we want.
    return candidates[-1]


def extract_deterministic_actor(
    policy, algo_name: str
) -> torch.nn.Module:
    """Return an ``nn.Module`` mapping ``[B, obs_dim] -> [B, action_dim]``."""
    algo = algo_name.lower()
    gaussian = {"ppo", "ppo_rnn", "ppo_adapt", "mappo", "mappo_old", "happo", "ppo_priv"}
    if algo in gaussian:
        feed_forward = _find_gaussian_feed_forward(policy.actor)
        module = _GaussianDetActor(feed_forward)
    elif algo == "sac":
        module = _ForwardDetActor(policy.actor.module, pass_deterministic=True)
    elif algo in {"td3", "matd3"}:
        module = _ForwardDetActor(policy.actor.module, pass_deterministic=False)
    else:
        raise NotImplementedError(
            f"Deterministic-actor extraction is not implemented for algo "
            f"'{algo_name}'. Supported: {sorted(gaussian | {'sac', 'td3', 'matd3'})}."
        )
    return module.eval()


# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors scripts/play.py)
# ---------------------------------------------------------------------------
def _load_state(path: str, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "policy_state_dict"):
            if key in state:
                return state[key]
    return state


def load_checkpoint_into_policy(policy, ckpt_path: str, device) -> None:
    """Load ``ckpt_path`` into ``policy`` (strict, falling back to non-strict)."""
    state = _load_state(ckpt_path, device)
    try:
        policy.load_state_dict(state)
        logging.info("Loaded checkpoint (strict) from %s", ckpt_path)
    except RuntimeError as err:
        logging.warning("Strict load failed (%s); retrying with strict=False.", err)
        missing, unexpected = policy.load_state_dict(state, strict=False)
        logging.warning(
            "Loaded with strict=False. Missing: %d, Unexpected: %d",
            len(missing), len(unexpected),
        )


# ---------------------------------------------------------------------------
# Metadata assembly
# ---------------------------------------------------------------------------
def build_metadata(cfg, base_env, obs_dim: int, action_dim: int) -> ic.PolicyMetadata:
    """Read the observation/CTBR parameters off the live env and config."""
    pursuer_cfg = cfg.task.pursuer
    evader_cfg = cfg.task.evader
    obs_cfg = cfg.task.observation

    obs_cfg = ic.ObsConfig(
        use_ab_world_frame=bool(obs_cfg.get("use_world_frame_pos", False)),
        use_relative_velocity=bool(obs_cfg.get("use_evader_rel_lin_vel", False)),
        use_previous_action=bool(obs_cfg.get("use_previous_action", True)),
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    expected = obs_cfg.expected_obs_dim()
    if expected != obs_dim:
        raise RuntimeError(
            f"Observation layout mismatch: flags imply obs_dim={expected} but "
            f"the env reports {obs_dim}. Aborting to avoid a silently wrong "
            f"deployment observation."
        )

    # Pull CTBR decode parameters straight off the controller the task built.
    controller = getattr(base_env, "pursuer_controller", None) or getattr(base_env, "controller")
    ctbr_cfg = ic.CTBRConfig(
        target_clip=float(controller.target_clip),
        min_thrust_ratio=float(controller.min_thrust_ratio),
        max_thrust_ratio=float(controller.max_thrust_ratio),
        lpf_coef=float(controller.LPF_coef),
        dt=float(cfg.sim.dt * cfg.sim.substeps),
    )

    return ic.PolicyMetadata(
        artifact_version=ic.ARTIFACT_VERSION,
        algo=str(cfg.algo.name).lower(),
        obs=obs_cfg,
        ctbr=ctbr_cfg,
        sim_dt=float(cfg.sim.dt),
        notes={
            "task": str(cfg.task.name),
            "checkpoint": str(cfg.get("checkpoint", "")),
            "pursuer_model": str(pursuer_cfg.get("model", "")),
            "action_transform": str(cfg.task.get("action_transform", "")),
        },
    )


@hydra.main(config_path=_SCRIPTS_DIR, config_name="play", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    ckpt_path = cfg.get("checkpoint", None)
    if not ckpt_path:
        raise ValueError(
            "No checkpoint provided. Pass checkpoint=/path/to/checkpoint_final.pt"
        )
    ckpt_path = os.path.abspath(os.path.expanduser(str(ckpt_path)))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    export_dir = cfg.get("export_dir", None) or os.path.join(
        _THIS_DIR, "artifacts", f"{cfg.task.name}_{str(cfg.algo.name).lower()}"
    )
    export_dir = os.path.abspath(os.path.expanduser(str(export_dir)))
    os.makedirs(export_dir, exist_ok=True)

    # Isaac Sim must be initialised before importing the env registry.
    cfg.headless = True
    simulation_app = init_simulation_app(cfg)

    try:
        from omni_drones.envs.isaac_env import IsaacEnv
        from omni_drones.utils.torchrl.transforms import (
            PIDRateController,
            RateController,
            ravel_composite,
        )

        env_class = IsaacEnv.REGISTRY[cfg.task.name]
        base_env = env_class(cfg, headless=True)

        transforms = [InitTracker()]
        if cfg.task.get("ravel_obs", False):
            transforms.append(
                ravel_composite(base_env.observation_spec, ("agents", "observation"))
            )
        if cfg.task.get("ravel_obs_central", False):
            transforms.append(
                ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
            )

        action_transform = cfg.task.get("action_transform", None)
        if action_transform is not None:
            action_transform = action_transform.lower()
            controller = getattr(base_env, "controller", None) or getattr(
                base_env, "pursuer_controller", None
            )
            if controller is None:
                raise RuntimeError("Action transform requires a controller.")
            if action_transform.startswith("rate"):
                transforms.append(RateController(controller.to(base_env.device)))
            elif action_transform == "pidrate":
                transforms.append(PIDRateController(controller))
            else:
                raise NotImplementedError(
                    f"export_policy only supports rate/PIDrate action transforms; "
                    f"got '{action_transform}'."
                )

        env = TransformedEnv(base_env, Compose(*transforms))
        env.set_seed(int(cfg.get("seed", 0)))

        algo_name = str(cfg.algo.name).lower()
        if algo_name in ("sac", "td3"):
            policy = ALGOS[algo_name](cfg.algo, env.agent_spec["drone"], device=base_env.device)
        else:
            policy = ALGOS[algo_name](
                cfg.algo,
                env.observation_spec,
                env.action_spec,
                env.reward_spec,
                device=base_env.device,
            )

        load_checkpoint_into_policy(policy, ckpt_path, base_env.device)
        if hasattr(policy, "eval"):
            policy.eval()

        obs_dim = env.observation_spec[("agents", "observation")].shape[-1]
        action_dim = env.action_spec[("agents", "action")].shape[-1]

        det_actor = extract_deterministic_actor(policy, algo_name).to(base_env.device)

        # Trace to a standalone TorchScript module.
        example = torch.zeros(1, obs_dim, device=base_env.device)
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            traced = torch.jit.trace(det_actor, example, check_trace=False)

            # Numerically validate the trace against the eager module.
            probe = torch.randn(64, obs_dim, device=base_env.device)
            eager_out = det_actor(probe)
            traced_out = traced(probe)
        max_err = (eager_out - traced_out).abs().max().item()
        if max_err > 1e-4:
            raise RuntimeError(
                f"TorchScript trace diverges from eager module (max abs error "
                f"{max_err:.3e}). Refusing to export a mismatched policy."
            )
        logging.info("Trace validated (max abs error %.3e).", max_err)

        ts_path, meta_path = ic.artifact_paths(export_dir)
        traced.save(ts_path)
        metadata = build_metadata(cfg, base_env, obs_dim, action_dim)
        ic.save_metadata(metadata, meta_path)

        logging.info("Exported TorchScript policy -> %s", ts_path)
        logging.info("Exported metadata          -> %s", meta_path)
        print(f"\nExport complete:\n  {ts_path}\n  {meta_path}\n")
        print(OmegaConf.to_yaml({"obs_dim": obs_dim, "action_dim": action_dim,
                                 "metadata": metadata.to_dict()}))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    # play.yaml declares `hydra.searchpath: [file://../cfg]`, which Hydra
    # resolves relative to the *current working directory* (it works because
    # play.py is run from scripts/). This exporter is typically run from the
    # repo root, so inject an absolute searchpath to the repo's cfg/ dir unless
    # the user already provided one.
    if not any(arg.startswith("hydra.searchpath") for arg in sys.argv[1:]):
        _cfg_dir = os.path.join(_REPO_ROOT, "cfg")
        sys.argv.append(f"hydra.searchpath=[file://{_cfg_dir}]")
    main()
