"""Export AMSPB PEG policies trained by train_pursuer.py to deploy artifacts.

This exporter mirrors scripts/amspb_peg/train_pursuer.py environment/policy
construction, then extracts a deterministic actor and writes:
- policy_ts.pt
- metadata.json

The artifact is intended to be consumed by scripts/amspb_peg/deploy/
intercept_controller.py.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict

import hydra
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AMSPB_DIR = os.path.dirname(_THIS_DIR)
_SCRIPTS_DIR = os.path.dirname(_AMSPB_DIR)
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)

if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from scripts.deploy.export_policy import (  # noqa: E402
    extract_deterministic_actor,
    load_checkpoint_into_policy,
)
import scripts.deploy.intercept_common as ic  # noqa: E402

from omni_drones import init_simulation_app  # noqa: E402
from omni_drones.learning import ALGOS  # noqa: E402
from omni_drones.utils.torchrl.transforms import (  # noqa: E402
    FromDiscreteAction,
    FromMultiDiscreteAction,
    AMSPBRateController,
    VelController,
)


def _as_list(value: Any) -> list[float]:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    return [float(v) for v in tensor.reshape(-1).tolist()]


def _build_pursuit_notes(cfg, base_env, role: str) -> Dict[str, Any]:
    pursuit_obs = {
        "time_encoding": bool(cfg.task.time_encoding),
        "time_encoding_dim": int(getattr(base_env, "time_encoding_dim", 1)),
        "include_distances": bool(cfg.task.include_distances),
        "include_closing_velocity": bool(cfg.task.include_closing_velocity),
        "include_heading_to_target": bool(cfg.task.include_heading_to_target),
        "include_last_action": bool(cfg.task.include_last_action),
        "include_effort": bool(cfg.task.include_effort),
        "use_body_rates": bool(cfg.task.use_body_rates),
        "prev_traj_steps": int(getattr(base_env, "prev_traj_steps", 0)),
        "prev_evader_traj_steps": int(getattr(base_env, "prev_evader_traj_steps", 0)),
        "prev_evader_steps_fix": int(getattr(base_env, "prev_evader_steps_fix", 4)),
        "max_episode_length": int(base_env.max_episode_length),
        "k_p": _as_list(base_env.K_P),
        "k_v": _as_list(base_env.K_V),
        "k_omega": _as_list(base_env.K_OMEGA),
        "k_rp": _as_list(base_env.K_RP),
        "arena_center": _as_list(base_env.arena_center),
    }

    return {
        "task": str(cfg.task.name),
        "role": str(role),
        "checkpoint": str(cfg.get("checkpoint", "")),
        "action_transform": str(cfg.task.get("action_transform", "None")),
        "pursuit_obs": pursuit_obs,
    }


def _build_metadata(cfg, base_env, obs_dim: int, action_dim: int, role: str) -> ic.PolicyMetadata:
    controller = getattr(base_env, "cont_pursuer", None)
    if controller is None:
        raise RuntimeError("Failed to find base_env.cont_pursuer while building metadata.")

    ctbr_cfg = ic.CTBRConfig(
        target_clip=float(getattr(controller, "target_clip", 1.0)),
        min_thrust_ratio=float(getattr(controller, "min_thrust_ratio", 0.0)),
        max_thrust_ratio=float(getattr(controller, "max_thrust_ratio", 0.9)),
        lpf_coef=float(getattr(controller, "LPF_coef", 1.0)),
        dt=float(cfg.sim.dt),
    )

    # Reuse PolicyMetadata schema for compatibility with deploy loaders.
    # Pursuit-specific observation structure is stored in notes["pursuit_obs"].
    obs_cfg = ic.ObsConfig(
        use_ab_world_frame=False,
        use_relative_velocity=False,
        use_previous_action=False,
        obs_dim=int(obs_dim),
        action_dim=int(action_dim),
    )

    notes = _build_pursuit_notes(cfg, base_env, role)

    return ic.PolicyMetadata(
        artifact_version=ic.ARTIFACT_VERSION,
        algo=str(cfg.algo.name).lower(),
        obs=obs_cfg,
        ctbr=ctbr_cfg,
        sim_dt=float(cfg.sim.dt),
        notes=notes,
    )


def _build_env(cfg):
    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=True)

    transforms = [InitTracker()]
    action_transform = cfg.task.get("action_transform", None)
    if action_transform is not None and not str(action_transform).startswith("None"):
        action_transform = str(action_transform).lower()
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromMultiDiscreteAction(nbins=nbins))
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromDiscreteAction(nbins=nbins))
        elif action_transform.startswith("velocity"):
            transforms.append(
                VelController(
                    controller=base_env.cont_pursuer,
                    max_vel=base_env.K_V,
                    in_keys_inv=("agents", "pursuer_state"),
                )
            )
        elif action_transform.startswith("rate"):
            transforms.append(
                AMSPBRateController(
                    controller=base_env.cont_pursuer,
                    in_keys_inv=("agents", "pursuer_state"),
                )
            )
        elif action_transform.startswith("thrust"):
            pass
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).eval()
    env.set_seed(int(cfg.seed))
    return base_env, env


@hydra.main(version_base=None, config_path=_AMSPB_DIR, config_name="train_pursuer")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    ckpt_path = cfg.get("checkpoint", None)
    if not ckpt_path:
        raise ValueError("Missing checkpoint path. Pass checkpoint=/path/to/checkpoint_final.pt")
    ckpt_path = os.path.abspath(os.path.expanduser(str(ckpt_path)))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    role = str(cfg.get("role", "pursuer")).lower()
    if role not in {"pursuer", "evader"}:
        raise ValueError("role must be one of: pursuer, evader")

    default_export = os.path.join(_THIS_DIR, "artifacts", f"{cfg.task.name}_{cfg.algo.name}_{role}".lower())
    export_dir = os.path.abspath(os.path.expanduser(str(cfg.get("export_dir", default_export))))
    os.makedirs(export_dir, exist_ok=True)

    cfg.headless = True
    simulation_app = init_simulation_app(cfg)

    try:
        base_env, env = _build_env(cfg)

        algo_name = str(cfg.algo.name).lower()
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

        obs_dim = int(env.observation_spec[("agents", "observation")].shape[-1])
        action_dim = int(env.action_spec[("agents", "action")].shape[-1])

        if action_dim != 4:
            raise RuntimeError(
                "AMSPB deploy controller expects a 4D CTBR action head; "
                f"got action_dim={action_dim}."
            )

        det_actor = extract_deterministic_actor(policy, algo_name).to(base_env.device).eval()

        example = torch.zeros(1, obs_dim, device=base_env.device)
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            traced = torch.jit.trace(det_actor, example, check_trace=False)

            probe = torch.randn(64, obs_dim, device=base_env.device)
            eager_out = det_actor(probe)
            traced_out = traced(probe)

        max_err = (eager_out - traced_out).abs().max().item()
        if max_err > 1e-4:
            raise RuntimeError(
                f"TorchScript trace diverges from eager module (max abs error {max_err:.3e})."
            )

        ts_path, meta_path = ic.artifact_paths(export_dir)
        traced.save(ts_path)

        metadata = _build_metadata(cfg, base_env, obs_dim, action_dim, role)
        ic.save_metadata(metadata, meta_path)

        logging.info("Exported TorchScript policy -> %s", ts_path)
        logging.info("Exported metadata          -> %s", meta_path)
        logging.info("Trace max abs error: %.3e", max_err)
        print(f"\nExport complete:\n  {ts_path}\n  {meta_path}\n")

    finally:
        simulation_app.close()


if __name__ == "__main__":
    if not any(arg.startswith("hydra.searchpath") for arg in sys.argv[1:]):
        cfg_dir = os.path.join(_REPO_ROOT, "cfg")
        sys.argv.append(f"hydra.searchpath=[file://{cfg_dir}]")
    main()
