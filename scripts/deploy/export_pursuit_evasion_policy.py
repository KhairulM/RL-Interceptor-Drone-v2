# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# See the LICENSE file at the repository root for full terms.

"""Export a trained PursuitEvasion policy to a deployment artifact.

The PursuitEvasion counterpart of [export_policy.py](export_policy.py). It runs
inside the **Isaac Sim** training environment (Python 3.11 ``.venv``), builds
the task/policy exactly the way ``scripts/play.py`` does, loads the checkpoint,
extracts the *deterministic* actor, and serialises it to a self-contained
TorchScript module plus a ``metadata.json`` describing the PursuitEvasion
observation layout and CTBR decoding parameters.

The resulting artifact depends only on ``torch`` and is consumed by
[pursuit_evasion_controller.py](pursuit_evasion_controller.py) in the separate
Crazyswarm2 environment.

Because the PursuitEvasion observation is role-symmetric, the exported artifact
records which ``role`` (pursuer / evader) it was trained as; the controller uses
that to decide which drone is "self" and which is the opponent.

Usage (from the repository root, with the Isaac ``.venv`` active)::

    python scripts/deploy/export_pursuit_evasion_policy.py \\
        task=PursuitEvasion algo=ppo task.role=pursuer \\
        checkpoint=outputs/selfplay/checkpoints/pursuer_stage4.pt \\
        +export_dir=scripts/deploy/artifacts/pe_pursuer_stage4

``export_dir`` defaults to ``deploy/artifacts/<task>_<role>_<algo>``.
"""

import logging
import os
import sys

import hydra
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

# Make the sibling deploy modules importable regardless of CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pursuit_evasion_common as pec  # noqa: E402

from omni_drones import init_simulation_app  # noqa: E402

# Registers every algorithm's structured config with Hydra's ConfigStore; must
# happen before ``@hydra.main`` composes the config so ``algo=ppo`` resolves.
from omni_drones.learning import ALGOS  # noqa: E402

from omni_drones.utils.checkpoint_config import apply_training_config  # noqa: E402

# The deterministic-actor extraction and checkpoint loading are algorithm
# concerns, not task concerns, so they are shared with the Intercept exporter.
from export_policy import (  # noqa: E402
    extract_deterministic_actor,
    load_checkpoint_into_policy,
)

_SCRIPTS_DIR = os.path.split(_THIS_DIR)[0]  # parent of deploy/
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)


# ---------------------------------------------------------------------------
# Metadata assembly
# ---------------------------------------------------------------------------
def build_metadata(cfg, base_env, obs_dim: int, action_dim: int) -> pec.PursuitEvasionMetadata:
    """Read the observation/CTBR/arena parameters off the live env and config."""
    task_obs_cfg = cfg.task.observation
    role = str(cfg.task.get("role", "pursuer")).lower()
    if role not in pec.ROLES:
        raise ValueError(f"task.role must be one of {list(pec.ROLES)}; got '{role}'.")

    obs_cfg = pec.PursuitEvasionObsConfig(
        use_previous_action=bool(task_obs_cfg.get("use_previous_action", True)),
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

    # The task exposes the RL-controlled side's rate controller as `controller`.
    controller = getattr(base_env, "controller")
    ctbr_cfg = pec.CTBRConfig(
        target_clip=float(controller.target_clip),
        min_thrust_ratio=float(controller.min_thrust_ratio),
        max_thrust_ratio=float(controller.max_thrust_ratio),
        lpf_coef=float(controller.LPF_coef),
        dt=float(cfg.sim.dt * cfg.sim.substeps),
    )

    arena = cfg.task.get("arena", {})
    opponent_ids = [
        str(p.get("id", "")) for p in (cfg.task.opponent.get("policies", None) or [])
    ]

    return pec.PursuitEvasionMetadata(
        artifact_version=pec.PE_ARTIFACT_VERSION,
        algo=str(cfg.algo.name).lower(),
        role=role,
        obs=obs_cfg,
        ctbr=ctbr_cfg,
        sim_dt=float(cfg.sim.dt),
        arena_half_xy=float(arena.get("half_xy", 5.0)),
        arena_z_min=float(arena.get("z_min", 0.5)),
        arena_z_max=float(arena.get("z_max", 4.5)),
        # Evaluation-time capture radius (the curriculum only applies in training).
        capture_radius=float(cfg.task.get("success_radius_eval", 0.1)),
        notes={
            "task": str(cfg.task.name),
            "checkpoint": str(cfg.get("checkpoint", "")),
            "drone_model": str(cfg.task.get(role, {}).get("model", "")),
            "action_transform": str(cfg.task.get("action_transform", "")),
            "trained_against": ",".join(opponent_ids),
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
            "No checkpoint provided. Pass checkpoint=/path/to/checkpoint.pt"
        )
    ckpt_path = os.path.abspath(os.path.expanduser(str(ckpt_path)))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build with the checkpoint's training config (observation layout,
    # action_transform, sim dt/substeps) rather than current YAML defaults.
    # CLI overrides (including task.role) still win.
    apply_training_config(cfg, ckpt_path)

    role = str(cfg.task.get("role", "pursuer")).lower()
    export_dir = cfg.get("export_dir", None) or os.path.join(
        _THIS_DIR, "artifacts",
        f"{cfg.task.name}_{role}_{str(cfg.algo.name).lower()}",
    )
    export_dir = os.path.abspath(os.path.expanduser(str(export_dir)))
    os.makedirs(export_dir, exist_ok=True)

    # A single env is enough to read specs + controller parameters.
    cfg.headless = True
    cfg.task.env.num_envs = 1
    simulation_app = init_simulation_app(cfg)

    try:
        from omni_drones.envs.isaac_env import IsaacEnv
        from omni_drones.utils.torchrl.transforms import (
            PIDRateController,
            RateController,
        )

        env_class = IsaacEnv.REGISTRY[cfg.task.name]
        base_env = env_class(cfg, headless=True)

        transforms = [InitTracker()]
        action_transform = cfg.task.get("action_transform", None)
        if action_transform is not None:
            action_transform = action_transform.lower()
            controller = getattr(base_env, "controller", None)
            if controller is None:
                raise RuntimeError("Action transform requires a controller.")
            if action_transform.startswith("rate"):
                transforms.append(RateController(controller.to(base_env.device)))
            elif action_transform == "pidrate":
                transforms.append(PIDRateController(controller))
            else:
                raise NotImplementedError(
                    f"Only rate/PIDrate action transforms are supported; "
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

        example = torch.zeros(1, obs_dim, device=base_env.device)
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            traced = torch.jit.trace(det_actor, example, check_trace=False)

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

        ts_path, meta_path = pec.artifact_paths(export_dir)
        traced.save(ts_path)
        metadata = build_metadata(cfg, base_env, obs_dim, action_dim)
        pec.save_metadata(metadata, meta_path)

        logging.info("Exported TorchScript policy -> %s", ts_path)
        logging.info("Exported metadata          -> %s", meta_path)
        print(f"\nExport complete ({role}):\n  {ts_path}\n  {meta_path}\n")
        print(OmegaConf.to_yaml({"obs_dim": obs_dim, "action_dim": action_dim,
                                 "metadata": metadata.to_dict()}))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    # play.yaml declares `hydra.searchpath: [file://../cfg]`, resolved relative
    # to the CWD. This exporter is typically run from the repo root, so inject
    # an absolute searchpath to the repo's cfg/ dir unless one is already given.
    if not any(a.startswith("hydra.searchpath") for a in sys.argv[1:]):
        sys.argv.append(f"hydra.searchpath=[file://{os.path.join(_REPO_ROOT, 'cfg')}]")
    main()
