"""Pin a run's config to the one its checkpoint was trained with.

``scripts/play.py``, ``scripts/evaluate.py`` and ``scripts/deploy/export_policy.py``
must build the env with the *same* config the policy was trained with, not the
current ``cfg/task/Intercept.yaml`` defaults. Some flags (e.g.
``use_evader_rel_dist``) change the observation *meaning* without changing
``obs_dim``, and ``action_transform`` / ``sim.dt`` change how the policy's output
is interpreted and how the world evolves -- a stale default silently evaluates
the policy off-distribution.

Every training run saves its resolved config next to the checkpoint: a nested
``train_config.yaml`` written by ``train.py`` (preferred), or W&B's flattened
``files/config.yaml`` (fallback for older runs). The helpers here load that
snapshot and overlay it onto the run cfg.

These functions use only ``OmegaConf`` (and, lazily, ``hydra`` for CLI
overrides) -- no Isaac / torch -- so they are safe to import and call before the
``SimulationApp`` is created.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def find_training_config(ckpt_path: str) -> Optional[str]:
    """Locate the training-config snapshot saved alongside a checkpoint.

    Prefers a nested ``train_config.yaml`` (written by ``train.py``); falls back
    to W&B's flattened ``config.yaml``. Checkpoints may live in ``files/`` or a
    nested ``files/files/``, so search the checkpoint directory and a few
    parents. Returns ``None`` when nothing is found.
    """
    directory = os.path.dirname(os.path.abspath(ckpt_path))
    for _ in range(4):
        for name in ("train_config.yaml", "config.yaml"):
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return candidate
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent
    return None


def _wandb_flat_to_nested(data: dict) -> dict:
    """Un-flatten a W&B ``config.yaml`` (``a.b.c: {value: X}``) into nested dicts."""
    nested: dict = {}
    for key, entry in data.items():
        if key == "_wandb" or not isinstance(key, str):
            continue
        value = entry.get("value") if isinstance(entry, dict) and "value" in entry else entry
        node = nested
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        # Guard against a scalar shadowing a nested branch (e.g. both `algo` and
        # `algo.name` present); prefer the nested branch.
        if not isinstance(node, dict):
            continue
        node[parts[-1]] = value
    return nested


def load_training_sections(
    config_path: str, sections: Sequence[str] = ("task", "algo")
) -> Dict[str, dict]:
    """Return the requested top-level ``sections`` from a training-config snapshot.

    Handles both the nested (``train_config.yaml``) and W&B-flattened
    (``config.yaml``) layouts. Missing sections are simply omitted.

    The snapshot is parsed with OmegaConf rather than ``yaml.safe_load`` so that
    dot-less scientific notation (e.g. ``success_radius_lr: 2e-6``) is read as a
    float. PyYAML's 1.1 parser treats it as a *string*, which would silently
    corrupt numeric config and crash the env (``str * float``).
    """
    loaded = OmegaConf.load(config_path)
    raw = OmegaConf.to_container(loaded, resolve=False)
    if not isinstance(raw, dict):
        return {}

    # Detect the W&B flattened format: keys like "task.reward_precision_weight".
    is_wandb_flat = any(
        isinstance(k, str) and "." in k and isinstance(v, dict) and "value" in v
        for k, v in raw.items()
    )
    nested = _wandb_flat_to_nested(raw) if is_wandb_flat else raw

    out: Dict[str, dict] = {}
    for section in sections:
        value = nested.get(section)
        # Some hydra dumps wrap each entry as {"value": ...}.
        if isinstance(value, dict) and set(value.keys()) == {"value"}:
            value = value["value"]
        if isinstance(value, dict):
            out[section] = value
    return out


def read_training_obs_flags(config_path: str) -> dict:
    """Extract the ``task.observation.*`` block from a training-config snapshot."""
    sections = load_training_sections(config_path, sections=("task",))
    task = sections.get("task", {})
    obs = task.get("observation") if isinstance(task, dict) else None
    return dict(obs) if isinstance(obs, dict) else {}


def collect_cli_overrides(prefixes: Sequence[str] = ("task.", "algo.")) -> List[str]:
    """Return CLI overrides (dotlist) that target one of ``prefixes``.

    These are re-applied on top of the trained config so explicit user tweaks
    (e.g. ``task.evader.trajectory_types=[random]`` or ``task.env.num_envs=1``)
    still win. Returns ``[]`` outside a Hydra run.
    """
    try:
        from hydra.core.hydra_config import HydraConfig

        overrides = list(HydraConfig.get().overrides.task)
    except Exception:
        return []

    dotlist: List[str] = []
    for override in overrides:
        if "=" not in override:
            continue  # e.g. defaults-list override `task=Intercept`
        cleaned = override.lstrip("+~")
        key = cleaned.split("=", 1)[0]
        if any(key.startswith(prefix) for prefix in prefixes):
            dotlist.append(cleaned)
    return dotlist


def apply_training_config(
    cfg,
    ckpt_path: str,
    sections: Sequence[str] = ("task", "algo"),
    apply_cli: bool = True,
) -> bool:
    """Pin ``cfg`` to the config the checkpoint at ``ckpt_path`` was trained on.

    Overlays the trained ``task``/``algo`` (and re-syncs ``cfg.sim`` / ``cfg.env``
    from the trained ``task.sim`` / ``task.env``) so the policy is always built
    and rolled out with its training config, regardless of later edits to
    ``cfg/task/``. Explicit CLI ``task.*`` / ``algo.*`` overrides are re-applied
    on top when ``apply_cli`` is set.

    Returns ``True`` when a snapshot was found and applied, ``False`` (with a
    warning) when none was found so callers fall back to the live config.
    """
    if not ckpt_path:
        return False

    config_path = find_training_config(ckpt_path)
    if config_path is None:
        logger.warning(
            "No training-config snapshot (train_config.yaml / config.yaml) found "
            "next to %s. Falling back to the live config in cfg/task/. "
            "Observation/reward/dynamics may NOT match how this policy was "
            "trained.",
            ckpt_path,
        )
        return False

    trained = load_training_sections(config_path, sections=sections)
    if not trained:
        logger.warning(
            "No %s section found in %s; using the live config.",
            "/".join(sections),
            config_path,
        )
        return False

    trained_task = trained.get("task")
    if trained_task is not None:
        trained_task_cfg = OmegaConf.create(trained_task)
        live_name = cfg.task.get("name", None) if "task" in cfg else None
        if live_name == trained_task_cfg.get("name", None):
            # Same task: overlay the trained values, keeping any newer structural
            # keys the live task defines.
            cfg.task = OmegaConf.merge(cfg.task, trained_task_cfg)
        else:
            logger.warning(
                "Live task '%s' differs from the checkpoint's task '%s'; using "
                "the trained one. Pass task=%s so Hydra composes the right "
                "defaults (controllers, etc.).",
                live_name,
                trained_task_cfg.get("name", None),
                trained_task_cfg.get("name", None),
            )
            cfg.task = trained_task_cfg

    trained_algo = trained.get("algo")
    if trained_algo is not None:
        if "algo" in cfg:
            cfg.algo = OmegaConf.merge(cfg.algo, OmegaConf.create(trained_algo))
        else:
            cfg.algo = OmegaConf.create(trained_algo)

    # Re-apply explicit CLI task.*/algo.* overrides on top of the trained base.
    if apply_cli:
        cfg.merge_with_dotlist(collect_cli_overrides(("task.", "algo.")))

    # play.yaml / evaluate.yaml wire `sim: ${task.sim}` and `env: ${task.env}`,
    # but that interpolation was already resolved into independent copies;
    # refresh them so the sim/env used at runtime match the (possibly
    # CLI-overridden) trained task. `task.env.num_envs=1` therefore propagates.
    if "task" in cfg:
        if "sim" in cfg.task:
            cfg.sim = cfg.task.sim
        if "env" in cfg.task:
            cfg.env = cfg.task.env

    # The env reads the top-level `cfg.env` / `cfg.sim` (not `cfg.task.env`), so
    # apply direct `env.*` / `sim.*` CLI overrides AFTER the re-sync above; this
    # lets the intuitive `env.num_envs=1` win too, not just `task.env.num_envs=1`.
    if apply_cli:
        cfg.merge_with_dotlist(collect_cli_overrides(("sim.", "env.")))

    logger.info(
        "Using training config from %s (task='%s', algo='%s').",
        config_path,
        cfg.task.get("name", "?") if "task" in cfg else "?",
        cfg.algo.get("name", "?") if "algo" in cfg else "?",
    )
    return True


def apply_training_obs_config(cfg, ckpt_path: str) -> None:
    """Backward-compatible shim: pin only ``cfg.task.observation`` from the snapshot.

    Prefer :func:`apply_training_config`, which also pins ``action_transform``
    and the sim/dynamics. Kept for callers that only need the observation layout.
    """
    if not ckpt_path:
        return
    config_path = find_training_config(ckpt_path)
    if config_path is None:
        logger.warning(
            "No config snapshot found next to the checkpoint; falling back to the "
            "current task.observation config. Verify the flags match training."
        )
        return
    flags = read_training_obs_flags(config_path)
    if not flags:
        logger.warning(
            "No task.observation.* entries in %s; using current config.", config_path
        )
        return
    for name, value in flags.items():
        cfg.task.observation[name] = value
    logger.info(
        "Using training observation config from %s: %s", config_path, dict(flags)
    )
