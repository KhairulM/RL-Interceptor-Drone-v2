import datetime
import json
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import hydra
import torch
from omegaconf import OmegaConf
from setproctitle import setproctitle
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm import tqdm

from omni_drones import init_simulation_app
from omni_drones.learning import ALGOS
from omni_drones.utils.checkpoint_config import apply_training_config
from omni_drones.utils.torchrl import Collector, EpisodeStats
from omni_drones.utils.torchrl.transforms import (
    FromDiscreteAction,
    FromMultiDiscreteAction,
    PIDRateController,
    RateController,
    ravel_composite,
)

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


FILE_PATH = os.path.dirname(__file__)


class _NoOpRun:
    def __init__(self):
        self.name = "omnidrones-evaluate"

    def log(self, *args, **kwargs):
        return None


@dataclass
class FairnessSignature:
    obs_dim: int
    action_dim: int
    action_transform: str
    dt: float
    max_episode_length: int


class ClassicalPolicy:
    def __init__(self, controller, base_env):
        self.controller = controller
        self.base_env = base_env

    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, tensordict):
        # Classical baselines output the same pre-tanh CTBR action as the RL
        # policy and run through the task's PIDrate transform, so the proven
        # on-board rate PID handles stabilization. Guidance needs the relative
        # state, so read the evader's true position/velocity from the env.
        drone_state = tensordict[("info", "drone_state")]
        done = tensordict.get("done", None)
        evader_state = self.base_env.evader.get_state()
        evader_pos = evader_state[..., 0:3]
        evader_vel = evader_state[..., 7:10]
        action = self.controller(drone_state, evader_pos, evader_vel, done)
        tensordict.set(("agents", "action"), action)
        return tensordict


def _resolve_existing_path(path: str, expect: str):
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

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized not in seen:
            seen.add(normalized)
            unique_candidates.append(normalized)

    for candidate in unique_candidates:
        if expect == "dir" and os.path.isdir(candidate):
            return candidate
        if expect == "file" and os.path.isfile(candidate):
            return candidate
    return None


def _infer_wandb_run_id_from_run_dir(run_dir: str):
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


def _init_wandb_for_eval(cfg):
    assert wandb is not None
    wandb_cfg = cfg.wandb

    run_id = wandb_cfg.get("run_id", None)
    if run_id is not None:
        return wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            mode=wandb_cfg.mode,
            id=run_id,
            resume="allow",
        )

    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    run_name = f"{wandb_cfg.run_name}/{time_str}"
    return wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        mode=wandb_cfg.mode,
        group=wandb_cfg.group,
        name=run_name,
        job_type="evaluate",
        tags=wandb_cfg.tags,
    )


def _flatten_key(key: Any) -> str:
    if isinstance(key, tuple):
        return ".".join(map(str, key))
    return str(key)


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    vals = list(values)
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, math.sqrt(var)


def _build_env(cfg):
    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms: list = [InitTracker()]

    if cfg.task.get("ravel_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))  # type: ignore[arg-type]
    if cfg.task.get("ravel_obs_central", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation_central")))  # type: ignore[arg-type]

    action_transform = str(cfg.task.get("action_transform", "none"))
    normalized = action_transform.lower()

    if normalized.startswith("multidiscrete"):
        nbins = int(normalized.split(":")[1])
        transforms.append(FromMultiDiscreteAction(nbins=nbins))
    elif normalized.startswith("discrete"):
        nbins = int(normalized.split(":")[1])
        transforms.append(FromDiscreteAction(nbins=nbins))
    elif normalized.startswith("rate"):
        controller = getattr(base_env, "controller", None) or getattr(base_env, "pursuer_controller", None)
        if controller is None:
            raise RuntimeError("Rate action transform requires a task controller")
        transforms.append(RateController(controller.to(base_env.device)))
    elif normalized == "pidrate":
        controller = getattr(base_env, "controller", None) or getattr(base_env, "pursuer_controller", None)
        if controller is None:
            raise RuntimeError("PIDRate action transform requires a task controller")
        transforms.append(PIDRateController(controller))

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(int(cfg.seed))
    return base_env, env


def _build_rl_policy(cfg, env, base_env):
    algo_name = cfg.algo.name.lower()
    if algo_name in ["sac", "td3"]:
        policy = ALGOS[algo_name](cfg.algo, env.agent_spec["drone"], device=base_env.device)
    else:
        policy = ALGOS[algo_name](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device,
        )

    ckpt_path = cfg.get("checkpoint", None)
    if ckpt_path is None:
        raise ValueError("RL evaluation requires checkpoint=... or wandb_run_dir=...")

    resolved = _resolve_existing_path(ckpt_path, expect="file")
    if resolved is None:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(resolved, map_location=base_env.device)
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "policy_state_dict" in state:
            state = state["policy_state_dict"]

    try:
        policy.load_state_dict(state)
    except RuntimeError as e:
        err_msg = str(e)
        # If the mismatch is due to a changed observation layout (e.g., adding
        # previous_action), hint at the correct CLI override instead of falling
        # through to strict=False which also fails on shape mismatches.
        if "size mismatch" in err_msg and "actor.module" in err_msg:
            raise RuntimeError(
                f"Checkpoint observation dim does not match the current env config.\n"
                f"This usually means the checkpoint was trained with a different\n"
                f"observation layout (e.g., use_previous_action=true vs false).\n\n"
                f"Try overriding the flag to match the checkpoint:\n"
                f"  evaluate.py ... task.observation.use_previous_action=false\n"
            ) from e
        policy.load_state_dict(state, strict=False)

    return policy


def _build_classical_policy(method: str, cfg, base_env):
    from omni_drones.controllers.apf_controller import APFController
    from omni_drones.controllers.intercept_baseline_common import GeometricCTBR
    from omni_drones.controllers.kinematic_mpc_controller import KinematicMPCController
    from omni_drones.controllers.nonlinear_mpc_controller import NonlinearMPCController
    from omni_drones.controllers.proportional_navigation_controller import (
        ProportionalNavigationController,
    )
    from omni_drones.controllers.pure_pursuit_controller import PurePursuitController

    # Drive the baselines through the pursuer's proven PIDrate CTBR stack via a
    # geometric outer loop. Calibrate the hover throttle from the drone params:
    # per-rotor max thrust KF = max_rot_vel^2 * force_const, hover per rotor =
    # m*g/4, and thrust ~ throttle^2 -> hover_throttle = sqrt(m*g/(4*KF)).
    pc = base_env.pursuer_controller
    mass = float(base_env.pursuer.params["mass"])
    g = 9.81
    per_rotor_max = float(pc.max_thrusts.reshape(-1)[0])
    hover_throttle = math.sqrt(mass * g / (4.0 * per_rotor_max))
    # Gains tuned for the thrust-limited Crazyflie pursuer (thrust-to-weight
    # ~1.4). The attitude loop is deliberately gentle (k_att, tilt) so the drone
    # does not tumble on this low-authority, motor-lagged airframe, and the
    # position loop is well damped (kv >= kp) for a clean, overshoot-free
    # approach. Paired with the altitude-priority thrust allocation in
    # GeometricCTBR, this keeps the pursuer airborne instead of trading its
    # altitude away to chase horizontally (which cratered every baseline before).
    ctbr = GeometricCTBR(
        mass=mass,
        g=g,
        hover_throttle=hover_throttle,
        target_clip=float(pc.target_clip),
        min_ratio=float(pc.min_thrust_ratio),
        max_ratio=float(pc.max_thrust_ratio),
        kp=8.0,
        kv=7.0,
        k_att=6.0,
        k_yaw=1.0,
        max_tilt_deg=28.0,
        # Integral action on altitude nulls the steady-state undershoot from the
        # per-episode thrust-to-weight randomisation (otherwise the docking
        # baselines hover ~0.2 m below the target, just outside the 0.1 m radius).
        ki_z=10.0,
        i_limit_z=1.0,
    )

    # Raw calibration shared with the sampling-based nonlinear MPC.
    ctbr_params = {
        "g": g,
        "hover_throttle": hover_throttle,
        "target_clip": float(pc.target_clip),
        "min_ratio": float(pc.min_thrust_ratio),
        "max_ratio": float(pc.max_thrust_ratio),
    }

    if method == "pure_pursuit":
        controller = PurePursuitController(ctbr, cfg.task)
    elif method == "pn":
        controller = ProportionalNavigationController(ctbr, cfg.task, dt=float(cfg.sim.dt))
    elif method == "kinematic_mpc":
        controller = KinematicMPCController(ctbr, cfg.task, dt=float(cfg.sim.dt))
    elif method == "nonlinear_mpc":
        controller = NonlinearMPCController(ctbr_params, cfg.task, dt=float(cfg.sim.dt))
    elif method == "apf":
        controller = APFController(ctbr, cfg.task)
    else:
        raise ValueError(f"Unknown classical method: {method}")

    controller.reset(int(base_env.num_envs), base_env.device)
    return ClassicalPolicy(controller, base_env)


def _fairness_signature(cfg, env) -> FairnessSignature:
    obs_dim = int(env.observation_spec[("agents", "observation")].shape[-1])
    action_dim = int(env.action_spec[("agents", "action")].shape[-1])
    return FairnessSignature(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_transform=str(cfg.task.get("action_transform", "none")).lower(),
        dt=float(cfg.sim.dt),
        max_episode_length=int(cfg.task.env.max_episode_length),
    )


def _assert_fairness(reference: FairnessSignature, current: FairnessSignature, method: str):
    if reference != current:
        raise RuntimeError(
            "Fairness check failed for method "
            f"{method}. Expected {reference}, got {current}."
        )


def _apply_scenario(cfg, scenario):
    run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    run_cfg.task.evader.trajectory_types = list(scenario.trajectory_types)
    run_cfg.task.evader.speed_range = list(scenario.speed_range)
    run_cfg.task.evader.spawn_distance_range = list(scenario.spawn_distance_range)

    num_envs = int(cfg.eval.num_envs)
    run_cfg.task.env.num_envs = num_envs
    run_cfg.env.num_envs = num_envs
    return run_cfg


def _run_method_on_env(cfg, base_env, env, method: str, run_label: str) -> Dict[str, float]:
    if method == "rl":
        policy = _build_rl_policy(cfg, env, base_env)
    else:
        policy = _build_classical_policy(method, cfg, base_env)
    # All methods output the same pre-tanh CTBR action and run through the
    # transformed (PIDrate) env, so the flight pipeline is identical to RL.
    rollout_env = env

    if hasattr(policy, "eval"):
        policy.eval()
    base_env.eval()
    env.eval()

    steps_per_env = int(cfg.eval.steps_per_env)
    frames_per_batch_steps = int(cfg.eval.frames_per_batch_steps)
    frames_per_batch = int(base_env.num_envs) * max(frames_per_batch_steps, 1)
    total_frames = int(base_env.num_envs) * max(steps_per_env, 1)
    max_iters = int(cfg.eval.max_iters)

    planned_iters = math.ceil(total_frames / frames_per_batch)
    if max_iters > 0:
        planned_iters = min(planned_iters, max_iters)

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys)  # type: ignore[arg-type]

    collector = Collector(
        rollout_env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=max(total_frames, frames_per_batch),
        device=cfg.sim.device,
        return_same_td=True,
        trust_policy=True,
    )

    metrics_lists: Dict[str, List[float]] = defaultdict(list)

    # Per-episode pursuer-speed accumulators (m/s). "travelling" speed is over the
    # whole flight; "intercepting" speed is the instant of a successful catch, so
    # it is collected only for episodes whose success_rate latched to 1.
    travel_mean_vals: List[float] = []      # per-episode mean speeds
    travel_peak = float("-inf")             # max over per-episode peak speeds
    intercept_vals: List[float] = []        # per-successful-episode catch speeds

    def _col(td, name: str) -> Optional[torch.Tensor]:
        tensor = td.get(("stats", name), None)
        return tensor.float().reshape(-1) if tensor is not None else None

    with set_exploration_type(ExplorationType.MODE):
        for i, data in enumerate(tqdm(collector, total=planned_iters, desc=run_label)):
            episode_stats.add(data.to_tensordict())
            if len(episode_stats) >= base_env.num_envs:
                popped = episode_stats.pop()
                for key, value in popped.items(True, True):
                    metric_name = _flatten_key(key)
                    metrics_lists[metric_name].append(float(value.float().mean().item()))

                mean_travel = _col(popped, "mean_travel_v")
                max_travel = _col(popped, "max_travel_v")
                intercept = _col(popped, "intercept_v")
                success = _col(popped, "success_rate")
                if mean_travel is not None:
                    travel_mean_vals.extend(mean_travel.tolist())
                if max_travel is not None and max_travel.numel():
                    travel_peak = max(travel_peak, float(max_travel.max().item()))
                if intercept is not None and success is not None:
                    caught = intercept[success > 0.5]
                    if caught.numel():
                        intercept_vals.extend(caught.tolist())
            if i + 1 >= planned_iters:
                break

    aggregated = {
        metric: (sum(values) / max(len(values), 1))
        for metric, values in metrics_lists.items()
    }
    aggregated["rollout_steps"] = float(planned_iters * frames_per_batch_steps)
    aggregated["mean_travel_velocity"] = (
        sum(travel_mean_vals) / len(travel_mean_vals) if travel_mean_vals else float("nan")
    )
    aggregated["max_travel_velocity"] = (
        travel_peak if travel_peak != float("-inf") else float("nan")
    )
    aggregated["mean_intercept_velocity"] = (
        sum(intercept_vals) / len(intercept_vals) if intercept_vals else float("nan")
    )
    aggregated["max_intercept_velocity"] = (
        max(intercept_vals) if intercept_vals else float("nan")
    )
    return aggregated


def _derive_metrics(metrics: Dict[str, float], dt: float) -> Dict[str, float]:
    success_rate = float(metrics.get("stats.success_rate", float("nan")))
    episode_len = float(metrics.get("stats.episode_len", float("nan")))
    intercept_time_s = episode_len * dt if not math.isnan(episode_len) else float("nan")

    return {
        "success_rate": success_rate,
        "intercept_time_s": intercept_time_s,
        "miss_distance": float(metrics.get("stats.distance", float("nan"))),
        "episode_return": float(metrics.get("stats.return", float("nan"))),
        # Pursuer speed (m/s): travelling = over the flight, intercepting = at the
        # instant of a successful catch. Computed per-run in _run_method_on_env.
        "mean_travel_velocity": float(metrics.get("mean_travel_velocity", float("nan"))),
        "max_travel_velocity": float(metrics.get("max_travel_velocity", float("nan"))),
        "mean_intercept_velocity": float(metrics.get("mean_intercept_velocity", float("nan"))),
        "max_intercept_velocity": float(metrics.get("max_intercept_velocity", float("nan"))),
    }


def _write_summary_json(output_dir: str, payload: Dict[str, Any]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"evaluate_summary_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def _clear_robot_registry() -> None:
    from omni_drones.robots.robot import RobotBase

    RobotBase._robots.clear()


@hydra.main(config_path=FILE_PATH, config_name="evaluate", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    wandb_run_dir = cfg.get("wandb_run_dir", None)
    if wandb_run_dir:
        resolved_wandb_run_dir = _resolve_existing_path(wandb_run_dir, expect="dir")
        if resolved_wandb_run_dir is None:
            raise FileNotFoundError(f"wandb_run_dir not found: {wandb_run_dir}")

        if cfg.get("checkpoint", None) is None:
            cfg.checkpoint = os.path.join(resolved_wandb_run_dir, "files", "checkpoint_final.pt")
            logging.info("Using checkpoint inferred from wandb_run_dir: %s", cfg.checkpoint)

        inferred_run_id = _infer_wandb_run_id_from_run_dir(resolved_wandb_run_dir)
        if inferred_run_id and cfg.wandb.get("run_id", None) is None:
            cfg.wandb.run_id = inferred_run_id

    # The RL policy must be evaluated with the config it was trained with
    # (observation layout, action_transform, sim dynamics), not the current
    # Intercept.yaml defaults. The benchmark scenarios below still override the
    # evader trajectory/speed/spawn-distance and num_envs on top (see
    # _apply_scenario). Classical baselines read the true env state and are
    # unaffected by the observation layout.
    ckpt_for_cfg = cfg.get("checkpoint", None)
    if ckpt_for_cfg:
        resolved_ckpt = _resolve_existing_path(ckpt_for_cfg, expect="file") or ckpt_for_cfg
        apply_training_config(cfg, resolved_ckpt)

    simulation_app = init_simulation_app(cfg)

    # Import isaacsim-dependent controllers after SimulationApp is instantiated
    from omni_drones.controllers.intercept_baseline_common import GeometricCTBR
    from omni_drones.controllers.kinematic_mpc_controller import KinematicMPCController
    from omni_drones.controllers.nonlinear_mpc_controller import NonlinearMPCController
    from omni_drones.controllers.proportional_navigation_controller import (
        ProportionalNavigationController,
    )
    from omni_drones.controllers.pure_pursuit_controller import PurePursuitController

    use_wandb = wandb is not None and str(cfg.wandb.mode).lower() != "disabled"
    if use_wandb:
        run = _init_wandb_for_eval(cfg)
    else:
        run = _NoOpRun()

    setproctitle(run.name or f"{cfg.task.name}-evaluate")
    print(OmegaConf.to_yaml(cfg))

    methods = [str(m).lower() for m in cfg.eval.methods]
    seeds = [int(s) for s in cfg.eval.seeds]
    scenarios = list(cfg.eval.scenarios)

    runs = []
    summary_bucket: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(list)

    for scenario in scenarios:
        scenario_name = str(scenario.name)
        for seed in seeds:
            _clear_robot_registry()
            run_cfg = _apply_scenario(cfg, scenario)
            run_cfg.seed = seed
            base_env, env = _build_env(run_cfg)
            signature_ref: Optional[FairnessSignature] = None
            try:
                signature_ref = _fairness_signature(run_cfg, env)
                for method in methods:
                    if bool(cfg.eval.strict_fairness):
                        _assert_fairness(signature_ref, _fairness_signature(run_cfg, env), method)

                    metrics = _run_method_on_env(
                        run_cfg,
                        base_env=base_env,
                        env=env,
                        method=method,
                        run_label=f"{scenario_name}|seed={seed}|{method}",
                    )

                    derived = _derive_metrics(metrics, dt=float(run_cfg.sim.dt))
                    record = {
                        "scenario": scenario_name,
                        "seed": seed,
                        "method": method,
                        **metrics,
                        **derived,
                    }
                    runs.append(record)
                    summary_bucket[(scenario_name, method)].append(record)
            finally:
                simulation_close = getattr(base_env, "close", None)
                if callable(simulation_close):
                    simulation_close()
                _clear_robot_registry()

    summaries = []
    for (scenario_name, method), records in summary_bucket.items():
        success_mean, success_std = _mean_std([r["success_rate"] for r in records])
        time_mean, time_std = _mean_std([r["intercept_time_s"] for r in records])
        miss_mean, miss_std = _mean_std([r["miss_distance"] for r in records])

        # Velocity summaries (m/s): avg = mean across seeds, max = peak across
        # seeds (max of the per-run peaks). NaN-only records (e.g. no successful
        # interception) are dropped before aggregating.
        def _avg(key: str) -> float:
            vals = [r[key] for r in records if not math.isnan(r[key])]
            return sum(vals) / len(vals) if vals else float("nan")

        def _peak(key: str) -> float:
            vals = [r[key] for r in records if not math.isnan(r[key])]
            return max(vals) if vals else float("nan")

        summaries.append(
            {
                "scenario": scenario_name,
                "method": method,
                "n": len(records),
                "success_rate_mean": success_mean,
                "success_rate_std": success_std,
                "intercept_time_s_mean": time_mean,
                "intercept_time_s_std": time_std,
                "miss_distance_mean": miss_mean,
                "miss_distance_std": miss_std,
                "avg_travel_velocity": _avg("mean_travel_velocity"),
                "max_travel_velocity": _peak("max_travel_velocity"),
                "avg_intercept_velocity": _avg("mean_intercept_velocity"),
                "max_intercept_velocity": _peak("max_intercept_velocity"),
            }
        )

    payload = {
        "generated_at": datetime.datetime.now().isoformat(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "methods": methods,
        "seeds": seeds,
        "runs": runs,
        "summary": summaries,
    }

    output_dir = _resolve_existing_path(str(cfg.eval.output_dir), expect="dir")
    if output_dir is None:
        output_dir = os.path.normpath(os.path.join(FILE_PATH, str(cfg.eval.output_dir)))

    json_path = _write_summary_json(output_dir, payload)
    logging.info("Wrote benchmark summary to %s", json_path)

    if use_wandb:
        assert wandb is not None
        columns = [
            "scenario",
            "method",
            "n",
            "success_rate_mean",
            "success_rate_std",
            "intercept_time_s_mean",
            "intercept_time_s_std",
            "miss_distance_mean",
            "miss_distance_std",
            "avg_travel_velocity",
            "max_travel_velocity",
            "avg_intercept_velocity",
            "max_intercept_velocity",
        ]
        table_data = [[row[col] for col in columns] for row in summaries]
        table = wandb.Table(columns=columns, data=table_data)
        run.log({"evaluate/summary_table": table, "evaluate/runs": len(runs)})
        wandb.finish()

    print(f"Evaluation complete. Summary JSON: {json_path}")
    simulation_app.close()


if __name__ == "__main__":
    main()
