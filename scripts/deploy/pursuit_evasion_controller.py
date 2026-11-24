"""PursuitEvasion policy controller built on the drone/mocap helpers.

The PursuitEvasion counterpart of
[intercept_controller_v2.py](intercept_controller_v2.py). It loads an artifact
produced by [export_pursuit_evasion_policy.py](export_pursuit_evasion_policy.py)
and flies it on a Crazyflie (CrazySim or real hardware).

Because the PursuitEvasion observation is role-symmetric, one controller covers
both sides: the artifact's ``role`` decides which drone is "self". The opponent
can be driven three ways (``opponent.mode`` in the config):

* ``policy``   - a second PursuitEvasion artifact flying the opposing role, so
                 both learned policies compete on hardware (the real self-play
                 deployment);
* ``cf``       - a real Crazyflie tracked by mocap but commanded elsewhere;
* ``scripted`` - hover / linear / circular / pure_pursuit / flee position
                 setpoints, mirroring the seed heuristics the population was
                 trained against.

Policies that observe extra estimator outputs carry the estimator in their
artifact and run it here, once per control tick, exactly as in training: the
ESN-OA reservoir (``metadata.obs.use_esn``) and the learned opponent-acceleration
estimator with its online RLS readout (``metadata.obs.use_adapter``).

Episode endings mirror training: capture, either drone crashing, the evader
leaving the arena (the pursuer may leave freely), or the training timeout.

Run with:  python pursuit_evasion_controller.py [--config config_pursuit_evasion.yaml]
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

import pursuit_evasion_common as pec
from drone import CrazyflieDrone, ScriptedDrone, DronePosePublisher
from mocap import MocapReceiver, MocapTfPublisher
from safety import Arena, Level, MocapGate, SafetyConfig, SafetyMonitor

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)

warnings.filterwarnings(
    'ignore',
    message=r'Using legacy TYPE_HOVER_LEGACY\. Please update your crazyflie-firmware\.',
    category=DeprecationWarning,
    module=r'cflib\.crazyflie\.commander',
)
warnings.filterwarnings(
    "ignore",
    message="The supervisor subsystem requires CRTP protocol version 12",
)

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'config_pursuit_evasion.yaml'
)

OPPONENT_MODES = ('policy', 'cf', 'scripted')
SCRIPTED_MOTIONS = ('hover', 'linear', 'circular', 'pure_pursuit', 'flee')
# Legacy heuristic names (as in PursuitEvasion.HEURISTIC_ALIASES).
MOTION_ALIASES = {'pursue': 'pure_pursuit'}


# ===========================================================================
# Configuration
#
# One dataclass per YAML section. The field defaults are the values used when a
# key is omitted; unknown keys are rejected so a typo cannot silently fall back
# to a default. The drone (`pursuer`, `evader`), `mocap` and `safety` sections
# are parsed by their own modules (intercept_common.py, safety.py).
# ===========================================================================
def _section(cls, data, name: str):
    """Build the config dataclass ``cls`` from the YAML mapping ``data``."""
    data = {} if data is None else data
    if not isinstance(data, dict):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    known = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(f"Unknown key(s) {unknown} in config section '{name}'; valid keys: {known}.")
    values = {}
    for f in dataclasses.fields(cls):
        if data.get(f.name) is None:      # omitted or `key:` / `key: null` -> default
            continue
        value = data[f.name]
        # YAML writes 1 for 1.0 and the like; coerce scalars to the declared type.
        if f.type in ('float', float):
            value = float(value)
        elif f.type in ('bool', bool):
            value = bool(value)
        elif f.type in ('str', str):
            value = str(value).strip()
        values[f.name] = value
    return cls(**values)


def _vec3(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"Config key '{name}' must have exactly 3 elements.")
    return arr


@dataclass
class PolicyConfig:
    """`policy`: the learned drone this controller flies."""

    # Folder with policy_ts.pt + metadata.json (+ esn.pt / adapter_basis.pt).
    # Its metadata `role` decides which drone section (pursuer | evader) is "self".
    artifact_dir: str = ''


@dataclass
class ScriptedMotionConfig:
    """`opponent.scripted`: position setpoints for a scripted opponent."""

    type: str = 'hover'          # hover | linear | circular | pure_pursuit | flee
    yaw_deg: float = 0.0         # yaw of every setpoint (deg)
    # Start / hover point [x, y, z] in mocap world coordinates (m); flee holds its z.
    # Omitted: [2, 0, flight.takeoff_height].
    anchor: Optional[list] = None
    speed: float = 1.0           # linear and flee speed (m/s)
    direction: list = field(default_factory=lambda: [1.0, 0.0, 0.0])   # linear, world frame
    radius: float = 1.5          # circular orbit radius (m)
    omega: float = 1.0           # circular angular rate (rad/s)
    lookahead: float = 0.5       # flee setpoint lead time (s)

    def __post_init__(self) -> None:
        self.type = MOTION_ALIASES.get(self.type.lower(), self.type.lower())
        if self.type not in SCRIPTED_MOTIONS:
            raise ValueError(
                f"Config key 'opponent.scripted.type' must be one of {list(SCRIPTED_MOTIONS)}; "
                f"got '{self.type}'.")
        direction = _vec3(self.direction, 'opponent.scripted.direction')
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            raise ValueError("Config key 'opponent.scripted.direction' must be nonzero.")
        self.direction = direction / norm
        if self.anchor is not None:
            self.anchor = _vec3(self.anchor, 'opponent.scripted.anchor')


@dataclass
class OpponentConfig:
    """`opponent`: how the other drone is driven."""

    mode: str = 'scripted'       # policy | cf | scripted
    artifact_dir: str = ''       # mode=policy: artifact of the OPPOSING role
    scripted: ScriptedMotionConfig = field(default_factory=ScriptedMotionConfig)

    def __post_init__(self) -> None:
        self.mode = self.mode.lower()
        if self.mode not in OPPONENT_MODES:
            raise ValueError(
                f"Config key 'opponent.mode' must be one of {list(OPPONENT_MODES)}; got '{self.mode}'.")
        if self.mode == 'policy' and not self.artifact_dir:
            raise ValueError("Config key 'opponent.artifact_dir' is required when opponent.mode is 'policy'.")
        if isinstance(self.scripted, dict) or self.scripted is None:
            self.scripted = _section(ScriptedMotionConfig, self.scripted, 'opponent.scripted')


@dataclass
class ArenaConfig:
    """`arena`: where the trained arena (metadata arena_half_xy / z_min / z_max) sits in the mocap world."""

    # Arena centre in mocap world coordinates; the policy observes positions relative to it.
    origin: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Scripted opponent setpoints are kept this far inside the walls (m).
    scripted_margin: float = 0.3

    def __post_init__(self) -> None:
        self.origin = _vec3(self.origin, 'arena.origin')


@dataclass
class EpisodeConfig:
    """`episode`: when a run ends. The first four mirror the training episode endings."""

    stop_on_capture: bool = True
    capture_radius: float = 0.0          # m; 0 = the trained radius (metadata.capture_radius)
    # Training confines only the evader: its exit ends the episode, the pursuer may leave freely.
    stop_on_evader_exit: bool = True
    # Stop after metadata.max_episode_length control steps (evader win in training).
    stop_on_timeout: bool = True
    state_timeout: float = 0.5           # s; stop if a policy-relevant state is this stale
    min_altitude: float = 0.15           # m; stop if a real drone is lower


@dataclass
class FlightConfig:
    """`flight`: take-off, landing and the control loop."""

    takeoff_height: float = 1.0          # m, both drones
    takeoff_duration: float = 3.0        # s
    land_duration: float = 3.0           # s
    control_dt: float = 0.0              # s; 0 = the training control period (metadata.ctbr.dt)
    # First-order low-pass on the commanded body rates (time constant, s; 0 = off).
    # The policy was trained with the rate loop stepped at the 50 Hz control
    # period; the CrazySim / real firmware rate loop runs ~10x faster and tracks
    # raw setpoints far more crisply, which can turn aggressive rate commands
    # into visible roll oscillation. Smoothing the setpoint narrows the
    # closed-loop rate bandwidth back toward the trained response. Try
    # 0.02-0.05 and increase until the oscillation stops; decrease if the chase
    # gets sluggish. Thrust is never filtered.
    command_rate_lpf_tau: float = 0.0


@dataclass
class LoggingConfig:
    """`logging`: console output and ROS 2 TF."""

    log_commands: bool = True            # one line per tick: altitude, distance, CTBR command
    # Dump raw states + named observation components (throttled to debug_period_s)
    # to compare the deploy observation against the sim env for a matched state.
    debug_obs: bool = False
    debug_period_s: float = 0.5
    # Publish mocap poses on /tf (parent frame: mocap.world_frame).
    publish_tf: bool = False


@dataclass
class ControllerConfig:
    """The controller sections of the YAML (everything except the drone, mocap and safety sections)."""

    policy: PolicyConfig
    opponent: OpponentConfig
    arena: ArenaConfig
    episode: EpisodeConfig
    flight: FlightConfig
    logging: LoggingConfig

    SECTIONS = {'policy': PolicyConfig, 'opponent': OpponentConfig, 'arena': ArenaConfig,
                'episode': EpisodeConfig, 'flight': FlightConfig, 'logging': LoggingConfig}
    OTHER_SECTIONS = ('pursuer', 'evader', 'mocap', 'safety')

    @classmethod
    def from_yaml_dict(cls, config: dict, path: str) -> 'ControllerConfig':
        if 'controller' in config or 'artifact_dir' in config:
            raise ValueError(
                f"{path} uses the old layout (top-level 'artifact_dir' + a 'controller' section). "
                "Move its keys into the policy / opponent / arena / episode / flight / logging "
                "sections; see config_pursuit_evasion.yaml.")
        unknown = sorted(set(config) - set(cls.SECTIONS) - set(cls.OTHER_SECTIONS))
        if unknown:
            raise ValueError(f'Unknown top-level section(s) {unknown} in {path}.')
        sections = {name: _section(sec, config.get(name), name) for name, sec in cls.SECTIONS.items()}
        if not sections['policy'].artifact_dir:
            raise ValueError(f"Config must set 'policy.artifact_dir' in {path}.")
        return cls(**sections)


def _read_yaml(path: str) -> dict:
    import yaml

    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


# ===========================================================================
# Learned agent
# ===========================================================================
def _load_policy(artifact_dir: str):
    """Load a TorchScript PursuitEvasion policy plus its metadata."""
    ts_path, meta_path = pec.artifact_paths(artifact_dir)
    if not (os.path.isfile(ts_path) and os.path.isfile(meta_path)):
        raise FileNotFoundError(
            f'Missing artifact(s) under {artifact_dir}: expected '
            f'{pec.POLICY_TS_FILENAME} and {pec.METADATA_FILENAME}. '
            f'Run export_pursuit_evasion_policy.py first.'
        )
    metadata = pec.load_metadata(meta_path)
    # Deploy on the CPU even when a GPU is present: the actor is a tiny MLP, so
    # host<->device transfer would only add latency and jitter to the control
    # loop, and deterministic timing is what keeps the body-rate commands
    # matched to training.
    device = torch.device('cpu')
    # Batch-1 inference: extra intra-op threads only contend with the sim/radio (20 threads -> ~20 ms/tick).
    torch.set_num_threads(1)
    policy = torch.jit.load(ts_path, map_location=device).eval()
    logger.info(
        '[pe] Loaded %s %s policy (obs_dim=%d) from %s',
        metadata.algo, metadata.role, metadata.obs.obs_dim, ts_path,
    )
    return metadata, policy, device


class PolicyAgent:
    """A drone flown by an exported PursuitEvasion policy.

    Bundles the TorchScript actor and its metadata with the per-run state its
    observation depends on (previous action, optional ESN / opponent adapter)
    and the optional body-rate low-pass on its commands. Used for the
    controller's own drone and, in ``opponent.mode: policy``, for the opponent.
    """

    def __init__(self, artifact_dir: str, rate_lpf_tau: float = 0.0,
                 control_dt: Optional[float] = None) -> None:
        artifact_dir = os.path.abspath(os.path.expanduser(artifact_dir))
        self.metadata, self.policy, self.device = _load_policy(artifact_dir)
        self.role = self.metadata.role
        # Deploy control period; defaults to the training one.
        self.control_dt = control_dt or self.metadata.ctbr.dt
        obs_cfg = self.metadata.obs
        # Fresh per run, like a training episode start: ESN W_out=0, adapter readout at its learned mean.
        self.esn = pec.load_esn(artifact_dir, self.device) if obs_cfg.use_esn else None
        self.adapter = (pec.load_opponent_adapter(artifact_dir, self.metadata.adapter, self.device)
                        if obs_cfg.use_adapter else None)
        if self.adapter is not None:
            a = self.metadata.adapter
            logger.info(
                '[pe] %s opponent adapter: %s/%s H=%d K=%d, RLS %s (lambda=%.3f, delta=%g)',
                self.role, a.kind, a.target, a.horizon, a.history,
                'frozen' if a.freeze else 'online', a.forgetting, a.delta,
            )
        self.previous_action = torch.zeros(obs_cfg.action_dim, dtype=torch.float32, device=self.device)
        self.rate_lpf_tau = rate_lpf_tau
        self.rate_lpf_alpha = (1.0 - math.exp(-self.control_dt / rate_lpf_tau)
                               if rate_lpf_tau > 0.0 else 1.0)
        self._filtered_rates: Optional[torch.Tensor] = None

    def act(
        self, own: pec.DroneState, opponent: pec.DroneState, arena_origin: np.ndarray,
    ) -> 'tuple[torch.Tensor, pec.CTBRCommand, torch.Tensor]':
        """Run the policy once (advancing the ESN / adapter by one control step).

        Returns the tanh-scaled action, the CTBR command, and the exact
        observation vector fed to the network.
        """
        def _t(value) -> torch.Tensor:
            return torch.as_tensor(value, dtype=torch.float32, device=self.device)

        # The env observes positions relative to the arena origin.
        self_pos = _t(own.pos - arena_origin)
        self_quat = _t(own.quat_wxyz)
        self_vel = _t(own.lin_vel)
        opp_pos = _t(opponent.pos - arena_origin)
        opp_vel = _t(opponent.lin_vel)
        obs = pec.build_observation(
            self.metadata.obs,
            self_pos=self_pos,
            self_quat_wxyz=self_quat,
            self_lin_vel_world=self_vel,
            self_ang_vel_body=_t(own.ang_vel),
            opponent_pos=opp_pos,
            opponent_lin_vel_world=opp_vel,
            previous_action=self.previous_action,
        )
        if self.esn is not None:
            target = pec.build_esn_target(self_pos, self_quat, self_vel, opp_pos, opp_vel)
            obs = pec.esn_augment(self.esn, obs, target)
        if self.adapter is not None:
            # The adapter is pursuer-only, so "self" is the pursuer here. Its
            # feature scale uses the training control period, not the deploy one.
            z = pec.build_adapter_input(self_pos, self_vel, opp_pos, opp_vel)
            features = pec.adapter_step(self.adapter, self.metadata.adapter, z, self_quat,
                                        self.metadata.ctbr.dt)
            obs = torch.cat([obs.reshape(-1), features])
        obs = obs.reshape(1, self.metadata.obs.obs_dim)

        with torch.no_grad():
            raw_action = self.policy(obs)
        return (
            torch.tanh(raw_action).squeeze(0),
            pec.decode_action_to_ctbr(raw_action, self.metadata.ctbr),
            obs.reshape(-1),
        )

    def send(self, drone, action: torch.Tensor, command: pec.CTBRCommand) -> pec.CTBRCommand:
        """Commit ``action`` as the next observation's previous action and send ``command`` (smoothed)."""
        # previous_action feeds the observation, so store the raw (unfiltered)
        # policy action; the low-pass only shapes the setpoint sent to the firmware.
        self.previous_action = action.detach()
        command = self._smooth(command)
        drone.send_ctbr(command)
        return command

    def _smooth(self, command: pec.CTBRCommand) -> pec.CTBRCommand:
        """Low-pass the commanded body rates in place (no-op when command_rate_lpf_tau is 0)."""
        if self.rate_lpf_alpha >= 1.0:
            return command
        rates = command.body_rate_deg
        prev = self._filtered_rates if self._filtered_rates is not None else rates.clone()
        filt = self.rate_lpf_alpha * rates + (1.0 - self.rate_lpf_alpha) * prev
        self._filtered_rates = filt.detach()
        command.body_rate_deg = filt
        return command


# ===========================================================================
# Debug output
# ===========================================================================
def _quat_to_rpy_deg(quat_wxyz: np.ndarray) -> 'tuple[float, float, float]':
    """Roll/pitch/yaw in degrees from an ``(w, x, y, z)`` quaternion (for logs)."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(np.degrees([roll, pitch, yaw]))


def _log_obs_breakdown(
    metadata: pec.PursuitEvasionMetadata,
    own: pec.DroneState,
    opponent: pec.DroneState,
    arena_origin: np.ndarray,
    obs: torch.Tensor,
) -> None:
    """Dump the raw states and the named observation components.

    Use ``logging.debug_obs: true`` and hover both drones (opponent static)
    to check the deploy observation against what the sim env produces for the
    same physical state. The single most important line is ``rel_pos(body)``:
    if the two drones are physically metres apart but this reads ~0 (or a wrong
    offset), the two Crazyflies' position estimates are NOT in a shared world
    frame (e.g. each CrazySim EKF is relative to its own spawn), which makes the
    relative-position observation garbage and the policy chase a phantom.
    """
    o = obs.detach().cpu().numpy().reshape(-1)
    # Component offsets mirror pec.build_observation's concat order.
    r = pec.rotation_obs_dim(metadata.obs.rot_repr)
    self_pos = o[0:3]
    rot = o[3:3 + r]
    lin_vel_b = o[3 + r:6 + r]
    ang_vel_b = o[6 + r:9 + r]
    rel_pos_b = o[9 + r:12 + r]
    rel_vel_b = o[12 + r:15 + r]
    prev_act = o[15 + r:19 + r] if metadata.obs.use_previous_action else np.zeros(4)
    # Estimator outputs follow the base observation: [ESN(6)] [adapter(3)].
    extra = o[metadata.obs.base_obs_dim():]
    extra_lines = ''
    if metadata.obs.use_esn:
        extra_lines += '\n  esn_pred      =' + np.array2string(
            extra[:metadata.obs.esn_dim], precision=2, sign='+', suppress_small=True)
        extra = extra[metadata.obs.esn_dim:]
    if metadata.obs.use_adapter:
        label = 'opp_acc/10(body)' if metadata.adapter.target == 'dvel' else 'opp_disp(body)'
        extra_lines += f'\n  {label}=' + np.array2string(
            extra[:metadata.obs.adapter_dim], precision=3, sign='+', suppress_small=True)

    own_rpy = _quat_to_rpy_deg(own.quat_wxyz)
    world_rel = (opponent.pos - arena_origin) - (own.pos - arena_origin)

    logger.info(
        '[pe:dbg] --- raw states (world frame) ---\n'
        '  own   pos=[%+.2f %+.2f %+.2f] rpy(deg)=[%+.0f %+.0f %+.0f] '
        'vel=[%+.2f %+.2f %+.2f] gyro(rad/s)=[%+.2f %+.2f %+.2f]\n'
        '  opp   pos=[%+.2f %+.2f %+.2f] vel=[%+.2f %+.2f %+.2f]\n'
        '  rel_pos(world)=[%+.2f %+.2f %+.2f] |rel|=%.2fm\n'
        '[pe:dbg] --- observation components (network input) ---\n'
        '  self_pos      =[%+.2f %+.2f %+.2f]\n'
        '  rot(%s)%s=%s\n'
        '  lin_vel(body) =[%+.2f %+.2f %+.2f]\n'
        '  ang_vel(body) =[%+.2f %+.2f %+.2f]\n'
        '  rel_%s(body) =[%+.2f %+.2f %+.2f]\n'
        '  rel_vel(body) =[%+.2f %+.2f %+.2f]\n'
        '  prev_action   =[%+.2f %+.2f %+.2f %+.2f]%s',
        own.pos[0], own.pos[1], own.pos[2], own_rpy[0], own_rpy[1], own_rpy[2],
        own.lin_vel[0], own.lin_vel[1], own.lin_vel[2],
        own.ang_vel[0], own.ang_vel[1], own.ang_vel[2],
        opponent.pos[0], opponent.pos[1], opponent.pos[2],
        opponent.lin_vel[0], opponent.lin_vel[1], opponent.lin_vel[2],
        world_rel[0], world_rel[1], world_rel[2], float(np.linalg.norm(world_rel)),
        self_pos[0], self_pos[1], self_pos[2],
        metadata.obs.rot_repr, ' ' * max(0, 9 - len(metadata.obs.rot_repr)),
        np.array2string(rot, precision=2, sign='+', suppress_small=True),
        lin_vel_b[0], lin_vel_b[1], lin_vel_b[2],
        ang_vel_b[0], ang_vel_b[1], ang_vel_b[2],
        'pos' if metadata.obs.use_rel_dist else 'hdg',
        rel_pos_b[0], rel_pos_b[1], rel_pos_b[2],
        rel_vel_b[0], rel_vel_b[1], rel_vel_b[2],
        prev_act[0], prev_act[1], prev_act[2], prev_act[3],
        extra_lines,
    )


class _FixedRateClock:
    """Fixed-rate scheduling anchored to an absolute deadline.

    The policy was trained at a fixed control period; a plain sleep(dt) would
    stretch it by the compute time and jitter, holding each body-rate command
    longer than in training.
    """

    def __init__(self, dt: float) -> None:
        self.dt = dt
        self.next_tick = time.perf_counter()
        self.ticks = 0
        self.overruns = 0

    def wait(self) -> None:
        self.next_tick += self.dt
        self.ticks += 1
        slack = self.next_tick - time.perf_counter()
        if slack > 0.0:
            time.sleep(slack)
            return
        self.overruns += 1
        self.next_tick = time.perf_counter()
        if self.overruns % 50 == 1:
            logger.warning(
                '[pe] Control loop overran its %.1f ms budget by %.1f ms (%d/%d ticks late); '
                'the policy is running slower than its trained %.1f Hz.',
                self.dt * 1e3, -slack * 1e3, self.overruns, self.ticks, 1.0 / self.dt,
            )


# ===========================================================================
# Controller
# ===========================================================================
class PursuitEvasionController:
    def __init__(self, args: argparse.Namespace) -> None:
        self.config_path = os.path.abspath(os.path.expanduser(args.config))
        config = _read_yaml(self.config_path)
        self.cfg = cfg = ControllerConfig.from_yaml_dict(config, self.config_path)

        # -- policies --------------------------------------------------------
        lpf_tau = cfg.flight.command_rate_lpf_tau
        self.agent = PolicyAgent(cfg.policy.artifact_dir, lpf_tau, cfg.flight.control_dt)
        self.metadata = self.agent.metadata
        self.control_dt = self.agent.control_dt
        self.role = self.agent.role
        self.opponent_role = pec.opponent_role(self.role)

        self.opponent_mode = cfg.opponent.mode
        self.opponent_agent: Optional[PolicyAgent] = None
        if self.opponent_mode == 'policy':
            self.opponent_agent = PolicyAgent(cfg.opponent.artifact_dir, lpf_tau, self.control_dt)
            if self.opponent_agent.role != self.opponent_role:
                raise ValueError(
                    f"Opponent artifact declares role '{self.opponent_agent.role}' but the agent is "
                    f"'{self.role}', so the opponent must be '{self.opponent_role}'.")

        # -- episode / arena ---------------------------------------------------
        self.arena_origin = cfg.arena.origin
        self.capture_radius = cfg.episode.capture_radius or self.metadata.capture_radius
        # Training truncates at max_episode_length steps (an evader win).
        self.episode_duration = self.metadata.max_episode_length * self.metadata.ctbr.dt
        self.land_duration = cfg.flight.land_duration
        self.motion = cfg.opponent.scripted
        anchor = self.motion.anchor if self.motion.anchor is not None else [2.0, 0.0, cfg.flight.takeoff_height]
        self.motion_anchor = np.array(anchor, dtype=np.float64)

        # -- drones / mocap ----------------------------------------------------
        self.self_config = pec.load_drone_config_from_yaml(self.config_path, self.role)
        self.opponent_drone_config = pec.load_drone_config_from_yaml(self.config_path, self.opponent_role)
        self.mocap_config = pec.load_mocap_config_from_yaml(self.config_path)
        self._rigid_ids = {
            'pursuer': self.mocap_config.pursuer_rigid_body_id,
            'evader': self.mocap_config.evader_rigid_body_id,
        }

        # Safety layer (SAFETY_PLAN.md): per-tick checks over both drones, graded responses, pre-flight checks.
        self.safety_cfg = SafetyConfig.from_dict(config.get('safety'))
        self.safety = SafetyMonitor(self.safety_cfg, Arena(
            origin=self.arena_origin, half_xy=self.metadata.arena_half_xy,
            z_min=self.metadata.arena_z_min, z_max=self.metadata.arena_z_max))
        self._fault = None                       # latched safety Verdict that ended the run
        self._inject = dict(self.safety_cfg.inject or {})
        self._inject_kind = str(self._inject.get('kind', 'none')).lower()
        self._frozen_opponent = None

        self.drone_pose_pub = DronePosePublisher(world_frame=self.mocap_config.world_frame)
        self.own = CrazyflieDrone(self.self_config, pose_publisher=self.drone_pose_pub)
        opponent_cls = ScriptedDrone if self.opponent_mode == 'scripted' else CrazyflieDrone
        self.opponent = opponent_cls(self.opponent_drone_config, pose_publisher=self.drone_pose_pub)

        self.mocap_tf_publisher: Optional[MocapTfPublisher] = None
        self.mocap_receiver: Optional[MocapReceiver] = None

        self._motion_start_time = 0.0
        self._position_setpoint = self.motion_anchor.copy()
        self._last_debug_time = 0.0

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:
        cflib_crtp = importlib.import_module('cflib.crtp')
        cflib_crtp.init_drivers()

        try:
            self._connect_and_setup()
            self._setup_mocap()
            self._arm_and_takeoff()
            self._initialize_motion_anchor()
            self._motion_start_time = time.time()
            self._position_setpoint = self.motion_anchor.copy()
            self._log_run_start()

            clock = _FixedRateClock(self.control_dt)
            while self.own.connected and self.opponent.connected:
                own_state = self.own.get_state()
                opponent_state = self.opponent.get_state()
                if self.opponent_mode == 'scripted':
                    self._send_scripted_setpoint(own_state)
                if own_state is None or opponent_state is None:
                    time.sleep(self.control_dt)
                    continue

                now = time.time()
                own_state, opponent_state = self._apply_injection(own_state, opponent_state, now)
                if self._episode_over(own_state, opponent_state, now):
                    break
                if not self._step_policies(own_state, opponent_state, now):
                    break

                self.own.publish_pose()
                if self.opponent_mode != 'scripted':
                    self.opponent.publish_pose()
                clock.wait()
        except KeyboardInterrupt:
            logger.info('[pe] Stopping.')
        finally:
            self.shutdown()

    def _log_run_start(self) -> None:
        logger.info(
            '[pe] Running %s policy at %.1f Hz vs %s (%s). Ctrl+C to stop.',
            self.role, 1.0 / self.control_dt, self.opponent_role,
            self.opponent_mode if self.opponent_mode != 'scripted' else f'scripted:{self.motion.type}',
        )
        if self.agent.rate_lpf_alpha < 1.0:
            logger.info(
                '[pe] Body-rate command low-pass ON (tau=%.3fs, alpha=%.2f at %.1f Hz): smoothing the '
                'setpoint sent to the firmware rate loop to match the trained 50 Hz rate response.',
                self.agent.rate_lpf_tau, self.agent.rate_lpf_alpha, 1.0 / self.control_dt,
            )

    def _episode_over(self, own_state: pec.DroneState, opponent_state: pec.DroneState, now: float) -> bool:
        """Check the run-ending conditions (stale state, crash, evader exit, timeout, capture); log the first hit."""
        ep = self.cfg.episode
        real_opponent = self.opponent_mode != 'scripted'
        # SAFETY_PLAN B2: the opponent's state can go stale too (policy / cf mode).
        for role, state, monitored in ((self.role, own_state, True),
                                       (self.opponent_role, opponent_state, real_opponent)):
            if monitored and now - state.stamp > ep.state_timeout:
                logger.warning('[pe] %s state timed out (%.2f s); stopping.', role, now - state.stamp)
                return True
        for role, state, monitored in ((self.role, own_state, True),
                                       (self.opponent_role, opponent_state, real_opponent)):
            if monitored and state.pos[2] < ep.min_altitude:
                logger.warning('[pe] %s below min altitude (%.2f m); stopping.', role, state.pos[2])
                return True
        evader_state = own_state if self.role == 'evader' else opponent_state
        if ep.stop_on_evader_exit and self._outside_arena(evader_state.pos):
            logger.warning(
                '[pe] evader left the trained arena at [%.2f, %.2f, %.2f]; '
                'stopping (episode end in training).', *evader_state.pos)
            return True
        if (ep.stop_on_timeout and self.episode_duration > 0.0
                and now - self._motion_start_time >= self.episode_duration):
            logger.info('[pe] Training episode length reached (%.1f s): evader survived; stopping.',
                        self.episode_duration)
            return True
        distance = float(np.linalg.norm(opponent_state.pos - own_state.pos))
        if ep.stop_on_capture and distance <= self.capture_radius:
            logger.info('[pe] Capture radius reached (%.2f m <= %.2f m); stopping.',
                        distance, self.capture_radius)
            return True
        return False

    def _step_policies(self, own_state: pec.DroneState, opponent_state: pec.DroneState, now: float) -> bool:
        """Run the policies, check safety, send the commands. Returns False on a safety fault."""
        action, command, obs = self.agent.act(own_state, opponent_state, self.arena_origin)
        if self._inject_kind == 'nan_action' and self._inject_active(now):
            action = action * float('nan')
        opp_action = opp_command = None
        if self.opponent_agent is not None:
            opp_action, opp_command, _ = self.opponent_agent.act(
                opponent_state, own_state, self.arena_origin)

        action_finite = bool(torch.isfinite(action).all() and torch.isfinite(obs).all()
                             and (opp_action is None or torch.isfinite(opp_action).all()))
        verdict = self.safety.check(
            self._monitored_states(own_state, opponent_state), now,
            action_finite=action_finite, mocap=self._mocap_health())
        if verdict.level >= Level.HOLD:
            self._fault = verdict
            logger.error('[pe] SAFETY %s: %s', verdict.level.name, ' | '.join(verdict.reasons))
            return False

        command = self.agent.send(self.own, action, command)
        log = self.cfg.logging
        if log.debug_obs and now - self._last_debug_time >= log.debug_period_s:
            _log_obs_breakdown(self.metadata, own_state, opponent_state, self.arena_origin, obs)
            self._last_debug_time = now
        if self.opponent_agent is not None:
            self.opponent_agent.send(self.opponent, opp_action, opp_command)
        if log.log_commands:
            rates = command.body_rate_deg.detach().cpu().numpy().reshape(-1)
            logger.info(
                '[pe] alt=%.2fm dist=%.2fm rates(deg/s)=[%+.0f,%+.0f,%+.0f] thrust_pwm=%.0f',
                own_state.pos[2], float(np.linalg.norm(opponent_state.pos - own_state.pos)),
                rates[0], rates[1], rates[2], float(command.thrust_pwm.item()),
            )
        return True

    # -- scripted opponent motion ------------------------------------------
    def _send_scripted_setpoint(self, own_state: Optional[pec.DroneState]) -> None:
        self._position_setpoint = self._scripted_setpoint(own_state)
        self.opponent.send_position_setpoint(
            self._position_setpoint[0], self._position_setpoint[1], self._position_setpoint[2],
            yaw_deg=self.motion.yaw_deg,
        )

    def _scripted_setpoint(self, own_state: Optional[pec.DroneState]) -> np.ndarray:
        """Position setpoint for the scripted opponent, clamped to the arena."""
        elapsed = time.time() - self._motion_start_time
        anchor = self.motion_anchor
        m = self.motion

        if m.type == 'hover':
            target = anchor.copy()
        elif m.type == 'linear':
            target = anchor + m.direction * (m.speed * elapsed)
        elif m.type == 'circular':
            angle = m.omega * elapsed
            target = anchor + np.array([m.radius * np.cos(angle), m.radius * np.sin(angle), 0.0])
        elif m.type == 'pure_pursuit':
            # Fly at the agent's current position (no lead).
            target = (own_state.pos.copy() if own_state is not None
                      else self._position_setpoint.copy())
        else:  # flee
            # Horizontal escape at the held anchor altitude, as in training:
            # a 3D flee dives whenever the pursuer is above.
            current = self._position_setpoint
            if own_state is None:
                target = current.copy()
            else:
                away = current - own_state.pos
                away[2] = 0.0
                norm = float(np.linalg.norm(away))
                away = away / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
                target = current + away * (m.speed * m.lookahead)
            target[2] = anchor[2]

        return self._clamp_to_arena(target)

    def _clamp_to_arena(self, position: np.ndarray) -> np.ndarray:
        """Clamp a world-frame setpoint into the trained arena volume."""
        margin = self.cfg.arena.scripted_margin
        half = self.metadata.arena_half_xy - margin
        local = position - self.arena_origin
        local[0] = float(np.clip(local[0], -half, half))
        local[1] = float(np.clip(local[1], -half, half))
        local[2] = float(np.clip(local[2], self.metadata.arena_z_min + margin,
                                 self.metadata.arena_z_max - margin))
        return local + self.arena_origin

    def _outside_arena(self, position: np.ndarray) -> bool:
        local = position - self.arena_origin
        return bool(
            abs(local[0]) > self.metadata.arena_half_xy
            or abs(local[1]) > self.metadata.arena_half_xy
            or local[2] < self.metadata.arena_z_min
            or local[2] > self.metadata.arena_z_max
        )

    # -- mocap / connection / take-off ---------------------------------------
    def _setup_mocap(self) -> None:
        if not self.mocap_config.enabled:
            return
        if self.cfg.logging.publish_tf:
            self.mocap_tf_publisher = MocapTfPublisher(world_frame=self.mocap_config.world_frame)
        self.mocap_receiver = MocapReceiver(
            self.mocap_config, tf_publisher=self.mocap_tf_publisher,
            gate=MocapGate(self.safety_cfg))

        rigid_ids = self._rigid_ids
        self.mocap_receiver.register(rigid_ids[self.role], self.own, frame_id=self.role)
        if self.opponent_mode != 'scripted':
            self.mocap_receiver.register(
                rigid_ids[self.opponent_role], self.opponent, frame_id=self.opponent_role)
        self.mocap_receiver.start()

    def _connect_and_setup(self) -> None:
        self.own.connect()
        self.own.setup()
        self.opponent.connect()
        self.opponent.setup()

    def _monitored_states(self, own_state=None, opponent_state=None) -> dict:
        """States the safety layer checks: the agent, plus the opponent when it is a real drone."""
        states = {self.role: own_state if own_state is not None else self.own.get_state()}
        if self.opponent_mode != 'scripted':
            states[self.opponent_role] = (opponent_state if opponent_state is not None
                                          else self.opponent.get_state())
        return states

    def _mocap_health(self) -> Optional[dict]:
        if self.mocap_receiver is None:
            return None
        roles = [self.role] + ([self.opponent_role] if self.opponent_mode != 'scripted' else [])
        return {r: self.mocap_receiver.health(self._rigid_ids[r]) for r in roles}

    def _arm_and_takeoff(self) -> None:
        status = {'own': self.own.arm() or {}, 'opp': self.opponent.arm() or {}}
        # SAFETY_PLAN B4: with strict_preflight a failed mocap alignment / estimator convergence or a bad start
        # position blocks the take-off instead of only logging a warning.
        problems = []
        for key, drone, tracked in (('own', self.own, True),
                                    ('opp', self.opponent, self.opponent_mode != 'scripted')):
            if not tracked:
                continue
            if self.mocap_config.enabled and not status[key].get('mocap_aligned', False):
                problems.append(f'{drone.name}: mocap orientation not aligned')
            if not status[key].get('estimator_converged', False):
                problems.append(f'{drone.name}: estimator not converged')
        if self.safety_cfg.enabled:
            problems += self.safety.preflight(self._monitored_states(), time.time())
        if problems:
            if self.safety_cfg.strict_preflight:
                raise RuntimeError('[pe] Pre-flight checks failed, not taking off: ' + '; '.join(problems))
            logger.warning('[pe] Pre-flight problems (strict_preflight off, continuing): %s', '; '.join(problems))
        flight = self.cfg.flight
        self.own.takeoff(flight.takeoff_height, duration=flight.takeoff_duration)
        self.opponent.takeoff(flight.takeoff_height, duration=flight.takeoff_duration)

    def _initialize_motion_anchor(self) -> None:
        takeoff_height = self.cfg.flight.takeoff_height
        if self.opponent_mode == 'scripted':
            self.motion_anchor[2] = takeoff_height
            return
        deadline = time.time() + 2.0
        while time.time() < deadline:
            state = self.opponent.get_state()
            if state is not None:
                self.motion_anchor = state.pos.copy()
                break
            time.sleep(0.05)
        self.motion_anchor[2] = max(self.motion_anchor[2], takeoff_height)

    # -- fault injection / shutdown --------------------------------------------
    def _apply_injection(self, own_state, opponent_state, now):
        """Simulator fault injection (safety.inject): corrupt the states after inject.at_s seconds."""
        kind = self._inject_kind
        if kind in ('none', 'nan_action') or not self._inject_active(now):
            return own_state, opponent_state
        if kind == 'stale_opponent':
            if self._frozen_opponent is None:
                self._frozen_opponent = opponent_state
                logger.warning('[pe] INJECT: freezing the %s state', self.opponent_role)
            return own_state, self._frozen_opponent
        if kind == 'position_jump':
            return dataclasses.replace(own_state, pos=own_state.pos + np.array([3.0, 0.0, 0.0])), opponent_state
        if kind == 'tilt':
            return dataclasses.replace(own_state, quat_wxyz=np.array([0.7071068, 0.7071068, 0.0, 0.0])), \
                opponent_state
        raise ValueError(f'Unknown safety.inject.kind {kind!r}')

    def _inject_active(self, now: float) -> bool:
        return self._inject_kind != 'none' and now - self._motion_start_time >= float(self._inject.get('at_s', 5.0))

    def _hold(self, drones, duration: float) -> None:
        """Hold each real drone at its current position (and yaw) for ``duration`` seconds."""
        targets = []
        for d in drones:
            s = d.get_state() if isinstance(d, CrazyflieDrone) else None
            if s is not None and np.all(np.isfinite(s.pos)):
                targets.append((d, s.pos.copy(), _quat_to_rpy_deg(s.quat_wxyz)[2]))
        if targets:
            logger.warning('[pe] HOLD for %.1f s at the current positions.', duration)
        end = time.time() + duration
        while targets and time.time() < end:
            for d, p, yaw in targets:
                try:
                    d.send_position_setpoint(p[0], p[1], p[2], yaw_deg=yaw)
                except Exception:
                    pass
            time.sleep(self.control_dt)

    def shutdown(self) -> None:
        drones = [self.own, self.opponent]
        fault = self._fault
        if fault is not None and fault.level >= Level.EMERGENCY:
            for role in fault.emergency_roles:
                d = self.own if role == self.role else self.opponent
                try:
                    d.emergency_stop()
                except Exception:
                    logger.exception('[pe] emergency stop failed for %s', role)
                drones = [x for x in drones if x is not d]
        if fault is not None and fault.level == Level.HOLD:
            self._hold(drones, self.safety_cfg.hold_s)
        # SAFETY_PLAN B1: start every landing first, then wait once. Landing the drones one after the other left
        # the second one without setpoints for the whole first landing (> 2 s: the firmware cuts its motors).
        landing = []
        for d in drones:
            try:
                if d.land_async(self.land_duration):
                    landing.append(d)
            except Exception:
                logger.exception('[pe] landing failed for %s', getattr(d, 'name', d))
        if landing:
            time.sleep(self.land_duration + 1.0)
        for drone in (self.own, self.opponent):
            for action in ('disarm', 'disconnect'):
                try:
                    getattr(drone, action)()
                except Exception:
                    pass

        if self.mocap_receiver is not None:
            self.mocap_receiver.stop()
            self.mocap_receiver = None
        if self.mocap_tf_publisher is not None:
            self.mocap_tf_publisher.close()
            self.mocap_tf_publisher = None


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description='Run a PursuitEvasion policy using the drone/mocap helpers.'
    )
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG_PATH,
        help=f'Path to the YAML configuration file (default: {DEFAULT_CONFIG_PATH}).',
    )
    args = parser.parse_args(argv)
    PursuitEvasionController(argparse.Namespace(config=args.config)).run()


if __name__ == '__main__':
    main()
