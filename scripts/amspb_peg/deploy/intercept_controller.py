"""Run AMSPB pursuer deploy artifacts with the existing intercept runtime loop.

This module reuses scripts/deploy/intercept_controller_v2.py for radio/mocap/
control-loop infrastructure, but swaps policy observation construction so it
matches Pursuit observations from scripts/amspb_peg/train_pursuer.py.
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch

import scripts.deploy.intercept_common as ic

# NOTE: scripts.deploy.intercept_controller_v2 (and its cflib/mocap/ROS deps) is
# imported lazily inside main() so that the pure-torch observation/command logic
# in this module can be imported and unit-tested outside the Crazyswarm runtime.


class PursuitObservationBuilder:
    """Reconstructs Pursuit observations from live pursuer/evader states."""

    def __init__(self, pursuit_obs_cfg: dict, device: torch.device):
        self.device = device

        self.time_encoding = bool(pursuit_obs_cfg.get("time_encoding", True))
        self.time_encoding_dim = int(pursuit_obs_cfg.get("time_encoding_dim", 1))
        self.include_distances = bool(pursuit_obs_cfg.get("include_distances", True))
        self.include_closing_velocity = bool(pursuit_obs_cfg.get("include_closing_velocity", True))
        self.include_heading_to_target = bool(pursuit_obs_cfg.get("include_heading_to_target", False))
        self.include_last_action = bool(pursuit_obs_cfg.get("include_last_action", False))
        self.include_effort = bool(pursuit_obs_cfg.get("include_effort", False))
        self.use_body_rates = bool(pursuit_obs_cfg.get("use_body_rates", True))

        self.prev_traj_steps = int(pursuit_obs_cfg.get("prev_traj_steps", 0))
        self.prev_evader_traj_steps = int(pursuit_obs_cfg.get("prev_evader_traj_steps", 0))
        self.prev_evader_steps_fix = int(
            pursuit_obs_cfg.get(
                "prev_evader_steps_fix",
                max(4, self.prev_evader_traj_steps),
            )
        )
        self.max_episode_length = max(1, int(pursuit_obs_cfg.get("max_episode_length", 600)))

        self.k_p = torch.as_tensor(pursuit_obs_cfg.get("k_p", [1.0, 1.0, 1.0]), device=device, dtype=torch.float32).view(1, 3)
        self.k_v = torch.as_tensor(pursuit_obs_cfg.get("k_v", [1.0, 1.0, 1.0]), device=device, dtype=torch.float32).view(1, 3)
        self.k_omega = torch.as_tensor(pursuit_obs_cfg.get("k_omega", [1.0, 1.0, 1.0]), device=device, dtype=torch.float32).view(1, 3)
        self.k_rp = torch.as_tensor(pursuit_obs_cfg.get("k_rp", [1.0, 1.0, 1.0]), device=device, dtype=torch.float32).view(1, 3)
        self.arena_center = torch.as_tensor(pursuit_obs_cfg.get("arena_center", [0.0, 0.0, 0.0]), device=device, dtype=torch.float32).view(1, 3)

        self._step = 0
        self._rel_hist: deque[torch.Tensor] = deque(maxlen=max(1, self.prev_evader_steps_fix))
        self._pos_hist: deque[torch.Tensor] = deque(maxlen=max(1, self.prev_traj_steps))

    def _seed_histories(self, rel_pos: torch.Tensor, pursuer_pos: torch.Tensor) -> None:
        if len(self._rel_hist) == 0:
            for _ in range(max(1, self.prev_evader_steps_fix)):
                self._rel_hist.append(rel_pos.detach().clone())

        if self.prev_traj_steps > 0 and len(self._pos_hist) == 0:
            for _ in range(self.prev_traj_steps):
                self._pos_hist.append(pursuer_pos.detach().clone())

    def _heading_from_quat(self, quat_wxyz: torch.Tensor) -> torch.Tensor:
        rot = ic.quaternion_to_rotation_matrix(quat_wxyz).reshape(1, 3, 3)
        return rot[..., :, 0]

    def build(
        self,
        pursuer: ic.DroneState,
        evader: ic.DroneState,
        previous_action: Optional[torch.Tensor],
        obs_dim: int,
    ) -> torch.Tensor:
        pursuer_pos = torch.as_tensor(pursuer.pos, dtype=torch.float32, device=self.device).view(1, 3)
        pursuer_quat = torch.as_tensor(pursuer.quat_wxyz, dtype=torch.float32, device=self.device).view(1, 4)
        pursuer_lin_vel = torch.as_tensor(pursuer.lin_vel, dtype=torch.float32, device=self.device).view(1, 3)
        pursuer_ang_vel = torch.as_tensor(pursuer.ang_vel, dtype=torch.float32, device=self.device).view(1, 3)
        evader_pos = torch.as_tensor(evader.pos, dtype=torch.float32, device=self.device).view(1, 3)
        evader_quat = torch.as_tensor(evader.quat_wxyz, dtype=torch.float32, device=self.device).view(1, 4)
        evader_lin_vel = torch.as_tensor(evader.lin_vel, dtype=torch.float32, device=self.device).view(1, 3)

        # -- Reference-frame alignment with Pursuit training --------------------
        # `DroneState.lin_vel` is the Crazyflie kalman.statePX/PY/PZ estimate, which
        # is expressed in the *body* frame (same convention the Intercept task feeds
        # to the network as `drone.vel_b`). Pursuit, however, observes *world*-frame
        # linear velocity (`drone.vel_w`, via `split_state`) and a world-frame
        # closing velocity, so we rotate body->world with the body-to-world rotation
        # matrix R (columns = body axes in world). `DroneState.ang_vel` comes from
        # the gyro and is already body-frame, matching Pursuit's
        # `body_rates = quat_rotate_inverse(rot, ang_vel_world)`.
        #
        # NOTE: we deliberately do *not* use `ic.quat_rotate_inverse` here for the
        # body<->world mapping. The rotation matrix R from
        # `quaternion_to_rotation_matrix` is verified body->world, keeping this path
        # independent of any frame convention in the shared helper.
        pursuer_rot_bw = ic.quaternion_to_rotation_matrix(pursuer_quat).reshape(3, 3)
        evader_rot_bw = ic.quaternion_to_rotation_matrix(evader_quat).reshape(3, 3)
        pursuer_lin_vel_world = (pursuer_rot_bw @ pursuer_lin_vel.view(3, 1)).view(1, 3)
        evader_lin_vel_world = (evader_rot_bw @ evader_lin_vel.view(3, 1)).view(1, 3)

        rel_pos = evader_pos - pursuer_pos
        self._seed_histories(rel_pos, pursuer_pos)

        obs = []

        if self.time_encoding:
            t = torch.tensor(
                [[float(self._step) / float(self.max_episode_length)]],
                dtype=torch.float32,
                device=self.device,
            )
            obs.append(t.expand(-1, self.time_encoding_dim))

        if self.prev_evader_traj_steps > 0:
            hist = torch.cat(list(self._rel_hist), dim=0)
            offset = self.prev_evader_steps_fix - self.prev_evader_traj_steps
            hist = hist[offset:]
            hist_norm = hist / self.k_rp
            obs.append(hist_norm.reshape(1, -1))

        rel_norm = rel_pos / self.k_rp
        obs.append(rel_norm)

        if self.include_distances:
            if self.prev_evader_traj_steps > 0:
                hist_dist = torch.norm(hist_norm, dim=-1, keepdim=True)
                obs.append(hist_dist.reshape(1, -1))
            obs.append(torch.norm(rel_norm, dim=-1, keepdim=True))

        if self.include_closing_velocity:
            # Training: -(self.lin_vel - v_evader) / K_V, both in the world frame.
            closing_velocity = -(pursuer_lin_vel_world - evader_lin_vel_world) / self.k_v
            obs.append(closing_velocity)

        error_pos_unit = rel_pos / (torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-6)
        heading = self._heading_from_quat(pursuer_quat)
        heading_tt = (heading * error_pos_unit).sum(-1, keepdim=True)
        if self.include_heading_to_target:
            obs.append(heading_tt)

        if self.prev_traj_steps > 0:
            pos_hist = torch.cat(list(self._pos_hist), dim=0)
            z_hist = ((pos_hist - self.arena_center) / self.k_rp)[..., 2:3]
            obs.append(z_hist.reshape(1, -1))

        pos_norm = (pursuer_pos - self.arena_center) / self.k_p
        obs.append(pos_norm[..., 2:3])

        obs.append(pursuer_rot_bw.reshape(1, 9))
        # Training observes world-frame linear velocity (drone.vel_w / K_V).
        obs.append(pursuer_lin_vel_world / self.k_v)

        if self.use_body_rates:
            # The gyro already reports body-frame rates, which is exactly the
            # training body_rates = quat_rotate_inverse(rot, ang_vel_world).
            obs.append(pursuer_ang_vel / self.k_omega)
        else:
            # Training's use_body_rates=False branch observes world-frame angular
            # velocity (the raw drone.vel_w angular component).
            ang_vel_world = (pursuer_rot_bw @ pursuer_ang_vel.view(3, 1)).view(1, 3)
            obs.append(ang_vel_world / self.k_omega)

        if self.include_last_action:
            # Under the AMSPBRateController transform, Pursuit stores the
            # *post-transform rotor commands* as prev_action (the transform
            # replaces ("agents","action") before _pre_sim_step). The deploy only
            # has the 4D CTBR action, so feeding it here would silently mismatch
            # training. Refuse rather than fly on a wrong observation.
            raise NotImplementedError(
                "include_last_action is not supported for AMSPB rate deployment: "
                "training observes post-transform rotor commands, which are not "
                "reconstructable from the CTBR action. Retrain with "
                "include_last_action=false or extend the builder."
            )

        if self.include_effort:
            obs.append(torch.zeros(1, 1, dtype=torch.float32, device=self.device))

        obs_tensor = torch.cat(obs, dim=-1)
        if obs_tensor.shape[-1] != obs_dim:
            raise RuntimeError(
                f"Pursuit observation dim mismatch: built {obs_tensor.shape[-1]}, expected {obs_dim}."
            )

        self._rel_hist.append(rel_pos.detach().clone())
        if self.prev_traj_steps > 0:
            self._pos_hist.append(pursuer_pos.detach().clone())
        self._step += 1

        return obs_tensor


_BUILDERS: Dict[int, PursuitObservationBuilder] = {}


def _get_builder(policy: torch.nn.Module, metadata: ic.PolicyMetadata, device: torch.device) -> PursuitObservationBuilder:
    key = id(policy)
    if key not in _BUILDERS:
        pursuit_obs_cfg = metadata.notes.get("pursuit_obs", None)
        if not isinstance(pursuit_obs_cfg, dict):
            raise RuntimeError(
                "Missing pursuit_obs metadata. Re-export with scripts/amspb_peg/deploy/export_policy.py"
            )
        _BUILDERS[key] = PursuitObservationBuilder(pursuit_obs_cfg, device)
    return _BUILDERS[key]


def _build_command(
    policy: torch.nn.Module,
    metadata: ic.PolicyMetadata,
    pursuer: ic.DroneState,
    evader: ic.DroneState,
    previous_action: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
):
    role = str(metadata.notes.get("role", "pursuer")).lower()
    if role != "pursuer":
        raise NotImplementedError(
            f"Role '{role}' is not implemented yet. Current deploy path supports pursuer only."
        )

    builder = _get_builder(policy, metadata, device)
    obs = builder.build(
        pursuer=pursuer,
        evader=evader,
        previous_action=previous_action,
        obs_dim=int(metadata.obs.obs_dim),
    )

    with torch.no_grad():
        raw_action = policy(obs)

    return torch.tanh(raw_action).squeeze(0), ic.decode_action_to_ctbr(raw_action, metadata.ctbr)


def _default_config_path() -> str:
    local_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    legacy_cfg = os.path.join(repo_root, "scripts", "deploy", "config_v2.yaml")
    if os.path.isfile(local_cfg):
        return local_cfg
    return legacy_cfg


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run AMSPB pursuer policy artifact using the intercept runtime loop."
    )
    parser.add_argument(
        "--config",
        default=_default_config_path(),
        help="Path to controller YAML config. If omitted, uses AMSPB local config.yaml when present, else scripts/deploy/config_v2.yaml.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Artifact directory containing policy_ts.pt and metadata.json. If omitted, reads artifact_dir from config.",
    )
    args = parser.parse_args(argv)

    import scripts.deploy.intercept_controller_v2 as base_controller

    config_path = os.path.abspath(os.path.expanduser(args.config))
    artifact_dir = args.artifact_dir
    if artifact_dir is None:
        artifact_dir = base_controller._load_artifact_dir(config_path)

    controller_args = argparse.Namespace(
        config=config_path,
        artifact_dir=artifact_dir,
    )

    original_build_command = base_controller._build_command
    try:
        base_controller._build_command = _build_command
        controller = base_controller.InterceptController(controller_args)
        controller.run()
    finally:
        base_controller._build_command = original_build_command


if __name__ == "__main__":
    main()
