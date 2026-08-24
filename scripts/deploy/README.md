# Intercept policy deployment (CrazySim + Crazyswarm2)

This folder contains everything needed to run a trained **Intercept** policy
checkpoint on a Crazyflie in **CrazySim**, commanded through **Crazyswarm2**.

Because the training stack (Isaac Sim, Python 3.11) and the Crazyswarm2 stack
(ROS 2 Humble + cflib, Python 3.10) live in **separate virtual environments**
(see the repo [README](../README.md)), deployment is a **two-step** process:

```
 ┌─────────────────────────┐        artifact         ┌──────────────────────────┐
 │ export_policy.py         │   policy_ts.pt +        │ intercept_controller_v2.py │
 │ (.venv, Python 3.11,     │──►  metadata.json    ──►│ (.venv-crazyswarm, 3.10,  │
 │  Isaac Sim + torchrl)    │                         │  ROS 2 + torch, no Isaac) │
 └─────────────────────────┘                         └──────────────────────────┘
```

1. **Export** (Isaac env): load the checkpoint with the exact same machinery as
   `scripts/play.py` (so it works for *any* RL algorithm — ppo, mappo, happo,
   sac, td3), extract the deterministic actor, and serialise it to a
   self-contained **TorchScript** module plus a small `metadata.json`.
2. **Deploy** (Crazyswarm2 env): the lightweight cflib controller loads that artifact
   (only `torch` + `numpy` needed), rebuilds the Intercept observation from live
   drone state, runs the policy, decodes the action into a
   collective-thrust + body-rate (CTBR) command, and streams it to the drone.

## Files

| File | Runs in | Purpose |
|---|---|---|
| [intercept_common.py](intercept_common.py) | both | Pure `torch`/stdlib core: observation construction + CTBR decoding + metadata IO. Mirrors the training code so the two stay in sync. |
| [export_policy.py](export_policy.py) | `.venv` (3.11) | Load checkpoint, extract deterministic actor, export TorchScript + metadata. |
| [intercept_controller_v2.py](intercept_controller_v2.py) | `.venv-crazyswarm` (3.10) | cflib controller that flies the policy on the Crazyflie. |
| [test_intercept_common.py](test_intercept_common.py) | any env with `torch` | Unit tests for the observation/CTBR math. |

## Step 1 — Export the policy (Isaac `.venv`)

```bash
cd ~/Projects/RLInterceptorDrone
source .venv/bin/activate

python deploy/export_policy.py \
    task=Intercept algo=ppo headless=true \
    checkpoint=scripts/outputs/<date>/<time>/checkpoint_final.pt \
    export_dir=deploy/artifacts/intercept_ppo
```

Notes:

- Set `algo=` to whatever algorithm produced the checkpoint. Supported for
  extraction: `ppo`, `mappo`, `happo`, `sac`, `td3` (and their aliases).
- Use the **same** task overrides you trained/evaluated with if they change the
  observation layout (`task.observation.use_world_frame_pos`,
  `task.observation.include_evader_rel_lin_vel`,
  `task.observation.include_previous_action`). The exporter records these in
  `metadata.json` and the controller enforces them.
- Re-export after changing the relative-heading frame. Current Intercept
  artifacts use metadata version 3, where the target heading is in the
  pursuer body frame; older artifacts are intentionally rejected.
- The exporter numerically validates the TorchScript trace against the eager
  policy before writing it; a mismatch aborts the export.

Output (`deploy/artifacts/intercept_ppo/`):

```
policy_ts.pt      # standalone TorchScript deterministic actor
metadata.json     # obs layout + CTBR decode params + provenance
```

## Step 2 — Run the controller (Crazyswarm2 `.venv-crazyswarm`)

Start CrazySim and the Crazyswarm2 server first (see the repo README, e.g.
`./scripts/launch_crazyswarm_udp.sh`). Then, in another shell:

```bash
cd ~/Projects/RLInterceptorDrone
source .venv-crazyswarm/bin/activate
source /opt/ros/humble/setup.bash
source crazyswarm2_ws/install/setup.bash
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.10/site-packages:$PYTHONPATH"

# torch + numpy must be available in .venv-crazyswarm:
#   pip install torch numpy   # CPU wheels are sufficient

python scripts/deploy/intercept_controller_v2.py \
  --config scripts/deploy/config_v2.yaml
```

The controller connects, arms, and takes off both configured drones. Set the
artifact directory, radio URIs, mocap settings, and evader behavior in the YAML
configuration before running it.

### Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `artifact_dir` | *(required)* | Folder with `policy_ts.pt` + `metadata.json`. |
| `controller.evader_source` | `cf` | `cf` for a tracked target or `scripted` for generated motion. |
| `controller.evader_motion.type` | `hover` | `hover`, `linear`, or Intercept-aligned turning `random`; CSV `trajectory` remains available for legacy use. |
| `controller.control_dt` | `0.02` | Policy and evader-motion period in seconds. |
| `controller.takeoff_height` | `1.0` | Takeoff height for both drones in metres. |
| `pursuer.uri` / `evader.uri` | *(config)* | cflib radio or CrazySim UDP URI. |
| `min_altitude` | `0.15` | Safety cutoff (mirrors the training "misbehave" floor). |
| `state_timeout` | `0.5` | Stop the drone if pose is stale for this long (s). |

The controller rotates the on-board Kalman **world-frame** velocity
(`kalman.statePX/Y/Z`) into the pursuer body frame before building the policy
observation. Its body rates come directly from the gyro; both are always part
of the observation. The evader's world-frame velocity is only needed when the
policy's `metadata.json` enables `use_relative_velocity`.

## Command fidelity (important)

The Intercept policy outputs a **CTBR** command: a collective thrust plus body
rates `[roll_rate, pitch_rate, yaw_rate]` (deg/s). The decoding in
`decode_action_to_ctbr` reproduces the training-time `PIDRateController`
transform exactly.

The v2 controller uses cflib's low-level `send_setpoint` command directly. Its
setup selects Crazyflie rate mode for roll and pitch, then sends the decoded
roll, pitch, yaw rates and collective thrust. Confirm `rate_sign` and firmware
rate-mode behavior on a restrained test before flight.

## Testing the core

```bash
# any environment with torch:
python deploy/test_intercept_common.py
```

These tests cross-check the rotation-matrix / observation layout / CTBR decoding
against independent reference implementations, so you can catch drift from the
training code without needing Isaac Sim or ROS.
