# Intercept policy deployment (CrazySim + Crazyswarm2)

This folder contains everything needed to run a trained **Intercept** policy
checkpoint on a Crazyflie in **CrazySim**, commanded through **Crazyswarm2**.

Because the training stack (Isaac Sim, Python 3.11) and the Crazyswarm2 stack
(ROS 2 Humble + cflib, Python 3.10) live in **separate virtual environments**
(see the repo [README](../README.md)), deployment is a **two-step** process:

```
 ┌─────────────────────────┐        artifact         ┌──────────────────────────┐
 │ export_policy.py         │   policy_ts.pt +        │ intercept_controller.py   │
 │ (.venv, Python 3.11,     │──►  metadata.json    ──►│ (.venv-crazyswarm, 3.10,  │
 │  Isaac Sim + torchrl)    │                         │  ROS 2 + torch, no Isaac) │
 └─────────────────────────┘                         └──────────────────────────┘
```

1. **Export** (Isaac env): load the checkpoint with the exact same machinery as
   `scripts/play.py` (so it works for *any* RL algorithm — ppo, mappo, happo,
   sac, td3), extract the deterministic actor, and serialise it to a
   self-contained **TorchScript** module plus a small `metadata.json`.
2. **Deploy** (Crazyswarm2 env): a lightweight ROS 2 node loads that artifact
   (only `torch` + `numpy` needed), rebuilds the Intercept observation from live
   drone state, runs the policy, decodes the action into a
   collective-thrust + body-rate (CTBR) command, and streams it to the drone.

## Files

| File | Runs in | Purpose |
|---|---|---|
| [intercept_common.py](intercept_common.py) | both | Pure `torch`/stdlib core: observation construction + CTBR decoding + metadata IO. Mirrors the training code so the two stay in sync. |
| [export_policy.py](export_policy.py) | `.venv` (3.11) | Load checkpoint, extract deterministic actor, export TorchScript + metadata. |
| [intercept_controller.py](intercept_controller.py) | `.venv-crazyswarm` (3.10) | ROS 2 node that flies the policy on the Crazyflie. |
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
  observation layout (e.g. `task.pursuer.use_ab_world_frame`,
  `task.pursuer.use_rot_speed`, `task.evader.use_relative_velocity`). The
  exporter records these in `metadata.json` and the controller enforces them.
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

python deploy/intercept_controller.py --ros-args \
    -p artifact_dir:=deploy/artifacts/intercept_ppo \
    -p pursuer_name:=cf1 \
    -p evader_source:=scripted
```

**Take off first.** The policy assumes the pursuer is already airborne (it was
trained spawning at ~1.6 m). Take off with your usual Crazyswarm2 tooling before
starting this node, or extend the node to call the `takeoff` service.

### Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `artifact_dir` | *(required)* | Folder with `policy_ts.pt` + `metadata.json`. |
| `pursuer_name` | `cf1` | Crazyflie name (topic namespace) of the interceptor. |
| `evader_name` | `cf2` | Name of the target Crazyflie (when `evader_source=cf`). |
| `evader_source` | `scripted` | `scripted` (internal trajectory) or `cf` (read `/<evader>/pose`). |
| `command_mode` | `attitude` | `attitude` (`cmd_vel_legacy`) or `fullstate` (experimental). |
| `pursuer_pose_topic` | `/<pursuer>/pose` | Source of pursuer pose (`geometry_msgs/PoseStamped`). |
| `evader_pose_topic` | `/<evader>/pose` | Source of evader pose. |
| `evader_speed` / `evader_start` / `evader_dir` | `3.0` / `[3,0,1.6]` / `[1,0,0]` | Scripted evader motion. |
| `max_tilt_deg` | `30` | Attitude-setpoint clamp (attitude backend). |
| `min_altitude` | `0.15` | Safety cutoff (mirrors the training "misbehave" floor). |
| `state_timeout` | `0.5` | Stop the drone if pose is stale for this long (s). |
| `vel_lpf` | `0.4` | Low-pass factor for finite-difference velocity. |

The controller estimates the pursuer's **world-frame linear velocity** by
finite-differencing its pose (works with just `PoseStamped`; no odometry/twist
topic required). Angular velocity and evader velocity are only needed when the
policy's `metadata.json` enables `use_rot_speed` / `use_relative_velocity`.

## Command fidelity (important)

The Intercept policy outputs a **CTBR** command: a collective thrust plus body
rates `[roll_rate, pitch_rate, yaw_rate]` (deg/s). The decoding in
`decode_action_to_ctbr` reproduces the training-time `PIDRateController`
transform exactly.

However, Crazyswarm2's cflib backend does **not** expose a native body-rate
setpoint. The available streaming topics are:

- `cmd_vel_legacy` (`geometry_msgs/Twist`) → attitude (roll/pitch **angles**),
  yaw-rate, thrust;
- `cmd_full_state`, `cmd_hover`, `cmd_position`, `cmd_velocity_world`.

So the last-mile command is an **approximation**:

- `attitude` (default): integrates the roll/pitch **rate** commands one control
  step ahead of the measured attitude and sends the result as an attitude
  setpoint via `cmd_vel_legacy`; yaw-rate and thrust pass through natively. This
  works with a stock Crazyswarm2 + CrazySim install but is not a perfect
  reproduction of the trained rate-control interface.
- `fullstate` (experimental): forwards the body rates as the `omega`
  feed-forward of `cmd_full_state` while holding position — for experimentation
  only.

For a **faithful** deployment, use a CrazySim / firmware build that accepts CTBR
directly and add a `CommandBackend` that publishes to it. The `CommandBackend`
abstraction in [intercept_controller.py](intercept_controller.py) is the single
place to plug this in — no other code needs to change.

## Testing the core

```bash
# any environment with torch:
python deploy/test_intercept_common.py
```

These tests cross-check the rotation-matrix / observation layout / CTBR decoding
against independent reference implementations, so you can catch drift from the
training code without needing Isaac Sim or ROS.
