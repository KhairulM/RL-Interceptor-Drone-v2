# AMSPB Pursuit — Real Crazyflie Deployment

Deploy a **pursuer** policy trained by
[`scripts/amspb_peg/train_pursuer.py`](../train_pursuer.py) (task `Pursuit`,
`policy: amspbrate`) onto a real Crazyflie.

These scripts reuse the Crazyswarm2 runtime infrastructure from
[`scripts/deploy/`](../../deploy/) (radio link, mocap/NatNet streaming, CTBR
dispatch, evader motion) and only swap in the **Pursuit** observation layout and
the **AMSPBRateController** action decode.

```
export_policy.py         Isaac-side exporter: checkpoint -> policy_ts.pt + metadata.json
intercept_controller.py  Crazyswarm-side runtime: builds Pursuit obs, runs policy, sends CTBR
config.yaml              Runtime config (artifact path, drones, mocap, evader motion)
test_pursuit_obs.py      Regression test: deploy obs == training obs (torch only)
```

## Two environments

Same split as `scripts/deploy/` (see that README for full setup):

* **Isaac Sim venv** (Python 3.11) — runs `export_policy.py`. It rebuilds the
  training env to read the exact observation/action specs and normalization
  constants, then traces a deterministic TorchScript actor.
* **Crazyswarm2 venv** (Python 3.10, `cflib` + ROS 2) — runs
  `intercept_controller.py`. Depends only on `torch`, `numpy`, `cflib`, `rclpy`.

`intercept_common.py` (shared, torch-only) is imported by both.

## 1. Export a checkpoint (Isaac venv)

```bash
# from repo root, Isaac venv active
python -m scripts.amspb_peg.deploy.export_policy \
    checkpoint=/abs/path/to/checkpoint_final.pt \
    task.pursuer_model.policy=amspbrate \
    role=pursuer
# -> scripts/amspb_peg/deploy/artifacts/pursuit_ppo_pursuer/{policy_ts.pt,metadata.json}
```

Notes:
* Pass the **same task overrides** you trained with so the observation dimension
  matches. The exporter asserts a 4D CTBR action head and verifies the traced
  actor against eager execution (max abs error < 1e-4).
* Override the output folder with `export_dir=...`.
* `metadata.json` records the full observation layout under `notes.pursuit_obs`
  (the flags + `K_P/K_V/K_OMEGA/K_RP/arena_center`) and the CTBR decode under
  `ctbr`. The runtime rebuilds the observation from these — nothing is hard-coded.

## 2. Run on hardware (Crazyswarm venv)

Edit [`config.yaml`](config.yaml) (URIs, mocap server/rigid-body ids,
`artifact_dir`, evader motion), then:

```bash
# from repo root, Crazyswarm venv active
python -m scripts.amspb_peg.deploy.intercept_controller \
    --config scripts/amspb_peg/deploy/config.yaml
# or point directly at an artifact:
python -m scripts.amspb_peg.deploy.intercept_controller --artifact-dir <dir>
```

Both drones take off to `controller.takeoff_height`, then the pursuer runs the
policy at `control_dt` while the evader follows `controller.evader_motion`
(`hover` / `random` / `trajectory`) or is a real mocap-tracked Crazyflie
(`evader_source: cf`). Ctrl+C lands and disarms both.

## Observation & action mapping (the part that must match training)

Observation (24-dim for the default config: `time_encoding`,
`include_distances`, `include_closing_velocity`, `use_body_rates` on; histories
off), built in `PursuitObservationBuilder`:

| component        | dim | training source (`pursuit.py`)                     |
|------------------|-----|----------------------------------------------------|
| time encoding    | 1   | `progress_buf / max_episode_length`                |
| relative position| 3   | `(evader_pos - pursuer_pos) / K_RP`                |
| distance         | 1   | `‖rel‖`                                            |
| closing velocity | 3   | `-(v_pursuer − v_evader) / K_V`  **(world frame)** |
| altitude         | 1   | `((pos − arena_center) / K_P)[z]`                  |
| rotation matrix  | 9   | `quaternion_to_rotation_matrix(rot)`               |
| linear velocity  | 3   | `v_pursuer / K_V`  **(world frame)**               |
| body rates       | 3   | `quat_rotate_inverse(rot, ω_world) / K_OMEGA`      |

**Reference frames — the critical detail.** `DroneState.lin_vel` comes from the
Crazyflie `kalman.statePX/PY/PZ` estimate, which is **body frame** (the same
convention the *Intercept* task consumes as `drone.vel_b`). But **Pursuit** feeds
the network **world**-frame linear/closing velocity (`drone.vel_w`). So the
builder rotates body→world with the body-to-world rotation matrix. Angular
velocity, in contrast, is read from the **gyro** (already body frame), which
matches Pursuit's `body_rates`, so it is used directly with no extra rotation.
`test_pursuit_obs.py` locks this in.

Action decode (`AMSPBRateController` → `decode_action_to_ctbr`): the 4D action is
`tanh`-squashed, then

* body-rate setpoint (deg/s) = `target_rate * 180 * target_clip`  (training: `*π` rad/s),
* collective thrust ratio     = `clip((target_thrust + 1)/2, min_thrust_ratio, max_thrust_ratio)`,
  sent to the firmware as `ratio * 65535` PWM.

The firmware runs the on-board rate PID; we only send the setpoint.

## Caveats before flying

* **Arena scale.** `train_pursuer.yaml` uses `arena_size: 2.0` → a ~12 m wide,
  6 m tall arena centered at `z ≈ 3.5 m`. In a small lab the altitude/position
  observations sit far from what the policy saw during training. Either train
  with a lab-sized arena or verify behavior in sim at your intended flight
  volume first.
* **Thrust scaling** is the dominant sim-to-real knob. The exporter writes
  `max_thrust_ratio = 1.0` (faithful to `AMSPBRateController`, which has no upper
  clip). Lower it in `metadata.json` as a safety cap, and expect to hand-tune the
  thrust mapping for your drone's mass/battery.
* **Evader velocity.** With `evader_source: cf`, the evader's onboard body-frame
  velocity is rotated to world using its mocap orientation. With scripted/hover
  motion the commanded velocity is already world-frame (identity orientation).

## Testing

```bash
python -m scripts.amspb_peg.deploy.test_pursuit_obs   # torch only, no Isaac/ROS
```

## Known issue in shared `intercept_common`

`scripts/deploy/intercept_common.quat_rotate_inverse` has a **sign error**: its
cross-product term is `+2w(q×v)` (the *forward* rotation) where the true inverse
is `-2w(q×v)` (cf. `omni_drones/utils/torch.quat_rotate_inverse`, `a - b + c`).
It returns `R·v`, not `Rᵀ·v`. This deploy path deliberately avoids that function
and derives body→world from the (verified-correct) rotation matrix instead. The
bug affects the *Intercept* deploy's `use_relative_velocity` path and should be
fixed there separately.
