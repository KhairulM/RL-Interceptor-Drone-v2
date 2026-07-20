# Interceptor Benchmark: RL vs PN and MPC

This tutorial compares a trained RL interceptor with traditional guidance baselines under the same environment settings.

## What Is Included

- RL policy evaluation from checkpoint
- Pure pursuit baseline
- Proportional Navigation (PN) baseline
- Lightweight kinematic MPC baseline
- Shared fairness checks across all methods
- JSON summary export and optional Weights & Biases table logging

## New Files

- `scripts/evaluate.py`: benchmark runner
- `scripts/evaluate.yaml`: benchmark configuration
- `omni_drones/controllers/pure_pursuit_controller.py`
- `omni_drones/controllers/proportional_navigation_controller.py`
- `omni_drones/controllers/kinematic_mpc_controller.py`
- `omni_drones/controllers/intercept_baseline_common.py`
- `omni_drones/controllers/cfg/pure_pursuit_intercept.yaml`
- `omni_drones/controllers/cfg/pn_intercept.yaml`
- `omni_drones/controllers/cfg/kinematic_mpc_intercept.yaml`

## Run The Benchmark

From `scripts/`:

```bash
python evaluate.py \
  task=Intercept \
  algo=ppo \
  checkpoint=/absolute/path/to/checkpoint_final.pt \
  headless=true
```

Or infer checkpoint from a W&B run folder:

```bash
python evaluate.py \
  task=Intercept \
  algo=ppo \
  wandb_run_dir=/absolute/path/to/wandb/run-YYYYMMDD_HHMMSS-<id> \
  headless=true
```

## Key Configuration Knobs

Edit `scripts/evaluate.yaml`:

- `eval.methods`: method list (`rl`, `pure_pursuit`, `pn`, `kinematic_mpc`)
- `eval.seeds`: benchmark seeds
- `eval.num_envs`: vectorized env count for evaluation
- `eval.steps_per_env`: rollout horizon per method/seed/scenario
- `eval.scenarios`: stress-grid presets
- `eval.strict_fairness`: enforce identical obs/action/horizon/dt signatures
- `eval.output_dir`: JSON output location

### Baseline Controller Parameters

Tune in:

- `omni_drones/controllers/cfg/pure_pursuit_intercept.yaml`
- `omni_drones/controllers/cfg/pn_intercept.yaml`
- `omni_drones/controllers/cfg/kinematic_mpc_intercept.yaml`

## Output Artifacts

The runner writes:

- one aggregated JSON summary per benchmark run
- optional W&B summary table (`evaluate/summary_table`)

Summary metrics include:

- `success_rate`
- `intercept_time_s`
- `interception_speed`
- `miss_distance`

## Fairness Policy

For each `(scenario, seed)`, all methods must share:

- observation dimension
- action dimension
- action transform mode
- simulation timestep (`dt`)
- max episode length

If any mismatch appears, evaluation stops with an error.

## How The Baselines Fly

All methods (RL and classical) output the same pre-tanh CTBR action
(`[body_rate(3), thrust(1)]`) and are stabilized by the pursuer's on-board
`PIDRateController` via the task's `PIDrate` action transform — the exact same
low-level flight pipeline the RL policy uses. The classical baselines add a
geometric outer loop (`GeometricCTBR` in
[intercept_baseline_common.py](omni_drones/controllers/intercept_baseline_common.py))
that converts a guidance target (position/velocity) into that CTBR action, with
the hover throttle calibrated from the drone parameters.

Each guidance law differs only in the target it picks:

- **Pure pursuit**: aim at the evader's current position.
- **Proportional Navigation (PN)**: aim at the constant-bearing intercept point
  (evader position led by its velocity over the estimated time-to-go).
- **Kinematic MPC**: predict the evader position over the time-to-go and track
  it, matching the evader velocity at intercept.

> Note: the classical guidance laws use the full relative state (range +
> bearing + evader velocity), which PN/MPC fundamentally require. The RL policy
> is trained on bearing-only observations. The `LeePositionController` is not
> used for the pursuer because it is only marginally stable on the perturbed,
> motor-lagged Crazyflie; the tuned on-board rate PID is used instead.

### Outer-loop gains

The geometric outer-loop gains (`kp`, `kv`, `k_att`, `k_yaw`, `max_tilt_deg`)
are set where `GeometricCTBR` is constructed in
[scripts/evaluate.py](scripts/evaluate.py). Increase `kp`/`k_att` for a tighter
intercept; lower them if the approach oscillates.

## Suggested Workflow

1. Run quick smoke benchmark with 1-2 seeds and one scenario.
2. Tune PN/MPC YAML gains until behavior is stable.
3. Run full configured benchmark and compare mean/std by scenario.
4. Keep the same seeds/scenarios across model revisions for trend tracking.
