# thesis_F14_rls_s200: self-play pursuer with the opponent-acceleration estimator + RLS

- `pursuer/`: F14 Stage E RLS arm, seed 200, stage 24
  (`scripts/selfplay/outputs/thesis-F14-v2d-acc-rls-26-s200/checkpoints/pursuer_stage24.pt`). Its observation is
  31-D: the 28-D PursuitEvasion observation plus the adapter output.
  - The adapter is the learned acceleration estimator `nf_dvel_H10` (F5 estimator, `adapter_basis.pt`) with an
    online RLS readout: history 10, horizon 10, delta 0.01, forgetting 0.999.
  - The adapter output is the predicted mean evader acceleration over the next 10 steps, x0.1, in the body frame.
  - `metadata.json` carries the adapter block (`artifact_version` 2). The controller runs the adapter once per
    control tick, as in training.
- `evader/`: the same campaign's final evader (`evader_stage25.pt`; evaders are trained at odd stages). Plain 28-D
  observation.
- `evader_stage23/`: the same campaign's second-to-last evader (`evader_stage23.pt`), plain 28-D observation
  (exported 2026-11-24: `export_noisaac.py evader_stage23`).

Exported without Isaac by `export_noisaac.py`, the same method as `../thesis_F8b_roncero_s0`:
- Each checkpoint loads strictly and has TorchScript trace error 0.
- The task metadata (arena 4 x 4 m, z 0.1-3.0 m, capture 0.10 m, `pidrate` CTBR at 50 Hz, 1500-step episodes) is
  F8's; F14 uses the same task values.

## Why this pursuer

Selected from the F21 RLS ablation (`experiments/thesis/F21_rls_ablation/results.json`) and F14/F15/F20 in
`experiments/thesis/RESULTS.md`. The `baseline` arm of F21 is this F14 Stage E RLS policy. Seed 200 ranks first on
every criterion:

| criterion | seed 200 |
|---|---|
| G1 + own-campaign J, 0.10 m / 0.15 m | 1.876 / 1.961 (G1 0.892, own J 0.984) |
| switching evaders: post-switch capture / post-switch J | 0.767 / 0.905 (both the highest) |
| fixed-evader capture | 0.768 |
| lure J (F20, RLS vs its own exploiter) | 0.674 (other seeds 0.071 / 0.048) |
| F14 G1 / G2 gap vs plain | +0.051 / +0.357 (the only seed with a positive G1 gap) |
| F15 post-switch J, RLS minus F8 | +0.220 |

Caveats:
- Choosing the best of 3 seeds is a post hoc selection.
- The live RLS update adds little over the frozen estimator in F21.
- Training had no domain randomisation and no sensor noise.

## Run

In `scripts/deploy/config_pursuit_evasion.yaml` (the default config):

```yaml
policy:
  artifact_dir: artifacts/thesis_F14_rls_s200/pursuer
opponent:
  mode: policy
  artifact_dir: artifacts/thesis_F14_rls_s200/evader
```

Set `logging.debug_obs: true` to print each observation component, including the estimator output
(`opp_acc/10(body)`).
