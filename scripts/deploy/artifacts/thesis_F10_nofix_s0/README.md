# thesis_F10_nofix_s0: F10 seed-0 final evader (for switching-evader tests)

- `evader/`: `scripts/selfplay/outputs/thesis-F10-nofix-s0/checkpoints/evader_stage11.pt`, the final evader of the
  13-stage F10 campaign (thesis rules, `algo.gae_cut_at_truncation` off). Plain 28-D observation.
- Exported without Isaac by `export_noisaac.py`: the checkpoint loads strictly, TorchScript trace error 0. The task
  metadata (arena 4 x 4 m, z 0.1-3.0 m, `pidrate` CTBR at 50 Hz) is copied from `../thesis_F8_roncero_s0/evader`.

## Why this evader

The F14 / F15 switching-evader evaluations of the F14 RLS seed-200 pursuer switch between the final evaders of the
F8 and F10 seed-0 campaigns (`_switch_pair` in `scripts/experiments/thesis/exps/f14_generalization_sp.py`), at 0.5,
1.0 or 1.5 s into the episode, in both directions. Those two evaders are `../thesis_F8_roncero_s0/evader` (stage 25)
and this one.

## Run

```yaml
opponent:
  mode: policy
  artifact_dir: artifacts/thesis_F8_roncero_s0/evader   # first evader
  switch:
    artifact_dir: artifacts/thesis_F10_nofix_s0/evader  # second evader
    after_s: 1.0                                        # s after the chase starts (thesis: 0.5, 1.0, 1.5)
```

Swap the two `artifact_dir` values for F10 to F8.
