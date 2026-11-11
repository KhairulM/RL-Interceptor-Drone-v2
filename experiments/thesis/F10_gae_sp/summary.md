# F10_gae_sp summary

Final pursuer (stage 12) of thesis-F10-nofix-s0 (thesis rules, Hedge both roles, 13 x 50M, seed 0, no GAE truncation fix) vs F9 control (the same with the fix) and F3 hedge s0. All evals clean under the thesis rules; exploitability_thesis = 1 - J vs a fresh 100M-frame thesis-rules exploiter. 1 seed, descriptive.

Pre-registered (a), in-campaign lure rate (mean over evader stages 3-11): no fix 0.81, control 0.36. (b), pursuer crash vs the exploiter: see the table.

## Capture radius 0.10 m

| metric | F10 no fix | F9 control (fix) | no fix - fix | F3 hedge s0 (9 x 300M, no fix) |
|---|---|---|---|---|
| heldout_rl | 0.933 | 0.893 | +0.040 | 0.791 |
| heldout_worst_cluster | 0.890 | 0.859 | +0.031 | 0.692 |
| other_f3_evaders | 0.826 | 0.766 | +0.060 | 0.871 |
| other_f9_evaders | 0.890 | 0.866 | +0.024 | - |
| own_campaign_evaders | 0.813 | 0.933 | -0.120 | 0.957 |
| train | 0.838 | 0.688 | +0.150 | 0.594 |
| heldout_scripted | 0.968 | 0.918 | +0.049 | 0.808 |
| heldout_scripted_pursuer_crash | 0.033 | 0.085 | -0.052 | 0.197 |
| x3_exploiters | 0.993 | 0.983 | +0.010 | 0.955 |
| exploitability_thesis | 0.172 | 0.440 | -0.268 | 0.974 |
| exploiter_thesis_pursuer_crash | 0.031 | 0.056 | -0.025 | 0.975 |
| exploiter_thesis_timeout | 0.143 | 0.385 | -0.243 | - |

## Capture radius 0.15 m

| metric | F10 no fix | F9 control (fix) | no fix - fix | F3 hedge s0 (9 x 300M, no fix) |
|---|---|---|---|---|
| heldout_rl | 0.996 | 0.989 | +0.007 | 0.960 |
| heldout_worst_cluster | 0.991 | 0.973 | +0.018 | 0.874 |
| other_f3_evaders | 0.957 | 0.948 | +0.009 | 0.957 |
| other_f9_evaders | 0.982 | 0.972 | +0.011 | - |
| own_campaign_evaders | 0.964 | 0.995 | -0.031 | 0.993 |
| train | 0.949 | 0.840 | +0.109 | 0.759 |
| heldout_scripted | 0.998 | 0.978 | +0.020 | 0.905 |
| heldout_scripted_pursuer_crash | 0.002 | 0.023 | -0.021 | 0.096 |
| x3_exploiters | 1.000 | 0.998 | +0.002 | 0.987 |
| exploitability_thesis | 0.004 | 0.095 | -0.091 | 0.773 |
| exploiter_thesis_pursuer_crash | 0.004 | 0.008 | -0.004 | 0.774 |
| exploiter_thesis_timeout | 0.000 | 0.086 | -0.086 | - |

## In-campaign lure rates (pursuer crash caused by each evader stage in the stage >= 2 pursuers)

| campaign | evader stage | mean | min - max | n pursuers |
|---|---|---|---|---|
| thesis-F10-nofix-s0 | 3 | 0.75 | 0.75 - 0.75 | 1 |
| thesis-F10-nofix-s0 | 5 | 0.85 | 0.85 - 0.86 | 2 |
| thesis-F10-nofix-s0 | 7 | 0.88 | 0.87 - 0.90 | 3 |
| thesis-F10-nofix-s0 | 9 | 0.80 | 0.56 - 0.93 | 4 |
| thesis-F10-nofix-s0 | 11 | 0.75 | 0.47 - 0.94 | 5 |
| thesis-F9-control-s0 | 3 | 0.09 | 0.09 - 0.09 | 1 |
| thesis-F9-control-s0 | 5 | 0.41 | 0.27 - 0.54 | 2 |
| thesis-F9-control-s0 | 7 | 0.37 | 0.24 - 0.60 | 3 |
| thesis-F9-control-s0 | 9 | 0.42 | 0.16 - 0.63 | 4 |
| thesis-F9-control-s0 | 11 | 0.53 | 0.22 - 0.69 | 5 |

## Inside the campaign (each stage's eval vs its training opponents)

**thesis-F10-nofix-s0**

| stage | role | vs | capture | pursuer crash | evader crash + OOB | timeout |
|---|---|---|---|---|---|---|
| 0 | pursuer | evader hover | 0.92 | 0.01 | 0.08 | 0.00 |
| 0 | pursuer | evader circular | 0.87 | 0.00 | 0.12 | 0.00 |
| 0 | pursuer | evader flee | 0.88 | 0.01 | 0.11 | 0.00 |
| 0 | pursuer | evader weave | 0.89 | 0.01 | 0.10 | 0.00 |
| 1 | evader | pursuer pure_pursuit | 0.05 | 0.15 | 0.08 | 0.71 |
| 1 | evader | pursuer apf | 0.31 | 0.26 | 0.14 | 0.29 |
| 2 | pursuer | evader stage1 | 0.92 | 0.02 | 0.07 | 0.00 |
| 3 | evader | pursuer stage0 | 0.08 | 0.01 | 0.03 | 0.88 |
| 3 | evader | pursuer stage2 | 0.18 | 0.75 | 0.08 | 0.00 |
| 4 | pursuer | evader stage1 | 0.91 | 0.02 | 0.08 | 0.00 |
| 4 | pursuer | evader stage3 | 0.91 | 0.06 | 0.03 | 0.00 |
| 5 | evader | pursuer stage0 | 0.03 | 0.00 | 0.03 | 0.94 |
| 5 | evader | pursuer stage2 | 0.07 | 0.85 | 0.09 | 0.00 |
| 5 | evader | pursuer stage4 | 0.10 | 0.86 | 0.06 | 0.00 |
| 6 | pursuer | evader stage1 | 0.90 | 0.02 | 0.08 | 0.00 |
| 6 | pursuer | evader stage3 | 0.92 | 0.06 | 0.03 | 0.00 |
| 6 | pursuer | evader stage5 | 0.85 | 0.12 | 0.04 | 0.00 |
| 7 | evader | pursuer stage0 | 0.15 | 0.01 | 0.03 | 0.81 |
| 7 | evader | pursuer stage2 | 0.06 | 0.90 | 0.04 | 0.00 |
| 7 | evader | pursuer stage4 | 0.08 | 0.88 | 0.05 | 0.00 |
| 7 | evader | pursuer stage6 | 0.09 | 0.87 | 0.06 | 0.00 |
| 8 | pursuer | evader stage1 | 0.85 | 0.03 | 0.12 | 0.00 |
| 8 | pursuer | evader stage3 | 0.89 | 0.06 | 0.05 | 0.00 |
| 8 | pursuer | evader stage5 | 0.83 | 0.10 | 0.08 | 0.00 |
| 8 | pursuer | evader stage7 | 0.50 | 0.27 | 0.25 | 0.00 |
| 9 | evader | pursuer stage0 | 0.19 | 0.01 | 0.04 | 0.77 |
| 9 | evader | pursuer stage2 | 0.04 | 0.93 | 0.03 | 0.00 |
| 9 | evader | pursuer stage4 | 0.08 | 0.89 | 0.04 | 0.00 |
| 9 | evader | pursuer stage6 | 0.11 | 0.84 | 0.06 | 0.00 |
| 9 | evader | pursuer stage8 | 0.29 | 0.56 | 0.16 | 0.00 |
| 10 | pursuer | evader stage1 | 0.86 | 0.02 | 0.13 | 0.00 |
| 10 | pursuer | evader stage3 | 0.91 | 0.04 | 0.05 | 0.00 |
| 10 | pursuer | evader stage5 | 0.78 | 0.13 | 0.11 | 0.00 |
| 10 | pursuer | evader stage7 | 0.52 | 0.24 | 0.26 | 0.00 |
| 10 | pursuer | evader stage9 | 0.54 | 0.29 | 0.19 | 0.00 |
| 11 | evader | pursuer stage0 | 0.10 | 0.00 | 0.04 | 0.86 |
| 11 | evader | pursuer stage2 | 0.04 | 0.94 | 0.03 | 0.00 |
| 11 | evader | pursuer stage4 | 0.08 | 0.88 | 0.04 | 0.00 |
| 11 | evader | pursuer stage6 | 0.12 | 0.84 | 0.06 | 0.00 |
| 11 | evader | pursuer stage8 | 0.22 | 0.64 | 0.15 | 0.01 |
| 11 | evader | pursuer stage10 | 0.33 | 0.47 | 0.21 | 0.01 |
| 12 | pursuer | evader stage1 | 0.87 | 0.02 | 0.12 | 0.00 |
| 12 | pursuer | evader stage3 | 0.93 | 0.03 | 0.04 | 0.00 |
| 12 | pursuer | evader stage5 | 0.69 | 0.20 | 0.12 | 0.00 |
| 12 | pursuer | evader stage7 | 0.52 | 0.27 | 0.24 | 0.00 |
| 12 | pursuer | evader stage9 | 0.63 | 0.21 | 0.17 | 0.00 |
| 12 | pursuer | evader stage11 | 0.57 | 0.26 | 0.17 | 0.00 |

