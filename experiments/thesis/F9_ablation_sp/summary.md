# F9_ablation_sp summary

Final pursuers (stage 12) of four campaigns, seed 0, Hedge both roles, 13 stages x 50M, GAE truncation fix: control = thesis rules; no_timeout = timeout gives 0 to both roles; beta = pursuer capture reward x Roncero's closing-speed factor; endings = crashes / evader leaving are truncations. All evals clean under the thesis rules; exploitability_thesis = 1 - J vs a fresh exploiter trained under the thesis rules (luring pays). other_f3 / other_f9 / own evaders exclude stage-1 evaders. Reference columns: F8 (26 x 50M, all of Roncero's rules) and F3 hedge (9 x 300M, thesis rules), seed 0. 1 seed, descriptive.

## Capture radius 0.10 m

| metric | control | no_timeout | beta | endings | no_timeout - control | beta - control | endings - control | F8 roncero s0 | F3 hedge s0 |
|---|---|---|---|---|---|---|---|---|---|
| heldout_rl | 0.893 | 0.916 | 0.941 | 0.890 | +0.024 | +0.049 | -0.003 | 0.948 | 0.791 |
| heldout_worst_cluster | 0.859 | 0.884 | 0.914 | 0.849 | +0.025 | +0.054 | -0.010 | 0.917 | 0.692 |
| other_f3_evaders | 0.766 | 0.845 | 0.836 | 0.702 | +0.079 | +0.071 | -0.064 | 0.887 | 0.871 |
| other_f9_evaders | 0.866 | 0.887 | 0.879 | 0.714 | +0.021 | +0.013 | -0.151 | - | - |
| own_campaign_evaders | 0.933 | 0.872 | 0.959 | 0.993 | -0.061 | +0.026 | +0.060 | 0.916 | 0.957 |
| train | 0.688 | 0.718 | 0.859 | 0.756 | +0.030 | +0.171 | +0.069 | 0.877 | 0.594 |
| heldout_scripted | 0.918 | 0.917 | 0.969 | 0.976 | -0.001 | +0.050 | +0.057 | 0.983 | 0.808 |
| heldout_scripted_pursuer_crash | 0.085 | 0.085 | 0.033 | 0.025 | -0.000 | -0.053 | -0.061 | 0.018 | 0.197 |
| x3_exploiters | 0.983 | 0.983 | 0.996 | 0.999 | -0.000 | +0.012 | +0.016 | 0.999 | 0.955 |
| exploitability_thesis | 0.440 | 0.950 | 0.370 | 0.823 | +0.510 | -0.070 | +0.384 | 0.631 | 0.974 |
| exploiter_thesis_pursuer_crash | 0.056 | 0.955 | 0.395 | 0.834 | +0.898 | +0.339 | +0.778 | 0.010 | 0.975 |
| exploiter_thesis_timeout | 0.385 | 0.000 | 0.000 | 0.000 | -0.385 | -0.385 | -0.385 | - | - |

## Capture radius 0.15 m

| metric | control | no_timeout | beta | endings | no_timeout - control | beta - control | endings - control | F8 roncero s0 | F3 hedge s0 |
|---|---|---|---|---|---|---|---|---|---|
| heldout_rl | 0.989 | 0.995 | 0.996 | 0.992 | +0.005 | +0.006 | +0.003 | 0.995 | 0.960 |
| heldout_worst_cluster | 0.973 | 0.965 | 0.992 | 0.989 | -0.008 | +0.019 | +0.017 | 0.993 | 0.874 |
| other_f3_evaders | 0.948 | 0.970 | 0.965 | 0.924 | +0.023 | +0.017 | -0.024 | 0.975 | 0.957 |
| other_f9_evaders | 0.972 | 0.980 | 0.975 | 0.901 | +0.008 | +0.003 | -0.071 | - | - |
| own_campaign_evaders | 0.995 | 0.981 | 0.997 | 1.000 | -0.013 | +0.002 | +0.005 | 0.939 | 0.993 |
| train | 0.840 | 0.847 | 0.959 | 0.928 | +0.006 | +0.119 | +0.087 | 0.963 | 0.759 |
| heldout_scripted | 0.978 | 0.978 | 0.995 | 0.995 | +0.000 | +0.017 | +0.017 | 0.997 | 0.905 |
| heldout_scripted_pursuer_crash | 0.023 | 0.022 | 0.005 | 0.005 | -0.001 | -0.018 | -0.018 | 0.003 | 0.096 |
| x3_exploiters | 0.998 | 0.999 | 1.000 | 1.000 | +0.001 | +0.002 | +0.002 | 1.000 | 0.987 |
| exploitability_thesis | 0.095 | 0.571 | 0.085 | 0.127 | +0.477 | -0.010 | +0.032 | 0.000 | 0.773 |
| exploiter_thesis_pursuer_crash | 0.008 | 0.572 | 0.090 | 0.128 | +0.563 | +0.081 | +0.119 | 0.000 | 0.774 |
| exploiter_thesis_timeout | 0.086 | 0.000 | 0.000 | 0.000 | -0.086 | -0.086 | -0.086 | - | - |

## In-campaign lure rates (pursuer crash caused by each evader stage in the stage >= 2 pursuers)

| arm | evader stage | mean | min - max | n pursuers |
|---|---|---|---|---|
| control | 3 | 0.09 | 0.09 - 0.09 | 1 |
| control | 5 | 0.41 | 0.27 - 0.54 | 2 |
| control | 7 | 0.37 | 0.24 - 0.60 | 3 |
| control | 9 | 0.42 | 0.16 - 0.63 | 4 |
| control | 11 | 0.53 | 0.22 - 0.69 | 5 |
| no_timeout | 3 | 0.67 | 0.67 - 0.67 | 1 |
| no_timeout | 5 | 0.82 | 0.78 - 0.86 | 2 |
| no_timeout | 7 | 0.75 | 0.50 - 0.91 | 3 |
| no_timeout | 9 | 0.73 | 0.56 - 0.93 | 4 |
| no_timeout | 11 | 0.79 | 0.65 - 0.93 | 5 |
| beta | 3 | 0.03 | 0.03 - 0.03 | 1 |
| beta | 5 | 0.04 | 0.03 - 0.05 | 2 |
| beta | 7 | 0.20 | 0.05 - 0.38 | 3 |
| beta | 9 | 0.28 | 0.02 - 0.63 | 4 |
| beta | 11 | 0.33 | 0.03 - 0.71 | 5 |
| endings | 3 | 0.00 | 0.00 - 0.00 | 1 |
| endings | 5 | 0.02 | 0.01 - 0.03 | 2 |
| endings | 7 | 0.05 | 0.00 - 0.08 | 3 |
| endings | 9 | 0.04 | 0.00 - 0.07 | 4 |
| endings | 11 | 0.03 | 0.00 - 0.06 | 5 |

Pre-registered measure (a): in-campaign lure rate, mean over evader stages 3-11 of each stage's mean over the stage >= 2 pursuers (lure present if >= 0.50): control 0.36, no_timeout 0.75, beta 0.18, endings 0.03


## Inside the campaigns (each stage's eval vs its training opponents)

**thesis-F9-control-s0**

| stage | role | vs | capture | pursuer crash | evader crash + OOB | timeout |
|---|---|---|---|---|---|---|
| 0 | pursuer | evader hover | 0.92 | 0.01 | 0.07 | 0.00 |
| 0 | pursuer | evader circular | 0.86 | 0.01 | 0.14 | 0.00 |
| 0 | pursuer | evader flee | 0.89 | 0.01 | 0.11 | 0.00 |
| 0 | pursuer | evader weave | 0.89 | 0.01 | 0.11 | 0.00 |
| 1 | evader | pursuer pure_pursuit | 0.09 | 0.15 | 0.04 | 0.72 |
| 1 | evader | pursuer apf | 0.33 | 0.22 | 0.08 | 0.37 |
| 2 | pursuer | evader stage1 | 0.95 | 0.01 | 0.03 | 0.00 |
| 3 | evader | pursuer stage0 | 0.15 | 0.01 | 0.20 | 0.64 |
| 3 | evader | pursuer stage2 | 0.30 | 0.09 | 0.25 | 0.37 |
| 4 | pursuer | evader stage1 | 0.95 | 0.02 | 0.03 | 0.00 |
| 4 | pursuer | evader stage3 | 0.85 | 0.02 | 0.13 | 0.00 |
| 5 | evader | pursuer stage0 | 0.10 | 0.01 | 0.09 | 0.80 |
| 5 | evader | pursuer stage2 | 0.11 | 0.27 | 0.16 | 0.47 |
| 5 | evader | pursuer stage4 | 0.39 | 0.54 | 0.08 | 0.00 |
| 6 | pursuer | evader stage1 | 0.94 | 0.02 | 0.04 | 0.00 |
| 6 | pursuer | evader stage3 | 0.86 | 0.01 | 0.13 | 0.00 |
| 6 | pursuer | evader stage5 | 0.90 | 0.07 | 0.04 | 0.00 |
| 7 | evader | pursuer stage0 | 0.22 | 0.01 | 0.06 | 0.71 |
| 7 | evader | pursuer stage2 | 0.26 | 0.24 | 0.16 | 0.36 |
| 7 | evader | pursuer stage4 | 0.30 | 0.60 | 0.12 | 0.00 |
| 7 | evader | pursuer stage6 | 0.64 | 0.28 | 0.09 | 0.00 |
| 8 | pursuer | evader stage1 | 0.93 | 0.02 | 0.05 | 0.00 |
| 8 | pursuer | evader stage3 | 0.86 | 0.01 | 0.13 | 0.00 |
| 8 | pursuer | evader stage5 | 0.93 | 0.03 | 0.04 | 0.00 |
| 8 | pursuer | evader stage7 | 0.87 | 0.08 | 0.06 | 0.00 |
| 9 | evader | pursuer stage0 | 0.17 | 0.01 | 0.06 | 0.76 |
| 9 | evader | pursuer stage2 | 0.12 | 0.16 | 0.16 | 0.57 |
| 9 | evader | pursuer stage4 | 0.22 | 0.63 | 0.17 | 0.00 |
| 9 | evader | pursuer stage6 | 0.33 | 0.55 | 0.14 | 0.00 |
| 9 | evader | pursuer stage8 | 0.51 | 0.35 | 0.16 | 0.00 |
| 10 | pursuer | evader stage1 | 0.91 | 0.02 | 0.07 | 0.00 |
| 10 | pursuer | evader stage3 | 0.83 | 0.01 | 0.16 | 0.00 |
| 10 | pursuer | evader stage5 | 0.95 | 0.02 | 0.03 | 0.00 |
| 10 | pursuer | evader stage7 | 0.90 | 0.05 | 0.05 | 0.00 |
| 10 | pursuer | evader stage9 | 0.77 | 0.13 | 0.11 | 0.00 |
| 11 | evader | pursuer stage0 | 0.10 | 0.02 | 0.07 | 0.81 |
| 11 | evader | pursuer stage2 | 0.09 | 0.22 | 0.15 | 0.54 |
| 11 | evader | pursuer stage4 | 0.17 | 0.69 | 0.15 | 0.01 |
| 11 | evader | pursuer stage6 | 0.21 | 0.68 | 0.12 | 0.00 |
| 11 | evader | pursuer stage8 | 0.25 | 0.63 | 0.14 | 0.00 |
| 11 | evader | pursuer stage10 | 0.48 | 0.39 | 0.16 | 0.00 |
| 12 | pursuer | evader stage1 | 0.89 | 0.03 | 0.09 | 0.00 |
| 12 | pursuer | evader stage3 | 0.82 | 0.01 | 0.17 | 0.00 |
| 12 | pursuer | evader stage5 | 0.94 | 0.03 | 0.04 | 0.00 |
| 12 | pursuer | evader stage7 | 0.88 | 0.05 | 0.07 | 0.00 |
| 12 | pursuer | evader stage9 | 0.78 | 0.11 | 0.13 | 0.00 |
| 12 | pursuer | evader stage11 | 0.68 | 0.18 | 0.17 | 0.00 |

**thesis-F9-no_timeout-s0**

| stage | role | vs | capture | pursuer crash | evader crash + OOB | timeout |
|---|---|---|---|---|---|---|
| 0 | pursuer | evader hover | 0.92 | 0.00 | 0.08 | 0.00 |
| 0 | pursuer | evader circular | 0.87 | 0.00 | 0.12 | 0.00 |
| 0 | pursuer | evader flee | 0.89 | 0.00 | 0.11 | 0.00 |
| 0 | pursuer | evader weave | 0.89 | 0.00 | 0.11 | 0.00 |
| 1 | evader | pursuer pure_pursuit | 0.17 | 0.15 | 0.04 | 0.65 |
| 1 | evader | pursuer apf | 0.28 | 0.26 | 0.09 | 0.37 |
| 2 | pursuer | evader stage1 | 0.95 | 0.01 | 0.04 | 0.00 |
| 3 | evader | pursuer stage0 | 0.09 | 0.01 | 0.03 | 0.87 |
| 3 | evader | pursuer stage2 | 0.25 | 0.67 | 0.08 | 0.01 |
| 4 | pursuer | evader stage1 | 0.95 | 0.01 | 0.04 | 0.00 |
| 4 | pursuer | evader stage3 | 0.94 | 0.04 | 0.02 | 0.00 |
| 5 | evader | pursuer stage0 | 0.14 | 0.00 | 0.02 | 0.84 |
| 5 | evader | pursuer stage2 | 0.14 | 0.78 | 0.08 | 0.01 |
| 5 | evader | pursuer stage4 | 0.12 | 0.86 | 0.03 | 0.00 |
| 6 | pursuer | evader stage1 | 0.92 | 0.02 | 0.06 | 0.00 |
| 6 | pursuer | evader stage3 | 0.93 | 0.05 | 0.03 | 0.00 |
| 6 | pursuer | evader stage5 | 0.91 | 0.07 | 0.03 | 0.00 |
| 7 | evader | pursuer stage0 | 0.15 | 0.01 | 0.05 | 0.79 |
| 7 | evader | pursuer stage2 | 0.11 | 0.50 | 0.13 | 0.28 |
| 7 | evader | pursuer stage4 | 0.05 | 0.91 | 0.05 | 0.00 |
| 7 | evader | pursuer stage6 | 0.10 | 0.85 | 0.08 | 0.00 |
| 8 | pursuer | evader stage1 | 0.91 | 0.02 | 0.07 | 0.00 |
| 8 | pursuer | evader stage3 | 0.94 | 0.04 | 0.03 | 0.00 |
| 8 | pursuer | evader stage5 | 0.93 | 0.05 | 0.02 | 0.00 |
| 8 | pursuer | evader stage7 | 0.68 | 0.24 | 0.10 | 0.00 |
| 9 | evader | pursuer stage0 | 0.10 | 0.00 | 0.04 | 0.86 |
| 9 | evader | pursuer stage2 | 0.07 | 0.58 | 0.09 | 0.27 |
| 9 | evader | pursuer stage4 | 0.04 | 0.93 | 0.04 | 0.00 |
| 9 | evader | pursuer stage6 | 0.08 | 0.87 | 0.07 | 0.00 |
| 9 | evader | pursuer stage8 | 0.33 | 0.56 | 0.14 | 0.00 |
| 10 | pursuer | evader stage1 | 0.90 | 0.02 | 0.08 | 0.00 |
| 10 | pursuer | evader stage3 | 0.94 | 0.03 | 0.03 | 0.00 |
| 10 | pursuer | evader stage5 | 0.96 | 0.03 | 0.02 | 0.00 |
| 10 | pursuer | evader stage7 | 0.81 | 0.13 | 0.07 | 0.00 |
| 10 | pursuer | evader stage9 | 0.67 | 0.25 | 0.11 | 0.00 |
| 11 | evader | pursuer stage0 | 0.13 | 0.00 | 0.04 | 0.83 |
| 11 | evader | pursuer stage2 | 0.07 | 0.71 | 0.09 | 0.13 |
| 11 | evader | pursuer stage4 | 0.04 | 0.93 | 0.05 | 0.00 |
| 11 | evader | pursuer stage6 | 0.06 | 0.91 | 0.05 | 0.00 |
| 11 | evader | pursuer stage8 | 0.16 | 0.74 | 0.13 | 0.00 |
| 11 | evader | pursuer stage10 | 0.26 | 0.65 | 0.13 | 0.00 |
| 12 | pursuer | evader stage1 | 0.89 | 0.03 | 0.08 | 0.00 |
| 12 | pursuer | evader stage3 | 0.92 | 0.04 | 0.04 | 0.00 |
| 12 | pursuer | evader stage5 | 0.95 | 0.03 | 0.02 | 0.00 |
| 12 | pursuer | evader stage7 | 0.74 | 0.21 | 0.08 | 0.00 |
| 12 | pursuer | evader stage9 | 0.76 | 0.17 | 0.09 | 0.00 |
| 12 | pursuer | evader stage11 | 0.62 | 0.28 | 0.12 | 0.00 |

**thesis-F9-beta-s0**

| stage | role | vs | capture | pursuer crash | evader crash + OOB | timeout |
|---|---|---|---|---|---|---|
| 0 | pursuer | evader hover | 0.93 | 0.00 | 0.07 | 0.00 |
| 0 | pursuer | evader circular | 0.87 | 0.00 | 0.12 | 0.00 |
| 0 | pursuer | evader flee | 0.89 | 0.00 | 0.11 | 0.00 |
| 0 | pursuer | evader weave | 0.89 | 0.00 | 0.10 | 0.00 |
| 1 | evader | pursuer pure_pursuit | 0.09 | 0.15 | 0.04 | 0.72 |
| 1 | evader | pursuer apf | 0.33 | 0.22 | 0.08 | 0.37 |
| 2 | pursuer | evader stage1 | 0.96 | 0.01 | 0.03 | 0.00 |
| 3 | evader | pursuer stage0 | 0.03 | 0.00 | 0.04 | 0.92 |
| 3 | evader | pursuer stage2 | 0.46 | 0.03 | 0.13 | 0.38 |
| 4 | pursuer | evader stage1 | 0.95 | 0.01 | 0.04 | 0.00 |
| 4 | pursuer | evader stage3 | 0.94 | 0.02 | 0.05 | 0.00 |
| 5 | evader | pursuer stage0 | 0.47 | 0.00 | 0.08 | 0.45 |
| 5 | evader | pursuer stage2 | 0.50 | 0.03 | 0.12 | 0.35 |
| 5 | evader | pursuer stage4 | 0.47 | 0.05 | 0.17 | 0.32 |
| 6 | pursuer | evader stage1 | 0.94 | 0.01 | 0.05 | 0.00 |
| 6 | pursuer | evader stage3 | 0.92 | 0.02 | 0.06 | 0.00 |
| 6 | pursuer | evader stage5 | 0.93 | 0.02 | 0.06 | 0.00 |
| 7 | evader | pursuer stage0 | 0.09 | 0.01 | 0.05 | 0.85 |
| 7 | evader | pursuer stage2 | 0.12 | 0.05 | 0.09 | 0.74 |
| 7 | evader | pursuer stage4 | 0.11 | 0.16 | 0.08 | 0.65 |
| 7 | evader | pursuer stage6 | 0.47 | 0.38 | 0.15 | 0.00 |
| 8 | pursuer | evader stage1 | 0.95 | 0.01 | 0.04 | 0.00 |
| 8 | pursuer | evader stage3 | 0.92 | 0.02 | 0.06 | 0.00 |
| 8 | pursuer | evader stage5 | 0.94 | 0.01 | 0.05 | 0.00 |
| 8 | pursuer | evader stage7 | 0.85 | 0.06 | 0.09 | 0.00 |
| 9 | evader | pursuer stage0 | 0.10 | 0.01 | 0.04 | 0.85 |
| 9 | evader | pursuer stage2 | 0.14 | 0.02 | 0.09 | 0.76 |
| 9 | evader | pursuer stage4 | 0.16 | 0.19 | 0.08 | 0.57 |
| 9 | evader | pursuer stage6 | 0.27 | 0.63 | 0.10 | 0.00 |
| 9 | evader | pursuer stage8 | 0.57 | 0.30 | 0.14 | 0.00 |
| 10 | pursuer | evader stage1 | 0.92 | 0.01 | 0.06 | 0.00 |
| 10 | pursuer | evader stage3 | 0.91 | 0.02 | 0.07 | 0.00 |
| 10 | pursuer | evader stage5 | 0.92 | 0.01 | 0.07 | 0.00 |
| 10 | pursuer | evader stage7 | 0.86 | 0.05 | 0.09 | 0.00 |
| 10 | pursuer | evader stage9 | 0.84 | 0.08 | 0.09 | 0.00 |
| 11 | evader | pursuer stage0 | 0.08 | 0.00 | 0.05 | 0.87 |
| 11 | evader | pursuer stage2 | 0.17 | 0.03 | 0.05 | 0.75 |
| 11 | evader | pursuer stage4 | 0.13 | 0.17 | 0.04 | 0.66 |
| 11 | evader | pursuer stage6 | 0.21 | 0.71 | 0.08 | 0.01 |
| 11 | evader | pursuer stage8 | 0.36 | 0.51 | 0.14 | 0.00 |
| 11 | evader | pursuer stage10 | 0.64 | 0.23 | 0.14 | 0.00 |
| 12 | pursuer | evader stage1 | 0.92 | 0.02 | 0.06 | 0.00 |
| 12 | pursuer | evader stage3 | 0.90 | 0.03 | 0.08 | 0.00 |
| 12 | pursuer | evader stage5 | 0.91 | 0.02 | 0.07 | 0.00 |
| 12 | pursuer | evader stage7 | 0.86 | 0.05 | 0.09 | 0.00 |
| 12 | pursuer | evader stage9 | 0.89 | 0.05 | 0.07 | 0.00 |
| 12 | pursuer | evader stage11 | 0.85 | 0.08 | 0.07 | 0.00 |

**thesis-F9-endings-s0**

| stage | role | vs | capture | pursuer crash | evader crash + OOB | timeout |
|---|---|---|---|---|---|---|
| 0 | pursuer | evader hover | 0.90 | 0.00 | 0.09 | 0.00 |
| 0 | pursuer | evader circular | 0.86 | 0.00 | 0.14 | 0.00 |
| 0 | pursuer | evader flee | 0.87 | 0.00 | 0.13 | 0.00 |
| 0 | pursuer | evader weave | 0.87 | 0.00 | 0.13 | 0.00 |
| 1 | evader | pursuer pure_pursuit | 0.06 | 0.14 | 0.03 | 0.77 |
| 1 | evader | pursuer apf | 0.21 | 0.18 | 0.07 | 0.55 |
| 2 | pursuer | evader stage1 | 0.94 | 0.00 | 0.06 | 0.00 |
| 3 | evader | pursuer stage0 | 0.10 | 0.00 | 0.04 | 0.85 |
| 3 | evader | pursuer stage2 | 0.07 | 0.00 | 0.04 | 0.89 |
| 4 | pursuer | evader stage1 | 0.93 | 0.00 | 0.06 | 0.00 |
| 4 | pursuer | evader stage3 | 0.96 | 0.00 | 0.04 | 0.00 |
| 5 | evader | pursuer stage0 | 0.10 | 0.01 | 0.04 | 0.85 |
| 5 | evader | pursuer stage2 | 0.08 | 0.01 | 0.06 | 0.85 |
| 5 | evader | pursuer stage4 | 0.32 | 0.03 | 0.04 | 0.61 |
| 6 | pursuer | evader stage1 | 0.94 | 0.01 | 0.05 | 0.00 |
| 6 | pursuer | evader stage3 | 0.96 | 0.00 | 0.03 | 0.00 |
| 6 | pursuer | evader stage5 | 0.97 | 0.01 | 0.02 | 0.00 |
| 7 | evader | pursuer stage0 | 0.24 | 0.01 | 0.08 | 0.67 |
| 7 | evader | pursuer stage2 | 0.13 | 0.00 | 0.05 | 0.81 |
| 7 | evader | pursuer stage4 | 0.28 | 0.08 | 0.04 | 0.60 |
| 7 | evader | pursuer stage6 | 0.55 | 0.08 | 0.03 | 0.34 |
| 8 | pursuer | evader stage1 | 0.95 | 0.00 | 0.04 | 0.00 |
| 8 | pursuer | evader stage3 | 0.97 | 0.00 | 0.03 | 0.00 |
| 8 | pursuer | evader stage5 | 0.96 | 0.01 | 0.03 | 0.00 |
| 8 | pursuer | evader stage7 | 0.97 | 0.01 | 0.02 | 0.00 |
| 9 | evader | pursuer stage0 | 0.24 | 0.01 | 0.04 | 0.71 |
| 9 | evader | pursuer stage2 | 0.16 | 0.00 | 0.02 | 0.82 |
| 9 | evader | pursuer stage4 | 0.25 | 0.03 | 0.03 | 0.69 |
| 9 | evader | pursuer stage6 | 0.42 | 0.05 | 0.02 | 0.50 |
| 9 | evader | pursuer stage8 | 0.90 | 0.07 | 0.04 | 0.00 |
| 10 | pursuer | evader stage1 | 0.97 | 0.00 | 0.03 | 0.00 |
| 10 | pursuer | evader stage3 | 0.98 | 0.00 | 0.02 | 0.00 |
| 10 | pursuer | evader stage5 | 0.97 | 0.01 | 0.02 | 0.00 |
| 10 | pursuer | evader stage7 | 0.98 | 0.01 | 0.01 | 0.00 |
| 10 | pursuer | evader stage9 | 0.98 | 0.01 | 0.01 | 0.00 |
| 11 | evader | pursuer stage0 | 0.47 | 0.01 | 0.03 | 0.49 |
| 11 | evader | pursuer stage2 | 0.13 | 0.00 | 0.01 | 0.86 |
| 11 | evader | pursuer stage4 | 0.28 | 0.05 | 0.02 | 0.65 |
| 11 | evader | pursuer stage6 | 0.61 | 0.05 | 0.02 | 0.32 |
| 11 | evader | pursuer stage8 | 0.92 | 0.06 | 0.02 | 0.00 |
| 11 | evader | pursuer stage10 | 0.98 | 0.01 | 0.01 | 0.00 |
| 12 | pursuer | evader stage1 | 0.97 | 0.00 | 0.03 | 0.00 |
| 12 | pursuer | evader stage3 | 0.98 | 0.00 | 0.02 | 0.00 |
| 12 | pursuer | evader stage5 | 0.97 | 0.01 | 0.02 | 0.00 |
| 12 | pursuer | evader stage7 | 0.98 | 0.01 | 0.01 | 0.00 |
| 12 | pursuer | evader stage9 | 0.98 | 0.01 | 0.01 | 0.00 |
| 12 | pursuer | evader stage11 | 0.99 | 0.01 | 0.00 | 0.00 |

