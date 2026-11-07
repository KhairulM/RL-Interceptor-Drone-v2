# F7_floor_sp summary

Final pursuers (stage 8), clean, 1 campaign each (seed 0; descriptive). plain = thesis-F2-plain (0.10 m, no floor penalty); r15 = 0.15 m capture radius in training; floor_r15 = 0.15 m + pursuer floor penalty. Evaluated at 0.10 m with per-episode files; J at 0.15 m is the exact re-score (`rescore.py`). exploitability = 1 - J vs a fresh exploiter trained under the 0.15 m rules; *_pursuer_crash = pursuer crash rate. other / own campaign and F7 evaders exclude stage-1 evaders.

## Capture radius 0.15 m

| metric | plain | floor_r15 | r15 | floor_r15 - r15 | r15 - plain | floor_r15 - plain |
|---|---|---|---|---|---|---|
| exploitability | 0.002 | 0.968 | 0.963 | +0.005 | +0.961 | +0.966 |
| exploiter_pursuer_crash | 0.006 | 0.970 | 0.965 | +0.005 | +0.959 | +0.964 |
| heldout_rl | 0.959 | 0.827 | 0.934 | -0.107 | -0.025 | -0.132 |
| heldout_worst_cluster | 0.914 | 0.751 | 0.886 | -0.135 | -0.028 | -0.163 |
| other_campaign_evaders | 0.963 | 0.922 | 0.942 | -0.020 | -0.021 | -0.041 |
| other_f7_evaders | 0.901 | 0.850 | 0.844 | +0.006 | -0.056 | -0.051 |
| own_campaign_evaders | 0.968 | 0.966 | 0.944 | +0.022 | -0.024 | -0.002 |
| train | 0.766 | 0.741 | 0.731 | +0.010 | -0.035 | -0.025 |
| heldout_scripted | 0.928 | 0.912 | 0.900 | +0.011 | -0.028 | -0.017 |
| heldout_scripted_pursuer_crash | 0.072 | 0.089 | 0.100 | -0.011 | +0.028 | +0.017 |
| x3_exploiters | 0.988 | 0.983 | 0.975 | +0.008 | -0.013 | -0.005 |

## Capture radius 0.10 m

| metric | plain | floor_r15 | r15 | floor_r15 - r15 | r15 - plain | floor_r15 - plain |
|---|---|---|---|---|---|---|
| exploitability | 0.022 | 0.989 | 0.989 | +0.000 | +0.967 | +0.967 |
| exploiter_pursuer_crash | 0.036 | 0.991 | 0.991 | +0.000 | +0.955 | +0.956 |
| heldout_rl | 0.777 | 0.490 | 0.641 | -0.150 | -0.136 | -0.287 |
| heldout_worst_cluster | 0.646 | 0.303 | 0.518 | -0.215 | -0.128 | -0.343 |
| other_campaign_evaders | 0.850 | 0.650 | 0.708 | -0.059 | -0.141 | -0.200 |
| other_f7_evaders | 0.744 | 0.496 | 0.564 | -0.069 | -0.180 | -0.249 |
| own_campaign_evaders | 0.918 | 0.620 | 0.592 | +0.028 | -0.325 | -0.298 |
| train | 0.601 | 0.520 | 0.510 | +0.009 | -0.091 | -0.082 |
| heldout_scripted | 0.834 | 0.791 | 0.772 | +0.019 | -0.063 | -0.044 |
| heldout_scripted_pursuer_crash | 0.171 | 0.213 | 0.231 | -0.018 | +0.060 | +0.042 |
| x3_exploiters | 0.956 | 0.914 | 0.902 | +0.013 | -0.054 | -0.041 |

## Inside the campaigns (each stage's eval vs its training opponents; capture radius 0.15 m)

**thesis-F7-floor_r15-s0**

| stage | role | vs | capture | pursuer crash | evader crash + OOB | timeout |
|---|---|---|---|---|---|---|
| 0 | pursuer | evader hover | 0.97 | 0.01 | 0.03 | 0.00 |
| 0 | pursuer | evader circular | 0.92 | 0.02 | 0.07 | 0.00 |
| 0 | pursuer | evader flee | 0.94 | 0.01 | 0.05 | 0.00 |
| 0 | pursuer | evader weave | 0.94 | 0.01 | 0.05 | 0.00 |
| 1 | evader | pursuer pure_pursuit | 0.15 | 0.14 | 0.04 | 0.67 |
| 1 | evader | pursuer apf | 0.47 | 0.20 | 0.07 | 0.27 |
| 2 | pursuer | evader stage1 | 0.98 | 0.01 | 0.01 | 0.00 |
| 3 | evader | pursuer stage0 | 0.08 | 0.87 | 0.05 | 0.01 |
| 3 | evader | pursuer stage2 | 0.04 | 0.95 | 0.01 | 0.00 |
| 4 | pursuer | evader stage1 | 0.99 | 0.00 | 0.01 | 0.00 |
| 4 | pursuer | evader stage3 | 0.99 | 0.01 | 0.00 | 0.00 |
| 5 | evader | pursuer stage0 | 0.08 | 0.76 | 0.09 | 0.08 |
| 5 | evader | pursuer stage2 | 0.01 | 0.99 | 0.01 | 0.00 |
| 5 | evader | pursuer stage4 | 0.08 | 0.91 | 0.02 | 0.00 |
| 6 | pursuer | evader stage1 | 0.98 | 0.01 | 0.01 | 0.00 |
| 6 | pursuer | evader stage3 | 1.00 | 0.00 | 0.00 | 0.00 |
| 6 | pursuer | evader stage5 | 0.96 | 0.04 | 0.00 | 0.00 |
| 7 | evader | pursuer stage0 | 0.13 | 0.67 | 0.08 | 0.13 |
| 7 | evader | pursuer stage2 | 0.01 | 0.99 | 0.00 | 0.00 |
| 7 | evader | pursuer stage4 | 0.04 | 0.96 | 0.00 | 0.00 |
| 7 | evader | pursuer stage6 | 0.06 | 0.93 | 0.01 | 0.00 |
| 8 | pursuer | evader stage1 | 0.98 | 0.01 | 0.01 | 0.00 |
| 8 | pursuer | evader stage3 | 1.00 | 0.00 | 0.00 | 0.00 |
| 8 | pursuer | evader stage5 | 0.98 | 0.02 | 0.00 | 0.00 |
| 8 | pursuer | evader stage7 | 0.92 | 0.08 | 0.01 | 0.00 |

**thesis-F7-r15-s0**

| stage | role | vs | capture | pursuer crash | evader crash + OOB | timeout |
|---|---|---|---|---|---|---|
| 0 | pursuer | evader hover | 0.97 | 0.01 | 0.03 | 0.00 |
| 0 | pursuer | evader circular | 0.91 | 0.02 | 0.07 | 0.00 |
| 0 | pursuer | evader flee | 0.94 | 0.01 | 0.05 | 0.00 |
| 0 | pursuer | evader weave | 0.94 | 0.01 | 0.05 | 0.00 |
| 1 | evader | pursuer pure_pursuit | 0.15 | 0.14 | 0.04 | 0.67 |
| 1 | evader | pursuer apf | 0.47 | 0.20 | 0.07 | 0.27 |
| 2 | pursuer | evader stage1 | 0.98 | 0.01 | 0.01 | 0.00 |
| 3 | evader | pursuer stage0 | 0.20 | 0.63 | 0.10 | 0.08 |
| 3 | evader | pursuer stage2 | 0.06 | 0.94 | 0.01 | 0.00 |
| 4 | pursuer | evader stage1 | 0.98 | 0.00 | 0.01 | 0.00 |
| 4 | pursuer | evader stage3 | 0.98 | 0.02 | 0.00 | 0.00 |
| 5 | evader | pursuer stage0 | 0.09 | 0.70 | 0.15 | 0.07 |
| 5 | evader | pursuer stage2 | 0.01 | 0.98 | 0.01 | 0.00 |
| 5 | evader | pursuer stage4 | 0.06 | 0.94 | 0.02 | 0.00 |
| 6 | pursuer | evader stage1 | 0.98 | 0.01 | 0.01 | 0.00 |
| 6 | pursuer | evader stage3 | 0.99 | 0.01 | 0.00 | 0.00 |
| 6 | pursuer | evader stage5 | 0.95 | 0.04 | 0.00 | 0.00 |
| 7 | evader | pursuer stage0 | 0.16 | 0.63 | 0.16 | 0.07 |
| 7 | evader | pursuer stage2 | 0.01 | 0.98 | 0.01 | 0.00 |
| 7 | evader | pursuer stage4 | 0.02 | 0.97 | 0.01 | 0.00 |
| 7 | evader | pursuer stage6 | 0.12 | 0.87 | 0.01 | 0.00 |
| 8 | pursuer | evader stage1 | 0.98 | 0.01 | 0.01 | 0.00 |
| 8 | pursuer | evader stage3 | 0.99 | 0.01 | 0.00 | 0.00 |
| 8 | pursuer | evader stage5 | 0.97 | 0.03 | 0.00 | 0.00 |
| 8 | pursuer | evader stage7 | 0.86 | 0.14 | 0.01 | 0.00 |

