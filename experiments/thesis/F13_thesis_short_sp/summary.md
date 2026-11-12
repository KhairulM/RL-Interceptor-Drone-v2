# F13_thesis_short_sp summary

Final pursuers (stage 12) of thesis-F9-control-s0 / thesis-F13-short-s{100,200} (thesis rules, GAE fix, Hedge both roles, 13 x 50M) and F11's v2d final pursuers (stage 24). Evals clean under the thesis rules; exploitability_thesis = 1 - J vs a 100M-frame thesis-rules exploiter (F11: exploitability_old, same recipe); _v2d vs a v2d-rules exploiter (F11: F11c's v2 exploiter). cross_play = J vs the other rule set's campaign evaders. 0.10 m primary.

**Decision (pre-registered)**: v2d better on 4 of 4 -> **resume F12**

| measure | v2d mean | F13 mean | v2d better |
|---|---|---|---|
| heldout_rl | 0.872 | 0.871 | True |
| strong_finals_mean | 0.798 | 0.601 | True |
| exploitability_thesis | 0.458 | 0.699 | True |
| cross_play | 0.913 | 0.723 | True |

In-campaign lure rate (a) (mean over evader stages 3-11): thesis-F9-control-s0 0.36, thesis-F13-short-s100 0.14, thesis-F13-short-s200 0.15

## Capture radius 0.10 m

| metric | f13/0 | f13/100 | f13/200 | f11/0 | f11/1 |
|---|---|---|---|---|---|
| heldout_rl | 0.893 | 0.894 | 0.826 | 0.875 | 0.869 |
| heldout_worst_cluster | 0.859 | 0.854 | 0.757 | 0.838 | 0.832 |
| strong_finals_mean | 0.656 | 0.583 | 0.563 | 0.845 | 0.751 |
| strong_finals_min | 0.404 | 0.376 | 0.275 | 0.501 | 0.346 |
| vs_f8_finals | 0.792 | 0.713 | 0.804 | 0.868 | 0.956 |
| vs_f3_hedge_finals | 0.521 | 0.466 | 0.439 | 0.858 | 0.765 |
| other_f3_evaders | 0.766 | 0.664 | 0.619 | 0.892 | 0.846 |
| other_f13_evaders | 0.937 | 0.831 | 0.827 | - | - |
| own_campaign_evaders | 0.934 | 0.946 | 0.932 | 0.952 | 0.957 |
| cross_play | 0.817 | 0.703 | 0.650 | 0.912 | 0.914 |
| train | 0.688 | 0.717 | 0.694 | 0.833 | 0.678 |
| heldout_scripted | 0.918 | 0.948 | 0.947 | 0.976 | 0.971 |
| heldout_scripted_pursuer_crash | 0.085 | 0.054 | 0.055 | 0.025 | 0.030 |
| x3_exploiters | 0.983 | 0.992 | 0.992 | 0.998 | 0.999 |
| exploitability_thesis | 0.440 | 0.915 | 0.741 | 0.293 | 0.624 |
| exploiter_thesis_pursuer_crash | 0.056 | 0.925 | 0.751 | 0.309 | 0.636 |
| exploiter_thesis_timeout | 0.385 | 0.000 | 0.000 | - | - |
| exploitability_v2d | 0.429 | 0.775 | 0.461 | - | - |
| exploiter_v2d_pursuer_crash | 0.263 | 0.396 | 0.208 | - | - |
| exploiter_v2d_timeout | 0.168 | 0.380 | 0.254 | - | - |

F11 stage-12 pursuers (same stage count as F13): f11/0: heldout_rl 0.930, strong_finals_mean 0.877, cross_play 0.917; f11/1: heldout_rl 0.797, strong_finals_mean 0.677, cross_play 0.845

## Capture radius 0.15 m

| metric | f13/0 | f13/100 | f13/200 | f11/0 | f11/1 |
|---|---|---|---|---|---|
| heldout_rl | 0.989 | 0.991 | 0.977 | 0.980 | 0.987 |
| heldout_worst_cluster | 0.973 | 0.982 | 0.969 | 0.976 | 0.982 |
| strong_finals_mean | 0.891 | 0.850 | 0.808 | 0.947 | 0.922 |
| strong_finals_min | 0.710 | 0.735 | 0.674 | 0.789 | 0.773 |
| vs_f8_finals | 0.943 | 0.898 | 0.954 | 0.973 | 0.993 |
| vs_f3_hedge_finals | 0.840 | 0.785 | 0.691 | 0.930 | 0.908 |
| other_f3_evaders | 0.948 | 0.911 | 0.869 | 0.972 | 0.964 |
| other_f13_evaders | 0.996 | 0.964 | 0.957 | - | - |
| own_campaign_evaders | 0.995 | 0.996 | 0.996 | 0.996 | 0.988 |
| cross_play | 0.981 | 0.953 | 0.935 | 0.985 | 0.989 |
| train | 0.840 | 0.881 | 0.866 | 0.927 | 0.814 |
| heldout_scripted | 0.978 | 0.988 | 0.987 | 0.996 | 0.995 |
| heldout_scripted_pursuer_crash | 0.023 | 0.013 | 0.013 | 0.004 | 0.005 |
| x3_exploiters | 0.998 | 1.000 | 0.999 | 1.000 | 1.000 |
| exploitability_thesis | 0.095 | 0.476 | 0.206 | 0.031 | 0.208 |
| exploiter_thesis_pursuer_crash | 0.008 | 0.477 | 0.208 | 0.033 | 0.188 |
| exploiter_thesis_timeout | 0.086 | 0.000 | 0.000 | - | - |
| exploitability_v2d | 0.016 | 0.329 | 0.007 | - | - |
| exploiter_v2d_pursuer_crash | 0.015 | 0.193 | 0.007 | - | - |
| exploiter_v2d_timeout | 0.000 | 0.137 | 0.000 | - | - |

F11 stage-12 pursuers (same stage count as F13): f11/0: heldout_rl 0.990, strong_finals_mean 0.958, cross_play 0.982; f11/1: heldout_rl 0.960, strong_finals_mean 0.838, cross_play 0.966

## In-campaign lure rates

| campaign | evader stage | mean | min - max | n pursuers |
|---|---|---|---|---|
| thesis-F9-control-s0 | 3 | 0.09 | 0.09 - 0.09 | 1 |
| thesis-F9-control-s0 | 5 | 0.41 | 0.27 - 0.54 | 2 |
| thesis-F9-control-s0 | 7 | 0.37 | 0.24 - 0.60 | 3 |
| thesis-F9-control-s0 | 9 | 0.42 | 0.16 - 0.63 | 4 |
| thesis-F9-control-s0 | 11 | 0.53 | 0.22 - 0.69 | 5 |
| thesis-F13-short-s100 | 3 | 0.04 | 0.04 - 0.04 | 1 |
| thesis-F13-short-s100 | 5 | 0.06 | 0.04 - 0.07 | 2 |
| thesis-F13-short-s100 | 7 | 0.08 | 0.03 - 0.14 | 3 |
| thesis-F13-short-s100 | 9 | 0.19 | 0.05 - 0.32 | 4 |
| thesis-F13-short-s100 | 11 | 0.31 | 0.06 - 0.59 | 5 |
| thesis-F13-short-s200 | 3 | 0.07 | 0.07 - 0.07 | 1 |
| thesis-F13-short-s200 | 5 | 0.20 | 0.19 - 0.22 | 2 |
| thesis-F13-short-s200 | 7 | 0.11 | 0.10 - 0.13 | 3 |
| thesis-F13-short-s200 | 9 | 0.17 | 0.14 - 0.21 | 4 |
| thesis-F13-short-s200 | 11 | 0.20 | 0.18 - 0.23 | 5 |

## Inside the campaigns

**thesis-F13-short-s100**

| stage | role | vs | capture | pursuer crash | evader crash + OOB | timeout |
|---|---|---|---|---|---|---|
| 0 | pursuer | evader hover | 0.93 | 0.00 | 0.07 | 0.00 |
| 0 | pursuer | evader circular | 0.88 | 0.00 | 0.12 | 0.00 |
| 0 | pursuer | evader flee | 0.89 | 0.00 | 0.11 | 0.00 |
| 0 | pursuer | evader weave | 0.89 | 0.00 | 0.11 | 0.00 |
| 1 | evader | pursuer pure_pursuit | 0.06 | 0.15 | 0.05 | 0.74 |
| 1 | evader | pursuer apf | 0.37 | 0.24 | 0.08 | 0.30 |
| 2 | pursuer | evader stage1 | 0.94 | 0.01 | 0.05 | 0.00 |
| 3 | evader | pursuer stage0 | 0.22 | 0.00 | 0.05 | 0.73 |
| 3 | evader | pursuer stage2 | 0.59 | 0.04 | 0.11 | 0.26 |
| 4 | pursuer | evader stage1 | 0.95 | 0.01 | 0.04 | 0.00 |
| 4 | pursuer | evader stage3 | 0.94 | 0.03 | 0.04 | 0.00 |
| 5 | evader | pursuer stage0 | 0.14 | 0.00 | 0.07 | 0.79 |
| 5 | evader | pursuer stage2 | 0.31 | 0.04 | 0.11 | 0.54 |
| 5 | evader | pursuer stage4 | 0.45 | 0.07 | 0.11 | 0.37 |
| 6 | pursuer | evader stage1 | 0.95 | 0.01 | 0.04 | 0.00 |
| 6 | pursuer | evader stage3 | 0.91 | 0.03 | 0.06 | 0.00 |
| 6 | pursuer | evader stage5 | 0.90 | 0.05 | 0.06 | 0.00 |
| 7 | evader | pursuer stage0 | 0.11 | 0.00 | 0.04 | 0.84 |
| 7 | evader | pursuer stage2 | 0.19 | 0.03 | 0.07 | 0.71 |
| 7 | evader | pursuer stage4 | 0.30 | 0.07 | 0.10 | 0.54 |
| 7 | evader | pursuer stage6 | 0.76 | 0.14 | 0.10 | 0.00 |
| 8 | pursuer | evader stage1 | 0.94 | 0.01 | 0.05 | 0.00 |
| 8 | pursuer | evader stage3 | 0.91 | 0.04 | 0.06 | 0.00 |
| 8 | pursuer | evader stage5 | 0.92 | 0.04 | 0.05 | 0.00 |
| 8 | pursuer | evader stage7 | 0.89 | 0.06 | 0.06 | 0.00 |
| 9 | evader | pursuer stage0 | 0.12 | 0.00 | 0.04 | 0.84 |
| 9 | evader | pursuer stage2 | 0.22 | 0.05 | 0.09 | 0.64 |
| 9 | evader | pursuer stage4 | 0.35 | 0.08 | 0.08 | 0.50 |
| 9 | evader | pursuer stage6 | 0.54 | 0.32 | 0.14 | 0.01 |
| 9 | evader | pursuer stage8 | 0.60 | 0.31 | 0.10 | 0.00 |
| 10 | pursuer | evader stage1 | 0.94 | 0.02 | 0.05 | 0.00 |
| 10 | pursuer | evader stage3 | 0.90 | 0.04 | 0.07 | 0.00 |
| 10 | pursuer | evader stage5 | 0.89 | 0.05 | 0.06 | 0.00 |
| 10 | pursuer | evader stage7 | 0.91 | 0.05 | 0.05 | 0.00 |
| 10 | pursuer | evader stage9 | 0.88 | 0.09 | 0.04 | 0.00 |
| 11 | evader | pursuer stage0 | 0.11 | 0.00 | 0.03 | 0.86 |
| 11 | evader | pursuer stage2 | 0.15 | 0.06 | 0.06 | 0.73 |
| 11 | evader | pursuer stage4 | 0.18 | 0.10 | 0.07 | 0.65 |
| 11 | evader | pursuer stage6 | 0.41 | 0.41 | 0.18 | 0.01 |
| 11 | evader | pursuer stage8 | 0.32 | 0.59 | 0.10 | 0.00 |
| 11 | evader | pursuer stage10 | 0.52 | 0.42 | 0.08 | 0.00 |
| 12 | pursuer | evader stage1 | 0.93 | 0.01 | 0.05 | 0.00 |
| 12 | pursuer | evader stage3 | 0.91 | 0.03 | 0.07 | 0.00 |
| 12 | pursuer | evader stage5 | 0.89 | 0.04 | 0.07 | 0.00 |
| 12 | pursuer | evader stage7 | 0.90 | 0.04 | 0.06 | 0.00 |
| 12 | pursuer | evader stage9 | 0.89 | 0.06 | 0.05 | 0.00 |
| 12 | pursuer | evader stage11 | 0.82 | 0.12 | 0.06 | 0.00 |

**thesis-F13-short-s200**

| stage | role | vs | capture | pursuer crash | evader crash + OOB | timeout |
|---|---|---|---|---|---|---|
| 0 | pursuer | evader hover | 0.92 | 0.00 | 0.08 | 0.00 |
| 0 | pursuer | evader circular | 0.87 | 0.00 | 0.13 | 0.00 |
| 0 | pursuer | evader flee | 0.89 | 0.00 | 0.11 | 0.00 |
| 0 | pursuer | evader weave | 0.88 | 0.00 | 0.11 | 0.00 |
| 1 | evader | pursuer pure_pursuit | 0.04 | 0.13 | 0.09 | 0.74 |
| 1 | evader | pursuer apf | 0.33 | 0.17 | 0.14 | 0.36 |
| 2 | pursuer | evader stage1 | 0.92 | 0.00 | 0.08 | 0.00 |
| 3 | evader | pursuer stage0 | 0.10 | 0.01 | 0.04 | 0.85 |
| 3 | evader | pursuer stage2 | 0.50 | 0.07 | 0.10 | 0.32 |
| 4 | pursuer | evader stage1 | 0.91 | 0.01 | 0.09 | 0.00 |
| 4 | pursuer | evader stage3 | 0.94 | 0.02 | 0.05 | 0.00 |
| 5 | evader | pursuer stage0 | 0.42 | 0.01 | 0.06 | 0.51 |
| 5 | evader | pursuer stage2 | 0.43 | 0.22 | 0.07 | 0.29 |
| 5 | evader | pursuer stage4 | 0.61 | 0.19 | 0.10 | 0.11 |
| 6 | pursuer | evader stage1 | 0.89 | 0.01 | 0.10 | 0.00 |
| 6 | pursuer | evader stage3 | 0.89 | 0.04 | 0.07 | 0.00 |
| 6 | pursuer | evader stage5 | 0.92 | 0.04 | 0.04 | 0.00 |
| 7 | evader | pursuer stage0 | 0.17 | 0.00 | 0.05 | 0.77 |
| 7 | evader | pursuer stage2 | 0.17 | 0.13 | 0.09 | 0.61 |
| 7 | evader | pursuer stage4 | 0.28 | 0.10 | 0.12 | 0.50 |
| 7 | evader | pursuer stage6 | 0.66 | 0.11 | 0.15 | 0.08 |
| 8 | pursuer | evader stage1 | 0.89 | 0.01 | 0.11 | 0.00 |
| 8 | pursuer | evader stage3 | 0.89 | 0.04 | 0.07 | 0.00 |
| 8 | pursuer | evader stage5 | 0.92 | 0.04 | 0.04 | 0.00 |
| 8 | pursuer | evader stage7 | 0.88 | 0.08 | 0.05 | 0.00 |
| 9 | evader | pursuer stage0 | 0.14 | 0.01 | 0.05 | 0.80 |
| 9 | evader | pursuer stage2 | 0.19 | 0.15 | 0.09 | 0.57 |
| 9 | evader | pursuer stage4 | 0.32 | 0.14 | 0.11 | 0.43 |
| 9 | evader | pursuer stage6 | 0.56 | 0.17 | 0.14 | 0.14 |
| 9 | evader | pursuer stage8 | 0.70 | 0.21 | 0.10 | 0.00 |
| 10 | pursuer | evader stage1 | 0.87 | 0.01 | 0.12 | 0.00 |
| 10 | pursuer | evader stage3 | 0.89 | 0.04 | 0.07 | 0.00 |
| 10 | pursuer | evader stage5 | 0.92 | 0.04 | 0.05 | 0.00 |
| 10 | pursuer | evader stage7 | 0.87 | 0.07 | 0.06 | 0.00 |
| 10 | pursuer | evader stage9 | 0.78 | 0.14 | 0.09 | 0.00 |
| 11 | evader | pursuer stage0 | 0.11 | 0.00 | 0.06 | 0.82 |
| 11 | evader | pursuer stage2 | 0.17 | 0.18 | 0.11 | 0.55 |
| 11 | evader | pursuer stage4 | 0.35 | 0.20 | 0.14 | 0.32 |
| 11 | evader | pursuer stage6 | 0.59 | 0.19 | 0.17 | 0.07 |
| 11 | evader | pursuer stage8 | 0.64 | 0.23 | 0.11 | 0.02 |
| 11 | evader | pursuer stage10 | 0.60 | 0.23 | 0.18 | 0.00 |
| 12 | pursuer | evader stage1 | 0.88 | 0.01 | 0.11 | 0.00 |
| 12 | pursuer | evader stage3 | 0.91 | 0.04 | 0.06 | 0.00 |
| 12 | pursuer | evader stage5 | 0.93 | 0.03 | 0.04 | 0.00 |
| 12 | pursuer | evader stage7 | 0.89 | 0.06 | 0.05 | 0.00 |
| 12 | pursuer | evader stage9 | 0.86 | 0.09 | 0.05 | 0.00 |
| 12 | pursuer | evader stage11 | 0.79 | 0.13 | 0.09 | 0.00 |

