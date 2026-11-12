# F11_v2_sp summary

Final pursuers (stage 24) of thesis-F11v2d-s{0,100}: thesis v2 rules (truncation endings, no evader timeout bonus, plain-difference shaping 7 / 10, no closing-speed factor, GAE fix), Hedge both roles, 26 x 50M. Settings from the pilot: {'k_app': 7.0, 'w_prec': 10.0, 'w_smooth': 0.5, 'w_cmd': 0.2} (J vs scripted 0.989). Evals clean under the thesis rules; exploitability_v2 / _old = 1 - J vs a fresh 100M-frame exploiter trained under thesis v2 / the old thesis rules (luring pays). strong_finals = F8d's 16 final evaders. 2 seeds, descriptive.

Pre-registered (a), in-campaign lure rate (mean over evader stages 3-11): thesis-F11v2d-s0 0.03, thesis-F11v2d-s100 0.15

## Capture radius 0.10 m

| metric | v2/0 | v2/1 |
|---|---|---|
| heldout_rl | 0.875 | 0.869 |
| heldout_worst_cluster | 0.838 | 0.832 |
| strong_finals_mean | 0.845 | 0.751 |
| strong_finals_min | 0.501 | 0.346 |
| vs_f8_finals | 0.868 | 0.956 |
| vs_f3_hedge_finals | 0.858 | 0.765 |
| other_f3_evaders | 0.892 | 0.846 |
| other_f11_evaders | 0.906 | 0.942 |
| own_campaign_evaders | 0.952 | 0.957 |
| train | 0.833 | 0.678 |
| heldout_scripted | 0.976 | 0.971 |
| heldout_scripted_pursuer_crash | 0.025 | 0.030 |
| x3_exploiters | 0.998 | 0.999 |
| exploitability_v2 | 0.501 | 0.140 |
| exploiter_v2_pursuer_crash | 0.016 | 0.028 |
| exploiter_v2_timeout | 0.486 | 0.113 |
| exploitability_old | 0.293 | 0.624 |
| exploiter_old_pursuer_crash | 0.309 | 0.636 |
| exploiter_old_timeout | 0.002 | 0.030 |

## Capture radius 0.15 m

| metric | v2/0 | v2/1 |
|---|---|---|
| heldout_rl | 0.980 | 0.987 |
| heldout_worst_cluster | 0.976 | 0.982 |
| strong_finals_mean | 0.947 | 0.922 |
| strong_finals_min | 0.789 | 0.773 |
| vs_f8_finals | 0.973 | 0.993 |
| vs_f3_hedge_finals | 0.930 | 0.908 |
| other_f3_evaders | 0.972 | 0.964 |
| other_f11_evaders | 0.993 | 0.993 |
| own_campaign_evaders | 0.996 | 0.988 |
| train | 0.927 | 0.814 |
| heldout_scripted | 0.996 | 0.995 |
| heldout_scripted_pursuer_crash | 0.004 | 0.005 |
| x3_exploiters | 1.000 | 1.000 |
| exploitability_v2 | 0.156 | 0.113 |
| exploiter_v2_pursuer_crash | 0.001 | 0.004 |
| exploiter_v2_timeout | 0.156 | 0.109 |
| exploitability_old | 0.031 | 0.208 |
| exploiter_old_pursuer_crash | 0.033 | 0.188 |
| exploiter_old_timeout | 0.000 | 0.029 |

## In-campaign lure rates

| campaign | evader stage | mean | min - max | n pursuers |
|---|---|---|---|---|
| thesis-F11v2d-s0 | 3 | 0.00 | 0.00 - 0.00 | 1 |
| thesis-F11v2d-s0 | 5 | 0.00 | 0.00 - 0.00 | 2 |
| thesis-F11v2d-s0 | 7 | 0.01 | 0.00 - 0.02 | 3 |
| thesis-F11v2d-s0 | 9 | 0.10 | 0.00 - 0.22 | 4 |
| thesis-F11v2d-s0 | 11 | 0.04 | 0.00 - 0.11 | 5 |
| thesis-F11v2d-s0 | 13 | 0.04 | 0.01 - 0.05 | 6 |
| thesis-F11v2d-s0 | 15 | 0.05 | 0.00 - 0.10 | 7 |
| thesis-F11v2d-s0 | 17 | 0.10 | 0.00 - 0.19 | 8 |
| thesis-F11v2d-s0 | 19 | 0.07 | 0.00 - 0.13 | 9 |
| thesis-F11v2d-s0 | 21 | 0.09 | 0.00 - 0.16 | 10 |
| thesis-F11v2d-s0 | 23 | 0.08 | 0.00 - 0.15 | 11 |
| thesis-F11v2d-s0 | 25 | 0.06 | 0.00 - 0.15 | 12 |
| thesis-F11v2d-s100 | 3 | 0.00 | 0.00 - 0.00 | 1 |
| thesis-F11v2d-s100 | 5 | 0.00 | 0.00 - 0.00 | 2 |
| thesis-F11v2d-s100 | 7 | 0.01 | 0.00 - 0.02 | 3 |
| thesis-F11v2d-s100 | 9 | 0.32 | 0.00 - 0.67 | 4 |
| thesis-F11v2d-s100 | 11 | 0.41 | 0.00 - 0.89 | 5 |
| thesis-F11v2d-s100 | 13 | 0.43 | 0.00 - 0.97 | 6 |
| thesis-F11v2d-s100 | 15 | 0.50 | 0.00 - 0.97 | 7 |
| thesis-F11v2d-s100 | 17 | 0.36 | 0.00 - 0.87 | 8 |
| thesis-F11v2d-s100 | 19 | 0.36 | 0.00 - 0.84 | 9 |
| thesis-F11v2d-s100 | 21 | 0.30 | 0.00 - 0.81 | 10 |
| thesis-F11v2d-s100 | 23 | 0.20 | 0.00 - 0.57 | 11 |
| thesis-F11v2d-s100 | 25 | 0.40 | 0.01 - 0.75 | 12 |

## Inside the campaigns

**thesis-F11v2d-s0**

| stage | role | vs | capture | pursuer crash | evader crash + OOB | timeout |
|---|---|---|---|---|---|---|
| 0 | pursuer | evader hover | 0.90 | 0.00 | 0.09 | 0.00 |
| 0 | pursuer | evader circular | 0.86 | 0.00 | 0.14 | 0.00 |
| 0 | pursuer | evader flee | 0.87 | 0.00 | 0.13 | 0.00 |
| 0 | pursuer | evader weave | 0.87 | 0.00 | 0.13 | 0.00 |
| 1 | evader | pursuer pure_pursuit | 0.05 | 0.14 | 0.03 | 0.78 |
| 1 | evader | pursuer apf | 0.25 | 0.18 | 0.08 | 0.50 |
| 2 | pursuer | evader stage1 | 0.94 | 0.00 | 0.05 | 0.00 |
| 3 | evader | pursuer stage0 | 0.08 | 0.00 | 0.06 | 0.86 |
| 3 | evader | pursuer stage2 | 0.20 | 0.00 | 0.11 | 0.69 |
| 4 | pursuer | evader stage1 | 0.95 | 0.00 | 0.05 | 0.00 |
| 4 | pursuer | evader stage3 | 0.90 | 0.01 | 0.09 | 0.00 |
| 5 | evader | pursuer stage0 | 0.02 | 0.00 | 0.07 | 0.90 |
| 5 | evader | pursuer stage2 | 0.03 | 0.00 | 0.05 | 0.92 |
| 5 | evader | pursuer stage4 | 0.28 | 0.00 | 0.08 | 0.63 |
| 6 | pursuer | evader stage1 | 0.94 | 0.00 | 0.06 | 0.00 |
| 6 | pursuer | evader stage3 | 0.90 | 0.01 | 0.09 | 0.00 |
| 6 | pursuer | evader stage5 | 0.92 | 0.00 | 0.07 | 0.00 |
| 7 | evader | pursuer stage0 | 0.02 | 0.00 | 0.01 | 0.97 |
| 7 | evader | pursuer stage2 | 0.04 | 0.00 | 0.01 | 0.95 |
| 7 | evader | pursuer stage4 | 0.11 | 0.02 | 0.02 | 0.85 |
| 7 | evader | pursuer stage6 | 0.26 | 0.02 | 0.05 | 0.67 |
| 8 | pursuer | evader stage1 | 0.95 | 0.00 | 0.04 | 0.00 |
| 8 | pursuer | evader stage3 | 0.92 | 0.01 | 0.07 | 0.00 |
| 8 | pursuer | evader stage5 | 0.92 | 0.01 | 0.07 | 0.00 |
| 8 | pursuer | evader stage7 | 0.97 | 0.01 | 0.02 | 0.00 |
| 9 | evader | pursuer stage0 | 0.00 | 0.00 | 0.02 | 0.97 |
| 9 | evader | pursuer stage2 | 0.01 | 0.00 | 0.02 | 0.97 |
| 9 | evader | pursuer stage4 | 0.07 | 0.02 | 0.09 | 0.83 |
| 9 | evader | pursuer stage6 | 0.15 | 0.15 | 0.04 | 0.66 |
| 9 | evader | pursuer stage8 | 0.65 | 0.22 | 0.06 | 0.08 |
| 10 | pursuer | evader stage1 | 0.95 | 0.00 | 0.05 | 0.00 |
| 10 | pursuer | evader stage3 | 0.91 | 0.01 | 0.08 | 0.00 |
| 10 | pursuer | evader stage5 | 0.93 | 0.01 | 0.06 | 0.00 |
| 10 | pursuer | evader stage7 | 0.96 | 0.01 | 0.03 | 0.00 |
| 10 | pursuer | evader stage9 | 0.94 | 0.03 | 0.03 | 0.00 |
| 11 | evader | pursuer stage0 | 0.02 | 0.00 | 0.04 | 0.94 |
| 11 | evader | pursuer stage2 | 0.01 | 0.00 | 0.02 | 0.96 |
| 11 | evader | pursuer stage4 | 0.07 | 0.01 | 0.04 | 0.89 |
| 11 | evader | pursuer stage6 | 0.06 | 0.02 | 0.05 | 0.87 |
| 11 | evader | pursuer stage8 | 0.08 | 0.11 | 0.05 | 0.76 |
| 11 | evader | pursuer stage10 | 0.64 | 0.06 | 0.07 | 0.23 |
| 12 | pursuer | evader stage1 | 0.96 | 0.00 | 0.04 | 0.00 |
| 12 | pursuer | evader stage3 | 0.92 | 0.01 | 0.07 | 0.00 |
| 12 | pursuer | evader stage5 | 0.92 | 0.01 | 0.08 | 0.00 |
| 12 | pursuer | evader stage7 | 0.98 | 0.00 | 0.02 | 0.00 |
| 12 | pursuer | evader stage9 | 0.94 | 0.03 | 0.03 | 0.00 |
| 12 | pursuer | evader stage11 | 0.94 | 0.03 | 0.03 | 0.00 |
| 13 | evader | pursuer stage0 | 0.00 | 0.00 | 0.03 | 0.96 |
| 13 | evader | pursuer stage2 | 0.05 | 0.01 | 0.05 | 0.88 |
| 13 | evader | pursuer stage4 | 0.11 | 0.05 | 0.08 | 0.76 |
| 13 | evader | pursuer stage6 | 0.06 | 0.05 | 0.05 | 0.83 |
| 13 | evader | pursuer stage8 | 0.05 | 0.03 | 0.05 | 0.87 |
| 13 | evader | pursuer stage10 | 0.28 | 0.04 | 0.09 | 0.60 |
| 13 | evader | pursuer stage12 | 0.62 | 0.03 | 0.07 | 0.28 |
| 14 | pursuer | evader stage1 | 0.94 | 0.00 | 0.06 | 0.00 |
| 14 | pursuer | evader stage3 | 0.92 | 0.01 | 0.07 | 0.00 |
| 14 | pursuer | evader stage5 | 0.92 | 0.01 | 0.07 | 0.00 |
| 14 | pursuer | evader stage7 | 0.97 | 0.01 | 0.02 | 0.00 |
| 14 | pursuer | evader stage9 | 0.95 | 0.03 | 0.03 | 0.00 |
| 14 | pursuer | evader stage11 | 0.95 | 0.02 | 0.04 | 0.00 |
| 14 | pursuer | evader stage13 | 0.94 | 0.02 | 0.04 | 0.00 |
| 15 | evader | pursuer stage0 | 0.00 | 0.00 | 0.03 | 0.96 |
| 15 | evader | pursuer stage2 | 0.03 | 0.00 | 0.10 | 0.87 |
| 15 | evader | pursuer stage4 | 0.15 | 0.07 | 0.12 | 0.67 |
| 15 | evader | pursuer stage6 | 0.09 | 0.03 | 0.11 | 0.78 |
| 15 | evader | pursuer stage8 | 0.07 | 0.03 | 0.06 | 0.83 |
| 15 | evader | pursuer stage10 | 0.18 | 0.04 | 0.11 | 0.68 |
| 15 | evader | pursuer stage12 | 0.30 | 0.10 | 0.10 | 0.50 |
| 15 | evader | pursuer stage14 | 0.62 | 0.09 | 0.10 | 0.20 |
| 16 | pursuer | evader stage1 | 0.94 | 0.00 | 0.05 | 0.00 |
| 16 | pursuer | evader stage3 | 0.92 | 0.01 | 0.07 | 0.00 |
| 16 | pursuer | evader stage5 | 0.93 | 0.01 | 0.06 | 0.00 |
| 16 | pursuer | evader stage7 | 0.98 | 0.01 | 0.02 | 0.00 |
| 16 | pursuer | evader stage9 | 0.95 | 0.03 | 0.03 | 0.00 |
| 16 | pursuer | evader stage11 | 0.94 | 0.02 | 0.03 | 0.00 |
| 16 | pursuer | evader stage13 | 0.95 | 0.02 | 0.04 | 0.00 |
| 16 | pursuer | evader stage15 | 0.93 | 0.03 | 0.04 | 0.00 |
| 17 | evader | pursuer stage0 | 0.01 | 0.00 | 0.02 | 0.97 |
| 17 | evader | pursuer stage2 | 0.07 | 0.00 | 0.11 | 0.82 |
| 17 | evader | pursuer stage4 | 0.11 | 0.10 | 0.08 | 0.71 |
| 17 | evader | pursuer stage6 | 0.08 | 0.03 | 0.06 | 0.82 |
| 17 | evader | pursuer stage8 | 0.12 | 0.14 | 0.05 | 0.69 |
| 17 | evader | pursuer stage10 | 0.21 | 0.10 | 0.08 | 0.61 |
| 17 | evader | pursuer stage12 | 0.31 | 0.19 | 0.07 | 0.43 |
| 17 | evader | pursuer stage14 | 0.56 | 0.14 | 0.08 | 0.22 |
| 17 | evader | pursuer stage16 | 0.67 | 0.11 | 0.08 | 0.14 |
| 18 | pursuer | evader stage1 | 0.92 | 0.00 | 0.08 | 0.00 |
| 18 | pursuer | evader stage3 | 0.92 | 0.01 | 0.07 | 0.00 |
| 18 | pursuer | evader stage5 | 0.92 | 0.01 | 0.07 | 0.00 |
| 18 | pursuer | evader stage7 | 0.98 | 0.01 | 0.02 | 0.00 |
| 18 | pursuer | evader stage9 | 0.95 | 0.02 | 0.03 | 0.00 |
| 18 | pursuer | evader stage11 | 0.94 | 0.03 | 0.04 | 0.00 |
| 18 | pursuer | evader stage13 | 0.95 | 0.02 | 0.04 | 0.00 |
| 18 | pursuer | evader stage15 | 0.94 | 0.02 | 0.05 | 0.00 |
| 18 | pursuer | evader stage17 | 0.93 | 0.04 | 0.04 | 0.00 |
| 19 | evader | pursuer stage0 | 0.00 | 0.00 | 0.03 | 0.96 |
| 19 | evader | pursuer stage2 | 0.01 | 0.00 | 0.03 | 0.96 |
| 19 | evader | pursuer stage4 | 0.06 | 0.07 | 0.08 | 0.78 |
| 19 | evader | pursuer stage6 | 0.07 | 0.07 | 0.06 | 0.80 |
| 19 | evader | pursuer stage8 | 0.08 | 0.13 | 0.03 | 0.76 |
| 19 | evader | pursuer stage10 | 0.22 | 0.08 | 0.07 | 0.64 |
| 19 | evader | pursuer stage12 | 0.23 | 0.07 | 0.05 | 0.64 |
| 19 | evader | pursuer stage14 | 0.40 | 0.11 | 0.07 | 0.42 |
| 19 | evader | pursuer stage16 | 0.26 | 0.05 | 0.07 | 0.62 |
| 19 | evader | pursuer stage18 | 0.64 | 0.05 | 0.07 | 0.24 |
| 20 | pursuer | evader stage1 | 0.91 | 0.01 | 0.08 | 0.00 |
| 20 | pursuer | evader stage3 | 0.92 | 0.01 | 0.08 | 0.00 |
| 20 | pursuer | evader stage5 | 0.91 | 0.00 | 0.09 | 0.00 |
| 20 | pursuer | evader stage7 | 0.98 | 0.01 | 0.01 | 0.00 |
| 20 | pursuer | evader stage9 | 0.94 | 0.02 | 0.03 | 0.00 |
| 20 | pursuer | evader stage11 | 0.92 | 0.04 | 0.04 | 0.00 |
| 20 | pursuer | evader stage13 | 0.95 | 0.01 | 0.04 | 0.00 |
| 20 | pursuer | evader stage15 | 0.92 | 0.03 | 0.05 | 0.00 |
| 20 | pursuer | evader stage17 | 0.93 | 0.04 | 0.03 | 0.00 |
| 20 | pursuer | evader stage19 | 0.92 | 0.03 | 0.04 | 0.00 |
| 21 | evader | pursuer stage0 | 0.01 | 0.00 | 0.03 | 0.96 |
| 21 | evader | pursuer stage2 | 0.01 | 0.00 | 0.03 | 0.96 |
| 21 | evader | pursuer stage4 | 0.07 | 0.15 | 0.05 | 0.73 |
| 21 | evader | pursuer stage6 | 0.07 | 0.10 | 0.07 | 0.76 |
| 21 | evader | pursuer stage8 | 0.05 | 0.16 | 0.03 | 0.76 |
| 21 | evader | pursuer stage10 | 0.17 | 0.08 | 0.07 | 0.68 |
| 21 | evader | pursuer stage12 | 0.23 | 0.08 | 0.06 | 0.63 |
| 21 | evader | pursuer stage14 | 0.43 | 0.13 | 0.06 | 0.38 |
| 21 | evader | pursuer stage16 | 0.34 | 0.06 | 0.09 | 0.52 |
| 21 | evader | pursuer stage18 | 0.63 | 0.05 | 0.11 | 0.21 |
| 21 | evader | pursuer stage20 | 0.58 | 0.08 | 0.07 | 0.27 |
| 22 | pursuer | evader stage1 | 0.93 | 0.01 | 0.07 | 0.00 |
| 22 | pursuer | evader stage3 | 0.92 | 0.01 | 0.07 | 0.00 |
| 22 | pursuer | evader stage5 | 0.93 | 0.00 | 0.07 | 0.00 |
| 22 | pursuer | evader stage7 | 0.97 | 0.01 | 0.02 | 0.00 |
| 22 | pursuer | evader stage9 | 0.94 | 0.03 | 0.03 | 0.00 |
| 22 | pursuer | evader stage11 | 0.93 | 0.02 | 0.05 | 0.00 |
| 22 | pursuer | evader stage13 | 0.95 | 0.01 | 0.04 | 0.00 |
| 22 | pursuer | evader stage15 | 0.94 | 0.02 | 0.04 | 0.00 |
| 22 | pursuer | evader stage17 | 0.92 | 0.04 | 0.04 | 0.00 |
| 22 | pursuer | evader stage19 | 0.92 | 0.03 | 0.04 | 0.00 |
| 22 | pursuer | evader stage21 | 0.92 | 0.03 | 0.04 | 0.01 |
| 23 | evader | pursuer stage0 | 0.00 | 0.00 | 0.02 | 0.98 |
| 23 | evader | pursuer stage2 | 0.01 | 0.00 | 0.01 | 0.98 |
| 23 | evader | pursuer stage4 | 0.09 | 0.03 | 0.04 | 0.84 |
| 23 | evader | pursuer stage6 | 0.11 | 0.10 | 0.05 | 0.74 |
| 23 | evader | pursuer stage8 | 0.12 | 0.14 | 0.05 | 0.69 |
| 23 | evader | pursuer stage10 | 0.16 | 0.15 | 0.07 | 0.63 |
| 23 | evader | pursuer stage12 | 0.16 | 0.09 | 0.05 | 0.69 |
| 23 | evader | pursuer stage14 | 0.44 | 0.09 | 0.07 | 0.39 |
| 23 | evader | pursuer stage16 | 0.24 | 0.05 | 0.05 | 0.67 |
| 23 | evader | pursuer stage18 | 0.40 | 0.06 | 0.09 | 0.45 |
| 23 | evader | pursuer stage20 | 0.28 | 0.09 | 0.04 | 0.59 |
| 23 | evader | pursuer stage22 | 0.57 | 0.04 | 0.05 | 0.33 |
| 24 | pursuer | evader stage1 | 0.94 | 0.00 | 0.06 | 0.00 |
| 24 | pursuer | evader stage3 | 0.92 | 0.02 | 0.07 | 0.00 |
| 24 | pursuer | evader stage5 | 0.92 | 0.01 | 0.07 | 0.00 |
| 24 | pursuer | evader stage7 | 0.97 | 0.01 | 0.02 | 0.00 |
| 24 | pursuer | evader stage9 | 0.93 | 0.04 | 0.03 | 0.00 |
| 24 | pursuer | evader stage11 | 0.92 | 0.03 | 0.05 | 0.00 |
| 24 | pursuer | evader stage13 | 0.95 | 0.02 | 0.03 | 0.00 |
| 24 | pursuer | evader stage15 | 0.93 | 0.02 | 0.04 | 0.00 |
| 24 | pursuer | evader stage17 | 0.91 | 0.05 | 0.04 | 0.00 |
| 24 | pursuer | evader stage19 | 0.92 | 0.04 | 0.04 | 0.00 |
| 24 | pursuer | evader stage21 | 0.92 | 0.04 | 0.04 | 0.01 |
| 24 | pursuer | evader stage23 | 0.92 | 0.03 | 0.03 | 0.01 |
| 25 | evader | pursuer stage0 | 0.00 | 0.00 | 0.00 | 0.99 |
| 25 | evader | pursuer stage2 | 0.00 | 0.00 | 0.02 | 0.98 |
| 25 | evader | pursuer stage4 | 0.07 | 0.01 | 0.03 | 0.89 |
| 25 | evader | pursuer stage6 | 0.11 | 0.06 | 0.06 | 0.77 |
| 25 | evader | pursuer stage8 | 0.11 | 0.14 | 0.04 | 0.71 |
| 25 | evader | pursuer stage10 | 0.20 | 0.15 | 0.06 | 0.59 |
| 25 | evader | pursuer stage12 | 0.23 | 0.06 | 0.04 | 0.68 |
| 25 | evader | pursuer stage14 | 0.45 | 0.06 | 0.09 | 0.40 |
| 25 | evader | pursuer stage16 | 0.27 | 0.05 | 0.06 | 0.62 |
| 25 | evader | pursuer stage18 | 0.39 | 0.05 | 0.07 | 0.50 |
| 25 | evader | pursuer stage20 | 0.42 | 0.09 | 0.05 | 0.44 |
| 25 | evader | pursuer stage22 | 0.47 | 0.05 | 0.04 | 0.44 |
| 25 | evader | pursuer stage24 | 0.72 | 0.05 | 0.04 | 0.19 |

**thesis-F11v2d-s100**

| stage | role | vs | capture | pursuer crash | evader crash + OOB | timeout |
|---|---|---|---|---|---|---|
| 0 | pursuer | evader hover | 0.89 | 0.00 | 0.11 | 0.00 |
| 0 | pursuer | evader circular | 0.84 | 0.00 | 0.16 | 0.00 |
| 0 | pursuer | evader flee | 0.86 | 0.00 | 0.14 | 0.00 |
| 0 | pursuer | evader weave | 0.86 | 0.00 | 0.14 | 0.00 |
| 1 | evader | pursuer pure_pursuit | 0.03 | 0.13 | 0.05 | 0.79 |
| 1 | evader | pursuer apf | 0.25 | 0.19 | 0.08 | 0.48 |
| 2 | pursuer | evader stage1 | 0.97 | 0.00 | 0.03 | 0.00 |
| 3 | evader | pursuer stage0 | 0.18 | 0.00 | 0.07 | 0.74 |
| 3 | evader | pursuer stage2 | 0.12 | 0.00 | 0.05 | 0.83 |
| 4 | pursuer | evader stage1 | 0.95 | 0.01 | 0.04 | 0.00 |
| 4 | pursuer | evader stage3 | 0.96 | 0.00 | 0.04 | 0.00 |
| 5 | evader | pursuer stage0 | 0.32 | 0.00 | 0.03 | 0.65 |
| 5 | evader | pursuer stage2 | 0.03 | 0.00 | 0.01 | 0.95 |
| 5 | evader | pursuer stage4 | 0.27 | 0.00 | 0.02 | 0.71 |
| 6 | pursuer | evader stage1 | 0.94 | 0.01 | 0.05 | 0.00 |
| 6 | pursuer | evader stage3 | 0.94 | 0.01 | 0.05 | 0.00 |
| 6 | pursuer | evader stage5 | 0.97 | 0.01 | 0.02 | 0.00 |
| 7 | evader | pursuer stage0 | 0.02 | 0.00 | 0.02 | 0.96 |
| 7 | evader | pursuer stage2 | 0.00 | 0.00 | 0.02 | 0.98 |
| 7 | evader | pursuer stage4 | 0.07 | 0.00 | 0.03 | 0.90 |
| 7 | evader | pursuer stage6 | 0.30 | 0.02 | 0.08 | 0.60 |
| 8 | pursuer | evader stage1 | 0.95 | 0.01 | 0.05 | 0.00 |
| 8 | pursuer | evader stage3 | 0.95 | 0.01 | 0.05 | 0.00 |
| 8 | pursuer | evader stage5 | 0.98 | 0.01 | 0.02 | 0.00 |
| 8 | pursuer | evader stage7 | 0.97 | 0.01 | 0.02 | 0.00 |
| 9 | evader | pursuer stage0 | 0.09 | 0.01 | 0.02 | 0.88 |
| 9 | evader | pursuer stage2 | 0.00 | 0.00 | 0.01 | 0.98 |
| 9 | evader | pursuer stage4 | 0.08 | 0.07 | 0.02 | 0.82 |
| 9 | evader | pursuer stage6 | 0.29 | 0.55 | 0.03 | 0.14 |
| 9 | evader | pursuer stage8 | 0.29 | 0.67 | 0.02 | 0.02 |
| 10 | pursuer | evader stage1 | 0.97 | 0.00 | 0.03 | 0.00 |
| 10 | pursuer | evader stage3 | 0.96 | 0.00 | 0.04 | 0.00 |
| 10 | pursuer | evader stage5 | 0.98 | 0.00 | 0.01 | 0.00 |
| 10 | pursuer | evader stage7 | 0.97 | 0.01 | 0.02 | 0.00 |
| 10 | pursuer | evader stage9 | 0.98 | 0.01 | 0.01 | 0.00 |
| 11 | evader | pursuer stage0 | 0.04 | 0.00 | 0.01 | 0.95 |
| 11 | evader | pursuer stage2 | 0.01 | 0.00 | 0.01 | 0.98 |
| 11 | evader | pursuer stage4 | 0.02 | 0.21 | 0.01 | 0.77 |
| 11 | evader | pursuer stage6 | 0.04 | 0.85 | 0.01 | 0.11 |
| 11 | evader | pursuer stage8 | 0.07 | 0.89 | 0.01 | 0.03 |
| 11 | evader | pursuer stage10 | 0.35 | 0.10 | 0.02 | 0.53 |
| 12 | pursuer | evader stage1 | 0.97 | 0.00 | 0.03 | 0.00 |
| 12 | pursuer | evader stage3 | 0.96 | 0.00 | 0.04 | 0.00 |
| 12 | pursuer | evader stage5 | 0.99 | 0.00 | 0.01 | 0.00 |
| 12 | pursuer | evader stage7 | 0.97 | 0.01 | 0.02 | 0.00 |
| 12 | pursuer | evader stage9 | 0.97 | 0.02 | 0.01 | 0.00 |
| 12 | pursuer | evader stage11 | 0.99 | 0.01 | 0.01 | 0.00 |
| 13 | evader | pursuer stage0 | 0.04 | 0.00 | 0.01 | 0.95 |
| 13 | evader | pursuer stage2 | 0.01 | 0.00 | 0.01 | 0.97 |
| 13 | evader | pursuer stage4 | 0.01 | 0.21 | 0.01 | 0.77 |
| 13 | evader | pursuer stage6 | 0.01 | 0.97 | 0.01 | 0.01 |
| 13 | evader | pursuer stage8 | 0.02 | 0.90 | 0.01 | 0.07 |
| 13 | evader | pursuer stage10 | 0.36 | 0.13 | 0.03 | 0.49 |
| 13 | evader | pursuer stage12 | 0.48 | 0.36 | 0.02 | 0.14 |
| 14 | pursuer | evader stage1 | 0.96 | 0.00 | 0.04 | 0.00 |
| 14 | pursuer | evader stage3 | 0.96 | 0.00 | 0.04 | 0.00 |
| 14 | pursuer | evader stage5 | 0.99 | 0.00 | 0.01 | 0.00 |
| 14 | pursuer | evader stage7 | 0.98 | 0.00 | 0.02 | 0.00 |
| 14 | pursuer | evader stage9 | 0.96 | 0.02 | 0.02 | 0.00 |
| 14 | pursuer | evader stage11 | 0.99 | 0.00 | 0.01 | 0.00 |
| 14 | pursuer | evader stage13 | 0.98 | 0.01 | 0.01 | 0.00 |
| 15 | evader | pursuer stage0 | 0.04 | 0.01 | 0.02 | 0.93 |
| 15 | evader | pursuer stage2 | 0.01 | 0.00 | 0.04 | 0.95 |
| 15 | evader | pursuer stage4 | 0.02 | 0.41 | 0.01 | 0.56 |
| 15 | evader | pursuer stage6 | 0.02 | 0.97 | 0.01 | 0.00 |
| 15 | evader | pursuer stage8 | 0.03 | 0.94 | 0.02 | 0.01 |
| 15 | evader | pursuer stage10 | 0.34 | 0.30 | 0.05 | 0.31 |
| 15 | evader | pursuer stage12 | 0.29 | 0.63 | 0.03 | 0.05 |
| 15 | evader | pursuer stage14 | 0.69 | 0.27 | 0.02 | 0.01 |
| 16 | pursuer | evader stage1 | 0.96 | 0.00 | 0.04 | 0.00 |
| 16 | pursuer | evader stage3 | 0.96 | 0.00 | 0.03 | 0.00 |
| 16 | pursuer | evader stage5 | 0.99 | 0.00 | 0.01 | 0.00 |
| 16 | pursuer | evader stage7 | 0.98 | 0.00 | 0.02 | 0.00 |
| 16 | pursuer | evader stage9 | 0.96 | 0.03 | 0.02 | 0.00 |
| 16 | pursuer | evader stage11 | 0.99 | 0.00 | 0.00 | 0.00 |
| 16 | pursuer | evader stage13 | 0.98 | 0.01 | 0.01 | 0.00 |
| 16 | pursuer | evader stage15 | 0.92 | 0.04 | 0.03 | 0.00 |
| 17 | evader | pursuer stage0 | 0.06 | 0.00 | 0.06 | 0.88 |
| 17 | evader | pursuer stage2 | 0.02 | 0.00 | 0.04 | 0.94 |
| 17 | evader | pursuer stage4 | 0.12 | 0.15 | 0.06 | 0.67 |
| 17 | evader | pursuer stage6 | 0.08 | 0.87 | 0.02 | 0.03 |
| 17 | evader | pursuer stage8 | 0.10 | 0.80 | 0.04 | 0.06 |
| 17 | evader | pursuer stage10 | 0.32 | 0.19 | 0.09 | 0.41 |
| 17 | evader | pursuer stage12 | 0.38 | 0.54 | 0.05 | 0.03 |
| 17 | evader | pursuer stage14 | 0.73 | 0.20 | 0.05 | 0.02 |
| 17 | evader | pursuer stage16 | 0.80 | 0.12 | 0.06 | 0.02 |
| 18 | pursuer | evader stage1 | 0.95 | 0.00 | 0.05 | 0.00 |
| 18 | pursuer | evader stage3 | 0.96 | 0.00 | 0.04 | 0.00 |
| 18 | pursuer | evader stage5 | 0.99 | 0.00 | 0.01 | 0.00 |
| 18 | pursuer | evader stage7 | 0.98 | 0.01 | 0.02 | 0.00 |
| 18 | pursuer | evader stage9 | 0.96 | 0.02 | 0.01 | 0.00 |
| 18 | pursuer | evader stage11 | 0.99 | 0.00 | 0.01 | 0.00 |
| 18 | pursuer | evader stage13 | 0.98 | 0.01 | 0.01 | 0.00 |
| 18 | pursuer | evader stage15 | 0.94 | 0.04 | 0.02 | 0.00 |
| 18 | pursuer | evader stage17 | 0.93 | 0.05 | 0.02 | 0.00 |
| 19 | evader | pursuer stage0 | 0.07 | 0.00 | 0.02 | 0.91 |
| 19 | evader | pursuer stage2 | 0.01 | 0.00 | 0.05 | 0.94 |
| 19 | evader | pursuer stage4 | 0.03 | 0.23 | 0.03 | 0.71 |
| 19 | evader | pursuer stage6 | 0.04 | 0.84 | 0.02 | 0.10 |
| 19 | evader | pursuer stage8 | 0.07 | 0.69 | 0.02 | 0.21 |
| 19 | evader | pursuer stage10 | 0.17 | 0.19 | 0.03 | 0.61 |
| 19 | evader | pursuer stage12 | 0.27 | 0.60 | 0.04 | 0.10 |
| 19 | evader | pursuer stage14 | 0.53 | 0.28 | 0.06 | 0.13 |
| 19 | evader | pursuer stage16 | 0.53 | 0.21 | 0.07 | 0.19 |
| 19 | evader | pursuer stage18 | 0.71 | 0.15 | 0.04 | 0.10 |
| 20 | pursuer | evader stage1 | 0.95 | 0.01 | 0.05 | 0.00 |
| 20 | pursuer | evader stage3 | 0.95 | 0.00 | 0.05 | 0.00 |
| 20 | pursuer | evader stage5 | 0.98 | 0.00 | 0.01 | 0.00 |
| 20 | pursuer | evader stage7 | 0.97 | 0.01 | 0.03 | 0.00 |
| 20 | pursuer | evader stage9 | 0.96 | 0.02 | 0.03 | 0.00 |
| 20 | pursuer | evader stage11 | 0.99 | 0.00 | 0.01 | 0.00 |
| 20 | pursuer | evader stage13 | 0.97 | 0.01 | 0.01 | 0.00 |
| 20 | pursuer | evader stage15 | 0.94 | 0.03 | 0.04 | 0.00 |
| 20 | pursuer | evader stage17 | 0.93 | 0.04 | 0.04 | 0.00 |
| 20 | pursuer | evader stage19 | 0.89 | 0.06 | 0.05 | 0.00 |
| 21 | evader | pursuer stage0 | 0.04 | 0.00 | 0.06 | 0.90 |
| 21 | evader | pursuer stage2 | 0.01 | 0.00 | 0.20 | 0.79 |
| 21 | evader | pursuer stage4 | 0.03 | 0.36 | 0.13 | 0.48 |
| 21 | evader | pursuer stage6 | 0.05 | 0.81 | 0.06 | 0.09 |
| 21 | evader | pursuer stage8 | 0.08 | 0.65 | 0.10 | 0.17 |
| 21 | evader | pursuer stage10 | 0.21 | 0.25 | 0.10 | 0.44 |
| 21 | evader | pursuer stage12 | 0.57 | 0.28 | 0.06 | 0.08 |
| 21 | evader | pursuer stage14 | 0.70 | 0.14 | 0.08 | 0.08 |
| 21 | evader | pursuer stage16 | 0.42 | 0.19 | 0.13 | 0.25 |
| 21 | evader | pursuer stage18 | 0.51 | 0.22 | 0.14 | 0.14 |
| 21 | evader | pursuer stage20 | 0.75 | 0.10 | 0.13 | 0.03 |
| 22 | pursuer | evader stage1 | 0.95 | 0.01 | 0.05 | 0.00 |
| 22 | pursuer | evader stage3 | 0.96 | 0.00 | 0.04 | 0.00 |
| 22 | pursuer | evader stage5 | 0.99 | 0.00 | 0.01 | 0.00 |
| 22 | pursuer | evader stage7 | 0.98 | 0.00 | 0.01 | 0.00 |
| 22 | pursuer | evader stage9 | 0.97 | 0.01 | 0.02 | 0.00 |
| 22 | pursuer | evader stage11 | 0.99 | 0.00 | 0.01 | 0.00 |
| 22 | pursuer | evader stage13 | 0.98 | 0.01 | 0.01 | 0.00 |
| 22 | pursuer | evader stage15 | 0.94 | 0.03 | 0.03 | 0.00 |
| 22 | pursuer | evader stage17 | 0.95 | 0.03 | 0.02 | 0.00 |
| 22 | pursuer | evader stage19 | 0.94 | 0.04 | 0.02 | 0.00 |
| 22 | pursuer | evader stage21 | 0.93 | 0.03 | 0.04 | 0.00 |
| 23 | evader | pursuer stage0 | 0.03 | 0.00 | 0.13 | 0.83 |
| 23 | evader | pursuer stage2 | 0.00 | 0.00 | 0.13 | 0.87 |
| 23 | evader | pursuer stage4 | 0.05 | 0.32 | 0.08 | 0.55 |
| 23 | evader | pursuer stage6 | 0.06 | 0.57 | 0.03 | 0.34 |
| 23 | evader | pursuer stage8 | 0.10 | 0.46 | 0.06 | 0.39 |
| 23 | evader | pursuer stage10 | 0.20 | 0.13 | 0.07 | 0.59 |
| 23 | evader | pursuer stage12 | 0.52 | 0.29 | 0.04 | 0.16 |
| 23 | evader | pursuer stage14 | 0.71 | 0.09 | 0.05 | 0.15 |
| 23 | evader | pursuer stage16 | 0.52 | 0.15 | 0.09 | 0.25 |
| 23 | evader | pursuer stage18 | 0.47 | 0.13 | 0.10 | 0.30 |
| 23 | evader | pursuer stage20 | 0.67 | 0.08 | 0.16 | 0.10 |
| 23 | evader | pursuer stage22 | 0.91 | 0.03 | 0.06 | 0.01 |
| 24 | pursuer | evader stage1 | 0.96 | 0.00 | 0.04 | 0.00 |
| 24 | pursuer | evader stage3 | 0.96 | 0.00 | 0.04 | 0.00 |
| 24 | pursuer | evader stage5 | 0.99 | 0.00 | 0.01 | 0.00 |
| 24 | pursuer | evader stage7 | 0.99 | 0.00 | 0.01 | 0.00 |
| 24 | pursuer | evader stage9 | 0.98 | 0.01 | 0.01 | 0.00 |
| 24 | pursuer | evader stage11 | 0.99 | 0.00 | 0.00 | 0.00 |
| 24 | pursuer | evader stage13 | 0.98 | 0.01 | 0.01 | 0.00 |
| 24 | pursuer | evader stage15 | 0.95 | 0.03 | 0.02 | 0.00 |
| 24 | pursuer | evader stage17 | 0.97 | 0.02 | 0.02 | 0.00 |
| 24 | pursuer | evader stage19 | 0.95 | 0.03 | 0.02 | 0.00 |
| 24 | pursuer | evader stage21 | 0.94 | 0.03 | 0.03 | 0.00 |
| 24 | pursuer | evader stage23 | 0.93 | 0.03 | 0.04 | 0.00 |
| 25 | evader | pursuer stage0 | 0.02 | 0.01 | 0.03 | 0.94 |
| 25 | evader | pursuer stage2 | 0.00 | 0.01 | 0.05 | 0.93 |
| 25 | evader | pursuer stage4 | 0.01 | 0.36 | 0.02 | 0.61 |
| 25 | evader | pursuer stage6 | 0.01 | 0.75 | 0.01 | 0.23 |
| 25 | evader | pursuer stage8 | 0.02 | 0.71 | 0.04 | 0.24 |
| 25 | evader | pursuer stage10 | 0.06 | 0.32 | 0.04 | 0.58 |
| 25 | evader | pursuer stage12 | 0.28 | 0.46 | 0.02 | 0.24 |
| 25 | evader | pursuer stage14 | 0.33 | 0.23 | 0.05 | 0.39 |
| 25 | evader | pursuer stage16 | 0.17 | 0.46 | 0.06 | 0.31 |
| 25 | evader | pursuer stage18 | 0.14 | 0.68 | 0.03 | 0.15 |
| 25 | evader | pursuer stage20 | 0.22 | 0.50 | 0.07 | 0.22 |
| 25 | evader | pursuer stage22 | 0.55 | 0.14 | 0.07 | 0.24 |
| 25 | evader | pursuer stage24 | 0.57 | 0.21 | 0.07 | 0.15 |

