# F8_roncero_sp summary

Final pursuers (stage 8). roncero/k = F8 campaign seed k (Roncero's reward and endings, Hedge both roles, 9 x 50M); hedge/k = F3 hedge-both seed k (thesis rules, Hedge both roles, 9 x 300M). All evals clean under the thesis rules except vs_exploiter_roncero (Roncero's rules). exploitability_<rules> = 1 - J vs a fresh exploiter trained under those rules; for hedge/k the thesis-rules exploiter is F3c's. other_f3 / other_f8 / own evaders exclude stage-1 evaders. J at 0.15 m is the exact re-score (eval-only). 2 seeds, descriptive; F8 = 26 stages x 50M (13 generations), F3 = 9 stages x 300M.

## Capture radius 0.10 m

| metric | roncero/0 | roncero/1 | hedge/0 | hedge/1 | roncero/0 - hedge/0 | roncero/1 - hedge/1 |
|---|---|---|---|---|---|---|
| heldout_rl | 0.424 | 0.597 | 0.791 | 0.882 | -0.367 | -0.284 |
| heldout_worst_cluster | 0.211 | 0.515 | 0.692 | 0.816 | -0.482 | -0.301 |
| other_f3_evaders | 0.852 | 0.819 | 0.871 | 0.906 | -0.019 | -0.087 |
| other_f8_evaders | 0.946 | 0.987 | 0.898 | 0.919 | +0.048 | +0.068 |
| own_campaign_evaders | 0.996 | 0.986 | 0.957 | 0.961 | +0.039 | +0.024 |
| train | 0.804 | 0.825 | 0.594 | 0.640 | +0.210 | +0.184 |
| heldout_scripted | 0.991 | 0.971 | 0.808 | 0.853 | +0.183 | +0.118 |
| heldout_scripted_pursuer_crash | 0.009 | 0.018 | 0.197 | 0.149 | -0.187 | -0.132 |
| x3_exploiters | 0.541 | 0.577 | 0.955 | 0.973 | -0.414 | -0.395 |
| exploitability_thesis | 0.755 | 0.770 | 0.974 | 0.985 | -0.219 | -0.215 |
| exploiter_thesis_pursuer_crash | 0.002 | 0.002 | 0.975 | 0.988 | -0.973 | -0.986 |
| exploitability_roncero | 0.557 | 0.886 | 0.760 | 0.932 | -0.202 | -0.046 |
| exploiter_roncero_pursuer_crash | 0.001 | 0.000 | 0.115 | 0.021 | -0.114 | -0.020 |
| exploiter_roncero_timeout | 0.546 | 0.879 | 0.000 | 0.009 | +0.546 | +0.870 |

## Capture radius 0.15 m

| metric | roncero/0 | roncero/1 | hedge/0 | hedge/1 | roncero/0 - hedge/0 | roncero/1 - hedge/1 |
|---|---|---|---|---|---|---|
| heldout_rl | 0.576 | 0.817 | 0.960 | 0.989 | -0.383 | -0.172 |
| heldout_worst_cluster | 0.432 | 0.778 | 0.874 | 0.921 | -0.442 | -0.142 |
| other_f3_evaders | 0.956 | 0.942 | 0.957 | 0.973 | -0.001 | -0.031 |
| other_f8_evaders | 0.994 | 1.000 | 0.980 | 0.987 | +0.014 | +0.012 |
| own_campaign_evaders | 0.999 | 0.999 | 0.993 | 0.995 | +0.006 | +0.004 |
| train | 0.911 | 0.941 | 0.759 | 0.810 | +0.152 | +0.132 |
| heldout_scripted | 0.999 | 0.999 | 0.905 | 0.936 | +0.094 | +0.063 |
| heldout_scripted_pursuer_crash | 0.001 | 0.001 | 0.096 | 0.065 | -0.095 | -0.064 |
| x3_exploiters | 0.861 | 0.944 | 0.987 | 0.992 | -0.126 | -0.049 |
| exploitability_thesis | 0.177 | 0.132 | 0.773 | 0.855 | -0.596 | -0.722 |
| exploiter_thesis_pursuer_crash | 0.000 | 0.000 | 0.774 | 0.856 | -0.774 | -0.856 |
| exploitability_roncero | 0.526 | 0.027 | 0.540 | 0.734 | -0.014 | -0.708 |
| exploiter_roncero_pursuer_crash | 0.000 | 0.000 | 0.081 | 0.013 | -0.080 | -0.013 |
| exploiter_roncero_timeout | 0.517 | 0.021 | 0.000 | 0.007 | +0.517 | +0.014 |

## In-campaign lure rates (pursuer crash caused by each evader stage in the stage >= 2 pursuers)

Stages 3 / 5 / 7 are comparable across F8 and F3 (F3 has 9 stages, F8 26).

| campaign | evader stage | mean | min - max | n pursuers |
|---|---|---|---|---|
| thesis-F8-roncero-s0 | 3 | 0.00 | 0.00 - 0.00 | 1 |
| thesis-F8-roncero-s0 | 5 | 0.00 | 0.00 - 0.00 | 2 |
| thesis-F8-roncero-s0 | 7 | 0.00 | 0.00 - 0.00 | 3 |
| thesis-F8-roncero-s0 | 9 | 0.00 | 0.00 - 0.01 | 4 |
| thesis-F8-roncero-s0 | 11 | 0.04 | 0.00 - 0.16 | 5 |
| thesis-F8-roncero-s0 | 13 | 0.02 | 0.00 - 0.06 | 6 |
| thesis-F8-roncero-s0 | 15 | 0.07 | 0.00 - 0.28 | 7 |
| thesis-F8-roncero-s0 | 17 | 0.05 | 0.00 - 0.23 | 8 |
| thesis-F8-roncero-s0 | 19 | 0.11 | 0.00 - 0.39 | 9 |
| thesis-F8-roncero-s0 | 21 | 0.05 | 0.00 - 0.27 | 10 |
| thesis-F8-roncero-s0 | 23 | 0.03 | 0.00 - 0.18 | 11 |
| thesis-F8-roncero-s0 | 25 | 0.01 | 0.00 - 0.04 | 12 |
| thesis-F8-roncero-s100 | 3 | 0.00 | 0.00 - 0.00 | 1 |
| thesis-F8-roncero-s100 | 5 | 0.00 | 0.00 - 0.00 | 2 |
| thesis-F8-roncero-s100 | 7 | 0.00 | 0.00 - 0.00 | 3 |
| thesis-F8-roncero-s100 | 9 | 0.02 | 0.00 - 0.06 | 4 |
| thesis-F8-roncero-s100 | 11 | 0.00 | 0.00 - 0.02 | 5 |
| thesis-F8-roncero-s100 | 13 | 0.01 | 0.00 - 0.03 | 6 |
| thesis-F8-roncero-s100 | 15 | 0.01 | 0.00 - 0.03 | 7 |
| thesis-F8-roncero-s100 | 17 | 0.01 | 0.00 - 0.05 | 8 |
| thesis-F8-roncero-s100 | 19 | 0.00 | 0.00 - 0.01 | 9 |
| thesis-F8-roncero-s100 | 21 | 0.01 | 0.00 - 0.02 | 10 |
| thesis-F8-roncero-s100 | 23 | 0.01 | 0.00 - 0.04 | 11 |
| thesis-F8-roncero-s100 | 25 | 0.01 | 0.00 - 0.03 | 12 |
| thesis-F3-hedge-both-s0 | 3 | 0.97 | 0.97 - 0.97 | 1 |
| thesis-F3-hedge-both-s0 | 5 | 0.98 | 0.98 - 0.98 | 2 |
| thesis-F3-hedge-both-s0 | 7 | 0.98 | 0.96 - 0.99 | 3 |
| thesis-F3-hedge-both-s1 | 3 | 0.95 | 0.95 - 0.95 | 1 |
| thesis-F3-hedge-both-s1 | 5 | 0.98 | 0.97 - 0.98 | 2 |
| thesis-F3-hedge-both-s1 | 7 | 0.97 | 0.96 - 0.98 | 3 |

## Inside the F8 campaigns (each stage's eval vs its training opponents, Roncero's rules)

**thesis-F8-roncero-s0**

| stage | role | vs | capture | pursuer crash | evader crash + OOB | timeout |
|---|---|---|---|---|---|---|
| 0 | pursuer | evader hover | 0.89 | 0.00 | 0.10 | 0.00 |
| 0 | pursuer | evader circular | 0.82 | 0.01 | 0.17 | 0.00 |
| 0 | pursuer | evader flee | 0.85 | 0.01 | 0.15 | 0.00 |
| 0 | pursuer | evader weave | 0.85 | 0.01 | 0.14 | 0.00 |
| 1 | evader | pursuer pure_pursuit | 0.00 | 0.11 | 0.21 | 0.56 |
| 1 | evader | pursuer apf | 0.06 | 0.13 | 0.49 | 0.13 |
| 2 | pursuer | evader stage1 | 0.42 | 0.00 | 0.56 | 0.00 |
| 3 | evader | pursuer stage0 | 0.11 | 0.01 | 0.11 | 0.72 |
| 3 | evader | pursuer stage2 | 0.02 | 0.00 | 0.11 | 0.85 |
| 4 | pursuer | evader stage1 | 0.37 | 0.00 | 0.60 | 0.00 |
| 4 | pursuer | evader stage3 | 0.87 | 0.00 | 0.12 | 0.00 |
| 5 | evader | pursuer stage0 | 0.06 | 0.00 | 0.07 | 0.85 |
| 5 | evader | pursuer stage2 | 0.05 | 0.00 | 0.07 | 0.86 |
| 5 | evader | pursuer stage4 | 0.13 | 0.00 | 0.08 | 0.77 |
| 6 | pursuer | evader stage1 | 0.37 | 0.00 | 0.61 | 0.00 |
| 6 | pursuer | evader stage3 | 0.88 | 0.00 | 0.11 | 0.00 |
| 6 | pursuer | evader stage5 | 0.90 | 0.01 | 0.08 | 0.00 |
| 7 | evader | pursuer stage0 | 0.14 | 0.00 | 0.18 | 0.21 |
| 7 | evader | pursuer stage2 | 0.25 | 0.00 | 0.10 | 0.50 |
| 7 | evader | pursuer stage4 | 0.03 | 0.00 | 0.11 | 0.76 |
| 7 | evader | pursuer stage6 | 0.30 | 0.00 | 0.12 | 0.54 |
| 8 | pursuer | evader stage1 | 0.40 | 0.00 | 0.56 | 0.00 |
| 8 | pursuer | evader stage3 | 0.89 | 0.00 | 0.10 | 0.00 |
| 8 | pursuer | evader stage5 | 0.89 | 0.01 | 0.09 | 0.00 |
| 8 | pursuer | evader stage7 | 0.82 | 0.00 | 0.16 | 0.00 |
| 9 | evader | pursuer stage0 | 0.12 | 0.01 | 0.15 | 0.35 |
| 9 | evader | pursuer stage2 | 0.02 | 0.00 | 0.11 | 0.84 |
| 9 | evader | pursuer stage4 | 0.02 | 0.00 | 0.11 | 0.84 |
| 9 | evader | pursuer stage6 | 0.05 | 0.00 | 0.16 | 0.74 |
| 9 | evader | pursuer stage8 | 0.19 | 0.01 | 0.33 | 0.26 |
| 10 | pursuer | evader stage1 | 0.39 | 0.00 | 0.58 | 0.00 |
| 10 | pursuer | evader stage3 | 0.88 | 0.01 | 0.11 | 0.00 |
| 10 | pursuer | evader stage5 | 0.87 | 0.01 | 0.10 | 0.00 |
| 10 | pursuer | evader stage7 | 0.78 | 0.01 | 0.18 | 0.00 |
| 10 | pursuer | evader stage9 | 0.72 | 0.01 | 0.24 | 0.00 |
| 11 | evader | pursuer stage0 | 0.03 | 0.00 | 0.11 | 0.29 |
| 11 | evader | pursuer stage2 | 0.01 | 0.00 | 0.09 | 0.79 |
| 11 | evader | pursuer stage4 | 0.00 | 0.00 | 0.09 | 0.86 |
| 11 | evader | pursuer stage6 | 0.02 | 0.03 | 0.33 | 0.53 |
| 11 | evader | pursuer stage8 | 0.04 | 0.03 | 0.12 | 0.79 |
| 11 | evader | pursuer stage10 | 0.21 | 0.16 | 0.23 | 0.30 |
| 12 | pursuer | evader stage1 | 0.38 | 0.00 | 0.59 | 0.00 |
| 12 | pursuer | evader stage3 | 0.87 | 0.01 | 0.11 | 0.00 |
| 12 | pursuer | evader stage5 | 0.87 | 0.01 | 0.11 | 0.00 |
| 12 | pursuer | evader stage7 | 0.80 | 0.01 | 0.18 | 0.00 |
| 12 | pursuer | evader stage9 | 0.72 | 0.01 | 0.25 | 0.00 |
| 12 | pursuer | evader stage11 | 0.74 | 0.02 | 0.20 | 0.00 |
| 13 | evader | pursuer stage0 | 0.01 | 0.00 | 0.88 | 0.05 |
| 13 | evader | pursuer stage2 | 0.01 | 0.00 | 0.10 | 0.88 |
| 13 | evader | pursuer stage4 | 0.00 | 0.00 | 0.10 | 0.88 |
| 13 | evader | pursuer stage6 | 0.01 | 0.01 | 0.19 | 0.77 |
| 13 | evader | pursuer stage8 | 0.03 | 0.02 | 0.40 | 0.47 |
| 13 | evader | pursuer stage10 | 0.04 | 0.06 | 0.19 | 0.66 |
| 13 | evader | pursuer stage12 | 0.11 | 0.02 | 0.20 | 0.63 |
| 14 | pursuer | evader stage1 | 0.38 | 0.00 | 0.55 | 0.00 |
| 14 | pursuer | evader stage3 | 0.87 | 0.01 | 0.11 | 0.00 |
| 14 | pursuer | evader stage5 | 0.86 | 0.01 | 0.12 | 0.00 |
| 14 | pursuer | evader stage7 | 0.80 | 0.00 | 0.17 | 0.00 |
| 14 | pursuer | evader stage9 | 0.74 | 0.01 | 0.22 | 0.00 |
| 14 | pursuer | evader stage11 | 0.81 | 0.01 | 0.16 | 0.00 |
| 14 | pursuer | evader stage13 | 0.79 | 0.01 | 0.18 | 0.00 |
| 15 | evader | pursuer stage0 | 0.01 | 0.00 | 0.38 | 0.36 |
| 15 | evader | pursuer stage2 | 0.00 | 0.00 | 0.08 | 0.91 |
| 15 | evader | pursuer stage4 | 0.00 | 0.00 | 0.07 | 0.90 |
| 15 | evader | pursuer stage6 | 0.02 | 0.03 | 0.15 | 0.32 |
| 15 | evader | pursuer stage8 | 0.00 | 0.01 | 0.44 | 0.44 |
| 15 | evader | pursuer stage10 | 0.09 | 0.28 | 0.22 | 0.27 |
| 15 | evader | pursuer stage12 | 0.11 | 0.13 | 0.12 | 0.49 |
| 15 | evader | pursuer stage14 | 0.17 | 0.02 | 0.14 | 0.09 |
| 16 | pursuer | evader stage1 | 0.40 | 0.00 | 0.55 | 0.00 |
| 16 | pursuer | evader stage3 | 0.87 | 0.00 | 0.12 | 0.00 |
| 16 | pursuer | evader stage5 | 0.86 | 0.01 | 0.12 | 0.00 |
| 16 | pursuer | evader stage7 | 0.83 | 0.00 | 0.15 | 0.00 |
| 16 | pursuer | evader stage9 | 0.76 | 0.01 | 0.21 | 0.00 |
| 16 | pursuer | evader stage11 | 0.81 | 0.01 | 0.16 | 0.00 |
| 16 | pursuer | evader stage13 | 0.80 | 0.01 | 0.18 | 0.00 |
| 16 | pursuer | evader stage15 | 0.88 | 0.01 | 0.09 | 0.00 |
| 17 | evader | pursuer stage0 | 0.01 | 0.00 | 0.48 | 0.01 |
| 17 | evader | pursuer stage2 | 0.00 | 0.00 | 0.11 | 0.85 |
| 17 | evader | pursuer stage4 | 0.00 | 0.00 | 0.11 | 0.85 |
| 17 | evader | pursuer stage6 | 0.01 | 0.01 | 0.21 | 0.17 |
| 17 | evader | pursuer stage8 | 0.01 | 0.02 | 0.75 | 0.12 |
| 17 | evader | pursuer stage10 | 0.05 | 0.23 | 0.23 | 0.39 |
| 17 | evader | pursuer stage12 | 0.03 | 0.03 | 0.15 | 0.67 |
| 17 | evader | pursuer stage14 | 0.12 | 0.05 | 0.19 | 0.23 |
| 17 | evader | pursuer stage16 | 0.11 | 0.03 | 0.16 | 0.35 |
| 18 | pursuer | evader stage1 | 0.40 | 0.00 | 0.54 | 0.00 |
| 18 | pursuer | evader stage3 | 0.86 | 0.00 | 0.12 | 0.00 |
| 18 | pursuer | evader stage5 | 0.86 | 0.01 | 0.12 | 0.00 |
| 18 | pursuer | evader stage7 | 0.83 | 0.00 | 0.14 | 0.00 |
| 18 | pursuer | evader stage9 | 0.76 | 0.00 | 0.20 | 0.00 |
| 18 | pursuer | evader stage11 | 0.81 | 0.01 | 0.15 | 0.00 |
| 18 | pursuer | evader stage13 | 0.76 | 0.01 | 0.21 | 0.00 |
| 18 | pursuer | evader stage15 | 0.84 | 0.01 | 0.12 | 0.00 |
| 18 | pursuer | evader stage17 | 0.81 | 0.03 | 0.14 | 0.00 |
| 19 | evader | pursuer stage0 | 0.02 | 0.02 | 0.29 | 0.59 |
| 19 | evader | pursuer stage2 | 0.00 | 0.00 | 0.09 | 0.91 |
| 19 | evader | pursuer stage4 | 0.00 | 0.00 | 0.07 | 0.91 |
| 19 | evader | pursuer stage6 | 0.05 | 0.09 | 0.13 | 0.70 |
| 19 | evader | pursuer stage8 | 0.03 | 0.09 | 0.45 | 0.39 |
| 19 | evader | pursuer stage10 | 0.04 | 0.39 | 0.25 | 0.21 |
| 19 | evader | pursuer stage12 | 0.15 | 0.16 | 0.17 | 0.44 |
| 19 | evader | pursuer stage14 | 0.14 | 0.15 | 0.26 | 0.06 |
| 19 | evader | pursuer stage16 | 0.05 | 0.03 | 0.14 | 0.31 |
| 19 | evader | pursuer stage18 | 0.20 | 0.03 | 0.12 | 0.45 |
| 20 | pursuer | evader stage1 | 0.42 | 0.00 | 0.54 | 0.00 |
| 20 | pursuer | evader stage3 | 0.87 | 0.01 | 0.12 | 0.00 |
| 20 | pursuer | evader stage5 | 0.85 | 0.01 | 0.13 | 0.00 |
| 20 | pursuer | evader stage7 | 0.82 | 0.00 | 0.15 | 0.00 |
| 20 | pursuer | evader stage9 | 0.77 | 0.00 | 0.20 | 0.00 |
| 20 | pursuer | evader stage11 | 0.79 | 0.01 | 0.17 | 0.00 |
| 20 | pursuer | evader stage13 | 0.75 | 0.01 | 0.22 | 0.00 |
| 20 | pursuer | evader stage15 | 0.85 | 0.01 | 0.12 | 0.00 |
| 20 | pursuer | evader stage17 | 0.83 | 0.02 | 0.13 | 0.00 |
| 20 | pursuer | evader stage19 | 0.83 | 0.02 | 0.14 | 0.00 |
| 21 | evader | pursuer stage0 | 0.05 | 0.01 | 0.49 | 0.24 |
| 21 | evader | pursuer stage2 | 0.00 | 0.00 | 0.09 | 0.89 |
| 21 | evader | pursuer stage4 | 0.00 | 0.00 | 0.14 | 0.83 |
| 21 | evader | pursuer stage6 | 0.02 | 0.04 | 0.20 | 0.72 |
| 21 | evader | pursuer stage8 | 0.03 | 0.05 | 0.40 | 0.45 |
| 21 | evader | pursuer stage10 | 0.04 | 0.27 | 0.25 | 0.40 |
| 21 | evader | pursuer stage12 | 0.06 | 0.09 | 0.17 | 0.65 |
| 21 | evader | pursuer stage14 | 0.07 | 0.02 | 0.39 | 0.42 |
| 21 | evader | pursuer stage16 | 0.03 | 0.02 | 0.28 | 0.50 |
| 21 | evader | pursuer stage18 | 0.05 | 0.02 | 0.30 | 0.49 |
| 21 | evader | pursuer stage20 | 0.15 | 0.01 | 0.30 | 0.51 |
| 22 | pursuer | evader stage1 | 0.43 | 0.00 | 0.52 | 0.00 |
| 22 | pursuer | evader stage3 | 0.87 | 0.00 | 0.12 | 0.00 |
| 22 | pursuer | evader stage5 | 0.85 | 0.01 | 0.13 | 0.00 |
| 22 | pursuer | evader stage7 | 0.81 | 0.00 | 0.16 | 0.00 |
| 22 | pursuer | evader stage9 | 0.78 | 0.00 | 0.20 | 0.00 |
| 22 | pursuer | evader stage11 | 0.79 | 0.01 | 0.18 | 0.00 |
| 22 | pursuer | evader stage13 | 0.73 | 0.01 | 0.23 | 0.00 |
| 22 | pursuer | evader stage15 | 0.84 | 0.02 | 0.12 | 0.00 |
| 22 | pursuer | evader stage17 | 0.82 | 0.02 | 0.13 | 0.00 |
| 22 | pursuer | evader stage19 | 0.79 | 0.03 | 0.14 | 0.00 |
| 22 | pursuer | evader stage21 | 0.66 | 0.02 | 0.24 | 0.06 |
| 23 | evader | pursuer stage0 | 0.05 | 0.01 | 0.44 | 0.46 |
| 23 | evader | pursuer stage2 | 0.01 | 0.00 | 0.12 | 0.87 |
| 23 | evader | pursuer stage4 | 0.01 | 0.00 | 0.17 | 0.82 |
| 23 | evader | pursuer stage6 | 0.02 | 0.02 | 0.19 | 0.75 |
| 23 | evader | pursuer stage8 | 0.01 | 0.03 | 0.44 | 0.50 |
| 23 | evader | pursuer stage10 | 0.04 | 0.18 | 0.27 | 0.49 |
| 23 | evader | pursuer stage12 | 0.03 | 0.04 | 0.23 | 0.70 |
| 23 | evader | pursuer stage14 | 0.03 | 0.03 | 0.36 | 0.57 |
| 23 | evader | pursuer stage16 | 0.02 | 0.01 | 0.22 | 0.72 |
| 23 | evader | pursuer stage18 | 0.02 | 0.01 | 0.23 | 0.72 |
| 23 | evader | pursuer stage20 | 0.04 | 0.01 | 0.24 | 0.70 |
| 23 | evader | pursuer stage22 | 0.07 | 0.01 | 0.18 | 0.73 |
| 24 | pursuer | evader stage1 | 0.41 | 0.00 | 0.55 | 0.00 |
| 24 | pursuer | evader stage3 | 0.87 | 0.00 | 0.12 | 0.00 |
| 24 | pursuer | evader stage5 | 0.85 | 0.00 | 0.13 | 0.00 |
| 24 | pursuer | evader stage7 | 0.81 | 0.00 | 0.16 | 0.00 |
| 24 | pursuer | evader stage9 | 0.78 | 0.00 | 0.19 | 0.00 |
| 24 | pursuer | evader stage11 | 0.80 | 0.01 | 0.17 | 0.00 |
| 24 | pursuer | evader stage13 | 0.77 | 0.01 | 0.21 | 0.00 |
| 24 | pursuer | evader stage15 | 0.87 | 0.01 | 0.10 | 0.00 |
| 24 | pursuer | evader stage17 | 0.85 | 0.01 | 0.12 | 0.00 |
| 24 | pursuer | evader stage19 | 0.81 | 0.02 | 0.14 | 0.00 |
| 24 | pursuer | evader stage21 | 0.73 | 0.02 | 0.22 | 0.01 |
| 24 | pursuer | evader stage23 | 0.72 | 0.01 | 0.24 | 0.00 |
| 25 | evader | pursuer stage0 | 0.04 | 0.02 | 0.16 | 0.77 |
| 25 | evader | pursuer stage2 | 0.00 | 0.00 | 0.14 | 0.53 |
| 25 | evader | pursuer stage4 | 0.00 | 0.00 | 0.16 | 0.12 |
| 25 | evader | pursuer stage6 | 0.01 | 0.01 | 0.14 | 0.26 |
| 25 | evader | pursuer stage8 | 0.01 | 0.01 | 0.17 | 0.74 |
| 25 | evader | pursuer stage10 | 0.00 | 0.01 | 0.15 | 0.62 |
| 25 | evader | pursuer stage12 | 0.01 | 0.02 | 0.23 | 0.57 |
| 25 | evader | pursuer stage14 | 0.01 | 0.01 | 0.23 | 0.74 |
| 25 | evader | pursuer stage16 | 0.01 | 0.00 | 0.15 | 0.82 |
| 25 | evader | pursuer stage18 | 0.01 | 0.01 | 0.16 | 0.80 |
| 25 | evader | pursuer stage20 | 0.02 | 0.00 | 0.12 | 0.84 |
| 25 | evader | pursuer stage22 | 0.03 | 0.00 | 0.12 | 0.85 |
| 25 | evader | pursuer stage24 | 0.13 | 0.04 | 0.16 | 0.65 |

**thesis-F8-roncero-s100**

| stage | role | vs | capture | pursuer crash | evader crash + OOB | timeout |
|---|---|---|---|---|---|---|
| 0 | pursuer | evader hover | 0.89 | 0.00 | 0.10 | 0.00 |
| 0 | pursuer | evader circular | 0.85 | 0.00 | 0.14 | 0.00 |
| 0 | pursuer | evader flee | 0.85 | 0.00 | 0.15 | 0.00 |
| 0 | pursuer | evader weave | 0.86 | 0.00 | 0.14 | 0.00 |
| 1 | evader | pursuer pure_pursuit | 0.01 | 0.11 | 0.18 | 0.57 |
| 1 | evader | pursuer apf | 0.11 | 0.14 | 0.42 | 0.15 |
| 2 | pursuer | evader stage1 | 0.58 | 0.00 | 0.39 | 0.00 |
| 3 | evader | pursuer stage0 | 0.07 | 0.00 | 0.08 | 0.44 |
| 3 | evader | pursuer stage2 | 0.06 | 0.00 | 0.09 | 0.84 |
| 4 | pursuer | evader stage1 | 0.60 | 0.00 | 0.36 | 0.00 |
| 4 | pursuer | evader stage3 | 0.89 | 0.01 | 0.09 | 0.00 |
| 5 | evader | pursuer stage0 | 0.03 | 0.00 | 0.24 | 0.02 |
| 5 | evader | pursuer stage2 | 0.02 | 0.00 | 0.55 | 0.24 |
| 5 | evader | pursuer stage4 | 0.09 | 0.00 | 0.34 | 0.41 |
| 6 | pursuer | evader stage1 | 0.58 | 0.00 | 0.39 | 0.00 |
| 6 | pursuer | evader stage3 | 0.90 | 0.00 | 0.09 | 0.00 |
| 6 | pursuer | evader stage5 | 0.76 | 0.00 | 0.17 | 0.00 |
| 7 | evader | pursuer stage0 | 0.02 | 0.00 | 0.12 | 0.00 |
| 7 | evader | pursuer stage2 | 0.01 | 0.00 | 0.58 | 0.17 |
| 7 | evader | pursuer stage4 | 0.06 | 0.00 | 0.27 | 0.22 |
| 7 | evader | pursuer stage6 | 0.09 | 0.00 | 0.16 | 0.25 |
| 8 | pursuer | evader stage1 | 0.60 | 0.00 | 0.38 | 0.00 |
| 8 | pursuer | evader stage3 | 0.89 | 0.01 | 0.09 | 0.00 |
| 8 | pursuer | evader stage5 | 0.80 | 0.01 | 0.15 | 0.00 |
| 8 | pursuer | evader stage7 | 0.86 | 0.01 | 0.12 | 0.00 |
| 9 | evader | pursuer stage0 | 0.02 | 0.00 | 0.12 | 0.35 |
| 9 | evader | pursuer stage2 | 0.00 | 0.00 | 0.10 | 0.89 |
| 9 | evader | pursuer stage4 | 0.02 | 0.01 | 0.07 | 0.88 |
| 9 | evader | pursuer stage6 | 0.04 | 0.03 | 0.09 | 0.41 |
| 9 | evader | pursuer stage8 | 0.18 | 0.06 | 0.10 | 0.31 |
| 10 | pursuer | evader stage1 | 0.60 | 0.00 | 0.36 | 0.00 |
| 10 | pursuer | evader stage3 | 0.90 | 0.01 | 0.09 | 0.00 |
| 10 | pursuer | evader stage5 | 0.81 | 0.01 | 0.16 | 0.00 |
| 10 | pursuer | evader stage7 | 0.87 | 0.01 | 0.10 | 0.00 |
| 10 | pursuer | evader stage9 | 0.90 | 0.01 | 0.06 | 0.00 |
| 11 | evader | pursuer stage0 | 0.01 | 0.00 | 0.14 | 0.79 |
| 11 | evader | pursuer stage2 | 0.00 | 0.00 | 0.04 | 0.94 |
| 11 | evader | pursuer stage4 | 0.01 | 0.00 | 0.05 | 0.88 |
| 11 | evader | pursuer stage6 | 0.01 | 0.00 | 0.05 | 0.83 |
| 11 | evader | pursuer stage8 | 0.05 | 0.02 | 0.08 | 0.81 |
| 11 | evader | pursuer stage10 | 0.08 | 0.00 | 0.07 | 0.84 |
| 12 | pursuer | evader stage1 | 0.61 | 0.00 | 0.36 | 0.00 |
| 12 | pursuer | evader stage3 | 0.90 | 0.00 | 0.09 | 0.00 |
| 12 | pursuer | evader stage5 | 0.81 | 0.01 | 0.16 | 0.00 |
| 12 | pursuer | evader stage7 | 0.87 | 0.01 | 0.08 | 0.00 |
| 12 | pursuer | evader stage9 | 0.89 | 0.01 | 0.07 | 0.00 |
| 12 | pursuer | evader stage11 | 0.91 | 0.00 | 0.08 | 0.00 |
| 13 | evader | pursuer stage0 | 0.01 | 0.00 | 0.01 | 0.79 |
| 13 | evader | pursuer stage2 | 0.01 | 0.00 | 0.01 | 0.97 |
| 13 | evader | pursuer stage4 | 0.01 | 0.00 | 0.01 | 0.98 |
| 13 | evader | pursuer stage6 | 0.01 | 0.00 | 0.02 | 0.97 |
| 13 | evader | pursuer stage8 | 0.05 | 0.02 | 0.03 | 0.90 |
| 13 | evader | pursuer stage10 | 0.05 | 0.01 | 0.01 | 0.93 |
| 13 | evader | pursuer stage12 | 0.09 | 0.03 | 0.03 | 0.84 |
| 14 | pursuer | evader stage1 | 0.62 | 0.00 | 0.34 | 0.00 |
| 14 | pursuer | evader stage3 | 0.90 | 0.00 | 0.09 | 0.00 |
| 14 | pursuer | evader stage5 | 0.81 | 0.01 | 0.14 | 0.00 |
| 14 | pursuer | evader stage7 | 0.90 | 0.01 | 0.07 | 0.00 |
| 14 | pursuer | evader stage9 | 0.90 | 0.01 | 0.06 | 0.00 |
| 14 | pursuer | evader stage11 | 0.91 | 0.01 | 0.06 | 0.00 |
| 14 | pursuer | evader stage13 | 0.95 | 0.01 | 0.03 | 0.00 |
| 15 | evader | pursuer stage0 | 0.01 | 0.00 | 0.04 | 0.90 |
| 15 | evader | pursuer stage2 | 0.01 | 0.00 | 0.03 | 0.95 |
| 15 | evader | pursuer stage4 | 0.01 | 0.00 | 0.03 | 0.95 |
| 15 | evader | pursuer stage6 | 0.00 | 0.00 | 0.03 | 0.95 |
| 15 | evader | pursuer stage8 | 0.03 | 0.00 | 0.03 | 0.92 |
| 15 | evader | pursuer stage10 | 0.02 | 0.00 | 0.05 | 0.90 |
| 15 | evader | pursuer stage12 | 0.03 | 0.01 | 0.04 | 0.90 |
| 15 | evader | pursuer stage14 | 0.18 | 0.03 | 0.09 | 0.59 |
| 16 | pursuer | evader stage1 | 0.61 | 0.00 | 0.29 | 0.00 |
| 16 | pursuer | evader stage3 | 0.91 | 0.00 | 0.08 | 0.00 |
| 16 | pursuer | evader stage5 | 0.82 | 0.01 | 0.14 | 0.00 |
| 16 | pursuer | evader stage7 | 0.88 | 0.00 | 0.08 | 0.00 |
| 16 | pursuer | evader stage9 | 0.89 | 0.01 | 0.07 | 0.00 |
| 16 | pursuer | evader stage11 | 0.89 | 0.01 | 0.09 | 0.00 |
| 16 | pursuer | evader stage13 | 0.94 | 0.02 | 0.04 | 0.00 |
| 16 | pursuer | evader stage15 | 0.92 | 0.01 | 0.06 | 0.00 |
| 17 | evader | pursuer stage0 | 0.01 | 0.00 | 0.03 | 0.87 |
| 17 | evader | pursuer stage2 | 0.01 | 0.00 | 0.02 | 0.97 |
| 17 | evader | pursuer stage4 | 0.01 | 0.00 | 0.05 | 0.93 |
| 17 | evader | pursuer stage6 | 0.00 | 0.00 | 0.04 | 0.95 |
| 17 | evader | pursuer stage8 | 0.01 | 0.00 | 0.05 | 0.93 |
| 17 | evader | pursuer stage10 | 0.01 | 0.00 | 0.07 | 0.90 |
| 17 | evader | pursuer stage12 | 0.02 | 0.00 | 0.08 | 0.87 |
| 17 | evader | pursuer stage14 | 0.09 | 0.05 | 0.10 | 0.66 |
| 17 | evader | pursuer stage16 | 0.12 | 0.01 | 0.17 | 0.67 |
| 18 | pursuer | evader stage1 | 0.63 | 0.00 | 0.33 | 0.00 |
| 18 | pursuer | evader stage3 | 0.92 | 0.00 | 0.08 | 0.00 |
| 18 | pursuer | evader stage5 | 0.82 | 0.01 | 0.15 | 0.00 |
| 18 | pursuer | evader stage7 | 0.90 | 0.00 | 0.08 | 0.00 |
| 18 | pursuer | evader stage9 | 0.91 | 0.01 | 0.06 | 0.00 |
| 18 | pursuer | evader stage11 | 0.91 | 0.00 | 0.08 | 0.00 |
| 18 | pursuer | evader stage13 | 0.94 | 0.01 | 0.04 | 0.00 |
| 18 | pursuer | evader stage15 | 0.93 | 0.01 | 0.05 | 0.00 |
| 18 | pursuer | evader stage17 | 0.90 | 0.01 | 0.08 | 0.00 |
| 19 | evader | pursuer stage0 | 0.03 | 0.00 | 0.02 | 0.86 |
| 19 | evader | pursuer stage2 | 0.00 | 0.00 | 0.00 | 0.99 |
| 19 | evader | pursuer stage4 | 0.05 | 0.00 | 0.01 | 0.93 |
| 19 | evader | pursuer stage6 | 0.01 | 0.01 | 0.02 | 0.91 |
| 19 | evader | pursuer stage8 | 0.00 | 0.00 | 0.01 | 0.98 |
| 19 | evader | pursuer stage10 | 0.01 | 0.00 | 0.04 | 0.94 |
| 19 | evader | pursuer stage12 | 0.01 | 0.00 | 0.05 | 0.93 |
| 19 | evader | pursuer stage14 | 0.26 | 0.00 | 0.05 | 0.64 |
| 19 | evader | pursuer stage16 | 0.03 | 0.00 | 0.04 | 0.93 |
| 19 | evader | pursuer stage18 | 0.04 | 0.00 | 0.09 | 0.86 |
| 20 | pursuer | evader stage1 | 0.66 | 0.00 | 0.28 | 0.00 |
| 20 | pursuer | evader stage3 | 0.90 | 0.00 | 0.09 | 0.00 |
| 20 | pursuer | evader stage5 | 0.82 | 0.01 | 0.16 | 0.00 |
| 20 | pursuer | evader stage7 | 0.90 | 0.01 | 0.08 | 0.00 |
| 20 | pursuer | evader stage9 | 0.91 | 0.01 | 0.07 | 0.00 |
| 20 | pursuer | evader stage11 | 0.90 | 0.01 | 0.08 | 0.00 |
| 20 | pursuer | evader stage13 | 0.95 | 0.01 | 0.04 | 0.00 |
| 20 | pursuer | evader stage15 | 0.91 | 0.02 | 0.05 | 0.00 |
| 20 | pursuer | evader stage17 | 0.87 | 0.02 | 0.08 | 0.00 |
| 20 | pursuer | evader stage19 | 0.89 | 0.00 | 0.10 | 0.00 |
| 21 | evader | pursuer stage0 | 0.04 | 0.00 | 0.08 | 0.76 |
| 21 | evader | pursuer stage2 | 0.01 | 0.00 | 0.03 | 0.96 |
| 21 | evader | pursuer stage4 | 0.10 | 0.00 | 0.03 | 0.87 |
| 21 | evader | pursuer stage6 | 0.12 | 0.00 | 0.02 | 0.84 |
| 21 | evader | pursuer stage8 | 0.01 | 0.00 | 0.03 | 0.96 |
| 21 | evader | pursuer stage10 | 0.02 | 0.01 | 0.10 | 0.79 |
| 21 | evader | pursuer stage12 | 0.03 | 0.01 | 0.04 | 0.90 |
| 21 | evader | pursuer stage14 | 0.10 | 0.02 | 0.05 | 0.73 |
| 21 | evader | pursuer stage16 | 0.02 | 0.00 | 0.02 | 0.95 |
| 21 | evader | pursuer stage18 | 0.04 | 0.01 | 0.03 | 0.92 |
| 21 | evader | pursuer stage20 | 0.07 | 0.01 | 0.07 | 0.83 |
| 22 | pursuer | evader stage1 | 0.66 | 0.00 | 0.26 | 0.00 |
| 22 | pursuer | evader stage3 | 0.90 | 0.00 | 0.08 | 0.00 |
| 22 | pursuer | evader stage5 | 0.81 | 0.01 | 0.16 | 0.00 |
| 22 | pursuer | evader stage7 | 0.90 | 0.00 | 0.08 | 0.00 |
| 22 | pursuer | evader stage9 | 0.89 | 0.01 | 0.08 | 0.00 |
| 22 | pursuer | evader stage11 | 0.90 | 0.01 | 0.08 | 0.00 |
| 22 | pursuer | evader stage13 | 0.94 | 0.01 | 0.04 | 0.00 |
| 22 | pursuer | evader stage15 | 0.89 | 0.02 | 0.06 | 0.00 |
| 22 | pursuer | evader stage17 | 0.85 | 0.02 | 0.09 | 0.00 |
| 22 | pursuer | evader stage19 | 0.91 | 0.00 | 0.08 | 0.00 |
| 22 | pursuer | evader stage21 | 0.88 | 0.01 | 0.07 | 0.00 |
| 23 | evader | pursuer stage0 | 0.01 | 0.00 | 0.07 | 0.55 |
| 23 | evader | pursuer stage2 | 0.00 | 0.00 | 0.04 | 0.95 |
| 23 | evader | pursuer stage4 | 0.00 | 0.00 | 0.06 | 0.93 |
| 23 | evader | pursuer stage6 | 0.01 | 0.00 | 0.06 | 0.92 |
| 23 | evader | pursuer stage8 | 0.01 | 0.00 | 0.06 | 0.93 |
| 23 | evader | pursuer stage10 | 0.01 | 0.00 | 0.07 | 0.91 |
| 23 | evader | pursuer stage12 | 0.04 | 0.00 | 0.07 | 0.88 |
| 23 | evader | pursuer stage14 | 0.18 | 0.00 | 0.06 | 0.71 |
| 23 | evader | pursuer stage16 | 0.07 | 0.01 | 0.04 | 0.87 |
| 23 | evader | pursuer stage18 | 0.10 | 0.02 | 0.12 | 0.72 |
| 23 | evader | pursuer stage20 | 0.18 | 0.04 | 0.12 | 0.57 |
| 23 | evader | pursuer stage22 | 0.12 | 0.02 | 0.11 | 0.67 |
| 24 | pursuer | evader stage1 | 0.65 | 0.00 | 0.29 | 0.00 |
| 24 | pursuer | evader stage3 | 0.91 | 0.00 | 0.08 | 0.00 |
| 24 | pursuer | evader stage5 | 0.82 | 0.01 | 0.15 | 0.00 |
| 24 | pursuer | evader stage7 | 0.89 | 0.00 | 0.08 | 0.00 |
| 24 | pursuer | evader stage9 | 0.90 | 0.01 | 0.07 | 0.00 |
| 24 | pursuer | evader stage11 | 0.91 | 0.00 | 0.07 | 0.00 |
| 24 | pursuer | evader stage13 | 0.95 | 0.01 | 0.04 | 0.00 |
| 24 | pursuer | evader stage15 | 0.91 | 0.02 | 0.05 | 0.00 |
| 24 | pursuer | evader stage17 | 0.86 | 0.02 | 0.08 | 0.00 |
| 24 | pursuer | evader stage19 | 0.93 | 0.00 | 0.06 | 0.00 |
| 24 | pursuer | evader stage21 | 0.92 | 0.01 | 0.05 | 0.00 |
| 24 | pursuer | evader stage23 | 0.90 | 0.01 | 0.07 | 0.00 |
| 25 | evader | pursuer stage0 | 0.02 | 0.01 | 0.19 | 0.08 |
| 25 | evader | pursuer stage2 | 0.01 | 0.00 | 0.04 | 0.94 |
| 25 | evader | pursuer stage4 | 0.01 | 0.00 | 0.04 | 0.92 |
| 25 | evader | pursuer stage6 | 0.01 | 0.00 | 0.07 | 0.59 |
| 25 | evader | pursuer stage8 | 0.00 | 0.00 | 0.06 | 0.92 |
| 25 | evader | pursuer stage10 | 0.02 | 0.01 | 0.12 | 0.80 |
| 25 | evader | pursuer stage12 | 0.01 | 0.00 | 0.11 | 0.84 |
| 25 | evader | pursuer stage14 | 0.08 | 0.02 | 0.08 | 0.74 |
| 25 | evader | pursuer stage16 | 0.03 | 0.00 | 0.07 | 0.86 |
| 25 | evader | pursuer stage18 | 0.05 | 0.00 | 0.11 | 0.83 |
| 25 | evader | pursuer stage20 | 0.11 | 0.01 | 0.14 | 0.69 |
| 25 | evader | pursuer stage22 | 0.11 | 0.00 | 0.19 | 0.57 |
| 25 | evader | pursuer stage24 | 0.20 | 0.03 | 0.18 | 0.54 |

