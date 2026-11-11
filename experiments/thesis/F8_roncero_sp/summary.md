# F8_roncero_sp summary

Final pursuers (roncero/0: stage 24; roncero/1: stage 24; hedge/0: stage 8; hedge/1: stage 8). roncero/k = F8 campaign seed k (Roncero's reward and endings, Hedge both roles, 26 stages x 50M = 13 generations); hedge/k = F3 hedge-both seed k (thesis rules, Hedge both roles, 9 x 300M). All evals clean under the thesis rules except vs_exploiter_roncero (Roncero's rules). exploitability_<rules> = 1 - J vs a fresh exploiter trained under those rules; for hedge/k the thesis-rules exploiter is F3c's. other_f3 / other_f8 / own evaders exclude stage-1 evaders. J at 0.15 m is the exact re-score (eval-only). 2 seeds, descriptive; F8 = 26 stages x 50M (13 generations), F3 = 9 stages x 300M.

## Capture radius 0.10 m

| metric | roncero/0 | roncero/1 | hedge/0 | hedge/1 | roncero/0 - hedge/0 | roncero/1 - hedge/1 |
|---|---|---|---|---|---|---|
| heldout_rl | 0.948 | 0.941 | 0.791 | 0.882 | +0.157 | +0.060 |
| heldout_worst_cluster | 0.917 | 0.893 | 0.692 | 0.816 | +0.225 | +0.078 |
| other_f3_evaders | 0.887 | 0.877 | 0.871 | 0.906 | +0.017 | -0.029 |
| other_f8_evaders | 0.877 | 0.935 | 0.826 | 0.874 | +0.051 | +0.061 |
| own_campaign_evaders | 0.916 | 0.940 | 0.957 | 0.961 | -0.041 | -0.022 |
| train | 0.877 | 0.902 | 0.594 | 0.640 | +0.283 | +0.262 |
| heldout_scripted | 0.983 | 0.990 | 0.808 | 0.853 | +0.175 | +0.136 |
| heldout_scripted_pursuer_crash | 0.018 | 0.011 | 0.197 | 0.149 | -0.179 | -0.139 |
| x3_exploiters | 0.999 | 0.999 | 0.955 | 0.973 | +0.044 | +0.026 |
| exploitability_thesis | 0.631 | 0.513 | 0.974 | 0.985 | -0.342 | -0.472 |
| exploiter_thesis_pursuer_crash | 0.010 | 0.030 | 0.975 | 0.988 | -0.965 | -0.957 |
| exploitability_roncero | 0.512 | 0.755 | 0.816 | 0.939 | -0.304 | -0.185 |
| exploiter_roncero_pursuer_crash | 0.001 | 0.006 | 0.106 | 0.023 | -0.105 | -0.017 |
| exploiter_roncero_pursuer_oob | 0.289 | 0.094 | 0.718 | 0.915 | -0.429 | -0.821 |
| exploiter_roncero_timeout | 0.241 | 0.658 | 0.001 | 0.004 | +0.240 | +0.654 |

## Capture radius 0.15 m

| metric | roncero/0 | roncero/1 | hedge/0 | hedge/1 | roncero/0 - hedge/0 | roncero/1 - hedge/1 |
|---|---|---|---|---|---|---|
| heldout_rl | 0.995 | 0.996 | 0.960 | 0.989 | +0.036 | +0.007 |
| heldout_worst_cluster | 0.993 | 0.992 | 0.874 | 0.921 | +0.119 | +0.071 |
| other_f3_evaders | 0.975 | 0.967 | 0.957 | 0.973 | +0.017 | -0.006 |
| other_f8_evaders | 0.950 | 0.987 | 0.959 | 0.979 | -0.009 | +0.008 |
| own_campaign_evaders | 0.939 | 0.971 | 0.993 | 0.995 | -0.054 | -0.024 |
| train | 0.963 | 0.974 | 0.759 | 0.810 | +0.204 | +0.165 |
| heldout_scripted | 0.997 | 0.999 | 0.905 | 0.936 | +0.092 | +0.063 |
| heldout_scripted_pursuer_crash | 0.003 | 0.002 | 0.096 | 0.065 | -0.093 | -0.064 |
| x3_exploiters | 1.000 | 1.000 | 0.987 | 0.992 | +0.013 | +0.008 |
| exploitability_thesis | 0.000 | 0.026 | 0.773 | 0.855 | -0.773 | -0.829 |
| exploiter_thesis_pursuer_crash | 0.000 | 0.003 | 0.774 | 0.856 | -0.774 | -0.854 |
| exploitability_roncero | 0.064 | 0.331 | 0.683 | 0.825 | -0.619 | -0.493 |
| exploiter_roncero_pursuer_crash | 0.000 | 0.001 | 0.085 | 0.017 | -0.085 | -0.016 |
| exploiter_roncero_pursuer_oob | 0.068 | 0.057 | 0.605 | 0.806 | -0.537 | -0.749 |
| exploiter_roncero_timeout | 0.003 | 0.275 | 0.000 | 0.004 | +0.003 | +0.271 |

## In-campaign lure rates (pursuer crash caused by each evader stage in the stage >= 2 pursuers)

Stages 3 / 5 / 7 are comparable across F8 and F3 (F3 has 9 stages, F8 26).

| campaign | evader stage | mean | min - max | n pursuers |
|---|---|---|---|---|
| thesis-F8b-roncero-s0 | 3 | 0.00 | 0.00 - 0.00 | 1 |
| thesis-F8b-roncero-s0 | 5 | 0.00 | 0.00 - 0.00 | 2 |
| thesis-F8b-roncero-s0 | 7 | 0.00 | 0.00 - 0.01 | 3 |
| thesis-F8b-roncero-s0 | 9 | 0.05 | 0.00 - 0.18 | 4 |
| thesis-F8b-roncero-s0 | 11 | 0.00 | 0.00 - 0.01 | 5 |
| thesis-F8b-roncero-s0 | 13 | 0.01 | 0.00 - 0.03 | 6 |
| thesis-F8b-roncero-s0 | 15 | 0.01 | 0.00 - 0.03 | 7 |
| thesis-F8b-roncero-s0 | 17 | 0.01 | 0.00 - 0.08 | 8 |
| thesis-F8b-roncero-s0 | 19 | 0.02 | 0.00 - 0.06 | 9 |
| thesis-F8b-roncero-s0 | 21 | 0.01 | 0.00 - 0.04 | 10 |
| thesis-F8b-roncero-s0 | 23 | 0.01 | 0.00 - 0.08 | 11 |
| thesis-F8b-roncero-s0 | 25 | 0.00 | 0.00 - 0.03 | 12 |
| thesis-F8b-roncero-s100 | 3 | 0.00 | 0.00 - 0.00 | 1 |
| thesis-F8b-roncero-s100 | 5 | 0.00 | 0.00 - 0.00 | 2 |
| thesis-F8b-roncero-s100 | 7 | 0.00 | 0.00 - 0.00 | 3 |
| thesis-F8b-roncero-s100 | 9 | 0.00 | 0.00 - 0.00 | 4 |
| thesis-F8b-roncero-s100 | 11 | 0.02 | 0.00 - 0.04 | 5 |
| thesis-F8b-roncero-s100 | 13 | 0.01 | 0.00 - 0.05 | 6 |
| thesis-F8b-roncero-s100 | 15 | 0.02 | 0.00 - 0.06 | 7 |
| thesis-F8b-roncero-s100 | 17 | 0.05 | 0.00 - 0.26 | 8 |
| thesis-F8b-roncero-s100 | 19 | 0.03 | 0.00 - 0.05 | 9 |
| thesis-F8b-roncero-s100 | 21 | 0.03 | 0.00 - 0.10 | 10 |
| thesis-F8b-roncero-s100 | 23 | 0.02 | 0.00 - 0.07 | 11 |
| thesis-F8b-roncero-s100 | 25 | 0.01 | 0.00 - 0.05 | 12 |
| thesis-F3-hedge-both-s0 | 3 | 0.97 | 0.97 - 0.97 | 1 |
| thesis-F3-hedge-both-s0 | 5 | 0.98 | 0.98 - 0.98 | 2 |
| thesis-F3-hedge-both-s0 | 7 | 0.98 | 0.96 - 0.99 | 3 |
| thesis-F3-hedge-both-s1 | 3 | 0.95 | 0.95 - 0.95 | 1 |
| thesis-F3-hedge-both-s1 | 5 | 0.98 | 0.97 - 0.98 | 2 |
| thesis-F3-hedge-both-s1 | 7 | 0.97 | 0.96 - 0.98 | 3 |

## Inside the F8 campaigns (each stage's eval vs its training opponents, Roncero's rules)

**thesis-F8b-roncero-s0**

| stage | role | vs | capture | pursuer crash | evader crash + OOB | timeout |
|---|---|---|---|---|---|---|
| 0 | pursuer | evader hover | 0.89 | 0.00 | 0.09 | 0.00 |
| 0 | pursuer | evader circular | 0.81 | 0.01 | 0.17 | 0.01 |
| 0 | pursuer | evader flee | 0.86 | 0.00 | 0.14 | 0.00 |
| 0 | pursuer | evader weave | 0.86 | 0.00 | 0.14 | 0.00 |
| 1 | evader | pursuer pure_pursuit | 0.00 | 0.09 | 0.12 | 0.66 |
| 1 | evader | pursuer apf | 0.03 | 0.12 | 0.21 | 0.43 |
| 2 | pursuer | evader stage1 | 0.74 | 0.00 | 0.23 | 0.00 |
| 3 | evader | pursuer stage0 | 0.09 | 0.00 | 0.25 | 0.06 |
| 3 | evader | pursuer stage2 | 0.04 | 0.00 | 0.47 | 0.21 |
| 4 | pursuer | evader stage1 | 0.68 | 0.00 | 0.25 | 0.00 |
| 4 | pursuer | evader stage3 | 0.79 | 0.00 | 0.17 | 0.00 |
| 5 | evader | pursuer stage0 | 0.75 | 0.00 | 0.18 | 0.03 |
| 5 | evader | pursuer stage2 | 0.01 | 0.00 | 0.02 | 0.96 |
| 5 | evader | pursuer stage4 | 0.04 | 0.00 | 0.03 | 0.79 |
| 6 | pursuer | evader stage1 | 0.67 | 0.00 | 0.26 | 0.00 |
| 6 | pursuer | evader stage3 | 0.83 | 0.00 | 0.15 | 0.00 |
| 6 | pursuer | evader stage5 | 0.95 | 0.01 | 0.03 | 0.00 |
| 7 | evader | pursuer stage0 | 0.30 | 0.00 | 0.01 | 0.51 |
| 7 | evader | pursuer stage2 | 0.01 | 0.00 | 0.01 | 0.98 |
| 7 | evader | pursuer stage4 | 0.02 | 0.00 | 0.02 | 0.48 |
| 7 | evader | pursuer stage6 | 0.17 | 0.01 | 0.05 | 0.76 |
| 8 | pursuer | evader stage1 | 0.67 | 0.00 | 0.26 | 0.00 |
| 8 | pursuer | evader stage3 | 0.88 | 0.00 | 0.10 | 0.00 |
| 8 | pursuer | evader stage5 | 0.97 | 0.01 | 0.02 | 0.00 |
| 8 | pursuer | evader stage7 | 0.97 | 0.00 | 0.02 | 0.00 |
| 9 | evader | pursuer stage0 | 0.23 | 0.08 | 0.05 | 0.50 |
| 9 | evader | pursuer stage2 | 0.06 | 0.03 | 0.02 | 0.85 |
| 9 | evader | pursuer stage4 | 0.01 | 0.00 | 0.01 | 0.93 |
| 9 | evader | pursuer stage6 | 0.02 | 0.00 | 0.02 | 0.96 |
| 9 | evader | pursuer stage8 | 0.07 | 0.18 | 0.04 | 0.72 |
| 10 | pursuer | evader stage1 | 0.66 | 0.00 | 0.25 | 0.00 |
| 10 | pursuer | evader stage3 | 0.88 | 0.00 | 0.08 | 0.00 |
| 10 | pursuer | evader stage5 | 0.98 | 0.00 | 0.02 | 0.00 |
| 10 | pursuer | evader stage7 | 0.98 | 0.00 | 0.02 | 0.00 |
| 10 | pursuer | evader stage9 | 0.96 | 0.01 | 0.02 | 0.00 |
| 11 | evader | pursuer stage0 | 0.04 | 0.02 | 0.04 | 0.09 |
| 11 | evader | pursuer stage2 | 0.00 | 0.00 | 0.03 | 0.00 |
| 11 | evader | pursuer stage4 | 0.01 | 0.00 | 0.03 | 0.32 |
| 11 | evader | pursuer stage6 | 0.01 | 0.00 | 0.04 | 0.65 |
| 11 | evader | pursuer stage8 | 0.01 | 0.00 | 0.04 | 0.90 |
| 11 | evader | pursuer stage10 | 0.04 | 0.01 | 0.05 | 0.70 |
| 12 | pursuer | evader stage1 | 0.67 | 0.00 | 0.24 | 0.00 |
| 12 | pursuer | evader stage3 | 0.87 | 0.00 | 0.10 | 0.00 |
| 12 | pursuer | evader stage5 | 0.98 | 0.00 | 0.01 | 0.00 |
| 12 | pursuer | evader stage7 | 0.98 | 0.00 | 0.02 | 0.00 |
| 12 | pursuer | evader stage9 | 0.97 | 0.01 | 0.02 | 0.00 |
| 12 | pursuer | evader stage11 | 0.93 | 0.01 | 0.04 | 0.00 |
| 13 | evader | pursuer stage0 | 0.01 | 0.00 | 0.01 | 0.95 |
| 13 | evader | pursuer stage2 | 0.01 | 0.00 | 0.01 | 0.96 |
| 13 | evader | pursuer stage4 | 0.01 | 0.00 | 0.02 | 0.93 |
| 13 | evader | pursuer stage6 | 0.01 | 0.00 | 0.01 | 0.95 |
| 13 | evader | pursuer stage8 | 0.01 | 0.00 | 0.02 | 0.95 |
| 13 | evader | pursuer stage10 | 0.04 | 0.01 | 0.02 | 0.78 |
| 13 | evader | pursuer stage12 | 0.06 | 0.03 | 0.05 | 0.67 |
| 14 | pursuer | evader stage1 | 0.69 | 0.00 | 0.24 | 0.00 |
| 14 | pursuer | evader stage3 | 0.87 | 0.00 | 0.08 | 0.00 |
| 14 | pursuer | evader stage5 | 0.98 | 0.00 | 0.01 | 0.00 |
| 14 | pursuer | evader stage7 | 0.98 | 0.00 | 0.01 | 0.00 |
| 14 | pursuer | evader stage9 | 0.97 | 0.00 | 0.02 | 0.00 |
| 14 | pursuer | evader stage11 | 0.92 | 0.02 | 0.05 | 0.00 |
| 14 | pursuer | evader stage13 | 0.93 | 0.02 | 0.04 | 0.00 |
| 15 | evader | pursuer stage0 | 0.01 | 0.00 | 0.06 | 0.59 |
| 15 | evader | pursuer stage2 | 0.00 | 0.00 | 0.15 | 0.45 |
| 15 | evader | pursuer stage4 | 0.01 | 0.00 | 0.14 | 0.41 |
| 15 | evader | pursuer stage6 | 0.02 | 0.00 | 0.16 | 0.67 |
| 15 | evader | pursuer stage8 | 0.01 | 0.00 | 0.13 | 0.64 |
| 15 | evader | pursuer stage10 | 0.02 | 0.01 | 0.11 | 0.64 |
| 15 | evader | pursuer stage12 | 0.03 | 0.02 | 0.16 | 0.37 |
| 15 | evader | pursuer stage14 | 0.09 | 0.03 | 0.14 | 0.51 |
| 16 | pursuer | evader stage1 | 0.72 | 0.00 | 0.24 | 0.00 |
| 16 | pursuer | evader stage3 | 0.90 | 0.00 | 0.08 | 0.00 |
| 16 | pursuer | evader stage5 | 0.98 | 0.00 | 0.01 | 0.00 |
| 16 | pursuer | evader stage7 | 0.99 | 0.00 | 0.01 | 0.00 |
| 16 | pursuer | evader stage9 | 0.97 | 0.00 | 0.02 | 0.00 |
| 16 | pursuer | evader stage11 | 0.93 | 0.01 | 0.05 | 0.00 |
| 16 | pursuer | evader stage13 | 0.92 | 0.01 | 0.05 | 0.00 |
| 16 | pursuer | evader stage15 | 0.90 | 0.00 | 0.08 | 0.00 |
| 17 | evader | pursuer stage0 | 0.00 | 0.00 | 0.01 | 0.80 |
| 17 | evader | pursuer stage2 | 0.00 | 0.00 | 0.03 | 0.88 |
| 17 | evader | pursuer stage4 | 0.00 | 0.00 | 0.02 | 0.84 |
| 17 | evader | pursuer stage6 | 0.01 | 0.00 | 0.03 | 0.91 |
| 17 | evader | pursuer stage8 | 0.00 | 0.00 | 0.03 | 0.88 |
| 17 | evader | pursuer stage10 | 0.01 | 0.01 | 0.04 | 0.78 |
| 17 | evader | pursuer stage12 | 0.04 | 0.08 | 0.06 | 0.52 |
| 17 | evader | pursuer stage14 | 0.04 | 0.02 | 0.03 | 0.49 |
| 17 | evader | pursuer stage16 | 0.03 | 0.00 | 0.06 | 0.89 |
| 18 | pursuer | evader stage1 | 0.70 | 0.00 | 0.23 | 0.00 |
| 18 | pursuer | evader stage3 | 0.89 | 0.00 | 0.08 | 0.00 |
| 18 | pursuer | evader stage5 | 0.98 | 0.00 | 0.01 | 0.00 |
| 18 | pursuer | evader stage7 | 0.98 | 0.00 | 0.01 | 0.00 |
| 18 | pursuer | evader stage9 | 0.98 | 0.00 | 0.02 | 0.00 |
| 18 | pursuer | evader stage11 | 0.92 | 0.01 | 0.05 | 0.00 |
| 18 | pursuer | evader stage13 | 0.92 | 0.02 | 0.04 | 0.00 |
| 18 | pursuer | evader stage15 | 0.86 | 0.01 | 0.09 | 0.00 |
| 18 | pursuer | evader stage17 | 0.10 | 0.01 | 0.05 | 0.80 |
| 19 | evader | pursuer stage0 | 0.00 | 0.00 | 0.03 | 0.95 |
| 19 | evader | pursuer stage2 | 0.00 | 0.00 | 0.15 | 0.47 |
| 19 | evader | pursuer stage4 | 0.00 | 0.00 | 0.30 | 0.32 |
| 19 | evader | pursuer stage6 | 0.00 | 0.00 | 0.03 | 0.92 |
| 19 | evader | pursuer stage8 | 0.00 | 0.00 | 0.03 | 0.95 |
| 19 | evader | pursuer stage10 | 0.01 | 0.01 | 0.08 | 0.13 |
| 19 | evader | pursuer stage12 | 0.06 | 0.06 | 0.07 | 0.70 |
| 19 | evader | pursuer stage14 | 0.02 | 0.04 | 0.04 | 0.01 |
| 19 | evader | pursuer stage16 | 0.01 | 0.03 | 0.09 | 0.86 |
| 19 | evader | pursuer stage18 | 0.04 | 0.01 | 0.09 | 0.84 |
| 20 | pursuer | evader stage1 | 0.70 | 0.00 | 0.23 | 0.00 |
| 20 | pursuer | evader stage3 | 0.90 | 0.00 | 0.07 | 0.00 |
| 20 | pursuer | evader stage5 | 0.98 | 0.00 | 0.02 | 0.00 |
| 20 | pursuer | evader stage7 | 0.98 | 0.00 | 0.01 | 0.00 |
| 20 | pursuer | evader stage9 | 0.97 | 0.00 | 0.02 | 0.00 |
| 20 | pursuer | evader stage11 | 0.92 | 0.01 | 0.05 | 0.00 |
| 20 | pursuer | evader stage13 | 0.91 | 0.01 | 0.05 | 0.00 |
| 20 | pursuer | evader stage15 | 0.82 | 0.01 | 0.10 | 0.00 |
| 20 | pursuer | evader stage17 | 0.79 | 0.02 | 0.07 | 0.03 |
| 20 | pursuer | evader stage19 | 0.84 | 0.03 | 0.09 | 0.01 |
| 21 | evader | pursuer stage0 | 0.01 | 0.00 | 0.02 | 0.88 |
| 21 | evader | pursuer stage2 | 0.00 | 0.00 | 0.03 | 0.81 |
| 21 | evader | pursuer stage4 | 0.01 | 0.00 | 0.02 | 0.82 |
| 21 | evader | pursuer stage6 | 0.00 | 0.00 | 0.03 | 0.91 |
| 21 | evader | pursuer stage8 | 0.00 | 0.00 | 0.04 | 0.91 |
| 21 | evader | pursuer stage10 | 0.01 | 0.01 | 0.03 | 0.29 |
| 21 | evader | pursuer stage12 | 0.02 | 0.03 | 0.13 | 0.60 |
| 21 | evader | pursuer stage14 | 0.01 | 0.04 | 0.10 | 0.08 |
| 21 | evader | pursuer stage16 | 0.02 | 0.03 | 0.16 | 0.58 |
| 21 | evader | pursuer stage18 | 0.04 | 0.01 | 0.10 | 0.45 |
| 21 | evader | pursuer stage20 | 0.03 | 0.01 | 0.19 | 0.58 |
| 22 | pursuer | evader stage1 | 0.69 | 0.00 | 0.24 | 0.00 |
| 22 | pursuer | evader stage3 | 0.89 | 0.00 | 0.07 | 0.00 |
| 22 | pursuer | evader stage5 | 0.98 | 0.00 | 0.01 | 0.00 |
| 22 | pursuer | evader stage7 | 0.98 | 0.00 | 0.01 | 0.00 |
| 22 | pursuer | evader stage9 | 0.97 | 0.00 | 0.02 | 0.00 |
| 22 | pursuer | evader stage11 | 0.92 | 0.01 | 0.05 | 0.00 |
| 22 | pursuer | evader stage13 | 0.90 | 0.01 | 0.07 | 0.00 |
| 22 | pursuer | evader stage15 | 0.81 | 0.01 | 0.11 | 0.00 |
| 22 | pursuer | evader stage17 | 0.83 | 0.02 | 0.08 | 0.01 |
| 22 | pursuer | evader stage19 | 0.91 | 0.01 | 0.07 | 0.00 |
| 22 | pursuer | evader stage21 | 0.88 | 0.01 | 0.06 | 0.00 |
| 23 | evader | pursuer stage0 | 0.00 | 0.00 | 0.04 | 0.94 |
| 23 | evader | pursuer stage2 | 0.00 | 0.00 | 0.02 | 0.94 |
| 23 | evader | pursuer stage4 | 0.00 | 0.00 | 0.02 | 0.90 |
| 23 | evader | pursuer stage6 | 0.01 | 0.00 | 0.03 | 0.94 |
| 23 | evader | pursuer stage8 | 0.00 | 0.00 | 0.02 | 0.92 |
| 23 | evader | pursuer stage10 | 0.01 | 0.00 | 0.02 | 0.91 |
| 23 | evader | pursuer stage12 | 0.05 | 0.02 | 0.14 | 0.52 |
| 23 | evader | pursuer stage14 | 0.03 | 0.08 | 0.16 | 0.14 |
| 23 | evader | pursuer stage16 | 0.03 | 0.03 | 0.22 | 0.60 |
| 23 | evader | pursuer stage18 | 0.05 | 0.00 | 0.26 | 0.42 |
| 23 | evader | pursuer stage20 | 0.05 | 0.01 | 0.30 | 0.49 |
| 23 | evader | pursuer stage22 | 0.04 | 0.00 | 0.25 | 0.63 |
| 24 | pursuer | evader stage1 | 0.70 | 0.00 | 0.25 | 0.00 |
| 24 | pursuer | evader stage3 | 0.91 | 0.00 | 0.07 | 0.00 |
| 24 | pursuer | evader stage5 | 0.98 | 0.00 | 0.01 | 0.00 |
| 24 | pursuer | evader stage7 | 0.98 | 0.00 | 0.01 | 0.00 |
| 24 | pursuer | evader stage9 | 0.98 | 0.00 | 0.02 | 0.00 |
| 24 | pursuer | evader stage11 | 0.91 | 0.01 | 0.05 | 0.00 |
| 24 | pursuer | evader stage13 | 0.91 | 0.01 | 0.06 | 0.00 |
| 24 | pursuer | evader stage15 | 0.82 | 0.02 | 0.10 | 0.00 |
| 24 | pursuer | evader stage17 | 0.87 | 0.01 | 0.06 | 0.02 |
| 24 | pursuer | evader stage19 | 0.91 | 0.02 | 0.06 | 0.00 |
| 24 | pursuer | evader stage21 | 0.91 | 0.01 | 0.05 | 0.00 |
| 24 | pursuer | evader stage23 | 0.89 | 0.01 | 0.06 | 0.01 |
| 25 | evader | pursuer stage0 | 0.00 | 0.00 | 0.02 | 0.96 |
| 25 | evader | pursuer stage2 | 0.00 | 0.00 | 0.01 | 0.96 |
| 25 | evader | pursuer stage4 | 0.00 | 0.00 | 0.03 | 0.85 |
| 25 | evader | pursuer stage6 | 0.00 | 0.00 | 0.02 | 0.96 |
| 25 | evader | pursuer stage8 | 0.00 | 0.00 | 0.01 | 0.97 |
| 25 | evader | pursuer stage10 | 0.00 | 0.00 | 0.02 | 0.82 |
| 25 | evader | pursuer stage12 | 0.04 | 0.00 | 0.10 | 0.58 |
| 25 | evader | pursuer stage14 | 0.06 | 0.03 | 0.06 | 0.54 |
| 25 | evader | pursuer stage16 | 0.02 | 0.00 | 0.09 | 0.72 |
| 25 | evader | pursuer stage18 | 0.06 | 0.00 | 0.07 | 0.56 |
| 25 | evader | pursuer stage20 | 0.02 | 0.00 | 0.07 | 0.66 |
| 25 | evader | pursuer stage22 | 0.01 | 0.00 | 0.07 | 0.85 |
| 25 | evader | pursuer stage24 | 0.02 | 0.00 | 0.15 | 0.72 |

**thesis-F8b-roncero-s100**

| stage | role | vs | capture | pursuer crash | evader crash + OOB | timeout |
|---|---|---|---|---|---|---|
| 0 | pursuer | evader hover | 0.89 | 0.01 | 0.10 | 0.00 |
| 0 | pursuer | evader circular | 0.85 | 0.01 | 0.14 | 0.00 |
| 0 | pursuer | evader flee | 0.86 | 0.01 | 0.13 | 0.00 |
| 0 | pursuer | evader weave | 0.86 | 0.00 | 0.14 | 0.00 |
| 1 | evader | pursuer pure_pursuit | 0.01 | 0.10 | 0.16 | 0.61 |
| 1 | evader | pursuer apf | 0.19 | 0.14 | 0.32 | 0.18 |
| 2 | pursuer | evader stage1 | 0.79 | 0.00 | 0.18 | 0.00 |
| 3 | evader | pursuer stage0 | 0.06 | 0.00 | 0.12 | 0.02 |
| 3 | evader | pursuer stage2 | 0.08 | 0.00 | 0.21 | 0.16 |
| 4 | pursuer | evader stage1 | 0.76 | 0.00 | 0.21 | 0.00 |
| 4 | pursuer | evader stage3 | 0.88 | 0.01 | 0.09 | 0.00 |
| 5 | evader | pursuer stage0 | 0.03 | 0.00 | 0.02 | 0.95 |
| 5 | evader | pursuer stage2 | 0.02 | 0.00 | 0.02 | 0.94 |
| 5 | evader | pursuer stage4 | 0.05 | 0.00 | 0.02 | 0.93 |
| 6 | pursuer | evader stage1 | 0.75 | 0.00 | 0.21 | 0.00 |
| 6 | pursuer | evader stage3 | 0.90 | 0.01 | 0.08 | 0.00 |
| 6 | pursuer | evader stage5 | 0.97 | 0.01 | 0.02 | 0.00 |
| 7 | evader | pursuer stage0 | 0.01 | 0.00 | 0.03 | 0.92 |
| 7 | evader | pursuer stage2 | 0.02 | 0.00 | 0.03 | 0.94 |
| 7 | evader | pursuer stage4 | 0.02 | 0.00 | 0.04 | 0.94 |
| 7 | evader | pursuer stage6 | 0.04 | 0.00 | 0.04 | 0.92 |
| 8 | pursuer | evader stage1 | 0.74 | 0.00 | 0.21 | 0.00 |
| 8 | pursuer | evader stage3 | 0.89 | 0.01 | 0.08 | 0.00 |
| 8 | pursuer | evader stage5 | 0.96 | 0.01 | 0.02 | 0.00 |
| 8 | pursuer | evader stage7 | 0.96 | 0.01 | 0.03 | 0.00 |
| 9 | evader | pursuer stage0 | 0.02 | 0.00 | 0.03 | 0.94 |
| 9 | evader | pursuer stage2 | 0.02 | 0.00 | 0.02 | 0.96 |
| 9 | evader | pursuer stage4 | 0.02 | 0.00 | 0.02 | 0.95 |
| 9 | evader | pursuer stage6 | 0.04 | 0.00 | 0.02 | 0.93 |
| 9 | evader | pursuer stage8 | 0.07 | 0.00 | 0.03 | 0.89 |
| 10 | pursuer | evader stage1 | 0.75 | 0.00 | 0.20 | 0.00 |
| 10 | pursuer | evader stage3 | 0.89 | 0.01 | 0.08 | 0.00 |
| 10 | pursuer | evader stage5 | 0.97 | 0.01 | 0.02 | 0.00 |
| 10 | pursuer | evader stage7 | 0.96 | 0.00 | 0.03 | 0.00 |
| 10 | pursuer | evader stage9 | 0.97 | 0.01 | 0.02 | 0.00 |
| 11 | evader | pursuer stage0 | 0.02 | 0.01 | 0.01 | 0.94 |
| 11 | evader | pursuer stage2 | 0.03 | 0.00 | 0.02 | 0.91 |
| 11 | evader | pursuer stage4 | 0.01 | 0.02 | 0.02 | 0.95 |
| 11 | evader | pursuer stage6 | 0.04 | 0.01 | 0.02 | 0.93 |
| 11 | evader | pursuer stage8 | 0.04 | 0.02 | 0.02 | 0.91 |
| 11 | evader | pursuer stage10 | 0.08 | 0.04 | 0.04 | 0.81 |
| 12 | pursuer | evader stage1 | 0.76 | 0.00 | 0.20 | 0.00 |
| 12 | pursuer | evader stage3 | 0.89 | 0.01 | 0.09 | 0.00 |
| 12 | pursuer | evader stage5 | 0.97 | 0.01 | 0.02 | 0.00 |
| 12 | pursuer | evader stage7 | 0.95 | 0.01 | 0.04 | 0.00 |
| 12 | pursuer | evader stage9 | 0.95 | 0.01 | 0.03 | 0.00 |
| 12 | pursuer | evader stage11 | 0.85 | 0.02 | 0.02 | 0.09 |
| 13 | evader | pursuer stage0 | 0.10 | 0.00 | 0.04 | 0.62 |
| 13 | evader | pursuer stage2 | 0.06 | 0.00 | 0.02 | 0.71 |
| 13 | evader | pursuer stage4 | 0.03 | 0.05 | 0.01 | 0.88 |
| 13 | evader | pursuer stage6 | 0.02 | 0.00 | 0.02 | 0.95 |
| 13 | evader | pursuer stage8 | 0.02 | 0.00 | 0.01 | 0.96 |
| 13 | evader | pursuer stage10 | 0.06 | 0.01 | 0.02 | 0.90 |
| 13 | evader | pursuer stage12 | 0.16 | 0.01 | 0.03 | 0.79 |
| 14 | pursuer | evader stage1 | 0.76 | 0.00 | 0.19 | 0.00 |
| 14 | pursuer | evader stage3 | 0.89 | 0.01 | 0.08 | 0.00 |
| 14 | pursuer | evader stage5 | 0.97 | 0.01 | 0.02 | 0.00 |
| 14 | pursuer | evader stage7 | 0.95 | 0.01 | 0.04 | 0.00 |
| 14 | pursuer | evader stage9 | 0.96 | 0.01 | 0.03 | 0.00 |
| 14 | pursuer | evader stage11 | 0.96 | 0.01 | 0.02 | 0.00 |
| 14 | pursuer | evader stage13 | 0.96 | 0.01 | 0.02 | 0.00 |
| 15 | evader | pursuer stage0 | 0.01 | 0.01 | 0.03 | 0.88 |
| 15 | evader | pursuer stage2 | 0.01 | 0.00 | 0.03 | 0.92 |
| 15 | evader | pursuer stage4 | 0.03 | 0.04 | 0.06 | 0.56 |
| 15 | evader | pursuer stage6 | 0.16 | 0.01 | 0.10 | 0.46 |
| 15 | evader | pursuer stage8 | 0.06 | 0.06 | 0.10 | 0.46 |
| 15 | evader | pursuer stage10 | 0.02 | 0.01 | 0.04 | 0.76 |
| 15 | evader | pursuer stage12 | 0.02 | 0.00 | 0.03 | 0.61 |
| 15 | evader | pursuer stage14 | 0.04 | 0.00 | 0.04 | 0.30 |
| 16 | pursuer | evader stage1 | 0.75 | 0.00 | 0.17 | 0.00 |
| 16 | pursuer | evader stage3 | 0.91 | 0.00 | 0.07 | 0.00 |
| 16 | pursuer | evader stage5 | 0.97 | 0.00 | 0.02 | 0.00 |
| 16 | pursuer | evader stage7 | 0.95 | 0.01 | 0.04 | 0.00 |
| 16 | pursuer | evader stage9 | 0.96 | 0.01 | 0.02 | 0.00 |
| 16 | pursuer | evader stage11 | 0.97 | 0.01 | 0.01 | 0.00 |
| 16 | pursuer | evader stage13 | 0.95 | 0.01 | 0.02 | 0.00 |
| 16 | pursuer | evader stage15 | 0.09 | 0.00 | 0.04 | 0.81 |
| 17 | evader | pursuer stage0 | 0.00 | 0.00 | 0.04 | 0.92 |
| 17 | evader | pursuer stage2 | 0.00 | 0.00 | 0.04 | 0.92 |
| 17 | evader | pursuer stage4 | 0.01 | 0.03 | 0.03 | 0.77 |
| 17 | evader | pursuer stage6 | 0.01 | 0.03 | 0.02 | 0.83 |
| 17 | evader | pursuer stage8 | 0.02 | 0.26 | 0.04 | 0.45 |
| 17 | evader | pursuer stage10 | 0.02 | 0.07 | 0.04 | 0.40 |
| 17 | evader | pursuer stage12 | 0.02 | 0.00 | 0.05 | 0.68 |
| 17 | evader | pursuer stage14 | 0.03 | 0.01 | 0.06 | 0.43 |
| 17 | evader | pursuer stage16 | 0.05 | 0.00 | 0.07 | 0.82 |
| 18 | pursuer | evader stage1 | 0.76 | 0.00 | 0.18 | 0.00 |
| 18 | pursuer | evader stage3 | 0.90 | 0.01 | 0.08 | 0.00 |
| 18 | pursuer | evader stage5 | 0.97 | 0.00 | 0.02 | 0.00 |
| 18 | pursuer | evader stage7 | 0.96 | 0.01 | 0.03 | 0.00 |
| 18 | pursuer | evader stage9 | 0.96 | 0.01 | 0.02 | 0.00 |
| 18 | pursuer | evader stage11 | 0.96 | 0.01 | 0.02 | 0.00 |
| 18 | pursuer | evader stage13 | 0.95 | 0.01 | 0.02 | 0.00 |
| 18 | pursuer | evader stage15 | 0.93 | 0.00 | 0.04 | 0.00 |
| 18 | pursuer | evader stage17 | 0.92 | 0.00 | 0.06 | 0.00 |
| 19 | evader | pursuer stage0 | 0.01 | 0.01 | 0.12 | 0.70 |
| 19 | evader | pursuer stage2 | 0.01 | 0.00 | 0.14 | 0.81 |
| 19 | evader | pursuer stage4 | 0.04 | 0.05 | 0.06 | 0.13 |
| 19 | evader | pursuer stage6 | 0.04 | 0.03 | 0.08 | 0.10 |
| 19 | evader | pursuer stage8 | 0.03 | 0.04 | 0.08 | 0.13 |
| 19 | evader | pursuer stage10 | 0.01 | 0.01 | 0.07 | 0.34 |
| 19 | evader | pursuer stage12 | 0.01 | 0.00 | 0.10 | 0.29 |
| 19 | evader | pursuer stage14 | 0.04 | 0.02 | 0.09 | 0.02 |
| 19 | evader | pursuer stage16 | 0.02 | 0.05 | 0.13 | 0.04 |
| 19 | evader | pursuer stage18 | 0.11 | 0.04 | 0.13 | 0.47 |
| 20 | pursuer | evader stage1 | 0.76 | 0.00 | 0.16 | 0.00 |
| 20 | pursuer | evader stage3 | 0.90 | 0.01 | 0.08 | 0.00 |
| 20 | pursuer | evader stage5 | 0.96 | 0.01 | 0.02 | 0.00 |
| 20 | pursuer | evader stage7 | 0.96 | 0.01 | 0.03 | 0.00 |
| 20 | pursuer | evader stage9 | 0.95 | 0.01 | 0.02 | 0.00 |
| 20 | pursuer | evader stage11 | 0.96 | 0.01 | 0.02 | 0.00 |
| 20 | pursuer | evader stage13 | 0.94 | 0.01 | 0.02 | 0.00 |
| 20 | pursuer | evader stage15 | 0.91 | 0.00 | 0.04 | 0.00 |
| 20 | pursuer | evader stage17 | 0.92 | 0.00 | 0.05 | 0.00 |
| 20 | pursuer | evader stage19 | 0.84 | 0.01 | 0.09 | 0.01 |
| 21 | evader | pursuer stage0 | 0.01 | 0.01 | 0.07 | 0.84 |
| 21 | evader | pursuer stage2 | 0.00 | 0.00 | 0.07 | 0.90 |
| 21 | evader | pursuer stage4 | 0.02 | 0.10 | 0.07 | 0.39 |
| 21 | evader | pursuer stage6 | 0.00 | 0.05 | 0.07 | 0.48 |
| 21 | evader | pursuer stage8 | 0.01 | 0.04 | 0.07 | 0.42 |
| 21 | evader | pursuer stage10 | 0.01 | 0.02 | 0.06 | 0.74 |
| 21 | evader | pursuer stage12 | 0.01 | 0.01 | 0.07 | 0.79 |
| 21 | evader | pursuer stage14 | 0.04 | 0.02 | 0.07 | 0.24 |
| 21 | evader | pursuer stage16 | 0.05 | 0.02 | 0.08 | 0.43 |
| 21 | evader | pursuer stage18 | 0.09 | 0.01 | 0.12 | 0.42 |
| 21 | evader | pursuer stage20 | 0.06 | 0.00 | 0.13 | 0.76 |
| 22 | pursuer | evader stage1 | 0.77 | 0.00 | 0.15 | 0.00 |
| 22 | pursuer | evader stage3 | 0.90 | 0.01 | 0.08 | 0.00 |
| 22 | pursuer | evader stage5 | 0.97 | 0.01 | 0.02 | 0.00 |
| 22 | pursuer | evader stage7 | 0.97 | 0.01 | 0.03 | 0.00 |
| 22 | pursuer | evader stage9 | 0.97 | 0.00 | 0.02 | 0.00 |
| 22 | pursuer | evader stage11 | 0.97 | 0.01 | 0.02 | 0.00 |
| 22 | pursuer | evader stage13 | 0.96 | 0.01 | 0.02 | 0.00 |
| 22 | pursuer | evader stage15 | 0.94 | 0.00 | 0.04 | 0.00 |
| 22 | pursuer | evader stage17 | 0.95 | 0.00 | 0.04 | 0.00 |
| 22 | pursuer | evader stage19 | 0.90 | 0.01 | 0.06 | 0.00 |
| 22 | pursuer | evader stage21 | 0.93 | 0.00 | 0.05 | 0.00 |
| 23 | evader | pursuer stage0 | 0.03 | 0.01 | 0.04 | 0.84 |
| 23 | evader | pursuer stage2 | 0.02 | 0.01 | 0.06 | 0.89 |
| 23 | evader | pursuer stage4 | 0.02 | 0.07 | 0.04 | 0.44 |
| 23 | evader | pursuer stage6 | 0.01 | 0.02 | 0.06 | 0.57 |
| 23 | evader | pursuer stage8 | 0.03 | 0.03 | 0.04 | 0.45 |
| 23 | evader | pursuer stage10 | 0.03 | 0.01 | 0.03 | 0.70 |
| 23 | evader | pursuer stage12 | 0.01 | 0.01 | 0.05 | 0.22 |
| 23 | evader | pursuer stage14 | 0.01 | 0.01 | 0.04 | 0.38 |
| 23 | evader | pursuer stage16 | 0.02 | 0.00 | 0.08 | 0.49 |
| 23 | evader | pursuer stage18 | 0.04 | 0.01 | 0.10 | 0.58 |
| 23 | evader | pursuer stage20 | 0.03 | 0.00 | 0.08 | 0.66 |
| 23 | evader | pursuer stage22 | 0.13 | 0.01 | 0.08 | 0.61 |
| 24 | pursuer | evader stage1 | 0.76 | 0.00 | 0.15 | 0.00 |
| 24 | pursuer | evader stage3 | 0.89 | 0.01 | 0.08 | 0.00 |
| 24 | pursuer | evader stage5 | 0.98 | 0.00 | 0.02 | 0.00 |
| 24 | pursuer | evader stage7 | 0.97 | 0.00 | 0.03 | 0.00 |
| 24 | pursuer | evader stage9 | 0.97 | 0.01 | 0.02 | 0.00 |
| 24 | pursuer | evader stage11 | 0.97 | 0.01 | 0.02 | 0.00 |
| 24 | pursuer | evader stage13 | 0.95 | 0.01 | 0.01 | 0.00 |
| 24 | pursuer | evader stage15 | 0.95 | 0.00 | 0.03 | 0.00 |
| 24 | pursuer | evader stage17 | 0.95 | 0.00 | 0.03 | 0.00 |
| 24 | pursuer | evader stage19 | 0.91 | 0.00 | 0.06 | 0.00 |
| 24 | pursuer | evader stage21 | 0.92 | 0.01 | 0.05 | 0.00 |
| 24 | pursuer | evader stage23 | 0.90 | 0.01 | 0.04 | 0.00 |
| 25 | evader | pursuer stage0 | 0.03 | 0.01 | 0.23 | 0.19 |
| 25 | evader | pursuer stage2 | 0.01 | 0.01 | 0.13 | 0.59 |
| 25 | evader | pursuer stage4 | 0.01 | 0.05 | 0.15 | 0.05 |
| 25 | evader | pursuer stage6 | 0.01 | 0.01 | 0.13 | 0.09 |
| 25 | evader | pursuer stage8 | 0.02 | 0.01 | 0.09 | 0.07 |
| 25 | evader | pursuer stage10 | 0.02 | 0.00 | 0.10 | 0.21 |
| 25 | evader | pursuer stage12 | 0.01 | 0.00 | 0.11 | 0.01 |
| 25 | evader | pursuer stage14 | 0.02 | 0.00 | 0.13 | 0.10 |
| 25 | evader | pursuer stage16 | 0.01 | 0.00 | 0.15 | 0.61 |
| 25 | evader | pursuer stage18 | 0.03 | 0.01 | 0.24 | 0.22 |
| 25 | evader | pursuer stage20 | 0.02 | 0.00 | 0.19 | 0.53 |
| 25 | evader | pursuer stage22 | 0.03 | 0.01 | 0.22 | 0.55 |
| 25 | evader | pursuer stage24 | 0.11 | 0.01 | 0.15 | 0.30 |

