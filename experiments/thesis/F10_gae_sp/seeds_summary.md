# F10 seeds (amendment 1) summary

Final pursuers (stage 12) of thesis-F10-nofix-s{0,100,200} (thesis rules, no GAE fix, Hedge both roles, 13 x 50M) vs F13 / F9 control (same with the fix, paired by seed). Evals clean under the thesis rules except f8_evaders_roncero (Roncero's rules). 3 seeds, descriptive.

## Capture radius 0.10 m

| metric | F10 s0 | F10 s100 | F10 s200 | F10 mean | F13 s0 | F13 s100 | F13 s200 | F13 mean |
|---|---|---|---|---|---|---|---|---|
| train | 0.838 | 0.656 | 0.775 | 0.756 | 0.688 | 0.717 | 0.694 | 0.700 |
| heldout_rl | 0.933 | 0.916 | 0.925 | 0.924 | 0.893 | 0.894 | 0.826 | 0.871 |
| heldout_scripted | 0.968 | 0.901 | 0.943 | 0.937 | 0.918 | 0.948 | 0.947 | 0.938 |
| x3_exploiters | 0.993 | 0.985 | 0.987 | 0.988 | 0.983 | 0.992 | 0.992 | 0.989 |
| other_f3_evaders | 0.826 | 0.718 | 0.776 | 0.773 | 0.766 | 0.664 | 0.619 | 0.683 |
| other_f9_evaders | 0.890 | 0.854 | 0.882 | 0.875 | - | - | - | - |
| own_campaign_evaders | 0.813 | 0.945 | 0.915 | 0.891 | 0.934 | 0.946 | 0.932 | 0.937 |
| heldout_worst_cluster | 0.890 | 0.871 | 0.900 | 0.887 | 0.859 | 0.854 | 0.757 | 0.824 |
| heldout_scripted_pursuer_crash | 0.033 | 0.102 | 0.060 | 0.065 | 0.085 | 0.054 | 0.055 | 0.065 |
| exploitability_thesis | 0.172 | 0.808 | 0.597 | 0.525 | 0.440 | 0.915 | 0.741 | 0.699 |
| exploiter_thesis_pursuer_crash | 0.031 | 0.825 | 0.572 | 0.476 | 0.056 | 0.925 | 0.751 | 0.577 |
| exploiter_thesis_timeout | 0.143 | 0.000 | 0.031 | 0.058 | 0.385 | 0.000 | 0.000 | 0.128 |
| strong_finals_mean | 0.828 | 0.605 | 0.699 | 0.711 | 0.656 | 0.583 | 0.563 | 0.601 |
| strong_finals_min | 0.658 | 0.402 | 0.486 | 0.515 | 0.404 | 0.376 | 0.275 | 0.352 |
| vs_f8_finals | 0.884 | 0.856 | 0.922 | 0.887 | 0.792 | 0.713 | 0.804 | 0.770 |
| vs_f3_hedge_finals | 0.824 | 0.424 | 0.560 | 0.603 | 0.521 | 0.466 | 0.439 | 0.475 |
| cross_play | 0.860 | 0.777 | 0.881 | 0.839 | 0.817 | 0.703 | 0.650 | 0.723 |
| other_f10_evaders | 0.868 | 0.772 | 0.847 | 0.829 | - | - | - | - |
| f8_evaders_thesis | 0.926 | 0.909 | 0.923 | 0.920 | - | - | - | - |
| f8_evaders_roncero | 0.707 | 0.803 | 0.745 | 0.752 | - | - | - | - |
| f8_evaders_roncero_pursuer_oob | 0.270 | 0.149 | 0.222 | 0.213 | - | - | - | - |

## Capture radius 0.15 m

| metric | F10 s0 | F10 s100 | F10 s200 | F10 mean | F13 s0 | F13 s100 | F13 s200 | F13 mean |
|---|---|---|---|---|---|---|---|---|
| train | 0.949 | 0.827 | 0.914 | 0.897 | 0.840 | 0.881 | 0.866 | 0.862 |
| heldout_rl | 0.996 | 0.995 | 0.995 | 0.995 | 0.989 | 0.991 | 0.977 | 0.986 |
| heldout_scripted | 0.998 | 0.968 | 0.990 | 0.985 | 0.978 | 0.988 | 0.987 | 0.984 |
| x3_exploiters | 1.000 | 0.999 | 0.999 | 0.999 | 0.998 | 1.000 | 0.999 | 0.999 |
| other_f3_evaders | 0.957 | 0.943 | 0.966 | 0.956 | 0.948 | 0.911 | 0.869 | 0.909 |
| other_f9_evaders | 0.982 | 0.977 | 0.980 | 0.980 | - | - | - | - |
| own_campaign_evaders | 0.964 | 0.996 | 0.992 | 0.984 | 0.995 | 0.996 | 0.996 | 0.995 |
| heldout_worst_cluster | 0.991 | 0.958 | 0.986 | 0.979 | 0.973 | 0.982 | 0.969 | 0.975 |
| heldout_scripted_pursuer_crash | 0.002 | 0.032 | 0.011 | 0.015 | 0.023 | 0.013 | 0.013 | 0.016 |
| exploitability_thesis | 0.004 | 0.547 | 0.277 | 0.276 | 0.095 | 0.476 | 0.206 | 0.259 |
| exploiter_thesis_pursuer_crash | 0.004 | 0.552 | 0.264 | 0.273 | 0.008 | 0.477 | 0.208 | 0.231 |
| exploiter_thesis_timeout | 0.000 | 0.000 | 0.016 | 0.005 | 0.086 | 0.000 | 0.000 | 0.029 |
| strong_finals_mean | 0.947 | 0.885 | 0.933 | 0.922 | 0.891 | 0.850 | 0.808 | 0.850 |
| strong_finals_min | 0.898 | 0.700 | 0.829 | 0.809 | 0.710 | 0.735 | 0.674 | 0.706 |
| vs_f8_finals | 0.982 | 0.972 | 0.988 | 0.981 | 0.943 | 0.898 | 0.954 | 0.932 |
| vs_f3_hedge_finals | 0.925 | 0.845 | 0.916 | 0.895 | 0.840 | 0.785 | 0.691 | 0.772 |
| cross_play | 0.995 | 0.971 | 0.995 | 0.987 | 0.981 | 0.953 | 0.935 | 0.957 |
| other_f10_evaders | 0.978 | 0.956 | 0.961 | 0.965 | - | - | - | - |
| f8_evaders_thesis | 0.987 | 0.989 | 0.989 | 0.989 | - | - | - | - |
| f8_evaders_roncero | 0.879 | 0.940 | 0.901 | 0.907 | - | - | - | - |
| f8_evaders_roncero_pursuer_oob | 0.118 | 0.056 | 0.096 | 0.090 | - | - | - | - |

## In-campaign lure rate (a), mean over evader stages 3-11

- f10/0: 0.81
- f10/100: 0.39
- f10/200: 0.33
- f13/0: 0.36
- f13/100: 0.14
- f13/200: 0.15

## F10 - F13 (mean over seed pairs, 0.10 m)

- heldout_rl: +0.054
- strong_finals_mean: +0.110
- exploitability_thesis: -0.173
- exploiter_thesis_pursuer_crash: -0.101
- cross_play: +0.116

## Pre-registered predictions (new seeds 100 / 200)

- 1_exploiter_crash_lt_0.50: did not hold
- 2_strong_finals_ge_0.80: did not hold
- 3_heldout_rl_ge_0.90: held
- 4_roncero_f8_evaders_below_f8: held
- 5_gae_fix_small: did not hold
