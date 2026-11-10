# F6b_div_sp summary

hedge_divb = F3 hedge + PSD-PSRO diversity regulariser for evaders (F6b form). F3 arms copied from F3_sampler_sp/results.json. Mean over 2 seeds [range]. other_campaign_evaders: for hedge_divb the F3 evaders (stage-1 excluded); for F3 arms as in F3. Descriptive (2 seeds).

| arm | train | heldout_rl | heldout_worst_cluster | heldout_scripted | x3_exploiters | other_campaign_evaders | own_campaign_evaders | exploitability | evader_vendi |
|---|---|---|---|---|---|---|---|---|---|
| hedge_divb | 0.737 [0.692, 0.781] | 0.917 [0.860, 0.974] | 0.831 [0.777, 0.884] | 0.904 [0.902, 0.907] | 0.985 [0.985, 0.985] | 0.954 [0.945, 0.963] | 0.996 [0.995, 0.996] | 0.741 [0.681, 0.801] | 2.359 [2.314, 2.403] |
| uniform | 0.715 [0.665, 0.766] | 0.952 [0.944, 0.959] | 0.880 [0.846, 0.914] | 0.901 [0.873, 0.928] | 0.986 [0.984, 0.988] | 0.940 [0.923, 0.957] | 0.979 [0.964, 0.994] | 0.752 [0.749, 0.755] | 2.449 [2.360, 2.539] |
| hedge | 0.784 [0.759, 0.810] | 0.974 [0.960, 0.989] | 0.897 [0.874, 0.921] | 0.920 [0.905, 0.936] | 0.990 [0.987, 0.992] | 0.965 [0.957, 0.973] | 0.994 [0.993, 0.995] | 0.814 [0.773, 0.855] | 2.475 [2.448, 2.502] |
| qd_dpp | 0.729 [0.699, 0.759] | 0.936 [0.900, 0.973] | 0.859 [0.842, 0.877] | 0.911 [0.900, 0.923] | 0.987 [0.986, 0.988] | 0.949 [0.938, 0.960] | 0.996 [0.994, 0.997] | 0.695 [0.632, 0.758] | 2.336 [2.291, 2.380] |
| pfsp | 0.787 [0.778, 0.796] | 0.944 [0.894, 0.994] | 0.884 [0.838, 0.931] | 0.930 [0.913, 0.948] | 0.986 [0.976, 0.995] | 0.898 [0.898, 0.899] | 0.930 [0.916, 0.945] | 0.497 [0.483, 0.512] | 3.019 [2.980, 3.057] |

Seed-paired differences:

| pair | train | heldout_rl | heldout_worst_cluster | heldout_scripted | x3_exploiters | other_campaign_evaders | own_campaign_evaders | exploitability | evader_vendi |
|---|---|---|---|---|---|---|---|---|---|
| hedge_divb-hedge | -0.048 [-0.117, +0.022] | -0.057 [-0.128, +0.014] | -0.067 [-0.143, +0.010] | -0.016 [-0.034, +0.003] | -0.005 [-0.008, -0.002] | -0.011 [-0.028, +0.005] | +0.002 [+0.001, +0.002] | -0.073 [-0.173, +0.028] | -0.117 [-0.135, -0.099] |
| hedge_divb-qd_dpp | +0.008 [-0.006, +0.022] | -0.019 [-0.039, +0.001] | -0.029 [-0.065, +0.007] | -0.007 [-0.021, +0.007] | -0.002 [-0.003, -0.001] | +0.005 [+0.003, +0.006] | -0.000 [-0.001, +0.001] | +0.046 [+0.043, +0.049] | +0.023 [-0.066, +0.112] |
| hedge_divb-uniform | +0.021 [+0.015, +0.028] | -0.035 [-0.084, +0.014] | -0.049 [-0.069, -0.029] | +0.004 [-0.021, +0.029] | -0.001 [-0.003, +0.000] | +0.014 [+0.006, +0.021] | +0.017 [+0.002, +0.031] | -0.010 [-0.067, +0.047] | -0.091 [-0.225, +0.044] |
| hedge_divb-pfsp | -0.050 [-0.085, -0.015] | -0.027 [-0.034, -0.020] | -0.053 [-0.060, -0.047] | -0.026 [-0.041, -0.011] | -0.001 [-0.011, +0.008] | +0.055 [+0.047, +0.064] | +0.065 [+0.051, +0.080] | +0.244 [+0.199, +0.289] | -0.660 [-0.667, -0.653] |

Vendi check (F3 values recomputed in the frozen space): thesis-F2-plain: 2.539, thesis-F3-hedge-both-s0: 2.448, thesis-F3-qd_dpp-both-s0: 2.380, thesis-F3-pfsp-both-s0: 2.980, thesis-F3-uniform-s1: 2.360, thesis-F3-hedge-both-s1: 2.502, thesis-F3-qd_dpp-both-s1: 2.291, thesis-F3-pfsp-both-s1: 3.057

hedge_divb - hedge_div (F6), seed-paired (mean [per seed]):

| metric | difference |
|---|---|
| train | +0.087 [+0.188, -0.015] |
| heldout_rl | +0.107 [+0.328, -0.113] |
| heldout_scripted | -0.037 [-0.060, -0.014] |
| x3_exploiters | -0.007 [-0.008, -0.005] |
| other_campaign_evaders | +0.232 [+0.394, +0.071] |
| own_campaign_evaders | +0.005 [+0.005, +0.005] |
| heldout_worst_cluster | +0.123 [+0.362, -0.116] |
| exploitability | +0.055 [+0.101, +0.009] |
| train_cap | +0.106 [+0.210, +0.002] |
| heldout_rl_cap | +0.115 [+0.355, -0.125] |
| heldout_scripted_cap | -0.020 [-0.040, +0.000] |
| x3_exploiters_cap | +0.019 [+0.024, +0.014] |
| other_campaign_evaders_cap | +0.275 [+0.462, +0.088] |
| own_campaign_evaders_cap | +0.015 [+0.018, +0.012] |
| heldout_worst_cluster_cap | +0.140 [+0.334, -0.055] |
| exploitability_cap | +0.053 [+0.098, +0.007] |
| evader_vendi | -0.859 [-0.995, -0.723] |
