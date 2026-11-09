# F6b_div_sp summary

hedge_divb = F3 hedge + PSD-PSRO diversity regulariser for evaders (F6b form). F3 arms copied from F3_sampler_sp/results.json. Mean over 2 seeds [range]. other_campaign_evaders: for hedge_divb the F3 evaders (stage-1 excluded); for F3 arms as in F3. Descriptive (2 seeds).

| arm | train | heldout_rl | heldout_worst_cluster | heldout_scripted | x3_exploiters | other_campaign_evaders | own_campaign_evaders | exploitability | evader_vendi |
|---|---|---|---|---|---|---|---|---|---|
| hedge_divb | 0.574 [0.547, 0.600] | 0.708 [0.597, 0.820] | 0.563 [0.406, 0.721] | 0.805 [0.798, 0.813] | 0.954 [0.950, 0.958] | 0.886 [0.882, 0.891] | 0.970 [0.969, 0.972] | 0.983 [0.980, 0.985] | 2.365 [2.335, 2.395] |
| uniform | 0.570 [0.540, 0.601] | 0.736 [0.696, 0.777] | 0.593 [0.540, 0.646] | 0.795 [0.756, 0.834] | 0.951 [0.946, 0.956] | 0.852 [0.843, 0.862] | 0.936 [0.909, 0.964] | 0.973 [0.972, 0.974] | 2.456 [2.376, 2.536] |
| hedge | 0.617 [0.594, 0.640] | 0.836 [0.791, 0.882] | 0.754 [0.692, 0.816] | 0.831 [0.808, 0.853] | 0.964 [0.955, 0.973] | 0.888 [0.871, 0.906] | 0.959 [0.957, 0.961] | 0.979 [0.974, 0.985] | 2.499 [2.454, 2.543] |
| qd_dpp | 0.562 [0.542, 0.582] | 0.696 [0.588, 0.803] | 0.541 [0.377, 0.705] | 0.821 [0.795, 0.847] | 0.958 [0.950, 0.966] | 0.878 [0.872, 0.884] | 0.969 [0.962, 0.977] | 0.977 [0.977, 0.977] | 2.350 [2.297, 2.403] |
| pfsp | 0.542 [0.511, 0.573] | 0.775 [0.639, 0.910] | 0.665 [0.503, 0.827] | 0.826 [0.786, 0.867] | 0.930 [0.895, 0.966] | 0.626 [0.580, 0.672] | 0.776 [0.750, 0.803] | 0.983 [0.982, 0.985] | 2.985 [2.948, 3.023] |

Seed-paired differences:

| pair | train | heldout_rl | heldout_worst_cluster | heldout_scripted | x3_exploiters | other_campaign_evaders | own_campaign_evaders | exploitability | evader_vendi |
|---|---|---|---|---|---|---|---|---|---|
| hedge_divb-hedge | -0.044 [-0.093, +0.006] | -0.128 [-0.285, +0.029] | -0.190 [-0.410, +0.029] | -0.025 [-0.055, +0.005] | -0.010 [-0.023, +0.004] | -0.002 [-0.015, +0.012] | +0.011 [+0.010, +0.012] | +0.003 [-0.005, +0.011] | -0.133 [-0.148, -0.119] |
| hedge_divb-qd_dpp | +0.012 [+0.005, +0.018] | +0.013 [+0.009, +0.017] | +0.023 [+0.016, +0.029] | -0.016 [-0.049, +0.018] | -0.004 [-0.016, +0.009] | +0.008 [-0.002, +0.019] | +0.001 [-0.005, +0.007] | +0.005 [+0.003, +0.008] | +0.016 [-0.067, +0.099] |
| hedge_divb-uniform | +0.003 [-0.001, +0.007] | -0.028 [-0.099, +0.043] | -0.030 [-0.134, +0.075] | +0.010 [-0.022, +0.042] | +0.003 [+0.003, +0.004] | +0.034 [+0.021, +0.048] | +0.034 [+0.008, +0.060] | +0.010 [+0.009, +0.011] | -0.091 [-0.201, +0.019] |
| hedge_divb-pfsp | +0.031 [+0.027, +0.036] | -0.066 [-0.090, -0.042] | -0.101 [-0.106, -0.097] | -0.021 [-0.054, +0.012] | +0.024 [-0.007, +0.054] | +0.260 [+0.211, +0.310] | +0.194 [+0.166, +0.222] | -0.001 [-0.005, +0.003] | -0.620 [-0.628, -0.613] |

Vendi check (F3 values recomputed in the frozen space): thesis-F2-plain: 2.536, thesis-F3-hedge-both-s0: 2.454, thesis-F3-qd_dpp-both-s0: 2.403, thesis-F3-pfsp-both-s0: 2.948, thesis-F3-uniform-s1: 2.376, thesis-F3-hedge-both-s1: 2.543, thesis-F3-qd_dpp-both-s1: 2.297, thesis-F3-pfsp-both-s1: 3.023

hedge_divb - hedge_div (F6), seed-paired (mean [per seed]):

| metric | difference |
|---|---|
| train | +0.148 [+0.215, +0.081] |
| heldout_rl | +0.134 [+0.402, -0.134] |
| heldout_scripted | -0.022 [-0.064, +0.021] |
| x3_exploiters | +0.006 [-0.007, +0.019] |
| other_campaign_evaders | +0.469 [+0.597, +0.341] |
| own_campaign_evaders | +0.014 [+0.009, +0.020] |
| heldout_worst_cluster | +0.097 [+0.457, -0.263] |
| exploitability | +0.004 [+0.007, +0.000] |
| train_cap | +0.212 [+0.260, +0.164] |
| heldout_rl_cap | +0.166 [+0.395, -0.064] |
| heldout_scripted_cap | +0.128 [+0.100, +0.156] |
| x3_exploiters_cap | +0.238 [+0.268, +0.208] |
| other_campaign_evaders_cap | +0.527 [+0.648, +0.406] |
| own_campaign_evaders_cap | +0.074 [+0.060, +0.088] |
| heldout_worst_cluster_cap | +0.128 [+0.420, -0.164] |
| exploitability_cap | -0.004 [+0.001, -0.008] |
| evader_vendi | -0.888 [-0.995, -0.781] |
