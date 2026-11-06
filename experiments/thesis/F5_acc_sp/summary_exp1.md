# F5_acc_sp summary (Exp 1 only: Exp 2 / qd_dpp_rls paused 11-05, user)

Final pursuer (stage 8) of each campaign; mean over the campaign seeds (uniform_rls / uniform_plain: seeds 0, 100, 300 after amendment 2; uniform_fd 1) [95 % bootstrap CI over seeds; with 2 values it is their range]. Campaign evaders include the seed-300 campaigns' evaders (amendment 2). *_rls = learned acceleration estimate + RLS (F5); *_plain = plain observation (F2 / F3); uniform_fd = finite-difference acceleration (F2-acc). noisy = A3c sensor noise at eval; *_frozen = F5 pursuer evaluated with RLS frozen. exploitability = 1 - J (or capture) vs a fresh exploiter. J = capture + evader crash + evader out of bounds; cap = capture only. other / own campaign evaders exclude stage-1 evaders (amendment 1). Estimator retrained (amendment 1): `estimator/summary.md`.

## J, clean

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.871 [0.812, 0.921] | 0.789 [0.708, 0.839] | 0.835 [0.742, 0.895] | 0.957 [0.949, 0.969] | 0.619 [0.559, 0.656] | 0.867 [0.840, 0.898] | 0.975 [0.964, 0.982] | 0.972 [0.953, 0.983] |
| uniform_plain | 0.780 [0.696, 0.866] | 0.661 [0.540, 0.797] | 0.862 [0.837, 0.899] | 0.974 [0.972, 0.977] | 0.595 [0.540, 0.644] | 0.808 [0.756, 0.834] | 0.955 [0.946, 0.964] | 0.946 [0.918, 0.967] |
| uniform_fd | 0.893 [0.893, 0.893] | 0.827 [0.827, 0.827] | 0.657 [0.657, 0.657] | 0.965 [0.965, 0.965] | 0.495 [0.495, 0.495] | 0.876 [0.876, 0.876] | 0.986 [0.986, 0.986] | - |

## J, noisy

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.731 [0.717, 0.745] | 0.614 [0.597, 0.645] | 0.640 [0.575, 0.700] | 0.944 [0.929, 0.956] | 0.539 [0.497, 0.563] | 0.827 [0.789, 0.846] | 0.923 [0.896, 0.937] | 0.750 [0.657, 0.834] |
| uniform_plain | 0.645 [0.548, 0.746] | 0.487 [0.348, 0.655] | 0.780 [0.765, 0.807] | 0.976 [0.972, 0.978] | 0.541 [0.483, 0.581] | 0.762 [0.686, 0.811] | 0.903 [0.867, 0.924] | 0.865 [0.851, 0.892] |
| uniform_fd | 0.541 [0.541, 0.541] | 0.440 [0.440, 0.440] | 0.293 [0.293, 0.293] | 0.751 [0.751, 0.751] | 0.398 [0.398, 0.398] | 0.805 [0.805, 0.805] | 0.892 [0.892, 0.892] | - |

## J, clean_frozen

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.872 [0.809, 0.926] | 0.791 [0.701, 0.853] | 0.823 [0.720, 0.894] | 0.942 [0.923, 0.955] | 0.618 [0.554, 0.657] | 0.867 [0.836, 0.896] | 0.980 [0.973, 0.985] | 0.965 [0.952, 0.973] |

## J, noisy_frozen

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.700 [0.681, 0.736] | 0.575 [0.534, 0.638] | 0.744 [0.715, 0.768] | 0.916 [0.893, 0.938] | 0.564 [0.531, 0.592] | 0.818 [0.788, 0.839] | 0.912 [0.884, 0.934] | 0.824 [0.795, 0.845] |

## J, seed-paired differences

| contrast | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls:clean - uniform_plain:clean | +0.091 [+0.055, +0.116] | +0.128 [+0.042, +0.174] | -0.027 [-0.157, +0.046] | -0.018 [-0.028, -0.005] | +0.024 [-0.085, +0.102] | +0.059 [+0.005, +0.142] | +0.020 [+0.008, +0.034] | +0.026 [+0.013, +0.035] |
| uniform_rls:noisy - uniform_plain:noisy | +0.086 [-0.014, +0.198] | +0.127 [-0.058, +0.296] | -0.140 [-0.232, -0.065] | -0.032 [-0.050, -0.022] | -0.002 [-0.085, +0.079] | +0.065 [-0.021, +0.159] | +0.020 [-0.028, +0.071] | -0.115 [-0.235, -0.017] |
| uniform_fd:clean - uniform_plain:clean | +0.117 [+0.117, +0.117] | +0.181 [+0.181, +0.181] | -0.193 [-0.193, -0.193] | -0.010 [-0.010, -0.010] | -0.106 [-0.106, -0.106] | +0.042 [+0.042, +0.042] | +0.030 [+0.030, +0.030] | - |
| uniform_fd:noisy - uniform_plain:noisy | -0.100 [-0.100, -0.100] | -0.017 [-0.017, -0.017] | -0.472 [-0.472, -0.472] | -0.226 [-0.226, -0.226] | -0.161 [-0.161, -0.161] | -0.005 [-0.005, -0.005] | -0.032 [-0.032, -0.032] | - |
| uniform_rls:clean - uniform_rls:clean_frozen | -0.001 [-0.005, +0.004] | -0.002 [-0.014, +0.007] | +0.012 [+0.001, +0.022] | +0.015 [+0.003, +0.027] | +0.001 [-0.001, +0.005] | -0.000 [-0.007, +0.004] | -0.004 [-0.009, -0.002] | +0.007 [+0.002, +0.011] |
| uniform_rls:noisy - uniform_rls:noisy_frozen | +0.031 [+0.009, +0.049] | +0.039 [+0.006, +0.063] | -0.103 [-0.140, -0.069] | +0.027 [+0.018, +0.035] | -0.025 [-0.034, -0.012] | +0.009 [+0.002, +0.018] | +0.011 [+0.003, +0.017] | -0.073 [-0.173, +0.039] |

## Capture only, clean

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.765 [0.648, 0.902] | 0.622 [0.477, 0.758] | 0.761 [0.684, 0.813] | 0.975 [0.962, 0.983] | 0.523 [0.510, 0.541] | 0.746 [0.674, 0.803] | 0.853 [0.798, 0.910] | 0.947 [0.918, 0.974] |
| uniform_plain | 0.638 [0.578, 0.680] | 0.496 [0.399, 0.582] | 0.789 [0.764, 0.830] | 0.980 [0.978, 0.983] | 0.488 [0.460, 0.510] | 0.640 [0.616, 0.658] | 0.751 [0.724, 0.777] | 0.898 [0.857, 0.922] |
| uniform_fd | 0.869 [0.869, 0.869] | 0.771 [0.771, 0.771] | 0.606 [0.606, 0.606] | 0.975 [0.975, 0.975] | 0.461 [0.461, 0.461] | 0.814 [0.814, 0.814] | 0.909 [0.909, 0.909] | - |

## Capture only, noisy

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.607 [0.543, 0.688] | 0.475 [0.402, 0.561] | 0.481 [0.459, 0.514] | 0.969 [0.959, 0.984] | 0.388 [0.358, 0.430] | 0.643 [0.552, 0.773] | 0.605 [0.495, 0.767] | 0.597 [0.505, 0.770] |
| uniform_plain | 0.508 [0.440, 0.566] | 0.356 [0.259, 0.468] | 0.648 [0.638, 0.661] | 0.983 [0.979, 0.985] | 0.397 [0.373, 0.418] | 0.532 [0.500, 0.575] | 0.540 [0.517, 0.565] | 0.742 [0.709, 0.800] |
| uniform_fd | 0.416 [0.416, 0.416] | 0.319 [0.319, 0.319] | 0.153 [0.153, 0.153] | 0.957 [0.957, 0.957] | 0.275 [0.275, 0.275] | 0.615 [0.615, 0.615] | 0.467 [0.467, 0.467] | - |

## Capture only, clean_frozen

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.764 [0.651, 0.905] | 0.626 [0.479, 0.777] | 0.742 [0.657, 0.804] | 0.963 [0.937, 0.982] | 0.514 [0.499, 0.533] | 0.741 [0.663, 0.811] | 0.847 [0.786, 0.910] | 0.932 [0.908, 0.960] |

## Capture only, noisy_frozen

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.562 [0.528, 0.630] | 0.443 [0.419, 0.484] | 0.528 [0.494, 0.581] | 0.952 [0.936, 0.968] | 0.388 [0.354, 0.448] | 0.635 [0.567, 0.745] | 0.530 [0.418, 0.680] | 0.625 [0.565, 0.729] |

## Capture only, seed-paired differences

| contrast | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls:clean - uniform_plain:clean | +0.127 [+0.069, +0.222] | +0.127 [+0.078, +0.175] | -0.029 [-0.146, +0.039] | -0.005 [-0.021, +0.005] | +0.035 [+0.008, +0.050] | +0.106 [+0.027, +0.145] | +0.102 [+0.074, +0.132] | +0.049 [+0.027, +0.061] |
| uniform_rls:noisy - uniform_plain:noisy | +0.099 [+0.071, +0.123] | +0.118 [+0.093, +0.144] | -0.167 [-0.202, -0.124] | -0.014 [-0.020, -0.001] | -0.009 [-0.041, +0.029] | +0.110 [-0.023, +0.250] | +0.064 [-0.070, +0.249] | -0.145 [-0.294, +0.052] |
| uniform_fd:clean - uniform_plain:clean | +0.212 [+0.212, +0.212] | +0.265 [+0.265, +0.265] | -0.169 [-0.169, -0.169] | -0.004 [-0.004, -0.004] | -0.033 [-0.033, -0.033] | +0.167 [+0.167, +0.167] | +0.185 [+0.185, +0.185] | - |
| uniform_fd:noisy - uniform_plain:noisy | -0.102 [-0.102, -0.102] | -0.023 [-0.023, -0.023] | -0.485 [-0.485, -0.485] | -0.027 [-0.027, -0.027] | -0.143 [-0.143, -0.143] | +0.040 [+0.040, +0.040] | -0.099 [-0.099, -0.099] | - |
| uniform_rls:clean - uniform_rls:clean_frozen | +0.001 [-0.003, +0.009] | -0.004 [-0.019, +0.010] | +0.019 [+0.009, +0.027] | +0.013 [+0.002, +0.025] | +0.008 [+0.007, +0.011] | +0.005 [-0.008, +0.012] | +0.006 [-0.000, +0.012] | +0.015 [+0.010, +0.022] |
| uniform_rls:noisy - uniform_rls:noisy_frozen | +0.044 [+0.014, +0.060] | +0.031 [-0.017, +0.077] | -0.047 [-0.122, +0.005] | +0.017 [+0.013, +0.023] | +0.000 [-0.018, +0.015] | +0.007 [-0.015, +0.028] | +0.074 [+0.059, +0.087] | -0.028 [-0.078, +0.041] |

