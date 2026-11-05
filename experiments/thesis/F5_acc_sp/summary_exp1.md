# F5_acc_sp summary (Exp 1 only: Exp 2 / qd_dpp_rls paused 11-05, user)

Final pursuer (stage 8) of each campaign; mean over the campaign seeds (2; uniform_fd 1) [95 % bootstrap CI; with 2 seeds the CI is the range of the two values]. *_rls = learned acceleration estimate + RLS (F5); *_plain = plain observation (F2 / F3); uniform_fd = finite-difference acceleration (F2-acc). noisy = A3c sensor noise at eval; *_frozen = F5 pursuer evaluated with RLS frozen. exploitability = 1 - J (or capture) vs a fresh exploiter. J = capture + evader crash + evader out of bounds; cap = capture only. other / own campaign evaders exclude stage-1 evaders (amendment 1). Estimator retrained (amendment 1): `estimator/summary.md`.

## J, clean

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.846 [0.812, 0.879] | 0.764 [0.708, 0.820] | 0.879 [0.867, 0.892] | 0.960 [0.951, 0.969] | 0.649 [0.641, 0.656] | 0.869 [0.840, 0.898] | 0.972 [0.964, 0.980] | 0.967 [0.953, 0.980] |
| uniform_plain | 0.736 [0.696, 0.777] | 0.593 [0.540, 0.646] | 0.845 [0.841, 0.849] | 0.973 [0.972, 0.974] | 0.570 [0.540, 0.601] | 0.795 [0.756, 0.834] | 0.951 [0.946, 0.956] | 0.942 [0.918, 0.967] |
| uniform_fd | 0.893 [0.893, 0.893] | 0.827 [0.827, 0.827] | 0.643 [0.643, 0.643] | 0.965 [0.965, 0.965] | 0.495 [0.495, 0.495] | 0.876 [0.876, 0.876] | 0.986 [0.986, 0.986] | - |

## J, noisy

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.731 [0.717, 0.745] | 0.622 [0.600, 0.645] | 0.666 [0.642, 0.689] | 0.942 [0.929, 0.956] | 0.560 [0.557, 0.563] | 0.817 [0.789, 0.846] | 0.917 [0.896, 0.937] | 0.708 [0.657, 0.759] |
| uniform_plain | 0.595 [0.548, 0.641] | 0.403 [0.348, 0.457] | 0.771 [0.763, 0.779] | 0.978 [0.977, 0.978] | 0.521 [0.483, 0.559] | 0.748 [0.686, 0.811] | 0.895 [0.867, 0.924] | 0.872 [0.853, 0.892] |
| uniform_fd | 0.541 [0.541, 0.541] | 0.440 [0.440, 0.440] | 0.285 [0.285, 0.285] | 0.751 [0.751, 0.751] | 0.398 [0.398, 0.398] | 0.805 [0.805, 0.805] | 0.892 [0.892, 0.892] | - |

## J, clean_frozen

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.845 [0.809, 0.881] | 0.760 [0.701, 0.820] | 0.872 [0.854, 0.890] | 0.951 [0.948, 0.955] | 0.649 [0.642, 0.657] | 0.866 [0.836, 0.896] | 0.977 [0.973, 0.982] | 0.962 [0.952, 0.971] |

## J, noisy_frozen

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.709 [0.681, 0.736] | 0.596 [0.553, 0.638] | 0.760 [0.753, 0.766] | 0.916 [0.893, 0.938] | 0.581 [0.570, 0.592] | 0.813 [0.788, 0.839] | 0.909 [0.884, 0.934] | 0.838 [0.830, 0.845] |

## J, seed-paired differences

| contrast | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls:clean - uniform_plain:clean | +0.110 [+0.103, +0.116] | +0.171 [+0.168, +0.174] | +0.034 [+0.026, +0.042] | -0.013 [-0.021, -0.005] | +0.078 [+0.054, +0.102] | +0.074 [+0.005, +0.142] | +0.021 [+0.008, +0.034] | +0.024 [+0.013, +0.035] |
| uniform_rls:noisy - uniform_plain:noisy | +0.137 [+0.075, +0.198] | +0.220 [+0.143, +0.296] | -0.105 [-0.137, -0.074] | -0.036 [-0.050, -0.022] | +0.039 [-0.002, +0.079] | +0.069 [-0.021, +0.159] | +0.021 [-0.028, +0.071] | -0.164 [-0.235, -0.094] |
| uniform_fd:clean - uniform_plain:clean | +0.117 [+0.117, +0.117] | +0.181 [+0.181, +0.181] | -0.206 [-0.206, -0.206] | -0.010 [-0.010, -0.010] | -0.106 [-0.106, -0.106] | +0.042 [+0.042, +0.042] | +0.030 [+0.030, +0.030] | - |
| uniform_fd:noisy - uniform_plain:noisy | -0.100 [-0.100, -0.100] | -0.017 [-0.017, -0.017] | -0.478 [-0.478, -0.478] | -0.226 [-0.226, -0.226] | -0.161 [-0.161, -0.161] | -0.005 [-0.005, -0.005] | -0.032 [-0.032, -0.032] | - |
| uniform_rls:clean - uniform_rls:clean_frozen | +0.001 [-0.001, +0.004] | +0.003 [-0.000, +0.007] | +0.007 [+0.002, +0.013] | +0.009 [+0.003, +0.015] | -0.001 [-0.001, -0.000] | +0.003 [+0.002, +0.004] | -0.005 [-0.009, -0.002] | +0.005 [+0.002, +0.009] |
| uniform_rls:noisy - uniform_rls:noisy_frozen | +0.022 [+0.009, +0.035] | +0.026 [+0.006, +0.047] | -0.094 [-0.111, -0.077] | +0.026 [+0.018, +0.035] | -0.021 [-0.029, -0.012] | +0.004 [+0.002, +0.006] | +0.008 [+0.003, +0.012] | -0.130 [-0.173, -0.086] |

## Capture only, clean

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.697 [0.648, 0.746] | 0.555 [0.477, 0.633] | 0.803 [0.787, 0.818] | 0.982 [0.980, 0.983] | 0.525 [0.510, 0.541] | 0.718 [0.674, 0.761] | 0.825 [0.798, 0.852] | 0.934 [0.918, 0.949] |
| uniform_plain | 0.617 [0.578, 0.657] | 0.452 [0.399, 0.506] | 0.779 [0.775, 0.783] | 0.979 [0.978, 0.980] | 0.477 [0.460, 0.494] | 0.632 [0.616, 0.647] | 0.738 [0.724, 0.753] | 0.889 [0.857, 0.922] |
| uniform_fd | 0.869 [0.869, 0.869] | 0.771 [0.771, 0.771] | 0.590 [0.590, 0.590] | 0.975 [0.975, 0.975] | 0.461 [0.461, 0.461] | 0.814 [0.814, 0.814] | 0.909 [0.909, 0.909] | - |

## Capture only, noisy

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.566 [0.543, 0.589] | 0.432 [0.402, 0.461] | 0.492 [0.474, 0.510] | 0.974 [0.964, 0.984] | 0.368 [0.358, 0.377] | 0.578 [0.552, 0.603] | 0.524 [0.495, 0.552] | 0.511 [0.505, 0.517] |
| uniform_plain | 0.479 [0.440, 0.518] | 0.300 [0.259, 0.342] | 0.647 [0.639, 0.655] | 0.985 [0.984, 0.985] | 0.395 [0.373, 0.418] | 0.537 [0.500, 0.575] | 0.552 [0.539, 0.565] | 0.755 [0.709, 0.800] |
| uniform_fd | 0.416 [0.416, 0.416] | 0.319 [0.319, 0.319] | 0.150 [0.150, 0.150] | 0.957 [0.957, 0.957] | 0.275 [0.275, 0.275] | 0.615 [0.615, 0.615] | 0.467 [0.467, 0.467] | - |

## Capture only, clean_frozen

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.694 [0.651, 0.737] | 0.551 [0.479, 0.623] | 0.788 [0.770, 0.806] | 0.975 [0.969, 0.982] | 0.516 [0.499, 0.533] | 0.706 [0.663, 0.750] | 0.816 [0.786, 0.846] | 0.918 [0.908, 0.927] |

## Capture only, noisy_frozen

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.529 [0.528, 0.529] | 0.423 [0.419, 0.427] | 0.511 [0.510, 0.512] | 0.959 [0.951, 0.968] | 0.358 [0.354, 0.363] | 0.581 [0.567, 0.595] | 0.456 [0.418, 0.493] | 0.574 [0.565, 0.583] |

## Capture only, seed-paired differences

| contrast | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls:clean - uniform_plain:clean | +0.080 [+0.069, +0.090] | +0.102 [+0.078, +0.127] | +0.023 [+0.012, +0.035] | +0.003 [+0.000, +0.005] | +0.048 [+0.047, +0.050] | +0.086 [+0.027, +0.145] | +0.087 [+0.074, +0.099] | +0.044 [+0.027, +0.061] |
| uniform_rls:noisy - uniform_plain:noisy | +0.087 [+0.071, +0.103] | +0.131 [+0.119, +0.144] | -0.155 [-0.181, -0.129] | -0.010 [-0.020, -0.001] | -0.027 [-0.041, -0.014] | +0.040 [-0.023, +0.103] | -0.028 [-0.070, +0.013] | -0.243 [-0.294, -0.192] |
| uniform_fd:clean - uniform_plain:clean | +0.212 [+0.212, +0.212] | +0.265 [+0.265, +0.265] | -0.194 [-0.194, -0.194] | -0.004 [-0.004, -0.004] | -0.033 [-0.033, -0.033] | +0.167 [+0.167, +0.167] | +0.185 [+0.185, +0.185] | - |
| uniform_fd:noisy - uniform_plain:noisy | -0.102 [-0.102, -0.102] | -0.023 [-0.023, -0.023] | -0.489 [-0.489, -0.489] | -0.027 [-0.027, -0.027] | -0.143 [-0.143, -0.143] | +0.040 [+0.040, +0.040] | -0.099 [-0.099, -0.099] | - |
| uniform_rls:clean - uniform_rls:clean_frozen | +0.003 [-0.003, +0.009] | +0.004 [-0.002, +0.010] | +0.014 [+0.012, +0.017] | +0.006 [+0.002, +0.010] | +0.009 [+0.007, +0.011] | +0.012 [+0.011, +0.012] | +0.009 [+0.006, +0.012] | +0.016 [+0.010, +0.022] |
| uniform_rls:noisy - uniform_rls:noisy_frozen | +0.037 [+0.014, +0.060] | +0.008 [-0.017, +0.033] | -0.019 [-0.036, -0.002] | +0.015 [+0.013, +0.017] | +0.009 [+0.004, +0.015] | -0.003 [-0.015, +0.009] | +0.068 [+0.059, +0.077] | -0.063 [-0.078, -0.048] |

