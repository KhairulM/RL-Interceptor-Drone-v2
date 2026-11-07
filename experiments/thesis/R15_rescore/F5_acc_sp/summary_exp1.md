# F5_acc_sp summary (Exp 1 only: Exp 2 / qd_dpp_rls paused 11-05, user)

Final pursuer (stage 8) of each campaign; mean over the campaign seeds (uniform_rls / uniform_plain: seeds 0, 100, 300 after amendment 2; uniform_fd 1) [95 % bootstrap CI over seeds; with 2 values it is their range]. Campaign evaders include the seed-300 campaigns' evaders (amendment 2). *_rls = learned acceleration estimate + RLS (F5); *_plain = plain observation (F2 / F3); uniform_fd = finite-difference acceleration (F2-acc). noisy = A3c sensor noise at eval; *_frozen = F5 pursuer evaluated with RLS frozen. exploitability = 1 - J (or capture) vs a fresh exploiter. J = capture + evader crash + evader out of bounds; cap = capture only. other / own campaign evaders exclude stage-1 evaders (amendment 1). Estimator retrained (amendment 1): `estimator/summary.md`.

## J, clean

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.986 [0.975, 0.997] | 0.937 [0.924, 0.955] | 0.965 [0.948, 0.978] | 0.645 [0.554, 0.766] | 0.809 [0.758, 0.836] | 0.947 [0.933, 0.965] | 0.993 [0.991, 0.994] | 0.995 [0.990, 0.999] |
| uniform_plain | 0.965 [0.944, 0.992] | 0.889 [0.846, 0.914] | 0.958 [0.939, 0.973] | 0.772 [0.749, 0.812] | 0.738 [0.665, 0.783] | 0.908 [0.873, 0.928] | 0.987 [0.984, 0.988] | 0.984 [0.968, 0.996] |
| uniform_fd | 0.996 [0.996, 0.996] | 0.950 [0.950, 0.950] | 0.941 [0.941, 0.941] | 0.327 [0.327, 0.327] | 0.775 [0.775, 0.775] | 0.956 [0.956, 0.956] | 0.996 [0.996, 0.996] | - |

## J, noisy

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.934 [0.906, 0.952] | 0.893 [0.853, 0.923] | 0.894 [0.870, 0.917] | 0.728 [0.684, 0.751] | 0.725 [0.717, 0.740] | 0.939 [0.914, 0.960] | 0.979 [0.963, 0.991] | 0.967 [0.959, 0.974] |
| uniform_plain | 0.883 [0.827, 0.941] | 0.804 [0.719, 0.902] | 0.935 [0.917, 0.949] | 0.824 [0.814, 0.837] | 0.689 [0.623, 0.729] | 0.898 [0.848, 0.927] | 0.976 [0.965, 0.982] | 0.973 [0.951, 0.991] |
| uniform_fd | 0.776 [0.776, 0.776] | 0.717 [0.717, 0.717] | 0.519 [0.519, 0.519] | 0.641 [0.641, 0.641] | 0.555 [0.555, 0.555] | 0.936 [0.936, 0.936] | 0.965 [0.965, 0.965] | - |

## J, clean_frozen

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.985 [0.971, 0.996] | 0.940 [0.922, 0.950] | 0.962 [0.945, 0.976] | 0.580 [0.450, 0.695] | 0.808 [0.758, 0.836] | 0.951 [0.933, 0.967] | 0.995 [0.994, 0.996] | 0.996 [0.991, 0.999] |

## J, noisy_frozen

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.914 [0.884, 0.937] | 0.862 [0.821, 0.894] | 0.912 [0.903, 0.920] | 0.691 [0.667, 0.725] | 0.734 [0.727, 0.741] | 0.932 [0.904, 0.957] | 0.973 [0.952, 0.988] | 0.966 [0.949, 0.990] |

## J, seed-paired differences

| contrast | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls:clean - uniform_plain:clean | +0.021 [+0.005, +0.030] | +0.049 [+0.011, +0.109] | +0.007 [-0.002, +0.015] | -0.127 [-0.258, +0.011] | +0.071 [+0.054, +0.093] | +0.040 [+0.005, +0.092] | +0.006 [+0.003, +0.010] | +0.011 [+0.003, +0.022] |
| uniform_rls:noisy - uniform_plain:noisy | +0.051 [+0.011, +0.116] | +0.089 [+0.021, +0.183] | -0.041 [-0.079, -0.021] | -0.095 [-0.136, -0.064] | +0.036 [-0.012, +0.096] | +0.041 [-0.013, +0.112] | +0.003 [-0.018, +0.026] | -0.007 [-0.018, +0.008] |
| uniform_fd:clean - uniform_plain:clean | +0.037 [+0.037, +0.037] | +0.037 [+0.037, +0.037] | -0.021 [-0.021, -0.021] | -0.428 [-0.428, -0.428] | +0.009 [+0.009, +0.009] | +0.028 [+0.028, +0.028] | +0.008 [+0.008, +0.008] | - |
| uniform_fd:noisy - uniform_plain:noisy | -0.105 [-0.105, -0.105] | -0.074 [-0.074, -0.074] | -0.420 [-0.420, -0.420] | -0.173 [-0.173, -0.173] | -0.161 [-0.161, -0.161] | +0.009 [+0.009, +0.009] | -0.016 [-0.016, -0.016] | - |
| uniform_rls:clean - uniform_rls:clean_frozen | +0.001 [-0.001, +0.003] | -0.002 [-0.014, +0.005] | +0.003 [+0.002, +0.005] | +0.065 [+0.018, +0.104] | +0.000 [-0.004, +0.004] | -0.003 [-0.008, +0.000] | -0.002 [-0.003, -0.001] | -0.000 [-0.001, +0.001] |
| uniform_rls:noisy - uniform_rls:noisy_frozen | +0.020 [+0.006, +0.033] | +0.031 [+0.009, +0.051] | -0.018 [-0.042, -0.003] | +0.037 [+0.017, +0.070] | -0.009 [-0.019, -0.000] | +0.006 [+0.004, +0.010] | +0.006 [+0.003, +0.011] | +0.001 [-0.017, +0.017] |

## Capture only, clean

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.979 [0.966, 0.991] | 0.877 [0.862, 0.891] | 0.961 [0.941, 0.973] | 0.653 [0.561, 0.772] | 0.781 [0.710, 0.827] | 0.902 [0.885, 0.916] | 0.926 [0.920, 0.935] | 0.992 [0.988, 0.995] |
| uniform_plain | 0.956 [0.934, 0.983] | 0.810 [0.774, 0.833] | 0.952 [0.936, 0.964] | 0.775 [0.752, 0.815] | 0.704 [0.637, 0.738] | 0.849 [0.820, 0.866] | 0.910 [0.909, 0.911] | 0.974 [0.955, 0.987] |
| uniform_fd | 0.990 [0.990, 0.990] | 0.898 [0.898, 0.898] | 0.938 [0.938, 0.938] | 0.330 [0.330, 0.330] | 0.765 [0.765, 0.765] | 0.917 [0.917, 0.917] | 0.936 [0.936, 0.936] | - |

## Capture only, noisy

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.923 [0.893, 0.945] | 0.857 [0.827, 0.877] | 0.873 [0.860, 0.893] | 0.743 [0.692, 0.784] | 0.681 [0.653, 0.696] | 0.888 [0.859, 0.906] | 0.872 [0.836, 0.891] | 0.954 [0.945, 0.959] |
| uniform_plain | 0.868 [0.813, 0.924] | 0.767 [0.709, 0.815] | 0.925 [0.912, 0.935] | 0.828 [0.819, 0.841] | 0.654 [0.597, 0.683] | 0.833 [0.791, 0.859] | 0.879 [0.875, 0.884] | 0.960 [0.932, 0.976] |
| uniform_fd | 0.752 [0.752, 0.752] | 0.684 [0.684, 0.684] | 0.456 [0.456, 0.456] | 0.803 [0.803, 0.803] | 0.508 [0.508, 0.508] | 0.865 [0.865, 0.865] | 0.766 [0.766, 0.766] | - |

## Capture only, clean_frozen

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.978 [0.963, 0.990] | 0.882 [0.857, 0.899] | 0.956 [0.937, 0.970] | 0.589 [0.456, 0.701] | 0.779 [0.706, 0.822] | 0.906 [0.885, 0.918] | 0.931 [0.925, 0.939] | 0.993 [0.989, 0.995] |

## Capture only, noisy_frozen

| arm | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls | 0.898 [0.866, 0.920] | 0.843 [0.800, 0.864] | 0.884 [0.865, 0.901] | 0.709 [0.678, 0.731] | 0.686 [0.653, 0.715] | 0.876 [0.842, 0.896] | 0.830 [0.781, 0.859] | 0.948 [0.934, 0.974] |

## Capture only, seed-paired differences

| contrast | heldout_rl | heldout_worst_cluster | other_campaign_evaders | exploitability | train | heldout_scripted | x3_exploiters | own_campaign_evaders |
|---|---|---|---|---|---|---|---|---|
| uniform_rls:clean - uniform_plain:clean | +0.024 [+0.008, +0.033] | +0.067 [+0.029, +0.116] | +0.009 [+0.003, +0.018] | -0.122 [-0.254, +0.013] | +0.077 [+0.068, +0.090] | +0.053 [+0.019, +0.096] | +0.016 [+0.010, +0.026] | +0.019 [+0.007, +0.033] |
| uniform_rls:noisy - uniform_plain:noisy | +0.055 [+0.021, +0.118] | +0.090 [+0.050, +0.168] | -0.052 [-0.075, -0.033] | -0.084 [-0.131, -0.057] | +0.028 [+0.011, +0.056] | +0.054 [-0.000, +0.115] | -0.007 [-0.048, +0.014] | -0.006 [-0.018, +0.013] |
| uniform_fd:clean - uniform_plain:clean | +0.039 [+0.039, +0.039] | +0.065 [+0.065, +0.065] | -0.017 [-0.017, -0.017] | -0.428 [-0.428, -0.428] | +0.027 [+0.027, +0.027] | +0.051 [+0.051, +0.051] | +0.026 [+0.026, +0.026] | - |
| uniform_fd:noisy - uniform_plain:noisy | -0.114 [-0.114, -0.114] | -0.093 [-0.093, -0.093] | -0.471 [-0.471, -0.471] | -0.017 [-0.017, -0.017] | -0.176 [-0.176, -0.176] | +0.006 [+0.006, +0.006] | -0.118 [-0.118, -0.118] | - |
| uniform_rls:clean - uniform_rls:clean_frozen | +0.001 [-0.001, +0.004] | -0.005 [-0.021, +0.005] | +0.004 [+0.003, +0.006] | +0.064 [+0.016, +0.105] | +0.002 [-0.003, +0.005] | -0.004 [-0.010, +0.001] | -0.005 [-0.006, -0.004] | -0.001 [-0.001, -0.000] |
| uniform_rls:noisy - uniform_rls:noisy_frozen | +0.025 [+0.011, +0.037] | +0.014 [+0.004, +0.027] | -0.011 [-0.041, +0.007] | +0.034 [+0.014, +0.064] | -0.005 [-0.018, +0.004] | +0.012 [+0.009, +0.016] | +0.042 [+0.031, +0.055] | +0.006 [-0.017, +0.025] |

