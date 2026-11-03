# A3f_specnorm summary

Mean error (m/s): acc = evader velocity change over 0.2 s (A3a), vel = current evader velocity (A3d); positions-only inputs; noisy = A3a sensor-noise model; val = train-pool evaders (20 % of envs), test = held-out test-set evaders. MLP = unregularised A3a / A3d model; spec / frob = + beta * sum ||W||^2 with the spectral / Frobenius norm, beta* chosen on val in the basis' training condition (clean basis: clean val; aug basis: noisy val).

| target | set | inputs | MLP_clean | spec_clean | frob_clean | MLP_aug | spec_aug | frob_aug |
|---|---|---|---|---|---|---|---|---|
| acc | val | clean | 0.2414 | 0.2428 | 0.2480 | 0.2465 | 0.2476 | 0.2516 |
| acc | val | noisy | 0.2808 | 0.2786 | 0.2784 | 0.2493 | 0.2503 | 0.2538 |
| acc | test | clean | 0.2884 | 0.2884 | 0.2902 | 0.2901 | 0.2892 | 0.2895 |
| acc | test | noisy | 0.3309 | 0.3276 | 0.3244 | 0.2931 | 0.2921 | 0.2922 |
| vel | val | clean | 0.1052 | 0.1117 | 0.1103 | 0.1390 | 0.1391 | 0.1430 |
| vel | val | noisy | 0.2808 | 0.2805 | 0.2772 | 0.2170 | 0.2185 | 0.2202 |
| vel | test | clean | 0.1261 | 0.1286 | 0.1269 | 0.1614 | 0.1619 | 0.1635 |
| vel | test | noisy | 0.2918 | 0.2905 | 0.2881 | 0.2324 | 0.2347 | 0.2347 |

Lipschitz bound prod ||W_l||_2 (normalised inputs):

| target | MLP_clean | spec_clean | frob_clean | MLP_aug | spec_aug | frob_aug |
|---|---|---|---|---|---|---|
| acc | 6.4 | 3.8 | 3.1 | 5.9 | 2.9 | 2.2 |
| vel | 36.5 | 30.1 | 29.1 | 36.7 | 15.5 | 27.9 |

beta* (val grid in choice of pred_err.json): acc/spec_clean 0.0001, acc/frob_clean 0.0001, acc/spec_aug 0.0001, acc/frob_aug 0.0001, vel/spec_clean 0.0001, vel/frob_clean 0.0001, vel/spec_aug 0.001, vel/frob_aug 0.0001

Consistency (re-evaluated MLP - stored A3a / A3d, should be ~0): max |diff| 0.00e+00
