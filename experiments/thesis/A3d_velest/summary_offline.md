# A3d_velest offline summary

Mean error |v - v_hat| (m/s) of the evader's current world velocity from a 10-step window of [rel_pos, own_pos, own_vel] (no evader velocity). noisy = sensor-noise model on the inputs (0.02 m, 0.02 m, 0.05 m/s). FD = first difference of the evader position estimate; FD_ema alpha = 0.1 (tuned on noisy val). measured_vel = error of A3c's noisy measured velocity (rel_lin_vel_std 0.08).

| set | inputs | zero | FD_raw | FD_ema | NF_clean | NF_aug | measured_vel |
|---|---|---|---|---|---|---|---|
| val | clean | 0.9006 | 0.0094 | 0.2686 | 0.1052 | 0.1390 | - |
| val | noisy | 0.9006 | 3.1800 | 0.4389 | 0.2808 | 0.2170 | 0.1276 |
| test | clean | 0.8680 | 0.0096 | 0.2635 | 0.1261 | 0.1614 | - |
| test | noisy | 0.8680 | 3.1813 | 0.4345 | 0.2918 | 0.2324 | 0.1276 |

Pre-registered choice: {'basis': 'clean', 'ema_alpha': 0.1, 'ema_grid_val_noisy': {'0.02': 1.0310863256454468, '0.05': 0.6183803677558899, '0.1': 0.43892189860343933, '0.2': 0.5298760533332825, '0.3': 0.7558691501617432, '0.5': 1.3069792985916138, '1.0': 3.1800456047058105}, 'model_path': '/home/user/Projects/RLInterceptorDrone/experiments/thesis/A3d_velest/models/vel_nf_clean.pt'}
