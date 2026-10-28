# A3a_posonly summary

Mean error |y - y_hat| (m/s) of the evader's velocity change over H steps, estimator inputs WITHOUT evader velocity. FD = second difference of the evader position estimate, held for H steps. noisy = sensor-noise model on all inputs (rel_pos 0.02 m, own_pos 0.02 m, own_vel 0.05 m/s).

| H | set | inputs | zero | FD | NF_clean | NF_aug | FD_ema |
|---|---|---|---|---|---|---|---|
| 5 | val | clean | 0.1773 | 0.0573 | 0.1271 | 0.1300 | - |
| 5 | val | noisy | 0.1773 | 27.6280 | 0.1526 | 0.1316 | 0.8468 |
| 5 | test | clean | 0.1803 | 0.0705 | 0.1526 | 0.1554 | - |
| 5 | test | noisy | 0.1803 | 27.6445 | 0.1801 | 0.1571 | 0.8492 |
| 10 | val | clean | 0.3400 | 0.1674 | 0.2414 | 0.2465 | - |
| 10 | val | noisy | 0.3400 | 55.2576 | 0.2808 | 0.2493 | 1.6961 |
| 10 | test | clean | 0.3395 | 0.1969 | 0.2884 | 0.2901 | - |
| 10 | test | noisy | 0.3395 | 55.2877 | 0.3309 | 0.2931 | 1.6987 |

Pre-registered choice: {'gain_vs_FD_clean_val': {5: -1.2186129382487851, 10: -0.44183271839943816}, 'H_star': 10, 'basis': 'aug', 'model_path': '/home/user/Projects/RLInterceptorDrone/experiments/thesis/A3a_posonly/models/pos_nf_H10_aug.pt', 'fd_ema_alpha': 0.05}
