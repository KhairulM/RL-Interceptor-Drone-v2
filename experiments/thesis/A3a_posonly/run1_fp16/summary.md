# A3a_posonly summary

Mean error |y - y_hat| (m/s) of the evader's velocity change over H steps, estimator inputs WITHOUT evader velocity. FD = second difference of the evader position estimate, held for H steps. noisy = sensor-noise model on all inputs (rel_pos 0.02 m, own_pos 0.02 m, own_vel 0.05 m/s).

| H | set | inputs | zero | FD | NF_clean | NF_aug | FD_ema |
|---|---|---|---|---|---|---|---|
| 5 | val | clean | 0.1773 | 0.2589 | 0.1270 | 0.1300 | - |
| 5 | val | noisy | 0.1773 | 27.6290 | 0.1525 | 0.1316 | 0.8469 |
| 5 | test | clean | 0.1803 | 0.2648 | 0.1526 | 0.1554 | - |
| 5 | test | noisy | 0.1803 | 27.6457 | 0.1801 | 0.1571 | 0.8492 |
| 10 | val | clean | 0.3400 | 0.5390 | 0.2414 | 0.2465 | - |
| 10 | val | noisy | 0.3400 | 55.2597 | 0.2808 | 0.2493 | 1.6962 |
| 10 | test | clean | 0.3395 | 0.5596 | 0.2884 | 0.2901 | - |
| 10 | test | noisy | 0.3395 | 55.2901 | 0.3309 | 0.2931 | 1.6988 |

Pre-registered choice: {'gain_vs_FD_clean_val': {5: 0.509271690281687, 10: 0.5521241742425038}, 'H_star': 10, 'basis': 'aug', 'model_path': '/home/user/Projects/RLInterceptorDrone/experiments/thesis/A3a_posonly/models/pos_nf_H10_aug.pt', 'fd_ema_alpha': 0.05}
