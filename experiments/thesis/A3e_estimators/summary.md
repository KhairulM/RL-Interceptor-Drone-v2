# A3e_estimators summary

Mean error (m/s) from inputs without evader velocity; noisy = sensor-noise model (0.02 m, 0.02 m, 0.05 m/s). MLP = 10-step window (A3a / A3d), FD / FD_ema / MLP reused from A3a / A3d. KF q* (tuned on noisy val): vel 10, acc 1.

## Evader velocity (A3d target)

| set | inputs | zero | FD | FD_ema | KF | MLP_clean | MLP_aug | GRU_clean | GRU_aug | LSTM_clean | LSTM_aug |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val | clean | 0.9006 | 0.0094 | 0.2686 | 0.1176 | 0.1052 | 0.1390 | 0.0513 | 0.1200 | 0.0675 | 0.1333 |
| val | noisy | 0.9006 | 3.1800 | 0.4389 | 0.2811 | 0.2808 | 0.2170 | 0.7798 | 0.1698 | 0.5534 | 0.1784 |
| test | clean | 0.8680 | 0.0096 | 0.2635 | 0.1233 | 0.1261 | 0.1614 | 0.0602 | 0.1390 | 0.0777 | 0.1492 |
| test | noisy | 0.8680 | 3.1813 | 0.4345 | 0.2880 | 0.2918 | 0.2324 | 0.8000 | 0.1861 | 0.5676 | 0.1929 |

## Evader velocity change over 10 steps (A3a target)

| set | inputs | zero | FD | FD_ema | KF | MLP_clean | MLP_aug | GRU_clean | GRU_aug | LSTM_clean | LSTM_aug |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val | clean | 0.3400 | 0.1674 | - | 0.3477 | 0.2414 | 0.2465 | 0.2152 | 0.2267 | 0.2400 | 0.2435 |
| val | noisy | 0.3400 | 55.2576 | 1.6961 | 0.3965 | 0.2808 | 0.2493 | 0.3860 | 0.2297 | 0.2594 | 0.2448 |
| test | clean | 0.3395 | 0.1969 | - | 0.3379 | 0.2884 | 0.2901 | 0.2444 | 0.2554 | 0.2819 | 0.2866 |
| test | noisy | 0.3395 | 55.2877 | 1.6987 | 0.3852 | 0.3309 | 0.2931 | 0.4194 | 0.2589 | 0.3016 | 0.2881 |
