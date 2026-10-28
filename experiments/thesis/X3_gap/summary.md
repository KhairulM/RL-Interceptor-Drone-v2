# X3_gap summary

Generalist mean J on train pool per seed: 0.748, 0.640, 0.562, 0.787, 0.872

| evader | group | J gen s0 | J gen seeds | J specialist | Delta |
|---|---|---|---|---|---|
| v10/evader:s1 | heldout_rl | 0.976 | 0.98, 0.97, 0.96, 0.97, 0.99 | 0.989 | 0.013 |
| v10/evader:s3 | heldout_rl | 0.805 | 0.80, 0.62, 0.51, 0.89, 0.93 | 0.981 | 0.176 |
| v10/evader:s5 | heldout_rl | 0.310 | 0.31, 0.05, 0.07, 0.42, 0.91 | 0.358 | 0.048 |
| v10/evader:s7 | heldout_rl | 0.284 | 0.28, 0.46, 0.21, 0.65, 0.98 | 0.625 | 0.341 |
| v10/evader:s9 | heldout_rl | 0.607 | 0.61, 0.36, 0.24, 0.88, 0.97 | 0.973 | 0.365 |
| v10/evader:s11 | heldout_rl | 0.641 | 0.64, 0.20, 0.21, 0.91, 0.99 | 0.954 | 0.313 |
| heur/evader:zigzag | heldout_scripted | 0.999 | 1.00, 1.00, 1.00, 1.00, 1.00 | 0.998 | -0.000 |
| heur/evader:fast_flee | heldout_scripted | 0.998 | 1.00, 1.00, 1.00, 1.00, 1.00 | 0.999 | 0.001 |
| heur/evader:flee>circular@0.5 | heldout_scripted | 0.997 | 1.00, 0.99, 1.00, 0.99, 0.99 | 0.996 | -0.001 |
| heur/evader:circular>hover@0.5 | heldout_scripted | 0.997 | 1.00, 1.00, 1.00, 0.99, 1.00 | 0.997 | -0.000 |
| heur/evader:hover>flee@0.5 | heldout_scripted | 0.997 | 1.00, 1.00, 1.00, 1.00, 1.00 | 0.998 | 0.001 |
| heur/evader:flee>weave@0.5 | heldout_scripted | 0.998 | 1.00, 1.00, 1.00, 1.00, 1.00 | 0.999 | 0.001 |
| exploit/evader:g0e0 | exploiter | 0.239 | 0.24, 0.17, 0.21, 0.48, 0.86 | 0.998 | 0.759 |
| exploit/evader:g0e1 | exploiter | 0.220 | 0.22, 0.26, 0.38, 0.53, 0.99 | 0.998 | 0.777 |
| exploit/evader:g0e2 | exploiter | 0.427 | 0.43, 0.49, 0.39, 0.82, 1.00 | 0.999 | 0.572 |

Evaders with Delta >= 0.15: 7 (v10/evader:s3, v10/evader:s7, v10/evader:s9, v10/evader:s11, exploit/evader:g0e0, exploit/evader:g0e1, exploit/evader:g0e2)
