# X3_gap summary

Generalist mean J on train pool per seed: 0.817, 0.760, 0.685, 0.873, 0.955

| evader | group | J gen s0 | J gen seeds | J specialist | Delta |
|---|---|---|---|---|---|
| v10/evader:s1 | heldout_rl | 0.999 | 1.00, 1.00, 1.00, 1.00, 1.00 | 1.000 | 0.001 |
| v10/evader:s3 | heldout_rl | 0.985 | 0.98, 0.88, 0.71, 1.00, 1.00 | 1.000 | 0.015 |
| v10/evader:s5 | heldout_rl | 0.562 | 0.56, 0.10, 0.10, 0.71, 0.96 | 0.379 | -0.182 |
| v10/evader:s7 | heldout_rl | 0.689 | 0.69, 0.79, 0.56, 0.95, 1.00 | 0.733 | 0.044 |
| v10/evader:s9 | heldout_rl | 0.892 | 0.89, 0.68, 0.61, 0.98, 1.00 | 0.998 | 0.106 |
| v10/evader:s11 | heldout_rl | 0.937 | 0.94, 0.44, 0.42, 1.00, 1.00 | 0.998 | 0.061 |
| heur/evader:zigzag | heldout_scripted | 1.000 | 1.00, 1.00, 1.00, 1.00, 1.00 | 1.000 | 0.000 |
| heur/evader:fast_flee | heldout_scripted | 1.000 | 1.00, 1.00, 1.00, 1.00, 1.00 | 1.000 | -0.000 |
| heur/evader:flee>circular@0.5 | heldout_scripted | 1.000 | 1.00, 1.00, 1.00, 1.00, 1.00 | 1.000 | 0.000 |
| heur/evader:circular>hover@0.5 | heldout_scripted | 0.999 | 1.00, 1.00, 1.00, 1.00, 1.00 | 1.000 | 0.001 |
| heur/evader:hover>flee@0.5 | heldout_scripted | 1.000 | 1.00, 1.00, 1.00, 1.00, 1.00 | 1.000 | -0.000 |
| heur/evader:flee>weave@0.5 | heldout_scripted | 1.000 | 1.00, 1.00, 1.00, 1.00, 1.00 | 1.000 | -0.000 |
| exploit/evader:g0e0 | exploiter | 0.528 | 0.53, 0.50, 0.43, 0.84, 0.98 | 1.000 | 0.472 |
| exploit/evader:g0e1 | exploiter | 0.637 | 0.64, 0.65, 0.83, 0.94, 1.00 | 1.000 | 0.363 |
| exploit/evader:g0e2 | exploiter | 0.975 | 0.97, 0.90, 0.90, 1.00, 1.00 | 1.000 | 0.025 |

Evaders with Delta >= 0.15: 2 (exploit/evader:g0e0, exploit/evader:g0e1)
