# X1_crossplay summary

14 pursuers x 32 evaders; episodes per cell: min 256, median 1745.

## Transitivity

| subset | additive-logit R² | rank-reversal fraction (delta 0.1) |
|---|---|---|
| all | 0.777 | 0.0448 |
| rl_pursuers_only | 0.835 | 0.0236 |
| capture_all | 0.831 | 0.0138 |

## Pursuers (mean / min neutralisation over evaders)

| pursuer | mean neut | min neut | hardest evader | mean capture | mean pursuer crash |
|---|---|---|---|---|---|
| v9/pursuer:s14 | 0.797 | 0.040 | v9/evader:s15 | 0.749 | 0.206 |
| v9/pursuer:s16 | 0.864 | 0.032 | v9/evader:s17 | 0.806 | 0.140 |
| v9/pursuer:s18 | 0.904 | 0.045 | v9/evader:s19 | 0.836 | 0.100 |
| v9/pursuer:s20 | 0.909 | 0.711 | v11/evader:s7 | 0.837 | 0.097 |
| v10/pursuer:s6 | 0.284 | 0.004 | v10/evader:s7 | 0.002 | 0.001 |
| v10/pursuer:s8 | 0.269 | 0.000 | v9/evader:s13 | 0.016 | 0.002 |
| v10/pursuer:s10 | 0.279 | 0.000 | v10/evader:s7 | 0.032 | 0.000 |
| v10/pursuer:s12 | 0.352 | 0.000 | v9/evader:s13 | 0.138 | 0.000 |
| v11/pursuer:s6 | 0.572 | 0.020 | v9/evader:s13 | 0.546 | 0.429 |
| v11/pursuer:s8 | 0.723 | 0.039 | v11/evader:s9 | 0.680 | 0.281 |
| v11/pursuer:s10 | 0.786 | 0.037 | v11/evader:s11 | 0.736 | 0.218 |
| v11/pursuer:s12 | 0.868 | 0.216 | v9/evader:s19 | 0.801 | 0.137 |
| heur/pursuer:pure_pursuit | 0.273 | 0.004 | v9/evader:s19 | 0.206 | 0.079 |
| heur/pursuer:apf | 0.567 | 0.179 | v9/evader:s15 | 0.420 | 0.330 |

## Evaders (mean neutralisation over pursuers; lower = harder)

| evader | mean neut | max neut | min neut |
|---|---|---|---|
| v9/evader:s19 | 0.151 | 0.906 | 0.004 |
| v9/evader:s17 | 0.220 | 0.891 | 0.000 |
| v9/evader:s15 | 0.285 | 0.921 | 0.000 |
| v9/evader:s13 | 0.317 | 0.888 | 0.000 |
| v11/evader:s11 | 0.342 | 0.880 | 0.012 |
| v9/evader:s11 | 0.360 | 0.855 | 0.004 |
| v11/evader:s9 | 0.392 | 0.959 | 0.012 |
| v9/evader:s9 | 0.425 | 0.907 | 0.004 |
| v11/evader:s7 | 0.493 | 0.885 | 0.012 |
| v10/evader:s7 | 0.495 | 0.941 | 0.000 |
| v10/evader:s9 | 0.548 | 0.950 | 0.004 |
| v9/evader:s7 | 0.558 | 0.950 | 0.055 |
| v10/evader:s11 | 0.623 | 0.982 | 0.012 |
| v10/evader:s5 | 0.631 | 0.939 | 0.107 |
| heur/evader:circular | 0.647 | 0.971 | 0.136 |
| heur/evader:flee>circular@0.5 | 0.648 | 0.957 | 0.159 |
| v11/evader:s5 | 0.668 | 0.985 | 0.136 |
| v9/evader:s5 | 0.692 | 0.993 | 0.043 |
| v9/evader:s3 | 0.700 | 0.993 | 0.023 |
| v11/evader:s3 | 0.709 | 0.995 | 0.019 |
| heur/evader:circular>hover@0.5 | 0.734 | 0.968 | 0.223 |
| heur/evader:hover | 0.744 | 0.980 | 0.109 |
| heur/evader:weave | 0.745 | 0.974 | 0.192 |
| heur/evader:hover>flee@0.5 | 0.747 | 0.969 | 0.218 |
| heur/evader:flee | 0.753 | 0.977 | 0.224 |
| heur/evader:flee>weave@0.5 | 0.755 | 0.974 | 0.265 |
| heur/evader:zigzag | 0.760 | 0.978 | 0.238 |
| v10/evader:s3 | 0.762 | 0.996 | 0.100 |
| heur/evader:fast_flee | 0.762 | 0.977 | 0.284 |
| v11/evader:s1 | 0.882 | 0.999 | 0.066 |
| v10/evader:s1 | 0.883 | 1.000 | 0.050 |
| v9/evader:s1 | 0.884 | 1.000 | 0.062 |

Figures: fig_payoff_neut.png, fig_payoff_capture.png. Raw: payoff.npz, evals/*.json.
