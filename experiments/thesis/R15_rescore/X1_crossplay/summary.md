# X1_crossplay summary

14 pursuers x 32 evaders; episodes per cell: min 256, median 1745.

## Transitivity

| subset | additive-logit R² | rank-reversal fraction (delta 0.1) |
|---|---|---|
| all | 0.822 | 0.0279 |
| rl_pursuers_only | 0.848 | 0.0095 |
| capture_all | 0.883 | 0.0050 |

## Pursuers (mean / min neutralisation over evaders)

| pursuer | mean neut | min neut | hardest evader | mean capture | mean pursuer crash |
|---|---|---|---|---|---|
| v9/pursuer:s14 | 0.928 | 0.374 | v9/evader:s19 | 0.899 | 0.073 |
| v9/pursuer:s16 | 0.952 | 0.405 | v9/evader:s17 | 0.919 | 0.049 |
| v9/pursuer:s18 | 0.972 | 0.414 | v9/evader:s19 | 0.932 | 0.029 |
| v9/pursuer:s20 | 0.981 | 0.921 | v9/evader:s11 | 0.943 | 0.019 |
| v10/pursuer:s6 | 0.285 | 0.004 | v10/evader:s7 | 0.003 | 0.001 |
| v10/pursuer:s8 | 0.289 | 0.000 | v9/evader:s13 | 0.039 | 0.001 |
| v10/pursuer:s10 | 0.308 | 0.004 | v9/evader:s11 | 0.064 | 0.000 |
| v10/pursuer:s12 | 0.378 | 0.000 | v9/evader:s13 | 0.173 | 0.000 |
| v11/pursuer:s6 | 0.801 | 0.250 | v9/evader:s13 | 0.786 | 0.199 |
| v11/pursuer:s8 | 0.861 | 0.393 | v11/evader:s9 | 0.836 | 0.140 |
| v11/pursuer:s10 | 0.921 | 0.397 | v9/evader:s13 | 0.893 | 0.081 |
| v11/pursuer:s12 | 0.963 | 0.684 | v9/evader:s17 | 0.923 | 0.038 |
| heur/pursuer:pure_pursuit | 0.309 | 0.004 | v9/evader:s19 | 0.255 | 0.073 |
| heur/pursuer:apf | 0.708 | 0.203 | v9/evader:s15 | 0.621 | 0.250 |

## Evaders (mean neutralisation over pursuers; lower = harder)

| evader | mean neut | max neut | min neut |
|---|---|---|---|
| v9/evader:s19 | 0.368 | 0.984 | 0.004 |
| v9/evader:s17 | 0.396 | 0.973 | 0.000 |
| v9/evader:s15 | 0.439 | 0.985 | 0.000 |
| v9/evader:s13 | 0.464 | 0.983 | 0.000 |
| v11/evader:s11 | 0.506 | 0.986 | 0.016 |
| v9/evader:s11 | 0.525 | 0.969 | 0.004 |
| v11/evader:s9 | 0.547 | 0.999 | 0.016 |
| v9/evader:s9 | 0.592 | 0.990 | 0.004 |
| v10/evader:s7 | 0.617 | 0.997 | 0.004 |
| v11/evader:s7 | 0.639 | 0.993 | 0.016 |
| v9/evader:s7 | 0.640 | 0.998 | 0.055 |
| heur/evader:circular | 0.692 | 0.994 | 0.136 |
| v11/evader:s5 | 0.701 | 1.000 | 0.171 |
| v10/evader:s9 | 0.701 | 0.999 | 0.004 |
| heur/evader:flee>circular@0.5 | 0.704 | 0.988 | 0.187 |
| v9/evader:s5 | 0.718 | 1.000 | 0.058 |
| v9/evader:s3 | 0.730 | 1.000 | 0.039 |
| v11/evader:s3 | 0.736 | 1.000 | 0.047 |
| v10/evader:s11 | 0.755 | 1.000 | 0.012 |
| v10/evader:s5 | 0.765 | 0.999 | 0.115 |
| heur/evader:hover | 0.776 | 0.997 | 0.109 |
| heur/evader:weave | 0.780 | 0.992 | 0.192 |
| heur/evader:flee | 0.782 | 0.994 | 0.228 |
| heur/evader:flee>weave@0.5 | 0.784 | 0.993 | 0.265 |
| heur/evader:hover>flee@0.5 | 0.784 | 0.990 | 0.218 |
| v10/evader:s3 | 0.784 | 1.000 | 0.115 |
| heur/evader:circular>hover@0.5 | 0.785 | 0.990 | 0.227 |
| heur/evader:zigzag | 0.793 | 0.994 | 0.242 |
| heur/evader:fast_flee | 0.793 | 0.994 | 0.286 |
| v10/evader:s1 | 0.923 | 1.000 | 0.205 |
| v11/evader:s1 | 0.923 | 1.000 | 0.225 |
| v9/evader:s1 | 0.924 | 1.000 | 0.229 |

Figures: fig_payoff_neut.png, fig_payoff_capture.png. Raw: payoff.npz, evals/*.json.
