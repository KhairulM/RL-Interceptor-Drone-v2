# F3 transitivity per sampler (post-hoc, 2026-11-09)

X1's measures: additive-logit R^2 (one strength per player) and rank-reversal fraction (delta 0.1).

## Within each campaign: pursuer stages 2-8 x RL evader stages 1-7 (in-campaign evals, 0.10 m)

| sampler | campaign | R^2 (J) | reversals (J) | R^2 (capture) | reversals (capture) |
|---|---|---|---|---|---|
| uniform | thesis-F2-plain | 0.731 | 0.000 | 0.683 | 0.139 |
| uniform | thesis-F3-uniform-s1 | 0.725 | 0.000 | 0.718 | 0.000 |
| hedge | thesis-F3-hedge-both-s0 | 0.651 | 0.000 | 0.632 | 0.000 |
| hedge | thesis-F3-hedge-both-s1 | 0.690 | 0.000 | 0.666 | 0.000 |
| qd_dpp | thesis-F3-qd_dpp-both-s0 | 0.671 | 0.000 | 0.644 | 0.083 |
| qd_dpp | thesis-F3-qd_dpp-both-s1 | 0.673 | 0.000 | 0.707 | 0.000 |
| pfsp | thesis-F3-pfsp-both-s0 | 0.388 | 0.278 | 0.420 | 0.278 |
| pfsp | thesis-F3-pfsp-both-s1 | 0.382 | 0.194 | 0.348 | 0.444 |

## Across campaigns: 8 F3 final pursuers x each sampler's RL evaders (stage 1 excluded; F3c evals)

| evader set | n evaders | R^2 0.10 m | reversals 0.10 m | R^2 0.15 m | reversals 0.15 m |
|---|---|---|---|---|---|
| uniform | 6 | 0.921 | 0.007 | 0.839 | 0.000 |
| hedge | 6 | 0.924 | 0.005 | 0.829 | 0.000 |
| qd_dpp | 6 | 0.902 | 0.000 | 0.779 | 0.000 |
| pfsp | 6 | 0.748 | 0.105 | 0.747 | 0.045 |
| all | 24 | 0.835 | 0.032 | 0.765 | 0.010 |

## Within-campaign J matrices (rows pursuer stages 2-8, columns evader stages 1-7)

**thesis-F2-plain**

| P \ E | E1 | E3 | E5 | E7 |
|---|---|---|---|---|
| P2 | 0.99 | 0.04 | 0.01 | 0.02 |
| P4 | 0.99 | 0.98 | 0.04 | 0.02 |
| P6 | 0.98 | 0.99 | 0.95 | 0.07 |
| P8 | 0.95 | 0.99 | 0.96 | 0.79 |

**thesis-F3-uniform-s1**

| P \ E | E1 | E3 | E5 | E7 |
|---|---|---|---|---|
| P2 | 0.99 | 0.06 | 0.10 | 0.03 |
| P4 | 0.99 | 0.99 | 0.14 | 0.02 |
| P6 | 0.99 | 1.00 | 0.95 | 0.12 |
| P8 | 0.97 | 0.99 | 0.99 | 0.92 |

**thesis-F3-hedge-both-s0**

| P \ E | E1 | E3 | E5 | E7 |
|---|---|---|---|---|
| P2 | 0.99 | 0.03 | 0.02 | 0.02 |
| P4 | 0.99 | 0.99 | 0.03 | 0.01 |
| P6 | 0.97 | 0.99 | 0.97 | 0.04 |
| P8 | 0.95 | 0.98 | 0.98 | 0.91 |

**thesis-F3-hedge-both-s1**

| P \ E | E1 | E3 | E5 | E7 |
|---|---|---|---|---|
| P2 | 0.99 | 0.05 | 0.03 | 0.03 |
| P4 | 0.99 | 0.98 | 0.03 | 0.02 |
| P6 | 0.97 | 0.99 | 0.98 | 0.05 |
| P8 | 0.97 | 0.99 | 0.98 | 0.90 |

**thesis-F3-qd_dpp-both-s0**

| P \ E | E1 | E3 | E5 | E7 |
|---|---|---|---|---|
| P2 | 0.99 | 0.04 | 0.01 | 0.02 |
| P4 | 0.99 | 0.99 | 0.03 | 0.02 |
| P6 | 0.98 | 0.99 | 0.97 | 0.05 |
| P8 | 0.95 | 0.99 | 0.99 | 0.92 |

**thesis-F3-qd_dpp-both-s1**

| P \ E | E1 | E3 | E5 | E7 |
|---|---|---|---|---|
| P2 | 0.99 | 0.07 | 0.03 | 0.05 |
| P4 | 0.99 | 0.99 | 0.05 | 0.03 |
| P6 | 0.96 | 0.99 | 0.97 | 0.06 |
| P8 | 0.98 | 0.99 | 0.99 | 0.95 |

**thesis-F3-pfsp-both-s0**

| P \ E | E1 | E3 | E5 | E7 |
|---|---|---|---|---|
| P2 | 0.99 | 0.04 | 0.09 | 0.27 |
| P4 | 0.82 | 0.99 | 0.02 | 0.12 |
| P6 | 0.87 | 0.92 | 0.98 | 0.03 |
| P8 | 0.95 | 0.86 | 0.65 | 0.91 |

**thesis-F3-pfsp-both-s1**

| P \ E | E1 | E3 | E5 | E7 |
|---|---|---|---|---|
| P2 | 0.99 | 0.04 | 0.27 | 0.09 |
| P4 | 0.85 | 0.99 | 0.04 | 0.35 |
| P6 | 0.96 | 0.90 | 0.97 | 0.03 |
| P8 | 0.92 | 0.92 | 0.35 | 0.98 |

