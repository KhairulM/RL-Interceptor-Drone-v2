# F8 transitivity and evader diversity (post-hoc, 2026-11-10; no new runs)

X1's measures: additive-logit R^2 (one strength per player) and rank-reversal fraction (delta 0.1).

## 1. Within each campaign (in-campaign evals, training rules, 0.10 m)

| campaign | block | R^2 (J) | reversals (J) | R^2 (capture) | reversals (capture) | pursuers beat all older | evaders beat all older | newest pursuer min J vs older |
|---|---|---|---|---|---|---|---|---|
| thesis-F8-roncero-s0 | full 12x13 | 0.721 | 0.020 | 0.760 | 0.000 | 1.00 | 0.83 | 0.95 |
| thesis-F8-roncero-s0 | matched P2-8 x E1-7 4x4 | 0.627 | 0.028 | 0.521 | 0.028 | 1.00 | 1.00 | 0.96 |
| thesis-F8-roncero-s100 | full 12x13 | 0.717 | 0.003 | 0.718 | 0.002 | 1.00 | 0.83 | 0.93 |
| thesis-F8-roncero-s100 | matched P2-8 x E1-7 4x4 | 0.559 | 0.111 | 0.713 | 0.000 | 1.00 | 0.33 | 0.95 |
| thesis-F3-hedge-both-s0 | matched P2-8 x E1-7 4x4 | 0.651 | 0.000 | 0.632 | 0.000 | 1.00 | 1.00 | 0.91 |
| thesis-F3-hedge-both-s1 | matched P2-8 x E1-7 4x4 | 0.690 | 0.000 | 0.666 | 0.000 | 1.00 | 1.00 | 0.90 |

## 2. Across campaigns (F3c / F8c evals, thesis rules; probes: thesis-F3-hedge-both-s0, thesis-F3-hedge-both-s1, thesis-F8-roncero-s0, thesis-F8-roncero-s100 (F3 final pursuers; F8 pursuer stage 8, which F8c evaluated instead of stage 24); stage-1 evaders excluded; F8 evaders are stages 3-7 only)

| evader set | n | R^2 0.10 m | reversals 0.10 m | R^2 0.15 m | reversals 0.15 m |
|---|---|---|---|---|---|
| F8 evaders | 6 | 0.850 | 0.000 | 0.787 | 0.000 |
| F3 all samplers | 24 | 0.717 | 0.055 | 0.695 | 0.000 |
| all | 30 | 0.512 | 0.050 | 0.644 | 0.000 |

## 3a. Evader Vendi in a shared 4-probe space (stages 1-7, 4 evaders per campaign, max 4)

| campaign | Vendi (this space) | F3 official (8 probes) |
|---|---|---|
| thesis-F2-plain | 2.350 | 2.536 |
| thesis-F3-uniform-s1 | 2.072 | 2.376 |
| thesis-F3-hedge-both-s0 | 1.991 | 2.454 |
| thesis-F3-hedge-both-s1 | 2.126 | 2.543 |
| thesis-F3-qd_dpp-both-s0 | 2.127 | 2.403 |
| thesis-F3-qd_dpp-both-s1 | 1.962 | 2.297 |
| thesis-F3-pfsp-both-s0 | 2.867 | 2.948 |
| thesis-F3-pfsp-both-s1 | 2.891 | 3.023 |
| thesis-F8-roncero-s0 | 1.536 | - |
| thesis-F8-roncero-s100 | 2.044 | - |

## 3b. Evader Vendi within each campaign (descriptors vs the campaign's own pursuers)

| campaign | descriptor space | evader set | n | Vendi |
|---|---|---|---|---|
| thesis-F8-roncero-s0 | all evaders vs P2-24 | all evaders vs P2-24 | 13 | 3.845 |
| thesis-F8-roncero-s0 | all evaders vs P2-24 | first 4 (E1-7) | 4 | 2.054 |
| thesis-F8-roncero-s0 | all evaders vs P2-24 | last 4 (E19-25) | 4 | 2.237 |
| thesis-F8-roncero-s0 | E1-7 vs P2-8 | E1-7 vs P2-8 | 4 | 2.511 |
| thesis-F8-roncero-s100 | all evaders vs P2-24 | all evaders vs P2-24 | 13 | 3.330 |
| thesis-F8-roncero-s100 | all evaders vs P2-24 | first 4 (E1-7) | 4 | 1.497 |
| thesis-F8-roncero-s100 | all evaders vs P2-24 | last 4 (E19-25) | 4 | 2.093 |
| thesis-F8-roncero-s100 | E1-7 vs P2-8 | E1-7 vs P2-8 | 4 | 2.329 |
| thesis-F3-hedge-both-s0 | E1-7 vs P2-8 | E1-7 vs P2-8 | 4 | 2.401 |
| thesis-F3-hedge-both-s1 | E1-7 vs P2-8 | E1-7 vs P2-8 | 4 | 2.384 |

## Within-campaign J matrices, full blocks (rows pursuer stages, columns evader stages)

**thesis-F8-roncero-s0**

| P \ E | E1 | E3 | E5 | E7 | E9 | E11 | E13 | E15 | E17 | E19 | E21 | E23 | E25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P2 | 0.97 | 0.13 | 0.11 | 0.35 | 0.13 | 0.10 | 0.11 | 0.08 | 0.11 | 0.09 | 0.10 | 0.13 | 0.14 |
| P4 | 0.96 | 0.99 | 0.21 | 0.14 | 0.14 | 0.10 | 0.11 | 0.07 | 0.11 | 0.07 | 0.14 | 0.18 | 0.17 |
| P6 | 0.98 | 0.99 | 0.98 | 0.42 | 0.20 | 0.35 | 0.20 | 0.17 | 0.22 | 0.18 | 0.22 | 0.21 | 0.15 |
| P8 | 0.96 | 0.99 | 0.98 | 0.98 | 0.51 | 0.16 | 0.43 | 0.45 | 0.76 | 0.48 | 0.43 | 0.46 | 0.18 |
| P10 | 0.96 | 0.98 | 0.98 | 0.96 | 0.96 | 0.43 | 0.23 | 0.32 | 0.28 | 0.28 | 0.28 | 0.32 | 0.16 |
| P12 | 0.96 | 0.98 | 0.98 | 0.97 | 0.96 | 0.94 | 0.31 | 0.22 | 0.19 | 0.32 | 0.23 | 0.26 | 0.24 |
| P14 | 0.93 | 0.98 | 0.97 | 0.97 | 0.95 | 0.96 | 0.97 | 0.32 | 0.31 | 0.40 | 0.46 | 0.39 | 0.24 |
| P16 | 0.94 | 0.99 | 0.98 | 0.97 | 0.97 | 0.97 | 0.98 | 0.98 | 0.27 | 0.19 | 0.32 | 0.25 | 0.16 |
| P18 | 0.94 | 0.98 | 0.97 | 0.97 | 0.96 | 0.97 | 0.97 | 0.96 | 0.95 | 0.31 | 0.36 | 0.25 | 0.17 |
| P20 | 0.95 | 0.98 | 0.98 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.96 | 0.97 | 0.45 | 0.28 | 0.14 |
| P22 | 0.95 | 0.99 | 0.98 | 0.96 | 0.97 | 0.97 | 0.96 | 0.96 | 0.95 | 0.93 | 0.90 | 0.25 | 0.14 |
| P24 | 0.96 | 0.99 | 0.98 | 0.96 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.95 | 0.95 | 0.97 | 0.29 |

**thesis-F8-roncero-s100**

| P \ E | E1 | E3 | E5 | E7 | E9 | E11 | E13 | E15 | E17 | E19 | E21 | E23 | E25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P2 | 0.97 | 0.15 | 0.57 | 0.59 | 0.10 | 0.04 | 0.02 | 0.04 | 0.02 | 0.01 | 0.04 | 0.05 | 0.05 |
| P4 | 0.96 | 0.99 | 0.42 | 0.32 | 0.09 | 0.06 | 0.02 | 0.04 | 0.05 | 0.06 | 0.12 | 0.06 | 0.06 |
| P6 | 0.96 | 0.99 | 0.93 | 0.24 | 0.12 | 0.06 | 0.02 | 0.03 | 0.04 | 0.03 | 0.14 | 0.07 | 0.08 |
| P8 | 0.97 | 0.98 | 0.95 | 0.97 | 0.27 | 0.13 | 0.07 | 0.06 | 0.06 | 0.01 | 0.04 | 0.06 | 0.06 |
| P10 | 0.95 | 0.99 | 0.96 | 0.97 | 0.96 | 0.15 | 0.06 | 0.07 | 0.08 | 0.05 | 0.12 | 0.08 | 0.15 |
| P12 | 0.96 | 0.99 | 0.97 | 0.96 | 0.96 | 0.99 | 0.12 | 0.07 | 0.11 | 0.06 | 0.07 | 0.11 | 0.12 |
| P14 | 0.96 | 0.99 | 0.95 | 0.97 | 0.96 | 0.97 | 0.98 | 0.26 | 0.19 | 0.31 | 0.15 | 0.24 | 0.17 |
| P16 | 0.90 | 0.99 | 0.96 | 0.96 | 0.96 | 0.98 | 0.98 | 0.97 | 0.29 | 0.07 | 0.05 | 0.11 | 0.10 |
| P18 | 0.95 | 0.99 | 0.97 | 0.98 | 0.97 | 0.99 | 0.99 | 0.97 | 0.97 | 0.13 | 0.07 | 0.22 | 0.16 |
| P20 | 0.94 | 0.99 | 0.97 | 0.98 | 0.98 | 0.98 | 0.98 | 0.96 | 0.95 | 0.99 | 0.14 | 0.30 | 0.24 |
| P22 | 0.92 | 0.99 | 0.96 | 0.97 | 0.97 | 0.97 | 0.98 | 0.95 | 0.94 | 0.99 | 0.95 | 0.23 | 0.30 |
| P24 | 0.93 | 0.99 | 0.97 | 0.97 | 0.97 | 0.97 | 0.99 | 0.96 | 0.95 | 0.99 | 0.97 | 0.97 | 0.38 |

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

