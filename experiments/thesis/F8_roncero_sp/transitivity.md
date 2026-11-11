# F8 transitivity and evader diversity (post-hoc, 2026-11-10, re-run 2026-11-11 on thesis-F8b-*; no new runs)

X1's measures: additive-logit R^2 (one strength per player) and rank-reversal fraction (delta 0.1).

## 1. Within each campaign (in-campaign evals, training rules, 0.10 m)

| campaign | block | R^2 (J) | reversals (J) | R^2 (capture) | reversals (capture) | pursuers beat all older | evaders beat all older | newest pursuer min J vs older |
|---|---|---|---|---|---|---|---|---|
| thesis-F8b-roncero-s0 | full 12x13 | 0.731 | 0.008 | 0.739 | 0.000 | 0.92 | 0.92 | 0.92 |
| thesis-F8b-roncero-s0 | matched P2-8 x E1-7 4x4 | 0.670 | 0.000 | 0.637 | 0.000 | 1.00 | 0.67 | 0.93 |
| thesis-F8b-roncero-s100 | full 12x13 | 0.673 | 0.003 | 0.718 | 0.002 | 0.92 | 1.00 | 0.91 |
| thesis-F8b-roncero-s100 | matched P2-8 x E1-7 4x4 | 0.606 | 0.000 | 0.603 | 0.000 | 1.00 | 1.00 | 0.95 |
| thesis-F3-hedge-both-s0 | matched P2-8 x E1-7 4x4 | 0.651 | 0.000 | 0.632 | 0.000 | 1.00 | 1.00 | 0.91 |
| thesis-F3-hedge-both-s1 | matched P2-8 x E1-7 4x4 | 0.690 | 0.000 | 0.666 | 0.000 | 1.00 | 1.00 | 0.90 |

## 2. Across campaigns (F3c / F8c evals, thesis rules; probes: thesis-F3-hedge-both-s0, thesis-F3-hedge-both-s1, thesis-F8b-roncero-s0, thesis-F8b-roncero-s100 (final pursuers: F3 stage 8, F8 stage 24); stage-1 evaders excluded; F8 evaders stages 3-25)

| evader set | n | R^2 0.10 m | reversals 0.10 m | R^2 0.15 m | reversals 0.15 m |
|---|---|---|---|---|---|
| F8 evaders | 24 | 0.619 | 0.050 | 0.533 | 0.001 |
| F3 all samplers | 24 | 0.560 | 0.048 | 0.626 | 0.000 |
| all | 48 | 0.362 | 0.064 | 0.517 | 0.002 |

## 3a. Evader Vendi in a shared 4-probe space (4 evaders per row, max 4; F8 as E1-7 and E19-25)

| campaign | Vendi (this space) | F3 official (8 probes) |
|---|---|---|
| thesis-F2-plain | 2.164 | 2.536 |
| thesis-F3-uniform-s1 | 1.968 | 2.376 |
| thesis-F3-hedge-both-s0 | 2.011 | 2.454 |
| thesis-F3-hedge-both-s1 | 2.217 | 2.543 |
| thesis-F3-qd_dpp-both-s0 | 2.003 | 2.403 |
| thesis-F3-qd_dpp-both-s1 | 2.020 | 2.297 |
| thesis-F3-pfsp-both-s0 | 2.616 | 2.948 |
| thesis-F3-pfsp-both-s1 | 2.799 | 3.023 |
| thesis-F8b-roncero-s0 E1-7 | 1.659 | - |
| thesis-F8b-roncero-s0 E19-25 | 1.991 | - |
| thesis-F8b-roncero-s100 E1-7 | 1.567 | - |
| thesis-F8b-roncero-s100 E19-25 | 2.771 | - |

## 3b. Evader Vendi within each campaign (descriptors vs the campaign's own pursuers)

| campaign | descriptor space | evader set | n | Vendi |
|---|---|---|---|---|
| thesis-F8b-roncero-s0 | all evaders vs P2-24 | all evaders vs P2-24 | 13 | 3.514 |
| thesis-F8b-roncero-s0 | all evaders vs P2-24 | first 4 (E1-7) | 4 | 1.912 |
| thesis-F8b-roncero-s0 | all evaders vs P2-24 | last 4 (E19-25) | 4 | 1.707 |
| thesis-F8b-roncero-s0 | E1-7 vs P2-8 | E1-7 vs P2-8 | 4 | 2.462 |
| thesis-F8b-roncero-s100 | all evaders vs P2-24 | all evaders vs P2-24 | 13 | 3.514 |
| thesis-F8b-roncero-s100 | all evaders vs P2-24 | first 4 (E1-7) | 4 | 1.809 |
| thesis-F8b-roncero-s100 | all evaders vs P2-24 | last 4 (E19-25) | 4 | 1.729 |
| thesis-F8b-roncero-s100 | E1-7 vs P2-8 | E1-7 vs P2-8 | 4 | 2.356 |
| thesis-F3-hedge-both-s0 | E1-7 vs P2-8 | E1-7 vs P2-8 | 4 | 2.401 |
| thesis-F3-hedge-both-s1 | E1-7 vs P2-8 | E1-7 vs P2-8 | 4 | 2.384 |

## Within-campaign J matrices, full blocks (rows pursuer stages, columns evader stages)

**thesis-F8b-roncero-s0**

| P \ E | E1 | E3 | E5 | E7 | E9 | E11 | E13 | E15 | E17 | E19 | E21 | E23 | E25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P2 | 0.96 | 0.51 | 0.04 | 0.02 | 0.07 | 0.03 | 0.02 | 0.15 | 0.03 | 0.16 | 0.04 | 0.03 | 0.01 |
| P4 | 0.93 | 0.96 | 0.06 | 0.03 | 0.02 | 0.04 | 0.03 | 0.15 | 0.02 | 0.30 | 0.02 | 0.03 | 0.04 |
| P6 | 0.93 | 0.98 | 0.99 | 0.22 | 0.04 | 0.05 | 0.03 | 0.17 | 0.04 | 0.03 | 0.03 | 0.04 | 0.02 |
| P8 | 0.93 | 0.98 | 0.99 | 0.99 | 0.11 | 0.05 | 0.03 | 0.14 | 0.03 | 0.03 | 0.04 | 0.02 | 0.02 |
| P10 | 0.91 | 0.96 | 0.99 | 0.99 | 0.98 | 0.09 | 0.06 | 0.13 | 0.05 | 0.08 | 0.04 | 0.03 | 0.03 |
| P12 | 0.91 | 0.96 | 0.99 | 0.99 | 0.99 | 0.97 | 0.11 | 0.19 | 0.11 | 0.13 | 0.15 | 0.19 | 0.14 |
| P14 | 0.93 | 0.95 | 0.99 | 0.99 | 0.99 | 0.97 | 0.97 | 0.23 | 0.07 | 0.06 | 0.11 | 0.20 | 0.12 |
| P16 | 0.95 | 0.97 | 1.00 | 0.99 | 0.99 | 0.98 | 0.97 | 0.97 | 0.10 | 0.09 | 0.17 | 0.24 | 0.11 |
| P18 | 0.93 | 0.97 | 1.00 | 0.99 | 1.00 | 0.97 | 0.96 | 0.94 | 0.15 | 0.13 | 0.15 | 0.31 | 0.13 |
| P20 | 0.93 | 0.97 | 0.99 | 0.99 | 0.99 | 0.97 | 0.96 | 0.92 | 0.87 | 0.93 | 0.21 | 0.35 | 0.09 |
| P22 | 0.93 | 0.96 | 1.00 | 0.99 | 0.99 | 0.97 | 0.97 | 0.92 | 0.90 | 0.97 | 0.94 | 0.28 | 0.08 |
| P24 | 0.94 | 0.98 | 1.00 | 0.99 | 0.99 | 0.96 | 0.96 | 0.92 | 0.92 | 0.97 | 0.95 | 0.96 | 0.17 |

**thesis-F8b-roncero-s100**

| P \ E | E1 | E3 | E5 | E7 | E9 | E11 | E13 | E15 | E17 | E19 | E21 | E23 | E25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P2 | 0.97 | 0.28 | 0.05 | 0.05 | 0.03 | 0.05 | 0.07 | 0.04 | 0.04 | 0.14 | 0.07 | 0.07 | 0.14 |
| P4 | 0.98 | 0.97 | 0.07 | 0.06 | 0.04 | 0.03 | 0.04 | 0.09 | 0.05 | 0.09 | 0.10 | 0.07 | 0.16 |
| P6 | 0.95 | 0.98 | 0.99 | 0.08 | 0.06 | 0.05 | 0.04 | 0.26 | 0.04 | 0.12 | 0.07 | 0.06 | 0.14 |
| P8 | 0.95 | 0.97 | 0.99 | 0.99 | 0.10 | 0.06 | 0.03 | 0.16 | 0.06 | 0.11 | 0.07 | 0.06 | 0.12 |
| P10 | 0.95 | 0.97 | 0.99 | 0.99 | 0.99 | 0.12 | 0.07 | 0.07 | 0.05 | 0.08 | 0.07 | 0.06 | 0.12 |
| P12 | 0.95 | 0.97 | 0.99 | 0.99 | 0.98 | 0.88 | 0.19 | 0.05 | 0.07 | 0.11 | 0.08 | 0.06 | 0.12 |
| P14 | 0.96 | 0.97 | 0.99 | 0.99 | 0.98 | 0.98 | 0.98 | 0.08 | 0.09 | 0.13 | 0.11 | 0.06 | 0.14 |
| P16 | 0.92 | 0.98 | 0.99 | 0.99 | 0.98 | 0.99 | 0.96 | 0.13 | 0.12 | 0.16 | 0.12 | 0.10 | 0.16 |
| P18 | 0.94 | 0.98 | 0.99 | 0.99 | 0.98 | 0.98 | 0.97 | 0.96 | 0.98 | 0.24 | 0.21 | 0.15 | 0.27 |
| P20 | 0.92 | 0.98 | 0.98 | 0.99 | 0.98 | 0.98 | 0.96 | 0.95 | 0.97 | 0.92 | 0.19 | 0.11 | 0.21 |
| P22 | 0.93 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 0.98 | 0.98 | 0.99 | 0.96 | 0.97 | 0.21 | 0.25 |
| P24 | 0.91 | 0.97 | 0.99 | 0.99 | 0.99 | 0.99 | 0.96 | 0.98 | 0.98 | 0.97 | 0.97 | 0.94 | 0.26 |

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

