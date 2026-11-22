# F21 RLS self-play ablation

Descriptive, three paired training seeds. No best-arm selection.
Switch endpoints below remain at 0.10 m; only generalization/exploiter endpoints are re-scored.

| Contrast | Metric | s0 | s100 | s200 | Mean | Decision |
|---|---|---|---|---|---|---|
| fast_minus_baseline | g1 | +0.008 | +0.169 | -0.092 | +0.028 | inconclusive |
| fast_minus_baseline | own_j | +0.033 | -0.355 | -0.093 | -0.138 | inconclusive |
| fast_minus_baseline | lure_j | -0.024 | +0.627 | -0.644 | -0.014 | inconclusive |
| fast_minus_baseline | post_switch_capture | +0.009 | +0.022 | -0.037 | -0.002 | inconclusive |
| fast_minus_baseline | post_switch_neutralization | -0.015 | +0.070 | -0.082 | -0.009 | inconclusive |
| fast_minus_baseline | live_minus_frozen_post_switch_capture | -0.008 | -0.004 | -0.019 | -0.010 | inconclusive |
| cap_minus_baseline | g1 | -0.212 | +0.066 | -0.093 | -0.080 | inconclusive |
| cap_minus_baseline | own_j | +0.032 | +0.099 | -0.609 | -0.159 | inconclusive |
| cap_minus_baseline | lure_j | -0.046 | -0.020 | -0.578 | -0.214 | notable_decrease |
| cap_minus_baseline | post_switch_capture | -0.194 | +0.026 | -0.125 | -0.098 | inconclusive |
| cap_minus_baseline | post_switch_neutralization | -0.178 | +0.011 | -0.192 | -0.120 | inconclusive |
| cap_minus_baseline | live_minus_frozen_post_switch_capture | -0.016 | +0.006 | -0.012 | -0.007 | inconclusive |
| fast_cap_minus_fast | g1 | +0.032 | -0.180 | +0.005 | -0.048 | inconclusive |
| fast_cap_minus_fast | own_j | -0.003 | +0.452 | +0.107 | +0.185 | inconclusive |
| fast_cap_minus_fast | lure_j | +0.001 | -0.593 | +0.066 | -0.175 | inconclusive |
| fast_cap_minus_fast | post_switch_capture | -0.010 | -0.020 | -0.080 | -0.037 | inconclusive |
| fast_cap_minus_fast | post_switch_neutralization | +0.022 | -0.076 | -0.075 | -0.043 | inconclusive |
| fast_cap_minus_fast | live_minus_frozen_post_switch_capture | +0.012 | -0.008 | +0.005 | +0.003 | inconclusive |
| fast_cap_minus_cap | g1 | +0.251 | -0.077 | +0.007 | +0.060 | inconclusive |
| fast_cap_minus_cap | own_j | -0.002 | -0.001 | +0.622 | +0.206 | inconclusive |
| fast_cap_minus_cap | lure_j | +0.023 | +0.054 | -0.000 | +0.025 | inconclusive |
| fast_cap_minus_cap | post_switch_capture | +0.193 | -0.023 | +0.007 | +0.059 | inconclusive |
| fast_cap_minus_cap | post_switch_neutralization | +0.184 | -0.017 | +0.035 | +0.067 | inconclusive |
| fast_cap_minus_cap | live_minus_frozen_post_switch_capture | +0.021 | -0.017 | -0.002 | +0.000 | inconclusive |
| fast_cap_gain_minus_fast_cap | g1 | -0.062 | +0.089 | +0.001 | +0.009 | inconclusive |
| fast_cap_gain_minus_fast_cap | own_j | -0.799 | +0.001 | +0.002 | -0.265 | inconclusive |
| fast_cap_gain_minus_fast_cap | lure_j | -0.002 | -0.029 | -0.059 | -0.030 | inconclusive |
| fast_cap_gain_minus_fast_cap | post_switch_capture | -0.147 | -0.050 | -0.034 | -0.077 | notable_decrease |
| fast_cap_gain_minus_fast_cap | post_switch_neutralization | -0.176 | +0.032 | +0.006 | -0.046 | inconclusive |
| fast_cap_gain_minus_fast_cap | live_minus_frozen_post_switch_capture | +0.035 | -0.004 | -0.008 | +0.008 | inconclusive |

Predictions (held): {'P1': True, 'P2': True, 'P3': False, 'P4': False}
