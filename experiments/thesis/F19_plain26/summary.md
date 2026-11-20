# F19 plain-26 vs RLS-26 and F8 (0.10 m)

| Contrast | Metric | Seed 0 | Seed 100 | Seed 200 | Mean | Verdict |
|---|---|---:|---:|---:|---:|---|
| plain_minus_rls | strong_j | +0.164 | +0.149 | -0.037 | +0.092 | inconclusive |
| plain_minus_rls | cross_j | -0.035 | -0.007 | +0.022 | -0.007 | inconclusive |
| plain_minus_rls | g1 | +0.065 | +0.071 | -0.008 | +0.043 | inconclusive |
| plain_minus_rls | own_exploitability | +0.467 | +0.040 | +0.457 | +0.321 | notable |
| plain_minus_rls | post_switch_capture | -0.247 | -0.071 | -0.173 | -0.163 | notable |
| plain_minus_rls | post_switch_j | -0.017 | -0.018 | -0.122 | -0.052 | notable |
| plain_minus_rls | fp_time_s | +8.261 | +2.451 | +3.303 | +4.672 | notable |
| plain_minus_f8 | g1 | +0.017 | -0.048 | +0.055 | +0.008 | inconclusive |
| plain_minus_f8 | strong_j | -0.014 | -0.127 | +0.109 | -0.011 | inconclusive |
| plain_minus_f8 | own_exploitability | +0.163 | -0.567 | +0.398 | -0.002 | inconclusive |
| plain_minus_f8 | lure_j | +0.338 | -0.110 | +0.395 | +0.208 | inconclusive |
| plain_minus_f8 | post_switch_capture | -0.160 | -0.035 | +0.059 | -0.046 | inconclusive |
| plain_minus_f8 | post_switch_j | -0.016 | -0.085 | +0.098 | -0.001 | inconclusive |
| plain_minus_f8 | fp_time_s | +3.548 | -0.694 | -2.582 | +0.091 | inconclusive |

| Seed | Strong J | Cross J | G1 | Own expl. | Lure J | Lure crash | Post-switch capture | Post-switch J | FP time (s) | Fixed capture |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.843 | 0.931 | 0.887 | 0.501 | 0.707 | 0.309 | 0.444 | 0.868 | 18.06 | 0.425 |
| 100 | 0.748 | 0.968 | 0.858 | 0.140 | 0.376 | 0.636 | 0.600 | 0.813 | 12.97 | 0.614 |
| 200 | 0.895 | 0.957 | 0.926 | 0.473 | 0.819 | 0.075 | 0.594 | 0.783 | 13.07 | 0.567 |
