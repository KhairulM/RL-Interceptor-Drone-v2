# F14 stageC: selected-rule estimator comparison

Selected rules: v2d. At 0.10 m; G1 = 0.5 x strong-set J + 0.5 x campaign-balanced cross-seed J; G2 = own-rules exploiter J minus F8 own-rules exploiter J.
| Arm | Seed | Strong J | Cross J | G1 gap vs F8 | G2 gap vs F8 | Own exploitability |
|---|---:|---:|---:|---:|---:|---:|
| plain | 0 | 0.873 | 0.960 | +0.034 | +0.099 | 0.275 |
| plain | 100 | 0.679 | 0.912 | -0.087 | +0.357 | 0.016 |
| plain | 200 | 0.904 | 0.953 | +0.046 | +0.291 | 0.082 |
| rls | 0 | 0.646 | 0.959 | -0.080 | +0.371 | 0.002 |
| rls | 100 | 0.607 | 0.965 | -0.096 | +0.371 | 0.002 |
| rls | 200 | 0.873 | 0.945 | +0.027 | +0.350 | 0.023 |
| no_rls | 0 | 0.586 | 0.819 | -0.180 | -0.353 | 0.726 |
| no_rls | 100 | 0.679 | 0.858 | -0.114 | +0.372 | 0.001 |
| no_rls | 200 | 0.673 | 0.860 | -0.116 | -0.294 | 0.667 |

Mean RLS G1 gap: -0.050; mean RLS G2 gap: +0.364; RLS - no-RLS G1: +0.087.
At 0.15 m, mean RLS G1 gap: -0.023; mean RLS G2 gap: +0.120.
26-stage continuation required: True (trigger: either mean gap is non-positive).
