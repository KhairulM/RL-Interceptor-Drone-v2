# F14 stageE: selected-rule estimator comparison

Selected rules: v2d. At 0.10 m; G1 = 0.5 x strong-set J + 0.5 x campaign-balanced cross-seed J; G2 = own-rules exploiter J minus F8 own-rules exploiter J.
| Arm | Seed | Strong J | Cross J | G1 gap vs F8 | G2 gap vs F8 | Own exploitability |
|---|---:|---:|---:|---:|---:|---:|
| rls | 0 | 0.679 | 0.966 | -0.060 | +0.339 | 0.034 |
| rls | 100 | 0.599 | 0.975 | -0.095 | +0.274 | 0.100 |
| rls | 200 | 0.932 | 0.935 | +0.051 | +0.357 | 0.016 |
| no_rls | 0 | 0.482 | 0.830 | -0.226 | -0.440 | 0.813 |
| no_rls | 100 | 0.481 | 0.785 | -0.249 | +0.371 | 0.002 |
| no_rls | 200 | 0.738 | 0.933 | -0.047 | +0.369 | 0.004 |

Mean RLS G1 gap: -0.035; mean RLS G2 gap: +0.323; RLS - no-RLS G1: +0.140.
At 0.15 m, mean RLS G1 gap: -0.019; mean RLS G2 gap: +0.107.
