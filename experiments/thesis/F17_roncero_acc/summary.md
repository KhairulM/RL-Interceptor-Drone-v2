# F17: F8 rules + learned acceleration estimator/RLS on the pursuer

Paired per-seed F17 - F8 (stage-24 pursuers, 0.10 m). notable if mean paired F17-F8 >= 0.05 and all three seeds share its sign.

| Metric | s0 | s100 | s200 | Mean | Verdict |
|---|---:|---:|---:|---:|---|
| g1 (0.10 m) | -0.013 | -0.013 | +0.055 | +0.010 | inconclusive |
| g2_own_j (0.10 m) | -0.428 | -0.132 | -0.528 | -0.363 | notable_degradation |
| strong_j (0.10 m) | -0.015 | +0.016 | +0.117 | +0.039 | inconclusive |
| cross_j (0.10 m) | -0.011 | -0.041 | -0.007 | -0.020 | inconclusive |
| thesis_lure_j (0.10 m) | -0.319 | -0.105 | +0.568 | +0.048 | inconclusive |
| g1 (0.15 m) | +0.002 | -0.019 | +0.018 | +0.000 | inconclusive |
| g2_own_j (0.15 m) | -0.475 | -0.539 | -0.452 | -0.489 | notable_degradation |
| live - frozen G1 (0.10 m) | -0.006 | +0.013 | +0.012 | +0.006 | inconclusive |
| switch post_switch_neutralization | -0.056 | +0.037 | +0.173 | +0.051 | inconclusive |
| switch post_switch_capture | +0.051 | +0.016 | +0.227 | +0.098 | notable_improvement |
| switch full_episode_j | -0.053 | +0.036 | +0.178 | +0.053 | inconclusive |
| switch fixed_j_r010 | -0.015 | +0.032 | +0.180 | +0.066 | inconclusive |
| switch live_minus_frozen_post_switch_j | -0.009 | +0.016 | +0.004 | +0.004 | inconclusive |

| Seed | F17 G1 | F8 G1 | F17 own J | F8 own J | F17 thesis-lure J | F8 thesis-lure J |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.857 | 0.870 | 0.233 | 0.662 | 0.050 | 0.369 |
| 100 | 0.894 | 0.906 | 0.162 | 0.294 | 0.382 | 0.487 |
| 200 | 0.926 | 0.871 | 0.397 | 0.925 | 0.992 | 0.424 |

Caveat: cross-campaign J uses each family's own other-seed evaders (F14 style), so the F17 and F8 cross components face different evader banks. The estimator was fitted on F5 thesis-rule data.
