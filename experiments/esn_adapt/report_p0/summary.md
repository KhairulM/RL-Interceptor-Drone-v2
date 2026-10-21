# ESN adaptation report (p0)

Capture rate, mean +/- 95% bootstrap CI over seeds (equal weight per evader).

| method | T0 | T1 | T2 | T3 | train s |
|---|---|---|---|---|---|
| M0_ppo | 0.653 +/- 0.006 | 0.880 +/- 0.007 | 0.012 +/- 0.001 | 0.907 +/- 0.005 | 1044 |
| M1_esn_shared | 0.670 +/- 0.015 | 0.873 +/- 0.013 | 0.009 +/- 0.003 | 0.901 +/- 0.012 | 1075 |
| M2_esn_per_env | 0.651 +/- 0.003 | 0.883 +/- 0.003 | 0.015 +/- 0.005 | 0.913 +/- 0.003 | 1605 |
| M3_esn_shared_adapt | 0.663 +/- 0.002 | 0.887 +/- 0.005 | 0.014 +/- 0.005 | 0.918 +/- 0.004 | 1611 |
| M4_gru | 0.572 +/- 0.006 | 0.727 +/- 0.010 | 0.024 +/- 0.010 | 0.744 +/- 0.027 | 1328 |

Capture-rate difference vs M0_ppo (mean over seeds, held-out T1+T2+T3):

- M1_esn_shared: -0.005
- M2_esn_per_env: +0.004
- M3_esn_shared_adapt: +0.007
- M4_gru: -0.102

Figures: bars.png, heatmap_capture.png, heatmap_own_error.png
Not yet produced (need logging not implemented): learning curves, adaptation-latency curves, trajectories/GIFs.
