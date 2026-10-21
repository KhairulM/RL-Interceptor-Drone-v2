# ESN adaptation report (smoke)

Capture rate, mean +/- 95% bootstrap CI over seeds (equal weight per evader).

| method | T0 | T1 | T2 | T3 | train s |
|---|---|---|---|---|---|
| M0_ppo | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.001 +/- 0.000 | 150 |
| M1_esn_shared | 0.001 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 135 |
| M2_esn_per_env | 0.001 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 153 |
| M3_esn_shared_adapt | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 142 |

Capture-rate difference vs M0_ppo (mean over seeds, held-out T1+T2+T3):

- M1_esn_shared: -0.000
- M2_esn_per_env: -0.000
- M3_esn_shared_adapt: -0.000

Figures: bars.png, heatmap_capture.png, heatmap_own_error.png
Not yet produced (need logging not implemented): learning curves, adaptation-latency curves, trajectories/GIFs.
