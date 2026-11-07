# F4_estimator_sp summary (v2)

Pursuer observes an *estimated* evader velocity (no true velocity), trained and evaluated with sensor noise (except *_clean). ema = EMA-filtered finite-difference velocity (A3d V_ema); nf = learned velocity + learned acceleration estimate (A3d V_nf). Mean over 3 campaign seeds [95 % bootstrap CI]; paired = same campaign seed. *_no_s1 = post-hoc (11-04): stage-1 evaders excluded (byte-identical across same-seed campaigns).

| metric | ema | nf | nf - ema (paired) |
|---|---|---|---|
| exploitability | 0.430 [0.304, 0.494] | 0.470 [0.398, 0.516] | +0.040 [-0.095, +0.212] |
| heldout_rl | 0.437 [0.386, 0.486] | 0.607 [0.465, 0.843] | +0.170 [+0.028, +0.404] |
| heldout_rl_clean | 0.456 [0.397, 0.516] | 0.621 [0.496, 0.848] | +0.165 [+0.002, +0.394] |
| heldout_scripted | 0.960 [0.910, 0.997] | 0.962 [0.939, 0.992] | +0.002 [-0.018, +0.029] |
| other_campaign_evaders | 0.718 [0.611, 0.783] | 0.875 [0.732, 0.978] | +0.156 [+0.121, +0.218] |
| other_campaign_evaders_no_s1 | 0.628 [0.488, 0.712] | 0.835 [0.645, 0.971] | +0.207 [+0.156, +0.287] |
| own_campaign_evaders | 0.910 [0.838, 0.946] | 0.953 [0.925, 0.987] | +0.043 [-0.019, +0.107] |
| own_campaign_evaders_no_s1 | 0.880 [0.785, 0.928] | 0.938 [0.903, 0.984] | +0.058 [-0.024, +0.143] |
| train | 0.566 [0.430, 0.673] | 0.622 [0.497, 0.802] | +0.057 [-0.106, +0.208] |
| x3_exploiters | 0.713 [0.515, 0.938] | 0.983 [0.958, 0.997] | +0.269 [+0.059, +0.443] |
| x3_exploiters_clean | 0.776 [0.620, 0.966] | 0.988 [0.968, 0.999] | +0.212 [+0.033, +0.348] |
