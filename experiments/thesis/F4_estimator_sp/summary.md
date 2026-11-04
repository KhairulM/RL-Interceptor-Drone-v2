# F4_estimator_sp summary (v2)

Pursuer observes an *estimated* evader velocity (no true velocity), trained and evaluated with sensor noise (except *_clean). ema = EMA-filtered finite-difference velocity (A3d V_ema); nf = learned velocity + learned acceleration estimate (A3d V_nf). Mean over 3 campaign seeds [95 % bootstrap CI]; paired = same campaign seed. *_no_s1 = post-hoc (11-04): stage-1 evaders excluded (byte-identical across same-seed campaigns).

| metric | ema | nf | nf - ema (paired) |
|---|---|---|---|
| exploitability | 0.675 [0.552, 0.786] | 0.872 [0.829, 0.911] | +0.197 [+0.125, +0.277] |
| heldout_rl | 0.272 [0.243, 0.311] | 0.394 [0.302, 0.562] | +0.122 [-0.009, +0.301] |
| heldout_rl_clean | 0.280 [0.249, 0.321] | 0.414 [0.311, 0.572] | +0.134 [-0.010, +0.300] |
| heldout_scripted | 0.913 [0.834, 0.974] | 0.878 [0.825, 0.949] | -0.035 [-0.106, +0.027] |
| other_campaign_evaders | 0.499 [0.415, 0.543] | 0.695 [0.581, 0.830] | +0.196 [+0.132, +0.291] |
| other_campaign_evaders_no_s1 | 0.367 [0.274, 0.417] | 0.618 [0.463, 0.788] | +0.252 [+0.189, +0.371] |
| own_campaign_evaders | 0.744 [0.719, 0.779] | 0.770 [0.699, 0.863] | +0.025 [-0.037, +0.084] |
| own_campaign_evaders_no_s1 | 0.683 [0.633, 0.753] | 0.712 [0.629, 0.839] | +0.029 [-0.035, +0.086] |
| train | 0.396 [0.302, 0.459] | 0.452 [0.364, 0.613] | +0.056 [-0.081, +0.186] |
| x3_exploiters | 0.459 [0.262, 0.703] | 0.900 [0.791, 0.957] | +0.441 [+0.254, +0.541] |
| x3_exploiters_clean | 0.517 [0.321, 0.755] | 0.911 [0.804, 0.965] | +0.394 [+0.210, +0.490] |
