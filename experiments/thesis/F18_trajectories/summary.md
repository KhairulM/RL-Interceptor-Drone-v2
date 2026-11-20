# F18 offline estimator replay

Mean |y - y_hat| (m/s) of the estimator replayed on recorded trajectories of the stage-24 RLS pursuer (live RLS reproduces the pursuer's own estimate; frozen = theta held at theta_bar on the same trajectories). Drift = median ||theta_t - theta_bar||_F / ||theta_bar||_F.

| Recording | Error live | Error frozen | Live vs frozen | Drift 1 s | Drift 2 s | Drift at end |
|---|---:|---:|---:|---:|---:|---:|
| T2_rls_s0_cross | 0.223 | 0.239 | -0.067 | 0.044 | 0.077 | 0.075 |
| T2_rls_s0_fixed_earlier | 0.254 | 0.267 | -0.047 | 0.048 | 0.082 | 0.103 |
| T2_rls_s0_fixed_f8ev | 0.342 | 0.368 | -0.071 | 0.067 | 0.133 | 0.140 |
| T2_rls_s0_strong | 0.260 | 0.272 | -0.042 | 0.044 | 0.087 | 0.104 |
| T2_rls_s100_cross | 0.216 | 0.231 | -0.068 | 0.043 | 0.071 | 0.082 |
| T2_rls_s100_fixed_earlier | 0.254 | 0.260 | -0.022 | 0.043 | 0.079 | 0.107 |
| T2_rls_s100_fixed_f8ev | 0.304 | 0.323 | -0.061 | 0.052 | 0.099 | 0.130 |
| T2_rls_s100_strong | 0.246 | 0.258 | -0.045 | 0.042 | 0.083 | 0.106 |
| T2_rls_s200_cross | 0.188 | 0.206 | -0.087 | 0.041 | 0.060 | 0.073 |
| T2_rls_s200_fixed_earlier | 0.177 | 0.189 | -0.064 | 0.037 | 0.056 | 0.095 |
| T2_rls_s200_fixed_f8ev | 0.209 | 0.222 | -0.062 | 0.048 | 0.073 | 0.120 |
| T2_rls_s200_strong | 0.194 | 0.205 | -0.053 | 0.039 | 0.061 | 0.090 |
