# F16_roncero_population summary

Hedge = F8b (reused training, fresh common evaluation); QD-DPP / PSRO = new campaigns from the same F8b stage-0/1 bootstrap (PSD dropped after its smoke). Three paired seeds, descriptive. G1 = 0.5 strong-bank J + 0.5 cross-campaign J (other seeds, arms / campaigns / stages weighted equally). Mixtures: exact expectation of one pursuer sampled per episode. Evaluated pursuer stochastic; deterministic strong-bank J in the last table. 0.15 m values are re-scores of the same episodes (upper bounds; all policies trained at 0.10 m).

## r0.10/thesis

| arm/seed | final G1 | strong | cross | v10 | F8 anchor | worst cluster | own att. | bank worst | mix G1 | mix own att. | mix bank worst | coverage exploitability |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hedge/0 | 0.814 | 0.799 | 0.829 | 0.907 | 0.831 | 0.848 | 0.361 | 0.251 | 0.666 | 0.580 | 0.580 | 0.290 |
| hedge/100 | 0.898 | 0.852 | 0.943 | 0.927 | 0.904 | 0.888 | 0.485 | 0.485 | 0.735 | 0.596 | 0.427 | 0.471 |
| hedge/200 | 0.832 | 0.747 | 0.917 | 0.889 | 0.945 | 0.848 | 0.895 | 0.784 | 0.691 | 0.560 | 0.375 | 0.173 |
| qd_dpp/0 | 0.869 | 0.857 | 0.881 | 0.942 | 0.802 | 0.916 | 0.156 | 0.156 | 0.694 | 0.146 | 0.146 | 0.758 |
| qd_dpp/100 | 0.773 | 0.691 | 0.856 | 0.786 | 0.834 | 0.630 | 0.486 | 0.288 | 0.685 | 0.286 | 0.286 | 0.561 |
| qd_dpp/200 | 0.861 | 0.819 | 0.904 | 0.925 | 0.872 | 0.851 | 0.326 | 0.326 | 0.662 | 0.245 | 0.245 | 0.473 |
| psro/0 | 0.544 | 0.332 | 0.756 | 0.409 | 0.708 | 0.259 | 0.478 | 0.478 | 0.213 | 0.073 | 0.058 | 0.333 |
| psro/100 | 0.729 | 0.576 | 0.882 | 0.748 | 0.847 | 0.647 | 0.311 | 0.124 | 0.729 | 0.124 | 0.124 | 0.349 |
| psro/200 | 0.579 | 0.331 | 0.826 | 0.425 | 0.735 | 0.255 | 0.362 | 0.259 | 0.650 | 0.233 | 0.093 | 0.569 |

| contrast | metric | s0 | s100 | s200 | mean | decision |
|---|---|---|---|---|---|---|
| qd_dpp-hedge | final_G1 | +0.055 | -0.124 | +0.030 | -0.013 | inconclusive |
| qd_dpp-hedge | final_strong | +0.058 | -0.161 | +0.072 | -0.011 | inconclusive |
| qd_dpp-hedge | final_cross | +0.052 | -0.087 | -0.013 | -0.016 | inconclusive |
| qd_dpp-hedge | mixture_G1 | +0.027 | -0.050 | -0.029 | -0.017 | inconclusive |
| qd_dpp-hedge | mixture_attack_worst | -0.435 | -0.141 | -0.130 | -0.235 | notable decrease |
| qd_dpp-hedge | final_attack_worst | -0.095 | -0.197 | -0.458 | -0.250 | notable decrease |
| psro-hedge | final_G1 | -0.271 | -0.168 | -0.253 | -0.231 | notable decrease |
| psro-hedge | final_strong | -0.467 | -0.276 | -0.416 | -0.386 | notable decrease |
| psro-hedge | final_cross | -0.074 | -0.060 | -0.090 | -0.075 | notable decrease |
| psro-hedge | mixture_G1 | -0.454 | -0.006 | -0.041 | -0.167 | notable decrease |
| psro-hedge | mixture_attack_worst | -0.522 | -0.303 | -0.282 | -0.369 | notable decrease |
| psro-hedge | final_attack_worst | +0.227 | -0.360 | -0.525 | -0.219 | inconclusive |

## r0.10/roncero

| arm/seed | final G1 | strong | cross | v10 | F8 anchor | worst cluster | own att. | bank worst | mix G1 | mix own att. | mix bank worst | coverage exploitability |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hedge/0 | 0.787 | 0.773 | 0.802 | 0.909 | 0.798 | 0.859 | 0.327 | 0.221 | 0.629 | 0.372 | 0.372 | 0.512 |
| hedge/100 | 0.834 | 0.799 | 0.869 | 0.915 | 0.794 | 0.874 | 0.251 | 0.251 | 0.685 | 0.432 | 0.254 | 0.696 |
| hedge/200 | 0.764 | 0.685 | 0.844 | 0.861 | 0.837 | 0.787 | 0.151 | 0.151 | 0.650 | 0.260 | 0.260 | 0.595 |
| qd_dpp/0 | 0.829 | 0.831 | 0.828 | 0.927 | 0.749 | 0.918 | 0.181 | 0.169 | 0.651 | 0.124 | 0.124 | 0.786 |
| qd_dpp/100 | 0.739 | 0.668 | 0.809 | 0.792 | 0.780 | 0.657 | 0.351 | 0.180 | 0.646 | 0.122 | 0.122 | 0.812 |
| qd_dpp/200 | 0.835 | 0.815 | 0.855 | 0.926 | 0.794 | 0.861 | 0.283 | 0.283 | 0.621 | 0.225 | 0.225 | 0.532 |
| psro/0 | 0.451 | 0.272 | 0.630 | 0.404 | 0.531 | 0.264 | 0.205 | 0.205 | 0.177 | 0.066 | 0.046 | 0.713 |
| psro/100 | 0.637 | 0.530 | 0.745 | 0.703 | 0.648 | 0.565 | 0.194 | 0.122 | 0.637 | 0.122 | 0.122 | 0.562 |
| psro/200 | 0.476 | 0.287 | 0.665 | 0.403 | 0.535 | 0.235 | 0.215 | 0.215 | 0.562 | 0.125 | 0.068 | 0.688 |

| contrast | metric | s0 | s100 | s200 | mean | decision |
|---|---|---|---|---|---|---|
| qd_dpp-hedge | final_G1 | +0.042 | -0.095 | +0.071 | +0.006 | inconclusive |
| qd_dpp-hedge | final_strong | +0.058 | -0.130 | +0.131 | +0.020 | inconclusive |
| qd_dpp-hedge | final_cross | +0.026 | -0.060 | +0.011 | -0.008 | inconclusive |
| qd_dpp-hedge | mixture_G1 | +0.022 | -0.040 | -0.029 | -0.016 | inconclusive |
| qd_dpp-hedge | mixture_attack_worst | -0.248 | -0.132 | -0.035 | -0.138 | notable decrease |
| qd_dpp-hedge | final_attack_worst | -0.053 | -0.071 | +0.132 | +0.003 | inconclusive |
| psro-hedge | final_G1 | -0.336 | -0.196 | -0.288 | -0.274 | notable decrease |
| psro-hedge | final_strong | -0.501 | -0.269 | -0.397 | -0.389 | notable decrease |
| psro-hedge | final_cross | -0.172 | -0.124 | -0.179 | -0.158 | notable decrease |
| psro-hedge | mixture_G1 | -0.452 | -0.048 | -0.088 | -0.196 | notable decrease |
| psro-hedge | mixture_attack_worst | -0.325 | -0.132 | -0.192 | -0.217 | notable decrease |
| psro-hedge | final_attack_worst | -0.016 | -0.129 | +0.064 | -0.027 | inconclusive |

## r0.15/thesis

| arm/seed | final G1 | strong | cross | v10 | F8 anchor | worst cluster | own att. | bank worst | mix G1 | mix own att. | mix bank worst | coverage exploitability |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hedge/0 | 0.936 | 0.928 | 0.944 | 0.987 | 0.939 | 0.978 | 0.942 | 0.379 | 0.802 | 0.628 | 0.628 | 0.184 |
| hedge/100 | 0.964 | 0.939 | 0.990 | 0.989 | 0.979 | 0.981 | 0.668 | 0.668 | 0.855 | 0.707 | 0.541 | 0.322 |
| hedge/200 | 0.940 | 0.911 | 0.970 | 0.982 | 0.997 | 0.971 | 0.900 | 0.900 | 0.818 | 0.706 | 0.494 | 0.068 |
| qd_dpp/0 | 0.945 | 0.931 | 0.959 | 0.997 | 0.911 | 0.996 | 0.232 | 0.232 | 0.808 | 0.257 | 0.257 | 0.515 |
| qd_dpp/100 | 0.919 | 0.891 | 0.948 | 0.963 | 0.938 | 0.930 | 0.587 | 0.329 | 0.833 | 0.368 | 0.368 | 0.411 |
| qd_dpp/200 | 0.945 | 0.918 | 0.973 | 0.992 | 0.955 | 0.981 | 0.381 | 0.381 | 0.783 | 0.337 | 0.337 | 0.328 |
| psro/0 | 0.760 | 0.605 | 0.914 | 0.809 | 0.879 | 0.766 | 0.563 | 0.563 | 0.312 | 0.180 | 0.085 | 0.228 |
| psro/100 | 0.913 | 0.849 | 0.976 | 0.945 | 0.965 | 0.901 | 0.957 | 0.301 | 0.913 | 0.916 | 0.301 | 0.189 |
| psro/200 | 0.770 | 0.602 | 0.938 | 0.762 | 0.881 | 0.650 | 0.409 | 0.395 | 0.816 | 0.265 | 0.117 | 0.433 |

| contrast | metric | s0 | s100 | s200 | mean | decision |
|---|---|---|---|---|---|---|
| qd_dpp-hedge | final_G1 | +0.009 | -0.045 | +0.005 | -0.010 | inconclusive |
| qd_dpp-hedge | final_strong | +0.003 | -0.049 | +0.007 | -0.013 | inconclusive |
| qd_dpp-hedge | final_cross | +0.015 | -0.042 | +0.003 | -0.008 | inconclusive |
| qd_dpp-hedge | mixture_G1 | +0.006 | -0.022 | -0.034 | -0.017 | inconclusive |
| qd_dpp-hedge | mixture_attack_worst | -0.371 | -0.174 | -0.157 | -0.234 | notable decrease |
| qd_dpp-hedge | final_attack_worst | -0.147 | -0.340 | -0.519 | -0.335 | notable decrease |
| psro-hedge | final_G1 | -0.176 | -0.052 | -0.171 | -0.133 | notable decrease |
| psro-hedge | final_strong | -0.323 | -0.090 | -0.309 | -0.241 | notable decrease |
| psro-hedge | final_cross | -0.030 | -0.013 | -0.033 | -0.025 | inconclusive |
| psro-hedge | mixture_G1 | -0.489 | +0.058 | -0.001 | -0.144 | inconclusive |
| psro-hedge | mixture_attack_worst | -0.543 | -0.240 | -0.378 | -0.387 | notable decrease |
| psro-hedge | final_attack_worst | +0.184 | -0.367 | -0.505 | -0.230 | inconclusive |

## r0.15/roncero

| arm/seed | final G1 | strong | cross | v10 | F8 anchor | worst cluster | own att. | bank worst | mix G1 | mix own att. | mix bank worst | coverage exploitability |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hedge/0 | 0.922 | 0.914 | 0.930 | 0.993 | 0.919 | 0.991 | 0.931 | 0.352 | 0.778 | 0.422 | 0.422 | 0.417 |
| hedge/100 | 0.941 | 0.912 | 0.969 | 0.991 | 0.950 | 0.988 | 0.368 | 0.368 | 0.824 | 0.555 | 0.354 | 0.507 |
| hedge/200 | 0.911 | 0.882 | 0.940 | 0.981 | 0.959 | 0.967 | 0.178 | 0.178 | 0.792 | 0.486 | 0.471 | 0.317 |
| qd_dpp/0 | 0.925 | 0.924 | 0.926 | 0.999 | 0.887 | 0.997 | 0.272 | 0.264 | 0.778 | 0.240 | 0.240 | 0.572 |
| qd_dpp/100 | 0.895 | 0.860 | 0.929 | 0.964 | 0.920 | 0.938 | 0.457 | 0.225 | 0.806 | 0.194 | 0.194 | 0.610 |
| qd_dpp/200 | 0.927 | 0.902 | 0.953 | 0.986 | 0.928 | 0.971 | 0.338 | 0.338 | 0.757 | 0.319 | 0.319 | 0.359 |
| psro/0 | 0.697 | 0.566 | 0.828 | 0.790 | 0.753 | 0.735 | 0.323 | 0.323 | 0.283 | 0.164 | 0.066 | 0.623 |
| psro/100 | 0.851 | 0.810 | 0.893 | 0.936 | 0.856 | 0.902 | 0.935 | 0.275 | 0.851 | 0.918 | 0.275 | 0.312 |
| psro/200 | 0.712 | 0.581 | 0.842 | 0.751 | 0.756 | 0.664 | 0.267 | 0.267 | 0.744 | 0.163 | 0.089 | 0.528 |

| contrast | metric | s0 | s100 | s200 | mean | decision |
|---|---|---|---|---|---|---|
| qd_dpp-hedge | final_G1 | +0.003 | -0.046 | +0.016 | -0.009 | inconclusive |
| qd_dpp-hedge | final_strong | +0.009 | -0.052 | +0.020 | -0.007 | inconclusive |
| qd_dpp-hedge | final_cross | -0.004 | -0.040 | +0.012 | -0.010 | inconclusive |
| qd_dpp-hedge | mixture_G1 | +0.000 | -0.017 | -0.035 | -0.017 | inconclusive |
| qd_dpp-hedge | mixture_attack_worst | -0.183 | -0.160 | -0.152 | -0.165 | notable decrease |
| qd_dpp-hedge | final_attack_worst | -0.088 | -0.143 | +0.159 | -0.024 | inconclusive |
| psro-hedge | final_G1 | -0.225 | -0.089 | -0.200 | -0.171 | notable decrease |
| psro-hedge | final_strong | -0.348 | -0.102 | -0.301 | -0.251 | notable decrease |
| psro-hedge | final_cross | -0.102 | -0.076 | -0.098 | -0.092 | notable decrease |
| psro-hedge | mixture_G1 | -0.496 | +0.028 | -0.048 | -0.172 | inconclusive |
| psro-hedge | mixture_attack_worst | -0.357 | -0.079 | -0.382 | -0.272 | notable decrease |
| psro-hedge | final_attack_worst | -0.029 | -0.093 | +0.089 | -0.011 | inconclusive |

## Deterministic-actor final pursuer vs the strong bank (secondary)

| arm/seed | 0.10 m | 0.15 m |
|---|---|---|
| hedge/0 | 0.856 | 0.956 |
| hedge/100 | 0.875 | 0.957 |
| hedge/200 | 0.782 | 0.934 |
| qd_dpp/0 | 0.905 | 0.968 |
| qd_dpp/100 | 0.705 | 0.893 |
| qd_dpp/200 | 0.877 | 0.952 |
| psro/0 | 0.358 | 0.647 |
| psro/100 | 0.616 | 0.874 |
| psro/200 | 0.391 | 0.659 |
