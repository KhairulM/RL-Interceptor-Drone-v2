# F8d_heldout summary

J (pursuer neutralisation rate) of F8 pursuer stages 8 / 16 / 24 (13 x 50M schedule: 250M / 450M / 650M pursuer frames) and the F3 hedge final pursuers (stage 8, 1.5B pursuer frames), clean under the thesis rules. strong_finals = the final evader of 16 campaigns none of them trained with (thesis-F2-plain, thesis-F3-uniform-s1, thesis-F3-qd_dpp-both-s0, thesis-F3-qd_dpp-both-s1, thesis-F3-pfsp-both-s0, thesis-F3-pfsp-both-s1, thesis-F6-hedge_div-s0, thesis-F6-hedge_div-s1, thesis-F6b-hedge_divb-s0, thesis-F6b-hedge_divb-s1, thesis-F7-r15-s0, thesis-F7-floor_r15-s0, thesis-F9-control-s0, thesis-F9-no_timeout-s0, thesis-F9-beta-s0, thesis-F9-endings-s0); vs_f3_hedge_finals / vs_f8_finals exclude the pursuer's own campaign. 2 seeds, descriptive.

## Capture radius 0.10 m

| metric | roncero/0/P8 | roncero/0/P16 | roncero/0/P24 | roncero/1/P8 | roncero/1/P16 | roncero/1/P24 | hedge/0 | hedge/1 |
|---|---|---|---|---|---|---|---|---|
| heldout_rl | 0.889 | 0.960 | 0.948 | 0.925 | 0.968 | 0.941 | 0.791 | 0.882 |
| heldout_worst_cluster | 0.828 | 0.938 | 0.917 | 0.871 | 0.961 | 0.893 | 0.692 | 0.816 |
| x3_exploiters | 0.998 | 0.999 | 0.999 | 0.995 | 0.999 | 0.999 | 0.955 | 0.973 |
| strong_finals_mean | 0.753 | 0.864 | 0.858 | 0.849 | 0.861 | 0.875 | 0.669 | 0.749 |
| strong_finals_min | 0.179 | 0.662 | 0.653 | 0.482 | 0.620 | 0.603 | 0.326 | 0.446 |
| strong_finals_pursuer_crash | 0.100 | 0.139 | 0.152 | 0.078 | 0.147 | 0.142 | 0.357 | 0.275 |
| vs_f3_hedge_finals | 0.899 | 0.845 | 0.867 | 0.947 | 0.866 | 0.870 | 0.740 | 0.846 |
| vs_f8_finals | 0.553 | 0.222 | 0.863 | 0.190 | 0.780 | 0.864 | 0.783 | 0.817 |

Per strong final evader (rows) x pursuer (columns):

| final evader | roncero/0/P8 | roncero/0/P16 | roncero/0/P24 | roncero/1/P8 | roncero/1/P16 | roncero/1/P24 | hedge/0 | hedge/1 |
|---|---|---|---|---|---|---|---|---|
| thesis-F2-plain/evader:s7 | 0.89 | 0.88 | 0.85 | 0.94 | 0.82 | 0.87 | 0.75 | 0.78 |
| thesis-F3-uniform-s1/evader:s7 | 0.90 | 0.85 | 0.87 | 0.91 | 0.88 | 0.85 | 0.87 | 0.92 |
| thesis-F3-qd_dpp-both-s0/evader:s7 | 0.89 | 0.91 | 0.91 | 0.92 | 0.90 | 0.91 | 0.85 | 0.84 |
| thesis-F3-qd_dpp-both-s1/evader:s7 | 0.91 | 0.86 | 0.90 | 0.85 | 0.91 | 0.85 | 0.90 | 0.94 |
| thesis-F3-pfsp-both-s0/evader:s7 | 0.76 | 0.92 | 0.91 | 0.92 | 0.93 | 0.94 | 0.38 | 0.60 |
| thesis-F3-pfsp-both-s1/evader:s7 | 0.60 | 0.88 | 0.83 | 0.91 | 0.86 | 0.86 | 0.44 | 0.52 |
| thesis-F6-hedge_div-s0/evader:s7 | 0.93 | 0.96 | 0.96 | 0.91 | 0.96 | 0.97 | 0.53 | 0.70 |
| thesis-F6-hedge_div-s1/evader:s7 | 0.85 | 0.92 | 0.93 | 0.95 | 0.90 | 0.93 | 0.63 | 0.75 |
| thesis-F6b-hedge_divb-s0/evader:s7 | 0.91 | 0.88 | 0.89 | 0.92 | 0.89 | 0.90 | 0.88 | 0.87 |
| thesis-F6b-hedge_divb-s1/evader:s7 | 0.88 | 0.87 | 0.91 | 0.94 | 0.89 | 0.89 | 0.88 | 0.93 |
| thesis-F7-r15-s0/evader:s7 | 0.50 | 0.85 | 0.81 | 0.81 | 0.90 | 0.90 | 0.57 | 0.68 |
| thesis-F7-floor_r15-s0/evader:s7 | 0.18 | 0.89 | 0.80 | 0.79 | 0.86 | 0.88 | 0.33 | 0.45 |
| thesis-F9-control-s0/evader:s11 | 0.76 | 0.84 | 0.82 | 0.70 | 0.75 | 0.84 | 0.57 | 0.72 |
| thesis-F9-no_timeout-s0/evader:s11 | 0.51 | 0.66 | 0.65 | 0.81 | 0.62 | 0.60 | 0.53 | 0.55 |
| thesis-F9-beta-s0/evader:s11 | 0.63 | 0.81 | 0.83 | 0.84 | 0.74 | 0.84 | 0.68 | 0.81 |
| thesis-F9-endings-s0/evader:s11 | 0.94 | 0.85 | 0.85 | 0.48 | 0.97 | 0.99 | 0.93 | 0.93 |

## Capture radius 0.15 m

| metric | roncero/0/P8 | roncero/0/P16 | roncero/0/P24 | roncero/1/P8 | roncero/1/P16 | roncero/1/P24 | hedge/0 | hedge/1 |
|---|---|---|---|---|---|---|---|---|
| heldout_rl | 0.977 | 0.998 | 0.995 | 0.992 | 0.995 | 0.996 | 0.960 | 0.989 |
| heldout_worst_cluster | 0.949 | 0.997 | 0.993 | 0.982 | 0.994 | 0.992 | 0.874 | 0.921 |
| x3_exploiters | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.987 | 0.992 |
| strong_finals_mean | 0.898 | 0.966 | 0.959 | 0.937 | 0.955 | 0.955 | 0.879 | 0.927 |
| strong_finals_min | 0.471 | 0.915 | 0.898 | 0.604 | 0.833 | 0.820 | 0.671 | 0.795 |
| strong_finals_pursuer_crash | 0.028 | 0.037 | 0.042 | 0.026 | 0.047 | 0.050 | 0.123 | 0.076 |
| vs_f3_hedge_finals | 0.979 | 0.945 | 0.952 | 0.991 | 0.953 | 0.935 | 0.929 | 0.958 |
| vs_f8_finals | 0.582 | 0.231 | 0.953 | 0.469 | 0.981 | 0.976 | 0.945 | 0.959 |

Per strong final evader (rows) x pursuer (columns):

| final evader | roncero/0/P8 | roncero/0/P16 | roncero/0/P24 | roncero/1/P8 | roncero/1/P16 | roncero/1/P24 | hedge/0 | hedge/1 |
|---|---|---|---|---|---|---|---|---|
| thesis-F2-plain/evader:s7 | 0.98 | 0.97 | 0.94 | 0.99 | 0.95 | 0.94 | 0.93 | 0.95 |
| thesis-F3-uniform-s1/evader:s7 | 0.99 | 0.96 | 0.97 | 0.99 | 0.98 | 0.94 | 0.99 | 1.00 |
| thesis-F3-qd_dpp-both-s0/evader:s7 | 0.97 | 0.98 | 0.97 | 0.98 | 0.97 | 0.96 | 0.98 | 0.97 |
| thesis-F3-qd_dpp-both-s1/evader:s7 | 0.99 | 0.94 | 0.96 | 0.96 | 0.97 | 0.92 | 0.99 | 1.00 |
| thesis-F3-pfsp-both-s0/evader:s7 | 0.95 | 0.98 | 0.98 | 0.98 | 0.97 | 0.98 | 0.67 | 0.85 |
| thesis-F3-pfsp-both-s1/evader:s7 | 0.94 | 0.96 | 0.94 | 0.97 | 0.94 | 0.95 | 0.74 | 0.79 |
| thesis-F6-hedge_div-s0/evader:s7 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 0.90 | 0.95 |
| thesis-F6-hedge_div-s1/evader:s7 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.92 | 0.97 |
| thesis-F6b-hedge_divb-s0/evader:s7 | 0.99 | 0.95 | 0.97 | 0.98 | 0.97 | 0.96 | 0.97 | 0.97 |
| thesis-F6b-hedge_divb-s1/evader:s7 | 0.97 | 0.96 | 0.97 | 0.99 | 0.97 | 0.96 | 0.98 | 0.99 |
| thesis-F7-r15-s0/evader:s7 | 0.63 | 0.95 | 0.94 | 0.92 | 0.96 | 0.96 | 0.80 | 0.90 |
| thesis-F7-floor_r15-s0/evader:s7 | 0.47 | 0.98 | 0.93 | 0.94 | 0.96 | 0.97 | 0.72 | 0.84 |
| thesis-F9-control-s0/evader:s11 | 0.90 | 0.97 | 0.95 | 0.83 | 0.94 | 0.96 | 0.82 | 0.87 |
| thesis-F9-no_timeout-s0/evader:s11 | 0.74 | 0.92 | 0.90 | 0.94 | 0.83 | 0.82 | 0.77 | 0.82 |
| thesis-F9-beta-s0/evader:s11 | 0.87 | 0.94 | 0.95 | 0.94 | 0.89 | 0.97 | 0.90 | 0.96 |
| thesis-F9-endings-s0/evader:s11 | 1.00 | 1.00 | 0.99 | 0.60 | 1.00 | 1.00 | 1.00 | 1.00 |

