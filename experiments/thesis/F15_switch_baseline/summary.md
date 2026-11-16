# F15: F8 pursuer on learned-policy switches

Descriptive means over three training seeds. Post-switch outcomes condition on episodes alive at the recorded switch; do not interpret them as same-state counterfactuals. Post-switch J is scored at 0.10 m; 0.15 m is a full-episode exact re-score only because it can change which episodes would have survived to switch.
| Arm | Post-switch J | Capture | Full J (0.10 m) | Full J (0.15 m) | Alive at switch | Capture time (s) | Failure-penalized capture time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| f8 | 0.822 | 0.592 | 0.822 | 0.953 | 0.330 | 4.434 | 14.612 |
| rls | 0.874 | 0.710 | 0.876 | 0.980 | 0.309 | 2.210 | 10.031 |
| rls_frozen | 0.864 | 0.700 | 0.867 | 0.979 | 0.309 | 2.206 | 10.270 |
| no_rls | 0.717 | 0.569 | 0.715 | 0.915 | 0.297 | 1.723 | 13.465 |

Fixed-policy controls (mean of seed-balanced F8 and F10 fixed-policy jobs):
| Arm | J (0.10 m) | J (0.15 m) | Capture (0.10 m) |
|---|---:|---:|---:|
| f8 | 0.809 | 0.950 | 0.573 |
| rls | 0.875 | 0.979 | 0.726 |
| rls_frozen | 0.863 | 0.976 | 0.714 |
| no_rls | 0.700 | 0.909 | 0.569 |

Paired per-seed differences versus RLS (control minus RLS), seeds 0/100/200:
| Control | Post-switch J | Capture | Full-episode J |
|---|---|---|---|
| f8 | [-0.0010389188925424842, 0.06655751665433252, -0.22046400606632233] | [-0.08668772379557288, -0.03539333740870165, -0.23124012847741449] | [-0.0014631450176239014, 0.06319168210029602, -0.2232685387134552] |
| rls_frozen | [-0.01067407925923658, -0.004885435104370117, -0.012184977531433105] | [-0.01675589879353845, -0.0007833639780681079, -0.01022118330001831] | [-0.010382086038589478, -0.0052995383739471436, -0.012072712182998657] |
| no_rls | [-0.16435481111208594, -0.17402214805285132, -0.13062407573064172] | [-0.20767682790756226, -0.14504280189673113, -0.0706101506948471] | [-0.1732410490512848, -0.1798458993434906, -0.1308073103427887] |

Full-episode ending rates across randomized-switch conditions (flags can overlap):
| Arm | Capture | Pursuer crash | Evader crash | Evader OOB | Timeout | Pursuer OOB |
|---|---:|---:|---:|---:|---:|---:|
| f8 | 0.591 | 0.164 | 0.067 | 0.165 | 0.022 | 0.166 |
| rls | 0.724 | 0.131 | 0.047 | 0.107 | 0.001 | 0.161 |
| rls_frozen | 0.714 | 0.139 | 0.048 | 0.106 | 0.001 | 0.164 |
| no_rls | 0.582 | 0.296 | 0.047 | 0.086 | 0.000 | 0.241 |

Seed-level switch means:
| Arm | Seed | Post-switch J | Capture | Full J | Alive at switch |
|---|---:|---:|---:|---:|---:|
| f8 | 0 | 0.883 | 0.604 | 0.884 | 0.331 |
| f8 | 100 | 0.898 | 0.636 | 0.900 | 0.329 |
| f8 | 200 | 0.684 | 0.536 | 0.682 | 0.330 |
| rls | 0 | 0.884 | 0.691 | 0.886 | 0.290 |
| rls | 100 | 0.832 | 0.671 | 0.837 | 0.307 |
| rls | 200 | 0.905 | 0.767 | 0.905 | 0.330 |
| rls_frozen | 0 | 0.874 | 0.674 | 0.875 | 0.290 |
| rls_frozen | 100 | 0.827 | 0.670 | 0.831 | 0.307 |
| rls_frozen | 200 | 0.893 | 0.757 | 0.893 | 0.330 |
| no_rls | 0 | 0.720 | 0.483 | 0.713 | 0.282 |
| no_rls | 100 | 0.658 | 0.526 | 0.657 | 0.297 |
| no_rls | 200 | 0.774 | 0.696 | 0.775 | 0.313 |
