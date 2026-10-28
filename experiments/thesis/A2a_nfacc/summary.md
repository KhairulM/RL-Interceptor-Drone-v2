# A2a_nfacc summary

Mean error |y - y_hat| (m/s) of the evader's velocity change over H steps (= H dt x mean acceleration).
val = 20 % train-pool envs not used to train the basis; test = held-out RL, scripted, exploiters.

| H | set | zero | FD_persistence | NF_frozen | NF_RLS_S1 | NF_RLS_S2 |
|---|---|---|---|---|---|---|
| 1 | val | 0.0363 | 0.0045 | 0.0076 | 0.0054 | 0.0053 |
| 1 | test | 0.0371 | 0.0055 | 0.0088 | 0.0062 | 0.0062 |
| 5 | val | 0.1773 | 0.0522 | 0.0545 | 0.0501 | 0.0471 |
| 5 | test | 0.1803 | 0.0643 | 0.0679 | 0.0603 | 0.0561 |
| 10 | val | 0.3401 | 0.1599 | 0.1442 | 0.1377 | 0.1313 |
| 10 | test | 0.3396 | 0.1885 | 0.1744 | 0.1568 | 0.1488 |

Relative error reduction of NF_RLS_S1 vs FD persistence (val): H=1: -0.198, H=5: +0.040, H=10: +0.139

**H* = 10** (pre-registered rule).

Tuned RLS: H=1: {'delta': 10.0, 'forgetting': 0.98}, H=5: {'delta': 0.1, 'forgetting': 0.995}, H=10: {'delta': 0.01, 'forgetting': 0.999}
