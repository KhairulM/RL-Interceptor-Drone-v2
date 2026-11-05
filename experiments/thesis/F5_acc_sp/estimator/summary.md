# F5 estimator (A2a rules) summary

Mean error |y - y_hat| (m/s) of the evader's velocity change over H steps (= H dt x mean acceleration).
val = 20 % of the recorded training envs, not used to train the basis; test = held-out RL, scripted, exploiters.

| H | set | zero | FD_persistence | NF_frozen | NF_RLS_S1 | NF_RLS_S2 |
|---|---|---|---|---|---|---|
| 1 | val | 0.0384 | 0.0039 | 0.0065 | 0.0045 | 0.0046 |
| 1 | test | 0.0347 | 0.0050 | 0.0087 | 0.0058 | 0.0059 |
| 5 | val | 0.1892 | 0.0454 | 0.0448 | 0.0398 | 0.0384 |
| 5 | test | 0.1684 | 0.0592 | 0.0628 | 0.0562 | 0.0526 |
| 10 | val | 0.3676 | 0.1431 | 0.1195 | 0.1099 | 0.1073 |
| 10 | test | 0.3169 | 0.1741 | 0.1582 | 0.1433 | 0.1373 |

Relative error reduction of NF_RLS_S1 vs FD persistence (val): H=1: -0.174, H=5: +0.122, H=10: +0.232

**H* = 10** (pre-registered rule).

Tuned RLS: H=1: {'delta': 10.0, 'forgetting': 0.98}, H=5: {'delta': 0.1, 'forgetting': 0.995}, H=10: {'delta': 0.01, 'forgetting': 0.999}
