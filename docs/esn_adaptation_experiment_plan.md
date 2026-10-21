# Experiment plan: does online adaptation (ESN-OA and alternatives) help the pursuer against unseen evaders?

Executable by an autonomous agent. Every step has a command or file to produce and a machine-checkable
"done" criterion. Human supervision happens at the **review gates** (section 8) via the generated report.

## 1. Question and hypotheses

Setting: 1v1 Crazyflie pursuit-evasion ([PursuitEvasion.yaml](../cfg/task/PursuitEvasion.yaml),
[pursuit_evasion.py](../omni_drones/envs/single/pursuit_evasion.py)). The pursuer is trained with PPO; at test time it
meets evader strategies not seen in training and **cannot update weights by gradient** (except the explicit
upper-bound baseline).

- **H1** An ESN-OA pursuer (reservoir + online RLS predicting the evader's next rel. distance/velocity, appended to obs;
  [esn.py](../omni_drones/utils/esn.py)) has a higher capture rate / lower time-to-capture than a plain MLP PPO
  pursuer on held-out evaders.
- **H2** The gain comes from *online* adaptation, not from the extra input: `readout=per_env` / `shared_adapt` beat the
  frozen `shared` readout, most visibly right after a mid-episode strategy switch.
- **H3** The reservoir matters: it beats an online RLS on raw observation history (same readout, no reservoir).
- **H4** ESN-OA is competitive with other rapid-adaptation modules (section 3) at lower cost (no BPTT, no privileged info).

## 2. Existing assets (verified in repo)

| Need | Where |
|---|---|
| ESN modes `per_env` / `shared` / `shared_adapt`, `esn_pred_err` stat | `task.esn.*`, `BatchedESN` |
| Recurrent PPO (GRU/LSTM) | `algo=ppo_gru`, `algo=ppo_lstm` ([ppo_rnn.py](../omni_drones/learning/ppo/ppo_rnn.py)) |
| RMA-style encoder/adaptation phases | `algo=ppo_adapt` ([ppo_adapt.py](../omni_drones/learning/ppo/ppo_adapt.py)) |
| Frozen opponent pool + heuristics `hover/circular/pursue/flee`, RL evader checkpoints | `task.opponent.policies`, [selfplay/](../scripts/selfplay) |
| Per-opponent eval (capture, TTC, distance, effort, crash/OOB) as JSON | `scripts/selfplay/run_stage.py mode=eval` |
| One Isaac process per run (RTX 5090, 32 GB) | run sequentially, never in parallel |

## 3. Methods under test (pursuer-side adaptation modules)

All are trained with identical PPO hyper-parameters, reward, arena, domain randomisation and frame budget.
Only the adaptation module differs. Priority: **P0** must run, **P1** run if time budget remains, **P2** optional.

| ID | Method | Adaptation mechanism (test time, no gradient unless noted) | Implementation | Pri |
|---|---|---|---|---|
| M0 | PPO-MLP | none (lower bound) | `algo=ppo`, `esn.enabled=false` | P0 |
| M1 | ESN-OA `shared` | frozen readout fitted centrally in training | `esn.readout=shared` | P0 |
| M2 | ESN-OA `per_env` | per-episode RLS readout | `esn.readout=per_env` | P0 |
| M3 | ESN-OA `shared_adapt` | shared init + per-env RLS | `esn.readout=shared_adapt` | P0 |
| M4 | RL² / recurrent PPO (GRU) | hidden state implicitly infers evader | `algo=ppo_gru` | P0 |
| M5 | RLS on raw history (ESN ablation) | same RLS readout over stacked last-K obs, `reservoir_size` replaced by identity features | new `esn.feature=raw_history` option in `BatchedESN` | P0 |
| M6 | History-window encoder (TCN/1D-conv over last K obs) | feed-forward temporal inference of evader behaviour | new encoder in `omni_drones/learning/ppo/` | P1 |
| M7 | RMA-style two-phase | phase 1: policy sees privileged evader descriptor (speed, type one-hot/params); phase 2: adaptation module regresses the latent from the K-step obs/action history | `algo=ppo_adapt`, `phase=encoder -> adaptation`; env must expose the privileged key | P1 |
| M8 | LIAM-style opponent modelling | encoder-decoder over local history, decoder reconstructs evader obs/action; latent conditions the policy | new module, aux loss | P1 |
| M9 | Kalman/EKF evader predictor (non-learned) | constant-velocity/acceleration EKF forecast of the evader appended to obs | new obs feature in env, no learning | P1 |
| M10 | PEARL-style context latent | probabilistic context encoder with variational posterior sampled online | new module | P2 |
| M11 | Bayesian Policy Reuse over specialist policies | belief over known evader types from observed signal, pick the matching specialist | uses existing pool + checkpoints | P2 |
| M12 | Test-time PPO fine-tuning (upper bound, needs gradients) | fine-tune last layers for N steps on the new evader | `run_stage.py mode=train init_policy=...` with a small frame budget | P2 |

Notes: M5 isolates the reservoir (H3); M9 is a cheap, strong "is learning even needed" control; M12 bounds what
adaptation can reach. A policy trained with a different obs layout cannot share a pool, so each method uses its own
checkpoint dir.

## 4. Evader scenarios

**Training pool (all methods identical):** heuristics `hover, circular, pursue, flee` plus RL evaders from an existing
self-play db (e.g. `selfplay_v6_fix_reward_and_bounds` `evader_stage{3,7}.pt`; list actual files in Phase 0).

**Test sets (never seen in training):**

| Set | Contents | Purpose |
|---|---|---|
| T0 in-distribution | training pool | sanity: no method should regress |
| T1 new heuristics | `weave` (sinusoidal lateral), `zigzag` (random heading flips every 0.5-1.5 s), `spiral_climb`, `fast_flee` (speed x1.5) | held-out strategies |
| T2 held-out RL evaders | later-generation checkpoints (e.g. `evader_stage11.pt`) not in the training pool | the realistic case |
| T3 strategy switch | evader changes strategy at t = 2 s inside a 5 s episode (`flee -> circular`, `circular -> hover`, `RL -> weave`) | measures **adaptation latency** |
| T4 dynamics shift | evader mass/thrust/time-constant outside the training randomisation range | robustness |

Required env work (new, small): add the T1 heuristics to `_heuristic_targets`, a `switch` opponent kind
(`metadata: {sequence:[a,b], switch_time_s}`) in the opponent bank, and a `task.heuristic.speed_scale`. Add unit
tests (reset/shape/arena containment) before any training.

## 5. Protocol

- Seeds: 3 training seeds per method (5 for P0 if the budget allows). Eval: >= 4096 episodes per (method, seed, scenario)
  with `randomization.eval`.
- Frame budget: set after Phase 0 throughput measurement; target equal frames for all methods
  (initial guess 100M frames; the self-play default of 500M is too costly for ~12 methods x 3 seeds).
- Capture radius 0.1 m, episode 1500 steps, as in the task config.
- Training and eval outputs go to `experiments/esn_adapt/<method>/seed<k>/` (checkpoint, `_esn.pt` sidecar, `eval_<scenario>.json`,
  training log CSV). Each run writes `DONE` only after the eval JSON is validated.
- Eval-time fairness: frozen weights, identical seeds and spawn states across methods
  (same `seed`, `eval_seed`), no reward shaping differences.

## 6. Metrics

Per (method, seed, scenario): capture rate; time-to-capture (mean, median, censored at timeout); own crash rate; out-of-bounds
forced-error rate; mean distance; motor effort; `esn_pred_err` (ESN methods).
Adaptation-specific (T3): **recovery time** = steps after the switch until the rolling capture-approach rate (d(distance)/dt < 0
over a 0.3 s window) returns to its pre-switch level; **post-switch capture rate**; prediction error vs time since
switch (all methods with a predictor: M1-M3, M5, M8, M9).
Aggregation: mean +/- 95 % bootstrap CI over seeds, paired differences vs M0, Welch/Mann-Whitney p-values noted but
CIs preferred. Add per-env `esn_pred_err` and capture rate time series logging in `eval` mode (new, small).

## 7. Visualisation (outputs for the human supervisor)

Produced by one script `scripts/experiments/esn_adapt/report.py`, which reads all eval JSON/CSV and writes
`experiments/esn_adapt/report/` (`index.html` + `summary.md` + PNGs + `summary.json`):

1. **Bar chart**: capture rate with CI per method x scenario set (T0-T4), grouped; M0 line for reference.
2. **Heatmap**: methods x individual evaders (capture rate), with unseen evaders marked.
3. **Time-to-capture** violin/box plots per method.
4. **Learning curves** (reward, capture rate, `esn_pred_err`) per method with seed bands (from training logs / wandb export).
5. **Adaptation curves (T3)**: mean distance and prediction error vs time with the switch time marked; table of recovery times.
6. **ESN diagnostics**: prediction error vs episode time (per_env vs shared vs frozen), reservoir-size/leak-rate ablation lines.
7. **Trajectory plots**: 3D top-3 / worst-3 episodes for M0 vs best method on the same evader and spawn, coloured by time,
   saved as PNG; plus an MP4/GIF of one episode per scenario (matplotlib, offscreen; no Isaac viewer needed).
8. **Cost table**: parameters, training wall-clock, per-step inference latency, memory (per-env RLS state is
   `num_envs x reservoir^2` floats) so ESN overhead is visible.
9. `summary.md` starts with a one-paragraph auto-generated verdict per hypothesis H1-H4 (supported / not supported /
   inconclusive with numbers) and a "questions for the human" list.

## 8. Execution phases and review gates

State is kept in `experiments/esn_adapt/state.json` (phase, run table with status pending/running/done/failed, retries).
The driver `scripts/experiments/esn_adapt/run_all.py` is idempotent: it skips `DONE` runs, retries a crashed Isaac worker
(2x, as in selfplay), and stops at each gate by writing `GATE_<n>.md` and exiting. The agent continues only if the human
creates `GATE_<n>.APPROVED` (or, in fully unattended mode, if all automatic checks listed pass; else it halts and reports).

| Phase | Work | Done when |
|---|---|---|
| 0 Recon | list available checkpoints, measure frames/s for M0 and M2 (4096 envs), VRAM with `reservoir_size=500` | `phase0.json` with throughput + chosen frame budget |
| 1 Env extensions | T1 heuristics, `switch` opponent, M5/M9 options, eval time-series logging; unit tests | tests pass; 200-step smoke rollouts per scenario without NaN/arena escape |
| 2 Smoke | 2M-frame training + short eval of every method | every method trains, saves a ckpt, eval JSON parses |
| **Gate 1** | report of smoke results (`report.py --smoke`) | auto: all methods finite loss, capture > 0 on hover. Human: scenarios look sensible (GIFs) |
| 3 P0 runs | M0-M5, 3 seeds each, then evals on T0-T4 | all `DONE` |
| **Gate 2** | full report for P0 | auto: M0 capture on T0 >= 0.8 (else the training is broken: stop). Human: confirm/redirect H1-H3 conclusions and which P1/P2 methods to add |
| 4 P1 runs | M6-M9 | all `DONE` |
| 5 Ablations (ESN only) | reservoir_size {100,250,500}, leak_rate {0.1,0.3,0.6}, forgetting_factor {0.95,0.99,0.999}, 1 seed on T1+T3, then 3 seeds for the best | ablation figures |
| 6 P2 runs (if approved) | M10-M12 | all `DONE` |
| **Gate 3** | final report | human sign-off |

Failure handling: if a method's training diverges, re-run once with a different seed, then log it as failed in the report
(never silently drop). If throughput makes the budget infeasible, shrink frames, seeds or the P1/P2 set in that order and note it in
the report. Do not change PPO hyper-parameters per method without recording it in `state.json`.

## 9. Commands (templates, per run)

```bash
# train (pursuer vs the shared training pool)
python scripts/selfplay/run_stage.py mode=train task.role=pursuer algo=<algo> \
  task.esn.enabled=<bool> task.esn.readout=<mode> seed=<k> total_frames=<F> \
  opponent_spec_path=experiments/esn_adapt/pools/train_pool.json \
  stage_ckpt=experiments/esn_adapt/<method>/seed<k>/policy.pt wandb.mode=disabled

# eval on a scenario set (same seed for all methods)
python scripts/selfplay/run_stage.py mode=eval task.role=pursuer algo=<algo> \
  eval_policy=experiments/esn_adapt/<method>/seed<k>/policy.pt \
  opponent_spec_path=experiments/esn_adapt/pools/<T0..T4>.json \
  stage_out=experiments/esn_adapt/<method>/seed<k>/eval_<T>.json

# report
python scripts/experiments/esn_adapt/report.py --root experiments/esn_adapt
```

Verify the exact override names against [run_stage.yaml](../scripts/selfplay/run_stage.yaml) in Phase 0 before relying on them.

## 10. Risks and open points

- ESN changes `obs_dim`; a pool of frozen *pursuers* is not needed here, but never mix ESN/non-ESN checkpoints.
- `shared` readout frozen at deploy may look like "no adaptation"; that is intended (M1 vs M2/M3 isolates it).
- Fixed pursuer speed/thrust limits mean some evaders (`fast_flee`) are unwinnable; report the oracle upper bound
  from M12 and a pure-pursuit baseline ([pure_pursuit_controller.py](../omni_drones/controllers/pure_pursuit_controller.py))
  so "low capture" is interpretable.
- Compute: ~12 methods x 3 seeds is large; the budget decision is made in Phase 0, not assumed.
- Sim-to-real is out of scope; real-flight ESN logs exist in `scripts/deploy/logs_esn_module/` and could be added as an appendix figure.
