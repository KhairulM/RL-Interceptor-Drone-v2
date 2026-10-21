# Experiment plan v2: online adaptation of a pursuer to unseen evader strategies

Supersedes the method list of [esn_adaptation_experiment_plan.md](./esn_adaptation_experiment_plan.md) (v1). v1's
infrastructure ([run_all.py](../scripts/experiments/esn_adapt/run_all.py), [report.py](../scripts/experiments/esn_adapt/report.py),
`experiments/esn_adapt/`) is reused. v2 is driven by what the P0 results show.

## 0. What P0 already tells us (5 methods x 3 seeds, 100M frames, `experiments/esn_adapt/p0`)

| Finding | Evidence | Consequence |
|---|---|---|
| **No method effect** | All ESN variants are within +/-0.01 of M0 on every set | Nothing has been shown yet, neither for nor against the hypothesis |
| **Ceiling on "unseen" heuristics** | T1 (weave/zigzag/fast_flee) and T3 (switches) are captured at 0.87-0.95, the same as the training heuristics | These evaders are not novel in any way that matters to the pursuer: a non-adaptive policy already wins, so adaptation cannot help |
| **RL evaders look broken / the metric is wrong** | vs `stage3/7/9/11` capture is 0.01-0.21, but 78-95 % of episodes end with **evader crash/OOB** within ~1.5 s. In the v6 self-play db the v6 pursuers captured the same `evader_s3/s7` at 0.92-0.95 | Either the replayed RL evaders are mis-wired in the current env (obs/action layout drift since v6), or the pursuer learned to force evader errors. In both cases "capture rate" on T2 does not measure adaptation |
| **ESN feature is near-redundant** | Target = current `rel_dist`/`rel_vel` ([pursuit_evasion.py](../omni_drones/envs/single/pursuit_evasion.py), `_agent_observation`), which is already in the obs; a 1-step (10 ms) forecast is ~= the input | The policy has no reason to use it. Predict something it *cannot* see: future evader state at 0.3-1 s, or the evader's latent mode |
| **Episodes are very short** | Mean time to capture ~1.2-1.8 s | Within-episode adaptation has ~150 steps to work in. Real deployment = the *same* evader met over many encounters, so **across-episode** adaptation is the more realistic and more promising setting |
| **GRU is under-trained, not "worse"** | M4 is behind on in-distribution evaders too (0.57 vs 0.65 on T0) | Same frame budget is unfair to recurrent PPO; tune it before drawing conclusions |

**Bottom line:** the bottleneck is not the number of methods. It is the absence of a measurable **adaptation gap**.
Fix the testbed first (Phases A-B), then add methods (Phase D).

## 1. Hypotheses (reformulated around the gap)

Let `J(pi, E)` = neutralization rate of pursuer `pi` vs evader `E`. Let `pi_G` = non-adaptive generalist (M0) and
`pi*_E` = a specialist fine-tuned on `E` (oracle).
Gap `Delta_E = J(pi*_E, E) - J(pi_G, E)`, **gap closed** `GC_A(E) = (J(pi_A, E) - J(pi_G, E)) / Delta_E`.

- **H0 (premise, the user's hypothesis)**: there exist realistic held-out evaders with `Delta_E >= 0.25`.
  If H0 fails in sim, that is itself the result (the generalist is robust) and the rest is moot.
- **H1**: adaptive methods reach `GC > 0` on held-out evaders, with no loss on in-distribution evaders.
- **H2 (it is *adaptation*)**: the benefit grows with experience (time-in-episode, or episode index in the
  across-episode setting) and disappears when the adaptation state is frozen/reset (ablation).
- **H3 (mechanism)**: ranking of families (in-context inference vs explicit opponent model vs policy reuse vs
  test-time gradient RL) on GC, sample efficiency (episodes to adapt), and onboard cost.

## 2. Phase A: validity fixes (do before any new training)

| # | Task | Done when |
|---|---|---|
| A1 | **RL-evader replay check**: evaluate v6 `pursuer_stage10.pt` vs v6 `evader_stage{3,7,11}.pt` in the current env | Outcomes reproduce `selfplay_v6.../db/matchups.json` within ~0.05. If not, diff the obs layout (`rot_repr`, prev action, time encoding, ESN dims) and action transform against the v6 config and fix the opponent loader |
| A2 | **Metric**: add `neutralized = capture OR evader_crash OR evader_oob` (an interceptor that forces a crash has succeeded) and report capture separately. Add per-step logging of distance + outcome time in eval | `report.py` shows both, and T2 is interpretable |
| A3 | **ESN target audit**: compare `esn_pred_err` with a constant-velocity extrapolation baseline at the same horizon | Number logged. If ESN <= CV baseline, its feature carries no information |
| A4 | **Fair recurrent baseline**: GRU with tuned BPTT length / lr and 2x frames, separate from the equal-frame comparison | M4 matches M0 on T0 (otherwise exclude it and say so) |
| A5 | **Noise on**: run evals with `task.observation.include_noise=true` (+ optional action latency) since the claim is about real deployment | Evals run under the real-world noise model |

## 3. Phase B: build an evader suite with a real gap

The training pool must contain evaders whose **best response differs** (otherwise a policy learns to ignore any
context, which is likely why ESN is unused). The test set must contain evaders that beat `pi_G`.

**Evader families** (heuristic ones are new branches in `_heuristic_targets`, parametrised so held-out *parameters*
are possible):

| Family | Behaviour | Hidden params (train range / held-out range) |
|---|---|---|
| F1 flee-type (exists) | flee / weave / zigzag / fast_flee | speed, lateral freq |
| F2 **juke** (reactive) | holds or drifts, then dodges laterally/vertically when the pursuer is within `r_trigger` and closing faster than `v_trigger` | `r_trigger`, dodge direction bias, dodge accel |
| F3 **predictive dodge** | aims away from the pursuer's *extrapolated* intercept point (exploits pure-pursuit / PN-like policies) | lookahead, gain |
| F4 **terrain/arena use** | corner/wall hugging, low-altitude skimming (pursuer risks a crash), ceiling | preferred region, margin |
| F5 **dynamics shift** | evader with higher `max_tilt_deg` / thrust or lower mass than in training | agility scale |
| F6 **RL exploiters** | PPO evaders trained as **best responses to frozen `pi_G`** (`run_stage.py mode=train task.role=evader`, pool = {pi_G}), several seeds and reward variants | a gap by construction; the most realistic "new strategy" |
| F7 (real world only) | **human-piloted** evader Crazyflie / joystick | the true out-of-distribution test |

**Splits**
- `Train`: F1 + F2/F3 with parameters inside the train range + 2 RL exploiters (seed a, b).
- `Test-interp`: F2/F3 held-out parameters inside the range.
- `Test-extrap`: F2/F3 parameters outside the range, F5.
- `Test-family`: F4 + RL exploiters of held-out seeds/reward variants (never seen).
- `Switch`: F1 -> F2/F3 at t = 1.0 s (shorter than v1's 2 s: most episodes end before 2 s).

**Gap measurement**: for each test evader, fine-tune `pi_G` for N frames (e.g. 20M) -> specialist `pi*_E`. Keep
evaders with `J(pi_G) <= 0.6` and `J(pi*) >= 0.85`. Specialists are reused as the M12 oracle and by M11 (policy reuse).

**Gate B**: at least 6 test evaders across at least 2 families with `Delta >= 0.25`. If you cannot find them,
stop and report H0 as unsupported in sim (and try F7 in the real world before concluding).

**Important**: retrain the generalist `pi_G` (M0) on the *new* `Train` pool before measuring gaps. Then train F6
exploiters against that retrained policy.

## 4. Phase C: upper-bound diagnostics (cheap, decisive)

Before building complex adaptive modules, check whether the *information* they would infer is useful at all.

| ID | Oracle | Tells you |
|---|---|---|
| O1 | Privileged evader params / family one-hot in the obs (train + test) | Max achievable by *context inference* methods (M4-M10). If O1 does not close the gap, no inference module will |
| O2 | True future evader position at +0.3 / +0.6 / +1.0 s in the obs | Max achievable by *predictive-feature* methods (ESN, Kalman, LIAM decoder) and which horizon matters. This sets the ESN target horizon |
| O3 | Specialist `pi*_E` (from Phase B) | Max achievable by *test-time RL* |

**Gate C**: pick the method families whose oracle closes at least 50 % of the gap. Deprioritise the rest.

## 5. Phase D: methods (taxonomy, each answers a different question)

Tested in three deployment **settings**:
- **S1 within-episode**: adaptation state is reset at each episode (v1 setting).
- **S2 across-episode**: K = 30 consecutive episodes vs the same evader without a state reset. Curve = J vs episode index.
  This needs a flag, e.g. `task.esn.reset_on_episode=false`, and a GRU hidden state carried across resets.
- **S3 switch**: mid-episode strategy change, with recovery time as the metric.

| ID | Family | Method | Adapts via | Pri |
|---|---|---|---|---|
| M0 | none | PPO-MLP (generalist, new pool) | - | P0 |
| M0h | none | PPO-MLP + **frame stack of the same history length** the adaptive methods see | - (control: "is it just more input?") | P0 |
| B1 | classical | PN / pure pursuit / kinematic MPC ([controllers/](../omni_drones/controllers)) | - (reference for interpretability) | P0 |
| M1-3 | explicit model | **ESN-OA v2**: target = evader state at the O2-best horizon (or evader accel); optionally feed `W_out` low-rank projection as an opponent embedding; `shared_adapt` + `per_env` | online RLS | P0 |
| M5 | explicit model | RLS on raw history (no reservoir) | online RLS | P0 (isolates the reservoir) |
| M9 | explicit model | **IMM Kalman filter** (CV / CA / coordinated-turn models); appends forecast + mode probabilities | Bayesian filter | P0 (strong non-learned control) |
| M4 | in-context | Recurrent PPO / RL^2 (GRU, tuned in A4; S2 carries hidden state across episodes) | hidden state | P0 |
| M6 | in-context | Transformer/TCN history encoder over last K steps | attention | P1 |
| M7 | in-context | **RMA**: teacher with O1 privileged params -> student regresses the latent from history (`algo=ppo_adapt`) | latent estimate | P1 (if Gate C favours O1) |
| M8 | explicit model | LIAM-style VAE opponent model (reconstruct evader obs/actions) | latent | P1 |
| M11 | policy reuse | **Bayesian Policy Reuse** over Phase-B specialists + `pi_G` (belief from the observed evader trajectory likelihood) | belief update | P1 (cheap, specialists already exist) |
| M12 | test-time RL | **Online PPO fine-tuning** of the last layers / LoRA with a KL anchor to `pi_G`, using the capture/crash reward | gradients | P1 (this is "online RL" literally) |
| M13 | test-time RL | Meta-learned init (Reptile over train evaders) + M12 | gradients | P2 |
| M10 | in-context | PEARL context posterior | latent | P2 |

Note on terms: M1-M11 adapt **without** weight updates (in-context / filtering), and M12-M13 do **online RL**.
State in the paper which one you claim. ESN-OA is online *supervised* learning of a predictor, not online RL.

For M12/M13 the x-axis is **real-world budget**: J after 1 / 5 / 20 / 100 episodes vs the new evader, plus the
own-crash rate during fine-tuning (safety). This is what a real deployment can afford.

## 6. Metrics and statistics

- Primary: neutralization rate, **gap closed GC**, in-distribution regression.
- Secondary: capture rate, time to neutralize (censored), own crash/OOB, motor effort, min distance.
- Adaptation: J vs time-in-episode (S1), J vs episode index (S2, area under the curve), recovery time (S3),
  prediction error vs horizon (predictive methods vs O2).
- Cost: params, onboard/offboard inference latency at the control rate, memory (per-env RLS = `reservoir^2`).
- Stats: 5 seeds for P0 methods (P0 runs took ~17-27 min each, so 5 seeds are affordable), >= 2048 episodes per
  (method, seed, evader), paired eval seeds. Report IQM + stratified bootstrap CI (`rliable`), and per-evader heatmaps.

## 7. Phase E: ablations and real-world pilot

- ESN: horizon {1, 30, 60, 100 steps}, reservoir {100, 250, 500}, leak {0.1, 0.3, 0.6}, forgetting {0.95, 0.99, 0.999}
  on `Test-*` + `Switch`, 1 seed each, then 3 seeds for the best.
- History-length ablation for M0h/M4/M6 (is it memory length or architecture?).
- Real world (Crazyflie, [deploy/](../scripts/deploy)): M0 vs the best 2 methods vs B1. Evaders: 2 scripted (one F2/F3 from
  the gap set, one held-out) on a second Crazyflie + F7 human pilot. 20 encounters per pair in S2 order, with
  capture/forced-error/crash logged from mocap. This directly tests the motivating claim.

## 8. Order, compute and gates

| Phase | Est. GPU time | Gate |
|---|---|---|
| A validity | ~0.5 day (evals + 1 GRU retune) | A1 reproduces v6; T2 interpretable |
| B gap suite | ~1-1.5 days (new M0, ~8 exploiters, ~10 specialists at ~20-25 min each) | **Gate B**: at least 6 evaders with Delta >= 0.25 |
| C oracles | ~0.5 day (O1, O2 x3 horizons, 3 seeds) | **Gate C**: which families to pursue |
| D P0 methods x 5 seeds x S1/S2/S3 | ~1.5 days | review before P1 |
| D P1, E ablations | ~2 days | - |
| E real world | lab time | - |

Run Isaac jobs sequentially (one GPU). Extend [run_all.py](../scripts/experiments/esn_adapt/run_all.py) with new
`METHODS`/`POOLS`, an `--setting {S1,S2,S3}` eval flag, and a `--phase {A,B,C,D}`.
Write results to `experiments/adapt_v2/`.

## 9. Code changes required (small to medium)

1. `pursuit_evasion.py`: F2-F5 heuristics with params in `metadata`, neutralization outcome, per-step eval
   time series, `esn.horizon` (delayed-target RLS: fit at t on the target observed at t, using features from
   t - H), oracle obs keys for O1/O2 (eval + train), optional no-reset of adaptation state (S2).
2. `esn.py`: delayed-target buffer for horizon H, optional `W_out` embedding output.
3. `run_stage.py`: `mode=finetune` with a small budget + KL anchor (M12), and repeated-episode eval (S2).
4. New modules: IMM filter feature (M9), history encoder (M6), BPR selector (M11).
5. `report.py`: GC, S2 curves, S3 recovery, oracle comparison, rliable aggregates.

Unit tests for every new heuristic (reset, shapes, arena containment, no NaN in 200 steps) before training.

## 10. Risks

- **No gap exists** (H0 false): a legitimate result. Strengthen with F6 exploiters and real-world F7 before concluding.
- **Exploiters are degenerate** (they win by crashing the pursuer into the floor): keep them only if neutralization
  of the *specialist* is high, i.e. the exploit is counterable.
- **Context ignored in training**: if the train pool has no strategy-dependent best response, every adaptive method
  collapses to M0. O1 detects this early.
- **Short episodes** limit S1: emphasise S2 (across-episode), which matches the real deployment story.
