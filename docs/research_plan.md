# Research plan: diversity-aware opponent sampling and online opponent adaptation for competitive drone interception

Single source of truth for the thesis experiments (started 2026-10-26, experiment freeze 2026-11-05, thesis due
~2026-11-09). Results go to [experiments/thesis/RESULTS.md](../experiments/thesis/RESULTS.md) and are reported as
measured, including null and negative results. Predictions below are written **before** running anything, so they
can be checked against the outcome.

## 0. Starting point (verified in the repo on 2026-10-26)

| Fact | Evidence | Consequence for this plan |
|---|---|---|
| 1v1 Crazyflie pursuit-evasion, CTBR actions, 4 x 4 x 2.9 m arena, control step 0.02 s (sim dt 0.01 x 2 substeps, 50 Hz), 1500-step (30 s) max episode | [PursuitEvasion.yaml](../cfg/task/PursuitEvasion.yaml), `cfg/base/sim_base.yaml` | Same setting as Roncero 2026 (also Crazyflie) |
| Episodes are short: ~90 steps (~1.8 s) mean episode length. The self-play metric reports ~0.9 s because it multiplies steps by `sim.dt` instead of `sim.dt x substeps`; thesis evals use 0.02 s per step | `v11-uniform-mix/db/matchups.json` | Little time to adapt within an episode, so **across-episode** adaptation (same evader met repeatedly) is the main adaptation setting |
| Population self-play with a multi-metric hardness sampler (win rate, TTC, min distance, effort, crash) | [policy_db.py](../omni_drones/utils/selfplay/policy_db.py), [train_selfplay.yaml](../scripts/selfplay/train_selfplay.yaml) | Baseline sampler already exists. Gavin 2026 already uses an outcome-aware (time + crash) PFSP, so "multi-metric" alone is not novel |
| 4 recent campaigns with checkpoints: `v9-min-distance` (21 stages), `v10-without-warm-start` (13), `v11-uniform-mix` (13), `selfplay_v6_fix_reward_and_bounds` (12) | `scripts/selfplay/outputs/` | About 30 RL evaders and 30 RL pursuers are available without new self-play |
| One self-play stage takes ~1.3 h (500M frames + eval); one 100M-frame single-pursuer run takes ~17 min | checkpoint timestamps, `esn_adapt/report_p0` | Full self-play campaigns (~17 h) are only affordable once or twice. Most experiments train **one pursuer against a fixed evader pool** |
| ESN-OA (reservoir + online RLS) gave no gain: at most +0.007 capture over PPO; GRU -0.10 | [summary.md](../experiments/esn_adapt/report_p0/summary.md) | The ESN target (next-step rel. distance/velocity) is already in the observation. The held-out heuristics were not hard. RL-evader replay may be broken (T2 ~1 % capture with 78-95 % evader crash/OOB). These were found in an earlier review, not yet fixed |
| Eval uses ~680 episodes per matchup | `matchups.json` | 95 % CI ~ +/-0.03. Thesis evals count every episode that starts in the first 1500 steps of a 3000-step rollout (~15x more episodes, consistent estimator; `eval_cutoff_steps` in [run_stage.py](../scripts/selfplay/run_stage.py)) |
| Compute: one RTX 5090 (32 GB), 20 CPU cores | `nvidia-smi` | Isaac jobs run one at a time |

## 1. Research questions and hypotheses

**RQ1 (diversity).** Does sampling opponents by *quality and diversity* produce pursuers that are more robust to
unseen evaders than sampling by hardness alone (PFSP / Hedge / multi-metric)?

- H1a: Self-play pools are redundant. Consecutive warm-started stages produce near-duplicate evaders, so the number
  of distinct behaviour clusters is much smaller than the pool size.
- H1b: Hardness-only samplers are **not robust to duplicates**. Copying an opponent increases its sampling mass
  linearly, which hurts performance on the other opponents. Under a quality-diversity DPP sampler the mass of a
  duplicated cluster is bounded (§2.1).
- H1c: The QD sampler improves **worst-case** capture on held-out evaders and lowers exploitability, without hurting
  in-distribution capture by more than 0.02.

**RQ2 (adaptation).** Can a pursuer adapt online to an unseen evader, and does an adaptive-control-style learner
(learned basis + online RLS, as in Neural-Fly) beat a random reservoir (ESN), recurrence and non-adaptive policies?

- H2a (premise): there are evaders with a real adaptation gap `Delta_E = J(specialist_E, E) - J(generalist, E) >= 0.15`.
- H2b (information value): an oracle given the evader's behaviour descriptor or its current acceleration (intent)
  captures more than the plain generalist. If no oracle helps, no adaptation module can, and RQ2 becomes a negative
  result with an explanation.
- H2c: NF-OA lowers the evader-prediction error versus ESN-OA at the same horizon and closes more of `Delta_E`.
- H2d (it is *adaptation*): the gain increases with the number of encounters (across-episode) and disappears when the
  RLS state is frozen.

`J` = neutralisation rate = capture + evader crash + evader OOB (the interceptor forced a failure). Capture alone is
reported too.

## 2. Methods

### 2.1 Selection (provisional; confirmed at gate G2 after the literature step L0)

| ID | Method | Based on | Why it is novel for this topic | Priority |
|---|---|---|---|---|
| **QD-DPP** | Quality-diversity DPP opponent sampler | Perez-Nieves et al. ICML 2021 (DPP diversity in PSRO); Kulesza & Taskar 2012 (quality-diversity L-ensemble); Liu et al. NeurIPS 2021 (response + behavioural diversity) | All three drone PE papers prioritise by difficulty only. Diversity is claimed (AgilePE, Roncero) but not measured or optimised | P0 |
| **NF-OA** | Neural-Fly-style opponent adaptation: offline-learned basis `phi` + online RLS on the last linear layer, predicting the evader's future relative position | O'Connell et al. Sci. Robotics 2022 (DAIML + composite adaptation); ESN-OA (arXiv 2602.06326) | Roncero 2026 leaves "online adaptation" as future work. Learned-basis online adaptation of an **opponent model** for agile interception has not been reported | P0 |
| **BayesID** | Bayesian opponent identification over diversity prototypes, with a prototype-conditioned pursuer and a conservative fallback | Bayesian Policy Reuse (Rosman 2016); GSCU (Fu et al. ICML 2022); RMA (Kumar et al. RSS 2021) | Links RQ1 and RQ2: prototypes are the DPP clusters | P1 (only if G1 shows headroom) |

**QD-DPP sampler.** For opponents `i = 1..N`:
- **Quality** `q_i`: the existing multi-metric hardness of `i` for the current learner, mapped to (0, 1].
- **Behaviour descriptor** `b_i`, z-normalised (defined in X2): response vector against a fixed probe set of pursuers plus trajectory statistics.
- **Similarity** `S_ij = exp(-||b_i - b_j||^2 / 2 sigma^2)`, with `sigma` set to the median pairwise distance.
- **L-ensemble** `L = diag(q) S diag(q)`.
- **Sampling** `p_i ∝ K_ii` with `K = L (L + I)^-1` (DPP marginal inclusion probabilities), mixed with the existing `uniform_mix` floor.

N <= 60, so this is one small matrix inverse per stage.

Exact copies make the corresponding block of `L` rank one. For `m` copies of quality `q`, the cluster's total
inclusion mass is `m q^2 / (1 + m q^2)`, which is bounded by 1. For q = 1 it is at most 2x the mass of a single copy,
whereas under softmax/PFSP sampling it grows linearly in `m`. Implemented in `policy_db.py` as a new `sampler: qd_dpp`
option; the existing samplers stay as baselines.

**NF-OA.**
1. **Offline.** Roll out the generalist pursuer against the train evaders. Learn `phi(h_t) in R^d`, with d = 32 and `h_t` = the last 10 steps of relative position and velocity plus the pursuer's own velocity. Train it with per-evader linear heads `theta_e`, as a multi-task regression of the evader's relative position at `t + H`. Optional: a DAIML adversarial identity loss.
2. **Online.** `theta` starts at the mean head `theta_bar`. It is updated by RLS with forgetting `lambda`, using the delayed target (the target for time `t - H` becomes known at time `t`). The prediction `y_hat` is appended to the observation.
3. **Implementation.** This replaces the random reservoir in [esn.py](../omni_drones/utils/esn.py) (`BatchedESN` already has per-env RLS) with a learned feature map: new option `esn.feature: learned`.
4. **Across-episode setting (S2).** `esn.reset_on_episode: false` keeps `theta` across consecutive episodes against the same evader.

**BayesID.**
1. **Prototypes.** Take the K = 4-6 medoid evaders of the DPP clusters.
2. **Belief.** For each prototype `k`, run its frozen policy on the evader's state (full state, as in mocap) to predict the evader's acceleration. Score the observed acceleration with a Gaussian likelihood, and update the belief with forgetting.
3. **Policy.** The pursuer is trained with the privileged prototype one-hot and gets the belief at test time.
4. **Fallback.** If `max(belief) < tau`, it uses the generalist instead (GSCU's "conservative when uncertain").

### 2.2 Baselines

| ID | Baseline | Role |
|---|---|---|
| S-U | Uniform FSP sampler (`uniform_mix: 1`) | AgilePE FSP |
| S-W | Win-rate-only Hedge / PFSP (`metric_weights` = win_rate only) | Roncero AMSPBH, AgilePE PFSP |
| S-M | Current multi-metric softmax sampler | Gavin-style outcome-aware PFSP / your v11 |
| M0 | PPO-MLP, no adaptation | Lower bound |
| M0h | PPO-MLP + frame stack of the same 10-step history | "Is it just more input?" |
| ESO | Finite-difference / extended-state-observer estimate of evader acceleration appended to obs | Classical adaptive-observer baseline (APN-style) |
| ESN | Existing ESN-OA `shared_adapt`, same target and horizon as NF-OA | Random-basis ablation of NF-OA |
| GRU | `algo=ppo_gru` with 2x frames | Recurrent in-context baseline (only if time allows; P1) |
| PN / PP / MPC | Existing classical pursuers ([controllers/](../omni_drones/controllers)) | Interpretable reference |

## 3. Repository layout

```
docs/research_plan.md                    # this file
scripts/experiments/thesis/
    run.py                               # resumable driver: python scripts/experiments/thesis/run.py --exp X1 [--seeds 3]
    common.py                            # pool builders, run_stage wrappers, DONE markers, status.json
    exps/<id>.py                         # one module per experiment (X0 ... F1)
    report.py                            # tables + figures -> experiments/thesis/<ID>/summary.md
experiments/thesis/
    RESULTS.md                           # aggregated, truthful results (auto tables + written interpretation)
    L0_literature/notes.md               # what each paper did, novelty check
    X0_replay/  X1_crossplay/  X2_descriptors/  X3_gap/  X4_oracles/
    D1_sampler/  D1b_duplicates/  D2_exploitability/  D3_campaign/
    A1_nfoa/  A2_bayesid/
    F1_final/
```

Every experiment directory contains `config.json` (exact overrides, git hash), `status.json`, raw `*.json` evals,
`summary.md` and `fig_*.png`. Checkpoints go to `experiments/thesis/<ID>/ckpt/` (added to `.gitignore` in the first implementation step).

## 4. Experiments

GPU hours are estimates for sequential Isaac runs. "Pred." is the expected outcome stated before the run.

### Phase X: diagnostics (days 1-3)

| ID | Question | Design | Outputs | Pred. | GPU h |
|---|---|---|---|---|---|
| **X0_replay** | Are stored RL policies replayed correctly in the current env? | Re-evaluate 3 stored matchups per campaign (v6, v9, v10, v11) with 2048 episodes; compare with `matchups.json` | `replay_check.json`, table of `|Delta|` per campaign | v9/v10/v11 within 0.05. v6 (older obs layout) fails and is excluded | 0.5 |
| **X1_crossplay** | Is the game transitive? Do campaigns find different strategies? | Payoff matrix: ~20 pursuers (last 5 stages of v9/v10/v11 + PN/PP/MPC) x all valid evaders (~30 RL + 8 heuristics), 2048 eps each. Fit an Elo/Bradley-Terry model; count 3-cycles | `payoff_neut.npy`, `payoff_capture.npy`, `fig_payoff.png`, `transitivity.json` (Elo R², # cycles) | Mostly transitive (Elo R² 0.8-0.9), with real cross-campaign failures (some final pursuer < 0.6 vs another campaign's evader) | 2 |
| **X2_descriptors** | How redundant are self-play pools? | Behaviour descriptor per evader: (a) response vector vs a probe set {PN, PP, MPC, v9/v10/v11 final pursuers}; (b) trajectory statistics (mean and std of speed, altitude, lateral acceleration, distance to arena centre, bearing to pursuer, dominant jink frequency), logged as running sums in eval. Clustering (k-medoids + silhouette), DPP log-det per campaign | `descriptors.parquet`, `fig_umap.png`, `clusters.json`, `redundancy.json` | **H1a:** consecutive stages are near-duplicates; 4-7 clusters across ~38 evaders; each campaign covers 2-3 clusters | 0 (reuses X1 rollouts) |
| **X3_gap** | Do unseen evaders exist that the generalist loses to but a specialist beats (H2a)? | Generalist `pi_G` = M0 trained 150M frames on the train split (§5). Test evaders: held-out campaign evaders, switch heuristics, and **exploiters** (evader PPO best responses to frozen `pi_G`, 3 seeds x 50M frames). Specialist = `pi_G` fine-tuned 20M frames on each test evader | `gap.json` (`J_G`, `J_spec`, `Delta` per evader), `fig_gap.png` | Exploiters: `Delta` >= 0.25 for >= 2 of 3. Held-out RL evaders: 0.05-0.2. Heuristics: < 0.05 (as in P0) | 3 |
| **X4_oracles** | Is opponent information worth anything (H2b)? | Train pursuers with privileged obs, 3 seeds x 150M: **O1** evader descriptor `b_i` (from X2); **O2** evader true world acceleration (intent); **O3** evader true relative position at +0.3 s (future; computed from a 15-step delayed replay buffer, valid offline only). Eval on train, held-out and X3 exploiters | `oracle.json`, `fig_oracle.png` | O2 (intent) helps most (+0.05-0.15 on exploiters / switches), O1 helps little in-distribution (< 0.03), O3 sets the target horizon for NF-OA | 4 |

**Gate G1 (end of day 3).**
- If at least one oracle closes at least 30 % of `Delta` on the X3 evaders, run Phase A as planned.
- If no oracle helps, or no evader has `Delta >= 0.15`, Phase A shrinks to A1 with M0 / ESO / ESN / NF-OA only. That chapter then reports a characterised negative result: the generalist is information-saturated in this full-state setting. The thesis weight moves to Phase D.

### Phase L: literature (day 2-3, in parallel with GPU diagnostics)

| ID | Task | Output |
|---|---|---|
| **L0** | Read: AgilePE, Roncero 2026, Gavin 2026; Perez-Nieves 2021; Liu 2021, 2022; Yao 2023 (PSD-PSRO); Balduzzi 2019; Neural-Fly 2022; RMA 2021; GSCU 2022; BPR 2016; Al-Shedivat 2018; ESN-OA. Novelty search: "DPP / diversity opponent sampling drone pursuit", "online opponent adaptation quadrotor interception" | `L0_literature/notes.md`: per paper method, setting, limits, and what our method changes. **Gate G2:** confirm or swap QD-DPP / NF-OA / BayesID |

### Phase D: diversity-aware sampling (days 4-8)

**Pool split (§5) is fixed before any D-run.**

| ID | Question | Design | Outputs | Pred. | GPU h |
|---|---|---|---|---|---|
| **D1_sampler** | H1c: does QD-DPP give more robust pursuers? | Train one pursuer per sampler {S-U, S-W, S-M, QD-DPP} on the **fixed** train pool, 150M frames, 3 seeds (5 if time). Sampler re-evaluated every 25M frames (mini-generations), with a cheap 256-episode eval vs the pool to update hardness | `train_curves.json`, `eval_heldout.json`, `fig_sampler_bars.png` (mean, worst-cluster, in-distribution) | QD-DPP >= S-M on held-out worst-cluster by 0.03-0.08; equal mean in-distribution (+/-0.02); S-U worst on hard evaders | 6 |
| **D1b_duplicates** | H1b: robustness to duplicated opponents | Same as D1 but the train pool has one cluster duplicated 5x (copies of the same checkpoints). Samplers S-M and QD-DPP, 3 seeds | `dup.json`, `fig_dup.png` (capture on non-duplicated clusters, dup vs no-dup) | S-M drops 0.05-0.10 on non-duplicated clusters, QD-DPP drops < 0.02 | 2.5 |
| **D2_exploitability** | Are QD-DPP pursuers harder to exploit? | For each D1 final pursuer (4 samplers x 3 seeds), train a fresh exploiter evader, 50M frames, 1 seed. Exploitability = exploiter's evasion rate (1 - `J`) | `exploit.json`, `fig_exploit.png` | QD-DPP lowest exploitability, but the difference vs S-M may be within the CI | 3 |
| **D3_campaign** (P1) | Does QD-DPP help in full self-play? | One campaign with `sampler: qd_dpp`, 9 stages x 300M frames, same seeds/heuristics as `v11-uniform-mix` (baseline). Cross-play final pursuers vs the other campaign's evaders, plus a D2-style exploiter each | `campaign_compare.json`, win-rate matrices, per-stage diversity log-det | QD-DPP evader pool has higher log-det (more clusters); final pursuer at least as strong in cross-play | 8 |

### Phase A: online adaptation (days 5-10; GPU runs interleave with Phase D)

Evaluation settings for every A-method:
- **S1 within-episode:** the adaptation state is reset at each episode.
- **S2 across-episode:** 30 consecutive episodes against the same evader with no reset. Plot `J` vs episode index and report the area under the curve.
- **S3 switch:** the evader switches strategy at t = 0.5 s. Report recovery time.

| ID | Question | Design | Outputs | Pred. | GPU h |
|---|---|---|---|---|---|
| **A1a_basis** | Can a learned basis predict the evader better than a reservoir? | Collect 50k episodes of `pi_G` vs the train evaders; train `phi` + per-evader heads (offline, CPU/GPU, no Isaac). Compare open-loop prediction error of: CV extrapolation, ESN+RLS, raw-history RLS, NF frozen `theta_bar`, NF+RLS; horizons {0.1, 0.3, 0.5} s; on train and held-out evaders | `pred_err.json`, `fig_pred_err_vs_horizon.png`, `fig_err_vs_time.png` | NF+RLS < ESN+RLS by 30-50 % at 0.3 s on held-out; NF frozen in between; CV good at 0.1 s | 1 |
| **A1b_policy** | H2c/H2d: does it improve interception? | Train pursuers {M0, M0h, ESO, ESN, NF-frozen, NF-OA} on the train pool, 150M frames, 3 seeds. Eval S1/S2/S3 on train, held-out, X3 exploiters | `adapt_eval.json`, `fig_gap_closed.png`, `fig_S2_curves.png`, `fig_S3_recovery.png`, inference latency table | NF-OA closes 20-40 % of `Delta` on exploiters in S2, about 0 in-distribution; NF-frozen gets about half of that; ESN about 0 (as in P0). Within-episode (S1) gains small because episodes are short | 8 |
| **A2a_nfacc** (added 10-27, user choice B) | Can the learned basis + RLS estimate the evader's *acceleration* better than a finite difference? | Same NF/RLS machinery, new target: evader velocity change over the next H steps (= predicted mean acceleration x H dt). Offline on the A1a trajectories; H in {1, 5, 10}; RLS tuned on train-pool validation envs; compared with finite-difference persistence (last-step acceleration) | `A2a_nfacc/pred_err.json`, `summary.md` | Learned predictor beats finite-difference persistence for H >= 5; small gain from RLS | 0.5 |
| **A2b_nfacc** (added 10-27) | Does a learned acceleration estimate help the pursuer as much as the finite-difference input (O2a)? | Pursuers with the NF acceleration estimate (scaled like O2a) as input, frozen and with RLS, 3 seeds, noise-free; S1/S2 evals | `A2b_nfacc/summary.md` | Close to O2a (within 0.05) noise-free; RLS gain small | 7 |
| **A3a_posonly** (added 10-28, user request) | Without the evader's velocity, does a learned estimator of its acceleration beat finite differences of positions? Clean and noisy inputs | Re-record generalist trajectories with the pursuer's own position. Estimator input window [rel_pos, own_pos, own_vel] (no evader velocity), target = true evader velocity change over H in {5, 10}, frozen basis (online RLS needs the unobservable evader velocity as target). Basis trained on clean inputs and on noise-augmented inputs. Baselines: zero, second-difference-of-position persistence, raw and EMA-filtered (alpha tuned on val). Noise model = `observation.noise` stds. Pre-registered: H* = H with the largest error reduction of the frozen basis vs FD persistence on clean val (ties -> 10); the policy uses the noise-augmented basis if its noisy-val error is lower and its clean-val error <= 1.1x the clean basis's | `A3a_posonly/summary.md` | Learned beats FD clearly with noise, roughly equal without | 0.5 |
| **A3b_posonly** | Same question in closed loop | Pursuers whose observation has **no evader velocity** (`observation.agent_rel_vel=false`), train pool, uniform, 150M frames, noise-free training, 3 seeds: P_none (no estimator), P_fd (second-difference acceleration, `oracle.accel_source=pos`), P_nf (frozen learned estimator, `adapt.z_mode=pos`). Eval noise-free and with sensor noise (P_fd also with the tuned EMA) | `A3b_posonly/summary.md` | P_fd ~ P_nf noise-free; P_nf > P_fd with noise; both > P_none | 6 |
| **A3c_noise_ref** | How do the velocity-based policies degrade under sensor noise? | Eval only: M0, finite-difference acceleration input (F1 SU_acc), NF-acc frozen and NF-acc + RLS (A2b) with `observation.include_noise=true` | `A3c_noise_ref/summary.md` | Finite-difference acceleration degrades most | 1 |
| **A3d_velest** (added 10-29, user choice after A3b/A3c, pre-registered 10-29 11:30 before any A3d result) | Can a learned estimator recover the evader's *velocity* (and acceleration) from noisy positions well enough to replace a measured velocity? | Offline: A3a trajectories; window [rel_pos, own_pos, own_vel] (K = 10), target = evader's current velocity; frozen basis trained clean / noise-augmented (A3a selection rule); baselines: first difference of the position estimate, raw and EMA (alpha tuned on noisy val); reference: A3c's measured velocity error. Closed loop: train pool, uniform, 150M, noise-free training, 3 seeds; the estimate replaces the true relative velocity (`observation.rel_vel_source`). **V_ema** = tuned EMA difference; **V_nf** = learned velocity + A3a learned acceleration (one shared noisy measurement per step). Eval clean / noisy; refs M0 (A3c), P_none (A3b) | `A3d_velest/summary_offline.md`, `summary.md` | Offline: NF_aug < FD_ema on noisy val/test; noise-free FD_raw best. Closed loop, noisy eval: V_nf > V_ema on train and held-out RL; both far above P_none (0.29); both below M0 noisy (0.70) | 3 |
| ~~A2_bayesid~~ | dropped to P2 at gate G2 (BPR / GSCU already cover it) | | | | |

### Phase F: final system and write-up (days 10-11 GPU, days 11-14 writing)

| ID | Design | Outputs |
|---|---|---|
| **F1_final** (redesigned 10-27, user choice B) | 2x2 factorial {uniform, QD-DPP sampler} x {plain obs, opponent-acceleration input}, 5 seeds, noise-free; existing runs reused (X3, D1, X4 O2a). A QD-DPP + learned-acceleration cell is added only if A2b beats O2a. Main effects, interaction (seed-paired) | `F1_final/summary.md`, `results.json` |

| **F2a_exploit_acc** (added 10-28, user request, pre-registered before A3b/A3c/D1c results) | Is the acceleration-input pursuer less exploitable? | One fresh exploiter (plain-obs evader, PPO, 100M frames, seed 200 + s as in D2) against each F1 `SU + acc` pursuer, seeds 0-4. Exploitability = 1 - J vs its exploiter. Seed-paired comparison with D2 `S-U` (same seeds, plain obs) | `F2a_exploit_acc/summary.md` | Lower than plain (D2 S-U 0.672), mean 0.3-0.5, paired CI excludes 0; still clearly > 0 |
| **F2b_campaigns** (same) | Does the acceleration input help in co-evolution, the protocol of the reference papers? | Two self-play campaigns (`train_selfplay.py`), identical except the input: **acc** (both roles observe the opponent's finite-difference acceleration; frozen opponents too, `oracle.opp_side_accel`) vs **plain**. Uniform opponent sampling (`uniform_mix: 1`), 9 stages x 300M frames, pursuer first, warm start, seed 0, scripted seeds pursuer {pure_pursuit, apf}, evader {hover, circular, flee, weave} (the thesis train heuristics). Noise-free. 1 seed per arm: descriptive, no significance claim | campaign DBs in `scripts/selfplay/outputs/thesis-F2-{acc,plain}` | - |
| **F2c_final_eval** (same) | Final comparison | Every RL pursuer of both campaigns vs every RL evader of both campaigns (cross-play matrix, Roncero-style heatmap); final pursuer of each campaign vs train pool and thesis test set (v10, held-out scripted, X3 exploiters); one fresh plain-obs exploiter (100M frames) per final pursuer | `F2c_final_eval/summary.md`, `fig_crossplay.png` | acc final pursuer >= plain final pursuer + 0.1 J on the other campaign's evaders and on the test set, and lower exploitability. Within-campaign J need not rise, since the acc evaders also observe the pursuer's acceleration |

| **F3_sampler_sp** (added 10-28, user request, pre-registered 20:30 before any F2/F3 result) | RQ1 under the reference papers' protocol: do diversity-aware or hardness-prioritised opponent samplers beat uniform in full self-play? | Campaigns as F2b "plain" (plain obs, 9 stages x 300M, evader side uniform, multi-episode evals with behaviour statistics). Only the **pursuer's** opponent sampler changes: **uniform**; **hedge** = Roncero eq. (12) exactly (gamma 0.7, eta 2, challenge c = 1 - J of the latest pursuer); **qd_dpp** = the same Hedge quality re-weighted by a DPP over evader behaviour descriptors (mean/std of the 9 per-step behaviour features over the evader's recorded matchups, z-normalised over the pool; trajectory block only, unlike X2, because the probe set changes during a campaign). 3 seeds each (campaign seed 0/100/200; uniform seed 0 = F2b plain). Final pursuer of each campaign vs train pool, thesis test set, all 36 campaign evaders, and one fresh exploiter. Primary: held-out RL J, worst held-out cluster, J vs other campaigns' evaders, exploitability; seed-paired differences qd_dpp - hedge, qd_dpp - uniform, hedge - uniform | `F3_sampler_sp/summary.md` | Roncero predicts hedge > uniform; D1 predicts hedge <= uniform and qd_dpp ~ uniform. With gamma = 0.7 the three sampling distributions are close (70 % uniform), so all paired differences are expected within +/- 0.05 and mostly not significant with 3 seeds |

| **F4_estimator_sp** (added 10-28, user request, pre-registered 21:30 before A3b/A3c/F2/F3 results) | RQ2 under realistic sensing in full self-play: when the pursuer cannot observe the evader's velocity and its sensors are noisy, does the learned acceleration estimator help? | Campaigns (uniform sampling, 9 stages x 300M, 3 seeds: campaign seed 0/100/200). Pursuer: no evader velocity (`agent_rel_vel=false`), sensor noise in training and eval (`include_noise=true`). Arms: **nf** = + the A3a estimator (frozen, noise-augmented basis, H = 10, as A3b P_nf); **none** = no estimator. Evader: full clean state (as in the reference papers). Frozen pursuers seen by the evader get the same inputs with clean sensors (`opp_drop_rel_vel`, `adapt.opp_side`). F4c: final pursuer of each campaign (noisy) vs train pool, test set, all 6 F4 campaigns' evaders, the test set without noise, and one fresh exploiter. Primary: seed-paired nf - none on held-out RL, X3 exploiters, other campaigns' evaders, exploitability | `F4_estimator_sp/summary.md` | nf > none by >= 0.1 on held-out RL and X3 exploiters, CI excluding 0 (A3a: only the learned estimator carries acceleration information under noise); exploitability lower for nf |
| **F4_estimator_sp v2** (redesign 10-29 11:30, user choice, before any F4 campaign started; replaces the row above) | Same question with a velocity estimate: A3b showed that without the evader velocity every pursuer is weak (~0.29) and an acceleration estimate alone does not help, so v1 would compare two crippled pursuers | As v1 (uniform, 9 stages x 300M, 3 seeds, sensor noise in pursuer training and eval, evader full clean state), but both arms observe an *estimated* evader velocity in place of the true one: **nf** = A3d V_nf inputs (learned velocity + learned acceleration); **ema** = A3d V_ema inputs (tuned EMA finite difference, the classical baseline). Frozen pursuers seen by the evader get the same estimators with clean inputs (`opp_rel_vel_source`, `adapt.opp_side`). F4c and metrics as v1; primary: seed-paired nf - ema | `F4_estimator_sp/summary.md` | nf > ema on held-out RL and X3 exploiters by >= 0.05; exploitability lower for nf. Low confidence (3 seeds, no fixed-pool result yet) |

Final method for F2 = uniform sampling + finite-difference opponent acceleration (velocity source). The learned estimator (A2b) is not used in F2: it is noise-free-equivalent to the finite difference (A2b), and a symmetric learned estimator for the evader role does not exist. Sensor-noise robustness is reported separately (A3b/A3c).

Total estimated GPU time is about 40 h for P0 plus about 11 h for P1, which fits the window with room for reruns.

## 5. Protocol details

- **Pool split (train / held-out), pre-registered on 2026-10-26 before any X1 data was seen.**
  - Held-out RL evaders: the `v10-without-warm-start` campaign. It is the only lineage trained without warm-starting, so it is the most independent test.
  - Train RL evaders: `v9-min-distance` + `v11-uniform-mix` (+ `v6` only if X0 shows it replays validly).
  - Train heuristics: {hover, circular, flee, weave}. Held-out heuristics: {zigzag, fast_flee} and the switch forms {flee>circular, circular>hover, hover>flee, flee>weave} at `@0.5`.
  - Held-out also includes the X3 exploiters.
  - If X0 shows v10 does not replay validly, the held-out RL set becomes the last 5 stages of v9 (and train keeps v9's first half). This is the only allowed change.
  - Note: heuristic time is `progress x sim.dt`, i.e. half the real time. `@0.5` switches at 50 steps (1 s real); the old `@2.0` switches at 200 steps, after most episodes have ended.
- **Fair comparison.** Identical PPO hyperparameters, reward, DR, frame budget and eval seeds within an experiment. Only the factor under test changes.
- **Statistics.**
  - Seeds: 3 per cell (5 in F1).
  - Episodes: >= 2048 per (policy, evader) pair.
  - Aggregation: report IQM with stratified bootstrap 95 % CIs (`rliable`) over seeds x evaders, plus paired differences vs the baseline.
  - Rule: a claim is "supported" only if the CI of the paired difference excludes 0.
- **Metrics.**
  - Neutralisation `J` (primary) and capture rate.
  - Right-censored time to neutralise (15 s on failure).
  - Pursuer crash and OOB rates, motor effort.
  - Worst-cluster `J` (min over held-out clusters).
  - Exploitability (D2).
  - Gap closed `GC = (J_A - J_G) / Delta`.
  - Prediction error vs horizon.
  - Inference latency per step.
- **Observation noise (P2, moved on 10-27).** Evaluation with `task.observation.include_noise=true` (noisy
  finite-difference acceleration, optional EMA `task.oracle.accel_ema`; noisy adapter inputs) is P2, run only if
  time remains after F1 and A2.
- **Honesty rules.** No metric or test set is changed after seeing results without saying so in RESULTS.md. Every run that crashed or was excluded is listed. Predictions in this file are not edited after the fact.

## 6. Code changes

| Change | File | Test |
|---|---|---|
| Neutralisation outcome, 2048-episode evals, trajectory-statistic accumulators | [run_stage.py](../scripts/selfplay/run_stage.py), [pursuit_evasion.py](../omni_drones/envs/single/pursuit_evasion.py) | `tests/test_selfplay_metrics.py` extended |
| Privileged obs keys: descriptor (O1), evader acceleration (O2), future rel. position (O3) | `pursuit_evasion.py` | shape / no-NaN test |
| `sampler: qd_dpp` and win-rate-only option, descriptor loading | [policy_db.py](../omni_drones/utils/selfplay/policy_db.py) | `tests/test_qd_dpp.py`: bounded cluster mass under duplication (matches the closed form), uniform-quality case, reduces to quality-only when `S = I` |
| Mini-generation sampler refresh inside a single training run | `run_stage.py` | smoke run |
| `esn.feature: learned`, `esn.horizon`, `esn.reset_on_episode` | [esn.py](../omni_drones/utils/esn.py), `pursuit_evasion.py` | RLS converges on a synthetic linear target; delayed target alignment |
| ESO acceleration feature | `pursuit_evasion.py` | matches finite-difference on a scripted evader |
| BayesID belief module (P1) | new `omni_drones/utils/bayes_id.py` | belief concentrates on the true prototype in a scripted test |
| Experiment driver and report | `scripts/experiments/thesis/` | smoke run of every experiment at 2M frames |

## 7. Schedule

| Day (date) | Work | Gate |
|---|---|---|
| 1 (10-26) | Code: metrics, descriptors, oracle keys; X0 | - |
| 2 (10-27) | X1 + X2 (GPU); L0 reading in parallel | - |
| 3 (10-28) | X3, X4 | **G1**, **G2** |
| 4-5 (10-29, 10-30) | QD-DPP code + tests; D1, D1b; NF-OA code; A1a | - |
| 6-7 (10-31, 11-01) | D2; A1b | - |
| 8-9 (11-02, 11-03) | A2 or D3 (P1, whichever G1 favours) | - |
| 10 (11-04) | F1 | - |
| 11 (11-05) | Experiment freeze, RESULTS.md complete | - |
| 12-14 | Thesis writing (not covered by this plan) | - |

## 8. Risks

| Risk | Mitigation |
|---|---|
| Old RL evaders do not replay correctly (X0) | Exclude those campaigns; v9-v11 alone still give ~22 evaders |
| No adaptation gap (H2a false) | Report it as a finding with oracle evidence (X4); the thesis weight moves to RQ1 |
| QD-DPP and S-M are within noise | Report it. D1b still tests the duplicate-robustness mechanism directly; increase seeds to 5 for D1 |
| Exploiters win by forcing pursuer crashes, not by evading | Report crash/OOB breakdown. Exploitability uses `1 - J` and the crash share is shown separately |
| Isaac crashes / time overruns | Resumable driver with DONE markers; P1 items are dropped first (D3, then A2, then GRU) |
| Real-world validation | Out of scope for the 2 weeks (needs the lab). Deploy code exists in [scripts/deploy/](../scripts/deploy) for later |

## 9. Deviations from this plan (logged as they happen)

| Date | Change | Reason |
|---|---|---|
| 10-29 | F4 redesigned (v2) and A3d added, pre-registered 11:30 before any A3d result and before F4 started (only the 3-stage v1 smoke ran). Env: `observation.rel_vel_source` / `opp_rel_vel_source` (truth / nf / fd_ema), adapter target `vel`. F4 now starts with A3d's closed-loop runs (shared lanes), then a v2 smoke, then the campaigns | A3b: without evader velocity no acceleration input helps; A3c: learned acceleration survives noise. User chose to make the estimator recover the velocity too |
| 10-29 | F2a analysis rebuilt from the eval files on disk (bug: a lane that found a seed locked by the other lane left it out of its own summary; 3 of 5 seeds were shown) | Bookkeeping fix, no rerun |
| 10-26 | X4 O3 (future-position oracle) dropped | Needs a delayed-observation mechanism; A1a's offline horizon study answers the same question |
| 10-26 | X4 O2 split into O2a (measured acceleration) and O2b (latest command) | O2a is computable from consecutive full-state observations, so it doubles as the ESO / finite-difference baseline for A1b |
| 10-26 | BayesID dropped from P1 to P2 (gate G2) | L0: concept already covered by BPR / Bayes-ToMoP / GSCU |
| 10-26 | D1 samplers refresh opponent probabilities inside one training run every 25M frames (from the episodes played), not via separate mini-generation evals | Saves an Isaac start-up + eval per refresh. Episodes in flight at a refresh (~1-2 %) are credited to the new opponent |
| 10-26 | S-U (uniform sampler), M0 and the X3 generalist are the same runs | Identical configuration; trained once |
| 10-26 | A1b compares M0h (frame stack), ESN, NF-frozen, NF-OA (+ reused M0, O2a); GRU not run unless time remains | Budget |
| 10-26 | Clustering searches k = 2..8 | Fixed before seeing X1/X2 results |
| 10-26 | D1, D2 and A1b go to 5 seeds (generalist/S-U seeds 3-4 added via `X3 --seeds 5`); X4 stays at 3 seeds | Decided after X3 showed large seed variance (generalist train J 0.75 / 0.64 / 0.56) and before any 3-seed D1/A1b result was available |
| 10-26 | The D1 "exploiter" column is biased against S-U (the X3 exploiters were trained against S-U seed 0); D2 is the fair exploitability comparison | Noted when the first D1 numbers came in |
| 10-27 | A1a: RLS uses a bounded covariance (trace <= initial) and (delta, forgetting) tuned per feature kind and horizon on 20 % held-out train-pool envs; A1b uses the tuned values | The pre-registered defaults (delta 10, forgetting 0.98) wound up and diverged (run 1 kept as `A1a_basis/*_run1_untuned.*`) |
| 10-27 | **D1c_nofloor (exploratory, added after seeing D1b)**: S-M vs QD-DPP with `uniform_mix = 0`, normal and duplicated pool, 3 seeds | D1b's 0.5 uniform floor is not duplicate-robust, so D1b could not isolate the DPP mechanism. Reported as exploratory |
| 10-27 | F1 redesigned (user choice B): 2x2 {uniform, QD-DPP} x {plain, acceleration input}, 5 seeds, noise-free; new A2a/A2b (learned acceleration estimator, NF target = evader velocity change); sensor-noise tests moved to P2 | A1b showed online adaptation adds nothing reliable while the explicit acceleration input is far better; the user chose option B without noise |
| 10-27 | D1c interrupted after 2 of 12 runs (kept) and requeued after A2 | Lane A capacity given to F1 / A2 |
| 10-28 | A3a/A3b/A3c added (positions-only observation; sensor-noise evaluation partly promoted from P2 to P1) | User request after A2b |
| 10-28 | F4 added (learned-estimator campaigns, user: run it regardless of A3b/A3c, because real interceptors rarely measure the evader's velocity or acceleration), pre-registered 21:30. Env flags `observation.opp_drop_rel_vel`, `adapt.opp_side`; orchestrator `role_task_overrides`. Queue order F2a, F2b, F4, F3, then final evals; if time runs short, F3 seed 2 is dropped first | User request |
| 10-28 | F3 added (user chose 3 samplers x 3 seeds in self-play), pre-registered 20:30. Campaign evals (F2b too, before it started) use the multi-episode estimator with behaviour statistics (`eval_extra_overrides`). RQ2 in self-play (learned estimator campaign): decided after A3b/A3c | User: compare the diversity sampler with hardness-only samplers under the papers' protocol |
| 10-28 | F2a/F2b/F2c added ("final run"; user chose exploitability test + two campaigns), pre-registered at 19:30 before A3b/A3c/D1c results. Replaces D3 (QD-DPP campaign): F1 showed no sampler effect, so the campaign tests the acceleration input instead. New env flag `oracle.opp_side_accel` | User request: a final publishable method, compared with the reference papers' co-evolution protocol |
| 10-28 | A3a re-run (run 1 kept as `A3a_posonly/run1_fp16/`) with trajectories recorded in float32; same pre-registered rule | Run 1's recorder stored positions as float16 (~2 mm steps), which makes the clean second-difference baseline worse than predicting zero (0.26 vs 0.18 m/s at H=5) and also degrades the learned estimator's clean inputs. Found before any A3b run used run 1's choice. A1a/A2a also used float16 recordings (velocity-based; smaller effect, not re-run) |
