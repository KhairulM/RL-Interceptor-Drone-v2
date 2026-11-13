# Research plan: diversity-aware opponent sampling and online opponent adaptation for competitive drone interception

Single source of truth for the thesis experiments (started 2026-10-26; freeze planned for 2026-11-05, moved to 2026-11-08 on 11-04 for F5, later moves in §9). Results go to [experiments/thesis/RESULTS.md](../experiments/thesis/RESULTS.md), reported as measured, including null and negative results. Day-to-day status: [experiments/thesis/PROGRESS.md](../experiments/thesis/PROGRESS.md).

**Pre-registration rules.** Every experiment entry in §4 is a pre-registration written **before** running anything. Its predictions are never edited; later changes are amendments, timestamped in the entry and logged in §9. On 2026-11-12 the file was reformatted (the §4 table became one entry per experiment; other sections condensed): predictions, success criteria and predictions stated in amendments are verbatim; design text was shortened without changing its meaning.

Working title (chosen 10-28): *Beyond Hard Opponents: Diversity-Aware Opponent Sampling and Learned Opponent Models for Agile Quadrotor Pursuit–Evasion*. Reference papers: AgilePE (Tang et al. 2026), Roncero et al. 2026 (AMSPBH), Gavin & Bronz (ICUAS 2026).

## Contents

0. [Starting point](#0-starting-point) · 1. [Research questions](#1-research-questions-and-hypotheses) · 2. [Methods](#2-methods) · 3. [Repository layout](#3-repository-layout)
4. [Experiments](#4-experiments): [X diagnostics](#phase-x-diagnostics) · [L literature](#phase-l-literature) · [D sampling](#phase-d-diversity-aware-sampling) · [A adaptation](#phase-a-online-adaptation) · [F final / self-play](#phase-f-final-system-and-self-play)
5. [Protocol details](#5-protocol-details) · 6. [Code changes](#6-code-changes-original-plan) · 7. [Schedule](#7-schedule) · 8. [Risks](#8-risks) · 9. [Deviation log](#9-deviation-log)

## 0. Starting point

Verified in the repo on 2026-10-26.

- **Setting.** 1v1 Crazyflie pursuit-evasion, CTBR actions, 4 x 4 x 2.9 m arena, control step 0.02 s (sim dt 0.01 x 2 substeps, 50 Hz), 1500-step (30 s) max episode ([PursuitEvasion.yaml](../cfg/task/PursuitEvasion.yaml)). Same setting as Roncero 2026 (also Crazyflie).
- **Short episodes.** ~90 steps (~1.8 s) mean (the self-play metric reports ~0.9 s because it multiplies steps by `sim.dt` instead of `sim.dt x substeps`; thesis evals use 0.02 s per step). So **across-episode** adaptation is the main adaptation setting.
- **Existing code and checkpoints.** Multi-metric hardness sampler (win rate, TTC, min distance, effort, crash) in [policy_db.py](../omni_drones/utils/selfplay/policy_db.py) (Gavin 2026 already uses an outcome-aware PFSP, so "multi-metric" alone is not novel). Campaigns `v9-min-distance` (21 stages), `v10-without-warm-start` (13), `v11-uniform-mix` (13), `selfplay_v6_fix_reward_and_bounds` (12): about 30 RL evaders and 30 RL pursuers.
- **Cost.** One self-play stage ~1.3 h (500M frames + eval); one 100M-frame single-pursuer run ~17 min. Full campaigns (~17 h) are affordable only once or twice, so most experiments train **one pursuer against a fixed evader pool**.
- **Prior ESN-OA result.** At most +0.007 capture over PPO; GRU -0.10 ([summary.md](../experiments/esn_adapt/report_p0/summary.md)). Its target was already in the observation, the held-out heuristics were not hard, and RL-evader replay may be broken (T2 ~1 % capture with 78-95 % evader crash/OOB).
- **Eval size and compute.** ~680 episodes per matchup (95 % CI ~ +/-0.03); thesis evals count every episode that starts in the first 1500 steps of a 3000-step rollout (~15x more; `eval_cutoff_steps`). One RTX 5090 (32 GB), 20 CPU cores.

## 1. Research questions and hypotheses

**RQ1 (diversity).** Does sampling opponents by *quality and diversity* produce pursuers that are more robust to unseen evaders than sampling by hardness alone (PFSP / Hedge / multi-metric)?
- H1a: Self-play pools are redundant. Consecutive warm-started stages produce near-duplicate evaders, so the number of distinct behaviour clusters is much smaller than the pool size.
- H1b: Hardness-only samplers are **not robust to duplicates**. Copying an opponent increases its sampling mass linearly, which hurts performance on the other opponents. Under a quality-diversity DPP sampler the mass of a duplicated cluster is bounded (§2).
- H1c: The QD sampler improves **worst-case** capture on held-out evaders and lowers exploitability, without hurting in-distribution capture by more than 0.02.

**RQ2 (adaptation).** Can a pursuer adapt online to an unseen evader, and does an adaptive-control-style learner (learned basis + online RLS, as in Neural-Fly) beat a random reservoir (ESN), recurrence and non-adaptive policies?
- H2a (premise): there are evaders with a real adaptation gap `Delta_E = J(specialist_E, E) - J(generalist, E) >= 0.15`.
- H2b (information value): an oracle given the evader's behaviour descriptor or its current acceleration (intent) captures more than the plain generalist. If no oracle helps, no adaptation module can, and RQ2 becomes a negative result with an explanation.
- H2c: NF-OA lowers the evader-prediction error versus ESN-OA at the same horizon and closes more of `Delta_E`.
- H2d (it is *adaptation*): the gain increases with the number of encounters (across-episode) and disappears when the RLS state is frozen.

`J` = neutralisation rate = capture + evader crash + evader OOB (the interceptor forced a failure). Capture alone is reported too.

## 2. Methods

Selection confirmed at gate G2 (after L0): QD-DPP and NF-OA kept; BayesID (Bayesian identification over diversity prototypes; BPR, GSCU, RMA) dropped to P2.

**QD-DPP** (Perez-Nieves et al. ICML 2021; Kulesza & Taskar 2012; Liu et al. NeurIPS 2021). Novelty: all three drone PE papers prioritise by difficulty only; diversity is claimed (AgilePE, Roncero) but not measured or optimised. For opponents `i = 1..N` (N <= 60):
- quality `q_i` = existing multi-metric hardness for the current learner, mapped to (0, 1]; descriptor `b_i` (z-normalised, X2) = response vector vs a fixed probe set + trajectory statistics;
- similarity `S_ij = exp(-||b_i - b_j||^2 / 2 sigma^2)`, `sigma` = median pairwise distance; `L = diag(q) S diag(q)`; `p_i ∝ K_ii` with `K = L (L + I)^-1` (DPP marginal inclusion), mixed with the `uniform_mix` floor;
- `m` exact copies of quality `q` get total mass `m q^2 / (1 + m q^2)`, bounded by 1 (for q = 1 at most 2x one copy), whereas softmax/PFSP mass grows linearly in `m`. Implemented as `sampler: qd_dpp` in `policy_db.py`.

**NF-OA** (O'Connell et al. Sci. Robotics 2022; ESN-OA arXiv 2602.06326). Novelty: Roncero 2026 leaves "online adaptation" as future work; learned-basis online adaptation of an **opponent model** for agile interception is not reported.
1. Offline: roll out the generalist vs the train evaders; learn `phi(h_t) in R^d`, d = 32, `h_t` = last 10 steps of relative position and velocity + own velocity, with per-evader heads `theta_e` (multi-task regression of the evader's relative position at `t + H`).
2. Online: `theta` starts at the mean head `theta_bar`, updated by RLS with forgetting `lambda` on the delayed target; the prediction `y_hat` is appended to the observation.
3. Implementation: learned feature map in place of the reservoir in [esn.py](../omni_drones/utils/esn.py) (`esn.feature: learned`); `esn.reset_on_episode: false` keeps `theta` across episodes (S2).

**Baselines.** S-U uniform FSP (`uniform_mix: 1`; AgilePE FSP) · S-W win-rate-only Hedge / PFSP (Roncero AMSPBH, AgilePE PFSP) · S-M current multi-metric softmax (Gavin-style) · M0 PPO-MLP (lower bound) · M0h M0 + 10-step frame stack ("is it just more input?") · ESO finite-difference / extended-state-observer evader acceleration in the obs (classical) · ESN existing ESN-OA, same target and horizon (random-basis ablation) · GRU `algo=ppo_gru` (P1, only if time) · PN / PP / MPC classical pursuers ([controllers/](../omni_drones/controllers)).

## 3. Repository layout

- Driver `scripts/experiments/thesis/run.py --exp <ID> [--seeds 3] [--smoke]` (resumable, DONE markers); `common.py` (pools, run_stage wrappers, locks, Isaac slot cap), `analysis.py` (bootstrap CIs), `git_sync.sh` (commit + push after each experiment), `exps/<group>.py`.
- Outputs `experiments/thesis/<ID>/` (e.g. `X0_replay/`, `D1_sampler/`, `A3c_noise_ref/`, `F2c_final_eval/`, `F3_sampler_sp/`) with raw `*.json` evals (exact opponent specs), logs, `summary.md`, figures; `L0_literature/`, `RESULTS.md`, `PROGRESS.md`. Campaigns: `scripts/selfplay/outputs/thesis-F*/`. Checkpoints and logs are git-ignored.

## 4. Experiments

Each entry: question, design, outputs, predictions (verbatim), amendments, status (RESULTS.md section). In ID order.

### Phase X: diagnostics

#### X0_replay
- **Question.** Are stored RL policies replayed correctly in the current env?
- **Design.** Re-evaluate 3 stored matchups per campaign (v6, v9, v10, v11) with 2048 episodes; compare with `matchups.json`. **Outputs.** `replay_check.json`, table of `|Delta|` per campaign.
- **Predictions.** v9/v10/v11 within 0.05. v6 (older obs layout) fails and is excluded
- **Status.** Done. RESULTS § X0_replay.

#### X1_crossplay
- **Question.** Is the game transitive? Do campaigns find different strategies?
- **Design.** Payoff matrix: ~20 pursuers (last 5 stages of v9/v10/v11 + PN/PP/MPC) x all valid evaders (~30 RL + 8 heuristics), 2048 eps each; Elo/Bradley-Terry fit; count 3-cycles. **Outputs.** `payoff_neut.npy`, `payoff_capture.npy`, `fig_payoff.png`, `transitivity.json`.
- **Predictions.** Mostly transitive (Elo R² 0.8-0.9), with real cross-campaign failures (some final pursuer < 0.6 vs another campaign's evader)
- **Status.** Done. RESULTS § X1_crossplay.

#### X2_descriptors
- **Question.** How redundant are self-play pools?
- **Design.** Descriptor per evader: (a) response vector vs probe set {PN, PP, MPC, v9/v10/v11 final pursuers}; (b) trajectory statistics (mean/std of speed, altitude, lateral acceleration, distance to arena centre, bearing to pursuer, dominant jink frequency). k-medoids + silhouette; DPP log-det per campaign. Reuses X1 rollouts. **Outputs.** `descriptors.parquet`, `fig_umap.png`, `clusters.json`, `redundancy.json`.
- **Predictions.** **H1a:** consecutive stages are near-duplicates; 4-7 clusters across ~38 evaders; each campaign covers 2-3 clusters
- **Amendments.** 10-26: clustering searches k = 2..8 (fixed before seeing X1/X2 results).
- **Status.** Done. RESULTS § X2_descriptors.

#### X3_gap
- **Question.** Do unseen evaders exist that the generalist loses to but a specialist beats (H2a)?
- **Design.** Generalist `pi_G` = M0, 150M frames on the train split (§5). Test evaders: held-out campaign evaders, switch heuristics, **exploiters** (evader PPO best responses to frozen `pi_G`, 3 seeds x 50M). Specialist = `pi_G` fine-tuned 20M frames per test evader. **Outputs.** `gap.json`, `fig_gap.png`.
- **Predictions.** Exploiters: `Delta` >= 0.25 for >= 2 of 3. Held-out RL evaders: 0.05-0.2. Heuristics: < 0.05 (as in P0)
- **Amendments.** 10-26: S-U, M0 and the X3 generalist are the same runs; seeds 3-4 added (`X3 --seeds 5`).
- **Status.** Done. RESULTS § X3_gap.

#### X4_oracles
- **Question.** Is opponent information worth anything (H2b)?
- **Design.** Privileged-obs pursuers, 3 seeds x 150M: **O1** evader descriptor `b_i`; **O2** evader true world acceleration (intent); **O3** evader relative position at +0.3 s (offline only). Eval on train, held-out, X3 exploiters. **Outputs.** `oracle.json`, `fig_oracle.png`.
- **Predictions.** O2 (intent) helps most (+0.05-0.15 on exploiters / switches), O1 helps little in-distribution (< 0.03), O3 sets the target horizon for NF-OA
- **Amendments.** 10-26: O3 dropped (needs a delayed-observation mechanism; A1a answers it); O2 split into O2a (measured acceleration, doubles as the ESO baseline) and O2b (latest command).
- **Status.** Done. RESULTS § X4_oracles (with gate G1).

**Gate G1.** If at least one oracle closes at least 30 % of `Delta` on the X3 evaders, run Phase A as planned. If no oracle helps, or no evader has `Delta >= 0.15`, Phase A shrinks to A1 with M0 / ESO / ESN / NF-OA only and reports a characterised negative result; the thesis weight moves to Phase D.

### Phase L: literature

#### L0 (gate G2)
- **Task.** Read AgilePE, Roncero 2026, Gavin 2026, Perez-Nieves 2021, Liu 2021 / 2022, Yao 2023 (PSD-PSRO), Balduzzi 2019, Neural-Fly 2022, RMA 2021, GSCU 2022, BPR 2016, Al-Shedivat 2018, ESN-OA; novelty search. **Output.** `L0_literature/notes.md`; gate G2 confirms or swaps QD-DPP / NF-OA / BayesID.
- **Status.** Done (BayesID dropped to P2). RESULTS § L0 literature / gate G2.

### Phase D: diversity-aware sampling

Pool split (§5) fixed before any D-run.

#### D1_sampler
- **Question.** H1c: does QD-DPP give more robust pursuers?
- **Design.** One pursuer per sampler {S-U, S-W, S-M, QD-DPP} on the **fixed** train pool, 150M frames, 3 seeds (5 if time); sampler refreshed every 25M frames. **Outputs.** `train_curves.json`, `eval_heldout.json`, `fig_sampler_bars.png`.
- **Predictions.** QD-DPP >= S-M on held-out worst-cluster by 0.03-0.08; equal mean in-distribution (+/-0.02); S-U worst on hard evaders
- **Amendments.** 10-26: refresh from the episodes played inside one run (no separate evals); 5 seeds (after X3's seed variance, before any 3-seed result); the D1 "exploiter" column is biased against S-U, D2 is the fair comparison.
- **Status.** Done (5 seeds). RESULTS § D1_sampler.

#### D1b_duplicates
- **Question.** H1b: robustness to duplicated opponents.
- **Design.** As D1, one cluster duplicated 5x; S-M and QD-DPP, 3 seeds. **Outputs.** `dup.json`, `fig_dup.png`.
- **Predictions.** S-M drops 0.05-0.10 on non-duplicated clusters, QD-DPP drops < 0.02
- **Status.** Done. RESULTS § D1b_duplicates.

#### D1c_nofloor (exploratory; added 10-27 after seeing D1b)
- **Question.** Is the DPP mechanism duplicate-robust without the uniform floor (D1b's 0.5 floor is not)?
- **Design.** S-M vs QD-DPP, `uniform_mix = 0`, normal and duplicated pool, 3 seeds. **Outputs.** `D1c_nofloor/summary.md`.
- **Predictions.** none (exploratory)
- **Amendments.** 10-27: interrupted after 2 of 12 runs (kept), requeued after A2.
- **Status.** Done. RESULTS § D1c_nofloor.

#### D2_exploitability
- **Question.** Are QD-DPP pursuers harder to exploit?
- **Design.** Per D1 final pursuer, a fresh exploiter evader, 50M frames. Exploitability = 1 - `J`. **Outputs.** `exploit.json`, `fig_exploit.png`.
- **Predictions.** QD-DPP lowest exploitability, but the difference vs S-M may be within the CI
- **Status.** Done (5 seeds). RESULTS § D2_exploitability.

#### D3_campaign (P1)
- **Question.** Does QD-DPP help in full self-play?
- **Design.** One `sampler: qd_dpp` campaign, 9 stages x 300M, as `v11-uniform-mix` (baseline); cross-play + one exploiter each. **Outputs.** `campaign_compare.json`, per-stage log-det.
- **Predictions.** QD-DPP evader pool has higher log-det (more clusters); final pursuer at least as strong in cross-play
- **Status.** Dropped 10-28, replaced by F2b / F3.

### Phase A: online adaptation

Settings: **S1** within-episode (state reset per episode); **S2** across-episode (30 consecutive episodes vs the same evader, area under `J` vs episode index); **S3** switch (evader switches at t = 0.5 s; recovery time).

#### A1a_basis
- **Question.** Can a learned basis predict the evader better than a reservoir?
- **Design.** 50k episodes of `pi_G` vs train evaders; offline open-loop error of CV, ESN+RLS, raw-history RLS, NF frozen `theta_bar`, NF+RLS; horizons {0.1, 0.3, 0.5} s; train and held-out. **Outputs.** `pred_err.json`, error-vs-horizon / time figures.
- **Predictions.** NF+RLS < ESN+RLS by 30-50 % at 0.3 s on held-out; NF frozen in between; CV good at 0.1 s
- **Amendments.** 10-27: bounded RLS covariance, (delta, forgetting) tuned on 20 % held-out train-pool envs, used by A1b; the defaults (delta 10, forgetting 0.98) diverged (run 1 kept).
- **Status.** Done. RESULTS § A1a_basis.

#### A1b_policy
- **Question.** H2c/H2d: does it improve interception?
- **Design.** Pursuers {M0, M0h, ESO, ESN, NF-frozen, NF-OA}, train pool, 150M, 3 seeds; S1/S2/S3 on train, held-out, X3 exploiters. **Outputs.** `adapt_eval.json`, gap-closed / S2 / S3 figures, latency table.
- **Predictions.** NF-OA closes 20-40 % of `Delta` on exploiters in S2, about 0 in-distribution; NF-frozen gets about half of that; ESN about 0 (as in P0). Within-episode (S1) gains small because episodes are short
- **Amendments.** 10-26: M0 and O2a reused, GRU only if time; 5 seeds.
- **Status.** Done (5 seeds). RESULTS § A1b_policy.

#### A2a_nfacc (added 10-27, user choice B)
- **Question.** Can the learned basis + RLS estimate the evader's *acceleration* better than a finite difference?
- **Design.** NF/RLS with target = evader velocity change over the next H steps, offline on A1a data; H in {1, 5, 10}; RLS tuned on validation envs; vs finite-difference persistence. **Outputs.** `A2a_nfacc/pred_err.json`, `summary.md`.
- **Predictions.** Learned predictor beats finite-difference persistence for H >= 5; small gain from RLS
- **Status.** Done. RESULTS § A2a_nfacc.

#### A2b_nfacc (added 10-27)
- **Question.** Does a learned acceleration estimate help the pursuer as much as the finite-difference input (O2a)?
- **Design.** Pursuers with the NF acceleration estimate (scaled like O2a), frozen and with RLS, 3 seeds, noise-free; S1/S2. **Outputs.** `A2b_nfacc/summary.md`.
- **Predictions.** Close to O2a (within 0.05) noise-free; RLS gain small
- **Status.** Done (5 seeds). RESULTS § A2b_nfacc.

#### A2_bayesid
- **Status.** Dropped to P2 at gate G2 (BPR / GSCU already cover it).

#### A3a_posonly (added 10-28, user request)
- **Question.** Without the evader's velocity, does a learned estimator of its acceleration beat finite differences of positions? Clean and noisy inputs.
- **Design.** New generalist trajectories with own position; window [rel_pos, own_pos, own_vel], target = evader velocity change over H in {5, 10}; frozen basis, trained clean and noise-augmented; baselines zero, second-difference persistence, raw and EMA (alpha tuned on val); noise = `observation.noise` stds. Rule: H* = H with the largest error reduction vs FD persistence on clean val (ties -> 10); use the augmented basis if its noisy-val error is lower and its clean-val error <= 1.1x the clean basis's. **Outputs.** `A3a_posonly/summary.md`.
- **Predictions.** Learned beats FD clearly with noise, roughly equal without
- **Amendments.** 10-28: re-run in float32, same rule (run 1 kept as `A3a_posonly/run1_fp16/`): float16 positions (~2 mm steps) made the clean second-difference baseline worse than zero; found before any A3b run used it.
- **Status.** Done. RESULTS § A3a_posonly.

#### A3b_posonly
- **Question.** Same question in closed loop.
- **Design.** No evader velocity in the obs, train pool, uniform, 150M, noise-free training, 3 seeds: P_none, P_fd (second-difference acceleration), P_nf (frozen learned estimator); eval noise-free and noisy (P_fd also with EMA). **Outputs.** `A3b_posonly/summary.md`.
- **Predictions.** P_fd ~ P_nf noise-free; P_nf > P_fd with noise; both > P_none
- **Status.** Done. RESULTS § A3b_posonly.

#### A3c_noise_ref
- **Question.** How do the velocity-based policies degrade under sensor noise?
- **Design.** Eval only, `observation.include_noise=true`: M0, finite-difference acceleration input (F1 SU_acc), NF-acc frozen and + RLS (A2b). **Outputs.** `A3c_noise_ref/summary.md`.
- **Predictions.** Finite-difference acceleration degrades most
- **Status.** Done. RESULTS § A3c_noise_ref.

#### A3d_velest (added 10-29, user choice after A3b/A3c; pre-registered 10-29 ~11:00, before the offline run started 11:03; pushed 11:05)
- **Question.** Can a learned estimator recover the evader's *velocity* (and acceleration) from noisy positions well enough to replace a measured velocity?
- **Design.** Offline: A3a data, window K = 10, target = evader's current velocity, clean / augmented basis (A3a rule); baselines first difference, raw and EMA; reference A3c's measured velocity error. Closed loop: train pool, uniform, 150M, noise-free training, 3 seeds, the estimate replaces the true relative velocity: **V_ema** (tuned EMA difference) vs **V_nf** (learned velocity + A3a learned acceleration); eval clean / noisy; refs M0 (A3c), P_none (A3b). **Outputs.** `A3d_velest/summary_offline.md`, `summary.md`.
- **Predictions.** Offline: NF_aug < FD_ema on noisy val/test; noise-free FD_raw best. Closed loop, noisy eval: V_nf > V_ema on train and held-out RL; both far above P_none (0.29); both below M0 noisy (0.70)
- **Status.** Done. RESULTS § A3d_velest.

#### A3e_estimators (added 10-29, user request; pre-registered 10-29 20:19 before any A3e run; offline only)
- **Question.** Is the 10-step MLP window the right estimator, or do GRU / LSTM (memory over the episode) or a Kalman filter estimate the evader's velocity and 0.2 s-ahead velocity change better from noisy positions?
- **Design.** A3a / A3d data, split, noise and metric (MLP / FD / FD_ema numbers reused). GRU (hidden 96), LSTM (hidden 80), ~30k parameters like the MLP (32k), clean / aug training, truncated BPTT 100, 8 epochs. KF: per-axis constant-acceleration filter on own_pos + rel_pos, R = 0.02^2 + 0.02^2 m^2, white-jerk q tuned on noisy val (grid 0.1 ... 1e5). No closed loop. **Outputs.** `A3e_estimators/summary.md`, `pred_err.json`.
- **Predictions.** (1) Noise-free velocity: KF < MLP (the KF is near exact like FD). (2) Noisy velocity: best of GRU / LSTM (aug) beats MLP_aug by >= 10 %; KF within +/- 20 % of MLP_aug; KF and the recurrent nets all beat FD_ema. (3) Noisy velocity change (acceleration): GRU / LSTM aug <= MLP_aug; KF worse than MLP_aug. Consequence: if any alternative beats the MLP used in the closed loop (velocity: MLP_clean, per the A3d rule; acceleration: MLP_aug) by >= 10 % on noisy test, this is reported as a limitation of A3d / F4 (closed loop = MLP vs EMA); no rerun before the freeze
- **Status.** Done. RESULTS § A3e_estimators.

#### A3f_specnorm (added 11-03, user request after Pan et al. RA-L 2026; pre-registered 11-03 ~14:25 before any A3f run (smoke 14:27); offline only)
- **Question.** Does a spectral-norm penalty on the estimator (bounding the Lipschitz constant, after Neural Lander / Bartlett 2017) make it more robust to sensor noise and held-out evaders?
- **Design.** A3a / A3d data and metric, same MLP (32k parameters), 8 epochs. Loss + beta * sum_l ||W_l||_2^2 over the 3 basis layers and the head (exact SVD); control: Frobenius norm (weight decay). beta in {1e-4, 1e-3, 1e-2}, chosen on val in the basis' own training condition. Reference: unregularised A3a / A3d models; Lipschitz bound prod_l ||W_l||_2 reported. **Outputs.** `A3f_specnorm/summary.md`, `pred_err.json`.
- **Predictions.** (1) The spectral penalty lowers the Lipschitz bound by >= 2x at beta*. (2) Clean basis, noisy test: spec beats the unregularised clean MLP by >= 10 % (bounded noise amplification), for both targets. (3) Aug basis: spec within +/- 5 % of the unregularised aug MLP on noisy test (noise augmentation already regularises). (4) Clean test (held-out evaders): spec within +/- 5 % of unregularised. (5) spec <= frob on noisy test for the clean basis. Consequence: if spec beats the estimator used in closed loop (acc: MLP_aug, vel: MLP_clean) by >= 10 % on noisy test, it is reported as a candidate improvement for A3d / F4; no closed-loop rerun before the freeze
- **Status.** Done. RESULTS § A3f_specnorm.

### Phase F: final system and self-play

Conventions: "plain" = plain observation; "Hedge" = Roncero eq. (12) exactly, gamma 0.7, eta 2; an exploiter is a fresh evader trained against one final pursuer.

#### F1_final (redesigned 10-27, user choice B)
- **Question.** Final comparison: sampler x opponent-acceleration input.
- **Design.** 2x2 {uniform, QD-DPP} x {plain obs, opponent-acceleration input}, 5 seeds, noise-free; reuses X3, D1, X4 O2a; a QD-DPP + learned-acceleration cell only if A2b beats O2a; main effects and interaction (seed-paired). **Outputs.** `F1_final/summary.md`, `results.json`.
- **Predictions.** - (none recorded)
- **Status.** Done. RESULTS § F1_final.

#### F2a_exploit_acc (added 10-28, user request, pre-registered 19:30 before A3b/A3c/D1c results)
- **Question.** Is the acceleration-input pursuer less exploitable?
- **Design.** One fresh plain-obs exploiter (seed 200 + s, as in D2) per F1 `SU + acc` pursuer, seeds 0-4; seed-paired vs D2 `S-U`. **Outputs.** `F2a_exploit_acc/summary.md`.
- **Predictions.** Lower than plain (D2 S-U 0.672), mean 0.3-0.5, paired CI excludes 0; still clearly > 0
- **Amendments.** 10-29: analysis rebuilt from the eval files (3 of 5 seeds had been shown); no rerun.
- **Status.** Done (5 seeds). RESULTS § F2a_exploit_acc.

#### F2b_campaigns, F2c_final_eval (same pre-registration as F2a; replace D3)
- **Question.** F2b: does the acceleration input help in co-evolution, the reference papers' protocol? F2c: final comparison.
- **Design.** Two campaigns differing only in the input: **acc** (both roles, frozen opponents too, observe the opponent's finite-difference acceleration) vs **plain**; uniform sampling, 9 stages x 300M, pursuer first, warm start, seed 0, scripted seeds pursuer {pure_pursuit, apf}, evader {hover, circular, flee, weave}; noise-free; 1 seed per arm, descriptive. F2c: cross-play of all RL pursuers vs all RL evaders; final pursuers vs train pool and test set (v10, held-out scripted, X3 exploiters); one plain-obs exploiter (100M frames) per final pursuer. Method choice: uniform + finite-difference acceleration (the learned A2b estimator is noise-free-equivalent and has no evader-role counterpart). **Outputs.** `thesis-F2-{acc,plain}` DBs, `F2c_final_eval/summary.md`.
- **Predictions.** F2b: -. F2c: acc final pursuer >= plain final pursuer + 0.1 J on the other campaign's evaders and on the test set, and lower exploitability. Within-campaign J need not rise, since the acc evaders also observe the pursuer's acceleration
- **Status.** Done. RESULTS § F2b_campaigns / F2c_final_eval.

#### F3_sampler_sp (added 10-28, user request, pre-registered 20:30 before any F2/F3 result)
- **Question.** RQ1 under the reference papers' protocol: do diversity-aware or hardness-prioritised opponent samplers beat uniform in full self-play?
- **Design.** Campaigns as F2b plain (9 stages x 300M, multi-episode evals with behaviour statistics); only the pursuer's sampler changes: **uniform**; **hedge** (challenge c = 1 - J of the latest pursuer); **qd_dpp** = Hedge quality re-weighted by a DPP over evader behaviour descriptors (mean/std of the 9 per-step behaviour features, z-normalised over the pool; trajectory block only). 3 seeds (campaign seed 0/100/200; uniform seed 0 = F2b plain). F3c: final pursuers vs train pool, test set, all 36 campaign evaders, one exploiter. Primary: held-out RL J, worst held-out cluster, J vs other campaigns' evaders, exploitability; seed-paired qd_dpp - hedge, qd_dpp - uniform, hedge - uniform. **Outputs.** `F3_sampler_sp/summary.md`.
- **Predictions.** Roncero predicts hedge > uniform; D1 predicts hedge <= uniform and qd_dpp ~ uniform. With gamma = 0.7 the three sampling distributions are close (70 % uniform), so all paired differences are expected within +/- 0.05 and mostly not significant with 3 seeds
- **Amendments** (user, all before any F3 result):
  1. 10-29 19:40: + arm **pfsp** = Hedge with gamma 0, eta 2 (p_j ~ (1 - J_j)^2; Gavin's 50 % latest-opponent share not included); seed 2 dropped, **2 seeds per sampler**; extra pairs pfsp - uniform, qd_dpp - pfsp; descriptive. Prediction for pfsp (from D1 / D1c): pfsp <= uniform on held-out RL and worst held-out cluster, the highest max sampling probability of the four samplers, and qd_dpp >= pfsp.
  2. 10-29 20:13: + metric **evader_vendi** (Vendi score of each campaign's 4 RL evaders in one shared descriptor space, from the F3c cross-evaluations), testing AgilePE's claim that PFSP self-play sustains tactical diversity. No directional prediction from our data (the evader side samples uniformly in every arm, so the pursuer sampler acts only indirectly); AgilePE's claim implies pfsp >= uniform. Max value 4 per campaign, 2 seeds: descriptive
  3. 11-03 ~14:30: sampler on **both roles**, as in the reference papers (evader-side challenge of pursuer j = J(pursuer j, latest evader)); qd_dpp descriptors of pursuers = the same 9 features with the pursuer as subject; new `thesis-F3-{hedge,qd_dpp,pfsp}-both-s{0,1}`; pursuer-only campaigns kept, not analysed; seed 1 only if started before 11-04 08:00. Predictions unchanged (all differences within +/- 0.05 of uniform; pfsp <= uniform; qd_dpp >= pfsp); evader_vendi now tests AgilePE's PFSP-diversity claim directly (pfsp >= uniform).
  4. 11-04 14:45: 2 seeds for every arm (seed 100 added for qd_dpp and pfsp); F3c waits for all 8 campaigns. Predictions unchanged.
  5. 11-04 15:25: every outcome metric also with capture only (`<metric>_cap`, secondary); J stays primary. No prediction for the capture-only columns
  6. 11-04 21:11: stage-1 evaders are byte-identical across same-seed campaigns, so other / own campaign evader metrics exclude them (3 per campaign remain); evader_vendi keeps all 4. Predictions unchanged.
- **Status.** Done (2 seeds per sampler). RESULTS § F3_sampler_sp / F3c.

#### F4_estimator_sp v1 (superseded by v2 before any F4 campaign ran; added 10-28, user request, pre-registered 21:30 before A3b/A3c/F2/F3 results)
- **Question.** RQ2 under realistic sensing in self-play: without the evader's velocity and with noisy sensors, does the learned acceleration estimator help?
- **Design.** Uniform campaigns, 9 stages x 300M, campaign seeds 0/100/200; pursuer without evader velocity, sensor noise in training and eval; arms **nf** (+ A3a estimator, H = 10) vs **none**; evader full clean state. F4c: final pursuers vs train pool, test set (noisy and clean), all 6 F4 campaigns' evaders, one exploiter; primary seed-paired nf - none.
- **Predictions.** nf > none by >= 0.1 on held-out RL and X3 exploiters, CI excluding 0 (A3a: only the learned estimator carries acceleration information under noise); exploitability lower for nf
- **Status.** Superseded (only the 3-stage v1 smoke ran).

#### F4_estimator_sp v2 (redesign 10-29 ~11:00, user choice, before any F4 campaign; pushed 11:05, commit 55bbbac)
- **Question.** Same question with a velocity estimate (A3b: without the evader velocity every pursuer is weak, ~0.29, so v1 would compare two crippled pursuers).
- **Design.** As v1, but both arms observe an *estimated* evader velocity: **nf** = A3d V_nf inputs vs **ema** = A3d V_ema inputs; frozen pursuers seen by the evader use clean inputs. Primary: seed-paired nf - ema. **Outputs.** `F4_estimator_sp/summary.md`.
- **Predictions.** nf > ema on held-out RL and X3 exploiters by >= 0.05; exploitability lower for nf. Low confidence (3 seeds, no fixed-pool result yet)
- **Amendments.** 11-02: analysis rebuilt (the driver used the v1 arms); no data changed. 11-04: post-hoc `_no_s1` columns.
- **Status.** Done (3 seeds per arm). RESULTS § F4_estimator_sp v2.

#### F5_acc_sp (added 11-04, user request, pre-registered 11-04 17:40 before any F5 run; freeze moved to 11-08)
- **Question.** Exp 1: does the learned acceleration estimate (velocity observed) help the pursuer in self-play vs no acceleration input? Exp 2: the thesis pursuer (QD-DPP + learned acceleration + RLS) vs the reference papers' pursuers (uniform / Hedge / PFSP, plain). Clean and noisy; does online RLS matter at deployment?
- **Design.** New campaigns (F3 protocol, clean training, seeds 0 / 100): pursuer gets the A2b estimate + tuned RLS (`NFacc_OA`) on top of the plain observation; evader plain. Arms **uniform_rls** (Exp 1) and **qd_dpp_rls** (QD-DPP both roles; Exp 2), 2 seeds each. Reused: uniform_plain = F2-plain + F3-uniform-s1; hedge / pfsp / qd_dpp_plain = F3 `-both`; uniform_fd = F2-acc (1 seed). F5c: final pursuers (stage 8) vs train pool, test set and the RL evaders of the 12 plain-evader campaigns, clean and noisy (A3c noise, eval only); one 100M-frame exploiter per F5 pursuer (existing F3c / F2c exploiters for reused pursuers); **RLS ablation (B1)**: RLS frozen at the mean head at eval. Metrics as F3. Contrasts: uniform_rls - uniform_plain; qd_dpp_rls - {uniform, hedge, pfsp, qd_dpp}_plain and - uniform_rls; RLS on - frozen. 2 seeds: descriptive. **Outputs.** `F5_acc_sp/summary.md`, `results.json`.
- **Predictions.** (1) Clean, Exp 1: uniform_rls - uniform_plain >= +0.05 on held-out RL (A2b +0.20 on the fixed pool; F2b finite difference +0.12 in self-play), but no gain on other campaigns' evaders (F2b: -0.12) and exploitability within +/- 0.05 (F2b: both ~0.97). (2) Noisy: plain pursuers lose <= 0.05 (A3c M0 -0.03); uniform_fd falls below uniform_plain (A3c finite difference -0.37); uniform_rls stays >= uniform_plain on held-out RL (A3c +0.12). (3) Exp 2: no sampler effect, qd_dpp_rls - uniform_rls within +/- 0.05 (D1, F1, F3 in-campaign data), so qd_dpp_rls - {uniform, hedge}_plain ~ the Exp 1 acceleration effect, and qd_dpp_rls - pfsp_plain larger (F3 in-campaign data: pfsp weaker). (4) RLS: clean, rls - frozen-at-eval within +/- 0.03 (A2b); noisy, rls > frozen-at-eval by >= 0.03 on the train pool and lower exploitability (A3c, there with separately trained policies). Low confidence throughout.
- **Amendments** (user):
  1. 11-04 21:11 (before any F5 run): estimator **retrained** on self-play evaders F5c never evaluates: a new plain generalist (X3 settings, seed 0) trained on the train pool + the 7 unique RL evaders of the F4 seed-2 campaigns records trajectories vs them and the 4 F2-acc evaders (no checkpoint shared with any evaluation evader; file hash); A2a rules re-applied (ties -> 5); output `F5_acc_sp/estimator/`; stage-1 evaders excluded. Predictions unchanged; they were derived from A2b's estimator and now concern the retrained one.
  2. 11-06 12:45 (decided **after** the 2-seed Exp 1 results were seen): third Exp 1 seed pair, seed **300** (`thesis-F5-uniform_rls-s2`, `thesis-F5-uniform_plain-s2`), each with a fresh 100M-frame exploiter; all Exp 1 pursuers also evaluated vs their evaders (`*_campaign_evaders_s300.json`). All Exp 1 contrasts and predictions are then reported with 3 seed pairs (uniform_fd stays 1 seed), whatever they show; the 2-seed analysis is kept unchanged as `summary_exp1_2seeds.md`. With 3 pairs the bracket is a bootstrap CI over 3 values (coarse); a difference whose CI excludes 0 is reported as supported with that caveat. Disclosure: adding seeds after seeing results risks optional stopping; no further Exp 1 seeds will be added, whatever this pair shows. Exp 2 unchanged (paused)
- **Other changes.** 11-05 14:38: Exp 2 paused (user); F5c evaluates Exp 1 alone.
- **Status.** Exp 1 done (3 seed pairs), RESULTS § F5_acc_sp Exp 1 / F5c. Exp 2 paused.

#### F6_div_sp (added 11-05, user request after the post-hoc F3 check; pre-registered 11-05 ~16:00 before any F6 run; exploratory, designed after seeing F3c)
- **Question.** Can QD-DPP self-play produce more diverse evaders while its pursuer stays at least as good as Hedge (Roncero)?
- **Design.** Arm **qd_dpp_div** = F3 qd_dpp (both roles, plain, seeds 0 / 100) + PSD-PSRO's diversity regulariser (Yao et al. NeurIPS 2023) for evaders only: each evader after the first maximises beta * min(min_j d_j(s), cap) as a loss term and a per-step intrinsic reward, j = earlier RL evaders of its campaign, d_j = mean-only KL 0.5 sum_k ((mu - mu_j)/sigma_j)^2, cap = 2. Beta pilot (training data only) from F3 qd_dpp-both-s0 evader stage 5, 100M frames per beta in {0, 0.1, 1.0}: the largest beta with J vs its training pursuers <= J(beta=0) + 0.05 and final mean min-distance >= cap/2 (else the largest meeting the J condition); beta is not changed after F6c. F6c = F3c for each final pursuer + the 8 F3 final pursuers vs the F6 evaders; evader_vendi in F3's frozen descriptor space; seed-paired vs {hedge, qd_dpp, uniform, pfsp}; 2 seeds: descriptive. **Outputs.** `F6_div_sp/summary.md`, `results.json`, `pilot/choice.json`.
- **Predictions.** (1) Mechanism: mean div_distance >= cap/2 at the end of evader stages 3, 5, 7. (2) Diversity: evader_vendi qd_dpp_div > qd_dpp (2.350) and > hedge (2.499) on both seeds, >= +0.3 vs qd_dpp. (3) Pursuer (user target, non-inferiority): qd_dpp_div - hedge >= -0.05 on held-out RL and worst held-out cluster; qd_dpp_div - qd_dpp >= 0 on held-out RL and other campaigns' evaders. (4) Exploitability unchanged (0.97-0.98, as every F3 sampler). Not predicted: superiority over Hedge (2 seeds cannot show it; F3 run-to-run spread up to 0.29). Low confidence throughout; results reported whatever they are
- **Amendments.** 1. 11-05 16:12 (user, before any F6 run): sampler **Hedge** (both roles) instead of QD-DPP, so the arm differs from F3 hedge only by the regulariser (QD-DPP sampled like Hedge anyway, TV 0.010); arm **hedge_div** (`thesis-F6-hedge_div-s{0,1}`), pilot from F3 hedge-both-s0; primary contrast **hedge_div - hedge**. Predictions restated: (1) unchanged; (2) evader_vendi hedge_div > hedge (2.499) and > qd_dpp (2.350) on both seeds, >= +0.3 vs hedge; (3) hedge_div - hedge >= -0.05 on held-out RL and worst held-out cluster, >= 0 on other campaigns' evaders; (4) unchanged. Consequence: the thesis method no longer contains the DPP sampler; whether F5 Exp 2 is rerun with Hedge + regulariser is decided after F6
- **Status.** Done. RESULTS § F6_div_sp / F6c.

#### F6b_div_sp (added 11-08, user request after F6c / F7c; pre-registered 2026-11-08 15:40 before any F6b run; exploratory)
- **Question.** Does the F6 diversity term, redesigned so it neither saturates nor pays a per-step survival bonus, give more diverse evaders while the pursuer stays as good as under Hedge?
- **Design.** Arm **hedge_divb** = F6 hedge_div with only the term changed: d_j(s) = RMS over the 4 action dims of mu(s) - mu_j(s); bonus = 1 - exp(-d / tau), d = min_j d_j; intrinsic reward beta * bonus with its batch mean subtracted. Pilot (training data only; F3 hedge-both-s0 evader stage 5, seed 900, 100M frames): (i) beta = 0, tau = its mean d over the last 10 updates; (ii) beta in {0.03, 0.1, 0.3}. Keep betas with J <= J(0) + 0.05 and mean timeout rate <= timeout(0) + 0.10; choose the largest kept beta that is active (final mean d >= 1.25 tau and final mean bonus <= 0.9), else the largest kept; if none is kept, F6b stops. F6bc = F6c (exploiter seeds 650 / 651) + the 8 F3 final pursuers vs the F6b evaders. Primary contrast **hedge_divb - hedge**; secondary vs hedge_div and the other F3 arms. J at 0.10 m primary, also at 0.15 m. 2 seeds: descriptive. **Outputs.** `F6b_div_sp/summary.md`, `results.json`, `vs_f6.json`, `pilot/choice.json`.
- **Predictions.** (1) Mechanism: at the end of evader stages 3 / 5 / 7 (last 10 updates) mean bonus <= 0.9 and mean d >= tau. (2) Win mode kept: in-campaign, evader stages 3 / 5 / 7 make the stage >= 2 pursuers crash in >= 0.79 of episodes (F2 / F3 / F5: 0.79-0.99; F6: 0.22-0.45, timeout 0.31-0.66). (3) Diversity: evader Vendi > hedge (2.499) on both seeds, mean difference >= +0.2. (4) Pursuer (non-inferiority, as F6 (3)): hedge_divb - hedge >= -0.05 on held-out RL and the worst held-out cluster, >= 0 on other campaigns' evaders; and hedge_divb - hedge_div > 0 on other campaigns' evaders on both seeds. (5) Exploitability unchanged (0.97-0.98). **Success = (3) and (4)** ("more diverse, as good as Hedge"). Low confidence: F3 (PFSP: more diverse, more specialised pursuer) and F6 suggest diversity does not by itself help the pursuer. **Consequence**: success -> the term is a thesis method candidate (F5 Exp 2 with it: user decides); otherwise RQ1 is answered negatively (diversity in sampling, D1 / F3, and in training, F6 / F6b, does not beat Hedge), Hedge is the sampler of the final setup, freeze (user decides). Starts now on both Isaac slots (nothing else running); freeze moves to ~11-10
- **Amendments.** 1. 11-09 01:30 (user, before any F6b result; v1 pilot values and campaign stages 0-4 seen): the beta = 0 run became unstable at its end, so v1's tau was 0.653 (~9x the typical value) and the fallback chose beta = 0.3. Now tau = median d over the last quarter of the beta = 0 run (= 0.0727); the pilot guard's reference = the stricter of the beta = 0 run and F3 hedge-both-s0's own stage-5 evader (J 0.184, timeout 0.134). v1 campaigns (stopped at stage 5) archived; betas re-run; rule, predictions and F6bc unchanged.
- **Status.** Done. RESULTS § F6b_div_sp / F6bc.

#### F7_floor_sp (added 11-06, user request after the F5 Exp 1 post-hoc outcome analysis; pre-registered 11-06 19:30 before any F7 run; exploratory, 1 seed per arm)
- **Question.** Do evaders beat self-play pursuers by luring them into the floor, and does (a) a pursuer floor penalty and (b) a 0.15 m capture radius in training remove that weakness?
- **Design.** Two campaigns, F3 uniform protocol (campaign seed 0), differing from `thesis-F2-plain` (baseline) only in: **floor_r15** = 0.15 m capture radius in training and in-campaign evals + pursuer floor penalty `task.reward_floor_weight=1.0` (0 at 0.6 m -> 1 at the 0.1 m floor); **r15** = the radius only. F7c: final pursuers of F2-plain, r15, floor_r15 evaluated at 0.10 m vs train pool, test set, F5c campaign evaders, both F7 campaigns' evaders, and one 100M-frame exploiter per pursuer trained under the 0.15 m rules; every J at **0.15 m** (primary; exact re-score) and 0.10 m. Primary: exploitability and pursuer-crash rate vs that exploiter, held-out RL, worst held-out cluster, other campaigns' evaders. Contrasts floor_r15 - r15, r15 - plain, floor_r15 - plain (descriptive). **Outputs.** `F7_floor_sp/summary.md`, `results.json`.
- **Predictions.** From the F5 post-hoc analysis (every evader win vs self-play pursuers is a pursuer crash; evader stages lure 0.79-0.99; pool-trained pursuers are not lured): (1) floor_r15: pursuer crash vs its exploiter < 0.50 (F5c plain vs its exploiter: 0.97) and exploitability at 0.15 m <= r15 - 0.10. (2) r15: its exploiter still wins mainly by pursuer crash (>= 0.5 of its wins), exploitability at 0.15 m within +/- 0.10 of F2-plain's vs its 0.15 m exploiter. (3) Pursuer crash rate vs held-out scripted evaders: floor_r15 < 0.05 (F5c uniform_plain 0.21). (4) Held-out RL J at 0.15 m: floor_r15 >= r15 - 0.05. (5) In-campaign: in floor_r15, evader stages 3 / 5 / 7 make the stage >= 2 pursuers crash in < 0.50 of episodes (F5 / F2 / F3: 0.79-0.99). **Decision rule** (user decides about the 11-08 freeze after F7c): if (1) holds and floor_r15's held-out RL and other-campaign J at 0.15 m are >= F2-plain's - 0.05, the floor penalty + 0.15 m radius are recommended for any later F6 / F5 Exp 2 runs; if only r15 lowers exploitability, the radius alone; otherwise the rules stay. Queued after F5 Exp 1 (amendment 2) finishes, before any further F6 / F5 Exp 2 runs (F6 campaigns were already running; F6c is evaluation only)
- **Status.** Done; the rules stay (0.10 m radius, no floor penalty). RESULTS § F7_floor_sp / F7c.

#### F8_roncero_sp (added 11-09, user request after the F7 write-up; pre-registered 2026-11-09 12:21 before any F8 run; exploratory)
- **Question.** Do evaders still beat self-play pursuers by luring them into the floor when the game uses Roncero et al. 2026's reward and episode endings (is the weakness a product of the thesis's rules, or of the pursuer)?
- **Design.** Campaigns `thesis-F8-roncero-s{0,100}`: our drone, observations, arena and 0.10 m radius; Hedge on both roles; **9 stages x 50M**; `reward_mode=roncero` = Roncero v3 Eqs. (8)-(11) with his coefficients (k_br 0.001, k_sm 0.002, k_b 10, k_c 10, k_app 0.25, k_s 0.0025, v_safe 2 m/s, alpha_close 3): -10 for own out-of-bounds, capture bonus scaled by the closing-speed factor, evader per-step survival, -10 if captured, nothing when the pursuer crashes; capture terminal, every crash / out-of-arena and the timeout are truncations (`algo.gae_cut_at_truncation=true`). Not copied: his drone, arena (4 x 5 x 2 m), domain randomisation. F8c: final pursuers under the thesis rules vs train pool, test set, F3 and F8 campaign evaders, plus two 100M-frame exploiters each, thesis rules (seeds 820 / 821) and Roncero rules (830 / 831); F3 hedge-both s0 / s1 also get a Roncero-rules exploiter (832 / 833). In-campaign lure rate = pursuer crash caused by evader stages 3 / 5 / 7 in the stage >= 2 pursuers. J at 0.10 m and 0.15 m; contrast vs hedge, same seed. **Outputs.** `F8_roncero_sp/summary.md`, `results.json`.
- **Predictions.** (1) In-campaign: F8 evader stages 3 / 5 / 7 make the stage >= 2 pursuers crash in < 0.50 of episodes on both seeds (F3 hedge: 0.95-0.99), because Roncero's evader gains nothing from a pursuer crash. (2) Competing explanations, decided by the thesis-rules exploiter: **H-rules** (the weakness is created by the thesis rules): F8 pursuers' crash rate vs it < 0.50 on both seeds; **H-pursuer** (Roncero's rules only hide it): >= 0.50. Predicted: H-pursuer (low confidence; F6's hedge_div pursuers, trained without lure pressure, crash 0.48-0.75 vs luring F3 evaders). (3) Under Roncero's rules the F3 hedge pursuers' exploiter wins mostly by timeout, not pursuer crash (crash share of its wins < 0.5): his rules would not reveal the lure. No prediction for held-out J (budget confound). **Use**: H-rules -> report that the floor-lure exploit depends on the episode-ending rules (the reference papers would not see it); H-pursuer -> the low-thrust pursuer is lure-vulnerable whatever the rules, and his evaluation hides it. Queued after F6bc on two new lanes (both Isaac slots)
- **Amendments.**
  1. 11-09 12:28 (user, before any F8 run): **26 stages x 50M** (Roncero's 13 generations; 650M frames per agent; final pursuer = stage 24). Prediction (1) applies to evader stages 3 / 5 / 7 (comparable with F3's 9 stages) and is also reported for every later evader stage; (2)-(3) unchanged.
  2. 11-10 17:05 (loader bug): F8c had evaluated pursuer stage 8 and evader stages 1-7 only; re-run pending.
  3. 11-10 19:25 (user, before any re-run result): k_sm was 0 in every Roncero-rules run (computed per substep); fixed; F8 re-run with the same design and seeds as `thesis-F8b-roncero-s{0,100}`, with all Roncero-rules exploiters retrained. Predictions unchanged. The k_sm = 0 campaigns are kept as a 'Roncero without smoothness' variant.
- **Status.** Done (`thesis-F8b-*`). RESULTS § F8_roncero_sp / F8c.

#### F8d_heldout (added 11-11, user request: v10 no longer separates the strong pursuers; pre-registered 2026-11-11 12:21 before any F8d eval; eval only, 2 seeds, descriptive)
- **Question.** Is the F8 pursuer still better than Hedge's against stronger held-out learned evaders, and how early does it overtake?
- **Design.** Pursuers: F8b stages 8 / 16 / 24 (250M / 450M / 650M pursuer frames) and the F3 hedge-both final pursuers (1.5B frames). Opponents (thesis rules, 0.10 m and 0.15 m): the test set and the final evaders of 20 campaigns, a 'strong' set of 16 that none of these pursuers trained with (F2-plain, F3 uniform-s1 / qd_dpp / pfsp, F6, F6b, F7 r15 / floor_r15, F9 x4) plus the F3 hedge and F8 finals (own campaign excluded). **Outputs.** `F8d_heldout/summary.md`, `results.json`.
- **Predictions.** (1) The strong set is harder than v10: strong-finals mean J < v10 held-out RL J for all four final pursuers (F8 s0 / s1 stage 24, hedge s0 / s1). (2) On the strong set the F8 final pursuer beats Hedge's by >= 0.05 on both seeds (low confidence). (3) F8 held-out RL and strong-finals J rise from stage 8 to 24 on both seeds, and F8 stage 8 is below Hedge's final pursuer on held-out RL on both seeds. Descriptive (2 seeds).
- **Status.** Done. RESULTS § F8_roncero_sp, subsection F8d.

#### F9_ablation_sp (added 11-10, user request after F8; pre-registered 2026-11-10 15:19 before any F9 run; exploratory, 1 seed)
- **Question.** Which part of Roncero et al.'s rules removes the floor-lure exploit that F8 removed (one change at a time on the thesis rules)?
- **Design.** Campaigns `thesis-F9-{control,no_timeout,beta,endings}-s0`, Hedge both roles, **13 stages x 50M**, seed 0, GAE truncation fix: **control** = thesis rules; **no_timeout** = timeout gives 0 to both roles (thesis: pursuer -100, evader +100); **beta** = capture reward x exp(-3 max(0, v_close - 2 m/s)); **endings** = every ending except capture becomes a truncation, rewards unchanged. Not tested: Roncero's weak distance shaping and reward magnitudes. F9c: final pursuers (stage 12) vs train pool, test set, F3 and F9 campaign evaders, one 100M-frame thesis-rules exploiter (seeds 840-843). Measures: (a) in-campaign lure rate (evader stages 3 / 5 / 7 / 9 / 11 vs stage >= 2 pursuers); (b) pursuer crash vs the exploiter. **Outputs.** `F9_ablation_sp/summary.md`, `results.json`.
- **Predictions.** Lure present = (a) >= 0.50 mean over evader stages 3-11, (b) >= 0.50. (1) control: (a) and (b) present (as F3). (2) endings: (a) absent (the evader gains nothing from a pursuer crash), (b) present (the pursuer's reward is unchanged; F6's lure-free pursuers were lurable). (3) no_timeout: (a) present (a pursuer crash still beats likely capture for the evader); (b) present (low confidence). (4) beta: (a) present; (b) absent (slower final approaches avoid dives; low confidence). **Use**: an arm with (b) absent identifies a pursuer-side cause of F8's lure-resistance; an arm with (a) absent an evader-side cause. If no single change gives (b) absent, the pursuer-side effect needs a combination or an untested component (shaping, schedule). 1 seed: descriptive
- **Status.** Done. RESULTS § F9_ablation_sp / F9c.

#### F10_gae_sp (added 11-11, user request after F9; pre-registered 2026-11-11 11:45 before any F10 run; exploratory, 1 seed)
- **Question.** Why is F9's control pursuer (13 x 50M, GAE fix) hardly lurable (exploiter crash 0.056) when F3's hedge pursuer (9 x 300M, no fix) is fully lurable (0.975): the GAE fix or the schedule?
- **Design.** `thesis-F10-nofix-s0` = F9 control with `algo.gae_cut_at_truncation` off. F10c as F9c (exploiter seed 850), measures (a) and (b). Reference: F9 control (same seed and schedule), F3 hedge s0. **Outputs.** `F10_gae_sp/summary.md`, `results.json`.
- **Predictions.** Decision on (b): **H-fix** = (b) >= 0.50 (without the fix the pursuer is lurable again, so the GAE fix causes F9 control's resistance); **H-schedule** = (b) < 0.50 (the 50M-frame / 13-stage schedule does). **Predicted: H-schedule (low confidence)**: the fix only changes the advantage at timeout steps (under the thesis rules the timeout is the only truncation), and final self-play pursuers almost never time out (F5: timeout 0.000 vs every group), while the schedule changes stage length 6x. Also predicted under H-schedule: (a) within +/- 0.15 of control's 0.36. 1 seed: descriptive; either outcome is reported.
- **Status.** Done. RESULTS § F10_gae_sp / F10c.
- **Amendment 1 (user, 11-13 11:45; pre-registered before any new F10 run): seeds 100 / 200 (`F10s` / `F10sc`).** Reason: F10 s0's final pursuer looks strong (strong finals mean 0.828, worst 0.658; eval-only addendum 11-13) but F13 showed large seed variance under the same schedule with the fix (exploitability 0.440 on s0 vs 0.915 / 0.741 on s100 / s200). Decided after seeing F10 s0 and F13; disclosed as such; no further F10 seeds afterwards. Runs before the F12 resume (user, 11-13), which F13's decision rule calls for next.
  - **Design.** Campaigns `thesis-F10-nofix-s{100,200}` = `thesis-F10-nofix-s0` with seeds 100 / 200 (thesis rules, `algo.gae_cut_at_truncation=false` set explicitly, Hedge both roles, 13 x 50M, warm start), so each is paired by seed with `thesis-F13-short-s{100,200}` (same with the fix); seed 0 pairs with F9 control. **F10sc**: final pursuers (stage 12) of all three F10 campaigns, clean under the thesis rules, vs train pool, test set, F3 campaigns' evaders, F9 campaigns' + own evaders (as F10c), F8d's 20 finals (16 strong), the F11 campaigns' evaders, the other F10 campaigns' evaders, a fresh 100M-frame thesis-rules exploiter without the fix (F10c recipe; seeds 920 / 921; s0 reuses F10c's), and all F8 campaign evaders under the thesis rules and under Roncero's rules (`f8_roncero_sp.RULES`). Existing s0 eval files are reused. J at 0.10 m (primary) and 0.15 m. **Outputs.** `F10_gae_sp/seeds_summary.md`, `seeds_results.json` (F10's `results.json` / `summary.md` unchanged).
  - **Predictions (new seeds 100 / 200).** (1) H-schedule replicates: pursuer crash vs the thesis-rules exploiter < 0.50 on both (F10 s0 0.031; low confidence: F13's s100 / s200 with the fix gave 0.925 / 0.751). (2) Strong-finals mean >= 0.80 on both (s0 0.828; low confidence). (3) Held-out RL (v10) >= 0.90 on both (s0 0.933). (4) Under Roncero's rules, J vs the F8 evaders is below F8's own reference (0.824-0.843) on both. (5) GAE fix effect, means over the three seed pairs F10 vs F9 control / F13: |exploitability_thesis difference| < 0.15 and |strong-finals mean difference| < 0.10 (H-schedule; low confidence). Descriptive (3 seeds, no CIs).
  - **Status.** Running (11-13, lanes U / V).

#### F11_v2_sp (added 11-11, user decision after F8-F10; pre-registered 2026-11-11 17:06 before any F11 run; 2 seeds, descriptive)
- **Question.** Does self-play under revised thesis rules ('thesis v2') give pursuers that are not lurable, generalise at least as well as F8 / F10 and are hard to exploit?
- **Design (v2 as registered).** Thesis rules with (1) Roncero-style endings (capture terminal; pursuer crash, evader crash and evader leaving are truncations; the pursuer may leave the arena freely); (2) no timeout terms for either role; other terminal rewards unchanged, no closing-speed factor; (3) potential-based shaping gamma Phi(s') - Phi(s), gamma 0.99, Phi = -k_app d + w_prec exp(-5 d), k_app 2.5 (was 7), w_prec 5 (was 10); smoothness 0.5, command rate 0.2; GAE fix; (4) Hedge both roles, **26 stages x 50M**. Pilot `thesis-F11p-s0` (13 x 50M; rule: shaping > 20 -> halve the weights; J vs the 4 scripted training evaders < 0.80 -> double them; regularisation > 20 -> halve smoothness / command rate), then seeds 0 / 100. F11c: final pursuers (stage 24) vs train pool, test set, F3 campaign evaders, F8d's 20 final evaders, both F11 campaigns' evaders, and two 100M-frame exploiters each: thesis v2 rules (seeds 860 / 861) and the old rules without the GAE fix (870 / 871). J at 0.10 m and 0.15 m. **Outputs.** `F11_v2_sp/summary.md`, `results.json`, `pilot_decision.json`.
- **Predictions.** Both seeds: (1) in-campaign lure rate (a) (mean over evader stages 3-11) < 0.10; (2) pursuer crash vs the old-rules exploiter < 0.10; (3) exploitability vs the old-rules exploiter <= 0.172 (F10; low confidence); (4) held-out RL >= 0.93; (5) strong-finals mean >= F8's (0.858 / 0.875) - 0.05. Descriptive (2 seeds)
- **Amendments and stops.**
  1. 11-11 17:18, amendment 1 (after the smoke, before any pilot result): the shaping-size clause is dropped (the undiscounted potential-based sum does not measure dominance; the discounted shaped return is bounded by ~21). Other clauses unchanged; pilot restarted.
  2. 11-11 19:55, stopped: the pilot never learned to chase (capture 0.00 at every stage, the pursuer drifts out of the arena): potential-based shaping pays +(1 - gamma) k_app d per step and nothing pulls the pursuer back. Two pilot-rule bugs (J vs scripted NaN; regularisation clause fired). Campaigns stopped in stage 0.
  3. 11-11 20:01, amendment 2 (user, before any v2b run): **v2b** = plain-difference shaping (approach 2.5 x delta d, precision 5 x delta exp(-5 d)), everything else as registered; **gate**: the stage-0 pursuer's capture rate vs the 4 scripted training evaders must be >= 0.80; pilot rule: double the shaping if the final pilot pursuer's J vs those evaders < 0.80, no other clause. Predictions (1)-(5) unchanged.
  4. 11-11 20:16, v2b gate failed (capture 0.001-0.003): shaping 2.5 / 5 too weak against the 0.5 / 0.2 penalties without a timeout penalty.
  5. 11-12 00:43, amendment 3 (user, before any v2c run): **v2c** = v2b with the original shaping (7 / 10); everything else unchanged.
  6. 11-12 00:58, v2c gate failed (0.781 < 0.80); diagnosis 01:48: one-stage reruns reproduce earlier stage-0 results exactly (no code or simulator regression); endings + no timeout terms together remove the pursuer's urgency (capture 0.18, timeouts 0.42).
  7. 11-12 09:16, amendment 4 (user, before any v2d run): **v2d** = v2c but only the evader's timeout bonus is removed; the pursuer keeps -100 at timeout (`task.ablation.no_timeout_evader`); everything else unchanged. Campaigns `thesis-F11v2d-*`.
- **Status.** Done (campaigns 11-12 20:06, F11c 21:33; RESULTS.md [F11](../experiments/thesis/RESULTS.md#f11), 11-13 00:10): predictions (1)-(4) did not hold, (5) held on s0 only.

#### F12_acc_v2_sp (added 11-12, user request; pre-registered 2026-11-12 13:43 before any F12 run; queued after F11; 2 seeds, descriptive)
- **Question.** Does the learned acceleration estimator still help the pursuer under the thesis v2d rules?
- **Design.** `thesis-F12-acc-s{0,100}` = F11 v2d (rules, pilot settings, Hedge, 26 x 50M, seeds) with F5's pursuer input (learned acceleration + online RLS); gate as F11. F12c: final pursuers vs train pool, test set, the default held-out set (v10 + F8d's 16 strong finals), F3 / F11 / F12 campaign evaders, two 100M-frame exploiters (v2d rules, seeds 880 / 881; old rules without the GAE fix, 890 / 891); RLS-frozen ablation. Paired by seed with `thesis-F11v2d-s{0,100}`. **Outputs.** `F12_acc_v2_sp/summary.md`, `results.json`.
- **Predictions.** (1) acc - plain on the default held-out set >= +0.03 on both seeds (low confidence; F5: +0.091 on v10 under the old rules, F11 near the v10 ceiling). (2) Exploitability difference vs the old-rules exploiter within +/- 0.10. (3) RLS frozen vs on within +/- 0.02 (F5). Descriptive (2 seeds)
- **Status.** Stopped 11-13 01:14 (user) after 10 of 26 stages per campaign, to run F13 first (resumable). Resume if v2d beats F13's setup (F13 decision rule); otherwise F12 is re-run on F13's setup (new entry).

#### F13_thesis_short_sp (added 11-13, user decision after F11; pre-registered 2026-11-13 01:16 before any F13 run; 3 seeds, descriptive)
- **Question.** Is the original thesis rule set with short stages (thesis rules, GAE fix, Hedge both roles, 13 x 50M) a robust protocol on more seeds, and does thesis v2d (F11) beat it?
- **Design.** Campaigns `thesis-F13-short-s{100,200}` = F9 control exactly (thesis rules from the task YAML, `algo.gae_cut_at_truncation=true`, Hedge both roles, 13 stages x 50M, warm start, the same seed heuristics); seed 0 is `thesis-F9-control-s0` (same configuration, not re-run). No smoke or gate (same code path as F9 control and F11). **F13c**: final pursuers (stage 12) of the three campaigns, clean under the thesis rules, vs train pool, test set (v10 held-out RL, held-out scripted, X3 exploiters), F3 campaigns' evaders, F8d's 16 strong finals and the F3 hedge / F8 finals, the F13 campaigns' evaders (own and other), the F11 campaigns' evaders, and two 100M-frame exploiters each: thesis rules without the GAE fix (F3c / F9c / F11c recipe; seeds 900 / 901; seed 0 reuses F9c's exploiter) and v2d rules (F11c recipe; seeds 910 / 911 / 912). Cross-play and stage control, eval only: the F11 final pursuers (stage 24) vs the F13 campaigns' evaders; the F11 stage-12 pursuers (same number of stages as F13) vs the test set, the strong finals and the F13 evaders. J at 0.10 m (primary) and 0.15 m. **Outputs.** `F13_thesis_short_sp/summary.md`, `results.json`.
- **Decision rule (user, 11-13 01:14).** Four measures at 0.10 m, mean over seeds (F13: 0 / 100 / 200; F11: 0 / 100): M1 held-out RL (v10); M2 strong-finals mean (16); M3 exploitability vs the thesis-rules exploiter (lower is better); M4 J vs the other rule set's campaign evaders (F11 pursuers vs F13 evaders; F13 pursuers vs F11 evaders). **v2d beats F13's setup if its mean is better on at least 3 of the 4.** Then F12 is resumed; otherwise F12 is re-run on F13's setup (new pre-registration). The F11 stage-12 control is reported, not used in the rule.
- **Predictions.** (1) Pursuer crash vs the thesis-rules exploiter < 0.10 on both new seeds (F9 control 0.056, F10 0.031). (2) Held-out RL >= 0.90 on both new seeds (F9 control 0.893, F10 0.933; low confidence). (3) v2d does not beat F13's setup (rule above). (4) In-campaign lure rate (a) (mean over evader stages 3-11) between 0.10 and 0.60 on both new seeds (F9 control 0.36; low confidence). Descriptive (3 seeds; the decision uses means, no CIs)
- **Status.** Done (campaigns ~05:30, F13c 07:43; RESULTS.md [F13](../experiments/thesis/RESULTS.md#f13), 11-13 09:24): decision rule favoured v2d on 4/4 measures, so resume F12; F10 amendment 1 (more seeds) runs first (user, 11-13).

#### F14_generalization_sp (user request, 2026-11-13 15:39; prospective follow-up to F8/F10/F11/F13; exploratory)

- **Question.** With a 13-stage (13 x 50M) self-play budget, does the learned acceleration estimator with online RLS under either the original thesis rules or thesis v2d improve generalization and reduce exploitability relative to the 26 x 50M Roncero-rule F8 pursuer? If the 13-stage estimator arm does not improve on F8, does resuming the same training state to 26 stages close the gap? Does online RLS help when an evader changes strategy during an episode?
- **Disclosure / scope.** The hypothesis and this design follow inspection of existing F8/F10/F11/F13 results, so F14 is an exploratory follow-up, not a prospectively registered test of those earlier results. Register before F14-specific training/evaluation. Report all results, including the 13-stage result if 26-stage continuation is triggered. No post-result changes to the test set, metrics, weighting, or trigger.
- **Rule-selection baseline (Stage A).** Compare plain thesis-rule F10 final pursuers at stage 12 (seeds 0/100/200) and plain v2d F11 pursuers at stage 12 (the 13th stage, 0-indexed; seeds 0/100/200) against the F8 final pursuers at stage 24 (seeds 0/100/200). F10 seeds 0/100/200 and F11 seeds 0/100 already exist; F11 seed 200 is a new v2d campaign stopped after 13 stages for the stage-matched comparison. Add F8 seed 200 (Roncero rules, Hedge both roles, GAE fix, 26 x 50M). Existing F8 seeds 0/100 are retained. The F10 seed expansion/evaluations already underway are not modified.
- **Generalization test (G1).** Evaluate each final pursuer, including estimator arms later in Stage C, on:
  1. F8d's fixed 16 strong final RL evaders, an unseen, multi-lineage set spanning different rules/rewards/strategies. Report each evader and source-family results.
  2. Every RL evader stage from other seeds/campaigns trained under the same self-play rules as the evaluated pursuer; exclude its own campaign. Average within each campaign first, then average campaigns equally, so F8's 26-stage pool does not outweigh a 13-stage candidate merely by containing more policies.
  The G1 score is the equal-weight mean of the two group means (0.5 x strong-set J + 0.5 x cross-seed/campaign J); G1 gap = candidate G1 - F8 G1. Report both component gaps as well as G1. Use neutralization J at 0.10 m as primary and 0.15 m as secondary.
- **Common evaluation outcomes.** All generalization and exploiter evaluations use the original thesis episode-ending semantics, regardless of the pursuer or exploiter's training rules: capture, either drone crash, evader out-of-bounds, or timeout ends an episode; pursuer out-of-bounds alone does not. A pursuer that leaves and later returns continues the episode; if it fails to capture before timeout, it is a non-capture/timeout. Report pursuer out-of-bounds separately; never count it as an evader win. This specifically requires re-evaluating the existing F8 final pursuers against the existing Roncero-trained exploiters with thesis evaluation semantics: current F8 Roncero-rule evaluations end when the pursuer leaves and cannot be rescored as-is.
- **Exploitability test (G2).** For each final pursuer, train fresh 100M-frame RL exploiters against that exact checkpoint:
  - Thesis-rule lure exploiter: original thesis rewards/episode rules, GAE truncation fix off, the F3c/F9c recipe.
  - Same-rules exploiter: rules matching the pursuer's self-play campaign (F8 Roncero, F10 thesis/no-fix, F11 v2d).
  For a thesis/no-fix pursuer these are one and the same exploiter, not duplicate runs. Evaluate every exploiter under the common thesis episode-ending semantics above. G2 is J versus the candidate's same-rules exploiter minus mean F8 J versus F8's same-rules exploiter; positive is better for the candidate. Also report J, 1-J, pursuer crash, evader crash/out-of-bounds, timeout, and pursuer out-of-bounds for each exploiter. Use 0.10 m primary / 0.15 m secondary.
- **Rule choice.** First publish per-seed and mean G1 and G2 gaps (with G1's two components); do not show a combined score in place of them. Then calculate G1 + G2 and select the rule set with the larger sum for the estimator study. Equal weights are intentional: both gaps are J differences on the same scale. In a tie, report no winner and use v2d (F12's preregistered setup) as the operational default, without claiming it won. Rule choice uses the means; seed-level variation and uncertainty remain visible.
- **Stage C: estimator arms at 13 x 50M, selected rules, seeds 0/100/200.** Compare: (a) plain observation; (b) learned acceleration estimator + online RLS; (c) the same learned estimator with RLS frozen during training (no online RLS updates). Keep Hedge on both roles, warm start, rules, PPO settings, and seeds matched; seed 0 is included. For v2d, resume the paused F12 seed 0/100 campaigns through stage 12 and add the seed-200 13-stage campaign. For thesis rules, reuse F10 plain pursuers and train the two estimator arms. All selected-arm pursuers get the G1 and G2 evaluations. Additionally evaluate the RLS-trained pursuer with RLS frozen at evaluation to isolate the value of test-time updating. F12's existing 26-stage plan is superseded only for this interim stage-12 evaluation; retain its 26-stage continuation as the fallback below.
- **Within-episode adaptation (Stage D).** Implement a learned-policy switch evaluation (not the existing scripted heuristic switches): an evader runs a late-stage F8 (timeout/avoidance) RL policy, switches at a randomized 0.5/1.0/1.5 s to a late-stage F10 (luring) RL policy, and a reciprocal F10-to-F8 condition. No policy is trained on the evaluated pursuer. Include fixed-policy controls for each constituent evader and evaluate 2,048 episodes per condition. Compare estimator+RLS, estimator trained without RLS, and estimator+RLS with RLS frozen at evaluation. Primary adaptation outcomes: capture/J and time-to-capture among episodes still active at the switch; also report full-episode J and all ending rates, including pursuer OOB separately. The switch harness must verify the policy actually changes within an episode, preserve observation/action compatibility, and pass automated tests plus a smoke evaluation before full runs.
- **26-stage fallback (Stage E).** Trigger if the selected-rule 13-stage estimator+RLS arm fails to have both G1 gap > 0 and G2 gap > 0 versus F8 (either non-positive gap triggers the extension). Preserve and report all 13-stage results. Extend the estimator+RLS and no-RLS arms to 26 x 50M from a snapshot/clone of the stage-12 self-play database, continuing at stage 13 through stage 25; final pursuer is stage 24. Do not retrain stages 0-12 or overwrite the 13-stage campaign. Verify resumed configs and weights against the original before launch. The plain selected-rule control already has 26-stage equivalents for v2d (F11); if thesis is selected, extend the F10 plain campaigns as an additional schedule control. Re-run G1/G2 and adaptation for the stage-24 finals with fresh exploiters.
- **Statistics.** Three training seeds per arm/reference. Report each seed, mean, and range; report paired within-seed contrasts for plain/RLS/no-RLS. Do not use episode count as the sample size for training-seed uncertainty or claim a narrow population CI from n=3. For episode-level proportions, provide Wilson intervals as evaluation-sampling uncertainty only; report the 95% stratified-bootstrap IQM interval and paired differences where supported by the existing thesis analysis tooling, labeling seed-level limitations. Each (pursuer, evader) matchup targets at least 2,048 episodes; reuse an identical eval seed across conditions where possible.
- **Predictions.** (1) Selected 13-stage estimator+RLS arm has G1 > F8 (both component groups reported). (2) It has G2 > 0 under the same-rules exploiter. (3) Online RLS improves post-switch capture/J over both the frozen-at-eval and no-RLS-trained controls. (4) If the 13-stage arm fails the joint G1/G2 rule, the 26-stage continuation improves the failed gap(s) relative to its own 13-stage checkpoint. Directional predictions are low-confidence; report contrary outcomes.
- **Execution order/status.** (1) Finish F10 seeds 100/200 and F10sc already running; (2) implement/test F14 evaluators and switch harness; (3) train/evaluate F8 seed 200 and F11 v2d seed 200, then complete Stage A evaluations including F8's no-OOB reruns and fresh exploiters; (4) publish G1/G2 separately, then select by sum; (5) train/evaluate selected-rule 13-stage estimator arms; (6) run adaptation; (7) if triggered, extend to 26 stages and repeat. At most two Isaac jobs at once. F12 remains paused until Stage A/rule selection; no F14-specific training or evaluation began before this registration.
- **Outputs.** `F14_generalization_sp/results.json`, `summary.md`, switch-eval artifacts and per-seed campaign records; update `RESULTS.md` with measured numbers only and `PROGRESS.md` with current jobs/decisions.

## 5. Protocol details

- **Pool split, pre-registered on 2026-10-26 before any X1 data was seen.** Held-out RL evaders: `v10-without-warm-start` (the only lineage trained without warm-starting). Train RL evaders: `v9-min-distance` + `v11-uniform-mix` (+ `v6` only if X0 shows it replays validly). Train heuristics {hover, circular, flee, weave}; held-out heuristics {zigzag, fast_flee} and switch forms {flee>circular, circular>hover, hover>flee, flee>weave} at `@0.5` (50 steps, 1 s real); held-out also includes the X3 exploiters. If v10 did not replay validly, held-out would be the last 5 stages of v9 (the only allowed change).
- **Default held-out test from 11-12 (user):** new experiments report held-out RL on v10 + F8d's 16 strong final evaders (`f8d_heldout.heldout_v2`), next to v10 alone.
- **Fair comparison.** Identical PPO hyperparameters, reward, DR, frame budget and eval seeds within an experiment; only the factor under test changes.
- **Statistics.** 3 seeds per cell (5 in F1); >= 2048 episodes per (policy, evader) pair; IQM with stratified bootstrap 95 % CIs (`rliable`), paired differences vs the baseline. A claim is "supported" only if the CI of the paired difference excludes 0.
- **Metrics.** `J` (primary) and capture rate; right-censored time to neutralise (15 s on failure); pursuer crash / OOB, effort; worst-cluster `J`; exploitability; gap closed `GC = (J_A - J_G) / Delta`; prediction error vs horizon; latency.
- **Capture radius.** Training and primary eval at 0.10 m; from 11-06 every closed-loop J is also re-scored at 0.15 m.
- **GAE truncation fix.** Default on from 11-11; experiments registered earlier are pinned to their original setting (`common.gae_pin`).
- **Observation noise.** Moved to P2 on 10-27, promoted on 10-28/10-29 (A3a-A3d, F4).
- **Honesty rules.** No metric or test set is changed after seeing results without saying so in RESULTS.md. Every run that crashed or was excluded is listed. Predictions in this file are not edited after the fact.

## 6. Code changes (original plan)

Planned 10-26 (later changes in §9): neutralisation outcome, 2048-episode evals, trajectory statistics (`run_stage.py`, `pursuit_evasion.py`); privileged obs keys O1-O3; `sampler: qd_dpp` (`tests/test_qd_dpp.py`: bounded cluster mass under duplication); in-run sampler refresh; `esn.feature: learned`; ESO feature; experiment driver (2M-frame smoke run of every experiment).

## 7. Schedule

Original 14-day schedule (10-26): days 1-3 X0-X4 with L0 (gates G1, G2); days 4-9 Phases D and A; day 10 (11-04) F1; day 11 (11-05) freeze with RESULTS.md complete; days 12-14 writing. The freeze moved to 11-08 (on 11-04, for F5), then to ~11-10 (F6b, F8). Actual dates: PROGRESS.md.

## 8. Risks

Registered 10-26 (mitigation in brackets): old evaders do not replay (exclude them); no adaptation gap (report it with X4 oracle evidence, weight moves to RQ1); QD-DPP within noise of S-M (report it; D1b tests the mechanism; 5 seeds); exploiters win by forcing pursuer crashes (crash/OOB breakdown shown separately); Isaac crashes / overruns (resumable driver; drop P1 items first: D3, then A2, then GRU); real-world validation out of scope (deploy code in [scripts/deploy/](../scripts/deploy)).

## 9. Deviation log

Chronological. Amendment details are in the §4 entries.

| Date | Change | Reason |
|---|---|---|
| 10-26 | X4: O3 dropped; O2 split into O2a / O2b | Needs delayed obs (A1a answers it); O2a = ESO baseline |
| 10-26 | BayesID P1 -> P2 (gate G2) | L0: covered by BPR / Bayes-ToMoP / GSCU |
| 10-26 | D1 sampler refresh inside the run every 25M frames | Saves an Isaac start-up + eval per refresh |
| 10-26 | S-U = M0 = X3 generalist (same runs) | Identical configuration |
| 10-26 | A1b arms fixed; GRU only if time | Budget |
| 10-26 | Clustering k = 2..8 | Fixed before X1/X2 results |
| 10-26 | D1, D2, A1b at 5 seeds; X4 at 3 | X3 seed variance, before any 3-seed result |
| 10-26 | D1 exploiter column biased against S-U; D2 is fair | Noted at the first D1 numbers |
| 10-27 | A1a: bounded, tuned RLS | Defaults diverged |
| 10-27 | D1c_nofloor added (exploratory, after D1b) | D1b's floor hides the DPP mechanism |
| 10-27 | F1 redesigned (choice B); A2a/A2b added; noise tests to P2 | A1b: adaptation adds nothing reliable |
| 10-27 | D1c interrupted after 2 of 12 runs, requeued | Lane capacity to F1 / A2 |
| 10-28 | A3a/A3b/A3c added | User request after A2b |
| 10-28 | A3a re-run in float32 | float16 recorder bug |
| 10-28 | F2a/F2b/F2c added (19:30), replacing D3 | User: final method under the papers' protocol |
| 10-28 | F3 added (20:30); campaign evals use the multi-episode estimator | User: samplers in self-play |
| 10-28 | F4 added (21:30) | User: interceptors rarely measure evader velocity |
| 10-29 | F4 v2 + A3d (~11:00, commit 55bbbac) | A3b: no acceleration input helps without velocity |
| 10-29 | F2a analysis rebuilt (3 of 5 seeds shown) | Bookkeeping fix |
| 10-29 | F3 amendments 1 (19:40) and 2 (20:13) | User |
| 10-29 | A3e added (20:19) | User: why not recurrent? |
| 11-02 | F4c analysis rebuilt | Bookkeeping bug |
| 11-02 | Power loss 11-01 08:48 - 11-02 10:19; F3 campaigns resumed | External event |
| 11-02 | Resumed F3 campaigns ran extra stages; F3c uses stages 0-8; `max_total_stages` guard | Bug, cost ~16 GPU-h |
| 11-03 | F3 amendment 3 (both roles) | User: match the reference papers |
| 11-03 | A3f added | User, after Pan et al. 2026 |
| 11-04 | F3 amendments 4 (14:45) and 5 (15:25) | User |
| 11-04 | F5 added (17:40); freeze 11-05 -> 11-08 | User |
| 11-04 | F3 amendment 6 / F5 amendment 1 (21:11); F4c post-hoc `_no_s1` | User: avoid data leakage |
| 11-05 | F5 Exp 2 paused (14:38); post-hoc: F3 QD-DPP sampled like Hedge (mean TV 0.010) | User: improve QD-DPP first |
| 11-05 | F6 added (~16:00); F6 amendment 1 (16:12) | User |
| 11-05 | F5c evaluates Exp 1 alone | Exp 1 does not depend on Exp 2 |
| 11-06 | F5 amendment 2 (12:45) | User, after the 2-seed results |
| 11-06 | F7 added (19:30); all closed-loop J re-scored at 0.15 m | Every evader win is a pursuer crash; 0.10 m < body-contact reach 0.145 m |
| 11-08 | F6b added (15:40); after F7c the rules stay; freeze to ~11-10 | User: redesign F6 |
| 11-09 | F6b amendment 1 (01:30) | Unstable tau |
| 11-09 | F8 added (12:21); `roncero_reward.py`, `gae_cut_at_truncation`; F8 amendment 1 (12:28) | User: would the reference papers see luring? |
| 11-10 | F9 added (15:19) | User: which Roncero component |
| 11-10 | F8c loader fix (17:05); F8 amendment 2 (19:25, k_sm fix, re-run) | Bugs found in checks |
| 11-11 | F10 added (11:45); F8d added (12:21) | User |
| 11-11 | GAE truncation fix default on (16:29); earlier experiments pinned | F10: luring 0.81 without vs 0.36 with |
| 11-11 | F11 added (17:06); amendment 1 (17:18); stopped (19:55); amendment 2 (20:01); v2b gate failed (20:16) | Pilot / gate results |
| 11-12 | F11 amendment 3 (00:43); v2c gate failed (00:58); amendment 4 (09:16) | Gate results |
| 11-12 | Default held-out test and F12 added (13:43) | User |
| 11-12 | File reformatted; predictions verbatim | Readability |
| 11-13 | F12 stopped after 10 stages (01:14); F13 added (01:16) | User: test the thesis rules with 13 x 50M before F12 |
| 11-13 | F10 amendment 1: seeds 100 / 200 before the F12 resume (11:45) | User, after F10 s0's strong-finals eval and F13's seed variance |
| 11-13 | F14 registered (15:39) | User: select recommended design choices, detail the plan, then execute |
