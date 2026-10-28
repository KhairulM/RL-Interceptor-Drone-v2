# Results (thesis experiments)

Measured results only. Each section is filled when the experiment finishes, with exact numbers, CIs, number of
seeds/episodes, and any deviation from [the plan](../../docs/research_plan.md). Predictions are in the plan and are
compared here as "prediction held / did not hold".

## Summary (updated 2026-10-28 19:45; refreshed after every new result)

**How to read the numbers.** J = neutralisation rate. `0.676 [0.533, 0.846]` = mean over training seeds and its
95 % confidence interval (CI). CIs are percentile bootstraps over seeds (resample the per-seed values with replacement
2000 times, take the 2.5th / 97.5th percentiles of the mean; `analysis.bootstrap_ci`). For "A minus B" tables the
per-seed differences (same seed index) are bootstrapped. A difference is called "supported" when its CI excludes 0.
With 3-5 seeds the bootstrap CI is coarse and tends to be too narrow, so a CI that only just excludes 0 is weak
evidence.

J = neutralisation rate (capture, or the evader crashes / leaves the arena). "Interim" = fewer seeds than final.

**Setting and diagnostics (final)**
- Stored self-play checkpoints v9 / v10 / v11 replay correctly; v6 does not under the current env and is excluded (X0).
- The 1v1 game is mostly transitive: one rating per player explains 78-84 % of the outcome variance (X1).
- Self-play pools are redundant: 22 RL evaders amount to 3.8 effectively distinct behaviours (Vendi score); there are
  7 behaviour clusters (X2).
- A real gap to unseen evaders exists: 7 of 15 test evaders lose by >= 0.15 J to the generalist compared with a
  specialist (exploiters: 0.57-0.78; held-out v10 evaders: 0.18-0.37) (X3).
- Most of that gap is **missing opponent state**: adding the evader's acceleration (a finite difference, computable
  from two observations) raises J from 0.65 to 0.94 on the train pool, from 0.47 to 0.77 on held-out RL and from 0.31
  to 0.97 on exploiters. Conditioning on an opponent descriptor fails out of distribution (J ~= 0 on 3 novel
  evaders) (X4).

**RQ1: diversity-aware opponent sampling (QD-DPP)**
- D1 (final, 5 seeds): QD-DPP beats both hardness-prioritised samplers on train and held-out evaders, with paired CIs
  excluding 0. vs the current multi-metric sampler: held-out RL +0.185 [+0.110, +0.259], worst held-out cluster
  +0.176 [+0.049, +0.302]. vs win-rate PFSP: +0.131 / +0.116. **Against uniform sampling the gain is small and
  not significant** (held-out RL +0.042 [-0.069, +0.131]). Hardness-only prioritisation is worse than uniform; the
  diversity term repairs it.
- D1b (3 seeds): the DPP does bound the sampling mass of a duplicated opponent (0.21 vs 0.29), but QD-DPP did
  **not** keep its performance better under duplication (held-out RL -0.23 vs -0.10 for S-M). H1b not supported.
  D1c (exploratory, no uniform floor): both samplers collapse (train J -0.29); in that regime duplication hurts
  S-M (held-out RL -0.05, worst cluster -0.10, CIs exclude 0) but not QD-DPP, i.e. the duplicate robustness shows only
  when hardness dominates the sampling.
- D2 (final, 5 seeds): every sampler's pursuer is highly exploitable by a fresh exploiter (mean 0.44-0.69); no
  sampler reduces exploitability reliably (QD-DPP 0.630 vs uniform 0.672, n.s.). One exploiter run failed (S-W seed 4).

**RQ2: online opponent adaptation (NF-OA)**
- A1a (offline): a learned opponent basis halves the prediction error of constant-velocity extrapolation at every
  horizon. Random-reservoir (ESN) and raw-history RLS are worse than constant velocity beyond 0.1 s. Online RLS on top
  of the basis adds a small, consistent gain (6-16 %), largest when the readout is kept across episodes. RLS with the
  pre-registered defaults diverged and had to be bounded and tuned.
- A1b (final, 5 seeds): in closed loop **no adapter beats M0** (NF-OA: train +0.04, held-out RL -0.03, both n.s.);
  online adaptation over the frozen basis is not significant and across-episode adaptation changes nothing (<= 0.005).
  ESN and a raw frame stack are significantly *worse* than M0. The explicit acceleration input is far better than
  all of them.
- A2a (offline): a learned estimator of the evader's *future acceleration* (0.2 s ahead) beats the finite-difference
  estimate by 7 % (frozen) to 21 % (online RLS kept across episodes); here online adaptation clearly helps
  (-10 to -15 % error vs frozen). For one step ahead the finite difference is better. A2b (policies with this input)
  is queued.
- A2b (3 seeds): feeding the policy the **learned estimate of the evader's acceleration** (0.2 s ahead) gives large
  gains over M0 (+0.25 train, +0.34 held-out RL, +0.67 exploiters; CIs exclude 0) and is on par with the
  finite-difference acceleration input (held-out RL 0.82-0.88 vs 0.77, n.s.). None of its 6 runs fails on the later
  v10 evaders (finite difference: 3 of 10 runs fail). Online RLS still adds nothing in closed loop.
- A3a (offline, no evader velocity observed): noise-free, the second difference of the evader position
  estimates its acceleration 1.5-2.2x better than a learned estimator. With realistic sensor noise (2 cm) position
  differences are useless (errors 28-55 m/s; even EMA-filtered, worse than predicting zero), and only the
  noise-augmented learned estimator beats zero (-13 to -26 %). A3b tests this in closed loop.
- Overall reading: what the pursuer lacks against unseen evaders is the opponent's current/near-future acceleration.
  A learned opponent model helps when it predicts that quantity; online adaptation of the model does not add to
  closed-loop performance in this (noise-free, full-state) setting.

**Final 2x2 (F1, 5 seeds)**
- {uniform, QD-DPP} x {plain, acceleration input}: the **acceleration input is the dominant factor** (train +0.20,
  exploiters +0.46; CIs exclude 0). The sampler has no main effect. With the acceleration input, QD-DPP is not better
  than uniform (slightly lower on held-out RL, driven by one extra seed that fails on the v10 lineage). Held-out
  generalisation is bimodal per seed: a policy either handles the later v10 evaders (0.82-0.97) or fails on all of
  them (0.10-0.29).

**Pending**: A2b seeds 3-4 (running, relaunched after an OOM crash on 10-28); A3b (positions-only
policies, 3 seeds) and A3c (noise evaluation of the velocity-based policies) queued; final runs F2a (exploitability
of the acceleration-input pursuer) and F2b/F2c (self-play campaigns with vs without the acceleration input) queued.

## Relation to the reference papers (updated 2026-10-28 19:10)

Reference papers (the thesis is positioned against these three):
**[AgilePE]** Tang et al. 2026, arXiv:2608.14135; **[Roncero]** Sanchez Roncero et al. 2026 (AMSPBH),
arXiv:2506.02849v3; **[Gavin]** Gavin & Bronz, ICUAS 2026 (DOI 10.1109/ICUAS69441.2026.11598610).
Numbers are **not** comparable across papers (different arenas, capture definitions, metrics, simulators); the
comparison is between *claims and mechanisms*, tested in our setting. "Status" says whether our data agrees,
disagrees under our protocol, extends the claim, or does not test it.

**Setting comparison**

| | ours | Roncero | AgilePE | Gavin |
|---|---|---|---|---|
| game | 1v1 | 1v1 | 1v1 | 3v1, net capture |
| platform / sim | Crazyflie, OmniDrones (Isaac) | Crazyflie brushless, Isaac Lab | quadrotor, OmniDrones + RK4 kinematic backend | quadrotor, JAX sim |
| arena | 4 x 4 x 2.9 m | 4 x 5 x 2 m | 6 x 6 x 3 m | 32 x 32 x 16 m (evader 12 x 12 x 6) |
| action / rate | CTBR, 50 Hz | CTBR, 50 Hz (500 Hz rate loop) | CTBR, 60 Hz | CTBR, 100 Hz |
| pursuer observes | own state, opponent rel. position + velocity (full state, memoryless MLP) | same (memoryless MLP) | own state + **temporal window** of opponent state, FOV-masked | own state, opponent rel. position + velocity |
| opponent sampler | uniform, win-rate softmax, multi-metric softmax, QD-DPP (ours) | Hedge over win rate (gamma 0.7, eta 2) | PFSP over win rate | PFSP over outcome score (time + crash), f = (1-x)^2, 50 % latest |
| outcome metric | J = capture + evader crash + evader OOB | pursuer win = capture or evader OOB | in-FOV rate, capture rate | catch / evade / crash rates, censored time to catch |
| seeds, episodes | 3-5 seeds, >= 2048 eps per pair (typ. ~5k) | 3 seeds, 3072 eps per matchup | not stated | not stated, 10k eps per pair |
| real hardware | **no** | yes (42 flights) | yes | no |

**Claims and what our data shows**

| # | claim in the reference papers | our evidence | status |
|---|---|---|---|
| 1 | Prioritising hard opponents (Hedge / PFSP) improves on uniform population sampling. Roncero Fig. 5: Hedge beats uniform AMSPB by +0.05 to +0.27 win rate in 8 of 9 settings (3 seeds). AgilePE: PFSP stronger than FSP (single runs). | D1, 5 seeds, fixed 20-evader pool: win-rate softmax (S-W, closest to Hedge/PFSP) minus uniform: train **-0.054 [-0.081, -0.026]**, held-out RL **-0.088 [-0.145, -0.035]**, worst held-out cluster **-0.088 [-0.147, -0.030]**, worse on 5/5 seeds. Multi-metric softmax (S-M, closest to Gavin's outcome score): train **-0.150 [-0.259, -0.047]**, exploiters **-0.166 [-0.302, -0.047]**, held-out RL -0.143 [-0.306, +0.021]. D2: no sampler lowers exploitability. | **Does not replicate under our protocol.** Differences: we train one pursuer against a fixed pool (no co-evolving generations); our uniform floor is 0.5 (Roncero 0.7) and we use a softmax (T = 0.9) instead of a power eta = 2. We did not run latest-opponent-only self-play (Roncero's AMSDRL, Gavin's SP), so we do not test "population beats latest-only". D1c (exploratory): with **no** uniform floor, hardness prioritisation collapses (train -0.29); this agrees with Roncero's advice to keep gamma large, and is a warning for PFSP without a uniform share (AgilePE does not report one). |
| 2 | Self-play with PFSP "sustains tactical diversity" (AgilePE); the population "contains distinct pursuit and evasion behaviours" (Roncero). Neither paper measures diversity. | X2: 22 RL evaders from three self-play campaigns amount to **3.8 effectively distinct behaviours** (Vendi score), 7 clusters; consecutive stages are 36 % closer than random same-campaign pairs. | **Extends:** first quantitative measure; the pool is diverse across campaigns but highly redundant within one. |
| 3 | Our addition: sample by quality *and* diversity (QD-DPP). No reference paper uses diversity in sampling. | D1: QD-DPP beats both hardness samplers (vs S-M held-out RL +0.185 [+0.110, +0.259]; vs S-W +0.131 [+0.038, +0.223]) but is **not** better than uniform (+0.042 [-0.069, +0.131]). D1b: bounds the sampling mass of duplicates but does not protect performance. F1: no sampler effect once the acceleration input is present. D2: not less exploitable. | **New, mostly negative:** diversity repairs the damage hardness prioritisation does in our setting, nothing more. |
| 4 | Population training generalises to unseen opponents (Roncero: unseen APF evader; Gavin: SP/PFSP pursuers vs other evaders). | Consistent for scripted opponents: held-out scripted evaders (zigzag, fast flee, 4 switching evaders) are neutralised at J 0.99-1.00 by every pursuer. **Not** for learned opponents: against the independent v10 lineage the generalist loses up to 0.37 J to a specialist; fresh exploiters (PPO best responses, 100M frames) reach J 0.22-0.43 against it (X3), and the mean exploitability per sampler is 0.44-0.69 (D2; one failed exploiter included). | **Consistent for scripted, contradicted for learned opponents.** None of the three papers reports exploitability or held-out RL lineages. |
| 5 | Seeds may converge to different strategies (Roncero, limitations). | Generalist (uniform sampler) train J 0.56-0.87 across 5 seeds; held-out generalisation is **bimodal per seed**: 1-2 of 5 seeds fail on all later v10 evaders (J 0.10-0.29) while the others reach 0.82-0.97 (F1). | **Extends:** quantifies it; with 3 seeds (Roncero) effects of +/- 0.1 can come from one seed. |
| 6 | Observation design: Roncero and Gavin give the pursuer opponent position + velocity to a memoryless MLP; Gavin leaves "trajectory prediction" out of scope; AgilePE adds a temporal window of opponent state. | Adding the opponent's acceleration (finite difference of two consecutive observations) is the **largest effect in this thesis**: train 0.722 -> 0.940 (+0.199 [+0.128, +0.270]), exploiters 0.497 -> 0.983 (+0.464 [+0.239, +0.638]), held-out RL 0.634 -> 0.838 (CI includes 0 because of the bimodal seeds) (F1, 5 seeds). A learned acceleration estimate does the same (A2b). A raw 10-step history stack (our analogue of a temporal window) was *worse* than no history (train -0.254 [-0.355, -0.152]) (A1b). | **New:** a memoryless pursuer with position + velocity (Roncero, Gavin) is limited by the missing opponent acceleration far more than by opponent sampling. Caveat: AgilePE's window length and architecture are not given, so the history-stack result is not a test of AgilePE itself. |
| 7 | Future work in Roncero: online adaptation, recurrent policies. | Online RLS adaptation of a learned opponent model (NF-OA), within and across episodes, adds nothing in closed loop (A1b, A2b); a random reservoir (ESN) is worse than no adaptation. Recurrent policies (GRU) **not tested**. | **Answers part of it (negative)** in a noise-free full-state setting. |
| 8 | Sensor noise: AgilePE reports robustness under real sensor noise (mocap + IMU); Roncero and Gavin assume perfect state. | A3a (offline): with 2 cm position noise, acceleration from position differences is useless (error 28-55 m/s); a noise-augmented learned estimator is the only method better than zero. Closed loop: A3b / A3c pending. | **Pending.** |
| 9 | Real-world transfer (all three), multi-pursuer coordination (Gavin), FOV-limited sensing (AgilePE). | Not studied. | **Not tested.** |

## X0_replay (done 2026-10-26)

Final pursuer of each campaign re-evaluated against 3 RL evaders recorded in its `matchups.json`, with the new
multi-episode estimator (cutoff 1500 steps, 4096 envs). Full table: [X0_replay/summary.md](X0_replay/summary.md).

| campaign | reproduced (\|delta capture\| <= 0.05) | max \|delta\| | max \|z\| | verdict |
|---|---|---|---|---|
| v6 | 1 / 3 | 0.906 | 119.7 | **invalid**: evaders leave the arena in 85-99.8 % of episodes under the current env rules; excluded |
| v9 | 3 / 3 | 0.022 | 1.0 | valid |
| v10 | 3 / 3 | 0.009 | 0.8 | valid |
| v11 | 3 / 3 | 0.030 | 2.0 | valid |

- Estimator check (v11): multi-episode capture 0.982 / 0.793 / 0.837 (21.7k-23.6k episodes) vs classic first-episode
  eval 0.982 / 0.794 / 0.827 (~1.37k episodes). They agree within sampling error.
- Prediction (plan): "v9/v10/v11 within 0.05, v6 fails". **Held.**
- Side observation: `v10/evader:s7` is neutralised only 1.7 % of the time by the v10 final pursuer (capture 0.4 %).

## L0 literature / gate G2 (done 2026-10-26)

Notes: [L0_literature/notes.md](L0_literature/notes.md). Novelty verdicts (research agent, arXiv-verified):

| method | novel in general? | novel for agile quadrotor PE? | decision |
|---|---|---|---|
| QD-DPP opponent sampling | likely, as a sampling rule (DPP diversity is known as a best-response regulariser) | yes | keep (main) |
| NF-OA opponent adaptation | mechanism known (Neural-Fly, RMA), not for opponent modelling | yes | keep (main) |
| BayesID | no (BPR / Bayes-ToMoP / GSCU) | likely | dropped to P2 |

## X1_crossplay (done 2026-10-26)

14 pursuers (last 4 stages of v9 / v10 / v11 + scripted pure pursuit and APF) x 32 evaders (22 RL + 10 scripted),
multi-episode eval, 256-3,400 episodes per cell (median 1,745). Primary metric: neutralisation J.
Details: [X1_crossplay/summary.md](X1_crossplay/summary.md), heatmaps `fig_payoff_neut.png`, `fig_payoff_capture.png`.

| subset | additive-logit R² (one strength per player) | rank-reversal fraction (delta 0.1) |
|---|---|---|
| all 14 pursuers | 0.777 | 0.045 |
| RL pursuers only | 0.835 | 0.024 |
| all, capture rate instead of J | 0.831 | 0.014 |

- The game is **mostly transitive**: one strength axis explains 78-84 % of the logit-J variance, and only 1-4 % of
  (pursuer pair, evader pair) comparisons reverse. Prediction "Elo R² 0.8-0.9": held for RL pursuers (0.835),
  slightly below for all pursuers (0.777).
- Campaign final pursuers differ strongly. v9/s20: mean J 0.909, worst evader 0.711. v11/s12: mean 0.868, worst
  0.216 (vs v9/evader:s19). v10 pursuers: mean J 0.27-0.35, capture <= 0.14 (the no-warm-start campaign did not
  produce a working pursuer). Prediction "some final pursuer < 0.6 vs another campaign's evader": **held** (v11/s12
  0.216 vs v9/s19; all v10 pursuers).
- Hardest evaders across all pursuers: v9/evader:s19 (mean J 0.151), s17 (0.220), s15 (0.285).

## X2_descriptors (done 2026-10-26)

Descriptor per evader = response block (J and censored time-to-neutralise vs the 14 X1 pursuers) + trajectory
block (mean/std of 9 per-step behaviour features), equal block weights. Details:
[X2_descriptors/summary.md](X2_descriptors/summary.md), `fig_mds.png`.

| group | n | Vendi score (effective number of distinct evaders) |
|---|---|---|
| all | 32 | 4.54 |
| RL | 22 | 3.79 |
| scripted | 10 | 1.84 |
| v9 / v10 / v11 | 10 / 6 / 6 | 3.32 / 2.30 / 2.59 |

- Mean descriptor distance: consecutive stages 1.15, any two evaders of the same campaign 1.79, across campaigns
  1.83. Consecutive stages are 36 % closer than an average same-campaign pair.
- k-medoids (k = 2..8, silhouette): best k = 7 (silhouette 0.461). Stage-1 evaders of all three campaigns fall
  together (all seeded from the same scripted opponents). v10 s7/s9/s11 form their own cluster with no training
  evader in it; v10 s1/s3/s5 share clusters with training evaders.
- Prediction H1a "4-7 clusters, consecutive stages near-duplicates, each campaign covers 2-3 clusters": clusters
  **held** (7); redundancy **held** (22 RL evaders = 3.8 effective); "2-3 clusters per campaign" **did not hold**
  (v9 spans 4 clusters, v10 3, v11 3).
- Pre-registered split written to `X2_descriptors/split.json`: train = v9 + v11 RL evaders + {hover, circular,
  flee, weave} (20); held-out = v10 RL (6) + {zigzag, fast_flee, 4 switch evaders @0.5} (6) + X3 exploiters.

## X3_gap (done 2026-10-26)

Generalist pi_G = pursuer trained from scratch on the train split, uniform opponents, 150M frames, seeds 0-2.
Exploiters = evaders trained from scratch (100M frames) against frozen pi_G seed 0. Specialist = pi_G seed 0
fine-tuned 20M frames on one test evader, evaluated only against that evader. Full table:
[X3_gap/summary.md](X3_gap/summary.md).

- Generalist mean J on the train pool: **0.748 / 0.640 / 0.562** (seeds 0 / 1 / 2). Seed variance is large.
- Gap Delta = J(specialist) - J(pi_G seed 0):

| group | evaders | Delta |
|---|---|---|
| held-out scripted (zigzag, fast_flee, 4 switch evaders @0.5) | 6 | -0.001 to 0.001 (pi_G already ~1.0) |
| held-out RL v10 | 6 | 0.013, 0.176, 0.048, **0.341**, **0.365**, **0.313** (s1, s3, s5, s7, s9, s11) |
| exploiters | 3 | **0.759, 0.777, 0.572** (pi_G J 0.239 / 0.220 / 0.427 -> specialist 0.998-0.999) |

- **7 evaders with Delta >= 0.15** (v10 s3/s7/s9/s11, all 3 exploiters). H2a (a real adaptation gap exists) **supported**.
- Exploiters win by evading, not by forcing crashes: vs pi_G seed 0 they reach timeout in 57-76 % of episodes,
  with 0 % pursuer crash. They also beat the other generalist seeds (J 0.17-0.49), so they are not seed-specific.
- Predictions: "exploiters Delta >= 0.25 for >= 2 of 3" **held** (3/3); "held-out RL Delta 0.05-0.2" **did not hold**
  (3 of 6 are 0.31-0.37); "heuristics Delta < 0.05" **held**.
- Caveat: specialists were not evaluated on the train pool, so this measures the reachable gain on one evader,
  not a policy that is good on everything.

## X4_oracles (done 2026-10-27 04:43) and gate G1

Pursuers trained like the X3 generalist (train split, uniform, 150M frames, 3 seeds) plus one extra input.
Mean J (per-seed in brackets); gap closed = mean over the 7 evaders with Delta >= 0.15 of
(J_method - J_M0 seed-mean) / Delta. Full: [X4_oracles/summary.md](X4_oracles/summary.md).

| input | train | held-out RL | held-out scripted | exploiters* | gap closed (n evaders) |
|---|---|---|---|---|---|
| none (M0 = X3 generalist) | 0.650 [0.75, 0.64, 0.56] | 0.473 [0.60, 0.45, 0.37] | 0.998 | 0.309 [0.30, 0.31, 0.32] | 0.00 (7) |
| O1 opponent descriptor (8-D PCA) | 0.541 [0.54, 0.54, 0.54] | 0.430 [0.41, 0.42, 0.47] | 0.995 | n/a | -0.34 (4) |
| **O2a opponent acceleration** (finite difference of true velocity) | **0.939** [0.96, 0.95, 0.91] | **0.773** [0.46, 0.93, 0.93] | 0.992 | **0.973** [0.93, 1.00, 0.99] | **1.08** (7) |
| O2b opponent's latest CTBR command | 0.740 [0.59, 0.81, 0.83] | 0.337 [0.29, 0.35, 0.37] | 0.998 | 0.322 [0.30, 0.25, 0.42] | -0.22 (7) |

\* exploiters were trained against M0 seed 0. Against them, M0 seeds 1-2 reach 0.17-0.49, O2a 0.87-1.00.

- **O2a is not privileged**: it is computable from two consecutive full-state observations. The memoryless MLP
  pursuer cannot form it. The large gain (train +0.29, held-out RL +0.30, exploiters +0.66) says most of the X3
  "adaptation gap" is **missing opponent state (its current acceleration)**, which a policy needs to intercept a
  manoeuvring target.
- **O1 hurts and fails catastrophically out of distribution**: J ~= 0.00 against v10 s7/s9/s11 on all 3 seeds
  (their descriptors are outside the training range), versus 0.21-0.64 for M0. Conditioning on opponent identity
  overfits.
- **O2b** helps in-distribution (+0.09) but not on held-out evaders (-0.14).
- Caveat: the oracle inputs are noise-free. With sensor noise (velocity std 0.05-0.08 m/s at 50 Hz) a finite
  difference is very noisy, so F1 must evaluate under observation noise with a filtered estimate.
- Predictions: "O2 helps most (+0.05-0.15)": O2a helped far more (+0.30 to +0.66); O2b did not help on held-out,
  so the direction held for O2a and the magnitude did not. "O1 helps little in-distribution (< 0.03)": **did not
  hold** (O1 is -0.11 in-distribution and collapses out of distribution).
- **Gate G1: passed** (an oracle closes >= 30 % of the gap: O2a 108 %). Phase A runs as planned. A1b must beat the
  O2a / ESO input (reused as baseline "ESO_O2a"), not only M0. M0h (frame stack) and NF-frozen separate "more
  history" from "online adaptation".

## D1_sampler (FINAL: 5 seeds, done 2026-10-27 20:49)

Pursuer trained on the train split (20 evaders), 150M frames, opponent probabilities refreshed every 25M frames
(temperature 0.9, uniform floor 0.5). S-U = X3 generalists (seeds 0-4). Mean J, 95 % bootstrap CI over 5 seeds
([D1_sampler/summary.md](D1_sampler/summary.md)):

| sampler | train | held-out RL (v10) | held-out scripted | exploiters* | held-out worst cluster | held-out worst evader |
|---|---|---|---|---|---|---|
| S-U uniform | 0.722 [0.625, 0.813] | 0.634 [0.447, 0.824] | 0.997 | 0.497 [0.306, 0.750] | 0.529 [0.326, 0.750] | 0.345 [0.104, 0.641] |
| S-W win rate | 0.668 [0.552, 0.784] | 0.546 [0.365, 0.761] | 0.995 | 0.466 [0.302, 0.675] | 0.440 [0.236, 0.690] | 0.272 [0.048, 0.604] |
| S-M multi-metric | 0.572 [0.457, 0.700] | 0.492 [0.337, 0.646] | 0.985 | 0.331 [0.161, 0.603] | 0.381 [0.214, 0.563] | 0.202 [0.050, 0.381] |
| **QD-DPP** | **0.752** [0.689, 0.812] | **0.676** [0.533, 0.846] | 0.996 | 0.496 [0.298, 0.750] | **0.556** [0.387, 0.755] | **0.358** [0.146, 0.640] |

\* exploiters were trained against S-U seed 0 (biased against S-U); D2 is the fair test.

Seed-paired differences QD-DPP minus baseline (mean [95 % bootstrap CI], seeds with QD better):

| vs | train | held-out RL | held-out worst cluster | held-out worst evader |
|---|---|---|---|---|
| S-M (current method) | **+0.180 [+0.090, +0.274]** 5/5 | **+0.185 [+0.110, +0.259]** 5/5 | **+0.176 [+0.049, +0.302]** 4/5 | **+0.156 [+0.020, +0.318]** 4/5 |
| S-W (PFSP / Hedge-style) | **+0.084 [+0.010, +0.189]** 4/5 | **+0.131 [+0.038, +0.223]** 5/5 | **+0.116 [+0.023, +0.214]** 4/5 | **+0.087 [+0.024, +0.185]** 5/5 |
| S-U uniform | +0.031 [-0.020, +0.104] 2/5 | +0.042 [-0.069, +0.131] 4/5 | +0.027 [-0.076, +0.105] 4/5 | +0.013 [-0.045, +0.057] 4/5 |

Sampling distributions (all refreshes, all seeds): effective number of opponents exp(entropy) out of 20:
S-W 19.6, S-M 18.8, QD-DPP 19.2; mean max probability 0.062 / 0.081 / 0.085.

- **QD-DPP beats both hardness-prioritised samplers** (S-M, S-W) on train, held-out RL, worst cluster and worst
  evader; the paired CIs exclude 0 (supported under the plan's rule).
- **QD-DPP vs uniform: not supported.** Mean differences are positive (+0.01 to +0.04) but the CIs include 0; seed 3
  is reversed.
- Hardness-only prioritisation **hurts** relative to uniform (S-M held-out RL 0.492 vs 0.634). Adding the diversity
  term repairs this. The advantage is not simply flatter sampling: S-W is flatter than QD-DPP yet worse.
- Caveat: all distributions are close to uniform (uniform floor 0.5, temperature 0.9) while the J differences are
  large. With seed variance this large, the effect sizes are uncertain even where the CI excludes 0.
- Interim (3-seed) reading "QD ahead of every baseline on held-out RL on all seeds" did not survive for uniform:
  seed 3 reverses it.
- Prediction H1c "QD >= S-M on held-out worst cluster by 0.03-0.08, equal in-distribution": QD's advantage over S-M
  is larger than predicted (+0.176) and also present in-distribution (+0.180), so the "equal" part **did not hold**.

## D1b_duplicates (done 2026-10-27 06:32; 3 seeds)

Train pool + 5 extra copies of the hardest train RL evader by X3 generalist J (**v9/evader:s15**), i.e. 6 copies
among 25 opponents. Samplers S-M and QD-DPP, same settings as D1 (uniform floor 0.5).
[D1b_duplicates/summary.md](D1b_duplicates/summary.md).

Sampling mass on the duplicated cluster (5 refreshes x 3 seeds): **S-M 0.255-0.314, QD-DPP 0.199-0.217**
(uniform share would be 0.24; the 0.5 uniform floor alone gives the copies 0.12). The DPP bound works as designed.

Performance change caused by duplication (seed-paired, D1b minus D1, mean over 3 seeds):

| sampler | train (non-duplicated evaders) | held-out RL | held-out worst cluster | exploiters |
|---|---|---|---|---|
| S-M | -0.161 [-0.306, -0.096, -0.079] | -0.096 [-0.371, +0.051, +0.032] | -0.134 | +0.097 |
| QD-DPP | -0.137 [-0.108, -0.154, -0.149] | **-0.230** [-0.326, -0.226, -0.138] | **-0.237** | -0.198 |

Absolute J under duplication: S-M train 0.411, held-out RL 0.374, worst cluster 0.241; QD-DPP 0.580, 0.370, 0.218.

- **H1b not supported in performance terms.** Lower sampling mass on the duplicates did not make QD-DPP more robust:
  it lost more than S-M on held-out evaders (-0.23 vs -0.10). Prediction "S-M drops 0.05-0.10, QD-DPP < 0.02":
  S-M's drop held (-0.10 held-out, -0.16 train); QD-DPP's **did not hold**.
- Limits of this test: (1) the 0.5 uniform floor is not duplicate-robust and controls half the probability mass;
  (2) 3 seeds with large seed variance (S-M seed 0 in D1 was an outlier, 0.769 train J), which inflates S-M's
  apparent drop on seed 0 and nothing else.
- Possible follow-up (not run): D1b with `uniform_mix = 0`.

## D1c_nofloor (done 2026-10-28 19:25; EXPLORATORY, designed after seeing D1b; 3 seeds; no pre-registered prediction)

D1b repeated without the uniform floor (`uniform_mix = 0`): S-M vs QD-DPP, normal pool ("nodup") and the pool with
5 extra copies of v9/evader:s15 ("dup"). Otherwise as D1 (150M frames, refresh every 25M, temperature 0.9).
[D1c_nofloor/nodup/summary.md](D1c_nofloor/nodup/summary.md), [dup/summary.md](D1c_nofloor/dup/summary.md).

| pool | sampler | train | held-out RL | held-out scripted | exploiters | worst held-out cluster |
|---|---|---|---|---|---|---|
| nodup | S-M | 0.283 [0.269, 0.307] | 0.363 [0.318, 0.395] | 0.612 [0.539, 0.649] | 0.450 [0.339, 0.554] | 0.230 [0.203, 0.254] |
| nodup | QD-DPP | 0.420 [0.402, 0.455] | 0.390 [0.366, 0.406] | 0.910 [0.875, 0.928] | 0.260 [0.202, 0.326] | 0.273 [0.243, 0.292] |
| dup | S-M | 0.262 [0.211, 0.291] | 0.310 [0.238, 0.360] | 0.473 [0.265, 0.580] | 0.574 [0.521, 0.647] | 0.131 [0.026, 0.208] |
| dup | QD-DPP | 0.387 [0.372, 0.400] | 0.386 [0.374, 0.398] | 0.940 [0.926, 0.950] | 0.336 [0.264, 0.477] | 0.252 [0.237, 0.264] |

Seed-paired differences (mean [95 % CI], 3 seeds):

| comparison | train | held-out RL | worst held-out cluster | exploiters |
|---|---|---|---|---|
| no floor - floor (D1, same seeds), S-M | **-0.288 [-0.494, -0.148]** | -0.107 [-0.330, +0.019] | -0.144 [-0.422, +0.005] | **+0.246 [+0.153, +0.403]** |
| no floor - floor (D1, same seeds), QD-DPP | **-0.297 [-0.372, -0.236]** | **-0.210 [-0.409, -0.067]** | -0.182 [-0.396, +0.005] | -0.139 [-0.401, +0.067] |
| dup - nodup, S-M | -0.022 [-0.057, +0.016] | **-0.053 [-0.080, -0.017]** | **-0.100 [-0.177, -0.026]** | +0.124 [+0.002, +0.309] |
| dup - nodup, QD-DPP | **-0.033 [-0.066, -0.002]** | -0.004 [-0.032, +0.020] | -0.021 [-0.055, +0.013] | +0.076 [-0.062, +0.275] |
| QD - S-M, nodup | **+0.137 [+0.129, +0.148]** | +0.027 [-0.011, +0.080] | **+0.042 [+0.009, +0.080]** | **-0.190 [-0.257, -0.085]** |
| QD - S-M, dup | **+0.126 [+0.081, +0.188]** | **+0.076 [+0.026, +0.161]** | **+0.122 [+0.048, +0.238]** | **-0.238 [-0.380, -0.044]** |

Sampling mass on the duplicated evader (6 copies of 25; uniform share 0.24): S-M 0.29-0.36, QD-DPP 0.16-0.20.
Mass on the 4 scripted training evaders (uniform share 0.20): 0.04-0.09 for both samplers.

- **Removing the uniform floor is harmful for both samplers**: train J drops by about 0.29 for both (CIs exclude 0);
  the pursuer is starved of the scripted training evaders (4-9 % of the mass) and even fails held-out scripted
  evaders (S-M 0.61, against 0.985 with the floor). Its J against the training RL evaders also drops (0.15-0.31 vs
  0.32-0.72 with the floor). The two D1c runs from 10-27, before any later code change, show the same collapse, so
  this is not a code-change artefact.
- **Without the floor, the DPP's duplicate robustness shows in performance** (the mechanism H1b predicted): duplication
  lowers S-M's held-out RL J by 0.053 and its worst-cluster J by 0.100 (CIs exclude 0), and leaves QD-DPP's
  unchanged (-0.004, -0.021, n.s.). QD-DPP keeps the duplicated evader's mass below its uniform share
  (0.16-0.20 vs 0.24); S-M inflates it (0.29-0.36).
- QD-DPP is better than S-M on train, held-out scripted and the worst cluster, and **worse on the X3 exploiters**
  (-0.19 / -0.24).
- Reading: H1b holds as a mechanism only when the hardness term controls most of the sampling mass, which is also
  the regime where hardness prioritisation is worst. With a 0.5 floor (D1b) the mechanism does not show. Exploratory:
  designed after D1b, 3 seeds.

## D2_exploitability (FINAL: 5 seeds, done 2026-10-28 01:05)

One fresh exploiter evader (PPO from scratch, 100M frames) per D1 pursuer; exploitability = 1 - J(pursuer, its
exploiter). [D2_exploitability/summary.md](D2_exploitability/summary.md).

| sampler | exploitability per seed (0-4) | mean [95 % CI] | capture vs exploiter per seed |
|---|---|---|---|
| S-U uniform | 0.697, 0.803, 0.513, 0.644, 0.705 | 0.672 [0.586, 0.745] | 0.25, 0.15, 0.45, 0.24, 0.26 |
| S-W win rate | 0.609, 0.551, 0.460, 0.574, **0.008** | 0.440 [0.219, 0.583] | 0.10, 0.13, 0.12, 0.37, 0.77 |
| S-M multi-metric | 0.830, 0.709, 0.495, 0.709, 0.713 | 0.691 [0.582, 0.782] | 0.15, 0.21, 0.45, 0.25, 0.25 |
| QD-DPP | 0.520, 0.794, 0.791, 0.559, 0.484 | 0.630 [0.513, 0.745] | 0.14, 0.15, 0.15, 0.12, 0.48 |

Seed-paired QD-DPP minus baseline: vs S-U -0.043 [-0.176, +0.129]; vs S-M -0.062 [-0.246, +0.132];
vs S-W +0.189 [+0.007, +0.371].

- **No sampler reduces exploitability reliably.** QD-DPP vs uniform and vs S-M: differences not significant.
- S-W's lower mean is driven by seed 4, where the exploiter **failed to learn** (exploitability 0.008, pursuer
  captures 77 %). Without that seed S-W's mean is 0.549. The QD-DPP vs S-W difference (CI barely excluding 0)
  therefore says more about one failed exploiter than about the samplers.
- Every pursuer remains highly exploitable (typically 0.5-0.8); exploiters win by evading until timeout.
- Limitation: one exploiter per pursuer. Exploiter training itself is noisy (one of 20 failed), so the measure has
  extra variance beyond the pursuer seeds.
- Prediction "QD-DPP lowest exploitability, maybe within CI": QD-DPP has the second-lowest mean; differences are
  within the CIs except vs S-W (opposite direction, driven by a failed exploiter). **Did not hold.**

## A1a_basis (done 2026-10-27 05:22; offline, open-loop)

Trajectories of the X3 generalist (seed 0) vs the train pool and vs the test set (1,024 evenly spaced envs, ~3,000
steps each). Target: evader relative displacement over H steps. The learned basis (BasisNet, 32 features, per-evader
heads + shared head) is trained on train-pool trajectories (80 % of envs). RLS (delta, forgetting) is tuned per
feature kind and H on the other 20 % of train-pool envs; the test set is not used for tuning.
Full: [A1a_basis/summary.md](A1a_basis/summary.md); tuned values: `A1a_basis/tuned_rls.json`.

**Run 1 (untuned, pre-registered defaults delta 10, forgetting 0.98) failed**: NF+RLS was worse than the frozen
basis (H 15 test: 0.198 vs 0.038 m) and diverged when kept across episodes (S2: 2.24 m), a covariance wind-up.
Kept as `summary_run1_untuned.md`. Fix: bounded-covariance RLS (trace <= initial trace) + tuning on train-pool
validation envs (plan deviation, §9).

Run 2, mean |y - y_hat| (m) on the **test** set (held-out RL, scripted, exploiters):

| H (steps / s) | CV (const. velocity) | raw RLS | ESN RLS | NF frozen | NF RLS (S1) | NF RLS (S2) |
|---|---|---|---|---|---|---|
| 5 / 0.1 | 0.0098 | 0.0083 | 0.0093 | 0.0044 | 0.0039 | **0.0037** |
| 15 / 0.3 | 0.0740 | 0.0899 | 0.0883 | 0.0376 | 0.0373 | **0.0354** |
| 25 / 0.5 | 0.1799 | 0.2169 | 0.2177 | 0.0984 | 0.0956 | **0.0889** |

- The **learned basis** is the main effect: NF frozen has 45-55 % lower error than CV at every horizon. The random
  reservoir (ESN) and raw-history RLS are worse than CV for H >= 15.
- **Online RLS on top of the basis** adds a small, consistent improvement on held-out evaders: S1 -1 % to -11 %,
  S2 (readout kept across episodes against the same evader) -6 % to -16 % vs NF frozen.
- Pre-registered horizon rule -> **H\* = 5** (largest gain over CV: +0.607).
- Error grows with episode index for every method. This is a selection effect (envs reaching a high episode index
  are those with short episodes), not evidence for or against adaptation.
- Predictions: "NF+RLS < ESN+RLS by 30-50 % at 0.3 s on held-out" **held** (0.0373 vs 0.0883, -58 %, larger than
  predicted); "NF frozen in between" **did not hold** (NF frozen is nearly as good as NF+RLS); "CV good at 0.1 s"
  partially (CV beats ESN at 0.1 s, but NF is 2x better).

## A1b_policy (FINAL: 5 seeds, done 2026-10-28 09:58)

Pursuers on the train pool (uniform, 150M frames) with the adapter output appended to the observation: predicted
evader displacement over H* = 5 steps (0.1 s), tuned RLS. S1: adapter reset every episode; S2: adapter kept across
the episodes of an env (same evader). M0 = X3 generalists (5 seeds). ESO_O2a = X4 acceleration-input pursuers
(3 seeds here; the 5-seed version is F1 "SU + acc"). Mean J, 95 % bootstrap CI over seeds
([A1b_policy/summary.md](A1b_policy/summary.md)):

| method | train | held-out RL | held-out scripted | exploiters* |
|---|---|---|---|---|
| M0 | 0.722 [0.625, 0.813] | 0.634 [0.447, 0.824] | 0.997 | 0.497 [0.306, 0.750] |
| ESO_O2a (3 seeds) | **0.939** [0.915, 0.958] | **0.773** [0.461, 0.933] | 0.992 | **0.973** [0.933, 0.996] |
| M0h frame stack | 0.468 [0.446, 0.490] | 0.346 [0.329, 0.370] | 0.998 | 0.614 [0.514, 0.713] |
| ESN S1 | 0.577 [0.508, 0.646] | 0.362 [0.312, 0.409] | 0.996 | 0.333 [0.266, 0.395] |
| ESN S2 | (same policy) | 0.366 [0.316, 0.414] | 0.996 | 0.333 [0.267, 0.391] |
| NF frozen | 0.743 [0.711, 0.775] | 0.481 [0.401, 0.585] | 0.997 | 0.429 [0.266, 0.620] |
| NF-OA S1 | 0.765 [0.736, 0.795] | 0.606 [0.441, 0.781] | 0.997 | 0.549 [0.319, 0.778] |
| NF-OA S2 | (same policy) | 0.604 [0.441, 0.784] | 0.997 | 0.544 [0.311, 0.777] |

\* exploiters were trained against M0 seed 0.

Seed-paired differences (mean [95 % CI]):

| comparison | train | held-out RL | exploiters |
|---|---|---|---|
| NF-OA - M0 | +0.044 [-0.062, +0.155] | -0.028 [-0.312, +0.281] | +0.052 [-0.324, +0.366] |
| NF frozen - M0 | +0.021 [-0.081, +0.115] | -0.153 [-0.372, +0.045] | -0.068 [-0.429, +0.279] |
| NF-OA - NF frozen (online adaptation) | +0.023 [-0.027, +0.073] | +0.125 [-0.117, +0.389] | +0.120 [-0.211, +0.391] |
| NF-OA S2 - S1 (across episodes) | 0.000 | -0.002 [-0.008, +0.006] | -0.005 [-0.013, +0.002] |
| ESN - M0 | **-0.145 [-0.260, -0.029]** | **-0.272 [-0.458, -0.085]** | -0.164 [-0.428, +0.040] |
| frame stack - M0 | **-0.254 [-0.355, -0.152]** | **-0.288 [-0.482, -0.115]** | +0.117 [-0.058, +0.293] |

- With 5 seeds, **neither NF variant is better than M0** (all CIs include 0). The 3-seed reading "the learned basis
  helps a little" did not survive (seeds 3-4 of M0 are strong).
- **Online adaptation (NF-OA vs NF frozen) is not significant**; across-episode adaptation (S2) changes J by <= 0.005.
- **ESN and the raw frame stack are significantly worse than M0** on train and held-out RL.
- The explicit acceleration input remains far better than every adapter (see F1 for its 5-seed numbers).
- J vs episode index rises for every method, M0 included (episode-length selection effect, not adaptation).
- Predictions: "NF-OA closes 20-40 % of Delta in S2": **did not hold**. "ESN about 0": **did not hold** (worse than
  M0). H2c (NF-OA beats ESN): **supported** (held-out RL 0.606 vs 0.362). H2d (gain grows with experience, vanishes
  when frozen): **not supported**.
- Design caveat: at H* = 0.1 s the predicted displacement is dominated by relative velocity x 0.1 s, which the policy
  already observes; A2 retargets the same machinery to the evader's acceleration.

## A2a_nfacc (done 2026-10-27 21:19; offline, open-loop)

Learned estimator of the evader's acceleration: same basis + RLS machinery as A1a, new target = change of the evader's
own velocity over the next H steps (H dt x mean acceleration). Same trajectories as A1a; RLS tuned on the train-pool
validation envs (20 % of envs not used to train the basis). [A2a_nfacc/summary.md](A2a_nfacc/summary.md).
Mean error (m/s); "FD persistence" = last-step finite-difference acceleration held for H steps (the offline analogue
of the O2a input); "zero" = constant velocity.

| H (steps / s) | set | zero | FD persistence | NF frozen | NF RLS (S1) | NF RLS (S2) |
|---|---|---|---|---|---|---|
| 1 / 0.02 | val | 0.0363 | **0.0045** | 0.0076 | 0.0054 | 0.0053 |
| 1 / 0.02 | test | 0.0371 | **0.0055** | 0.0088 | 0.0062 | 0.0062 |
| 5 / 0.1 | val | 0.1773 | 0.0522 | 0.0545 | 0.0501 | **0.0471** |
| 5 / 0.1 | test | 0.1803 | 0.0643 | 0.0679 | 0.0603 | **0.0561** |
| 10 / 0.2 | val | 0.3401 | 0.1599 | 0.1442 | 0.1377 | **0.1313** |
| 10 / 0.2 | test | 0.3396 | 0.1885 | 0.1744 | 0.1568 | **0.1488** |

- For one step ahead the finite difference is essentially exact and the learned estimator is worse (+13 % to +60 %).
- For 0.1-0.2 s ahead the learned estimator beats FD persistence; with online RLS the gain grows: at H = 10 on the
  test set, frozen -7 %, RLS S1 -17 %, RLS S2 (kept across episodes) -21 %.
- Here **online adaptation helps clearly** over the frozen basis (H = 10 test: -10 % S1, -15 % S2), more than for
  the displacement target in A1a.
- Pre-registered rule -> **H\* = 10** (val gain over FD: H=1 -0.198, H=5 +0.040, H=10 +0.139).
- Prediction "learned beats FD for H >= 5, small RLS gain": held at H = 10, marginal at H = 5 (S1 -6 %); the RLS
  gain was larger than "small" (10-15 %).

## A2b_nfacc (done 2026-10-28 11:04; 3 seeds; seeds 3-4 queued)

Pursuers on the train pool (uniform, 150M frames, noise-free) whose observation includes the **learned estimate of
the evader's mean acceleration over the next 0.2 s** (A2a, H* = 10), scaled exactly like the finite-difference input
O2a. Frozen basis vs basis + online RLS (tuned). [A2b_nfacc/summary.md](A2b_nfacc/summary.md). Mean J, 95 % bootstrap
CI over 3 seeds:

| method | train | held-out RL | held-out scripted | exploiters* |
|---|---|---|---|---|
| M0 | 0.650 [0.562, 0.748] | 0.473 [0.369, 0.604] | 0.998 | 0.309 [0.295, 0.324] |
| finite-difference acceleration (O2a) | **0.939** [0.915, 0.958] | 0.773 [0.461, 0.933] | 0.992 | 0.973 [0.933, 0.996] |
| NF-OA displacement target (A1b) | 0.764 [0.732, 0.811] | 0.595 [0.435, 0.887] | 0.997 | 0.515 [0.276, 0.878] |
| **NF-acc frozen** | 0.878 [0.857, 0.914] | **0.881** [0.801, 0.943] | 0.997 | **0.992** [0.985, 0.996] |
| NF-acc + RLS (S1) | 0.901 [0.836, 0.955] | 0.817 [0.737, 0.889] | 0.997 | 0.983 [0.966, 0.993] |
| NF-acc + RLS (S2) | (same policy) | 0.816 [0.739, 0.888] | 0.997 | 0.984 [0.968, 0.993] |

\* exploiters were trained against M0 seed 0.

Seed-paired differences (mean [95 % CI], 3 seeds):

| comparison | train | held-out RL | exploiters |
|---|---|---|---|
| NF-acc + RLS - M0 | **+0.251 [+0.164, +0.394]** | **+0.344 [+0.285, +0.378]** | **+0.673 [+0.658, +0.693]** |
| NF-acc frozen - O2a | **-0.062 [-0.101, -0.032]** | +0.108 [-0.025, +0.341] | +0.018 [-0.001, +0.053] |
| NF-acc + RLS - O2a | -0.039 [-0.110, +0.040] | +0.044 [-0.196, +0.429] | +0.009 [-0.029, +0.056] |
| NF-acc + RLS - NF-acc frozen (online adaptation) | +0.023 [-0.078, +0.092] | -0.065 [-0.206, +0.088] | -0.009 [-0.028, +0.003] |
| S2 - S1 | 0.000 | -0.001 | +0.001 |

Held-out v10 evaders s1/s3/s5/s7/s9/s11 per run (S1):
NF-acc frozen 0.99 0.94 0.57 0.66 0.80 0.85 | 1.00 0.97 0.79 0.84 0.88 0.92 | 0.99 0.96 0.90 0.88 0.97 0.94;
NF-acc + RLS 0.99 0.95 0.74 0.84 0.91 0.91 | 0.99 0.95 0.71 0.66 0.84 0.80 | 1.00 0.97 0.52 0.62 0.63 0.69.

- Retargeting the learned basis from displacement (A1b) to the evader's acceleration turns it from "no gain" into a
  **large gain over M0** (+0.25 train, +0.34 held-out RL, +0.67 exploiters).
- The learned estimate is **on par with the finite-difference input** in this noise-free setting: slightly lower on
  train (frozen -0.06, significant), higher on held-out RL (+0.11, not significant), equal on exploiters.
- None of the 6 NF-acc runs falls into the "fails on the later v10 lineage" mode: the lowest value on v10 s5-s11 is
  0.52. With the finite-difference input 3 of 10 runs (F1) fail there (0.10-0.29). Suggestive, not significant
  (0/6 vs 3/10, Fisher exact p ~= 0.25).
- **Online adaptation again adds nothing** (RLS vs frozen n.s.; S2 = S1), although it reduced the open-loop
  prediction error by 10-15 % in A2a.
- Prediction "close to O2a (within 0.05) noise-free; RLS gain small": held on held-out RL and exploiters, slightly
  outside on train for the frozen variant (-0.062); RLS gain small/absent **held**.

## A3a_posonly (done 2026-10-28 18:46; offline, open-loop)

Can a learned estimator recover the evader's acceleration **without observing its velocity**? Same X3-generalist
trajectories (seed 0, 1,024 envs, re-recorded with the pursuer's own position). Estimator input: 10-step window of
[relative position, own position, own velocity]; target = change of the evader's velocity over the next H steps;
frozen basis (online RLS would need the unobservable evader velocity as its target). Two bases: trained on clean
inputs ("clean") or on inputs with the sensor-noise model added ("aug"). Baselines: zero (constant velocity) and FD =
second difference of the evader position, held for H steps, raw or EMA-filtered (alpha tuned on noisy val).
Noise model = `observation.noise` stds: relative and own position 0.02 m, own velocity 0.05 m/s.
[A3a_posonly/summary.md](A3a_posonly/summary.md). Mean error (m/s):

| H (steps / s) | set | inputs | zero | FD | FD + EMA | NF clean | NF aug |
|---|---|---|---|---|---|---|---|
| 5 / 0.1 | val | clean | 0.177 | **0.057** | - | 0.127 | 0.130 |
| 5 / 0.1 | val | noisy | 0.177 | 27.6 | 0.847 | 0.153 | **0.132** |
| 5 / 0.1 | test | clean | 0.180 | **0.071** | - | 0.153 | 0.155 |
| 5 / 0.1 | test | noisy | 0.180 | 27.6 | 0.849 | 0.180 | **0.157** |
| 10 / 0.2 | val | clean | 0.340 | **0.167** | - | 0.241 | 0.247 |
| 10 / 0.2 | val | noisy | 0.340 | 55.3 | 1.696 | 0.281 | **0.249** |
| 10 / 0.2 | test | clean | 0.340 | **0.197** | - | 0.288 | 0.290 |
| 10 / 0.2 | test | noisy | 0.340 | 55.3 | 1.699 | 0.331 | **0.293** |

- **Noise-free, finite differences of position win clearly**: FD error is 2.2x (H=5) and 1.5x (H=10) lower than the
  learned estimator. Without the velocity input the learned estimator is much worse than A2a's velocity-based one
  (H=10 val 0.241 vs 0.144), while position-FD is about as good as velocity-FD (0.167 vs 0.160).
- **With sensor noise, position FD is useless**: 2 cm position noise differentiated twice at 50 Hz gives errors of
  28-55 m/s. The best EMA (alpha 0.05, the smallest value in the pre-registered grid, so the grid edge) is still
  4-5x worse than predicting zero.
- The **noise-augmented learned estimator is the only method better than zero under noise** (-26 % val, -13 to
  -14 % test). The clean-trained basis degrades under noise (+20 % val error); the augmented basis costs only 2 %
  on clean inputs.
- Pre-registered choices: H\* = 10 (both "gains vs FD" are negative, -1.22 at H=5 and -0.44 at H=10; the rule
  takes the larger), basis = aug, FD EMA alpha = 0.05. A3b uses these.
- Prediction "learned beats FD clearly with noise, roughly equal without": the noisy part **held**; the noise-free
  part **did not hold** (FD is clearly better).
- Run 1 was invalid: the trajectory recorder stored positions as float16 (~2 mm steps at 2-3 m), which inflated
  the noise-free FD error to 0.26-0.56 m/s (worse than zero). Re-recorded in float32 (run 1 kept in
  `A3a_posonly/run1_fp16/`; the learned-estimator errors were unchanged, and the pre-registered choices were the
  same in both runs). A1a / A2a used the same float16 recorder; their inputs are velocity-based, where the effect
  is much smaller (e.g. velocity FD one step ahead, 0.0045 m/s in A2a, is partly quantisation), and they were not
  re-run.

## F1_final (done 2026-10-28 04:22; 5 seeds, noise-free)

2x2 factorial {uniform (SU), QD-DPP (QD) sampler} x {plain obs, opponent-acceleration input (acc, finite difference
as in X4 O2a)}, train pool, 150M frames. Reused runs: SU_plain = X3 generalists, QD_plain = D1 QD, SU_acc seeds 0-2 =
X4 O2a. [F1_final/summary.md](F1_final/summary.md). Mean J (95 % bootstrap CI over 5 seeds):

| cell | train | held-out RL (v10) | held-out scripted | exploiters* |
|---|---|---|---|---|
| SU + plain | 0.722 [0.625, 0.813] | 0.634 [0.447, 0.824] | 0.997 | 0.497 [0.306, 0.750] |
| QD + plain | 0.752 [0.689, 0.812] | 0.676 [0.533, 0.846] | 0.996 | 0.496 [0.298, 0.750] |
| SU + acc | **0.940** [0.926, 0.951] | **0.838** [0.645, 0.942] | 0.993 | **0.983** [0.958, 0.997] |
| QD + acc | 0.933 [0.909, 0.959] | 0.730 [0.536, 0.921] | 0.993 | 0.937 [0.864, 0.988] |

\* exploiters were trained against SU + plain seed 0.

Seed-paired effects (mean [95 % CI]):

| effect | train | held-out RL | exploiters |
|---|---|---|---|
| acceleration input (main effect) | **+0.199 [+0.128, +0.270]** | +0.129 [-0.106, +0.330] | **+0.464 [+0.239, +0.638]** |
| sampler QD - SU (main effect) | +0.012 [-0.022, +0.066] | -0.033 [-0.121, +0.052] | -0.023 [-0.101, +0.042] |
| interaction | -0.037 [-0.084, +0.008] | -0.150 [-0.386, +0.041] | -0.045 [-0.270, +0.157] |
| QD - SU, plain obs | +0.031 [-0.020, +0.104] | +0.042 [-0.069, +0.131] | -0.001 |
| QD - SU, acc input | -0.006 [-0.039, +0.030] | -0.108 [-0.295, -0.005] | -0.046 [-0.094, -0.009] |

Held-out RL is **bimodal** for the acceleration-input policies. Each seed either handles the later v10 evaders
(s5-s11: J 0.82-0.97) or fails on all of them (J 0.10-0.29):

| seed | SU + acc on v10 s1/s3/s5/s7/s9/s11 | QD + acc on v10 s1/s3/s5/s7/s9/s11 |
|---|---|---|
| 0 | 0.99 0.99 **0.14 0.29 0.11 0.24** | 0.95 0.97 **0.10 0.25 0.21 0.14** |
| 1 | 0.99 0.93 0.88 0.89 0.91 0.95 | 0.99 0.95 0.87 0.90 0.96 0.95 |
| 2 | 0.99 0.94 0.87 0.92 0.93 0.95 | 0.99 0.99 **0.10 0.29 0.11 0.25** |
| 3 | 0.99 0.94 0.92 0.95 0.97 0.97 | 0.99 0.95 0.86 0.87 0.92 0.94 |
| 4 | 0.99 0.96 0.82 0.89 0.89 0.92 | 0.99 0.97 0.74 0.88 0.92 0.90 |

- **The acceleration input is the dominant factor**: +0.20 on train and +0.46 on exploiters (CIs exclude 0).
- **The sampler has no main effect.** With the plain observation QD-DPP is +0.03-0.04 (n.s., as in D1). With the
  acceleration input QD-DPP is slightly lower; the held-out RL CI excludes 0 (-0.108), but this comes from one extra
  seed in the failing mode (QD 2/5 vs SU 1/5 seeds), which is not meaningful with 5 seeds.
- Once the opponent's acceleration is observable, diversity-aware sampling brings no measurable benefit in this
  setting.
- Generalisation to the v10 lineage is a seed-level, all-or-nothing property: about 1 in 3-5 runs fails on it,
  regardless of sampler.
