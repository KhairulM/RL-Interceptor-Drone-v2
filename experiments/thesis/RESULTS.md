# Results (thesis experiments)

Measured results only. Each section is filled when the experiment finishes, with exact numbers, CIs, number of
seeds/episodes, and any deviation from [the plan](../../docs/research_plan.md). Predictions are in the plan and are
compared here as "prediction held / did not hold".

## Summary (updated 2026-10-29 18:50; details in the sections below)

J = neutralisation rate (capture, evader crash or evader out of bounds), mean over seeds; [ ] = 95 % bootstrap CI over
seeds; "supported" = paired CI excludes 0 (coarse with 3-5 seeds). M0 = plain pursuer (observes evader position +
velocity, no acceleration).

**Diagnostics (X0-X4)**
- The game is mostly transitive; self-play pools are redundant (22 RL evaders = 3.8 distinct behaviours).
- The pursuer loses up to 0.37 J to a specialist on unseen learned evaders. Most of this gap is missing opponent state:
  adding the evader's acceleration lifts J 0.65 -> 0.94 (train), 0.47 -> 0.77 (held-out RL), 0.31 -> 0.97 (exploiters).

**RQ1: opponent sampling**
- QD-DPP beats hardness-prioritised samplers (held-out RL +0.19 vs multi-metric, +0.13 vs win-rate PFSP) but **not
  uniform** (+0.04, n.s.); hardness-only prioritisation is worse than uniform (D1, 5 seeds).
- The DPP caps a duplicated opponent's sampling mass but does not protect performance (D1b); duplication robustness
  appears only without a uniform floor (D1c, exploratory).
- No sampler lowers exploitability (0.44-0.69, D2); with the acceleration input the sampler has no effect (F1).

**RQ2: opponent model / online adaptation**
- Noise-free: a learned acceleration estimate equals the finite difference (A2b, n.s.), and both beat M0 (+0.18 train,
  +0.23 held-out RL, +0.49 exploiters). Online RLS adds nothing (A1b, A2b); ESN and raw history are worse than none.
- Sensor noise (policies trained clean, A3c): the finite-difference input collapses (train 0.94 -> 0.33); learned +
  RLS holds (0.80) and beats it on every split (+0.20 to +0.49). **RLS beats the frozen estimator** (+0.09 train, +0.24
  exploiters): the only closed-loop gain from online adaptation so far.
- No evader velocity observed (A3b): every pursuer is weak (~0.29 vs 0.65); under noise the finite-difference pursuer
  crashes in ~92 % of episodes; the learned estimator is noise-proof but no better than no estimator. A3d (offline):
  a learned velocity estimate from noisy positions is 35-50 % more accurate than an EMA filter.

**Final runs**
- F1 (5 seeds): the acceleration input dominates (train +0.20, exploiters +0.46); held-out results are bimodal per seed.
- F2a (5 seeds): the acceleration input lowers exploitability 0.672 -> 0.397 (paired -0.275 [-0.522, -0.044]).

**Running / pending**: A3d closed loop (since 18:15) -> F4 v2 -> F3 -> F2c / F4c / F3c. F2b campaigns are done and
are compared in F2c.

## Relation to the reference papers (updated 2026-10-29 11:10)

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
| 6 | Observation design: Roncero and Gavin give the pursuer opponent position + velocity to a memoryless MLP; Gavin leaves "trajectory prediction" out of scope; AgilePE adds a temporal window of opponent state. | Adding the opponent's acceleration (finite difference of two consecutive observations) is the **largest effect in this thesis**: train 0.722 -> 0.940 (+0.199 [+0.128, +0.270]), exploiters 0.497 -> 0.983 (+0.464 [+0.239, +0.638]), held-out RL 0.634 -> 0.838 (CI includes 0 because of the bimodal seeds) (F1, 5 seeds). A learned acceleration estimate does the same (A2b, 5 seeds: all paired differences to the finite difference within +/- 0.04, n.s.). It also lowers exploitability: 0.672 -> 0.397, paired -0.275 [-0.522, -0.044] (F2a). A raw 10-step history stack (our analogue of a temporal window) was *worse* than no history (train -0.254 [-0.355, -0.152]) (A1b). | **New:** a memoryless pursuer with position + velocity (Roncero, Gavin) is limited by the missing opponent acceleration far more than by opponent sampling. Caveat: AgilePE's window length and architecture are not given, so the history-stack result is not a test of AgilePE itself. |
| 7 | Future work in Roncero: online adaptation, recurrent policies. | Noise-free, online RLS adaptation of a learned opponent model adds nothing in closed loop (A1b, A2b; within or across episodes); a random reservoir (ESN) is worse than no adaptation. **Under sensor noise it helps**: RLS vs frozen basis +0.09 train, +0.24 exploiters, CIs exclude 0 (A3c, policies trained noise-free). Recurrent policies (GRU) **not tested**. | **New, conditional:** online opponent-model adaptation pays off under sensor noise, not with perfect state (Roncero's setting). |
| 8 | Sensor noise: AgilePE reports robustness under real sensor noise (mocap + IMU); Roncero and Gavin assume perfect state. | With 2 cm / 8 cm/s noise (eval only, policies trained clean): the finite-difference acceleration input collapses (train 0.94 -> 0.33, below the plain pursuer 0.70); the learned estimator + RLS keeps 0.80 and beats it by +0.47 train / +0.38 held-out RL / +0.49 exploiters (A3c, 5 seeds). Without any evader velocity (A3b), every pursuer is weak (~0.29) and the learned estimator is noise-proof but not better than no estimator. | **New:** opponent-acceleration inputs must be learned, not differenced, once sensing is noisy; the perfect-state results of Roncero and Gavin would not transfer to such inputs. Limited to eval-time noise until F4. |
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

## A2b_nfacc (done 2026-10-28 21:10; 5 seeds; 3-seed version 11:04 superseded)

Pursuers on the train pool (uniform, 150M frames, noise-free) whose observation includes the **learned estimate of
the evader's mean acceleration over the next 0.2 s** (A2a, H* = 10), scaled exactly like the finite-difference input
O2a. Frozen basis vs basis + online RLS (tuned). [A2b_nfacc/summary.md](A2b_nfacc/summary.md). Mean J, 95 % bootstrap
CI over 5 seeds:

| method | train | held-out RL | held-out scripted | exploiters* |
|---|---|---|---|---|
| M0 | 0.722 [0.625, 0.813] | 0.634 [0.447, 0.824] | 0.997 | 0.497 [0.306, 0.750] |
| finite-difference acceleration (O2a) | **0.940** [0.926, 0.951] | 0.838 [0.645, 0.942] | 0.993 | 0.983 [0.958, 0.997] |
| NF-OA displacement target (A1b) | 0.765 [0.736, 0.795] | 0.606 [0.441, 0.781] | 0.997 | 0.549 [0.319, 0.778] |
| **NF-acc frozen** | 0.909 [0.871, 0.948] | 0.838 [0.739, 0.917] | 0.996 | 0.977 [0.951, 0.993] |
| NF-acc + RLS (S1) | 0.903 [0.869, 0.936] | **0.859** [0.797, 0.918] | 0.996 | **0.987** [0.976, 0.994] |
| NF-acc + RLS (S2) | (same policy) | 0.859 [0.799, 0.917] | 0.996 | 0.988 [0.978, 0.995] |

\* exploiters were trained against M0 seed 0.

Seed-paired differences (mean [95 % CI], 5 seeds):

| comparison | train | held-out RL | exploiters |
|---|---|---|---|
| NF-acc + RLS - M0 | **+0.181 [+0.093, +0.294]** | **+0.225 [+0.091, +0.355]** | **+0.490 [+0.243, +0.677]** |
| NF-acc frozen - M0 | **+0.188 [+0.111, +0.264]** | +0.204 [-0.060, +0.457] | **+0.480 [+0.199, +0.685]** |
| NF-acc frozen - O2a | -0.030 [-0.068, +0.007] | +0.001 [-0.167, +0.187] | -0.006 [-0.044, +0.028] |
| NF-acc + RLS - O2a | -0.037 [-0.077, +0.009] | +0.021 [-0.133, +0.246] | +0.004 [-0.018, +0.033] |
| NF-acc + RLS - NF-acc frozen (online adaptation) | -0.007 [-0.066, +0.053] | +0.021 [-0.121, +0.181] | +0.010 [-0.014, +0.042] |
| S2 - S1 | 0.000 | -0.000 | +0.001 |

Held-out RL per seed: O2a 0.46 0.93 0.93 0.96 0.91; NF-acc frozen 0.80 0.90 0.94 0.90 0.65; NF-acc + RLS 0.89 0.82
0.74 0.89 0.96. Held-out v10 evaders s1/s3/s5/s7/s9/s11 per run (S1):
NF-acc frozen 0.99 0.94 0.57 0.66 0.80 0.85 | 1.00 0.97 0.79 0.84 0.88 0.92 | 0.99 0.96 0.90 0.88 0.97 0.94 |
0.99 0.94 0.80 0.85 0.87 0.93 | 0.99 0.98 0.34 0.60 0.52 0.48;
NF-acc + RLS 0.99 0.95 0.74 0.84 0.91 0.91 | 0.99 0.95 0.71 0.66 0.84 0.80 | 1.00 0.97 0.52 0.62 0.63 0.69 |
0.98 0.94 0.78 0.80 0.91 0.89 | 0.99 0.94 0.95 0.93 0.97 0.98.

- Retargeting the learned basis from displacement (A1b) to the evader's acceleration turns it from "no gain" into a
  **large gain over M0** (with RLS: +0.18 train, +0.23 held-out RL, +0.49 exploiters, all CIs exclude 0; frozen: same
  size, held-out RL CI includes 0 because of seed 4).
- The learned estimate is **equivalent to the finite-difference input** in this noise-free setting: every paired
  difference is within +/- 0.04 in the mean and every CI includes 0. (The 3-seed "frozen is lower on train,
  -0.06 significant" no longer holds with 5 seeds: -0.030 [-0.068, +0.007].)
- Failure mode on the later v10 lineage: with the finite-difference input 3 of 10 runs (F1) collapse there
  (0.10-0.29). With the learned estimate none of 10 runs collapses that far, but one (frozen seed 4) is degraded
  (0.34-0.60). Fewer severe failures, but the 3-seed statement "no failures" is weakened; not significant.
- **Online adaptation again adds nothing** (RLS vs frozen n.s. on every split; S2 = S1), although it reduced the
  open-loop prediction error by 10-15 % in A2a.
- Prediction "close to O2a (within 0.05) noise-free; RLS gain small": **held** on all splits with 5 seeds; RLS
  gain small/absent **held**.

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

## A3b_posonly (done 2026-10-29 02:15; 3 seeds; closed loop)

Pursuers whose observation has **no evader velocity** (`observation.agent_rel_vel=false`; relative position only),
train pool, uniform, 150M frames, noise-free training. P_none: nothing added. P_fd: second difference of the evader
position (`oracle.accel_source=pos`). P_nf: frozen learned estimator from A3a (aug basis, H = 10, 10-step window of
relative position / own position / own velocity). Note: both estimators give the evader's **acceleration**, not its
velocity; the policy is memoryless. Eval noise-free ("clean") and with the sensor-noise model ("noisy"); P_fd also
with the tuned EMA ("noisyema"; 3 seeds, third seed added 10-29 18:40).
[A3b_posonly/summary.md](A3b_posonly/summary.md). Mean J, 95 % bootstrap CI over 3 seeds:

| policy / eval | train | held-out RL | held-out scripted | exploiters* |
|---|---|---|---|---|
| M0 (velocity observed, ref., clean) | 0.650 [0.562, 0.748] | 0.473 [0.369, 0.604] | 0.998 | 0.309 [0.295, 0.324] |
| P_none clean | 0.289 [0.280, 0.300] | 0.296 [0.244, 0.331] | 0.866 [0.824, 0.914] | 0.374 [0.256, 0.528] |
| P_none noisy | 0.288 [0.283, 0.293] | 0.291 [0.239, 0.322] | 0.863 [0.825, 0.907] | 0.381 [0.269, 0.530] |
| P_fd clean | **0.378** [0.342, 0.413] | 0.215 [0.200, 0.223] | **0.984** [0.982, 0.987] | 0.268 [0.261, 0.281] |
| P_fd noisy | 0.073 [0.043, 0.113] | 0.084 [0.042, 0.140] | 0.086 [0.073, 0.099] | 0.343 [0.221, 0.470] |
| P_fd noisy + EMA | 0.254 [0.248, 0.260] | 0.162 [0.152, 0.181] | 0.874 [0.819, 0.907] | 0.341 [0.278, 0.378] |
| P_nf clean | 0.303 [0.269, 0.321] | 0.269 [0.252, 0.278] | 0.902 [0.850, 0.936] | 0.233 [0.222, 0.253] |
| P_nf noisy | 0.302 [0.269, 0.321] | 0.263 [0.255, 0.268] | 0.898 [0.852, 0.928] | 0.229 [0.222, 0.239] |

\* exploiters were trained against M0 seed 0 (a velocity-observing pursuer).

Capture rate / pursuer crash rate (mean over 3 seeds; added 10-29 18:40 from the same eval files). J also counts
evader crashes and evader out-of-bounds, so J can be well above the capture rate:

| policy / eval | train | held-out RL | held-out scripted | exploiters |
|---|---|---|---|---|
| P_none clean | 0.177 / 0.010 | 0.187 / 0.009 | 0.676 / 0.016 | 0.026 / 0.005 |
| P_none noisy | 0.178 / 0.010 | 0.186 / 0.010 | 0.678 / 0.017 | 0.027 / 0.004 |
| P_fd clean | 0.286 / 0.013 | 0.132 / 0.011 | 0.796 / 0.014 | 0.076 / 0.008 |
| P_fd noisy | **0.001 / 0.928** | **0.001 / 0.917** | **0.001 / 0.915** | 0.001 / 0.664 |
| P_fd noisy + EMA | 0.168 / 0.014 | 0.085 / 0.014 | 0.627 / 0.021 | 0.139 / 0.016 |
| P_nf clean | 0.198 / 0.004 | 0.171 / 0.004 | 0.700 / 0.004 | 0.035 / 0.002 |
| P_nf noisy | 0.198 / 0.004 | 0.170 / 0.005 | 0.698 / 0.004 | 0.038 / 0.001 |

- The P_fd noisy collapse is a **crash** failure, not a missed-intercept failure: the pursuer crashes in ~92 % of
  episodes and captures almost never (0.001). Its J of 0.34 against the exploiters therefore comes from evader
  crashes / out-of-bounds, not from captures.

Seed-paired differences (mean [95 % CI], 3 seeds):

| comparison | train | held-out RL | held-out scripted | exploiters |
|---|---|---|---|---|
| P_nf - P_none (clean) | +0.014 [-0.031, +0.041] | -0.027 [-0.062, +0.032] | +0.036 [-0.064, +0.095] | **-0.140 [-0.302, -0.003]** |
| P_fd - P_none (clean) | **+0.089 [+0.062, +0.127]** | **-0.081 [-0.131, -0.021]** | **+0.119 [+0.070, +0.163]** | -0.105 [-0.267, +0.025] |
| P_nf - P_none (noisy) | +0.014 [-0.023, +0.037] | -0.028 [-0.056, +0.026] | +0.035 [-0.055, +0.090] | **-0.152 [-0.308, -0.030]** |
| P_nf - P_fd (noisy) | **+0.229 [+0.156, +0.277]** | **+0.178 [+0.125, +0.226]** | **+0.812 [+0.753, +0.855]** | -0.113 [-0.231, +0.007] |
| P_fd noisy - clean | **-0.305 [-0.350, -0.267]** | **-0.130 [-0.158, -0.083]** | **-0.898 [-0.910, -0.885]** | +0.074 [-0.043, +0.189] |
| P_nf noisy - clean | -0.001 [-0.004, +0.000] | -0.006 [-0.011, +0.003] | -0.003 [-0.009, +0.002] | -0.004 [-0.014, +0.006] |

- **Not observing the evader's velocity is crippling**: train J 0.29 vs 0.65 for M0 (seeds 0-2, not seed-paired runs
  but the same seeds/pool/frames). Neither acceleration estimate makes up for it: the policy is memoryless and gets
  acceleration but no velocity.
- **Under sensor noise, the position second difference destroys the policy** (train 0.38 -> 0.07; even the scripted
  evaders drop from 0.98 to 0.09). EMA filtering recovers part of it (0.25, still below P_none). The learned
  estimator is **unaffected by noise** (all noisy - clean differences <= 0.006) and beats FD under noise by +0.23
  train / +0.18 held-out RL (CIs exclude 0). This is the closed-loop confirmation of A3a's noise result.
- But the learned estimator **does not beat no estimator at all** (P_nf - P_none n.s. on train / held-out, and
  *worse* on the exploiters, -0.14 to -0.15, CI excludes 0). Noise-free, FD helps on train and scripted evaders
  (+0.09 / +0.12) but hurts on held-out RL (-0.08).
- Prediction "P_fd ~ P_nf noise-free; P_nf > P_fd with noise; both > P_none": "P_nf > P_fd with noise" **held**;
  "P_fd ~ P_nf noise-free" **did not hold** (FD higher on train/scripted, lower on held-out RL); "both > P_none"
  **did not hold**.
- Design limitation (found only now, stated plainly): A3b removes the evader velocity *and* offers only an
  acceleration estimate, so it tests "acceleration without velocity". It does not test a learned estimator that
  recovers the evader's velocity from noisy positions. The pre-registered F4 campaigns use the same inputs and
  therefore inherit this limitation (see PROGRESS.md, 10-29 02:40).

## A3c_noise_ref (done 2026-10-29 03:47; eval only, 5 seeds)

The velocity-observing policies (all **trained noise-free**) evaluated with the sensor-noise model
(`observation.include_noise=true`: 2 cm position, 8 cm/s relative velocity, IMU noise; the acceleration inputs are
computed from the noisy observations). [A3b_posonly/summary.md](A3b_posonly/summary.md) (shared table). Mean J, 95 %
bootstrap CI over 5 seeds:

| policy | train clean -> noisy | held-out RL clean -> noisy | held-out scripted noisy | exploiters clean -> noisy |
|---|---|---|---|---|
| M0 | 0.722 -> 0.695 | 0.634 -> 0.602 | 0.996 | 0.497 -> 0.408 |
| finite-difference acceleration | 0.940 -> **0.328** | 0.838 -> **0.342** | 0.797 | 0.983 -> 0.450 |
| NF-acc frozen | 0.909 -> 0.710 | 0.838 -> 0.646 | 0.976 | 0.977 -> 0.697 |
| NF-acc + RLS (S1) | 0.903 -> **0.796** [0.756, 0.844] | 0.859 -> **0.718** [0.584, 0.840] | 0.993 | 0.987 -> **0.941** [0.913, 0.962] |

Seed-paired differences under noise (mean [95 % CI], 5 seeds):

| comparison (noisy eval) | train | held-out RL | held-out scripted | exploiters |
|---|---|---|---|---|
| NF-acc + RLS - finite difference | **+0.468 [+0.360, +0.575]** | **+0.376 [+0.202, +0.557]** | **+0.197 [+0.117, +0.301]** | **+0.492 [+0.376, +0.610]** |
| NF-acc frozen - finite difference | **+0.382 [+0.346, +0.416]** | **+0.304 [+0.245, +0.360]** | **+0.179 [+0.097, +0.279]** | **+0.247 [+0.075, +0.404]** |
| NF-acc + RLS - M0 | +0.101 [-0.002, +0.233] | **+0.116 [+0.015, +0.217]** | -0.003 [-0.005, -0.001] | **+0.533 [+0.311, +0.686]** |
| NF-acc frozen - M0 | +0.015 [-0.084, +0.114] | +0.044 [-0.233, +0.298] | -0.021 [-0.033, -0.009] | +0.289 [-0.064, +0.499] |
| finite difference - M0 | **-0.367 [-0.492, -0.254]** | **-0.260 [-0.496, -0.038]** | **-0.199 [-0.303, -0.121]** | +0.042 [-0.181, +0.205] |
| **NF-acc + RLS - NF-acc frozen** (online adaptation) | **+0.086 [+0.008, +0.164]** | +0.072 [-0.134, +0.307] | **+0.018 [+0.007, +0.028]** | **+0.244 [+0.110, +0.395]** |

Degradation noisy - clean (same policy): finite difference train -0.611, held-out RL -0.496, exploiters -0.533;
NF-acc frozen -0.200 / -0.192 / -0.280; NF-acc + RLS -0.107 / -0.142 / -0.046; M0 -0.027 / -0.032 / -0.089 (all
CIs exclude 0).

- **The finite-difference acceleration input, the best input noise-free, becomes the worst under noise**: it falls
  below M0 (-0.37 train, -0.26 held-out RL) and even loses scripted evaders (0.80).
- **The learned estimator with online RLS is the most robust policy with an acceleration input** and the only one
  that stays above M0 under noise: held-out RL +0.12 and exploiters +0.53 over M0 (CIs exclude 0); train +0.10
  (CI touches 0). It beats the finite difference on every split by +0.20 to +0.49.
- **Online adaptation helps under noise** (RLS - frozen: train +0.09, exploiters +0.24, CIs exclude 0), the first
  closed-loop gain from online adaptation in this thesis (noise-free it added nothing: A1b, A2b). A plausible
  reason: the RLS readout is refit on the noisy inputs it actually receives, while the frozen basis was trained on
  clean ones. This is an interpretation, not tested.
- M0 itself is nearly noise-proof (-0.03), so the acceleration inputs are where noise robustness is decided.
- Caveat: all policies were trained without noise; policies trained with noise may learn to discount a noisy
  finite difference. F4 (self-play with noise in training) is where that is tested.
- Prediction "finite-difference acceleration degrades most": **held**.

## A3d_velest (offline done 2026-10-29 11:16; closed loop queued at the start of F4)

Can a learned estimator recover the evader's **velocity** from noisy positions? A3a trajectories; input = 10-step
window of [relative position, own position, own velocity] (no evader velocity); target = the evader's current world
velocity. [A3d_velest/summary_offline.md](A3d_velest/summary_offline.md). Mean error |v - v_hat| in m/s (evader speed
scale: predicting zero gives 0.87-0.90):

| set | inputs | first difference (raw) | first difference + EMA (alpha 0.1, tuned) | learned, clean basis | learned, noise-aug. basis |
|---|---|---|---|---|---|
| val | clean | **0.009** | 0.269 | 0.105 | 0.139 |
| val | noisy | 3.180 | 0.439 | 0.281 | **0.217** |
| test | clean | **0.010** | 0.264 | 0.126 | 0.161 |
| test | noisy | 3.181 | 0.435 | 0.292 | **0.232** |

Reference: the measured relative velocity of A3c (8 cm/s noise per axis) has error 0.128.

- Noise-free, differencing positions is almost exact (0.01 m/s); the learned estimator is worse (0.11-0.13).
- With 2 cm position noise, raw differencing is useless (3.2 m/s); the EMA filter brings it to 0.43-0.44 m/s
  (with lag). The learned estimator is **35 % (clean basis) to 50 % (noise-augmented basis) more accurate** than the
  tuned filter. Both are 2-3x worse than a measured velocity with A3c's noise level.
- Pre-registered basis rule (A3a's: "aug" only if better on noisy val *and* clean-val error <= 1.1x "clean"):
  "aug" is better on noisy val but its clean-val error is 1.32x, so the rule selects the **clean basis**. A3d's closed
  loop and F4 v2 use it as registered, although the augmented basis would be more accurate under noise.
- Prediction "NF_aug < FD_ema on noisy val/test; noise-free FD_raw best": **held**.

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
## F2a_exploit_acc (done 2026-10-29 06:19; 5 seeds)

Is the pursuer with the opponent-acceleration input harder to exploit? One fresh exploiter evader (PPO from
scratch, 100M frames, plain observation, seed 200 + s as in D2) against each F1 "uniform + acceleration" pursuer;
compared seed-paired with D2's uniform plain-observation pursuers (same seeds, same exploiter seeds).
[F2a_exploit_acc/summary.md](F2a_exploit_acc/summary.md).

| seed | plain exploitability (D2 S-U) | acc exploitability | acc: capture / timeout vs exploiter |
|---|---|---|---|
| 0 | 0.697 | 0.632 | 0.33 / 0.63 |
| 1 | 0.803 | 0.070 | 0.24 / 0.07 (exploiter mostly crashes / leaves the arena) |
| 2 | 0.513 | 0.569 | 0.37 / 0.53 |
| 3 | 0.644 | 0.439 | 0.49 / 0.42 |
| 4 | 0.705 | 0.276 | 0.70 / 0.27 |
| mean [95 % CI] | 0.672 [0.586, 0.745] | **0.397 [0.218, 0.568]** | |

Seed-paired acc - plain: **-0.275 [-0.522, -0.044]** (n = 5). Without seed 1, where the exploiter failed to learn to
fly safely (J 0.93 although it is captured only 24 % of the time): -0.161 [-0.338, -0.005] (n = 4).

- The acceleration input **lowers exploitability** (CI excludes 0, also without the failed exploiter), in addition
  to its gains on train / held-out / X3 exploiters (F1). It is the only intervention in this thesis that reduced
  exploitability; no sampler did (D2).
- The pursuer stays exploitable: a fresh exploiter still escapes neutralisation in 28-63 % of episodes on 4 of 5
  seeds, mostly by surviving to timeout.
- Prediction "lower than plain (0.672), mean 0.3-0.5, paired CI excludes 0; still clearly > 0": **held**
  (0.397; one seed's exploiter failed, as one of 20 did in D2).
- Bookkeeping: the driver's first summary showed 3 of 5 seeds (lane-lock bug); rebuilt from the eval files on disk.

