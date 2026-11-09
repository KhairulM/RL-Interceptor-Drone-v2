# Results (thesis experiments)

Measured results only. Each section is filled when the experiment finishes, with exact numbers, CIs, number of
seeds/episodes, and any deviation from [the plan](../../docs/research_plan.md). Predictions are in the plan and are
compared here as "prediction held / did not hold".

## Summary (updated 2026-11-09 11:54; details in the sections below)

J = neutralisation rate (capture, evader crash or evader out of bounds), mean over seeds; [ ] = 95 % bootstrap CI over
seeds; "supported" = paired CI excludes 0 (coarse with 3-5 seeds). M0 = plain pursuer (observes evader position +
velocity, no acceleration).

**Diagnostics (X0-X4)**
- The game is mostly transitive; self-play pools are redundant (22 RL evaders = 3.8 distinct behaviours).
- The pursuer loses up to 0.37 J to a specialist on unseen learned evaders. Most of this gap is missing opponent state:
  adding the evader's acceleration lifts J 0.65 -> 0.94 (train), 0.47 -> 0.77 (held-out RL), 0.31 -> 0.97 (exploiters).

**RQ1: opponent sampling**
- Fixed pool (D1, 5 seeds): QD-DPP beats hardness-prioritised samplers (held-out RL +0.19 vs multi-metric, +0.13 vs
  win-rate PFSP) but **not uniform** (+0.04, n.s.); hardness-only prioritisation is worse than uniform there.
- The DPP caps a duplicated opponent's sampling mass but does not protect performance (D1b); duplication robustness
  appears only without a uniform floor (D1c, exploratory).
- No sampler lowers exploitability (0.44-0.69, D2); with the acceleration input the sampler has no effect (F1).
- **Self-play, samplers on both roles (F3, 2 seeds, descriptive)**: **Hedge (Roncero) is best**: held-out RL +0.10 and
  worst held-out cluster +0.16 vs uniform on both seeds (opposite of the fixed-pool D1 result). QD-DPP is not better
  than uniform (-0.04 / -0.05) and below Hedge. PFSP yields the most diverse evader populations (Vendi 2.99 vs 2.46)
  but the most specialised pursuer (other campaigns' evaders -0.23). Every self-play pursuer stays ~fully exploitable
  (0.97-0.98). **Answer to RQ1: not robustly. QD-DPP beats badly tuned hardness prioritisation on a fixed pool (D1),
  but is not better than uniform (D1, F3) and is below Roncero's Hedge in self-play (F3, 2 seeds).**

**RQ2: opponent model / online adaptation**
- Noise-free: a learned acceleration estimate equals the finite difference (A2b, n.s.), and both beat M0 (+0.18 train,
  +0.23 held-out RL, +0.49 exploiters). Online RLS adds nothing (A1b, A2b); ESN and raw history are worse than none.
- Sensor noise (policies trained clean, A3c): the finite-difference input collapses (train 0.94 -> 0.33); learned +
  RLS holds (0.80) and beats it on every split (+0.20 to +0.49). **RLS beats the frozen estimator** (+0.09 train, +0.24
  exploiters): the only closed-loop gain from online adaptation so far.
- No evader velocity observed (A3b): every pursuer is weak (~0.29 vs 0.65); under noise the finite-difference pursuer
  crashes in ~92 % of episodes; the learned estimator is noise-proof but no better than no estimator. A3d (offline):
  a learned velocity estimate from noisy positions is 35-50 % more accurate than an EMA filter. A3d closed loop
  (3 seeds): an estimated velocity lifts train J from 0.29 to 0.38 (EMA) / 0.40 (learned; true velocity 0.62) but
  does not help on held-out RL evaders; learned vs EMA +0.02 to +0.04, CI includes 0 except held-out scripted.
- A3e (offline): a GRU trained on noisy inputs is the best estimator under noise (velocity 0.19 m/s vs 0.29 for the
  MLP used in closed loop, -36 %; acceleration -12 %); a Kalman filter only matches the clean MLP. So A3d / F4 use
  a weaker-than-best estimator.
- A3f (offline): a spectral-norm penalty on the estimator network (Pan et al. 2026) changes its error by
  <= 2 % at the pre-registered strength; stronger penalties trade clean accuracy for noise robustness, which training
  on noisy inputs achieves better. Not adopted.

**Final runs**
- F1 (5 seeds): the acceleration input dominates (train +0.20, exploiters +0.46); held-out results are bimodal per seed.
- F2a (5 seeds): the acceleration input lowers exploitability 0.672 -> 0.397 (paired -0.275 [-0.522, -0.044]).
- F2b / F2c (self-play, the reference papers' protocol; 1 campaign per arm, descriptive): with the acceleration input
  the final pursuer is better on held-out RL evaders (0.89 vs 0.78) but not on the other campaign's evaders (0.74 vs
  0.86) or the thesis train pool (0.50 vs 0.60); +0.07 on the whole test set. Pre-registered prediction (>= +0.1 and
  lower exploitability) did not hold. **Both self-play pursuers are ~fully exploitable (0.97), and their exploiters win
  by making the pursuer crash (97 %)**, unlike the timeout-based exploits of pool-trained pursuers (0.40-0.69). The
  plain campaign leapfrogs from stage to stage (each new stage beats the latest opponent stage) but does not forget:
  its ordering is transitive (F3 post-hoc transitivity); the acc campaign barely leapfrogs.
  Caveat: for these self-play pursuers the held-out v10 evaders are much easier than the train pool's late v9 / v11
  evaders (J 0.65-0.83 vs 0.12-0.34), so their held-out RL score is not evidence of better generalisation (X2 note).

- F4 (self-play, no true evader velocity, sensor noise; 3 seeds per arm): learned estimators (nf) vs EMA finite
  difference (ema). nf generalises better: other campaigns' evaders +0.196 [+0.132, +0.291], X3 exploiters +0.441
  [+0.254, +0.541], held-out RL +0.122 (n.s.). But nf is **more** exploitable (0.872 vs 0.675, +0.197 [+0.125, +0.277]):
  in 2 of 3 nf campaigns the exploiter makes the pursuer crash (88-92 %). Prediction held for X3 exploiters, not for
  exploitability.
- F5 Exp 1 (self-play, uniform sampler, evader velocity observed, **3 seed pairs**; the third pair was added after
  the first two were seen). Clean, the learned acceleration estimate + RLS lifts the final pursuer on held-out RL
  (+0.091 [+0.055, +0.116]) and the worst held-out cluster (+0.128 [+0.042, +0.174]); both are positive on all 3 pairs.
  It does **not** help against other campaigns' evaders (-0.027; seed 300 -0.157) and exploitability is unchanged
  (~0.96). The third pair weakened the 2-seed picture (other campaigns +0.034 -> -0.027).
  - Under sensor noise it is below plain against the self-play evaders (other campaigns -0.140 [-0.232, -0.065], all 3
    pairs). This comes mainly from running RLS (rls - frozen -0.103). **The A3c RLS gain is not reproduced in
    self-play.**
  - Post-hoc: every evader win is a pursuer crash (near-miss dive at a low evader; timeout 0 %), also against scripted
    evaders (0.13-0.21). Inside the campaigns each evader stage lures the current pursuers (0.79-0.99 crash) and the
    next pursuer stage patches it. Pool-trained generalists with the same reward are not lured (crash <= 0.03).
  - At a 0.15 m capture radius (eval-only re-score) the clean gain shrinks to held-out RL +0.021 and worst cluster
    +0.049, and exploitability drops to 0.65 (rls) / 0.77 (plain). The 0.10 m result is radius-sensitive.
- F6 (Hedge + PSD-PSRO diversity term for evaders, 2 seeds, descriptive): the evaders are more diverse (Vendi 3.25 vs
  Hedge 2.50, both seeds), but the pursuer is much weaker than Hedge's. Held-out RL -0.262, other campaigns' evaders
  -0.471, both seeds.
  - The term was saturated (KL 13-52 >> cap 2), so it acted as a +2 per-step survival bonus, 20 x the evader's
    survival reward.
  - Its evaders stopped luring, so the pursuer never learned to resist luring. Against F3's evaders, which do lure,
    it crashes in 0.48-0.75 of episodes (Hedge pursuer: 0.10-0.14).
  - Predictions held for diversity, not for the pursuer. The term is not usable in this form.

- F7 (0.15 m capture radius in training, with / without a pursuer floor penalty; 1 seed per arm, descriptive):
  **neither removes the luring weakness**. Both pursuers are fully exploitable under the 0.15 m rules (0.963 / 0.968,
  pursuer crash 0.97); the exploiters lure from farther away (closest distance ~0.24 m). The floor penalty lowers
  held-out RL (-0.107 vs r15 at 0.15 m); r15 is at or below F2-plain on every metric.
  - F2-plain's own 0.15 m exploiter failed to train (exploitability 0.002); its earlier 0.10 m exploiter still beats
    it under 0.15 m rules in 0.755 of episodes, so F2-plain is >= 0.755 exploitable there.
  - Predictions (1), (3), (4), (5) did not hold; (2) held in part. **Decision rule: the rules stay** (0.10 m, no floor
    penalty).

**0.15 m capture radius (eval-only exact re-score of every closed-loop eval; added to every section 2026-11-09 11:45).**
J at 0.15 m is an upper bound and exploitability a lower bound for a 0.15 m game, because every policy and exploiter
was trained for 0.10 m. Most directions hold; many 0.10 m gaps were partly final-approach precision. What changes:
- RQ1: QD-DPP still beats the hardness samplers on a fixed pool (D1) and still not uniform. Hedge - uniform in
  self-play shrinks to held-out RL +0.022, worst cluster +0.017 (mixed seeds) (F3). Pool-trained exploitability falls
  from 0.44-0.69 to 0.09-0.22 and no sampler differs (D2); self-play exploitability falls to 0.50-0.81 (F3).
- RQ2: the acceleration input still dominates but less (F1: train +0.153, exploiters +0.180, both CIs exclude 0;
  were +0.199 / +0.464). Learned and finite-difference acceleration stay equal noise-free (A2b; on train NF-acc is now
  marginally below, -0.02). Under noise NF-acc + RLS still beats M0 and the finite difference, but online adaptation
  gains less: RLS - frozen train +0.072 (CI includes 0), exploiters +0.073 (was +0.244) (A3c).
- Diagnostics: the adaptation gap mostly disappears for held-out RL evaders (largest 0.106); only 2 exploiters keep
  Delta >= 0.15 (X3). O2a's gains over M0 roughly halve but gate G1 still passes (X4). The game is more transitive
  (X1).
- Self-play: F2c acc exploitability 0.327 vs plain 0.746 (one exploiter each); F4 nf - ema held-out RL becomes
  significant (+0.170 [+0.028, +0.404]) and the exploitability difference disappears (both 0.43-0.47).

**Self-play only (0.10 m)**: see [Self-play synthesis](#self-play-synthesis-rq1-and-rq2-from-the-self-play-campaigns-only-010-m-written-2026-11-09) below: RQ1 no (Hedge best, descriptive); RQ2 partly (opponent model helps on held-out evaders, RLS does not).

**Running / pending**: F6b v2 (amendment 1, 11-09 01:30; beta 0.03, tau 0.0727): both campaigns at stage 6 of 9 at
11:10, F6bc expected ~20:30. F8 (Roncero's rules, 26 stages) queued behind it, results ~11-10 midday. No freeze
date (user, 11-09). F5 Exp 2 (paused) and the exploiter-augmented pool are the user's decision.

## Self-play synthesis: RQ1 and RQ2 from the self-play campaigns only (0.10 m; written 2026-11-09)

Only the self-play campaigns at the 0.10 m training radius: F2b / F2c, F3, F4, F5 Exp 1, F6, and X2's analysis of
the earlier self-play pools. The fixed-pool experiments (D1, D1b, D1c, D2, F1, F2a, A-series, X3, X4) are left out.
F7 is left out because its pursuers were trained for 0.15 m. F6b is still running. Numbers are copied from the
sections below; their 0.15 m versions are there too.

**How much the evidence can carry.**
- F2b / F2c (1 campaign per arm), F3 (2 seeds), F6 (2 seeds): descriptive; no difference can be called supported.
- F5 Exp 1 (3 seed pairs): a CI that excludes 0 counts as supported, with coarse 3-pair CIs and the disclosed
  optional-stopping risk of amendment 2.
- F4 (3 seeds per arm): the only fully inferential self-play experiment.
- So in self-play, RQ1 rests on descriptive results only; RQ2 has CI-backed evidence (F4, F5).

**RQ1: does quality + diversity opponent sampling give a more robust pursuer than hardness-only sampling?
In self-play: no. Roncero's Hedge is the best sampler; more evader diversity goes with a weaker pursuer.**

| evidence | result |
|---|---|
| Self-play pools are redundant (H1a) | Yes: 22 RL evaders from three campaigns = 3.8 effectively distinct behaviours (X2); F3 arm means 2.35-2.99 of 4 per campaign |
| Best sampler (F3, sampler on both roles, 2 seeds) | Hedge: vs uniform held-out RL +0.100, worst held-out cluster +0.161, positive on both seeds |
| QD-DPP (F3) | Not better than uniform (held-out RL -0.041, worst cluster -0.052); below Hedge (-0.140 / -0.213) |
| PFSP, hard opponents only (F3) | Most diverse evaders (Vendi +0.529 vs uniform), most specialised pursuer (other campaigns' evaders -0.226) |
| Diversity term in evader training (F6, Hedge + PSD-PSRO) | Evaders more diverse (Vendi 3.253 vs 2.499); pursuer much weaker (held-out RL -0.262, other campaigns' evaders -0.471) |
| Exploitability | 0.97-0.98 for every sampler (F3) and 0.979 for F6: no sampler changes it |

- H1c (QD sampling improves the worst case and lowers exploitability) did not hold in self-play.
- H1b (robustness to duplicated opponents) was tested on a fixed pool only (D1b, D1c), not in self-play.

**RQ2: does an opponent model / online adaptation help? In self-play: the opponent model helps against held-out
evaders; online adaptation (RLS) does not; neither lowers exploitability.**

| evidence | result |
|---|---|
| Finite-difference acceleration input (F2c, 1 seed per arm) | Held-out RL +0.116, test set +0.069; worse vs the other campaign's evaders (0.741 vs 0.863) and the train pool (-0.106); exploitability 0.965 vs 0.974; the acc campaign barely leapfrogs |
| Learned acceleration estimate + RLS, clean (F5 Exp 1, 3 pairs) | Held-out RL **+0.091 [+0.055, +0.116]**, worst cluster **+0.128 [+0.042, +0.174]**, positive on all 3 pairs; other campaigns' evaders -0.027 (CI includes 0); exploitability 0.957 vs 0.974 |
| Online adaptation, RLS vs the same estimator frozen (F5) | Clean: -0.001 on held-out RL. Under eval noise: held-out RL +0.031, other campaigns' evaders **-0.103 [-0.140, -0.069]** |
| Sensor noise at eval only (F5) | Learned estimate below plain vs self-play evaders (other campaigns **-0.140 [-0.232, -0.065]**); finite difference collapses (-0.472, 1 seed) |
| Noisy training, no true velocity (F4, 3 seeds) | Learned estimators vs EMA filter: other campaigns' evaders **+0.196 [+0.132, +0.291]**, X3 exploiters **+0.441 [+0.254, +0.541]**, held-out RL +0.122 (CI includes 0); but more exploitable, **+0.197 [+0.125, +0.277]** (0.872 vs 0.675), by pursuer crash in 2 of 3 seeds |

- The held-out gain from opponent-acceleration information appears in F2c, F5 (clean) and F4 (noisy). It does not
  carry over to other campaigns' self-play evaders except in F4, and it never lowers exploitability.
- RLS adds nothing clean and hurts against self-play evaders under noise, the opposite of the fixed-pool A3c result.
- H2a / H2b (adaptation gap, oracles) were measured on a fixed pool only (X3, X4).

**Common to both questions.** Every self-play pursuer with the true evader velocity stays ~0.94-0.98 exploitable
(F2c, F3, F5, F6), whatever the sampler, diversity term or opponent model, and the fresh exploiters win almost only by
pursuer crash. Inside the campaigns each evader stage lures the current pursuers (pursuer crash 0.79-0.99) and the
next pursuer stage patches it (F5 post-hoc). F4 (noisy, estimated velocity) is the exception (0.675-0.872, partly by
timeout). Interpretation (not tested): in self-play the pursuer's robustness is limited by this one exploitable
failure mode more than by opponent sampling or opponent modelling.

## Relation to the reference papers (updated 2026-11-09 11:45)

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

All numbers in this table are at the 0.10 m training radius. At 0.15 m (see the Summary and each section) rows 1, 6 and 7 weaken: Hedge - uniform in self-play +0.022 on held-out RL, acceleration input +0.153 train / +0.180 exploiters, RLS - frozen under noise +0.072 train (CI includes 0). Exploitability values at 0.15 m are lower bounds (exploiters trained for 0.10 m); F7 shows that exploiters trained under 0.15 m rules still fully exploit self-play pursuers trained for 0.15 m.

**Claims and what our data shows**

| # | claim in the reference papers | our evidence | status |
|---|---|---|---|
| 1 | Prioritising hard opponents (Hedge / PFSP) improves on uniform population sampling. Roncero Fig. 5: Hedge beats uniform AMSPB by +0.05 to +0.27 win rate in 8 of 9 settings (3 seeds). AgilePE: PFSP stronger than FSP (single runs). | D1, 5 seeds, fixed 20-evader pool: win-rate softmax (S-W, closest to Hedge/PFSP) minus uniform: train **-0.054 [-0.081, -0.026]**, held-out RL **-0.088 [-0.145, -0.035]**, worst held-out cluster **-0.088 [-0.147, -0.030]**, worse on 5/5 seeds. Multi-metric softmax (S-M, closest to Gavin's outcome score): train **-0.150 [-0.259, -0.047]**, exploiters **-0.166 [-0.302, -0.047]**, held-out RL -0.143 [-0.306, +0.021]. D2: no sampler lowers exploitability. In full self-play with the sampler on both roles (F3, 2 seeds, descriptive): Roncero's Hedge beats uniform on held-out RL (+0.100) and the worst held-out cluster (+0.161) on both seeds; PFSP without a uniform share is mixed on held-out RL (+0.038) and much worse against other campaigns' evaders (-0.226). | **Replicates directionally in self-play (Hedge, F3, 2 seeds), not on a fixed pool (D1).** PFSP without a uniform share specialises (F3, D1c). Differences of the fixed-pool test: we train one pursuer against a fixed pool (no co-evolving generations); our uniform floor is 0.5 (Roncero 0.7) and we use a softmax (T = 0.9) instead of a power eta = 2. We did not run latest-opponent-only self-play (Roncero's AMSDRL, Gavin's SP), so we do not test "population beats latest-only". D1c (exploratory): with **no** uniform floor, hardness prioritisation collapses (train -0.29); this agrees with Roncero's advice to keep gamma large, and is a warning for PFSP without a uniform share (AgilePE does not report one). |
| 2 | Self-play with PFSP "sustains tactical diversity" (AgilePE); the population "contains distinct pursuit and evasion behaviours" (Roncero). Neither paper measures diversity. | X2: 22 RL evaders from three self-play campaigns amount to **3.8 effectively distinct behaviours** (Vendi score), 7 clusters; consecutive stages are 36 % closer than random same-campaign pairs. F3 (2 seeds): evader Vendi per campaign (4 RL evaders, max 4): pfsp 2.985 > hedge 2.499 ~ uniform 2.456 > qd_dpp 2.350; pfsp - uniform +0.529 on both seeds. F6 (2 seeds): adding a PSD-PSRO diversity term for the evaders to Hedge raises evader Vendi to 3.253 (+0.754 vs Hedge, both seeds), but the term was saturated (a survival bonus), its evaders stopped luring, and its pursuer is much weaker (held-out RL -0.262, other campaigns' evaders -0.471). | **Extends:** first quantitative measure; the pool is diverse across campaigns but highly redundant within one. **Consistent with AgilePE:** PFSP self-play produces the most diverse evaders (F3), but its pursuer is the most specialised. **More evader diversity does not by itself give a better pursuer** (F3 pfsp, F6). |
| 3 | Our addition: sample by quality *and* diversity (QD-DPP). No reference paper uses diversity in sampling. | D1: QD-DPP beats both hardness samplers (vs S-M held-out RL +0.185 [+0.110, +0.259]; vs S-W +0.131 [+0.038, +0.223]) but is **not** better than uniform (+0.042 [-0.069, +0.131]). D1b: bounds the sampling mass of duplicates but does not protect performance. F1: no sampler effect once the acceleration input is present. D2: not less exploitable. F3 (self-play, both roles, 2 seeds): not better than uniform (held-out RL -0.041, worst cluster -0.052) and below Hedge (-0.140 / -0.213); produces the least diverse evaders (Vendi 2.350). F6 (diversity in training instead of sampling, 2 seeds): more diverse evaders, much weaker pursuer (see row 2). | **New, negative:** diversity repairs the damage of badly tuned hardness prioritisation on a fixed pool (D1); in self-play it is no better than uniform and worse than Roncero's Hedge (F3). |
| 4 | Population training generalises to unseen opponents (Roncero: unseen APF evader; Gavin: SP/PFSP pursuers vs other evaders). | Consistent for scripted opponents: held-out scripted evaders (zigzag, fast flee, 4 switching evaders) are neutralised at J 0.99-1.00 by every pursuer. **Not** for learned opponents: against the independent v10 lineage the generalist loses up to 0.37 J to a specialist; fresh exploiters (PPO best responses, 100M frames) reach J 0.22-0.43 against it (X3), and the mean exploitability per sampler is 0.44-0.69 (D2; one failed exploiter included). With the reference papers' own co-evolution protocol (F2b/F2c, 1 seed per arm), the final self-play pursuers neutralise held-out scripted evaders at only 0.83-0.88, the thesis train pool's v9 evaders at 0.34-0.48, and are **~fully exploitable (0.97)**: a fresh exploiter makes them crash in 97 % of episodes. The same holds for every later self-play pursuer (F3c 0.97-0.98; F5 Exp 1 uniform_rls 0.95-0.97, pursuer crash 95-97 % against its exploiter; F6 0.98). Training with a 0.15 m capture radius, with or without a pursuer floor penalty, does not change this: 0.963 / 0.968 vs exploiters trained under the 0.15 m rules, pursuer crash 0.97 (F7, 1 seed per arm). | **Consistent for scripted, contradicted for learned opponents.** None of the three papers reports exploitability or held-out RL lineages; under their protocol the gap is larger, not smaller. |
| 5 | Seeds may converge to different strategies (Roncero, limitations). | Generalist (uniform sampler) train J 0.56-0.87 across 5 seeds; held-out generalisation is **bimodal per seed**: 1-2 of 5 seeds fail on all later v10 evaders (J 0.10-0.29) while the others reach 0.82-0.97 (F1). | **Extends:** quantifies it; with 3 seeds (Roncero) effects of +/- 0.1 can come from one seed. |
| 6 | Observation design: Roncero and Gavin give the pursuer opponent position + velocity to a memoryless MLP; Gavin leaves "trajectory prediction" out of scope; AgilePE adds a temporal window of opponent state. | Adding the opponent's acceleration (finite difference of two consecutive observations) is the **largest effect in this thesis**: train 0.722 -> 0.940 (+0.199 [+0.128, +0.270]), exploiters 0.497 -> 0.983 (+0.464 [+0.239, +0.638]), held-out RL 0.634 -> 0.838 (CI includes 0 because of the bimodal seeds) (F1, 5 seeds). A learned acceleration estimate does the same (A2b, 5 seeds: all paired differences to the finite difference within +/- 0.04, n.s.). It also lowers exploitability: 0.672 -> 0.397, paired -0.275 [-0.522, -0.044] (F2a). A raw 10-step history stack (our analogue of a temporal window) was *worse* than no history (train -0.254 [-0.355, -0.152]) (A1b). Under co-evolution (F2c, 1 seed per arm, descriptive) the gain is smaller: final pursuer +0.12 on held-out RL, +0.07 on the whole test set, but worse on the other campaign's evaders (0.74 vs 0.86) and no lower exploitability (0.965 vs 0.974); the acc campaign also leapfrogs much less. With a learned estimate instead (F5 Exp 1, 3 seed pairs): held-out RL +0.091 [+0.055, +0.116], worst held-out cluster +0.128 [+0.042, +0.174] (all 3 pairs positive), but other campaigns' evaders -0.027 and train pool +0.024 (seed 300 negative on both), exploitability unchanged (0.96 vs 0.97). At a 0.15 m capture radius (eval-only re-score) the held-out gain shrinks to +0.021. | **New:** a memoryless pursuer with position + velocity (Roncero, Gavin) is limited by the missing opponent acceleration far more than by opponent sampling, when trained against a fixed pool; under co-evolution the gain is smaller and limited to held-out evaders with a learned estimate (F5 Exp 1, 3 pairs: held-out +0.09 / +0.13, other campaigns -0.03), mixed with finite differences (F2c, 1 seed), largely final-approach precision (0.15 m re-score), and it never lowers exploitability. Caveat: AgilePE's window length and architecture are not given, so the history-stack result is not a test of AgilePE itself. |
| 7 | Future work in Roncero: online adaptation, recurrent policies. | Noise-free, online RLS adaptation of a learned opponent model adds nothing in closed loop (A1b, A2b; within or across episodes); a random reservoir (ESN) is worse than no adaptation. **Under sensor noise it helps**: RLS vs frozen basis +0.09 train, +0.24 exploiters, CIs exclude 0 (A3c, policies trained noise-free). Recurrent policies (GRU) **not tested**; a recurrent *estimator* (GRU, trained on noisy inputs) is the most accurate opponent-state estimator under noise offline (A3e: velocity -36 % vs the MLP used in closed loop). In self-play (F5 Exp 1, 3 pairs): RLS vs frozen changes nothing clean (all within +/- 0.015); under noise it is mixed: held-out RL +0.031, but other campaigns' evaders -0.103 [-0.140, -0.069], train pool -0.025, exploitability +0.027. | **New, conditional:** online opponent-model adaptation pays off under sensor noise against a fixed pool (A3c), not with perfect state (Roncero's setting); in self-play (F5) it does not pay off. |
| 8 | Sensor noise: AgilePE reports robustness under real sensor noise (mocap + IMU); Roncero and Gavin assume perfect state. | With 2 cm / 8 cm/s noise (eval only, policies trained clean): the finite-difference acceleration input collapses (train 0.94 -> 0.33, below the plain pursuer 0.70); the learned estimator + RLS keeps 0.80 and beats it by +0.47 train / +0.38 held-out RL / +0.49 exploiters (A3c, 5 seeds). Without any evader velocity (A3b), every pursuer is weak (~0.29) and the learned estimator is noise-proof but not better than no estimator. With an *estimated* velocity from noisy positions (A3d, 3 seeds): train 0.29 -> 0.38-0.40 (true velocity 0.62), no gain on held-out RL; the learned estimate is not clearly better than an EMA filter (+0.02 train, n.s.). In self-play with noise in training and evaluation (F4, 3 seeds per arm, estimated velocity only), learned estimators generalise better than the EMA filter (other campaigns' evaders +0.196, X3 exploiters +0.441, CIs exclude 0) but make the pursuer more exploitable (+0.197; crash-inducing exploits in 2 of 3 seeds). Self-play with velocity observed (F5 Exp 1, 3 pairs, noise at eval only): plain pursuers lose 0.135 on held-out RL; the finite-difference pursuer falls below plain (held-out RL -0.100, other campaigns' evaders -0.472); the learned estimate keeps a held-out gain on average (+0.086, seed 300 -0.014) but is below plain against the self-play evaders (-0.140), through more pursuer crashes with RLS running. | **New:** opponent-acceleration inputs must be learned, not differenced, once sensing is noisy; the perfect-state results of Roncero and Gavin would not transfer to such inputs. Under co-evolution with noisy sensing the learned estimator trades better generalisation for higher exploitability. |
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

**At a 0.15 m capture radius** (eval-only exact re-score, [R15_rescore/X1_crossplay/summary.md](R15_rescore/X1_crossplay/summary.md); J values are upper bounds and exploitability values lower bounds for a 0.15 m game, since every policy and exploiter was trained for 0.10 m):

| subset | additive-logit R² (one strength per player) | rank-reversal fraction (delta 0.1) |
|---|---|---|
| all 14 pursuers | 0.822 | 0.028 |
| RL pursuers only | 0.848 | 0.010 |
| all, capture rate instead of J | 0.883 | 0.005 |

What changes at 0.15 m:
- The game is more transitive. R² is 0.822-0.883 (0.10 m: 0.777-0.835) and only 0.5-2.8 % of comparisons reverse
  (0.10 m: 1.4-4.5 %). The prediction "Elo R² 0.8-0.9" now holds for all pursuers too (0.822).
- Most of the gain is in the v9 and v11 pursuers (mean J +0.068 to +0.229) and APF (+0.141). The v10 pursuers and
  pure pursuit change by at most 0.036, so their failures are not near-misses.
- Campaign final pursuers: v9/s20 mean J 0.981, worst evader 0.921 (v9/evader:s11). v11/s12 mean 0.963, worst
  0.684 (v9/evader:s17), and 0.728 vs v9/evader:s19 (0.10 m: 0.216). v10 pursuers: mean J 0.285-0.378, capture
  <= 0.173. The prediction "some final pursuer < 0.6 vs another campaign's evader" still holds, but only through the
  v10 pursuers (v10/s12: 0.000 vs v9/evader:s13); the v11/s12 case no longer does.
- Hardest evaders are the same three, in the same order: v9/evader:s19 (mean J 0.368), s17 (0.396), s15 (0.439).

Findings (at 0.10 m):

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
- **Note (added 2026-11-04, after F2c; affects how held-out RL numbers are read):** the v10 RL evaders are
  **weaker than the train pool's late v9 / v11 evaders for pursuers that never trained on the pool**. v10 is the
  campaign without warm start whose pursuers never worked (X1: J 0.27-0.35), so its evaders never had to learn the
  lure-into-crash tactic of late v9 / v11 evaders (v9 runs to stage 19; X1's hardest evaders are v9 s19 / s17 / s15).
  Mean J by evader group (late = stage >= 7):

  | pursuers | v9 late | v11 late | v10 late | v10 early |
  |---|---|---|---|---|
  | F2b plain / acc final (self-play, never saw the pool) | 0.28 / 0.12 | 0.34 / 0.15 | 0.65 / 0.83 | 0.91 / 0.96 |
  | X1: last 4 stages of v9 and v11 (trained on v9 / v11 evaders) | 0.51 | 0.60 | 0.82 | 0.91 |
  | X1: scripted pure pursuit / APF | 0.21 | 0.22 | 0.24 | 0.28 |
  | D1 pool-trained (QD / S-M / S-W) | 0.41 / 0.19 / 0.31 | 0.82 / 0.47 / 0.64 | 0.61 / 0.41 / 0.45 | 0.74 / 0.57 / 0.64 |
  | F1 pool-trained with acceleration input (QD_acc / SU_acc) | 0.84 / 0.85 | 0.97 / 0.98 | 0.63 / 0.93 | 0.83 / 0.94 |

  So "held-out RL > train pool" for a self-play pursuer (F2c) reflects an easier held-out set, not better
  generalisation; for pool-trained pursuers (D1, F1) v10 late is the harder set, as expected for unseen opponents,
  and scripted pursuers find all groups alike. Not universal either: the F4 self-play pursuers score lower on held-out
  RL than on the train pool. The late v9 / v11 evaders beat the F2b pursuers by making them crash (pursuer crash
  0.68-0.89), not by crashing themselves (evader crash + OOB 0.02-0.14 in every group).

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

**At a 0.15 m capture radius** (eval-only exact re-score, [R15_rescore/X3_gap/summary.md](R15_rescore/X3_gap/summary.md); J values are upper bounds and exploitability values lower bounds for a 0.15 m game, since every policy and exploiter was trained for 0.10 m):

Generalist mean J on the train pool: **0.817 / 0.760 / 0.685** (seeds 0 / 1 / 2; 0.10 m: 0.748 / 0.640 / 0.562).

| group | evaders | Delta |
|---|---|---|
| held-out scripted (zigzag, fast_flee, 4 switch evaders @0.5) | 6 | -0.000 to 0.001 (pi_G already ~1.0) |
| held-out RL v10 | 6 | 0.001, 0.015, -0.182, 0.044, 0.106, 0.061 (s1, s3, s5, s7, s9, s11) |
| exploiters | 3 | **0.472, 0.363**, 0.025 (pi_G J 0.528 / 0.637 / 0.975 -> specialist 1.000) |

What changes at 0.15 m:
- The gap shrinks to **2 evaders with Delta >= 0.15** (exploiters g0e0, g0e1; 0.10 m: 7). No held-out RL evader
  keeps a gap of 0.15 (largest 0.106, v10 s9), because pi_G seed 0 gains more from the wider radius than the
  specialists do. On v10 s5 the specialist is now worse than pi_G (0.379 vs 0.562, Delta -0.182). H2a is still
  supported, but only by the exploiters.
- Predictions: "exploiters Delta >= 0.25 for >= 2 of 3" still **holds** (0.472, 0.363). "Held-out RL Delta
  0.05-0.2" still does not hold, now because 4 of 6 are below 0.05 (only 0.061 and 0.106 are in range).
  "Heuristics Delta < 0.05" holds.
- Exploiters are much weaker. Against pi_G seed 0 they reach timeout in 2.5-47.2 % of episodes (0.10 m:
  57.3-77.6 %), with 0 % pursuer crash. g0e2 is no longer an exploiter (pi_G seed 0 J 0.975). Against generalist
  seeds 1-2 they concede J 0.43-0.90 (0.10 m: 0.17-0.49).

Findings (at 0.10 m):

- **7 evaders with Delta >= 0.15** (v10 s3/s7/s9/s11, all 3 exploiters). H2a (a real adaptation gap exists) **supported**.
- Exploiters win by evading, not by forcing crashes: vs pi_G seed 0 they reach timeout in 57-78 % of episodes,
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

**At a 0.15 m capture radius** (eval-only exact re-score, [R15_rescore/X4_oracles/summary.md](R15_rescore/X4_oracles/summary.md); J values are upper bounds and exploitability values lower bounds for a 0.15 m game, since every policy and exploiter was trained for 0.10 m):

| input | train | held-out RL | held-out scripted | exploiters* | gap closed (n evaders)\*\* |
|---|---|---|---|---|---|
| none (M0 = X3 generalist) | 0.754 [0.82, 0.76, 0.69] | 0.686 [0.84, 0.65, 0.57] | 0.999 | 0.704 [0.71, 0.68, 0.72] | 0.00 (2) |
| O1 opponent descriptor (8-D PCA) | 0.542 [0.54, 0.54, 0.54] | 0.431 [0.41, 0.42, 0.47] | 1.000 | n/a | n/a (0) |
| **O2a opponent acceleration** (finite difference of true velocity) | **0.987** [0.99, 0.99, 0.98] | **0.848** [0.57, 0.99, 0.99] | 1.000 | **0.997** [0.99, 1.00, 1.00] | **0.94** (2) |
| O2b opponent's latest CTBR command | 0.838 [0.72, 0.90, 0.89] | 0.396 [0.36, 0.40, 0.42] | 1.000 | 0.546 [0.45, 0.45, 0.74] | -0.48 (2) |

\* Against the exploiters, M0 seeds 1-2 reach 0.43-0.90, O2a 0.98-1.00.
\*\* Gap closed uses Delta re-computed at 0.15 m (X3 above): only exploiters g0e0 and g0e1 keep Delta >= 0.15. O1
was not evaluated against the exploiters, so it has no gap evaders.

What changes at 0.15 m:
- O2a is still the best input everywhere, but its gain over M0 roughly halves, because M0 gains more from the wider
  radius: train +0.233 (0.10 m: +0.289), held-out RL +0.162 (+0.300), exploiters +0.293 (+0.664). O2a seed 0 is
  still weak on held-out RL (0.57).
- O1 barely moves (train 0.542, held-out RL 0.431), so its deficit to M0 grows: -0.213 in distribution (0.10 m:
  -0.109). It still collapses on v10 s7/s9/s11 (J 0.001-0.021 on all 3 seeds, vs 0.415-0.937 for M0). These
  failures are not near-misses.
- O2b still helps in distribution (+0.084) and hurts more on held-out RL (-0.291; 0.10 m: -0.136). It is now also
  worse than M0 on the exploiters (-0.158; 0.10 m: +0.013).
- **Gate G1 still passes**: O2a closes 94 % of the gap (bar: 30 %). But the gap set shrinks from 7 evaders to 2,
  both exploiters trained against M0 seed 0 at 0.10 m, so at 0.15 m G1 rests on 2 evaders.

Findings (at 0.10 m):

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

**At a 0.15 m capture radius** (eval-only exact re-score, [R15_rescore/D1_sampler/summary.md](R15_rescore/D1_sampler/summary.md); J values are upper bounds and exploitability values lower bounds for a 0.15 m game, since every policy and exploiter was trained for 0.10 m):

| sampler | train | held-out RL (v10) | held-out scripted | exploiters* | held-out worst cluster | held-out worst evader |
|---|---|---|---|---|---|---|
| S-U uniform | 0.818 [0.738, 0.900] | 0.799 [0.655, 0.942] | 0.999 | 0.806 [0.702, 0.924] | 0.702 [0.513, 0.890] | 0.489 [0.195, 0.783] |
| S-W win rate | 0.764 [0.655, 0.866] | 0.698 [0.524, 0.872] | 0.998 | 0.751 [0.584, 0.911] | 0.586 [0.385, 0.808] | 0.328 [0.070, 0.676] |
| S-M multi-metric | 0.666 [0.552, 0.783] | 0.639 [0.457, 0.827] | 0.996 | 0.545 [0.327, 0.807] | 0.518 [0.304, 0.736] | 0.329 [0.073, 0.590] |
| **QD-DPP** | **0.848** [0.793, 0.903] | **0.853** [0.756, 0.948] | 0.999 | **0.824** [0.712, 0.943] | **0.760** [0.627, 0.894] | **0.529** [0.266, 0.792] |

Seed-paired differences QD-DPP minus baseline at 0.15 m (mean [95 % bootstrap CI], seeds with QD better):

| vs | train | held-out RL | held-out worst cluster | held-out worst evader |
|---|---|---|---|---|
| S-M (current method) | **+0.181 [+0.090, +0.270]** 5/5 | **+0.214 [+0.075, +0.350]** 5/5 | **+0.243 [+0.082, +0.416]** 4/5 | **+0.200 [+0.037, +0.391]** 4/5 |
| S-W (PFSP / Hedge-style) | **+0.084 [+0.009, +0.181]** 4/5 | **+0.155 [+0.060, +0.246]** 5/5 | **+0.174 [+0.081, +0.268]** 5/5 | **+0.201 [+0.080, +0.370]** 5/5 |
| S-U uniform | +0.029 [-0.018, +0.094] 2/5 | +0.054 [-0.013, +0.114] 4/5 | +0.059 [-0.003, +0.126] 4/5 | +0.040 [-0.020, +0.087] 4/5 |

What changes at 0.15 m:
- QD-DPP still beats S-M and S-W: all eight paired CIs still exclude 0. The held-out gaps grow, e.g. vs S-M worst
  cluster +0.176 -> +0.243 and vs S-W worst evader +0.087 -> +0.201.
- QD-DPP vs uniform is still not supported: all four CIs include 0 (worst cluster +0.059 [-0.003, +0.126], only
  just). Seed 3 is still reversed (held-out RL -0.066).
- Hardness-only prioritisation still hurts relative to uniform: S-M held-out RL 0.639 vs S-U 0.799.
- J against the X3 exploiters rises most (S-U 0.497 -> 0.806, QD-DPP 0.496 -> 0.824): in about 0.3 of those episodes
  the closest distance was between 0.10 and 0.15 m. QD-DPP now has the highest exploiter J, but the gap to S-U is small.

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

**At a 0.15 m capture radius** (eval-only exact re-score, [R15_rescore/D1b_duplicates/summary.md](R15_rescore/D1b_duplicates/summary.md); J values are upper bounds and exploitability values lower bounds for a 0.15 m game, since every policy and exploiter was trained for 0.10 m). Performance change caused by duplication (seed-paired, D1b minus D1, mean over 3 seeds [per-seed values]):

| sampler | train (non-duplicated evaders) | held-out RL | held-out worst cluster | exploiters |
|---|---|---|---|---|
| S-M | -0.180 [-0.266, -0.149, -0.124] | -0.144 [-0.436, +0.006, -0.001] | -0.173 | -0.016 |
| QD-DPP | -0.138 [-0.096, -0.194, -0.123] | **-0.257** [-0.261, -0.356, -0.153] | **-0.243** | -0.433 |

Absolute J under duplication at 0.15 m: S-M train 0.482, held-out RL 0.481, worst cluster 0.333; QD-DPP 0.679,
0.540, 0.422.

What changes at 0.15 m:
- H1b is still not supported: QD-DPP still loses more held-out J under duplication than S-M (held-out RL -0.257 vs
  -0.144, worst cluster -0.243 vs -0.173).
- S-M's held-out drop grows (-0.096 -> -0.144) and now exceeds the predicted 0.05-0.10. It still comes from seed 0
  alone (-0.436; seeds 1-2 +0.006 and -0.001).
- In absolute terms QD-DPP is now ahead of S-M under duplication on held-out RL (0.540 vs 0.481) and the worst
  cluster (0.422 vs 0.333); at 0.10 m the two were level (0.370 vs 0.374; 0.218 vs 0.241).
- On the exploiters, QD-DPP's drop grows (-0.198 -> -0.433) and S-M's change goes from +0.097 to -0.016.

Findings (at 0.10 m):

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

**At a 0.15 m capture radius** (eval-only exact re-score, [R15_rescore/D1c_nofloor/nodup/summary.md](R15_rescore/D1c_nofloor/nodup/summary.md), [dup/summary.md](R15_rescore/D1c_nofloor/dup/summary.md); J values are upper bounds and exploitability values lower bounds for a 0.15 m game, since every policy and exploiter was trained for 0.10 m):

| pool | sampler | train | held-out RL | held-out scripted | exploiters | worst held-out cluster |
|---|---|---|---|---|---|---|
| nodup | S-M | 0.332 [0.327, 0.341] | 0.449 [0.399, 0.477] | 0.755 [0.712, 0.808] | 0.466 [0.357, 0.564] | 0.307 [0.242, 0.341] |
| nodup | QD-DPP | 0.466 [0.436, 0.500] | 0.465 [0.457, 0.478] | 0.921 [0.884, 0.950] | 0.441 [0.326, 0.615] | 0.310 [0.274, 0.330] |
| dup | S-M | 0.300 [0.225, 0.344] | 0.386 [0.267, 0.464] | 0.583 [0.307, 0.729] | 0.593 [0.546, 0.661] | 0.241 [0.073, 0.334] |
| dup | QD-DPP | 0.442 [0.412, 0.464] | 0.475 [0.468, 0.484] | 0.952 [0.943, 0.961] | 0.507 [0.436, 0.609] | 0.295 [0.256, 0.356] |

Seed-paired differences at 0.15 m (mean [95 % CI], 3 seeds; floor = D1 at 0.15 m, same seeds):

| comparison | train | held-out RL | worst held-out cluster | exploiters |
|---|---|---|---|---|
| no floor - floor (D1, same seeds), S-M | **-0.330 [-0.517, -0.199]** | **-0.176 [-0.438, -0.023]** | -0.200 [-0.505, +0.006] | -0.002 [-0.282, +0.170] |
| no floor - floor (D1, same seeds), QD-DPP | **-0.351 [-0.397, -0.318]** | **-0.332 [-0.477, -0.234]** | **-0.356 [-0.559, -0.241]** | **-0.368 [-0.629, -0.075]** |
| dup - nodup, S-M | -0.031 [-0.102, +0.018] | **-0.063 [-0.132, -0.014]** | **-0.066 [-0.169, -0.003]** | **+0.127 [+0.008, +0.305]** |
| dup - nodup, QD-DPP | -0.025 [-0.052, +0.029] | +0.010 [-0.006, +0.024] | -0.015 [-0.068, +0.026] | +0.067 [-0.179, +0.283] |
| QD - S-M, nodup | **+0.135 [+0.108, +0.159]** | +0.016 [-0.020, +0.079] | +0.003 [-0.063, +0.088] | -0.025 [-0.151, +0.052] |
| QD - S-M, dup | **+0.141 [+0.068, +0.240]** | **+0.088 [+0.004, +0.204]** | +0.054 [-0.062, +0.283] | -0.086 [-0.184, +0.063] |

What changes at 0.15 m:
- Removing the floor is still harmful, and more clearly so: train J drops by 0.330 (S-M) and 0.351 (QD-DPP), CIs
  exclude 0. For QD-DPP all four CIs now exclude 0. S-M's apparent exploiter gain (+0.246) disappears (-0.002, n.s.).
  S-M still fails held-out scripted evaders (0.755, against 0.996 with the floor).
- The duplicate-robustness reading still holds but is weaker: duplication lowers S-M's held-out RL J by 0.063 and its
  worst-cluster J by 0.066 (CIs exclude 0, the latter only just; 0.100 at 0.10 m), and leaves QD-DPP unchanged
  (+0.010, -0.015, n.s.).
- QD-DPP's advantage over S-M shrinks to train J (+0.135 / +0.141) and held-out RL with duplicates (+0.088
  [+0.004, +0.204]). The worst-cluster advantage is no longer significant (nodup +0.042 -> +0.003, dup +0.122 ->
  +0.054).
- QD-DPP's deficit on the X3 exploiters (-0.190 / -0.238 at 0.10 m) is no longer significant (-0.025 / -0.086): it was
  mostly near misses between 0.10 and 0.15 m.

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

**At a 0.15 m capture radius** (eval-only exact re-score, [R15_rescore/D2_exploitability/summary.md](R15_rescore/D2_exploitability/summary.md); an upper bound on J, so a lower bound on exploitability, since every pursuer and exploiter was trained for 0.10 m):

| sampler | exploitability per seed (0-4) | mean [95 % CI] | capture vs exploiter per seed |
|---|---|---|---|
| S-U uniform | 0.004, 0.196, 0.015, 0.105, 0.130 | 0.090 [0.029, 0.151] | 0.98, 0.78, 0.95, 0.79, 0.84 |
| S-W win rate | 0.284, 0.253, 0.157, 0.103, 0.000 | 0.159 [0.071, 0.246] | 0.46, 0.45, 0.47, 0.88, 0.81 |
| S-M multi-metric | 0.176, 0.256, 0.062, 0.317, 0.284 | 0.219 [0.130, 0.291] | 0.81, 0.67, 0.89, 0.65, 0.70 |
| QD-DPP | 0.231, 0.015, 0.066, 0.357, 0.037 | 0.141 [0.035, 0.256] | 0.49, 0.95, 0.89, 0.33, 0.94 |

Seed-paired QD-DPP minus baseline at 0.15 m: vs S-U +0.051 [-0.094, +0.196]; vs S-M -0.078 [-0.193, +0.038];
vs S-W -0.018 [-0.146, +0.124].

What changes at 0.15 m:
- Exploitability collapses from 0.44-0.69 to 0.09-0.22: most exploiter wins at 0.10 m were timeouts in which the
  evader came within 0.15 m. These exploiters were trained for 0.10 m; exploiters trained for 0.15 m would keep more
  distance, so this does not show that the pursuers are hard to exploit at 0.15 m.
- Still no sampler reduces exploitability reliably: all three QD-DPP paired CIs include 0. The QD-DPP vs S-W
  difference (+0.189, CI barely excluding 0) disappears (-0.018).
- The ranking changes: S-U now has the lowest mean (0.090) and S-M the highest (0.219), with QD-DPP second (0.141).
  With 5 seeds and wide CIs, this ordering is not reliable.

Findings (at 0.10 m):

- **No sampler reduces exploitability reliably.** QD-DPP vs uniform and vs S-M: differences not significant.
- S-W's lower mean is driven by seed 4, where the exploiter **failed to learn** (exploitability 0.008, pursuer
  captures 77 %). Without that seed S-W's mean is 0.548. The QD-DPP vs S-W difference (CI barely excluding 0)
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

**At a 0.15 m capture radius** (eval-only exact re-score, [R15_rescore/A1b_policy/summary.md](R15_rescore/A1b_policy/summary.md); J values are upper bounds and exploitability values lower bounds for a 0.15 m game, since every policy and exploiter was trained for 0.10 m):

| method | train | held-out RL | held-out scripted | exploiters* |
|---|---|---|---|---|
| M0 | 0.818 [0.738, 0.900] | 0.799 [0.655, 0.942] | 0.999 | 0.806 [0.702, 0.924] |
| ESO_O2a (3 seeds) | **0.987** [0.977, 0.992] | **0.848** [0.566, 0.993] | 1.000 | **0.997** [0.992, 1.000] |
| M0h frame stack | 0.536 [0.505, 0.579] | 0.431 [0.410, 0.453] | 1.000 | 0.828 [0.759, 0.895] |
| ESN S1 | 0.699 [0.642, 0.755] | 0.545 [0.481, 0.608] | 0.999 | 0.631 [0.581, 0.676] |
| ESN S2 | (same policy) | 0.547 [0.480, 0.615] | 0.999 | 0.631 [0.574, 0.678] |
| NF frozen | 0.837 [0.816, 0.868] | 0.701 [0.604, 0.797] | 1.000 | 0.809 [0.697, 0.922] |
| NF-OA S1 | 0.853 [0.824, 0.883] | 0.795 [0.670, 0.925] | 1.000 | 0.856 [0.764, 0.948] |
| NF-OA S2 | (same policy) | 0.791 [0.663, 0.922] | 1.000 | 0.855 [0.762, 0.949] |

Seed-paired differences at 0.15 m (mean [95 % CI]; computed from the per-seed values in
[R15_rescore/A1b_policy/results.json](R15_rescore/A1b_policy/results.json) with the same bootstrap):

| comparison | train | held-out RL | exploiters |
|---|---|---|---|
| NF-OA - M0 | +0.034 [-0.065, +0.140] | -0.004 [-0.210, +0.220] | +0.050 [-0.103, +0.182] |
| NF frozen - M0 | +0.019 [-0.069, +0.098] | -0.098 [-0.265, +0.069] | +0.003 [-0.205, +0.199] |
| NF-OA - NF frozen (online adaptation) | +0.016 [-0.026, +0.058] | +0.094 [-0.127, +0.316] | +0.047 [-0.092, +0.152] |
| NF-OA S2 - S1 (across episodes) | 0.000 | -0.004 [-0.011, +0.001] | -0.001 [-0.008, +0.005] |
| ESN - M0 | **-0.119 [-0.215, -0.024]** | **-0.254 [-0.406, -0.101]** | **-0.175 [-0.280, -0.078]** |
| frame stack - M0 | **-0.282 [-0.376, -0.166]** | **-0.367 [-0.499, -0.224]** | +0.023 [-0.076, +0.129] |

What changes at 0.15 m:
- The NF and online-adaptation findings are unchanged. Neither NF variant beats M0 (all CIs include 0; NF-OA - M0 on
  held-out RL is -0.004). NF-OA - NF frozen stays n.s. on every split, and S2 changes J by <= 0.004.
- ESN is now also significantly worse than M0 on the exploiters (-0.175 [-0.280, -0.078]; at 0.10 m the CI included
  0). The frame stack's exploiter edge over M0 shrinks from +0.117 to +0.023 (n.s.), and its deficit on train and
  held-out RL grows (-0.282, -0.367). H2c (NF-OA beats ESN) still holds: held-out RL 0.795 vs 0.545.
- The acceleration input (ESO_O2a) stays first in every column, but its lead is much smaller on held-out RL: 0.848 vs
  0.795 for NF-OA and 0.799 for M0 (at 0.10 m: 0.773 vs 0.606 and 0.634). On train (0.987 vs 0.853) and exploiters
  (0.997 vs 0.856) it is still far ahead of every adapter.

Findings (at 0.10 m):

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

**At a 0.15 m capture radius** (eval-only exact re-score, [R15_rescore/A2b_nfacc/summary.md](R15_rescore/A2b_nfacc/summary.md); J values are upper bounds and exploitability values lower bounds for a 0.15 m game, since every policy and exploiter was trained for 0.10 m):

| method | train | held-out RL | held-out scripted | exploiters* |
|---|---|---|---|---|
| M0 | 0.818 [0.738, 0.900] | 0.799 [0.655, 0.942] | 0.999 | 0.806 [0.702, 0.924] |
| finite-difference acceleration (O2a) | **0.988** [0.982, 0.991] | 0.904 [0.735, 0.992] | 1.000 | 0.998 [0.995, 1.000] |
| NF-OA displacement target (A1b) | 0.853 [0.824, 0.883] | 0.795 [0.670, 0.925] | 1.000 | 0.856 [0.764, 0.948] |
| **NF-acc frozen** | 0.967 [0.945, 0.989] | 0.933 [0.862, 0.984] | 1.000 | 0.997 [0.992, 0.999] |
| NF-acc + RLS (S1) | 0.962 [0.937, 0.983] | **0.958** [0.918, 0.987] | 1.000 | **0.998** [0.996, 0.999] |
| NF-acc + RLS (S2) | (same policy) | 0.958 [0.923, 0.986] | 1.000 | 0.998 [0.996, 0.999] |

Seed-paired differences at 0.15 m (mean [95 % CI], 5 seeds; computed from the per-seed values in
[R15_rescore/A2b_nfacc/results.json](R15_rescore/A2b_nfacc/results.json) with the same bootstrap):

| comparison | train | held-out RL | exploiters |
|---|---|---|---|
| NF-acc + RLS - M0 | **+0.144 [+0.075, +0.231]** | **+0.159 [+0.044, +0.274]** | **+0.193 [+0.076, +0.295]** |
| NF-acc frozen - M0 | **+0.149 [+0.086, +0.219]** | +0.135 [-0.052, +0.330] | **+0.191 [+0.069, +0.298]** |
| NF-acc frozen - O2a | **-0.021 [-0.041, -0.000]**† | +0.029 [-0.115, +0.211] | -0.001 [-0.007, +0.003] |
| NF-acc + RLS - O2a | **-0.025 [-0.054, -0.000]**† | +0.053 [-0.065, +0.240] | +0.000 [-0.003, +0.004] |
| NF-acc + RLS - NF-acc frozen (online adaptation) | -0.005 [-0.040, +0.032] | +0.024 [-0.061, +0.125] | +0.001 [-0.003, +0.007] |
| S2 - S1 | 0.000 | +0.000 | -0.000 |

† upper bounds -0.0004 and -0.0002: the CI only just excludes 0.

Held-out RL per seed at 0.15 m: O2a 0.57 0.99 0.99 0.99 0.98; NF-acc frozen 0.93 0.98 1.00 0.97 0.79; NF-acc + RLS
0.97 0.95 0.88 0.98 1.00. Held-out v10 evaders s1/s3/s5/s7/s9/s11 per run (S1), re-scored with `rescore.py`:
NF-acc frozen 1.00 1.00 0.78 0.87 0.94 0.98 | 1.00 1.00 0.95 0.95 0.98 1.00 | 1.00 1.00 0.99 0.99 1.00 1.00 |
1.00 1.00 0.92 0.95 0.95 0.99 | 1.00 1.00 0.56 0.80 0.72 0.69;
NF-acc + RLS 1.00 1.00 0.90 0.96 0.97 0.99 | 1.00 1.00 0.93 0.89 0.95 0.95 | 1.00 1.00 0.77 0.77 0.82 0.94 |
1.00 1.00 0.95 0.97 0.99 1.00 | 1.00 1.00 0.99 0.99 1.00 1.00.

What changes at 0.15 m:
- The gain over M0 still holds but is smaller: with RLS +0.144 train, +0.159 held-out RL, +0.193 exploiters (all CIs
  exclude 0; at 0.10 m +0.181, +0.225, +0.490). Frozen: the held-out RL CI still includes 0 (+0.135 [-0.052, +0.330]).
- Equivalence to the finite-difference input mostly holds. Held-out RL and exploiter differences stay n.s. On train
  the learned estimate is now marginally lower (frozen -0.021, RLS -0.025; CIs only just exclude 0), so "every CI
  includes 0" no longer holds strictly. The prediction "within 0.05 of O2a" holds except NF-acc + RLS on held-out RL
  (+0.053, in its favour).
- The failure-mode contrast is unchanged. The finite-difference runs that collapse on v10 s5-s11 (F1 SU seed 0, QD
  seeds 0 and 2) stay at 0.19-0.70. The worst learned-estimate run (frozen seed 4) reaches 0.56-0.80.
- Online adaptation still adds nothing (RLS - frozen n.s. on every split; S2 = S1).

Findings (at 0.10 m):

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

**At a 0.15 m capture radius** (eval-only exact re-score, [R15_rescore/A3b_posonly/summary.md](R15_rescore/A3b_posonly/summary.md); J values are upper bounds and exploitability values lower bounds for a 0.15 m game, since every policy and exploiter was trained for 0.10 m). The M0 reference row (seeds 0-2) is from [R15_rescore/A3d_velest/summary.md](R15_rescore/A3d_velest/summary.md). The capture / crash rates and the seed-paired differences are not in the R15 summaries; they were computed from the re-scored eval files with `rescore.py`, with the same seed pairing and bootstrap as the 0.10 m tables. Mean J, 95 % bootstrap CI over 3 seeds:

| policy / eval | train | held-out RL | held-out scripted | exploiters* |
|---|---|---|---|---|
| M0 (velocity observed, ref., clean) | 0.754 [0.685, 0.817] | 0.686 [0.567, 0.844] | 0.999 | 0.704 [0.683, 0.717] |
| P_none clean | 0.348 [0.322, 0.382] | 0.383 [0.336, 0.408] | 0.905 [0.881, 0.952] | 0.408 [0.306, 0.557] |
| P_none noisy | 0.345 [0.325, 0.373] | 0.374 [0.335, 0.397] | 0.904 [0.879, 0.946] | 0.412 [0.318, 0.550] |
| P_fd clean | **0.517** [0.441, 0.593] | 0.350 [0.309, 0.372] | **0.991** [0.990, 0.992] | 0.418 [0.369, 0.448] |
| P_fd noisy | 0.075 [0.045, 0.114] | 0.086 [0.044, 0.142] | 0.089 [0.075, 0.103] | 0.344 [0.222, 0.471] |
| P_fd noisy + EMA | 0.335 [0.335, 0.335] | 0.307 [0.271, 0.344] | 0.936 [0.882, 0.968] | 0.538 [0.408, 0.610] |
| P_nf clean | 0.363 [0.333, 0.379] | 0.363 [0.346, 0.375] | 0.928 [0.889, 0.954] | 0.291 [0.278, 0.309] |
| P_nf noisy | 0.362 [0.333, 0.379] | 0.352 [0.346, 0.357] | 0.930 [0.891, 0.954] | 0.283 [0.272, 0.305] |

Capture rate / pursuer crash rate at 0.15 m (mean over 3 seeds):

| policy / eval | train | held-out RL | held-out scripted | exploiters |
|---|---|---|---|---|
| P_none clean | 0.258 / 0.003 | 0.300 / 0.002 | 0.744 / 0.005 | 0.074 / 0.003 |
| P_none noisy | 0.257 / 0.003 | 0.291 / 0.002 | 0.747 / 0.005 | 0.072 / 0.002 |
| P_fd clean | 0.464 / 0.004 | 0.310 / 0.004 | 0.850 / 0.008 | 0.250 / 0.004 |
| P_fd noisy | **0.002 / 0.927** | **0.004 / 0.916** | **0.004 / 0.912** | 0.004 / 0.663 |
| P_fd noisy + EMA | 0.280 / 0.006 | 0.258 / 0.005 | 0.778 / 0.010 | 0.374 / 0.008 |
| P_nf clean | 0.284 / 0.000 | 0.289 / 0.000 | 0.768 / 0.001 | 0.103 / 0.000 |
| P_nf noisy | 0.282 / 0.000 | 0.286 / 0.000 | 0.773 / 0.001 | 0.101 / 0.000 |

Seed-paired differences at 0.15 m (mean [95 % CI], 3 seeds):

| comparison | train | held-out RL | held-out scripted | exploiters |
|---|---|---|---|---|
| P_nf - P_none (clean) | +0.015 [-0.049, +0.056] | -0.020 [-0.062, +0.040] | +0.023 [-0.063, +0.073] | -0.117 [-0.279, +0.004] |
| P_fd - P_none (clean) | **+0.169 [+0.119, +0.252]** | -0.033 [-0.098, +0.037] | **+0.087 [+0.041, +0.110]** | +0.010 [-0.119, +0.143] |
| P_nf - P_none (noisy) | +0.018 [-0.040, +0.050] | -0.022 [-0.051, +0.020] | +0.026 [-0.055, +0.074] | **-0.129 [-0.278, -0.013]** |
| P_nf - P_fd (noisy) | **+0.288 [+0.219, +0.330]** | **+0.266 [+0.213, +0.312]** | **+0.842 [+0.789, +0.879]** | -0.061 [-0.166, +0.050] |
| P_fd noisy - clean | **-0.442 [-0.528, -0.396]** | **-0.264 [-0.297, -0.230]** | **-0.903 [-0.915, -0.890]** | -0.075 [-0.147, +0.023] |
| P_nf noisy - clean | -0.001 [-0.003, +0.000] | **-0.011 [-0.021, -0.000]** | +0.003 [-0.000, +0.006] | **-0.008 [-0.014, -0.004]** |

What changes at 0.15 m:
- Not observing the evader's velocity is still crippling. The train gap to M0 grows: P_none 0.345-0.348 vs M0 0.754 (was 0.29 vs 0.65).
- The noise collapse of the position second difference still holds and is larger: P_fd noisy - clean is -0.442 on train (was -0.305). P_fd noisy barely gains from the larger radius (0.073 -> 0.075 train; capture 0.002, pursuer crash 0.93), which confirms it is a crash failure. P_nf still beats P_fd under noise (+0.288 train, +0.266 held-out RL). P_nf stays nearly noise-free (all noisy - clean differences within 0.011), but two of these tiny differences now exclude 0 (-0.011 held-out RL, -0.008 exploiters).
- Noise-free FD vs no estimator: the train gain grows (+0.089 -> +0.169) and the scripted gain stays (+0.087). The held-out RL loss shrinks and its CI now includes 0 (-0.081 -> -0.033), so "FD hurts on held-out RL" no longer holds at 0.15 m.
- The learned estimator still does not beat no estimator: P_nf - P_none is n.s. on train and held-out. On the exploiters the clean deficit's CI now touches 0 (-0.117 [-0.279, +0.004]); the noisy one still excludes 0 (-0.129). The prediction outcomes do not change.

Findings (at 0.10 m):

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

**At a 0.15 m capture radius** (eval-only exact re-score, [R15_rescore/A3b_posonly/summary.md](R15_rescore/A3b_posonly/summary.md) (shared table); J values are upper bounds and exploitability values lower bounds for a 0.15 m game, since every policy and exploiter was trained for 0.10 m). The seed-paired differences and the degradations are not in the R15 summary; they were computed from the re-scored eval files with `rescore.py`, with the same seed pairing and bootstrap as the 0.10 m table. Mean J, 95 % bootstrap CI over 5 seeds:

| policy | train clean -> noisy | held-out RL clean -> noisy | held-out scripted noisy | exploiters clean -> noisy |
|---|---|---|---|---|
| M0 | 0.818 -> 0.793 | 0.799 -> 0.770 | 0.999 | 0.806 -> 0.738 |
| finite-difference acceleration | 0.988 -> **0.502** | 0.904 -> **0.599** | 0.921 | 0.998 -> 0.679 |
| NF-acc frozen | 0.967 -> 0.826 | 0.933 -> 0.867 | 0.997 | 0.997 -> 0.919 |
| NF-acc + RLS (S1) | 0.962 -> **0.898** [0.854, 0.944] | 0.958 -> **0.899** [0.790, 0.971] | 1.000 | 0.998 -> **0.992** [0.985, 0.997] |

Seed-paired differences under noise at 0.15 m (mean [95 % CI], 5 seeds):

| comparison (noisy eval) | train | held-out RL | held-out scripted | exploiters |
|---|---|---|---|---|
| NF-acc + RLS - finite difference | **+0.396 [+0.272, +0.518]** | **+0.299 [+0.118, +0.517]** | **+0.078 [+0.048, +0.126]** | **+0.313 [+0.193, +0.460]** |
| NF-acc frozen - finite difference | **+0.324 [+0.264, +0.387]** | **+0.267 [+0.155, +0.409]** | **+0.075 [+0.045, +0.122]** | **+0.240 [+0.095, +0.393]** |
| NF-acc + RLS - M0 | **+0.105 [+0.021, +0.216]** | **+0.129 [+0.038, +0.221]** | **+0.001 [+0.000, +0.001]** | **+0.255 [+0.134, +0.374]** |
| NF-acc frozen - M0 | +0.032 [-0.035, +0.091] | +0.097 [-0.121, +0.314] | -0.002 [-0.005, +0.000] | +0.181 [-0.000, +0.327] |
| finite difference - M0 | **-0.292 [-0.425, -0.181]** | -0.170 [-0.425, +0.078] | **-0.078 [-0.125, -0.047]** | -0.059 [-0.172, +0.043] |
| **NF-acc + RLS - NF-acc frozen** (online adaptation) | +0.072 [-0.010, +0.158] | +0.032 [-0.112, +0.195] | **+0.003 [+0.001, +0.006]** | **+0.073 [+0.023, +0.134]** |

Degradation noisy - clean at 0.15 m (same policy): finite difference train -0.486, held-out RL -0.305, exploiters -0.319;
NF-acc frozen -0.141 / -0.067 / -0.078; NF-acc + RLS -0.064 / -0.059 / -0.006; M0 -0.025 / -0.029 / -0.068 (all
CIs exclude 0).

What changes at 0.15 m:
- The finite difference is still the worst input under noise and still degrades most (train -0.486 vs at most -0.141 for the others), so the prediction still holds. It stays below M0 on train (-0.292), but the held-out RL deficit's CI now includes 0 (-0.170 [-0.425, +0.078]). Its scripted J under noise is 0.921 (was 0.797).
- NF-acc + RLS is still the most robust acceleration input and now beats M0 under noise on every split. The train CI now excludes 0 (+0.105 [+0.021, +0.216]; was +0.101, CI touching 0). The exploiter margin halves (+0.533 -> +0.255), because M0's noisy exploiter J rises more (0.408 -> 0.738) than the RLS policy's (0.941 -> 0.992, near the ceiling).
- The online-adaptation gain (RLS - frozen) shrinks. On train it is +0.072 and its CI now includes 0 (was +0.086, excluding 0). On the exploiters it falls from +0.244 to +0.073 (CI still excludes 0). So at 0.15 m the gain is clear only on the exploiters, and smaller.

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

## A3d_velest (offline done 2026-10-29 11:16; closed loop done 22:00, 3 seeds)

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

### A3d closed loop (done 2026-10-29 22:00; 3 seeds)

Pursuers that observe an **estimated** evader velocity instead of the true one (`observation.rel_vel_source`), train
pool, uniform, 150M frames, noise-free training. **V_ema**: EMA-filtered first difference (alpha 0.1). **V_nf**:
frozen learned velocity (clean basis, as the rule chose) **plus** the A3a learned acceleration (one shared noisy
measurement per step). References: M0 observes the true velocity (A3c; noisy eval adds 8 cm/s velocity noise);
P_none observes no velocity (A3b). [A3d_velest/summary.md](A3d_velest/summary.md). Mean J [95 % CI over 3 seeds]:

| policy / eval | train | held-out RL | held-out scripted | exploiters* |
|---|---|---|---|---|
| M0 noisy (true velocity + noise) | 0.622 [0.529, 0.736] | 0.439 [0.311, 0.589] | 0.996 | 0.273 [0.229, 0.339] |
| P_none noisy (no velocity) | 0.288 [0.283, 0.293] | 0.291 [0.239, 0.322] | 0.863 [0.825, 0.907] | 0.381 [0.269, 0.530] |
| V_ema clean | 0.423 [0.380, 0.458] | 0.267 [0.225, 0.313] | 0.992 [0.987, 0.996] | 0.192 [0.137, 0.248] |
| V_ema noisy | 0.383 [0.344, 0.404] | 0.222 [0.185, 0.259] | 0.959 [0.917, 0.987] | 0.158 [0.124, 0.199] |
| V_nf clean | 0.431 [0.391, 0.506] | 0.271 [0.251, 0.303] | 0.996 [0.993, 0.998] | 0.277 [0.159, 0.452] |
| V_nf noisy | 0.402 [0.379, 0.448] | 0.254 [0.233, 0.280] | 0.985 [0.982, 0.988] | 0.195 [0.146, 0.242] |

\* exploiters were trained against M0 seed 0.

Seed-paired differences (noisy eval unless stated):

| comparison | train | held-out RL | held-out scripted | exploiters |
|---|---|---|---|---|
| V_nf - V_ema | +0.019 [-0.021, +0.044] | +0.032 [-0.026, +0.096] | **+0.026 [+0.002, +0.068]** | +0.037 [-0.004, +0.119] |
| V_nf - V_ema (clean) | +0.008 [-0.040, +0.049] | +0.004 [-0.062, +0.078] | **+0.004 [+0.001, +0.011]** | +0.085 [-0.033, +0.315] |
| V_nf - P_none | **+0.114 [+0.092, +0.155]** | -0.036 [-0.089, +0.041] | **+0.122 [+0.075, +0.160]** | **-0.186 [-0.384, -0.027]** |
| V_ema - P_none | **+0.095 [+0.056, +0.117]** | **-0.069 [-0.088, -0.055]** | **+0.096 [+0.067, +0.130]** | **-0.224 [-0.380, -0.145]** |
| V_nf - M0 | **-0.220 [-0.357, -0.149]** | **-0.185 [-0.356, -0.061]** | **-0.011 [-0.014, -0.007]** | **-0.078 [-0.097, -0.055]** |
| V_ema - M0 | **-0.239 [-0.336, -0.185]** | **-0.217 [-0.330, -0.087]** | **-0.036 [-0.079, -0.009]** | **-0.116 [-0.215, -0.053]** |

Capture rate / evader out-of-bounds rate, noisy eval (mean over 3 seeds, same eval files; outcome flags are not
mutually exclusive):

| policy | train | held-out RL | held-out scripted | exploiters |
|---|---|---|---|---|
| M0 | 0.522 / 0.049 | 0.344 / 0.047 | 0.820 / 0.078 | 0.151 / 0.084 |
| P_none | 0.178 / 0.084 | 0.186 / 0.084 | 0.678 / 0.081 | **0.027 / 0.297** |
| V_ema | 0.284 / 0.067 | 0.158 / 0.035 | 0.771 / 0.085 | 0.050 / 0.066 |
| V_nf | 0.283 / 0.093 | 0.176 / 0.049 | 0.800 / 0.079 | 0.085 / 0.067 |

**At a 0.15 m capture radius** (eval-only exact re-score, [R15_rescore/A3d_velest/summary.md](R15_rescore/A3d_velest/summary.md); J values are upper bounds and exploitability values lower bounds for a 0.15 m game, since every policy and exploiter was trained for 0.10 m). The capture / out-of-bounds rates are not in the R15 summary; they were computed from the re-scored eval files with `rescore.py`. Mean J [95 % CI over 3 seeds]:

| policy / eval | train | held-out RL | held-out scripted | exploiters* |
|---|---|---|---|---|
| M0 noisy (true velocity + noise) | 0.731 [0.660, 0.796] | 0.645 [0.484, 0.834] | 0.999 | 0.627 [0.549, 0.676] |
| P_none noisy (no velocity) | 0.345 [0.325, 0.373] | 0.374 [0.335, 0.397] | 0.904 [0.879, 0.946] | 0.412 [0.318, 0.550] |
| V_ema clean | 0.552 [0.470, 0.622] | 0.421 [0.362, 0.487] | 0.997 [0.993, 1.000] | 0.382 [0.282, 0.483] |
| V_ema noisy | 0.499 [0.431, 0.551] | 0.341 [0.311, 0.393] | 0.988 [0.969, 0.999] | 0.275 [0.189, 0.390] |
| V_nf clean | 0.519 [0.443, 0.652] | 0.380 [0.293, 0.525] | 0.999 [0.999, 1.000] | 0.443 [0.278, 0.693] |
| V_nf noisy | 0.484 [0.427, 0.585] | 0.347 [0.288, 0.447] | 0.998 [0.997, 0.998] | 0.324 [0.219, 0.435] |

Seed-paired differences at 0.15 m (noisy eval unless stated):

| comparison | train | held-out RL | held-out scripted | exploiters |
|---|---|---|---|---|
| V_nf - V_ema | -0.015 [-0.087, +0.033] | +0.006 [-0.104, +0.129] | +0.009 [-0.001, +0.029] | +0.049 [-0.071, +0.246] |
| V_nf - V_ema (clean) | -0.033 [-0.122, +0.031] | -0.041 [-0.195, +0.111] | +0.002 [-0.000, +0.007] | +0.061 [-0.126, +0.411] |
| V_nf - P_none | **+0.140 [+0.102, +0.211]** | -0.027 [-0.102, +0.112] | **+0.094 [+0.051, +0.119]** | -0.088 [-0.331, +0.116] |
| V_ema - P_none | **+0.154 [+0.096, +0.190]** | -0.034 [-0.086, +0.002] | **+0.085 [+0.051, +0.120]** | -0.137 [-0.303, +0.021] |
| V_nf - M0 | **-0.247 [-0.369, -0.153]** | **-0.298 [-0.545, -0.170]** | **-0.001 [-0.002, -0.000]** | **-0.303 [-0.337, -0.241]** |
| V_ema - M0 | **-0.232 [-0.282, -0.187]** | **-0.304 [-0.441, -0.173]** | **-0.011 [-0.030, -0.001]** | **-0.352 [-0.488, -0.266]** |

Capture rate / evader out-of-bounds rate at 0.15 m, noisy eval (mean over 3 seeds):

| policy | train | held-out RL | held-out scripted | exploiters |
|---|---|---|---|---|
| M0 | 0.670 / 0.030 | 0.599 / 0.025 | 0.863 / 0.058 | 0.521 / 0.071 |
| P_none | 0.257 / 0.068 | 0.291 / 0.067 | 0.747 / 0.067 | **0.072 / 0.285** |
| V_ema | 0.431 / 0.048 | 0.303 / 0.019 | 0.843 / 0.064 | 0.178 / 0.059 |
| V_nf | 0.392 / 0.074 | 0.294 / 0.033 | 0.849 / 0.063 | 0.224 / 0.061 |

What changes at 0.15 m:
- An estimated velocity recovers a little more of the train gap: 36-40 % (0.345 -> 0.484-0.499 vs 0.731; was 28-34 %). It still does not generalise. On held-out RL neither estimator beats no velocity (-0.027, -0.034), and EMA's deficit, significant at 0.10 m, now has a CI that includes 0. The gap to M0 on held-out RL widens (about -0.30, was -0.19 to -0.22), because M0 gains more from the larger radius.
- The learned estimator no longer leads EMA. Under noise V_nf - V_ema is -0.015 on train, and the scripted gain (+0.026, the only CI excluding 0 at 0.10 m) becomes +0.009 [-0.001, +0.029]. No V_nf - V_ema CI excludes 0, clean or noisy. The prediction "V_nf > V_ema" still does not hold, now not even in direction on train.
- The exploiter deficits against P_none shrink and their CIs now include 0 (V_nf -0.088, V_ema -0.137). P_none's exploiter J is still mostly evader out-of-bounds (0.285), not capture (0.072).
- "Both below M0" still holds on every split. The held-out RL and exploiter margins grow (-0.30 to -0.35); the train margins are similar (-0.232 and -0.247, was -0.239 and -0.220).

Findings (at 0.10 m):

- An estimated velocity recovers 28-34 % of the gap between no velocity and the true velocity on the training pool
  (0.29 -> 0.38-0.40 vs 0.62) and most of it on scripted evaders (0.86 -> 0.96-0.99 vs 1.00). It does **not** generalise:
  on held-out RL evaders neither estimator beats no velocity (EMA is worse, -0.07), and both stay far below M0.
- The exploiter "loss" against P_none is not a capture loss: P_none captures exploiters in 3 % of episodes (V_nf
  9 %); its J of 0.38 comes from exploiters leaving the arena (evader OOB 0.30). Those exploiters were trained against
  a velocity-observing pursuer; P_none behaves differently enough that they fly out of bounds.
- Learned vs EMA: the learned arm is higher on every split, but the CI excludes 0 only on held-out scripted (+0.03).
  Offline its velocity error is 33 % lower (0.29 vs 0.44 m/s); in closed loop this gives at most +0.02 to +0.04 J.
- Noise costs both arms little (clean -> noisy: V_ema -0.04, V_nf -0.03 train); the policies were trained clean.
- Confound: V_nf also receives the learned acceleration, V_ema does not; V_nf - V_ema is therefore not a pure
  estimator comparison (it is an upper bound on the effect of the velocity estimator alone). Limitation from A3e: a
  noise-trained GRU would estimate velocity 36 % better than the MLP used here.
- Prediction "noisy eval: V_nf > V_ema on train and held-out RL; both far above P_none (0.29); both below M0 noisy":
  "V_nf > V_ema" **did not hold** (right direction, CIs include 0); "far above P_none" **held only on train /
  held-out scripted**, not on held-out RL or exploiters; "both below M0" **held** (the plan quoted M0 noisy as 0.70;
  in these eval files it is 0.62, which does not change the outcome).

## A3e_estimators (done 2026-10-29 20:28; offline, one training run per model)

Is the 10-step MLP window the right estimator architecture? Same data, split, noise draws, targets and masks as A3a /
A3d (the zero-baseline errors match exactly). GRU (hidden 96, 31k parameters) and LSTM (hidden 80, 29k) with memory
over the whole episode, trained on clean or noise-augmented inputs; constant-acceleration Kalman filter (KF) on the
evader-position measurement, process noise tuned on noisy val (q = 10 for velocity, 1 for acceleration). MLP (32k),
FD and FD + EMA reused from A3a / A3d. [A3e_estimators/summary.md](A3e_estimators/summary.md). Mean error in m/s,
**test** set (val gives the same ordering):

| target / inputs | zero | FD + EMA | KF | MLP clean | MLP aug | GRU clean | GRU aug | LSTM clean | LSTM aug |
|---|---|---|---|---|---|---|---|---|---|
| velocity, clean | 0.868 | 0.264 | 0.123 | 0.126 | 0.161 | **0.060** | 0.139 | 0.078 | 0.149 |
| velocity, noisy | 0.868 | 0.435 | 0.288 | 0.292 | 0.232 | 0.800 | **0.186** | 0.568 | 0.193 |
| 0.2 s velocity change, clean | 0.340 | - | 0.338 | 0.288 | 0.290 | **0.244** | 0.255 | 0.282 | 0.287 |
| 0.2 s velocity change, noisy | 0.340 | 1.699 | 0.385 | 0.331 | 0.293 | 0.419 | **0.259** | 0.302 | 0.288 |

(Noise-free, raw FD is still best for velocity: 0.010.)

- **A recurrent estimator trained on noisy inputs is the most accurate under noise**: GRU aug velocity 0.186 vs 0.232
  for the MLP aug (-20 %) and 0.292 for the MLP clean basis used in the closed loop (-36 %); acceleration 0.259 vs
  0.293 (-12 %). LSTM aug is close to GRU aug (velocity -17 %, acceleration -2 % vs MLP aug). Noise-free, the GRU is
  also better (velocity 0.060 vs 0.126).
- **Recurrent nets trained on clean inputs break under noise** (GRU clean velocity 0.80, worse than the EMA filter's
  0.43): noise augmentation matters far more for them than for the MLP.
- **The Kalman filter is no better than the clean MLP** (velocity 0.288 vs 0.292 under noise; 0.123 vs 0.126 clean)
  and 24 % worse than the MLP aug. For the acceleration target it is worse than predicting zero (0.385 vs 0.340), like
  every position-differencing method here. It beats the EMA filter (0.288 vs 0.435), so EMA is the weaker classical
  baseline.
- Predictions: (1) "noise-free velocity: KF < MLP" **did not hold** (equal: test 0.123 vs 0.126, val 0.118 vs
  0.105); (2) "best recurrent (aug) beats MLP aug by >= 10 % on noisy velocity" **held** (-20 %); "KF within +/- 20 %
  of MLP aug" **did not hold** (+24 %); "KF and recurrent nets beat FD_ema" **held** for the KF and the aug nets, not
  for the clean-trained nets; (3) "noisy acceleration: GRU / LSTM aug <= MLP aug; KF worse than MLP aug" **held**.
- Pre-registered consequence: GRU aug beats the closed-loop MLP by >= 10 % (velocity -36 %, acceleration -12 %).
  **This is a limitation of A3d / F4**: their learned arm uses the MLP, not the best available estimator, so
  closed-loop gains of the learned arm over EMA are a lower bound for a recurrent estimator. Not rerun before the
  freeze.
- Caveats: one training run per model (no seed spread); 8 epochs with the last epoch kept, and the clean GRU was
  still improving at epoch 8; offline error only, not closed-loop performance. A measured velocity with A3c's noise
  (error 0.128) is still better than every estimator from positions.

## A3f_specnorm (done 2026-11-03 14:41; offline, one training run per model)

User request after Pan et al. (RA-L 2026, "Learning on the Fly"), who train their residual-dynamics network with a
spectral-norm penalty. Pre-registered (plan, A3f row) before any run. The A3a / A3d positions-only MLP estimator is
retrained with loss + beta * sum_l ||W_l||^2 over the basis' 3 layers and the shared head: **spec** = largest singular
value (Pan et al.), **frob** = Frobenius norm (plain weight decay, the control). beta in {1e-4, 1e-3, 1e-2}, chosen on
val in the basis' training condition. Same data, split, noise draws, masks and metric as A3a / A3d; the unregularised
models re-evaluated with the A3f code match the stored A3a / A3d numbers exactly (max |diff| 0). Files:
`A3f_specnorm/summary.md`, `pred_err.json`.

Mean error (m/s); acc = evader velocity change over 0.2 s, vel = current evader velocity; test = held-out evaders.

| target | test inputs | MLP clean | spec clean | frob clean | MLP aug | spec aug | frob aug |
|---|---|---|---|---|---|---|---|
| acc | clean | 0.2884 | 0.2884 (0.0 %) | 0.2902 | 0.2901 | 0.2892 (-0.3 %) | 0.2895 |
| acc | noisy | 0.3309 | 0.3276 (-1.0 %) | 0.3244 | 0.2931 | 0.2921 (-0.3 %) | 0.2922 |
| vel | clean | 0.1261 | 0.1286 (+2.0 %) | 0.1269 | 0.1614 | 0.1619 (+0.3 %) | 0.1635 |
| vel | noisy | 0.2918 | 0.2905 (-0.4 %) | 0.2881 | 0.2324 | 0.2347 (+1.0 %) | 0.2347 |

Lipschitz bound prod ||W_l||_2 (normalised inputs), MLP -> spec at beta*: acc clean 6.4 -> 3.8, acc aug 5.9 -> 2.9,
vel clean 36.5 -> 30.1, vel aug 36.7 -> 15.5 (frob lowers it about as much: 3.1, 2.2, 29.1, 27.9). beta* = 1e-4 (the
smallest grid value) for 7 of 8 models (vel spec aug: 1e-3): on val, any stronger penalty is worse.

- Predictions: (1) "spec lowers the Lipschitz bound >= 2x" **held in 2 of 4** (aug bases 2.0x / 2.4x; clean bases
  1.7x / 1.2x). (2) "clean basis, noisy test: spec >= 10 % better" **did not hold** (-1.0 % acc, -0.4 % vel).
  (3) "aug basis within +/- 5 %" **held** (-0.3 %, +1.0 %). (4) "clean test within +/- 5 %" **held** (0.0 to +2.0 %).
  (5) "spec <= frob on noisy test, clean basis" **did not hold** (frob slightly better on both targets).
- Pre-registered consequence: spec does not beat the closed-loop estimators (acc MLP aug -0.3 %, vel MLP clean -0.4 %)
  by >= 10 %, so it is **not a candidate** for A3d / F4. The spectral penalty does not help this estimator.
- Exploratory (not the pre-registered beta): larger penalties make the clean-trained basis more noise-robust at a cost
  on clean inputs, e.g. acc spec beta 1e-2: noisy test -10.6 %, clean val +8.0 %; vel frob beta 1e-2: noisy test
  -9.7 %, clean val +60 %. Noise-augmented training does better on both counts (acc MLP aug: noisy test -11.4 % vs MLP
  clean, clean val +2.1 %), and A3e's GRU aug is better still. So in this setting the regulariser buys noise
  robustness only by trading off accuracy, which training on noisy inputs achieves more cheaply.
- Caveats: one training run per model, so differences of 1-2 % are likely within seed noise; beta* sits at the grid
  edge, so the val-optimal penalty may be below 1e-4 (i.e. about zero); offline error only. Setting differs from Pan
  et al.: their network predicts the robot's own residual acceleration from its state and action, ours estimates an
  opponent's state from noisy position history.

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

**At a 0.15 m capture radius** (eval-only exact re-score, [R15_rescore/F1_final/summary.md](R15_rescore/F1_final/summary.md); J values are upper bounds and exploitability values lower bounds for a 0.15 m game, since every policy and exploiter was trained for 0.10 m):

| cell | train | held-out RL (v10) | held-out scripted | exploiters* |
|---|---|---|---|---|
| SU + plain | 0.818 [0.738, 0.900] | 0.799 [0.655, 0.942] | 0.999 | 0.806 [0.702, 0.924] |
| QD + plain | 0.848 [0.793, 0.903] | 0.853 [0.756, 0.948] | 0.999 | 0.824 [0.712, 0.943] |
| SU + acc | **0.988** [0.982, 0.991] | **0.904** [0.735, 0.992] | 1.000 | **0.998** [0.995, 1.000] |
| QD + acc | 0.984 [0.974, 0.993] | 0.841 [0.689, 0.987] | 1.000 | 0.991 [0.977, 0.999] |

Seed-paired effects at 0.15 m (mean [95 % CI]):

| effect | train | held-out RL | exploiters |
|---|---|---|---|
| acceleration input (main effect) | **+0.153 [+0.094, +0.211]** | +0.047 [-0.147, +0.203] | **+0.180 [+0.086, +0.262]** |
| sampler QD - SU (main effect) | +0.013 [-0.016, +0.051] | -0.005 [-0.065, +0.051] | +0.006 [-0.061, +0.064] |
| interaction | -0.033 [-0.086, +0.008] | -0.117 [-0.301, +0.012] | -0.025 [-0.163, +0.119] |
| QD - SU, plain obs | +0.029 [-0.018, +0.094] | +0.054 [-0.013, +0.114] | +0.018 [-0.120, +0.145] |
| QD - SU, acc input | -0.004 [-0.017, +0.009] | -0.063 [-0.214, +0.024] | **-0.007 [-0.018, -0.001]** |

Held-out v10 evaders s1/s3/s5/s7/s9/s11 per seed at 0.15 m (re-scored with `rescore.py` from the per-episode files):

| seed | SU + acc on v10 s1/s3/s5/s7/s9/s11 | QD + acc on v10 s1/s3/s5/s7/s9/s11 |
|---|---|---|
| 0 | 1.00 1.00 **0.21 0.60 0.27 0.32** | 1.00 1.00 **0.19 0.70 0.44 0.31** |
| 1 | 1.00 1.00 0.97 0.97 0.98 0.99 | 1.00 1.00 0.96 0.98 1.00 1.00 |
| 2 | 1.00 1.00 0.98 0.98 0.99 1.00 | 1.00 1.00 **0.34 0.63 0.30 0.55** |
| 3 | 1.00 1.00 0.99 0.99 0.99 1.00 | 1.00 1.00 0.97 0.96 0.99 1.00 |
| 4 | 1.00 1.00 0.95 0.96 0.98 1.00 | 1.00 1.00 0.94 0.98 0.99 1.00 |

What changes at 0.15 m:
- The acceleration input is still the dominant factor on train (+0.153) and exploiters (+0.180), with CIs excluding
  0, but both effects shrink (from +0.199 and +0.464). The exploiter effect shrinks most because the plain-obs
  policies gain most from the larger radius (SU + plain on exploiters 0.497 -> 0.806).
- The sampler still has no main effect (every main-effect CI includes 0). The one notable QD - SU difference at
  0.10 m (acc input, held-out RL, -0.108 [-0.295, -0.005]) now includes 0: -0.063 [-0.214, +0.024]. On the exploiters
  QD is still lower with the acc input, but only by -0.007 [-0.018, -0.001].
- The v10 bimodality persists. The same seeds fail (SU seed 0; QD seeds 0 and 2) at 0.19-0.70 on s5-s11, while the
  other seeds reach 0.94-1.00.

Findings (at 0.10 m):

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

**At a 0.15 m capture radius** (eval-only exact re-score, [R15_rescore/F2a_exploit_acc/summary.md](R15_rescore/F2a_exploit_acc/summary.md); J values are upper bounds and exploitability values lower bounds for a 0.15 m game, since every policy and exploiter was trained for 0.10 m):

| seed | plain exploitability (D2 S-U) | acc exploitability | acc: capture / timeout vs exploiter |
|---|---|---|---|
| 0 | 0.004 | 0.033 | 0.93 / 0.03 |
| 1 | 0.196 | 0.002 | 0.55 / 0.00 (exploiter mostly crashes / leaves the arena) |
| 2 | 0.015 | 0.020 | 0.97 / 0.02 |
| 3 | 0.105 | 0.008 | 0.97 / 0.01 |
| 4 | 0.130 | 0.007 | 0.97 / 0.01 |
| mean [95 % CI] | 0.090 [0.029, 0.151] | **0.014 [0.005, 0.025]** | |

Seed-paired acc - plain: **-0.076 [-0.146, -0.006]** (n = 5). Without seed 1: -0.047 [-0.110, +0.017] (n = 4).

What changes at 0.15 m:
- Most of the exploiters' wins at 0.10 m were near misses. Exploitability falls from 0.672 to 0.090 (plain) and from
  0.397 to 0.014 (acc). The acc pursuer captures 0.93-0.97 of episodes on the 4 seeds whose exploiter learned to fly.
- Acc is still less exploitable than plain (-0.076, CI excludes 0), but the gap is much smaller (-0.275 at 0.10 m).
  Without the failed seed-1 exploiter the CI now includes 0 (-0.047 [-0.110, +0.017]), so the "also without the failed
  exploiter" part of the finding does not hold at 0.15 m.
- "The pursuer stays exploitable" does not hold at 0.15 m: acc exploitability is 0.002-0.033 on every seed against these exploiters. This is a lower bound: no exploiter was
  trained under 0.15 m rules for these pursuers.

Findings (at 0.10 m):

- The acceleration input **lowers exploitability** (CI excludes 0, also without the failed exploiter), in addition
  to its gains on train / held-out / X3 exploiters (F1). It is the only intervention in this thesis that reduced
  exploitability; no sampler did (D2).
- The pursuer stays exploitable: a fresh exploiter still escapes neutralisation in 28-63 % of episodes on 4 of 5
  seeds, mostly by surviving to timeout.
- Prediction "lower than plain (0.672), mean 0.3-0.5, paired CI excludes 0; still clearly > 0": **held**
  (0.397; one seed's exploiter failed, as one of 20 did in D2).
- Bookkeeping: the driver's first summary showed 3 of 5 seeds (lane-lock bug); rebuilt from the eval files on disk.


## F2b_campaigns / F2c_final_eval (campaigns done 2026-10-29 18:33; F2c done 2026-10-30 14:29; 1 seed per arm, descriptive)

Two self-play campaigns with the reference papers' co-evolution protocol, identical except the observation: **acc**
(both roles observe the opponent's finite-difference acceleration, frozen opponents too) vs **plain**. Uniform
opponent sampling, 9 stages x 300M frames, pursuer first, warm start, seed 0, noise-free. Pre-registered as 1 seed per
arm: no significance claims. [F2c_final_eval/summary.md](F2c_final_eval/summary.md), `fig_crossplay.png`.

Final pursuer of each campaign (stage 8), J:

| evaluated against | acc final | plain final | acc - plain |
|---|---|---|---|
| own campaign's RL evaders | 0.993 | 0.924 | |
| other campaign's RL evaders | 0.741 (plain evaders) | 0.863 (acc evaders) | |
| thesis test set, all 15 (pre-registered comparison) | 0.905 | 0.836 | +0.069 |
| - held-out RL (v10), mean / min | 0.893 / 0.809 | 0.777 / 0.566 | +0.116 |
| - held-out scripted, mean / min | 0.876 / 0.830 | 0.834 / 0.788 | +0.042 |
| - X3 exploiters, mean / min | 0.986 / 0.968 | 0.956 / 0.931 | +0.030 |
| thesis train pool (never seen by either campaign), mean / min | 0.495 / 0.050 | 0.601 / 0.067 | -0.106 |
| - of which v9 / v11 RL evaders / scripted | 0.340 / 0.494 / 0.887 | 0.481 / 0.640 / 0.845 | |
| exploitability (fresh exploiter, 100M frames) | 0.965 | 0.974 | -0.009 |
| - outcomes vs exploiter: capture / pursuer crash / timeout | 0.025 / 0.968 / 0.00 | 0.022 / 0.975 / 0.00 | |

Cross-play (rows = pursuer stages, columns = evader stages; full matrix in the summary):
- **plain** leapfrogs: each new stage beats the latest opponent stage (called "cycling" here before 11-09; the ordering
  is transitive with no forgetting, see the F3 post-hoc transitivity). Plain P2 is beaten by every later plain evader
  (J 0.01-0.03), P4 by E5/E7 (0.04 / 0.02), P6 by E7 (0.07); only the final P8 handles all plain evaders (0.79-0.99).
- **acc** shows almost no within-campaign leapfrogging: every acc pursuer from P4 on neutralises every earlier and later
  acc evader at >= 0.84, and P8 at >= 0.99. But acc pursuers fail on the late plain evaders (P8 vs plain E7: 0.30;
  P2-P6 vs plain E7: 0.17-0.23).
- So the acc campaign's evaders are easier for every pursuer (plain P8 vs acc evaders 0.73-0.95), and the acc final
  pursuer is the less robust of the two against the other campaign (worst cell 0.30 vs 0.73).

**At a 0.15 m capture radius** (eval-only exact re-score, [R15_rescore/F2c_final_eval/summary.md](R15_rescore/F2c_final_eval/summary.md); J values are upper bounds and exploitability values lower bounds for a 0.15 m game, since every policy and exploiter was trained for 0.10 m):

| evaluated against | acc final | plain final | acc - plain |
|---|---|---|---|
| own campaign's RL evaders | 0.999 | 0.974 | |
| other campaign's RL evaders | 0.944 (plain evaders) | 0.970 (acc evaders) | |
| thesis test set, all 15 (pre-registered comparison) | 0.980 | 0.953 | +0.028 |
| - held-out RL (v10), mean / min | 0.996 / 0.992 | 0.959 / 0.898 | +0.037 |
| - held-out scripted, mean / min | 0.956 / 0.945 | 0.928 / 0.902 | +0.028 |
| - X3 exploiters, mean / min | 0.996 / 0.989 | 0.988 / 0.967 | +0.008 |
| thesis train pool (never seen by either campaign), mean / min | 0.775 / 0.293 | 0.766 / 0.332 | +0.009 |
| - of which v9 / v11 RL evaders / scripted | 0.674 / 0.818 / 0.961 | 0.674 / 0.808 / 0.933 | |
| exploitability (fresh exploiter, 100M frames) | 0.327 | 0.746 | -0.419 |
| - outcomes vs exploiter: capture / pursuer crash / timeout | 0.670 / 0.327 / 0.00 | 0.252 / 0.746 / 0.00 | |

The test-set mean, the train-pool subsets and the exploiter outcome split are not in the summary; they were computed
with `rescore.rates_at` on the F2c eval files (the same code reproduces the 0.10 m values above).

Cross-play at 0.15 m (full matrix in the R15 summary):
- **plain** still leapfrogs, but much less: plain P2 vs later plain evaders 0.29-0.64 (was 0.01-0.03), P4 vs E5 / E7
  0.78 / 0.47 (was 0.04 / 0.02), P6 vs E7 0.76 (was 0.07); P8 handles all plain evaders (0.91-1.00).
- **acc**: every acc pursuer from P4 on neutralises every acc evader at >= 0.95, P8 at 1.00. Acc pursuers are still
  weakest on plain E7 (P8 0.79; P2-P6 0.46-0.75).
- Plain P8 vs acc evaders 0.92-0.99, so the acc final pursuer is still the less robust against the other campaign
  (worst cell 0.79 vs 0.92).

What changes at 0.15 m:
- **The exploitability result changes.** At 0.10 m both pursuers were ~fully exploitable (0.965 / 0.974) through
  pursuer crashes. At 0.15 m the acc pursuer captures its exploiter in 0.670 of episodes (exploitability 0.327), the
  plain pursuer in 0.252 (0.746). So the acc exploiter's wins were mostly near misses, while the plain exploiter still
  wins mostly by making the pursuer crash (0.746). One exploiter per pursuer and one seed per arm, so this is
  descriptive.
- The acc advantage on the test set shrinks from +0.069 to +0.028, and on held-out RL from +0.116 to +0.037. The
  prediction's +0.1 part still does not hold, now also not on the held-out RL subset; its "lower exploitability"
  part holds at 0.15 m (-0.419),
  against one 0.10 m-trained exploiter per pursuer (both values are lower bounds).
- The acc pursuer's train-pool deficit disappears (-0.106 -> +0.009; v9 evaders 0.674 in both arms). On the other
  campaign's evaders acc is still lower (0.944 vs 0.970, was 0.741 vs 0.863).

Findings:
- Under co-evolution the acceleration input improves the final pursuer on held-out RL evaders (+0.12), much less on
  the rest of the test set, and **not** on the other campaign's evaders or the train pool (-0.11).
- **Both self-play pursuers are almost fully exploitable (0.97), and the exploiters win differently than in D2 / F2a:
  by making the pursuer crash into the ground (97 % of episodes), not by surviving to timeout.** Pursuers trained on the
  fixed thesis pool (D2, F2a) were exploitable at 0.40-0.69, mostly through timeouts. This suggests the self-play
  pursuers learned aggressive near-ground pursuit that a best response can exploit. One exploiter per pursuer.
- Both self-play pursuers do poorly on the thesis train pool (v9 evaders: 0.34 / 0.48), which they never trained
  against; pursuers trained directly on that pool reach 0.72 (plain) / 0.94 (acc) (F1).
- Prediction "acc final >= plain final + 0.1 J on the other campaign's evaders and on the test set, and lower
  exploitability": **did not hold.** Other campaign's evaders: acc is lower on either reading. Test set: +0.069 (< 0.1;
  only the held-out RL subset passes, +0.116). Exploitability: 0.965 vs 0.974 is no meaningful difference.
- Limitation: one seed per arm. In F1 the acceleration result was bimodal across seeds, so a single campaign per arm
  cannot separate an input effect from a seed effect.

## F3_sampler_sp / F3c (final eval done 2026-11-05 07:42; 2 seeds per sampler, descriptive by pre-registration)

RQ1 under the reference papers' protocol: full self-play campaigns (plain observation, 9 stages x 300M frames,
pursuer first, warm start, noise-free) that differ only in the opponent sampler, used **by both roles** (amendment 3):
**uniform**; **hedge** = Roncero eq. (12) exactly (gamma 0.7, eta 2); **pfsp** = Hedge with gamma 0, p ~ (1 - J)^2
(the hard-opponent core of AgilePE / Gavin); **qd_dpp** = the same Hedge quality re-weighted by a DPP over behaviour
descriptors. Campaign seeds 0 / 100 (uniform seed 0 = F2b plain). Final pursuer (stage 8) of each campaign.
other / own campaign evaders exclude the stage-1 evaders, which are identical across same-seed campaigns
(amendment 6). With 2 seeds the bracket is only the range of the two values; **no difference can be called
supported** (pre-registered). [F3_sampler_sp/summary.md](F3_sampler_sp/summary.md); per-campaign values in
`results.json`. The earlier pursuer-only F3 campaigns (hedge s0/s1, qd_dpp s0, pfsp s0) are kept on disk, not analysed.

| final pursuer, J | uniform | hedge | qd_dpp | pfsp |
|---|---|---|---|---|
| held-out RL (v10) | 0.736 [0.696, 0.777] | **0.836** [0.791, 0.882] | 0.696 [0.588, 0.803] | 0.775 [0.639, 0.910] |
| held-out worst cluster | 0.593 [0.540, 0.646] | **0.754** [0.692, 0.816] | 0.541 [0.377, 0.705] | 0.665 [0.503, 0.827] |
| other campaigns' evaders | 0.852 [0.843, 0.862] | 0.888 [0.871, 0.906] | 0.878 [0.872, 0.884] | **0.626** [0.580, 0.672] |
| own campaign's evaders | 0.936 | 0.959 | 0.969 | **0.776** |
| thesis train pool | 0.570 | 0.617 | 0.562 | 0.542 |
| held-out scripted | 0.795 | 0.831 | 0.821 | 0.826 |
| X3 exploiters | 0.951 | 0.964 | 0.958 | 0.930 |
| exploitability (fresh exploiter) | 0.973 | 0.979 | 0.977 | 0.983 |
| evader Vendi (distinct RL evaders, max 4) | 2.456 | 2.499 | 2.350 | **2.985** |
| sampling: max probability (mean) | 0.521 | 0.664 | 0.664 | 0.988 |

Per seed (seed 0 / seed 1), held-out RL: uniform 0.777 / 0.696; hedge 0.791 / 0.882; qd_dpp 0.803 / 0.588;
pfsp 0.910 / 0.639. Worst held-out cluster: uniform 0.646 / 0.540; hedge 0.692 / 0.816; qd_dpp 0.705 / 0.377;
pfsp 0.827 / 0.503.

Seed-paired differences (mean of the 2 pairs [range]):

| pair | held-out RL | worst held-out cluster | other campaigns' evaders | train pool | exploitability | evader Vendi |
|---|---|---|---|---|---|---|
| hedge - uniform | +0.100 [+0.014, +0.186] | +0.161 [+0.046, +0.276] | +0.036 [+0.009, +0.063] | +0.047 [-0.007, +0.101] | +0.007 | +0.043 |
| qd_dpp - uniform | -0.041 [-0.108, +0.026] | -0.052 [-0.163, +0.059] | +0.026 [+0.023, +0.029] | -0.008 | +0.004 | -0.107 |
| qd_dpp - hedge | -0.140 [-0.293, +0.013] | -0.213 [-0.439, +0.013] | -0.010 | -0.055 [-0.098, -0.012] | -0.002 | -0.149 |
| pfsp - uniform | +0.038 [-0.057, +0.133] | +0.072 [-0.037, +0.181] | **-0.226** [-0.262, -0.190] | -0.028 | +0.011 | **+0.529** [+0.412, +0.647] |
| qd_dpp - pfsp | -0.079 [-0.107, -0.051] | -0.124 [-0.126, -0.122] | **+0.252** [+0.213, +0.292] | +0.020 | -0.006 | -0.636 |

Capture-only (secondary, amendment 5) gives the same ordering: hedge - uniform held-out RL +0.112, worst cluster
+0.153; qd_dpp - uniform -0.019 / -0.035; pfsp - uniform on other campaigns' evaders -0.290.

**At a 0.15 m capture radius** (eval-only exact re-score, [R15_rescore/F3_sampler_sp/summary.md](R15_rescore/F3_sampler_sp/summary.md); J values are upper bounds and exploitability values lower bounds for a 0.15 m game, since every policy and exploiter was trained for 0.10 m):

| final pursuer, J | uniform | hedge | qd_dpp | pfsp |
|---|---|---|---|---|
| held-out RL (v10) | 0.952 [0.944, 0.959] | **0.974** [0.960, 0.989] | 0.936 [0.900, 0.973] | 0.944 [0.894, 0.994] |
| held-out worst cluster | 0.880 [0.846, 0.914] | **0.897** [0.874, 0.921] | 0.859 [0.842, 0.877] | 0.884 [0.838, 0.931] |
| other campaigns' evaders | 0.940 [0.923, 0.957] | 0.965 [0.957, 0.973] | 0.949 [0.938, 0.960] | **0.898** [0.898, 0.899] |
| own campaign's evaders | 0.979 | 0.994 | 0.996 | **0.930** |
| thesis train pool | 0.715 | 0.784 | 0.729 | 0.787 |
| held-out scripted | 0.901 | 0.920 | 0.911 | 0.930 |
| X3 exploiters | 0.986 | 0.990 | 0.987 | 0.986 |
| exploitability (fresh exploiter) | 0.752 | 0.814 | 0.695 | **0.497** |
| evader Vendi (distinct RL evaders, max 4) | 2.449 | 2.475 | 2.336 | **3.019** |

Evader Vendi changes slightly because its descriptors include J vs the F3 pursuers. The sampling probabilities are
training statistics and do not change.

Per seed (seed 0 / seed 1), held-out RL: uniform 0.959 / 0.944; hedge 0.960 / 0.989; qd_dpp 0.973 / 0.900;
pfsp 0.994 / 0.894. Worst held-out cluster: uniform 0.914 / 0.846; hedge 0.874 / 0.921; qd_dpp 0.877 / 0.842;
pfsp 0.931 / 0.838.

Seed-paired differences at 0.15 m (mean of the 2 pairs [range]):

| pair | held-out RL | worst held-out cluster | other campaigns' evaders | train pool | exploitability | evader Vendi |
|---|---|---|---|---|---|---|
| hedge - uniform | +0.022 [+0.000, +0.044] | +0.017 [-0.040, +0.075] | +0.025 [+0.001, +0.050] | +0.069 [-0.007, +0.145] | +0.062 [+0.019, +0.106] | +0.026 [-0.091, +0.143] |
| qd_dpp - uniform | -0.016 [-0.045, +0.013] | -0.021 [-0.037, -0.004] | +0.009 [+0.003, +0.015] | +0.013 [-0.007, +0.034] | -0.056 [-0.116, +0.004] | -0.113 [-0.159, -0.068] |
| qd_dpp - hedge | -0.038 [-0.089, +0.013] | -0.038 [-0.079, +0.003] | -0.016 [-0.035, +0.002] | -0.056 [-0.111, -0.000] | -0.119 [-0.222, -0.015] | -0.139 [-0.211, -0.068] |
| pfsp - uniform | -0.008 [-0.050, +0.034] | +0.004 [-0.009, +0.017] | **-0.042** [-0.058, -0.025] | +0.072 [+0.030, +0.113] | -0.254 [-0.266, -0.243] | **+0.569** [+0.441, +0.697] |
| qd_dpp - pfsp | -0.008 [-0.021, +0.006] | -0.025 [-0.054, +0.004] | **+0.051** [+0.040, +0.061] | -0.058 [-0.079, -0.037] | +0.198 [+0.150, +0.246] | -0.683 [-0.765, -0.600] |

Bold marks the cells bolded in the 0.10 m tables where the same reading still holds. Capture-only at 0.15 m: hedge -
uniform held-out RL +0.022, worst cluster +0.006; qd_dpp - uniform -0.020 / +0.007; pfsp - uniform on other
campaigns' evaders -0.048.

What changes at 0.15 m:
- Most of Hedge's advantage over uniform is near misses. Held-out RL falls from +0.100 to +0.022 (seed 0: +0.000),
  and the worst cluster from +0.161 to +0.017, now mixed across seeds (-0.040 / +0.075). Hedge still has the highest
  mean on held-out RL and the worst cluster, but the "higher on both seeds" claim no longer holds for the worst
  cluster.
- PFSP's specialisation shrinks but remains. Its deficit on other campaigns' evaders shrinks from -0.226 to -0.042
  (still negative on both seeds), and its own-campaign J is 0.930 vs 0.979-0.996. QD-DPP stays close to uniform
  (-0.016 held-out RL), and its gap to Hedge shrinks from -0.140 to -0.038.
- "No sampler changes exploitability" does not hold at 0.15 m. Exploitability is 0.50-0.81, not 0.97-0.98, and PFSP's
  pursuer is the least exploitable (pfsp - uniform -0.254, both seeds); Hedge is more exploitable than uniform
  (+0.062, both seeds). The exploiters were trained for 0.10 m, so these are lower bounds on exploitability in a
  0.15 m game.
- Against the pre-registered "all paired differences within +/- 0.05": at 0.15 m every difference vs uniform on
  held-out RL, the worst cluster and other campaigns' evaders is within it, but the train pool (hedge +0.069, pfsp
  +0.072) and exploitability (hedge +0.062, qd_dpp -0.056, pfsp -0.254) are not.

Findings:
- **Hedge (Roncero) is the best sampler in self-play on average** (the single best held-out run is PFSP seed 0, 0.910): higher than uniform on held-out RL (+0.100) and the worst
  held-out cluster (+0.161) on **both** seeds. The size differs strongly between
  seeds (held-out RL +0.014 / +0.186); J is also higher on average on the other metrics. This is the direction Roncero reports and the **opposite** of our fixed-pool
  result D1 (win-rate softmax worse than uniform on 5/5 seeds); the protocols differ (co-evolution, both roles,
  Hedge eta = 2 with gamma 0.7 vs softmax with a 0.5 floor on a fixed pool).
- **QD-DPP does not improve on uniform** (held-out RL -0.041, worst cluster -0.052, mixed across seeds) and is below
  Hedge on average (held-out RL -0.140: seed 0 +0.012, seed 1 -0.294); on other campaigns' evaders it is level with
  Hedge and slightly above uniform (+0.026). Its seed 1 is the worst held-out result of all 8 campaigns (0.588 / worst cluster 0.377).
- **PFSP (no uniform share) produces the most diverse evader populations** (Vendi 2.985 vs 2.35-2.50, both seeds) but
  the most specialised pursuer: it is clearly worse against the other campaigns' evaders (-0.226 vs uniform, both
  seeds) and even against its own campaign's evaders (0.776 vs 0.94-0.97). On held-out RL it is mixed (+0.133 / -0.057).
- **No sampler changes exploitability**: every self-play pursuer is ~fully exploitable (0.97-0.98), as in F2c.
- Predictions (pre-registered 10-28 / amendments 1-3): "all paired differences within +/- 0.05": **did not hold**
  for hedge - uniform (+0.10 / +0.16) and pfsp on other campaigns' evaders (-0.23); held for exploitability and most
  qd_dpp - uniform differences. "D1: hedge <= uniform": **did not hold** (Roncero's "hedge > uniform" is the observed
  direction). "qd_dpp ~ uniform": **held** on held-out RL (-0.041), marginally outside on the worst cluster (-0.052). "pfsp <= uniform on
  held-out RL and worst cluster": **did not hold** on the means (+0.038 / +0.072, mixed seeds). "pfsp has the highest
  max sampling probability": **held** (0.988). "qd_dpp >= pfsp": **did not hold** on held-out RL / worst cluster
  (-0.079 / -0.124), **held** on other campaigns' evaders (+0.252). AgilePE's claim "PFSP sustains diversity"
  (evader Vendi pfsp >= uniform): **consistent** (+0.529, both seeds).
- Limitations: 2 seeds per sampler (descriptive only); one exploiter per pursuer; the sampler only reweights a
  population of 4-5 RL opponents per role, so the distributions differ less than the names suggest (except pfsp).

### F3 post-hoc: transitivity per sampler (added 2026-11-09; not pre-registered, no new runs)

X1's measures on the F3 campaigns: additive-logit R² (one strength per player) and rank-reversal fraction (delta 0.1;
a pursuer pair that is ordered one way on one evader and the other way on another).
[F3_sampler_sp/transitivity.md](F3_sampler_sp/transitivity.md) (matrices included), `transitivity.json`,
`exps/f3_transitivity.py`.

Within each campaign (pursuer stages 2-8 x RL evader stages 1-7, from the in-campaign stage evals, 0.10 m only; pursuer
stage 0 is left out because evader stage 1 never played it):

| sampler | R² (J), seed 0 / 1 | reversals (J), seed 0 / 1 | final pursuer (P8) vs older evaders, min J, seed 0 / 1 |
|---|---|---|---|
| uniform | 0.731 / 0.725 | 0.000 / 0.000 | 0.79 / 0.92 (vs E7) |
| hedge | 0.651 / 0.690 | 0.000 / 0.000 | 0.91 / 0.90 (vs E7) |
| qd_dpp | 0.671 / 0.673 | 0.000 / 0.000 | 0.92 / 0.95 (vs E7) |
| pfsp | 0.388 / 0.382 | **0.278 / 0.194** | **0.65 / 0.35** (vs E5) |

Across campaigns (the 8 F3 final pursuers x each sampler's RL evaders, stage 1 excluded; F3c evals):

| evader set | n evaders | R² 0.10 m (0.15 m) | reversals 0.10 m (0.15 m) |
|---|---|---|---|
| uniform | 6 | 0.921 (0.839) | 0.007 (0.000) |
| hedge | 6 | 0.924 (0.829) | 0.005 (0.000) |
| qd_dpp | 6 | 0.902 (0.779) | 0.000 (0.000) |
| pfsp | 6 | **0.748** (0.747) | **0.105** (0.045) |
| all 24 | 24 | 0.835 (0.765) | 0.032 (0.010) |

- **Uniform, Hedge and QD-DPP campaigns are ordinally transitive**: every new stage beats all older opponents
  (staircase matrices: P_k vs older evaders 0.79-1.00, vs newer evaders 0.01-0.14), no rank reversals, and the final
  pursuer still beats every older evader (min 0.79-0.95, always against the latest one, E7). Their R² is only 0.65-0.73 because a 4 x 4 step-shaped matrix with cells
  near 0.01 / 0.99 fits an additive logit poorly; the reversal fraction is the better measure here.
- **PFSP campaigns are not transitive**: reversals 0.19-0.28, and the final pursuer has forgotten an older evader
  (P8 vs E5 0.65 / 0.35). This is the forgetting that Roncero attributes to latest-opponent training; PFSP without a
  uniform share behaves like it.
- Across campaigns the game is mostly transitive (all 24 evaders: R² 0.835, reversals 0.032, close to X1's 0.777-0.835 /
  0.024-0.045). PFSP's evaders are the exception (R² 0.748, reversals 0.105), consistent with their higher diversity
  (Vendi 2.985) and with PFSP's most specialised pursuer.
- At 0.15 m the reversals fall to 0.000-0.045; R² also falls because most cells approach 1, leaving less variance to
  explain.
- Terminology (applied 2026-11-09): the stage-to-stage pattern previously called "cycling" in F2c (each new evader beats
  the latest pursuer, the next pursuer beats it back) is ordinally transitive in uniform / Hedge / QD-DPP campaigns,
  so it is now called leapfrogging; only PFSP shows forgetting (a true cycle back to an older opponent).
- Limitations: 2 seeds; 4 x 4 within-campaign matrices, half from pursuer-side and half from evader-side evals.

## F4_estimator_sp v2 (campaigns done 2026-10-31 16:36; F4c done 22:09; analysis fixed 2026-11-02 10:25; 3 seeds per arm)

Self-play campaigns (reference papers' protocol: uniform sampling, 9 stages x 300M frames, pursuer first) in which
the pursuer **never observes the true evader velocity** and is trained and evaluated with sensor noise. Both arms get
an *estimated* evader velocity: **ema** = EMA-filtered finite difference of noisy positions (A3d V_ema, the classical
baseline); **nf** = learned velocity estimate + learned acceleration estimate (A3d V_nf). Frozen pursuers seen by the
evader get the same estimators with clean inputs. 3 campaign seeds per arm (0 / 100 / 200), final pursuer (stage 8)
of each. [F4_estimator_sp/summary.md](F4_estimator_sp/summary.md), per-campaign values in `results.json`.

Bookkeeping: the driver's first summary (10-31 22:09) still used the v1 arm names ("none" vs "nf"), so the ema arm was
missing; rebuilt from the eval files on disk with the pre-registered contrast nf - ema (no data changed).

| final pursuer vs | ema | nf | nf - ema (seed-paired) |
|---|---|---|---|
| thesis train pool | 0.396 [0.302, 0.459] | 0.452 [0.364, 0.613] | +0.056 [-0.081, +0.186] |
| held-out RL (v10) | 0.272 [0.243, 0.311] | 0.394 [0.302, 0.562] | +0.122 [-0.009, +0.301] |
| held-out RL, noise-free eval | 0.280 [0.249, 0.321] | 0.414 [0.311, 0.572] | +0.134 [-0.010, +0.300] |
| held-out scripted | 0.913 [0.834, 0.974] | 0.878 [0.825, 0.949] | -0.035 [-0.106, +0.027] |
| X3 exploiters | 0.459 [0.262, 0.703] | **0.900** [0.791, 0.957] | **+0.441 [+0.254, +0.541]** |
| X3 exploiters, noise-free eval | 0.517 [0.321, 0.755] | 0.911 [0.804, 0.965] | **+0.394 [+0.210, +0.490]** |
| own campaign's RL evaders | 0.744 [0.719, 0.779] | 0.770 [0.699, 0.863] | +0.025 [-0.037, +0.084] |
| other F4 campaigns' RL evaders | 0.499 [0.415, 0.543] | **0.695** [0.581, 0.830] | **+0.196 [+0.132, +0.291]** |
| exploitability (fresh exploiter, 100M frames) | **0.675** [0.552, 0.786] | 0.872 [0.829, 0.911] | **+0.197 [+0.125, +0.277]** (nf worse) |

How the fresh exploiters win (fractions of episodes):

| campaign | J vs exploiter | capture | pursuer crash | timeout | evader crash / OOB |
|---|---|---|---|---|---|
| ema-s0 / s1 / s2 | 0.448 / 0.214 / 0.313 | 0.13 / 0.12 / 0.14 | 0.00 / 0.00 / 0.00 | 0.55 / 0.78 / 0.68 | 0.32 / 0.09 / 0.17 |
| nf-s0 / s1 / s2 | 0.171 / 0.089 / 0.123 | 0.13 / 0.05 / 0.09 | 0.01 / **0.92 / 0.88** | 0.82 / 0.00 / 0.00 | 0.04 / 0.04 / 0.03 |

**At a 0.15 m capture radius** (eval-only exact re-score, [R15_rescore/F4_estimator_sp/summary.md](R15_rescore/F4_estimator_sp/summary.md); J values are upper bounds and exploitability values lower bounds for a 0.15 m game, since every policy and exploiter was trained for 0.10 m):

| final pursuer vs | ema | nf | nf - ema (seed-paired) |
|---|---|---|---|
| thesis train pool | 0.566 [0.430, 0.673] | 0.622 [0.497, 0.802] | +0.057 [-0.106, +0.208] |
| held-out RL (v10) | 0.437 [0.386, 0.486] | **0.607** [0.465, 0.843] | **+0.170 [+0.028, +0.404]** |
| held-out RL, noise-free eval | 0.456 [0.397, 0.516] | 0.621 [0.496, 0.848] | **+0.165 [+0.002, +0.394]** |
| held-out scripted | 0.960 [0.910, 0.997] | 0.962 [0.939, 0.992] | +0.002 [-0.018, +0.029] |
| X3 exploiters | 0.713 [0.515, 0.938] | **0.983** [0.958, 0.997] | **+0.269 [+0.059, +0.443]** |
| X3 exploiters, noise-free eval | 0.776 [0.620, 0.966] | 0.988 [0.968, 0.999] | **+0.212 [+0.033, +0.348]** |
| own campaign's RL evaders | 0.910 [0.838, 0.946] | 0.953 [0.925, 0.987] | +0.043 [-0.019, +0.107] |
| other F4 campaigns' RL evaders | 0.718 [0.611, 0.783] | **0.875** [0.732, 0.978] | **+0.156 [+0.121, +0.218]** |
| exploitability (fresh exploiter, 100M frames) | 0.430 [0.304, 0.494] | 0.470 [0.398, 0.516] | +0.040 [-0.095, +0.212] |

How the fresh exploiters win at 0.15 m (fractions of episodes; computed with `rescore.rescore` on the F4 exploiter
eval files, which reproduces the 0.10 m table above):

| campaign | J vs exploiter | capture | pursuer crash | timeout | evader crash / OOB |
|---|---|---|---|---|---|
| ema-s0 / s1 / s2 | 0.696 / 0.507 / 0.506 | 0.39 / 0.46 / 0.41 | 0.00 / 0.00 / 0.00 | 0.30 / 0.49 / 0.49 | 0.30 / 0.04 / 0.10 |
| nf-s0 / s1 / s2 | 0.484 / 0.505 / 0.602 | 0.45 / 0.49 / 0.60 | 0.00 / **0.50 / 0.40** | 0.51 / 0.00 / 0.00 | 0.03 / 0.01 / 0.00 |

Post-hoc, stage-1 evaders excluded, at 0.15 m: other campaigns' evaders ema 0.628 [0.488, 0.712], nf 0.835
[0.645, 0.971], nf - ema **+0.207 [+0.156, +0.287]**; own campaign ema 0.880 [0.785, 0.928], nf 0.938 [0.903, 0.984],
nf - ema +0.058 [-0.024, +0.143].

What changes at 0.15 m:
- The nf generalisation advantage holds and now also reaches held-out RL: +0.170 [+0.028, +0.404] (noise-free eval
  +0.165 [+0.002, +0.394]); at 0.10 m both CIs just included 0. The X3-exploiter gap shrinks from +0.441 to +0.269
  and the other-campaigns gap from +0.196 to +0.156; both CIs still exclude 0.
- **"nf is more exploitable" does not hold at 0.15 m**: +0.040 [-0.095, +0.212], CI includes 0 (was +0.197). Both
  arms are at 0.43-0.47. nf-s1 / s2 exploiters still win mainly by making the pursuer crash (0.50 / 0.40), but the
  pursuer now captures 0.45-0.60 of the time against nf exploiters and 0.39-0.46 against ema exploiters. These exploiters were trained for 0.10 m, so both
  exploitability values are lower bounds.
- For the prediction: held-out RL now **holds** (+0.170, CI excludes 0); X3 exploiters still **holds** (+0.269);
  exploitability lower for nf still does not hold (no difference).
- Absolute level: both arms are still well below the full-state F1 pursuers on held-out RL (0.44-0.61 vs 0.80-0.90 in
  F1 at 0.15 m, [R15_rescore/F1_final/summary.md](R15_rescore/F1_final/summary.md)).

Findings (at 0.10 m):

- With learned estimators the self-play pursuer **generalises better**: to the other campaigns' evaders (+0.196) and
  to the X3 exploiters (+0.441), both CIs exclude 0. On held-out RL it is +0.122 (CI just includes 0). Train pool and
  held-out scripted: no difference. Noise-free evaluation gives the same picture, so this is not only noise robustness.
- But the nf pursuers are **more exploitable** (+0.197, CI excludes 0). In 2 of 3 nf campaigns the exploiter makes the
  pursuer crash into the ground (88-92 %), the failure mode already seen in the F2c self-play pursuers. The ema
  exploiters win by survival to timeout, and partly fail themselves (evader crash / OOB 9-32 %), which lowers the ema
  arm's exploitability. Captures against the exploiter are similarly low in both arms (0.05-0.14).
- Absolute level: without the true evader velocity both arms remain far below the full-state pursuers (held-out RL
  0.27-0.39 vs 0.63-0.84 in F1), consistent with A3b / A3d.
- On the fixed pool (A3d closed loop) nf and ema did not differ (+0.02 to +0.03, n.s.); under co-evolution nf is
  clearly better at generalisation. The co-evolving opponents (evaders trained against the nf / ema pursuers) differ
  between arms, so the arms are compared on the shared test sets and on each other's evaders, not on their own pool.
- Prediction "nf > ema on held-out RL and X3 exploiters by >= 0.05; exploitability lower for nf (low confidence)":
  X3 exploiters **held** (+0.441); held-out RL **held in direction and size, not significant** (+0.122, CI includes 0);
  exploitability **did not hold** (nf higher by +0.197).
- **Post-hoc (11-04, plan §9): stage-1 evaders excluded.** Stage-1 evaders are byte-identical across campaigns with the same seed (here nf-sK and ema-sK share theirs), so "other campaigns' evaders" contained a copy of each pursuer's own stage-1 opponent. Without them (3 evaders per campaign): other campaigns' evaders ema 0.367 [0.274, 0.417], nf 0.618 [0.463, 0.788], nf - ema **+0.252 [+0.189, +0.371]**; own campaign ema 0.683 [0.633, 0.753], nf 0.712 [0.629, 0.839], nf - ema +0.029 [-0.035, +0.086]. Both arms are lower (the stage-1 evaders are easy) and the nf advantage is larger; conclusions unchanged. The registered numbers above are unchanged (`*_no_s1` columns in `results.json` / `summary.md`).
- Limitation: 3 seeds, one exploiter per pursuer; the estimator is the clean-trained MLP chosen by the A3d rule, not the
  better noise-trained GRU found later (A3e).

## F5_acc_sp Exp 1 / F5c (3 seed pairs; campaigns done 2026-11-07 06:45; F5c done 08:31; written up 2026-11-07 17:45; descriptive)

RQ2 under the reference papers' protocol: does the learned acceleration estimate (A2b machinery, velocity observed)
help the pursuer in full self-play? New arm **uniform_rls**: F3 protocol (uniform sampler on both roles, plain
observation + the learned estimate of the evader's mean acceleration over the next 0.2 s, online RLS on the last
layer, 9 stages x 300M frames, pursuer first, warm start, clean training; frozen pursuers seen by the evader run the
same estimator on clean inputs), campaign seeds 0 / 100 / 300. **uniform_plain** = F2-plain (seed 0) + F3-uniform-s1
(seed 100) + the new uniform_plain-s2 (seed 300); **uniform_fd** = F2-acc (finite-difference acceleration, 1 seed).
Final pursuer (stage 8) of each campaign, evaluated clean and with the A3c sensor-noise model (eval only); RLS ablation
= the same uniform_rls pursuer with RLS frozen at eval (B1). [ ] = 95 % bootstrap CI over the 3 seeds (coarse);
amendment 2 reports a difference whose CI excludes 0 as supported, with that caveat.
[F5_acc_sp/summary_exp1.md](F5_acc_sp/summary_exp1.md); per-seed values in `results_exp1.json`; the 2-seed
analysis is kept as [summary_exp1_2seeds.md](F5_acc_sp/summary_exp1_2seeds.md); every J at 0.15 m in
[R15_rescore/F5_acc_sp/summary_exp1.md](R15_rescore/F5_acc_sp/summary_exp1.md).

Deviations:
- (a) Amendment 1: the estimator was retrained before the campaigns. H* = 10 by the A2a rule. NF + RLS (S1) error is
  0.110 m/s vs 0.143 for finite-difference persistence on val (-23 %); on test 0.143 vs 0.174
  ([estimator/summary.md](F5_acc_sp/estimator/summary.md)).
- (b) The user paused Exp 2 (`qd_dpp_rls`) on 11-05, so F5c was split (11-05 16:08) and Exp 1 is analysed alone.
  "Campaign evaders" are the RL evaders of the 12 non-paused plain-evader campaigns (8 F3, 2 F5 uniform_rls seeds 0 /
  100, and the 2 seed-300 campaigns), stage-1 evaders excluded.
- (c) **Amendment 2** (11-06 12:45, decided after the 2-seed results were seen; optional-stopping risk disclosed, no
  further seeds): a third seed pair (seed 300), and every Exp 1 pursuer re-evaluated against the new campaigns' evaders.

The 2-seed numbers that this section had before (11-06 12:40) are superseded. Per-seed values below are in seed order
0 / 100 / 300.

**0.15 m column:** the same evals re-scored at a 0.15 m capture radius (exact, eval-only; `rescore.py`; see the F5
post-hoc section). These are upper bounds for a 0.15 m game, because the opponents and exploiters were trained for 0.10 m.

| final pursuer, J at 0.10 m (0.15 m) | uniform_rls clean | uniform_plain clean | uniform_fd clean | uniform_rls noisy | uniform_plain noisy | uniform_fd noisy |
|---|---|---|---|---|---|---|
| held-out RL (v10) | **0.871** [0.812, 0.921] (0.986) | 0.780 [0.696, 0.866] (0.965) | 0.893 (0.996) | **0.731** [0.717, 0.745] (0.934) | 0.645 [0.548, 0.746] (0.883) | 0.541 (0.776) |
| held-out worst cluster | **0.789** [0.708, 0.839] (0.937) | 0.661 [0.540, 0.797] (0.889) | 0.827 (0.950) | **0.614** [0.597, 0.645] (0.893) | 0.487 [0.348, 0.655] (0.804) | 0.440 (0.717) |
| other campaigns' evaders | 0.835 [0.742, 0.895] (0.965) | 0.862 [0.837, 0.899] (0.958) | 0.657 (0.941) | 0.640 [0.575, 0.700] (0.894) | **0.780** [0.765, 0.807] (0.935) | 0.293 (0.519) |
| own campaign's evaders | 0.972 (0.995) | 0.946 (0.984) | - | 0.750 (0.967) | **0.865** (0.973) | - |
| thesis train pool | 0.619 (0.809) | 0.595 (0.738) | 0.495 (0.775) | 0.539 (0.725) | 0.541 (0.689) | 0.398 (0.555) |
| held-out scripted | 0.867 (0.947) | 0.808 (0.908) | 0.876 (0.956) | 0.827 (0.939) | 0.762 (0.898) | 0.805 (0.936) |
| X3 exploiters | 0.975 (0.993) | 0.955 (0.987) | 0.986 (0.996) | 0.923 (0.979) | 0.903 (0.976) | 0.892 (0.965) |
| exploitability (fresh exploiter) | 0.957 (0.645) | 0.974 (0.772) | 0.965 (0.327) | 0.944 (0.728) | 0.976 (0.824) | 0.751 (0.641) |

Per seed (0 / 100 / 300), clean:
- held-out RL: uniform_rls 0.879 / 0.812 / 0.921; uniform_plain 0.777 / 0.696 / 0.866.
- other campaigns' evaders: uniform_rls 0.895 / 0.868 / 0.742; uniform_plain 0.850 / 0.837 / 0.899.

Seed-paired differences (3 pairs [CI]; uniform_fd: 1 pair), at 0.10 m (0.15 m):

| pair | held-out RL | worst held-out cluster | other campaigns' evaders | own campaign | train pool | exploitability |
|---|---|---|---|---|---|---|
| rls - plain, clean | **+0.091** [+0.055, +0.116] (+0.021 [+0.005, +0.030]) | **+0.128** [+0.042, +0.174] (+0.049 [+0.011, +0.109]) | -0.027 [-0.157, +0.046] (+0.007 [-0.002, +0.015]) | +0.026 [+0.013, +0.035] (+0.011) | +0.024 [-0.085, +0.102] (+0.071 [+0.054, +0.093]) | -0.018 [-0.028, -0.005] (-0.127 [-0.258, +0.011]) |
| rls - plain, noisy | +0.086 [-0.014, +0.198] (+0.051 [+0.011, +0.116]) | +0.127 [-0.058, +0.296] (+0.089 [+0.021, +0.183]) | **-0.140** [-0.232, -0.065] (-0.041 [-0.079, -0.021]) | **-0.115** [-0.235, -0.017] (-0.007) | -0.002 [-0.085, +0.079] (+0.036) | -0.032 [-0.050, -0.022] (-0.095 [-0.136, -0.064]) |
| fd - plain, clean | +0.117 (+0.037) | +0.181 (+0.037) | -0.193 (-0.021) | - | -0.106 (+0.009) | -0.010 (-0.428) |
| fd - plain, noisy | -0.100 (-0.105) | -0.017 (-0.074) | **-0.472** (-0.420) | - | -0.161 (-0.161) | -0.226 (-0.173) |
| rls - rls frozen, clean | -0.001 [-0.005, +0.004] (+0.001) | -0.002 (-0.002) | +0.012 [+0.001, +0.022] (+0.003) | +0.007 (-0.000) | +0.001 (+0.000) | +0.015 [+0.003, +0.027] (+0.065 [+0.018, +0.104]) |
| rls - rls frozen, noisy | +0.031 [+0.009, +0.049] (+0.020) | +0.039 [+0.006, +0.063] (+0.031) | **-0.103** [-0.140, -0.069] (-0.018 [-0.042, -0.003]) | -0.073 [-0.173, +0.039] (+0.001) | -0.025 [-0.034, -0.012] (-0.009) | +0.027 [+0.018, +0.035] (+0.037 [+0.017, +0.070]) |

Capture-only (secondary) has the same signs. rls - plain on held-out RL: +0.127 clean, +0.099 noisy. On other
campaigns' evaders: -0.029 clean, -0.167 noisy.

Findings:
- **Clean: the learned acceleration estimate helps on held-out evaders, on all 3 seed pairs.**
  - Held-out RL +0.091 [+0.055, +0.116]; worst held-out cluster +0.128 [+0.042, +0.174]. Both CIs exclude 0.
  - Smaller supported gains: own campaign +0.026, X3 exploiters +0.020, held-out scripted +0.059.
  - It does **not** help against the other campaigns' evaders (-0.027; +0.046 / +0.031 / -0.157 per pair) or on the
    train pool (+0.024; seed 300: -0.085). Exploitability is ~unchanged (-0.018).
  - The third pair weakened the 2-seed result: other campaigns' evaders went from +0.034 to -0.027, train pool from
    +0.078 to +0.024, held-out RL from +0.110 to +0.091.
  - The seed-300 uniform_rls pursuer is the weakest against the other campaigns' evaders (0.742). It loses only by
    crashing: pursuer crash 0.265, timeout 0.000 (seeds 0 / 100: crash 0.119 / 0.149). Its worst opponents are the
    stage-7 evaders of other F5 campaigns (J 0.29-0.35).
- **Noisy: the learned estimate keeps a held-out gain on average but falls below plain against the self-play evaders.**
  - Held-out RL +0.086, worst cluster +0.127; both CIs include 0, because seed 300 is -0.014 / -0.058.
  - Against self-play evaders it is below plain: other campaigns -0.140 [-0.232, -0.065] and own -0.115
    [-0.235, -0.017], negative on all 3 pairs.
  - Finite differencing collapses under noise: fd - plain is -0.100 on held-out RL and -0.472 on other campaigns'
    evaders.
- **Running RLS under noise causes most of the noisy loss against self-play evaders**: rls - frozen, other campaigns'
  evaders -0.103 [-0.140, -0.069] (all 3 pairs).
  - RLS under noise helps a little on held-out RL (+0.031) and the worst cluster (+0.039).
  - It is worse on the train pool (-0.025) and on exploitability (+0.027).
  - Clean, RLS changes nothing (all |difference| <= 0.015), as in A1b / A2b.
  - The A3c result (RLS beats frozen under noise; pool-trained pursuers) is **not reproduced** in self-play.
- **Every pursuer stays ~fully exploitable at 0.10 m** (0.94-0.98). The fresh exploiters win by making the pursuer
  crash in 0.92-0.98 of the episodes (timeout 0). The one exception is uniform_fd under noise (0.751; pursuer crash
  0.80).
- Plain self-play pursuers are noise-sensitive: from clean to noisy, held-out RL drops -0.135 and the worst cluster
  -0.174 (other campaigns -0.082, train -0.054). For comparison, M0 in A3c dropped -0.03.
- **At 0.15 m (eval-only re-score) the clean advantage of the estimate shrinks to final-approach precision.**
  - Held-out RL +0.021 [+0.005, +0.030]; worst cluster +0.049 [+0.011, +0.109]; train pool +0.071 [+0.054, +0.093].
    All CIs exclude 0, but the gains are small.
  - Under noise the deficit against campaign evaders mostly disappears: other campaigns -0.041, own -0.007.
  - Exploitability falls to 0.65 (rls), 0.77 (plain) and 0.33 (fd): many exploiter wins are near misses inside 0.15 m.
    These values are upper bounds, because the exploiters were trained for 0.10 m.
- **Predictions (pre-registered 11-04 17:40), re-checked with 3 pairs:**
  - (1) Clean:
    - "rls - plain >= +0.05 on held-out RL": **held**. +0.091, CI excludes 0, every pair >= +0.055.
    - "no gain on other campaigns' evaders": **held** (-0.027).
    - "exploitability within +/- 0.05": **held** (-0.018).
  - (2) Noisy:
    - "plain pursuers lose <= 0.05": **did not hold**. Held-out RL -0.135, worst cluster -0.174, other campaigns
      -0.082, train -0.054. It held only for exploitability.
    - "fd falls below plain": **held**.
    - "rls >= plain on held-out RL": **held on the mean** (+0.086), not on every pair (seed 300: -0.014).
  - (4) RLS:
    - "clean, rls - frozen within +/- 0.03": **held**.
    - "noisy, rls > frozen by >= 0.03 on the train pool and lower exploitability": **did not hold** (-0.025;
      exploitability +0.027).
  - Prediction (3) belongs to Exp 2.
- Limitations:
  - 3 seed pairs (uniform_fd 1); the third pair was added after the first two were seen.
  - One exploiter per pursuer, trained against the RLS-on pursuer and reused for the frozen evaluation.
  - For self-play pursuers the held-out v10 evaders are easier than the train pool's late evaders (F2c caveat), so the
    worst-cluster and other-campaign columns matter more than the held-out mean.
  - The estimator is the MLP; A3e found a noise-trained GRU to be more accurate.
  - Noise is applied only at eval.

### F5 Exp 1, post-hoc: how evaders beat the self-play pursuers (added 2026-11-06 17:40; descriptive, not pre-registered)

Written before the seed-300 pair finished: the numbers below are pooled over seeds 0 / 100 only (the 2-seed
F5c). The seed-300 pursuers also lose only by crashing (timeout 0; see the findings above).

From existing eval files only ([F5_acc_sp/outcomes_posthoc.md](F5_acc_sp/outcomes_posthoc.md), generated by
`exps/f5_outcomes.py`; pooled over seeds, weighted by episodes, stage-1 evaders excluded from campaign evaders). An
evader wins (episode not neutralised) only by a pursuer crash or a timeout.

- **Every evader win against the final F5 Exp 1 pursuers is a pursuer crash; timeout is 0.000 for every arm, evader
  group and noise condition.** Pursuer crash rate, clean, uniform_rls / uniform_plain: own campaign 0.039 / 0.064,
  other campaigns 0.130 / 0.162, held-out scripted 0.134 / 0.208, held-out RL 0.154 / 0.248, train pool RL (v9 / v11)
  0.386 / 0.478, fresh exploiter 0.970 / 0.974 (uniform_fd: other campaigns 0.345, exploiter 0.968). The pursuer
  wins by capture (0.71-0.93 against non-exploiter groups), in ~1.2-1.5 s.
- **Crash episodes are near misses on low targets** (clean, episode means, crash vs capture episodes): same length
  (1.2-1.6 s vs 1.2-1.5 s), closest approach 0.13-0.18 m before the crash (capture radius 0.1 m), evader lower
  (1.0-1.2 m vs 1.4-1.6 m), pursuer vertical speed about twice as high (1.3-1.7 vs 0.6-1.1 m/s). The pursuer dives at
  the evader, misses, and cannot pull up. **Scripted evaders, which cannot lure deliberately, also win only this way**
  (0.13-0.21), so the dive is a pursuer weakness, not only an evader tactic; learned evaders and exploiters trigger it
  more often.
- **Part of the non-exploiter pursuer crashes may be body contact, not a clean miss.** The Crazyflie collider
  (`cf2x_isaac.usd`) is one 0.10 x 0.10 x 0.03 m box per drone (rotors have none), and the two drones in an env are
  not collision-filtered, so the bodies can touch without a capture at centre distances 0.10-0.145 m, depending on
  relative orientation (capture radius 0.1 m; closest distance sampled at 50 Hz, physics at 100 Hz). Against
  campaign, test and train-pool evaders, 0.25-0.78 of the pursuer-crash episodes come within 0.145 m, and in the
  closest bin (0.10-0.12 m) the evader crashes in the same step in 0.03-0.21 of them, vs 0.00-0.03 beyond 0.145 m,
  consistent with collisions. Against the fresh exploiters only 0.17-0.29 come within 0.145 m and both crash in
  <= 0.07: **the exploit is mainly a clean dive-and-miss.** Contact is not logged, so this is consistent with, not
  proof of, collisions; a real Crazyflie would also crash on such contact.
- **Inside the campaigns, every evader stage learns to lure the current RL pursuers, and the next pursuer stage patches
  it** (4 campaigns: F5 uniform_rls s0 / s1, F2-plain, F3-uniform-s1; each stage's eval vs its training opponents).
  Evader stages 3, 5 and 7 make the stage >= 2 pursuers crash in 0.79-0.99 of episodes (one exception: stage 3 of
  uniform_rls-s1 vs pursuer 2, 0.47); against the stage-0 pursuer (trained on scripted evaders only) they win by
  timeout instead (0.25-0.74). The following pursuer stage captures those evaders in 0.88-0.98 of episodes. The
  leapfrogging does not shrink from stage 3 to stage 7, and the final pursuer is still crashed by the last evader stage in
  0.06-0.24 of episodes. The campaign ends with a pursuer stage, so no evader is trained against the final pursuer;
  the fresh exploiter (from scratch, 100M frames, final pursuer only) plays that role and reaches 0.97.
- **Same reward, no crash weakness in pool-trained pursuers**: the X3 generalists (5 seeds, trained on the fixed thesis
  pool) crash in 0.001-0.004 of episodes against scripted evaders and 0.009-0.033 against RL evaders; evaders beat
  them by timeout (exploiters: timeout 0.04-0.70, crash <= 0.002). The reward settings were not changed during the
  thesis, so the dive is not explained by the reward alone; the self-play opponent distribution is the leading
  explanation (not tested: training length, warm start and sampler also differ).
- **Capture radius 0.15 m instead of 0.10 m (eval-only re-score, added 2026-11-06 18:25; section 5 of
  `outcomes_posthoc.md`).** Exact for the same policies: an episode is identical up to the first step with centre
  distance <= 0.15 m, where a 0.15 m radius would end it as a capture. J at 0.10 m from the episode files reproduces
  `results_exp1.json` exactly. Clean, 0.10 -> 0.15 m: exploitability uniform_rls 0.960 -> 0.690, uniform_plain
  0.973 -> 0.752, uniform_fd 0.965 -> 0.327; held-out RL 0.846 -> 0.981 / 0.736 -> 0.952; other campaigns' evaders
  0.879 -> 0.962 / 0.845 -> 0.948. **The rls - plain advantage shrinks**: held-out RL +0.110 -> +0.030, worst cluster
  +0.171 -> +0.060, other campaigns +0.034 -> +0.014 (train pool unchanged, +0.078 -> +0.079). Noisy: held-out RL
  +0.137 -> +0.071, worst cluster +0.220 -> +0.123, and the noisy deficit against campaign evaders mostly disappears
  (other -0.105 -> -0.020, own -0.164 -> -0.005). So much of the learned estimate's gain is converting passes at
  0.10-0.15 m into captures (final-approach precision), and much of its noisy loss is near misses inside 0.15 m.
  Pool-trained X3 generalists gain too (X3 exploiters, seeds 0-3: 0.30-0.61 -> 0.68-0.92; held-out RL 0.37-0.96 ->
  0.57-0.99): their exploiters' timeouts also contain close passes. Caveat: policies were trained (and exploiters
  trained) for 0.10 m; opponents trained against a 0.15 m radius would keep more distance, so these J values are upper
  bounds for a 0.15 m game, not its result. The 0.10 m radius matches Roncero et al.
- Interpretation (hypothesis): each self-play pursuer stage only has to counter 2-4 specific RL evaders, starting from
  the previous aggressive pursuer, so patching each lure is cheaper than general caution near the ground, which would
  cost fast captures against everything else. Limitations: episode means, not trajectories (the last second is not
  observed); the lower evader altitude in crash episodes may partly be spawn altitude; post-hoc.

## F6_div_sp / F6c (campaigns done 2026-11-06 11:42 / 20:23; F6c done 2026-11-07 04:13; written up 17:45; 2 seeds, descriptive)

**Question:** can a diversity term for the evaders produce more diverse evaders while the pursuer stays as good as
under Roncero's Hedge?

**Setup:**
- Arm **hedge_div** = F3 `hedge` (plain obs, Hedge gamma 0.7 / eta 2 on both roles, 9 stages x 300M, seeds 0 / 100)
  + PSD-PSRO's policy-space diversity regulariser for the evaders only (amendment 1: Hedge instead of QD-DPP).
- Every RL evader after the first maximises beta * min(min_j d_j(s), cap), as a loss term and as a per-step intrinsic
  reward. d_j = mean-only KL to each earlier RL evader of its campaign; cap = 2.
- beta from the training-only pilot (F3 hedge-both-s0, evader stage 5, 100M frames per beta):
  - J vs training pursuers 0.498 (beta 0), 0.101 (beta 0.1), 0.187 (beta 1.0).
  - End distance 165 (beta 0.1) and 39 (beta 1.0).
  - The rule chose **beta = 1.0** ([pilot/choice.json](F6_div_sp/pilot/choice.json)). Both regularised pilot
    evaders beat their pursuers more often than beta = 0.
- F6c = F3c for each final pursuer (train pool, test set, campaign evaders, a fresh 100M-frame exploiter), plus the 8
  F3 final pursuers vs the F6 evaders (probes, used for the evader Vendi score).
- Primary contrast: hedge_div - hedge (same seeds). With 2 seeds, [ ] is the range and nothing can be called
  supported. [F6_div_sp/summary.md](F6_div_sp/summary.md); 0.15 m re-score:
  [R15_rescore/F6_div_sp/summary.md](R15_rescore/F6_div_sp/summary.md).

| final pursuer, J at 0.10 m (0.15 m) | hedge_div | hedge | uniform | qd_dpp | pfsp |
|---|---|---|---|---|---|
| held-out RL (v10) | 0.575 [0.418, 0.731] (0.810) | 0.836 (0.974) | 0.736 (0.952) | 0.696 (0.936) | 0.775 (0.944) |
| held-out worst cluster | 0.467 [0.264, 0.669] (0.708) | 0.754 (0.897) | 0.593 (0.880) | 0.541 (0.859) | 0.665 (0.884) |
| other campaigns' evaders (F3, stage 1 excl.) | **0.417** [0.285, 0.549] (0.721) | 0.888 (0.965) | 0.852 (0.940) | 0.878 (0.949) | 0.626 (0.898) |
| own campaign's evaders | 0.956 (0.991) | 0.959 (0.994) | 0.936 (0.979) | 0.969 (0.996) | 0.776 (0.930) |
| thesis train pool | 0.426 (0.650) | 0.617 (0.784) | 0.570 (0.715) | 0.562 (0.729) | 0.542 (0.787) |
| held-out scripted | 0.827 (0.941) | 0.831 (0.920) | 0.795 (0.901) | 0.821 (0.911) | 0.826 (0.930) |
| X3 exploiters | 0.948 (0.991) | 0.964 (0.990) | 0.951 (0.986) | 0.958 (0.987) | 0.930 (0.986) |
| exploitability | 0.979 (0.686) | 0.979 (0.814) | 0.973 (0.752) | 0.977 (0.695) | 0.983 (0.497) |
| evader Vendi (max 4) | **3.253** [3.176, 3.330] | 2.499 | 2.456 | 2.350 | 2.985 |

hedge_div - hedge (2 pairs [range]), at 0.10 m (0.15 m):

| metric | difference |
|---|---|
| held-out RL | -0.262 [-0.372, -0.151] (-0.165 [-0.314, -0.015]) |
| worst held-out cluster | -0.287 [-0.428, -0.147] (-0.189) |
| other campaigns' evaders | -0.471 [-0.586, -0.356] (-0.244 [-0.388, -0.100]) |
| train pool | -0.191 (-0.134) |
| exploitability | -0.001 (-0.128) |
| evader Vendi | +0.754 [+0.633, +0.876] |

Against qd_dpp / uniform / pfsp, hedge_div is lower on other campaigns' evaders by -0.461 / -0.435 / -0.209 (both
seeds) and has higher evader Vendi by +0.904 / +0.797 / +0.268.

Findings:
- **The diversity term produced more diverse evaders but a much weaker pursuer.**
  - Evader Vendi is 3.33 / 3.18, above every F3 campaign (2.30-3.02) on both seeds.
  - The hedge_div pursuer is below Hedge on held-out RL (-0.262), the worst cluster (-0.287), other campaigns'
    evaders (-0.471) and the train pool (-0.191), on both seeds.
  - Held-out scripted, X3 exploiters (-0.016, mixed seeds), own campaign and exploitability are ~unchanged.
- **The regulariser was saturated, so it acted as a survival bonus, not as a diversity gradient.**
  - Over the last 10 logged updates of evader stages 3 / 5 / 7, the mean min-KL distance was 52 / 15 / 28 (seed 0)
    and 31 / 13 / 18 (seed 1), always far above cap = 2. The lowest logged batch mean was 4.4.
  - The mean intrinsic reward per stage was 1.98-2.00 per step (maximum 2.0). So min(d, cap) was at the cap for almost
    every state, and the loss term had almost no gradient.
  - What remained is about +2 per step while the episode lasts: 20 x the evader's survival reward (`reward_step`
    0.1). Any episode end, including a pursuer crash (neutral in the game reward), ends this stream.
  - The diversity measured by Vendi is therefore a side effect of a different objective, not of the KL gradient. A
    larger cap (or a normalised distance) would be needed to test the PSD-PSRO term itself.
- **The F6 evaders stopped luring, so the hedge_div pursuers never learned to resist it.**
  - Inside the hedge_div campaigns (each stage's eval vs its training opponents), evader stages 3 / 5 / 7 beat the
    stage >= 2 pursuers by a mix of timeout (0.31-0.66) and pursuer crash (0.22-0.45). In F3 hedge-both-s0 the
    same stages make the pursuer crash in 0.96-0.99 of episodes, timeout 0.00.
  - The final hedge_div pursuers then meet the F3 evaders, which do lure. They crash in 0.753 (seed 0) / 0.482
    (seed 1) of those episodes; F3 hedge pursuers crash in 0.144 / 0.105. Timeout is 0 in all four.
  - Against the 8 F3 pursuers the F6 evaders (stage 1 excluded) are about as hard as the F3 evaders: mean J 0.801,
    pursuer crash 0.200; F3 pursuers vs other F3 campaigns' evaders: J 0.811.
  - Interpretation (post-hoc): with the term on, luring stops paying for the evader, and the anti-lure pressure that
    made Hedge's pursuer robust disappears.
- **At 0.15 m (eval-only re-score):**
  - Much of hedge_div's deficit is near misses: hedge_div - hedge is -0.165 on held-out RL and -0.244 on other
    campaigns' evaders. It is still negative on both seeds.
  - hedge_div's exploitability falls to 0.686 vs Hedge's 0.814. This is an upper bound: the exploiters were trained
    for 0.10 m.
- **Predictions (pre-registered 11-05 ~16:00, amendment 1 16:12):**
  - (1) Mechanism, mean distance >= cap/2 at the end of evader stages 3 / 5 / 7: **held**, trivially (13-52 >> 1); the
    term was saturated (see above).
  - (2) Diversity, Vendi above hedge (2.499) and qd_dpp (2.350) on both seeds and >= +0.3 vs hedge: **held**
    (+0.754; both seeds above every F3 value).
  - (3) Pursuer:
    - hedge_div - hedge >= -0.05 on held-out RL and the worst cluster: **did not hold** (-0.262 / -0.287, both
      seeds).
    - >= 0 on other campaigns' evaders: **did not hold** (-0.471).
  - (4) Exploitability unchanged (0.97-0.98): **held** (0.979).
  - The user's target (more diversity at no pursuer cost) was not met.
- **Consequence for the plan:**
  - The regulariser at cap 2 / beta 1 is not a usable thesis method in its current form.
  - Whether F5 Exp 2 is rerun with Hedge + regulariser was to be decided after F6. On this result it should not be,
    unless the term is redesigned (cap / normalisation) and re-piloted.
  - F7 (floor penalty, 0.15 m radius) may change the luring dynamics that this finding depends on, so the decision
    waits for F7c (user).
- Limitations:
  - 2 seeds.
  - Vendi is computed in F3's frozen descriptor space, which includes J vs the F3 pursuers, so a different way of
    winning (timeout vs lure) counts as diversity.
  - The in-campaign stage evals are vs each stage's own training opponents only.
  - div_distance is a batch mean, not per state.

## F7_floor_sp / F7c (campaigns done 2026-11-07 21:22; F7c done 22:30; written up 2026-11-08 14:45; 1 seed per arm, descriptive)

**Question:** evaders beat self-play pursuers by luring them into the floor (F5 post-hoc). Does (a) a pursuer floor
penalty and (b) a 0.15 m capture radius in training remove that weakness?

**Setup** (pre-registered 11-06 19:30):
- Two new campaigns with the F3 uniform protocol (plain obs, uniform sampler on both roles, 9 stages x 300M, campaign
  seed 0), differing from `thesis-F2-plain` (the baseline, trained at 0.10 m) only in:
  - **r15**: capture radius 0.15 m in training and in the in-campaign evals.
  - **floor_r15**: the same + pursuer floor penalty `reward_floor_weight` = 1.0 (the evader's floor ramp: 0 at 0.6 m
    -> 1 per step at the 0.1 m floor). The pursuer's own crash already costs -100.
- F7c: the three final pursuers (stage 8), clean, evaluated at 0.10 m with per-episode files and re-scored exactly at
  0.15 m. Opponents: train pool, test set, the F5c campaign evaders, both F7 campaigns' RL evaders (stage 1 excluded),
  and **one fresh exploiter per pursuer trained under the 0.15 m rules** (100M frames, seeds 800 / 801 / 802).
- **0.15 m is the primary radius here** (the F7 pursuers were trained for it). The 0.10 m numbers are given in
  parentheses; for the F7 pursuers they are off their training radius, for F2-plain the 0.15 m numbers are a re-score.
- [F7_floor_sp/summary.md](F7_floor_sp/summary.md), [results.json](F7_floor_sp/results.json).

| final pursuer, J at 0.15 m (0.10 m) | F2-plain | r15 | floor_r15 |
|---|---|---|---|
| exploitability vs its 0.15 m exploiter | 0.002 (0.022) **failed exploiter, see below** | **0.963** (0.989) | **0.968** (0.989) |
| pursuer crash vs that exploiter | 0.006 (0.036) | 0.965 (0.991) | 0.970 (0.991) |
| F2-plain vs its earlier 0.10 m exploiter (F5c), exact re-score | **0.755** (0.974) | | |
| held-out RL (v10) | 0.959 (0.777) | 0.934 (0.641) | 0.827 (0.490) |
| held-out worst cluster | 0.914 (0.646) | 0.886 (0.518) | 0.751 (0.303) |
| other campaigns' evaders (F5c sets) | 0.963 (0.850) | 0.942 (0.708) | 0.922 (0.650) |
| other F7 campaign's evaders | 0.901 (0.744) (both F7) | 0.844 (0.564) | 0.850 (0.496) |
| own campaign's evaders | 0.968 (0.918) | 0.944 (0.592) | 0.966 (0.620) |
| thesis train pool | 0.766 (0.601) | 0.731 (0.510) | 0.741 (0.520) |
| held-out scripted | 0.928 (0.834) | 0.900 (0.772) | 0.912 (0.791) |
| pursuer crash vs held-out scripted | 0.072 (0.171) | 0.100 (0.231) | 0.089 (0.213) |
| X3 exploiters | 0.988 (0.956) | 0.975 (0.902) | 0.983 (0.914) |

Contrasts at 0.15 m (0.10 m), 1 seed: floor_r15 - r15: held-out RL -0.107 (-0.150), worst cluster -0.135 (-0.215),
other campaigns -0.020 (-0.059), exploitability +0.005 (+0.000); train / scripted / own / X3 within +/- 0.03.
r15 - plain: held-out RL -0.025 (-0.136), worst cluster -0.028 (-0.128), other campaigns -0.021 (-0.141), other F7
evaders -0.056 (-0.180), train -0.035 (-0.091).

**F2-plain's 0.15 m exploiter failed to train (checked 11-08):**
- All three exploiters were set up identically (from scratch, 100M frames, 762 iterations); only the opponent and
  seed differ. The F7 exploiters learned to lure within the first ~12 % of training (pursuer crash 0.87 at that point,
  0.93-0.95 at the end). F2-plain's exploiter never did: pursuer crash <= 0.043 at every logged update, and capture 0.89-0.99 from the 21st
  of 263 updates on.
- Under the 0.10 m rules all 12 earlier exploiters (F3c, F5c, including F2-plain's, seed 500) learned to lure
  (pursuer crash > 0.5 after 14-165 of 263-381 logged updates, 0.92-0.98 at the end).
- Luring F2-plain does not need the evader inside 0.15 m: in 77 % of the episodes in which its 0.10 m exploiter made
  it crash, the closest distance stayed > 0.15 m (median 0.178 m). Because the re-score is exact for the same two
  policies, that exploiter beats F2-plain under the 0.15 m rules in **0.755** of episodes, all by pursuer crash.
- So F2-plain's exploitability in the 0.15 m game is **>= 0.755**, not 0.002. The pre-registered "r15 - plain"
  exploitability contrast (+0.961) is an artefact of the failed exploiter; against the best known evader it is
  <= +0.208.

Findings:
- **Neither change removes the luring weakness.**
  - Both F7 pursuers are fully exploitable in the game they were trained for (0.963 / 0.968), and their exploiters
    win almost only by pursuer crash (0.965 / 0.970).
  - Those exploiters lure from farther away: the median closest distance in crash episodes is 0.244 m (r15) /
    0.233 m (floor_r15), and only 2-3 % of them come within 0.15 m. For the 0.10 m exploiters of F5c it was
    0.142-0.198 m. Post-hoc reading: a larger capture radius pushes the lure outwards; the pursuer still follows into
    the floor.
  - Inside both campaigns, evader stages 3 / 5 / 7 still make the stage >= 2 pursuers crash in 0.91-0.99 (floor_r15)
    / 0.87-0.98 (r15) of episodes, as in F2 / F3 / F5 at 0.10 m (0.79-0.99); the next pursuer stage again patches the
    latest evader (stage 8 vs stage 7: pursuer crash 0.08 / 0.14).
- **The floor penalty makes the pursuer worse on held-out RL evaders, and changes little else.**
  - floor_r15 - r15 at 0.15 m: held-out RL -0.107, worst cluster -0.135; train pool, held-out scripted, own campaign
    and X3 exploiters within +/- 0.03; crash rate vs held-out scripted 0.089 vs 0.100.
  - Post-hoc reading: the penalty adds at most 1 per step near the floor, next to a -100 crash penalty the pursuer
    already has. It adds shaping, not a new incentive against crashing, and the evaders still find a lure.
- **Training at 0.15 m does not help the pursuer.** r15 is at or below F2-plain on every metric at 0.15 m (held-out
  RL -0.025, other campaigns -0.021, other F7 evaders -0.056, train -0.035), and lower at 0.10 m (-0.05 to
  -0.33), because it learned to close only to 0.15 m.
- **Cross-play:** the F7 evaders lure F2-plain less than they lure the F7 pursuers. At 0.15 m F2-plain crashes in
  0.124 (floor_r15 evaders) / 0.079 (r15 evaders) of episodes; each F7 pursuer crashes in 0.157 / 0.161 of episodes
  against the other F7 campaign's evaders, 0.035 / 0.058 against its own.
- **Predictions (pre-registered 11-06 19:30; J at 0.15 m):**
  - (1) floor_r15 pursuer crash vs its exploiter < 0.50 and exploitability <= r15 - 0.10: **did not hold** (0.970;
    0.968 vs 0.963).
  - (2) r15's exploiter wins mainly by pursuer crash: **held** (pursuer crash 0.965, exploitability 0.963). r15 exploitability within +/- 0.10
    of F2-plain's: **did not hold as measured** (0.963 vs 0.002), but that measurement is invalid (failed exploiter);
    with the best known evader F2-plain is >= 0.755, so this part is **undecided** (it holds if F2-plain's true value
    is >= 0.863).
  - (3) floor_r15 pursuer crash vs held-out scripted < 0.05: **did not hold** (0.089; 0.213 at 0.10 m).
  - (4) floor_r15 held-out RL >= r15 - 0.05: **did not hold** (0.827 vs 0.934).
  - (5) In floor_r15, evader stages 3 / 5 / 7 make the stage >= 2 pursuers crash in < 0.50: **did not hold**
    (0.91-0.99).
- **Decision rule (pre-registered):** (1) did not hold, so the floor penalty + 0.15 m package is not recommended. r15
  does not lower exploitability either (0.963 vs >= 0.755 for F2-plain), so neither is the radius alone. **The rules
  stay** (0.10 m capture radius in training, no floor penalty; J still reported at both radii). This does not depend
  on the failed F2-plain exploiter. Freeze and F5 Exp 2 / F6 redesign: the user decides.
- Limitations:
  - 1 campaign per arm and 1 exploiter per pursuer, one of which failed; no second exploiter seed for F2-plain.
  - One floor-penalty weight (1.0).
  - F2-plain was trained at 0.10 m, so the r15 - plain contrast mixes the training radius with the evaluation radius.
  - The in-campaign evals are vs each stage's own training opponents only.
