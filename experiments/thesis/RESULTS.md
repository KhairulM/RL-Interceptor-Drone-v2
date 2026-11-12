# Results (thesis experiments)

Measured results only. Each section is filled when the experiment finishes, with exact numbers, CIs, number of seeds/episodes, and any deviation from [the plan](../../docs/research_plan.md). Predictions are in the plan and are compared here as "prediction held / did not hold".

**Standing conventions (apply to every section).**
- J = neutralisation rate (capture, evader crash or evader out of bounds), mean over seeds; [ ] = 95 % bootstrap CI over seeds (with 2 seeds, the range of the two values); "supported" = paired CI excludes 0 (coarse with 3-5 seeds). M0 = plain pursuer (observes evader position + velocity, no acceleration).
- Training radius and primary results: 0.10 m. **0.15 m values are eval-only exact re-scores** of the same episodes ([R15_rescore/](R15_rescore/), `rescore.py`). Every policy and exploiter was trained for 0.10 m, so **J at 0.15 m is an upper bound and exploitability a lower bound for a 0.15 m game**. Sections below write "(0.15 m convention)" for this. Exception: F7 trained its pursuers and exploiters at 0.15 m.

## Summary (updated 2026-11-13 00:10; details in the sections below)

One finding per bullet, 0.10 m unless marked; numbers are copied from the linked sections.

**Diagnostics (X0-X4)**
- The game is mostly transitive; self-play pools are redundant (22 RL evaders = 3.8 distinct behaviours) ([X1](#x1), [X2](#x2)).
- The pursuer loses up to 0.37 J to a specialist on unseen learned evaders. Most of this gap is missing opponent state: adding the evader's acceleration lifts J 0.65 -> 0.94 (train), 0.47 -> 0.77 (held-out RL), 0.31 -> 0.97 (exploiters) ([X3](#x3), [X4](#x4)).

**RQ1: opponent sampling**
- Fixed pool (5 seeds): QD-DPP beats hardness-prioritised samplers (held-out RL +0.19 vs multi-metric, +0.13 vs win-rate PFSP) but **not uniform** (+0.04, n.s.); hardness-only prioritisation is worse than uniform there ([D1](#d1)).
- The DPP caps a duplicated opponent's sampling mass but does not protect performance ([D1b](#d1b)); duplication robustness appears only without a uniform floor ([D1c](#d1c), exploratory).
- No sampler lowers exploitability (0.44-0.69, [D2](#d2)); with the acceleration input the sampler has no effect ([F1](#f1)).
- Self-play, samplers on both roles (2 seeds, descriptive): **Hedge (Roncero) is best** (held-out RL +0.10, worst held-out cluster +0.16 vs uniform, both seeds); QD-DPP is not better than uniform (-0.04 / -0.05); PFSP gives the most diverse evaders (Vendi 2.99 vs 2.46) but the most specialised pursuer (other campaigns' evaders -0.23); every pursuer stays ~fully exploitable (0.97-0.98) ([F3](#f3)).
- A diversity term in evader training does not help: F6 makes the evaders more diverse (Vendi 3.25 vs Hedge 2.50) but the pursuer much weaker (held-out RL -0.262, other campaigns -0.471) ([F6](#f6)); the redesigned F6b restores luring but not diversity (Vendi 2.365 vs 2.499) and stays below Hedge (held-out RL -0.128, worst cluster -0.190) ([F6b](#f6b)).
- **Answer to RQ1: not robustly. QD-DPP beats badly tuned hardness prioritisation on a fixed pool (D1), but is not better than uniform (D1, F3) and is below Roncero's Hedge in self-play (F3, 2 seeds).**

**RQ2: opponent model / online adaptation**
- Noise-free: a learned acceleration estimate equals the finite difference (n.s.), and both beat M0 (+0.18 train, +0.23 held-out RL, +0.49 exploiters) ([A2b](#a2b)). Online RLS adds nothing; ESN and raw history are worse than none ([A1b](#a1b)).
- On the fixed pool the acceleration input dominates (train +0.20, exploiters +0.46; held-out bimodal per seed) ([F1](#f1)) and lowers exploitability 0.672 -> 0.397 (paired -0.275 [-0.522, -0.044]) ([F2a](#f2a)).
- Sensor noise (policies trained clean): the finite-difference input collapses (train 0.94 -> 0.33); learned + RLS holds (0.80) and beats it on every split (+0.20 to +0.49). **RLS beats the frozen estimator** (+0.09 train, +0.24 exploiters): the only closed-loop gain from online adaptation ([A3c](#a3c)).
- No evader velocity observed: every pursuer is weak (~0.29 vs 0.65) ([A3b](#a3b)); an estimated velocity lifts train J to 0.38 (EMA) / 0.40 (learned; true velocity 0.62) but does not help on held-out RL ([A3d](#a3d)). Offline, a noise-trained GRU is the best estimator (velocity -36 % vs the closed-loop MLP) ([A3e](#a3e)); a spectral-norm penalty changes the error by <= 2 %, not adopted ([A3f](#a3f)).
- Self-play, finite-difference acceleration (1 campaign per arm): better on held-out RL (0.89 vs 0.78), not on the other campaign's evaders (0.74 vs 0.86); prediction did not hold ([F2b / F2c](#f2)).
- Self-play with noise and no true velocity (3 seeds per arm): learned estimators beat the EMA filter on other campaigns' evaders (+0.196 [+0.132, +0.291]) and X3 exploiters (+0.441 [+0.254, +0.541]) but are **more** exploitable (0.872 vs 0.675) ([F4](#f4)).
- Self-play, learned estimate + RLS (3 seed pairs): clean, held-out RL +0.091 [+0.055, +0.116] and worst cluster +0.128 [+0.042, +0.174] on all 3 pairs, not other campaigns' evaders (-0.027); under noise below plain (other campaigns -0.140 [-0.232, -0.065]), mainly from running RLS (-0.103). **The A3c RLS gain is not reproduced in self-play** ([F5](#f5)).

**Self-play robustness and the floor-lure exploit (F2-F10, F8d)**
- Under the reference papers' protocol (9 x 300M) every self-play pursuer is ~fully exploitable (0.94-0.98; F2c, F3, F5, F6, F6b) and the exploiters win almost only by making it crash (97 % in F2c), a near-miss dive at a low evader. Inside the campaigns each evader stage lures the current pursuers (0.79-0.99 crash) and the next pursuer stage patches it; pool-trained pursuers are not lured (crash <= 0.03) ([F5 post-hoc](#f5-posthoc)).
- A 0.15 m training radius, with or without a pursuer floor penalty, does not remove the weakness (exploitability 0.963 / 0.968 vs exploiters trained under the 0.15 m rules, 1 seed per arm). **Decision rule: the rules stay** (0.10 m, no floor penalty) ([F7](#f7)).
- Under Roncero et al.'s reward and episode endings (26 stages x 50M, 2 seeds) the exploit disappears: no evader stage lures, and a thesis-rules exploiter crashes the final pursuer in only 0.010 / 0.030 of episodes (F3 hedge 0.975 / 0.988). Vs F3 hedge: held-out RL +0.157 / +0.060, exploitability 0.631 / 0.513 by near-miss timeouts. The game stays as transitive as F3's ([F8](#f8), [post-hoc](#f8-posthoc)).
- Against the final evaders of 16 other campaigns the final F8 pursuer beats Hedge's by +0.189 / +0.126, and F8 already beats Hedge's final pursuer at stage 8 (250M pursuer frames vs Hedge's 1.5B) ([F8d](#f8d)).
- One Roncero component at a time (1 seed): **the control arm (thesis rules, GAE truncation fix, 13 x 50M) is already hardly lurable** (lure 0.36, exploiter crash 0.056, exploitability 0.440); removing the timeout terms restores luring (0.75; exploiter crash 0.955); truncation endings stop evaders from learning to lure (0.03) but leave the pursuer lurable (0.834) ([F9](#f9)).
- **The schedule, not the GAE fix, makes the pursuer lure-resistant**: without the fix the evaders lure (0.81), yet the final pursuer is not lurable (crash 0.031) and is the least exploitable thesis-rules self-play pursuer so far (0.172). F3-F7's ~0.97 exploitability is mainly a 9 x 300M schedule effect ([F10](#f10)).
- Thesis v2d rules (truncation endings, no evader timeout bonus, 26 x 50M, 2 seeds): the pursuer is not lured by a v2d-rules exploiter (crash 0.016 / 0.028) but **is lured by a thesis-rules exploiter (crash 0.309 / 0.636; F8 0.010 / 0.030, F10 0.031)** and generalises less well (held-out RL 0.875 / 0.869 vs F8 0.948 / 0.941; strong finals 0.845 / 0.751 vs 0.858 / 0.875). 4 of 5 predictions did not hold, (5) held on one seed; v2d improves on neither F8 nor F10 ([F11](#f11)).

**Capture radius 0.15 m** (0.15 m convention; F6b added 2026-11-10; F8 reports both radii itself)
- Most directions hold; many 0.10 m gaps were partly final-approach precision.
- RQ1: QD-DPP still beats the hardness samplers on a fixed pool and still not uniform (D1); Hedge - uniform in self-play shrinks to held-out RL +0.022 (F3). Exploitability falls to 0.09-0.22 pool-trained (D2) and 0.50-0.81 in self-play (F3).
- RQ2: the acceleration input still dominates but less (train +0.153, exploiters +0.180; F1). Under noise RLS - frozen gains less (train +0.072, CI includes 0; exploiters +0.073; A3c). F5's clean held-out RL gain shrinks to +0.021; F4 nf - ema held-out RL becomes significant (+0.170 [+0.028, +0.404]) and the exploitability difference disappears.
- Diagnostics and F2c: the adaptation gap rests on 2 exploiters (X3); gate G1 still passes (X4); the game is more transitive (X1). F2c acc exploitability 0.327 vs plain 0.746 (one exploiter each).

**Running / pending (updated 2026-11-13 01:20)**: F13 (original thesis rules, GAE fix, Hedge, 13 x 50M; seeds 100 / 200 + F9 control as seed 0) running, results ~08:30; decides whether F12 (stopped after 10 of 26 stages) is resumed on v2d or re-run on F13's setup. Default held-out test for new experiments: v10 + F8d's 16 strong finals. No freeze date (user, 11-09). Open (user): opponent sensor noise in the noisy runs (A3b-d, F4, F5; audit), the seed-CI method, F5 Exp 2 (paused), the exploiter-augmented pool, a sampler / diversity comparison under v2d.

## Contents

- Cross-experiment: [Self-play synthesis](#synthesis) · [Relation to the reference papers](#refs)
- Diagnostics: [X0 replay](#x0) · [L0 literature](#l0) · [X1 cross-play](#x1) · [X2 descriptors](#x2) · [X3 gap](#x3) · [X4 oracles / G1](#x4)
- RQ1 on a fixed pool: [D1 sampler](#d1) · [D1b duplicates](#d1b) · [D1c no floor](#d1c) · [D2 exploitability](#d2)
- RQ2 opponent model: [A1a](#a1a) · [A1b](#a1b) · [A2a](#a2a) · [A2b](#a2b) · [A3a](#a3a) · [A3b](#a3b) · [A3c](#a3c) · [A3d](#a3d) · [A3e](#a3e) · [A3f](#a3f)
- Final runs on a fixed pool: [F1](#f1) · [F2a](#f2a)
- Self-play: [F2b / F2c](#f2) · [F3](#f3) ([transitivity](#f3-posthoc)) · [F4](#f4) · [F5](#f5) ([post-hoc](#f5-posthoc)) · [F6](#f6) · [F7](#f7) · [F6b](#f6b) · [F8](#f8) ([post-hoc](#f8-posthoc), [F8d](#f8d)) · [F9](#f9) · [F10](#f10) · [F11](#f11)

<a id="synthesis"></a>

## Self-play synthesis: RQ1 and RQ2 from the self-play campaigns only (0.10 m; written 2026-11-09)

Only the self-play campaigns at the 0.10 m training radius: F2b / F2c, F3, F4, F5 Exp 1, F6, F6b, and X2's analysis of the earlier self-play pools. Fixed-pool experiments and F7 (trained for 0.15 m) are left out. Numbers are copied from the sections below; their 0.15 m versions are there.

**How much the evidence can carry.** F2b / F2c (1 campaign per arm), F3, F6 and F6b (2 seeds each): descriptive, no difference can be called supported. F5 Exp 1 (3 seed pairs): a CI that excludes 0 counts as supported, with coarse CIs and the disclosed optional-stopping risk of amendment 2. F4 (3 seeds per arm): the only fully inferential self-play experiment. So in self-play, RQ1 rests on descriptive results only; RQ2 has CI-backed evidence (F4, F5).

**RQ1: does quality + diversity opponent sampling give a more robust pursuer than hardness-only sampling? In self-play: no. Roncero's Hedge is the best sampler; more evader diversity goes with a weaker pursuer.**

| evidence | result |
|---|---|
| Self-play pools are redundant (H1a) | Yes: 22 RL evaders from three campaigns = 3.8 effectively distinct behaviours (X2); F3 arm means 2.35-2.99 of 4 per campaign |
| Best sampler (F3, sampler on both roles, 2 seeds) | Hedge: vs uniform held-out RL +0.100, worst held-out cluster +0.161, positive on both seeds |
| QD-DPP (F3) | Not better than uniform (held-out RL -0.041, worst cluster -0.052); below Hedge (-0.140 / -0.213) |
| PFSP, hard opponents only (F3) | Most diverse evaders (Vendi +0.529 vs uniform), most specialised pursuer (other campaigns' evaders -0.226) |
| Diversity term in evader training (F6, Hedge + PSD-PSRO) | Evaders more diverse (Vendi 3.253 vs 2.499); pursuer much weaker (held-out RL -0.262, other campaigns' evaders -0.471) |
| Redesigned diversity term (F6b, 2 seeds) | Did not increase diversity (Vendi 2.365 vs 2.499); held-out RL -0.128, worst cluster -0.190, other campaigns -0.002 vs Hedge. In-campaign lure crash rates restored to 0.949-0.991. |
| Exploitability | 0.97-0.98 for every sampler (F3); 0.979 for F6 and 0.983 for F6b: no sampler or diversity term changes it materially |

- H1c (QD sampling improves the worst case and lowers exploitability) did not hold in self-play.
- H1b (robustness to duplicated opponents) was tested on a fixed pool only (D1b, D1c), not in self-play.

**RQ2: does an opponent model / online adaptation help? In self-play: the opponent model helps against held-out evaders; online adaptation (RLS) does not; neither lowers exploitability.**

| evidence | result |
|---|---|
| Finite-difference acceleration input (F2c, 1 seed per arm) | Held-out RL +0.116, test set +0.069; worse vs the other campaign's evaders (0.741 vs 0.863) and the train pool (-0.106); exploitability 0.965 vs 0.974; the acc campaign barely leapfrogs |
| Learned acceleration estimate + RLS, clean (F5 Exp 1, 3 pairs) | Held-out RL **+0.091 [+0.055, +0.116]**, worst cluster **+0.128 [+0.042, +0.174]**, positive on all 3 pairs; other campaigns' evaders -0.027 (CI includes 0); exploitability 0.957 vs 0.974 |
| Online adaptation, RLS vs the same estimator frozen (F5) | Clean: -0.001 on held-out RL. Under eval noise: held-out RL +0.031, other campaigns' evaders **-0.103 [-0.140, -0.069]** |
| Sensor noise at eval only (F5) | Learned estimate below plain vs self-play evaders (other campaigns **-0.140 [-0.232, -0.065]**); finite difference collapses (-0.472, 1 seed) |
| Noisy training, no true velocity (F4, 3 seeds) | Learned estimators vs EMA filter: other campaigns' evaders **+0.196 [+0.132, +0.291]**, X3 exploiters **+0.441 [+0.254, +0.541]**, held-out RL +0.122 (CI includes 0); but more exploitable, **+0.197 [+0.125, +0.277]** (0.872 vs 0.675), by pursuer crash in 2 of 3 seeds |

- The held-out gain from opponent-acceleration information appears in F2c, F5 (clean) and F4 (noisy). It does not carry over to other campaigns' self-play evaders except in F4, and it never lowers exploitability.
- RLS adds nothing clean and hurts against self-play evaders under noise, the opposite of the fixed-pool A3c result.
- H2a / H2b (adaptation gap, oracles) were measured on a fixed pool only (X3, X4).

**Common to both questions.** Every self-play pursuer with the true evader velocity stays ~0.94-0.98 exploitable (F2c, F3, F5, F6, F6b), whatever the sampler, diversity term or opponent model, and the fresh exploiters win almost only by pursuer crash; F4 (noisy, estimated velocity) is the exception (0.675-0.872, partly by timeout). Interpretation (not tested): in self-play the pursuer's robustness is limited by this one failure mode more than by opponent sampling or opponent modelling. **F8 / F9 (added 2026-11-10, rewritten 11-11)**: the failure mode depends on the training setup. Under Roncero et al.'s rules the final pursuers are not lured even by a thesis-rules exploiter (pursuer crash 0.010 / 0.030); the thesis rules with 50M-frame stages are already hardly lurable, with or without the GAE fix (exploiter crash 0.056 / 0.031, F9 / F10), so the robustness ceiling seen in F2-F7 is mainly a schedule effect (9 x 300M), not the rules or the sampler.

<a id="refs"></a>

## Relation to the reference papers (updated 2026-11-11 11:30)

Reference papers (the thesis is positioned against these three): **[AgilePE]** Tang et al. 2026, arXiv:2608.14135; **[Roncero]** Sanchez Roncero et al. 2026 (AMSPBH), arXiv:2506.02849v3; **[Gavin]** Gavin & Bronz, ICUAS 2026 (DOI 10.1109/ICUAS69441.2026.11598610). Numbers are **not** comparable across papers (different arenas, capture definitions, metrics, simulators); the comparison is between *claims and mechanisms*, tested in our setting. "Status" says whether our data agrees, disagrees under our protocol, extends the claim, or does not test it.

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
| 1 | Prioritising hard opponents (Hedge / PFSP) improves on uniform population sampling. Roncero Fig. 5: Hedge beats uniform AMSPB by +0.05 to +0.27 win rate in 8 of 9 settings (3 seeds). AgilePE: PFSP stronger than FSP (single runs). | D1 (5 seeds, fixed 20-evader pool), minus uniform: win-rate softmax (S-W, closest to Hedge/PFSP) train **-0.054 [-0.081, -0.026]**, held-out RL **-0.088 [-0.145, -0.035]**, worst held-out cluster **-0.088 [-0.147, -0.030]**, worse on 5/5 seeds; multi-metric softmax (S-M, closest to Gavin's outcome score) train **-0.150 [-0.259, -0.047]**, exploiters **-0.166 [-0.302, -0.047]**, held-out RL -0.143 [-0.306, +0.021]. D2: no sampler lowers exploitability. Self-play, sampler on both roles (F3, 2 seeds): Hedge beats uniform on held-out RL (+0.100) and the worst held-out cluster (+0.161) on both seeds; PFSP without a uniform share is mixed on held-out RL (+0.038) and much worse against other campaigns' evaders (-0.226). | **Replicates directionally in self-play (Hedge, F3, 2 seeds), not on a fixed pool (D1).** PFSP without a uniform share specialises (F3, D1c). The fixed-pool test differs: one pursuer against a fixed pool (no co-evolving generations), uniform floor 0.5 (Roncero 0.7), softmax (T = 0.9) instead of a power eta = 2. Latest-opponent-only self-play (Roncero's AMSDRL, Gavin's SP) was not run, so "population beats latest-only" is not tested. D1c (exploratory): with **no** uniform floor hardness prioritisation collapses (train -0.29), in line with Roncero's advice to keep gamma large; a warning for PFSP without a uniform share (AgilePE does not report one). |
| 2 | Self-play with PFSP "sustains tactical diversity" (AgilePE); the population "contains distinct pursuit and evasion behaviours" (Roncero). Neither paper measures diversity. | X2: 22 RL evaders from three self-play campaigns = **3.8 effectively distinct behaviours** (Vendi score), 7 clusters; consecutive stages are 36 % closer than random same-campaign pairs. F3 (2 seeds): evader Vendi per campaign (max 4) pfsp 2.985 > hedge 2.499 ~ uniform 2.456 > qd_dpp 2.350; pfsp - uniform +0.529 on both seeds. F6: a PSD-PSRO diversity term on Hedge raises Vendi to 3.253 (+0.754), but the term was saturated (a survival bonus), its evaders stopped luring, and its pursuer is much weaker (held-out RL -0.262, other campaigns' evaders -0.471). F6b (redesigned, unsaturated): the evaders keep luring but Vendi is 2.365 vs Hedge 2.499, and the pursuer stays below Hedge on held-out RL (-0.128, mixed seeds). | **Extends:** first quantitative measure; the pool is diverse across campaigns but highly redundant within one. **Consistent with AgilePE:** PFSP self-play produces the most diverse evaders (F3), but its pursuer is the most specialised. **More evader diversity does not by itself give a better pursuer** (F3 pfsp, F6), and a diversity term weak enough to keep the evaders effective adds no diversity (F6b). |
| 3 | Our addition: sample by quality *and* diversity (QD-DPP). No reference paper uses diversity in sampling. | D1: QD-DPP beats both hardness samplers (vs S-M held-out RL +0.185 [+0.110, +0.259]; vs S-W +0.131 [+0.038, +0.223]) but is **not** better than uniform (+0.042 [-0.069, +0.131]). D1b: bounds the sampling mass of duplicates, does not protect performance. F1: no sampler effect once the acceleration input is present. D2: not less exploitable. F3 (self-play, 2 seeds): not better than uniform (held-out RL -0.041, worst cluster -0.052), below Hedge (-0.140 / -0.213), least diverse evaders (Vendi 2.350). F6 / F6b: see row 2. | **New, negative:** diversity repairs the damage of badly tuned hardness prioritisation on a fixed pool (D1); in self-play it is no better than uniform and worse than Roncero's Hedge (F3). |
| 4 | Population training generalises to unseen opponents (Roncero: unseen APF evader; Gavin: SP/PFSP pursuers vs other evaders). | Scripted opponents: held-out scripted evaders are neutralised at J 0.99-1.00 by every pursuer. Learned opponents: against the independent v10 lineage the generalist loses up to 0.37 J to a specialist; fresh exploiters reach J 0.22-0.43 against it (X3); mean exploitability per sampler 0.44-0.69 (D2). With the reference papers' co-evolution protocol (F2b/F2c, 1 seed per arm) the final pursuers neutralise held-out scripted evaders at only 0.83-0.88 and the train pool's v9 evaders at 0.34-0.48, and are **~fully exploitable (0.97)** by pursuer crash (97 %); so is every later self-play pursuer (F3c 0.97-0.98; F5 Exp 1 0.95-0.97; F6 0.98), and a 0.15 m training radius does not change this (F7: 0.963 / 0.968). **Under Roncero's own reward and episode endings (F8, 2 seeds) the crash exploit disappears**: a thesis-rules exploiter crashes the final pursuer in only 0.010 / 0.030 of episodes, the pursuer beats F3 hedge on held-out RL (+0.157 / +0.060), and exploitability is 0.631 / 0.513 by near-miss timeouts. F9 (1 seed): the thesis rules with the GAE truncation fix and 50M-frame stages are already hardly lurable (exploiter crash 0.056); without the timeout terms the pursuer is lurable again (0.955). Gavin's rules give the evader the same lure incentive as the thesis rules (literature notes). | **Consistent for scripted, contradicted for learned opponents.** None of the three papers reports exploitability or held-out RL lineages; under their protocol the gap is larger, not smaller. **Extends Roncero:** his rules remove the crash exploit that the thesis and Gavin's rules allow and give a pursuer that generalises better than Hedge under the thesis rules (F8); part of the effect comes from the training setup (GAE truncation fix, 50M-frame stages; F9). |
| 5 | Seeds may converge to different strategies (Roncero, limitations). | Generalist (uniform sampler) train J 0.56-0.87 across 5 seeds; held-out generalisation is **bimodal per seed**: 1-2 of 5 seeds fail on all later v10 evaders (J 0.10-0.29) while the others reach 0.74-0.97 (F1). | **Extends:** quantifies it; with 3 seeds (Roncero) effects of +/- 0.1 can come from one seed. |
| 6 | Observation design: Roncero and Gavin give the pursuer opponent position + velocity to a memoryless MLP; Gavin leaves "trajectory prediction" out of scope; AgilePE adds a temporal window of opponent state. | Adding the opponent's acceleration (finite difference of two consecutive observations) is the **largest effect in this thesis**: train 0.722 -> 0.940 (+0.199 [+0.128, +0.270]), exploiters 0.497 -> 0.983 (+0.464 [+0.239, +0.638]), held-out RL 0.634 -> 0.838 (CI includes 0, bimodal seeds) (F1, 5 seeds). A learned estimate does the same (A2b: within +/- 0.04, n.s.). Exploitability 0.672 -> 0.397, paired -0.275 [-0.522, -0.044] (F2a). A raw 10-step history stack (our analogue of a temporal window) is *worse* than no history (train -0.254 [-0.355, -0.152]) (A1b). Under co-evolution the gain is smaller: F2c (1 seed per arm) held-out RL +0.12, whole test set +0.07, worse on the other campaign's evaders (0.74 vs 0.86), exploitability 0.965 vs 0.974; F5 Exp 1 (learned estimate, 3 pairs) held-out RL +0.091 [+0.055, +0.116], worst held-out cluster +0.128 [+0.042, +0.174], other campaigns' evaders -0.027, exploitability unchanged (0.96 vs 0.97). At 0.15 m the held-out gain shrinks to +0.021. | **New:** a memoryless pursuer with position + velocity (Roncero, Gavin) is limited by the missing opponent acceleration far more than by opponent sampling, when trained against a fixed pool; under co-evolution the gain is smaller, limited to held-out evaders, largely final-approach precision (0.15 m re-score), and it never lowers exploitability. Caveat: AgilePE's window length and architecture are not given, so the history-stack result is not a test of AgilePE itself. |
| 7 | Future work in Roncero: online adaptation, recurrent policies. | Noise-free, online RLS adaptation of a learned opponent model adds nothing in closed loop (A1b, A2b); a random reservoir (ESN) is worse than no adaptation. **Under sensor noise it helps**: RLS vs frozen basis +0.09 train, +0.24 exploiters, CIs exclude 0 (A3c, policies trained noise-free). Recurrent policies (GRU) **not tested**; a recurrent *estimator* (GRU, trained on noisy inputs) is the most accurate opponent-state estimator under noise offline (A3e: velocity -36 % vs the MLP used in closed loop). In self-play (F5 Exp 1, 3 pairs) RLS vs frozen changes nothing clean (all within +/- 0.015); under noise held-out RL +0.031, but other campaigns' evaders -0.103 [-0.140, -0.069]. | **New, conditional:** online opponent-model adaptation pays off under sensor noise against a fixed pool (A3c), not with perfect state (Roncero's setting); in self-play (F5) it does not pay off. |
| 8 | Sensor noise: AgilePE reports robustness under real sensor noise (mocap + IMU); Roncero and Gavin assume perfect state. | With 2 cm / 8 cm/s noise (eval only, policies trained clean) the finite-difference acceleration input collapses (train 0.94 -> 0.33, below the plain pursuer 0.70); the learned estimator + RLS keeps 0.80 and beats it by +0.47 train / +0.38 held-out RL / +0.49 exploiters (A3c, 5 seeds). Without any evader velocity every pursuer is weak (~0.29) (A3b); an *estimated* velocity from noisy positions gives train 0.29 -> 0.38-0.40 (true velocity 0.62), no gain on held-out RL (A3d). Self-play with noise in training and evaluation (F4): learned estimators generalise better than the EMA filter (other campaigns' evaders +0.196, X3 exploiters +0.441) but make the pursuer more exploitable (+0.197). Self-play with noise at eval only (F5): the finite-difference pursuer falls below plain (other campaigns' evaders -0.472); the learned estimate is below plain against the self-play evaders (-0.140), through more pursuer crashes with RLS running. | **New:** opponent-acceleration inputs must be learned, not differenced, once sensing is noisy; the perfect-state results of Roncero and Gavin would not transfer to such inputs. Under co-evolution with noisy sensing the learned estimator trades better generalisation for higher exploitability. |
| 9 | Real-world transfer (all three), multi-pursuer coordination (Gavin), FOV-limited sensing (AgilePE). | Not studied. | **Not tested.** |

<a id="x0"></a>

## X0_replay (done 2026-10-26)

Final pursuer of each campaign re-evaluated against 3 RL evaders recorded in its `matchups.json`, with the new multi-episode estimator (cutoff 1500 steps, 4096 envs). Full table: [X0_replay/summary.md](X0_replay/summary.md).

| campaign | reproduced (\|delta capture\| <= 0.05) | max \|delta\| | max \|z\| | verdict |
|---|---|---|---|---|
| v6 | 1 / 3 | 0.906 | 119.7 | **invalid**: evaders leave the arena in 85-99.8 % of episodes under the current env rules; excluded |
| v9 | 3 / 3 | 0.022 | 1.0 | valid |
| v10 | 3 / 3 | 0.009 | 0.8 | valid |
| v11 | 3 / 3 | 0.030 | 2.0 | valid |

- Estimator check (v11): multi-episode capture 0.982 / 0.793 / 0.837 (21.7k-23.6k episodes) vs classic first-episode eval 0.982 / 0.794 / 0.827 (~1.37k episodes). They agree within sampling error.
- Prediction (plan): "v9/v10/v11 within 0.05, v6 fails". **Held.**
- Side observation: `v10/evader:s7` is neutralised only 1.7 % of the time by the v10 final pursuer (capture 0.4 %).

<a id="l0"></a>

## L0 literature / gate G2 (done 2026-10-26)

Notes: [L0_literature/notes.md](L0_literature/notes.md). Novelty verdicts (research agent, arXiv-verified):

| method | novel in general? | novel for agile quadrotor PE? | decision |
|---|---|---|---|
| QD-DPP opponent sampling | likely, as a sampling rule (DPP diversity is known as a best-response regulariser) | yes | keep (main) |
| NF-OA opponent adaptation | mechanism known (Neural-Fly, RMA), not for opponent modelling | yes | keep (main) |
| BayesID | no (BPR / Bayes-ToMoP / GSCU) | likely | dropped to P2 |

<a id="x1"></a>

## X1_crossplay (done 2026-10-26)

14 pursuers (last 4 stages of v9 / v10 / v11 + scripted pure pursuit and APF) x 32 evaders (22 RL + 10 scripted), multi-episode eval, 256-3,482 episodes per cell (median 1,745). Primary metric: neutralisation J. Details: [X1_crossplay/summary.md](X1_crossplay/summary.md), heatmaps `fig_payoff_neut.png`, `fig_payoff_capture.png`.

| subset | additive-logit R² (one strength per player) | rank-reversal fraction (delta 0.1) |
|---|---|---|
| all 14 pursuers | 0.777 | 0.045 |
| RL pursuers only | 0.835 | 0.024 |
| all, capture rate instead of J | 0.831 | 0.014 |

**At 0.15 m** ([R15_rescore/X1_crossplay/summary.md](R15_rescore/X1_crossplay/summary.md); 0.15 m convention):

| subset | additive-logit R² (one strength per player) | rank-reversal fraction (delta 0.1) |
|---|---|---|
| all 14 pursuers | 0.822 | 0.028 |
| RL pursuers only | 0.848 | 0.010 |
| all, capture rate instead of J | 0.883 | 0.005 |

What changes at 0.15 m:
- The game is more transitive; the prediction "Elo R² 0.8-0.9" now holds for all pursuers too (0.822).
- The gain is in the v9 and v11 pursuers (mean J +0.068 to +0.229) and APF (+0.141). The v10 pursuers and pure pursuit change by at most 0.036, so their failures are not near-misses.
- Campaign final pursuers: v9/s20 mean J 0.981, worst evader 0.921 (v9/evader:s11); v11/s12 mean 0.963, worst 0.684 (v9/evader:s17), 0.728 vs v9/evader:s19; v10 pursuers mean J 0.285-0.378, capture <= 0.173. "Some final pursuer < 0.6 vs another campaign's evader" still holds, but only through the v10 pursuers (v10/s12: 0.000 vs v9/evader:s13).
- Hardest evaders are the same three, in the same order: v9/evader:s19 (mean J 0.368), s17 (0.396), s15 (0.439).

Findings (at 0.10 m):

- The game is **mostly transitive**: one strength axis explains 78-84 % of the logit-J variance, and only 1-4 % of (pursuer pair, evader pair) comparisons reverse. Prediction "Elo R² 0.8-0.9": held for RL pursuers (0.835), slightly below for all pursuers (0.777).
- Campaign final pursuers differ strongly. v9/s20: mean J 0.909, worst evader 0.711. v11/s12: mean 0.868, worst 0.216 (vs v9/evader:s19). v10 pursuers: mean J 0.27-0.35, capture <= 0.14 (the no-warm-start campaign did not produce a working pursuer). Prediction "some final pursuer < 0.6 vs another campaign's evader": **held** (v11/s12 0.216 vs v9/s19; all v10 pursuers).
- Hardest evaders across all pursuers: v9/evader:s19 (mean J 0.151), s17 (0.220), s15 (0.285).

<a id="x2"></a>

## X2_descriptors (done 2026-10-26)

Descriptor per evader = response block (J and censored time-to-neutralise vs the 14 X1 pursuers) + trajectory block (mean/std of 9 per-step behaviour features), equal block weights. Details: [X2_descriptors/summary.md](X2_descriptors/summary.md), `fig_mds.png`.

| group | n | Vendi score (effective number of distinct evaders) |
|---|---|---|
| all | 32 | 4.54 |
| RL | 22 | 3.79 |
| scripted | 10 | 1.84 |
| v9 / v10 / v11 | 10 / 6 / 6 | 3.32 / 2.30 / 2.59 |

- Mean descriptor distance: consecutive stages 1.15, any two evaders of the same campaign 1.79, across campaigns 1.83 (consecutive stages are 36 % closer than an average same-campaign pair).
- k-medoids (k = 2..8, silhouette): best k = 7 (silhouette 0.461). Stage-1 evaders of all three campaigns fall together (all seeded from the same scripted opponents); v10 s7/s9/s11 form their own cluster with no training evader in it; v10 s1/s3/s5 share clusters with training evaders.
- Prediction H1a "4-7 clusters, consecutive stages near-duplicates, each campaign covers 2-3 clusters": clusters **held** (7); redundancy **held** (22 RL evaders = 3.8 effective); "2-3 clusters per campaign" **did not hold** (v9 spans 4 clusters, v10 3, v11 3).
- Pre-registered split written to `X2_descriptors/split.json`: train = v9 + v11 RL evaders + {hover, circular, flee, weave} (20); held-out = v10 RL (6) + {zigzag, fast_flee, 4 switch evaders @0.5} (6) + X3 exploiters.
- **Note (added 2026-11-04, after F2c; affects how held-out RL numbers are read):** the v10 RL evaders are **weaker than the train pool's late v9 / v11 evaders for pursuers that never trained on the pool**. v10 is the campaign without warm start whose pursuers never worked (X1: J 0.27-0.35), so its evaders never had to learn the lure-into-crash tactic of late v9 / v11 evaders. Mean J by evader group (late = stage >= 7):

  | pursuers | v9 late | v11 late | v10 late | v10 early |
  |---|---|---|---|---|
  | F2b plain / acc final (self-play, never saw the pool) | 0.28 / 0.12 | 0.34 / 0.15 | 0.65 / 0.83 | 0.91 / 0.96 |
  | X1: last 4 stages of v9 and v11 (trained on v9 / v11 evaders) | 0.51 | 0.60 | 0.82 | 0.91 |
  | X1: scripted pure pursuit / APF | 0.21 | 0.22 | 0.24 | 0.28 |
  | D1 pool-trained (QD / S-M / S-W) | 0.41 / 0.19 / 0.31 | 0.82 / 0.47 / 0.64 | 0.61 / 0.41 / 0.45 | 0.74 / 0.57 / 0.64 |
  | F1 pool-trained with acceleration input (QD_acc / SU_acc) | 0.84 / 0.85 | 0.97 / 0.98 | 0.63 / 0.93 | 0.83 / 0.94 |

  So "held-out RL > train pool" for a self-play pursuer (F2c) reflects an easier held-out set, not better generalisation; for pool-trained pursuers (D1, F1) v10 late is the harder set, as expected for unseen opponents, and scripted pursuers find all groups alike. Not universal: the F4 self-play pursuers score lower on held-out RL than on the train pool. The late v9 / v11 evaders beat the F2b pursuers by making them crash (pursuer crash 0.68-0.89), not by crashing themselves (evader crash + OOB 0.02-0.14 in every group).

<a id="x3"></a>

## X3_gap (done 2026-10-26)

Generalist pi_G = pursuer trained from scratch on the train split, uniform opponents, 150M frames, seeds 0-2. Exploiters = evaders trained from scratch (100M frames) against frozen pi_G seed 0. Specialist = pi_G seed 0 fine-tuned 20M frames on one test evader, evaluated only against that evader. Full table: [X3_gap/summary.md](X3_gap/summary.md).

- Generalist mean J on the train pool: **0.748 / 0.640 / 0.562** (seeds 0 / 1 / 2). Seed variance is large.
- Gap Delta = J(specialist) - J(pi_G seed 0):

| group | evaders | Delta |
|---|---|---|
| held-out scripted (zigzag, fast_flee, 4 switch evaders @0.5) | 6 | -0.001 to 0.001 (pi_G already ~1.0) |
| held-out RL v10 | 6 | 0.013, 0.176, 0.048, **0.341**, **0.365**, **0.313** (s1, s3, s5, s7, s9, s11) |
| exploiters | 3 | **0.759, 0.777, 0.572** (pi_G J 0.239 / 0.220 / 0.427 -> specialist 0.998-0.999) |

**At 0.15 m** ([R15_rescore/X3_gap/summary.md](R15_rescore/X3_gap/summary.md); 0.15 m convention): generalist mean J on the train pool **0.817 / 0.760 / 0.685** (seeds 0 / 1 / 2).

| group | evaders | Delta |
|---|---|---|
| held-out scripted (zigzag, fast_flee, 4 switch evaders @0.5) | 6 | -0.000 to 0.001 (pi_G already ~1.0) |
| held-out RL v10 | 6 | 0.001, 0.015, -0.182, 0.044, 0.106, 0.061 (s1, s3, s5, s7, s9, s11) |
| exploiters | 3 | **0.472, 0.363**, 0.025 (pi_G J 0.528 / 0.637 / 0.975 -> specialist 1.000) |

What changes at 0.15 m:
- The gap shrinks to **2 evaders with Delta >= 0.15** (exploiters g0e0, g0e1; 0.10 m: 7). No held-out RL evader keeps a gap of 0.15 (largest 0.106, v10 s9), because pi_G seed 0 gains more from the wider radius than the specialists do; on v10 s5 the specialist is now worse than pi_G (0.379 vs 0.562). H2a is still supported, but only by the exploiters.
- Predictions: "exploiters Delta >= 0.25 for >= 2 of 3" still **holds**; "held-out RL Delta 0.05-0.2" still does not hold, now because 4 of 6 are below 0.05; "heuristics Delta < 0.05" holds.
- Exploiters are much weaker: against pi_G seed 0 they reach timeout in 2.5-47.2 % of episodes (0.10 m: 57.3-77.6 %), 0 % pursuer crash; g0e2 is no longer an exploiter; against generalist seeds 1-2 they concede J 0.43-0.90.

Findings (at 0.10 m):

- **7 evaders with Delta >= 0.15** (v10 s3/s7/s9/s11, all 3 exploiters). H2a (a real adaptation gap exists) **supported**.
- Exploiters win by evading, not by forcing crashes: vs pi_G seed 0 they reach timeout in 57-78 % of episodes, with 0 % pursuer crash. They also beat generalist seeds 1-2 (J 0.17-0.49) and mostly seed 3 (0.48-0.82), but not seed 4 (0.86-1.00), so they are only partly seed-specific.
- Predictions: "exploiters Delta >= 0.25 for >= 2 of 3" **held** (3/3); "held-out RL Delta 0.05-0.2" **did not hold** (3 of 6 are 0.31-0.37); "heuristics Delta < 0.05" **held**.
- Caveat: specialists were not evaluated on the train pool, so this measures the reachable gain on one evader, not a policy that is good on everything.

<a id="x4"></a>

## X4_oracles (done 2026-10-27 04:43) and gate G1

Pursuers trained like the X3 generalist (train split, uniform, 150M frames, 3 seeds) plus one extra input. Mean J (per-seed in brackets); gap closed = mean over the 7 evaders with Delta >= 0.15 of (J_method - J_M0 seed-mean) / Delta. Full: [X4_oracles/summary.md](X4_oracles/summary.md).

| input | train | held-out RL | held-out scripted | exploiters* | gap closed (n evaders) |
|---|---|---|---|---|---|
| none (M0 = X3 generalist) | 0.650 [0.75, 0.64, 0.56] | 0.473 [0.60, 0.45, 0.37] | 0.998 | 0.309 [0.30, 0.31, 0.32] | 0.00 (7) |
| O1 opponent descriptor (8-D PCA) | 0.541 [0.54, 0.54, 0.54] | 0.430 [0.41, 0.42, 0.47] | 0.995 | n/a | -0.34 (4) |
| **O2a opponent acceleration** (finite difference of true velocity) | **0.939** [0.96, 0.95, 0.91] | **0.773** [0.46, 0.93, 0.93] | 0.992 | **0.973** [0.93, 1.00, 0.99] | **1.08** (7) |
| O2b opponent's latest CTBR command | 0.740 [0.59, 0.81, 0.83] | 0.337 [0.29, 0.35, 0.37] | 0.998 | 0.322 [0.30, 0.25, 0.42] | -0.22 (7) |

\* exploiters were trained against M0 seed 0. Against them, M0 seeds 1-2 reach 0.17-0.49, O2a 0.87-1.00.

**At 0.15 m** ([R15_rescore/X4_oracles/summary.md](R15_rescore/X4_oracles/summary.md); 0.15 m convention):

| input | train | held-out RL | held-out scripted | exploiters* | gap closed (n evaders)\*\* |
|---|---|---|---|---|---|
| none (M0 = X3 generalist) | 0.754 [0.82, 0.76, 0.69] | 0.686 [0.84, 0.65, 0.57] | 0.999 | 0.704 [0.71, 0.68, 0.72] | 0.00 (2) |
| O1 opponent descriptor (8-D PCA) | 0.542 [0.54, 0.54, 0.54] | 0.431 [0.41, 0.42, 0.47] | 1.000 | n/a | n/a (0) |
| **O2a opponent acceleration** (finite difference of true velocity) | **0.987** [0.99, 0.99, 0.98] | **0.848** [0.57, 0.99, 0.99] | 1.000 | **0.997** [0.99, 1.00, 1.00] | **0.94** (2) |
| O2b opponent's latest CTBR command | 0.838 [0.72, 0.90, 0.89] | 0.396 [0.36, 0.40, 0.42] | 1.000 | 0.546 [0.45, 0.45, 0.74] | -0.48 (2) |

\* Against the exploiters, M0 seeds 1-2 reach 0.43-0.90, O2a 0.98-1.00. \*\* Gap closed uses Delta re-computed at 0.15 m (X3 above): only exploiters g0e0 and g0e1 keep Delta >= 0.15. O1 was not evaluated against the exploiters, so it has no gap evaders.

What changes at 0.15 m:
- O2a is still the best input everywhere, but its gain over M0 roughly halves because M0 gains more from the wider radius: train +0.233 (0.10 m: +0.289), held-out RL +0.162 (+0.300), exploiters +0.293 (+0.664). O2a seed 0 is still weak on held-out RL.
- O1 barely moves, so its deficit to M0 grows (-0.213 in distribution; 0.10 m: -0.109). It still collapses on v10 s7/s9/s11 (J 0.001-0.021 on all 3 seeds, vs 0.415-0.937 for M0): not near-misses.
- O2b still helps in distribution (+0.084), hurts more on held-out RL (-0.291; 0.10 m: -0.136), and is now also worse than M0 on the exploiters (-0.158; 0.10 m: +0.013).
- **Gate G1 still passes** (O2a closes 94 % of the gap; bar 30 %), but the gap set shrinks from 7 evaders to 2, both exploiters trained against M0 seed 0 at 0.10 m.

Findings (at 0.10 m):

- **O2a is not privileged**: it is computable from two consecutive full-state observations, but the memoryless MLP pursuer cannot form it. Its large gain (train +0.29, held-out RL +0.30, exploiters +0.66) says most of the X3 "adaptation gap" is **missing opponent state (its current acceleration)**, which a policy needs to intercept a manoeuvring target.
- **O1 hurts and fails catastrophically out of distribution**: J ~= 0.00 against v10 s7/s9/s11 on all 3 seeds (their descriptors are outside the training range), versus 0.20-0.64 for M0. Conditioning on opponent identity overfits.
- **O2b** helps in-distribution (+0.09) but not on held-out evaders (-0.14).
- Caveat: the oracle inputs are noise-free; with sensor noise (velocity std 0.05-0.08 m/s at 50 Hz) a finite difference is very noisy, so F1 must evaluate under observation noise with a filtered estimate.
- Predictions: "O2 helps most (+0.05-0.15)": O2a helped far more (+0.30 to +0.66); O2b did not help on held-out, so the direction held for O2a and the magnitude did not. "O1 helps little in-distribution (< 0.03)": **did not hold** (O1 is -0.11 in-distribution and collapses out of distribution).
- **Gate G1: passed** (an oracle closes >= 30 % of the gap: O2a 108 %). Phase A runs as planned. A1b must beat the O2a / ESO input (reused as baseline "ESO_O2a"), not only M0. M0h (frame stack) and NF-frozen separate "more history" from "online adaptation".

<a id="d1"></a>

## D1_sampler (FINAL: 5 seeds, done 2026-10-27 20:49)

Pursuer trained on the train split (20 evaders), 150M frames, opponent probabilities refreshed every 25M frames (temperature 0.9, uniform floor 0.5). S-U = X3 generalists (seeds 0-4). Mean J, 95 % bootstrap CI over 5 seeds ([D1_sampler/summary.md](D1_sampler/summary.md)):

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

**At 0.15 m** ([R15_rescore/D1_sampler/summary.md](R15_rescore/D1_sampler/summary.md); 0.15 m convention):

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
- QD-DPP still beats S-M and S-W (all eight paired CIs exclude 0), and the held-out gaps grow.
- QD-DPP vs uniform is still not supported (all four CIs include 0; the worst cluster only just). Seed 3 is still reversed (held-out RL -0.066).
- Hardness-only prioritisation still hurts relative to uniform.
- J against the X3 exploiters rises most: in about 0.3 of those episodes the closest distance was between 0.10 and 0.15 m. QD-DPP now has the highest exploiter J, but the gap to S-U is small.

Sampling distributions (all refreshes, all seeds): effective number of opponents exp(entropy) out of 20: S-W 19.6, S-M 18.8, QD-DPP 19.2; mean max probability 0.062 / 0.081 / 0.085.

- **QD-DPP beats both hardness-prioritised samplers** (S-M, S-W) on train, held-out RL, worst cluster and worst evader; the paired CIs exclude 0 (supported under the plan's rule).
- **QD-DPP vs uniform: not supported** (mean differences +0.01 to +0.04, CIs include 0; seed 3 is reversed). The interim 3-seed reading "QD ahead of every baseline on held-out RL on all seeds" did not survive for uniform.
- Hardness-only prioritisation **hurts** relative to uniform; adding the diversity term repairs this. The advantage is not simply flatter sampling: S-W is flatter than QD-DPP yet worse.
- Caveat: all distributions are close to uniform (uniform floor 0.5, temperature 0.9) while the J differences are large; with seed variance this large, the effect sizes are uncertain even where the CI excludes 0.
- Prediction H1c "QD >= S-M on held-out worst cluster by 0.03-0.08, equal in-distribution": QD's advantage over S-M is larger than predicted (+0.176) and also present in-distribution (+0.180), so the "equal" part **did not hold**.

<a id="d1b"></a>

## D1b_duplicates (done 2026-10-27 06:32; 3 seeds)

Train pool + 5 extra copies of the hardest train RL evader by X3 generalist J (**v9/evader:s15**), i.e. 6 copies among 25 opponents. Samplers S-M and QD-DPP, same settings as D1 (uniform floor 0.5). [D1b_duplicates/summary.md](D1b_duplicates/summary.md).

Sampling mass on the duplicated cluster (5 refreshes x 3 seeds): **S-M 0.255-0.314, QD-DPP 0.199-0.217** (uniform share would be 0.24; the 0.5 uniform floor alone gives the copies 0.12). The DPP bound works as designed.

Performance change caused by duplication (seed-paired, D1b minus D1, mean over 3 seeds; train (D1's train mean also without v9/evader:s15; corrected 2026-11-10, was mixed groups)):

| sampler | train (non-duplicated evaders) | held-out RL | held-out worst cluster | exploiters |
|---|---|---|---|---|
| S-M | -0.187 [-0.340, -0.120, -0.101] | -0.096 [-0.371, +0.051, +0.032] | -0.134 | +0.097 |
| QD-DPP | -0.166 [-0.139, -0.182, -0.177] | **-0.230** [-0.326, -0.226, -0.138] | **-0.237** | -0.198 |

Absolute J under duplication: S-M train 0.411, held-out RL 0.374, worst cluster 0.241; QD-DPP 0.580, 0.370, 0.218.

**At 0.15 m** ([R15_rescore/D1b_duplicates/summary.md](R15_rescore/D1b_duplicates/summary.md); 0.15 m convention). Performance change caused by duplication (seed-paired, D1b minus D1, mean over 3 seeds [per-seed values]):

| sampler | train (non-duplicated evaders) | held-out RL | held-out worst cluster | exploiters |
|---|---|---|---|---|
| S-M | -0.208 [-0.295, -0.178, -0.150] | -0.144 [-0.436, +0.006, -0.001] | -0.173 | -0.016 |
| QD-DPP | -0.165 [-0.123, -0.225, -0.148] | **-0.257** [-0.261, -0.356, -0.153] | **-0.243** | -0.433 |

Absolute J under duplication at 0.15 m: S-M train 0.482, held-out RL 0.481, worst cluster 0.333; QD-DPP 0.679, 0.540, 0.422.

What changes at 0.15 m:
- H1b is still not supported: QD-DPP still loses more held-out J under duplication than S-M.
- S-M's held-out drop grows and now exceeds the predicted 0.05-0.10; it still comes from seed 0 alone.
- In absolute terms QD-DPP is now ahead of S-M under duplication on held-out RL and the worst cluster (at 0.10 m the two were level).
- On the exploiters QD-DPP's drop grows (-0.198 -> -0.433) and S-M's change goes from +0.097 to -0.016.

Findings (at 0.10 m):

- **H1b not supported in performance terms.** Lower sampling mass on the duplicates did not make QD-DPP more robust: it lost more than S-M on held-out evaders (-0.23 vs -0.10). Prediction "S-M drops 0.05-0.10, QD-DPP < 0.02": S-M's drop held (-0.10 held-out, -0.19 train); QD-DPP's **did not hold**.
- Limits of this test: (1) the 0.5 uniform floor is not duplicate-robust and controls half the probability mass; (2) 3 seeds with large seed variance (S-M seed 0 in D1 was an outlier, 0.769 train J), which inflates S-M's apparent drop on seed 0 and nothing else.
- Possible follow-up (not run): D1b with `uniform_mix = 0`.

<a id="d1c"></a>

## D1c_nofloor (done 2026-10-28 19:25; EXPLORATORY, designed after seeing D1b; 3 seeds; no pre-registered prediction)

D1b repeated without the uniform floor (`uniform_mix = 0`): S-M vs QD-DPP, normal pool ("nodup") and the pool with 5 extra copies of v9/evader:s15 ("dup"). Otherwise as D1 (150M frames, refresh every 25M, temperature 0.9). [D1c_nofloor/nodup/summary.md](D1c_nofloor/nodup/summary.md), [dup/summary.md](D1c_nofloor/dup/summary.md).

| pool | sampler | train | held-out RL | held-out scripted | exploiters | worst held-out cluster |
|---|---|---|---|---|---|---|
| nodup | S-M | 0.283 [0.269, 0.307] | 0.363 [0.318, 0.395] | 0.612 [0.539, 0.649] | 0.450 [0.339, 0.554] | 0.230 [0.203, 0.254] |
| nodup | QD-DPP | 0.420 [0.402, 0.455] | 0.390 [0.366, 0.406] | 0.910 [0.875, 0.928] | 0.260 [0.202, 0.326] | 0.273 [0.243, 0.292] |
| dup | S-M | 0.262 [0.211, 0.291] | 0.310 [0.238, 0.360] | 0.473 [0.265, 0.580] | 0.574 [0.521, 0.647] | 0.131 [0.026, 0.208] |
| dup | QD-DPP | 0.387 [0.372, 0.400] | 0.386 [0.374, 0.398] | 0.940 [0.926, 0.950] | 0.336 [0.264, 0.477] | 0.252 [0.237, 0.264] |

Seed-paired differences (mean [95 % CI], 3 seeds):

| comparison | train | held-out RL | worst held-out cluster | exploiters |
|---|---|---|---|---|
| no floor - floor (D1, same seeds), S-M | **-0.315 [-0.527, -0.171]** | -0.107 [-0.330, +0.019] | -0.144 [-0.422, +0.005] | **+0.246 [+0.153, +0.403]** |
| no floor - floor (D1, same seeds), QD-DPP | **-0.326 [-0.404, -0.264]** | **-0.210 [-0.409, -0.067]** | -0.182 [-0.396, +0.005] | -0.139 [-0.401, +0.067] |
| dup - nodup, S-M | -0.022 [-0.057, +0.016] | **-0.053 [-0.080, -0.017]** | **-0.100 [-0.177, -0.026]** | +0.124 [+0.002, +0.309] |
| dup - nodup, QD-DPP | **-0.033 [-0.066, -0.002]** | -0.004 [-0.032, +0.020] | -0.021 [-0.055, +0.013] | +0.076 [-0.062, +0.275] |
| QD - S-M, nodup | **+0.137 [+0.129, +0.148]** | +0.027 [-0.011, +0.080] | **+0.042 [+0.009, +0.080]** | **-0.190 [-0.257, -0.085]** |
| QD - S-M, dup | **+0.126 [+0.081, +0.188]** | **+0.076 [+0.026, +0.161]** | **+0.122 [+0.048, +0.238]** | **-0.238 [-0.380, -0.044]** |

**At 0.15 m** ([R15_rescore/D1c_nofloor/nodup/summary.md](R15_rescore/D1c_nofloor/nodup/summary.md), [dup/summary.md](R15_rescore/D1c_nofloor/dup/summary.md); 0.15 m convention):

| pool | sampler | train | held-out RL | held-out scripted | exploiters | worst held-out cluster |
|---|---|---|---|---|---|---|
| nodup | S-M | 0.332 [0.327, 0.341] | 0.449 [0.399, 0.477] | 0.755 [0.712, 0.808] | 0.466 [0.357, 0.564] | 0.307 [0.242, 0.341] |
| nodup | QD-DPP | 0.466 [0.436, 0.500] | 0.465 [0.457, 0.478] | 0.921 [0.884, 0.950] | 0.441 [0.326, 0.615] | 0.310 [0.274, 0.330] |
| dup | S-M | 0.300 [0.225, 0.344] | 0.386 [0.267, 0.464] | 0.583 [0.307, 0.729] | 0.593 [0.546, 0.661] | 0.241 [0.073, 0.334] |
| dup | QD-DPP | 0.442 [0.412, 0.464] | 0.475 [0.468, 0.484] | 0.952 [0.943, 0.961] | 0.507 [0.436, 0.609] | 0.295 [0.256, 0.356] |

Seed-paired differences at 0.15 m (mean [95 % CI], 3 seeds; floor = D1 at 0.15 m, same seeds):

| comparison | train | held-out RL | worst held-out cluster | exploiters |
|---|---|---|---|---|
| no floor - floor (D1, same seeds), S-M | **-0.358 [-0.546, -0.225]** | **-0.176 [-0.438, -0.023]** | -0.200 [-0.505, +0.006] | -0.002 [-0.282, +0.170] |
| no floor - floor (D1, same seeds), QD-DPP | **-0.379 [-0.424, -0.349]** | **-0.332 [-0.477, -0.234]** | **-0.356 [-0.559, -0.241]** | **-0.368 [-0.629, -0.075]** |
| dup - nodup, S-M | -0.031 [-0.102, +0.018] | **-0.063 [-0.132, -0.014]** | **-0.066 [-0.169, -0.003]** | **+0.127 [+0.008, +0.305]** |
| dup - nodup, QD-DPP | -0.025 [-0.052, +0.029] | +0.010 [-0.006, +0.024] | -0.015 [-0.068, +0.026] | +0.067 [-0.179, +0.283] |
| QD - S-M, nodup | **+0.135 [+0.108, +0.159]** | +0.016 [-0.020, +0.079] | +0.003 [-0.063, +0.088] | -0.025 [-0.151, +0.052] |
| QD - S-M, dup | **+0.141 [+0.068, +0.240]** | **+0.088 [+0.004, +0.204]** | +0.054 [-0.062, +0.283] | -0.086 [-0.184, +0.063] |

What changes at 0.15 m:
- Removing the floor is still harmful, and more clearly so (for QD-DPP all four CIs now exclude 0). S-M's apparent exploiter gain disappears (n.s.); S-M still fails held-out scripted evaders (0.755, against 0.996 with the floor).
- The duplicate-robustness reading still holds but is weaker: duplication lowers S-M's held-out RL and worst-cluster J (CIs exclude 0, the latter only just) and leaves QD-DPP unchanged.
- QD-DPP's advantage over S-M shrinks to train J and held-out RL with duplicates; the worst-cluster advantage is no longer significant.
- QD-DPP's deficit on the X3 exploiters is no longer significant: it was mostly near misses between 0.10 and 0.15 m.

Sampling mass on the duplicated evader (6 copies of 25; uniform share 0.24): S-M 0.29-0.36, QD-DPP 0.16-0.20. Mass on the 4 scripted training evaders (uniform share 0.16 in the 25-opponent pool): 0.03-0.11 per refresh for S-M, 0.07-0.13 for QD-DPP.

- **Removing the uniform floor is harmful for both samplers**: train J drops by about 0.32 for both (CIs exclude 0; D1's train mean also without v9/evader:s15, corrected 2026-11-10). The pursuer is starved of the scripted training evaders (3-13 % of the mass) and even fails held-out scripted evaders (S-M 0.61, against 0.985 with the floor); its J against the training RL evaders also drops (0.15-0.31 vs 0.32-0.72 with the floor). The two D1c runs from 10-27, before any later code change, show the same collapse, so this is not a code-change artefact.
- **Without the floor, the DPP's duplicate robustness shows in performance** (the mechanism H1b predicted): duplication lowers S-M's held-out RL and worst-cluster J (CIs exclude 0) and leaves QD-DPP's unchanged (n.s.). QD-DPP keeps the duplicated evader's mass below its uniform share (0.16-0.20 vs 0.24); S-M inflates it (0.29-0.36).
- QD-DPP is better than S-M on train, held-out scripted and the worst cluster, and **worse on the X3 exploiters** (-0.19 / -0.24).
- Reading: H1b holds as a mechanism only when the hardness term controls most of the sampling mass, which is also the regime where hardness prioritisation is worst. With a 0.5 floor (D1b) the mechanism does not show. Exploratory: designed after D1b, 3 seeds.

<a id="d2"></a>

## D2_exploitability (FINAL: 5 seeds, done 2026-10-28 01:05)

One fresh exploiter evader (PPO from scratch, 100M frames) per D1 pursuer; exploitability = 1 - J(pursuer, its exploiter). [D2_exploitability/summary.md](D2_exploitability/summary.md).

| sampler | exploitability per seed (0-4) | mean [95 % CI] | capture vs exploiter per seed |
|---|---|---|---|
| S-U uniform | 0.697, 0.803, 0.513, 0.644, 0.705 | 0.672 [0.586, 0.745] | 0.25, 0.15, 0.45, 0.24, 0.26 |
| S-W win rate | 0.609, 0.551, 0.460, 0.574, **0.008** | 0.440 [0.219, 0.583] | 0.10, 0.13, 0.12, 0.37, 0.77 |
| S-M multi-metric | 0.830, 0.709, 0.495, 0.709, 0.713 | 0.691 [0.582, 0.782] | 0.15, 0.21, 0.45, 0.25, 0.25 |
| QD-DPP | 0.520, 0.794, 0.791, 0.559, 0.484 | 0.630 [0.513, 0.745] | 0.14, 0.15, 0.15, 0.12, 0.48 |

Seed-paired QD-DPP minus baseline: vs S-U -0.043 [-0.176, +0.129]; vs S-M -0.062 [-0.246, +0.132]; vs S-W +0.189 [+0.007, +0.371].

**At 0.15 m** ([R15_rescore/D2_exploitability/summary.md](R15_rescore/D2_exploitability/summary.md); 0.15 m convention):

| sampler | exploitability per seed (0-4) | mean [95 % CI] | capture vs exploiter per seed |
|---|---|---|---|
| S-U uniform | 0.004, 0.196, 0.015, 0.105, 0.130 | 0.090 [0.029, 0.151] | 0.98, 0.78, 0.95, 0.79, 0.84 |
| S-W win rate | 0.284, 0.253, 0.157, 0.103, 0.000 | 0.159 [0.071, 0.246] | 0.46, 0.45, 0.47, 0.88, 0.81 |
| S-M multi-metric | 0.176, 0.256, 0.062, 0.317, 0.284 | 0.219 [0.130, 0.291] | 0.81, 0.67, 0.89, 0.65, 0.70 |
| QD-DPP | 0.231, 0.015, 0.066, 0.357, 0.037 | 0.141 [0.035, 0.256] | 0.49, 0.95, 0.89, 0.33, 0.94 |

Seed-paired QD-DPP minus baseline at 0.15 m: vs S-U +0.051 [-0.094, +0.196]; vs S-M -0.078 [-0.193, +0.038]; vs S-W -0.018 [-0.146, +0.124].

What changes at 0.15 m:
- Exploitability collapses from 0.44-0.69 to 0.09-0.22: most exploiter wins at 0.10 m were timeouts in which the evader came within 0.15 m. Exploiters trained for 0.15 m would keep more distance, so this does not show that the pursuers are hard to exploit at 0.15 m.
- Still no sampler reduces exploitability reliably (all three QD-DPP paired CIs include 0); the QD-DPP vs S-W difference disappears.
- The ranking changes (S-U lowest, S-M highest, QD-DPP second); with 5 seeds and wide CIs this ordering is not reliable.

Findings (at 0.10 m):

- **No sampler reduces exploitability reliably.** QD-DPP vs uniform and vs S-M: differences not significant.
- S-W's lower mean is driven by seed 4, where the exploiter **failed to learn** (exploitability 0.008, pursuer captures 77 %). Without that seed S-W's mean is 0.548. The QD-DPP vs S-W difference (CI barely excluding 0) therefore says more about one failed exploiter than about the samplers.
- Every pursuer remains highly exploitable (typically 0.5-0.8); exploiters win by evading until timeout.
- Limitation: one exploiter per pursuer. Exploiter training itself is noisy (one of 20 failed), so the measure has extra variance beyond the pursuer seeds.
- Prediction "QD-DPP lowest exploitability, maybe within CI": QD-DPP has the second-lowest mean; differences are within the CIs except vs S-W (opposite direction, driven by a failed exploiter). **Did not hold.**

<a id="a1a"></a>

## A1a_basis (done 2026-10-27 05:22; offline, open-loop)

Trajectories of the X3 generalist (seed 0) vs the train pool and vs the test set (1,024 evenly spaced envs, ~3,000 steps each). Target: evader relative displacement over H steps. The learned basis (BasisNet, 32 features, per-evader heads + shared head) is trained on train-pool trajectories (80 % of envs); RLS (delta, forgetting) is tuned per feature kind and H on the other 20 %; the test set is not used for tuning. Full: [A1a_basis/summary.md](A1a_basis/summary.md); tuned values: `A1a_basis/tuned_rls.json`.

**Run 1 (untuned, pre-registered defaults delta 10, forgetting 0.98) failed** by covariance wind-up (NF+RLS worse than the frozen basis, H 15 test 0.198 vs 0.038 m; diverged across episodes, S2 2.24 m); kept as `summary_run1_untuned.md`. Fix: bounded-covariance RLS (trace <= initial trace) + tuning on train-pool validation envs (plan deviation, §9).

Run 2, mean |y - y_hat| (m) on the **test** set (held-out RL, scripted, exploiters):

| H (steps / s) | CV (const. velocity) | raw RLS | ESN RLS | NF frozen | NF RLS (S1) | NF RLS (S2) |
|---|---|---|---|---|---|---|
| 5 / 0.1 | 0.0098 | 0.0083 | 0.0093 | 0.0044 | 0.0039 | **0.0037** |
| 15 / 0.3 | 0.0740 | 0.0899 | 0.0883 | 0.0376 | 0.0373 | **0.0354** |
| 25 / 0.5 | 0.1799 | 0.2169 | 0.2177 | 0.0984 | 0.0956 | **0.0889** |

- The **learned basis** is the main effect: NF frozen has 45-55 % lower error than CV at every horizon. ESN and raw-history RLS are worse than CV for H >= 15.
- **Online RLS on top of the basis** adds a small, consistent improvement on held-out evaders: S1 -1 % to -11 %, S2 (readout kept across episodes against the same evader) -6 % to -16 % vs NF frozen.
- Pre-registered horizon rule -> **H\* = 5** (largest gain over CV: +0.607).
- Error grows with episode index for every method: a selection effect (envs reaching a high episode index have short episodes), not evidence for or against adaptation.
- Predictions: "NF+RLS < ESN+RLS by 30-50 % at 0.3 s on held-out" **held** (0.0373 vs 0.0883, -58 %, larger than predicted); "NF frozen in between" **did not hold** (NF frozen is nearly as good as NF+RLS); "CV good at 0.1 s" partially (CV beats ESN at 0.1 s, but NF is 2x better).

<a id="a1b"></a>

## A1b_policy (FINAL: 5 seeds, done 2026-10-28 09:58)

Pursuers on the train pool (uniform, 150M frames) with the adapter output appended to the observation: predicted evader displacement over H* = 5 steps (0.1 s), tuned RLS. S1: adapter reset every episode; S2: adapter kept across the episodes of an env (same evader). M0 = X3 generalists (5 seeds). ESO_O2a = X4 acceleration-input pursuers (3 seeds here; the 5-seed version is F1 "SU + acc"). Mean J, 95 % bootstrap CI over seeds ([A1b_policy/summary.md](A1b_policy/summary.md)):

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

**At 0.15 m** ([R15_rescore/A1b_policy/summary.md](R15_rescore/A1b_policy/summary.md); 0.15 m convention):

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

Seed-paired differences at 0.15 m (mean [95 % CI]; computed from the per-seed values in [R15_rescore/A1b_policy/results.json](R15_rescore/A1b_policy/results.json) with the same bootstrap):

| comparison | train | held-out RL | exploiters |
|---|---|---|---|
| NF-OA - M0 | +0.034 [-0.065, +0.140] | -0.004 [-0.210, +0.220] | +0.050 [-0.103, +0.182] |
| NF frozen - M0 | +0.019 [-0.069, +0.098] | -0.098 [-0.265, +0.069] | +0.003 [-0.205, +0.199] |
| NF-OA - NF frozen (online adaptation) | +0.016 [-0.026, +0.058] | +0.094 [-0.127, +0.316] | +0.047 [-0.092, +0.152] |
| NF-OA S2 - S1 (across episodes) | 0.000 | -0.004 [-0.011, +0.001] | -0.001 [-0.008, +0.005] |
| ESN - M0 | **-0.119 [-0.215, -0.024]** | **-0.254 [-0.406, -0.101]** | **-0.175 [-0.280, -0.078]** |
| frame stack - M0 | **-0.282 [-0.376, -0.166]** | **-0.367 [-0.499, -0.224]** | +0.023 [-0.076, +0.129] |

What changes at 0.15 m:
- The NF and online-adaptation findings are unchanged: neither NF variant beats M0 (all CIs include 0), NF-OA - NF frozen stays n.s. on every split, and S2 changes J by <= 0.004.
- ESN is now also significantly worse than M0 on the exploiters. The frame stack's exploiter edge over M0 shrinks to n.s. and its train / held-out RL deficit grows. H2c (NF-OA beats ESN) still holds: held-out RL 0.795 vs 0.545.
- ESO_O2a stays first in every column, but its lead on held-out RL is much smaller (0.848 vs 0.795 for NF-OA and 0.799 for M0); on train and exploiters it is still far ahead of every adapter.

Findings (at 0.10 m):

- With 5 seeds **neither NF variant is better than M0** (all CIs include 0); the 3-seed reading "the learned basis helps a little" did not survive (seeds 3-4 of M0 are strong).
- **Online adaptation (NF-OA vs NF frozen) is not significant**; across-episode adaptation (S2) changes J by <= 0.005.
- **ESN and the raw frame stack are significantly worse than M0** on train and held-out RL; the explicit acceleration input remains far better than every adapter (F1).
- J vs episode index rises for every method, M0 included (episode-length selection effect, not adaptation).
- Predictions: "NF-OA closes 20-40 % of Delta in S2": **did not hold**. "ESN about 0": **did not hold** (worse than M0). H2c (NF-OA beats ESN): **supported** (held-out RL 0.606 vs 0.362). H2d (gain grows with experience, vanishes when frozen): **not supported**.
- Design caveat: at H* = 0.1 s the predicted displacement is dominated by relative velocity x 0.1 s, which the policy already observes; A2 retargets the same machinery to the evader's acceleration.

<a id="a2a"></a>

## A2a_nfacc (done 2026-10-27 21:19; offline, open-loop)

Learned estimator of the evader's acceleration: same basis + RLS machinery as A1a, new target = change of the evader's own velocity over the next H steps (H dt x mean acceleration). Same trajectories as A1a; RLS tuned on the train-pool validation envs (20 % of envs not used to train the basis). [A2a_nfacc/summary.md](A2a_nfacc/summary.md). Mean error (m/s); "FD persistence" = last-step finite-difference acceleration held for H steps (the offline analogue of the O2a input); "zero" = constant velocity.

| H (steps / s) | set | zero | FD persistence | NF frozen | NF RLS (S1) | NF RLS (S2) |
|---|---|---|---|---|---|---|
| 1 / 0.02 | val | 0.0363 | **0.0045** | 0.0076 | 0.0054 | 0.0053 |
| 1 / 0.02 | test | 0.0371 | **0.0055** | 0.0088 | 0.0062 | 0.0062 |
| 5 / 0.1 | val | 0.1773 | 0.0522 | 0.0545 | 0.0501 | **0.0471** |
| 5 / 0.1 | test | 0.1803 | 0.0643 | 0.0679 | 0.0603 | **0.0561** |
| 10 / 0.2 | val | 0.3401 | 0.1599 | 0.1442 | 0.1377 | **0.1313** |
| 10 / 0.2 | test | 0.3396 | 0.1885 | 0.1744 | 0.1568 | **0.1488** |

- For one step ahead the finite difference is essentially exact and the learned estimator is worse (+13 % to +60 %). For 0.1-0.2 s ahead the learned estimator beats FD persistence, and online RLS adds to it: at H = 10 on the test set, frozen -7 %, RLS S1 -17 %, RLS S2 -21 %; so here **online adaptation helps clearly** over the frozen basis (-10 % S1, -15 % S2), more than for the displacement target in A1a.
- Pre-registered rule -> **H\* = 10** (val gain over FD: H=1 -0.198, H=5 +0.040, H=10 +0.139).
- Prediction "learned beats FD for H >= 5, small RLS gain": held at H = 10, marginal at H = 5 (S1 -6 %); the RLS gain was larger than "small" (10-15 %).

<a id="a2b"></a>

## A2b_nfacc (done 2026-10-28 21:10; 5 seeds; 3-seed version 11:04 superseded)

Pursuers on the train pool (uniform, 150M frames, noise-free) whose observation includes the **learned estimate of the evader's mean acceleration over the next 0.2 s** (A2a, H* = 10), scaled exactly like the finite-difference input O2a. Frozen basis vs basis + online RLS (tuned). [A2b_nfacc/summary.md](A2b_nfacc/summary.md). Mean J, 95 % bootstrap CI over 5 seeds:

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

Held-out RL per seed: O2a 0.46 0.93 0.93 0.96 0.91; NF-acc frozen 0.80 0.90 0.94 0.90 0.65; NF-acc + RLS 0.89 0.82 0.74 0.89 0.96. Held-out v10 evaders s1/s3/s5/s7/s9/s11 per run (S1): NF-acc frozen 0.99 0.94 0.57 0.66 0.80 0.85 | 1.00 0.97 0.79 0.84 0.88 0.92 | 0.99 0.96 0.90 0.88 0.97 0.94 | 0.99 0.94 0.80 0.85 0.87 0.93 | 0.99 0.98 0.34 0.60 0.52 0.48; NF-acc + RLS 0.99 0.95 0.74 0.84 0.91 0.91 | 0.99 0.95 0.71 0.66 0.84 0.80 | 1.00 0.97 0.52 0.62 0.63 0.69 | 0.98 0.94 0.78 0.80 0.91 0.89 | 0.99 0.94 0.95 0.93 0.97 0.98.

**At 0.15 m** ([R15_rescore/A2b_nfacc/summary.md](R15_rescore/A2b_nfacc/summary.md); 0.15 m convention):

| method | train | held-out RL | held-out scripted | exploiters* |
|---|---|---|---|---|
| M0 | 0.818 [0.738, 0.900] | 0.799 [0.655, 0.942] | 0.999 | 0.806 [0.702, 0.924] |
| finite-difference acceleration (O2a) | **0.988** [0.982, 0.991] | 0.904 [0.735, 0.992] | 1.000 | 0.998 [0.995, 1.000] |
| NF-OA displacement target (A1b) | 0.853 [0.824, 0.883] | 0.795 [0.670, 0.925] | 1.000 | 0.856 [0.764, 0.948] |
| **NF-acc frozen** | 0.967 [0.945, 0.989] | 0.933 [0.862, 0.984] | 1.000 | 0.997 [0.992, 0.999] |
| NF-acc + RLS (S1) | 0.962 [0.937, 0.983] | **0.958** [0.918, 0.987] | 1.000 | **0.998** [0.996, 0.999] |
| NF-acc + RLS (S2) | (same policy) | 0.958 [0.923, 0.986] | 1.000 | 0.998 [0.996, 0.999] |

Seed-paired differences at 0.15 m (mean [95 % CI], 5 seeds; computed from the per-seed values in [R15_rescore/A2b_nfacc/results.json](R15_rescore/A2b_nfacc/results.json) with the same bootstrap):

| comparison | train | held-out RL | exploiters |
|---|---|---|---|
| NF-acc + RLS - M0 | **+0.144 [+0.075, +0.231]** | **+0.159 [+0.044, +0.274]** | **+0.193 [+0.076, +0.295]** |
| NF-acc frozen - M0 | **+0.149 [+0.086, +0.219]** | +0.135 [-0.052, +0.330] | **+0.191 [+0.069, +0.298]** |
| NF-acc frozen - O2a | **-0.021 [-0.041, -0.000]**† | +0.029 [-0.115, +0.211] | -0.001 [-0.007, +0.003] |
| NF-acc + RLS - O2a | **-0.025 [-0.054, -0.000]**† | +0.053 [-0.065, +0.240] | +0.000 [-0.003, +0.004] |
| NF-acc + RLS - NF-acc frozen (online adaptation) | -0.005 [-0.040, +0.032] | +0.024 [-0.061, +0.125] | +0.001 [-0.003, +0.007] |
| S2 - S1 | 0.000 | +0.000 | -0.000 |

† upper bounds -0.0004 and -0.0002: the CI only just excludes 0.

Held-out RL per seed at 0.15 m: O2a 0.57 0.99 0.99 0.99 0.98; NF-acc frozen 0.93 0.98 1.00 0.97 0.79; NF-acc + RLS 0.97 0.95 0.88 0.98 1.00. Held-out v10 evaders s1/s3/s5/s7/s9/s11 per run (S1), re-scored with `rescore.py`: NF-acc frozen 1.00 1.00 0.78 0.87 0.94 0.98 | 1.00 1.00 0.95 0.95 0.98 1.00 | 1.00 1.00 0.99 0.99 1.00 1.00 | 1.00 1.00 0.92 0.95 0.95 0.99 | 1.00 1.00 0.56 0.80 0.72 0.69; NF-acc + RLS 1.00 1.00 0.90 0.96 0.97 0.99 | 1.00 1.00 0.93 0.89 0.95 0.95 | 1.00 1.00 0.77 0.77 0.82 0.94 | 1.00 1.00 0.95 0.97 0.99 1.00 | 1.00 1.00 0.99 0.99 1.00 1.00.

What changes at 0.15 m:
- The gain over M0 still holds but is smaller (with RLS all CIs exclude 0; at 0.10 m +0.181, +0.225, +0.490). Frozen: the held-out RL CI still includes 0.
- Equivalence to the finite-difference input mostly holds: held-out RL and exploiter differences stay n.s.; on train the learned estimate is now marginally lower (CIs only just exclude 0), so "every CI includes 0" no longer holds strictly. "Within 0.05 of O2a" holds except NF-acc + RLS on held-out RL (+0.053, in its favour).
- The failure-mode contrast is unchanged: the finite-difference runs that collapse on v10 s5-s11 (F1 SU seed 0, QD seeds 0 and 2) stay at 0.19-0.70; the worst learned-estimate run (frozen seed 4) reaches 0.56-0.80.
- Online adaptation still adds nothing (RLS - frozen n.s. on every split; S2 = S1).

Findings (at 0.10 m):

- Retargeting the learned basis from displacement (A1b) to the evader's acceleration turns it from "no gain" into a **large gain over M0** (with RLS all CIs exclude 0; frozen: same size, held-out RL CI includes 0 because of seed 4).
- The learned estimate is **equivalent to the finite-difference input** noise-free: every paired difference is within +/- 0.04 in the mean and every CI includes 0 (the 3-seed "frozen is lower on train, -0.06 significant" no longer holds with 5 seeds).
- Failure mode on the later v10 lineage: with the finite-difference input 3 of 10 runs (F1) collapse there (0.10-0.29); with the learned estimate none of 10 does, but one (frozen seed 4) is degraded (0.34-0.60). Fewer severe failures, but not significant.
- **Online adaptation again adds nothing** (S2 = S1), although it reduced the open-loop prediction error by 10-15 % in A2a.
- Prediction "close to O2a (within 0.05) noise-free; RLS gain small": **held** on all splits with 5 seeds; RLS gain small/absent **held**.

<a id="a3a"></a>

## A3a_posonly (done 2026-10-28 18:46; offline, open-loop)

Can a learned estimator recover the evader's acceleration **without observing its velocity**? Same X3-generalist trajectories (seed 0, 1,024 envs, re-recorded with the pursuer's own position). Input: 10-step window of [relative position, own position, own velocity]; target = change of the evader's velocity over the next H steps; frozen basis (online RLS would need the unobservable evader velocity as its target), trained on clean ("clean") or noise-augmented ("aug") inputs. Baselines: zero (constant velocity) and FD = second difference of the evader position, held for H steps, raw or EMA-filtered (alpha tuned on noisy val). Noise model = `observation.noise` stds: relative and own position 0.02 m, own velocity 0.05 m/s. [A3a_posonly/summary.md](A3a_posonly/summary.md). Mean error (m/s):

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

- **Noise-free, finite differences of position win clearly** (FD error 2.2x lower at H=5, 1.5x at H=10). Without the velocity input the learned estimator is much worse than A2a's velocity-based one (H=10 val 0.241 vs 0.144), while position-FD is about as good as velocity-FD (0.167 vs 0.160).
- **With sensor noise, position FD is useless** (2 cm noise differentiated twice at 50 Hz gives 28-55 m/s); the best EMA (alpha 0.05, the edge of the pre-registered grid) is still 4-5x worse than predicting zero.
- The **noise-augmented learned estimator is the only method better than zero under noise** (-26 % val, -13 to -14 % test); the clean-trained basis degrades under noise (+20 % val error), the augmented one costs only 2 % on clean inputs.
- Pre-registered choices: H\* = 10 (both "gains vs FD" are negative, -1.22 at H=5 and -0.44 at H=10; the rule takes the larger), basis = aug, FD EMA alpha = 0.05. A3b uses these.
- Prediction "learned beats FD clearly with noise, roughly equal without": the noisy part **held**; the noise-free part **did not hold** (FD is clearly better).
- Run 1 was invalid (the trajectory recorder stored positions as float16, ~2 mm steps at 2-3 m, inflating the noise-free FD error to 0.26-0.56 m/s); re-recorded in float32, run 1 kept in `A3a_posonly/run1_fp16/`. The learned-estimator errors and the pre-registered choices were the same in both runs. A1a / A2a used the same recorder with velocity-based inputs, where the effect is much smaller (e.g. velocity FD one step ahead, 0.0045 m/s in A2a, is partly quantisation); they were not re-run.

<a id="a3b"></a>

## A3b_posonly (done 2026-10-29 02:15; 3 seeds; closed loop)

Pursuers whose observation has **no evader velocity** (`observation.agent_rel_vel=false`; relative position only), train pool, uniform, 150M frames, noise-free training. P_none: nothing added. P_fd: second difference of the evader position (`oracle.accel_source=pos`). P_nf: frozen A3a learned estimator (aug basis, H = 10). Both estimators give the evader's **acceleration**, not its velocity; the policy is memoryless. Eval noise-free ("clean") and with the sensor-noise model ("noisy"); P_fd also with the tuned EMA ("noisyema"; third seed added 10-29 18:40). [A3b_posonly/summary.md](A3b_posonly/summary.md). Mean J, 95 % bootstrap CI over 3 seeds:

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

Capture rate / pursuer crash rate (mean over 3 seeds; added 10-29 18:40 from the same eval files). J also counts evader crashes and evader out-of-bounds, so J can be well above the capture rate:

| policy / eval | train | held-out RL | held-out scripted | exploiters |
|---|---|---|---|---|
| P_none clean | 0.177 / 0.010 | 0.187 / 0.009 | 0.676 / 0.016 | 0.026 / 0.005 |
| P_none noisy | 0.178 / 0.010 | 0.186 / 0.010 | 0.678 / 0.017 | 0.027 / 0.004 |
| P_fd clean | 0.286 / 0.013 | 0.132 / 0.011 | 0.796 / 0.014 | 0.076 / 0.008 |
| P_fd noisy | **0.001 / 0.928** | **0.001 / 0.917** | **0.001 / 0.915** | 0.001 / 0.664 |
| P_fd noisy + EMA | 0.168 / 0.014 | 0.085 / 0.014 | 0.627 / 0.021 | 0.139 / 0.016 |
| P_nf clean | 0.198 / 0.004 | 0.171 / 0.004 | 0.700 / 0.004 | 0.035 / 0.002 |
| P_nf noisy | 0.198 / 0.004 | 0.170 / 0.005 | 0.698 / 0.004 | 0.038 / 0.001 |

- The P_fd noisy collapse is a **crash** failure, not a missed-intercept failure: the pursuer crashes in ~92 % of episodes and captures almost never (0.001). Its J of 0.34 against the exploiters therefore comes from evader crashes / out-of-bounds, not from captures.

Seed-paired differences (mean [95 % CI], 3 seeds):

| comparison | train | held-out RL | held-out scripted | exploiters |
|---|---|---|---|---|
| P_nf - P_none (clean) | +0.014 [-0.031, +0.041] | -0.027 [-0.062, +0.032] | +0.036 [-0.064, +0.095] | **-0.140 [-0.302, -0.003]** |
| P_fd - P_none (clean) | **+0.089 [+0.062, +0.127]** | **-0.081 [-0.131, -0.021]** | **+0.119 [+0.070, +0.163]** | -0.105 [-0.267, +0.025] |
| P_nf - P_none (noisy) | +0.014 [-0.023, +0.037] | -0.028 [-0.056, +0.026] | +0.035 [-0.055, +0.090] | **-0.152 [-0.308, -0.030]** |
| P_nf - P_fd (noisy) | **+0.229 [+0.156, +0.277]** | **+0.178 [+0.125, +0.226]** | **+0.812 [+0.753, +0.855]** | -0.113 [-0.231, +0.007] |
| P_fd noisy - clean | **-0.305 [-0.350, -0.267]** | **-0.130 [-0.158, -0.083]** | **-0.898 [-0.910, -0.885]** | +0.074 [-0.043, +0.189] |
| P_nf noisy - clean | -0.001 [-0.004, +0.000] | -0.006 [-0.011, +0.003] | -0.003 [-0.009, +0.002] | -0.004 [-0.014, +0.006] |

**At 0.15 m** ([R15_rescore/A3b_posonly/summary.md](R15_rescore/A3b_posonly/summary.md); 0.15 m convention). The M0 row (seeds 0-2) is from [R15_rescore/A3d_velest/summary.md](R15_rescore/A3d_velest/summary.md); the capture / crash rates and the paired differences were computed from the re-scored eval files with `rescore.py`, with the same seed pairing and bootstrap as at 0.10 m. Mean J, 95 % bootstrap CI over 3 seeds:

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
- Not observing the evader's velocity is still crippling; the train gap to M0 grows (P_none 0.345-0.348 vs M0 0.754).
- The noise collapse of the position second difference still holds and is larger (P_fd noisy - clean -0.442 on train). P_fd noisy barely gains from the larger radius (capture 0.002, pursuer crash 0.93): a crash failure. P_nf still beats P_fd under noise and stays nearly noise-free (all noisy - clean within 0.011), though two of these tiny differences now exclude 0.
- Noise-free FD vs no estimator: the train gain grows; the held-out RL loss shrinks and its CI now includes 0, so "FD hurts on held-out RL" no longer holds at 0.15 m.
- The learned estimator still does not beat no estimator (n.s. on train and held-out; on the exploiters the clean deficit's CI now touches 0, the noisy one still excludes 0). The prediction outcomes do not change.

Findings (at 0.10 m):

- **Not observing the evader's velocity is crippling**: train J 0.29 vs 0.65 for M0 (seeds 0-2, not seed-paired runs but the same seeds/pool/frames). Neither acceleration estimate makes up for it: the memoryless policy gets acceleration but no velocity.
- **Under sensor noise, the position second difference destroys the policy** (train 0.38 -> 0.07; scripted evaders 0.98 -> 0.09); EMA filtering recovers part of it (0.25, still below P_none). The learned estimator is **unaffected by noise** (all noisy - clean differences <= 0.006) and beats FD under noise (CIs exclude 0): the closed-loop confirmation of A3a's noise result.
- But the learned estimator **does not beat no estimator at all** (n.s. on train / held-out, *worse* on the exploiters, -0.14 to -0.15, CI excludes 0). Noise-free, FD helps on train and scripted evaders but hurts on held-out RL.
- Prediction "P_fd ~ P_nf noise-free; P_nf > P_fd with noise; both > P_none": "P_nf > P_fd with noise" **held**; "P_fd ~ P_nf noise-free" **did not hold** (FD higher on train/scripted, lower on held-out RL); "both > P_none" **did not hold**.
- Design limitation (found only now, stated plainly): A3b removes the evader velocity *and* offers only an acceleration estimate, so it tests "acceleration without velocity", not a learned estimator that recovers the evader's velocity from noisy positions. The pre-registered F4 campaigns use the same inputs and inherit this limitation (see PROGRESS.md, 10-29 02:40).

<a id="a3c"></a>

## A3c_noise_ref (done 2026-10-29 03:47; eval only, 5 seeds)

The velocity-observing policies (all **trained noise-free**) evaluated with the sensor-noise model (`observation.include_noise=true`: 2 cm position, 8 cm/s relative velocity, IMU noise; the acceleration inputs are computed from the noisy observations). [A3b_posonly/summary.md](A3b_posonly/summary.md) (shared table). Mean J, 95 % bootstrap CI over 5 seeds:

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

Degradation noisy - clean (same policy): finite difference train -0.611, held-out RL -0.496, exploiters -0.533; NF-acc frozen -0.200 / -0.192 / -0.280; NF-acc + RLS -0.107 / -0.142 / -0.046; M0 -0.027 / -0.032 / -0.089 (all CIs exclude 0).

**At 0.15 m** ([R15_rescore/A3b_posonly/summary.md](R15_rescore/A3b_posonly/summary.md), shared table; 0.15 m convention). The seed-paired differences and the degradations were computed from the re-scored eval files with `rescore.py` (same pairing and bootstrap). Mean J, 95 % bootstrap CI over 5 seeds:

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

Degradation noisy - clean at 0.15 m (same policy): finite difference train -0.486, held-out RL -0.305, exploiters -0.319; NF-acc frozen -0.141 / -0.067 / -0.078; NF-acc + RLS -0.064 / -0.059 / -0.006; M0 -0.025 / -0.029 / -0.068 (all CIs exclude 0).

What changes at 0.15 m:
- The finite difference is still the worst input under noise and still degrades most, so the prediction still holds. It stays below M0 on train, but the held-out RL deficit's CI now includes 0; its scripted J under noise is 0.921 (was 0.797).
- NF-acc + RLS is still the most robust acceleration input and now beats M0 under noise on every split (the train CI now excludes 0). The exploiter margin halves because M0's noisy exploiter J rises more (0.408 -> 0.738) than the RLS policy's (0.941 -> 0.992, near the ceiling).
- The online-adaptation gain (RLS - frozen) shrinks: on train its CI now includes 0; on the exploiters it falls from +0.244 to +0.073 (CI still excludes 0). So at 0.15 m the gain is clear only on the exploiters, and smaller.

Findings (at 0.10 m):

- **The finite-difference acceleration input, the best input noise-free, becomes the worst under noise**: below M0 (-0.37 train, -0.26 held-out RL), even losing scripted evaders (0.80).
- **The learned estimator with online RLS is the most robust policy with an acceleration input** and the only one that stays above M0 under noise (held-out RL +0.12, exploiters +0.53, CIs exclude 0; train +0.10, CI touches 0); it beats the finite difference on every split by +0.20 to +0.49.
- **Online adaptation helps under noise** (RLS - frozen: train +0.09, exploiters +0.24, CIs exclude 0), the first closed-loop gain from online adaptation in this thesis (noise-free it added nothing: A1b, A2b). Plausible reason (an interpretation, not tested): the RLS readout is refit on the noisy inputs it actually receives, while the frozen basis was trained on clean ones.
- M0 itself is nearly noise-proof (-0.03), so the acceleration inputs are where noise robustness is decided.
- Caveat: all policies were trained without noise; policies trained with noise may learn to discount a noisy finite difference. F4 (self-play with noise in training) is where that is tested.
- Prediction "finite-difference acceleration degrades most": **held**.

<a id="a3d"></a>

## A3d_velest (offline done 2026-10-29 11:16; closed loop done 22:00, 3 seeds)

Can a learned estimator recover the evader's **velocity** from noisy positions? A3a trajectories; input = 10-step window of [relative position, own position, own velocity] (no evader velocity); target = the evader's current world velocity. [A3d_velest/summary_offline.md](A3d_velest/summary_offline.md). Mean error |v - v_hat| in m/s (evader speed scale: predicting zero gives 0.87-0.90):

| set | inputs | first difference (raw) | first difference + EMA (alpha 0.1, tuned) | learned, clean basis | learned, noise-aug. basis |
|---|---|---|---|---|---|
| val | clean | **0.009** | 0.269 | 0.105 | 0.139 |
| val | noisy | 3.180 | 0.439 | 0.281 | **0.217** |
| test | clean | **0.010** | 0.264 | 0.126 | 0.161 |
| test | noisy | 3.181 | 0.435 | 0.292 | **0.232** |

Reference: the measured relative velocity of A3c (8 cm/s noise per axis) has error 0.128.

- Noise-free, differencing positions is almost exact (0.01 m/s); the learned estimator is worse (0.11-0.13).
- With 2 cm position noise raw differencing is useless (3.2 m/s); the EMA filter brings it to 0.43-0.44 m/s (with lag). The learned estimator is **35 % (clean basis) to 50 % (noise-augmented basis) more accurate** than the tuned filter; both are 2-3x worse than a measured velocity with A3c's noise level.
- Pre-registered basis rule (A3a's: "aug" only if better on noisy val *and* clean-val error <= 1.1x "clean"): "aug" is better on noisy val but its clean-val error is 1.32x, so the rule selects the **clean basis**. A3d's closed loop and F4 v2 use it as registered, although the augmented basis would be more accurate under noise.
- Prediction "NF_aug < FD_ema on noisy val/test; noise-free FD_raw best": **held**.

### A3d closed loop (done 2026-10-29 22:00; 3 seeds)

Pursuers that observe an **estimated** evader velocity instead of the true one (`observation.rel_vel_source`), train pool, uniform, 150M frames, noise-free training. **V_ema**: EMA-filtered first difference (alpha 0.1). **V_nf**: frozen learned velocity (clean basis, as the rule chose) **plus** the A3a learned acceleration (one shared noisy measurement per step). References: M0 observes the true velocity (A3c; noisy eval adds 8 cm/s velocity noise); P_none observes no velocity (A3b). [A3d_velest/summary.md](A3d_velest/summary.md). Mean J [95 % CI over 3 seeds]:

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

Capture rate / evader out-of-bounds rate, noisy eval (mean over 3 seeds, same eval files; outcome flags are not mutually exclusive):

| policy | train | held-out RL | held-out scripted | exploiters |
|---|---|---|---|---|
| M0 | 0.522 / 0.049 | 0.344 / 0.047 | 0.820 / 0.078 | 0.151 / 0.084 |
| P_none | 0.178 / 0.084 | 0.186 / 0.084 | 0.678 / 0.081 | **0.027 / 0.297** |
| V_ema | 0.284 / 0.067 | 0.158 / 0.035 | 0.771 / 0.085 | 0.050 / 0.066 |
| V_nf | 0.283 / 0.093 | 0.176 / 0.049 | 0.800 / 0.079 | 0.085 / 0.067 |

**At 0.15 m** ([R15_rescore/A3d_velest/summary.md](R15_rescore/A3d_velest/summary.md); 0.15 m convention). The capture / out-of-bounds rates were computed from the re-scored eval files with `rescore.py`. Mean J [95 % CI over 3 seeds]:

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
- An estimated velocity recovers a little more of the train gap (36-40 %; 0.10 m: 28-34 %) but still does not generalise: on held-out RL neither estimator beats no velocity, and EMA's deficit (significant at 0.10 m) is now n.s. The held-out RL gap to M0 widens because M0 gains more from the larger radius.
- The learned estimator no longer leads EMA: no V_nf - V_ema CI excludes 0, clean or noisy (the 0.10 m scripted gain becomes +0.009 [-0.001, +0.029]); "V_nf > V_ema" still does not hold, now not even in direction on train.
- The exploiter deficits against P_none are now n.s.; P_none's exploiter J is still mostly evader out-of-bounds (0.285), not capture (0.072).
- "Both below M0" still holds on every split; the held-out RL and exploiter margins grow, the train margins are similar.

Findings (at 0.10 m):

- An estimated velocity recovers 28-34 % of the gap between no velocity and the true velocity on the training pool (0.29 -> 0.38-0.40 vs 0.62) and most of it on scripted evaders (0.86 -> 0.96-0.99 vs 1.00). It does **not** generalise: on held-out RL evaders neither estimator beats no velocity (EMA is worse, -0.07), and both stay far below M0.
- The exploiter "loss" against P_none is not a capture loss: P_none captures exploiters in 3 % of episodes (V_nf 9 %); its J of 0.38 comes from exploiters leaving the arena (evader OOB 0.30), since those exploiters were trained against a velocity-observing pursuer.
- Learned vs EMA: higher on every split, but the CI excludes 0 only on held-out scripted (+0.03); offline the velocity error is 33 % lower (0.29 vs 0.44 m/s), in closed loop this gives at most +0.02 to +0.04 J. Noise costs both arms little (clean -> noisy: V_ema -0.04, V_nf -0.03 train); the policies were trained clean.
- Confound: V_nf also receives the learned acceleration, V_ema does not, so V_nf - V_ema is an upper bound on the effect of the velocity estimator alone. Limitation from A3e: a noise-trained GRU would estimate velocity 36 % better than the MLP used here.
- Prediction "noisy eval: V_nf > V_ema on train and held-out RL; both far above P_none (0.29); both below M0 noisy": "V_nf > V_ema" **did not hold** (right direction, CIs include 0); "far above P_none" **held only on train / held-out scripted**, not on held-out RL or exploiters; "both below M0" **held** (the plan quoted M0 noisy as 0.70; in these eval files it is 0.62, which does not change the outcome).

<a id="a3e"></a>

## A3e_estimators (done 2026-10-29 20:28; offline, one training run per model)

Is the 10-step MLP window the right estimator architecture? Same data, split, noise draws, targets and masks as A3a / A3d (the zero-baseline errors match exactly). GRU (hidden 96, 31k parameters) and LSTM (hidden 80, 29k) with memory over the whole episode, trained on clean or noise-augmented inputs; constant-acceleration Kalman filter (KF) on the evader-position measurement, process noise tuned on noisy val (q = 10 for velocity, 1 for acceleration). MLP (32k), FD and FD + EMA reused from A3a / A3d. [A3e_estimators/summary.md](A3e_estimators/summary.md). Mean error in m/s, **test** set (val gives the same ordering):

| target / inputs | zero | FD + EMA | KF | MLP clean | MLP aug | GRU clean | GRU aug | LSTM clean | LSTM aug |
|---|---|---|---|---|---|---|---|---|---|
| velocity, clean | 0.868 | 0.264 | 0.123 | 0.126 | 0.161 | **0.060** | 0.139 | 0.078 | 0.149 |
| velocity, noisy | 0.868 | 0.435 | 0.288 | 0.292 | 0.232 | 0.800 | **0.186** | 0.568 | 0.193 |
| 0.2 s velocity change, clean | 0.340 | - | 0.338 | 0.288 | 0.290 | **0.244** | 0.255 | 0.282 | 0.287 |
| 0.2 s velocity change, noisy | 0.340 | 1.699 | 0.385 | 0.331 | 0.293 | 0.419 | **0.259** | 0.302 | 0.288 |

(Noise-free, raw FD is still best for velocity: 0.010.)

- **A recurrent estimator trained on noisy inputs is the most accurate under noise**: GRU aug velocity -20 % vs the MLP aug and -36 % vs the MLP clean basis used in the closed loop; acceleration -12 %. LSTM aug is close to GRU aug (velocity -17 %, acceleration -2 % vs MLP aug). Noise-free, the GRU is also better.
- **Recurrent nets trained on clean inputs break under noise** (GRU clean velocity 0.80, worse than the EMA filter): noise augmentation matters far more for them than for the MLP.
- **The Kalman filter is no better than the clean MLP** and 24 % worse than the MLP aug; for the acceleration target it is worse than predicting zero, like every position-differencing method here. It beats the EMA filter, so EMA is the weaker classical baseline.
- Predictions: (1) "noise-free velocity: KF < MLP" **did not hold** (equal: test 0.123 vs 0.126, val 0.118 vs 0.105); (2) "best recurrent (aug) beats MLP aug by >= 10 % on noisy velocity" **held** (-20 %); "KF within +/- 20 % of MLP aug" **did not hold** (+24 %); "KF and recurrent nets beat FD_ema" **held** for the KF and the aug nets, not for the clean-trained nets; (3) "noisy acceleration: GRU / LSTM aug <= MLP aug; KF worse than MLP aug" **held**.
- Pre-registered consequence: GRU aug beats the closed-loop MLP by >= 10 % (velocity -36 %, acceleration -12 %). **This is a limitation of A3d / F4**: their learned arm uses the MLP, not the best available estimator, so closed-loop gains of the learned arm over EMA are a lower bound for a recurrent estimator. Not rerun before the freeze.
- Caveats: one training run per model (no seed spread); 8 epochs with the last epoch kept, and the clean GRU was still improving at epoch 8; offline error only. A measured velocity with A3c's noise (error 0.128) is still better than every estimator from positions.

<a id="a3f"></a>

## A3f_specnorm (done 2026-11-03 14:41; offline, one training run per model)

User request after Pan et al. (RA-L 2026, "Learning on the Fly"), who train their residual-dynamics network with a spectral-norm penalty; pre-registered (plan, A3f row) before any run. The A3a / A3d positions-only MLP estimator is retrained with loss + beta * sum_l ||W_l||^2 over the basis' 3 layers and the shared head: **spec** = largest singular value (Pan et al.), **frob** = Frobenius norm (plain weight decay, the control); beta in {1e-4, 1e-3, 1e-2}, chosen on val in the basis' training condition. Same data, split, noise draws, masks and metric as A3a / A3d (the unregularised models re-evaluated with the A3f code match the stored numbers exactly, max |diff| 0). Files: `A3f_specnorm/summary.md`, `pred_err.json`.

Mean error (m/s); acc = evader velocity change over 0.2 s, vel = current evader velocity; test = held-out evaders.

| target | test inputs | MLP clean | spec clean | frob clean | MLP aug | spec aug | frob aug |
|---|---|---|---|---|---|---|---|
| acc | clean | 0.2884 | 0.2884 (0.0 %) | 0.2902 | 0.2901 | 0.2892 (-0.3 %) | 0.2895 |
| acc | noisy | 0.3309 | 0.3276 (-1.0 %) | 0.3244 | 0.2931 | 0.2921 (-0.3 %) | 0.2922 |
| vel | clean | 0.1261 | 0.1286 (+2.0 %) | 0.1269 | 0.1614 | 0.1619 (+0.3 %) | 0.1635 |
| vel | noisy | 0.2918 | 0.2905 (-0.4 %) | 0.2881 | 0.2324 | 0.2347 (+1.0 %) | 0.2347 |

Lipschitz bound prod ||W_l||_2 (normalised inputs), MLP -> spec at beta*: acc clean 6.4 -> 3.8, acc aug 5.9 -> 2.9, vel clean 36.5 -> 30.1, vel aug 36.7 -> 15.5 (frob lowers it about as much: 3.1, 2.2, 29.1, 27.9). beta* = 1e-4 (the smallest grid value) for 7 of 8 models (vel spec aug: 1e-3): on val, any stronger penalty is worse.

- Predictions: (1) "spec lowers the Lipschitz bound >= 2x" **held in 2 of 4** (aug bases 2.0x / 2.4x; clean bases 1.7x / 1.2x). (2) "clean basis, noisy test: spec >= 10 % better" **did not hold** (-1.0 % acc, -0.4 % vel). (3) "aug basis within +/- 5 %" **held** (-0.3 %, +1.0 %). (4) "clean test within +/- 5 %" **held** (0.0 to +2.0 %). (5) "spec <= frob on noisy test, clean basis" **did not hold** (frob slightly better on both targets).
- Pre-registered consequence: spec does not beat the closed-loop estimators (acc MLP aug -0.3 %, vel MLP clean -0.4 %) by >= 10 %, so it is **not a candidate** for A3d / F4.
- Exploratory (not the pre-registered beta): larger penalties make the clean-trained basis more noise-robust at a cost on clean inputs (acc spec beta 1e-2: noisy test -10.6 %, clean val +8.0 %; vel frob beta 1e-2: noisy test -9.7 %, clean val +60 %). Noise-augmented training does better on both counts (acc MLP aug: noisy test -11.4 % vs MLP clean, clean val +2.1 %), and A3e's GRU aug is better still.
- Caveats: one training run per model, so differences of 1-2 % are likely within seed noise; beta* sits at the grid edge, so the val-optimal penalty may be below 1e-4 (i.e. about zero); offline error only. Pan et al.'s network predicts the robot's own residual acceleration from its state and action; ours estimates an opponent's state from noisy position history.

<a id="f1"></a>

## F1_final (done 2026-10-28 04:22; 5 seeds, noise-free)

2x2 factorial {uniform (SU), QD-DPP (QD) sampler} x {plain obs, opponent-acceleration input (acc, finite difference as in X4 O2a)}, train pool, 150M frames. Reused runs: SU_plain = X3 generalists, QD_plain = D1 QD, SU_acc seeds 0-2 = X4 O2a. [F1_final/summary.md](F1_final/summary.md). Mean J (95 % bootstrap CI over 5 seeds):

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

Held-out RL is **bimodal** for the acceleration-input policies. Each seed either handles the later v10 evaders (s5-s11: J 0.74-0.97) or fails on all of them (J 0.10-0.29):

| seed | SU + acc on v10 s1/s3/s5/s7/s9/s11 | QD + acc on v10 s1/s3/s5/s7/s9/s11 |
|---|---|---|
| 0 | 0.99 0.99 **0.14 0.29 0.11 0.24** | 0.95 0.97 **0.10 0.25 0.21 0.14** |
| 1 | 0.99 0.93 0.88 0.89 0.91 0.95 | 0.99 0.95 0.87 0.90 0.96 0.95 |
| 2 | 0.99 0.94 0.87 0.92 0.93 0.95 | 0.99 0.99 **0.10 0.29 0.11 0.25** |
| 3 | 0.99 0.94 0.92 0.95 0.97 0.97 | 0.99 0.95 0.86 0.87 0.92 0.94 |
| 4 | 0.99 0.96 0.82 0.89 0.89 0.92 | 0.99 0.97 0.74 0.88 0.92 0.90 |

**At 0.15 m** ([R15_rescore/F1_final/summary.md](R15_rescore/F1_final/summary.md); 0.15 m convention):

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
- The acceleration input is still the dominant factor on train and exploiters (CIs exclude 0), but both effects shrink; the exploiter effect most, because the plain-obs policies gain most from the larger radius (SU + plain on exploiters 0.497 -> 0.806).
- The sampler still has no main effect. The one notable QD - SU difference at 0.10 m (acc input, held-out RL) now includes 0; on the exploiters QD is still lower with the acc input, but only by -0.007.
- The v10 bimodality persists: the same seeds fail (SU seed 0; QD seeds 0 and 2) at 0.19-0.70 on s5-s11, while the other seeds reach 0.94-1.00.

Findings (at 0.10 m):

- **The acceleration input is the dominant factor**: +0.20 on train and +0.46 on exploiters (CIs exclude 0).
- **The sampler has no main effect.** With the plain observation QD-DPP is +0.03-0.04 (n.s., as in D1). With the acceleration input QD-DPP is slightly lower; the held-out RL CI excludes 0 (-0.108), but this comes from one extra seed in the failing mode (QD 2/5 vs SU 1/5 seeds), which is not meaningful with 5 seeds.
- Once the opponent's acceleration is observable, diversity-aware sampling brings no measurable benefit in this setting.
- Generalisation to the v10 lineage is a seed-level, all-or-nothing property: 1 of 5 (SU + acc) and 2 of 5 (QD + acc) runs fail on it, regardless of sampler.

<a id="f2a"></a>

## F2a_exploit_acc (done 2026-10-29 06:19; 5 seeds)

Is the pursuer with the opponent-acceleration input harder to exploit? One fresh exploiter evader (PPO from scratch, 100M frames, plain observation, seed 200 + s as in D2) against each F1 "uniform + acceleration" pursuer; compared seed-paired with D2's uniform plain-observation pursuers (same seeds, same exploiter seeds). [F2a_exploit_acc/summary.md](F2a_exploit_acc/summary.md).

| seed | plain exploitability (D2 S-U) | acc exploitability | acc: capture / timeout vs exploiter |
|---|---|---|---|
| 0 | 0.697 | 0.632 | 0.33 / 0.63 |
| 1 | 0.803 | 0.070 | 0.24 / 0.07 (exploiter mostly crashes / leaves the arena) |
| 2 | 0.513 | 0.569 | 0.37 / 0.53 |
| 3 | 0.644 | 0.439 | 0.49 / 0.42 |
| 4 | 0.705 | 0.276 | 0.70 / 0.27 |
| mean [95 % CI] | 0.672 [0.586, 0.745] | **0.397 [0.218, 0.568]** | |

Seed-paired acc - plain: **-0.275 [-0.522, -0.044]** (n = 5). Without seed 1, where the exploiter failed to learn to fly safely (J 0.93 although it is captured only 24 % of the time): -0.161 [-0.338, -0.004] (n = 4).

**At 0.15 m** ([R15_rescore/F2a_exploit_acc/summary.md](R15_rescore/F2a_exploit_acc/summary.md); 0.15 m convention):

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
- Most of the exploiters' wins at 0.10 m were near misses; the acc pursuer captures 0.93-0.97 of episodes on the 4 seeds whose exploiter learned to fly.
- Acc is still less exploitable than plain (CI excludes 0), but the gap is much smaller. Without the failed seed-1 exploiter the CI now includes 0, so the "also without the failed exploiter" part of the finding does not hold at 0.15 m.
- "The pursuer stays exploitable" does not hold at 0.15 m against these exploiters (acc 0.002-0.033 on every seed); this is a lower bound, since no exploiter was trained under 0.15 m rules for these pursuers.

Findings (at 0.10 m):

- The acceleration input **lowers exploitability** (CI excludes 0, also without the failed exploiter), in addition to its gains on train / held-out / X3 exploiters (F1). It was the only intervention in this thesis that reduced exploitability as of 10-29; no sampler did (D2). [Later also: F4's EMA filter (0.675) and F8's Roncero rules.]
- The pursuer stays exploitable: a fresh exploiter still escapes neutralisation in 28-63 % of episodes on 4 of 5 seeds, mostly by surviving to timeout.
- Prediction "lower than plain (0.672), mean 0.3-0.5, paired CI excludes 0; still clearly > 0": **held** (0.397; one seed's exploiter failed, as one of 20 did in D2).
- Bookkeeping: the driver's first summary showed 3 of 5 seeds (lane-lock bug); rebuilt from the eval files on disk.

<a id="f2"></a>

## F2b_campaigns / F2c_final_eval (campaigns done 2026-10-29 18:33; F2c done 2026-10-30 14:29; 1 seed per arm, descriptive)

Two self-play campaigns with the reference papers' co-evolution protocol, identical except the observation: **acc** (both roles observe the opponent's finite-difference acceleration, frozen opponents too) vs **plain**. Uniform opponent sampling, 9 stages x 300M frames, pursuer first, warm start, seed 0, noise-free. Pre-registered as 1 seed per arm: no significance claims. [F2c_final_eval/summary.md](F2c_final_eval/summary.md), `fig_crossplay.png`. Final pursuer of each campaign (stage 8), J:

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
- **plain** leapfrogs: each new stage beats the latest opponent stage (called "cycling" here before 11-09; the ordering is transitive with no forgetting, see the F3 post-hoc transitivity). Plain P2 is beaten by every later plain evader (J 0.01-0.03), P4 by E5/E7 (0.04 / 0.02), P6 by E7 (0.07); only the final P8 handles all plain evaders (0.79-0.99).
- **acc** shows almost no within-campaign leapfrogging (every acc pursuer from P4 on neutralises every acc evader at >= 0.84, P8 at >= 0.99) but fails on the late plain evaders (P8 vs plain E7: 0.30; P2-P6: 0.17-0.23). So the acc evaders are easier for every pursuer (plain P8 vs acc evaders 0.73-0.95), and the acc final pursuer is the less robust of the two against the other campaign (worst cell 0.30 vs 0.73).

**At 0.15 m** ([R15_rescore/F2c_final_eval/summary.md](R15_rescore/F2c_final_eval/summary.md); 0.15 m convention):

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

The test-set mean, the train-pool subsets and the exploiter outcome split are not in the summary; they were computed with `rescore.rates_at` on the F2c eval files (the same code reproduces the 0.10 m values above).

Cross-play at 0.15 m (full matrix in the R15 summary): **plain** still leapfrogs, but much less (P2 vs later plain evaders 0.29-0.64, P4 vs E5 / E7 0.78 / 0.47, P6 vs E7 0.76; P8 handles all plain evaders, 0.91-1.00). **acc**: every acc pursuer from P4 on neutralises every acc evader at >= 0.95, P8 at 1.00; acc pursuers are still weakest on plain E7 (P8 0.79; P2-P6 0.46-0.75). Plain P8 vs acc evaders 0.92-0.99, so the acc final pursuer is still the less robust against the other campaign (worst cell 0.79 vs 0.92).

What changes at 0.15 m:
- **The exploitability result changes.** At 0.15 m the acc pursuer captures its exploiter in 0.670 of episodes, the plain pursuer in 0.252: the acc exploiter's wins were mostly near misses, while the plain exploiter still wins mostly by making the pursuer crash. One exploiter per pursuer and one seed per arm, so this is descriptive.
- The acc advantage shrinks (test set +0.028, held-out RL +0.037), so the prediction's +0.1 part still does not hold, now also not on the held-out RL subset; its "lower exploitability" part holds at 0.15 m (-0.419), against one 0.10 m-trained exploiter per pursuer (both values are lower bounds).
- The acc pursuer's train-pool deficit disappears (v9 evaders 0.674 in both arms); on the other campaign's evaders acc is still lower.

Findings:
- Under co-evolution the acceleration input improves the final pursuer on held-out RL evaders (+0.12), much less on the rest of the test set, and **not** on the other campaign's evaders or the train pool (-0.11).
- **Both self-play pursuers are almost fully exploitable (0.97), and the exploiters win by making the pursuer crash into the ground (97 % of episodes), not by surviving to timeout** as against the fixed-pool pursuers (D2, F2a: 0.40-0.69, mostly timeouts). This suggests the self-play pursuers learned aggressive near-ground pursuit that a best response can exploit. One exploiter per pursuer.
- Both self-play pursuers do poorly on the thesis train pool (v9 evaders: 0.34 / 0.48), which they never trained against; pursuers trained directly on that pool reach 0.72 (plain) / 0.94 (acc) (F1).
- Prediction "acc final >= plain final + 0.1 J on the other campaign's evaders and on the test set, and lower exploitability": **did not hold.** Other campaign's evaders: acc is lower on either reading. Test set: +0.069 (< 0.1; only the held-out RL subset passes, +0.116). Exploitability: 0.965 vs 0.974 is no meaningful difference.
- Limitation: one seed per arm. In F1 the acceleration result was bimodal across seeds, so a single campaign per arm cannot separate an input effect from a seed effect.

<a id="f3"></a>

## F3_sampler_sp / F3c (final eval done 2026-11-05 07:42; 2 seeds per sampler, descriptive by pre-registration)

RQ1 under the reference papers' protocol: full self-play campaigns (plain observation, 9 stages x 300M frames, pursuer first, warm start, noise-free) that differ only in the opponent sampler, used **by both roles** (amendment 3): **uniform**; **hedge** = Roncero eq. (12) exactly (gamma 0.7, eta 2); **pfsp** = Hedge with gamma 0, p ~ (1 - J)^2 (the hard-opponent core of AgilePE / Gavin); **qd_dpp** = the same Hedge quality re-weighted by a DPP over behaviour descriptors. Campaign seeds 0 / 100 (uniform seed 0 = F2b plain); final pursuer (stage 8). Other / own campaign evaders exclude the stage-1 evaders, which are identical across same-seed campaigns (amendment 6). With 2 seeds **no difference can be called supported** (pre-registered). [F3_sampler_sp/summary.md](F3_sampler_sp/summary.md); per-campaign values in `results.json`. The earlier pursuer-only F3 campaigns (hedge s0/s1, qd_dpp s0, pfsp s0) are kept on disk, not analysed.

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

Per seed (seed 0 / seed 1), held-out RL: uniform 0.777 / 0.696; hedge 0.791 / 0.882; qd_dpp 0.803 / 0.588; pfsp 0.910 / 0.639. Worst held-out cluster: uniform 0.646 / 0.540; hedge 0.692 / 0.816; qd_dpp 0.705 / 0.377; pfsp 0.827 / 0.503.

Seed-paired differences (mean of the 2 pairs [range]):

| pair | held-out RL | worst held-out cluster | other campaigns' evaders | train pool | exploitability | evader Vendi |
|---|---|---|---|---|---|---|
| hedge - uniform | +0.100 [+0.014, +0.186] | +0.161 [+0.046, +0.276] | +0.036 [+0.009, +0.063] | +0.047 [-0.007, +0.101] | +0.007 | +0.043 |
| qd_dpp - uniform | -0.041 [-0.108, +0.026] | -0.052 [-0.163, +0.059] | +0.026 [+0.023, +0.029] | -0.008 | +0.004 | -0.107 |
| qd_dpp - hedge | -0.140 [-0.293, +0.013] | -0.213 [-0.439, +0.013] | -0.010 | -0.055 [-0.098, -0.012] | -0.002 | -0.149 |
| pfsp - uniform | +0.038 [-0.057, +0.133] | +0.072 [-0.037, +0.181] | **-0.226** [-0.262, -0.190] | -0.028 | +0.011 | **+0.529** [+0.412, +0.647] |
| qd_dpp - pfsp | -0.079 [-0.107, -0.051] | -0.124 [-0.126, -0.122] | **+0.252** [+0.213, +0.292] | +0.020 | -0.006 | -0.636 |

Capture-only (secondary, amendment 5) gives the same ordering: hedge - uniform held-out RL +0.112, worst cluster +0.153; qd_dpp - uniform -0.019 / -0.035; pfsp - uniform on other campaigns' evaders -0.290.

**At 0.15 m** ([R15_rescore/F3_sampler_sp/summary.md](R15_rescore/F3_sampler_sp/summary.md); 0.15 m convention):

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

Evader Vendi changes slightly because its descriptors include J vs the F3 pursuers. The sampling probabilities are training statistics and do not change.

Per seed (seed 0 / seed 1), held-out RL: uniform 0.959 / 0.944; hedge 0.960 / 0.989; qd_dpp 0.973 / 0.900; pfsp 0.994 / 0.894. Worst held-out cluster: uniform 0.914 / 0.846; hedge 0.874 / 0.921; qd_dpp 0.877 / 0.842; pfsp 0.931 / 0.838.

Seed-paired differences at 0.15 m (mean of the 2 pairs [range]):

| pair | held-out RL | worst held-out cluster | other campaigns' evaders | train pool | exploitability | evader Vendi |
|---|---|---|---|---|---|---|
| hedge - uniform | +0.022 [+0.000, +0.044] | +0.017 [-0.040, +0.075] | +0.025 [+0.001, +0.050] | +0.069 [-0.007, +0.145] | +0.062 [+0.019, +0.106] | +0.026 [-0.091, +0.143] |
| qd_dpp - uniform | -0.016 [-0.045, +0.013] | -0.021 [-0.037, -0.004] | +0.009 [+0.003, +0.015] | +0.013 [-0.007, +0.034] | -0.056 [-0.116, +0.004] | -0.113 [-0.159, -0.068] |
| qd_dpp - hedge | -0.038 [-0.089, +0.013] | -0.038 [-0.079, +0.003] | -0.016 [-0.035, +0.002] | -0.056 [-0.111, -0.000] | -0.119 [-0.222, -0.015] | -0.139 [-0.211, -0.068] |
| pfsp - uniform | -0.008 [-0.050, +0.034] | +0.004 [-0.009, +0.017] | **-0.042** [-0.058, -0.025] | +0.072 [+0.030, +0.113] | -0.254 [-0.266, -0.243] | **+0.569** [+0.441, +0.697] |
| qd_dpp - pfsp | -0.008 [-0.021, +0.006] | -0.025 [-0.054, +0.004] | **+0.051** [+0.040, +0.061] | -0.058 [-0.079, -0.037] | +0.198 [+0.150, +0.246] | -0.683 [-0.765, -0.600] |

Bold marks the cells bolded in the 0.10 m tables where the same reading still holds. Capture-only at 0.15 m: hedge - uniform held-out RL +0.022, worst cluster +0.006; qd_dpp - uniform -0.020 / +0.007; pfsp - uniform on other campaigns' evaders -0.048.

What changes at 0.15 m:
- Most of Hedge's advantage over uniform is near misses: held-out RL +0.022 (seed 0: +0.000), worst cluster +0.017, now mixed across seeds. Hedge still has the highest mean on held-out RL and the worst cluster, but "higher on both seeds" no longer holds for the worst cluster.
- PFSP's specialisation shrinks but remains (other campaigns' evaders still negative on both seeds; own campaign 0.930 vs 0.979-0.996). QD-DPP stays close to uniform, and its gap to Hedge shrinks.
- "No sampler changes exploitability" does not hold at 0.15 m: exploitability is 0.50-0.81, PFSP's pursuer is the least exploitable (both seeds) and Hedge is more exploitable than uniform (both seeds). These are lower bounds (0.15 m convention).
- Against the pre-registered "all paired differences within +/- 0.05": at 0.15 m every difference vs uniform on held-out RL, the worst cluster and other campaigns' evaders is within it, but the train pool and exploitability differences are not.

Findings:
- **Hedge (Roncero) is the best sampler in self-play on average** (the single best held-out run is PFSP seed 0, 0.910): higher than uniform on held-out RL and the worst held-out cluster on **both** seeds, though the size differs strongly between seeds (held-out RL +0.014 / +0.186); J is also higher on average on the other metrics. This is the direction Roncero reports and the **opposite** of our fixed-pool result D1 (win-rate softmax worse than uniform on 5/5 seeds); the protocols differ (co-evolution, both roles, Hedge eta = 2 with gamma 0.7 vs softmax with a 0.5 floor on a fixed pool).
- **QD-DPP does not improve on uniform** (mixed across seeds) and is below Hedge on average (held-out RL seed 0 +0.013, seed 1 -0.293); on other campaigns' evaders it is level with Hedge and slightly above uniform. Its seed 1 is the worst held-out result of all 8 campaigns.
- **PFSP (no uniform share) produces the most diverse evader populations** (both seeds) but the most specialised pursuer: clearly worse against the other campaigns' evaders (both seeds) and even against its own campaign's evaders (0.776 vs 0.94-0.97); mixed on held-out RL (+0.133 / -0.057).
- **No sampler changes exploitability**: every self-play pursuer is ~fully exploitable (0.97-0.98), as in F2c.
- Predictions (pre-registered 10-28 / amendments 1-3): "all paired differences within +/- 0.05": **did not hold** for hedge - uniform (+0.10 / +0.16) and pfsp on other campaigns' evaders (-0.23); held for exploitability and most qd_dpp - uniform differences. "D1: hedge <= uniform": **did not hold** (Roncero's "hedge > uniform" is the observed direction). "qd_dpp ~ uniform": **held** on held-out RL (-0.041), marginally outside on the worst cluster (-0.052). "pfsp <= uniform on held-out RL and worst cluster": **did not hold** on the means (+0.038 / +0.072, mixed seeds). "pfsp has the highest max sampling probability": **held** (0.988). "qd_dpp >= pfsp": **did not hold** on held-out RL / worst cluster (-0.079 / -0.124), **held** on other campaigns' evaders (+0.252). AgilePE's claim "PFSP sustains diversity" (evader Vendi pfsp >= uniform): **consistent** (+0.529, both seeds).
- Limitations: 2 seeds per sampler (descriptive only); one exploiter per pursuer; the sampler only reweights a population of 4-5 RL opponents per role, so the distributions differ less than the names suggest (except pfsp).

<a id="f3-posthoc"></a>

### F3 post-hoc: transitivity per sampler (added 2026-11-09; not pre-registered, no new runs)

X1's measures on the F3 campaigns: additive-logit R² (one strength per player) and rank-reversal fraction (delta 0.1; a pursuer pair ordered one way on one evader and the other way on another). [F3_sampler_sp/transitivity.md](F3_sampler_sp/transitivity.md) (matrices included), `transitivity.json`, `exps/f3_transitivity.py`. Within each campaign: pursuer stages 2-8 x RL evader stages 1-7, from the in-campaign stage evals, 0.10 m only (pursuer stage 0 is left out because evader stage 1 never played it):

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

- **Uniform, Hedge and QD-DPP campaigns are ordinally transitive**: staircase matrices (P_k vs older evaders 0.79-1.00, vs newer evaders 0.01-0.14), no rank reversals, and the final pursuer still beats every older evader (always worst against the latest one, E7). Their R² is only 0.65-0.73 because a 4 x 4 step-shaped matrix with cells near 0.01 / 0.99 fits an additive logit poorly; the reversal fraction is the better measure here.
- **PFSP campaigns are not transitive**: the final pursuer has forgotten an older evader. This is the forgetting that Roncero attributes to latest-opponent training; PFSP without a uniform share behaves like it.
- Across campaigns the game is mostly transitive (close to X1's 0.777-0.835 / 0.024-0.045); PFSP's evaders are the exception, consistent with their higher diversity (Vendi 2.985) and with PFSP's most specialised pursuer. At 0.15 m the reversals fall to 0.000-0.045; R² also falls because most cells approach 1.
- Terminology (applied 2026-11-09): the stage-to-stage pattern previously called "cycling" in F2c is ordinally transitive in uniform / Hedge / QD-DPP campaigns, so it is now called leapfrogging; only PFSP shows forgetting (a true cycle back to an older opponent).
- Limitations: 2 seeds; 4 x 4 within-campaign matrices, half from pursuer-side and half from evader-side evals.

<a id="f4"></a>

## F4_estimator_sp v2 (campaigns done 2026-10-31 16:36; F4c done 22:09; analysis fixed 2026-11-02 10:25; 3 seeds per arm)

Self-play campaigns (reference papers' protocol: uniform sampling, 9 stages x 300M frames, pursuer first) in which the pursuer **never observes the true evader velocity** and is trained and evaluated with sensor noise. Both arms get an *estimated* evader velocity: **ema** = EMA-filtered finite difference of noisy positions (A3d V_ema, the classical baseline); **nf** = learned velocity estimate + learned acceleration estimate (A3d V_nf). Frozen pursuers seen by the evader get the same estimators with clean inputs. 3 campaign seeds per arm (0 / 100 / 200), final pursuer (stage 8). [F4_estimator_sp/summary.md](F4_estimator_sp/summary.md), per-campaign values in `results.json`. Bookkeeping: the driver's first summary (10-31 22:09) still used the v1 arm names, so the ema arm was missing; rebuilt from the eval files with the pre-registered contrast nf - ema (no data changed).

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

**At 0.15 m** ([R15_rescore/F4_estimator_sp/summary.md](R15_rescore/F4_estimator_sp/summary.md); 0.15 m convention):

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

How the fresh exploiters win at 0.15 m (fractions of episodes; computed with `rescore.rescore` on the F4 exploiter eval files, which reproduces the 0.10 m table above):

| campaign | J vs exploiter | capture | pursuer crash | timeout | evader crash / OOB |
|---|---|---|---|---|---|
| ema-s0 / s1 / s2 | 0.696 / 0.507 / 0.506 | 0.39 / 0.46 / 0.41 | 0.00 / 0.00 / 0.00 | 0.30 / 0.49 / 0.49 | 0.30 / 0.04 / 0.10 |
| nf-s0 / s1 / s2 | 0.484 / 0.505 / 0.602 | 0.45 / 0.49 / 0.60 | 0.00 / **0.50 / 0.40** | 0.51 / 0.00 / 0.00 | 0.03 / 0.01 / 0.00 |

Post-hoc, stage-1 evaders excluded, at 0.15 m: other campaigns' evaders ema 0.628 [0.488, 0.712], nf 0.835 [0.645, 0.971], nf - ema **+0.207 [+0.156, +0.287]**; own campaign ema 0.880 [0.785, 0.928], nf 0.938 [0.903, 0.984], nf - ema +0.058 [-0.024, +0.143].

What changes at 0.15 m:
- The nf generalisation advantage holds and now also reaches held-out RL (CI excludes 0; at 0.10 m it just included 0). The X3-exploiter and other-campaigns gaps shrink; both CIs still exclude 0.
- **"nf is more exploitable" does not hold at 0.15 m** (CI includes 0; both arms at 0.43-0.47). nf-s1 / s2 exploiters still win mainly by making the pursuer crash, but the pursuer now captures 0.45-0.60 of the time against nf exploiters and 0.39-0.46 against ema exploiters. Lower bounds (0.15 m convention).
- For the prediction: held-out RL now **holds**; X3 exploiters still **holds**; exploitability lower for nf still does not hold (no difference).
- Both arms are still well below the full-state F1 pursuers on held-out RL (0.44-0.61 vs 0.80-0.90 in F1 at 0.15 m).

Findings (at 0.10 m):

- With learned estimators the self-play pursuer **generalises better** to the other campaigns' evaders and the X3 exploiters (CIs exclude 0); held-out RL +0.122 (CI just includes 0); train pool and held-out scripted: no difference. Noise-free evaluation gives the same picture, so this is not only noise robustness.
- But the nf pursuers are **more exploitable** (CI excludes 0). In 2 of 3 nf campaigns the exploiter makes the pursuer crash into the ground (88-92 %), the F2c failure mode. The ema exploiters win by survival to timeout and partly fail themselves (evader crash / OOB 9-32 %), which lowers the ema arm's exploitability. Captures against the exploiter are similarly low in both arms (0.05-0.14).
- Without the true evader velocity both arms remain far below the full-state pursuers (held-out RL 0.27-0.39 vs 0.63-0.84 in F1), consistent with A3b / A3d.
- On the fixed pool (A3d closed loop) nf and ema did not differ (+0.02 to +0.03, n.s.); under co-evolution nf is clearly better at generalisation. The co-evolving opponents differ between arms, so the arms are compared on the shared test sets and on each other's evaders, not on their own pool.
- Prediction "nf > ema on held-out RL and X3 exploiters by >= 0.05; exploitability lower for nf (low confidence)": X3 exploiters **held** (+0.441); held-out RL **held in direction and size, not significant** (+0.122, CI includes 0); exploitability **did not hold** (nf higher by +0.197).
- **Post-hoc (11-04, plan §9): stage-1 evaders excluded.** Stage-1 evaders are byte-identical across campaigns with the same seed (nf-sK and ema-sK share theirs), so "other campaigns' evaders" contained a copy of each pursuer's own stage-1 opponent. Without them: other campaigns' evaders ema 0.367 [0.274, 0.417], nf 0.618 [0.463, 0.788], nf - ema **+0.252 [+0.189, +0.371]**; own campaign ema 0.683 [0.633, 0.753], nf 0.712 [0.629, 0.839], nf - ema +0.029 [-0.035, +0.086]. Conclusions unchanged; the registered numbers above are unchanged (`*_no_s1` columns in `results.json` / `summary.md`).
- Limitation: 3 seeds, one exploiter per pursuer; the estimator is the clean-trained MLP chosen by the A3d rule, not the better noise-trained GRU found later (A3e).

<a id="f5"></a>

## F5_acc_sp Exp 1 / F5c (3 seed pairs; campaigns done 2026-11-07 06:45; F5c done 08:31; written up 2026-11-07 17:45; descriptive)

RQ2 under the reference papers' protocol: does the learned acceleration estimate (A2b machinery, velocity observed) help the pursuer in full self-play? New arm **uniform_rls**: F3 protocol (uniform sampler on both roles, plain observation + the learned estimate of the evader's mean acceleration over the next 0.2 s, online RLS on the last layer, 9 stages x 300M frames, pursuer first, warm start, clean training; frozen pursuers seen by the evader run the same estimator on clean inputs), campaign seeds 0 / 100 / 300. **uniform_plain** = F2-plain (seed 0) + F3-uniform-s1 (seed 100) + the new uniform_plain-s2 (seed 300); **uniform_fd** = F2-acc (finite-difference acceleration, 1 seed). Final pursuer (stage 8), evaluated clean and with the A3c sensor-noise model (eval only); RLS ablation = the same pursuer with RLS frozen at eval (B1). Amendment 2 reports a difference whose 3-seed CI excludes 0 as supported (coarse). [F5_acc_sp/summary_exp1.md](F5_acc_sp/summary_exp1.md); per-seed values in `results_exp1.json`; 2-seed analysis kept as [summary_exp1_2seeds.md](F5_acc_sp/summary_exp1_2seeds.md); every J at 0.15 m in [R15_rescore/F5_acc_sp/summary_exp1.md](R15_rescore/F5_acc_sp/summary_exp1.md).

Deviations: (a) Amendment 1: the estimator was retrained before the campaigns (H* = 10 by the A2a rule; NF + RLS (S1) error 0.110 m/s vs 0.143 for finite-difference persistence on val, -23 %; on test 0.143 vs 0.174; [estimator/summary.md](F5_acc_sp/estimator/summary.md)). (b) The user paused Exp 2 (`qd_dpp_rls`) on 11-05, so F5c was split (11-05 16:08) and Exp 1 is analysed alone; "campaign evaders" are the RL evaders of the 12 non-paused plain-evader campaigns (8 F3, 2 F5 uniform_rls seeds 0 / 100, and the 2 seed-300 campaigns), stage-1 evaders excluded. (c) **Amendment 2** (11-06 12:45, decided after the 2-seed results were seen; optional-stopping risk disclosed, no further seeds): a third seed pair (seed 300), and every Exp 1 pursuer re-evaluated against the new campaigns' evaders. The 2-seed numbers (11-06 12:40) are superseded. Per-seed values below are in seed order 0 / 100 / 300; 0.15 m values in parentheses (0.15 m convention; see the F5 post-hoc section).

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

Capture-only (secondary) has the same signs. rls - plain on held-out RL: +0.127 clean, +0.099 noisy. On other campaigns' evaders: -0.029 clean, -0.167 noisy.

Findings:
- **Clean: the learned acceleration estimate helps on held-out evaders, on all 3 seed pairs**: held-out RL +0.091 and worst held-out cluster +0.128, both CIs exclude 0; smaller supported gains: own campaign +0.026, X3 exploiters +0.020, held-out scripted +0.059. It does **not** help against the other campaigns' evaders (-0.027; +0.046 / +0.031 / -0.157 per pair) or on the train pool (+0.024; seed 300: -0.085). Exploitability is ~unchanged (-0.018).
  - The third pair weakened the 2-seed result: other campaigns' evaders went from +0.034 to -0.027, train pool from +0.078 to +0.024, held-out RL from +0.110 to +0.091.
  - The seed-300 uniform_rls pursuer is the weakest against the other campaigns' evaders (0.742). It loses only by crashing: pursuer crash 0.265, timeout 0.000 (seeds 0 / 100: crash 0.119 / 0.149). Its worst opponents are the stage-7 evaders of other F5 campaigns (J 0.29-0.35).
- **Noisy: the learned estimate keeps a held-out gain on average but falls below plain against the self-play evaders.** Held-out RL +0.086, worst cluster +0.127 (both CIs include 0, because seed 300 is -0.014 / -0.058); other campaigns -0.140 and own -0.115, negative on all 3 pairs. Finite differencing collapses under noise (fd - plain -0.100 on held-out RL, -0.472 on other campaigns' evaders).
- **Running RLS under noise causes most of the noisy loss against self-play evaders** (rls - frozen, other campaigns' evaders -0.103, all 3 pairs). Under noise RLS helps a little on held-out RL (+0.031) and the worst cluster (+0.039) and is worse on the train pool (-0.025) and exploitability (+0.027). Clean, RLS changes nothing (all |difference| <= 0.015), as in A1b / A2b. The A3c result (RLS beats frozen under noise; pool-trained pursuers) is **not reproduced** in self-play.
- **Every pursuer stays ~fully exploitable at 0.10 m** (0.94-0.98): the fresh exploiters make the pursuer crash in 0.92-0.98 of the episodes (timeout 0). The one exception is uniform_fd under noise (0.751; pursuer crash 0.80).
- Plain self-play pursuers are noise-sensitive: from clean to noisy, held-out RL -0.135 and worst cluster -0.174 (other campaigns -0.082, train -0.054); M0 in A3c dropped -0.03.
- **At 0.15 m the clean advantage of the estimate shrinks to final-approach precision**: the gains are small though their CIs exclude 0; under noise the deficit against campaign evaders mostly disappears; exploitability falls to 0.65 (rls), 0.77 (plain) and 0.33 (fd), lower bounds: many exploiter wins are near misses inside 0.15 m.
- **Predictions (pre-registered 11-04 17:40), re-checked with 3 pairs:**
  - (1) Clean:
    - "rls - plain >= +0.05 on held-out RL": **held**. +0.091, CI excludes 0, every pair >= +0.055.
    - "no gain on other campaigns' evaders": **held** (-0.027).
    - "exploitability within +/- 0.05": **held** (-0.018).
  - (2) Noisy:
    - "plain pursuers lose <= 0.05": **did not hold**. Held-out RL -0.135, worst cluster -0.174, other campaigns -0.082, train -0.054. It held only for exploitability.
    - "fd falls below plain": **held**.
    - "rls >= plain on held-out RL": **held on the mean** (+0.086), not on every pair (seed 300: -0.014).
  - (4) RLS:
    - "clean, rls - frozen within +/- 0.03": **held**.
    - "noisy, rls > frozen by >= 0.03 on the train pool and lower exploitability": **did not hold** (-0.025; exploitability +0.027).
  - Prediction (3) belongs to Exp 2.
- Limitations:
  - 3 seed pairs (uniform_fd 1); the third pair was added after the first two were seen.
  - One exploiter per pursuer, trained against the RLS-on pursuer and reused for the frozen evaluation.
  - For self-play pursuers the held-out v10 evaders are easier than the train pool's late evaders (F2c caveat), so the worst-cluster and other-campaign columns matter more than the held-out mean.
  - The estimator is the MLP; A3e found a noise-trained GRU to be more accurate.
  - Noise is applied only at eval.

<a id="f5-posthoc"></a>

### F5 Exp 1, post-hoc: how evaders beat the self-play pursuers (added 2026-11-06 17:40; descriptive, not pre-registered)

Written before the seed-300 pair finished: pooled over seeds 0 / 100 only (the 2-seed F5c); the seed-300 pursuers also lose only by crashing (timeout 0). From existing eval files only ([F5_acc_sp/outcomes_posthoc.md](F5_acc_sp/outcomes_posthoc.md), `exps/f5_outcomes.py`; pooled over seeds, weighted by episodes, stage-1 evaders excluded from campaign evaders). An evader wins (episode not neutralised) only by a pursuer crash or a timeout.

- **Every evader win against the final F5 Exp 1 pursuers is a pursuer crash; timeout is 0.000 for every arm, evader group and noise condition.** Pursuer crash rate, clean, uniform_rls / uniform_plain: own campaign 0.039 / 0.064, other campaigns 0.130 / 0.162, held-out scripted 0.134 / 0.208, held-out RL 0.154 / 0.248, train pool RL (v9 / v11) 0.386 / 0.478, fresh exploiter 0.970 / 0.974 (uniform_fd: other campaigns 0.345, exploiter 0.968). The pursuer wins by capture (0.46-0.93 against non-exploiter groups), in ~1.1-1.7 s.
- **Crash episodes are near misses on low targets** (clean, crash vs capture episodes): same length (1.2-2.1 s vs 1.1-1.7 s), closest approach 0.13-0.18 m before the crash, evader lower (1.0-1.2 m vs 1.4-1.6 m), pursuer vertical speed about twice as high (1.3-1.7 vs 0.6-1.1 m/s): the pursuer dives at the evader, misses, and cannot pull up. **Scripted evaders, which cannot lure deliberately, also win only this way** (0.13-0.21), so the dive is a pursuer weakness, not only an evader tactic; learned evaders and exploiters trigger it more often.
- **Part of the non-exploiter pursuer crashes may be body contact, not a clean miss.** The Crazyflie collider (`cf2x_isaac.usd`) is one 0.10 x 0.10 x 0.03 m box per drone and the two drones are not collision-filtered, so the bodies can touch without a capture at centre distances 0.10-0.145 m (closest distance sampled at 50 Hz, physics at 100 Hz). Against campaign, test and train-pool evaders, 0.25-0.78 of the pursuer-crash episodes come within 0.145 m, and in the closest bin (0.10-0.12 m) the evader crashes in the same step in 0.03-0.21 of them (0.00-0.03 beyond 0.145 m). Against the fresh exploiters only 0.17-0.29 come within 0.145 m and both crash in <= 0.07: **the exploit is mainly a clean dive-and-miss.** Contact is not logged (consistent with, not proof of, collisions); a real Crazyflie would also crash on such contact.
- **Inside the campaigns, every evader stage learns to lure the current RL pursuers, and the next pursuer stage patches it** (F5 uniform_rls s0 / s1, F2-plain, F3-uniform-s1). Evader stages 3, 5 and 7 make the stage >= 2 pursuers crash in 0.79-0.99 of episodes (one exception: stage 3 of uniform_rls-s1 vs pursuer 2, 0.47) and beat the stage-0 pursuer by timeout instead (0.25-0.74); the next pursuer stage captures them in 0.68-0.98 of episodes (stage 8 vs evader stage 7: 0.68-0.90). The leapfrogging does not shrink from stage 3 to stage 7, and the final pursuer is still crashed by the last evader stage in 0.06-0.24 of episodes. No evader is trained against the final pursuer; the fresh exploiter plays that role and reaches 0.97.
- **Same reward, no crash weakness in pool-trained pursuers**: the X3 generalists (5 seeds) crash in 0.001-0.004 of episodes against scripted and 0.009-0.033 against RL evaders, and are beaten by timeout (exploiters: timeout 0.04-0.70, crash <= 0.002). The reward settings were not changed during the thesis, so the self-play opponent distribution is the leading explanation (not tested: training length, warm start and sampler also differ).
- **Capture radius 0.15 m instead of 0.10 m** (eval-only re-score, added 2026-11-06 18:25; section 5 of `outcomes_posthoc.md`, 2-seed values; the 3-seed values are in the F5c table above). Exact for the same policies: an episode is identical up to the first step with centre distance <= 0.15 m, where a 0.15 m radius would end it as a capture; J at 0.10 m from the episode files reproduces `results_exp1.json` exactly. On 2 seeds the rls - plain advantage shrinks (held-out RL +0.110 -> +0.030, worst cluster +0.171 -> +0.060) and the noisy deficit against campaign evaders mostly disappears (other -0.105 -> -0.020, own -0.164 -> -0.005). So much of the learned estimate's gain is converting passes at 0.10-0.15 m into captures (final-approach precision), and much of its noisy loss is near misses inside 0.15 m. Pool-trained X3 generalists gain too (X3 exploiters, seeds 0-3: 0.30-0.61 -> 0.68-0.92; held-out RL 0.37-0.96 -> 0.57-0.99): their exploiters' timeouts also contain close passes. The 0.10 m radius matches Roncero et al.
- Interpretation (hypothesis): each self-play pursuer stage only has to counter 2-4 specific RL evaders, starting from the previous aggressive pursuer, so patching each lure is cheaper than general caution near the ground. Limitations: episode means, not trajectories; the lower evader altitude in crash episodes may partly be spawn altitude; post-hoc.

<a id="f6"></a>

## F6_div_sp / F6c (campaigns done 2026-11-06 11:42 / 20:23; F6c done 2026-11-07 04:13; written up 17:45; 2 seeds, descriptive)

**Question:** can a diversity term for the evaders produce more diverse evaders while the pursuer stays as good as under Roncero's Hedge?

**Setup:** arm **hedge_div** = F3 `hedge` (plain obs, Hedge gamma 0.7 / eta 2 on both roles, 9 stages x 300M, seeds 0 / 100) + PSD-PSRO's policy-space diversity regulariser for the evaders only (amendment 1: Hedge instead of QD-DPP). Every RL evader after the first maximises beta * min(min_j d_j(s), cap), as a loss term and as a per-step intrinsic reward; d_j = mean-only KL to each earlier RL evader of its campaign; cap = 2. beta from the training-only pilot (F3 hedge-both-s0, evader stage 5, 100M frames per beta): J vs training pursuers 0.498 (beta 0), 0.101 (beta 0.1), 0.187 (beta 1.0); end distance 165 (beta 0.1) and 39 (beta 1.0); the rule chose **beta = 1.0** ([pilot/choice.json](F6_div_sp/pilot/choice.json)). F6c = F3c for each final pursuer plus the 8 F3 final pursuers vs the F6 evaders (probes, used for the evader Vendi score). Primary contrast: hedge_div - hedge (same seeds; 2 seeds, descriptive). [F6_div_sp/summary.md](F6_div_sp/summary.md); 0.15 m: [R15_rescore/F6_div_sp/summary.md](R15_rescore/F6_div_sp/summary.md).

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

Against qd_dpp / uniform / pfsp, hedge_div is lower on other campaigns' evaders by -0.461 / -0.435 / -0.209 (both seeds) and has higher evader Vendi by +0.904 / +0.797 / +0.268.

Findings:
- **The diversity term produced more diverse evaders but a much weaker pursuer.** Evader Vendi is 3.33 / 3.18, above every F3 campaign (2.30-3.02) on both seeds. The hedge_div pursuer is below Hedge on held-out RL, the worst cluster, other campaigns' evaders and the train pool, on both seeds; held-out scripted, X3 exploiters (-0.016, mixed seeds), own campaign and exploitability are ~unchanged.
- **The regulariser was saturated, so it acted as a survival bonus, not as a diversity gradient.** Over the last 10 logged updates of evader stages 3 / 5 / 7 the mean min-KL distance was 52 / 15 / 28 (seed 0) and 31 / 13 / 18 (seed 1), always far above cap = 2 (lowest logged batch mean 4.4); the mean intrinsic reward per stage was 1.98-2.00 per step (maximum 2.0), so the loss term had almost no gradient. What remained is about +2 per step while the episode lasts, 20 x the evader's survival reward (`reward_step` 0.1); any episode end, including a pursuer crash (neutral in the game reward), ends this stream. The Vendi diversity is therefore a side effect of a different objective; a larger cap (or a normalised distance) would be needed to test the PSD-PSRO term itself.
- **The F6 evaders stopped luring, so the hedge_div pursuers never learned to resist it.** Inside the hedge_div campaigns evader stages 3 / 5 / 7 beat the stage >= 2 pursuers by a mix of timeout (0.31-0.66) and pursuer crash (0.22-0.45); in F3 hedge-both-s0 the same stages crash the pursuer in 0.96-0.99 of episodes, timeout 0.00. Against the F3 evaders, which do lure, the final hedge_div pursuers crash in 0.753 (seed 0) / 0.482 (seed 1) of episodes (F3 hedge pursuers 0.144 / 0.105; timeout 0 in all four). For the 8 F3 pursuers the F6 evaders are about as hard as the F3 evaders (mean J 0.801, pursuer crash 0.200; vs other F3 campaigns' evaders 0.811). Interpretation (post-hoc): with the term on, luring stops paying for the evader, and the anti-lure pressure that made Hedge's pursuer robust disappears.
- **At 0.15 m** much of hedge_div's deficit is near misses, but hedge_div - hedge is still negative on both seeds; its exploitability (a lower bound) falls below Hedge's.
- **Predictions (pre-registered 11-05 ~16:00, amendment 1 16:12):**
  - (1) Mechanism, mean distance >= cap/2 at the end of evader stages 3 / 5 / 7: **held**, trivially (13-52 >> 1); the term was saturated (see above).
  - (2) Diversity, Vendi above hedge (2.499) and qd_dpp (2.350) on both seeds and >= +0.3 vs hedge: **held** (+0.754; both seeds above every F3 value).
  - (3) Pursuer:
    - hedge_div - hedge >= -0.05 on held-out RL and the worst cluster: **did not hold** (-0.262 / -0.287, both seeds).
    - >= 0 on other campaigns' evaders: **did not hold** (-0.471).
  - (4) Exploitability unchanged (0.97-0.98): **held** (0.979).
  - The user's target (more diversity at no pursuer cost) was not met.
- **Consequence for the plan:** the regulariser at cap 2 / beta 1 is not a usable thesis method in its current form; F5 Exp 2 should not be rerun with Hedge + regulariser unless the term is redesigned (cap / normalisation) and re-piloted; since F7 may change the luring dynamics, that decision waited for F7c (user).
- Limitations: 2 seeds; Vendi is computed in F3's frozen descriptor space, which includes J vs the F3 pursuers, so a different way of winning (timeout vs lure) counts as diversity; in-campaign stage evals are vs each stage's own training opponents only; div_distance is a batch mean, not per state.

<a id="f7"></a>

## F7_floor_sp / F7c (campaigns done 2026-11-07 21:22; F7c done 22:30; written up 2026-11-08 14:45; 1 seed per arm, descriptive)

**Question:** evaders beat self-play pursuers by luring them into the floor (F5 post-hoc). Does (a) a pursuer floor penalty and (b) a 0.15 m capture radius in training remove that weakness?

**Setup** (pre-registered 11-06 19:30): two campaigns with the F3 uniform protocol (plain obs, uniform sampler on both roles, 9 stages x 300M, campaign seed 0) that differ from `thesis-F2-plain` (the baseline, trained at 0.10 m) only in **r15** (capture radius 0.15 m in training and in the in-campaign evals) and **floor_r15** (the same + pursuer floor penalty `reward_floor_weight` = 1.0: the evader's floor ramp, 0 at 0.6 m -> 1 per step at the 0.1 m floor; the pursuer's own crash already costs -100). F7c: the three final pursuers (stage 8), clean, evaluated at 0.10 m with per-episode files and re-scored exactly at 0.15 m, vs train pool, test set, the F5c campaign evaders, both F7 campaigns' RL evaders (stage 1 excluded) and **one fresh exploiter per pursuer trained under the 0.15 m rules** (100M frames, seeds 800 / 801 / 802). **0.15 m is the primary radius here**; 0.10 m numbers in parentheses (off the training radius for the F7 pursuers; for F2-plain the 0.15 m numbers are a re-score). [F7_floor_sp/summary.md](F7_floor_sp/summary.md), [results.json](F7_floor_sp/results.json).

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

Contrasts at 0.15 m (0.10 m), 1 seed: floor_r15 - r15: held-out RL -0.107 (-0.150), worst cluster -0.135 (-0.215), other campaigns -0.020 (-0.059), exploitability +0.005 (+0.000); train / scripted / own / X3 within +/- 0.03. r15 - plain: held-out RL -0.025 (-0.136), worst cluster -0.028 (-0.128), other campaigns -0.021 (-0.141), other F7 evaders -0.056 (-0.180), train -0.035 (-0.091).

**F2-plain's 0.15 m exploiter failed to train (checked 11-08).** All three exploiters were set up identically (from scratch, 100M frames, 762 iterations). The F7 exploiters learned to lure within the first ~12 % of training (pursuer crash 0.87 at that point, 0.93-0.95 at the end); F2-plain's never did (pursuer crash <= 0.043 at every logged update, capture 0.89-0.99 from the 21st of 263 updates on). Under the 0.10 m rules all 12 earlier exploiters (F3c, F5c, including F2-plain's, seed 500) learned to lure (pursuer crash > 0.5 after 14-165 of 263-381 logged updates, 0.92-0.98 at the end). Luring F2-plain does not need the evader inside 0.15 m: in 77 % of the episodes in which its 0.10 m exploiter made it crash, the closest distance stayed > 0.15 m (median 0.178 m), so (exact re-score) that exploiter beats F2-plain under the 0.15 m rules in **0.755** of episodes, all by pursuer crash. F2-plain's exploitability in the 0.15 m game is therefore **>= 0.755**, not 0.002; the pre-registered "r15 - plain" exploitability contrast (+0.961) is an artefact of the failed exploiter (<= +0.208 against the best known evader).

Findings:
- **Neither change removes the luring weakness.** Both F7 pursuers are fully exploitable in the game they were trained for, and their exploiters win almost only by pursuer crash. Those exploiters lure from farther away (median closest distance in crash episodes 0.244 m (r15) / 0.233 m (floor_r15), only 2-3 % within 0.15 m; F5c's 0.10 m exploiters 0.142-0.198 m): post-hoc reading, a larger capture radius pushes the lure outwards and the pursuer still follows into the floor. Inside both campaigns evader stages 3 / 5 / 7 still crash the stage >= 2 pursuers in 0.91-0.99 (floor_r15) / 0.87-0.98 (r15) of episodes, as at 0.10 m (0.79-0.99), and the next pursuer stage again patches the latest evader (stage 8 vs stage 7: pursuer crash 0.08 / 0.14).
- **The floor penalty makes the pursuer worse on held-out RL evaders, and changes little else** (train pool, held-out scripted, own campaign and X3 exploiters within +/- 0.03). Post-hoc reading: the penalty adds at most 1 per step near the floor next to the -100 crash penalty the pursuer already has; it adds shaping, not a new incentive against crashing, and the evaders still find a lure.
- **Training at 0.15 m does not help the pursuer**: r15 is at or below F2-plain on every metric at 0.15 m, and lower at 0.10 m (-0.05 to -0.33), because it learned to close only to 0.15 m.
- **Cross-play:** the F7 evaders lure F2-plain less than they lure the F7 pursuers. At 0.15 m F2-plain crashes in 0.124 (floor_r15 evaders) / 0.079 (r15 evaders) of episodes; each F7 pursuer crashes in 0.161 / 0.157 of episodes against the other F7 campaign's evaders, 0.035 / 0.058 against its own.
- **Predictions (pre-registered 11-06 19:30; J at 0.15 m):**
  - (1) floor_r15 pursuer crash vs its exploiter < 0.50 and exploitability <= r15 - 0.10: **did not hold** (0.970; 0.968 vs 0.963).
  - (2) r15's exploiter wins mainly by pursuer crash: **held** (pursuer crash 0.965, exploitability 0.963). r15 exploitability within +/- 0.10 of F2-plain's: **did not hold as measured** (0.963 vs 0.002), but that measurement is invalid (failed exploiter); with the best known evader F2-plain is >= 0.755, so this part is **undecided** (it holds if F2-plain's true value is >= 0.863).
  - (3) floor_r15 pursuer crash vs held-out scripted < 0.05: **did not hold** (0.089; 0.213 at 0.10 m).
  - (4) floor_r15 held-out RL >= r15 - 0.05: **did not hold** (0.827 vs 0.934).
  - (5) In floor_r15, evader stages 3 / 5 / 7 make the stage >= 2 pursuers crash in < 0.50: **did not hold** (0.91-0.99).
- **Decision rule (pre-registered):** (1) did not hold, so the floor penalty + 0.15 m package is not recommended. r15 does not lower exploitability either (0.963 vs >= 0.755 for F2-plain), so neither is the radius alone. **The rules stay** (0.10 m capture radius in training, no floor penalty; J still reported at both radii). This does not depend on the failed F2-plain exploiter. Freeze and F5 Exp 2 / F6 redesign: the user decides.
- Limitations: 1 campaign per arm and 1 exploiter per pursuer, one of which failed (no second exploiter seed for F2-plain); one floor-penalty weight (1.0); F2-plain was trained at 0.10 m, so the r15 - plain contrast mixes the training radius with the evaluation radius; in-campaign evals are vs each stage's own training opponents only.

<a id="f6b"></a>

## F6b_div_sp / F6bc (redesigned diversity term; campaigns done 2026-11-09 15:14; F6bc done 16:09; written up 2026-11-10 14:50; 2 seeds, descriptive)

**Question:** with F6's diversity term redesigned so that it neither saturates nor pays a survival bonus, are the evaders more diverse while the pursuer stays as good as under Hedge?

**Setup** (pre-registered 11-08 15:40; amendment 1, 11-09 01:30): arm **hedge_divb** = F6 `hedge_div` (F3 `hedge` settings, Hedge on both roles, 9 stages x 300M, seeds 0 / 100) with only the term changed: d = RMS action-mean distance to the nearest earlier RL evader (action units, no division by sigma), bonus = 1 - exp(-d / tau), zero-mean (batch-centred) intrinsic reward, plus the bonus in the loss. The action-space distance and smooth bonus remove F6's hard cap plateau; batch-centering removes the positive mean intrinsic reward that acted as a survival bonus in F6.
- Pilot (training data only): tau = 0.0727 (median distance over the last quarter of the beta = 0 run; amendment 1, after v1's last-10 mean caught an end-of-run instability, 0.653). Guard reference = the stricter of the beta = 0 run and F3 hedge-both-s0's stage-5 evader (J 0.184, timeout 0.134). Only **beta = 0.03** passed the timeout guard (timeout 0.217; beta 0.1: 0.237, beta 0.3: 0.268, limit 0.234); none passed the "active" test, so the largest kept beta was chosen (pre-registered fallback). Beta 0.1 missed the cutoff narrowly (0.23750 vs the 0.23369 limit, 0.00381 over); the rule has no uncertainty margin, so the selected beta is sensitive to this single pilot estimate. [pilot/choice.json](F6b_div_sp/pilot/choice.json). The v1 campaigns (tau 0.653) were stopped at stage 5 and are not analysed.
- F6bc = F6c (train pool, test set, F3 + F6b campaign evaders, fresh exploiter seeds 650 / 651, F3 pursuers as probes for the evader Vendi in F3's frozen space). [F6b_div_sp/summary.md](F6b_div_sp/summary.md), `vs_f6.json`; 0.15 m (re-score done 2026-11-10): [R15_rescore/F6b_div_sp/summary.md](R15_rescore/F6b_div_sp/summary.md) (F3c, F6c and F6bc re-analysed together).

| final pursuer, J at 0.10 m (0.15 m) | hedge_divb | hedge | uniform | hedge_div (F6) |
|---|---|---|---|---|
| held-out RL (v10) | 0.708 [0.597, 0.820] (0.917) | 0.836 (0.974) | 0.736 (0.952) | 0.575 (0.810) |
| held-out worst cluster | 0.563 [0.406, 0.721] (0.831) | 0.754 (0.897) | 0.593 (0.880) | 0.467 (0.708) |
| other campaigns' evaders (F3, stage 1 excl.) | 0.886 [0.882, 0.891] (0.954) | 0.888 (0.965) | 0.852 (0.940) | 0.417 (0.721) |
| own campaign's evaders | 0.970 (0.996) | 0.959 (0.994) | 0.936 (0.979) | 0.956 (0.991) |
| thesis train pool | 0.574 (0.737) | 0.617 (0.784) | 0.570 (0.715) | 0.426 (0.650) |
| held-out scripted | 0.805 (0.904) | 0.831 (0.920) | 0.795 (0.901) | 0.827 (0.941) |
| X3 exploiters | 0.954 (0.985) | 0.964 (0.990) | 0.951 (0.986) | 0.948 (0.991) |
| exploitability | 0.983 (0.741) | 0.979 (0.814) | 0.973 (0.752) | 0.979 (0.686) |
| evader Vendi (max 4) | **2.365** [2.335, 2.395] (2.359) | 2.499 (2.475) | 2.456 (2.449) | 3.253 |

Seed-paired differences (2 pairs [range]), at 0.10 m (0.15 m):

| metric | hedge_divb - hedge | hedge_divb - uniform | hedge_divb - hedge_div (F6) |
|---|---|---|---|
| held-out RL | -0.128 [-0.285, +0.029] (-0.057) | -0.028 (-0.035) | +0.134 [-0.134, +0.402] (+0.107) |
| worst held-out cluster | -0.190 [-0.410, +0.029] (-0.067) | -0.030 (-0.049) | +0.097 (+0.123) |
| other campaigns' evaders | -0.002 [-0.015, +0.012] (-0.011) | +0.034 (+0.014) | **+0.469 [+0.341, +0.597]** (+0.232) |
| train pool | -0.044 (-0.048) | +0.003 (+0.021) | +0.148 (+0.087) |
| exploitability | +0.003 (-0.073) | +0.010 (-0.010) | +0.004 (+0.055) |
| evader Vendi | **-0.133** [-0.148, -0.119] | -0.091 | -0.888 |

In-campaign pursuer crash caused by evader stages 3 / 5 / 7 in the stage >= 2 pursuers (each stage's own training opponents, 0.10 m): hedge_divb 0.97 / 0.98 / 0.98 (seed 0), 0.95 / 0.97 / 0.97 (seed 1); hedge 0.97 / 0.98 / 0.98, 0.95 / 0.98 / 0.97; uniform 0.97 / 0.98 / 0.97, 0.95 / 0.89 / 0.96; F6 hedge_div 0.33 / 0.33 / 0.27, 0.22 / 0.33 / 0.37. Term at the end of evader stages 3 / 5 / 7 (last 10 updates): distance 0.191 / 0.076 / 0.078 (seed 0), 0.162 / 0.083 / 0.092 (seed 1) vs tau 0.073; bonus 0.57-0.82.

Registered mechanism check, unrounded (last-10-update mean distance / bonus at evader stages 3 / 5 / 7): seed 0 0.19131 / 0.81538, 0.07591 / 0.57875, 0.07810 / 0.56771; seed 100 0.16239 / 0.79501, 0.08335 / 0.57924, 0.09173 / 0.60546. All distances exceed tau = 0.07266 and all mean bonuses are below 0.9. The logged batch-mean intrinsic reward was effectively zero (mean across updates about 1e-11; maximum absolute update mean < 2.5e-9), confirming centering was applied. Nuance: the stage-3 `div_sat` diagnostic (fraction of samples with bonus > 0.9) was 0.463 / 0.426 at the endpoint and fell to 0.059-0.146 at stages 5 / 7; the smooth bonus has no zero-gradient cap, but many stage-3 samples were still near its asymptote.

Findings:
- **The redesign fixed F6's failure, but the evaders are not more diverse.** The term stayed active and unsaturated, and the evaders kept luring exactly like Hedge's (later pursuer stages crashed in 0.949-0.991 of episodes, vs 0.22-0.45 for F6's evaders). But evader Vendi is 2.335 / 2.395, below matched Hedge (2.454 / 2.543) on both seeds and close to QD-DPP (2.350). At the only weight that kept the evaders winning the same way (beta 0.03), the term moved the policy by little more than one ordinary stage does.
- **The pursuer is like Hedge's against self-play evaders, mixed on held-out evaders.** Other campaigns' evaders -0.002 (seed 0 +0.012, seed 1 -0.015), own +0.011, X3 exploiters -0.010; held-out RL / worst cluster +0.029 / +0.029 on seed 0 and -0.285 / -0.410 on seed 1 [seed labels corrected 2026-11-10; the 11-09 text had them swapped]. Against the F3 evaders it crashes in 0.131 / 0.111 of episodes (F3 hedge 0.144 / 0.105), and its exploiters win by pursuer crash (0.986 / 0.981).
- **Compared with F6, the lure-resistance is back** (other campaigns' evaders +0.469, both seeds), but Vendi fell by 0.888: removing the saturation did not deliver the intended quality-diversity trade-off.
- The F6b evaders are about as hard for the 8 F3 pursuers as the F3 evaders (J 0.860, pursuer crash 0.153; F3 vs other F3 campaigns 0.811).
- At 0.15 m the deficits vs Hedge shrink (table above); exploitability 0.741 vs 0.814 (lower bounds; exploiters trained for 0.10 m).
- **Predictions (pre-registered 11-08 15:40):**
  - (1) Mechanism (bonus <= 0.9 and distance >= tau at the end of stages 3 / 5 / 7): **held**.
  - (2) Win mode kept (stages 3 / 5 / 7 crash the stage >= 2 pursuers in >= 0.79): **held** (0.95-0.98).
  - (3) Diversity (Vendi > hedge on both seeds, >= +0.2): **did not hold** (-0.133).
  - (4) Pursuer: hedge_divb - hedge >= -0.05 on held-out RL and worst cluster: **did not hold** on the mean (-0.128 / -0.190; seed 0 +0.029, seed 1 -0.285); >= 0 on other campaigns' evaders: **did not hold**, narrowly (-0.002); hedge_divb - hedge_div > 0 on other campaigns' evaders on both seeds: **held** (+0.341 / +0.597).
  - (5) Exploitability unchanged: **held** (0.983).
  - **Success = (3) and (4): not met.** Pre-registered consequence: RQ1 is answered negatively (diversity in sampling, D1 / F3, and in training, F6 / F6b, does not beat Hedge); Hedge is the sampler of the final setup.
- Limitations:
  - 2 seeds; the pilot's weight choice depended on amendment 1's guard reference (with the original reference, the rule would have chosen beta 0.1); one weight tested in campaigns.
  - Vendi in F3's frozen descriptor space; one exploiter per pursuer.

<a id="f8"></a>

## F8_roncero_sp / F8c (Roncero et al. 2026's rules; re-run with the k_sm fix: campaigns done 2026-11-11 03:39, F8c done 05:36; written up 2026-11-11 11:30; 2 seeds, descriptive)

**Question:** do evaders still beat self-play pursuers by luring them into the floor when the game uses Roncero et al.'s reward and episode endings? (Is the weakness created by the thesis rules, or is it in the pursuer?)

**Superseded first run (11-10), withdrawn.** It had two bugs ([AUDIT_2026-11-10.md](AUDIT_2026-11-10.md)): Roncero's k_sm smoothness term was 0 in the reward (computed once per physics substep), and F8c evaluated pursuer stage 8 instead of 24 (stage-capped policy loader). Its campaigns stay on disk as a "Roncero without smoothness" variant (in-campaign lure rates also 0.00); outputs in `F8_roncero_sp/archive_stage8_pursuer_20261110/` and `archive_ksm0_roncero_exploiters_20261110/`. Everything below is from the re-run (amendment 2, 11-10 19:25).

**Setup** (pre-registered 11-09 12:21; amendment 1, 12:28; amendment 2, 11-10 19:25):
- Campaigns `thesis-F8b-roncero-s{0,100}`: our Crazyflie, observations, 4 x 4 x 2.9 m arena and 0.10 m capture radius; Hedge on both roles (gamma 0.7, eta 2); **26 stages x 50M frames = Roncero's 13 generations** (650M frames per agent).
- `reward_mode=roncero` = Roncero v3 Eqs. (8)-(11) with his coefficients: shared regularisation (body rate k_br, action smoothness k_sm, -10 for the agent's own out-of-bounds: own crash or leaving the arena, both agents); pursuer capture bonus 10 x beta(closing speed > 2 m/s) + approach 0.25 x distance closed; evader +0.0025 per step and -10 if captured, **nothing when the pursuer fails**. **Capture is terminal; crashes, leaving the arena and the timeout are truncations** (the value target bootstraps; PPO `gae_cut_at_truncation`). Not copied: his drone, arena, domain randomisation.
- F8c: both final pursuers (stage 24) evaluated under the thesis rules vs train pool, test set, the F3 campaigns' evaders and all F8 campaigns' evaders (13 stages each), plus two fresh 100M-frame exploiters each: one under the **thesis rules** (a pursuer crash ends the game at 0 for the evader, so luring pays) and one under **Roncero's rules**. The F3 hedge-both s0 / s1 final pursuers get a Roncero-rules exploiter too (their thesis-rules exploiters are F3c's). [F8_roncero_sp/summary.md](F8_roncero_sp/summary.md), `results.json`.
- Comparison: roncero/k vs F3 hedge/k (same seed; thesis rules, 9 x 300M, no GAE truncation fix). Rules, capture-bonus scaling, schedule and the GAE fix all differ together; F9 (below) shows that the schedule and GAE fix alone already change the result a lot.

| final pursuer, J at 0.10 m (0.15 m) | roncero s0 | roncero s1 | hedge s0 | hedge s1 |
|---|---|---|---|---|
| held-out RL (v10) | 0.948 (0.995) | 0.941 (0.996) | 0.791 (0.960) | 0.882 (0.989) |
| held-out worst cluster | 0.917 (0.993) | 0.893 (0.992) | 0.692 (0.874) | 0.816 (0.921) |
| F3 campaigns' evaders (stage 1 excl.) | 0.887 (0.975) | 0.877 (0.967) | 0.871 (0.957) | 0.906 (0.973) |
| F8 campaigns' evaders (roncero: the other campaign's; stage 1 excl.) | 0.877 (0.950) | 0.935 (0.987) | 0.826 (0.959) | 0.874 (0.979) |
| own campaign's evaders (roncero: 12, hedge: 3) | 0.916 (0.939) | 0.940 (0.971) | 0.957 (0.993) | 0.961 (0.995) |
| thesis train pool | 0.877 (0.963) | 0.902 (0.974) | 0.594 (0.759) | 0.640 (0.810) |
| held-out scripted | 0.983 (0.997) | 0.990 (0.999) | 0.808 (0.905) | 0.853 (0.936) |
| pursuer crash vs held-out scripted | 0.018 (0.003) | 0.011 (0.002) | 0.197 (0.096) | 0.149 (0.065) |
| X3 exploiters | 0.999 (1.000) | 0.999 (1.000) | 0.955 (0.987) | 0.973 (0.992) |
| exploitability vs thesis-rules exploiter | 0.631 (0.000) | 0.513 (0.026) | 0.974 (0.773) | 0.985 (0.855) |
| exploitability vs Roncero-rules exploiter | 0.512 (0.064) | 0.755 (0.331) | 0.816 (0.683) | 0.939 (0.825) |

How the exploiters win (fractions of episodes at 0.10 m; the thesis-rules exploiter of hedge s0 / s1 is F3c's; leaving the arena is non-terminal under the thesis rules and ends the episode under Roncero's):

| pursuer | exploiter | capture | pursuer crash | pursuer leaves arena | timeout | evader crash / leaves |
|---|---|---|---|---|---|---|
| roncero s0 / s1 | thesis rules | 0.304 / 0.441 | **0.010 / 0.030** | 0.054 / 0.012 | 0.622 / 0.484 | 0.066 / 0.046 |
| roncero s0 / s1 | Roncero rules | 0.032 / 0.077 | 0.001 / 0.006 | 0.289 / 0.094 | 0.241 / 0.658 | 0.456 / 0.169 |
| hedge s0 / s1 | thesis rules (F3c) | | **0.975 / 0.988** | | | |
| hedge s0 / s1 | Roncero rules | 0.040 / 0.028 | 0.106 / 0.023 | **0.718 / 0.915** | 0.001 / 0.004 | 0.144 / 0.032 |

In-campaign pursuer crash caused by each evader stage in the stage >= 2 pursuers (its own training opponents, under Roncero's rules, 0.10 m): evader stages 3 / 5 / 7: **0.00 / 0.00 / 0.00** on both seeds (F3 hedge under the thesis rules: 0.97 / 0.98 / 0.98 and 0.95 / 0.98 / 0.97). Later evader stages 9-25: per-stage means 0.00-0.05 on both seeds (at most 0.18 against one pursuer in s0, 0.26 in s1).

Findings:
- **Under Roncero's rules the floor-lure exploit disappears** (as in the first run): no evader stage learns to lure, and the final pursuers are not lured **even by an exploiter trained under the thesis rules, where luring pays** (pursuer crash 0.010 / 0.030; F3 hedge pursuers 0.975 / 0.988).
- **The final F8 pursuer is better than F3 Hedge's on every opponent group except its own campaign**: held-out RL +0.157 / +0.060, worst cluster +0.225 / +0.078, train pool +0.283 / +0.262, held-out scripted +0.175 / +0.136 with almost no crashes, X3 exploiters +0.044 / +0.026; F3 campaigns' evaders level (+0.017 / -0.029); own campaign -0.041 / -0.022, but against its 12 own evaders (including stage 25, trained against it) vs Hedge's 3. The first run's "weaker against agile learned evaders" came from the stage-8 pursuer and does not hold for the final one.
- **It is much less exploitable, and what remains is near misses.** Thesis-rules exploiter: -0.342 / -0.472 vs Hedge, winning by timeout (0.622 / 0.484); at 0.15 m exploitability is **0.000 / 0.026** (lower bounds). Against seed 0 the Roncero-rules exploiter wins as much by crashing or leaving itself (0.456) as by timeout.
- **Roncero's rules would not have shown our pursuers' weakness as crashes**: his-rules exploiters beat the F3 hedge pursuers mainly by making them leave the arena (0.718 / 0.915), which the thesis rules never penalised, and rarely by crashing them (0.106 / 0.023).
- At 0.15 m the F8 pursuers are near perfect against every learned and scripted set (table above).
- **Predictions (pre-registered 11-09 12:21; unchanged by amendment 2):**
  - (1) In-campaign lure rate < 0.50 on both seeds: **held** (0.00).
  - (2) H-rules (crash vs the thesis-rules exploiter < 0.50) vs H-pursuer (>= 0.50): **H-rules held** (0.010 / 0.030). The predicted H-pursuer **did not hold**.
  - (3) Under Roncero's rules the F3 hedge pursuers' exploiter wins mostly by timeout, crash share of its wins < 0.5: the crash share **held** (0.13 / 0.02), "mostly by timeout" **did not hold** (it wins by the pursuer leaving the arena).
- **Consequence:** with Roncero's rules (plus the GAE truncation fix and 50M-frame stages) self-play produces a pursuer that is not lurable, beats Hedge's on held-out learned evaders, and is exploitable at 0.10 m only by near-miss timeouts. F9 shows that part of this comes from the GAE fix / schedule rather than from Roncero's rules.
- Limitations: 2 seeds, one exploiter per pursuer and rule set; F8 and F3 differ in rules, closing-speed factor, schedule (26 x 50M vs 9 x 300M) and the GAE truncation fix at once; the F3 pursuers' Roncero-rules evaluation uses rules they were not trained with (leaving the arena ends the game); own- and other-campaign evader sets differ in size between arms; in-campaign evals at the training radius only; our domain randomisation (mass, T/W, inertia, motor time constants; train and eval) and drone (T/W 1.44), not Roncero's.

<a id="f8-posthoc"></a>

### F8 post-hoc: transitivity and evader diversity (added 2026-11-10 17:05, re-run 2026-11-11 11:30 on the F8 re-run; not pre-registered, no new runs)

Question (user): under Roncero's rules, is there room for diversity-aware training (QD-DPP / PSD-PSRO style), which targets non-transitive games with several distinct strategies to cover? Same measures as the F3 post-hoc transitivity; in-campaign evals under each campaign's own rules at 0.10 m. [F8_roncero_sp/transitivity.md](F8_roncero_sp/transitivity.md) (matrices included), `transitivity.json`, `exps/f8_transitivity.py`. (The 11-10 version used the k_sm = 0 campaigns and stage-8 probes; archived with the stage-8 outputs.)

| | F8 s0 | F8 s1 | F3 hedge s0 | F3 hedge s1 |
|---|---|---|---|---|
| within campaign, P2-24 x E1-25: R² (J) / reversals | 0.731 / 0.008 | 0.673 / 0.003 | - | - |
| within campaign, P2-8 x E1-7: R² (J) / reversals | 0.670 / 0.000 | 0.606 / 0.000 | 0.651 / 0.000 | 0.690 / 0.000 |
| new pursuer stage beats every older evader (J >= 0.5) | 11 / 12 | 11 / 12 | 4 / 4 | 4 / 4 |
| newest pursuer, min J vs older evaders | 0.92 | 0.91 | 0.91 | 0.90 |
| new evader stage beats every older pursuer (J < 0.5) | 11 / 12 | 12 / 12 | 3 / 3 | 3 / 3 |
| evader Vendi, shared 4-probe space, E1-7 (max 4) | 1.659 | 1.567 | 2.011 | 2.217 |
| evader Vendi, shared 4-probe space, E19-25 (max 4) | 1.991 | 2.771 | - | - |
| evader Vendi, own-campaign space, E1-7 vs P2-8 (max 4) | 2.462 | 2.356 | 2.401 | 2.384 |
| evader Vendi, own-campaign space, all 13 evaders vs P2-24 | 3.514 | 3.514 | - | - |
| - in that space, first 4 (E1-7) / last 4 (E19-25) | 1.912 / 1.707 | 1.809 / 1.729 | - | - |

Shared space = F3's Vendi construction (J and censored time vs each probe, behaviour mean / std; one RBF kernel over all 32 F3 + 26 F8 RL evaders) with the 4 probes that played both F3 and F8 evaders (F3 hedge s0 / s1 and F8 s0 / s1 final pursuers); not F3's frozen 8-probe space. Vendi grows with the number of evaders, so F8 is reported for 4-evader subsets as F3. In this space the other F3 samplers give uniform 1.97-2.16, QD-DPP 2.00-2.02, PFSP 2.62-2.80. Across campaigns, same 4 probes (thesis rules): F8 evaders stages 3-25 R² 0.619, reversals 0.050; F3 evaders 0.560 / 0.048; all 48 RL evaders 0.362 / 0.064 (0.15 m: reversals 0.000-0.002).

Findings:
- **The F8 game is as transitive as F3 Hedge's**: over the full 26-stage campaigns only 0.3-0.8 % of comparisons reverse; every new pursuer stage beats every older evader except one per campaign (s0: pursuer 18 vs evader 17, J 0.15; s1: pursuer 16 vs evader 15, 0.13).
- **Early F8 evaders are no more diverse than F3's; one seed's late evaders are**: the last four reach 1.99 (s0, hedge-like) and 2.77 (s1, close to PFSP's 2.62-2.80) in the shared space, but not in the own-campaign space. The 13 evaders of a campaign behave like 3.5 distinct ones.
- Across campaigns the few reversals (0.05 at 0.10 m) vanish at 0.15 m: they are near misses.
- Cross-generation win-rate matrices (Roncero Fig. 3 style, added 11-11; `scripts/selfplay/plot_winrate_matrix.py` on each campaign's db, win_rate = pursuer wins / decisive episodes, in-campaign evals under Roncero's rules, 0.10 m): [s0](F8_roncero_sp/winrate_matrix_s0.png), [s1](F8_roncero_sp/winrate_matrix_s100.png). Generation g = pursuer stage 2(g-1), evader stage 2g-1. Every pursuer generation >= 2 beats every older evader generation (win rate 0.88-1.00) except one cell per seed, its newest opponent (s0: generation 10 vs 9, 0.11; s1: 9 vs 8, 0.10), which the next pursuer generation fixes (0.93 / 0.99); each evader generation beats the pursuer it trained against (pursuer win rate 0.03-0.32) and almost all older ones (exceptions: s0 pursuer generation 2 vs evader generation 6, 0.52; s1 pursuer generation 7 vs evader generation 13, 0.53).
- Interpretation (not tested): removing the lure did not make the game cyclic, so diversity-aware training probably has little room under Roncero's rules either, as under the thesis rules (QD-DPP in F3, F6, F6b did not beat Hedge); seed 1's late spread against the probes is the only hint of more variety.
- Limitations: 2 seeds; in-campaign evals only at the training radius and only between stages of one campaign; the shared space has 4 probes; the Vendi descriptors include J, so part of the spread is strength, not style.

<a id="f8d"></a>

### F8d: stronger held-out test and F8 learning curve (pre-registered 2026-11-11 12:21; eval only, done 13:21; written up 13:25; 2 seeds, descriptive)

**Question (user):** v10 no longer separates the strong pursuers (F8 0.94-0.95, everyone >= 0.96 at 0.15 m). Is F8's final pursuer still better than Hedge's against stronger held-out learned evaders, and how early does it overtake?

**Setup:** pursuers = F8 stages 8 / 16 / 24 (250M / 450M / 650M pursuer frames) and the F3 hedge-both final pursuers (stage 8, 1.5B pursuer frames; not matched by stage count, since F8's final pursuer already has fewer frames). Clean evals under the thesis rules vs the test set (F8c / F3c evals reused for F8 stage 24 / hedge) and vs the **final evader of 20 campaigns**: a common "strong" set of 16 none of these pursuers trained with (F2-plain, F3 uniform-s1, qd_dpp x2, pfsp x2, F6 x2, F6b x2, F7 r15 / floor_r15, F9 x4) plus the F3 hedge and F8 finals (own campaign excluded). [F8d_heldout/summary.md](F8d_heldout/summary.md) (per-evader tables), `results.json`, `exps/f8d_heldout.py`.

| J at 0.10 m (0.15 m) | F8 s0 P8 | F8 s0 P16 | F8 s0 P24 | F8 s1 P8 | F8 s1 P16 | F8 s1 P24 | hedge s0 | hedge s1 |
|---|---|---|---|---|---|---|---|---|
| held-out RL (v10) | 0.889 (0.977) | 0.960 (0.998) | 0.948 (0.995) | 0.925 (0.992) | 0.968 (0.995) | 0.941 (0.996) | 0.791 (0.960) | 0.882 (0.989) |
| held-out worst cluster | 0.828 (0.949) | 0.938 (0.997) | 0.917 (0.993) | 0.871 (0.982) | 0.961 (0.994) | 0.893 (0.992) | 0.692 (0.874) | 0.816 (0.921) |
| **strong finals, mean (16)** | 0.753 (0.898) | 0.864 (0.966) | **0.858** (0.959) | 0.849 (0.937) | 0.861 (0.955) | **0.875** (0.955) | **0.669** (0.879) | **0.749** (0.927) |
| strong finals, worst | 0.179 (0.471) | 0.662 (0.915) | 0.653 (0.898) | 0.482 (0.604) | 0.620 (0.833) | 0.603 (0.820) | 0.326 (0.671) | 0.446 (0.795) |
| strong finals, pursuer crash | 0.100 (0.028) | 0.139 (0.037) | 0.152 (0.042) | 0.078 (0.026) | 0.147 (0.047) | 0.142 (0.050) | 0.357 (0.123) | 0.275 (0.076) |
| F3 hedge finals | 0.899 (0.979) | 0.845 (0.945) | 0.867 (0.952) | 0.947 (0.991) | 0.866 (0.953) | 0.870 (0.935) | 0.740 (0.929) | 0.846 (0.958) |
| F8 finals (other seed for F8) | 0.553 (0.582) | 0.222 (0.231) | 0.863 (0.953) | 0.190 (0.469) | 0.780 (0.981) | 0.864 (0.976) | 0.783 (0.945) | 0.817 (0.959) |

Findings:
- **The strong set separates the pursuers where v10 does not**: mean J on it is 0.07-0.13 below v10 for every final pursuer, and the worst final evader (F7 floor_r15's, on both seeds) holds Hedge to 0.326 / 0.446.
- **F8's final pursuer beats Hedge's on the strong set: +0.189 / +0.126** (at 0.15 m +0.080 / +0.028), with a much higher worst case and about half the crash rate. The gain is largest against evaders that lure or press hard: the PFSP finals (0.83-0.94 vs 0.38-0.60), F6's (0.93-0.97 vs 0.53-0.75) and F7's (0.80-0.90 vs 0.33-0.68). Against the F3 uniform / qd_dpp, F6b and F9 endings finals the two are level (0.84-0.99 both). The hardest final for F8 is F9 no_timeout's (0.65 / 0.60; Hedge 0.53 / 0.55).
- **F8 overtakes early**: already at stage 8 (250M pursuer frames, 1/6 of Hedge's) it is above Hedge's final pursuer on v10 and on the strong set, with one weak spot on s0 (F7 floor_r15's final 0.18). By stage 16 it is at its final level (v10 peaks at stage 16).
- **F8's own late evaders stay hard for earlier F8 pursuers** (stages 8 / 16 vs the other seed's final evader: 0.19-0.78) but not for its final pursuers, consistent with the staircase.
- **Predictions (pre-registered 12:21):**
  - (1) strong set harder than v10 for all four final pursuers: **held** (by 0.07-0.13).
  - (2) F8 final >= Hedge final + 0.05 on the strong set, both seeds: **held** (+0.189 / +0.126; at 0.15 m only s0 clears 0.05, +0.080 / +0.028).
  - (3) F8 rises from stage 8 to 24 on held-out RL and the strong set, both seeds: **held** (not monotone: held-out RL peaks at stage 16); F8 stage 8 below Hedge's final on held-out RL, both seeds: **did not hold** (0.889 / 0.925 vs 0.791 / 0.882).
- Limitations: 2 seeds; one final evader per campaign; the strong set is 16 evaders from our own campaigns (all with thesis or near-thesis rules), not an external lineage; F8 vs F3 still differ in rules, schedule and the GAE fix (F9, F10); eval at the training radius only for the 0.10 m column.

<a id="f9"></a>

## F9_ablation_sp / F9c (one Roncero component at a time on the thesis rules; campaigns done 2026-11-10 19:31 (control, no_timeout) and 2026-11-11 09:28 (beta, endings); F9c done 10:58; written up 2026-11-11 11:30; 1 seed, descriptive)

**Question:** which part of Roncero et al.'s rules removes the floor-lure exploit that F8 removed?

**Setup** (pre-registered 11-10 15:19, before any F9 run; user request after F8):
- Four campaigns `thesis-F9-{control,no_timeout,beta,endings}-s0`, seed 0, our drone / observations / arena, 0.10 m radius, Hedge on both roles, **13 stages x 50M**, all with the GAE truncation fix (`algo.gae_cut_at_truncation`).
- **control** = thesis rules. **no_timeout** = the timeout gives 0 to both roles (thesis: pursuer -100, evader +100). **beta** = the pursuer's +100 capture reward x Roncero's closing-speed factor exp(-3 max(0, v_close - 2 m/s)). **endings** = every terminal ending except capture (crash of either agent, evader leaving) becomes a truncation (value bootstrapped), rewards unchanged. Not tested: Roncero's weak distance shaping and reward magnitudes.
- F9c: each final pursuer (stage 12) evaluated clean under the thesis rules vs train pool, test set, the 8 F3 campaigns' and the F9 campaigns' evaders (stage 1 excluded), and **one fresh 100M-frame thesis-rules exploiter** (luring pays). Measures: (a) in-campaign lure rate (pursuer crash caused by evader stages 3-11 in the stage >= 2 pursuers; mean over stages of each stage's mean); (b) pursuer crash vs the exploiter. "Lure present" = >= 0.50.
- Deviations: F9c uses the stage-loader fix of 11-10 17:05 (made before F9c ran), so it tests stage 12 and all F9 evaders, as pre-registered; beta / endings ran after the F8 re-run (user), with unchanged code for these arms (the k_sm fix touches only Roncero mode). [F9_ablation_sp/summary.md](F9_ablation_sp/summary.md), `results.json`.

| final pursuer (stage 12), J at 0.10 m (0.15 m) | control | no_timeout | beta | endings | F8 roncero s0 | F3 hedge s0 |
|---|---|---|---|---|---|---|
| held-out RL (v10) | 0.893 (0.989) | 0.916 (0.995) | 0.941 (0.996) | 0.890 (0.992) | 0.948 (0.995) | 0.791 (0.960) |
| held-out worst cluster | 0.859 (0.973) | 0.884 (0.965) | 0.914 (0.992) | 0.849 (0.989) | 0.917 (0.993) | 0.692 (0.874) |
| F3 campaigns' evaders | 0.766 (0.948) | 0.845 (0.970) | 0.836 (0.965) | 0.702 (0.924) | 0.887 (0.975) | 0.871 (0.957) |
| other F9 campaigns' evaders | 0.866 (0.972) | 0.887 (0.980) | 0.879 (0.975) | 0.714 (0.901) | - | - |
| own campaign's evaders | 0.933 (0.995) | 0.872 (0.981) | 0.959 (0.997) | 0.993 (1.000) | 0.916 (0.939) | 0.957 (0.993) |
| thesis train pool | 0.688 (0.840) | 0.718 (0.847) | 0.859 (0.959) | 0.756 (0.928) | 0.877 (0.963) | 0.594 (0.759) |
| held-out scripted | 0.918 (0.978) | 0.917 (0.978) | 0.969 (0.995) | 0.976 (0.995) | 0.983 (0.997) | 0.808 (0.905) |
| pursuer crash vs held-out scripted | 0.085 (0.023) | 0.085 (0.022) | 0.033 (0.005) | 0.025 (0.005) | 0.018 (0.003) | 0.197 (0.096) |
| X3 exploiters | 0.983 (0.998) | 0.983 (0.999) | 0.996 (1.000) | 0.999 (1.000) | 0.999 (1.000) | 0.955 (0.987) |
| exploitability (thesis-rules exploiter) | 0.440 (0.095) | **0.950** (0.571) | 0.370 (0.085) | **0.823** (0.127) | 0.631 (0.000) | 0.974 (0.773) |

How the thesis-rules exploiter wins (fractions of episodes, 0.10 m; leaving the arena is non-terminal here):

| arm | capture | pursuer crash (b) | timeout | evader crash / leaves | pursuer leaves arena |
|---|---|---|---|---|---|
| control | 0.318 | 0.056 | 0.385 | 0.243 | 0.682 |
| no_timeout | 0.036 | **0.955** | 0.000 | 0.015 | 0.053 |
| beta | 0.401 | 0.395 | 0.000 | 0.237 | 0.051 |
| endings | 0.130 | **0.834** | 0.000 | 0.047 | 0.483 |

In-campaign lure rate (a) per evader stage (mean over the stage >= 2 pursuers):

| arm | E3 | E5 | E7 | E9 | E11 | (a) mean |
|---|---|---|---|---|---|---|
| control | 0.09 | 0.41 | 0.37 | 0.42 | 0.53 | 0.36 |
| no_timeout | 0.67 | 0.82 | 0.75 | 0.73 | 0.79 | **0.75** |
| beta | 0.03 | 0.04 | 0.20 | 0.28 | 0.33 | 0.18 |
| endings | 0.00 | 0.02 | 0.05 | 0.04 | 0.03 | 0.03 |
| F3 hedge s0 (9 x 300M, no GAE fix) | 0.97 | 0.98 | 0.98 | | | |

Findings:
- **The control arm is the main surprise: the thesis rules alone, with the GAE truncation fix and 13 x 50M stages, give a pursuer that is hardly lurable.** In-campaign lure 0.36 (rising from 0.09 to 0.53 over the stages; F3 hedge 0.97-0.98), and the thesis-rules exploiter makes the final pursuer crash in only 0.056 of episodes (F3 hedge 0.975). It wins by timeout, mostly near misses (exploitability 0.440 -> 0.095 at 0.15 m), and drags the pursuer out of the arena in 0.682 of episodes (allowed under the thesis rules). Control differs from F3 hedge in two ways at once, the GAE fix and the schedule (50M vs 300M frames per stage, 13 vs 9 stages), and F9 cannot say which matters.
- **no_timeout restores F3-level luring on both measures** (lure 0.67-0.82 at every evader stage, exploiter crash 0.955).
- **endings stops evaders from learning to lure but leaves the pursuer lurable**: lure 0.00-0.05 at every stage, yet the exploiter makes the pursuer crash in 0.834 of episodes (0.127 exploitability at 0.15 m, so the crashes are dives that pass within 0.15 m). It is also the weakest against other campaigns' evaders. As with F6's lure-free pursuers, a pursuer that never meets lurers in training does not learn to resist them.
- **beta lowers in-campaign luring (0.18) and is best on easy opponents** (train +0.171, held-out RL +0.049, fewer crashes vs scripted), but its exploiter crashes it more than control's does (0.395 vs 0.056); exploitability is similar because control's exploiter wins by timeout instead.
- At 0.15 m every arm except no_timeout is near perfect on held-out evaders (0.989-0.996), with exploitability 0.085-0.127.
- **Predictions (pre-registered 11-10 15:19):**
  - (1) control: (a) and (b) present: **did not hold** ((a) 0.36, (b) 0.056).
  - (2) endings: (a) absent, (b) present: **held** (0.03; 0.834).
  - (3) no_timeout: (a) present, (b) present (low confidence): **held** (0.75; 0.955).
  - (4) beta: (a) present, (b) absent: (a) **did not hold** (0.18), (b) **held** (0.395).
- **Use (as registered):** an arm with (b) absent was to identify a pursuer-side cause of F8's lure-resistance. Since the control already has (b) absent, F9 does not isolate one Roncero component: under the F8/F9 schedule with the GAE fix the thesis rules already give a lure-resistant pursuer; removing the timeout terms makes it lurable again; truncation endings remove the evader's incentive, so the pursuer never learns to resist; beta sits in between.
- **Consequence for F8:** the F8 - F3 contrast mixes Roncero's rules with the GAE fix and the shorter stages. F10 (the 13 x 50M arm without the fix, below) shows the schedule makes the pursuer lure-resistant and the fix mainly reduces evader luring.
- Limitations: 1 seed per arm; one exploiter each; 13 stages (F8: 26); in-campaign evals at the training radius only; the "other F9 campaigns' evaders" set excludes a different campaign for each arm.

<a id="f10"></a>

## F10_gae_sp / F10c (F9 control without the GAE truncation fix; campaign done 2026-11-11 14:36, F10c done 15:08; written up 2026-11-11 16:25; 1 seed, descriptive)

**Question:** F9's control pursuer (thesis rules, 13 x 50M, GAE fix) is hardly lurable while F3's hedge pursuer (thesis rules, 9 x 300M, no fix) is fully lurable. Is that the GAE truncation fix or the schedule?

**Setup** (pre-registered 11-11 11:45, before any F10 run; user chose this single cell): campaign `thesis-F10-nofix-s0` = F9 control with `algo.gae_cut_at_truncation` off (worker command line checked: `task_overrides=[]`); seed 0, Hedge both roles, 13 stages x 50M, thesis rules. F10c as F9c: final pursuer (stage 12) clean under the thesis rules vs train pool, test set, F3 campaigns' evaders, the F9 campaigns' + its own evaders, and a fresh 100M-frame thesis-rules exploiter (seed 850). [F10_gae_sp/summary.md](F10_gae_sp/summary.md), `results.json`, `exps/f10_gae_sp.py`. Decision on (b): H-fix = (b) >= 0.50, H-schedule = (b) < 0.50.

| final pursuer (stage 12), J at 0.10 m (0.15 m) | F10: no fix | F9 control: fix | no fix - fix | F3 hedge s0 (9 x 300M, no fix) |
|---|---|---|---|---|
| held-out RL (v10) | 0.933 (0.996) | 0.893 (0.989) | +0.040 (+0.007) | 0.791 (0.960) |
| held-out worst cluster | 0.890 (0.991) | 0.859 (0.973) | +0.031 (+0.018) | 0.692 (0.874) |
| F3 campaigns' evaders | 0.826 (0.957) | 0.766 (0.948) | +0.060 (+0.009) | 0.871 (0.957) |
| F9 campaigns' evaders (control: the other three) | 0.890 (0.982) | 0.866 (0.972) | +0.024 (+0.011) | - |
| own campaign's evaders | 0.813 (0.964) | 0.933 (0.995) | -0.120 (-0.031) | 0.957 (0.993) |
| thesis train pool | 0.838 (0.949) | 0.688 (0.840) | +0.150 (+0.109) | 0.594 (0.759) |
| held-out scripted | 0.968 (0.998) | 0.918 (0.978) | +0.049 (+0.020) | 0.808 (0.905) |
| pursuer crash vs held-out scripted | 0.033 (0.002) | 0.085 (0.023) | -0.052 | 0.197 (0.096) |
| X3 exploiters | 0.993 (1.000) | 0.983 (0.998) | +0.010 | 0.955 (0.987) |
| exploitability (thesis-rules exploiter) | **0.172** (0.004) | 0.440 (0.095) | -0.268 | 0.974 (0.773) |
| (b) pursuer crash vs that exploiter | **0.031** (0.004) | 0.056 (0.008) | -0.025 | 0.975 (0.774) |
| exploiter wins by timeout | 0.143 (0.000) | 0.385 (0.086) | -0.243 | - |

(a) In-campaign lure rate per evader stage 3 / 5 / 7 / 9 / 11: no fix **0.75 / 0.85 / 0.88 / 0.80 / 0.75, mean 0.81**; F9 control 0.09 / 0.41 / 0.37 / 0.42 / 0.53, mean 0.36; F3 hedge s0 (stages 3 / 5 / 7) 0.97 / 0.98 / 0.98.

Findings:
- **The pursuer's lure-resistance comes from the schedule, not the GAE fix.** Without the fix the 13 x 50M pursuer is still not lurable by a fresh exploiter (F3 hedge 0.975) and is even less exploitable than with it; the exploiter mostly fails (capture 0.651).
- **The GAE fix changes the evaders instead**: without it, evaders learn to lure inside the campaign at F3-like rates (0.75-0.88 per stage). Under the thesis rules the only truncation is the timeout, the evader's win; the old GAE recursion leaked the next episode's advantages into the last steps of every timed-out episode, which plausibly blurs the credit for surviving and makes luring relatively more attractive (not tested).
- **So luring by in-campaign evaders does not make the pursuer lurable.** F10's pursuer met lurers at every stage and ended up lure-resistant; F3's, with the same rules and no fix but 300M-frame stages, stayed lurable. What differs is the schedule: 13 short stages (the pursuer re-trains every 50M frames against a pool that keeps adding lurers) versus 9 long ones (Hedge pursuer stages of 300M frames). Interpretation, not tested: with long stages each pursuer may over-fit to its training pool's current mix and each evader has 300M frames to find the dive-and-miss exploit; F7 and F6b showed that pool composition alone does not fix it.
- F10's pursuer is the least exploitable self-play pursuer under the thesis rules so far (next: F2c's acceleration-input pursuer, 0.327), but weaker against its own campaign's lurers.
- **Predictions (pre-registered 11:45):** (b) **H-schedule held** (0.031 < 0.50; predicted, low confidence). "(a) within +/- 0.15 of control's 0.36" **did not hold** (0.81).
- **Consequence:** F3-F7's ~0.97 exploitability is mainly a schedule effect (9 x 300M), not the GAE leak; the GAE leak raises in-campaign luring but not final-pursuer lurability. The thesis rules + a 13 x 50M schedule already give a pursuer about as robust as F8's (exploitability 0.172 vs F8 0.631 / 0.513 at 0.10 m; held-out RL 0.933 vs 0.948 / 0.941; 1 seed vs 2).
- Limitations: 1 seed; one exploiter; the schedule effect is inferred from F10 vs F3 hedge, which also differ in stage count (13 vs 9) and total frames (650M vs 1.5B pursuer frames), so "schedule" covers stage length, stage count and total budget together; F10 was not in F8d's strong set.

<a id="f11"></a>

## F11_v2_sp / F11c (thesis v2d rules; campaigns done 2026-11-12 20:06, F11c done 21:33; written up 2026-11-13 00:10; 2 seeds, descriptive)

**Question:** do revised thesis rules ('thesis v2') give self-play pursuers that are not lurable, generalise at least as well as F8 / F10 and are hard to exploit?

**Setup** (pre-registered 2026-11-11 17:06; amendments 1-4 in the plan; v2a, v2b and v2c stopped or failed their gates before any campaign result):
- **v2d rules** = the thesis rules with: every ending except capture (pursuer crash, evader crash, evader leaving) a truncation, rewards unchanged; **no evader timeout bonus** (the pursuer keeps -100 at timeout); plain-difference shaping approach 7 / precision 10; smoothness 0.5, command rate 0.2; no closing-speed factor; the pursuer may leave the arena freely; GAE truncation fix.
- Campaigns `thesis-F11v2d-s{0,100}`: Hedge both roles, 26 stages x 50M, our drone / arena / 0.10 m radius. Gate (stage-0 pursuer capture vs the 4 scripted training evaders >= 0.80): 0.876. Pilot `thesis-F11v2d-pilot` (13 x 50M): J vs scripted 0.989, settings kept ([F11_v2_sp/pilot_decision_v2d.json](F11_v2_sp/pilot_decision_v2d.json)).
- F11c: final pursuers (stage 24) evaluated clean under the thesis rules vs train pool, test set, F3 campaigns' evaders, F8d's 16 strong finals, the F3 hedge / F8 finals, both F11 campaigns' evaders, and two fresh 100M-frame exploiters each: one under the **v2d rules** (seeds 860 / 861) and one under the **thesis rules** without the GAE fix (870 / 871; same recipe as F3c / F9c / F10c, luring pays). [F11_v2_sp/summary.md](F11_v2_sp/summary.md), `results.json`, `exps/f11_v2_sp.py`.
- References (not paired): F8 (Roncero's rules, 26 x 50M, 2 seeds) and F10 (thesis rules, 13 x 50M, no GAE fix, 1 seed); their exploiters use the same thesis-rules recipe.

| final pursuer, J at 0.10 m (0.15 m) | v2d s0 | v2d s100 | F8 s0 / s1 | F10 |
|---|---|---|---|---|
| held-out RL (v10) | 0.875 (0.980) | 0.869 (0.987) | 0.948 / 0.941 | 0.933 |
| held-out worst cluster | 0.838 (0.976) | 0.832 (0.982) | 0.917 / 0.893 | 0.890 |
| **strong finals, mean (16)** | 0.845 (0.947) | 0.751 (0.922) | 0.858 / 0.875 | - |
| strong finals, worst | 0.501 (0.789) | 0.346 (0.773) | 0.653 / 0.603 | - |
| F8 finals | 0.868 (0.973) | 0.956 (0.993) | | |
| F3 hedge finals | 0.858 (0.930) | 0.765 (0.908) | 0.867 / 0.870 | |
| F3 campaigns' evaders | 0.892 (0.972) | 0.846 (0.964) | 0.887 / 0.877 | 0.826 |
| other F11 campaign's evaders | 0.906 (0.993) | 0.942 (0.993) | | |
| own campaign's evaders | 0.952 (0.996) | 0.957 (0.988) | 0.916 / 0.940 | 0.813 |
| thesis train pool | 0.833 (0.927) | 0.678 (0.814) | 0.877 / 0.902 | 0.838 |
| held-out scripted | 0.976 (0.996) | 0.971 (0.995) | 0.983 / 0.990 | 0.968 |
| pursuer crash vs held-out scripted | 0.025 (0.004) | 0.030 (0.005) | 0.018 / 0.011 | 0.033 |
| X3 exploiters | 0.998 (1.000) | 0.999 (1.000) | 0.999 / 0.999 | 0.993 |
| exploitability, v2d-rules exploiter | 0.501 (0.156) | 0.140 (0.113) | - | - |
| ... pursuer crash / timeout | 0.016 / 0.486 | 0.028 / 0.113 | | |
| **exploitability, thesis-rules exploiter** | 0.293 (0.031) | 0.624 (0.208) | 0.631 / 0.513 | 0.172 |
| ... **pursuer crash** / timeout | **0.309** / 0.002 | **0.636** / 0.030 | 0.010 / 0.030 (crash) | 0.031 (crash) |

(Outcome fractions come from separate per-episode flags and can overlap, so a crash fraction can exceed 1 - J.)

(a) In-campaign lure rate (pursuer crash caused by each evader stage in the stage >= 2 pursuers), evader stages 3 / 5 / 7 / 9 / 11: s0 0.00 / 0.00 / 0.01 / 0.10 / 0.04, **mean 0.03**; s100 0.00 / 0.00 / 0.01 / 0.32 / 0.41, **mean 0.15**. Later stages 13-25: s0 0.04-0.10; s100 0.20-0.50, mostly against pursuer stages 6 and 8 (crash 0.57-0.97), less against the latest pursuer (stage 24 vs evader stage 25: 0.21). References: F8 0.00, F9 control 0.36, F10 0.81.

Findings:
- **Under its own rules the v2d pursuer is not lured**: the v2d-rules exploiter crashes it in 0.016 / 0.028 of episodes and wins by timeout (0.486 / 0.113).
- **But it stays lurable when luring pays**: the thesis-rules exploiter crashes it in 0.309 / 0.636 of episodes (F8 0.010 / 0.030, F10 0.031). On this measure v2d is worse than both references, on both seeds.
- **It generalises less well than F8 and F10**: held-out RL 0.875 / 0.869 (F8 0.948 / 0.941, F10 0.933); strong finals level with F8 on s0 (0.845 vs 0.858) and 0.124 below on s100 (0.751 vs 0.875); train pool 0.833 / 0.678. At 0.15 m every learned evader set is >= 0.90 except the train pool of s100 (0.814).
- **The seeds differ**: s100's evaders learn to lure from stage 9 on, and its pursuer is weaker on the train pool and the strong finals and more lurable by the thesis-rules exploiter (0.636 vs 0.309), yet less exploitable under v2d rules (0.140 vs 0.501).
- Why v2d pursuers stay lurable while F10's (thesis rules with timeout terms, 13 x 50M) are not is not tested; F10 also differs in stage count (13 vs 26), the GAE fix and the evader's timeout bonus.
- **Predictions (pre-registered 11-11 17:06):**
  - (1) lure rate (a) < 0.10 on both seeds: **did not hold** (0.03 / 0.15).
  - (2) pursuer crash vs the thesis-rules exploiter < 0.10: **did not hold** (0.309 / 0.636).
  - (3) exploitability vs the thesis-rules exploiter <= 0.172: **did not hold** (0.293 / 0.624).
  - (4) held-out RL >= 0.93: **did not hold** (0.875 / 0.869).
  - (5) strong-finals mean >= F8's - 0.05 (0.808 / 0.825): **held on s0** (0.845), **not on s100** (0.751).
- **Consequence:** v2d does not improve on F10's thesis rules + 13 x 50M schedule or on F8's Roncero rules on any pre-registered measure. F12 (learned acceleration input) runs on v2d as pre-registered, so its comparison is with F11, not with F8 / F10.
- Limitations: 2 seeds; one exploiter per rule set and pursuer; F8 and F10 differ from F11 in more than the rules (F10 1 seed).
