# Literature Review & Novelty Check: Opponent Sampling, Online Opponent Adaptation, and Bayesian Opponent ID for Quadrotor Pursuit-Evasion

*Compiled 2026-10-26. All arXiv entries were fetched directly from `arxiv.org/abs/...` (title, authors, abstract, submission dates verified from page metadata). Venue claims not visible on the arXiv abstract page (e.g., journal acceptance) are flagged as unverified. Non-arXiv items were checked against publisher pages (PMLR, Springer, GitHub) where possible.*

*Updated 2026-11-04: every section-1 entry checked against the full text (results bullets added, contradicted fields corrected), one paper added (Gavin, Lacroix & Bronz 2025), citation chases run for (A), (B) and (C) with revised verdicts in section 2, reference list and caveats updated. Section 4 lists what is still unverified.*

*Added 2026-11-08 (after F6): a separate check of diversity methods that keep the trained agent as strong as a
hardness sampler (SMERL, DOMiNO, DGPO, RSPO / SIPO, AlphaStar exploiters, FXP, Minimax Exploiter, RPPO, PSRO
variants), with fit to our F6 failure modes and a ranked shortlist:
[diversity_quality_2026-11-08.md](diversity_quality_2026-11-08.md). About 30 searches, not systematic; unverified
claims are marked there.*

---

## 1. Per-paper summaries

*Note (2026-11-04): every entry was checked against the full text on 2026-11-04 and corrected where the text contradicted it. The **Results / conclusions** bullets quote the authors' own conclusions and limitations. Most **Stated limitations** fields written on 2026-10-26 hold our own assessment of each paper for this thesis, not limitations the authors state, unless they say "Authors' own".*

### AgilePE — Tang et al., 2026
- **Title:** *AgilePE: Autonomous UAV Pursuit-Evasion via Self-Play Reinforcement Learning*
- **Authors:** Wenhao Tang, Tianyang Chen, Zhejun Cui, Boyuan An, Jiayu Chen, Ruize Zhang, Huidong Liu, Tianyue Wu, Qingmin Liao, Fei Gao, Yu Wang, Chao Yu
- **Venue/Year:** arXiv preprint, submitted 14 Aug 2026; arXiv Comments: "8 pages, 7 figures. Under review" (cs.RO; cs.LG). No accepted venue as of 2026-11-04.
- **arXiv:** [2608.14135](https://arxiv.org/abs/2608.14135)
- **Summary:** Presents a full-stack UAV pursuit-evasion system: a policy mapping onboard state directly to CTBR commands (no trajectory planner), trained via competitive self-play using **Prioritized Fictitious Self-Play (PFSP)** against a "diversified opponent pool" to stabilize optimization and reduce policy oscillation. A hardware-aligned sim pipeline (actuator dynamics, comms latency, domain randomization) enables zero-shot sim-to-real transfer, reproducing dodge/flank tactics on real quadrotors.
- **Experimental setting:** Isaac-Sim-style high-fidelity simulator → zero-shot deployment on real two-agent quadrotor hardware; qualitative tactic analysis (dodging, flanking).
- **Results / conclusions (full text read, 2026-11-04):** All quantitative results are in simulation (OmniDrones/Isaac Sim, 6×6×3 m arena). The metric is the pursuer's **in-FOV rate** (evader kept inside a forward cone), not a capture rate. Naive SP exceeds 0.9 in-FOV late in training (Fig. 4), which the authors read as evader strategy collapse. In Table IIb/c the "Prey vs. Pred. k" row rises 0.28→0.81 under FSP and 0.53→0.88 under PFSP, and PFSP's predator iteration 6 stays >0.82 against all historical prey (Fig. 5b). Trade-off: PFSP agents become "slightly more vulnerable to simplistic scripted patterns" (Sec. IV-A.3, Table IIc). No seed counts or error bars are given, and there is no FSP-vs-PFSP head-to-head or comparison with other learning methods. Real-world deployment (Jetson Orin NX + PX4, 60 Hz policy) is **qualitative only** (Fig. 7 snapshots, no trial counts). Conclusion: PFSP "sustains tactical diversity and drives robust policy equilibria", and zero-shot deployment works. Stated limitations (Sec. VII): 1v1, obstacle-free, state-based observations. Naive SP used "multiple independent trials using different random seeds" (count not given). Caveats: (a) Sec. IV-A.2 says "from 0.28 (iteration 0) to 0.81 (iteration 4)", but Table IIb has 0.82 at iteration 4 and 0.81 at iteration 6; (b) the same "Prey vs. Pred. k" row is read as FSP evader *survival* (IV-A.2) and as a PFSP "superior capture rate of 0.88" (IV-A.3), which cannot both hold; (c) Sec. III-A / Fig. 3 describe the evader exploiting obstacles, while Sec. VII says experiments were "obstacle-free"; (d) the PFSP weighting is described inconsistently ("defeats well", Sec. III-C, vs. "failure rate", Sec. IV-A.3).
- **Stated limitations:** (Sec. VII) 1v1 only, obstacle-free, state-based observations. The PFSP sampling rule is only described in words (no formula) and inconsistently (see caveat (d) above).
- **Relation to candidates:** Directly competing/adjacent to **(A)** — its "diversified opponent pool" is the historical checkpoint pool, sampled uniformly (FSP) or by win rate (PFSP), with no DPP or other explicit diversity mechanism (full text checked 2026-11-04); this is the single closest **system-level** prior work to your overall thesis setting (CTBR, self-play, Isaac-Sim-class pipeline). Hardware is a Jetson Orin NX + PX4 quadrotor, not a Crazyflie-class vehicle.

### Roncero et al., 2026 — AMSPBH
- **Title:** *Learned Controllers for Agile Quadrotors in Pursuit-Evasion Games*
- **Authors:** Alejandro Sanchez Roncero, Yixi Cai, Olov Andersson, Petter Ogren (KTH)
- **Venue/Year:** arXiv, v1 submitted 3 Jun 2025, v3 20 Jun 2026; framed as a "letter" (consistent with an RA-L-style submission, but **acceptance venue not confirmed** on the abstract page)
- **arXiv:** [2506.02849](https://arxiv.org/abs/2506.02849)
- **Summary:** Studies 1v1 quadrotor PE with PPO self-play. Proposes **AMSPBH** (Asynchronous Multi-Stage Population-Based training with **Hedge sampling**), a PSRO variant where each generation trains one agent against frozen opponents sampled by a Hedge algorithm that upweights currently-hard opponents. Shows AMSPBH retains competence against older opponents, generalizes better than latest-opponent-only training, and yields a population with distinct pursuit/evasion behaviors.
- **Experimental setting:** Isaac-Sim-class simulator; Crazyflie brushless quadrotor hardware validation, zero-shot sim-to-real, up to 4.87 m/s flights, against handcrafted and learned adversaries.
- **Results / conclusions (full text read, 2026-11-04):** Setup: Isaac Lab, 4×5×2 m arena, 0.1 m capture radius, 13 generations × 50M frames per oracle, 3 seeds, 1024 episodes per matchup. AMSPBH's cross-generation win-rate heatmaps show later generations beating earlier ones, while AMSDRL (latest-opponent only) is erratic, which the authors attribute to forgetting and cycling (Fig. 3). In final cross-evaluation (Fig. 4, bars only) AMSPBH has the best balanced score (min of pursuer and evader score per seed) and is best against the unseen APF evader. AMSPB (uniform sampling) beats it only against trajectory evaders, and AMSDRL is "consistently weaker". In the Hedge-parameter ablation vs. uniform AMSPB (Fig. 5), 8 of 9 settings improve by +0.05 to +0.27. The one negative (−0.09 at γ=0.7, η=0.5) is called likely seed noise. The nominal γ=0.7, η=2 gives +0.25 and the highest average win rate against handcrafted baselines (0.93). Note that the sampler (Eq. 12) is p_j = γ/N + (1−γ)·c_j^η/Σc^η with c_j = 1 − win rate, so 70 % of the mass is uniform at the nominal setting. It is not multiplicative-weights Hedge. Hardware (Table II): 42 zero-shot Crazyflie-brushless flights, 2–5 per matchup. The AMSPBH pursuer caught hover 5/5, circle 5/5, APF 4/5, manual 3/3, AMSPBH 5/5 and AMSDRL 3/3, but AMSPB only 2/5. The AMSPBH evader escaped manual 3/3, FRPN 2/2 and AMSPB 2/3, but AMSDRL caught it 2/3. Peak 4.87 m/s and 6.08 rad/s. Conclusion: population training with Hedge mitigates forgetting/cycling and generalizes better than single-opponent training. Stated limitations (Sec. V): memoryless MLPs; seeds converge to different strategies, so deployment needs "policy selection, ensembles, or online adaptation"; perfect state information. The "limited improvement from additional best-response training" is framed as convergence, not as a limitation. There is no PFSP arm; ablations are only vs. uniform (AMSPB) and latest-only (AMSDRL).
- **Stated limitations:** (Sec. V, full text) memoryless feed-forward MLPs; different seeds converge to different strategies, so deployment needs "policy selection, ensembles, or online adaptation"; perfect state information. No DPP/diversity-based sampler is compared.
- **Relation to candidates:** **This is the paper your thesis most directly extends/competes with for (A).** Its Hedge sampler is exactly the "PFSP/Hedge" baseline your (A) claims to improve on (bounded mass vs. duplicate-inflatable Hedge weight).

### Gavin & Bronz, 2026
- **Title:** *Intercepting an Agile Target with Net-Carrying Drones using Competitive Multi-Agent Reinforcement Learning*
- **Authors:** Timothée Gavin (ENAC-LAB, LAAS-RIS; IAS, Thales LAS), Murat Bronz (ENAC)
- **Venue/Year:** 2026 International Conference on Unmanned Aircraft Systems (**ICUAS 2026**), Corfu, Greece, June 2026, pp. 759–766, DOI 10.1109/ICUAS69441.2026.11598610 (arXiv Journal-ref and the IEEE PDF; verified 2026-11-04). arXiv v1 7 Jul 2026.
- **arXiv:** [2607.05939](https://arxiv.org/abs/2607.05939)
- **Summary:** Multi-pursuer (net-carrying) vs. single-evader interception, trained with **MAPPO + PFSP**, using low-level CTBR commands. Compares to heuristic baselines on catch rate, time-to-catch, crash rate; ablations show PFSP → more robust/adaptive policies and that low-level (CTBR) control is crucial; qualitative analysis shows emergent cooperative pursuer tactics.
- **Experimental setting:** High-fidelity simulator, CTBR control, team pursuit (not 1v1).
- **Results / conclusions (full text read, 2026-11-04; the PDF is the published ICUAS 2026 version, DOI 10.1109/ICUAS69441.2026.11598610):** **Simulation only**: 3 net-carrying pursuers vs. 1 evader, JAX simulator, 32×32×16 m arena with the evader confined to 12×12×6 m. 10,000 episodes per pairing; seeds not reported (Table II, Sec. V). The PFSP-CTBR pursuer catches 88.1/91.3/93.3/76.9/88.4 % of the APF/noSP/SP/PFSP-vel/PFSP-CTBR evaders, with time-to-catch 2.82–4.10 s. The best heuristic (FRPN+APF) reaches 71.9/47.0/64.7/59.6/20.0 %. Against the PFSP-CTBR evader, SP (latest-opponent) pursuers catch 22.5 %, noSP 3.5 %, PFSP-vel 17.7 % and FRPN 2.9 %. The PFSP ablation is against SP and noSP (a fixed heuristic opponent), **not against uniform FSP**. PFSP is mixed 50 % latest opponent / 50 % PFSP with f(x) = (1−x)² over a graded score (Sec. IV-D). PFSP-vel vs. PFSP-CTBR supports the conclusion that CTBR control is crucial. The claim that the PFSP-CTBR evader has higher evade and lower crash rates "across all settings" is not strictly supported by Table II (e.g. vs. the noSP pursuer, the noSP evader evades 24.2 % vs. 11.9 %). Stated limitations: no real-world transfer yet, no perception/state estimation, obstacle-free arena with a constrained evader volume ("strong assumptions"), and rare encircling, attributed to early simple, less agile evaders leading to reactive strategies and to the confined evader volume ("A bigger arena could alleviate this issue", Sec. V-C). Sec. III-B claims PFSP works "by improving the diversity of the opponents' policies", but there is no explicit diversity mechanism.
- **Stated limitations:** See the Results bullet for the authors' own. For this thesis: team (3v1, not 1v1) setting, so less directly comparable to your 1v1 protocol; no opponent-diversity/DPP mechanism; no uniform-sampling arm.
- **Relation to candidates:** Confirms PFSP is the de facto baseline in this literature (relevant to **(A)**'s novelty claim vs. PFSP); CTBR + MARL + catastrophic-forgetting framing is close background context.

### Gavin, Lacroix & Bronz, 2025 — 1v1 net-capture interception (precursor to Gavin & Bronz 2026; added 2026-11-04)
- **Title:** *Agile Interception of a Flying Target using Competitive Reinforcement Learning*
- **Authors:** Timothée Gavin (ENAC-LAB / LAAS-RIS / Thales LAS), Simon Lacroix (LAAS-CNRS, RIS), Murat Bronz (ENAC)
- **Venue/Year:** arXiv v1 submitted 17 Mar 2026 (cs.RO; stat.ML); arXiv Journal-ref: "Conference on Artificial Intelligence for Defence, AMIAD, Nov 2025, Rennes, France" (cited as CAID 2025 in Gavin & Bronz 2026). No proceedings DOI found (**proceedings details unverified**).
- **arXiv:** [2603.16279](https://arxiv.org/abs/2603.16279)
- **Summary:** 1v1 interception of an agile evader by one net-carrying pursuer as competitive MARL. Separate PPO policies for pursuer and evader output CTBR commands. The critics are privileged (both states plus the opponent's action). Training uses a JAX simulator with drag, motor speeds, delays and a low-level controller; an SE(3) controller drives the heuristic baselines. Each policy trains against the opponent's **current** policy (no opponent pool).
- **Experimental setting:** Quantitative results in simulation only. Obstacle-free arena with the evader effectively confined to a central volume by a penalised buffer zone. Perfect state information, no domain randomisation, 10 s episodes at 100 Hz, memoryless 2×256 MLPs. Two arenas, each with its own trained pair: Large 40×40×14 m (evader 20×20×4 m) and Small 8×8×5 m (evader 6×6×4 m). Baselines: Pure Pursuit (PP) and FRPN pursuers; Hover and APF evaders. Hardware demo: one real Betaflight quadrotor pursuer against a *simulated* evader in an 8×8×5 m indoor arena (hardware-in-the-loop, OptiTrack 200 Hz), zero-shot.
- **Results / conclusions (full text read, 2026-11-04):** 10,000 episodes per pairing; seeds not reported (Table III, Sec. V-B). Against the learned evader, the learned pursuer catches 78.8 % (Small; PP 58.3 %, FRPN 37.7 %) and 66.5 % (Large; PP 24.6 %, FRPN 49.2 %), with time-to-catch 3.78 s vs. 5.29 / 6.88 s (Small) and 6.62 s vs. 8.32 / 6.72 s (Large). Against heuristic evaders it is much worse: Large Hover 20.7 % (PP 100 %, FRPN 97.5 %) and APF 34.0 % (FRPN 68.8 %); Small Hover 90.7 % (PP 96.4 %, FRPN 97.4 %). The authors attribute this to the pursuer having "overfitted to the strategies of the agile learned evader encountered during training" and to its observation having no history, so it "cannot infer that the evader is stationary" (Sec. V-B). The learned evader crashes far less than APF (0.3–4.1 % vs. 15.4–36.4 %). It evades PP / FRPN / the learned pursuer 40.0 / 59.9 / 16.5 % (Small) and 74.6 / 47.3 / 31.9 % (Large). The text's claim of "a higher evade rate … against all pursuers" is contradicted in two Table III cells: APF evades PP 47.9 % (Small) and the learned pursuer 50.4 % (Large), largely through pursuer crashes. Training curves (Fig. 4) show adaptation / counter-adaptation cycles that level off, read as "approaching a Nash equilibrium"; no exploitability is measured. The training budget is reported inconsistently (2 × 10^9 steps, about 1 h 35 min, in Sec. IV-C vs. 4 × 10^9 in Table II). Hardware: one 28 s flight with 7 catches and no crash or exit; no real-flight catch-rate statistics. Conclusion: competitive RL with CTBR outperforms heuristics against agile learned opponents, and the JAX simulator "proved crucial".
- **Stated limitations:** (Sec. VII) sim-to-real gap, no domain randomisation; overfitting to interaction patterns: "When training only against the latest version of the opponent, agents are prone to forget how to deal with previously encountered strategies", with the remedy "Expanding the diversity of opponent strategies met during training, for example via Self-Play or Population-Based Training"; perfect state information; 1v1 bounded obstacle-free arena only; quantitative evaluation mainly in simulation.
- **Relation to candidates:** **(A):** no opponent pool or sampler (latest-opponent only), so no sampler comparison. Its value is documented motivation: the authors blame latest-opponent training for forgetting and overfitting (the hovering-evader failure) and call for more opponent diversity. The ICUAS 2026 follow-up adds PFSP mixed 50/50 with the latest opponent, still with no uniform or diversity arm. **(B):** none; memoryless policies, and the authors link the stationary-evader failure to the missing history, which is exactly the gap an online opponent-adaptation module targets. **(C):** none; the pursuer "must expect it to flee at any moment" whatever the opponent type, a clear case where inferring the opponent's type (hover vs. flee) would help. Setting: 1v1 CTBR, 10 s horizon, 8×8×5 m indoor-scale arena, the closest Gavin paper to the thesis's 1v1 protocol. Its hardware evidence is weaker than Roncero's: one real pursuer against a simulated evader, one flight.

### Capture definitions in the reference papers (checked in the full PDFs, 2026-11-06)
- **Roncero et al.:** capture when the distance between the two **geometric centres** falls below r_c = **0.1 m** (Table I),
  terminal. The capture bonus is scaled by exp(-alpha_close * max(0, v_close - v_safe)), v_safe = 2 m/s, alpha_close = 3,
  "discouraging unsafe high-speed collisions"; the disabling mechanism ("a net or a physical collision") is left out of
  scope. Opponent crash / out of bounds is not rewarded. The thesis uses the same 0.1 m centre-to-centre radius.
- **Gavin, Lacroix & Bronz (1v1) and Gavin & Bronz (3v1):** capture when the evader's centre comes within a capture
  distance of the pursuer's **rigid circular net of radius R** (mounted below the pursuer, body-aligned). Numeric R and
  capture distance are not given in the text (not found by full-text search; possibly only in a figure). Body contact
  between agents is simulated and penalised softly (r_coll), not terminal (3v1: pursuer-pursuer contact is terminal).
- **AgilePE:** no capture event. The metric is the in-FOV rate (evader inside the pursuer's detection cone). Episodes
  terminate on "collision when the inter-agent distance drops below a safety threshold", with a proximity penalty
  P_close below a safety distance; the threshold value is not given in the text.

### Episode endings and crash-luring in the reference papers (checked in the full PDFs, 2026-11-09)
Why this matters: in the thesis, self-play evaders beat pursuers almost only by luring them into a floor crash (F2c,
F3, F5, F6, F7 in [RESULTS.md](../RESULTS.md)). Whether an evader gains from a pursuer crash depends on how the episode
ends.
- **Thesis rules:** any crash ends the episode (terminated, value 0); the evader gets 0 when the pursuer crashes, +100
  at the timeout and -100 if captured. Ending at 0 beats a likely capture, so luring pays.
- **Roncero et al. v3 (Sec. II-B, Eqs. 8-11):** capture is the only terminal event; out-of-bounds (incl. crashes) and
  the timeout are truncations whose value target bootstraps from the critic: "This prevents a pursuer crash from
  artificially reducing the evader's survival return." Each agent gets -10 only for its own out-of-bounds, nothing for the
  opponent's; the evader's capture penalty is -10 and its survival bonus 0.0025 per step; the pursuer's capture bonus
  is scaled down above a 2 m/s closing speed. A pursuer crash therefore neither helps nor hurts the evader: no lure
  incentive. His paper reports no crash rates. (His public code `KTH-RPL/AMSPB_PEG`, Aug 2025, differs from v3:
  Hummingbird T/W 3.42, larger arena, 0.5 m capture radius, no Hedge; not used.) Tested in thesis F8.
- **Gavin, Lacroix & Bronz (1v1, Sec. IV) and Gavin & Bronz (3v1, Sec. IV-C):** a ground crash or leaving the arena
  terminates the episode with a penalty for the failing agent; "neither the evader nor the pursuer receive a reward when
  the opponent reaches a failure state to promote actual pursuit-evasion behaviours rather than forcing the opponent to
  crash". The evader has a positive per-step term (distance to the net in 1v1, a survival reward in 3v1) and
  -lambda_catch = -10 if caught. Structurally the same incentive as the thesis: a pursuer crash ends the game at nothing
  for the evader, which beats a capture. Their tables are consistent with crash-heavy evader wins (not shown to be floor
  lures; their "crash" includes leaving the arena):
  - 1v1 Table III, vs their learned evader (evade rate counts timeouts and pursuer crashes): learned pursuer, small
    arena evade 16.5 % of which pursuer crash 14.2 %; large 31.9 % / 17.2 %. Pure pursuit: 40.0 % / 39.3 % (small),
    74.6 % / 73.5 % (large). FRPN: 59.9 % / 59.8 % (small).
  - 3v1 Table II, vs their PFSP evader: pursuer crash 69.7 % (FRPN), 84.3 % (pursuers trained without self-play), 32.1 %
    (latest-opponent self-play), 4.5 % (PFSP velocity-command pursuer, which instead loses by timeout, 77.6 %), 7.7 %
    (their final PFSP-CTBR pursuer, catch 88.4 %). No fresh-exploiter test, so the final pursuer's lure robustness is
    untested.
- **AgilePE (Sec. III-A/B):** a tracking game (evader inside the pursuer's FOV cone), no capture. Episodes end at
  timeout, out-of-bounds or a close-distance "collision"; the dense zero-sum reward's signs and what each agent receives
  at an episode end are not given, so whether a pursuer crash helps their evader cannot be determined. No crash rates
  reported.

### ESN-OA paper — arXiv 2602.06326
- **Title:** *Online Adaptive Reinforcement Learning with Echo State Networks for Non-Stationary Dynamics*
- **Authors:** Aoi Yoshimura, Gouhei Tanaka
- **Venue/Year:** arXiv, submitted 6 Feb 2026 (cs.LG); arXiv Comments: "Submitted to IJCNN 2026"
- **arXiv:** [2602.06326](https://arxiv.org/abs/2602.06326)
- **Summary:** Proposes an **Echo State Network (ESN) + online RLS readout** adaptation module: the ESN reservoir encodes recent observation history into a latent context; only the linear readout is updated online via RLS, requiring no backprop/pretraining/privileged info. Targets non-stationary *dynamics* (not opponents).
- **Experimental setting:** CartPole and HalfCheetah with abrupt disturbances/friction changes (non-adversarial, classic control/MuJoCo benchmarks) — **not** pursuit-evasion, **not** drones, **not** opponent modeling.
- **Results / conclusions (full text read, 2026-11-04):** 10 seeds throughout. CartPole is trained without wind and tested under periodic wind A·cos(ωt) (Table 1). SAC, DR and RMA collapse above A = 8.0. At A = 8.0, ESN-OA scores 930.2 ± 199.0 and ESN-OA-PT 1000.0 ± 0.0, vs. SAC 95.5, DR 541.6, RMA 459.0 and PAD 191.7. ESN-OA then drops to 402.3 (A = 8.5) and 182.7 (A = 9.0), below DR at both (501.5 and 461.2; ESN-OA-PT is best there), and at A = 9.5 DR is best (231.8 vs. ESN-OA 96.2, ESN-OA-PT 97.6 ± 12.5). This contradicts the text's "near-optimal across the tested range" and "even higher stability at extreme amplitudes (A = 9.5)" for ESN-OA-PT, so cite the table. HalfCheetah has friction stepped to 1–10× at t = 500 (Table 2). SAC is best at nominal friction (6540.5 vs. ESN-OA 5046.3); DR is best at F = 4 (5418.7); ESN-OA is best at F = 7 (5022.4) and F = 10 (3777.2 vs. DR 3164.3, SAC 3054.1, RMA 1956.5) and recovers "within a few dozen steps" after a 1×→10× switch (Fig. 5). Cost: 0.27 ms/step vs. PAD 0.61 ms (Fig. 3). Conclusion: an ESN context plus an RLS readout recovers from abrupt changes within "a few steps" (Sec. 6; Sec. 5.2.1 says "a few dozen steps"), without backprop, pretraining or privileged information (Sec. 6). Stated limitations (Sec. 6.4): RLS is sensitive to observation noise/outliers, and there are no explicit stability guarantees. Implementation note for the (B) baseline: the readout predicts ŝ_{t+1}, which is appended to the SAC input, and SAC is trained with the ESN module running ("no pretraining" applies to W_out only).
- **Stated limitations:** Authors' own: see the Results bullet (Sec. 6.4). For this thesis: only non-adversarial single-agent benchmarks (CartPole and HalfCheetah, which the paper calls "high-dimensional (17-dimensional)"); no multi-agent or opponent-adaptation claim is made by the authors.
- **Relation to candidates:** This is the literal **ESN+RLS baseline architecture for (B)** — but note the paper itself targets non-stationary / out-of-distribution dynamics (domain randomization is one of its baselines), not opponent modeling/pursuit-evasion. Your thesis would be the first to (a) repurpose this exact ESN+RLS readout mechanism as an *opponent* model and (b) compare it against a Neural-Fly-style offline-basis+RLS approach in a competitive drone PE setting. **This is a methodological transplant, not prior art for your specific use case.**

### Perez-Nieves et al., ICML 2021
- **Title:** *Modelling Behavioural Diversity for Learning in Open-Ended Games*
- **Authors:** Nicolas Perez-Nieves, Yaodong Yang, Oliver Slumbers, David Henry Mguni, Ying Wen, Jun Wang
- **Venue/Year:** ICML 2021 (PMLR v139); arXiv v2
- **arXiv:** [2103.07927](https://arxiv.org/abs/2103.07927)
- **Summary:** Gives a geometric interpretation of behavioural diversity and introduces a **DPP-based diversity metric** over policies. Incorporates this metric as a **regularizer in the best-response objective**, yielding "diverse fictitious play" and "diverse PSRO." Proves uniqueness of the diverse best response and convergence in two-player games; shows the DPP metric provably enlarges the "gamescape" (convex polytope of strategy mixtures).
- **Experimental setting:** Meta-games of 28 real-world games (Czarnecki et al. 2020, incl. the AlphaStar meta-game of 888 RL policies), a 7-Gaussian non-transitive mixture game, Colonel Blotto and random zero-sum games (no robotics/drones).
- **Results / conclusions (main paper read 2026-11-04; appendices not available locally):** The headline benchmark is meta-games of 28 real-world games (Czarnecki et al. 2020). On these, diverse PSRO is "at worst" as good as SP, PSRO, PSRO-rN and Pipeline-PSRO, is often less exploitable, and always more diverse. On the AlphaStar meta-game (888 RL policies) it reaches the lowest exploitability with < 50 policies and keeps approaching zero while the baselines saturate (Fig. 2; other games in App. E). In the 7-Gaussian non-transitive mixture game, PSRO/PSRO-rN cycle and fail while DPP-PSRO reaches about zero exploitability (Fig. 3). Diverse PSRO is less exploitable on Colonel Blotto (Fig. 4a), and diverse α-PSRO finds more SSCCs than α-PSRO (Fig. 4b). Results are plots only, with no numeric tables. Conclusion: "on tens of games, our methods achieve lower exploitability than PSRO variants by finding both effective and diverse strategies" (Sec. 6). The baselines' failure is attributed to cycling back to strategies already in the population. Note for (A): the paper itself says its diversity-aware best response "is similar in manner to selecting a diverse set of opponents" (Sec. 4), so the novelty argument for DPP-marginal opponent sampling should address this sentence.
- **Stated limitations:** (Our assessment, not the authors'.) DPP is used to *shape the training objective of the newest policy*, not to define a *sampling distribution over which frozen opponents to train against*; it never proposes sampling opponents ∝ DPP marginals K_ii, although it remarks that its diversity-aware best response "is similar in manner to selecting a diverse set of opponents" (Sec. 4).
- **Relation to candidates:** **Closest conceptual ancestor of (A)**, but uses DPP as a diversity **regularizer for best-response optimization**, not as an **opponent-sampling/curriculum distribution**. This is the key technical gap your (A) would need to argue it closes.

### Liu et al., NeurIPS 2021
- **Title:** *Towards Unifying Behavioral and Response Diversity for Open-ended Learning in Zero-sum Games* (arXiv title: *Unifying Behavioral and Response Diversity for Open-ended Learning in Zero-sum Games*)
- **Authors:** Xiangyu Liu, Hangtian Jia, Ying Wen, Yaodong Yang, Yujing Hu, Yingfeng Chen, Changjie Fan, Zhipeng Hu
- **Venue/Year:** NeurIPS 2021 (vol. 34, pp. 941–952). The NeurIPS author order differs from arXiv (Yang last: Liu, Jia, Wen, Hu, Chen, Fan, Hu, Yang).
- **arXiv:** [2106.04958](https://arxiv.org/abs/2106.04958)
- **Summary:** Unifies "Behavioral Diversity" (occupancy-measure discrepancy) and "Response Diversity" (reward-dynamics discrepancy across opponents) into one framework; shows most prior diversity metrics are BD-only or RD-only. Designs a diversity-promoting objective and "population effectivity" criterion for best-response solving.
- **Experimental setting:** AlphaStar888 empirical meta-game, non-transitive mixture model, Google Research Football (no drones/robotics).
- **Results / conclusions (full text arXiv v2 read, 2026-11-04):** On the AlphaStar888 meta-game, P-PSRO with the BD&RD regularizer has the lowest exploitability and highest population effectivity (PE) vs. SP, PSRO, PSRO_rN, P-PSRO and DPP-PSRO (Fig. 1a–b). On the non-transitive mixture game it explores most diversely (Fig. 2) and has the best or near-best PE (e.g. PE(25)×10² = −12.18 vs. P-PSRO −12.51, PSRO −54.59). Its final exploitability ×10² (13.26 ± 0.24) is no better than P-PSRO (13.21 ± 0.29) or DPP-PSRO (13.24 ± 0.33) (Table 2). The authors acknowledge this and argue for PE instead. In Google Research Football it has the largest goal difference vs. built-in bots at every difficulty (Fig. 3) The authors state "an average of 60% win-rate over other baseline methods" (Sec. 5), but App. Table 9 (F.3) gives 0.74 / 0.78 / 0.80 / 0.69 / 0.57 vs. SP / PSRO / PSRO_rN / PSRO w. BD / PSRO w. RD, a mean of about 0.72. DPP-PSRO was left out of GRF because it "cannot scale to such a complex setting". Conclusion: robust "to both highly non-transitive games and complex games like the Google Research Football environment" (Sec. 6).
- **Stated limitations:** Still a training-time diversity **objective**, not an explicit opponent-sampling rule for curriculum/self-play.
- **Relation to candidates:** Background for **(A)**'s behaviour-descriptor framing (BD/RD distinction is directly relevant to how you'd define the RBF kernel S over "response vector vs. probe pursuers + trajectory statistics").

### Liu et al., NeurIPS 2022
- **Title:** *A Unified Diversity Measure for Multiagent Reinforcement Learning*
- **Authors:** Zongkai Liu, Chao Yu, Yaodong Yang, Peng Sun, Zifan Wu, Yuan Li. Six authors per the proceedings page, proceedings BibTeX, OpenReview (H-6iczs__Ro) and DBLP; the camera-ready PDF title block omits Yuan Li.
- **Venue/Year:** NeurIPS 2022 (vol. 35, pp. 10339–10352; DOI 10.52202/068431-0751)
- **arXiv:** **No arXiv preprint found.** Verified only via NeurIPS proceedings / OpenReview / mlanthology.org — **cite via NeurIPS proceedings, not arXiv.**
- **Summary (checked against the proceedings PDF, 2026-11-04):** Proposes "Unified Diversity Measure" (UDM) generalizing prior BD/RD-type metrics; derives UDM-Fictitious Play and UDM-PSRO with convergence guarantees; evaluated against PSRO baselines on non-transitive games (see Results for where it does and does not beat them).
- **Experimental setting:** AlphaStar888 empirical meta-game, Blotto (3 areas, 10 coins), 7-Gaussian non-transitive mixture model.
- **Results / conclusions (full text read from the NeurIPS proceedings PDF, 2026-11-04; this verifies the entry against the primary source):** Experiments: AlphaStar888, Blotto (3 areas, 10 coins) and the 7-Gaussian non-transitive mixture model. Baselines: SP, PSRO, P-PSRO, PSRO-rN, FEP-PSRO, EC-PSRO and RPD-PSRO; UDM uses RD-type features and a cubic polynomial kernel. On AlphaStar888 it has "the smallest exploitability, largest population effectivity and the largest diversity" (Fig. 1). On Blotto it is only "as well as the diversity-aware baselines" and beats the non-diversity ones. On the mixture model it has the best PE at every opponent strength (PE(25)×10² = 1.81 vs. RPD-PSRO 1.39, PSRO −25.47). Its exploitability ×10² is 2.07 ± 0.37, about equal to P-PSRO 2.34, EC-PSRO 2.05 and RPD-PSRO 1.95, and far below PSRO's 33.71 (Table 3). Conclusion: UDM methods converge to a two-player NE and beat baselines on exploitability and PE, though Table 3 does not bear out the exploitability part (EC-PSRO 2.05 and RPD-PSRO 1.95 are slightly lower). RPD-PSRO and FEP-PSRO each appear in only one experiment. Limitation: two-player only, since UDM is expensive with n players (Sec. 6).
- **Stated limitations:** Authors' own (Sec. 6): two-player games only, because UDM is expensive to compute in n-player games; extension to n-player / general-sum / non-symmetric games is future work.
- **Relation to candidates:** Same family as Perez-Nieves/Yao — diversity as a training objective, not as an opponent-sampling distribution; relevant background for **(A)**.

### Yao et al., NeurIPS 2023 — PSD-PSRO
- **Title:** *Policy Space Diversity for Non-Transitive Games*
- **Authors:** Jian Yao, Weiming Liu, Haobo Fu, Yaodong Yang, Stephen McAleer, Qiang Fu, Wei Yang
- **Venue/Year:** NeurIPS 2023 (pp. 67771–67793; confirmed via proceedings / ACM DL, 2026-11-04); arXiv submitted 29 Jun 2023, revised 8 Nov 2023
- **arXiv:** [2306.16884](https://arxiv.org/abs/2306.16884)
- **Summary:** Proves that existing diversity metrics' improvement does **not** guarantee a better Nash-equilibrium approximation, and proposes a new diversity metric whose improvement does guarantee this, optimized from state-action samples alone; yields **PSD-PSRO**, shown empirically to be less exploitable than SOTA PSRO variants.
- **Experimental setting:** Standard non-transitive game suites (no robotics).
- **Results / conclusions (full text arXiv v2 read, 2026-11-04; NeurIPS 2023 acceptance confirmed via proceedings / ACM DL, pp. 67771–67793):** Games: AlphaStar888, non-transitive mixture, Leduc poker, Goofspiel and the Disc game (appendix). Baselines: PSRO, P-PSRO, PSRO_rN, DPP-PSRO and BD&RD-PSRO. On AlphaStar888 (Fig. 1) and Leduc (Fig. 3; there vs. PSRO, PSRO_rN and P-PSRO only), exploitability and PE drop faster than for the baselines. On the mixture game PSRO/PSRO_rN "get trapped" and PSD-PSRO ends "significantly lower" (Fig. 2). On Goofspiel (8 cards) it beats PSRO, PSRO_rN and P-PSRO at 0.578 / 0.642 / 0.630 (Table 1). The diversity weight λ matters a lot. Mixture exploitability ×100 is 40.88 at λ = 0 and 1.72 at λ = 2. On Leduc it is 0.590 at λ = 0, 0.356 at λ = 0.5 and 0.664 at λ = 5 (App. Tables 3–4). Conclusion: "significantly more effective in approximating a NE than state-of-the-art PSRO variants". Stated limitations (Sec. 7): no guidance on choosing λ, and not scaled to larger games.
- **Stated limitations:** Diversity used again as a best-response-training regularizer, not an opponent-sampling rule.
- **Relation to candidates:** Same family — relevant **theoretical backdrop** for why naive diversity metrics can be gamed (useful to motivate why K_ii-bounded sampling in (A) is theoretically preferable to ad hoc diversity bonuses), but not itself a sampling method.

### Balduzzi et al., ICML 2019
- **Title:** *Open-ended Learning in Symmetric Zero-sum Games*
- **Authors:** David Balduzzi, Marta Garnelo, Yoram Bachrach, Wojciech M. Czarnecki, Julien Perolat, Max Jaderberg, Thore Graepel
- **Venue/Year:** ICML 2019
- **arXiv:** [1901.08106](https://arxiv.org/abs/1901.08106)
- **Summary:** Introduces a geometric framework for open-ended learning in nontransitive zero-sum games, and **PSRO_rN ("rectified Nash response")**, which uses "game-theoretic niching" to build diverse, effective populations — a precursor to all the diversity-PSRO lineage above.
- **Experimental setting:** Colonel Blotto and Differentiable Lotto (synthetic resource-allocation games), plus the disc game for illustration; no robotics.
- **Results / conclusions (full text arXiv v2 read, 2026-11-04):** Games: Colonel Blotto (3 areas, 10 coins, MPO oracle) and Differentiable Lotto (9 customers, gradient-ascent oracle), at a fixed oracle-call budget. PSRO_rN has positive relative population performance vs. SP, PSRO_U and PSRO_N across a wide range of compute budgets (Fig. 4; plots only, values up to about 0.4). PSRO_U ≈ PSRO_N, and SP is worst (as a population, SP "slightly outperforms" PSRO_N/PSRO_U on Blotto). On Differentiable Lotto, PSRO_rN is the only method whose gamescape grows at every iteration (Fig. 5). Directly relevant to Hedge/PFSP: on the disc game, the authors argue that training against opponents one *loses* to shrinks the gamescape and "does not encourage diverse niches" (Sec. 4.2, Fig. 3C). Conclusion: PSRO_rN is a validated algorithm "for uncovering strategic diversity within functional form games". Its own failure mode is many extremely local niches (Sec. 4.2), and more complex games are left to future work (Sec. 6).
- **Stated limitations:** No robotics/continuous-control application; niching via rectified-Nash weighting, not DPP.
- **Relation to candidates:** Foundational background for **(A)** — the PSRO lineage your opponent pool inherits from.

### Vinyals et al., Nature 2019 — AlphaStar
- **Title:** *Grandmaster level in StarCraft II using multi-agent reinforcement learning*
- **Authors:** Oriol Vinyals, Igor Babuschkin, Wojciech M. Czarnecki, Michaël Mathieu, Andrew Dudzik, et al. (DeepMind)
- **Venue/Year:** *Nature* 575, 350–354 (2019); DOI 10.1038/s41586-019-1724-z
- **arXiv:** **None.** Confirmed via web search — this is a Nature-only publication with no arXiv preprint (the arXiv ID 1902.01724 that surfaces in searches, "AlphaStar: An Evolutionary Computation Perspective" by Arulkumaran et al., is a **different, third-party analysis paper**, not the primary source — do not conflate the two).
- **Summary:** Introduces the AlphaStar **League**: a population of main agents, main exploiters, and league exploiters trained via multi-agent RL with **Prioritized Fictitious Self-Play (PFSP)**, where opponent-sampling probability is f(P[A beats B]) / Σ f, with polynomial f_hard(x) = (1−x)^p (default) or f_var(x) = x(1−x) (Methods, "Prioritized fictitious self-play"), i.e. weighted toward opponents the current agent loses to; not a softmax. Not every agent type uses PFSP: main exploiters play the current main agents and use PFSP with f_var only half the time when their win probability is below 20 %.
- **Experimental setting:** Full StarCraft II, reaching Grandmaster level, validated online vs. human players.
- **Results / conclusions (full text read, 2026-11-04):** AlphaStar Final reached 6,275 / 6,048 / 5,835 MMR (Protoss / Terran / Zerg), Grandmaster for all three races and above 99.8 % of ranked players (Fig. 2b). The supervised-only agent reached 3,699 MMR (above 84 %). Ablations (Fig. 3; one map, Protoss vs. Protoss, 10^10 steps): adding main exploiters, then league exploiters, raises test Elo 1,540 → 1,693 → 1,824 and relative population performance 6 % → 35 % → 62 %. Opponent sampling gives test Elo PFSP+SP 1,540, SP 1,519, PFSP 1,273 and FSP 1,143. The forgetting measure (min win rate vs. past versions) is 71 / 46 / 70 / 69 %: "naive self-play has a high Elo, but is more forgetful". PFSP beats uniform FSP "under all measures considered" (Ext. Data Fig. 5). PFSP samples ∝ f(P[win]) with polynomial f_hard(x) = (1−x)^p or f_var(x) = x(1−x), not a softmax. Main agents use a 35 % SP / 50 % PFSP / 15 % PFSP-vs-forgotten-players-and-exploiters mix. The authors read the decreasing main-exploiter performance (with better results against held-out validation agents) as the main agent growing "increasingly robust", and the low Nash weight on old players as the league not cycling or regressing. Conclusion: AlphaStar's success "suggests that general-purpose machine learning algorithms may have a substantial effect on complex real-world problems" (abstract: "general-purpose learning methods that are in principle applicable to other complex domains"). Their caveats: f_hard can waste games on much stronger opponents, and the blind Battle.net games "do not directly measure its susceptibility to exploitation under repeated play."
- **Stated limitations:** Authors' own: see the Results bullet (f_hard wasting games on much stronger opponents; no measure of exploitability under repeated play). **Thesis hypothesis, not stated in the paper:** PFSP's hardness weighting can over-sample near-duplicate historical checkpoints of the same strategy, the failure mode **(A)** claims to fix via bounded DPP marginal inclusion probabilities.
- **Relation to candidates:** **The canonical PFSP reference** your (A) is positioned against; also the origin of "exploiters" as deliberately diversity-inducing opponents (conceptually adjacent to your hardness/behaviour-descriptor kernel).

### O'Connell et al., Science Robotics 2022 — Neural-Fly
- **Title:** *Neural-Fly Enables Rapid Learning for Agile Flight in Strong Winds*
- **Authors:** Michael O'Connell, Guanya Shi, Xichen Shi, Kamyar Azizzadenesheli, Anima Anandkumar, Yisong Yue, Soon-Jo Chung (Caltech)
- **Venue/Year:** *Science Robotics* 7(66), eabm6597 (2022); DOI 10.1126/scirobotics.abm6597
- **arXiv:** [2205.06908](https://arxiv.org/abs/2205.06908)
- **Summary:** Learns a shared aerodynamic feature basis φ offline across many wind conditions via **Domain Adversarially Invariant Meta-Learning (DAIML)** (per-condition linear coefficients solved by least squares with norm clipping on each batch, Alg. 1, + domain-adversarial loss to make φ condition-invariant), trained on 12 minutes of flight data (50 Hz, 6 wind conditions); at test time it adapts only a linear coefficient vector via a **composite adaptation law** (not literally RLS, but a Lyapunov-based adaptive-control update).
- **Experimental setting:** Real quadrotor flight in Caltech Real Weather Wind Tunnel, winds up to 43.6 km/h; cross-platform transfer.
- **Results / conclusions (full text read, 2026-11-04):** Figure-8 tracking (2.5 × 1.5 m, 6.28 s laps). Mean position error in cm (Table 1), Neural-Fly / INDI / L1 / nonlinear baseline: 2.9 / 6.3 / 4.2 / 10.8 at 0 m/s; 3.7 / 5.9 / 5.2 / 9.9 at 4.2 m/s; 5.3 / 8.2 / 11.1 / 14.7 at 8.5 m/s; 8.7 / 10.1 / 21.3 / 21.6 at 12.1 m/s; 6.9 / 10.3 / 11.6 / 28.2 at 8.5 + 2.4 sin(t) m/s. Neural-Fly is best in every condition, including winds above the 0–6.1 m/s training range (Fig. 6). Average improvements are 66 % vs. the nonlinear baseline, 42 % vs. L1 and 35 % vs. INDI. Neural-Fly-Transfer (basis learned on another drone) is close (9.4 cm at 12.1 m/s). Neural-Fly-Constant (trivial basis) is only about L1/INDI level, so the learned basis carries the gain. Outdoors, with wind measured up to 17 km/h (4.9 m/s), the error is 7.5 cm (Fig. 7). The 12 min of flight data is **offline DAIML training data** (6 wind conditions); per-wind coefficients are least-squares solved per batch. The whole adaptive controller "only takes 10 ms to compute" on a Raspberry Pi 4 (basis inference ~3.5 ms, adaptation update ~3.0 ms). Conclusion: splitting the model into a learned wind-invariant basis and low-dimensional linear weights enables fast adaptation to unknown, time-varying wind with exponential-stability guarantees, and the basis transfers across vehicles. On the adaptation law: its regularisation, prediction-error and covariance terms form a Kalman–Bucy filter, with a tracking-error term added.
- **Stated limitations:** Authors' own (Discussion): 15–30 ms code/communication delay plus attitude-tracking delay limit accuracy, and "further research is required to understand the coupling between planning and learning-based control near actuation limits". For this thesis: domain is wind/aerodynamics, not an adversarial opponent; adaptation law is a stability-guaranteed composite update, not RLS per se (a detail your (B) description should be precise about — RLS is the natural discrete-time analogue, but Neural-Fly's own update is a continuous-time composite adaptive law).
- **Relation to candidates:** **This is the literal namesake/method template for (B).** Your novelty rests entirely on: (i) swapping "wind condition" for "opponent evader identity" as the multi-task label, and (ii) swapping the composite adaptive law for RLS, and (iii) using the predicted feature as an **observation input to a PPO policy** rather than as a direct control correction term.

### Kumar et al., RSS 2021 — RMA
- **Title:** *RMA: Rapid Motor Adaptation for Legged Robots*
- **Authors:** Ashish Kumar, Zipeng Fu, Deepak Pathak, Jitendra Malik
- **Venue/Year:** RSS 2021 (DOI 10.15607/RSS.2021.XVII.011)
- **arXiv:** [2107.04034](https://arxiv.org/abs/2107.04034)
- **Summary:** Base policy + adaptation module; the adaptation module infers a latent "extrinsics" vector online from a history of states/actions (via supervised distillation from a privileged encoder), enabling sub-second adaptation to terrain/payload/wear changes without backprop at test time.
- **Experimental setting:** Unitree A1 quadruped, diverse real terrains (sand, mud, stairs, pebbles), trained in sim, zero-shot transfer.
- **Results / conclusions (full text read, 2026-11-04):** In simulation (parameters resampled within episodes; 3 seeds × 1000 episodes; Table II), RMA reaches 73.5 % success, normalized time-to-fall 0.85 and 1.34 m distance. Comparisons: robust/DR 62.4 %, SysID 56.5 %, AWR 41.7 % (needs 40k adaptation samples), RMA without adaptation 52.1 %, privileged expert 76.2 %. On the real A1 (5 trials per setting) it has no failures on sand, mud, dirt or tall vegetation. Success rates: 70 % down hiking-trail stairs, 80 % over cement/pebble piles (Fig. 1), 100 % on a 15 cm step-down and on the mattress, 80 % on uneven foam (Fig. 3 data table; the caption swaps the step-down and foam figures), and 90 % across an oily patch (Fig. 4). Methods that failed badly were given only 2 trials. It carries up to 12 kg (100 % of body weight). Turning off the adaptation module "substantially degrades performance", and the factory controller fails on uneven foam. Stated limitation (Sec. VI): the robot is blind, so sudden falls on stairs or legs blocked by rocks still cause failures; vision is future work.
- **Stated limitations:** Domain is terrain/embodiment adaptation (environment, not an adversarial opponent); the online module is a distilled feed-forward encoder, not RLS.
- **Relation to candidates:** Background precedent for **(B)**'s "privileged-information distillation → fast online adaptation module" paradigm, parallel to but methodologically distinct from Neural-Fly's basis+linear-adaptation approach.

### Fu et al., ICML 2022 — GSCU
- **Title:** *Greedy when Sure and Conservative when Uncertain about the Opponents*
- **Authors:** Haobo Fu, Ye Tian, Hongxiang Yu, Weiming Liu, Shuang Wu, Jiechao Xiong, Ying Wen, Kai Li, Junliang Xing, Qiang Fu, Wei Yang
- **Venue/Year:** ICML 2022 (PMLR v162, pp. 6829–6848)
- **arXiv:** **No arXiv version found** — verified via PMLR proceedings page (`proceedings.mlr.press/v162/fu22b.html`) and the official code repo `github.com/YeTianJHU/GSCU`.
- **Summary:** Learns opponent-policy embeddings offline, trains a single embedding-conditioned best response (not a discrete library of responses), computes an **online Bayesian posterior over the current opponent's embedding** (avoiding discrete type classification), and selects online between a greedy (embedding-conditioned) policy and a fixed conservative policy via an adversarial-bandit algorithm with regret guarantees.
- **Experimental setting (verified in the paper, App. D/E):** **Kuhn Poker** (OpenSpiel) and **Predator-Prey** (Multi-Agent Particle Environment) — i.e., GSCU's own benchmark **already includes a simplified pursuit-evasion-like domain** with "online Bayesian adaptation for seen/unseen/mix/adaptive opponent settings."
- **Results / conclusions (full text read from PMLR, 2026-11-04):** Domains: Kuhn poker and 2D Predator-Prey. In PP **the controlled agent is the prey (evader)**, facing 3 predators that share one of 8 rule-based, direction-biased policies (4 train, 4 test; App. E). Baselines: a fixed conservative policy π1* (NE in Kuhn, PPO-trained robust policy in PP), DRON, LIAM, Deep BPR+, PPO-Tracking and GSCU-Greedy. Settings: seen / unseen / mix / adaptive opponent sequences, 5 seeds. On Kuhn (Fig. 2): GSCU-Greedy is best on seen opponents. On unseen, π1* is best and GSCU second (π1* is one of GSCU's two EXP3 arms). GSCU and GSCU-Greedy are top on mix, and GSCU is best and "least exploitable" vs. the adaptive PPO opponent. PP gives "similar conclusions", though Tracking and π1* are competitive there. GSCU has the best average and worst-case return over the four settings (App. H, Fig. 12). Results are plots only, with no numeric tables. Conclusion: inferring a continuous embedding posterior "is more effective than inferring a categorical distribution over a discrete number of opponent policies". Deep BPR+ "sometimes fails to identify the right opponent in time" (Sec. 5.5, Fig. 6). This argues against BayesID's discrete prototype library and should be addressed in (C). Future work: the seen-vs-unseen gap (training-pool diversity), and testing against humans, since GSCU assumes opponents are "to some extent, exploitable and predictable".
- **Stated limitations:** Low-dimensional 2D particle dynamics (not agile/CTBR quadrotor flight); continuous embedding space rather than a small discrete library of named prototype/medoid evaders; greedy/conservative switching is a two-arm bandit, not a continuous confidence-weighted blend.
- **Relation to candidates:** **The single closest prior work to (C).** GSCU already implements (a) online Bayesian (variational) inference of a continuous opponent-policy embedding, explicitly "without making the discrete and ineffective decision which type the current opponent belongs to", and (b) a conservative-fallback-when-uncertain mechanism, in a predator-prey (pursuit-evasion) toy domain. Your (C)'s novelty would have to rest on the **discrete medoid/prototype library + behavior-cluster construction + agile quadrotor CTBR domain**, not on the Bayesian-ID-with-fallback idea itself, which is not new.

### Rosman, Hawasly & Ramamoorthy, 2016 — Bayesian Policy Reuse
- **Title:** *Bayesian Policy Reuse*
- **Authors:** Benjamin Rosman, Majd Hawasly, Subramanian Ramamoorthy
- **Venue/Year:** *Machine Learning* 104(1), 99–127 (2016; online 22 Feb 2016); DOI 10.1007/s10994-016-5547-y (confirmed via Crossref, 2026-11-04)
- **arXiv:** [1505.00284](https://arxiv.org/abs/1505.00284) (preprint, 2015)
- **Summary:** Formalizes policy reuse: given a **library of policies pre-learned offline**, choose online which library policy to reuse for a novel task instance, using observed performance "signals" and a Bayesian-choice formulation with exploration/exploitation-balancing selection functions (an inspiration for BPR+/Bayes-ToMoP/GSCU-style methods).
- **Experimental setting:** Three simulated domains: golf-club selection, telephone-banking dialogue personalisation, and a 26×26-grid drone-surveillance task with 68 poacher locations; not adversarial pursuit-evasion.
- **Results / conclusions (full text arXiv read, 2026-11-04):** Domains: golf-club selection, telephone-banking personalisation (20 language models) and a 68-task drone-surveillance grid (searching for poachers). In golf, greedy BPR gets the ball "within 10–15 yards of the hole" by the second shot on average (100 holes, Fig. 5). In telephone banking the (s,a,s′) signal gives the lowest regret, then (s,a,r), then total utility; all reach zero regret (Fig. 7). In surveillance, BPR-BE, BPR-KG and belief sampling converge in about 15 episodes, "approximately a quarter" of brute force. BPR-PI and ε-greedy do not converge within 50 episodes, and BPR-EI is fastest (Fig. 9, Table 3). BPR-EI also beats UCB1 and GP-UCB (Fig. 10), and regret drops with a larger library or more episodes (Fig. 11). Conclusion: the belief over task types allows "global updates over the policy space" that per-arm bandits cannot; the authors call it "first steps toward a practical solution". The Bayes-optimal *response* is what is intractable; the type belief itself is updated directly. "BPR-EI is fastest" is from Fig. 9 and its text; Table 3 lists only regret standard deviations.
- **Stated limitations:** The Bayes-optimal *response* is often intractable, so it is approximated with exploration heuristics (PI, EI, BE, KG); the belief over the finite set of types is updated directly; no opponent-specific or adversarial framing (task instances, not adversaries).
- **Relation to candidates:** **Foundational ancestor of (C)** — BayesID's "small set of prototype evaders" + Bayesian belief update over which prototype is active is structurally a direct descendant of BPR's "library + Bayesian choice" framework, just applied to an adversarial opponent rather than a task family.

### Al-Shedivat et al., ICLR 2018
- **Title:** *Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments*
- **Authors:** Maruan Al-Shedivat, Trapit Bansal, Yuri Burda, Ilya Sutskever, Igor Mordatch, Pieter Abbeel
- **Venue/Year:** ICLR 2018
- **arXiv:** [1710.03641](https://arxiv.org/abs/1710.03641)
- **Summary:** Casts continuous adaptation to nonstationary/adversarial environments as a learning-to-learn (gradient-based meta-learning, MAML-style) problem; introduces **RoboSumo**, a competitive multi-agent benchmark, and "iterated adaptation games"; shows meta-learners adapt faster than reactive baselines and are the fittest in a competing population.
- **Experimental setting:** Nonstationary six-leg locomotion (15 leg-pair environments, 3 held out) and RoboSumo (simulated 3D wrestling-like competitive agents); not drones/quadrotors.
- **Results / conclusions (full text arXiv read, 2026-11-04):** Experiments include nonstationary six-leg locomotion as well as RoboSumo. In locomotion (3 held-out leg pairs) the meta-learned policy is worse in episode 1. By episodes 6–7 the meta-updated policies "perform much better than the rest", while PPO-tracking does not beat no adaptation (Fig. 4). In RoboSumo iterated games against improving opponents, meta-learners usually raise their win rate within about 100 episodes (3 per round) while baselines often degrade (Fig. 5). Above 50 episodes per round, tracking catches up and the meta-learner stays roughly flat (Fig. 6). Meta-learners had the highest TrueSkill in every category of a 105-agent population (Fig. 7) and "end up dominating" a 1050-agent, 10-generation evolution (Fig. 8). App. E.1 Table 3 gives average win rates (95 % CI) over the last 25 rounds: meta-updates are best in 8 of 9 agent/opponent pairs (exception: Spider vs. Ant, PPO-tracking 48.0 % vs. 45.3 %), with meta-learner win rates mostly below 50 % (34.6–52.0 %); App. E.2 Table 4 gives TrueSkill ranks. Conclusion: meta-learned adaptation is the most efficient in the few-shot regime, and meta-learners are "the fittest". The authors' stated limitations: one-step-ahead meta-update; second-order gradients slow training "by an order of magnitude"; "unlikely to work with sparse rewards"; extra data is not used well (Fig. 6); and training diverges under large opponent shifts (only ≤ 4 opponent update steps per iteration gave non-negative reward, App. Fig. 11).
- **Stated limitations:** Authors' own: see the Results bullet. For this thesis: gradient-based meta-adaptation needs several interaction episodes per update (slower than a single-pass RLS update); no CTBR/aerial dynamics.
- **Relation to candidates:** Conceptual ancestor of **(B)**'s "online adaptation to opponent" framing, but via MAML-style gradient meta-learning rather than linear-basis+RLS; useful contrast case (gradient-based vs. basis+RLS-based online opponent adaptation) for your related-work discussion.

### Friedman & Dieng, 2023 — The Vendi Score
- **Title:** *The Vendi Score: A Diversity Evaluation Metric for Machine Learning*
- **Authors:** Dan Friedman, Adji Bousso Dieng
- **Venue/Year:** *Transactions on Machine Learning Research* (TMLR), 07/2023 (OpenReview g97OHbQyk1; stated on the arXiv v2 PDF title page)
- **arXiv:** [2210.02410](https://arxiv.org/abs/2210.02410)
- **Summary:** Defines the Vendi Score as the exponential of the Shannon entropy of the eigenvalues of a user-defined similarity matrix (closely related to DPP machinery — it is essentially an effective-rank/diversity functional of the same kernel matrix a DPP would use). Reference-free, domain-agnostic diversity metric.
- **Experimental setting:** Molecular generative modeling, image/text generative models, GAN mode-collapse diagnosis (no RL/self-play).
- **Results / conclusions (full text arXiv v2 read, 2026-11-04; venue confirmed: *TMLR* 07/2023, OpenReview g97OHbQyk1):** On MOSES molecule generators, the HMM has one of the highest IntDiv scores (0.871 vs. 0.855 for the data) but the lowest VS (250.9 vs. 403.9; App. Table 4), because it generates clusters of near-duplicates (Fig. 3). On StackedMNIST, Self-conditioned GAN and PresGAN cover all 1000 modes yet have lower VS than the data (746.7 and 866.6 vs. 943.7; Table 1). On images, VS agrees with IS/FID/precision/recall on CIFAR-10 and ImageNet 64×64. On LSUN Cat/Bedroom the diffusion model has the best quality but lower VS, and no model matches its training data's diversity (Table 2). For MS-COCO captions, VS ranks beam search < diverse beam search < human, with beam search "effectively" returning "only three distinct responses for every five" (Sec. 4.5). Conclusion: VS is an interpretable "effective number of unique elements" that needs no reference data. It depends on the similarity function and must be paired with a quality metric (Limitations). The authors note the DPP link themselves (Sec. 3), including that the DPP likelihood is 0 with duplicates.
- **Stated limitations:** A diversity *metric for evaluation*, not a *sampling mechanism* — doesn't itself propose using eigenvalues/marginals for sampling.
- **Relation to candidates:** Useful **evaluation tool** for (A) — you could report the Vendi Score of your opponent pool over training as a diagnostic, distinct from your DPP-based *sampling* mechanism (metric vs. mechanism is the key distinction to keep straight in the thesis).

### Kulesza & Taskar, 2012 — DPP survey
- **Title:** *Determinantal Point Processes for Machine Learning*
- **Authors:** Alex Kulesza, Ben Taskar
- **Venue/Year:** *Foundations and Trends in Machine Learning*, Vol. 5, No. 2–3, pp. 123–286 (2012); DOI 10.1561/2200000044 (Crossref)
- **arXiv:** [1207.6083](https://arxiv.org/abs/1207.6083)
- **Summary:** The standard tutorial/reference for DPPs: L-ensemble construction (L = quality·similarity·quality, i.e., your `a·diag(q) S diag(q)` is a direct instantiation of their quality/diversity decomposition), marginal kernel K, marginal inclusion probabilities K_ii, sampling algorithms, k-DPPs, and applications (diverse search results, document summarization, pose diversity).
- **Experimental setting:** Tutorial/survey, which also contains full empirical studies (see Results).
- **Results / conclusions (full text arXiv read, 2026-11-04):** Main conclusion: DPPs give "coherent probabilistic semantics" for diversity, in contrast to "heuristic diversification techniques" (Sec. 7), with "efficient and exact algorithms" for sampling, marginalization and conditioning where MRFs become intractable under negative correlations (Abstract). Finding the mode (MAP) is NP-hard (Sec. 2.4.5). Contributions: the quality/diversity decomposition and dual representation (Sec. 3), learning a quality model with fixed diversity (Sec. 4), k-DPPs, which "can capture distributions that a standard DPP cannot" (Sec. 5), and structured DPPs (Sec. 6). It does contain empirical studies: DUC 2003/2004 summarization (Sec. 4.2.1), image search with human diversity judgments (Sec. 5.3), multiple-pose estimation (Sec. 6.4) and news threading (Sec. 6.6.4). Sec. 7 lists two open questions (concavity of DPP entropy, 7.1; higher-order sums Σ det(L_Y)^p, 7.2) and research directions (7.3), including learning the similarity kernel from data, Hermitian/asymmetric kernels, conditional-independence structure, constrained inference, loopy SDPPs and n-best reranking. Citation threading (Sec. 6.6.3) is also evaluated.
- **Stated limitations:** See the open questions and research directions in the Results bullet (Sec. 7); MAP is NP-hard (Sec. 2.4.5).
- **Relation to candidates:** **This is the exact mathematical toolkit (A) builds on** — your L = a·diag(q)·S·diag(q) construction and the "marginal inclusion probability K_ii" sampling rule is precisely the quality-diversity decomposition and marginal-kernel machinery this monograph formalizes. Cite as the canonical DPP reference, not as prior art for opponent sampling specifically.

---

## 2. Novelty check

### (A) DPP-based opponent sampling (sample ∝ marginal K_ii, bounded against duplicates)

**Closest prior work found:**
- Perez-Nieves et al. 2021 (arXiv:2103.07927) and the Liu/Yao diversity lineage (2106.04958, NeurIPS'22 UDM, 2306.16884) all use **diversity metrics as a training-time regularizer** added to the best-response objective of the *newest* policy; they do **not** use DPP marginals to define a **sampling distribution over which frozen opponents from the existing pool to train against**. Corrections (2026-11-04, full text): Liu 2021's metric is a convex-hull-projection lower bound, not a DPP, and it **does** re-weight opponent sampling (new policy trains against opponent j ∝ |∂F/∂a_{n+1,j}|, mixed with the meta-strategy) to optimise that regulariser. That is a diversity-derived opponent distribution, but with no quality × similarity kernel and no duplicate bound. Perez-Nieves itself remarks that its diverse best response "is similar in manner to selecting a diverse set of opponents" (Sec. 4).
- Balduzzi et al. 2019 (PSRO_rN, 1901.08106) uses "game-theoretic niching" (rectified Nash weighting) for a related but distinct notion of niche-based opponent weighting — not DPP-marginal sampling.
- Vinyals et al. 2019 (AlphaStar League) and the PFSP/Hedge samplers in Roncero et al. 2026 / Gavin & Bronz 2026 / AgilePE 2026 all weight opponents by **hardness alone** (AlphaStar: polynomial f_hard = (1−x)^p; Roncero: γ-uniform mix + normalised c_j^η; Gavin & Bronz: 50 % latest + 50 % (1−x)²; AgilePE: win-rate-proportional, described only in words). None of these papers states the "duplicates can inflate sampling mass" failure mode; that is the thesis's own argument.
- **Added by the 2026-11-04 citation chase. DPPs over opponent/policy populations already exist (one-off subset selection, not a per-episode sampling rule):**
  - **FormulaZero**: Sinha, O'Kelly, Zheng, Mangharam, Duchi & Tedrake, *Distributionally Robust Online Adaptation via Offline Population Synthesis*, arXiv:2003.03900 (v2 Aug 2020; ICML 2020 venue not checked). Full text read. Replica-exchange MCMC self-play builds a population, the near-optimal tier goes into a DPP with an RBF kernel, and d diverse opponents are drawn **once**, as the opponent set for online bandit adaptation and testing. Quality is a hard filter, not part of L, and there are no K_ii marginals. **Closest robotics precedent (autonomous racing); must cite.**
  - **GTOSP**: Zheng, Zhuang, Betz & Mangharam, *Game-theoretic Objective Space Planning*, arXiv:2209.07758 (v2 Oct 2023). Full text read. A DPP subsamples the Pareto-near-optimal racing population into disjoint sets that play each other to collect self-play data; a one-off subset selection.
  - **ZSC-Eval**: Wang et al., arXiv:2310.05208, NeurIPS 2024 Datasets & Benchmarks. Full text read. A DPP over best-response behaviour features picks the evaluation-partner set; evaluation only, no quality term.
  - **TSDP**: Wang, Han, Zhang, Ran & Ji, *Diverse task sampling for robust MARL in perturbed environments*, Applied Soft Computing 2026, DOI 10.1016/j.asoc.2026.114831. A DPP selects representative training tasks and the paper cites Kulesza and Perez-Nieves. **Unverified (search snippets only, full text 403); the closest "training tasks" analogue, check manually** for quality weighting / marginals.
  - DTGSH (Dai et al., arXiv:2108.07887, PRICAI 2021; abstract via search only): DPP / k-DPP for HER trajectory and goal selection.
- **Diversity-aware or deduplicated sampling distributions over training populations (not DPP-marginal):**
  - **DIPLR**: Li, Varakantham & Li, arXiv:2301.08025, IJCAI 2023. Level replay P = ρ·P_D + (1−ρ)·P_R (Wasserstein-distance diversity rank + regret), motivated explicitly by similar high-regret levels making training redundant. An additive quality + diversity mixture, no duplicate bound.
  - **HSP**: Yu et al., *Hidden-Utility Self-Play*, arXiv:2302.01605, ICLR 2023. States that the pool "often contains a lot of similar policies… Duplicated policies simply slow down training", then greedily keeps the K most diverse and samples uniformly. **Closest statement of the duplicate motivation**; a hard filter with no quality term.
  - **ROMANCE**: Yuan et al., arXiv:2305.05909, AAAI 2023. Behaviour-distance threshold archive of attackers, then fitness-based selection. A hard-threshold cousin of quality × diversity.
  - **Distributed PSRO** (Tang et al., IEEE TNNLS 36(6), 2025): top-K quality truncation that "incorporate[s] diversity estimation directly into the sampling process". **Ambiguous, paywalled, check manually.** Also unchecked: LeMIP "via determinantal diversity" (KBS 2026, DOI 10.1016/j.knosys.2026.116948, no abstract available).
- **Checked, do not anticipate (A):** ROA-Star (NeurIPS 2023; DPP as evaluation only, PFSP for training), DvD (2002.00632), PDO (Jiang et al., ICRA 2024, 2403.11114; determinant diversity in a PBT archive, aerial dogfight testbed), Diverse Auto-Curriculum (2102.07659), MEP (2112.11701; quality-only rank sampling), QDSP/FMSP (Dharna, Lu & Clune, 2507.06466, RLC 2025; random opponents from a novelty-gated archive, 2D car-tag PE game), DNS (2201.13357; k-DPP over Q-ensemble members), all 30 citers of PSD-PSRO.
- No paper found, in any domain, that samples training opponents **per episode ∝ K_ii of a quality-weighted L-ensemble** (L = diag(q)·S·diag(q)) or claims a **bound on the sampling mass of near-duplicates** as a replacement for PFSP/Hedge.
- **Robotics/drone-specific:** no drone/UAV PE or interception paper uses diversity-aware opponent sampling. AgilePE, Roncero and Gavin & Bronz 2607.05939 use hardness-weighted PFSP/Hedge; Gavin, Lacroix & Bronz 2603.16279 trains against the latest opponent only and names opponent diversity as future work; Anoruo et al. 2607.02472 (farthest-point selection for evaluation sets only) and Louette, Sánchez Roncero et al. 2610.03830 (fixed heuristic evader pool) do not. UAV air-combat self-play (P3DP, IEEE TAES 61(5) 2025; NMD-FXP, IEEE TVT 2026; AOS-PPO, Aerospace 12(3):265 2025, abstract only; Wang et al., Front. Neurorobot. 2026; PDO) uses win-rate or meta-solver sampling with diversity at most as a regularizer.

**Verdict (revised 2026-11-04, after the citation chase):**
- **(i) In general: partially novel, with narrower wording.** Do **not** claim to be first to use DPPs for opponent selection (FormulaZero 2020, GTOSP 2022 and ZSC-Eval 2024 select opponent/partner subsets with DPPs). Do not claim to be first to use diversity-aware or deduplicated opponent sampling either (Liu 2021 UBRD, DIPLR, HSP, ROMANCE). The claim that survives: the **specific rule**, i.e. per-episode opponent sampling ∝ the DPP marginal K_ii of a quality-weighted L-ensemble, with a **bounded-duplicate-mass property**, used as a drop-in replacement for PFSP/Hedge. Cite FormulaZero, GTOSP, ZSC-Eval, DIPLR and HSP as the closest prior work. Still to check manually: TSDP, Distributed PSRO and LeMIP (paywalled).
- **(ii) For agile quadrotor PE: novel.** No drone/UAV PE or interception paper found using any diversity-aware opponent sampling; all use hardness-weighted PFSP/Hedge or latest-opponent training. This remains the most defensible novelty claim among the three.
- **Search scope (2026-11-04):** Semantic Scholar Graph API citations (rate-limited, but all seeds returned): Perez-Nieves 84 citers (all titles screened, ~15 abstracts read), PSD-PSRO 30 (all screened), Roncero 4 (all read), AgilePE 0, Kulesza & Taskar 1,485 (regex-filtered to 103 RL-specific titles, all screened; drone/UAV/pursuit regex over all 1,485 gave 8 hits, none relevant). Plus OpenAlex `cites:` (Perez-Nieves 23, PSD-PSRO 8, Kulesza 68 term-filtered; budget ran out before the full Kulesza list), about 11 arXiv-API `abs:determinantal AND …` queries (0–5 hits each, none relevant), and about 22 WebSearch queries. Google Scholar "cited by" was not usable, and Semantic Scholar may lag on Oct–Nov 2026 papers.

### (B) Online opponent adaptation via offline-learned basis + online RLS readout (Neural-Fly-style), vs. ESN+RLS and GRU

**Closest prior work found:**
- O'Connell et al. 2022 (Neural-Fly, arXiv:2205.06908) — the exact method template, but for **wind/aerodynamics**, not an **opponent**; adaptation law is a composite adaptive-control update, not literally RLS.
- Kumar et al. 2021 (RMA) — same "offline-privileged, online-fast-adapt" paradigm, different mechanism (distilled encoder, not basis+linear adaptation).
- Al-Shedivat et al. 2018 — gradient-based (MAML) online adaptation to a competitive opponent in RoboSumo; the clearest **prior work on adapting to an opponent online**, but mechanistically a world apart from basis+RLS (and far slower per-step).
- Yoshimura & Tanaka 2026 (ESN-OA, arXiv:2602.06326) — literally the ESN+online-RLS-readout architecture you propose as a baseline, but applied to **non-stationary dynamics (CartPole/HalfCheetah)**, not opponents.
- Fu et al. 2022 (GSCU) — learns an opponent **embedding** offline and updates a posterior over it online, which is conceptually "learn offline, adapt online to an opponent" but via a **Bayesian posterior over a learned embedding**, not a linear basis + RLS.
- **Added by the 2026-11-04 citation chase (supersedes the 10-26 "no paper found" bullet):**
  - **Same predictor mechanism, non-adversarial agents.** These predate Neural-Fly for this use:
    - Si, Wei & Liu, *AGen: Adaptable Generative Prediction Networks for Autonomous Driving*, IEEE IV 2019, DOI 10.1109/IVS.2019.8814238 (PDF read). The hidden layers of a parameter-shared PS-GAIL network are a fixed feature basis, and RLS-PAA with forgetting updates only the last layer per vehicle, giving a 60 % RMSE drop over a 2.5 s horizon (NGSIM). **Strongly anticipates the predictor mechanism**, but the predicted agents are not adversarial and the prediction does not feed an RL policy.
    - Cheng, Zhao, Liu & Tomizuka, *Human Motion Prediction using Semi-adaptable Neural Networks*, arXiv:1810.00781 (2018; PDF read): an offline NN with RLS-PAA last-layer adaptation for humans in HRI.
    - Same lineage: Abuduweili, Li & Liu, arXiv:1909.05089 (2019; multi-task model + nonlinear RLS-PAA for human trajectory/intent); Wang, Hu & Liu, arXiv:2112.06129 (2021; modified-EKF online adaptation of a driving-behaviour predictor, incl. which layer to adapt).
  - **Learned evader prediction in the RL observation, UAV pursuit-evasion (prior art for the "prediction → policy observation" part):**
    - **OPEN**: Chen et al., *Online Planning for Multi-UAV Pursuit-Evasion in Unknown Environments Using Deep RL*, arXiv:2409.15866, IEEE RA-L 2025 (full HTML read; same group as AgilePE). An LSTM trained supervised on rollouts predicts the evader's next K steps; the prediction is concatenated into the actor and critic observations; zero-shot deployment on 3 Crazyflie 2.1. The evader is a fixed potential-field heuristic, and there is no online weight adaptation. **The closest drone paper, and effectively the GRU baseline.**
    - **PSTO**: Li et al., arXiv:2603.24238 (Mar 2026), IEEE RA-L 2026 per OpenAlex (DOI 10.1109/LRA.2026.3700665; HTML read). An LSTM evader-trajectory predictor feeds an egocentric grid observation of a MARL policy; outdoor swarm tests; potential-field evaders; no online adaptation described.
    - Li, Feng & Zhang, *Safe Cooperative Decision-Making for Multi-UAV Pursuit–Evasion Games via Opponent Intent Inference*, *Sensors* 2026, DOI 10.3390/s26072243 (abstract only, via OpenAlex; publisher 403). Learned intent/subgoal inference plus trajectory prediction produce belief/risk features for a hierarchical SAC policy, targeting "non-stationary evader tactics". Online inference, but no weight adaptation mentioned.
    - Weaker or partial (abstract only): Song & Chen, *Designs* 2026, DOI 10.3390/designs10020026 (LSTM prediction of learning targets used as interception points for PPO); EO-MADDPG, Wang et al., *Aerospace* 13(3):296, 2026, DOI 10.3390/aerospace13030296 (evader-action prediction inside MADDPG; fixed noisy evaders); DPE-PER, Srisomboon & Lee, IEEE IoT-J 2026, DOI 10.1109/JIOT.2026.3667218 (target-direction prediction used only for replay prioritisation); MRPA, Su et al., *Drones* 8(11):655, 2024, DOI 10.3390/drones8110655 (model-based online identification of a proportional-navigation pursuer, then SAC adaptation).
  - **Online-adapted models of a target or opponent, other mechanisms:** Liang et al., *Learning to Guide*, arXiv:1904.06892, IEEE Access 2019 (meta-learned dynamics model adapted online to target manoeuvres inside MPPI guidance; not RL, not a quadrotor); LUCIDGames, arXiv:2011.08152 (UKF estimate of other agents' cost parameters in a game planner); Smyrnakis, arXiv:1301.3347 (EKF tracking of opponents' strategies in fictitious play); GrAMMI, arXiv:2306.11168 (graph-network adversary-state prediction in PE domains, no online weight adaptation); Shen & How, arXiv:2203.07562, IROS 2022 (adaptation against a regularised opponent model).
  - **Checked; no opponent model, prediction or online adaptation:** AgilePE (full HTML; opponent-state history window only), Roncero (opponent identity not in the observation), Gavin, Lacroix & Bronz 2603.16279 and Gavin & Bronz 2607.05939 (both: "trajectory prediction are not addressed"), Papuc et al. 2602.06925, Anoruo et al. 2607.02472 and Geles et al. 2605.22748 (last two abstract only).

**Verdict (revised 2026-11-04):**
- **(i) In general: partially novel.** The predictor mechanism (offline basis shared across agents + online RLS last-layer readout predicting *another agent's* motion) is prior art for humans and drivers (AGen 2019, Cheng et al. 2018), so it is not only a Neural-Fly transplant from wind. No paper was found that applies it to an **adversarial** opponent or feeds the prediction into an **RL policy**. Defensible wording: "first application of basis + RLS online adaptation to an adversarial opponent, as an input to an RL policy". Cite AGen and Cheng et al.
- **(ii) For agile quadrotor PE: novel only in a narrowed form.** A learned evader prediction in the RL observation is already published for UAV PE (OPEN, RA-L 2025, on Crazyflies; PSTO, RA-L 2026). The novelty must rest on the combination: (1) **online adaptation at deployment** (RLS readout); (2) adaptation to **learned / self-play, population-diverse opponents** rather than one heuristic evader; (3) a **multi-task basis across evaders** (optionally domain-adversarial); (4) the **1v1 agile CTBR** setting. The GRU baseline is close to the OPEN/PSTO predictors, which justifies including it. ESN-OA remains the right ESN+RLS baseline.
- **Search scope (2026-11-04):** Semantic Scholar returned HTTP 429 throughout and gave no data. Forward citations came from OpenAlex instead: 292 citers of Neural-Fly (plus OpenCitations cross-check), 252 of Al-Shedivat 2018, 1 of ESN-OA (not relevant), 24 of OPEN, and 0 for Roncero, AgilePE and both Gavin papers (OpenAlex coverage of 2026 papers likely lags, so their forward citations are effectively unchecked beyond web search). This was combined with about 15 WebSearch queries and 8 arXiv-API abstract-field queries. MDPI/Springer full texts were not accessible (403/login), so the mechanisms of the abstract-only items above are from abstracts.

### (C) Bayesian opponent identification over prototype/medoid evaders + conservative fallback

**Closest prior work found:**
- Fu et al. 2022 (GSCU) — **the closest prior work overall.** Already does online Bayesian-style posterior inference over opponent identity (via a continuous learned embedding, not discrete medoids) **and** an explicit conservative-fallback-when-uncertain mechanism (greedy-vs-conservative adversarial-bandit switch), evaluated in part on a **Predator-Prey** (pursuit-evasion-flavored) MAPE domain.
- Rosman, Hawasly & Ramamoorthy 2016 (Bayesian Policy Reuse) — the foundational "library of policies + Bayesian online selection" framework that both GSCU and your BayesID descend from, but for task instances, not adversaries.
- Yang, Hao, Meng, Zhang, Zheng & Zheng, *Towards Efficient Detection and Optimal Response against Sophisticated Opponents* (method: **Bayes-ToMoP**), IJCAI 2019, pp. 623–629, DOI 10.24963/ijcai.2019/88, arXiv:1809.04240 (verified from the full PDF and Crossref, 2026-11-04; an AAMAS 2019 extended abstract carries the "Bayes-ToMoP" title). Extends BPR with theory-of-mind reasoning to detect stationary, switching and BPR-based opponents and to learn a new response for an unseen opponent online; Theorem 1 guarantees detection for in-library opponents; a DQN variant is included. Domains: 7×7 grid **soccer** and 7×7 grid **"thieves and hunters"** (a pursuit-like catching game; the paper does not call it predator-prey), with 6 / 24 hand-designed opponent policies. Deep soccer win rates (Table 1): 99.82 % vs. a Bayes-ToMoP0 opponent (BPRs 49.78 %, DRON 74.75 %, Deep BPR+ 71.57 %), 98.48 % vs. a switching opponent, and 98.21 % vs. a stationary-switching opponent (slightly below BPRs 99.37 % and Deep BPR+ 99.49 %). Two-player, tabular/DQN, no fallback-to-generalist mechanism.
- **Added by the 2026-11-04 citation chase. Discrete-library opponent ID with a library-conditioned pursuer already exists in (simplified) UAV PE:**
  - Kalanther, Bostwick, Maheshwari & Sastry, *Evader-Agnostic Team-Based Pursuit Strategies in Partially-Observable Environments*, arXiv:2511.05812 (Nov 2025, "in submission"; full text read). 2 pursuer UAVs vs. 1 evader in a grid urban simulator. Offline DRL builds a level-k library of pursuer best responses; online, a supervised classifier identifies the library evader and the paired best response is deployed. Non-Bayesian, no fallback, grid world, library from level-k rather than clustering. **Partially anticipates (C).**
  - Kalanther, Bharvirkar, Bostwick, Maheshwari & Sastry, *Orchestrating Level-K Policies Against Unknown Opponents in Partially-Observable Dynamic Games*, arXiv:2610.04937 (Oct 2026; full text read). Same grid PE setting. Uses "idealized Bayesian posterior → deploy paired response" as a **baseline** (Eq. 3) and argues against it ("misalignment between opponent identification and return"). An RL-trained orchestrator beats offline and on-policy classifiers and picks the paired expert "only intermittently". **Directly challenges (C)'s identify-then-respond design; must cite and answer**, together with GSCU's argument that a continuous embedding beats a categorical posterior.
  - Guan, Maity, Kroninger & Tsiotras, ACC 2021, DOI 10.23919/ACC50511.2021.9483152, arXiv:2003.06954. Aerial PE in a stochastic wind field on a grid MDP; level-k hierarchy plus online maximum-likelihood inference of the opponent's level, then the matching countermeasure. No RL, no agile dynamics, no fallback. Partial.
  - Li, Feng & Zhang, *Sensors* 26(7):2243, 2026 (see (B) and the reference list; full text via Europe PMC PMC13075325). 2v2 planar double-integrator UAV PE with obstacles. A variational model infers a categorical posterior over 4 **hand-designed** evader modes (Markov switching) plus a Gaussian subgoal. Belief/risk features drive a hierarchical multi-agent SAC whose high level outputs a continuous sigmoid gate over 4 maneuver templates, with a QP safety layer. One fixed evader policy in training and evaluation (no unseen opponents). vs. CBC-TP Net, OPEN, RGATD3, NAGC and MSMAR-RL over 10 seeds: about +5–7 % normalized return, about 22–25 % fewer pursuer losses. **Partial:** a discrete opponent-mode belief continuously gating the pursuer, in UAV PE, but scripted modes, 2D point-mass and no generalist fallback.
  - Adjacent, abstract only: Kokolakis, Kanellopoulos & Vamvoudakis, ACC 2020, DOI 10.23919/ACC45564.2020.9147737 (UAV team learns an evader's level of intelligence; level-k via RL); Su et al., MRPA, *Drones* 8(11):655, 2024 (roles reversed: evader identifies a proportional-navigation pursuer); Ren et al., *Defence Technology* 27:308–317, 2023, DOI 10.1016/j.dt.2022.10.008 (dynamic Bayesian network infers tactical intention in multi-UAV air combat); Shima, Oshman & Shinar, JGCD 2002, DOI 10.2514/2.4961 (classical multiple-model Bayesian filter bank over discrete evasive-maneuver hypotheses in missile interception; worth one related-work sentence).
- **GSCU/BPR descendants outside robotics (abstracts read):** EAD, Zhu et al., IEEE ToG 2025, DOI 10.1109/TG.2025.3579719 (GSCU-like embedding + Bayesian belief + Exp3.P switch; Kuhn poker and grid predator-prey); Bayes-Lab, Wang et al., *Applied Sciences* 2022, DOI 10.3390/app12146953 (predator-prey); BPR-VCA, Fu et al., *Neural Networks* 199:108755, 2026, DOI 10.1016/j.neunet.2026.108755.
- **Closest to the cluster-based library step:** Lv, Zheng & Hao, *Intelligence & Robotics* 2(2):168–179, 2022, DOI 10.20517/ir.2022.09 (contrastive clustering of opponent-policy representations, soccer; no medoids, no drones).
- **Checked, no opponent ID:** EO-MADDPG (action prediction only), AT-Drone arXiv:2502.09762 (Crazyflie pursuit adapting to unseen *teammates*), Görür et al. AAMAS 2019 (Bayesian selection over human types for collaborative robots), Gavin et al. 2603.16279, Anoruo et al. 2607.02472, van Hamersveld et al. 2610.37470, Ryou et al. 2502.14231.
- No paper found that combines agile 3D / CTBR quadrotor dynamics, **learned behaviour-cluster medoid prototypes**, and a **confidence-weighted blend to a generalist**.

**Verdict (revised 2026-11-04):**
- **(i) In general: not novel, more firmly than before.** BPR 2016 → Bayes-ToMoP 2019 → GSCU 2022 → EAD 2025, and two papers (GSCU; Kalanther et al. 2026) argue explicitly against identify-then-deploy-paired-response. The remaining points of departure are the **medoid / behaviour-cluster prototype construction** and the **continuous confidence-weighted blend to a generalist**; these are modest.
- **(ii) For agile quadrotor PE: novel only in a narrowed form.** Discrete-library opponent ID with library-conditioned pursuers has been done in simplified UAV PE (grid world: Kalanther 2025/2026; level-k aerial grid MDP: Guan 2021; 2D belief-gated scripted modes: Li et al. 2026). State the domain claim as "agile CTBR quadrotor PE with learned behaviour-cluster prototypes and a confidence-weighted generalist fallback", **not** as "first Bayesian opponent ID for UAV PE". The thesis should also answer the GSCU / Kalanther 2026 critique, e.g. with an embedding or return-driven orchestrator baseline, or an argument for why posterior-weighted blending avoids the misalignment.
- **Search scope (2026-11-04):** forward citations of GSCU (Semantic Scholar 19; not indexed in OpenAlex), BPR (Semantic Scholar 92, OpenAlex 70, OpenCitations 53; about 100 unique) and Bayes-ToMoP (Semantic Scholar 12 + 47 across the AAMAS and IJCAI records, OpenAlex 42, OpenCitations 31; 56 unique). All titles were screened, and abstracts were read or keyword-filtered for robotics / UAV / pursuit / interception / predator-prey. Plus 13 WebSearch queries and Crossref title lookups. arXiv-API abstract queries returned almost nothing and were treated as unreliable. Not accessible: MDPI pages (403), a KBS 2025 multi-UAV Bayes-ToMoP citer (Cheng et al., DOI 10.1016/j.knosys.2025.114534) and an EAAI 2023 BPR citer (title-only screens). Google Scholar and ResearchGate were not tried.

### Novelty summary table

| Candidate | Core mechanism | Closest general prior work | Novel in general? | Novel for agile quadrotor PE? |
|---|---|---|---|---|
| **(A) QD-DPP opponent sampling** | Sample opponents ∝ DPP marginal K_ii, L = quality·RBF-kernel·quality | Diversity as a best-response regularizer: Perez-Nieves 2021, Liu 2021/2022, Yao 2023. DPP subset selection over opponent/partner populations: FormulaZero 2020, GTOSP 2022, ZSC-Eval 2024. Diversity-aware / deduplicated sampling: Liu 2021 UBRD, DIPLR 2023, HSP 2023, ROMANCE 2023. Hardness-only samplers: PFSP/Hedge (Vinyals 2019, Roncero 2026, AgilePE 2026, Gavin & Bronz 2026) | **Partially**: DPP opponent selection and diversity-aware sampling exist; the per-episode K_ii rule with bounded duplicate mass as a PFSP/Hedge replacement was not found (2026-11-04 chase). TSDP, Distributed PSRO and LeMIP still to check manually | **Yes**: no drone/UAV PE or interception paper uses diversity-aware opponent sampling |
| **(B) NF-OA (basis + online RLS opponent prediction)** | Offline multi-task/DAIML basis φ across evaders + online RLS on last layer θ, prediction fed to the pursuer's observation | Same predictor mechanism for humans/drivers: AGen 2019, Cheng et al. 2018 (non-adversarial). Same mechanism for wind: Neural-Fly 2022. Learned evader prediction in the RL observation of UAV PE: OPEN (RA-L 2025, Crazyflie), PSTO (RA-L 2026). ESN+RLS for dynamics: ESN-OA 2026 | **Partially**: first found for an *adversarial* opponent feeding an *RL policy* | **Narrowed yes**: only the combination (online RLS adaptation at deployment + learned/population opponents + multi-task basis + 1v1 agile CTBR) is new |
| **(C) BayesID (Bayesian ID over prototype/medoid evaders + conservative fallback)** | Bayesian posterior over a small discrete medoid library; confidence-weighted fallback to a generalist | GSCU 2022, BPR 2016, Bayes-ToMoP 2019, EAD 2025. UAV PE with discrete-library ID: Kalanther 2025/2026 (grid; 2026 argues against identify-then-respond), Guan 2021 (level-k aerial grid MDP), Li et al. 2026 (2D belief-gated scripted modes) | **No** | **Narrowed yes**: only with agile CTBR dynamics + learned cluster-medoid prototypes + confidence-weighted generalist blend; weakest of the three |

### Additional 2025–2026 papers to consider citing (adaptive/opponent-modelling quadrotor pursuit-evasion/interception)

- **AgilePE**, Tang et al. 2026, arXiv:2608.14135: directly in-scope (self-play + PFSP for agile UAV PE), **must-cite**.
- **Roncero et al. 2026 (AMSPBH)**, arXiv:2506.02849: the most direct baseline/competitor (Hedge-sampled population training for 1v1 quadrotor PE), **must-cite**.
- **Gavin & Bronz 2026**, ICUAS 2026, DOI 10.1109/ICUAS69441.2026.11598610 (arXiv:2607.05939): MAPPO+PFSP, CTBR, 3v1 interception; CTBR/PFSP background.
- **Gavin, Lacroix & Bronz 2025**, arXiv:2603.16279 (AMIAD 2025): 1v1 CTBR interception, latest-opponent training; documents the forgetting / overfitting motivation for (A), (B) and (C). See section 1.
- **OPEN**, Chen et al., arXiv:2409.15866, IEEE RA-L 2025: learned (LSTM) evader prediction in the RL observation on Crazyflies. **Must-cite for (B)** (closest drone prior art; the 10-26 note that it is "not opponent-modelling" was wrong).
- **PSTO**, Li et al., arXiv:2603.24238, IEEE RA-L 2026: LSTM evader-trajectory predictor feeding a MARL policy. Cite for (B).
- **Li, Feng & Zhang 2026**, *Sensors* 26(7):2243, DOI 10.3390/s26072243: opponent-intent inference with belief-gated hierarchical SAC in UAV PE. **Verified 2026-11-04** (full text via Europe PMC). Cite for (B) and (C).
- **Kalanther et al. 2025 / 2026**, arXiv:2511.05812 and arXiv:2610.04937: discrete-library opponent ID in grid UAV PE; the 2026 paper argues against identify-then-respond. **Must-cite for (C).**
- **FormulaZero** (arXiv:2003.03900) and **GTOSP** (arXiv:2209.07758): DPP selection of opponents from a quality-filtered population in autonomous racing. **Must-cite for (A).**
- **AGen** (IEEE IV 2019) and **Cheng et al.** (arXiv:1810.00781): offline basis + online RLS last-layer prediction of other agents. **Must-cite for (B).**
- **EO-MADDPG**, Wang, Wang, Bai, Ma, Sun & Li, *Aerospace* 13(3):296, 2026, DOI 10.3390/aerospace13030296 (verified from the Crossref abstract; full text not read). MADDPG + expert knowledge + an evader-action predictor trained against a fixed evader; 99.9 ± 0.3 % (3v1) and 97.5 ± 1.4 % (3v2) success vs. MADDPG 72.7 ± 6.0 % and 64.4 ± 34.4 %, 5 seeds. Optional; no opponent ID. A likely precursor is SSRN 2024, DOI 10.2139/ssrn.4853581 (inferred from title and authors).
- **"Autonomous Decision Making for UAV Cooperative Pursuit-Evasion Game with RL"**, arXiv:2411.02983 (2024): optional broader-context citation, not opponent-modelling-specific.

### (D) Open gaps relative to the three reference papers (added 2026-11-09, after F2-F7)

Ranked by novelty vs. AgilePE / Roncero / Gavin and by support from our own data (numbers from
[RESULTS.md](../RESULTS.md), 0.10 m unless noted). "Not found" below means not found in the ~30-query check of
[diversity_quality_2026-11-08.md](diversity_quality_2026-11-08.md) and the 11-04 citation chase, not a proven gap.

1. **Exploitability-aware self-play (strongest gap).**
   - The papers: Roncero reads the "limited improvement from additional best-response training" as convergence; Gavin,
     Lacroix & Bronz say training is "approaching a Nash equilibrium" without measuring exploitability; AgilePE says PFSP
     "drives robust policy equilibria". None measures exploitability or tests held-out learned lineages.
   - Our data: every self-play pursuer is ~0.94-0.98 exploitable by a fresh PPO exploiter, almost only by luring it
     into a floor crash (F2c, F3, F5, F6), for every sampler (uniform, Hedge, PFSP, QD-DPP), diversity term (F6), opponent
     model (F5) and the floor penalty / 0.15 m training radius (F7).
   - Candidate method: an exploiter-augmented pool (AlphaStar main exploiters; FXP counter-population; optionally
     Minimax-Exploiter shaping), i.e. evaders trained only to beat the current pursuer added to the Hedge pool. Not
     found applied to quadrotor PE.
2. **Dynamics-aware pursuit (closing the floor-lure exploit at its source).**
   - The lure works because the low thrust-to-weight Crazyflie cannot pull out of a dive. Roncero only scales the
     capture bonus by closing speed (v_safe = 2 m/s); Gavin and AgilePE do not address it.
   - Our data: a reward penalty does not fix it (F7 floor_r15: pursuer crash 0.970 vs its exploiter).
   - Candidate: a hard constraint (action filter limiting descent rate given altitude and available thrust, or a
     time-to-floor observation), evaluated by exploitability. Combines naturally with gap 1.
   - Episode-ending rules matter too (section 1, "Episode endings and crash-luring", 2026-11-09): Gavin's rules give
     the evader the same lure incentive as ours, and their tables show crash-heavy evader wins; Roncero's
     truncation-based endings remove the incentive (thesis F8 tests whether that removes the luring).
3. **Opponent-intent estimation under realistic sensing.**
   - The papers: Roncero and both Gavin papers use memoryless MLPs with perfect state; Gavin, Lacroix & Bronz blame the
     missing history for the hovering-evader failure; Roncero lists online adaptation as future work; AgilePE uses a
     temporal window and FOV masking but no explicit opponent model.
   - Our data: the opponent's acceleration is the largest effect in the thesis (F1); a learned estimate is
     noise-robust where finite differencing collapses (A3c, F4); in self-play it helps on held-out evaders (F5 clean
     +0.091 / +0.128) but online RLS adds nothing.
   - Prior art: **OPEN** (Chen et al., RA-L 2025) already feeds a learned evader prediction to an RL pursuer on
     Crazyflies, and PSTO (RA-L 2026) does the same with MARL, so learned opponent prediction is not new in itself. The
     open part is estimation under AgilePE-style partial observability (FOV limits, occlusion) and its effect in
     self-play and on exploitability.
4. **Capture definition and evaluation rigour (evaluation contribution, no new runs).**
   - Roncero (and the thesis) capture at 0.1 m centre-to-centre, but Crazyflie bodies can touch at up to 0.145 m. At
     0.15 m many gains shrink (F3 Hedge - uniform held-out RL +0.100 -> +0.022) and exploitability against the 0.10 m
     exploiters drops (a lower bound; F7 shows exploiters trained under 0.15 m rules still fully exploit).
   - Seeds: Roncero 3, AgilePE and Gavin none reported; F1 shows per-seed bimodality, and single-seed effects of
     +/- 0.1-0.3 occur in our data.
   - Candidate: a robustness protocol (held-out learned lineages, fresh exploiters, capture-radius sensitivity, seed
     counts).
5. **Not pursued here.**
   - Real hardware: Roncero has 42 Crazyflie flights, the thesis none. A weakness of the thesis, not a gap in the
     papers.
   - Multi-pursuer play (Gavin 3v1) and obstacles (all three are obstacle-free in their quantitative results): too far
     from the current setup.

### (E) Improving generalization to held-out and cross-campaign evaders (citation chase, 2026-11-15)

Scope: pursuer policy trained via alternating-stage iterated best-response self-play (PPO, Isaac
Sim/OmniDrones, 13 or 26 frozen stages @ 50M frames, stored evader population, learned
evader-acceleration estimator + online RLS readout). Target gaps: **G1a** = strong held-out
evaders trained under different rules/rewards/strategies (measured gap −0.10); **G1b** =
cross-campaign evaders, same self-play rules, different seed (measured gap +0.03, i.e. pursuer
still weak relative to in-population exploitability gains). PSRO-family diversity anchors
(Balduzzi 2019, Perez-Nieves 2021, Liu 2021/2022, PSD-PSRO Yao 2023, AlphaStar league) are
assumed already covered in your notes; they are cited here only where a specific mechanism
(PFSP, league-exploiter roles) is the operative lever, not re-summarized.

All entries were produced by four parallel research passes, each required to verify every paper
against a primary source (arXiv abstract page, OpenReview/PMLR/NeurIPS/ICLR proceedings, ACM/IEEE/
Nature/MDPI publisher page, or OpenAlex) via direct fetch, not search snippets alone. Verification
status is given per entry. A small number of entries could not be fetched directly (bot-walls /
403s) and are marked **partially verified** (metadata corroborated by an independent secondary
search) or **unverified** (flagged, do not cite without a manual check).

#### Fit to this pipeline (cross-checked against RESULTS.md, 2026-11-15)

The research pass did not know which mechanisms the thesis already tested. Its per-paper entries and shortlist are kept as written below; read them with these corrections.

- **Shortlist items 1-2 (PFSP / population sampling) are already in the pipeline.** v2d (F11) and the F14 campaigns sample both roles from the whole stored population with Roncero's Hedge sampler (`SAMPLERS["hedge"]` in `scripts/experiments/thesis/exps/f3_sampler_sp.py`, used by `f11_v2_sp.py`). F3 measured Hedge vs uniform held-out RL +0.100 and QD-DPP vs uniform -0.041, and hard-only PFSP gave the most specialised pursuer (other campaigns' evaders -0.226). More sampler tuning inside one campaign's population is unlikely to close G1b, and hard-only weighting may widen it.
- **Item 7 (several independent attackers) is already the F14 protocol** (two fresh 100M exploiters per pursuer, plus the strong held-out set and cross-campaign evaders). Keep it as measurement discipline; it is not a new intervention.
- **Item 3 (reward/rule randomization) can leak the G1a test set.** The strong held-out set is defined by different rules and rewards. Randomized training evaders must exclude the held-out rule families (leave-one-rule-family-out), or a G1a gain is test-set leakage.
- **Item 4 (cross-seed pooling, FCP-style) is the most direct untried G1b lever.** The Stage E seeds (0/100/200) already store separate evader populations, so a pooled-population continuation needs no new infrastructure. It must then be scored on evaders from a new, unpooled seed, because the current cross-campaign evaders become training opponents.
- **Model soups (3.9) do not apply across our seeds.** The campaigns are independently initialised; use policy distillation (3.8) for cross-seed consolidation.
- **MAESTRO (2.9, item 8) is the closest UED formulation, not a validated method for this domain.** Its experiments use LaserTag (grid) and MultiCarRacing, not continuous quadrotor control; "precisely MAESTRO's target setting" overstates the match.
- **Schedule length alone did not close G1a.** Going from 13 to 26 stages moved the RLS strong-set component only from -0.131 to -0.103 at 0.10 m (F14 Stage C vs Stage E). This points at population composition rather than more stages.

**Revised order for the thesis:** (i) cross-seed pooled-population continuation, scored on a fresh seed's evaders (G1b); (ii) leave-one-rule-family-out reward-randomized evader pool (G1a); (iii) plasticity diagnostics per stage, then L2 / partial reset if they show loss (both, cheap); (iv) distil the three seed pursuers into one (G1b, no new self-play); (v) learned opponent embedding feeding the RLS readout (both); (vi) MAESTRO-style joint curriculum as future work.

#### Family 1 — Opponent-population construction

Anchors re-cited only for mechanism-linking (not re-summarized): AlphaStar league/PFSP (Vinyals
et al. 2019), Balduzzi 2019, Perez-Nieves 2021, Liu 2021/2022, PSD-PSRO (Yao 2023).

##### 1.1 Fictitious Self-Play (NFSP)
- **Citation**: Heinrich, J., Silver, D. (2016). *Deep Reinforcement Learning from Self-Play in Imperfect-Information Games.* arXiv:1603.01121.
- **Link**: https://arxiv.org/abs/1603.01121 — **Verified**.
- Summary: combines fictitious play with DRL; agent keeps a best-response network and a supervised "average-strategy" network; mixes both at play time; validated on Leduc/Limit Hold'em.
- Mechanism: implicit population via an averaged historical-strategy network rather than a stored discrete set of opponents.
- Application: **G1b** — an NFSP-style average-policy head alongside the pursuer could reduce cycling/overfitting to the most recent frozen evader stage, by giving the pursuer an explicit "mixture of my own past best responses" target. Weak fit for **G1a** (assumes same game structure).
- Cost: **Medium** (second network + reservoir buffer of past action distributions).

##### 1.2 NeuPL — Neural Population Learning
- **Citation**: Liu, S., Marris, L., Hennes, D., Merel, J., Heess, N., Graepel, T. (2022). *NeuPL: Neural Population Learning.* ICLR 2022.
- **Link**: https://arxiv.org/abs/2202.07415 — **Verified**.
- Summary: represents an entire population as one conditional network f(obs, population-id); new members trained as approximate best responses while sharing weights/skills with prior members; proves population best-responses are not lost as the population grows.
- Mechanism: single shared conditional policy, conditioning index = opponent identity.
- Application: **G1b** (evaders share task structure across seeds, so conditioning transfers well); weaker for **G1a** if reward/rule structure differs enough that conditioning does not transfer.
- Cost: **High** (architecture redesign + joint multi-opponent training loop).

##### 1.3 Simplex-NeuPL
- **Citation**: Liu, S., Marris, L., Hennes, D., et al. (2022). *Simplex Neural Population Learning: Any-Mixture Bayes-Optimality in Symmetric Zero-sum Games.* ICML 2022.
- **Link**: https://arxiv.org/abs/2205.15879 — **Verified**.
- Summary: extends NeuPL so the conditional network learns best responses to any mixture (simplex point) over the population, not just discrete members; near-Bayes-optimal under opponent-identity uncertainty.
- Mechanism: continuous mixture-weight conditioning, trained against sampled simplex mixtures.
- Application: **G1a** — train the pursuer conditioned on mixture-weights over a stored evader archive that explicitly spans different reward/rule "species," optimizing robustness to any convex combination (closer to the unknown "distance" of a held-out evader from the training mixture). **G1b** gives a continuous way to interpolate robustness across same-rule seeds.
- Cost: **High** (mixture-conditioning architecture + mixture-sampling curriculum).

##### 1.4 Anytime PSRO
- **Citation**: McAleer, S., Wang, K., Lanier, J.B., Lanctot, M., Baldi, P., Sandholm, T., Fox, R. (2022). *Anytime PSRO for Two-Player Zero-Sum Games.* AAAI-22 RL-in-Games Workshop.
- **Link**: https://arxiv.org/abs/2201.07700 — **Verified**.
- Summary: fixes PSRO/Double-Oracle's non-monotonic exploitability by solving, each iteration, for the restricted-population distribution that minimizes exploitability against the full unrestricted game, guaranteeing monotonic decrease.
- Mechanism: regret-matching meta-strategy solver over an explicit growing population (classic PSRO structure retained).
- Application: usable as the **meta-strategy solver** layered on your existing stored-evader population without changing network architecture — track a payoff matrix between stored pursuer/evader checkpoints each stage and run regret-matching. Helps both regimes by making the *sampling distribution* over stored evaders provably non-regressive.
- Cost: **Medium** (payoff-matrix bookkeeping + solver; no architecture change).

##### 1.5 Self-Play PSRO (SP-PSRO)
- **Citation**: McAleer, S., Lanier, J.B., Wang, K., Baldi, P., Fox, R., Sandholm, T. (2022). *Self-Play PSRO: Toward Optimal Populations in Two-Player Zero-Sum Games.* arXiv:2207.06541.
- **Link**: https://arxiv.org/abs/2207.06541 — **Verified**.
- Summary: at each iteration adds an approximately-optimal *stochastic* policy (not only a deterministic best response) to the population; converges in far fewer iterations than APSRO/PSRO on benchmark zero-sum games.
- Mechanism: augments population growth with a stochastic, mixture-approximating policy trained via self-play against the meta-distribution.
- Application: could accelerate population growth in your alternating-stage pipeline — add one stochastic exploiter network per stage (approximating a distribution over counter-strategies) instead of only one deterministic evader, reaching broader coverage per costly Isaac Sim round. Helps **both** regimes.
- Cost: **Medium** (secondary PPO policy per stage with higher entropy/mixture target).

##### 1.6 Population Based Training (PBT)
- **Citation**: Jaderberg, M., Dalibard, V., Osindero, S., Czarnecki, W.M., et al. (2017). *Population Based Training of Neural Networks.* arXiv:1711.09846.
- **Link**: https://arxiv.org/abs/1711.09846 — **Verified**.
- Summary: asynchronous joint optimization of weights and hyperparameters across a population; periodic exploit (copy from better performer) / explore (perturb hyperparameters).
- Mechanism: diversity across parallel training runs differing in hyperparameters, not strategy per se.
- Application: **G1b** — run a PBT population of evader-training jobs with randomized hyperparameters (LR, entropy coef, reward-shaping weights) and periodic exploit/explore to produce a naturally diverse cross-seed evader population instead of hand-launching independent seeds.
- Cost: **Medium** (orchestration/scheduling only, no architecture change).

##### 1.7 MAP-Elites
- **Citation**: Mouret, J.-B., Clune, J. (2015). *Illuminating Search Spaces by Mapping Elites.* arXiv:1504.04909.
- **Link**: https://arxiv.org/abs/1504.04909 — **Verified**.
- Summary: discretizes a behavior-descriptor space into niches; keeps the single best ("elite") solution per niche; produces a map of performance across behavioral dimensions.
- Mechanism: quality-diversity archive indexed by behavior descriptors.
- Application: **G1a** — define behavior descriptors for evaders (evasion aggressiveness, altitude-change frequency, reward-regime if varying rules) and run MAP-Elites to guarantee coverage of qualitatively distinct evasion niches rather than relying on self-play's emergent (often narrow) diversity.
- Cost: **Medium-High** (behavior-descriptor engineering, archive data structure, mutation/selection loop; classic MAP-Elites uses ES/GA, not PPO gradients — needs a gradient-friendly variant, e.g. MEMES).

##### 1.8 Reward Randomization for Policy Gradient (RPG, Tang et al.)
- **Citation**: Tang, Z., Yu, C., Chen, B., Xu, H., Wang, X., Fang, F., Du, S.S., Wang, Y., Wu, Y. (2021). *Discovering Diverse Multi-Agent Strategic Behavior via Reward Randomization.* ICLR 2021.
- **Link**: https://arxiv.org/abs/2103.04564 — **Verified**.
- Summary: samples different reward functions and runs policy gradient under each to escape convergence to one equilibrium; discovers multiple interpretable strategies in temporal trust-dilemma games.
- Mechanism: diversity induced by perturbing the reward function/shaping coefficients across parallel runs.
- Application: **most direct match to constructing your G1a evader population** — systematically randomize evader reward/rule weighting (risk-aversion, energy cost, capture-avoidance margin) across many training runs; retain the diverse set as held-out generalization test evaders (or feed back into pursuer training).
- Cost: **Low-Medium** (parameterize evader reward, sample parameter vectors, launch parallel PPO runs; no architecture change).

##### 1.9 GAME — Generational Adversarial MAP-Elites
- **Citation**: Anne, T., et al. (2025/2026). *Adversarial Coevolutionary Illumination with Generational Adversarial MAP-Elites.* ALIFE 2025 / IEEE Trans. Evolutionary Computation (DOI 10.1109/TEVC.2026.3686956).
- **Link**: https://arxiv.org/abs/2502.01352 (ALIFE preprint; journal DOI above) — **Verified**.
- Summary: coevolves both sides via two alternating MAP-Elites archives; uses a vision-embedding model on rollout video as the behavior descriptor (no hand-crafted descriptors); validated on a multi-agent battle game, soft-robot wrestling, deck-building.
- Mechanism: dual archives, alternating evolution per side, descriptors from learned video embeddings.
- Application: structurally close to your alternating-stage design — replace each stage's single best pursuer/evader with an *archive* of diverse elites per side; targets **both** G1a/G1b without hand-designed descriptors.
- Cost: **High** (dual archives, evolutionary outer loop around PPO, vision-embedding pipeline on Isaac Sim video).

##### 1.10 MEMES — MAP-Elites with Multiple Parallel ES Emitters
- **Citation**: (authors unconfirmed in this pass). *Multiple Hands Make Light Work: Enhancing Quality and Diversity using MAP-Elites with Multiple Parallel Evolution Strategies.* 2023, arXiv:2303.06137.
- **Link**: https://arxiv.org/abs/2303.06137 — **Unverified** (arXiv ID/title corroborated via secondary citations only; author list not independently confirmed in this session — fetch the abstract page directly before citing).
- Summary (provisional): multiple parallel ES "emitters" (fitness-seeking vs. novelty-seeking) update a shared MAP-Elites archive.
- Application: if adopting a MAP-Elites evader archive, a more sample/parallel-efficient way to fill it, matching Isaac Sim's GPU-parallel rollout capability.
- Cost: **High** (ES-based, not off-the-shelf PPO-compatible).

##### 1.11 Mix-ME — Quality-Diversity for Multi-Agent Learning
- **Citation**: Ingvarsson, G., Samvelyan, M., Lim, B., Flageat, M., Cully, A., Rocktäschel, T. (2023). *Mix-ME: Quality-Diversity for Multi-Agent Learning.* NeurIPS 2023 ALOE Workshop.
- **Link**: https://arxiv.org/abs/2311.01829 — **Verified**.
- Summary: multi-agent MAP-Elites variant with a crossover operator mixing agents from different archived teams.
- Mechanism: archive of team-solutions, recombination across archived teams.
- Application: relevant mainly if extending to multi-pursuer/multi-evader swarms; could cheaply generate novel evader "teams" from an existing archive. Lower priority for strict 1v1.
- Cost: **Medium-High**.

##### 1.12 Self-Play Survey (organizing reference)
- **Citation**: Zhang, R., Xu, Z., Ma, C., Yu, C., Tu, W.-W., Tang, W., Huang, S., Ye, D., Ding, W., Yang, Y., Wang, Y. (2024, v4 2025). *A Survey on Self-play Methods in Reinforcement Learning.* arXiv:2408.01072.
- **Link**: https://arxiv.org/abs/2408.01072 — **Verified**.
- Summary: taxonomy unifying naive self-play, FSP, population-play, PSRO-style, league training; discusses open challenges (scalability, non-transitivity, generalization). Shares authors with the AgilePE pursuit-evasion paper below.
- Application: scaffolding citation for the thesis's related-work section on self-play families.
- Cost: N/A (survey).

##### 1.13 AgilePE — Autonomous UAV Pursuit-Evasion via Self-Play RL (high priority, directly on-topic)
- **Citation**: Tang, W., Chen, T., Cui, Z., An, B., Chen, J., Zhang, R., Liu, H., Wu, T., Liao, Q., Gao, F., Wang, Y., Yu, C. (2026). *AgilePE: Autonomous UAV Pursuit-Evasion via Self-Play Reinforcement Learning.* arXiv:2608.14135 (under review, submitted Aug 2026).
- **Link**: https://arxiv.org/abs/2608.14135 — **Verified**.
- Summary: agile CTBR-command UAV pursuit-evasion; competitive self-play with **Prioritized Fictitious Self-Play (PFSP) and a diversified opponent pool**, explicitly to reduce policy oscillation/overfitting to the most-recent opponent; sim-to-real pipeline with domain randomization, zero-shot hardware transfer.
- Mechanism: stored pool of historical/diverse opponents, PFSP prioritizes sampling opponents the current policy is weakest against (not just the latest checkpoint) — explicit opponent-sampling-distribution change, no architecture change needed.
- Application: **closest available match to your exact setup** (same control abstraction, same self-play framing). Directly suggests (1) maintaining an explicit diversified evader pool rather than "latest evader only," (2) PFSP-weighted sampling. Strong fit for **G1b**; the sim-to-real domain randomization is an analogy for **G1a** robustness to unmodeled "rules."
- Cost: **Low** (sampling-distribution change over an existing stored evader population, reusing a win-rate/exploitability estimate per evader).

##### 1.14 ReCord — Replay Coordination for Population-Based Training
- **Citation**: Jeon, H.-C., Kim, K.-J. (2026). *ReCord: Replay Coordination for Safe and Robust Population-Based Training in Autonomous Driving.* AI4Good Workshop, ICML 2026.
- **Link**: OpenReview id yf1h6EUF88 — **Unverified** (OpenReview bot-wall blocked direct fetch; title/mechanism corroborated only via secondary search; naming discrepancy "ReCord" vs. "Recorded-PBT" across sources needs resolving before citing).
- Summary (provisional): trains against *frozen, pre-recorded* partner trajectories from a diverse population rather than live reactive co-training, removing the ego policy's reliance on partner adaptation; improves zero-shot coordination with non-reactive/human partners.
- Application: **G1b** — probe/harden generalization by testing/training the pursuer against frozen, non-reactive replays of past evader checkpoints (not just live, co-adapting checkpoints), since the stored evader population was trained *with* the pursuer and may be implicitly exploitable via that co-adaptation.
- Cost: **Low** (data-pipeline change if evader trajectories are already recorded).

##### 1.15 OPEN — adjacent, environment-diversity axis (not opponent-population, cross-reference only)
- **Citation**: Chen, J., Yu, C., Li, G., Tang, W., Ji, S., Yang, X., Xu, B., Yang, H., Wang, Y. (2024/2025). *Online Planning for Multi-UAV Pursuit-Evasion in Unknown Environments Using Deep Reinforcement Learning.* IEEE RA-L 2025.
- **Link**: https://arxiv.org/abs/2409.15866 — **Verified**.
- Summary: evader-prediction-enhanced network for partial observability; an adaptive *environment* generator (not opponent-policy generator) injects scenario diversity; 100% capture rate on unseen scenarios in sim, zero-shot real-quadrotor deployment.
- Application: complementary generalization axis (scenario diversity) to combine with opponent-population methods; by itself does not address unseen-evader-*policy* generalization, since its evader is largely environment/obstacle-driven.
- Cost: N/A to this family; see Family 6 for fuller treatment.

##### Negative result (checked, excluded)
- arXiv:2504.02420, "On learning racing policies with reinforcement learning" (IROS 2025) — **Verified real** but **not relevant**: single-agent RC racing vs. MPC/human baselines with domain randomization, no opponent population or self-play component.

---

#### Family 2 — Robust / adversarial RL

##### 2.1 RARL — Robust Adversarial Reinforcement Learning
- **Citation**: Pinto, L., Davidson, J., Sukthankar, R., Gupta, A. (2017). *Robust Adversarial Reinforcement Learning.* ICML 2017.
- **Link**: https://arxiv.org/abs/1703.02702 — **Verified**.
- Summary: trains protagonist jointly with a destabilizing adversary applying disturbance forces; zero-sum minimax (H∞-control inspired); improves robustness to train/test mismatch on InvertedPendulum/HalfCheetah/Swimmer/Hopper/Walker2d.
- Mechanism: single learned force-disturbance adversary, alternating gradient updates.
- Application: your existing alternating-stage self-play loop *is* essentially this minimax structure already — RARL is mostly a reframing/ablation exercise. Mainly a **G1b** tool (single-adversary robustness to seed variance); weaker for **G1a** since a single adversary is itself exploitable by qualitatively new ones (see 2.2).
- Cost: **Low** (reframing of existing pipeline).

##### 2.2 Robust RL using Adversarial Populations
- **Citation**: Vinitsky, E., Du, Y., Parvate, K., Jang, K., Abbeel, P., Bayen, A. (2020). *Robust Reinforcement Learning using Adversarial Populations.* arXiv:2008.01825.
- **Link**: https://arxiv.org/abs/2008.01825 — **Verified**. Code: github.com/eugenevinitsky/robust_RL_multi_adversary.
- Summary: a *single* RARL-style adversary does not reliably robustify a policy — the resulting protagonist is exploitable by new/unseen adversaries; a *population* of independently-initialized co-trained adversaries, sampled uniformly during training, gives comparable robustness/OOD generalization to domain randomization without DR's hand-designed uncertainty set.
- Mechanism: population of co-trained adversaries, diversity from independent random init only (no explicit diversity regularizer).
- Application: **load-bearing anchor for the thesis's core claim** — literally the mechanism explaining why a pursuer trained against one self-play evader lineage fails on G1a and even G1b: a single adversary (the stored best-response evader) is exploitable by any sufficiently different one. Argues for sampling from a **population** of evaders, not just the latest best response.
- Cost: **Medium** (maintain a live population of N evader checkpoints, sample uniformly per episode during pursuer PPO updates — you likely already store a population, main cost is restructuring sampling).

##### 2.3 Adversarial Policies: Attacking Deep Reinforcement Learning
- **Citation**: Gleave, A., Dennis, M., Wild, C., Kant, N., Levine, S., Russell, S. (2020). *Adversarial Policies: Attacking Deep Reinforcement Learning.* ICLR 2020.
- **Link**: https://arxiv.org/abs/1905.10615 — **Verified**.
- Summary: an attacker acting as a co-player (not perturbing observations directly) creates physically-valid adversarial observations; demonstrated in zero-sum humanoid self-play games against victims specifically self-play-trained to be robust; wins reliably while behaving erratically; induces different victim-network activations than normal opponents.
- Mechanism: train attacker via RL against a frozen victim, black-box, exploiting the victim's narrow self-play opponent distribution.
- Application: this **is** your G1a evaluation methodology — train an "attacker" evader against your frozen final pursuer using a different algorithm/reward than your self-play rules and measure exploitability. Self-play-robust victims can still be exploited; cautions against treating high in-population win rate as true generalization.
- Cost: **Low** as an evaluation protocol; **Medium** if used as a training-time defense (requires an attacker-detection/retraining loop, see 2.4).

##### 2.4 Adversarial Policies Beat Superhuman Go AIs
- **Citation**: Wang, T.T., Gleave, A., Tseng, T., Pelrine, K., Belrose, N., Miller, J., Dennis, M.D., Duan, Y., Pogrebniak, V., Levine, S., Russell, S. (2023). *Adversarial Policies Beat Superhuman Go AIs.* ICML 2023.
- **Link**: https://arxiv.org/abs/2211.00241 — **Verified**.
- Summary: >97% win rate against KataGo at superhuman settings via an adversarial policy that plays Go poorly itself but tricks the victim into blunders; transfers zero-shot to other superhuman engines; the vulnerability **persists even after adversarially retraining KataGo specifically to defend against this attack**.
- Mechanism: iterated-best-response-style adversarial policy search against a frozen victim, discovering off-distribution "trap" states never visited during self-play.
- Application: strong cautionary anchor for **G1a** — even a pursuer unexploitable by its own self-play lineage (and by one round of adversarial fine-tuning against one found exploiter) can harbor exploits found by independently-trained evaders with different search/reward structures. Your G1a protocol should test multiple independent attacker-training runs, not assume one adversarial-retraining pass closes the gap.
- Cost: **Medium-High** to replicate as a rigorous held-out-attacker evaluation (several independent attacker runs with different algorithms/rewards).

##### 2.5 ATLA — Robust RL on State Observations with Learned Optimal Adversary
- **Citation**: Zhang, H., Chen, H., Boning, D., Hsieh, C.-J. (2021). *Robust Reinforcement Learning on State Observations with Learned Optimal Adversary.* ICLR 2021.
- **Link**: https://arxiv.org/abs/2101.08452 — **Verified**. Code: github.com/huanzhang12/ATLA_robust_RL.
- Summary: formalizes State-Adversarial MDPs; proves an optimal state-observation adversary exists for any fixed policy; ATLA alternately trains agent/adversary online via policy gradient (iterated best-response alternation, structurally close to your pipeline); shows recurrent (LSTM) policies are more robust than memoryless MLPs.
- Mechanism: alternating-stage adversarial training where the adversary perturbs observations under an L∞ budget.
- Application: less directly transferable (your "adversary" is a full co-player, not an observation perturber), but the **recurrent-policy robustness finding is directly actionable**: giving the pursuer an LSTM/GRU head tracking evader state history may itself improve generalization to G1a/G1b evaders whose motion statistics differ from the training population, independent of any adversary redesign.
- Cost: **Low** (recurrent architecture swap) for the memory finding; **Medium** if additionally implementing an observation-noise adversary sub-loop.

##### 2.6 PAIRED — Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design
- **Citation**: Dennis, M., Jaques, N., Vinitsky, E., Bayen, A., Russell, S., Critch, A., Levine, S. (2020). *Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design.* NeurIPS 2020.
- **Link**: https://arxiv.org/abs/2012.02096 — **Verified**.
- Summary: introduces Unsupervised Environment Design (UED); critiques domain randomization (no curriculum) and pure minimax env generation (unsolvable worst cases); adds an "antagonist" agent, generator maximizes regret = protagonist return − antagonist return, provably avoiding unsolvable levels and yielding a natural curriculum.
- Mechanism: 3-player game (protagonist, antagonist, environment generator), regret-based generator reward.
- Application: the regret principle generalizes from environment generation to **opponent** generation (formalized by MAESTRO, 2.9). For **G1a**, reinterpret "environment parameters" as evader reward/strategy hyperparameters and run a PAIRED-style generator proposing evader configurations that maximize pursuer-antagonist regret.
- Cost: **High** (third agent + separate generator policy over evader-configuration space + regret computation infrastructure).

##### 2.7 Robust PLR (PLR⊥) — Replay-Guided Adversarial Environment Design
- **Citation**: Jiang, M., Dennis, M., Parker-Holder, J., Foerster, J., Grefenstette, E., Rocktäschel, T. (2021). *Replay-Guided Adversarial Environment Design.* NeurIPS 2021.
- **Link**: https://arxiv.org/abs/2110.02439 — **Verified**.
- Summary: recasts Prioritized Level Replay as UED, unifying with PAIRED under "Dual Curriculum Design"; PLR⊥ stops policy updates on uncurated levels, training only on curated high-regret levels — improves convergence to Nash equilibria and zero-shot OOD transfer vs. plain PLR; provides theoretical robustness guarantees at Nash equilibria.
- Mechanism: replay buffer of levels/opponents prioritized by a cheap regret proxy (e.g. value-loss/GAE magnitude), far cheaper than training a full antagonist.
- Application: **much more implementable than PAIRED** for your setting — maintain a replay buffer of stored evader checkpoints across stages/seeds and prioritize replaying evaders with high current regret/value-loss against the pursuer, rather than uniform or most-recent sampling. Directly addresses **G1b** (surfaces "hard" seeds from past campaigns); partially addresses **G1a** if the evader population already spans reward variants.
- Cost: **Low-Medium** (sampling-weight change via a value-function TD-error regret proxy per stored evader; no antagonist agent needed).

##### 2.8 ACCEL — Evolving Curricula with Regret-Based Environment Design
- **Citation**: Parker-Holder, J., Jiang, M., Dennis, M., Samvelyan, M., Foerster, J., Grefenstette, E., Rocktäschel, T. (2022). *Evolving Curricula with Regret-Based Environment Design.* ICML 2022.
- **Link**: https://arxiv.org/abs/2203.01302 — **Verified**.
- Summary: combines regret-based UED with evolutionary level-editing — mutates/edits existing high-regret levels incrementally rather than generating from scratch; retains Nash-equilibrium robustness guarantees; outperforms PAIRED/PLR without domain-specific heuristics.
- Mechanism: evolutionary "edit" operator applied to high-regret levels sampled from a PLR-style buffer.
- Application: analogous to 2.7 but for evader *strategy parameters* — take highest-regret stored evader checkpoints and apply small parametric mutations (reward-weight tweaks, speed/maneuver-limit edits) to synthesize new, slightly harder/different evaders, building an open-ended curriculum targeting **G1a** more cheaply than training an antagonist from scratch.
- Cost: **Medium** (mutation operator over evader policy/reward parameters + regret-based selection; reuses PLR-style buffer).

##### 2.9 MAESTRO — Open-Ended Environment Design for Multi-Agent RL (single most on-point UED paper)
- **Citation**: Samvelyan, M., Khan, A., Dennis, M., et al. (2023). *MAESTRO: Open-Ended Environment Design for Multi-Agent Reinforcement Learning.* ICLR 2023.
- **Link**: https://arxiv.org/abs/2303.03376 — **Verified**.
- Summary: first UED method explicitly extended to two-player zero-sum multi-agent settings, arguing environment difficulty and co-player strength are interdependent and must be curated jointly; jointly generates adversarial curricula over both environment parameters and co-player policies, maintaining minimax-regret guarantees; outperforms strong baselines on competitive two-player games.
- Mechanism: joint regret-based curriculum over (environment, co-player) pairs — PAIRED/PLR machinery generalized so regret spans a population of co-players as well as environment configs.
- Application: **the single most directly on-point paper in this family** — your 1v1, zero-sum, continuous-control interception task is precisely MAESTRO's target setting. Replaces plain alternating self-play with a joint (env-config, evader-policy) regret curriculum: helps **G1a** (joint curriculum over evader reward/strategy space = "different rules" evaders) and **G1b** (regret-weighted sampling over the existing stored-evader population handles seed variance).
- Cost: **High** (joint environment+co-player regret curriculum, UED-style generator/sampler, regret estimation infra) — largest architectural change in this family, but best theoretically targeted to the exact thesis question; candidate for a dedicated chapter rather than a quick ablation.

##### 2.10 SPIRAL — Self-Play Incremental Racing Algorithm for Multi-Drone Competitions
- **Citation**: Akgun, O., et al. (2025). *SPIRAL: Self-Play Incremental Racing Algorithm for Learning in Multi-Drone Competitions.* IEEE ASYU 2025.
- **Link**: https://arxiv.org/abs/2510.22568 — **Verified**.
- Summary: self-play framework for multi-drone racing; drones compete against increasingly proficient past selves, curriculum emerges from self-play alone (no explicit regret/UED machinery); algorithm-agnostic.
- Mechanism: plain iterative self-play, closer to your baseline than to the robust-RL methods above.
- Application: a directly comparable drone-domain precedent for "what we started from"; its absence of any population/regret mechanism is itself evidence for why G1a/G1b gaps are expected.
- Cost: N/A to adopt (you already do iterative self-play); **Low** as a comparative baseline citation.

##### 2.11 Robust DRL Through Adversarial Attacks and Training: A Survey
- **Citation**: Schott, L., et al. (2024, rev. 2026). *Robust Deep Reinforcement Learning Through Adversarial Attacks and Training: A Survey.* arXiv:2403.00420.
- **Link**: https://arxiv.org/abs/2403.00420 — **Verified**.
- Summary: 83-page taxonomy of adversarial-attack/training methods for DRL by objective (observation, dynamics/action, reward); comparative taxonomy of ~15 algorithms.
- Application: framing/related-work backbone for positioning RARL/ATLA/population methods relative to each other and relative to G1a (rule/reward shift) vs. G1b (seed shift).
- Cost: N/A (survey).

##### 2.12 Adversarial ML Attacks and Defences in MARL (survey)
- **Citation**: Standen, M., Kim, J., Szabo, C. (2025). *Adversarial Machine Learning Attacks and Defences in Multi-Agent Reinforcement Learning.* ACM Computing Surveys 57(5), Article 124.
- **Link**: https://dl.acm.org/doi/10.1145/3708320 — **Partially verified** (direct DL fetch 403-blocked; author names/venue/issue corroborated via independent search).
- Summary: MARL-specific attack/defense taxonomy; covers population/ensemble-based defenses; explicitly identifies generalization-to-unseen-opponents as an open research gap.
- Application: grounds the G1a/G1b taxonomy itself within the broader MARL adversarial-robustness literature.
- Cost: N/A (survey).

##### 2.13 Robust MARL with Diverse Adversarial Agent Generation + Contrastive Policy Encoding (DAAG+CAPE)
- **Citation**: Ma, W., Chang, Y.-C., Shen, Y., Zhao, Z., Lin, C.-T. (2026, under review). *Robust Multi-Agent Reinforcement Learning with Diverse Adversarial Agent Generation and Contrastive Policy Encoding.* ICLR 2026 (OpenReview id h9isaqF956).
- **Link**: https://openreview.net/forum?id=h9isaqF956 — **Unverified** (OpenReview bot-wall blocked direct fetch both here and in the Family 6 pass; title/5-author list corroborated only via independent secondary search; "ICLR 2026 under review" status should be re-checked for acceptance/withdrawal before citing).
- Summary (provisional): trains against a population of diverse adversarial co-players generated via a co-evolution mechanism, paired with contrastive representation learning encoding distinct opponent "policy styles"; tested on Predator-Prey and SMAC; targets generalization-to-unseen-opponents directly.
- Mechanism: (a) diversity-promoting adversarial-agent generator (not just random-init population), (b) contrastive encoder letting the protagonist condition on an inferred latent opponent style.
- Application: relevant to **both** regimes if confirmed — the generator targets G1a (structurally different evader "styles"), the contrastive encoder gives online inference of "what kind of evader is this" useful at test time against both G1a and G1b evaders. Predator-Prey is architecturally the closest existing benchmark to a 1v1 interception task among surveyed papers.
- Cost: **Medium-High**, pending verification.

---

#### Family 3 — Cross-seed / cross-play generalization

##### 3.1 Other-Play for Zero-Shot Coordination
- **Citation**: Hu, H., Lerer, A., Peysakhovich, A., Foerster, J. (2020). *"Other-Play" for Zero-Shot Coordination.* ICML 2020.
- **Link**: https://arxiv.org/abs/2003.02979 — **Verified**.
- Summary: naive self-play converges to arbitrary, highly specialized conventions that fail against independently-trained partners; Other-Play exploits known symmetries in the environment's dynamics/reward to bias training toward symmetry-invariant strategies; scores higher than SP when paired with independently-trained (cross-play) partners/humans in Hanabi.
- Mechanism: augments the SP objective with an expectation over a symmetry group acting on the environment/observation space.
- Application: **G1b** — if the PE engagement geometry has an exploitable symmetry (e.g. reflection about the pursuer-evader line, left/right maneuver choice), an Other-Play-style symmetrization term could reduce dependence on arbitrary conventions established against one campaign's evader lineage. Less relevant to **G1a** (differs by rules/rewards, not symmetry-breaking tie-breaks).
- Cost: **Medium** (identify/implement a symmetry group for the 3D continuous drone state-action space, modify PPO loss to sample over it).

##### 3.2 Off-Belief Learning
- **Citation**: Hu, H., Lerer, A., Cui, B., Pineda, L., Brown, N., Foerster, J. (2021). *Off-Belief Learning.* ICML 2021.
- **Link**: https://arxiv.org/abs/2103.04000 — **Verified**.
- Summary: standard self-play policies rely on fragile multi-step reasoning about what a partner "must have meant" by past actions, which breaks under cross-play; OBL trains assuming past actions came from a fixed grounded policy (e.g. uniform random) but future actions come from the policy itself, converging to a unique grounded policy; can be iterated hierarchically.
- Mechanism: belief-conditioning trick decoupling the policy from self-generated conventions.
- Application: **G1b-relevant but harder to port.** The core insight ("don't let your policy rely on inferences only your own training partner would produce") maps onto not letting the pursuer's acceleration-estimator/RLS readout overfit to implicit assumptions baked in by one campaign's evader behavior distribution. Less directly actionable than Other-Play for a strictly adversarial (not cooperative) continuous-control game.
- Cost: **High** (re-deriving an OBL-style construction for an adversarial rather than Dec-POMDP-cooperative setting is a research contribution in itself).

##### 3.3 TrajeDi — Trajectory Diversity for Zero-Shot Coordination
- **Citation**: Lupu, A., Cui, B., Hu, H., Foerster, J. (2021). *Trajectory Diversity for Zero-Shot Coordination.* ICML 2021, PMLR v139:7204-7213.
- **Link**: https://proceedings.mlr.press/v139/lupu21a.html — **Verified** (PMLR canonical; no separate arXiv ID found).
- Summary: self-play best responses overfit to their own trajectory distribution; the fix is training a common best response against a diverse population of agents; TrajeDi is a differentiable generalization of Jensen-Shannon divergence used as a regularizer to maximize diversity among the training population; improves cross-play scores in Hanabi over plain self-play and non-regularized population BR.
- Mechanism: JSD-based diversity bonus added to the population training objective.
- Application: **directly G1b-relevant** — a template for building a stored evader population that is diversity-regularized across seeds/campaigns rather than just accumulated opportunistically; explicitly penalize evader-population homogeneity so the pursuer's BR does not overfit to one campaign's style.
- Cost: **Medium** (differentiable divergence between evader policies from sampled-state action distributions; needs a population-level diversity loss term).

##### 3.4 Fictitious Co-Play (FCP) — Collaborating with Humans without Human Data
- **Citation**: Strouse, D.J., McKee, K., Botvinick, M., Hughes, E., Everett, R. (2021). *Collaborating with Humans without Human Data.* NeurIPS 2021 (Spotlight).
- **Link**: https://arxiv.org/abs/2110.08176 — **Verified**.
- Summary: SP/population-play overfit to training partners; collecting human data is expensive; FCP trains a best response against a population of self-play agents **and their past checkpoints across training**, producing broad partner skill/style diversity without human data; outperforms SP/PP/BC-play baselines against novel agents/humans in Overcooked.
- Mechanism: best-response training against a checkpoint-mixture population (multiple independent SP runs × checkpoints along each run).
- Application: **core recipe for G1b.** This is essentially checkpoint/population mixing across seeds — train the pursuer as a best response not just to its own campaign's evader lineage but to a population built from *multiple independently-seeded self-play campaigns' checkpoints*. Given the existing stored evader population, extending it to include checkpoints from several independent self-play seeds and sampling pursuer training partners from that broader pool is close to a direct implementation.
- Cost: **Low-Medium** given existing infrastructure (mainly pooling checkpoints across campaigns/seeds + adjusting the opponent-sampling distribution; most of the plumbing already exists).

##### 3.5 RPM — Ranked Policy Memory
- **Citation**: Qiu, W., et al. (2022). *RPM: Generalizable Behaviors for Multi-Agent Reinforcement Learning.* arXiv:2210.09646 (ICLR 2023 preprint).
- **Link**: https://arxiv.org/abs/2210.09646 — **Verified**.
- Summary: MARL agents overfit to training partners and fail against agents behaving differently at evaluation; RPM maintains a ranked look-up memory of past policies (ranked by training episode return) and samples training partners from across quality tiers rather than only the frontier; up to 402% average improvement on Melting Pot generalization evaluations.
- Mechanism: return-ranked policy archive sampled as training partner (vs. recency-based sampling).
- Application: **G1b-relevant**, complementary to FCP — mixes *within* a run's history by return-rank rather than recency; combinable with FCP by applying RPM-style ranked sampling across the multi-seed/multi-campaign evader pool so the pursuer sees weak and strong historical evaders, not only each campaign's final policy.
- Cost: **Low** (sampling-strategy change given an existing stored evader population).

##### 3.6 RPPO — Learning Diverse Risk Preferences in Population-Based Self-Play
- **Citation**: Jiang, Z., Liu, Y., Ma, J., Li, K., Yang, Y., Yang, X., Liang, S., Zhao, D. (2023/2024). *Learning Diverse Risk Preferences in Population-Based Self-Play.* AAAI 2024.
- **Link**: https://arxiv.org/abs/2305.11476 — **Verified**.
- Summary: self-play optimizing only expected win-rate causes the population to collapse into homogeneous, locally-optimal strategies; RPPO interpolates between worst-case (risk-averse) and best-case (risk-seeking) policy optimization via a tunable risk parameter per population member, integrated into PBT; yields a population with diverse risk postures.
- Mechanism: distributional/CVaR-style RL objective with a per-member risk parameter + PBT scheduler.
- Application: **G1b** — if your stored evader population's diversity across campaigns is really just diversity of risk tolerance (aggressive vs. cautious evasion), explicitly inducing risk diversity via RPPO broadens the population beyond what random seeds alone produce.
- Cost: **Medium** (switch PPO's expected-return advantage to a risk-sensitive/CVaR-weighted advantage + a PBT-like scheduler).

##### 3.7 QDSP — Quality-Diversity Self-Play (directly on-domain)
- **Citation**: (authors per NeurIPS 2024 poster listing). *Quality-Diversity Self-Play.* NeurIPS 2024 poster.
- **Link**: https://neurips.cc/virtual/2024/100930 — **Verified** (abstract fetched directly from NeurIPS virtual page).
- Summary: evaluated explicitly in a **two-player asymmetric pursuer-evader simulation** with code-based policies; combines foundation-model-driven code search with quality-diversity principles to explore diverse, high-performing strategies within competing self-play populations; surpasses human-designed pursuer/evader policies and quality-only/diversity-only self-play ablations; discovered strategies span RL, heuristic search, MPC, tree search.
- Mechanism: foundation-model-guided code mutation/search layered on a QD (MAP-Elites-style) archive maintained per competing side.
- Application: **directly on-domain for G1a** — explicitly diversifies strategy *families* (MPC vs. heuristic vs. RL), exactly the "different rules/rewards/strategies" that define G1a; ensures the evader population used to train the pursuer spans genuinely different algorithmic families, not just different random seeds of the same PPO rule.
- Cost: **High** (code-generation/foundation-model search loop + QD archive infrastructure — a substantial new subsystem, though the pursuer-evader framing is an exact match).

##### 3.8 Policy Distillation
- **Citation**: Rusu, A.A., et al. (2015/2016). *Policy Distillation.* arXiv:1511.06295 (ICLR 2016).
- **Link**: https://arxiv.org/abs/1511.06295 — **Verified**.
- Summary: distills a trained RL policy (teacher) into a student network via supervised distillation of the teacher's output distribution; consolidates multiple task-specific policies into one multi-task network, sometimes outperforming the originals.
- Mechanism: KL-divergence matching of student to teacher action distribution, across one or many teachers.
- Application: **G1b** as the "ensemble → single robust policy" mechanism — if multiple independent self-play campaigns (different seeds) each produce a reasonable pursuer, distillation offers a path to consolidate N pursuer policies into one that generalizes across the union of campaign-specific evader lineages, avoiding the weight-space incompatibility risk of direct averaging.
- Cost: **Medium** (N teacher policies, data collection of states + teacher action distributions, supervised distillation phase; the RLS online-adaptation layer needs separate handling — re-distill or keep frozen as a wrapper).

##### 3.9 Model Soups
- **Citation**: Wortsman, M., Ilharco, G., Gadre, S.Y., et al. (2022). *Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time.* ICML 2022.
- **Link**: https://arxiv.org/abs/2203.05482 — **Verified**.
- Summary: averaging the weights of multiple fine-tuned models ("soup") instead of picking the single best, when models share a low-loss basin (fine-tuned from the same pretrained initialization); improves accuracy/robustness/OOD/zero-shot performance at zero extra inference cost.
- Mechanism: direct parameter-space averaging of independently fine-tuned checkpoints (uniform or greedy/incremental).
- Application: **G1b-relevant with a caveat.** Plausible if each campaign is a continuation of the same base pursuer policy with a different random evader-seed stream — souping across campaign-final checkpoints is a nearly-free way to reduce campaign-specific overfit. Not plausible for independently-initialized runs from scratch (different permutation-symmetric basins) without weight-matching (e.g. Git Re-Basin) first.
- Cost: **Low** if initializations are shared; **High** if not (requires weight-alignment research). Recommend validating basin-compatibility empirically (interpolate two campaign-final checkpoints, check for a loss barrier) before committing.

##### 3.10 AlphaStar League Training (anchor, mechanism-linked)
- **Citation**: Vinyals, O., et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning.* Nature 575, 350-354.
- **Link**: https://www.nature.com/articles/s41586-019-1724-z — **Partially verified** (bibliographic metadata well-established and cross-confirmed; direct Nature fetch returned HTTP 406 in this session).
- Summary: league of main agents, main exploiters, and league exploiters — exploiters' explicit job is to find and exploit weaknesses in the main agents, preventing strategic collapse/overfitting to the current population; periodic opponent-pool refresh maintains strategic diversity.
- Mechanism: three heterogeneous agent roles with different matchmaking/objective rules.
- Application: **both G1a and G1b.** The "exploiter" role is directly analogous to deliberately including evaders optimized specifically to beat the *current* pursuer (a harder, targeted version of G1a); periodic population refresh/pooling across league history addresses G1b's seed-lineage overfitting.
- Cost: **Medium-High** (add an explicit exploiter training track with periodic reset + matchmaking logic for population diversity atop the existing alternating-stage self-play).

---

#### Family 4 — Opponent-conditioned / adaptive policies that generalize to unseen opponents

Anchors re-cited only for mechanism-linking (not re-summarized, per your notes): LIAM
(Papoudakis et al.), GSCU (Fu et al.), Al-Shedivat et al. 2018 (MAML-opponent).

##### 4.1 LIAM — Agent Modelling under Partial Observability for Deep RL (anchor)
- **Citation**: Papoudakis, G., Christianos, F., Albrecht, S.V. (2021). *Agent Modelling under Partial Observability for Deep Reinforcement Learning.* NeurIPS 2021.
- **Link**: https://arxiv.org/abs/2006.09447 — **Verified**. (Note: the arXiv title is the full title above; "LIAM" is the in-paper method name — cite the full title.)
- One-line relevance: opponent embeddings learned from the controlled agent's own local observations/actions (no access to opponent internals at execution) — anchor for conditioning the pursuer's policy/RLS readout on an inferred evader-style latent, applicable to both G1a and G1b.

##### 4.2 GSCU — Greedy when Sure and Conservative when Uncertain about the Opponents (anchor)
- **Citation**: Fu, H., Tian, Y., Yu, H., Liu, W., Wu, S., Xiong, J., Wen, Y., Li, K., Xing, J., Fu, Q., Yang, W. (2022). *Greedy when Sure and Conservative when Uncertain about the Opponents.* ICML 2022, PMLR v162:6829-6848.
- **Link**: https://proceedings.mlr.press/v162/fu22b.html — **Verified**.
- One-line relevance: offline opponent-policy-embedding + conditional best response + online Bayesian posterior updating over embeddings + bandit-based greedy/conservative switching — anchor for G1a (conservative fallback vs. truly novel/adversarial evaders) and G1b (posterior update distinguishes same-rule-different-seed evaders).

##### 4.3 Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments (anchor, MAML-opponent)
- **Citation**: Al-Shedivat, M., Bansal, T., Burda, Y., Sutskever, I., Mordatch, I., Abbeel, P. (2017/2018). ICLR 2018.
- **Link**: https://arxiv.org/abs/1710.03641 — **Verified**.
- One-line relevance: MAML-style gradient-based meta-learning for few-shot adaptation against nonstationary/competitive opponents in RoboSumo — anchor for both regimes as a test-time few-shot fine-tuning baseline.

##### 4.4 Algorithm Distillation — In-context RL
- **Citation**: Laskin, M., Wang, L., Oh, J., Parisotto, E., Spencer, S., Steigerwald, R., Strouse, D.J., Hansen, S., Filos, A., Brooks, E., Gazeau, M., Sahni, H., Singh, S., Mnih, V. (2022). *In-context Reinforcement Learning with Algorithm Distillation.* arXiv:2210.14215 (ICLR 2023).
- **Link**: https://arxiv.org/abs/2210.14215 — **Verified**.
- Summary: treats "learning to RL" as sequence modeling — a causal transformer autoregressively predicts actions from entire learning histories generated by a source RL algorithm; at deployment, improves purely in-context (via the trajectory fed as context), no gradient updates at test time; learns a more sample-efficient RL algorithm than the one generating training data.
- Mechanism: causal transformer trained on concatenated across-episode learning trajectories.
- Application: **both G1a and G1b** — the cleanest "no online fine-tuning needed" alternative to meta-RL. If the pursuer's learning histories against diverse evaders (across rule-families and seeds) are logged, a transformer pursuer could adapt to a brand-new evader within an episode purely by conditioning on recent trajectory — subsuming the role of the existing RLS readout but learned end-to-end.
- Cost: **High** (generate/log learning-history data across many evader types/seeds, transformer backbone replacing/augmenting the current policy, likely hybridizing with the hand-designed RLS readout).

##### 4.5 TAO — Towards Offline Opponent Modeling with In-context Learning
- **Citation**: Jing, Y., Li, Y., Liu, H., Zang, Q., Fu, S., Fu, Q., Xing, J., Cheng, J. (2024). ICLR 2024.
- **Link**: ICLR proceedings PDF (OpenReview id 2SwHngthig) — **Verified** via proceedings PDF fetch; OpenReview HTML itself bot-blocked, author list cross-checked against a secondary search summary.
- Summary: formalizes Offline Opponent Modeling (OOM) — respond to opponents using only pre-collected offline data; TAO = offline policy-embedding learning + offline opponent-aware response-policy training + online deployment via in-context learning (no weight updates) adapting to a never-before-seen opponent from the deployment trajectory alone; theoretically equivalent to Bayesian posterior sampling over opponent identity.
- Mechanism: pretrained transformer with in-context embedding inference, frozen at test time, adapted purely via context.
- Application: **strong fit for both regimes.** Maps directly onto the existing stored evader population: build an offline interaction dataset against it (spanning rule-families for G1a, seeds for G1b), train the embedding+response transformer offline, deploy a pursuer inferring "which kind of evader is this" purely in-context against any new evader.
- Cost: **Medium-High** (substantial offline interaction dataset — achievable given existing infra — plus a new Transformer-based embedding/response architecture alongside the existing PPO+RLS pursuer).

##### 4.6 OMIS — Opponent Modeling with In-context Search
- **Citation**: Lu, D., Wu, Y. (2024). NeurIPS 2024.
- **Link**: NeurIPS proceedings abstract (hash 710445227fa8c1b6a9ceada902dd4741) — **Verified** via direct fetch of the official NeurIPS proceedings abstract; code at github.com/Luzy2019/omis.
- Summary: extends in-context opponent modeling with decision-time search over three pretrained in-context components — actor (best response), opponent imitator (predicts opponent actions), critic (state-value); at test time against unknown non-stationary opponents, runs search to refine the actor's choice on the fly; theoretically converges in opponent-policy recognition without search and provides improvement guarantees with search; outperforms pretrain-only and online-fine-tuning baselines.
- Mechanism: pretrained ICL transformer (actor + opponent-imitator + critic) combined with test-time decision search conditioned on in-context opponent inference.
- Application: **arguably the most directly transferable 2023-2026 paper found for both regimes.** The three-component decomposition maps almost one-to-one onto the pursuer's existing stack: the "opponent imitator" is directly analogous to the existing learned evader-acceleration estimator, and OMIS's in-context Bayesian-style update over it is a principled generalization of the existing online RLS readout (Neural-Fly style) — upgrading from linear-regression adaptation to a learned, in-context, search-refined adaptation explicitly trained to generalize to unseen non-stationary opponents.
- Cost: **Medium** — architectural overlap with existing components makes this an incremental evolution rather than a from-scratch subsystem; decision-time search adds runtime compute that must be budgeted for an onboard real-time drone controller.

##### 4.7 HOLA-Drone — cooperative ZSC, low direct relevance (cross-reference)
- **Citation**: Zhang, R., et al. (2024). *HOLA-Drone: Hypergraphic Open-ended Learning for Zero-Shot Multi-Drone Cooperative Pursuit.* arXiv:2409.08767.
- **Link**: https://arxiv.org/abs/2409.08767 — **Verified** (fetched via ar5iv mirror).
- Summary: solves *cooperative* multi-drone pursuit ZSC (drone pursuers coordinating with unseen *drone teammates*), not adaptation to unseen evaders; hypergraphic-form game formulation + open-ended learning objective adaptation; validated on real Crazyflie drones.
- Mechanism: hypergraph-based multi-agent compatibility modeling + open-ended learning objective adaptation (PSRO-style iterative best-response against an evolving population of unseen-partner proxies).
- Application: low direct relevance as framed (cooperative teammate generalization, not adversarial evader generalization), but cited as evidence that ZSC methodology is actively being ported into exactly your domain (drone pursuit, Isaac-Sim-adjacent real hardware). The mechanism could in principle be re-derived for the adversarial setting as a novel contribution.
- Cost: **High** if repurposed (re-deriving the hypergraphic formalism for adversarial, not cooperative, settings).

##### 4.8 AgilePE (see Family 1, entry 1.13 — also Family 3/4-relevant)
- Cross-reference: PFSP + diversified opponent pool is a concrete, already-validated-on-drones instantiation of Family 3's "checkpoint/population mixing," directly actionable for **G1b**. See 1.13 for full entry. Cost **Low-Medium**.

##### 4.9 Equilibrium Policy Generalization (EPG) — cross-graph zero-shot generalization in pursuit-evasion games
- **Citation**: Lu, R., Zhang, P., Shi, R., Zhu, Y., Zhao, D., Liu, Y., Wang, D., Alippi, C. (2025). *Equilibrium Policy Generalization: A Reinforcement Learning Framework for Cross-Graph Zero-Shot Generalization in Pursuit-Evasion Games.* NeurIPS 2025, pp.48810-48841.
- **Link**: https://arxiv.org/abs/2511.00811 — **Verified**.
- Summary: builds a dynamic-programming oracle providing a provably near-optimal pure-strategy Nash equilibrium for any single graph, then trains an RL policy across many graphs against each graph's own equilibrium oracle rather than self-play alone; zero-shot generalizes to unseen real-world graphs, matching fine-tuned SOTA baselines in some settings.
- Mechanism: equilibrium-oracle-guided training (train against a computed Nash-equilibrium opponent per condition, not a learned self-play opponent) + distance-feature engineering for cross-condition generalization.
- Application: **conceptually relevant to G1a** — training against a computed, principled equilibrium opponent anchors the pursuer to game-theoretically sound behavior rather than only its own self-play lineage's conventions. If an equilibrium/near-optimal evader oracle could be approximated for simplified interception geometries (via differential-game pursuit-evasion theory), mixing equilibrium-oracle opponents into the evader-training population alongside learned self-play evaders could reduce G1a overfitting to the specific rules/rewards used for learned held-out evaders.
- Cost: **High** (computing a reliable equilibrium oracle for continuous 3D drone dynamics is a nontrivial differential-game/optimal-control sub-problem — most valuable as a conceptual argument rather than directly portable code).

---

#### Family 5 — Regularization / generalization in deep RL vs. unseen opponents

##### 5.1 EMAgnet — Parameter-Space EMA Regularization for Policy Gradient Self-Play
- **Citation**: Lanier, J.B., et al. (2026). *EMAgnet: Parameter-Space EMA Regularization for Policy Gradient Self-Play in Large Games.* NExT-Game 2026 (ICML 2026 Workshop).
- **Link**: https://arxiv.org/abs/2606.23995 — **Verified**.
- Summary: replaces uniform-entropy "magnet" regularization with EMA-of-own-policy regularization, avoiding wasted regularization budget on dominated/useless actions; lower exploitability than uniform-magnet PPO self-play, largest gains in decoy-action-rich games.
- Finding: regularization-target choice (EMA vs. uniform) matters more than entropy coefficient magnitude.
- Application: **G1b** — reduces overfitting to the current frozen evader's micro-strategies while controlling policy drift; weaker fit for **G1a** (reward/rule shift).
- Cost: **Low-Medium** (EMA weight copy + KL/L2 penalty term added to the PPO loss).

##### 5.2-5.3 Entropy regularization (unverified leads — flag only)
- "Proximal Policy Optimization with Entropy Regularization" (IEEE, 2024), https://ieeexplore.ieee.org/document/10585473 — **Unverified** (paywalled, search-snippet only).
- "Entropy Regularization in Deep Reinforcement Learning: A Structured Survey," MDPI *Entropy* 28(7):811 (2026) — **Unverified** (not independently fetched; search-synthesis only).
- Gap: no paper found directly/quantitatively linking entropy coefficient to win-rate vs. held-out opponent in continuous-control self-play.

##### 5.4 DrAC — Automatic Data Augmentation for Generalization in Deep RL
- **Citation**: Raileanu, R., Goldstein, M., Yarats, D., Kostrikov, I., Fergus, R. (2020/2021). NeurIPS 2021.
- **Link**: https://arxiv.org/abs/2006.12862 — **Verified**.
- Summary: auto-selects augmentations and adds policy/value KL-consistency regularization between original/augmented observations; ~40% relative improvement on Procgen **test (unseen level)** performance (see Procgen results tables in the paper).
- Finding: the KL-consistency term, not the augmentation alone, is the generalization mechanism.
- Application: **strong for both G1a and G1b** — apply augmentations to the relative-kinematic observation space (mirroring, sensor noise, perturbed estimated evader acceleration) and add KL-consistency terms to the PPO loss; extend to the RLS estimator.
- Cost: **Medium** (define geometric augmentation operators valid for the PE state; add KL-consistency terms to PPO loss; extend to RLS estimator). Code: github.com/rraileanu/auto-drac.

##### 5.5 Policy-Aware Adversarial Data Augmentation
- **Citation**: Zhang, H., Guo, Y. (2021). *Generalization of Reinforcement Learning with Policy-Aware Adversarial Data Augmentation.* arXiv:2106.15587.
- **Link**: https://arxiv.org/abs/2106.15587 — **Verified**.
- Summary: generates adversarial trajectory data via the policy-gradient objective (targeting the current policy's weaknesses) and mixes it with real trajectories to prevent instability; designed for a low-diversity training-set regime.
- Finding: adversarial perturbation targets policy-gradient-sensitive directions rather than generic transforms.
- Application: **G1a** — matches the "fixed small frozen-evader-stage set but broader test-time opponent variety" problem; could perturb estimated evader accelerations rather than full re-simulation.
- Cost: **Medium-High** (changes the data-generation loop, not just the loss).
- Gap: no paper found applying DrAC/RAD-style augmentation specifically to multi-agent *opponent* generalization (vs. environment/level generalization) — potential novel-contribution angle.

##### 5.6 The Primacy Bias in Deep Reinforcement Learning
- **Citation**: Nikishin, E., Schwarzer, M., D'Oro, P., Bacon, P.-L., Courville, A. (2022). ICML 2022.
- **Link**: https://arxiv.org/abs/2205.07802 — **Verified**.
- Summary: agents overfit to early-encountered data; periodic partial network resets (keeping the replay buffer) restore performance; validated on Atari-100k, DMC (SAC/DrQ/SPR).
- Finding: early-interaction overfitting degrades subsequent learning — analogous to "locking in" on the first few frozen-evader stages.
- Application: **both regimes** — reset the last 1-2 layers of policy/value heads at stage-transition boundaries. **Caveat**: PPO is on-policy (no persistent replay buffer like SAC/DrQ); the benefit needs re-verification in an on-policy setting.
- Cost: **Low** (reinit last N layers at stage boundaries).

##### 5.7 Loss of Plasticity in Continual Deep Reinforcement Learning
- **Citation**: Abbas, Z., Zhao, R., Modayil, J., White, A., Machado, M.C. (2023). CoLLAs 2023, PMLR v232.
- **Link**: https://arxiv.org/abs/2303.07507 — **Verified**.
- Summary: value-based agents cycling through task sequences (Atari) progressively lose learning ability on later tasks; mechanism is activation-footprint sparsification causing diminishing gradients; CReLU activation substantially mitigates.
- Finding: quantified mechanism — sparsification over repeated task cycling, the closest verified analogue to cycling through frozen evader opponents.
- Application: **both regimes**, diagnostic — log activation sparsity/dead-unit fraction across the 13/26 stages; if rising, independent evidence of plasticity loss contributing to degraded generalization, distinct from pure opponent-overfitting.
- Cost: **Low** for diagnosis (logging only); **Low-Medium** for a CReLU-swap mitigation.

##### 5.8 Loss of Plasticity in Deep Continual Learning
- **Citation**: Dohare, S., Hernandez-Garcia, J.F., Lan, Q., Rahman, P., Mahmood, A.R., Sutton, R.S. (2024). *Nature* 632.
- **Link**: https://arxiv.org/abs/2306.13812 — **Verified** (arXiv content confirmed directly; Nature DOI 10.1038/s41586-024-07711-7 widely corroborated but publisher page itself not independently fetched in this pass, 406-blocked).
- Summary: binary classification accuracy drops from 89% to ~77% (linear-network level) by task 2000 in a continual sequence; L2 regularization + weight perturbation substantially eases the effect; introduces "continual backpropagation" (reinitializing least-used units).
- Finding: L2 weight decay is a cheap, broadly effective first-line mitigation (cheaper than full resets or CReLU).
- Application: **both regimes** — add L2/AdamW weight decay plus optional small Gaussian weight perturbation at stage boundaries; minimal code change.
- Cost: **Low**.
- **Honesty flag**: 5.6-5.8 were developed for single-agent/task-cycling settings, not multi-agent self-play — application to the pursuer is inference/extrapolation, not a directly verified opponent-generalization finding.

##### 5.9 AlphaStar PFSP (mechanism cross-reference — see Family 3, entry 3.10)
- Application restated for Family 5 framing: replace single-latest-frozen-evader sampling with PFSP-weighted sampling over the stored evader population (win-rate matrix → softmax(1-winrate) sampling distribution). **G1b**-focused; less direct fit for G1a (still only diversifies within self-play-produced population). Cost **Medium**.

##### 5.10 AgilePE (mechanism cross-reference — see Family 1, entry 1.13)
- Design-level evidence (not an ablation table) that single-frozen-opponent iterated best response (your current design) is prone to oscillation/overfitting; same PFSP + diversified-opponent-pool mechanism, directly transferable since the domain is near-identical (UAV pursuit-evasion, CTBR control). Cost **Low-Medium**; exact hyperparameters need full-PDF extraction beyond the abstract.

##### 5.11 Offline Fictitious Self-Play for Competitive Games (OFF-FSP)
- **Citation**: Chen, J., et al. (2024/2025). arXiv:2403.00841.
- **Link**: https://arxiv.org/abs/2403.00841 — **Verified**.
- Summary: offline/limited-data FSP; approximates Nash equilibrium while constraining best responses away from OOD actions via importance-sampling opponent diversity; validated on matrix games, poker, board games, a real human-robot task.
- Finding: provides a concrete exploitability metric framing the generalization gap as performance against a broad simulated opponent distribution vs. a narrow training distribution.
- Application: methodology reference — adopt exploitability-style evaluation for both G1a/G1b rather than win-rate alone.
- Cost: **Low** if only adopting the evaluation metric; **Medium-High** if adopting the offline best-response machinery.
- Gap: no controlled ablation found directly varying "stage length before overfitting sets in" with quantitative tables in game-theoretic or continuous-control self-play — a genuine open gap; a 13-vs-26-stage comparison could be a novel contribution.

##### 5.12 OCNR — Stabilizing Self-Play by Mitigating Iteration-Collapse With One-Class Novelty Rewards
- **Citation**: Lee, S., Nam, G., Lee, H., Lee, J. (2026). ICML 2026, PMLR 306:64837-64861.
- **Link**: https://proceedings.mlr.press/v306/lee26ay.html — **Verified**.
- Summary: LLM self-play domain (Challenger/Solver, math reasoning); names "persistent iteration-collapse" — performance improves early, regresses later, as the generator narrows onto familiar problems ("cross-iteration degeneration"); fix = a "Seen Detector" penalizing near-duplicate task regeneration.
- Application: cross-domain analogy (LLM text → aerial PE) — a named phenomenon anchor for "strategy collapse via narrowing opponent/task distribution"; could inspire a diversity-preserving evader-selection mechanism.
- Cost: **High** if directly adapted (needs a representation-space novelty detector over evader trajectories).

##### 5.13 R-Diverse — Mitigating Diversity Illusion in Self-Play LLM Training
- **Citation**: Li, G., et al. (2026). arXiv:2602.13103.
- **Link**: https://arxiv.org/abs/2602.13103 — **Verified**.
- Summary: names "Diversity Illusion" — surface-diverse but skill-collapsed training signals; splits into Local (cross-iteration mode cycling) vs. Surface (no underlying skill variation) Diversity Illusion; fix = Memory-Augmented Penalty + Skill-Aware Measurement.
- Finding: "Local Diversity Illusion → cross-iteration mode cycling" is the most precisely-named recent description of self-play strategy cycling.
- Application: diagnostic — cluster stored evader checkpoints by behavioral features to check for surface diversity but skill collapse (same evasion skill class, different seed noise only), which would explain a weak G1a and partial G1b generalization gap.
- Cost: **Medium** for diagnostic (analysis only, reuses stored population); **High** for full mitigation adapted to continuous control.

##### 5.14 Self-Play Survey (cross-reference — see Family 1, entry 1.12)
- Frame-setting citation for why iterated-best-response self-play risks both G1a (OOD reward-trained opponents) and G1b (PSRO/FSP-style overfitting within fixed rules).

**Honesty flag (important)**: no 2023-2026 paper was found that directly, empirically measures "self-play overfitting"/"strategy collapse" in game-theoretic or continuous-control RL, as distinct from the LLM self-improvement literature (where 5.12/5.13 explicitly measure and name it). State in the thesis that rigorous quantitative measurement of this phenomenon is more mature in LLM self-play (2026) than in game/robotics self-play — positioning your work as potentially closing that gap. Balduzzi et al. 2019 and the AlphaStar/PFSP line remain the standard game-RL anchors.

---

#### Family 6 — Pursuit-evasion / interception-specific (2023-2026)

**Note on the Roncero baseline**: a related paper, arXiv:2506.02849, "Learned Controllers for Agile Quadrotors in Pursuit-Evasion Games" (v1 Jun 2025, v3 Jun 2026) — **Verified**. It covers AMSPBH (Asynchronous Multi-Stage Population-Based training with Hedge sampling) and states in its abstract that "population-based training generalizes better across diverse and unseen strategies" vs. training against only the latest opponent — supporting evidence embedded directly in your cited baseline's lineage, worth citing precisely if not already in your notes.

##### 6.1 DualCL — A Dual Curriculum Learning Framework for Multi-UAV Pursuit-Evasion in Diverse Environments
- **Citation**: Chen, J., Li, G., Yu, C., Yang, X., Xu, B., Yang, H., Wang, Y. (2023/2024). arXiv:2312.12255.
- **Link**: https://arxiv.org/abs/2312.12255 — **Verified**.
- Summary: Intrinsic Parameter Curriculum Proposer + External Environment Generator targeting unresolved task-space regions; multi-pursuer vs. single fast evader, 3D, obstacles, dynamics-constrained UAVs; >90% capture rate, ≥27.5% capture-time reduction, explicitly reports "best zero-shot generalization" (see experiments section table) as the headline metric.
- Application: **G1b** — adopt an "External Opponent-Rule/Environment Generator" biasing training toward regions where the current pursuer is weak, layered on the existing alternating self-play.
- Cost: **Medium** (parameterized generator/sampler; reuses existing PPO pipeline and stored evader population).

##### 6.2 OPEN — Online Planning for Multi-UAV Pursuit-Evasion in Unknown Environments Using Deep RL
- **Citation**: Yu, C., Li, G., Tang, W., Ji, S., Yang, X., Xu, B., Yang, H., Wang, Y. (2024/2025). IEEE RA-L, accepted Jun 2025.
- **Link**: https://arxiv.org/abs/2409.15866 — **Verified**.
- Summary: evader-prediction-enhanced network for partial observability; adaptive environment generator for exploration/generalization; 100% capture rate on unseen scenarios in sim, zero-shot real-quadrotor validation (CTBR control).
- Finding: explicit unseen-scenario evaluation claim (100% capture rate).
- Application: **both regimes** — the evader-prediction module is conceptually identical to the pursuer's existing Neural-Fly-style RLS estimator; evidence supports extending it to predict evader *strategy class*, not just acceleration.
- Cost: **Medium** (estimator extension is a moderate add-on; the adaptive environment generator is new infra).

##### 6.3 ViPER — Visibility-Based Pursuit-Evasion via Reinforcement Learning
- **Citation**: Wang, Y., Cao, Y., Chiun, J., Koley, S., Pham, M., Sartoretti, G.A. (2024). CoRL 2024, PMLR v270, pp.4188-4200.
- **Link**: https://mlanthology.org/corl/2024/wang2024corl-viper/ — **Verified** via official proceedings page (BibTeX/volume/pages confirmed, cross-checked against GitHub marmotlab/ViPER and the CoRL listing; OpenReview id EPujQZWemk itself was bot-blocked).
- Summary: first neural solution trained explicitly against **worst-case/adversarial evaders** (evader knows and exploits the pursuer policy) in visibility-limited PE; graph attention network, multi-pursuer coordination; zero-shot real aerial search deployment.
- Finding: worst-case-adversary training generalizes better across team sizes/environments, outperforms non-learning planners under adversarial evasion.
- Application: **G1a** — periodically insert a worst-case/adversarial evader (optimized to exploit current pursuer blind spots, not just evade) into the stored evader population instead of relying solely on self-play-discovered evaders.
- Cost: **Medium-High** (separate worst-case-evader objective; explicit adversarial-perturbation search over evader policy space).

##### 6.4 EPG (see Family 4, entry 4.9) — cross-reference, repeated for Family 6 completeness.

##### 6.5 Distributed Pursuit-Evasion Game Decision-Making Based on Multi-Agent Deep RL
- **Citation**: (Beijing Institute of Technology-affiliated authors) (2025). *Electronics* (MDPI) 14(11):2141.
- **Link**: https://www.mdpi.com/2079-9292/14/11/2141 — **Verified** (direct fetch via r.jina.ai proxy due to a 403 on direct fetch).
- Summary: automatic curriculum learning + MAPPO; self-play trains pursuers/evaders simultaneously with encirclement reward shaping; "subgame curriculum learning"; fixed-wing UAV dynamics constraints (min velocity, turn radius).
- Finding: subgame curriculum (ordered sequence of increasingly hard self-play subgames) addresses exploration/convergence difficulty under dynamics constraints (indirect, not a direct unseen-evader benchmark).
- Application: **G1b** — structure alternating self-play as a curriculum (ordering early stages toward easier geometries, progressively increasing closing-speed/maneuverability mismatch) to reduce susceptibility to degenerate seed-specific equilibria.
- Cost: **Low-Medium** (reward/curriculum scheduling only, no new modules).

##### 6.6 Scalable Pursuit-Evasion Game for Multi-Fixed-Wing UAV Based on Dynamic Target Assignment and Hierarchical RL
- **Citation**: (Air Force Engineering University/BUPT-affiliated authors) (2025). *Drones* (MDPI) 10(1):5.
- **Link**: https://www.mdpi.com/2504-446X/10/1/5 — **Verified** (direct fetch via r.jina.ai proxy).
- Summary: explicitly frames "limitations of current deep RL methods in terms of transferability and generalization"; 3-layer hierarchy (target allocation → maneuver decision with trajectory prediction → flight control); validated 3v3→12v12 scaling.
- Finding: hierarchical decomposition targets transferability/generalization limitations, but the axis studied is team-size scaling, not strategy diversity.
- Application: lower priority for a 1v1 setting, but the trajectory-prediction + hierarchical RL split complements the existing RLS/Neural-Fly estimator design pattern.
- Cost: **High** if the full hierarchy is adopted; **Low** if only borrowing the maneuver/control decoupling idea.

##### 6.7 GE-DDPG — Guided Exploration RL for UAV Pursuit-Evasion Games
- **Citation**: (authors not visible, paywalled) (2024). Springer book chapter, DOI 10.1007/978-3-031-70684-4_22.
- **Link**: https://link.springer.com/chapter/10.1007/978-3-031-70684-4_22 — **Verified title/abstract/references**; **author list unconfirmed** (paywalled) — cite with caution.
- Summary: a 3D guidance-law baseline (proportional-navigation-like) used as "expert knowledge"; GE-DDPG = DDPG + guided exploration (behavior cloning from the expert law + Gaussian noise); improved capture time/performance vs. the baseline guidance law and vanilla DDPG.
- Application: seed early self-play stages with expert-guided exploration (a BC auxiliary loss against a geometric guidance law) to produce a more diverse/robust initial strategy basin before pure self-play, indirectly helping both regimes.
- Cost: **Low-Medium** (BC auxiliary loss, cheap given existing PPO infra).

##### 6.8 MRPA — Evade Unknown Pursuer via Pursuit Strategy Identification and Model Reference Policy Adaptation (evader-side, informative by symmetry)
- **Citation**: (Northwestern Polytechnical University-affiliated authors) (2024). *Drones* (MDPI) 8(11):655.
- **Link**: https://www.mdpi.com/2504-446X/8/11/655 — **Verified** (direct fetch via r.jina.ai proxy).
- Summary: explicitly states "evasion decision making based on RL is generally trained only for specific pursuers... exhibits poor generalizability" — a direct statement of the thesis's exact problem, mirrored from the evader side; proposes pursuer-attitude estimation + pursuit-strategy identification (inferring opponent guidance-law parameters from relative motion) + Model Reference Policy Adaptation.
- Finding: an ablation table quantifies the "vs. unknown pursuer" generalization gap and MRPA's improvement over a non-adaptive baseline.
- Application: **G1a** — build an analogous "evader-strategy identification" module (a compact embedding of evader strategy type: PN-like vs. jink/juke vs. learned-adversarial) feeding a Model-Reference Policy Adaptation layer, extending the existing RLS/Neural-Fly estimator beyond pure acceleration estimation.
- Cost: **Medium** (strategy-classification head trained on stored population + adaptation layer).

##### 6.9 Adaptive Interception in Dynamic Domains: Exploration of Hybrid RL in Pursuit-Evasion Tasks
- **Citation**: Akinmolayan, M., Josyula, D., Casbeer, D. (2026). Proceedings of the AAAI Symposium Series 8(1):2-10, DOI 10.1609/aaaiss.v8i1.42510.
- **Link**: https://ojs.aaai.org/index.php/AAAI-SS/article/view/42510 — **Verified** (direct fetch via r.jina.ai proxy).
- Summary: classical interception heuristics (Weintraub-style range-limited PE formulation) combined with PPO; adaptive curriculum + enhanced reward; extends to multi-agent coordinated pursuer teams vs. RL-trained evader flocks (adversarial self-play); reports **94% capture rate vs. 24% for classical approaches** (single-agent case; see the paper's results section for the exact table).
- Finding: clearest quantitative capture-rate comparison found in this pass for a PPO-based pursuer vs. adversarial self-play evaders.
- Application: **G1a** — hybrid classical-heuristic + RL blending as a low-risk hedge: when the evader-model estimator is uncertain/OOD, blend toward a geometric (PN/pure-pursuit) fallback that degrades gracefully against differently-trained evaders.
- Cost: **Low-Medium** (confidence-gated blend between learned action and geometric fallback, assuming a geometric baseline controller already exists in the OmniDrones pipeline).

##### 6.10 HJI Reachability in Reinforcement Learning: A Survey
- **Citation**: Ganai, M., et al. (2024). IEEE Open Journal of Control Systems, arXiv:2407.09645.
- **Link**: https://arxiv.org/abs/2407.09645 — **Verified** ("Accepted in IEEE OJ-CSYS" confirmed).
- Summary: surveys methods computing HJ reachable sets jointly with RL policy learning to scale to high-dimensional systems; reviews reachability value functions as safety filters or dense auxiliary reward/critic targets encoding worst-case opponent behavior.
- Application: **G1a** — train an auxiliary HJI-style value function (worst-case time-to-capture against any admissible bounded-acceleration evader, not just the self-play population) using the existing acceleration-bound estimator as the key ingredient for a reduced-order (3-5 state dim) relative-dynamics HJI reach-avoid computation; usable as an auxiliary critic loss or action-shaping safety filter.
- Cost: **High** (separate reachability solver, e.g. levelset/DeepReach-style tools, plus PPO integration).
- Unverified leads noted by the research pass (manual check recommended before citing): "Learning Robust Policies via Interpretable Hamilton-Jacobi Reachability" (OpenReview id JeIUT41IYa, CAPTCHA-blocked); a *Chinese Journal of Aeronautics* 2025 review (ScienceDirect S1000936125005461, CAPTCHA-blocked).

##### 6.11 RPG (Lauffer et al.) — Robust and Diverse Multi-Agent Learning via Rational Policy Gradient
- **Citation**: Lauffer, N., Shah, A., Carroll, M., Seshia, S.A., Russell, S., Dennis, M. (2025). NeurIPS 2025, arXiv:2511.09535.
- **Link**: https://arxiv.org/abs/2511.09535 — **Verified** ("Published at NeurIPS 2025" confirmed).
- Summary: identifies a failure mode where naive adversarial-robustness training causes agents to "self-sabotage" (create exploitable flaws, halting learning); introduces Rationality-preserving Policy Optimization + an RPG algorithm that finds adversarial examples/diverse policies while keeping the adversary a rational best-responder (opponent-shaping). **Note**: distinct from the Tang et al. 2021 Reward Randomization "RPG" in Family 1 — same acronym, different paper.
- Application: **both regimes** — a periodic "stress-test and add to population" step: run a constrained flaw-finding search (an evader optimized to exploit the current frozen pursuer, bounded by the same acceleration/dynamics limits) before adding to the stored population; diversifies against self-play collapse (G1b) and is not purely an artifact of the specific reward shaping used (partial G1a bridge).
- Cost: **Medium-High** (separate rational-adversary-search optimizer/opponent-shaping gradient; reuses existing PPO infra for the adversary itself).

##### 6.12 CLAM — Contrastive Learning-Based Agent Modeling for Deep RL
- **Citation**: Ma, W., Chang, Y.-C., Yang, J., Wang, Y.-K., Lin, C.-T. (2023/2025). arXiv:2401.00132.
- **Link**: https://arxiv.org/abs/2401.00132 — **Verified**.
- Summary: learns real-time representations of other agents' policies from only ego-agent local observations (no opponent internal-state access, no long history needed); evaluated in cooperative and competitive settings. (A "≥28% improvement" figure circulating in secondary sources was not independently re-verified against the paper's results section — treat with caution.)
- Application: **G1a** — a generalized upgrade path for the RLS estimator: learn a compact contrastive embedding of evader *policy/behavior* (not just physical acceleration) that discriminates/clusters behaviors contrastively, usable zero-shot against evaders never seen in training.
- Cost: **Medium** (self-supervised contrastive encoder trained over the stored evader population's trajectories, feeding the pursuer policy — parallel to or replacing the current RLS readout).

##### 6.13 DAAG+CAPE (see Family 2, entry 2.13 — cross-referenced, same unverified status)

##### 6.14 Battery-Aware Reinforcement Learning for Aggressive Quadrotor Flight (tangential, same Roncero research group)
- **Citation**: Roncero, A.S., Andersson, O., Ogren, P. (2026). Submitted to ICRA 2027, arXiv:2609.37316.
- **Link**: https://arxiv.org/abs/2609.37316 — **Verified**.
- Summary: studies exploiting extra available thrust as Crazyflie battery voltage drops (identified load-transient battery model); lists pursuit-evasion as a motivating application in the abstract, but experiments are circle-tracking/lap-racing, not PE generalization; 49% circle-tracking error reduction, 10% race-time reduction vs. stock-authority RL on real hardware.
- Application: not a strategic-generalization mechanism, but adding voltage/thrust-margin state to observations could expand the pursuer's achievable closing-speed envelope, indirectly helping capture rate against fast/aggressive unseen evaders in both regimes by widening the dynamic-capability margin rather than strategy generalization.
- Cost: **Low** (sim-only observation addition) to **Medium** (real-hardware validation).

##### Rejected / not usable as-is
- **PursuitVLA** ("An Executable Vision-Language-Action Policy for Cross-Map Zero-Shot Pursuit-Evasion Games," OpenReview id wUdio9L3lL) — appears to be an **ICLR 2027 submission**, not yet peer-reviewed/published; OpenReview page bot-blocked, info only via search-engine cache. **Status: unverified/unconfirmed publication status** — excluded from the numbered list; do not cite without independently confirming acceptance/arXiv posting.

---
#### Ranked shortlist (research pass, unedited; read with the fit note above)

Ranked by expected effect-per-cost given the existing alternating-stage self-play pipeline, PPO,
stored evader population, and learned evader-acceleration estimator + online RLS readout.

**1. PFSP-weighted / regret-weighted opponent sampling over the stored evader population.**
   - Regimes: **G1a (partial) + G1b (strong).**
   - Mechanism: replace "train against latest frozen evader stage" with a sampling distribution
     over the *entire* stored population, weighted by estimated pursuer win-rate deficit (PFSP,
     AlphaStar-style) or by a regret/value-loss proxy (Robust PLR⊥-style).
   - Expected effect: directly targets the "pursuer overfits its own campaign's opponent lineage"
     failure mode (G1b, measured +0.03 gap suggests weak but nonzero overfitting); AgilePE
     demonstrates this exact mechanism in an architecturally adjacent drone pursuit-evasion system.
   - Cost: **Low** — sampling-distribution change only, no architecture change, reuses existing
     stored population and win-rate bookkeeping.
   - Key citations: Vinyals et al. 2019 (AlphaStar league/PFSP, anchor); Jiang et al. 2021 (Robust
     PLR⊥); Tang et al. 2026 (AgilePE).

**2. Population-based adversary training (sample uniformly or regret-weighted from a *population*
   of evaders, not a single most-recent one) as the baseline hardening step.**
   - Regimes: **G1a (strong) + G1b (strong).**
   - Mechanism: Vinitsky et al.'s empirical result that a single adversary does not robustify a
     policy, directly explaining both measured gaps; maintaining/sampling a population closes this.
   - Expected effect: should move both the −0.10 G1a gap and the +0.03 G1b gap, since it is the
     load-bearing anchor explaining why single-lineage self-play underperforms against any
     sufficiently different (rule-shifted or seed-shifted) adversary.
   - Cost: **Medium** — requires restructuring the training loop to sample from the full stored
     population each episode/update rather than only the most-recent stage; you likely already
     store the population, so this is a sampling/infrastructure change, not new networks.
   - Key citations: Vinitsky et al. 2020 (Robust RL using Adversarial Populations); Pinto et al.
     2017 (RARL, as the single-adversary baseline being critiqued).

**3. Reward/rule randomization of the evader-generation process (RPG-style) to explicitly build
   a rule/reward-diverse G1a-style evader pool.**
   - Regimes: **G1a (strong), direct construction method.**
   - Mechanism: systematically randomize the evader's reward/rule weighting (risk-aversion, energy
     cost, capture-avoidance margin) across parallel training runs, keeping the resulting diverse
     evaders as a held-out-style pool, mirroring the "different rules/rewards" structure of your
     G1a test set but at controlled, graduated difficulty.
   - Expected effect: should reduce the −0.10 G1a gap directly, since it manufactures training-time
     exposure to the kind of reward/rule diversity your G1a test evaders embody, rather than
     relying on emergent self-play diversity (which tends to collapse to a narrow strategic basin,
     per the self-play-collapse literature in Family 5).
   - Cost: **Low-Medium** — parameterize the evader reward function, sample parameter vectors,
     launch parallel PPO runs; no architecture change, fits as a preprocessing step before/alongside
     the alternating-stage pipeline.
   - Key citations: Tang et al. 2021 (Reward Randomization for Policy Gradient, RPG); Mouret &
     Clune 2015 (MAP-Elites, as the archive structure to retain the resulting diversity).

**4. Fictitious Co-Play-style pooling of checkpoints across multiple independent self-play
   campaigns/seeds (not just within one campaign).**
   - Regimes: **G1b (strong), direct construction method.**
   - Mechanism: extend the stored evader population so that pursuer training partners are sampled
     from *multiple independently-seeded self-play campaigns' checkpoints*, not only the current
     campaign's lineage — this is the literal definition of what G1b tests against.
   - Expected effect: should close most of the residual G1b gap, since FCP was purpose-built for
     exactly this "overfits to one's own training partner population" failure mode and showed large
     gains (Melting Pot: up to 402% per RPM, a complementary ranked-sampling variant).
   - Cost: **Low-Medium** given existing infrastructure — mainly requires running (or re-using) a
     handful of independently-seeded self-play campaigns and pooling/ranked-sampling across their
     checkpoints during pursuer training.
   - Key citations: Strouse et al. 2021 (Fictitious Co-Play, FCP); Qiu et al. 2022 (RPM, ranked
     sampling refinement); Lupu et al. 2021 (TrajeDi, diversity-regularized population variant).

**5. Stage-boundary regularization against plasticity loss / early-stage overfitting (L2 weight
   decay, partial resets, CReLU) across the 13/26 alternating stages.**
   - Regimes: **G1a + G1b (both, as a general robustness-of-training floor, not a targeted fix).**
   - Mechanism: repeated frozen-opponent cycling is structurally close to the continual-learning
     task-cycling settings where plasticity loss and primacy bias are documented; add L2/AdamW
     weight decay and/or partial last-layer resets at each stage transition.
   - Expected effect: a cheap floor-raising intervention — diagnostic logging (activation sparsity,
     dead-unit fraction per stage) can directly test whether this explains part of both gaps before
     investing further; if confirmed, the fix is nearly free.
   - Cost: **Low** (weight decay / stage-boundary partial reset; logging-only diagnostic first).
   - Key citations: Nikishin et al. 2022 (Primacy Bias); Dohare et al. 2024 (Nature, Loss of
     Plasticity, L2+perturbation); Abbas et al. 2023 (Loss of Plasticity in Continual DRL, CReLU).
   - Caveat: all three anchors are single-agent/task-cycling results; application to multi-agent
     alternating self-play is extrapolation, not a verified transfer — treat as a cheap experiment
     to run, not a guaranteed fix, and report honestly either way.

**6. Contrastive / in-context opponent-style embedding feeding the existing RLS readout (upgrade
   path for the evader-acceleration estimator), e.g. OMIS- or CLAM-style.**
   - Regimes: **G1a (strong) + G1b (strong) — a test-time adaptation lever, not just a training-time one.**
   - Mechanism: replace or augment the RLS linear-regression adaptation with a learned embedding
     of evader policy/behavior (not just acceleration), trained in-context or via contrastive
     self-supervision over the stored population, enabling faster/better online inference of
     "what kind of evader is this" against a genuinely novel evader at deployment time.
   - Expected effect: directly targets the test-time generalization axis (rather than only the
     training-population-construction axis of interventions 1-4); OMIS's architectural overlap
     with your existing acceleration-estimator + RLS stack makes this the most natural "upgrade,"
     not a replacement subsystem.
   - Cost: **Medium** (OMIS-style) to **Medium-High** (full TAO-style offline transformer pipeline)
     — architectural overlap with existing components keeps this from being a from-scratch system,
     but decision-time search (OMIS) adds runtime compute that must be budgeted for an onboard
     real-time controller.
   - Key citations: Lu & Wu 2024 (OMIS); Jing et al. 2024 (TAO); Ma et al. 2023/2025 (CLAM).

**7. Adversarial-policy-style held-out-attacker evaluation protocol (train several independent
   attacker evaders with different algorithms/rewards against the frozen final pursuer) as the
   measurement backbone for G1a, reused iteratively during development.**
   - Regimes: **G1a (methodology, not a training fix per se, but informs which fix to prioritize).**
   - Mechanism: Gleave et al.'s adversarial-policies protocol, hardened by the Go-attack finding
     that one round of adversarial retraining does not close such gaps — run multiple independent
     attacker-training runs (not one) each development cycle to track whether an intervention is
     actually closing the G1a gap or just beating one incidental exploiter.
   - Expected effect: not a direct generalization fix, but prevents false confidence from a single
     "fresh-exploiter exploitability" number (which your current results already show can diverge
     from the held-out-evader gap: positive exploitability gain but negative G1a gap).
   - Cost: **Low** as an evaluation protocol (reuses existing training infra for the attacker
     policies); **Medium-High** only if also used as an iterative training-time defense loop.
   - Key citations: Gleave et al. 2020 (Adversarial Policies); Wang et al. 2023 (Adversarial
     Policies Beat Superhuman Go AIs, the persistence-of-vulnerability caution).

**8 (stretch, high cost/high ceiling). MAESTRO-style joint regret-based curriculum over
   (environment/evader-reward-config, evader-policy) pairs, replacing plain alternating self-play.**
   - Regimes: **G1a (strong) + G1b (strong) — theoretically the most complete single intervention.**
   - Mechanism: joint UED-style regret curriculum spanning both evader strategy-space configuration
     (covers G1a's "different rules/rewards") and population sampling (covers G1b's "different
     seed"), with provable minimax-regret guarantees.
   - Expected effect: highest theoretical ceiling of all interventions surveyed, explicitly
     targeting a 1v1 zero-sum continuous-control setting structurally identical to yours.
   - Cost: **High** — a dedicated chapter/follow-on paper rather than a drop-in ablation; requires
     a generator/sampler over evader-configuration space, an antagonist agent, and regret
     estimation infrastructure layered on the existing PPO pipeline.
   - Key citations: Samvelyan et al. 2023 (MAESTRO); Dennis et al. 2020 (PAIRED, foundational
     regret principle); Parker-Holder et al. 2022 (ACCEL, cheaper evolutionary variant).

**Suggested order of experimentation** (cheapest/fastest signal first): 5 → 1 → 3 → 4 → 7 → 2 → 6 → 8.

---

## 3. Consolidated reference list (BibTeX-style keys, with verification flags)

```bibtex
% ---- Candidate-adjacent drone/UAV PE papers (2025-2026) ----
@misc{tang2026agilepe,
  title        = {AgilePE: Autonomous UAV Pursuit-Evasion via Self-Play Reinforcement Learning},
  author       = {Tang, Wenhao and Chen, Tianyang and Cui, Zhejun and An, Boyuan and Chen, Jiayu and Zhang, Ruize and Liu, Huidong and Wu, Tianyue and Liao, Qingmin and Gao, Fei and Wang, Yu and Yu, Chao},
  year         = {2026},
  eprint       = {2608.14135},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2608.14135},
  note         = {Verified via arXiv abstract page; arXiv Comments "Under review" (checked 2026-11-04)}
}

@misc{roncero2026learned,
  title        = {Learned Controllers for Agile Quadrotors in Pursuit-Evasion Games},
  author       = {Roncero, Alejandro Sanchez and Cai, Yixi and Andersson, Olov and Ogren, Petter},
  year         = {2025},
  eprint       = {2506.02849},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2506.02849},
  note         = {v1 Jun 2025, v3 Jun 2026; framed as a "letter"; accepted-venue status unverified}
}

@inproceedings{gavin2026intercepting,
  title        = {Intercepting an Agile Target with Net-Carrying Drones using Competitive Multi-Agent Reinforcement Learning},
  author       = {Gavin, Timoth\'ee and Bronz, Murat},
  booktitle    = {2026 International Conference on Unmanned Aircraft Systems (ICUAS)},
  address      = {Corfu, Greece},
  year         = {2026},
  pages        = {759--766},
  doi          = {10.1109/ICUAS69441.2026.11598610},
  eprint       = {2607.05939},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2607.05939},
  note         = {Venue verified from the IEEE PDF and arXiv Journal-ref, 2026-11-04}
}

@inproceedings{gavin2025agile,
  title        = {Agile Interception of a Flying Target using Competitive Reinforcement Learning},
  author       = {Gavin, Timoth\'ee and Lacroix, Simon and Bronz, Murat},
  booktitle    = {Conference on Artificial Intelligence for Defence (AMIAD)},
  address      = {Rennes, France},
  year         = {2025},
  eprint       = {2603.16279},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2603.16279},
  note         = {Venue from arXiv Journal-ref; proceedings DOI not found}
}

@misc{yoshimura2026esnoa,
  title        = {Online Adaptive Reinforcement Learning with Echo State Networks for Non-Stationary Dynamics},
  author       = {Yoshimura, Aoi and Tanaka, Gouhei},
  year         = {2026},
  eprint       = {2602.06326},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2602.06326},
  note         = {Applied to CartPole/HalfCheetah dynamics, not opponent modeling}
}

@misc{liu2409multiuav,
  title        = {Online Planning for Multi-UAV Pursuit-Evasion in Unknown Environments Using Deep Reinforcement Learning},
  year         = {2024},
  eprint       = {2409.15866},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2409.15866},
  note         = {OPEN; IEEE RA-L 2025 (per arXiv comments). LSTM evader prediction in the RL observation, Crazyflie 2.1 hardware. Must-cite for (B). Authors (surnames, verified on arXiv): Chen, Yu, Li, Tang, Ji, Yang, Xu, Yang, Wang}
}

@misc{anon2411uavcoop,
  title        = {Autonomous Decision Making for UAV Cooperative Pursuit-Evasion Game with Reinforcement Learning},
  year         = {2024},
  eprint       = {2411.02983},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2411.02983},
  note         = {Optional broader-context citation}
}

% ---- Diversity / DPP / PSRO lineage (candidate A) ----
@inproceedings{perez-nieves2021modelling,
  title        = {Modelling Behavioural Diversity for Learning in Open-Ended Games},
  author       = {Perez-Nieves, Nicolas and Yang, Yaodong and Slumbers, Oliver and Mguni, David Henry and Wen, Ying and Wang, Jun},
  booktitle    = {ICML},
  year         = {2021},
  eprint       = {2103.07927},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2103.07927}
}

@inproceedings{liu2021unifying,
  title        = {Towards Unifying Behavioral and Response Diversity for Open-ended Learning in Zero-sum Games},
  author       = {Liu, Xiangyu and Jia, Hangtian and Wen, Ying and Hu, Yujing and Chen, Yingfeng and Fan, Changjie and Hu, Zhipeng and Yang, Yaodong},
  booktitle    = {NeurIPS},
  year         = {2021},
  volume       = {34},
  pages        = {941--952},
  note         = {Author order as in the NeurIPS proceedings (arXiv lists Yang fourth)},
  eprint       = {2106.04958},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2106.04958}
}

@inproceedings{liu2022unified,
  title        = {A Unified Diversity Measure for Multiagent Reinforcement Learning},
  author       = {Liu, Zongkai and Yu, Chao and Yang, Yaodong and Sun, Peng and Wu, Zifan and Li, Yuan},
  booktitle    = {NeurIPS},
  year         = {2022},
  volume       = {35},
  pages        = {10339--10352},
  doi          = {10.52202/068431-0751},
  url          = {https://proceedings.neurips.cc/paper_files/paper/2022/file/435cce71b4007699041dfffa4f034079-Paper-Conference.pdf},
  note         = {No arXiv preprint; full proceedings PDF read 2026-11-04. Six authors per proceedings metadata, OpenReview and DBLP; the PDF title block omits Yuan Li}
}

@inproceedings{yao2023policy,
  title        = {Policy Space Diversity for Non-Transitive Games},
  author       = {Yao, Jian and Liu, Weiming and Fu, Haobo and Yang, Yaodong and McAleer, Stephen and Fu, Qiang and Yang, Wei},
  booktitle    = {NeurIPS},
  year         = {2023},
  eprint       = {2306.16884},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2306.16884}
}

@inproceedings{balduzzi2019open,
  title        = {Open-ended Learning in Symmetric Zero-sum Games},
  author       = {Balduzzi, David and Garnelo, Marta and Bachrach, Yoram and Czarnecki, Wojciech M. and Perolat, Julien and Jaderberg, Max and Graepel, Thore},
  booktitle    = {ICML},
  year         = {2019},
  eprint       = {1901.08106},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/1901.08106}
}

@article{vinyals2019grandmaster,
  title        = {Grandmaster level in StarCraft II using multi-agent reinforcement learning},
  author       = {Vinyals, Oriol and Babuschkin, Igor and Czarnecki, Wojciech M. and Mathieu, Micha\"el and Dudzik, Andrew and others},
  journal      = {Nature},
  volume       = {575},
  pages        = {350--354},
  year         = {2019},
  doi          = {10.1038/s41586-019-1724-z},
  note         = {No arXiv preprint; Nature-only publication}
}

@article{kulesza2012determinantal,
  title        = {Determinantal Point Processes for Machine Learning},
  author       = {Kulesza, Alex and Taskar, Ben},
  journal      = {Foundations and Trends in Machine Learning},
  volume       = {5},
  number       = {2--3},
  pages        = {123--286},
  year         = {2012},
  eprint       = {1207.6083},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/1207.6083}
}

@article{friedman2023vendi,
  title        = {The Vendi Score: A Diversity Evaluation Metric for Machine Learning},
  author       = {Friedman, Dan and Dieng, Adji Bousso},
  year         = {2023},
  eprint       = {2210.02410},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2210.02410},
  journal      = {Transactions on Machine Learning Research},
  note         = {TMLR 07/2023, OpenReview g97OHbQyk1 (stated on the arXiv v2 PDF title page)}
}

% ---- Online adaptation lineage (candidate B) ----
@article{oconnell2022neuralfly,
  title        = {Neural-Fly Enables Rapid Learning for Agile Flight in Strong Winds},
  author       = {O'Connell, Michael and Shi, Guanya and Shi, Xichen and Azizzadenesheli, Kamyar and Anandkumar, Anima and Yue, Yisong and Chung, Soon-Jo},
  journal      = {Science Robotics},
  volume       = {7},
  number       = {66},
  pages        = {eabm6597},
  year         = {2022},
  doi          = {10.1126/scirobotics.abm6597},
  eprint       = {2205.06908},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2205.06908}
}

@inproceedings{kumar2021rma,
  title        = {RMA: Rapid Motor Adaptation for Legged Robots},
  author       = {Kumar, Ashish and Fu, Zipeng and Pathak, Deepak and Malik, Jitendra},
  booktitle    = {RSS},
  year         = {2021},
  doi          = {10.15607/RSS.2021.XVII.011},
  eprint       = {2107.04034},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2107.04034}
}

@inproceedings{al-shedivat2018continuous,
  title        = {Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments},
  author       = {Al-Shedivat, Maruan and Bansal, Trapit and Burda, Yuri and Sutskever, Ilya and Mordatch, Igor and Abbeel, Pieter},
  booktitle    = {ICLR},
  year         = {2018},
  eprint       = {1710.03641},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/1710.03641}
}

% ---- Bayesian opponent identification lineage (candidate C) ----
@inproceedings{fu2022greedy,
  title        = {Greedy when Sure and Conservative when Uncertain about the Opponents},
  author       = {Fu, Haobo and Tian, Ye and Yu, Hongxiang and Liu, Weiming and Wu, Shuang and Xiong, Jiechao and Wen, Ying and Li, Kai and Xing, Junliang and Fu, Qiang and Yang, Wei},
  booktitle    = {ICML},
  year         = {2022},
  volume       = {PMLR v162},
  pages        = {6829--6848},
  url          = {https://proceedings.mlr.press/v162/fu22b.html},
  note         = {No arXiv version; code at github.com/YeTianJHU/GSCU; benchmarks incl. Kuhn Poker and Predator-Prey}
}

@article{rosman2016bayesian,
  title        = {Bayesian Policy Reuse},
  author       = {Rosman, Benjamin and Hawasly, Majd and Ramamoorthy, Subramanian},
  journal      = {Machine Learning},
  volume       = {104},
  number       = {1},
  pages        = {99--127},
  year         = {2016},
  doi          = {10.1007/s10994-016-5547-y},
  eprint       = {1505.00284},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/1505.00284}
}

@inproceedings{yang2019bayestomop,
  title        = {Towards Efficient Detection and Optimal Response against Sophisticated Opponents},
  author       = {Yang, Tianpei and Hao, Jianye and Meng, Zhaopeng and Zhang, Chongjie and Zheng, Yan and Zheng, Ze},
  booktitle    = {IJCAI},
  year         = {2019},
  pages        = {623--629},
  doi          = {10.24963/ijcai.2019/88},
  eprint       = {1809.04240},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/1809.04240},
  note         = {Method name Bayes-ToMoP; verified from full PDF and Crossref 2026-11-04. An AAMAS 2019 extended abstract (pp. 2282--2284) carries the "Bayes-ToMoP" title}
}

% ---- Added by the 2026-11-04 citation chase (see section 2). Author lists only where verified; complete them from the source before use ----
@inproceedings{sinha2020formulazero,
  title        = {FormulaZero: Distributionally Robust Online Adaptation via Offline Population Synthesis},
  author       = {Sinha, Aman and O'Kelly, Matthew and Zheng, Hongrui and Mangharam, Rahul and Duchi, John and Tedrake, Russ},
  year         = {2020},
  eprint       = {2003.03900},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2003.03900},
  note         = {Candidate (A). Venue (ICML 2020) not checked}
}

@misc{zheng2022gtosp,
  title        = {Game-theoretic Objective Space Planning},
  author       = {Zheng, Hongrui and Zhuang, Zhijun and Betz, Johannes and Mangharam, Rahul},
  year         = {2022},
  eprint       = {2209.07758},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2209.07758},
  note         = {Candidate (A); DPP subset selection of racing opponents}
}

@inproceedings{si2019agen,
  title        = {AGen: Adaptable Generative Prediction Networks for Autonomous Driving},
  author       = {Si and Wei and Liu},
  booktitle    = {IEEE Intelligent Vehicles Symposium (IV)},
  year         = {2019},
  doi          = {10.1109/IVS.2019.8814238},
  note         = {Candidate (B); offline basis + online RLS-PAA last layer; PDF read 2026-11-04. Complete first names from source}
}

@misc{cheng2018semiadaptable,
  title        = {Human Motion Prediction using Semi-adaptable Neural Networks},
  author       = {Cheng and Zhao and Liu and Tomizuka},
  year         = {2018},
  eprint       = {1810.00781},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/1810.00781},
  note         = {Candidate (B); complete first names from source}
}

@article{li2026psto,
  title        = {PSTO (full title: check arXiv:2603.24238)},
  author       = {Li, Yude and Zhou, Zhexuan and Li, Huizhe and Sun, Yanke and Wu, Yenan and Lai, Yichen and Wang, Yiming and Gong, Youmin and Mei, Jie},
  journal      = {IEEE Robotics and Automation Letters},
  year         = {2026},
  doi          = {10.1109/LRA.2026.3700665},
  eprint       = {2603.24238},
  archivePrefix= {arXiv},
  note         = {Candidate (B); RA-L venue per OpenAlex; title to be filled in from source}
}

@article{li2026safe,
  title        = {Safe Cooperative Decision-Making for Multi-UAV Pursuit--Evasion Games via Opponent Intent Inference},
  author       = {Li, Wenxin and Feng, Yongxin and Zhang, Wenbo},
  journal      = {Sensors},
  volume       = {26},
  number       = {7},
  pages        = {2243},
  year         = {2026},
  doi          = {10.3390/s26072243},
  note         = {Candidates (B), (C); verified via Crossref and Europe PMC (PMC13075325) 2026-11-04}
}

@article{wang2026eomaddpg,
  title        = {EO-MADDPG: An Improved Reinforcement Learning Approach for Multi-UAV Pursuit--Evasion Games},
  author       = {Wang, Xiao and Wang, Mengyu and Bai, Xueqian and Ma, Zhe and Sun, Kewu and Li, Jiake},
  journal      = {Aerospace},
  volume       = {13},
  number       = {3},
  pages        = {296},
  year         = {2026},
  doi          = {10.3390/aerospace13030296},
  note         = {Verified via Crossref abstract only, 2026-11-04}
}

@misc{kalanther2025evader,
  title        = {Evader-Agnostic Team-Based Pursuit Strategies in Partially-Observable Environments},
  author       = {Kalanther and Bostwick and Maheshwari and Sastry},
  year         = {2025},
  eprint       = {2511.05812},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2511.05812},
  note         = {Candidate (C); complete first names from source}
}

@misc{kalanther2026orchestrating,
  title        = {Orchestrating Level-K Policies Against Unknown Opponents in Partially-Observable Dynamic Games},
  author       = {Kalanther and Bharvirkar and Bostwick and Maheshwari and Sastry},
  year         = {2026},
  eprint       = {2610.04937},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2610.04937},
  note         = {Candidate (C); argues against identify-then-respond. Complete first names from source}
}

% ---- Related population-learning architecture, found during novelty search ----
@inproceedings{liu2022neupl,
  title        = {NeuPL: Neural Population Learning},
  author       = {Liu, Siqi and Marris, Luke and Hennes, Daniel and Merel, Josh and Heess, Nicolas and Graepel, Thore},
  booktitle    = {ICLR},
  year         = {2022},
  eprint       = {2202.07415},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2202.07415},
  note         = {Discovered during novelty search; single conditional network for population, not DPP-sampling — relevant contrast for (A)}
}

% ---- Section (E): generalization to held-out / cross-campaign evaders (2026-11-15). Duplicates of existing keys omitted: use liu2022neupl, roncero2026learned, tang2026agilepe, vinyals2019grandmaster, fu2022greedy, al-shedivat2018continuous ----
@article{heinrich2016deep,
  title   = {Deep Reinforcement Learning from Self-Play in Imperfect-Information Games},
  author  = {Heinrich, Johannes and Silver, David},
  journal = {arXiv:1603.01121},
  year    = {2016},
  note    = {Verified via arXiv abstract page.}
}


@inproceedings{liu2022simplex,
  title     = {Simplex Neural Population Learning: Any-Mixture Bayes-Optimality in Symmetric Zero-sum Games},
  author    = {Liu, Siqi and Marris, Luke and Hennes, Daniel and others},
  booktitle = {ICML},
  year      = {2022},
  note      = {Verified via arXiv:2205.15879 abstract page.}
}


@inproceedings{mcaleer2022anytime,
  title     = {Anytime PSRO for Two-Player Zero-Sum Games},
  author    = {McAleer, Stephen and Wang, Kevin and Lanier, John B. and Lanctot, Marc and Baldi, Pierre and Sandholm, Tuomas and Fox, Roy},
  booktitle = {AAAI RL-in-Games Workshop},
  year      = {2022},
  note      = {Verified via arXiv:2201.07700 abstract page.}
}


@article{mcaleer2022selfplay,
  title   = {Self-Play PSRO: Toward Optimal Populations in Two-Player Zero-Sum Games},
  author  = {McAleer, Stephen and Lanier, John B. and Wang, Kevin and Baldi, Pierre and Fox, Roy and Sandholm, Tuomas},
  journal = {arXiv:2207.06541},
  year    = {2022},
  note    = {Verified via arXiv abstract page.}
}


@article{jaderberg2017population,
  title   = {Population Based Training of Neural Networks},
  author  = {Jaderberg, Max and Dalibard, Valentin and Osindero, Simon and Czarnecki, Wojciech M. and others},
  journal = {arXiv:1711.09846},
  year    = {2017},
  note    = {Verified via arXiv abstract page.}
}


@article{mouret2015illuminating,
  title   = {Illuminating Search Spaces by Mapping Elites},
  author  = {Mouret, Jean-Baptiste and Clune, Jeff},
  journal = {arXiv:1504.04909},
  year    = {2015},
  note    = {Verified via arXiv abstract page.}
}


@inproceedings{tang2021discovering,
  title     = {Discovering Diverse Multi-Agent Strategic Behavior via Reward Randomization},
  author    = {Tang, Zhenggang and Yu, Chao and Chen, Boyuan and Xu, Huazhe and Wang, Xiaolong and Fang, Fei and Du, Simon S. and Wang, Yu and Wu, Yi},
  booktitle = {ICLR},
  year      = {2021},
  note      = {Verified via arXiv:2103.04564 abstract page.}
}


@article{anne2025game,
  title   = {Adversarial Coevolutionary Illumination with Generational Adversarial MAP-Elites},
  author  = {Anne, Timoth{\'e}e and others},
  journal = {IEEE Transactions on Evolutionary Computation},
  year    = {2026},
  doi     = {10.1109/TEVC.2026.3686956},
  note    = {Verified via arXiv abstract page (ALIFE 2025 preprint) and journal DOI.}
}


@article{memes2023,
  title   = {Multiple Hands Make Light Work: Enhancing Quality and Diversity using MAP-Elites with Multiple Parallel Evolution Strategies},
  author  = {{Unconfirmed authors}},
  journal = {arXiv:2303.06137},
  year    = {2023},
  note    = {UNVERIFIED: arXiv ID/title corroborated via secondary citations only; author list not independently confirmed. Fetch abstract page before citing.}
}


@inproceedings{ingvarsson2023mixme,
  title     = {Mix-ME: Quality-Diversity for Multi-Agent Learning},
  author    = {Ingvarsson, Gauti and Samvelyan, Mikayel and Lim, Bryan and Flageat, Manon and Cully, Antoine and Rockt{\"a}schel, Tim},
  booktitle = {NeurIPS ALOE Workshop},
  year      = {2023},
  note      = {Verified via arXiv:2311.01829 abstract page.}
}


@article{zhang2024survey,
  title   = {A Survey on Self-play Methods in Reinforcement Learning},
  author  = {Zhang, Ruize and Xu, Zelai and Ma, Chengdong and Yu, Chao and Tu, Wei-Wei and Tang, Wenhao and Huang, Shiyu and Ye, Deheng and Ding, Wenbo and Yang, Yaodong and Wang, Yu},
  journal = {arXiv:2408.01072},
  year    = {2024},
  note    = {Verified via arXiv abstract page (v4 2025).}
}


@misc{jeon2026record,
  title  = {ReCord: Replay Coordination for Safe and Robust Population-Based Training in Autonomous Driving},
  author = {Jeon, Hyun-Chul and Kim, Kyung-Joong},
  year   = {2026},
  note   = {UNVERIFIED: OpenReview id yf1h6EUF88 blocked by bot-check; title/mechanism corroborated via secondary search only. Confirm naming ("ReCord" vs "Recorded-PBT") and authors before citing.}
}


@article{chen2025open,
  title   = {Online Planning for Multi-UAV Pursuit-Evasion in Unknown Environments Using Deep Reinforcement Learning},
  author  = {Chen, Jiayu and Yu, Chao and Li, Guosheng and Tang, Wenhao and Ji, Shilong and Yang, Xinyi and Xu, Botian and Yang, Huazhong and Wang, Yu},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2025},
  doi     = {10.1109/LRA.2025.3583620},
  note    = {Verified via arXiv:2409.15866 abstract page and journal DOI.}
}


@article{racing2025excluded,
  title   = {On Learning Racing Policies with Reinforcement Learning},
  author  = {{Unconfirmed authors}},
  journal = {arXiv:2504.02420},
  year    = {2025},
  note    = {Verified as real (IROS 2025) but excluded: single-agent racing vs. MPC/human baselines, no opponent population or self-play.}
}


@inproceedings{pinto2017robust,
  title     = {Robust Adversarial Reinforcement Learning},
  author    = {Pinto, Lerrel and Davidson, James and Sukthankar, Rahul and Gupta, Abhinav},
  booktitle = {ICML},
  year      = {2017},
  note      = {Verified via arXiv:1703.02702 abstract page.}
}


@article{vinitsky2020robust,
  title   = {Robust Reinforcement Learning using Adversarial Populations},
  author  = {Vinitsky, Eugene and Du, Yuqing and Parvate, Kanaad and Jang, Kathy and Abbeel, Pieter and Bayen, Alexandre},
  journal = {arXiv:2008.01825},
  year    = {2020},
  note    = {Verified via arXiv abstract page.}
}


@inproceedings{gleave2020adversarial,
  title     = {Adversarial Policies: Attacking Deep Reinforcement Learning},
  author    = {Gleave, Adam and Dennis, Michael and Wild, Cody and Kant, Neel and Levine, Sergey and Russell, Stuart},
  booktitle = {ICLR},
  year      = {2020},
  note      = {Verified via arXiv:1905.10615 abstract page.}
}


@inproceedings{wang2023adversarial,
  title     = {Adversarial Policies Beat Superhuman Go AIs},
  author    = {Wang, Tony T. and Gleave, Adam and Tseng, Tom and Pelrine, Kellin and Belrose, Nora and Miller, Joseph and Dennis, Michael D. and Duan, Yawen and Pogrebniak, Viktor and Levine, Sergey and Russell, Stuart},
  booktitle = {ICML},
  year      = {2023},
  note      = {Verified via arXiv:2211.00241 abstract page.}
}


@inproceedings{zhang2021robust,
  title     = {Robust Reinforcement Learning on State Observations with Learned Optimal Adversary},
  author    = {Zhang, Huan and Chen, Hongge and Boning, Duane and Hsieh, Cho-Jui},
  booktitle = {ICLR},
  year      = {2021},
  note      = {Verified via arXiv:2101.08452 abstract page.}
}


@inproceedings{dennis2020emergent,
  title     = {Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design},
  author    = {Dennis, Michael and Jaques, Natasha and Vinitsky, Eugene and Bayen, Alexandre and Russell, Stuart and Critch, Andrew and Levine, Sergey},
  booktitle = {NeurIPS},
  year      = {2020},
  note      = {Verified via arXiv:2012.02096 abstract page.}
}


@inproceedings{jiang2021replay,
  title     = {Replay-Guided Adversarial Environment Design},
  author    = {Jiang, Minqi and Dennis, Michael and Parker-Holder, Jack and Foerster, Jakob and Grefenstette, Edward and Rockt{\"a}schel, Tim},
  booktitle = {NeurIPS},
  year      = {2021},
  note      = {Verified via arXiv:2110.02439 abstract page.}
}


@inproceedings{parkerholder2022evolving,
  title     = {Evolving Curricula with Regret-Based Environment Design},
  author    = {Parker-Holder, Jack and Jiang, Minqi and Dennis, Michael and Samvelyan, Mikayel and Foerster, Jakob and Grefenstette, Edward and Rockt{\"a}schel, Tim},
  booktitle = {ICML},
  year      = {2022},
  note      = {Verified via arXiv:2203.01302 abstract page.}
}


@inproceedings{samvelyan2023maestro,
  title     = {MAESTRO: Open-Ended Environment Design for Multi-Agent Reinforcement Learning},
  author    = {Samvelyan, Mikayel and Khan, Akbir and Dennis, Michael and others},
  booktitle = {ICLR},
  year      = {2023},
  note      = {Verified via arXiv:2303.03376 abstract page.}
}


@inproceedings{akgun2025spiral,
  title     = {SPIRAL: Self-Play Incremental Racing Algorithm for Learning in Multi-Drone Competitions},
  author    = {Akgun, Onur and others},
  booktitle = {IEEE ASYU},
  year      = {2025},
  doi       = {10.1109/ASYU67174.2025.11208263},
  note      = {Verified via arXiv:2510.22568 abstract page.}
}


@article{schott2024robust,
  title   = {Robust Deep Reinforcement Learning Through Adversarial Attacks and Training: A Survey},
  author  = {Schott, Lucas and others},
  journal = {arXiv:2403.00420},
  year    = {2024},
  note    = {Verified via arXiv abstract page (v3 2026).}
}


@article{standen2025adversarial,
  title   = {Adversarial Machine Learning Attacks and Defences in Multi-Agent Reinforcement Learning},
  author  = {Standen, Maxwell and Kim, Joshua and Szabo, Claudia},
  journal = {ACM Computing Surveys},
  volume  = {57},
  number  = {5},
  year    = {2025},
  doi     = {10.1145/3708320},
  note    = {PARTIALLY VERIFIED: ACM DL fetch 403-blocked; metadata corroborated via secondary search.}
}


@misc{ma2026robust,
  title  = {Robust Multi-Agent Reinforcement Learning with Diverse Adversarial Agent Generation and Contrastive Policy Encoding},
  author = {Ma, Wenhao and Chang, Yu-Cheng and Shen, Yang and Zhao, Ziyi and Lin, Chin-Teng},
  year   = {2026},
  note   = {UNVERIFIED: OpenReview id h9isaqF956 blocked by bot-check; title/author list corroborated via secondary search only. Confirm ICLR 2026 acceptance status before citing.}
}


@inproceedings{hu2020otherplay,
  title     = {"Other-Play" for Zero-Shot Coordination},
  author    = {Hu, Hengyuan and Lerer, Adam and Peysakhovich, Alexander and Foerster, Jakob},
  booktitle = {ICML},
  year      = {2020},
  note      = {Verified via arXiv:2003.02979 abstract page.}
}


@inproceedings{hu2021offbelief,
  title     = {Off-Belief Learning},
  author    = {Hu, Hengyuan and Lerer, Adam and Cui, Brandon and Pineda, Luis and Brown, Noam and Foerster, Jakob},
  booktitle = {ICML},
  year      = {2021},
  note      = {Verified via arXiv:2103.04000 abstract page.}
}


@inproceedings{lupu2021trajedi,
  title     = {Trajectory Diversity for Zero-Shot Coordination},
  author    = {Lupu, Andrei and Cui, Brandon and Hu, Hengyuan and Foerster, Jakob},
  booktitle = {ICML},
  year      = {2021},
  note      = {Verified via PMLR v139:7204-7213 (no distinct arXiv ID found; cite PMLR as canonical).}
}


@inproceedings{strouse2021fcp,
  title     = {Collaborating with Humans without Human Data},
  author    = {Strouse, D. J. and McKee, Kevin and Botvinick, Matt and Hughes, Edward and Everett, Richard},
  booktitle = {NeurIPS},
  year      = {2021},
  note      = {Verified via arXiv:2110.08176 abstract page (Spotlight).}
}


@article{qiu2022rpm,
  title   = {RPM: Generalizable Behaviors for Multi-Agent Reinforcement Learning},
  author  = {Qiu, Wei and others},
  journal = {arXiv:2210.09646},
  year    = {2022},
  note    = {Verified via arXiv abstract page (ICLR 2023 preprint).}
}


@inproceedings{jiang2024rppo,
  title     = {Learning Diverse Risk Preferences in Population-Based Self-Play},
  author    = {Jiang, Ziyi and Liu, Yu and Ma, Jiarui and Li, Kun and Yang, Yaodong and Yang, Xinyi and Liang, Shuai and Zhao, Dongbin},
  booktitle = {AAAI},
  year      = {2024},
  note      = {Verified via arXiv:2305.11476 abstract page.}
}


@misc{qdsp2024neurips,
  title = {Quality-Diversity Self-Play},
  year  = {2024},
  note  = {NeurIPS 2024 poster. Verified via neurips.cc/virtual/2024/100930 abstract fetch.}
}


@article{rusu2015policydistillation,
  title   = {Policy Distillation},
  author  = {Rusu, Andrei A. and others},
  journal = {arXiv:1511.06295},
  year    = {2015},
  note    = {Verified via arXiv abstract page (ICLR 2016).}
}


@inproceedings{wortsman2022modelsoups,
  title     = {Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time},
  author    = {Wortsman, Mitchell and Ilharco, Gabriel and Gadre, Samir Yitzhak and others},
  booktitle = {ICML},
  year      = {2022},
  note      = {Verified via arXiv:2203.05482 abstract page.}
}


@inproceedings{papoudakis2021liam,
  title     = {Agent Modelling under Partial Observability for Deep Reinforcement Learning},
  author    = {Papoudakis, Georgios and Christianos, Filippos and Albrecht, Stefano V.},
  booktitle = {NeurIPS},
  year      = {2021},
  note      = {Verified via arXiv:2006.09447 abstract page. Anchor paper, already in notes as "LIAM."}
}


@article{laskin2022algorithmdistillation,
  title   = {In-context Reinforcement Learning with Algorithm Distillation},
  author  = {Laskin, Michael and Wang, Luyu and Oh, Junhyuk and Parisotto, Emilio and others},
  journal = {arXiv:2210.14215},
  year    = {2022},
  note    = {Verified via arXiv abstract page (ICLR 2023).}
}


@inproceedings{jing2024tao,
  title     = {Towards Offline Opponent Modeling with In-context Learning},
  author    = {Jing, Yuheng and Li, Yue and Liu, Hongru and Zang, Qingyuan and Fu, Siqi and Fu, Qiang and Xing, Jiangcheng and Cheng, Jun},
  booktitle = {ICLR},
  year      = {2024},
  note      = {Verified via ICLR proceedings PDF fetch; OpenReview HTML bot-blocked, author list cross-checked via secondary search.}
}


@inproceedings{lu2024omis,
  title     = {Opponent Modeling with In-context Search},
  author    = {Lu, Di and Wu, Yan},
  booktitle = {NeurIPS},
  year      = {2024},
  note      = {Verified via NeurIPS official proceedings abstract page fetch.}
}


@article{zhang2024holadrone,
  title   = {HOLA-Drone: Hypergraphic Open-ended Learning for Zero-Shot Multi-Drone Cooperative Pursuit},
  author  = {Zhang, Ruize and others},
  journal = {arXiv:2409.08767},
  year    = {2024},
  note    = {Verified via ar5iv mirror fetch. Cooperative (unseen teammates), not adversarial (unseen evaders) -- cross-reference only.}
}


@article{lu2025equilibriumpolicygeneralization,
  title   = {Equilibrium Policy Generalization: A Reinforcement Learning Framework for Cross-Graph Zero-Shot Generalization in Pursuit-Evasion Games},
  author  = {Lu, Runyu and Zhang, Peng and Shi, Ruochuan and Zhu, Yuanheng and Zhao, Dongbin and Liu, Yang and Wang, Dong and Alippi, Cesare},
  journal = {NeurIPS},
  pages   = {48810--48841},
  year    = {2025},
  note    = {Verified via arXiv:2511.00811 abstract page and NeurIPS proceedings cross-check.}
}


@article{lanier2026emagnet,
  title   = {EMAgnet: Parameter-Space EMA Regularization for Policy Gradient Self-Play in Large Games},
  author  = {Lanier, John B. and others},
  journal = {arXiv:2606.23995},
  year    = {2026},
  note    = {Verified via arXiv abstract page (ICML 2026 NExT-Game Workshop).}
}


@misc{ieee2024entropyreg,
  title  = {Proximal Policy Optimization with Entropy Regularization},
  year   = {2024},
  note   = {UNVERIFIED: paywalled (IEEE Xplore 10585473), search-snippet only.}
}


@misc{mdpi2026entropysurvey,
  title  = {Entropy Regularization in Deep Reinforcement Learning: A Structured Survey},
  journal = {Entropy (MDPI)},
  volume  = {28},
  number  = {7},
  year    = {2026},
  note    = {UNVERIFIED: not independently fetched, search-synthesis only.}
}


@inproceedings{raileanu2021drac,
  title     = {Automatic Data Augmentation for Generalization in Deep Reinforcement Learning},
  author    = {Raileanu, Roberta and Goldstein, Max and Yarats, Denis and Kostrikov, Ilya and Fergus, Rob},
  booktitle = {NeurIPS},
  year      = {2021},
  note      = {Verified via arXiv:2006.12862 abstract page. Procgen test-set improvement ~40% relative (see results tables).}
}


@article{zhang2021policyaware,
  title   = {Generalization of Reinforcement Learning with Policy-Aware Adversarial Data Augmentation},
  author  = {Zhang, Hanping and Guo, Yuhong},
  journal = {arXiv:2106.15587},
  year    = {2021},
  note    = {Verified via arXiv abstract page.}
}


@inproceedings{nikishin2022primacy,
  title     = {The Primacy Bias in Deep Reinforcement Learning},
  author    = {Nikishin, Evgenii and Schwarzer, Max and D'Oro, Pierluca and Bacon, Pierre-Luc and Courville, Aaron},
  booktitle = {ICML},
  year      = {2022},
  note      = {Verified via arXiv:2205.07802 abstract page.}
}


@inproceedings{abbas2023lossofplasticity,
  title     = {Loss of Plasticity in Continual Deep Reinforcement Learning},
  author    = {Abbas, Zaheer and Zhao, Rosie and Modayil, Joseph and White, Adam and Machado, Marlos C.},
  booktitle = {CoLLAs},
  year      = {2023},
  note      = {Verified via arXiv:2303.07507 abstract page (PMLR v232).}
}


@article{dohare2024lossofplasticity,
  title   = {Loss of Plasticity in Deep Continual Learning},
  author  = {Dohare, Shibhansh and Hernandez-Garcia, J. Fernando and Lan, Qingfeng and Rahman, Parash and Mahmood, A. Rupam and Sutton, Richard S.},
  journal = {Nature},
  volume  = {632},
  year    = {2024},
  doi     = {10.1038/s41586-024-07711-7},
  note    = {Verified via arXiv:2306.13812 abstract page; Nature publisher page itself not independently fetched (406 blocked).}
}


@article{chen2024offfsp,
  title   = {Offline Fictitious Self-Play for Competitive Games},
  author  = {Chen, Jingxiao and others},
  journal = {arXiv:2403.00841},
  year    = {2024},
  note    = {Verified via arXiv abstract page (v2 2025).}
}


@inproceedings{lee2026ocnr,
  title     = {Stabilizing Self-Play by Mitigating Iteration-Collapse With One-Class Novelty Rewards},
  author    = {Lee, Seungyoo and Nam, Giung and Lee, Hyungi and Lee, Juho},
  booktitle = {ICML},
  year      = {2026},
  note      = {Verified via PMLR v306:64837-64861. LLM self-play domain, cross-domain analogy only.}
}


@article{li2026rdiverse,
  title   = {R-Diverse: Mitigating Diversity Illusion in Self-Play LLM Training},
  author  = {Li, Gengsheng and others},
  journal = {arXiv:2602.13103},
  year    = {2026},
  note    = {Verified via arXiv abstract page. LLM self-play domain, cross-domain analogy only.}
}


@article{chen2024dualcl,
  title   = {A Dual Curriculum Learning Framework for Multi-UAV Pursuit-Evasion in Diverse Environments},
  author  = {Chen, Jiayu and Li, Guosheng and Yu, Chao and Yang, Xinyi and Xu, Botian and Yang, Huazhong and Wang, Yu},
  journal = {arXiv:2312.12255},
  year    = {2024},
  note    = {Verified via arXiv abstract page and HTML full text.}
}


@inproceedings{wang2024viper,
  title     = {ViPER: Visibility-Based Pursuit-Evasion via Reinforcement Learning},
  author    = {Wang, Yizhuo and Cao, Yuhong and Chiun, Jimmy and Koley, Subhadeep and Pham, Mandy and Sartoretti, Guillaume Adrien},
  booktitle = {CoRL},
  year      = {2024},
  note      = {Verified via mlanthology.org official proceedings page (PMLR v270, pp.4188--4200).}
}


@article{mdpi2025distributedpeg,
  title   = {Distributed Pursuit-Evasion Game Decision-Making Based on Multi-Agent Deep Reinforcement Learning},
  journal = {Electronics (MDPI)},
  volume  = {14},
  number  = {11},
  pages   = {2141},
  year    = {2025},
  note    = {Verified via direct fetch (r.jina.ai proxy due to 403 on direct access).}
}


@article{mdpi2025scalablepeg,
  title   = {Scalable Pursuit-Evasion Game for Multi-Fixed-Wing UAV Based on Dynamic Target Assignment and Hierarchical Reinforcement Learning},
  journal = {Drones (MDPI)},
  volume  = {10},
  number  = {1},
  pages   = {5},
  year    = {2025},
  note    = {Verified via direct fetch (r.jina.ai proxy).}
}


@incollection{springer2024geddpg,
  title   = {Guided Exploration Reinforcement Learning for UAV Pursuit-Evasion Games},
  year    = {2024},
  doi     = {10.1007/978-3-031-70684-4_22},
  note    = {Verified title/abstract/references via Springer fetch; author list unconfirmed (paywalled).}
}


@article{mdpi2024mrpa,
  title   = {Evade Unknown Pursuer via Pursuit Strategy Identification and Model Reference Policy Adaptation},
  journal = {Drones (MDPI)},
  volume  = {8},
  number  = {11},
  pages   = {655},
  year    = {2024},
  note    = {Verified via direct fetch (r.jina.ai proxy). Evader-side generalization gap, informative by symmetry.}
}


@article{akinmolayan2026hybrid,
  title   = {Adaptive Interception in Dynamic Domains: Exploration of Hybrid Reinforcement Learning in Pursuit-Evasion Tasks},
  author  = {Akinmolayan, Matthew and Josyula, Darsana and Casbeer, David},
  journal = {Proceedings of the AAAI Symposium Series},
  volume  = {8},
  number  = {1},
  pages   = {2--10},
  year    = {2026},
  doi     = {10.1609/aaaiss.v8i1.42510},
  note    = {Verified via direct fetch (r.jina.ai proxy) of AAAI OJS page.}
}


@article{ganai2024hji,
  title   = {Hamilton-Jacobi Reachability in Reinforcement Learning: A Survey},
  author  = {Ganai, Milan and others},
  journal = {IEEE Open Journal of Control Systems},
  year    = {2024},
  note    = {Verified via arXiv:2407.09645 abstract page.}
}


@article{lauffer2025rpg,
  title   = {Robust and Diverse Multi-Agent Learning via Rational Policy Gradient},
  author  = {Lauffer, Niklas and Shah, Ameesh and Carroll, Micah and Seshia, Sanjit A. and Russell, Stuart and Dennis, Michael},
  journal = {arXiv:2511.09535},
  year    = {2025},
  note    = {Verified via arXiv abstract page (NeurIPS 2025). Distinct from Tang et al. 2021 "RPG" (Reward Randomization) -- same acronym, different paper.}
}


@article{ma2023clam,
  title   = {Contrastive Learning-Based Agent Modeling for Deep Reinforcement Learning},
  author  = {Ma, Wenhao and Chang, Yu-Cheng and Yang, Jie and Wang, Yu-Kai and Lin, Chin-Teng},
  journal = {arXiv:2401.00132},
  year    = {2023},
  note    = {Verified via arXiv abstract page (v3 2025).}
}


@article{roncero2026battery,
  title   = {Battery-Aware Reinforcement Learning for Aggressive Quadrotor Flight},
  author  = {Roncero, Alejandro Sanchez and Andersson, Olov and Ogren, Petter},
  journal = {arXiv:2609.37316},
  year    = {2026},
  note    = {Verified via arXiv abstract page. Tangential (circle-tracking/racing, not PE generalization); same research group as the thesis baseline.}
}


@misc{pursuitvla2027,
  title  = {An Executable Vision-Language-Action Policy for Cross-Map Zero-Shot Pursuit-Evasion Games},
  year   = {2027},
  note   = {UNVERIFIED / NOT CITABLE: appears to be an ICLR 2027 submission, OpenReview id wUdio9L3lL, bot-blocked for direct fetch, info only via search cache. Do not cite without confirming acceptance/posting.}
}
```

---

## 4. Key caveats / what remains unverified (rewritten 2026-11-04)

Items 1–5 of the 2026-10-26 list are resolved. Liu 2022 was read from the proceedings PDF. Bayes-ToMoP, the *Sensors* paper (Li, Feng & Zhang 2026) and EO-MADDPG are verified (EO-MADDPG from its abstract only). Gavin & Bronz is confirmed as ICUAS 2026, and AgilePE is "Under review". The citation chases for (A), (B) and (C) were run (section 2, with search scope recorded under each verdict). What remains:

1. **Paywalled / blocked, check manually before finalising the novelty claims:**
   - (A): **TSDP** (Applied Soft Computing 2026, DOI 10.1016/j.asoc.2026.114831; DPP task sampling, mechanism from snippets only); **Distributed PSRO** (IEEE TNNLS 36(6), 2025; "diversity estimation directly into the sampling process"); **LeMIP** (KBS 2026, DOI 10.1016/j.knosys.2026.116948).
   - (B): full texts of the MDPI / Springer items (Song & Chen 2026, EO-MADDPG, DPE-PER, MRPA, Chen et al. 2025 Applied Intelligence DOI 10.1007/s10489-025-06396-3), read from abstracts only.
   - (C): KBS 2025 (Cheng et al., DOI 10.1016/j.knosys.2025.114534) and an EAAI 2023 BPR citer, screened by title only.
2. **Forward citations of the 2026 drone-PE papers are thin.** OpenAlex / Semantic Scholar returned 0–4 citers for Roncero, AgilePE and both Gavin papers, likely due to indexing lag. Re-run near submission.
3. **Semantic Scholar rate-limited** the (B) chase completely (OpenAlex used instead). Google Scholar "cited by" was not usable in any chase.
4. **Abstract-only claims** in section 2: Kokolakis 2020, Ren 2023, Shima 2002, MRPA 2024, Geles 2605.22748, Anoruo 2607.02472, CooT, CENIE, DTGSH, P3DP, NMD-FXP, AOS-PPO. FormulaZero's ICML 2020 venue was not checked.
5. **Reference list (section 3):** entries added on 2026-11-04 give author surnames only where first names were not verified. Complete them from the sources before citing.
6. **Section 1 field labels:** see the note at the top of section 1. Most "Stated limitations" fields hold our assessment; the authors' own limitations are in the Results / conclusions bullets.
7. **Text–table contradictions inside papers** (cite the tables, not the prose): ESN-OA (Table 1 vs. "near-optimal across the tested range"), AgilePE (Table II vs. Sec. IV-A.2/3), Gavin & Bronz ("across all settings" vs. Table II), Gavin et al. 2603.16279 ("against all pursuers" vs. Table III), RMA (Fig. 3 caption vs. its table), Liu 2021 ("60 %" vs. App. Table 9).

8. **Section (E) generalization chase (2026-11-15):** caveats from the research pass, verbatim.
   - **Unverified or not independently fetched (do not cite without a manual re-check):**
     - MEMES (arXiv:2303.06137) — author list not independently confirmed.
     - ReCord / Jeon & Kim (OpenReview yf1h6EUF88) — OpenReview bot-wall blocked; naming ("ReCord" vs. "Recorded-PBT") inconsistent across secondary sources.
     - Standen, Kim, Szabo 2025 (ACM Computing Surveys 57(5):124) — DL fetch 403-blocked; metadata-only corroboration.
     - Ma et al., DAAG+CAPE (OpenReview h9isaqF956, "ICLR 2026") — bot-wall blocked in two independent research passes; acceptance status unconfirmed and should be re-checked (session date is Nov 2026, i.e. after a nominal ICLR 2026 review cycle).
     - TAO (Jing et al. 2024) and OMIS (Lu & Wu 2024) — OpenReview/forum HTML blocked; verified instead via official ICLR/NeurIPS proceedings PDF/abstract fetches, which succeeded, but author order was not independently cross-checked against the primary OpenReview page.
     - "Proximal Policy Optimization with Entropy Regularization" (IEEE 10585473) and the MDPI 2026 entropy-regularization survey — paywalled/search-snippet only, not independently fetched.
     - GE-DDPG (Springer DOI 10.1007/978-3-031-70684-4_22) — title/abstract verified, but the author list is paywalled and unconfirmed.
     - PursuitVLA (OpenReview wUdio9L3lL) — appears to be an unreviewed ICLR 2027 submission; excluded from all substantive sections, listed only as a flagged non-citable lead.
     - AlphaStar (Vinyals et al. 2019, Nature) — bibliographic metadata is well-established and cross-confirmed by multiple independent sources, but the Nature publisher page itself returned HTTP 406 on direct fetch in this session; treat the DOI as correct but unconfirmed by direct primary-text read.
     - Dohare et al. 2024 (Nature 632) — arXiv preprint independently confirmed; the Nature publisher page itself was 406-blocked (DOI corroborated via widely consistent secondary sources only).

   - **Genuine literature gaps identified (useful framing for the thesis, not weaknesses in the search):**
     - No paper frames the generalization problem exactly as this thesis's G1a ("different rules/rewards") vs. G1b ("same rules, different seed") split — this appears to be a genuinely novel framing angle worth stating explicitly as a thesis contribution.
     - No game-RL/continuous-control paper was found that directly and quantitatively measures "self-play strategy collapse" or "iteration collapse" the way the 2026 LLM self-play literature does (OCNR, R-Diverse) — this measurement rigor currently exists mainly in the LLM self-improvement literature, not in game-theoretic or robotics self-play.
     - No controlled ablation was found varying "stage length before overfitting sets in" with quantitative tables, in either game-theoretic or continuous-control self-play — a 13-vs-26-stage comparison in this thesis could be a genuine contribution closing this gap.
     - No 2023-2026 paper was found applying UED/PAIRED/PLR-style regret curricula to drone/UAV pursuit-evasion specifically (only to Predator-Prey, SMAC, or grid/BipedalWalker domains, or to general drone racing without a UED framing) — a possible original-contribution opportunity.
     - No paper was found combining policy distillation or model-soup-style weight averaging with multi-seed self-play campaigns in a competitive (not cooperative) RL context specifically — the thesis's proposed use of these techniques for cross-seed pursuer consolidation would be a novel application, not a replication.
     - No autonomous-racing paper with population-based opponent diversity (as opposed to single-agent racing against MPC/human baselines) was found in the 2023-2026 window; this sub-area is a confirmed gap in this pass and may warrant a more targeted follow-up search if racing analogs are still wanted.
     - 2025/2026-dated arXiv IDs (e.g. 2608.xxxxx, 2609.xxxxx, 2602.xxxxx) resolved to real, fetchable abstract pages consistent with the session's Nov 2026 date, but several are explicitly "under review" preprints (AgilePE, DAAG+CAPE candidate, ReCord) — re-confirm publication/acceptance status closer to thesis submission.
