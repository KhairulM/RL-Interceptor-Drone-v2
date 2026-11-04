# Literature Review & Novelty Check: Opponent Sampling, Online Opponent Adaptation, and Bayesian Opponent ID for Quadrotor Pursuit-Evasion

*Compiled 2026-10-26. All arXiv entries were fetched directly from `arxiv.org/abs/...` (title, authors, abstract, submission dates verified from page metadata). Venue claims not visible on the arXiv abstract page (e.g., journal acceptance) are flagged as unverified. Non-arXiv items were checked against publisher pages (PMLR, Springer, GitHub) where possible.*

*Updated 2026-11-04: every section-1 entry checked against the full text (results bullets added, contradicted fields corrected), one paper added (Gavin, Lacroix & Bronz 2025), citation chases run for (A), (B) and (C) with revised verdicts in section 2, reference list and caveats updated. Section 4 lists what is still unverified.*

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
