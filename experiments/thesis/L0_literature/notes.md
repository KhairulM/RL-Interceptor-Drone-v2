# Literature Review & Novelty Check: Opponent Sampling, Online Opponent Adaptation, and Bayesian Opponent ID for Quadrotor Pursuit-Evasion

*Compiled 2026-10-26. All arXiv entries were fetched directly from `arxiv.org/abs/...` (title, authors, abstract, submission dates verified from page metadata). Venue claims not visible on the arXiv abstract page (e.g., journal acceptance) are flagged as unverified. Non-arXiv items were checked against publisher pages (PMLR, Springer, GitHub) where possible.*

---

## 1. Per-paper summaries

### AgilePE — Tang et al., 2026
- **Title:** *AgilePE: Autonomous UAV Pursuit-Evasion via Self-Play Reinforcement Learning*
- **Authors:** Wenhao Tang, Tianyang Chen, Zhejun Cui, Boyuan An, Jiayu Chen, Ruize Zhang, Huidong Liu, Tianyue Wu, Qingmin Liao, Fei Gao, Yu Wang, Chao Yu
- **Venue/Year:** arXiv preprint, submitted 14 Aug 2026 (no accepted-venue "Comments" field present on the abstract page — **venue status unverified**)
- **arXiv:** [2608.14135](https://arxiv.org/abs/2608.14135)
- **Summary:** Presents a full-stack UAV pursuit-evasion system: a policy mapping onboard state directly to CTBR commands (no trajectory planner), trained via competitive self-play using **Prioritized Fictitious Self-Play (PFSP)** against a "diversified opponent pool" to stabilize optimization and reduce policy oscillation. A hardware-aligned sim pipeline (actuator dynamics, comms latency, domain randomization) enables zero-shot sim-to-real transfer, reproducing dodge/flank tactics on real quadrotors.
- **Experimental setting:** Isaac-Sim-style high-fidelity simulator → zero-shot deployment on real two-agent quadrotor hardware; qualitative tactic analysis (dodging, flanking).
- **Stated limitations:** Not independently verifiable beyond the abstract (full text not fetched); abstract gives no quantitative ablation of the "diversified opponent pool" mechanism itself, so its exact sampling rule is unconfirmed from the abstract alone.
- **Relation to candidates:** Directly competing/adjacent to **(A)** — uses PFSP + a "diversified pool" but (per the abstract) not a DPP-based marginal-probability sampler; this is the single closest **system-level** prior work to your overall thesis setting (CTBR, self-play, Isaac-Sim-class pipeline, Crazyflie-class agility).

### Roncero et al., 2026 — AMSPBH
- **Title:** *Learned Controllers for Agile Quadrotors in Pursuit-Evasion Games*
- **Authors:** Alejandro Sanchez Roncero, Yixi Cai, Olov Andersson, Petter Ogren (KTH)
- **Venue/Year:** arXiv, v1 submitted 3 Jun 2025, v3 20 Jun 2026; framed as a "letter" (consistent with an RA-L-style submission, but **acceptance venue not confirmed** on the abstract page)
- **arXiv:** [2506.02849](https://arxiv.org/abs/2506.02849)
- **Summary:** Studies 1v1 quadrotor PE with PPO self-play. Proposes **AMSPBH** (Asynchronous Multi-Stage Population-Based training with **Hedge sampling**), a PSRO variant where each generation trains one agent against frozen opponents sampled by a Hedge algorithm that upweights currently-hard opponents. Shows AMSPBH retains competence against older opponents, generalizes better than latest-opponent-only training, and yields a population with distinct pursuit/evasion behaviors.
- **Experimental setting:** Isaac-Sim-class simulator; Crazyflie brushless quadrotor hardware validation, zero-shot sim-to-real, up to 4.87 m/s flights, against handcrafted and learned adversaries.
- **Stated limitations:** Abstract notes "additional best-response training gives limited improvement" past a point (diminishing returns of pure self-play); no DPP/diversity-metric opponent-sampling comparison is mentioned.
- **Relation to candidates:** **This is the paper your thesis most directly extends/competes with for (A).** Its Hedge sampler is exactly the "PFSP/Hedge" baseline your (A) claims to improve on (bounded mass vs. duplicate-inflatable Hedge weight).

### Gavin & Bronz, 2026
- **Title:** *Intercepting an Agile Target with Net-Carrying Drones using Competitive Multi-Agent Reinforcement Learning*
- **Authors:** Timothée Gavin (ENAC-LAB, LAAS-RIS), Murat Bronz (ENAC)
- **Venue/Year:** arXiv, submitted 7 Jul 2026 (Subjects: cs.RO; **no accepted-venue comment found**)
- **arXiv:** [2607.05939](https://arxiv.org/abs/2607.05939)
- **Summary:** Multi-pursuer (net-carrying) vs. single-evader interception, trained with **MAPPO + PFSP**, using low-level CTBR commands. Compares to heuristic baselines on catch rate, time-to-catch, crash rate; ablations show PFSP → more robust/adaptive policies and that low-level (CTBR) control is crucial; qualitative analysis shows emergent cooperative pursuer tactics.
- **Experimental setting:** High-fidelity simulator, CTBR control, team pursuit (not 1v1).
- **Stated limitations:** Abstract-level only; team (not 1v1) setting, so less directly comparable to your 1v1 protocol; no opponent-diversity/DPP mechanism.
- **Relation to candidates:** Confirms PFSP is the de facto baseline in this literature (relevant to **(A)**'s novelty claim vs. PFSP); CTBR + MARL + catastrophic-forgetting framing is close background context.

### ESN-OA paper — arXiv 2602.06326
- **Title:** *Online Adaptive Reinforcement Learning with Echo State Networks for Non-Stationary Dynamics*
- **Authors:** Aoi Yoshimura, Gouhei Tanaka
- **Venue/Year:** arXiv, submitted 6 Feb 2026
- **arXiv:** [2602.06326](https://arxiv.org/abs/2602.06326)
- **Summary:** Proposes an **Echo State Network (ESN) + online RLS readout** adaptation module: the ESN reservoir encodes recent observation history into a latent context; only the linear readout is updated online via RLS, requiring no backprop/pretraining/privileged info. Targets non-stationary *dynamics* (not opponents).
- **Experimental setting:** CartPole and HalfCheetah with abrupt disturbances/friction changes (non-adversarial, classic control/MuJoCo benchmarks) — **not** pursuit-evasion, **not** drones, **not** opponent modeling.
- **Stated limitations:** Evaluated only on low-dimensional, non-adversarial control benchmarks; no multi-agent or opponent-adaptation claim is made by the authors themselves.
- **Relation to candidates:** This is the literal **ESN+RLS baseline architecture for (B)** — but note the paper itself targets dynamics randomization, not opponent modeling/pursuit-evasion. Your thesis would be the first to (a) repurpose this exact ESN+RLS readout mechanism as an *opponent* model and (b) compare it against a Neural-Fly-style offline-basis+RLS approach in a competitive drone PE setting. **This is a methodological transplant, not prior art for your specific use case.**

### Perez-Nieves et al., ICML 2021
- **Title:** *Modelling Behavioural Diversity for Learning in Open-Ended Games*
- **Authors:** Nicolas Perez-Nieves, Yaodong Yang, Oliver Slumbers, David Henry Mguni, Ying Wen, Jun Wang
- **Venue/Year:** ICML 2021 (PMLR v139); arXiv v2
- **arXiv:** [2103.07927](https://arxiv.org/abs/2103.07927)
- **Summary:** Gives a geometric interpretation of behavioural diversity and introduces a **DPP-based diversity metric** over policies. Incorporates this metric as a **regularizer in the best-response objective**, yielding "diverse fictitious play" and "diverse PSRO." Proves uniqueness of the diverse best response and convergence in two-player games; shows the DPP metric provably enlarges the "gamescape" (convex polytope of strategy mixtures).
- **Experimental setting:** Tens of synthetic non-transitive normal-form/open-ended games (no robotics/drones).
- **Stated limitations:** DPP is used to *shape the training objective of the newest policy*, not to define a *sampling distribution over which frozen opponents to train against* — it never proposes sampling opponents ∝ DPP marginals K_ii.
- **Relation to candidates:** **Closest conceptual ancestor of (A)**, but uses DPP as a diversity **regularizer for best-response optimization**, not as an **opponent-sampling/curriculum distribution**. This is the key technical gap your (A) would need to argue it closes.

### Liu et al., NeurIPS 2021
- **Title:** *Towards Unifying Behavioral and Response Diversity for Open-ended Learning in Zero-sum Games* (arXiv title: *Unifying Behavioral and Response Diversity for Open-ended Learning in Zero-sum Games*)
- **Authors:** Xiangyu Liu, Hangtian Jia, Ying Wen, Yaodong Yang, Yujing Hu, Yingfeng Chen, Changjie Fan, Zhipeng Hu
- **Venue/Year:** NeurIPS 2021
- **arXiv:** [2106.04958](https://arxiv.org/abs/2106.04958)
- **Summary:** Unifies "Behavioral Diversity" (occupancy-measure discrepancy) and "Response Diversity" (reward-dynamics discrepancy across opponents) into one framework; shows most prior diversity metrics are BD-only or RD-only. Designs a diversity-promoting objective and "population effectivity" criterion for best-response solving.
- **Experimental setting:** Matrix games, non-transitive mixture model, Google Research Football (no drones/robotics).
- **Stated limitations:** Still a training-time diversity **objective**, not an explicit opponent-sampling rule for curriculum/self-play.
- **Relation to candidates:** Background for **(A)**'s behaviour-descriptor framing (BD/RD distinction is directly relevant to how you'd define the RBF kernel S over "response vector vs. probe pursuers + trajectory statistics").

### Liu et al., NeurIPS 2022
- **Title:** *A Unified Diversity Measure for Multiagent Reinforcement Learning*
- **Authors:** Zongkai Liu, Chao Yu, Yaodong Yang, Peng Sun, Zifan Wu, Yuan Li
- **Venue/Year:** NeurIPS 2022 (vol. 35, pp. 10339–10352)
- **arXiv:** **No arXiv preprint found.** Verified only via NeurIPS proceedings / OpenReview / mlanthology.org — **cite via NeurIPS proceedings, not arXiv.**
- **Summary (per proceedings abstract, not independently fetched from a primary PDF due to no arXiv mirror):** Proposes "Unified Diversity Measure" (UDM) generalizing prior BD/RD-type metrics; derives UDM-Fictitious Play and UDM-PSRO with convergence guarantees; shown to outperform PSRO baselines on non-transitive games.
- **Experimental setting:** Per secondary sources — synthetic non-transitive games (unverified against primary text).
- **Stated limitations:** Not independently verified (no primary-source fetch possible without arXiv mirror or OpenReview access, which was blocked by a bot-check wall during this session) — **flag as weakly verified, title/authors/venue confirmed via proceedings record only.**
- **Relation to candidates:** Same family as Perez-Nieves/Yao — diversity as a training objective, not as an opponent-sampling distribution; relevant background for **(A)**.

### Yao et al., NeurIPS 2023 — PSD-PSRO
- **Title:** *Policy Space Diversity for Non-Transitive Games*
- **Authors:** Jian Yao, Weiming Liu, Haobo Fu, Yaodong Yang, Stephen McAleer, Qiang Fu, Wei Yang
- **Venue/Year:** arXiv submitted 29 Jun 2023, revised 8 Nov 2023 (timing consistent with NeurIPS 2023, **venue acceptance not shown on the arXiv page itself**)
- **arXiv:** [2306.16884](https://arxiv.org/abs/2306.16884)
- **Summary:** Proves that existing diversity metrics' improvement does **not** guarantee a better Nash-equilibrium approximation, and proposes a new diversity metric whose improvement does guarantee this, optimized from state-action samples alone; yields **PSD-PSRO**, shown empirically to be less exploitable than SOTA PSRO variants.
- **Experimental setting:** Standard non-transitive game suites (no robotics).
- **Stated limitations:** Diversity used again as a best-response-training regularizer, not an opponent-sampling rule.
- **Relation to candidates:** Same family — relevant **theoretical backdrop** for why naive diversity metrics can be gamed (useful to motivate why K_ii-bounded sampling in (A) is theoretically preferable to ad hoc diversity bonuses), but not itself a sampling method.

### Balduzzi et al., ICML 2019
- **Title:** *Open-ended Learning in Symmetric Zero-sum Games*
- **Authors:** David Balduzzi, Marta Garnelo, Yoram Bachrach, Wojciech M. Czarnecki, Julien Perolat, Max Jaderberg, Thore Graepel
- **Venue/Year:** ICML 2019
- **arXiv:** [1901.08106](https://arxiv.org/abs/1901.08106)
- **Summary:** Introduces a geometric framework for open-ended learning in nontransitive zero-sum games, and **PSRO_rN ("rectified Nash response")**, which uses "game-theoretic niching" to build diverse, effective populations — a precursor to all the diversity-PSRO lineage above.
- **Experimental setting:** Two nontransitive resource-allocation games (synthetic, no robotics).
- **Stated limitations:** No robotics/continuous-control application; niching via rectified-Nash weighting, not DPP.
- **Relation to candidates:** Foundational background for **(A)** — the PSRO lineage your opponent pool inherits from.

### Vinyals et al., Nature 2019 — AlphaStar
- **Title:** *Grandmaster level in StarCraft II using multi-agent reinforcement learning*
- **Authors:** Oriol Vinyals, Igor Babuschkin, Wojciech M. Czarnecki, Michaël Mathieu, Andrew Dudzik, et al. (DeepMind)
- **Venue/Year:** *Nature* 575, 350–354 (2019); DOI 10.1038/s41586-019-1724-z
- **arXiv:** **None.** Confirmed via web search — this is a Nature-only publication with no arXiv preprint (the arXiv ID 1902.01724 that surfaces in searches, "AlphaStar: An Evolutionary Computation Perspective" by Arulkumaran et al., is a **different, third-party analysis paper**, not the primary source — do not conflate the two).
- **Summary:** Introduces the AlphaStar **League**: a population of main agents, main exploiters, and league exploiters trained via multi-agent RL with **Prioritized Fictitious Self-Play (PFSP)**, where opponent-sampling probability is weighted toward opponents the current agent is losing to (a softmax/Hedge-like weighting over win-rate-derived "hardness").
- **Experimental setting:** Full StarCraft II, reaching Grandmaster level, validated online vs. human players.
- **Stated limitations:** PFSP's hardness weighting can over-sample near-duplicate historical checkpoints of the same strategy (this is precisely the failure mode **(A)** claims to fix via bounded DPP marginal inclusion probabilities).
- **Relation to candidates:** **The canonical PFSP reference** your (A) is positioned against; also the origin of "exploiters" as deliberately diversity-inducing opponents (conceptually adjacent to your hardness/behaviour-descriptor kernel).

### O'Connell et al., Science Robotics 2022 — Neural-Fly
- **Title:** *Neural-Fly Enables Rapid Learning for Agile Flight in Strong Winds*
- **Authors:** Michael O'Connell, Guanya Shi, Xichen Shi, Kamyar Azizzadenesheli, Anima Anandkumar, Yisong Yue, Soon-Jo Chung (Caltech)
- **Venue/Year:** *Science Robotics* 7(66), eabm6597 (2022); DOI 10.1126/scirobotics.abm6597
- **arXiv:** [2205.06908](https://arxiv.org/abs/2205.06908)
- **Summary:** Learns a shared aerodynamic feature basis φ offline across many wind conditions via **Domain Adversarially Invariant Meta-Learning (DAIML)** (multi-task heads per wind condition + domain-adversarial loss to make φ condition-invariant), then at test time adapts only a linear coefficient vector via a **composite adaptation law** (not literally RLS, but a Lyapunov-based adaptive-control update) using only 12 minutes of flight data.
- **Experimental setting:** Real quadrotor flight in Caltech Real Weather Wind Tunnel, winds up to 43.6 km/h; cross-platform transfer.
- **Stated limitations:** Domain is wind/aerodynamics, not an adversarial opponent; adaptation law is a stability-guaranteed composite update, not RLS per se (a detail your (B) description should be precise about — RLS is the natural discrete-time analogue, but Neural-Fly's own update is a continuous-time composite adaptive law).
- **Relation to candidates:** **This is the literal namesake/method template for (B).** Your novelty rests entirely on: (i) swapping "wind condition" for "opponent evader identity" as the multi-task label, and (ii) swapping the composite adaptive law for RLS, and (iii) using the predicted feature as an **observation input to a PPO policy** rather than as a direct control correction term.

### Kumar et al., RSS 2021 — RMA
- **Title:** *RMA: Rapid Motor Adaptation for Legged Robots*
- **Authors:** Ashish Kumar, Zipeng Fu, Deepak Pathak, Jitendra Malik
- **Venue/Year:** RSS 2021 (DOI 10.15607/RSS.2021.XVII.011)
- **arXiv:** [2107.04034](https://arxiv.org/abs/2107.04034)
- **Summary:** Base policy + adaptation module; the adaptation module infers a latent "extrinsics" vector online from a history of states/actions (via supervised distillation from a privileged encoder), enabling sub-second adaptation to terrain/payload/wear changes without backprop at test time.
- **Experimental setting:** Unitree A1 quadruped, diverse real terrains (sand, mud, stairs, pebbles), trained in sim, zero-shot transfer.
- **Stated limitations:** Domain is terrain/embodiment adaptation (environment, not an adversarial opponent); the online module is a distilled feed-forward encoder, not RLS.
- **Relation to candidates:** Background precedent for **(B)**'s "privileged-information distillation → fast online adaptation module" paradigm, parallel to but methodologically distinct from Neural-Fly's basis+linear-adaptation approach.

### Fu et al., ICML 2022 — GSCU
- **Title:** *Greedy when Sure and Conservative when Uncertain about the Opponents*
- **Authors:** Haobo Fu, Ye Tian, Hongxiang Yu, Weiming Liu, Shuang Wu, Jiechao Xiong, Ying Wen, Kai Li, Junliang Xing, Qiang Fu, Wei Yang
- **Venue/Year:** ICML 2022 (PMLR v162, pp. 6829–6848)
- **arXiv:** **No arXiv version found** — verified via PMLR proceedings page (`proceedings.mlr.press/v162/fu22b.html`) and the official code repo `github.com/YeTianJHU/GSCU`.
- **Summary:** Learns opponent-policy embeddings offline, trains a single embedding-conditioned best response (not a discrete library of responses), computes an **online Bayesian posterior over the current opponent's embedding** (avoiding discrete type classification), and selects online between a greedy (embedding-conditioned) policy and a fixed conservative policy via an adversarial-bandit algorithm with regret guarantees.
- **Experimental setting (verified via GitHub repo structure):** **Kuhn Poker** (OpenSpiel) and **Predator-Prey** (Multi-Agent Particle Environment) — i.e., GSCU's own benchmark **already includes a simplified pursuit-evasion-like domain** with "online Bayesian adaptation for seen/unseen/mix/adaptive opponent settings."
- **Stated limitations:** Low-dimensional 2D particle dynamics (not agile/CTBR quadrotor flight); continuous embedding space rather than a small discrete library of named prototype/medoid evaders; greedy/conservative switching is a two-arm bandit, not a continuous confidence-weighted blend.
- **Relation to candidates:** **The single closest prior work to (C).** GSCU already implements (a) Bayesian online identification of opponent identity/type and (b) a conservative-fallback-when-uncertain mechanism, in a predator-prey (pursuit-evasion) toy domain. Your (C)'s novelty would have to rest on the **discrete medoid/prototype library + behavior-cluster construction + agile quadrotor CTBR domain**, not on the Bayesian-ID-with-fallback idea itself, which is not new.

### Rosman, Hawasly & Ramamoorthy, 2016 — Bayesian Policy Reuse
- **Title:** *Bayesian Policy Reuse*
- **Authors:** Benjamin Rosman, Majd Hawasly, Subramanian Ramamoorthy
- **Venue/Year:** *Machine Learning* 104(1), 99–127 (2016); DOI 10.1007/s10994-016-5547-y
- **arXiv:** [1505.00284](https://arxiv.org/abs/1505.00284) (preprint, 2015)
- **Summary:** Formalizes policy reuse: given a **library of policies pre-learned offline**, choose online which library policy to reuse for a novel task instance, using observed performance "signals" and a Bayesian-choice formulation with exploration/exploitation-balancing selection functions (an inspiration for BPR+/Bayes-ToMoP/GSCU-style methods).
- **Experimental setting:** Simulated interactive domains (grid-world-like task families), not robotics-specific or adversarial pursuit-evasion per se.
- **Stated limitations:** Exact Bayesian posterior often intractable; approximated via Bayesian-optimization-style selection functions; no opponent-specific or adversarial framing (task instances, not adversaries).
- **Relation to candidates:** **Foundational ancestor of (C)** — BayesID's "small set of prototype evaders" + Bayesian belief update over which prototype is active is structurally a direct descendant of BPR's "library + Bayesian choice" framework, just applied to an adversarial opponent rather than a task family.

### Al-Shedivat et al., ICLR 2018
- **Title:** *Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments*
- **Authors:** Maruan Al-Shedivat, Trapit Bansal, Yuri Burda, Ilya Sutskever, Igor Mordatch, Pieter Abbeel
- **Venue/Year:** ICLR 2018
- **arXiv:** [1710.03641](https://arxiv.org/abs/1710.03641)
- **Summary:** Casts continuous adaptation to nonstationary/adversarial environments as a learning-to-learn (gradient-based meta-learning, MAML-style) problem; introduces **RoboSumo**, a competitive multi-agent benchmark, and "iterated adaptation games"; shows meta-learners adapt faster than reactive baselines and are the fittest in a competing population.
- **Experimental setting:** RoboSumo (simulated 3D wrestling-like competitive agents), not drones/quadrotors.
- **Stated limitations:** Gradient-based meta-adaptation requires multiple interaction episodes per update (slower than a single-pass RLS update); no CTBR/aerial dynamics.
- **Relation to candidates:** Conceptual ancestor of **(B)**'s "online adaptation to opponent" framing, but via MAML-style gradient meta-learning rather than linear-basis+RLS; useful contrast case (gradient-based vs. basis+RLS-based online opponent adaptation) for your related-work discussion.

### Friedman & Dieng, 2023 — The Vendi Score
- **Title:** *The Vendi Score: A Diversity Evaluation Metric for Machine Learning*
- **Authors:** Dan Friedman, Adji Bousso Dieng
- **Venue/Year:** arXiv 2022/2023 (TMLR-track-style; **exact venue not independently confirmed in this session** beyond arXiv + OpenReview listing)
- **arXiv:** [2210.02410](https://arxiv.org/abs/2210.02410)
- **Summary:** Defines the Vendi Score as the exponential of the Shannon entropy of the eigenvalues of a user-defined similarity matrix (closely related to DPP machinery — it is essentially an effective-rank/diversity functional of the same kernel matrix a DPP would use). Reference-free, domain-agnostic diversity metric.
- **Experimental setting:** Molecular generative modeling, image/text generative models, GAN mode-collapse diagnosis (no RL/self-play).
- **Stated limitations:** A diversity *metric for evaluation*, not a *sampling mechanism* — doesn't itself propose using eigenvalues/marginals for sampling.
- **Relation to candidates:** Useful **evaluation tool** for (A) — you could report the Vendi Score of your opponent pool over training as a diagnostic, distinct from your DPP-based *sampling* mechanism (metric vs. mechanism is the key distinction to keep straight in the thesis).

### Kulesza & Taskar, 2012 — DPP survey
- **Title:** *Determinantal Point Processes for Machine Learning*
- **Authors:** Alex Kulesza, Ben Taskar
- **Venue/Year:** *Foundations and Trends in Machine Learning*, Vol. 5, No. 2–3, pp. 123–286 (2012)
- **arXiv:** [1207.6083](https://arxiv.org/abs/1207.6083)
- **Summary:** The standard tutorial/reference for DPPs: L-ensemble construction (L = quality·similarity·quality, i.e., your `a·diag(q) S diag(q)` is a direct instantiation of their quality/diversity decomposition), marginal kernel K, marginal inclusion probabilities K_ii, sampling algorithms, k-DPPs, and applications (diverse search results, document summarization, pose diversity).
- **Experimental setting:** Tutorial/survey — no new experiments of its own beyond illustrative examples.
- **Stated limitations:** N/A (survey).
- **Relation to candidates:** **This is the exact mathematical toolkit (A) builds on** — your L = a·diag(q)·S·diag(q) construction and the "marginal inclusion probability K_ii" sampling rule is precisely the quality-diversity decomposition and marginal-kernel machinery this monograph formalizes. Cite as the canonical DPP reference, not as prior art for opponent sampling specifically.

---

## 2. Novelty check

### (A) DPP-based opponent sampling (sample ∝ marginal K_ii, bounded against duplicates)

**Closest prior work found:**
- Perez-Nieves et al. 2021 (arXiv:2103.07927) and the Liu/Yao DPP-diversity lineage (2106.04958, NeurIPS'22 UDM, 2306.16884) all use **DPP-derived or DPP-like diversity metrics as a training-time regularizer** added to the best-response objective of the *newest* policy — they do **not** use DPP marginals to define a **sampling distribution over which frozen opponents from the existing pool to train against**.
- Balduzzi et al. 2019 (PSRO_rN, 1901.08106) uses "game-theoretic niching" (rectified Nash weighting) for a related but distinct notion of niche-based opponent weighting — not DPP-marginal sampling.
- Vinyals et al. 2019 (AlphaStar League) and the PFSP baselines in Roncero et al. 2026 / Gavin & Bronz 2026 / AgilePE 2026 all weight opponents by a **Hedge/softmax-over-hardness** rule, which is exactly the "duplicates can inflate sampling mass" failure mode you describe.
- No paper found (arXiv, Google-Scholar-style web search, GitHub code search) that explicitly defines opponent-sampling probability as the **marginal inclusion probability K_ii of an L-ensemble DPP** with L factored as quality(hardness)×similarity(behaviour kernel), in **any** domain (games, MARL, robotics) — this specific sampling rule (as opposed to DPP-as-regularizer) was not located.
- **Robotics/drone-specific:** none of the three 2026 drone-PE papers (AgilePE, AMSPBH, Gavin & Bronz) use DPP at all; all use Hedge- or PFSP-style softmax weighting.

**Verdict:**
- **(i) In general:** **Not fully novel as a "DPP exists for diversity in self-play" idea** — DPP-based diversity metrics are well established in the PSRO literature (2017–2023). **Likely novel as the specific sampling mechanism** (opponent-selection probability = marginal K_ii, with the explicit bounded-duplicate-mass property as the selling point against PFSP/Hedge) — searches did not surface a paper that samples opponents this way rather than using DPP as a best-response regularizer. Recommend a thorough Semantic Scholar citation-chase of Perez-Nieves/Kulesza-Taskar citing papers before finalizing this claim, since this is a narrow, easily-missed gap.
- **(ii) For agile quadrotor pursuit-evasion:** **Novel.** No drone/UAV PE paper found using any DPP-based mechanism (only Hedge/PFSP). This is your strongest, most defensible novelty claim among the three.

### (B) Online opponent adaptation via offline-learned basis + online RLS readout (Neural-Fly-style), vs. ESN+RLS and GRU

**Closest prior work found:**
- O'Connell et al. 2022 (Neural-Fly, arXiv:2205.06908) — the exact method template, but for **wind/aerodynamics**, not an **opponent**; adaptation law is a composite adaptive-control update, not literally RLS.
- Kumar et al. 2021 (RMA) — same "offline-privileged, online-fast-adapt" paradigm, different mechanism (distilled encoder, not basis+linear adaptation).
- Al-Shedivat et al. 2018 — gradient-based (MAML) online adaptation to a competitive opponent in RoboSumo; the clearest **prior work on adapting to an opponent online**, but mechanistically a world apart from basis+RLS (and far slower per-step).
- Yoshimura & Tanaka 2026 (ESN-OA, arXiv:2602.06326) — literally the ESN+online-RLS-readout architecture you propose as a baseline, but applied to **non-stationary dynamics (CartPole/HalfCheetah)**, not opponents.
- Fu et al. 2022 (GSCU) — learns an opponent **embedding** offline and updates a posterior over it online, which is conceptually "learn offline, adapt online to an opponent" but via a **Bayesian posterior over a learned embedding**, not a linear basis + RLS.
- No paper was found that (a) learns a Neural-Fly-style multi-task feature basis **per evader** (with or without a DAIML domain-adversarial loss) specifically to predict an **opponent's future relative position**, and (b) appends that prediction to a pursuer's RL observation, in **any** pursuit-evasion/interception/drone-racing context. Searches for "online opponent modeling RLS drone racing/pursuit evasion" returned only general surveys and tangential MARL opponent-modeling papers (e.g., EO-MADDPG) that use learned opponent-action predictors inside MADDPG-style critics, not a Neural-Fly-style basis+RLS predictor feeding a PPO observation.

**Verdict:**
- **(i) In general:** The **mechanism** (shared offline basis + online linear/RLS adaptation) is well established (Neural-Fly, RMA) but **not for opponent modeling** — it has only been applied to physical/environmental nonstationarity (wind, terrain, general dynamics). Using it to predict an **adversary's** behavior appears to be a genuine, if incremental, repurposing.
- **(ii) For agile quadrotor pursuit-evasion:** **Novel**, and arguably your most "engineering-novel, low-risk" contribution — it is a well-motivated transplant of a proven aerial-robotics adaptation mechanism into the opponent-modeling role, with no prior PE/drone-racing paper found doing this. The ESN+RLS and GRU baselines you propose are also appropriately chosen contemporary alternatives (ESN-OA paper gives you a strong, very recent, directly citable baseline architecture, just needs re-targeting from dynamics to opponents).

### (C) Bayesian opponent identification over prototype/medoid evaders + conservative fallback

**Closest prior work found:**
- Fu et al. 2022 (GSCU) — **the closest prior work overall.** Already does online Bayesian-style posterior inference over opponent identity (via a continuous learned embedding, not discrete medoids) **and** an explicit conservative-fallback-when-uncertain mechanism (greedy-vs-conservative adversarial-bandit switch), evaluated in part on a **Predator-Prey** (pursuit-evasion-flavored) MAPE domain.
- Rosman, Hawasly & Ramamoorthy 2016 (Bayesian Policy Reuse) — the foundational "library of policies + Bayesian online selection" framework that both GSCU and your BayesID descend from, but for task instances, not adversaries.
- Yang et al. 2019, Bayes-ToMoP (arXiv:1809.04240, IJCAI 2019 — found via search, not independently fetched/verified in this session) — extends BPR with recursive theory-of-mind reasoning about possibly-Bayesian opponents; same "library + Bayesian belief" lineage, evaluated on simulated soccer/predator-prey-type games (per search summary; **not independently verified against primary source**).
- No paper was found that explicitly uses **medoids of behavior clusters** as the prototype library (as opposed to a continuous embedding or a hand-specified discrete type set), nor one applying GSCU/Bayes-ToMoP/BPR-style Bayesian opponent ID to **agile quadrotor pursuit-evasion** specifically.

**Verdict:**
- **(i) In general:** **Not novel as a concept.** Bayesian opponent/type identification with a conservative-policy fallback under uncertainty is a well-established sub-area (BPR 2016 → Bayes-ToMoP 2019 → GSCU 2022). Your (C) would need to lean on the **medoid/cluster-based prototype construction** and the **continuous confidence-weighted blend to a generalist** (vs. GSCU's discrete greedy/conservative bandit switch) as the specific point of departure, and these are comparatively modest differentiators.
- **(ii) For agile quadrotor pursuit-evasion:** **Likely novel** in the specific domain — no drone/UAV paper was found applying GSCU/BPR/Bayes-ToMoP-style Bayesian opponent identification to CTBR-level agile flight. This is the **weakest of your three novelty claims in general ML/game-theory terms**, but still domain-novel for your thesis's target application.

### Novelty summary table

| Candidate | Core mechanism | Closest general prior work | Novel in general? | Novel for agile quadrotor PE? |
|---|---|---|---|---|
| **(A) QD-DPP opponent sampling** | Sample opponents ∝ DPP marginal K_ii, L = quality·RBF-kernel·quality | Perez-Nieves et al. 2021 (`2103.07927`), Liu 2021/2022, Yao 2023 (`2306.16884`) — DPP/diversity used as **best-response training regularizer**, not opponent-sampling distribution; PFSP/Hedge (Vinyals 2019, Roncero 2026, AgilePE 2026, Gavin & Bronz 2026) | **Partially** — DPP-for-diversity is old; DPP-marginal-as-*sampling-rule* with bounded-duplicate-mass appears unaddressed, but needs a deeper citation-chase to be certain | **Yes** — no drone/UAV PE paper uses any DPP mechanism |
| **(B) NF-OA (basis + online RLS opponent prediction)** | Offline multi-task/DAIML basis φ per evader + online RLS on last layer θ, prediction fed to pursuer's observation | O'Connell et al. 2022 Neural-Fly (`2205.06908`) — identical mechanism, applied to **wind**, not opponents; Kumar 2021 RMA — same paradigm, different mechanism; Yoshimura & Tanaka 2026 ESN-OA (`2602.06326`) — your proposed ESN+RLS baseline, applied to **dynamics**, not opponents | **Yes** (as opponent-modeling application) | **Yes** — no PE/drone-racing paper found using this mechanism for opponent prediction |
| **(C) BayesID (Bayesian ID over prototype/medoid evaders + conservative fallback)** | Bayesian posterior over small discrete medoid library; conservative fallback to generalist when uncertain | Fu et al. 2022 GSCU (PMLR v162, no arXiv) — online Bayesian-style ID + conservative fallback, incl. a predator-prey domain; Rosman et al. 2016 BPR (`1505.00284`); Bayes-ToMoP (IJCAI 2019, `1809.04240`, unverified here) | **No** — concept well-established 2016–2022 | **Likely yes** — no drone/UAV paper found, but weakest differentiation of the three |

### Additional 2025–2026 papers to consider citing (adaptive/opponent-modelling quadrotor pursuit-evasion/interception)

- **AgilePE**, Tang et al. 2026, arXiv:2608.14135 — directly in-scope (self-play + PFSP + diversified pool for agile UAV PE), **must-cite**.
- **Roncero et al. 2026 (AMSPBH)**, arXiv:2506.02849 — your most direct baseline/competitor (Hedge-sampled PSRO for 1v1 quadrotor PE), **must-cite**.
- **Gavin & Bronz 2026**, arXiv:2607.05939 — MAPPO+PFSP, CTBR, team interception; good for CTBR/PFSP background, not opponent-modeling per se.
- **"Online Planning for Multi-UAV Pursuit-Evasion in Unknown Environments Using Deep RL,"** arXiv:2409.15866 (v4, 2025) — verified title/abstract via arXiv; multi-UAV, MARL, real hardware deployment; relevant as recent UAV PE RL context but not single-opponent Bayesian/adaptive modeling — **optional, broader-context citation**.
- **"Autonomous Decision Making for UAV Cooperative Pursuit-Evasion Game with Reinforcement Learning,"** arXiv:2411.02983 (2024) — verified title/abstract; multi-role cooperative UAV PE decision-making — **optional, broader-context citation**, not opponent-modeling-specific.
- A paper titled **"Safe Cooperative Decision-Making for Multi-UAV Pursuit–Evasion"** (MDPI *Sensors*, discovered via search, belief/risk-gated hierarchical policy with opponent intent inference) was identified by search but **could not be fetched/verified in this session** (403 error from publisher) — flag as **unverified**, worth checking manually before citing.
- **EO-MADDPG** (MDPI, "Expert Knowledge and Opponent Modeling Multi-UAV DDPG") — discovered via search only, **not independently verified**; potentially relevant as an explicit-opponent-modeling MARL baseline for multi-UAV PE — check manually.

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
  note         = {Verified via arXiv abstract page; accepted-venue status unverified}
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

@misc{gavin2026intercepting,
  title        = {Intercepting an Agile Target with Net-Carrying Drones using Competitive Multi-Agent Reinforcement Learning},
  author       = {Gavin, Timoth\'ee and Bronz, Murat},
  year         = {2026},
  eprint       = {2607.05939},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2607.05939}
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
  note         = {Optional broader-context citation; multi-UAV, not opponent-ID-specific}
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
  author       = {Liu, Xiangyu and Jia, Hangtian and Wen, Ying and Yang, Yaodong and Hu, Yujing and Chen, Yingfeng and Fan, Changjie and Hu, Zhipeng},
  booktitle    = {NeurIPS},
  year         = {2021},
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
  url          = {https://proceedings.neurips.cc/paper_files/paper/2022/file/435cce71b4007699041dfffa4f034079-Paper-Conference.pdf},
  note         = {No arXiv preprint found; cite via NeurIPS proceedings. Weakly verified (abstract via secondary sources only).}
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
  note         = {Exact venue not independently confirmed beyond arXiv/OpenReview}
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
  title        = {Bayes-ToMoP: A Fast Detection and Best Response Algorithm Towards Sophisticated Opponents},
  author       = {Yang, Tianpei and Meng, Zhaopeng and Hao, Jianye and Zhang, Chongjie and Zheng, Yan},
  booktitle    = {IJCAI},
  year         = {2019},
  eprint       = {1809.04240},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/1809.04240},
  note         = {Found via web search only; NOT independently fetched/verified in this session — verify primary text before citing}
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

## 4. Key caveats / what remains unverified

1. **Liu et al. NeurIPS 2022 ("A Unified Diversity Measure")** — no arXiv mirror found; summary is based on the NeurIPS/OpenReview/mlanthology record, not a directly fetched primary PDF (an OpenReview bot-check blocked direct access in this session).
2. **Bayes-ToMoP (Yang et al., IJCAI 2019, arXiv:1809.04240)** — surfaced only via web search summaries, not independently fetched from the arXiv abstract page; treat the experimental-domain claims (simulated soccer/predator-prey) as **unverified** until you pull the primary source.
3. **MDPI "Safe Cooperative Decision-Making for Multi-UAV Pursuit–Evasion"** and **"EO-MADDPG"** — both surfaced via search as possibly relevant 2025-era opponent-modeling UAV papers, but the MDPI site returned a 403 on direct fetch in this session; **manually verify before citing**.
4. **Accepted-venue status** of AgilePE (2608.14135) and Gavin & Bronz (2607.05939) could not be confirmed beyond "arXiv preprint, Robotics (cs.RO)" — no "Comments: accepted at..." field was present on the abstract pages at fetch time; treat as preprints unless you independently confirm publication.
5. **The (A) novelty claim is the most time-sensitive to verify further** — I was not able to do a full backward/forward citation-chase (e.g., via Semantic Scholar's citation graph) of Perez-Nieves et al. and Kulesza & Taskar within this session; before finalizing your novelty claim for (A), I'd recommend one more targeted pass specifically searching citing papers of Perez-Nieves (2103.07927) for any that repurpose the DPP marginal kernel as a *sampling* (not *regularization*) mechanism.