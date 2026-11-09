# Quality-preserving diversity in self-play: a literature check for the pursuer/evader pipeline

Date: 2026-11-08. Sources: arXiv abstract and HTML pages, proceedings pages, and ar5iv full text where noted. Numbers come only from the source text I read. **[unverified]** marks a claim I could not confirm.

## Bottom line

1. **The literature splits into two families, and only one of them targets what we measured.**
   - *Diversity-regularised best responses* (DPP-PSRO, BD&RD, UDM, PSD-PSRO) were built for **non-transitive** games. They aim to lower the exploitability of the *population mixture*.
   - *League and exploiter structures* (AlphaStar main exploiters, FXP counter-population, Minimax Exploiter) target the **exploitability of the main agent**. That is our problem: the pursuer is exploitable at 0.96–0.98 by a lure into a floor crash.
   - Our game is mostly transitive, and the pool is redundant. The second family fits better.
2. **Quality-constrained diversity methods turn our "pilot guard" into a training-time constraint.** SMERL, DOMiNO and DGPO add the diversity reward only while the agent's return stays near-optimal, using an indicator gate or a Lagrange multiplier. This targets F6b's failure mode (b): diversity that changes *how* the evader wins.
   - Caveat: all of their evidence is single-agent or cooperative (locomotion, MPE, SMAC, Atari). None tests whether the *opponent* trained against the diverse population becomes stronger.
3. **Hedge is already the principled meta-solver.** Anytime PSRO uses multiplicative-weights updates (MWU, i.e. Hedge) over the restricted population, because that keeps exploitability from rising. Another sampler change alone (DPP, rectified Nash, PFSP-style hardness) is unlikely to beat Hedge. The gains have to come from *what goes into the pool*, not from how it is sampled.

---

## A. Quality-constrained diversity (candidates for the evader side / F6b)

### A1. SMERL: Structured MaxEnt RL
- **Citation:** S. Kumar, A. Kumar, S. Levine, C. Finn. "One Solution is Not All You Need: Few-Shot Extrapolation via Structured MaxEnt RL." NeurIPS 2020. arXiv:2010.14484. Verified.
- **Mechanism:** Trains a latent-conditioned policy π(a|s,z) with a DIAYN discriminator reward r̃ = log q(z|s) − log p(z). The diversity term is **gated**: r_SMERL = r + α·1[R(π_z) ≥ R* − ε]·r̃. R* comes from a separately trained SAC agent.
- **How quality is kept:** Hard gate. Diversity is rewarded only while the latent policy's return is within ε of optimal. Defaults: α = 10, ε = 0.1·R_SAC.
- **Evidence:** Single-agent MuJoCo (HalfCheetah-Goal, Walker, Hopper) under obstacle, force and motor-failure perturbations.
  - SMERL is reported "as robust or more robust" than SAC across environments. The main comparison (Fig. 4) is only plotted; there are no headline numbers.
  - The paper itself states sensitivity to ε and α.
- **Fit to our failure modes:**
  - (b) The gate is the direct fix: the term switches off as soon as the evader's win rate drops below the reference.
  - (a) The discriminator reward log q(z|s) is not divided by std, but it can also saturate once the discriminator is confident.
  - Cost: small (reward gate plus a reference win rate). Our F6b-off control run supplies R*.
  - It does not target pursuer exploitability directly.
- **Applied to pursuit-evasion or drones?** Not found.

### A2. DOMiNO: Diversity Optimization Maintaining Near Optimality
- **Citation:** T. Zahavy, Y. Schroecker, F. Behbahani, K. Baumli, S. Flennerhag, S. Hou, S. Singh. "Discovering Policies with DOMiNO: Diversity Optimization Maintaining Near Optimality." ICLR 2023. arXiv:2205.13521. Verified.
- **Mechanism:** Constrained MDP: maximise diversity of n policies subject to v_e^i ≥ α·v_e* for each i. The main experiments use α = 0.9; v* is the value of policy 1.
  - Diversity is measured between **successor features** (expected discounted state features), not per-state action distributions.
  - Variants: a repulsive/Hausdorff variant and a "Van der Waals" variant. The VDW reward goes to zero at a target distance ℓ₀, so policies stop being pushed apart once they are far enough.
- **How quality is kept:** A per-policy Lagrange multiplier μ_i, passed through a sigmoid, so the reward is a **convex combination** σ(μ_i)·r_e + (1−σ(μ_i))·r_d. Separate value heads handle r_e and r_d. μ falls when the constraint is satisfied and rises otherwise.
- **Evidence:** DM Control walker.stand/walk and dog. Lowering α buys more diversity.
  - Under large perturbations, K-shot return was about 1.5–2.0× the single-policy baseline (20 seeds).
  - The authors call it an empirical investigation; convergence of the three-player game is not proven.
- **Fit to our failure modes:**
  - (a) Distance on bounded feature expectations avoids the std-division saturation. VDW gives a smooth, bounded push, close to F6b's 1−exp(−d/τ).
  - (b) The Lagrangian restores extrinsic weight automatically when the evader starts winning less or differently. It is the adaptive version of our fixed small weight.
  - Cost: small-to-moderate (second value head, multiplier update, choice of features ϕ).
  - It does not target the pursuer directly.
- **Applied to pursuit-evasion or drones?** Not found.

### A3. DGPO: Diversity-Guided Policy Optimization
- **Citation:** W. Chen, S. Huang, Y. Chiang, T. Pearce, W.-W. Tu, T. Chen, J. Zhu. "DGPO: Discovering Multiple Strategies with Diversity-Guided Policy Optimization." AAAI 2024, 38(10):11390–11398. DOI 10.1609/aaai.v38i10.29019. arXiv:2207.05631. Verified.
- **Mechanism:** One latent-conditioned PPO network. Diversity is the *minimum* pairwise state-distribution KL, lower-bounded by a discriminator.
- **How quality is kept:** Two indicator masks:
  - mask_r = 1[J_div ≥ δ]: the extrinsic reward counts only once the policies are diverse enough.
  - mask_d = 1[J ≥ R_target]: the diversity reward stays active only while return is at least R_target = 0.9 × the MAPPO value.
- **Evidence:** MPE Spread (finds all optima; about 1.7× / 15× faster than RSPO on easy / hard), Atari Pong and Boxing, and SMAC 2s_vs_1sc and 3m. In SMAC the three strategies are aggressive, kiting, and kiting with a drift.
- **Fit to our setup:** PPO-native, and the R_target mask matches what we need for (b).
  - The δ mask has a risk: it withholds the extrinsic reward until the evader is diverse, which could again push it away from the lure.
  - Cost: small to moderate, but a latent-conditioned evader is a structural change.
- **Applied to pursuit-evasion or drones?** Not found.

### A4. RSPO and SIPO (iterative novelty with reward switching)
- **Citations:**
  - Z. Zhou, W. Fu, B. Zhang, Y. Wu. "Continuously Discovering Novel Strategies via Reward-Switching Policy Optimization." ICLR 2022. arXiv:2204.02246. Verified.
  - W. Fu, W. Du, J. Li, S. Chen, J. Zhang, Y. Wu. "Iteratively Learn Diverse Strategies with State Distance Information" (SIPO). NeurIPS 2023. arXiv:2310.14509. Verified.
- **Mechanism:** One new policy per iteration against frozen references.
  - A trajectory is accepted only if its NLL under every reference is at least δ. The threshold is set automatically as δ_j = α·D(π_rnd, π_j).
  - Accepted trajectories get **only extrinsic** reward; rejected ones get **only intrinsic** reward (−log π_j(a|s)).
  - SIPO replaces the action likelihood with state distance (RBF or Wasserstein).
- **How quality is kept:** Per-trajectory switching. The policy is optimised for task reward inside the novel region.
- **Evidence:**
  - SMAC 2c_vs_64zg: per-iteration win rates of 100, 84.4, 96.9, 100, 87.5 and 100%, with 4 visually distinct strategies (versus 2–3 for baselines).
  - MuJoCo population diversity was higher than DvD in 3 of 4 environments.
- **Fit to our setup:** **Poor as the main evader.** By design it forbids reusing the reference strategy, and the lure is the evader's *best* strategy.
  - It could fit as a *side* population of "non-lure" evaders added to the pool, while lure evaders stay in the pool.
  - Cost: high. The paper names sequential iterations and needed batches of roughly 6.4k–16k.

---

## B. League and exploiter structures (target the pursuer's exploitability directly)

### B1. AlphaStar league: main exploiters and league exploiters
- **Citation:** O. Vinyals, I. Babuschkin, W. M. Czarnecki, et al. "Grandmaster level in StarCraft II using multi-agent reinforcement learning." *Nature* 575:350–354, 2019. Verified via the DeepMind blog and a survey; the DOI 10.1038/s41586-019-1724-z is from memory, **[unverified]**.
- **Mechanism:**
  - Main agents train against themselves and all past players.
  - **Main exploiters** train only against the current main agents. They join the pool once they win often, and are then **reset**.
  - **League exploiters** train against the whole league to find global blind spots.
  - DeepMind's stated rationale: "playing to win is insufficient"; exploiters exist to expose the main agent's weaknesses.
  - The exact matchmaking percentages and PFSP weighting functions are **[unverified]**: I could not open the Methods section and did not quote them.
- **How quality is kept:** Role separation. Diversity-seeking and exploitative agents are *separate* from the main agent, so the main agent's objective is never diluted.
- **Evidence:** Grandmaster level, above 99.8% of ranked players. I found no clean, verified ablation isolating the exploiters' effect.
- **Fit to our setup:** **Best structural match.**
  - We already train fresh PPO best-response evaders to measure exploitability, and they find the lure 0.96–0.98 of the time.
  - Adding a main-exploiter evader per pursuer stage to the pool (Hedge still samples it) feeds the pursuer exactly the lure pressure it lacks.
  - It avoids both failure modes: exploiters use only extrinsic reward, so there is no saturation and no bonus.
  - Cost: one extra exploiter run per pursuer stage, reusing existing code.
  - **Caveat:** evaluation exploiters must stay held out and freshly trained, or the exploitability metric is contaminated.
- **Applied to drones?**
  - Quadrotor racing uses "league-based self-play": Geles, Bauersfeld, Wulfmeier, Scaramuzza, arXiv:2605.22748, 2026. The abstract does not say whether it includes exploiters.
  - An air-combat HMARL paper (arXiv:2510.11474) ends its curriculum with league play.
  - No pursuit-evasion paper found with explicit exploiters.

### B2. FXP: Fictitious Cross-Play
- **Citation:** Z. Xu, Y. Liang, C. Yu, Y. Wang, Y. Wu. "Fictitious Cross-Play: Learning Global Nash Equilibrium in Mixed Cooperative-Competitive Games." AAMAS 2023 (per the PDF header). arXiv:2310.03354. Verified.
- **Mechanism:** Two populations.
  - The **main** policy plays itself with η = 0.2. Otherwise it samples from main ∪ counter policies, using a meta-policy plus **win-rate-prioritised** sampling (hardest first).
  - **Counter** policies are best responses to the main population's past versions. They start from scratch or an early checkpoint so they don't inherit the main policy's local optimum, and use a similar-strength curriculum.
- **How quality is kept:** The main policy keeps its weights and only *gains* opponents. The counter-population supplies diverse responses aimed at its weaknesses.
- **Evidence:**
  - MAgent 3v3 Battle: approximate exploitability **10.62** versus 21.21–28.66 for baselines; Elo 1832 versus best baseline 1627.
  - Google Research Football 11v11: over 94% wins against Tikick and the hard built-in AI.
  - Framed for mixed team games; the paper does not test 2-player zero-sum.
- **Fit to our setup:** Essentially B1 without league exploiters, plus a concrete recipe: initialise counter-evaders from an *early* checkpoint so they rediscover lures instead of inheriting current evader habits. Same cost and same caveat as B1.
- **Applied to drones?** No.

### B3. Minimax Exploiter
- **Citation:** D. Bairamian, P. Marcotte, J. Romoff, G. Robert, D. Nowrouzezahrai. "Minimax Exploiter: A Data Efficient Approach for Competitive Self-Play." AAMAS 2024 (per ar5iv). arXiv:2311.17190. Verified.
- **Mechanism:** The exploiter's reward is the environment reward minus α·max_a Q_main(s', a), the frozen main agent's estimated value at the next state, applied to non-terminal steps only.
  - **The shaping is shifted so it is never positive**, explicitly "to prevent reward hacking through cycles of positive reward".
  - The main agent keeps the unmodified reward. A league runs with main and exploiters only (no league exploiter).
- **How quality is kept:** Only the exploiter's reward is changed; the main agent's objective is untouched.
- **Evidence:**
  - For Honor: 11 converged exploiter generations in 24 h versus at most 9 for alternatives, and 16 versus 13 in 100 h.
  - The final Minimax-trained main agent beat the Vanilla, Defensive and Aggressive main agents in 80.26%, 70.31% and 66.25% of duels.
  - Tic-Tac-Toe and Connect-4 also improve. Atari Boxing beats DQN-sparse but not DQN-dense.
- **Fit to our setup:**
  - It makes B1/B2 exploiters cheaper: with PPO, use −α·V_pursuer(s') from the frozen pursuer's critic.
  - Its "never positive" shift is the **literature precedent for our failure mode (b)**: a positive per-step shaping term creates a survival incentive. This supports F6b's zero-mean design; strictly non-positive is the stronger version.
  - Cost: small (load the pursuer critic in evader-exploiter training).
- **Applied to drones?** No.

### B4. Population-size robustness evidence: Czempin & Gleave (2022), and RAP
- **Citations:**
  - P. Czempin, A. Gleave. "Reducing Exploitability with Population Based Training." ICML 2022 Workshop on New Frontiers in Adversarial ML. arXiv:2208.05083. Verified.
  - E. Vinitsky, Y. Du, K. Parvate, K. Jang, P. Abbeel, A. Bayen. "Robust Reinforcement Learning using Adversarial Populations." arXiv:2008.01825, 2020. Verified; venue not confirmed.
- **Finding:** Robustness against *new* adversarial policies, measured as attacker timesteps needed, rises with opponent-population size.
  - Laser Tag: an attacker against the n = 80 victim needed almost 2× the timesteps of the self-play victim. The gains are noisy and diminishing.
  - Simple Push: little benefit beyond n = 2.
  - RAP: a single adversary leaves the policy "highly exploitable by new adversaries"; uniformly sampling a population of adversaries fixes this.
- **Relevance:** The literature recommends *more independently-initialised adversaries*, not more behavioural spread inside one adversary. This is consistent with our Vendi finding: 22 evaders behave like 3.8. Evidence is thin (low-dimensional games, workshop paper).

### B5. ROA-Star: goal-conditioned exploiters
- **Citation:** Huang et al. "A Robust and Opponent-Aware League Training Method for StarCraft II." NeurIPS 2023. Verified via the proceedings and poster page; full author list not checked.
- **Mechanism:** Exploiters are conditioned on a strategy statistic z (build orders from human replays), so they cover distinct weakness types. The paper also adds an opponent-modelling "scouting" reward.
- **Evidence:** Over 50% win rate in repeated games against top humans.
- **Fit to our setup:** An evader exploiter conditioned on a lure type (approach geometry or dive altitude) would *diversify the exploiters* with no bonus term. Cost is moderate. It is an idea for later, not a first pick.

---

## C. Risk-diverse populations (diversity without any intrinsic reward)

### C1. RPPO / RPBT: Learning Diverse Risk Preferences in Population-based Self-play
- **Citation:** Y. Jiang, Q. Liu, et al. "Learning Diverse Risk Preferences in Population-Based Self-Play." AAAI 2024, 38(11):12910–12918. DOI 10.1609/aaai.v38i11.29188. arXiv:2305.11476. Verified.
- **Mechanism:** PPO with an expectile advantage A_τ = 2α·E[τ[δ]₊ + (1−τ)[δ]₋].
  - τ = 0.5 is ordinary PPO. τ → 0 approaches worst-case (risk-averse) learning; τ → 1 approaches best-case.
  - RPBT: a population of 5 agents with τ ∈ {0.1, 0.4, 0.5, 0.6, 0.9}. Low-Elo agents copy a better agent and perturb τ by U[−0.2, 0.2]. Opponents are sampled uniformly from the members and their checkpoints.
  - **No intrinsic reward.**
- **How quality is kept:** Each agent maximises its own task objective; diversity comes only from different risk attitudes. PBT-style exploitation removes weak members.
- **Evidence (best-of-3 seeds by Elo, 200 games):**
  - Slimevolley win rates of 56 / 64 / 64 / 59 / 48 / 60 / 53% against PP / SP / MEP / TrajeDi / DvD / RR / PSRO.
  - SumoAnts: 63 / 67 / 58 / 63 / 54 / 56 / 53%.
  - Diversity was shown only by t-SNE.
  - Moderate evidence. It was not compared against PFSP or Hedge, and best-seed selection inflates the numbers.
- **Fit to our setup:**
  - It avoids **both** failure modes by construction: there is no bonus, and every evader still maximises its win rate, so lures survive.
  - Risk-seeking or risk-averse evaders give *different ways to win* (aggressive close lures versus conservative timeouts).
  - Separately, a **risk-averse pursuer** (τ < 0.5) weights negative TD errors (crashes) more heavily, which may directly reduce susceptibility to lures. This is my hypothesis; the paper does not test it.
  - Cost: a few lines in the GAE/advantage code; RPBT exploitation is optional.
- **Applied to drones?** Not found.

---

## D. Game-theoretic diversity regularisers and meta-solvers (likely ≤ Hedge for us)

| Method | Citation (verified) | What it does | Why it is unlikely to beat Hedge here |
|---|---|---|---|
| PSRO_rN | Balduzzi et al., "Open-ended Learning in Symmetric Zero-sum Games," ICML 2019, arXiv:1901.08106 | BR trained only against Nash-weighted opponents it beats or ties ("play to strengths") | Pipeline PSRO (McAleer et al., NeurIPS 2020, arXiv:2006.08555) shows Rectified PSRO is not guaranteed to converge, with a counterexample, and "rarely converges" in random normal-form games. It specialises agents, like our PFSP result. |
| DPP-PSRO | Perez-Nieves et al., "Modelling Behavioural Diversity for Learning in Open-Ended Games," ICML 2021, arXiv:2103.07927 | DPP diversity term in the BR | Already tested (QD-DPP): no better than uniform. |
| BD&RD-PSRO | Liu et al., NeurIPS 2021, arXiv:2106.04958 | Occupancy-measure behavioural diversity plus response diversity | Built for non-transitive games; argues exploitability is an imperfect metric. No mechanism targets single-agent exploitability. |
| UDM-PSRO | Z. Liu et al., "A Unified Diversity Measure for MARL," NeurIPS 2022 | Unifies effective/population diversity and expected cardinality | Same family as DPP and BD&RD; normal-form and open-ended game benchmarks. |
| PSD-PSRO | Yao et al., NeurIPS 2023, arXiv:2306.16884 | KL policy-space diversity in the BR; guarantees population-exploitability reduction | Already tested, and it hurt the pursuer. The guarantee concerns the *population's* exploitability, not the latest pursuer's. |
| Anytime PSRO | McAleer et al., arXiv:2201.07700 (venue **[unverified]**) | Restricted distribution updated by a **no-regret learner (MWU in deep RL)** against a BR | *Supports* Hedge: MWU is the anytime, low-exploitability meta-solver. In deep RL, APSRO "rarely increases" exploitability versus PSRO (no proof for the RL version). |
| SP-PSRO | McAleer et al., ICLR 2024 ("Toward Optimal Policy Population Growth..."), arXiv:2207.06541 | Also adds a time-averaged mixed strategy each iteration | Faster early convergence in tabular and DDQN games, but "can plateau higher than APSRO". Costs an extra RL run per iteration. |

## E. Other candidates checked (low applicability)

- **DvD.** Parker-Holder, Pacchiano, Choromanski, Roberts. "Effective Diversity in Population Based RL." NeurIPS 2020. arXiv:2002.00632.
  - Measures population volume in a behavioural embedding; a Thompson-sampling bandit adapts the diversity weight.
  - Built on ES and TD3, single-agent. In RPBT's competitive tests it was the strongest diversity baseline: RPBT won only 48% against it on Slimevolley.
  - The bandit-weight idea could replace F6b's fixed small weight, but it is a weaker guarantee than the gates in A1–A3.
- **Cooperative zero-shot coordination methods** optimise partner compatibility, not resistance to an adversary. Little transfer to us except FCP's lesson that past checkpoints at different skill levels are free diversity.
  - TrajeDi: Lupu, Cui, Hu, Foerster, ICML 2021.
  - Fictitious Co-Play: Strouse et al., NeurIPS 2021.
  - MEP: Zhao et al., AAAI 2023, arXiv:2112.11701.
  - LIPO: Charakorn et al., ICLR 2023.
- **QDSP / FMSP.** Dharna, Lu, Clune. "Foundation Model Self-Play: Open-Ended Strategy Innovation via Foundation Models." RLC 2025 / RLJ vol. 6. arXiv:2507.06466.
  - MAP-Elites archive of *foundation-model-written code policies* (Car Tag, Gandalf). Not applicable to PPO neural controllers.
- **QSD-PBT.** Wu, Yao, Fu et al. "Quality-Similar Diversity via Population Based RL." ICLR 2023. Verified via the ICLR poster page; full text not read.
  - Diversity among policies of *similar* quality, using an adaptive diversity loss. MuJoCo and Atari only.
- **NeuPL.** S. Liu, Marris, Hennes, Merel, Heess, Graepel. ICLR 2022. arXiv:2202.07415.
  - One conditional network represents a population of best responses to mixtures. Too large a restructure for us.

## F. Novelty note (pursuit-evasion and drones)

- The closest prior pipelines are:
  - **Sanchez Roncero et al.** (arXiv:2506.02849; AMSPB population training; Hedge per our setup).
  - **AgilePE**: Tang, …, Yu Wang, Chao Yu. "AgilePE: Autonomous UAV Pursuit-Evasion via Self-Play Reinforcement Learning." arXiv:2608.14135, Aug 2026. OmniDrones, PFSP, and "a diversified opponent pool"; the abstract does not specify the diversification mechanism.
- Related drone self-play: league-based multi-quadrotor racing (Geles et al., arXiv:2605.22748, 2026) and CRUISE curriculum self-play for drone racing (arXiv:2510.22570, 2025).
- **I found no paper applying any of the following to quadrotor pursuit-evasion:** quality-constrained diversity (SMERL, DOMiNO, DGPO, RSPO), main exploiters or counter-populations, Minimax-exploiter shaping, or risk-diverse populations.
- My search was ~30 queries, not a systematic review, so treat novelty claims as "not found".

---

## Ranked shortlist for our setting

**1. Exploiter-augmented pool: AlphaStar main exploiters / FXP counter-population, optionally with Minimax-Exploiter shaping.**
- **Why it targets the failure:** The pursuer's measured weakness is the lure into a floor crash, and fresh best-response evaders find it almost every time.
- **What changes:** At each pursuer stage, add one or more counter-evaders to the pool. Each is trained purely on the extrinsic reward against the latest pursuer (or the recent-pursuer mixture, as in FXP), initialised from an early checkpoint (FXP), and reset after joining (AlphaStar). Keep Hedge as the sampler; Anytime PSRO says MWU/Hedge is the right meta-solver.
- **Why it avoids our failure modes:** There is no diversity term to saturate and no bonus to change how evaders win. Lure pressure becomes explicit.
- **Diversity:** It is *response* diversity aimed at the pursuer's weaknesses, which is the kind Czempin & Gleave and RAP link to lower exploitability.
- **Cost and evidence:** One extra evader run per pursuer stage, reusing the exploiter code. Minimax shaping (−α·V_pursuer(s'), shifted ≤ 0) can make those runs cheaper. Evidence: FXP halves approximate exploitability in MAgent (10.62 vs ≥ 21.21), and AlphaStar uses the structure at scale.
- **Risks:**
  - The pursuer may over-specialise to the lure exploiters, as PFSP did. Hedge's γ = 0.7 uniform share is the mitigation.
  - The evaluation exploiters must stay held out.

**2. Quality-gated F6b: a SMERL gate or DOMiNO Lagrangian on the action-unit diversity term.**
- **What changes:** Keep F6b's bounded 1−exp(−d/τ) action-space distance and zero-mean centring. Replace "small fixed weight + pilot guard" with a training-time constraint:
  - Either the SMERL gate 1[rolling evader win rate vs pool ≥ (1−ε)·reference], where the reference comes from the no-diversity control at the same stage;
  - or the DOMiNO convex combination σ(μ)·r_e + (1−σ(μ))·r_d, with μ raised whenever the constraint is violated and a separate value head for r_d.
- **Additional safeguard (my suggestion, not from the literature):** Also gate on *how* the evader wins, e.g. a crash-win share at least the reference's, because our failure (b) kept the win rate but switched the win mode.
- **Why it ranks second:** It directly addresses both diagnosed failures (bounded distance, so no saturation; constrained, so no survival bonus) at low cost.
- **Evidence gap:** None of these papers tests whether the *opponent* gets stronger from a diverse population. The likely outcome is a more diverse evader pool with a pursuer roughly equal to Hedge, not clearly better.

**3. Risk-diverse evaders (and possibly a risk-averse pursuer) with RPPO.**
- **Mechanism:** Train evader stages, or parallel evaders, with different expectile τ. This yields behavioural diversity with **no intrinsic reward**, so every evader still maximises its win rate and keeps using lures.
- **Pursuer variant:** A τ < 0.5 pursuer learns from worst-case outcomes, which may resist lure-induced floor crashes. This is untested and my hypothesis.
- **Cost:** A few lines in the advantage computation.
- **Evidence:** Moderate and somewhat inflated (best-of-3 seeds; 53–67% win rates against SP, PSRO, MEP, TrajeDi and others; 48% against DvD on Slimevolley). It was not compared against PFSP or Hedge.
- **Why it ranks third:** It is the cheapest, cleanest way to get diversity without touching the reward. The risk-averse pursuer is a separate, falsifiable experiment.

## Which are most likely to be no better than Hedge

- **Diversity-regularised BRs** (DPP-PSRO, BD&RD, UDM, PSD-PSRO, DvD-style bonuses): these target non-transitive gamescape growth. Two of them already failed here, and our game is mostly transitive.
- **Pure sampler changes** (PSRO_rN, PFSP-style hardness, DPP sampling): Hedge/MWU is already the theoretically preferred meta-solver (Anytime PSRO), and rectified Nash has known non-convergence and specialisation.
- **RSPO and SIPO as the main evader:** they deliberately avoid the reference (lure) strategy and cost M sequential runs.
- **Cooperative ZSC methods and FM-based QD self-play:** they solve a different problem.
- **Even for the shortlist, evidence is thin.** No verified source shows a quality-constrained diversity method beating a tuned hardness-based sampler on the *main agent's exploitability* in a continuous-control zero-sum game. FXP (mixed team games) and Minimax Exploiter (For Honor, DQN) come closest, and both work by adding exploiters, not by diversity bonuses.

## Sources (main)
arxiv.org/abs/2010.14484 · arxiv.org/abs/2205.13521 · arxiv.org/abs/2207.05631 (AAAI DOI 10.1609/aaai.v38i10.29019) · arxiv.org/abs/2204.02246 · arxiv.org/abs/2310.14509 · deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/ · arxiv.org/abs/2408.01072 (self-play survey, league description) · arxiv.org/abs/2310.03354 · arxiv.org/abs/2311.17190 · arxiv.org/abs/2208.05083 · arxiv.org/abs/2008.01825 · neurips.cc/virtual/2023/poster/70220 (ROA-Star) · arxiv.org/abs/2305.11476 · arxiv.org/abs/1901.08106 · arxiv.org/abs/2006.08555 · arxiv.org/abs/2103.07927 · arxiv.org/abs/2106.04958 · proceedings.neurips.cc (UDM, NeurIPS 2022) · arxiv.org/abs/2306.16884 · arxiv.org/abs/2201.07700 · arxiv.org/abs/2207.06541 · arxiv.org/abs/2002.00632 · proceedings.mlr.press/v139/lupu21a.html · arxiv.org/abs/2112.11701 · iclr.cc/virtual/2023/poster/11264 (QSD-PBT) · sites.google.com/view/iclr-lipo-2023 · arxiv.org/abs/2507.06466 · arxiv.org/abs/2202.07415 · arxiv.org/abs/2506.02849 · arxiv.org/abs/2608.14135 · arxiv.org/abs/2605.22748 · arxiv.org/abs/2510.22570 · arxiv.org/abs/2510.11474
