# Progress report (thesis experiments)

Living log so that any agent or person can resume the work. Plan: [docs/research_plan.md](../../docs/research_plan.md).
Results: [RESULTS.md](./RESULTS.md). Update this file after every meaningful step (newest entries at the top of
the log).

## How to resume (read this first)

1. Read section "Current state" below, then the plan's section for the active experiment.
2. Check whether a job is still running: `nvidia-smi` and `ps aux | grep -E "run_stage|thesis/run.py" | grep -v grep`.
   Only one Isaac job may run at a time (single RTX 5090).
3. Every run writes a `DONE` marker in its directory. The driver skips finished runs, so re-running the same command
   resumes where it stopped: `python scripts/experiments/thesis/run.py --exp <ID>` (from the repo root, inside the
   project's Python env; see "Environment").
4. When an experiment finishes: generate its `summary.md`, copy the numbers into RESULTS.md (exact values, CIs),
   **update the "Summary" section at the top of RESULTS.md (and its timestamp)**, tick it in the table below, and add
   a log entry. The user asked for the summary to be refreshed after every new result.

## Environment

- Repo root: `/home/user/Projects/RLInterceptorDrone`, git base commit `786cedd` (plan work is uncommitted unless noted).
- Python: `env -u PYTHONPATH .venv/bin/python` (py3.11, torch 2.7, torchrl 0.11.1). **Unset PYTHONPATH**: the shell
  exports ROS humble (py3.10) paths that break pytest/imports. The driver strips it for subprocesses.
- Tests: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/` (all must pass).
- Git: work is on branch `thesis-experiments` (pushed to `origin` = github.com/KhairulM/OmniDrones). `run.py` calls
  `scripts/experiments/thesis/git_sync.sh` after every finished (non-smoke) experiment: commits code + text results and
  pushes (user request 10-28). `.pt` files, Isaac logs, locks and lane stdout are git-ignored.
- Isaac/OmniDrones runs: `scripts/selfplay/run_stage.py` (Hydra; `mode=train|eval`). The thesis driver wraps it.
- wandb disabled for thesis runs (`wandb.mode=disabled`).

## Current state

| ID | Status | Output dir | Notes |
|---|---|---|---|
| code-metrics | done (eval part) | - | multi-episode eval (`eval_cutoff_steps`), neutralisation, behaviour features: `omni_drones/utils/selfplay/eval_accumulator.py`, `run_stage.py::_run_eval_multi`; tests `tests/test_eval_accumulator.py`. Oracle obs keys: still to do (needed for X4) |
| X0_replay | **done** | `X0_replay/` | v9/v10/v11 valid, v6 invalid (excluded). In RESULTS.md |
| X1_crossplay | **done** | `X1_crossplay/` | In RESULTS.md |
| X2_descriptors | **done** | `X2_descriptors/` | In RESULTS.md; `split.json`, `descriptors.json`, `clusters.json` |
| L0 | **done** | `L0_literature/notes.md` | Research-agent review (verified arXiv fetches). Gate G2 decided (see Decisions) |
| X3_gap | **done** (19:02) | `X3_gap/` | `exps/x3_gap.py`: generalists (=M0=S-U), exploiters, specialists |
| X4_oracles | **done** (10-27 04:43) | `X4_oracles/` | O1 descriptor, O2a accel, O2b command (O3 dropped, see plan deviation). Env oracle keys `task.oracle.*` added. Gate G1 after this |
| D1_sampler / D1b | D1 **done, 5 seeds** (final); D1b **done** (H1b not supported) | `D1_sampler/`, `D1b_duplicates/` | `exps/d_sampling.py`; sampler `omni_drones/utils/selfplay/samplers.py` (tests `tests/test_samplers.py`); in-run refresh `run_stage.py::_OpponentRefresher` (`refresh.*` config) |
| D2_exploitability | **done, 5 seeds** (final) | `D2_exploitability/` | `exps/d_sampling.py::run_d2` |
| A1a / A1b | **done** (A1b final, 5 seeds) | `A1a_basis/`, `A1b_policy/` | `exps/a_adapt.py`; adapter `omni_drones/utils/opponent_adapt.py` (tests `tests/test_opponent_adapt.py`), env `task.adapt.*`, eval trajectory recording `eval_traj_out`. Offline pipeline smoke-tested on synthetic data |
| A2a / A2b (learned accel. estimator) | A2a **done**; A2b **done (5 seeds, 21:10)**, H*=10 | `A2a_nfacc/`, `A2b_nfacc/` | added 10-27 (user choice B) |
| D1c (exploratory) | **done** (10-28 19:25), in RESULTS.md | `D1c_nofloor/` | |
| A3a / A3b / A3c (positions-only + sensor noise) | A3a **done** (run 2, float32, 18:46; H*=10, aug basis, EMA 0.05). A3b **done** (02:15, 3 seeds); A3c **done** (03:47, 5 seeds) | `A3a_posonly/`, `A3b_posonly/`, `A3c_noise_ref/` | added 10-28 (user request); `exps/a3_posonly.py`; env `observation.agent_rel_vel`, `oracle.accel_source=pos`, `adapt.z_mode=pos` (smoke-tested) |
| A3d (learned velocity estimate from noisy positions) | offline **running** (11:40, no Isaac); closed loop (V_ema, V_nf x 3 seeds) runs at the start of F4 | `A3d_velest/` | added 10-29 (user choice after A3b/A3c), pre-registered 11:30; `exps/a3d_velest.py`; env `observation.rel_vel_source`, adapter target `vel` (unit-tested; first Isaac use = A3d closed loop) |
| F2a / F2b / F2c (final runs) | F2a **done** (06:19, 5 seeds); smoke tests F2a/F2b/F3/F4v1 passed 03:48-04:51; F2b **running** (both campaigns, since 05:50 / 06:19, ~10 h) | `F2a_exploit_acc/`, `scripts/selfplay/outputs/thesis-F2-{acc,plain}/`, `F2c_final_eval/` | pre-registered 10-28 19:30 (plan Phase F); `exps/f2_final.py`; env flag `oracle.opp_side_accel` |
| F3 / F3c (samplers in self-play) | queued after F2b on lanes C/D | `scripts/selfplay/outputs/thesis-F3-*`, `F3_sampler_sp/` | pre-registered 10-28 20:30; `exps/f3_sampler_sp.py`; orchestrator `pursuer_sampler: hedge|qd_dpp`, `eval_extra_overrides` |
| F4 / F4c **v2** (learned opponent model in self-play, estimated velocity + noise) | redesigned 10-29 11:30 (user choice): after F2b, A3d closed loop -> v2 smoke (both arms) -> 6 campaigns | `scripts/selfplay/outputs/thesis-F4-*`, `F4_estimator_sp/` | pre-registered 10-28 21:30; `exps/f4_estimator_sp.py`; env `observation.opp_drop_rel_vel`, `adapt.opp_side`; orchestrator `role_task_overrides` |
| D3 campaign (QD-DPP), BayesID | dropped (D3 replaced by F2b/F3) | | |
| F1_final | **done** (10-28 04:22, 5 seeds) | `F1_final/` | 2x2 factorial, 5 seeds, `exps/f1_final.py` (reuses X3 / D1 / X4 runs) |

Running jobs (relaunched 10-28 18:27, detached with `nohup setsid`, stdout in `laneA_out.txt` / `laneB_out.txt`):
lane A = `D1c --seeds 3` -> `A2b --seeds 5` -> (waits for `A3a_posonly/choice.json`) -> `A3b` -> `A3c`;
lane B = `A3a` -> `A2b --seeds 5` -> `A3b` -> `A3c`. Runs are shared via per-run lock files. **At most 2 Isaac
processes run at once across all drivers** (`common.MAX_ISAAC`, slot files in `.isaac_slots/`): each needs ~17.5 GB
of host RAM on the 62 GB machine. If anything died, re-run `run.py --exp <ID>` (resumable, concurrency-safe).
lane C = waits for lanes A and B to exit -> smoke tests `F2a`, `F2b`, `F3`, `F4` (`--smoke`, chained with &&) && touch
`F2a_exploit_acc/smoke/PASSED` -> `F2a` -> `F2b` -> `F4` -> `F3` -> `F2c` -> `F4c` -> `F3c`; lane D = waits for A/B and the PASSED marker,
then the same chain (stdout `laneC_out.txt`, `laneD_out.txt`). Campaigns (F2b: 2, F4: 6, F3: 8 new) are shared between lanes via
`<campaign>/DONE.lock`; each holds one Isaac slot for ~10 h. If a smoke test fails, both lanes stop: fix, rerun the chains.
Expected: F2 campaigns ~10-29 16:00, F4 ~10-30 22:00, F3 ~11-01 14:00, final evals ~11-01 night. If late, drop F3 seed 2.

## Decisions taken

- 2026-10-28 (user): working thesis title: "Beyond Hard Opponents: Diversity-Aware Opponent Sampling and Learned
  Opponent Models for Agile Quadrotor Pursuit-Evasion".
- 2026-10-28 (user): the thesis is positioned against three reference papers: AgilePE (Tang et al. 2026), Roncero
  et al. 2026 (AMSPBH), Gavin & Bronz (ICUAS 2026). Every result must be stated in relation to them, truthfully, in
  RESULTS.md section "Relation to the reference papers". After all experiments: optionally one final run for a
  publishable method, then answer RQ1/RQ2 and H1a-H2d explicitly.
- 2026-10-26 (gate G2, after L0): keep **QD-DPP** (A) and **NF-OA** (B) as the thesis methods. BayesID (C) is
  dropped to P2: it is essentially GSCU (Fu et al. 2022) moved to drones, the weakest novelty claim. Novelty of (A)
  as a *sampling* rule should still be citation-chased (Perez-Nieves 2021 citing papers) before the write-up.
- 2026-10-26: X4 O3 (future-position oracle) dropped; the A1a horizon study answers the same question offline.
- 2026-10-26: Isaac smoke test (256 envs, 1.5M frames) of refresh/qd_dpp + adapter(raw) + oracle accel + S2 eval +
  trajectory recording passed; smoke files deleted.

- 2026-10-26: Plan approved by the user. Old plan files `docs/esn_adaptation_experiment_plan.md` and
  `docs/online_adaptation_experiment_plan_v2.md` removed (`git rm`, staged, not committed).

## Important facts found during the work

- One env step = 0.02 s (sim dt 0.01 x 2 substeps). The self-play `avg_time_to_capture` uses 0.01 s/step (2x too
  small). Thesis evals use 0.02 s (`common.STEP_DT`). Max episode = 1500 steps = 30 s.
- Heuristic time inside the env is also `progress x sim.dt`: a switch evader `a>b@T` switches at T/0.01 steps, i.e.
  real time 2T. The P0 switch evaders (`@2.0`) switched after 200 steps, when most episodes are already over.
  Thesis switch evaders use `@0.5` (50 steps = 1 s real).
- Training cost: 150M frames vs the 20-evader train pool = 1960 s (~76k frames/s; each RL opponent is a separate
  forward pass). With two lanes (from 10-26 15:35) both are estimated to end around midday 10-27 (concurrent jobs run somewhat slower each).
- Multi-episode eval: ~100k episodes per 4096-env eval (~25 per env), ~2.7 min per eval job incl. Isaac start-up.
- v6 checkpoints replay with evaders leaving the arena in 85-99.8 % of episodes (current env rules differ from v6),
  so v6 is excluded from all pools.

## Log

(Timestamps of log entries written on 10-27 before 21:20 are approximate; `driver_log.txt` has exact times.)

- 2026-10-29 12:00: F2a done (5 seeds): acceleration input lowers exploitability 0.672 -> 0.397, paired -0.275 [-0.522, -0.044] (also without the one failed exploiter). Lane-lock bug in the F2a summary (3 of 5 seeds) fixed and re-analysed. User chose F4 redesign "estimator outputs velocity": added A3d (learned velocity + acceleration from noisy positions vs EMA finite difference; offline now, closed loop at the start of F4) and F4 v2 (arms nf vs ema), pre-registered in plan before any result. F2b running on both slots until ~16-17 h.
- 2026-10-29 04:00: A3c done (5 seeds, eval-time noise): finite-difference acceleration collapses under noise (train 0.94 -> 0.33, below M0); learned estimator + RLS holds (0.80), beats FD on every split (+0.20..+0.49) and M0 on held-out RL / exploiters; online RLS > frozen under noise (+0.09 train, +0.24 exploiters) - first closed-loop gain from adaptation. In RESULTS.md (summary, reference rows 7-8). F2a smoke started 03:48. F4 design question put to the user.
- 2026-10-29 02:40: A3b done (3 seeds): without the evader velocity all pursuers are weak (train ~0.29 vs M0 0.65). Under noise the position-difference input collapses (0.38 -> 0.07); the learned estimator is noise-proof and beats it (+0.23), but is not better than no estimator (worse on exploiters). In RESULTS.md. **Issue for F4 (needs user decision before F4 starts, ~12 h away):** F4 as pre-registered gives the pursuer no evader velocity and only an acceleration estimate, exactly the A3b inputs, so it will most likely compare two crippled pursuers (expected nf ~ none). Options to put to the user once A3c is in (~03:45): keep F4 as registered / redesign the estimator to also output the evader velocity / give both arms noisy velocity and compare the acceleration inputs. Last P_fd noisyema eval (seed 2) still running on lane B; A3c started 02:15 on lane A.
- 2026-10-28 23:25: A2b 5 seeds done: learned acceleration estimate = finite-difference input (all paired diffs within +/-0.04, n.s.; the 3-seed 'frozen lower on train' is gone); big gain over M0 holds; one frozen run partly degraded on v10 (0.34-0.60), so 'no v10 failures' weakened. RLS adds nothing. A3b seed 0 done for all 3 arms, seeds 1-2 running (ETA ~01:30), then A3c, then F2a smoke -> finals.
- 2026-10-28 22:00: user: push to GitHub after every finished experiment. Branch `thesis-experiments` created and pushed (558a8fb); auto-sync hook in `run.py`.
- 2026-10-28 21:45: D1c done: no floor -> both samplers collapse (train -0.29); duplicates then hurt S-M but not QD-DPP. In RESULTS.md (summary + reference-paper table).
- 2026-10-28 21:35: user: add the learned-estimator campaign regardless of A3b/A3c (real interceptors rarely measure
  evader velocity/acceleration). F4 pre-registered (positions only + sensor noise, nf vs none, 3 seeds). Also found and
  fixed: new env keys were missing from `cfg/task/PursuitEvasion.yaml` (Hydra would have rejected them). Lanes C/D
  relaunched with F4.
- 2026-10-28 20:35: user chose F3 = samplers in self-play (uniform / Roncero Hedge / QD-DPP x 3 seeds) and to decide an RQ2
  learned-estimator campaign after A3b/A3c. Orchestrator: `pursuer_sampler` (Hedge eq. 12, QD-DPP with Hedge quality),
  `eval_extra_overrides`; tests added (`tests/test_samplers.py`, 48 pass). Lanes C/D relaunched with F3 + F3 smoke.
- 2026-10-28 19:40: user chose final runs = exploitability test of acc pursuers (F2a) + two self-play campaigns acc vs
  plain (F2b) + final eval (F2c). Pre-registered in the plan. New env flag `oracle.opp_side_accel` (frozen opponents get
  the acceleration input; default off). Lanes C/D queued behind A/B with Isaac smoke tests first.
- 2026-10-28 19:10: RESULTS.md: added "Relation to the reference papers" (setting table + 9 claims). New paired test: win-rate PFSP/Hedge-like sampler is worse than uniform on 5/5 seeds.
- 2026-10-28 18:50: A3a done (run 2): noise-free position FD beats the learned estimator; with noise only the augmented learned estimator beats zero. In RESULTS.md. Lane B now on A2b.
- 2026-10-28 18:36: A3a run 1 finished, but the trajectory recorder stored positions as float16 -> clean FD baseline
  artefact (worse than zero). Recorder switched to float32 (`run_stage.py`), run 1 moved to `A3a_posonly/run1_fp16/`,
  lane B stopped (A2b seed 3 was 1 min in) and relaunched with A3a first. Plan §9 deviation logged.
- 2026-10-28 18:27: found all jobs dead. Cause: host-RAM OOM kills 15:40-16:14 (three Isaac processes at once:
  two lanes + A3a, ~17.5 GB each; the `No space left on device` lines in Isaac logs are inotify warnings, not the cause).
  Lost: A2b seeds 3-4, A3a test recording, D1c dup/QD/seed 2. Fix (already in `common.py`): global 2-slot Isaac cap.
  Relaunched both lanes.
- 2026-10-28 15:45: user asked for A3 (positions only, offline + policy + noise). Pre-registered in plan; code + Isaac smoke test done; evaluate() now also uses run locks. A3a started directly.
- 2026-10-28 11:06: A2b (3 seeds) done: learned acceleration estimate ~ finite-difference input, big gain over M0, no v10-lineage failures in 6 runs; online RLS adds nothing. Queued A2b seeds 3-4 on both lanes after D1c.
- 2026-10-28 10:05: A1b 5 seeds final: no adapter beats M0; ESN/frame stack significantly worse. Lane B idle -> D1c split across both lanes with run locks.
- 2026-10-28 04:25: F1 done: acceleration input dominant (+0.20 train, +0.46 exploiters); sampler no main effect; held-out bimodal per seed. Lane A now A2a (recompute) -> A2b -> D1c.
- 2026-10-28 01:09: D2 5 seeds final: no sampler reduces exploitability; S-W seed-4 exploiter failed to learn. Lane B now on A1b seeds 3-4.
- 2026-10-27 21:20: A2a done (offline): learned accel. estimator beats FD for H >= 5-10; RLS helps here (-10 to -15 %). H*=10 for A2b.
- 2026-10-27 21:20: user chose F1 option B without sensor noise (noise -> P2). D1c stopped (2/12 runs kept). D1 5 seeds final: QD > S-M and S-W (CIs exclude 0), QD vs uniform not significant. F1 launched on lane A. Plan updated (§4 F1/A2, §5 noise, §9).
- 2026-10-27 18:40: A1b done: NF basis helps a little, online RLS adds nothing reliable, S2 = S1; acceleration input (O2a) far better. RESULTS + summary updated. Lane A now on D1c.
- 2026-10-27 12:50: early A1b (M0h_stack, ESN; 3 seeds): frame stack hurts train/held-out J (0.45-0.50 / 0.33 vs M0 0.65 / 0.47); ESN ~ M0 or slightly worse; S2 ~ S1. Concern: with H*=5 the NF prediction ~ rel_vel x 0.1 s (already observed); the acceleration part is ~0.05 m. If NF_OA ~ M0, candidate follow-up: feed the residual over CV (= learned adaptive acceleration estimate) and compare to O2a, also under noise.
- 2026-10-27 12:45: D2 (3 seeds) done: all samplers highly exploitable (0.54-0.70); QD not lower. Lane B now on the 5-seed tail (X3 generalists 3-4 started 12:44).
- 2026-10-27 06:35: D1b done: DPP bounds duplicate mass (0.21 vs 0.29) but QD lost more held-out J under duplication than S-M (-0.23 vs -0.10). H1b not supported. Candidate follow-up: uniform_mix=0.
- 2026-10-27 05:23: A1a run 1 had RLS wind-up -> bounded-covariance RLS + tuning; run 2 done (RESULTS). A1b restarted with tuned values, H*=5.
- 2026-10-27 04:45: X4 done, gate G1 passed (O2a acceleration input closes 108 % of the gap; O1 descriptor collapses out of distribution). A1a started. TODO for F1: evaluate under obs noise, with noise also on the acceleration estimate.
- 2026-10-27 00:20: D1 (3 seeds) done; interim results in RESULTS.md (QD ahead on held-out RL in 3/3 seeds).
- 2026-10-26 19:20: decided 5 seeds for D1/D2/A1b (see plan §9). D1 seed-0 refresh logs checked: probabilities move towards v9 s13-s19 (max 0.06-0.11 vs uniform 0.05).
- 2026-10-26 19:03: X3 done: 7 evaders with Delta >= 0.15 (RESULTS.md). X4 started. Note: large seed variance of the generalist (train J 0.75/0.64/0.56) -> consider 5 seeds for the final comparisons.
- 2026-10-26 15:35: user: the RTX 5090 runs two Isaac jobs at once -> two lanes.
- 2026-10-26 13:10: X1, X2 done (RESULTS.md). X3 started.
- 2026-10-26 12:12: X0 done (RESULTS.md). Chain X1->X4 started.
- 2026-10-26 12:05: X0 started (one earlier attempt failed due to a bug I introduced in run_stage.py, fixed).
- 2026-10-26: Created plan, progress report, RESULTS skeleton, `.gitignore` entries for `experiments/thesis/**/ckpt/`.
