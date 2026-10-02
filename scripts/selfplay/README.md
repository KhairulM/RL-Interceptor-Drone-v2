# Pursuit–Evasion Self-Play

Population-based self-play for 1v1 quadrotor pursuit–evasion, implemented on top of
the OmniDrones / Isaac Sim stack in this repo. Each generation trains **one** side
with PPO against a **frozen** opponent drawn from a growing population, then
evaluates the new policy against the pool and adds it back. Opponents are sampled
with a **temperature-scaled, multi-metric softmax** over win rate,
time-to-capture, distance, motor effort, and the training agent's own
crash/out-of-bounds rate, so training focuses on whichever frozen opponents are
currently hardest.

---

## Components

| Component | Path |
|-----------|------|
| Task environment (shared, role-flagged) | `omni_drones/envs/single/pursuit_evasion.py` |
| Task config | `cfg/task/PursuitEvasion.yaml` |
| Policy database (JSON) | `omni_drones/utils/selfplay/policy_db.py` |
| Frozen opponent bank | `omni_drones/utils/selfplay/opponents.py` |
| Generational driver (orchestrator) | `scripts/selfplay/train_selfplay.py` (+ `.yaml`) |
| Per-stage worker (Isaac process) | `scripts/selfplay/run_stage.py` (+ `.yaml`) |

The task is registered via `omni_drones/envs/single/__init__.py` and is available
as `task=PursuitEvasion`.

---

## The task

A single shared environment `PursuitEvasion(IsaacEnv)` with a **`role` flag**
(`pursuer` or `evader`). The `role` side is trained with RL; the opponent is a
frozen policy stepped internally. Both drones are Crazyflies commanded via
collective-thrust-and-body-rate (CTBR) through the on-board `PIDRateController`
(`action_transform: pidrate`).

### Observation (per side, symmetric)

Built in the drone's own body frame so a frozen policy trained as one role can be
replayed as the other role's opponent:

1. agent position (3)
2. flattened rotation matrix (9)
3. body-frame linear velocity (3)
4. body-frame angular velocity (3)
5. opponent relative vector in the agent body frame (3) — the raw relative
   position when `observation.use_rel_dist: true` (default), or its unit heading
   `normalize(rel_dist)` when `false`
6. opponent relative linear velocity in the agent body frame (3)
7. previous action (4) — if `observation.use_previous_action`

Total: **24** (or **28** with previous action). A time encoding is appended to the
critic `state` when `observation.use_time_encoding` is set.

> Note: the deployment observation in `scripts/deploy/pursuit_evasion_common.py`
> must match component 5's `use_rel_dist` setting bit-for-bit, or the exported
> policy sees an out-of-distribution input on hardware.

Optional per-component Gaussian sensor noise is added to the **agent's**
observation (never the frozen opponent's) when `observation.include_noise: true`;
stds are configured under `observation.noise` (defaults matched to Intercept).

### Action

`[wx, wy, wz, T]` — desired body rates + normalized collective thrust, tracked by
the PID rate controller.

### Reward (asymmetric dense terms, symmetric terminal)

Terminal outcomes use a single magnitude `W = reward_terminal_weight`, plus
**partial credit** `s·W` (`s = reward_forced_error_scale`, default `0.25`) when
the *opponent* destroys itself:

* **Full win (+W):** pursuer → capture; evader → survived to the timeout.
* **Full loss (−W):** the agent's **own** crash to the ground
  (`z < minimum_altitude`) / NaN state, its **own** departure through a wall or
  ceiling, and (pursuer only) the timeout.
* **Forced error (+s·W):** the **opponent** crashed or left the arena. This is
  deliberately *not* a full win. At parity the pursuer simply farmed it —
  standing off and waiting for a scripted evader to fall out of the sky paid the
  same as a capture, so 93% of its "wins" were evader crashes and the capture
  rate collapsed to ~3%. Partial credit keeps the "you forced the error" signal
  while leaving capture strictly the best outcome. Symmetric for both roles.

| outcome | pursuer | evader |
|---|:--:|:--:|
| capture | **+W** | −W |
| timeout (evader survived) | −W | **+W** |
| opponent crashes | +s·W | +s·W |
| opponent leaves arena | +s·W | +s·W |
| own crash | −W | −W |
| own departure from arena | −W | −W |

A forced error is only credited when the episode was not already resolved as a
win or a loss for the agent, so simultaneous events (both drones hitting the
ground; a capture of an evader that is also below the crash altitude) resolve to
a single clean value in `{−W, 0, +s·W, +W}`.

Dense terms: the pursuer gets `approach` (distance-closing) + `precision`
(potential `exp(-k·d)`) and, when `use_distance` is set, a per-step distance
penalty (`reward_distance_weight · distance`); the evader gets a small per-step
survival reward (`reward_step`). Both get a `bounds` penalty that ramps up within
`bounds_margin` of a wall (so the boundary is learned before it is hit) and
**three complementary smoothness penalties**:

| penalty | on what | why it alone is not enough |
|---|---|---|
| `reward_body_rate_weight` | the drone's **actual** body angular velocity | at the 50 Hz training rate the inner loop never fully realises a saturated command within a step, so this barely fires |
| `reward_action_smoothness_weight` | **step-to-step change** of the CTBR action (jerk) | a *sustained* saturated command has no jerk, so it is ~free |
| `reward_cmd_rate_weight` | **magnitude** of the commanded body rate (`\|tanh action\|` over the 3 rate channels) | closes the gap above — the real / CrazySim firmware rate loop runs ~10× faster, tracks crisp saturated setpoints faithfully, and tumbles |

The third term is the anti-tumble lever: without it a policy can hold ±180 deg/s
setpoints essentially for free in training and then tumble on deployment.

Weights: `reward_approach_weight`, `reward_precision_weight`/`_scale`,
`reward_step`, `reward_body_rate_weight`, `reward_action_smoothness_weight`,
`reward_cmd_rate_weight`, `use_distance`/`reward_distance_weight`,
`reward_bounds_weight`, `bounds_margin`, `reward_terminal_weight`,
`reward_forced_error_scale`.

### Arena, capture, termination

- Confined box (env-local): `arena.half_xy`, `arena.z_min`, `arena.z_max`
  (default ≈ 10 m × 10 m × 4 m). Drawn as a boundary box when not headless
  (`arena.visualize: true`).
- Capture when inter-drone distance ≤ the **curriculum capture radius**.
- Terminate on capture, on **either** drone crashing to the ground
  (`z < minimum_altitude`), on **either** drone leaving the arena through a
  wall/ceiling, or on a NaN state; truncate at `env.max_episode_length` (an
  evader escape). Crashing or leaving is a full loss for the drone responsible
  and **partial credit** (`s·W`) for the other — never a full win (see Reward).

### Capture-radius curriculum

Like Intercept, the capture threshold shrinks linearly during training and is held
fixed for evaluation:

```
radius(step) = max(success_radius_init − success_radius_lr · global_step,
                   success_radius_end)          # training
radius       = success_radius_eval              # eval / not training
```

Logged as `stats.capture_radius`.

---

## Opponent pool

Each parallel env is assigned **one** frozen opponent for the whole run (fixed per
generation). Opponents are either:

- **RL policies** — frozen PPO checkpoints, run deterministically; their CTBR
  output is converted to motor commands via the opponent's `PIDRateController`.
- **Heuristics** — scripted position targets flown through the same proven
  CTBR + on-board rate-PID stack as the RL policies (`GeometricCTBR`; the plain
  `LeePositionController` is only marginally stable on the motor-lagged
  Crazyflie):
  - `hover` — hold spawn position
  - `circular` — orbit the spawn point (`heuristic.circular_radius`, `circular_omega`)
  - `pursue` — chase the other drone
  - `flee` — run **horizontally** away from the pursuer at a **held spawn
    altitude** (see below)

All heuristic targets are clamped into the arena shrunk by
`heuristic.safe_margin`, and the opponent controller's tilt is capped at
`heuristic.max_tilt_deg` (default 22°).

#### Why `flee` is built the way it is

A naive "fly directly away in 3D at full speed" flee destroys itself on a
thrust-limited Crazyflie, and a self-destructing opponent poisons training (it
ends episodes before any interception can happen). Measured in-sim, the naive
version self-destructed in **94%** of episodes and was captured only **6%** of
the time. Three properties fix it, and all three are needed:

1. **Horizontal-only escape at a held spawn altitude.** The 3D version pointed
   *down* whenever the pursuer was above (half of all spawns), making "flee" a
   dive command. A constant altitude setpoint can only ever command climb-or-hold.
2. **Outward speed faded to zero within `heuristic.brake_zone` of the keep-in
   edge** (`arena.half_xy − heuristic.wall_margin`). A target that recedes at
   full speed up to the wall builds momentum that a position push cannot brake:
   the drone punched through the wall *and* the hard braking tilt dumped the lift
   it needed, so it dived at the same time.
3. **A gentle, always-on inward push** (`heuristic.wall_gain`), kept *separate*
   from the faded outward drive. Folding the push into the same faded step kills
   it exactly where it is needed and the drone drifts into the wall.

Result: self-destruct **94% → ~11%**, captured **6% → ~89%** — an interceptable
opponent instead of one that removes itself. Interior flee speed is ~1.3 m/s.

Per-env assignment uses `omni_drones/utils/sampling.py::policy_sampling` with the
sampling probabilities from the policy database.

---

## Policy database

JSON files under the database directory (`policy_db.py`):

- `pursuer_pool.json`, `evader_pool.json` — one entry per policy:
  `{id, role, generation, kind ("rl"|"heuristic"), checkpoint_path, metadata}`
- `matchups.json` — per `(pursuer_id, evader_id)` pair. The five **sampling
  metrics** (`win_rate` is always the pursuer capture rate):

  | metric | meaning |
  |---|---|
  | `win_rate` | pursuer capture rate |
  | `avg_time_to_capture` | mean time-to-capture over captured episodes (falls back to the episode cap when there were none) |
  | `avg_distance` | mean inter-drone distance |
  | `avg_motor_effort` | mean accumulated motor effort |
  | `agent_error_rate` | fraction of episodes in which the **training agent itself** crashed or left the arena |

  plus diagnostics that do **not** feed sampling: `episodes`, `eval_role`,
  `avg_episode_len`, `agent_terminal_win_rate`, `agent_forced_error_rate`, and an
  `outcomes` block breaking every episode down by how it ended
  (`capture`, `pursuer_crash`, `evader_crash`, `pursuer_oob`, `evader_oob`,
  `timeout`, `forced_error`).

  The `outcomes` breakdown is what makes a bad matchup *diagnosable*: a zero
  capture rate reads identically whether the pursuer could not catch the evader
  or the evader removed itself before it could be caught. The four numeric
  metrics alone cannot tell those apart.

### Opponent sampling (multi-metric, temperature)

For training `role`, each candidate opponent's five matchup metrics (vs the latest
same-role reference policy) are min-max normalised across candidates, oriented so
a larger value means the opponent is currently **harder** for the training agent,
combined by `metric_weights`, and turned into probabilities with a softmax scaled
by `temperature`:

```
score_i = Σ_m  w_m · hardness_m(i)
p_i     = softmax(score_i / temperature)
```

Orientation flips by role for the four outcome metrics (a hard evader has low
capture rate / long capture time / large distance / high effort; a hard pursuer
is the opposite). **`agent_error_rate` is oriented `+1` for *both* roles**: a
harder opponent forces the training agent into riskier, less controlled flight,
so a higher agent crash/OOB rate flags a harder opponent regardless of which side
is learning.

`agent_error_rate` is not redundant with `win_rate` — it measures a different
axis of difficulty. In the v4 run the final pursuer errored most against
`evader:s5` (0.100) while still capturing it 89% of the time, yet barely errored
against the strongest evader `evader:s8` (0.035) despite capturing it only 5% —
that evader wins by clean evasion rather than by forcing crashes.

Low temperature → focus on the hardest opponents; high temperature → uniform.
Cold start (no reference / no metrics) falls back to uniform.

Defaults (`scripts/selfplay/train_selfplay.yaml`):

```yaml
temperature: 0.9
metric_weights:
  win_rate: 1.0
  avg_time_to_capture: 0.5
  avg_distance: 0.0        # disabled
  avg_motor_effort: 0.25
  agent_error_rate: 1.0
```

A weight of `0.0` leaves the metric recorded but out of the score, which is how
`avg_distance` is currently disabled.

---

## Running

Always use the Isaac-enabled environment:

```bash
source .venv/bin/activate
```

### Standalone training (single default opponent)

Trains the pursuer against the default hover heuristic — useful for smoke tests
and reward/observation tuning:

```bash
cd scripts
python train.py task=PursuitEvasion algo=ppo headless=true \
    task.env.num_envs=256 total_frames=50_000_000
```

Train the evader instead: `task.role=evader`. Note the num-envs override is
`task.env.num_envs=N`.

### Full generational self-play

The orchestrator seeds both pools with heuristics, then alternates sides. Each
stage spawns two isolated worker processes (train, then eval):

```bash
python scripts/selfplay/train_selfplay.py \
    n_stages=8 frames_per_stage=20_000_000 \
    num_envs=256 temperature=1.0
```

  By default, rerunning the command resumes from the persisted policy database.
  Completed stages are detected from their checkpoint names and matchup records;
  with pursuer stage 4 and evader stage 5 present, the next run starts at stage 6
  and trains the pursuer, warm-started from `pursuer_stage4.pt`, against the
  persisted evader population. An incomplete stage is retried without adding a
  duplicate policy entry. Use `fresh_start=true` to clear the pools and begin at
  stage 0 again.

Key orchestrator options (`scripts/selfplay/train_selfplay.yaml`): `db_dir`,
`ckpt_dir`, `work_dir`, `start_role`, `warm_start`, `temperature`,
`metric_weights`, `seed_heuristics`.

### Worker (invoked by the orchestrator; can be run directly)

```bash
# Train one stage vs an injected pool
python scripts/selfplay/run_stage.py mode=train task=PursuitEvasion algo=ppo \
    task.role=pursuer opponent_spec_path=<pool.json> \
    stage_ckpt=<out.pt> stage_out=<train_out.json> total_frames=20_000_000

# Evaluate one policy vs the pool -> per-opponent matchup metrics
python scripts/selfplay/run_stage.py mode=eval task=PursuitEvasion algo=ppo \
    task.role=pursuer eval_policy=<policy.pt> \
    opponent_spec_path=<pool.json> stage_out=<eval_out.json>
```

### Visualising the arena

Run non-headless (e.g. `headless=false` with a small `task.env.num_envs`) to see
the green boundary box drawn per env.

---

## Analysing a finished run

### Win-rate matrix (pursuer generation × evader generation)

`plot_winrate_matrix.py` turns a run's `matchups.json` into a
pursuer-generation × evader-generation heatmap of the pursuer capture rate. It
always writes a PNG next to the database and can additionally log to W&B:

```bash
python scripts/selfplay/plot_winrate_matrix.py \
    --db scripts/selfplay/outputs/<run>/db        # -> <run>/winrate_matrix.png

# also log to W&B (image panel + wandb.Table)
python scripts/selfplay/plot_winrate_matrix.py \
    --db scripts/selfplay/outputs/<run>/db --wandb \
    --run-name winrate-matrix-<run>
```

Options: `--metric` (any numeric matchup key, default `win_rate`), `--out`,
`--project`, `--entity`, `--run-name`.

A healthy population shows a **staircase**: each pursuer generation beats the
evaders it was trained against (lower triangle + diagonal) and loses to newer
evaders (upper triangle). Cells for pairs that were never evaluated render as
`-` — the table is sparse because each new policy is only evaluated against the
pool that existed when it was added.

> `wandb.plot.HeatMap` is not available in current wandb versions, so the heatmap
> is logged as an image panel; the accompanying `wandb.Table` is there to build
> custom charts from.

---

## Logged stats

`return`, `episode_len`, `distance`, `success_rate`, `capture`, `win_rate`,
`forced_error`, `timeout`, `time_to_capture`, `motor_effort`, `crash`,
`opp_crash`, `out_of_bounds`, `opp_out_of_bounds`, `capture_radius`, and the
per-term reward components (`reward_approach`, `reward_precision`,
`reward_distance`, `reward_body_rate`, `reward_action_smoothness`,
`reward_cmd_rate`, `reward_terminal`).

`crash` / `out_of_bounds` are always the **training agent's**; the opponent's are
under `opp_crash` / `opp_out_of_bounds`. `win_rate` counts only full wins, so a
gap between `win_rate` and `capture` + `forced_error` shows how much of the
agent's terminal income came from partial credit.

All terminal stats are latched in `_compute_state_and_obs` (before the stats are
cloned into the observation), otherwise the episode resets first and every
terminal stat logs as zero. The eval worker aggregates these per opponent into
the matchup metrics consumed by the sampler.

---

## Implementation notes / gotchas

* **Episode-boundary controller resets.** The tensordict handed to
  `_pre_sim_step` is the *action* td and carries no useful `done` (it is `False`
  throughout action processing), so a `reset_pid=done` never fires and PID /
  altitude-integrator state leaks across episodes. The env therefore latches its
  own `_just_reset` flag in `_reset_idx` for the scripted opponent, and the
  shared `PIDRateController` transform uses `is_init` (from `InitTracker`) with a
  `done` fallback. Symptom when broken: a stale integral worth ~23% of `g`
  carried into the next episode.
* **Training ≠ deployment rate loop.** Training steps the on-board rate loop at
  the 50 Hz control period; real/CrazySim firmware runs it ~10× faster and tracks
  aggressive setpoints far more crisply. This is why `reward_cmd_rate_weight`
  exists, and why the deploy controller offers `command_rate_lpf_tau`.
