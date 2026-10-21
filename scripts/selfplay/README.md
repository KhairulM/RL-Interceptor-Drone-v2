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
| Optional ESN opponent predictor | `omni_drones/utils/esn.py` |
| Generational driver (orchestrator) | `scripts/selfplay/train_selfplay.py` (+ `.yaml`) |
| Per-stage worker (Isaac process) | `scripts/selfplay/run_stage.py` (+ `.yaml`) |
| Win-rate matrix plot | `scripts/selfplay/plot_winrate_matrix.py` |

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
2. attitude, per `observation.rot_repr`: `matrix` — flattened rotation matrix
  (9, default); `quat` — canonical quaternion with `w ≥ 0` (4); `log` — rotation
  vector `log(R)` (3)
3. body-frame linear velocity (3)
4. body-frame angular velocity (3)
5. opponent relative vector in the agent body frame (3) — the raw relative
   position when `observation.use_rel_dist: true` (default), or its unit heading
   `normalize(rel_dist)` when `false`
6. opponent relative linear velocity in the agent body frame (3)
7. previous action (4) — if `observation.use_previous_action`
8. ESN prediction of the opponent's next relative position + velocity (6) — only
  when `esn.enabled` and this side is listed in `esn.roles` (see
  [ESN opponent predictor](#esn-opponent-predictor))

Total with the default matrix attitude: **24** (or **28** with previous action),
plus 6 with ESN. A time encoding is appended to the critic `state` when
`observation.use_time_encoding` is set.

The frozen opponent uses its own role's observation layout. With the default
`esn.roles: [pursuer]`, a frozen pursuer still receives its ESN prediction when
playing against a training evader, so the roles can have different `obs_dim`.

> Note: the deployment observation in `scripts/deploy/pursuit_evasion_common.py`
> must match `use_rel_dist`, `rot_repr`, and the ESN setting bit-for-bit, or the
> exported policy sees an out-of-distribution input on hardware.

Optional per-component Gaussian sensor noise is added to the **agent's**
observation (never the frozen opponent's) when `observation.include_noise: true`;
stds are configured under `observation.noise` (defaults matched to Intercept).

### Action

`[wx, wy, wz, T]` — desired body rates + normalized collective thrust, tracked by
the PID rate controller.

### Reward (asymmetric dense terms, symmetric terminal)

Terminal outcomes use a single magnitude `W = reward_terminal_weight`:

* **Full win (+W):** pursuer → capture; evader → survived to the timeout.
* **Full loss (−W):** the agent's **own** crash to the ground
  (`z < minimum_altitude`) / NaN position, (evader only) its departure through
  a wall or ceiling, and (pursuer only) the timeout.
* **Neutral (0):** the opponent crashing ends the episode without terminal
  reward for the agent. An evader leaving the arena is also neutral for the
  pursuer; the pursuer may leave and re-enter freely.

| outcome | pursuer | evader |
|---|:--:|:--:|
| capture | **+W** | −W |
| timeout (evader survived) | −W | **+W** |
| pursuer crashes | −W | 0 |
| evader crashes | 0 | −W |
| evader leaves arena | 0 | −W |
| pursuer leaves arena | 0 (episode continues) | 0 (episode continues) |

A capture takes precedence over a simultaneous loss. Every terminal reward is
therefore one of `{−W, 0, +W}`.

Dense terms: the pursuer gets `approach` (distance-closing) + `precision`
(potential `exp(-k·d)`) and, when `use_distance` is set, a per-step distance
penalty (`reward_distance_weight · distance`); the evader gets a per-step
survival reward (`reward_step`, default `0.1`) and an evader-only `bounds`
penalty that ramps up within `bounds_margin` of a wall. The pursuer has no bounds
penalty. Both get
**three complementary smoothness penalties**:

| penalty | on what | why it alone is not enough |
|---|---|---|
| `reward_body_rate_weight` | the drone's **actual** body angular velocity | at the 50 Hz training rate the inner loop never fully realises a saturated command within a step, so this barely fires |
| `reward_action_smoothness_weight` | **step-to-step change** of the CTBR action (jerk) | a *sustained* saturated command has no jerk, so it is ~free |
| `reward_cmd_rate_weight` | **magnitude** of the commanded body rate (`\|tanh action\|` over the 3 rate channels) | closes the gap above — the real / CrazySim firmware rate loop runs ~10× faster, tracks crisp saturated setpoints faithfully, and tumbles |

The third term is the anti-tumble lever: without it a policy can hold ±180 deg/s
setpoints essentially for free in training and then tumble on deployment.
In the current task config, `reward_body_rate_weight` is `0.0` (disabled), while
`reward_action_smoothness_weight` and `reward_cmd_rate_weight` are `0.5` and `0.2`.

Weights: `reward_approach_weight`, `reward_precision_weight`/`_scale`,
`reward_step`, `reward_body_rate_weight`, `reward_action_smoothness_weight`,
`reward_cmd_rate_weight`, `use_distance`/`reward_distance_weight`,
`reward_bounds_weight`, `bounds_margin`, `reward_terminal_weight`.

### Arena, capture, termination

- Confined box (env-local): `arena.half_xy`, `arena.z_min`, `arena.z_max`
  (default 4 m × 4 m, `z ∈ [0.1, 3.0]` m). Drawn as a boundary box when not
  headless (`arena.visualize: true`). Spawns are at least
  `min_spawn_separation` (default 1 m) apart.
- Capture when inter-drone distance ≤ the **curriculum capture radius**.
- Terminate on capture, either drone crashing to the ground or having a NaN
  position, or the evader leaving through a wall/ceiling. The pursuer may leave
  without terminating the episode. Truncate at `env.max_episode_length` (default
  1500 steps): a surviving evader wins and the pursuer loses.

### Capture-radius curriculum

Like Intercept, the capture threshold shrinks linearly during training and is held
fixed for evaluation:

```
radius(step) = max(success_radius_init − success_radius_lr · global_step,
                   success_radius_end)          # training
radius       = success_radius_eval              # eval / not training
```

Logged as `stats.capture_radius`. Current task defaults set the initial, final,
and evaluation radii to `0.1` m and the learning rate to `0`, so it stays fixed
unless overridden.

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
  - `pure_pursuit` — fly at the other drone's current position (no lead); the
    legacy name `pursue` is also accepted
  - `apf` — artificial potential field: braking-limited attractive velocity
    toward the other drone plus its velocity as feed-forward
    (`omni_drones/controllers/apf_controller.py`,
    `omni_drones/controllers/cfg/apf_intercept.yaml`)
  - `flee` — run **horizontally** away from the pursuer at a **held spawn
    altitude** (see below)
  - `fast_flee` — flee at 1.5× speed; `weave` and `zigzag` add lateral
    oscillations
  - switch forms such as `hover>flee@2.0` switch from the first behavior to the
    second after the specified episode time in seconds

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

Frozen RL opponents run only on the observations for environments assigned to
them, using actor-only inference (the critic is not needed to produce actions).
This avoids evaluating each policy across the full environment batch.

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
  | `agent_error_rate` | fraction of episodes in which the evaluated role's agent crashed or left the arena |

  plus diagnostics that do **not** feed sampling: `episodes`, `eval_role`,
  `avg_episode_len`, `agent_terminal_win_rate`, and an
  `outcomes` block breaking every episode down by how it ended
  (`capture`, `pursuer_crash`, `evader_crash`, `pursuer_oob`, `evader_oob`,
  `timeout`).

  The `outcomes` breakdown is what makes a bad matchup *diagnosable*: a zero
  capture rate reads identically whether the pursuer could not catch the evader
  or the evader removed itself before it could be caught. The five sampling
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

At startup, the orchestrator evaluates missing scripted-pursuer ×
scripted-evader matchups. A role's first RL generation samples scripted
opponents uniformly. Later generations sample only opposing RL policies using
the multi-metric softmax; scripted opponents are excluded from both training and
evaluation pools. Each new RL policy is evaluated uniformly against the same
eligible pool used for its training. Scripted policies and historical matchups
remain in the database for diagnostics. Resuming an interrupted first generation
still uses scripted opponents; resuming later generations uses RL opponents only.

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
stage trains one role in an isolated Isaac worker and evaluates the resulting
policy in a second worker against the same eligible opposite-role pool (scripted
for that role's first RL generation, RL-only afterward):

```bash
python scripts/selfplay/train_selfplay.py \
    run_name=<campaign> n_stages=8 frames_per_stage=20_000_000 \
    num_envs=256 temperature=1.0
```

By default, rerunning the command resumes from the persisted database. Completed
stages are detected from policy metadata/checkpoint names and matchup records;
with pursuer stage 4 and evader stage 5 present, the next run starts at stage 6
and trains the pursuer, warm-started from `pursuer_stage4.pt`. An incomplete
stage is retried without adding a duplicate policy entry. Use
`fresh_start=true` to clear that campaign's pools, checkpoints, and stage files.

Current defaults from `scripts/selfplay/train_selfplay.yaml`: `run_name` is
`selfplay_v8_no_forced_error_reward`, `fresh_start=false`, `n_stages=26`,
`num_envs=4096`, `frames_per_stage=500_000_000`, `start_role=pursuer`,
`warm_start=true`, `wandb_mode=online`, and `temperature=0.9`. Outputs go to
`scripts/selfplay/outputs/{run_name}/{db,checkpoints,stages}`; W&B runs are named
`{run_name}-{role}-stage{n}`. `task_overrides` are forwarded to train and eval
workers; use a separate campaign/database when changing observation dimensions,
such as enabling ESN.

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

# Evaluate a scripted pursuer (pure_pursuit | apf) instead of a checkpoint
python scripts/selfplay/run_stage.py mode=eval task=PursuitEvasion algo=ppo \
    task.role=pursuer eval_heuristic=apf \
    opponent_spec_path=<pool.json> stage_out=<eval_out.json>
```

The worker builds the trainable policy from `algo`; frozen RL opponents are
loaded as PPO policies. The training worker logs to W&B unless
`wandb.mode=disabled`.

### Visualising the arena

Run non-headless (e.g. `headless=false` with a small `task.env.num_envs`) to see
the green boundary box drawn per env.

---

## Analysing a finished run

### Win-rate matrix (pursuer generation × evader generation)

`plot_winrate_matrix.py` turns a run's `matchups.json` into a
pursuer-generation × evader-generation heatmap of pursuer capture rate for RL
policies (scripted heuristics are excluded). It always writes a PNG next to the
database and can additionally log an image, a `wandb.Table`, and an interactive
W&B heatmap when supported:

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

> If the installed W&B version cannot create `wandb.plot.HeatMap`, the plotter
> still logs the image and table.

---

## Logged stats

`return`, `episode_len`, `distance`, `success_rate`, `capture`, `win_rate`,
`timeout`, `time_to_capture`, `motor_effort`, `crash`,
`opp_crash`, `out_of_bounds`, `opp_out_of_bounds`, `capture_radius`, and the
per-term reward components (`reward_approach`, `reward_precision`,
`reward_distance`, `reward_body_rate`, `reward_action_smoothness`,
`reward_cmd_rate`, `reward_terminal`), plus `esn_pred_err` when ESN is enabled.

`crash` / `out_of_bounds` are always the **training agent's**; the opponent's are
under `opp_crash` / `opp_out_of_bounds`. `stats.win_rate` counts role-specific
full wins: for pursuer training it matches capture; for evader training it
counts episodes survived to timeout. The matchup database's `win_rate` is always
the pursuer capture rate.

All terminal stats are latched in `_compute_state_and_obs` (before the stats are
cloned into the observation), otherwise the episode resets first and every
terminal stat logs as zero. The eval worker aggregates these per opponent into
the matchup metrics consumed by the sampler.

---

## ESN opponent predictor

Optional ESN-OA support is controlled by `task.esn.enabled` (default `false`)
and `task.esn.roles` (default `[pursuer]`). For each enabled role, a batched
echo-state reservoir predicts the opponent's next relative position and
velocity (6 values) and appends them to that role's observation. The frozen
opponent uses the observation size for its own role, so ESN policies must only
be mixed with checkpoints trained using the same role-specific observation
layout.

`task.esn.readout` supports `per_env` (episode-reset online RLS), `shared`
(central ridge readout fit across environments during training), and
`shared_adapt` (shared initialization followed by per-environment RLS). The
default is `shared`. Shared readouts are saved beside each policy checkpoint as
`<checkpoint>_esn.pt`; the worker warm-starts from the initial/evaluation
checkpoint sidecar, and frozen RL opponents require their own sidecar.

Enable ESN through the orchestrator override, using a separate campaign from
non-ESN policies:

```bash
python scripts/selfplay/train_selfplay.py run_name=selfplay_esn \
  'task_overrides=[task.esn.enabled=true]'
```

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
* **Time accounting uses `sim.dt` without substeps.** With current `sim.dt: 0.01`
  and `sim.substeps: 2`, an environment step spans 0.02 s, but capture-time
  stats, the no-capture matchup fallback, and circular heuristic phase use
  `progress_buf * sim.dt`. Reported times are half wall-clock time, and the
  circular heuristic runs at half its configured angular rate.
