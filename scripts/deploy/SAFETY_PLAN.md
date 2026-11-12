# Safety plan: `pursuit_evasion_controller.py` (2026-11-12)

Goal: when the mocap, the state estimate, the radio link, the policy or the host goes wrong, both Crazyflies end in a
safe state (hovering, landed or motors off) instead of flying a bad command. Part B lists bugs found in the current
code; several of them are safety problems on their own.

## Status (2026-11-12)

Implemented: `safety.py` (SafetyMonitor: rows 1-4, 6-11 and 15 of A2; MocapGate), the high-severity bugs B1-B4 and
B10, the `safety:` config section, simulator fault injection (`safety.inject`). Not yet: battery (row 12), watchdog
thread (row 13), second Ctrl+C = emergency stop (row 14), EKF-vs-mocap residual and periodic full-pose updates
(rows 4-5), trajectory logging, bugs B5 partly (gate only), B6-B9 partly (B7, B8 via the monitor), B11-B17.
Tests: `tests/test_deploy_safety.py` (14) and `tests/test_deploy_shutdown.py` (3, with ROS / cflib stubbed).

CrazySim (cf2x_L250, F8b seed-0 pursuer vs evader, policy mode, mocap off, fault injected 2 s into the chase):

| scenario | first safety event | response | landing |
|---|---|---|---|
| none (policies only) | pursuer body rate 811 deg/s (policy tumbles ~2 s in) | EMERGENCY stop on the pursuer | evader landed |
| stale_opponent | evader state stale 0.12 s | HOLD 1 s | both landings started in the same second |
| nan_action | non-finite policy action | HOLD 1 s | both, same second |
| position_jump (+3 m) | pursuer outside the geofence | LAND | both, same second |
| tilt (90 deg) | pursuer tilt 90 deg | EMERGENCY stop on the pursuer | evader landed |

Two findings from the runs: (1) the F8b policies do not fly stably in CrazySim: without injected faults the pursuer
tumbles (tilt > 60 deg or body rate > 720 deg/s) or falls (-2.7 to -4.0 m/s at 0.6 m) 2-8 s into the chase, with and
without the 0.03 s rate filter, so hardware flights need a fix first (sim-to-real, not the safety layer); (2) the
fast-descent rule was changed from EMERGENCY to HOLD: cutting the motors at 0.6 m while falling makes the impact
harder, a position setpoint makes the firmware brake.

---

## Part A: planned safety features

### A1. Design

- A new module **`safety.py`** with a `SafetyMonitor` class: pure numpy, no cflib / ROS / torch, unit-testable. Every
  control tick the controller passes it both drones' states plus mocap health, and gets back one verdict:
  `OK`, `WARN`, `HOLD`, `LAND` or `EMERGENCY`, with a reason string.
- **Graded responses, always applied to both drones** (a fault on one drone must also stop the other from chasing or
  fleeing into it):

| Level | Action | Needs a trustworthy state estimate? |
|---|---|---|
| WARN | log only; keep flying | - |
| HOLD | stop the policy; position setpoint at the drone's current EKF position for `hold_s`, then LAND | yes |
| LAND | high-level `land()` on both drones **at the same time** | yes (onboard) |
| EMERGENCY | supervisor emergency stop (motors off) | no |

- Faults latch: once triggered the run does not resume. A single `safety:` config section holds every threshold, so
  they can be tuned in CrazySim first and tightened for hardware.

### A2. Failure modes, detection and response

| # | Failure | How it is detected (default threshold) | Response |
|---|---|---|---|
| 1 | Mocap frames stop for a rigid body (occlusion, Motive / network down) | host-side time since the last valid frame for that body > `mocap_timeout` (0.2 s) | HOLD (the EKF still holds position briefly), LAND if > 0.5 s |
| 2 | Mocap position jumps (reflection, mislabelled marker) | new frame vs the last accepted one: `|dp| > v_max * dt + jump_tol` (5 m/s, 0.05 m) | drop the frame (not forwarded to the EKF); HOLD after `max_rejects` in a row (5) |
| 3 | Rigid bodies swapped (pursuer and evader exchanged in Motive) | a body's new position is within 0.05 m of the **other** body's last position and far from its own | drop the frames; LAND both |
| 4 | Mocap gives a plausible but wrong pose, or the EKF diverges | residual between the latest mocap position and the EKF position > `ekf_residual` (0.15 m) for > 0.2 s; Kalman variances (`kalman.varPX/PY/PZ`) above a limit | LAND (EMERGENCY if the drone is also tilted or falling) |
| 5 | Wrong yaw after alignment (orientation sent only in the first 2 s) | EKF yaw vs mocap yaw > 15 deg | WARN; LAND above 30 deg. Also send the full pose (`send_extpose`) at a low rate (e.g. 5 Hz) instead of position only |
| 6 | Telemetry from a drone stops (radio dropouts) | per log block (position, attitude, gyro) time since the last packet > `telemetry_timeout` (0.1 s), checked for **both** drones | HOLD; LAND if > 0.3 s. If the link is lost the firmware levels after 0.5 s and cuts motors after 2 s by itself; document this |
| 7 | NaN / Inf in a state or a policy output | `np.isfinite` on the observation and the action | HOLD (EMERGENCY if the state itself is non-finite) |
| 8 | Tumbling or extreme attitude | tilt from the quaternion > `max_tilt` (60 deg), or body rate > `max_body_rate` (720 deg/s) | EMERGENCY |
| 9 | Flying out of the space (either drone; the training rule that only the evader is confined is separate) | hard geofence = arena + `geofence_margin` (0.3 m) incl. ceiling; predictive check `p + v * t_stop` (0.3 s) | HOLD at the boundary, LAND if outside the hard fence |
| 10 | Too low / hitting the floor | z < `min_altitude` (0.10 m) during the chase, or vertical speed < -`max_descent` (2.5 m/s) below 0.6 m | LAND below the threshold; HOLD (firmware brakes) if falling faster than the limit near the floor. Changed from EMERGENCY after the CrazySim test: motors off at 0.6 m makes the impact harder |
| 11 | Drones about to collide | centre distance < `min_separation` (0.20 m on hardware; the 0.10 m training capture radius means physical contact) | treat as capture: HOLD both, then LAND |
| 12 | Battery low | log `pm.vbat`; pre-flight >= 3.8 V; in flight < 3.3 V for 1 s | pre-flight: refuse to take off; in flight: LAND |
| 13 | Control loop stalls (host load, GC, blocked call) | separate watchdog thread: time since the last setpoint was sent > `loop_stall` (0.1 s) | watchdog sends HOLD on both drones; LAND if the loop does not recover in 0.5 s |
| 14 | Operator abort | Ctrl+C (first press) | LAND both concurrently; second press: EMERGENCY |
| 15 | Bad start | pre-flight checks before arming: both links up within a timeout, mocap frames arriving for both bodies, estimator converged, both drones inside the arena (no minimum separation: removed 11-12, user), battery OK | refuse to arm or take off (today these are only warnings) |

### A3. Code changes

1. **`safety.py` (new):** `SafetyConfig` (dataclass from the `safety:` YAML section), `SafetyMonitor.check(own, opp,
   mocap_health, action, now) -> Verdict`, `MocapGate` (per-body jump / swap / staleness gate, used by `mocap.py`).
2. **`mocap.py`:** run every frame through `MocapGate` before forwarding it to the EKF; keep per-body health (last valid
   time, frames accepted / rejected, last accepted position); send the full pose at a low rate instead of position
   only; read `_first_send_stamp` under the lock.
3. **`drone.py`:** per-block timestamps (position, attitude, gyro); log `pm.vbat` and the Kalman variances; a
   `connect(timeout)` that fails instead of waiting forever; `arm()` that can fail (strict mode) when mocap alignment
   or estimator convergence fails; `hold(pos)`, `land_async()` (start landing, return immediately) and
   `emergency_stop()`.
4. **`pursuit_evasion_controller.py`:** pre-flight checks; call `SafetyMonitor` every tick for both drones; apply the
   verdict to both drones; watchdog thread; shutdown that lands both drones concurrently and keeps sending setpoints
   to the drone still under low-level control until its landing starts.
5. **Config:** a `safety:` section with the defaults above, plus `strict_preflight: true` for hardware (false allowed
   in CrazySim where there is no mocap).
6. **Logging:** every safety event with time, level, reason and both states; implement the `trajectory:` logging the
   config already advertises (per-tick states, observation, action, command, safety level) for post-flight analysis.

### A4. Testing (before any hardware flight)

1. **Unit tests** (`tests/test_deploy_safety.py`, no cflib): each detector with synthetic states (jump, swap, stale,
   NaN, tilt, geofence, separation, low battery) and the verdict mapping; shutdown ordering with fake drones that record
   call times (both landings start within 0.1 s).
2. **CrazySim fault injection:** a `--inject` option that, during a run, freezes or jumps the opponent's state, injects
   a NaN, stalls the loop for 0.3 s, or drives a scripted opponent out of the fence; check each response visually and
   in the log. Mocap faults are tested by feeding synthetic frames into `MocapReceiver._on_rigid_body`.
3. **Hardware, staged:** (a) props off: pre-flight checks and mocap gating with the drones carried by hand (swap the
   rigid bodies, block markers); (b) one drone, scripted opponent, low speed limits; (c) both drones with policy mode.

### A5. Open questions for you

- Hardware separation stop: 0.20 m, or keep the 0.10 m training capture radius (contact likely)?
- On a single-drone fault, should the healthy drone HOLD or LAND?
- Is a safety net / soft floor available, which changes how aggressive the EMERGENCY (motors off) rule should be?

---

## Part B: bugs found in the current code

Severity: **high** = can cause a crash or uncontrolled flight; **medium** = wrong behaviour or a silent safety gap;
**low** = usability / clarity.

| # | Sev. | Where | Bug | Evidence | Fix |
|---|---|---|---|---|---|
| B1 | high | `shutdown()` | Drones are landed one after the other; `land()` sleeps `duration + 1` (4 s). In `policy` / `cf` mode the opponent was flying on low-level CTBR setpoints and receives nothing during those 4 s, so the firmware levels it after 0.5 s and **cuts its motors after 2 s** (it falls from ~1 m). | `drone.land()` sleeps; firmware `supervisor.c` COMMANDER_WDT_TIMEOUT_STABILIZE 500 ms / SHUTDOWN 2000 ms | start both landings first, then wait (`land_async`) |
| B2 | high | `run()` state checks | The staleness check covers only the agent's own state; the opponent's `stamp` is never checked. In `policy` mode the opponent drone keeps receiving commands computed from a frozen state of itself. | only `own_state.stamp` is compared with `state_timeout` (line 570) | check both drones (A2 row 6) |
| B3 | high | `drone.StateBuffer` + `mocap.py` | "State timeout" measures log-packet arrival from the Crazyflie, not mocap freshness. If OptiTrack stops, the EKF keeps streaming a drifting estimate with fresh timestamps, so the controller never notices. | stamps set in the log callbacks (`update_pos_vel`); mocap frames are forwarded with no health record | mocap health (A2 rows 1-4) |
| B4 | high | `drone.arm()` | Arming continues after "mocap orientation not aligned" or "Kalman filter not converged" (warnings only), so the policy can take off on an unconverged estimate. | `arm()` logs warnings and proceeds | strict pre-flight (A2 row 15) |
| B5 | medium | `mocap._on_rigid_body` | Every frame flagged `tracking_valid` is forwarded to the EKF unchecked (no jump or swap gate). | no gating before `send_mocap_pos` | `MocapGate` (A2 rows 2-3) |
| B6 | medium | `mocap._on_rigid_body` | Orientation is sent only for the first `orientation_align_time` (2 s) after the **first frame**, which usually arrives while the drone sits on the ground before arming. Yaw is then IMU-only for the whole flight and can drift; body-frame observations rotate with it. | `first_send_stamp` set on the first frame | low-rate full-pose updates (A2 row 5) |
| B7 | medium | `run()` | No check on attitude, body rate, NaN state or NaN action: a tumbling drone or a NaN observation keeps being commanded (a NaN thrust raises in `int(np.clip(nan))` and aborts mid-flight; NaN rates are sent as-is). | `send_ctbr` / `_build_command` have no finite checks | A2 rows 7-8 |
| B8 | medium | `run()` | Only the evader is fenced (training semantics). On hardware the pursuer can fly into the walls or the net. | `_outside_arena(evader_state.pos)` only | hardware geofence for both (A2 row 9) |
| B9 | medium | `run()` | Capture stops at 0.10 m centre distance: on real Crazyflies that is contact, so the drones collide before the stop. | `stop_on_capture` with the trained radius | `min_separation` (A2 row 11) |
| B10 | medium | `drone.connect()` | Waits forever if the link never comes up (e.g. a wrong URI or port). | `while not self.connected: time.sleep(0.1)` | timeout and error |
| B11 | medium | `shutdown()` in `cf` mode | The controller lands and disarms the opponent drone even though another process commands it in `cf` mode. | `shutdown()` loops over both drones | skip the opponent in `cf` mode, or make it configurable |
| B12 | low | `run()` | When a state is `None` the loop sleeps without advancing `next_tick`, so the next tick counts as an overrun and logs a misleading warning; no setpoint is sent meanwhile (the firmware levels after 0.5 s). | `continue` branch at line 565 | send a hold setpoint and keep the schedule |
| B13 | low | `drone.StateBuffer` | The gyro has no timestamp, so a stale gyro (body rates are in the observation) is never detected. | `update_gyro_deg` sets no stamp | per-block stamps |
| B14 | low | config | `pursuer/evader.state_timeout` and `min_altitude` in the drone sections are loaded into `DroneConfig` but never used; the controller uses `controller.state_timeout` / `min_altitude`. Editing the drone sections has no effect. | grep: only the controller-level values are read | remove or use them |
| B15 | low | config | `trajectory: save` is advertised but nothing in the controller implements trajectory logging. | no reference to `trajectory` in the controller | implement (A3 step 6) or remove |
| B16 | low | `mocap._on_rigid_body` | `_first_send_stamp` is read and written outside the lock (the rest of the per-body state is locked). | lines 163-166 | move under the lock |
| B17 | to check | config vs CrazySim | The config uses UDP ports 19850 / 19851; `sitl_multiagent_text.sh` starts the firmware on 19950 / 19951. An earlier CrazySim test stalled in the cflib handshake. Not verified which port cflib must use. | config lines 71 / 84; launcher | verify in CrazySim |

Not bugs, but worth knowing: on a radio link loss the host cannot do anything; the firmware itself levels the drone
after 0.5 s without setpoints and stops the motors after 2 s. Landing commands (`land()`) use the onboard estimator, so
they are only safe while the EKF is healthy; with a bad estimate the only safe action is a short hold followed by an
emergency stop near the floor.
