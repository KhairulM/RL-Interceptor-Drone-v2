# Flight session 2026-11-26: findings

Generated from `scripts/deploy/logs/flights/2026-11-26_*` with
`python scripts/deploy/analyze_flights.py scripts/deploy/logs/flights --date 2026-11-26 --out scripts/deploy/logs/analysis/2026-11-26`.
Per-flight numbers: `summary.csv`. Per-flight plots: `<flight>.png` (top view in arena coordinates, altitude,
distance/closing speed, tilt/speed; orange line = evader policy switch).
Operator review is in `scripts/deploy/logs/flights/annotations.json` (columns `reviewed_outcome`, `excluded`,
`review_note` in `summary.csv`); the raw logs are unchanged.

Pursuer in every flight: F14 RLS seed 200, stage 24 (trained with a 3.0 m ceiling, pursuer unconfined).
Capture radius 0.11 m. Arena origin (0, 1, 0). The geofence uses the pursuer's metadata: |x|, |y| <= 2.0 m and
z <= 3.0 m, plus a 0.1 m margin.

## Flights

| time | condition | outcome | logged s | min distance (m) |
|---|---|---|---|---|
| 15:07:49 | P: scripted APF | **capture** at 3.49 s | 3.49 | 0.125 (log), 0.109 (controller) |
| 15:12:39 | O23: F22 evader stage 23 | safety stop: pursuer mocap stale | 5.33 | 0.558 |
| ~~15:45:56~~ | O23 | *excluded (disregarded by the operator)*; logged: pursuer crash | 8.83 | 0.695 |
| 15:54:10 | O25: F22 evader final | safety stop: pursuer mocap stale | 6.55 | 0.594 |
| 15:56:18 | O25 | safety stop: pursuer mocap stale | 4.21 | 0.606 |
| 15:59:54 | O25 | evader about to leave the geofence | 3.71 | 0.619 |
| 16:04:50 | S2: F10 -> F8b @1 s | safety stop: pursuer mocap stale | 1.93 | 0.687 |
| 16:06:49 | S2 @1 s | safety stop: pursuer mocap stale | 1.75 | 0.785 |
| 16:12:04 | S1: F8b -> F10 @1 s | pursuer crash (tilt 62 deg) | 0.75 | 1.908 |
| 19:38:07 | S1 @1 s | pursuer crash (tilt 66 deg) | 0.49 | 1.970 |
| 19:42:03 | S2: F10 -> F8b @3 s | safety stop: evader state stale | 3.27 | 0.354 |
| 19:45:55 | S2 @3 s | safety stop: pursuer mocap stale | 1.75 | 0.834 |
| 19:50:36 | S2 @3 s | **capture (operator-observed contact)**; logged: evader crash (body rate 914 deg/s) | 4.01 | 0.171 |

Totals over the 12 counted flights (15:45 excluded):
- Captures: 2. One against the scripted APF evader (15:07); one against a learned evader (19:50, F10 -> F8b
  switch, 1.0 s after the switch, recorded as a capture from the operator's observation of contact).
- Stopped by a safety check, not a game outcome: 6 pursuer mocap stale, 1 evader state stale.
- Pursuer crashes (tilt): 2, both S1.
- Evader leaving the arena: 1.

Only 2 flights ran past an evader switch (19:42 and 19:50); no S1 flight reached its switch. Condition A
(AMSPBH, F8b evader) was not flown. The control loop ran at 50 Hz throughout (median tick 0.020 s; worst 0.034 s),
apart from one 4.2 s gap in the excluded 15:45 flight.

## Findings

1. **The pursuer loses mocap tracking above about 2.2 m. This was the most common way flights ended.** All 6
   "pursuer mocap stale" stops happened with the pursuer at 2.22-2.47 m, and each flight ended at its highest point.
   No flight that kept the pursuer below 2.2 m lost pursuer tracking. The F14 pursuer (trained with a 3.0 m
   ceiling and allowed to leave) climbs above the evader in most flights (max z 2.2-2.5 m in 8 of 12). The
   geofence ceiling is 3.1 m, so it never caught this. The usable tracking volume therefore looks like z <= ~2.2 m,
   which matches the 2.2 m ceiling of F23/F24. Consider lowering the geofence ceiling (or the pursuer's geofence
   margin) for F14/F22 policies.
2. **S1 (F8b evader first): the pursuer rolls over within 0.5-0.75 s, both times** (cause analysed in "Pursuer crash analysis" below). It is the same motion each
   time: roll goes negative (-40 to -49 deg), pitch goes negative, yaw rises, tilt passes 30 deg at 0.45-0.69 s,
   and the 60 deg tilt stop fires at 1.1-1.4 m altitude. The distance barely changes (2.0 -> 1.9 m), and the F8b
   evader is still slow (<1.3 m/s). This is the pursuer's own attitude runaway, not a reaction to a close evader.
   It did not happen in the first 0.75 s of any other counted flight.
3. **19:50:36 is a capture by contact, not an evader crash.** The operator saw the pursuer hit the evader. The
   log is consistent: under F8b control after the switch, the pursuer closed from 0.59 to 0.17 m, finishing by
   descending onto the evader from above. At the last tick it was 0.08 m away horizontally (less than one
   Crazyflie's width) and 0.15 m above, coming down at 0.28 m/s. The evader's 914 deg/s body-rate stop ended the
   run before the 0.11 m centre-distance capture check could fire. The log alone cannot separate contact from
   the pursuer's downwash as the cause of the spike. Implication: the centre-distance check can miss a capture
   when the pursuer comes in from above; contact (or the evader's emergency while the pursuer is within about
   0.2 m) is a candidate capture criterion for future flights.
4. **19:42:03:** the F10 evader dived at -1.7 m/s from 1.64 m to 0.36 m around the switch (the pursuer was 0.35 m
   away at 2.5 s), and the flight ended on stale evader state at 0.32 m.
5. Capture distance (15:07): the controller reported 0.109 m. The closest logged sample is 0.125 m (the log uses the
   on-board state estimate; the capture check likely ran on a different sample).
6. Metadata: `artifacts/thesis_F22_z2/evader_stage23/metadata.json` says z_max 3.0, but F22 trained with 2.0
   (the final `thesis_F22_z2/evader` says 2.0). The controller uses only the pursuer's arena metadata (geofence,
   scripted evader), so this did not change behaviour, but the artifact should be re-exported.

## Pursuer crash analysis (S1: 16:12:04 and 19:38:07)

**Method.** The flight CSVs do not record commands (`log_commands` only printed them to the flight machine's
console). So the deployed pursuer (same artifact, RLS adapter, previous action, 0.03 s command low-pass) was
re-run open-loop on the logged states. Body rates were rebuilt from consecutive logged attitudes. This gives an
approximation of the rate and thrust commands at each tick. Check: in all 10 counted flights without a pursuer
crash, the measured body rates follow the replayed commands (correlation 0.73-0.98 per axis, 0-3 ticks of lag),
so the replay and its sign conventions are sound. Scripts: `replay_pursuer.py`, `tracking.py`, `early.py`,
`t2w.py` and `err_vs_thrust.py` in the session scratchpad (not in the repo).

**What happened.** In both S1 flights the drone stopped following its rate commands within about 0.1-0.3 s of
policy hand-over:

| 19:38:07, t (s) | commanded roll / pitch / yaw (deg/s) | measured roll / pitch / yaw (deg/s) | thrust command |
|---|---|---|---|
| 0.00-0.25 (mean) | -12 / +31 / -12 | -36 / +5 / +18 | 0.97 |
| 0.25-0.50 (mean) | +35 / +50 / +20 | -151 / -150 / +33 | 0.92 |
| 0.48 | +66 / +63 / +96 | -345 / -374 / +37 | 0.85 |

The policy kept commanding a *positive* roll and pitch rate (it was trying to correct), but the drone rolled and
pitched the opposite way at an increasing rate until the 60 deg tilt stop. Over the whole flight, commanded vs
measured roll correlates at -0.83 (16:12) and -0.48 (19:38), against +0.50 to +0.96 in every other flight.
16:12:04 is the same, about 0.2 s later. **This is a loss of attitude control, not a manoeuvre the policy chose.**

**Most likely cause: motor saturation at near-full collective thrust.**
- The real pursuer reaches T/W about 1.45 at full command (fit over all counted flights). Hovering therefore
  takes about 69% collective (about 45,000 PWM). Above about 0.9 the achieved lift flattens (1.23 g measured vs
  about 1.38 g from the linear fit), consistent with the motors running out of range.
- The policy opens every flight by climbing at 0.95-0.97 collective (about 63,000 PWM of 65,535). That leaves
  almost no headroom for the differential motor thrust that roll, pitch and yaw need.
- Rate-tracking error rises steeply with commanded thrust in every flight (tilt < 30 deg, S1 excluded):

  | commanded thrust | <0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-1.0 |
  |---|---|---|---|---|---|
  | mean roll / pitch / yaw error (deg/s) | 10 / 8 / 10 | 9 / 8 / 13 | 11 / 11 / 14 | 20 / 17 / 14 | 27 / 21 / 22 |

  Yaw is the weakest axis. In most flights the first 0.25 s already shows +3 to +28 deg/s of measured yaw rate
  against a -11 to -17 deg/s command.
- The S1 flights held near-full thrust longest. In 0.25-0.5 s their mean command was 0.92-0.94 and was above
  0.95 for 38% of ticks; every other flight had dropped to 0.82-0.93, with 0-23% of ticks above 0.95. They
  were the flights in which the tracking error grew instead of recovering. Why the policy kept full thrust
  longer against the F8b evader is not clear from two flights.
- Both crashes diverged in the same direction (roll negative, pitch negative, yaw positive). Saturation exposes
  any airframe asymmetry, so a weaker motor or propeller on that pursuer is plausible.

**Why the simulator did not show it (corrected).** Training does model saturation. The sim uses a Crazyflie-style
rate PID, the legacy motor mixing (collective +/- roll/2 +/- pitch/2 +/- yaw per motor), per-motor clipping to
[0, 1], and the same T/W (0.12 N per motor x 4 / (0.034 kg x g) = 1.44; domain randomization 0.85-1.15x).
`max_thrust_ratio` is 1.0 (the asset comment says it was originally 0.9). What the sim does not have: (a) four
identical motors, so clipping weakens torque but never biases it toward one side; a real weak motor or prop does;
(b) the rate PID runs once per 50 Hz control step there, against about 500 Hz in the firmware, so saturation
transients and integrator wind-up behave differently; (c) the real firmware's mixer and motor response may differ
(firmware version not checked). The policy therefore learned that near-full collective is safe, under sim
conditions in which it is.

**What would confirm or fix it.**
1. Confirm: log the motor outputs (`motor.m1`-`m4`) and the rate setpoints/gyro on the drone next session, and
   save the controller console (it already prints the commands each tick).
2. Check the pursuer's motors and props (the identical divergence direction).
3. Deploy-side mitigation: clip the collective to about 0.85-0.90 (it can still climb at T/W about 1.2-1.3), and
   check whether the firmware's motor mixer keeps attitude priority over thrust when the motors saturate.
4. Training-side mitigation: train with `max_thrust_ratio` about 0.85-0.9 so the policy never uses the top of
   the motor range, and/or add per-motor thrust asymmetry to domain randomization.
