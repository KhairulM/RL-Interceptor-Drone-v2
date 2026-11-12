# Thesis F8 (re-run with the smoothness fix), seed 0: final pursuer and evader (exported 2026-11-12)

| folder | checkpoint | role |
|---|---|---|
| `pursuer/` | `scripts/selfplay/outputs/thesis-F8b-roncero-s0/checkpoints/pursuer_stage24.pt` | pursuer (final, stage 24) |
| `evader/` | `scripts/selfplay/outputs/thesis-F8b-roncero-s0/checkpoints/evader_stage25.pt` | evader (final, stage 25) |

Training: Hedge self-play, 26 stages x 50M frames, Roncero et al. 2026 reward and episode endings with the action-smoothness
term (k_sm) working (fixed 2026-11-10; the earlier `thesis_F8_roncero_s0` artifacts had k_sm = 0), GAE truncation fix,
arena 4 x 4 m, z 0.1-3.0 m, capture radius 0.10 m, Crazyflie model, `pidrate` CTBR at 50 Hz, no domain randomisation,
perfect state.

Export: `export_noisaac.py` (copy in this folder), because both Isaac slots and the RAM were taken by F11. It builds the
same PPO policy from specs (obs 28, action 4), loads the checkpoint strictly, uses `export_policy.extract_deterministic_actor`
and traces it; `metadata.json` comes from `../thesis_F8_roncero_s0/` (official Isaac exporter, same task config: the
observation layout, arena and CTBR parameters did not change). Validation: re-exporting the old F8 checkpoints this way
reproduces the Isaac-exported `thesis_F8_roncero_s0` policies exactly (max difference 0 on 512 random observations).

Run: set `artifact_dir: artifacts/thesis_F8b_roncero_s0/pursuer` (or `evader`) in `config_pursuit_evasion.yaml`, and the
opponent drone's `artifact_dir` to the other folder for policy-vs-policy.

In simulation (RESULTS.md, F8 and F8d): this pursuer neutralises held-out RL evaders (v10) in 0.948 of episodes and the
strong held-out set (final evaders of 16 other campaigns) in 0.858; it is not lurable (pursuer crash 0.010 vs an exploiter
that wants it to crash) and its exploitability (0.631 at 0.10 m) is almost entirely near-miss timeouts (0.000 at 0.15 m).
This evader beats every pursuer of its own campaign (it was trained last).

Caveats for flight tests:
- Roncero's regularisation is light (body rate k_br 0.001, smoothness k_sm 0.002), so commands may still be aggressive:
  start with `controller.command_rate_lpf_tau` 0.02-0.05.
- No domain randomisation or sensor noise in training.
- `notes.trained_against` in metadata.json shows the YAML default opponent, not the self-play pool.
