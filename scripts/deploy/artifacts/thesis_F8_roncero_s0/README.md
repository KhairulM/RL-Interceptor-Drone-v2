# Thesis F8, seed 0: final pursuer and evader (exported 2026-11-10)

| folder | checkpoint | role |
|---|---|---|
| `pursuer/` | `scripts/selfplay/outputs/thesis-F8-roncero-s0/checkpoints/pursuer_stage24.pt` | pursuer (final, stage 24) |
| `evader/` | `scripts/selfplay/outputs/thesis-F8-roncero-s0/checkpoints/evader_stage25.pt` | evader (final, stage 25) |

Exported with `scripts/deploy/export_pursuit_evasion_policy.py` (logs: `export_*_log.txt`; TorchScript trace error
0). The campaign saved no training-config snapshot, so the exporter used the live `cfg/task/PursuitEvasion.yaml`;
it is unchanged since before F8 started (11-09 15:13) except the F9 ablation block, which is off by default. The
actor observation code matches the 10-20 deploy-verified version for these settings (obs_dim 28: pos, R (9), body lin
vel, body ang vel, raw body-frame rel position, rel velocity, previous action).

Training: Hedge self-play, 26 stages x 50M frames, Roncero et al. 2026 reward and episode endings
(`task.reward_mode=roncero`), arena 4 x 4 m, z 0.1-3.0 m (metadata carries these limits), capture radius 0.10 m,
Crazyflie model, `pidrate` CTBR at 50 Hz, no domain randomisation, perfect state.

Run: set `artifact_dir: artifacts/thesis_F8_roncero_s0/pursuer` (or `evader`) in `config_pursuit_evasion.yaml`, and
the opponent drone's `artifact_dir` to the other folder for policy-vs-policy.

Caveats for flight tests:
- These come from the original F8 campaign, trained with the k_sm bug. F8 is being re-run with the fix as
  `thesis-F8b-roncero-s{0,100}` (done ~11-11 03:30); export its final policies for a like-for-like Roncero test.
- Roncero's action-smoothness term (k_sm) was 0 in practice during F8 training (computed per physics substep; found
  11-10). The only smoothness pressure was the body-rate penalty k_br = 0.001, so expect aggressive rate commands:
  start with `controller.command_rate_lpf_tau` 0.02-0.05 (see the controller config comment).
- These final policies have no F8c evaluation (F8c evaluated stage 8 of this campaign, and the re-run evaluates the
  F8b campaigns instead). In-campaign (Roncero rules, 0.10 m): this pursuer neutralises its
  campaign's older evaders in 0.93-0.99 of episodes; this evader beats every pursuer stage (pursuer J 0.14-0.29).
- `notes.trained_against` in metadata.json shows the YAML default opponent, not the self-play pool.
