# F23 seed 100 policy exports

These are deterministic actors from the completed F23 seed-100 self-play campaign:

| Directory | Source checkpoint | Observation |
|---|---|---|
| `pursuer/` | `pursuer_stage24.pt` | 31 values, including the acceleration estimator |
| `evader/` | `evader_stage25.pt` | 28 values |
| `evader_stage23/` | `evader_stage23.pt` | 28 values |

The pursuer bundle includes the F5 `nf_dvel_H10` basis and online RLS settings
(history/horizon 10, forgetting 0.999, delta 0.01, trace-cap ratio 1, reset per episode).
The metadata records the v3 training arena (4 x 4 m, altitude 0.1-2.2 m),
0.1 m capture radius, 750 control steps, and 0.02 s control period.
F23 used uniform self-play sampling, not Hedge.

The export uses the established F14 no-Isaac actor builder and F14 deployment
metadata, adjusted for v3's ceiling and horizon. Checkpoints are loaded strictly,
the traced and reloaded actors are compared against the extracted actors, and
the pursuer's adapter is loaded from the exported basis.

These policies have not been validated in real flight. The original observation,
control, and safety requirements must be verified before deployment.
