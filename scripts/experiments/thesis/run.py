"""Resumable driver for the thesis experiments (see docs/research_plan.md).

    .venv/bin/python scripts/experiments/thesis/run.py --exp X0

Runs Isaac jobs sequentially; finished jobs (existing output JSON / checkpoint) are skipped.
"""

import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "exps"))

EXPERIMENTS = {
    "X0": "x0_replay",
    "X1": "x1_crossplay",
    "X2": "x2_descriptors",
    "X3": "x3_gap",
    "X4": "x4_oracles",
    "D1": "d_sampling:run_d1",
    "D1b": "d_sampling:run_d1b",
    "D1c": "d_sampling:run_d1c",
    "D2": "d_sampling:run_d2",
    "A1a": "a_adapt:run_a1a",
    "A1b": "a_adapt:run_a1b",
    "F1": "f1_final",
    "A2a": "a2_nfacc:run_a2a",
    "A2b": "a2_nfacc:run_a2b",
    "A3a": "a3_posonly:run_a3a",
    "A3b": "a3_posonly:run_a3b",
    "A3c": "a3_posonly:run_a3c",
    "A3d": "a3d_velest:run_a3d",
    "A3e": "a3e_estimators:run_a3e",
    "A3f": "a3f_specnorm:run_a3f",
    "F2a": "f2_final:run_f2a",
    "F2b": "f2_final:run_f2b",
    "F2c": "f2_final:run_f2c",
    "F3": "f3_sampler_sp:run_f3",
    "F3c": "f3_sampler_sp:run_f3c",
    "F4": "f4_estimator_sp:run_f4",
    "F4c": "f4_estimator_sp:run_f4c",
    "F5": "f5_acc_sp:run_f5",
    "F5c": "f5_acc_sp:run_f5c",
    "F6": "f6_div_sp:run_f6",
    "F6c": "f6_div_sp:run_f6c",
    "F6b": "f6b_div_sp:run_f6b",
    "F6bc": "f6b_div_sp:run_f6bc",
    "F7": "f7_floor_sp:run_f7",
    "F7c": "f7_floor_sp:run_f7c",
    "F8": "f8_roncero_sp:run_f8",
    "F8c": "f8_roncero_sp:run_f8c",
    "F9": "f9_ablation_sp:run_f9",
    "F9c": "f9_ablation_sp:run_f9c",
    "F8d": "f8d_heldout:run_f8d",
    "F10": "f10_gae_sp:run_f10",
    "F10c": "f10_gae_sp:run_f10c",
    "F11": "f11_v2_sp:run_f11",
    "F11c": "f11_v2_sp:run_f11c",
    "F12": "f12_acc_sp:run_f12",
    "F12c": "f12_acc_sp:run_f12c",
    "F13": "f13_thesis_short_sp:run_f13",
    "F13c": "f13_thesis_short_sp:run_f13c",
    "F10s": "f10_gae_sp:run_f10s",
    "F10sc": "f10_gae_sp:run_f10sc",
}

# Registered before the GAE truncation fix became the PPO default (2026-11-11): these keep the old setting
# (common.gae_pin) unless their own overrides set it.
LEGACY_GAE = {"X0", "X1", "X2", "X3", "X4", "D1", "D1b", "D1c", "D2", "A1a", "A1b", "F1", "A2a", "A2b", "A3a", "A3b",
              "A3c", "A3d", "A3e", "A3f", "F2a", "F2b", "F2c", "F3", "F3c", "F4", "F4c", "F5", "F5c", "F6", "F6c",
              "F6b", "F6bc", "F7", "F7c", "F8", "F8c", "F9", "F9c", "F8d", "F10", "F10c"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="tiny frame budgets / few opponents")
    args = ap.parse_args()
    import common
    common.LEGACY_GAE = args.exp in LEGACY_GAE
    from common import log_progress
    log_progress(f"=== start {args.exp} ({' '.join(sys.argv[1:])})")
    mod, _, fn = EXPERIMENTS[args.exp].partition(":")
    getattr(importlib.import_module(mod), fn or "run")(args)
    log_progress(f"=== done {args.exp}")
    if not args.smoke:      # user request (10-28): push code + text results after every finished experiment
        import subprocess
        subprocess.run([os.path.join(HERE, "git_sync.sh"), f"thesis: {args.exp} finished"], check=False)


if __name__ == "__main__":
    main()
