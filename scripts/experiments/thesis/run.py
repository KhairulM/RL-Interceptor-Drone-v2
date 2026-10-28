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
    "F2a": "f2_final:run_f2a",
    "F2b": "f2_final:run_f2b",
    "F2c": "f2_final:run_f2c",
    "F3": "f3_sampler_sp:run_f3",
    "F3c": "f3_sampler_sp:run_f3c",
    "F4": "f4_estimator_sp:run_f4",
    "F4c": "f4_estimator_sp:run_f4c",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="tiny frame budgets / few opponents")
    args = ap.parse_args()
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
