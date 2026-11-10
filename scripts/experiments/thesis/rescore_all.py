"""Re-run every closed-loop experiment's analysis with all J at a larger capture radius (post-hoc, eval-only).

1. Mirror experiments/thesis into MIRROR: text files copied, .pt files symlinked, and every eval JSON that has a
   per-episode file replaced by its re-score at RADIUS (rescore.py; exact for the same policies).
2. Point common.OUT at the mirror, block every way to start a job (training, eval, campaign, subprocess, sleep, file
   writes through .pt symlinks), and call each experiment's run function with its original arguments. Finished jobs are
   skipped, so only the analyses run, now on the re-scored evals.
3. Copy every summary / results file that changed into experiments/thesis/R<radius in cm>_rescore/<dir>/.
Usage: env -u PYTHONPATH .venv/bin/python scripts/experiments/thesis/rescore_all.py <mirror dir> [radius] [EXP ...]
"""

import argparse
import filecmp
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "exps")]
import common  # noqa: E402
from rescore import rescore  # noqa: E402

ORIG = common.OUT
# Experiment -> original run.py arguments (driver_log "=== start" lines).
RUNS = [("X1", {}), ("X3", {"seeds": 5}), ("X4", {}), ("D1", {"seeds": 5}), ("D1b", {}), ("D1c", {"seeds": 3}),
        ("D2", {"seeds": 5}), ("F1", {"seeds": 5}), ("A1b", {"seeds": 5}), ("A2b", {"seeds": 5}), ("A3b", {}),
        ("A3c", {}), ("A3d", {}), ("F2a", {}), ("F2c", {}), ("F3c", {}), ("F4c", {}), ("F5c", {}), ("F6c", {}), ("F6bc", {})]


class Blocked(RuntimeError):
    pass


def build_mirror(mirror, radius):
    n = 0
    for root, dirs, files in os.walk(ORIG):
        rel = os.path.relpath(root, ORIG)
        top = rel.split(os.sep)[0]
        if top.startswith("R") and top.endswith("_rescore"):
            continue
        os.makedirs(os.path.join(mirror, rel), exist_ok=True)
        for f in files:
            src, dst = os.path.join(root, f), os.path.join(mirror, rel, f)
            if f.endswith(".lock") or os.path.lexists(dst):
                continue
            if f.endswith(".pt"):
                os.symlink(src, dst)
            elif f.endswith(".json") and os.path.isfile(src[:-len(".json")] + "_episodes.pt"):
                with open(dst, "w") as fh:
                    json.dump(rescore(src, radius), fh, indent=1)
                n += 1
            else:
                shutil.copy2(src, dst)
    return n


def block(mirror):
    def no(*a, **k):
        raise Blocked(f"blocked: would start a job ({a[:1]})")
    common._evaluate = common._train = common.run_stage = common._acquire_isaac_slot = no
    subprocess.run = subprocess.Popen = subprocess.call = subprocess.check_call = subprocess.check_output = no
    time.sleep = no
    import torch
    real_save = torch.save

    def save(obj, f, *a, **k):
        if isinstance(f, (str, os.PathLike)) and (os.path.islink(f) or not os.path.abspath(f).startswith(mirror)):
            raise Blocked(f"blocked: torch.save outside the mirror / through a symlink: {f}")
        return real_save(obj, f, *a, **k)
    torch.save = save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mirror")
    ap.add_argument("radius", nargs="?", type=float, default=0.15)
    ap.add_argument("exps", nargs="*")
    a = ap.parse_args()
    mirror = os.path.abspath(a.mirror)
    if not os.path.isdir(mirror):
        print(f"mirror: {build_mirror(mirror, a.radius)} eval files re-scored at {a.radius} m", flush=True)
    common.OUT = mirror                                  # before any experiment module is imported
    block(mirror)
    from run import EXPERIMENTS
    status = {}
    for exp, kw in RUNS:
        if a.exps and exp not in a.exps:
            continue
        args = argparse.Namespace(exp=exp, seeds=kw.get("seeds", 3), smoke=False)
        mod, _, fn = EXPERIMENTS[exp].partition(":")
        try:
            getattr(importlib.import_module(mod), fn or "run")(args)
            status[exp] = "ok"
        except Blocked as e:
            status[exp] = str(e)
        except Exception as e:                           # noqa: BLE001
            status[exp] = f"error: {type(e).__name__}: {e}"
            traceback.print_exc()
        print(f"[{exp}] {status[exp]}", flush=True)
    out = os.path.join(ORIG, f"R{round(a.radius * 100)}_rescore")
    changed = []
    for root, _, files in os.walk(mirror):
        rel = os.path.relpath(root, mirror)
        if os.sep + "evals" in os.sep + rel or rel.startswith("evals"):
            continue
        for f in files:
            if not (f.endswith(".md") or f.endswith(".json")) or f in ("PROGRESS.md", "RESULTS.md"):
                continue
            if f.endswith(".json") and os.path.lexists(os.path.join(root, f[:-len(".json")] + "_episodes.pt")):
                continue                                 # a re-scored eval, not an analysis output
            m, o = os.path.join(root, f), os.path.join(ORIG, rel, f)
            if os.path.isfile(o) and not os.path.islink(m) and not filecmp.cmp(m, o, shallow=False):
                os.makedirs(os.path.join(out, rel), exist_ok=True)
                shutil.copy2(m, os.path.join(out, rel, f))
                changed.append(os.path.join(rel, f))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "status.json"), "w") as fh:
        json.dump({"radius": a.radius, "status": status, "changed": sorted(changed)}, fh, indent=1)
    print(json.dumps(status, indent=1))
    print(f"{len(changed)} changed summary / results files copied to {out}")


if __name__ == "__main__":
    main()
