"""Export F23 seed 100's final pursuer and last two evaders without Isaac Sim."""

import os
import shutil
import sys

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path.insert(0, os.path.join(REPO, "scripts/deploy/artifacts/thesis_F14_rls_s200"))
from export_noisaac import ADAPTER, BASIS, build, pec  # noqa: E402

CAMPAIGN = os.path.join(REPO, "scripts/selfplay/outputs/thesis-F23-v3-acc-rls-s100")
REFERENCE = os.path.join(REPO, "scripts/deploy/artifacts/thesis_F14_rls_s200")
EXPORTS = (
    ("pursuer", "pursuer", "pursuer_stage24.pt", 31),
    ("evader", "evader", "evader_stage25.pt", 28),
    ("evader_stage23", "evader", "evader_stage23.pt", 28),
)


def export(name, role, checkpoint, obs_dim):
    ckpt = os.path.join(CAMPAIGN, "checkpoints", checkpoint)
    out = os.path.join(HERE, name)
    if os.path.exists(out):
        raise FileExistsError(f"Refusing to overwrite existing export: {out}")
    actor = build(ckpt, obs_dim)
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        traced = torch.jit.trace(actor, torch.zeros(1, obs_dim), check_trace=False)
        probe = torch.randn(512, obs_dim)
        err = (actor(probe) - traced(probe)).abs().max().item()
        if err >= 1e-5:
            raise RuntimeError(f"Trace mismatch for {ckpt}: {err}")

    meta = pec.load_metadata(os.path.join(REFERENCE, role, pec.METADATA_FILENAME))
    if meta.obs.obs_dim != obs_dim or meta.role != role:
        raise RuntimeError(f"Reference metadata has unexpected observation or role: {role}")
    if role == "pursuer" and meta.adapter != ADAPTER:
        raise RuntimeError("Reference metadata has unexpected adapter settings")
    meta.arena_z_max = 2.2
    meta.max_episode_length = 750
    meta.notes = {
        **meta.notes,
        "checkpoint": ckpt,
        "exported_by": "thesis_F23_rls_s100/export_noisaac.py; metadata adapted from F14",
        "task": "PursuitEvasionV3",
        "trained_against": "uniform self-play pool of thesis-F23-v3-acc-rls-s100",
    }

    os.makedirs(out)
    traced.save(os.path.join(out, pec.POLICY_TS_FILENAME))
    if role == "pursuer":
        shutil.copyfile(BASIS, os.path.join(out, pec.ADAPTER_BASIS_FILENAME))
        pec.load_opponent_adapter(out, meta.adapter)
    pec.save_metadata(meta, os.path.join(out, pec.METADATA_FILENAME))
    saved = torch.jit.load(os.path.join(out, pec.POLICY_TS_FILENAME))
    with torch.no_grad():
        saved_err = (actor(probe) - saved(probe)).abs().max().item()
    if saved_err >= 1e-5:
        raise RuntimeError(f"Saved actor mismatch for {ckpt}: {saved_err}")
    print(f"{checkpoint} -> {out}: obs={obs_dim}, max error={saved_err:.3g}", flush=True)


if __name__ == "__main__":
    for item in EXPORTS:
        export(*item)
