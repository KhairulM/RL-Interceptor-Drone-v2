"""Export the F10 seed-0 final evader (stage 11) WITHOUT Isaac Sim, for the switching-evader deploy test.

Same method as ../thesis_F14_rls_s200/export_noisaac.py: build PPOPolicy from hand-made specs, load the checkpoint
strictly, extract the deterministic actor with scripts/deploy/export_policy.py, trace it, and write metadata.json from
the Isaac-exported F8 evader reference (same task values and obs layout, obs 28). F10 differs from F8 only in its
training rules (thesis rules, no GAE fix), which the deploy metadata does not carry.

Usage (repo .venv, no Isaac):  .venv/bin/python scripts/deploy/artifacts/thesis_F10_nofix_s0/export_noisaac.py
"""
import os
import sys

import torch
from omegaconf import OmegaConf
from torchrl.data import Composite, UnboundedContinuous
from torchrl.envs.utils import ExplorationType, set_exploration_type

REPO = "/home/user/Projects/RLInterceptorDrone"
sys.path.insert(0, os.path.join(REPO, "scripts/deploy"))
import pursuit_evasion_common as pec  # noqa: E402
from export_policy import extract_deterministic_actor  # noqa: E402
from omni_drones.learning.ppo.ppo import PPOConfig, PPOPolicy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(REPO, "scripts/selfplay/outputs/thesis-F10-nofix-s0/checkpoints/evader_stage11.pt")
REFERENCE = os.path.join(REPO, "scripts/deploy/artifacts/thesis_F8_roncero_s0/evader")
OBS, ACT = 28, 4


def main():
    def spec(key, dim):
        return Composite({"agents": Composite({key: UnboundedContinuous((1, 1, dim))}, shape=(1,))}, shape=(1,))
    policy = PPOPolicy(OmegaConf.structured(PPOConfig), spec("observation", OBS), spec("action", ACT),
                       spec("reward", 1), device="cpu")
    policy.load_state_dict(torch.load(CKPT, map_location="cpu"))   # strict: any missing / unexpected key raises
    policy.eval()
    actor = extract_deterministic_actor(policy, "ppo")
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        traced = torch.jit.trace(actor, torch.zeros(1, OBS), check_trace=False)
        probe = torch.randn(512, OBS)
        err = (actor(probe) - traced(probe)).abs().max().item()
        assert err < 1e-5, err
    out = os.path.join(HERE, "evader")
    os.makedirs(out, exist_ok=True)
    traced.save(os.path.join(out, pec.POLICY_TS_FILENAME))

    meta = pec.load_metadata(os.path.join(REFERENCE, pec.METADATA_FILENAME))
    meta.artifact_version = pec.PE_ARTIFACT_VERSION
    meta.notes = {**meta.notes, "checkpoint": CKPT,
                  "exported_by": "thesis_F10_nofix_s0/export_noisaac.py (no Isaac); task metadata from " + REFERENCE,
                  "trained_against": "self-play pool of thesis-F10-nofix-s0 (thesis rules, no GAE fix)"}
    assert meta.role == "evader" and meta.obs.expected_obs_dim() == OBS
    pec.save_metadata(meta, os.path.join(out, pec.METADATA_FILENAME))
    print(f"exported {CKPT} -> {out} (obs {OBS}, trace err {err:.1e})")


if __name__ == "__main__":
    main()
