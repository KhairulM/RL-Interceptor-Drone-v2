"""Export the F14 Stage E RLS seed-200 pursuer (stage 24) and its campaign evader (stage 25) WITHOUT Isaac Sim.

Same approach as ../thesis_F8b_roncero_s0/export_noisaac.py (2026-11: the GPU, both Isaac slots and the RAM are
taken by F16): build PPOPolicy from hand-made specs, load each checkpoint strictly, extract the deterministic actor
with scripts/deploy/export_policy.py, trace it, and write metadata.json. The task values (arena, ctbr, dt, episode
length, obs flags) are identical to F8, so the metadata starts from the Isaac-exported F8 reference; the pursuer adds
the opponent adapter (obs 28 + 3 = 31) and its learned basis, adapter_basis.pt.

Adapter settings are the F14 RLS arm's task.adapt: kind=nf, target=dvel, history=10, horizon=10, delta=0.01,
forgetting=0.999, and the task defaults for the rest.

Usage (repo .venv, no Isaac):  .venv/bin/python scripts/deploy/artifacts/thesis_F14_rls_s200/export_noisaac.py
"""
import os
import shutil
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
CAMPAIGN = os.path.join(REPO, "scripts/selfplay/outputs/thesis-F14-v2d-acc-rls-26-s200")
REFERENCE = os.path.join(REPO, "scripts/deploy/artifacts/thesis_F8b_roncero_s0")
BASIS = os.path.join(REPO, "experiments/thesis/F5_acc_sp/estimator/models/nf_dvel_H10.pt")
ACT = 4

ADAPTER = pec.OpponentAdapterConfig(
    kind="nf", target="dvel", z_mode="vel", history=10, horizon=10,
    forgetting=0.999, delta=0.01, p_max_ratio=1.0, freeze=False, reset_on_episode=True,
)
EXPORTS = [
    # role, checkpoint, obs_dim
    ("pursuer", os.path.join(CAMPAIGN, "checkpoints/pursuer_stage24.pt"), 28 + 3),
    ("evader", os.path.join(CAMPAIGN, "checkpoints/evader_stage25.pt"), 28),
]


def build(ckpt, obs_dim):
    def spec(key, dim):
        return Composite({"agents": Composite({key: UnboundedContinuous((1, 1, dim))}, shape=(1,))}, shape=(1,))
    policy = PPOPolicy(OmegaConf.structured(PPOConfig), spec("observation", obs_dim), spec("action", ACT),
                       spec("reward", 1), device="cpu")
    policy.load_state_dict(torch.load(ckpt, map_location="cpu"))   # strict: any missing / unexpected key raises
    policy.eval()
    return extract_deterministic_actor(policy, "ppo")


def export(role, ckpt, obs_dim):
    out = os.path.join(HERE, role)
    actor = build(ckpt, obs_dim)
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        traced = torch.jit.trace(actor, torch.zeros(1, obs_dim), check_trace=False)
        probe = torch.randn(512, obs_dim)
        err = (actor(probe) - traced(probe)).abs().max().item()
        assert err < 1e-5, err
    os.makedirs(out, exist_ok=True)
    traced.save(os.path.join(out, pec.POLICY_TS_FILENAME))

    meta = pec.load_metadata(os.path.join(REFERENCE, role, pec.METADATA_FILENAME))
    meta.artifact_version = pec.PE_ARTIFACT_VERSION
    meta.obs.obs_dim = obs_dim
    meta.notes = {
        **meta.notes,
        "checkpoint": ckpt,
        "exported_by": "thesis_F14_rls_s200/export_noisaac.py (no Isaac); task metadata from " + REFERENCE,
        "trained_against": "self-play pool of thesis-F14-v2d-acc-rls-26-s200",
    }
    if role == "pursuer":
        meta.obs.use_adapter = True
        meta.adapter = ADAPTER
        shutil.copyfile(BASIS, os.path.join(out, pec.ADAPTER_BASIS_FILENAME))
        pec.load_opponent_adapter(out, ADAPTER)       # raises if the basis's history/horizon/target differ
    assert meta.obs.expected_obs_dim() == obs_dim, (meta.obs.expected_obs_dim(), obs_dim)
    meta_path = os.path.join(out, pec.METADATA_FILENAME)
    pec.save_metadata(meta, meta_path)
    assert pec.load_metadata(meta_path).role == role
    print(f"exported {ckpt} -> {out} (obs {obs_dim}, trace err {err:.1e})")


def main():
    for role, ckpt, obs_dim in EXPORTS:
        export(role, ckpt, obs_dim)


if __name__ == "__main__":
    main()
