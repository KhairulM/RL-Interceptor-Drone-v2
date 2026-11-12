"""Export a PursuitEvasion PPO policy to the deploy artifact format WITHOUT starting Isaac Sim (2026-11-12: both Isaac
slots and the RAM are taken by F11). Builds PPOPolicy from hand-made specs (obs 28, action 4, one agent), loads the
checkpoint strictly, extracts the deterministic actor with scripts/deploy/export_policy.py's own helper, traces it, and
copies metadata.json from a reference artifact exported by the official (Isaac) exporter for the same task config.

Usage: python export_noisaac.py <checkpoint> <out_dir> <reference_artifact_dir> [--compare <reference_policy_ts.pt>]
"""
import json
import os
import shutil
import sys

import torch
from omegaconf import OmegaConf
from torchrl.data import Composite, UnboundedContinuous
from torchrl.envs.utils import ExplorationType, set_exploration_type

REPO = "/home/user/Projects/RLInterceptorDrone"
sys.path.insert(0, os.path.join(REPO, "scripts/deploy"))
from export_policy import extract_deterministic_actor  # noqa: E402
from omni_drones.learning.ppo.ppo import PPOConfig, PPOPolicy  # noqa: E402

OBS, ACT = 28, 4


def build(ckpt):
    obs_spec = Composite({"agents": Composite({"observation": UnboundedContinuous((1, 1, OBS))}, shape=(1,))},
                         shape=(1,))
    act_spec = Composite({"agents": Composite({"action": UnboundedContinuous((1, 1, ACT))}, shape=(1,))},
                         shape=(1,))
    rew_spec = Composite({"agents": Composite({"reward": UnboundedContinuous((1, 1, 1))}, shape=(1,))}, shape=(1,))
    cfg = OmegaConf.structured(PPOConfig)
    policy = PPOPolicy(cfg, obs_spec, act_spec, rew_spec, device="cpu")
    state = torch.load(ckpt, map_location="cpu")
    policy.load_state_dict(state)                     # strict: any missing / unexpected key raises
    policy.eval()
    return extract_deterministic_actor(policy, "ppo")


def main():
    ckpt, out, ref = sys.argv[1:4]
    actor = build(ckpt)
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        traced = torch.jit.trace(actor, torch.zeros(1, OBS), check_trace=False)
        probe = torch.randn(512, OBS)
        err = (actor(probe) - traced(probe)).abs().max().item()
        assert err < 1e-5, err
        if "--compare" in sys.argv:
            refm = torch.jit.load(sys.argv[sys.argv.index("--compare") + 1], map_location="cpu")
            d = (traced(probe) - refm(probe)).abs().max().item()
            print(f"max |this - reference artifact| on 512 random obs: {d:.3e}")
    os.makedirs(out, exist_ok=True)
    traced.save(os.path.join(out, "policy_ts.pt"))
    meta = json.load(open(os.path.join(ref, "metadata.json")))
    meta["notes"]["checkpoint"] = os.path.abspath(ckpt)
    meta["notes"]["exported_by"] = "export_noisaac.py (no Isaac); metadata from " + os.path.abspath(ref)
    json.dump(meta, open(os.path.join(out, "metadata.json"), "w"), indent=2, sort_keys=True)
    print(f"exported {ckpt} -> {out} (trace err {err:.1e}, role {meta['role']})")


if __name__ == "__main__":
    main()
