# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Frozen opponent bank for population-based pursuit-evasion self-play.

Holds a mix of frozen RL policies (PPO checkpoints, evaluated deterministically)
and scripted heuristic labels, assigns each parallel environment to exactly one
opponent for the whole run (fixed per generation), and produces the raw CTBR
actions for the RL-assigned environments. Heuristic geometry (hover / circular /
pursue / flee) is computed inside the task env, which owns the controllers that
convert targets and CTBR into motor commands.
"""

from typing import Dict, List, Optional

import torch
from tensordict.tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from omni_drones.learning import ALGOS
from omni_drones.utils.sampling import policy_sampling


class FrozenPolicyBank:
    """Manages frozen opponent policies and their per-env assignment."""

    def __init__(
        self,
        policies: List[dict],
        observation_spec,
        action_spec,
        reward_spec,
        algo_cfg,
        device,
    ):
        self.device = device
        self.policies = list(policies)
        self.ids: List[str] = [p["id"] for p in self.policies]
        self.kind: Dict[str, str] = {p["id"]: p["kind"] for p in self.policies}
        self.heuristic: Dict[str, Optional[str]] = {
            p["id"]: (p.get("metadata") or {}).get("heuristic") for p in self.policies
        }
        self.action_dim = int(action_spec[("agents", "action")].shape[-1])

        # Instantiate + load each frozen RL policy once.
        self._rl = {}
        for p in self.policies:
            if p["kind"] != "rl":
                continue
            ckpt = p.get("checkpoint_path")
            if not ckpt:
                raise ValueError(f"rl policy {p['id']} has no checkpoint_path")
            policy = ALGOS["ppo"](
                algo_cfg, observation_spec, action_spec, reward_spec, device=device
            )
            state = torch.load(ckpt, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            try:
                policy.load_state_dict(state)
            except RuntimeError:
                policy.load_state_dict(state, strict=False)
            policy.eval()
            self._rl[p["id"]] = policy

        self.heuristic_names = sorted(
            {h for h in self.heuristic.values() if h is not None}
        )

        self.num_envs = 0
        self.assignment: Dict[str, torch.Tensor] = {}
        self.env_policy: Optional[torch.Tensor] = None
        self._heuristic_masks: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------- assignment
    def assign(self, probs: Dict[str, float], num_envs: int):
        """Fix each env to an opponent by sampling the id distribution once."""
        self.num_envs = num_envs
        # Only sample over ids that this bank actually holds.
        probs = {k: v for k, v in probs.items() if k in self.ids and v > 0}
        if not probs:
            raise ValueError("no valid opponent probabilities for this bank")
        mapping = policy_sampling(probs, num_envs)

        self.assignment = {}
        env_policy = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        for i, pid in enumerate(self.ids):
            idx = mapping.get(pid, [])
            if not idx:
                continue
            idx_t = torch.as_tensor(idx, device=self.device, dtype=torch.long)
            self.assignment[pid] = idx_t
            env_policy[idx_t] = i
        self.env_policy = env_policy

        self._heuristic_masks = {}
        for name in self.heuristic_names:
            mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
            for pid, hname in self.heuristic.items():
                if hname == name and pid in self.assignment:
                    mask[self.assignment[pid]] = True
            self._heuristic_masks[name] = mask
        return self.assignment

    def heuristic_masks(self) -> Dict[str, torch.Tensor]:
        """Bool mask [num_envs] of envs assigned to each heuristic name."""
        return self._heuristic_masks

    @property
    def rl_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for pid in self._rl:
            if pid in self.assignment:
                mask[self.assignment[pid]] = True
        return mask

    # --------------------------------------------------------------- actions
    @torch.no_grad()
    def rl_raw_actions(self, opp_obs: torch.Tensor) -> torch.Tensor:
        """Raw (pre-tanh) CTBR actions for RL-assigned envs.

        ``opp_obs`` is [num_envs, 1, obs_dim] built from the opponent's own
        perspective. Each frozen policy runs over the full batch; only its
        assigned rows are kept. Non-RL rows stay zero.
        """
        raw = torch.zeros(
            self.num_envs, 1, self.action_dim, device=self.device
        )
        if not self._rl:
            return raw
        with set_exploration_type(ExplorationType.MODE):
            for pid, policy in self._rl.items():
                idx = self.assignment.get(pid)
                if idx is None or idx.numel() == 0:
                    continue
                td = TensorDict(
                    {("agents", "observation"): opp_obs},
                    batch_size=[self.num_envs],
                    device=self.device,
                )
                policy(td)
                raw[idx] = td[("agents", "action")][idx]
        return raw
