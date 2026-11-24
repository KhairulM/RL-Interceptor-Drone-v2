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
pure_pursuit / apf / flee) is computed inside the task env, which owns the controllers that
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
        switch: Optional[dict] = None,
        resample: Optional[dict] = None,
        stochastic: bool = False,
    ):
        self.device = device
        self.switch = switch
        # Thesis F16: ``resample`` = {"seed": int} redraws each env's opponent from ``probs`` at every episode reset
        # (on_reset) and holds it for the episode; ``stochastic`` samples the frozen RL policies' actions instead of
        # taking the mean. Both off by default (fixed per-run assignment, deterministic opponents).
        self.resample = resample
        self.stochastic = bool(stochastic)
        self.resample_counts: Dict[str, int] = {}
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
        if self.switch is not None:
            first, second = self.switch["first_id"], self.switch["second_id"]
            if first == second or first not in self.ids or second not in self.ids:
                raise ValueError("switch policies must be two distinct policies in the opponent bank")
            if self.kind[first] != "rl" or self.kind[second] != "rl":
                raise ValueError("within-episode policy switching requires two RL opponents")
            switch_times = self.switch.get("times_s", [])
            if not switch_times or any(float(t) <= 0 for t in switch_times):
                raise ValueError("switch times must be positive seconds")
            self._switch_first = self.ids.index(first)
            self._switch_second = self.ids.index(second)
            self._switch_times = torch.as_tensor(switch_times, dtype=torch.float32, device=self.device)
            generator = torch.Generator(device="cpu").manual_seed(int(self.switch.get("seed", 0)))
            order = torch.randperm(num_envs, generator=generator).to(self.device)
            choices = order % self._switch_times.numel()
            self.switch_steps_s = self._switch_times[choices]
            self.env_policy = torch.full((num_envs,), self._switch_first, dtype=torch.long, device=self.device)
            self.assignment = {self.switch["group_id"]: torch.arange(num_envs, device=self.device)}
            self._heuristic_masks = {}
            return self.assignment
        # Only sample over ids that this bank actually holds.
        probs = {k: v for k, v in probs.items() if k in self.ids and v > 0}
        if not probs:
            raise ValueError("no valid opponent probabilities for this bank")
        if self.resample is not None:
            return self._assign_resample(probs, num_envs)
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

    def _assign_resample(self, probs: Dict[str, float], num_envs: int):
        if self.switch is not None:
            raise ValueError("episode-reset resampling and within-episode switching are exclusive")
        ids = [pid for pid in self.ids if pid in probs]
        if any(self.kind[pid] != "rl" for pid in ids):
            raise ValueError("episode-reset resampling supports RL opponents only")
        p = torch.tensor([float(probs[pid]) for pid in ids], dtype=torch.float64)
        if not bool(torch.isfinite(p).all()) or float(p.sum()) <= 0:
            raise ValueError(f"invalid resampling probabilities {probs}")
        self._resample_probs = {pid: float(v) for pid, v in zip(ids, (p / p.sum()).tolist())}
        self._resample_index = torch.tensor([self.ids.index(pid) for pid in ids], dtype=torch.long)
        self._resample_cdf = torch.cumsum(p / p.sum(), 0)
        self._resample_cdf[-1] = 1.0
        self._resample_gen = torch.Generator(device="cpu").manual_seed(int(self.resample.get("seed", 0)))
        self.resample_counts = {pid: 0 for pid in ids}
        self.env_policy = self._resample_index[self._draw(num_envs)].to(self.device)
        self._rebuild_assignment()
        self._heuristic_masks = {}
        return self.assignment

    def _draw(self, n: int) -> torch.Tensor:
        u = torch.rand(n, generator=self._resample_gen, dtype=torch.float64)
        return torch.searchsorted(self._resample_cdf, u, right=True).clamp(max=len(self._resample_cdf) - 1)

    def _rebuild_assignment(self):
        self.assignment = {}
        for i, pid in enumerate(self.ids):
            idx = (self.env_policy == i).nonzero().squeeze(-1)
            if idx.numel():
                self.assignment[pid] = idx

    def on_reset(self, env_ids: torch.Tensor):
        """Thesis F16: draw a new opponent for every env in ``env_ids`` (called at each episode reset)."""
        if self.resample is None or env_ids.numel() == 0:
            return
        k = self._draw(int(env_ids.numel()))
        self.env_policy[env_ids] = self._resample_index[k].to(self.device)
        for j, c in enumerate(torch.bincount(k, minlength=len(self._resample_index)).tolist()):
            self.resample_counts[self.ids[int(self._resample_index[j])]] += int(c)
        self._rebuild_assignment()

    def heuristic_masks(self) -> Dict[str, torch.Tensor]:
        """Bool mask [num_envs] of envs assigned to each heuristic name."""
        return self._heuristic_masks

    def update_switch(self, progress_buf: torch.Tensor, step_dt: float):
        if self.switch is None:
            return
        elapsed = progress_buf.reshape(-1).float() * float(step_dt)
        self.switch_steps = torch.ceil(self.switch_steps_s / float(step_dt)).long()
        switched = elapsed >= self.switch_steps_s
        first = torch.full_like(self.env_policy, self._switch_first)
        second = torch.full_like(self.env_policy, self._switch_second)
        self.env_policy = torch.where(switched, second, first)

    @property
    def rl_mask(self) -> torch.Tensor:
        if self.switch is not None:
            return self.env_policy >= 0
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
        perspective. Each frozen policy runs only over its assigned rows, and
        non-RL rows stay zero.
        """
        raw = torch.zeros(
            self.num_envs, 1, self.action_dim, device=self.device
        )
        if not self._rl:
            return raw
        mode = ExplorationType.RANDOM if getattr(self, "stochastic", False) else ExplorationType.MODE
        with set_exploration_type(mode):
            for pid, policy in self._rl.items():
                if self.switch is None:
                    idx = self.assignment.get(pid)
                else:
                    idx = (self.env_policy == self.ids.index(pid)).nonzero().squeeze(-1)
                if idx is None or idx.numel() == 0:
                    continue
                policy_obs = opp_obs.index_select(0, idx)
                td = TensorDict(
                    {("agents", "observation"): policy_obs},
                    batch_size=[idx.numel()],
                    device=self.device,
                )
                raw[idx] = policy.act(td)
        return raw
