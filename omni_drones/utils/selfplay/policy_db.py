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

"""JSON-backed policy database for population-based pursuit-evasion self-play.

Stores a growing population of pursuer and evader policies (one JSON file per
role) plus a matchup table (``matchups.json``) that records, for every
(pursuer, evader) pair that has been evaluated, the pursuer win rate, average
time-to-capture, average inter-drone distance, and average motor effort.

The matchup metrics drive opponent sampling: when training one side against a
frozen opponent pool, opponents are drawn from a softmax over a per-opponent
"hardness" score (a temperature-scaled, min-max-normalised weighted sum of the
four metrics), oriented so that opponents currently harder for the training
agent are sampled more often.
"""

import json
import math
import os
import tempfile
from typing import Dict, List, Optional

# The four metrics tracked per matchup. ``win_rate`` is always the *pursuer*
# capture rate (fraction of episodes ending in capture); the remaining three are
# averages over episodes of a matchup.
METRIC_KEYS = (
    "win_rate", "avg_time_to_capture", "avg_distance", "avg_motor_effort",
    "agent_error_rate",
)

# Roles.
PURSUER = "pursuer"
EVADER = "evader"


def _opponent_role(role: str) -> str:
    return EVADER if role == PURSUER else PURSUER


def _matchup_key(pursuer_id: str, evader_id: str) -> str:
    return f"{pursuer_id}|{evader_id}"


class PolicyDatabase:
    """Reads/writes the pursuer/evader pools and their matchup table."""

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        self._pools: Dict[str, dict] = {PURSUER: None, EVADER: None}
        self._matchups: Optional[dict] = None
        self.load()

    # ------------------------------------------------------------------ paths
    def pool_path(self, role: str) -> str:
        assert role in (PURSUER, EVADER), f"invalid role: {role}"
        return os.path.join(self.root_dir, f"{role}_pool.json")

    @property
    def matchups_path(self) -> str:
        return os.path.join(self.root_dir, "matchups.json")

    # ------------------------------------------------------------------- io
    def load(self):
        for role in (PURSUER, EVADER):
            path = self.pool_path(role)
            if os.path.exists(path):
                with open(path, "r") as f:
                    self._pools[role] = json.load(f)
            else:
                self._pools[role] = {"role": role, "policies": []}
        if os.path.exists(self.matchups_path):
            with open(self.matchups_path, "r") as f:
                self._matchups = json.load(f)
        else:
            self._matchups = {"matchups": {}}

    def save(self):
        for role in (PURSUER, EVADER):
            _atomic_write_json(self.pool_path(role), self._pools[role])
        _atomic_write_json(self.matchups_path, self._matchups)

    # -------------------------------------------------------------- policies
    def list_policies(self, role: str) -> List[dict]:
        return list(self._pools[role]["policies"])

    def policy_ids(self, role: str) -> List[str]:
        return [p["id"] for p in self._pools[role]["policies"]]

    def get_policy(self, role: str, policy_id: str) -> Optional[dict]:
        for p in self._pools[role]["policies"]:
            if p["id"] == policy_id:
                return p
        return None

    def latest_policy_id(self, role: str) -> Optional[str]:
        policies = self._pools[role]["policies"]
        return policies[-1]["id"] if policies else None

    def add_policy(
        self,
        role: str,
        checkpoint_path: Optional[str],
        kind: str = "rl",
        policy_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Append a policy to a pool and return its id.

        ``kind`` is ``"rl"`` (a saved PPO checkpoint) or ``"heuristic"`` (a
        scripted controller identified by ``metadata['heuristic']``).
        """
        assert role in (PURSUER, EVADER), f"invalid role: {role}"
        assert kind in ("rl", "heuristic"), f"invalid kind: {kind}"
        policies = self._pools[role]["policies"]
        generation = len(policies)
        if policy_id is None:
            tag = kind if kind == "heuristic" else "s"
            policy_id = f"{role}:{tag}{generation}"
        if any(p["id"] == policy_id for p in policies):
            raise ValueError(f"policy id already exists: {policy_id}")
        entry = {
            "id": policy_id,
            "role": role,
            "generation": generation,
            "kind": kind,
            "checkpoint_path": (
                os.path.abspath(checkpoint_path) if checkpoint_path else None
            ),
            "metadata": dict(metadata or {}),
        }
        policies.append(entry)
        self.save()
        return policy_id

    # -------------------------------------------------------------- matchups
    def record_matchup(self, pursuer_id: str, evader_id: str, metrics: Dict[str, float]):
        """Store/overwrite the metrics for a (pursuer, evader) matchup."""
        missing = [k for k in METRIC_KEYS if k not in metrics]
        if missing:
            raise KeyError(f"matchup metrics missing keys: {missing}")
        record = {k: float(metrics[k]) for k in METRIC_KEYS}
        record["episodes"] = int(metrics.get("episodes", 0))
        # Keep any diagnostic extras the evaluator attached (e.g. the
        # per-opponent termination breakdown under "outcomes"). Only METRIC_KEYS
        # feed opponent sampling; the rest are there to explain a matchup.
        for k, v in metrics.items():
            if k not in record:
                record[k] = v
        self._matchups["matchups"][_matchup_key(pursuer_id, evader_id)] = record
        self.save()

    def get_matchup(self, pursuer_id: str, evader_id: str) -> Optional[dict]:
        return self._matchups["matchups"].get(_matchup_key(pursuer_id, evader_id))

    # -------------------------------------------------------------- sampling
    def compute_sampling_probs(
        self,
        target_role: str,
        temperature: float = 1.0,
        metric_weights: Optional[Dict[str, float]] = None,
        reference_policy_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Probabilities over opponents for training ``target_role``.

        Opponents are the policies of the other role. Each opponent's score is a
        weighted sum of its four matchup metrics against ``reference_policy_id``
        (the latest ``target_role`` policy by default), each min-max normalised
        across the candidate opponents and oriented so a larger value means the
        opponent is currently *harder* for the training agent. Probabilities are
        ``softmax(score / temperature)``. Falls back to uniform when there is no
        usable reference or no recorded metrics.
        """
        opp_role = _opponent_role(target_role)
        opponents = self.policy_ids(opp_role)
        if not opponents:
            return {}
        if len(opponents) == 1:
            return {opponents[0]: 1.0}

        weights = dict(DEFAULT_METRIC_WEIGHTS)
        if metric_weights:
            weights.update(metric_weights)

        if reference_policy_id is None:
            reference_policy_id = self.latest_policy_id(target_role)

        # Collect each opponent's matchup metrics vs the reference policy.
        raw: Dict[str, Optional[dict]] = {}
        for opp in opponents:
            if reference_policy_id is None:
                raw[opp] = None
            elif target_role == PURSUER:
                raw[opp] = self.get_matchup(reference_policy_id, opp)
            else:
                raw[opp] = self.get_matchup(opp, reference_policy_id)

        if all(v is None for v in raw.values()):
            # Cold start / never evaluated: uniform.
            u = 1.0 / len(opponents)
            return {opp: u for opp in opponents}

        # Min-max normalise each metric across opponents (missing -> neutral).
        normed = _normalise_metrics(raw)
        orient = _HARDNESS_ORIENTATION[target_role]

        scores: Dict[str, float] = {}
        for opp in opponents:
            score = 0.0
            for m in METRIC_KEYS:
                hardness = normed[opp][m] if orient[m] > 0 else (1.0 - normed[opp][m])
                score += abs(weights.get(m, 0.0)) * hardness
            scores[opp] = score

        return _softmax(scores, temperature)


# Default relative importance of each metric in the sampling score. Orientation
# (which direction is "harder") is handled per role below, so weights are
# magnitudes.
DEFAULT_METRIC_WEIGHTS = {
    "win_rate": 1.0,
    "avg_time_to_capture": 0.5,
    "avg_distance": 0.0,
    "avg_motor_effort": 0.25,
    # Fraction of episodes in which the TRAINING agent itself crashed to the
    # ground or left the arena against this opponent. A harder opponent forces
    # the agent into riskier, less controlled flight, so a higher agent
    # crash/OOB rate flags a harder opponent (same orientation for both roles).
    "agent_error_rate": 1.0,
}

# +1: larger normalised metric == harder for the training agent; -1: inverse.
# win_rate is the pursuer capture rate.
_HARDNESS_ORIENTATION = {
    # Training the pursuer against evaders: a hard evader has low capture rate,
    # long time-to-capture, large distance, high effort to catch.
    PURSUER: {
        "win_rate": -1,
        "avg_time_to_capture": +1,
        "avg_distance": +1,
        "avg_motor_effort": +1,
        "agent_error_rate": +1,
    },
    # Training the evader against pursuers: a hard pursuer has high capture rate,
    # short time-to-capture, small distance, low effort.
    EVADER: {
        "win_rate": +1,
        "avg_time_to_capture": -1,
        "avg_distance": -1,
        "avg_motor_effort": -1,
        "agent_error_rate": +1,
    },
}


def _normalise_metrics(raw: Dict[str, Optional[dict]]) -> Dict[str, Dict[str, float]]:
    """Min-max normalise each metric to [0, 1] across opponents."""
    normed: Dict[str, Dict[str, float]] = {opp: {} for opp in raw}
    for m in METRIC_KEYS:
        vals = [raw[opp][m] for opp in raw if raw[opp] is not None]
        if vals:
            lo, hi = min(vals), max(vals)
            span = hi - lo
        else:
            lo, hi, span = 0.0, 0.0, 0.0
        for opp in raw:
            if raw[opp] is None or span <= 1e-9:
                normed[opp][m] = 0.5  # neutral when unknown or degenerate
            else:
                normed[opp][m] = (raw[opp][m] - lo) / span
    return normed


def _softmax(scores: Dict[str, float], temperature: float) -> Dict[str, float]:
    temperature = max(float(temperature), 1e-6)
    keys = list(scores.keys())
    logits = [scores[k] / temperature for k in keys]
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    total = sum(exps)
    if total <= 0:
        u = 1.0 / len(keys)
        return {k: u for k in keys}
    return {k: e / total for k, e in zip(keys, exps)}


def _atomic_write_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
