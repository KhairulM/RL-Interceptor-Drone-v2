"""Thesis F16: append-only two-role payoff store for the PSRO / PSD-PSRO restricted game (research_plan.md F16 C.1).

Every evaluation run appends one immutable record per (pursuer, evader) cell to ``payoffs.jsonl``: episode count,
per-role discounted Roncero return mean / std (both roles measured, never inferred by negation), event rates and
provenance (seed, run id, evaluating role, checkpoint hashes, config). Records are never rewritten; a cell's estimate
pools all its records (episode-weighted mean, pooled variance). Cells with fewer than ``min_episodes`` episodes block
``matrices``. Payoffs are returns divided by ``RETURN_SCALE`` (the capture magnitude k_c = 10).

Pure python + numpy: tests load this file by path.
"""

import hashlib
import json
import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

STORE_VERSION = "f16-payoff-store-v1"
RETURN_SCALE = 10.0


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class MissingCells(RuntimeError):
    pass


class PayoffStore:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.path = os.path.join(self.root, "payoffs.jsonl")

    def records(self) -> List[dict]:
        if not os.path.isfile(self.path):
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def run_ids(self) -> set:
        return {r["run_id"] for r in self.records()}

    def append(self, records: Sequence[dict]):
        """Append the cell records of one evaluation run (all-or-nothing; a run id can be stored only once)."""
        records = list(records)
        if not records:
            return
        run_ids = {r["run_id"] for r in records}
        if len(run_ids) != 1:
            raise ValueError("one append = one evaluation run")
        if run_ids & self.run_ids():
            raise ValueError(f"run {run_ids} already stored; records are immutable")
        for r in records:
            for k in ("pursuer", "evader", "n", "mean_pursuer", "std_pursuer", "mean_evader", "std_evader"):
                if k not in r:
                    raise KeyError(f"payoff record missing {k}")
            if not all(math.isfinite(float(r[k])) for k in ("mean_pursuer", "std_pursuer", "mean_evader",
                                                             "std_evader")):
                raise ValueError(f"non-finite payoff record {r}")
            if int(r["n"]) <= 0:
                raise ValueError("payoff record without episodes")
        with open(self.path, "a") as f:
            for r in records:
                f.write(json.dumps(dict(r, store_version=STORE_VERSION), sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def cell(self, pursuer: str, evader: str) -> Optional[dict]:
        """Pooled estimate of one cell (None if never evaluated)."""
        rs = [r for r in self.records() if r["pursuer"] == pursuer and r["evader"] == evader]
        return pool(rs) if rs else None

    def cells(self) -> Dict[tuple, dict]:
        groups: Dict[tuple, list] = {}
        for r in self.records():
            groups.setdefault((r["pursuer"], r["evader"]), []).append(r)
        return {k: pool(v) for k, v in groups.items()}

    def missing(self, pursuers: Sequence[str], evaders: Sequence[str], min_episodes: int) -> List[tuple]:
        c = self.cells()
        return [(p, e) for p in pursuers for e in evaders if c.get((p, e), {"n": 0})["n"] < min_episodes]

    def matrices(self, pursuers: Sequence[str], evaders: Sequence[str], min_episodes: int):
        """(U_P, U_E, N): scaled per-role payoff matrices (rows pursuers, columns evaders) and episode counts."""
        miss = self.missing(pursuers, evaders, min_episodes)
        if miss:
            raise MissingCells(f"{len(miss)} cells below {min_episodes} episodes: {miss[:5]}")
        c = self.cells()
        UP = np.array([[c[(p, e)]["mean_pursuer"] / RETURN_SCALE for e in evaders] for p in pursuers])
        UE = np.array([[c[(p, e)]["mean_evader"] / RETURN_SCALE for e in evaders] for p in pursuers])
        N = np.array([[c[(p, e)]["n"] for e in evaders] for p in pursuers])
        return UP, UE, N


def pool(records: Sequence[dict]) -> dict:
    """Episode-weighted mean and pooled (total) standard deviation of the per-role returns over several records."""
    n = np.array([float(r["n"]) for r in records])
    out = {"n": int(n.sum()), "records": len(records), "run_ids": sorted({r["run_id"] for r in records})}
    for role in ("pursuer", "evader"):
        m = np.array([float(r[f"mean_{role}"]) for r in records])
        s = np.array([float(r[f"std_{role}"]) for r in records])
        mean = float((n * m).sum() / n.sum())
        var = float((n * (s ** 2 + (m - mean) ** 2)).sum() / n.sum())
        out[f"mean_{role}"] = mean
        out[f"std_{role}"] = math.sqrt(max(var, 0.0))
        out[f"se_{role}"] = out[f"std_{role}"] / math.sqrt(n.sum())
    events = {}
    for r in records:
        for k, v in (r.get("events") or {}).items():
            events.setdefault(k, 0.0)
            events[k] += float(v) * float(r["n"])
    out["events"] = {k: v / n.sum() for k, v in events.items()}
    return out


def records_from_matchups(matchups: Dict[str, dict], agent_id: str, agent_role: str, run_id: str, seed: int,
                          provenance: dict) -> List[dict]:
    """Cell records from one eval run's ``summarize_matchups`` output (needs the "payoff" block)."""
    out = []
    for opp_id, m in matchups.items():
        pay = m.get("payoff")
        if not pay:
            raise KeyError(f"matchup {opp_id} has no payoff block (task.payoff_returns must be on)")
        p, e = (agent_id, opp_id) if agent_role == "pursuer" else (opp_id, agent_id)
        out.append({"pursuer": p, "evader": e, "n": int(pay["n"]), "mean_pursuer": pay["mean_pursuer"],
                    "std_pursuer": pay["std_pursuer"], "mean_evader": pay["mean_evader"],
                    "std_evader": pay["std_evader"], "events": dict(m.get("outcomes", {}),
                                                                    neutralization=m.get("neutralization_rate")),
                    "run_id": run_id, "seed": int(seed), "agent_role": agent_role, **provenance})
    return out
