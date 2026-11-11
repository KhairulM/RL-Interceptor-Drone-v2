"""Re-score an eval at a larger capture radius from its per-episode file (eval-only, exact for the same policies).

An episode is identical up to the first step whose centre distance is <= the new radius; there a capture at that radius
ends it. So an episode becomes a capture iff its closest sampled distance (`min_distance`) is <= the radius. Crashes,
leaving the arena and timeouts end an episode, so they happen at or after that step and are cleared for such episodes
(a same-step coincidence counts as a capture, as in the env). Not re-scored: times, motor effort, behaviour features and
pursuer out-of-bounds (non-terminal under the thesis rules, timing unknown; under Roncero's rules it ends the episode,
so pass pursuer_oob_terminal=True for those evals).
"""

import json
import os

import torch

_CACHE = {}


def rescore(json_path: str, radius: float, pursuer_oob_terminal: bool = False) -> dict:
    """The eval JSON with every matchup's outcome rates recomputed at `radius` (needs `<json>_episodes.pt`)."""
    key = (os.path.abspath(json_path), radius, pursuer_oob_terminal)
    if key in _CACHE:
        return _CACHE[key]
    with open(json_path) as f:
        d = json.load(f)
    ep_path = json_path[:-len(".json")] + "_episodes.pt"
    ep = torch.load(ep_path, map_location="cpu")
    E, assign = ep["episodes"], ep["assignment"]
    for opp, m in d["matchups"].items():
        sel = torch.isin(E["env"], assign[opp].cpu())
        e = {k: v[sel] for k, v in E.items()}
        pursuer = m["eval_role"] == "pursuer"
        flag = {"pursuer_crash": e["crash" if pursuer else "opp_crash"] > .5,
                "evader_crash": e["opp_crash" if pursuer else "crash"] > .5,
                "pursuer_oob": e["out_of_bounds" if pursuer else "opp_out_of_bounds"] > .5,
                "evader_oob": e["opp_out_of_bounds" if pursuer else "out_of_bounds"] > .5,
                "timeout": e["timeout"] > .5}
        cap0 = e["capture"] > .5
        cap = cap0 | (e["min_distance"] <= radius)
        new = cap & ~cap0
        for k in ("pursuer_crash", "evader_crash", "evader_oob", "timeout") + (("pursuer_oob",) if pursuer_oob_terminal else ()):
            flag[k] = flag[k] & ~new
        n = int(sel.sum())
        assert n == m["episodes"], f"{json_path} {opp}: {n} episodes in the file vs {m['episodes']} in the JSON"
        neut = cap | flag["evader_crash"] | flag["evader_oob"]
        loss = ~cap & (flag["pursuer_crash"] | flag["timeout"])
        wins, decisive = int(cap.sum()), int(cap.sum() + loss.sum())
        m.update(capture_rate=float(cap.float().mean()), neutralization_rate=float(neut.float().mean()),
                 win_rate=wins / decisive if decisive else None, decisive_episodes=decisive,
                 pursuer_crash_rate=float(flag["pursuer_crash"].float().mean()),
                 outcomes={"capture": float(cap.float().mean()), **{k: float(f.float().mean()) for k, f in flag.items()}})
    d["rescore"] = {"capture_radius": radius, "source": os.path.basename(json_path)}
    _CACHE[key] = d
    return d


def rates_at(json_path: str, key: str, radius: float, pursuer_oob_terminal: bool = False) -> dict:
    """{opponent: rate} at `radius`; key = a matchup field (e.g. neutralization_rate) or an `outcomes` field. {} if the
    eval does not exist."""
    if not os.path.isfile(json_path):
        return {}
    out = {}
    for opp, m in rescore(json_path, radius, pursuer_oob_terminal)["matchups"].items():
        out[opp] = m[key] if key in m else m["outcomes"][key]
    return out
