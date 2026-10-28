"""Multi-episode evaluation accumulator for pursuit-evasion (thesis evaluations).

Each env runs episodes back to back (auto-reset). Every episode that *starts* before ``cutoff_step`` is counted.
Whether an episode is included depends only on the lengths of the earlier episodes of that env, never on its own
outcome, so pooled outcome rates are consistent estimates of the per-episode rates. ``cutoff_step=1`` reproduces the
classic "first episode of every env" evaluation.

Also accumulates per-step evader behaviour features, summarised per episode as mean and std. They are the
trajectory part of the opponent behaviour descriptors (diversity analysis).

Pure torch on purpose: tests load this file by path without importing Isaac Sim.
"""

from typing import Dict, List, Optional

import torch

BEHAVIOUR_FEATURES = (
    "evader_speed",
    "evader_altitude",
    "evader_radius",      # horizontal distance from the arena centre
    "evader_vz_abs",
    "evader_accel",
    "evader_lat_accel",   # acceleration component orthogonal to the velocity
    "evader_turn_rate",   # |d heading / dt| of the horizontal velocity
    "flee_cos",           # cos(evader velocity, line of sight pursuer -> evader); +1 = fleeing straight away
    "distance",
)

# Stats copied from the env's terminal snapshot for every counted episode.
EPISODE_STATS = (
    "capture", "crash", "opp_crash", "out_of_bounds", "opp_out_of_bounds", "timeout",
    "episode_len", "min_distance", "motor_effort",
)


def behaviour_step_features(e_pos, e_vel, p_pos, prev_e_vel, step_dt: float) -> torch.Tensor:
    """Evader behaviour features for one step. Inputs are [N, 3] (env-local position, world velocity)."""
    zeros = torch.zeros(e_pos.shape[0], device=e_pos.device)
    speed = e_vel.norm(dim=-1)
    moving = speed > 0.1
    acc_vec = (e_vel - prev_e_vel) / step_dt
    v_hat = e_vel / speed.clamp_min(1e-6).unsqueeze(-1)
    lat_vec = acc_vec - (acc_vec * v_hat).sum(-1, keepdim=True) * v_hat
    lat = torch.where(moving, lat_vec.norm(dim=-1), zeros)

    h_now, h_prev = e_vel[:, :2], prev_e_vel[:, :2]
    cross = h_prev[:, 0] * h_now[:, 1] - h_prev[:, 1] * h_now[:, 0]
    dot = (h_prev * h_now).sum(-1)
    turn_ok = (h_now.norm(dim=-1) > 0.2) & (h_prev.norm(dim=-1) > 0.2)
    turn = torch.where(turn_ok, torch.atan2(cross, dot).abs() / step_dt, zeros)

    los = e_pos - p_pos
    dist = los.norm(dim=-1)
    flee = torch.where(moving, (e_vel * los).sum(-1) / (speed * dist).clamp_min(1e-6), zeros)

    return torch.stack([
        speed, e_pos[:, 2], e_pos[:, :2].norm(dim=-1), e_vel[:, 2].abs(),
        acc_vec.norm(dim=-1), lat, turn, flee, dist,
    ], dim=-1)


class EpisodeAccumulator:
    """Collects every counted episode's outcome stats and behaviour summary."""

    def __init__(self, num_envs: int, cutoff_step: int, device):
        self.num_envs = num_envs
        self.cutoff_step = int(cutoff_step)
        self.device = device
        nf = len(BEHAVIOUR_FEATURES)
        self.ep_start = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ep_index = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.f_sum = torch.zeros(num_envs, nf, device=device)
        self.f_sq = torch.zeros(num_envs, nf, device=device)
        self.f_n = torch.zeros(num_envs, device=device)
        self._rows: List[Dict[str, torch.Tensor]] = []

    def all_counted_done(self) -> bool:
        """True once no env is still inside an episode that will be counted."""
        return bool((self.ep_start >= self.cutoff_step).all())

    def step(self, t: int, done: torch.Tensor, stats: Dict[str, torch.Tensor],
             features: Optional[torch.Tensor] = None):
        """Call once per env step ``t`` (0-based) after stepping.

        ``done``: [N] bool. ``stats``: the env's ``("next", "stats")`` entries (terminal snapshot for done envs),
        each [N] or [N, 1]. ``features``: [N, F] behaviour features of the post-step state. For done envs the drone
        state has already been reset, so their features are not accumulated this step.
        """
        done = done.reshape(-1).bool()
        if features is not None:
            live = (~done).float().unsqueeze(-1)
            self.f_sum += features * live
            self.f_sq += features.square() * live
            self.f_n += live.squeeze(-1)

        finished = done & (self.ep_start < self.cutoff_step)
        if bool(finished.any()):
            idx = finished.nonzero().squeeze(-1)
            row = {k: stats[k].reshape(self.num_envs)[idx].float().clone() for k in EPISODE_STATS if k in stats}
            n = self.f_n[idx].clamp_min(1.0).unsqueeze(-1)
            mean = self.f_sum[idx] / n
            row["env"] = idx.clone()
            row["episode_index"] = self.ep_index[idx].clone()
            row["feat_mean"] = mean
            row["feat_std"] = (self.f_sq[idx] / n - mean.square()).clamp_min(0.0).sqrt()
            self._rows.append(row)

        if bool(done.any()):
            self.ep_start = torch.where(done, torch.full_like(self.ep_start, t + 1), self.ep_start)
            self.ep_index = self.ep_index + done.long()
            self.f_sum[done] = 0.0
            self.f_sq[done] = 0.0
            self.f_n[done] = 0.0

    def episodes(self) -> Dict[str, torch.Tensor]:
        """All counted episodes, concatenated (CPU tensors)."""
        if not self._rows:
            return {}
        keys = self._rows[0].keys()
        return {k: torch.cat([r[k] for r in self._rows]).cpu() for k in keys}


def summarize_matchups(episodes: Dict[str, torch.Tensor], assignment: Dict[str, torch.Tensor], role: str,
                       step_dt: float, max_episode_steps: int) -> Dict[str, dict]:
    """Per-opponent metrics from counted episodes. Rates are fractions of counted episodes.

    ``role`` is the evaluated agent's role. Win rate keeps the self-play definition (capture vs pursuer
    crash/timeout, over decisive episodes). Neutralisation = capture, evader crash or evader leaving the arena.
    Times are in seconds (``step_dt`` = sim dt x substeps); time to neutralise is censored at the max episode time.
    """
    if not episodes:
        return {}
    agent_is_pursuer = role == "pursuer"
    src = {
        "pursuer_crash": "crash" if agent_is_pursuer else "opp_crash",
        "evader_crash": "opp_crash" if agent_is_pursuer else "crash",
        "pursuer_oob": "out_of_bounds" if agent_is_pursuer else "opp_out_of_bounds",
        "evader_oob": "opp_out_of_bounds" if agent_is_pursuer else "out_of_bounds",
        "timeout": "timeout",
    }
    env = episodes["env"]
    max_time = max_episode_steps * step_dt
    out = {}
    for opp_id, env_ids in assignment.items():
        sel = torch.isin(env, env_ids.cpu())
        n = int(sel.sum().item())
        if n == 0:
            continue
        flag = {k: episodes[v][sel] > 0.5 for k, v in src.items()}
        cap = episodes["capture"][sel] > 0.5
        neut = cap | flag["evader_crash"] | flag["evader_oob"]
        ep_time = episodes["episode_len"][sel] * step_dt
        loss = ~cap & (flag["pursuer_crash"] | flag["timeout"])
        wins, decisive = int(cap.sum()), int(cap.sum() + loss.sum())
        record = {
            "episodes": n,
            "eval_role": role,
            "capture_rate": float(cap.float().mean()),
            "neutralization_rate": float(neut.float().mean()),
            "win_rate": wins / decisive if decisive > 0 else None,
            "decisive_episodes": decisive,
            "avg_time_to_capture": float(ep_time[cap].mean()) if bool(cap.any()) else max_time,
            "time_to_neutralize_censored": float(torch.where(neut, ep_time, torch.full_like(ep_time, max_time)).mean()),
            "avg_episode_time": float(ep_time.mean()),
            "avg_min_distance": float(episodes["min_distance"][sel].mean()),
            "avg_motor_effort": float(episodes["motor_effort"][sel].mean()),
            "pursuer_crash_rate": float(flag["pursuer_crash"].float().mean()),
            "outcomes": {"capture": float(cap.float().mean()),
                         **{k: float(f.float().mean()) for k, f in flag.items()}},
            "behaviour": {
                "features": list(BEHAVIOUR_FEATURES),
                "mean": episodes["feat_mean"][sel].mean(0).tolist(),
                "std": episodes["feat_std"][sel].mean(0).tolist(),
            },
        }
        out[opp_id] = record
    return out
