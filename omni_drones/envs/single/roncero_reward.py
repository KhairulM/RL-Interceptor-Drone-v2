"""Thesis F8: Sanchez Roncero et al. 2026 (arXiv:2506.02849v3) reward and episode endings, as pure functions.

Eq. (8)  shared:  r_reg = -k_br ||omega||^2 - k_sm ||a_rate - a_rate_prev||^2 - k_b * b      (b: own out-of-bounds)
Eq. (9)  pursuer: r_P = k_c * delta * beta + k_app * (d_prev - d) + r_reg
Eq. (10) beta = exp(-alpha_close * max(0, v_close - v_safe)),  v_close = closing speed along the line of sight
Eq. (11) evader:  r_E = k_s - k_c * delta + r_reg
Endings (Sec. II-B b): capture is terminal; out-of-bounds (here: a crash into the floor or leaving through a wall or
the ceiling, either agent) and the timeout are truncations (the value target bootstraps). Coefficients: Sec. III-A c.
No Isaac imports, so the logic is unit-tested (tests/test_roncero_reward.py).
"""

import torch

DEFAULTS = {"k_br": 0.001, "k_sm": 0.002, "k_b": 10.0, "k_c": 10.0, "k_app": 0.25, "k_s": 0.0025,
            "v_safe": 2.0, "alpha_close": 3.0}


def outcome(captured, agent_crash, opp_crash, agent_oob, opp_oob, time_limit):
    """(terminated, truncated, timeout), all [N, 1] bool. timeout = time limit reached with no other ending."""
    oob_any = agent_crash | opp_crash | agent_oob | opp_oob
    terminated = captured
    truncated = (time_limit | oob_any) & ~captured
    timeout = time_limit & ~captured & ~oob_any
    return terminated, truncated, timeout


def closing_speed(p_pursuer, v_pursuer, p_evader, v_evader):
    """v_close = -u^T (v_E - v_P), u = unit line of sight from pursuer to evader. [N, 1]"""
    los = p_evader - p_pursuer
    u = los / los.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return -(u * (v_evader - v_pursuer)).sum(-1, keepdim=True)


def reward(pursuer, captured, delta_dist, omega_b, rate_jerk_sq, own_oob, v_close, k=DEFAULTS):
    """Per-step reward [N, 1] for the learning agent. pursuer: bool role flag; delta_dist = d_prev - d; omega_b: body
    angular velocity [N, 3] (or [N, 1, 3]); rate_jerk_sq: ||a_rate - a_rate_prev||^2 [N, 1]; own_oob: [N, 1] bool."""
    omega_sq = (omega_b.reshape(omega_b.shape[0], -1) ** 2).sum(-1, keepdim=True)
    reg = -k["k_br"] * omega_sq - k["k_sm"] * rate_jerk_sq - k["k_b"] * own_oob.float()
    cap = captured.float()
    if pursuer:
        beta = torch.exp(-k["alpha_close"] * torch.relu(v_close - k["v_safe"]))
        return k["k_c"] * cap * beta + k["k_app"] * delta_dist + reg
    return k["k_s"] - k["k_c"] * cap + reg


def rate_jerk(rate, prev):
    """(||a_rate - a_rate_prev||^2 [N, 1], new previous rate) for Eq. (8)'s k_sm term. Call once per control step:
    IsaacEnv calls _pre_sim_step once per physics substep with the same action, so computing it there gave 0
    (bug in every Roncero-rules run before 2026-11-10)."""
    return ((rate - prev) ** 2).sum(-1, keepdim=True), rate.detach().clone()


def ablate(terminal, terminated, truncated, timeout, captured, beta, w_terminal, pursuer,
           no_timeout=False, closing_discount=False, truncate_failures=False):
    """Thesis F9: one Roncero component at a time on top of the thesis rules. Inputs are the thesis outcome
    (terminal reward [N, 1], terminated / truncated / timeout / captured [N, 1] bool, beta [N, 1] from Eq. (10)).
    no_timeout: the timeout gives 0 to both roles (thesis: pursuer -W, evader +W).
    closing_discount: the pursuer's capture reward is W * beta (thesis: W).
    truncate_failures: every terminal ending except capture (crash, evader leaving) becomes a truncation (value
    bootstrapped); its rewards are unchanged. Returns (terminal, terminated, truncated)."""
    if no_timeout:
        terminal = torch.where(timeout, torch.zeros_like(terminal), terminal)
    if closing_discount and pursuer:
        terminal = torch.where(captured, w_terminal * beta, terminal)
    if truncate_failures:
        fail = terminated & ~captured
        truncated = truncated | fail
        terminated = captured.clone()
    return terminal, terminated, truncated
