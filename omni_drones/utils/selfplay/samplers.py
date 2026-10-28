"""Opponent samplers for population-based training (thesis experiments D1/D1b).

softmax  existing hardness sampler: (1 - u) * softmax(score / T) + u / N, score = weighted min-max-normalised metrics
         (same definition as PolicyDatabase.compute_sampling_probs).
qd_dpp   quality-diversity DPP: L = a * diag(q) S diag(q), q_i^2 = softmax_i (the hardness softmax),
         S_ij = exp(-|b_i - b_j|^2 / 2 sigma^2) over behaviour descriptors b, a = N (unit mean diagonal).
         p_i proportional to the DPP marginal K_ii, K = L (L + I)^-1, then mixed with u / N.
         With S = I the order of p equals the hardness order; m exact copies of an opponent of quality q share a
         total marginal m a q^2 / (1 + m a q^2) <= 1, instead of growing linearly in m as under softmax.

Pure python/numpy; tests load it by path.
"""

import importlib.util
import math
import os
from typing import Dict, Optional, Sequence

import numpy as np

_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_db.py")
_spec = importlib.util.spec_from_file_location("_selfplay_policy_db", _DB)
_db = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_db)

METRIC_KEYS = _db.METRIC_KEYS


def hardness_scores(raw: Dict[str, Optional[dict]], target_role: str,
                    metric_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    weights = dict(_db.DEFAULT_METRIC_WEIGHTS)
    if metric_weights:
        weights.update(metric_weights)
    normed = _db._normalise_metrics(raw)
    orient = _db._HARDNESS_ORIENTATION[target_role]
    scores = {}
    for opp in raw:
        s = 0.0
        for m in METRIC_KEYS:
            h = normed[opp][m] if orient[m] > 0 else 1.0 - normed[opp][m]
            s += abs(weights.get(m, 0.0)) * h
        scores[opp] = s
    return scores


def softmax_probs(scores: Dict[str, float], temperature: float, uniform_mix: float) -> Dict[str, float]:
    p = _db._softmax(scores, temperature)
    n = len(p)
    return {k: (1 - uniform_mix) * v + uniform_mix / n for k, v in p.items()}


def dpp_marginals(quality: Sequence[float], S: np.ndarray, scale: float) -> np.ndarray:
    q = np.asarray(quality, dtype=float)
    L = scale * (q[:, None] * S * q[None, :])
    K = L @ np.linalg.inv(L + np.eye(len(q)))
    return np.clip(np.diag(K), 0.0, None)


def qd_dpp_probs(scores: Dict[str, float], descriptors: Dict[str, Sequence[float]], temperature: float,
                 uniform_mix: float, sigma: Optional[float] = None) -> Dict[str, float]:
    ids = list(scores)
    soft = _db._softmax(scores, temperature)
    q = np.sqrt(np.array([soft[i] for i in ids]))
    B = np.array([descriptors[i] for i in ids], dtype=float)
    D2 = ((B[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    if sigma is None:
        iu = np.triu_indices(len(ids), 1)
        d = np.sqrt(D2[iu])
        d = d[d > 1e-9]
        sigma = float(np.median(d)) if len(d) else 1.0
    S = np.exp(-D2 / (2 * sigma ** 2))
    k = dpp_marginals(q, S, scale=float(len(ids)))
    p = k / k.sum() if k.sum() > 0 else np.full(len(ids), 1.0 / len(ids))
    n = len(ids)
    return {i: float((1 - uniform_mix) * pi + uniform_mix / n) for i, pi in zip(ids, p)}


def hedge_scores(challenge: Dict[str, float], eta: float, eps: float = 1e-6) -> Dict[str, float]:
    """Log-weights eta * log(c) so that a softmax at temperature 1 gives c^eta / sum c^eta (Roncero et al. 2026)."""
    return {k: eta * math.log(max(float(c), eps)) for k, c in challenge.items()}


def hedge_probs(challenge: Dict[str, float], gamma: float, eta: float) -> Dict[str, float]:
    """Roncero et al. 2026, eq. (12): p_j = gamma / N + (1 - gamma) c_j^eta / sum_l c_l^eta, c_j = 1 - win rate.
    All-zero challenges give the uniform distribution."""
    return softmax_probs(hedge_scores(challenge, eta), 1.0, gamma)


def opponent_metrics(capture, pursuer_crash, timeout, ttc, min_dist, effort, max_time: float) -> dict:
    """PolicyDatabase-style metrics from per-episode arrays of one opponent (pursuer's view)."""
    capture, pursuer_crash, timeout = (np.asarray(x, dtype=bool) for x in (capture, pursuer_crash, timeout))
    loss = ~capture & (pursuer_crash | timeout)
    decisive = int(capture.sum() + loss.sum())
    return {
        "win_rate": float(capture.sum()) / decisive if decisive else None,
        "avg_time_to_capture": float(np.mean(np.asarray(ttc)[capture])) if capture.any() else max_time,
        "avg_min_distance": float(np.mean(min_dist)),
        "avg_motor_effort": float(np.mean(effort)),
        "pursuer_crash_rate": float(pursuer_crash.mean()),
    }


def compute_probs(sampler: str, raw: Dict[str, Optional[dict]], target_role: str, temperature: float,
                  uniform_mix: float, metric_weights: Optional[Dict[str, float]] = None,
                  descriptors: Optional[Dict[str, Sequence[float]]] = None) -> Dict[str, float]:
    if sampler == "uniform":
        return {k: 1.0 / len(raw) for k in raw}
    scores = hardness_scores(raw, target_role, metric_weights)
    if sampler == "softmax":
        return softmax_probs(scores, temperature, uniform_mix)
    if sampler == "qd_dpp":
        if descriptors is None:
            raise ValueError("qd_dpp needs descriptors")
        return qd_dpp_probs(scores, descriptors, temperature, uniform_mix)
    raise ValueError(f"unknown sampler {sampler!r}")
