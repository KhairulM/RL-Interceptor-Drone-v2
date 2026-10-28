"""Analysis helpers for the thesis experiments (numpy only; no Isaac)."""

from typing import Dict, List, Sequence, Tuple

import numpy as np


# ------------------------------------------------------------------ payoff / transitivity
def logit(p, eps: float = 0.01):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def additive_fit(P: np.ndarray) -> Dict[str, float]:
    """Fit logit(P_ij) ~ a_i - b_j (Elo/Bradley-Terry-style, one rating per pursuer and per evader).

    R^2 near 1 means the matchup outcomes are explained by a single strength axis (transitive game).
    """
    L = logit(P)
    fit = L.mean(1, keepdims=True) + L.mean(0, keepdims=True) - L.mean()
    ss_res = float(((L - fit) ** 2).sum())
    ss_tot = float(((L - L.mean()) ** 2).sum())
    return {"r2": 1 - ss_res / ss_tot if ss_tot > 0 else 1.0,
            "pursuer_rating": L.mean(1).tolist(), "evader_rating": (-L.mean(0)).tolist()}


def rank_reversals(P: np.ndarray, delta: float = 0.1) -> Dict[str, float]:
    """Fraction of (pursuer pair, evader pair) quadruples where a beats b by > delta on x but b beats a by > delta on y."""
    n_p, n_e = P.shape
    D = P[:, None, :] - P[None, :, :]                 # [a, b, x] = P_ax - P_bx
    pos = (D > delta).astype(float)
    neg = (D < -delta).astype(float)
    rev = np.einsum("abx,aby->ab", pos, neg)          # x where a>b, y where b>a
    iu = np.triu_indices(n_p, 1)
    total = len(iu[0]) * n_e * (n_e - 1) / 2
    count = float((rev + rev.transpose(1, 0))[iu].sum()) / 2  # each unordered (x, y) pair counted twice
    return {"reversal_fraction": count / total if total else 0.0, "delta": delta}


# ------------------------------------------------------------------ descriptors / diversity
def zscore(X: np.ndarray) -> np.ndarray:
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 1e-9, sd, 1.0)


def combine_blocks(blocks: Sequence[np.ndarray]) -> np.ndarray:
    """Z-score each block's columns and scale each block by 1/sqrt(n_cols) so blocks weigh equally in distances."""
    return np.concatenate([zscore(B) / np.sqrt(B.shape[1]) for B in blocks], axis=1)


def pairwise_dist(X: np.ndarray) -> np.ndarray:
    sq = (X ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0))


def rbf_kernel(D: np.ndarray, sigma: float = None) -> Tuple[np.ndarray, float]:
    if sigma is None:
        iu = np.triu_indices(len(D), 1)
        sigma = float(np.median(D[iu])) if len(iu[0]) else 1.0
    return np.exp(-D ** 2 / (2 * sigma ** 2)), sigma


def vendi_score(K: np.ndarray) -> float:
    """Effective number of distinct items (Friedman & Dieng 2023): exp(entropy of eigenvalues of K / n)."""
    n = len(K)
    if n == 0:
        return 0.0
    lam = np.linalg.eigvalsh(K / n)
    lam = lam[lam > 1e-12]
    return float(np.exp(-(lam * np.log(lam)).sum()))


def kmedoids(D: np.ndarray, k: int, n_init: int = 20, seed: int = 0, iters: int = 100) -> Tuple[np.ndarray, np.ndarray, float]:
    """Alternating k-medoids on a distance matrix. Returns (labels, medoid indices, total cost)."""
    rng = np.random.default_rng(seed)
    n = len(D)
    best = (None, None, np.inf)
    for _ in range(n_init):
        med = rng.choice(n, size=k, replace=False)
        for _ in range(iters):
            labels = np.argmin(D[:, med], axis=1)
            new = med.copy()
            for c in range(k):
                members = np.where(labels == c)[0]
                if len(members):
                    new[c] = members[np.argmin(D[np.ix_(members, members)].sum(1))]
            if np.array_equal(np.sort(new), np.sort(med)):
                break
            med = new
        labels = np.argmin(D[:, med], axis=1)
        cost = float(D[np.arange(n), med[labels]].sum())
        if cost < best[2]:
            best = (labels, med, cost)
    return best


def silhouette(D: np.ndarray, labels: np.ndarray) -> float:
    n = len(D)
    s = np.zeros(n)
    for i in range(n):
        same = (labels == labels[i])
        same[i] = False
        if not same.any():
            continue
        a = D[i, same].mean()
        b = min(D[i, labels == c].mean() for c in np.unique(labels) if c != labels[i])
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


def classical_mds(D: np.ndarray, dim: int = 2) -> np.ndarray:
    n = len(D)
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:dim]
    return V[:, idx] * np.sqrt(np.maximum(w[idx], 0))


# ------------------------------------------------------------------ statistics
def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0, stat=np.mean) -> Tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)])
    return float(stat(v)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def iqm(values: Sequence[float]) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(v[lo:hi].mean()) if hi > lo else float(v.mean())
