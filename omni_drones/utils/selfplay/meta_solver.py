"""Thesis F16: general-sum bimatrix Nash solver for the PSRO / PSD-PSRO restricted game (docs/research_plan.md, F16 C.2).

Lemke-Howson (von Stengel's polytope formulation) started from every one of the m + n initial dropped labels, on an
exact integer tableau (Bareiss pivoting) with the lexicographic minimum-ratio rule over the initial-basis columns, so
degenerate ties are resolved deterministically and pivoting cannot cycle. Payoffs are rounded to 1 / SCALE and shifted
to be >= 1 before pivoting (equilibria are invariant to the shift); every candidate is then verified on the given float
matrices: finite, normalised, nonnegative and both unilateral regrets <= ``tol``. Among the distinct verified
candidates the one with maximum joint entropy H(x) + H(y) is selected; ties (within 1e-12) go to the lexicographically
smallest (row support ids, column support ids). No verified equilibrium -> NoEquilibrium; there is no uniform, Hedge
or zero-sum fallback.

Pure numpy + python ints: tests load this file by path.
"""

import itertools
from typing import List, Sequence

import numpy as np

SOLVER_VERSION = "f16-lemke-howson-bareiss-lex-v1"
SCALE = 10 ** 9


class NoEquilibrium(RuntimeError):
    pass


def _check(A, B):
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    if A.ndim != 2 or A.shape != B.shape or min(A.shape) < 1:
        raise ValueError(f"payoff matrices must be two equal-shape non-empty 2-D arrays, got {A.shape} / {B.shape}")
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        raise ValueError("payoff matrices contain non-finite entries")
    return A, B


def _to_int(M):
    """Integer payoffs (rounded to 1 / SCALE), shifted to be >= 1 so both best-response polytopes are bounded."""
    I = [[int(round(float(v) * SCALE)) for v in row] for row in M]
    lo = min(min(r) for r in I)
    return [[v - lo + 1 for v in r] for r in I]


class _Tableau:
    """Rows ``slack/basic + sum_label coef * var = rhs`` stored as integers [rhs, coef(label 0), ...] over the common
    denominator ``det``. ``lex`` = tableau columns of the initial basis (lexicographic ratio test)."""

    def __init__(self, rows, basis, lex):
        self.T, self.basis, self.lex, self.det = rows, basis, lex, 1

    def ratio_row(self, c):
        cand = [i for i, row in enumerate(self.T) if row[c] > 0]
        if not cand:
            raise NoEquilibrium("unbounded pivot column")
        for k in [0] + self.lex:
            lo = cand[0]
            for i in cand[1:]:
                if self.T[i][k] * self.T[lo][c] < self.T[lo][k] * self.T[i][c]:
                    lo = i
            cand = [i for i in cand if self.T[i][k] * self.T[lo][c] == self.T[lo][k] * self.T[i][c]]
            if len(cand) == 1:
                return cand[0]
        raise NoEquilibrium("lexicographic ratio test did not resolve a tie")

    def pivot(self, r, c):
        T, p = self.T, self.T[r][c]
        for i in range(len(T)):
            if i != r:
                f = T[i][c]
                T[i] = [(p * a - f * b) // self.det for a, b in zip(T[i], T[r])]
        self.det = p
        self.basis[r] = c - 1


def _lemke_howson(Ai, Bi, m, n, k0, max_pivots=100000):
    """Path from the artificial equilibrium dropping label k0. Labels 0..m-1: row strategies; m..m+n-1: columns.
    Q = {y >= 0 : A y <= 1} (slack of row i carries label i, y_j label m + j); P = {x >= 0 : B^T x <= 1} (x_i label i,
    slack of column j label m + j). Returns the normalised (x, y)."""
    tQ = _Tableau([[1] + [1 if c == i else 0 for c in range(m)] + [Ai[i][j] for j in range(n)] for i in range(m)],
                  list(range(m)), [1 + c for c in range(m)])
    tP = _Tableau([[1] + [Bi[i][j] for i in range(m)] + [1 if c == j else 0 for c in range(n)] for j in range(n)],
                  [m + j for j in range(n)], [1 + m + c for c in range(n)])
    entering = k0
    tab = tP if k0 < m else tQ
    for _ in range(max_pivots):
        r = tab.ratio_row(entering + 1)
        leaving = tab.basis[r]
        tab.pivot(r, entering + 1)
        if leaving == k0:
            break
        entering = leaving
        tab = tQ if tab is tP else tP
    else:
        raise NoEquilibrium(f"Lemke-Howson from label {k0} exceeded {max_pivots} pivots")
    x, y = np.zeros(m), np.zeros(n)
    for r, lab in enumerate(tP.basis):
        if lab < m:
            x[lab] = tP.T[r][0] / tP.det
    for r, lab in enumerate(tQ.basis):
        if lab >= m:
            y[lab - m] = tQ.T[r][0] / tQ.det
    if x.sum() <= 0 or y.sum() <= 0:
        raise NoEquilibrium(f"Lemke-Howson from label {k0} ended at the artificial equilibrium")
    return x / x.sum(), y / y.sum()


def regrets(A, B, x, y):
    """(row regret, column regret) of (x, y); the row player maximises A, the column player maximises B."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.max(A @ y) - x @ A @ y), float(np.max(x @ B) - x @ B @ y)


def entropy_of(p) -> float:
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def solve(A, B, row_ids: Sequence[str], col_ids: Sequence[str], tol: float = 1e-6) -> dict:
    """Maximum-joint-entropy verified Nash equilibrium of the estimated restricted game. Row player = pursuer
    (payoff A), column player = evader (payoff B). Returns a JSON-serialisable record including every candidate."""
    A, B = _check(A, B)
    m, n = A.shape
    if len(row_ids) != m or len(col_ids) != n:
        raise ValueError("id lists do not match the payoff shape")
    if len(set(row_ids)) != m or len(set(col_ids)) != n:
        raise ValueError("duplicate policy ids")
    Ai, Bi = _to_int(A), _to_int(B)
    cands: List[dict] = []
    for k0 in range(m + n):
        try:
            x, y = _lemke_howson(Ai, Bi, m, n, k0)
        except NoEquilibrium as e:
            cands.append({"label": k0, "error": str(e), "verified": False})
            continue
        rr, rc = regrets(A, B, x, y)
        ok = bool(np.isfinite(x).all() and np.isfinite(y).all() and abs(x.sum() - 1) < 1e-9
                  and abs(y.sum() - 1) < 1e-9 and (x >= 0).all() and (y >= 0).all() and rr <= tol and rc <= tol)
        cands.append({"label": k0, "x": x.tolist(), "y": y.tolist(), "regret_row": rr, "regret_col": rc,
                      "verified": ok, "entropy": entropy_of(x) + entropy_of(y)})
    verified, seen = [], set()
    for c in cands:
        if c["verified"]:
            key = (tuple(np.round(c["x"], 9)), tuple(np.round(c["y"], 9)))
            if key not in seen:
                seen.add(key)
                verified.append(c)
    if not verified:
        raise NoEquilibrium(f"no verified equilibrium among {len(cands)} Lemke-Howson candidates")

    def support(c):
        return (tuple(sorted(row_ids[i] for i, v in enumerate(c["x"]) if v > 0)),
                tuple(sorted(col_ids[j] for j, v in enumerate(c["y"]) if v > 0)))

    best_h = max(c["entropy"] for c in verified)
    best = min((c for c in verified if c["entropy"] >= best_h - 1e-12), key=support)
    return {"solver": SOLVER_VERSION, "tol": tol, "row_ids": list(row_ids), "col_ids": list(col_ids),
            "x": dict(zip(row_ids, best["x"])), "y": dict(zip(col_ids, best["y"])),
            "selected_label": best["label"], "regret_row": best["regret_row"], "regret_col": best["regret_col"],
            "joint_entropy": best["entropy"], "n_verified_distinct": len(verified), "candidates": cands}


def support_enumeration(A, B, tol: float = 1e-9):
    """All equilibria of a small nondegenerate game by support enumeration (test reference; exponential)."""
    A, B = _check(A, B)
    m, n = A.shape
    out = []
    for k in range(1, min(m, n) + 1):
        for I in itertools.combinations(range(m), k):
            for J in itertools.combinations(range(n), k):
                rhs = np.zeros(k + 1)
                rhs[k] = 1
                My = np.zeros((k + 1, k + 1))
                My[:k, :k], My[:k, k], My[k, :k] = A[np.ix_(I, J)], -1, 1
                Mx = np.zeros((k + 1, k + 1))
                Mx[:k, :k], Mx[:k, k], Mx[k, :k] = B[np.ix_(I, J)].T, -1, 1
                try:
                    sy, sx = np.linalg.solve(My, rhs), np.linalg.solve(Mx, rhs)
                except np.linalg.LinAlgError:
                    continue
                if (sy[:k] < -tol).any() or (sx[:k] < -tol).any():
                    continue
                x, y = np.zeros(m), np.zeros(n)
                x[list(I)], y[list(J)] = sx[:k], sy[:k]
                rr, rc = regrets(A, B, x, y)
                if rr <= 1e-7 and rc <= 1e-7:
                    out.append((x, y))
    return out
