"""Pursuer distance shaping for PursuitEvasion (no Isaac imports, so it is unit-tested: tests/test_shaping.py).

gamma None (thesis rules before 2026-11-11): plain differences, approach w_app (d_prev - d) and precision
w_prec (exp(-c d) - exp(-c d_prev)). These equal Phi(s') - Phi(s) and so carry a hidden (1 - gamma) Phi(s') per step
under a discounted return.
gamma set (thesis v2): potential-based shaping F = gamma Phi(s') - Phi(s) (Ng, Harada & Russell, ICML 1999) with
Phi_app = -w_app d and Phi_prec = w_prec exp(-c d), and Phi(s') = 0 when the episode terminated (capture), so the
discounted shaped return telescopes to -Phi(s_0) and the optimal policy is unchanged. Truncations keep Phi(s')
(the value target bootstraps there).
"""

import torch


def shaping(prev_d, d, w_app, w_prec, scale, gamma=None, terminated=None):
    """(approach, precision) per-step shaping rewards [N, 1] for distances prev_d -> d [N, 1]."""
    e_prev, e = torch.exp(-scale * prev_d), torch.exp(-scale * d)
    if gamma is None:
        return w_app * (prev_d - d), w_prec * (e - e_prev)
    keep = 1.0 - terminated.float()                     # Phi(terminal state) = 0
    approach = w_app * prev_d - gamma * keep * w_app * d          # gamma * (-w_app d') - (-w_app d)
    precision = gamma * keep * w_prec * e - w_prec * e_prev
    return approach, precision
