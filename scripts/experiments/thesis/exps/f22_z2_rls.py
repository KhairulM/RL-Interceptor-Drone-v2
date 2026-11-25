"""F22: F14 RLS-26 campaign (v2d rules, learned acceleration estimator + RLS) in a 2 m-ceiling arena (research plan F22).

Training only, seed 0. Identical to the F21 baseline / F14 Stage E RLS-26 configuration except task.arena.z_max=2.0.
"""

import os

from common import OUT, log_progress
from f2_final import run_campaign
import f11_v2_sp as f11
import f14_generalization_sp as f14
import f21_rls_ablation as f21

F22 = os.path.join(OUT, "F22_z2_rls")
SEED = 0
Z_MAX = "task.arena.z_max=2.0"
CAMPAIGN = f"thesis-F22-v2d-acc-rls-z2-s{SEED}"
SMOKE = "thesis-F22-smoke-z2"


def rules():
    return f14._campaign_rules("v2d") + [Z_MAX]


def roles():
    return f21.roles("baseline")


def run_f22(args):
    os.makedirs(F22, exist_ok=True)
    run_campaign(SMOKE, rules(), f11.cfg(SEED, 2), smoke_stages=2, role_overrides=roles())
    f14._wait_campaign(SMOKE)
    log_progress("F22 smoke passed (both roles trained, z_max 2.0)")
    if args.smoke:
        return
    run_campaign(CAMPAIGN, rules(), f11.cfg(SEED, 26), role_overrides=roles())
    f14._wait_campaign(CAMPAIGN)
    log_progress(f"F22 campaign {CAMPAIGN} complete (26 stages)")
