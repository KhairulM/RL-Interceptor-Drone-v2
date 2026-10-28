"""X0_replay: do stored self-play policies replay correctly in the current env?

For each campaign, the final pursuer is re-evaluated against 3 RL evaders it was evaluated against during self-play
(earliest, middle, latest). The new multi-episode estimate is compared with the stored matchups.json value. For v11 a
classic first-episode eval (cutoff=1) is also run, to check that the multi-episode estimator agrees with it.
"""

import math
import os

from common import (CAMPAIGNS, OUT, campaign_db, evaluate, log_progress, rl_policies, save_json)

EXP = os.path.join(OUT, "X0_replay")
TOL = 0.05


def _pick_pairs(tag):
    pursuers = rl_policies(tag, "pursuer")
    evaders = rl_policies(tag, "evader")
    _, _, matchups = campaign_db(tag)
    final = pursuers[-1]
    by_db = {e["metadata"]["db_id"]: e for e in evaders}
    recorded = [(eid, m) for key, m in matchups.items()
                for pid, eid in [key.split("|")] if pid == final["metadata"]["db_id"] and eid in by_db]
    recorded.sort(key=lambda x: by_db[x[0]]["metadata"]["stage"])
    # Older schemas only store the capture fraction under "outcomes".
    for _, m in recorded:
        m.setdefault("capture_rate", (m.get("outcomes") or {}).get("capture"))
    if not recorded:
        return final, []
    idx = sorted({0, len(recorded) // 2, len(recorded) - 1})
    return final, [(by_db[recorded[i][0]], recorded[i][1]) for i in idx]


def run(args):
    rows = []
    for tag in CAMPAIGNS:
        final, pairs = _pick_pairs(tag)
        if not pairs:
            log_progress(f"X0 {tag}: no recorded RL matchups for final pursuer {final['id']}")
            continue
        opps = [e for e, _ in pairs]
        log_progress(f"X0 {tag}: {final['id']} vs {[e['id'] for e in opps]}")
        res = evaluate(os.path.join(EXP, tag, "multi.json"), opps, policy_ckpt=final["checkpoint_path"])
        first = None
        if tag == "v11":
            first = evaluate(os.path.join(EXP, tag, "first.json"), opps, policy_ckpt=final["checkpoint_path"],
                             cutoff=1, save_episodes=False)
        for e, stored in pairs:
            new = (res or {}).get("matchups", {}).get(e["id"])
            row = {"campaign": tag, "pursuer": final["id"], "evader": e["id"],
                   "stored_capture": stored.get("capture_rate"), "stored_episodes": stored.get("episodes"),
                   "stored_outcomes": stored.get("outcomes"), "ok": False}
            if new is not None:
                n0, n1 = stored.get("episodes") or 1, new["episodes"]
                p0, p1 = stored["capture_rate"], new["capture_rate"]
                se = math.sqrt(max(p0 * (1 - p0), 1e-4) / n0 + max(p1 * (1 - p1), 1e-4) / n1)
                row.update({"new_capture": p1, "new_neutralization": new["neutralization_rate"],
                            "new_episodes": n1, "new_outcomes": new["outcomes"],
                            "delta_capture": p1 - p0, "z": (p1 - p0) / se, "ok": abs(p1 - p0) <= TOL})
            if first is not None and e["id"] in first.get("matchups", {}):
                f = first["matchups"][e["id"]]
                row.update({"first_capture": f["capture_rate"], "first_episodes": f["episodes"]})
            rows.append(row)
    save_json(os.path.join(EXP, "replay_check.json"), rows)
    _write_summary(rows)


def _write_summary(rows):
    lines = ["# X0_replay summary", "",
             f"Final pursuer of each campaign vs 3 stored RL evaders; |delta capture| <= {TOL} counts as reproduced.",
             "", "| campaign | evader | stored cap (n) | new cap (n) | delta | z | new neut | first-ep cap (n) | ok |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "new_capture" not in r:
            lines.append(f"| {r['campaign']} | {r['evader']} | {r['stored_capture']:.3f} ({r['stored_episodes']}) "
                         f"| FAILED | | | | | no |")
            continue
        fc = f"{r['first_capture']:.3f} ({r['first_episodes']})" if "first_capture" in r else "-"
        lines.append(f"| {r['campaign']} | {r['evader']} | {r['stored_capture']:.3f} ({r['stored_episodes']}) | "
                     f"{r['new_capture']:.3f} ({r['new_episodes']}) | {r['delta_capture']:+.3f} | {r['z']:+.1f} | "
                     f"{r['new_neutralization']:.3f} | {fc} | {'yes' if r['ok'] else 'no'} |")
    with open(os.path.join(EXP, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
