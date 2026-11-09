"""F5 Exp 1, post-hoc (descriptive, added 11-06 after the write-up): how do evaders beat the self-play pursuers?

Reads existing eval files only (no Isaac). Writes experiments/thesis/F5_acc_sp/outcomes_posthoc.md:
  1. evader wins (1 - J) split into pursuer crash / timeout, per evader group, final F5c pursuers;
  2. pursuer-crash vs capture episodes (length, altitudes, pursuer |vz|, min distance), clean; 2b. body-contact reach;
  3. stage by stage inside the campaigns (each stage's own eval vs its training opponents);
  4. control: pool-trained X3 generalists (same reward), crash vs timeout;
  5. J re-scored at a 0.15 m capture radius (eval-only; exact for the same policies: an episode is identical up to the
     first step with centre distance <= 0.15 m, where a 0.15 m radius would end it as a capture).
Run: env -u PYTHONPATH .venv/bin/python scripts/experiments/thesis/exps/f5_outcomes.py
"""

import collections
import json
import os
import sys

import numpy as np
import torch

sys.path[:0] = [os.path.dirname(__file__), os.path.dirname(os.path.dirname(__file__))]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
F5 = os.path.join(ROOT, "experiments", "thesis", "F5_acc_sp")
EV = os.path.join(F5, "evals")
X3 = os.path.join(ROOT, "experiments", "thesis", "X3_gap", "evals")
CAMPAIGNS = os.path.join(ROOT, "scripts", "selfplay", "outputs")
ARMS = {"uniform_rls": ["thesis-F5-uniform_rls-s0", "thesis-F5-uniform_rls-s1"],
        "uniform_plain": ["thesis-F2-plain", "thesis-F3-uniform-s1"], "uniform_fd": ["thesis-F2-acc"]}
GROUPS = ["own campaign", "other campaigns", "train pool RL (v9/v11)", "held-out RL (v10)", "X3 exploiters",
          "train pool scripted", "held-out scripted", "fresh exploiter"]
FILES = ("campaign_evaders", "train", "test", "vs_exploiter")
DT = 0.02


def group(pursuer, f, opp):
    if f == "vs_exploiter":
        return "fresh exploiter"
    c, s = opp.split("/")
    if f == "campaign_evaders":
        return None if s.endswith(":s1") else ("own campaign" if c == pursuer else "other campaigns")
    if f == "test":
        return {"v10": "held-out RL (v10)", "heur": "held-out scripted", "exploit": "X3 exploiters"}[c]
    return "train pool scripted" if c == "heur" else "train pool RL (v9/v11)"


def outcome_table(cond):
    lines = [f"| arm | evader group | evader wins (1 - J) | pursuer crash | timeout | capture |", "|---" * 6 + "|"]
    for arm, ps in ARMS.items():
        agg = collections.defaultdict(collections.Counter)
        for p in ps:
            for f in FILES:
                path = os.path.join(EV, f"{p}_{cond}_{f}.json")
                if not os.path.isfile(path):
                    continue
                for opp, m in json.load(open(path))["matchups"].items():
                    g = group(p, f, opp)
                    if g is None:
                        continue
                    n, o, a = m["episodes"], m["outcomes"], agg[g]
                    a["n"] += n
                    a["win"] += n * (1 - m["neutralization_rate"])
                    a["pc"] += n * o["pursuer_crash"]
                    a["to"] += n * o["timeout"]
                    a["cap"] += n * o["capture"]
        for g in GROUPS:
            if g in agg:
                a = agg[g]
                lines.append(f"| {arm} | {g} | {a['win'] / a['n']:.3f} | {a['pc'] / a['n']:.3f} | "
                             f"{a['to'] / a['n']:.3f} | {a['cap'] / a['n']:.3f} |")
    return lines


def episode_table():
    lines = ["| arm | evader group | pursuer crash % | length s (crash / capture) | evader altitude m | "
             "pursuer altitude m | pursuer abs(vz) m/s | min distance in crash episodes m |", "|---" * 8 + "|"]
    for arm, ps in list(ARMS.items())[:2]:
        rows = collections.defaultdict(list)
        for p in ps:
            for f in FILES:
                d = torch.load(os.path.join(EV, f"{p}_clean_{f}_episodes.pt"))
                E = d["episodes"]
                for opp, envs in d["assignment"].items():
                    g = group(p, f, opp)
                    if g is not None:
                        sel = torch.isin(E["env"], envs.cpu())
                        rows[g].append({k: v[sel] for k, v in E.items()})
        for g in GROUPS:
            if g not in rows:
                continue
            E = {k: torch.cat([r[k] for r in rows[g]]) for k in rows[g][0]}
            pc, cap, fm = (E["crash"] > .5) & (E["capture"] < .5), E["capture"] > .5, E["feat_mean"]

            def pair(x, scale=1.0):
                return f"{float(x[pc].mean()) * scale:.2f} / {float(x[cap].mean()) * scale:.2f}"
            # feat_mean: 9 evader features, then the same 9 for the pursuer (1 = altitude, 3 = |vz|)
            lines.append(f"| {arm} | {g} | {100 * float(pc.float().mean()):.1f} | {pair(E['episode_len'], DT)} | "
                         f"{pair(fm[:, 1])} | {pair(fm[:, 10])} | {pair(fm[:, 12])} | "
                         f"{float(E['min_distance'][pc].mean()):.2f} |")
    return lines


CONTACT_NOTE = ("Pursuer-crash episodes by closest sampled distance vs the body-contact reach. Collider "
                "(cf2x_isaac.usd): one 0.10 x 0.10 x 0.03 m box per drone, rotors without colliders, drones in one env "
                "not collision-filtered, so the bodies can touch without a capture at centre distances 0.10-0.145 m "
                "(box half-diagonal 0.072 m), depending on relative orientation. 'both' = the evader crashed in the "
                "same step.")


def contact_table():
    lines = ["| pursuer | evader file | pursuer-crash episodes | share within 0.145 m | both crash: 0.10-0.12 m / "
             "0.12-0.145 m / >= 0.145 m |", "|---" * 5 + "|"]
    for ps in list(ARMS.values())[:2]:
        for p in ps:
            for f in FILES:
                E = torch.load(os.path.join(EV, f"{p}_clean_{f}_episodes.pt"))["episodes"]
                pc = (E["crash"] > .5) & (E["capture"] < .5)
                both, d = pc & (E["opp_crash"] > .5), E["min_distance"]
                bins = [pc & (d >= lo) & (d < hi) for lo, hi in ((0.1, 0.12), (0.12, 0.145), (0.145, 1e9))]
                lines.append(f"| {p} | {f} | {int(pc.sum())} | {float((pc & (d < 0.145)).sum()) / int(pc.sum()):.2f} | "
                             + " / ".join(f"{float((both & b).sum()) / max(int(b.sum()), 1):.2f}" for b in bins) + " |")
    return lines


def stage_table(camp, n_stages=9):
    pools = {r: {p["id"]: (os.path.basename(p["checkpoint_path"]).split("_")[1].split(".")[0]
                           if p["checkpoint_path"] else p["metadata"]["heuristic"])
                 for p in json.load(open(os.path.join(CAMPAIGNS, camp, "db", f"{r}_pool.json")))["policies"]}
             for r in ("pursuer", "evader")}
    lines = [f"**{camp}**", "", "| stage | role | vs | capture | pursuer crash | evader crash + OOB | timeout |",
             "|---" * 7 + "|"]
    for s in range(n_stages):
        role = "pursuer" if s % 2 == 0 else "evader"
        other = "evader" if role == "pursuer" else "pursuer"
        m = json.load(open(os.path.join(CAMPAIGNS, camp, "stages", f"stage{s}_{role}_eval_out.json")))["matchups"]
        for opp, r in m.items():
            o = r["outcomes"]
            lines.append(f"| {s} | {role} | {other} {pools[other][opp]} | {o['capture']:.2f} | {o['pursuer_crash']:.2f} | "
                         f"{o['evader_crash'] + o['evader_oob']:.2f} | {o['timeout']:.2f} |")
    return lines + [""]


def generalist_table():
    lines = ["| generalist seed | evader group | evader wins | pursuer crash | timeout |", "|---" * 5 + "|"]
    for s in range(5):
        for part in ("train", "test"):
            agg = collections.defaultdict(collections.Counter)
            for opp, r in json.load(open(os.path.join(X3, f"generalist{s}_{part}.json")))["matchups"].items():
                a, n = agg[opp.split("/")[0]], r["episodes"]
                a["n"] += n
                a["win"] += n * (1 - r["neutralization_rate"])
                a["pc"] += n * r["outcomes"]["pursuer_crash"]
                a["to"] += n * r["outcomes"]["timeout"]
            for g, a in agg.items():
                lines.append(f"| {s} | {part} {g} | {a['win'] / a['n']:.3f} | {a['pc'] / a['n']:.3f} | "
                             f"{a['to'] / a['n']:.3f} |")
    return lines


RADII = (0.10, 0.15)
METRICS = ["heldout_rl", "heldout_worst_cluster", "other_campaign_evaders", "own_campaign_evaders", "train",
           "heldout_scripted", "x3_exploiters", "exploitability"]


def _episode_rates(radius):
    """Per-opponent J at `radius` from the per-episode files (pursuer evaluated: opp_* = evader)."""
    def rates(path, key):
        ep = path[:-len(".json")] + "_episodes.pt"
        if not os.path.isfile(ep):
            return {}
        d = torch.load(ep)
        E = d["episodes"]
        hit = (E["capture"] > .5) | (E["min_distance"] <= radius)
        if key == "neutralization_rate":
            hit = hit | (E["opp_crash"] > .5) | (E["opp_out_of_bounds"] > .5)
        return {o: float(hit[torch.isin(E["env"], envs.cpu())].float().mean()) for o, envs in d["assignment"].items()}
    return rates


def radius_table():
    import f5_acc_sp as f5
    from a_adapt import test_set
    from x3_gap import load_split
    split = load_split()
    test = test_set(split)
    labels = json.load(open(os.path.join(ROOT, "experiments", "thesis", "X2_descriptors", "clusters.json")))["labels"]
    rows = {}                                              # (radius, arm, cond) -> {k: metrics}
    orig = f5._rates
    try:
        for r in RADII:
            f5._rates = _episode_rates(r)
            for a, k, c, pov, _ in f5.active_pursuers():
                for cond in ("clean", "noisy"):
                    m = f5._metrics(os.path.join(EV, f"{c}_{cond}"), c, split, test, labels, "neutralization_rate")
                    if m:
                        rows.setdefault((r, a, cond), {})[k] = m
    finally:
        f5._rates = orig
    # consistency: radius 0.10 from episodes must equal the published per-seed values (results_exp1.json)
    pub = json.load(open(os.path.join(F5, "results_exp1.json")))["aggregate"]
    dev = max(abs(np.mean([v[m] for v in per.values() if m in v]) - pub[f"J/{cond}/{a}"][m]["mean"])
              for (r, a, cond), per in rows.items() if r == RADII[0] for m in METRICS
              if f"J/{cond}/{a}" in pub and m in pub[f"J/{cond}/{a}"])
    mean = lambda r, a, cond, m: np.mean([v[m] for v in rows.get((r, a, cond), {}).values() if m in v] or [np.nan])
    arms = list(dict.fromkeys(a for _, a, _ in rows))
    lines = [f"Consistency: J at 0.10 m from the episode files vs `results_exp1.json`: max |diff| {dev:.1e}.", "",
             "| metric | " + " | ".join(f"{a} {cond}" for cond in ("clean", "noisy") for a in arms) + " |",
             "|---" * (1 + 2 * len(arms)) + "|"]
    for m in METRICS:
        lines.append(f"| {m} | " + " | ".join(f"{mean(RADII[0], a, cond, m):.3f} -> {mean(RADII[1], a, cond, m):.3f}"
                                             for cond in ("clean", "noisy") for a in arms) + " |")
    pair = lambda r, cond, m: np.mean([rows[(r, "uniform_rls", cond)][k][m] - rows[(r, "uniform_plain", cond)][k][m]
                                       for k in rows[(r, "uniform_rls", cond)] if k in rows[(r, "uniform_plain", cond)]])
    lines += ["", "Seed-paired uniform_rls - uniform_plain at 0.10 m -> 0.15 m:", "",
              "| metric | clean | noisy |", "|---|---|---|"]
    for m in METRICS:
        lines.append(f"| {m} | " + " | ".join(f"{pair(RADII[0], cond, m):+.3f} -> {pair(RADII[1], cond, m):+.3f}"
                                             for cond in ("clean", "noisy")) + " |")
    lines += ["", "Control, pool-trained X3 generalists (clean; mean over opponents, 0.10 m -> 0.15 m):", "",
              "| generalist seed | held-out RL (v10) | held-out scripted | X3 exploiters | train RL (v9 / v11) |",
              "|---|---|---|---|---|"]
    for s in range(5):
        J = {(r, part): _episode_rates(r)(os.path.join(X3, f"generalist{s}_{part}.json"), "neutralization_rate")
             for r in RADII for part in ("train", "test")}

        def cell(part, pre):
            return " -> ".join(f"{np.mean([v for o, v in J[(r, part)].items() if o.split('/')[0] in pre]):.3f}"
                               for r in RADII)
        lines.append(f"| {s} | {cell('test', ('v10',))} | {cell('test', ('heur',))} | {cell('test', ('exploit',))} | "
                     f"{cell('train', ('v9', 'v11'))} |")
    return lines


def main():
    out = ["# F5 Exp 1: how evaders beat the self-play pursuers (post-hoc, descriptive)", "",
           "Generated by `scripts/experiments/thesis/exps/f5_outcomes.py` from existing eval files. Outcome flags are "
           "per episode and can overlap; an evader wins (not neutralised) only by a pursuer crash or a timeout. "
           "Other / own campaign evaders exclude stage-1 evaders. Rates pooled over seeds, weighted by episodes.", ""]
    for cond in ("clean", "noisy"):
        out += [f"## 1. Final pursuers, evader wins by outcome ({cond})", ""] + outcome_table(cond) + [""]
    out += ["## 2. Pursuer-crash vs capture episodes (clean; episode means)", ""] + episode_table() + [""]
    out += ["## 2b. Body-contact reach (clean; closest distance sampled at 50 Hz, physics at 100 Hz)", "",
            CONTACT_NOTE, ""] + contact_table() + [""]
    out += ["## 3. Inside the campaigns (each stage's eval vs its training opponents)", ""]
    for camps in (ARMS["uniform_rls"], ARMS["uniform_plain"]):
        for c in camps:
            out += stage_table(c)
    out += ["## 4. Control: pool-trained X3 generalists (same reward)", ""] + generalist_table() + [""]
    out += ["## 5. J at a 0.15 m capture radius (eval-only re-score; mean over opponents, then over seeds, as in "
            "the F5 summary)", ""] + radius_table() + [""]
    path = os.path.join(F5, "outcomes_posthoc.md")
    with open(path, "w") as f:
        f.write("\n".join(out))
    print(path)


if __name__ == "__main__":
    main()
