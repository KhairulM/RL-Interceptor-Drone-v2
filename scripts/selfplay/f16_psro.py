"""Thesis F16: PSRO / PSD-PSRO generation orchestrator (docs/research_plan.md, F16 A.4 and C).

Starts from a bootstrap campaign directory that already holds exactly the stage-0 pursuer and stage-1 evader (cloned
from F8b by the thesis driver). Each generation g = 1..G:
  1. complete the restricted game: every (pursuer, evader) cell of the current RL populations gets >= min_episodes
     episodes of both roles' discounted Roncero returns (stochastic actors, append-only payoff store);
  2. solve it once (maximum-entropy verified general-sum Nash, meta_solver.py) and freeze the result in
     meta/gen{g}.json;
  3. train the pursuer oracle (stage 2g) against the frozen evader meta-strategy, then the evader oracle (stage 2g+1)
     against the frozen pursuer meta-strategy (pre-generation populations only; one opponent drawn per episode reset).
     arm=psd adds the own-policy-hull KL objective vs that role's pre-generation policies (run_stage hull_kl.*);
  4. add both policies.
After G generations the final game is completed and solved into meta/final.json (the deployed pursuer mixture).
Every step is idempotent: re-running resumes without duplicating policies, payoff records or meta-strategies.

Usage: python scripts/selfplay/f16_psro.py run_name=... arm=psd seed=0 hull.lam=0.1 task_overrides=[...]
"""

import importlib.util
import json
import math
import os
import sys
import time

from omegaconf import OmegaConf

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ts = _load("scripts/selfplay/train_selfplay.py", "f16_train_selfplay")
_ms = _load("omni_drones/utils/selfplay/meta_solver.py", "f16_meta_solver")
_ps = _load("omni_drones/utils/selfplay/payoff_store.py", "f16_payoff_store")
PURSUER, EVADER, PolicyDatabase = _ts.PURSUER, _ts.EVADER, _ts.PolicyDatabase

DEFAULTS = {
    "run_name": None, "out_root": None, "arm": "psro", "seed": 0, "generations": 12, "frames_per_stage": 50_000_000,
    "num_envs": 4096, "headless": True, "task": "PursuitEvasion", "algo": "ppo", "task_overrides": [],
    "warm_start": True, "worker_retries": 2, "python": None, "wandb_mode": "disabled",
    "payoff_min_episodes": 2048, "payoff_num_envs": 4096, "payoff_seed_base": 7_000_000,
    "max_episode_steps": 1500, "solver_tol": 1e-6,
    "hull": {"lam": 0.0, "n_interior": 16, "n_check": 64, "mc_samples": 8},
}


def rl_policies(db, role):
    return sorted((p for p in db.list_policies(role) if p["kind"] == "rl"), key=lambda p: _ts._policy_stage(p))


def cutoff_steps(num_envs, n_opponents, min_episodes, max_len):
    """Every env starts an episode at least every max_len steps, so counting starts before k * max_len gives >= k
    episodes per env; policy_sampling gives each of n opponents >= floor(num_envs / n) envs."""
    per = num_envs // n_opponents
    if per < 1:
        raise ValueError("more opponents than envs")
    return int(math.ceil(min_episodes / per)) * int(max_len)


def next_payoff_job(missing):
    """Greedy: the agent (pursuer row or evader column) with the most missing cells; ties -> pursuer, then order."""
    if not missing:
        return None
    rows, cols = {}, {}
    for p, e in missing:
        rows.setdefault(p, []).append(e)
        cols.setdefault(e, []).append(p)
    best_p = max(rows, key=lambda k: (len(rows[k]), -list(rows).index(k)))
    best_e = max(cols, key=lambda k: (len(cols[k]), -list(cols).index(k)))
    if len(rows[best_p]) >= len(cols[best_e]):
        return PURSUER, best_p, rows[best_p]
    return EVADER, best_e, cols[best_e]


class Campaign:
    def __init__(self, cfg, worker=None):
        self.cfg = cfg
        self.out = os.path.join(cfg.out_root or os.path.join(_THIS, "outputs"), str(cfg.run_name))
        self.db = PolicyDatabase(os.path.join(self.out, "db"))
        self.ckpt_dir = os.path.join(self.out, "checkpoints")
        self.work = os.path.join(self.out, "stages")
        self.meta_dir = os.path.join(self.out, "meta")
        self.store = _ps.PayoffStore(os.path.join(self.out, "payoffs"))
        for d in (self.ckpt_dir, self.work, self.meta_dir):
            os.makedirs(d, exist_ok=True)
        self.worker = worker or (lambda overrides: _ts._run_worker(self.cfg, overrides))
        if str(cfg.arm) not in ("psro", "psd"):
            raise ValueError(f"unknown arm {cfg.arm}")
        if str(cfg.arm) == "psd" and float(cfg.hull.lam) <= 0:
            raise ValueError("arm=psd needs hull.lam > 0")
        if str(cfg.arm) == "psro" and float(cfg.hull.lam) != 0:
            raise ValueError("arm=psro has no diversity term (hull.lam must be 0)")

    # ------------------------------------------------------------------ helpers
    def _sha(self, policy):
        md = policy.get("metadata") or {}
        return md.get("sha256") or _ps.sha256_file(policy["checkpoint_path"])

    def _check_bootstrap(self):
        ps, es = rl_policies(self.db, PURSUER), rl_policies(self.db, EVADER)
        if not ps or not es or _ts._policy_stage(ps[0]) != 0 or _ts._policy_stage(es[0]) != 1:
            raise RuntimeError("F16 campaign needs the cloned stage-0 pursuer / stage-1 evader bootstrap")
        for p in ps + es:
            md = p.get("metadata") or {}
            if md.get("sha256") and _ps.sha256_file(p["checkpoint_path"]) != md["sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch for {p['id']} (corrupt state)")

    def _overrides_common(self, role):
        c = self.cfg
        return [f"task={c.task}", f"algo={c.algo}", f"task.role={role}", f"headless={bool(c.headless)}",
                f"wandb.mode={c.wandb_mode}"] + list(c.task_overrides)

    # ------------------------------------------------------------------ payoffs
    def complete_game(self):
        while True:
            ps = [p["id"] for p in rl_policies(self.db, PURSUER)]
            es = [p["id"] for p in rl_policies(self.db, EVADER)]
            job = next_payoff_job(self.store.missing(ps, es, int(self.cfg.payoff_min_episodes)))
            if job is None:
                return ps, es
            self._payoff_eval(*job)

    def _payoff_eval(self, role, agent_id, opp_ids):
        c = self.cfg
        agent = self.db.get_policy(role, agent_id)
        opp_role = _ts._other(role)
        opps = [self.db.get_policy(opp_role, o) for o in opp_ids]
        stage = _ts._policy_stage(agent)
        attempt = len({r["run_id"] for r in self.store.records() if r.get("agent_id") == agent_id})
        run_id = f"{agent_id}#run{attempt}"
        seed = int(c.payoff_seed_base) + int(c.seed) * 10_000 + stage * 10 + attempt
        cutoff = cutoff_steps(int(c.payoff_num_envs), len(opps), int(c.payoff_min_episodes), int(c.max_episode_steps))
        base = os.path.join(self.out, "payoffs", "runs", f"{agent_id.replace(':', '_')}_run{attempt}")
        spec = {"policies": [{"id": o["id"], "kind": "rl", "checkpoint_path": o["checkpoint_path"],
                              "metadata": o.get("metadata", {})} for o in opps],
                "probs": {o["id"]: 1.0 / len(opps) for o in opps}, "stochastic": True}
        _ts._write_json(base + "_opp.json", spec)
        out = base + "_out.json"
        if not os.path.isfile(out):
            self.worker(["mode=eval"] + self._overrides_common(role) + [
                f"task.env.num_envs={int(c.payoff_num_envs)}", f"seed={seed}", f"opponent_spec_path={base}_opp.json",
                f"eval_policy={agent['checkpoint_path']}", f"stage_out={out}", f"eval_cutoff_steps={cutoff}",
                "eval_stochastic=true", "task.payoff_returns=true", f"eval_episodes_out={base}_episodes.pt"])
        with open(out) as f:
            matchups = json.load(f)["matchups"]
        prov = {"agent_id": agent_id, "agent_sha256": self._sha(agent), "cutoff_steps": cutoff,
                "num_envs": int(c.payoff_num_envs), "task_overrides": list(c.task_overrides),
                "opponent_sha256": {o["id"]: self._sha(o) for o in opps}, "time": time.strftime("%F %T")}
        self.store.append(_ps.records_from_matchups(matchups, agent_id, role, run_id, seed, prov))
        for opp_id, m in matchups.items():
            pair = (agent_id, opp_id) if role == PURSUER else (opp_id, agent_id)
            self.db.record_matchup(*pair, m)
        print(f"[payoff] {run_id}: {len(matchups)} cells, cutoff {cutoff}")

    # ------------------------------------------------------------------ meta-strategies
    def meta(self, tag, ps, es):
        path = os.path.join(self.meta_dir, f"{tag}.json")
        if os.path.isfile(path):
            with open(path) as f:
                m = json.load(f)
            if m["row_ids"] != ps or m["col_ids"] != es:
                raise RuntimeError(f"frozen meta-strategy {tag} was solved for a different population")
            return m
        UP, UE, N = self.store.matrices(ps, es, int(self.cfg.payoff_min_episodes))
        sol = _ms.solve(UP, UE, ps, es, tol=float(self.cfg.solver_tol))
        sol.update(U_pursuer=UP.tolist(), U_evader=UE.tolist(), episodes=N.tolist(), tag=tag,
                   time=time.strftime("%F %T"))
        _ts._write_json(path, sol)
        print(f"[meta] {tag}: pursuer {sol['x']} evader {sol['y']} (H={sol['joint_entropy']:.3f})")
        return sol

    # ------------------------------------------------------------------ oracles
    def oracle(self, role, stage, opp_probs, gen):
        c = self.cfg
        ckpt = os.path.join(self.ckpt_dir, f"{role}_stage{stage}.pt")
        train_out = os.path.join(self.work, f"stage{stage}_{role}_train_out.json")
        existing = [p for p in rl_policies(self.db, role) if _ts._policy_stage(p) == stage]
        if existing:
            return existing[0]
        own = rl_policies(self.db, role)                       # pre-generation population of this role
        if not (os.path.isfile(ckpt) and os.path.isfile(train_out)):
            opp_role = _ts._other(role)
            support = {k: float(v) for k, v in opp_probs.items() if v > 0}
            spec = _ts._pool_to_worker_spec(self.db, opp_role, support)
            spec["resample"] = {"seed": int(c.seed) * 1000 + stage}
            spec_path = os.path.join(self.work, f"stage{stage}_{role}_train_opp.json")
            _ts._write_json(spec_path, spec)
            ov = ["mode=train"] + self._overrides_common(role) + [
                f"task.env.num_envs={int(c.num_envs)}", f"total_frames={int(c.frames_per_stage)}",
                f"seed={int(c.seed) + stage}", f"stage={stage}", f"wandb.run_name={c.run_name}-{role}-stage{stage}",
                f"opponent_spec_path={spec_path}", f"stage_ckpt={ckpt}", f"stage_out={train_out}"]
            if bool(c.warm_start):
                ov.append(f"init_policy={own[-1]['checkpoint_path']}")
            if str(c.arm) == "psd":
                h = c.hull
                ov += [f"hull_kl.ckpts=[{','.join(p['checkpoint_path'] for p in own)}]", f"hull_kl.lam={float(h.lam)}",
                       f"hull_kl.n_interior={int(h.n_interior)}", f"hull_kl.n_check={int(h.n_check)}",
                       f"hull_kl.seed={int(c.seed) * 1000 + stage}", f"hull_kl.mc_samples={int(h.mc_samples)}",
                       f"hull_kl.log_path={os.path.join(self.work, f'stage{stage}_{role}_hull.jsonl')}"]
            self.worker(ov)
        with open(train_out) as f:
            path = json.load(f)["checkpoint_path"]
        pid = self.db.add_policy(role, path, kind="rl", metadata={"stage": stage, "generation": gen,
                                                                   "sha256": _ps.sha256_file(path)})
        print(f"[db] {pid} <- {path}")
        return self.db.get_policy(role, pid)

    # ------------------------------------------------------------------ main loop
    def run(self):
        self._check_bootstrap()
        for g in range(1, int(self.cfg.generations) + 1):
            sp, se = 2 * g, 2 * g + 1
            have_p = any(_ts._policy_stage(p) == sp for p in rl_policies(self.db, PURSUER))
            have_e = any(_ts._policy_stage(p) == se for p in rl_policies(self.db, EVADER))
            if have_p and have_e:
                continue
            # The frozen pre-generation populations (exclude a pursuer already added in an interrupted generation).
            ps = [p["id"] for p in rl_policies(self.db, PURSUER) if _ts._policy_stage(p) < sp]
            es = [p["id"] for p in rl_policies(self.db, EVADER) if _ts._policy_stage(p) < sp]
            path = os.path.join(self.meta_dir, f"gen{g}.json")
            if not os.path.isfile(path):
                if have_p:
                    raise RuntimeError(f"generation {g}: pursuer exists but its meta-strategy does not")
                ps, es = self.complete_game()
            m = self.meta(f"gen{g}", ps, es)
            print(f"\n===== generation {g}: stages {sp} (pursuer) / {se} (evader) =====")
            self.oracle(PURSUER, sp, m["y"], g)
            self.oracle(EVADER, se, m["x"], g)
        ps, es = self.complete_game()
        final = self.meta("final", ps, es)
        print("[done] final pursuer mixture:", final["x"])
        return final


def load_cfg(argv=None):
    cfg = OmegaConf.merge(OmegaConf.create(DEFAULTS), OmegaConf.from_cli(argv if argv is not None else sys.argv[1:]))
    if not cfg.run_name:
        raise ValueError("run_name is required")
    return cfg


def main(argv=None, worker=None):
    return Campaign(load_cfg(argv), worker).run()


if __name__ == "__main__":
    main()
