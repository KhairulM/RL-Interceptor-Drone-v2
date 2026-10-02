import os, math
import hydra, torch
from omegaconf import OmegaConf
from omni_drones import init_simulation_app

FILE_PATH = os.path.dirname(__file__)

@hydra.main(config_path=FILE_PATH, config_name="evaluate", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg); OmegaConf.set_struct(cfg, False)
    NENV = 64
    cfg.eval.num_envs=NENV; cfg.task.env.num_envs=NENV; cfg.env.num_envs=NENV
    sim_app = init_simulation_app(cfg)
    import evaluate as E

    scen = OmegaConf.create({"name":"linear","trajectory_types":["linear"],
                             "speed_range":[0.1,5.0],"spawn_distance_range":[0.5,7.0]})
    run_cfg = E._apply_scenario(cfg, scen)
    run_cfg.eval.num_envs=NENV; run_cfg.task.env.num_envs=NENV; run_cfg.env.num_envs=NENV
    run_cfg.seed=0
    base_env, env = E._build_env(run_cfg); base_env.eval(); env.eval()

    for method in ["pure_pursuit","pn"]:
        policy = E._build_classical_policy(method, run_cfg, base_env)
        td = env.reset()
        print(f"\n=== {method} on LINEAR (env0 geometry) ===")
        print("  step  dist  p_speed  e_speed  closing  lead_angle_deg  (angle between LOS and evader-vel; ~180=chasing tail)")
        for step in range(300):
            td = policy(td)
            out, td = env.step_and_maybe_reset(td)
            nxt = out["next"]; ds = nxt[("info","drone_state")].reshape(-1,13)
            ev = base_env.evader.get_state()[...,:13].reshape(-1,13)
            ppos=ds[:,0:3]; pvel=ds[:,7:10]; epos=ev[:,0:3]; evel=ev[:,7:10]
            rel = epos-ppos; dist = rel.norm(dim=-1)
            los = rel/dist.clamp_min(1e-6).unsqueeze(-1)
            closing = -((evel-pvel)*los).sum(-1)  # >0 closing
            e_spd = evel.norm(dim=-1); p_spd = pvel.norm(dim=-1)
            # angle between LOS (pursuer->evader) and evader velocity direction
            evd = evel/e_spd.clamp_min(1e-6).unsqueeze(-1)
            cosang = (los*evd).sum(-1).clamp(-1,1)
            ang = torch.rad2deg(torch.arccos(cosang))
            if step % 30 == 0:
                print(f"  {step:4d}  {dist[0]:.2f}   {p_spd[0]:.2f}    {e_spd[0]:.2f}    {closing[0]:+.2f}     {ang[0]:.0f}")
    base_env.close(); sim_app.close()

if __name__=="__main__":
    main()
