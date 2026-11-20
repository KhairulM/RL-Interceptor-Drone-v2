import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

_THESIS = Path(__file__).resolve().parents[1] / "scripts/experiments/thesis"
sys.path[:0] = [str(_THESIS), str(_THESIS / "exps")]
import f21_rls_ablation as f21  # noqa: E402


def _values(arm, freeze=False):
    return dict(o.split("=", 1) for o in f21.overrides(arm, freeze))


def test_registered_campaign_budget_and_parameter_isolation():
    assert len(f21.specs()) == 12
    assert len({f21.campaign(a, s) for a, s in f21.specs()}) == 12
    assert len(f21.specs()) * 26 * 50_000_000 == 15_600_000_000
    changed = {"task.adapt.forgetting", "task.adapt.delta", "task.adapt.p_max_ratio"}
    for arm in f21.PARAMS:
        lam, delta, cap = f21.PARAMS[arm]
        ov = _values(arm)
        assert tuple(float(ov[k]) for k in ("task.adapt.forgetting", "task.adapt.delta",
                                          "task.adapt.p_max_ratio")) == (lam, delta, cap)
        assert {k: v for k, v in ov.items() if k not in changed} == {
            k: v for k, v in _values("baseline").items() if k not in changed}
        assert len(f21.overrides(arm)) == len(ov)
        assert ov["task.adapt.freeze"] == "false"
        assert ov["task.adapt.reset_on_episode"] == "true"
    assert _values("frozen_train")["task.adapt.freeze"] == "true"


@pytest.mark.parametrize("arm", f21.NEW)
def test_both_roles_have_matching_parameters_and_evader_stays_plain(arm):
    roles = f21.roles(arm)
    pov, opp = [dict(o.split("=", 1) for o in roles[k]) for k in ("pursuer", "evader")]
    assert pov.pop("task.adapt.enabled") == "true"
    assert opp.pop("task.adapt.opp_side") == "true"
    assert "task.adapt.enabled" not in opp
    assert pov == opp
    for seed in f21.SEEDS:
        cfg = f21.f11.cfg(seed, 26)
        assert (cfg["n_stages"], cfg["max_total_stages"], cfg["frames_per_stage"]) == (26, 26, 50_000_000)
        assert cfg["pursuer_sampler"] == cfg["evader_sampler"] == "hedge"


def test_lanes_partition_training_and_evaluation_consistently(monkeypatch):
    lane_sets = []
    for lane in ("0", "1"):
        monkeypatch.setenv("F21_LANE", lane)
        train = set(f21.lane_specs())
        evaluation = set(f21.lane_specs(f21.ARMS))
        assert len(train) == 6
        assert train <= evaluation
        lane_sets.append(evaluation)
    assert not lane_sets[0] & lane_sets[1]
    assert lane_sets[0] | lane_sets[1] == set(f21.specs(f21.ARMS))
    monkeypatch.setenv("F21_LANE", "bad")
    with pytest.raises(ValueError):
        f21.lane_specs()


def test_uses_campaign_runner_not_fixed_pool_training():
    with patch.object(f21, "audit"), patch.object(f21, "_smoke"), \
            patch.object(f21, "lane_specs", return_value=[("fast", 100)]), \
            patch.object(f21.f14, "_campaign_rules", return_value=["v2d"]), \
            patch.object(f21.f14, "_wait_campaign"), patch.object(f21, "run_campaign") as run:
        f21.run_f21(SimpleNamespace(seeds=3, smoke=False))
    args, kwargs = run.call_args
    assert args[:2] == ("thesis-F21-v2d-fast-s100", ["v2d"])
    assert args[2]["seed"] == 100
    assert kwargs["role_overrides"] == f21.roles("fast")
    assert "smoke_stages" not in kwargs


def test_failed_smoke_blocks_main_campaigns():
    with patch.object(f21, "audit"), patch.object(f21, "_smoke", side_effect=RuntimeError("gate failed")), \
            patch.object(f21, "run_campaign") as run:
        with pytest.raises(RuntimeError, match="gate failed"):
            f21.run_f21(SimpleNamespace(seeds=3, smoke=False))
    run.assert_not_called()


def test_smoke_trains_both_roles_for_every_arm_and_evaluates_live_frozen(tmp_path):
    with patch.object(f21, "F21", str(tmp_path)), patch.object(f21, "run_campaign") as run, \
            patch.object(f21.f14, "_wait_campaign"), patch.object(f21.f14, "_ck", return_value="p.pt"), \
            patch.object(f21.f14, "_campaign_rules", return_value=["v2d"]), \
            patch.object(f21.f8, "all_stages", return_value=[{"id": "evader"}]), \
            patch.object(f21, "_wait"):
        def fake_eval(path, *args, **kwargs):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text('{"matchups": {"evader": {"episodes": 10}}}')

        with patch.object(f21, "evaluate", side_effect=fake_eval) as evaluate:
            f21._smoke()
    assert run.call_count == 4
    assert evaluate.call_count == 8
    for arm, call in zip(f21.NEW, run.call_args_list):
        assert call.kwargs == {"smoke_stages": 2, "role_overrides": f21.roles(arm)}
    for i, call in enumerate(evaluate.call_args_list):
        assert ("task.adapt.freeze=true" in call.kwargs["extra"]) == bool(i % 2)


def test_baseline_and_frozen_campaigns_are_reused():
    for arm, key in (("baseline", "rls"), ("frozen_train", "no_rls")):
        for seed in f21.SEEDS:
            assert f21.campaign(arm, seed) == f21.f14._stagee_campaigns("v2d")[key][f21.SEEDS.index(seed)]
            assert f21._reused_general(arm, seed, "lure").endswith("_stageE_thesis.json")


def test_cross_bank_is_shared_and_excludes_all_estimator_lineages():
    requested = []
    with patch.object(f21.f14, "_evaders", side_effect=lambda camps: requested.append(camps) or [{"id": "common"}]):
        assert f21.common_cross() == [{"id": "common"}]
    assert requested == [tuple(f21.f19.CAMPS[s] for s in f21.SEEDS)]
    assert not set(requested[0]) & {f21.campaign(a, s) for a, s in f21.specs(f21.ARMS)}


def test_exploiters_receive_the_matching_mirrored_adapter_and_unique_seeds():
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    with patch.object(f21.f14, "_ck", return_value="p.pt"), \
            patch.object(f21.f14, "_campaign_rules", return_value=["v2d"]), \
            patch.object(f21.f14, "_exploiter", side_effect=fake):
        for arm, seed in f21.specs():
            f21._exploiters(arm, seed)
    assert len({args[5] for args, _ in calls}) == len(calls) == 24
    for (arm, _), pair in zip(f21.specs(), [calls[i:i + 2] for i in range(0, len(calls), 2)]):
        assert pair[0][0][4] == ["v2d"]
        assert pair[1][0][4] == f21.f11.OLD_RULES_EXPLOITER
        assert all(kw["opponent_extra"] == f21.f5._mirror(f21.overrides(arm)) for _, kw in pair)


def test_switches_match_f15_and_frozen_eval_does_not_change_checkpoint():
    def pair(seed, direction):
        a = {"id": "f8/e", "kind": "rl"}
        b = {"id": "f10/e", "kind": "rl"}
        return (a, b) if direction == "f8_to_f10" else (b, a)

    with patch.object(f21.f14, "_ck", return_value="p.pt"), patch.object(f21.f14, "_switch_pair", side_effect=pair):
        ref = f21.f15._baseline_jobs()
        for arm, seed in f21.specs():
            for freeze in (False, True):
                jobs = f21.switch_jobs(arm, seed, freeze)
                assert len(jobs) == 4
                assert len({j["path"] for j in jobs}) == 4
                for j, b in zip(jobs, [x for x in ref if x["seed"] == seed]):
                    assert j["checkpoint"] == "p.pt"
                    assert (j["rng_seed"], j["policies"], j["switch"]) == (b["rng_seed"], b["policies"], b["switch"])
                    assert _values(arm, freeze)["task.adapt.freeze"] == str(freeze).lower()


def test_error_bins_exclude_targets_straddling_switch_and_reset():
    ages = torch.arange(20)
    steps = torch.full((20,), 5)
    mask = f21.error_window_mask(ages, steps, 10, 0, 0.1, 0.02)
    assert mask.nonzero().flatten().tolist() == [15, 16, 17, 18, 19]
    assert not f21.error_window_mask(torch.tensor([0, 1, 2]), torch.tensor([5, 5, 5]),
                                     10, 0, 0.5, 0.02).any()


def test_replay_resets_pending_targets_and_reports_missing_windows(tmp_path):
    oa = f21.f18._adapter_module()
    model = tmp_path / "basis.pt"
    net = oa.BasisNet(9, d=2, hidden=4)
    torch.save({"z_mean": torch.zeros(9), "z_std": torch.ones(9), "in_dim": 9, "d": 2, "hidden": 4,
                "phi": net.state_dict(), "theta0": torch.zeros(3, 3), "horizon": 2, "history": 1,
                "target": "dvel"}, model)
    z = torch.randn(20, 1, 9)
    done = torch.zeros(19, 1, dtype=torch.bool)
    done[9] = True
    with patch.object(f21.f18, "_rls_params", return_value={"history": 1, "horizon": 2, "model_path": str(model)}):
        out = f21.replay_arrays(z, done, torch.tensor([3]), 0.02, "fast")
    for mode in ("live", "frozen"):
        windows = out[mode]["error_windows"]
        assert windows[0]["count"] == 10  # origins 3..7 in each episode; not across the reset
        assert windows[1]["count"] == 0
        assert windows[1]["mean_mps"] is None
        assert out[mode]["snapshots"]["0.5"]["drift_initial_median"] is None


def test_contrast_uses_training_seeds_and_exact_threshold():
    assert f21.contrast([0.05, 0.05, 0.05])["verdict"] == "notable_increase"
    assert f21.contrast([-0.05, -0.05, -0.05])["verdict"] == "notable_decrease"
    assert f21.contrast([0.2, 0.2, -0.001])["verdict"] == "inconclusive"
    assert f21.contrast([0.01, 0.02, 0.03])["verdict"] == "inconclusive"


def test_analysis_writes_complete_shapes_and_interaction(tmp_path):
    (tmp_path / "evals").mkdir()
    (tmp_path / "audit.json").write_text("{}")
    for arm in f21.PARAMS:
        for seed in f21.SEEDS:
            (tmp_path / "evals" / f"{arm}_s{seed}_diagnostics.json").write_text(json.dumps({
                direction: {"live": {"snapshots": {"0.5": {"drift_switch_median": 0.1}}}}
                for direction in ("f8_to_f10", "f10_to_f8")}))
    metrics = ("g1", "own_j", "lure_j", "post_switch_capture", "post_switch_neutralization",
               "live_minus_frozen_post_switch_capture")
    with patch.object(f21, "F21", str(tmp_path)), \
            patch.object(f21, "score", side_effect=lambda a, s, r: dict.fromkeys(metrics, f21.ARMS.index(a) * 0.1)):
        f21.analyse()
    result = json.loads((tmp_path / "results.json").read_text())
    assert set(result["r0.10"]["scores"]) == set(f21.ARMS)
    assert len(result["r0.10"]["contrasts"]) == 5
    assert result["r0.10"]["lambda_cap_interaction"]["g1"]["mean"] == pytest.approx(0)
    assert set(result["predictions"]) == {"P1", "P2", "P3", "P4"}
    assert (tmp_path / "summary.md").is_file()
