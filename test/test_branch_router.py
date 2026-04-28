import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from branch_router import (  # noqa: E402
    build_branch_candidates,
    build_branch_snapshots,
    hedge_weight_trace,
    rank_blend_scores,
    route_branch_v1,
    route_branch_v2a,
    route_branch_v2b_overlay,
)
from branch_router_diagnostics import aggregate_guard_summary, paired_delta_distribution, swap_delta_reconciliation  # noqa: E402
from branch_router_validation import build_sweep_grid  # noqa: E402
from branch_router_v2c_shadow import _riskoff_candidates  # noqa: E402
from branch_router_v2d_riskoff_shadow import assert_runtime_unchanged, evaluate_v2d_window  # noqa: E402
from config import config as PROJECT_CONFIG  # noqa: E402


def _frame(prefix: str, scores=None, sigma=0.2, amp=0.2, drawdown=0.2):
    scores = scores or [1.0, 0.9, 0.8, 0.7, 0.6, 0.1, 0.0]
    n = len(scores)
    df = pd.DataFrame(
        {
            "stock_id": [f"{prefix}{i:03d}" for i in range(n)],
            "branch_score": scores,
            "sigma20": [sigma] * n,
            "amp20": [amp] * n,
            "max_drawdown20": [drawdown] * n,
            "ret5": [0.01] * n,
            "median_amount20": [1_000_000.0] * n,
            "grr_consensus_norm": [0.7] * n,
            "grr_consensus_count": [3] * n,
        }
    )
    df.attrs["score_col"] = "branch_score"
    return df


def _outputs():
    return {
        "current_aggressive": _frame("a", sigma=0.4, amp=0.4, drawdown=0.4),
        "trend_uncluttered": _frame("t", sigma=0.25, amp=0.25, drawdown=0.25),
        "legal_minrisk_hardened": _frame("m", sigma=0.10, amp=0.10, drawdown=0.10),
        "ai_hardware_mainline_v1": _frame("h", sigma=0.30, amp=0.30, drawdown=0.30),
        "grr_tail_guard": _frame("g", scores=[0.7, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64], sigma=0.18, amp=0.18, drawdown=0.18),
    }


def test_illegal_baseline_branches_filtered():
    outputs = _outputs()
    outputs["reference_baseline_branch"] = _frame("r")
    outputs["baseline_model_hybrid"] = _frame("b")
    decision = route_branch_v1(outputs, {"risk_off_score": 0.1}, {"min_confidence_to_switch": 999})
    assert decision.debug_info["illegal_branch_filtered"] is True
    assert "reference_baseline_branch" not in decision.branch_weights
    assert "baseline_model_hybrid" not in decision.branch_weights


def test_branch_router_returns_default_when_low_confidence():
    decision = route_branch_v1(_outputs(), {"risk_off_score": 0.1}, {"min_confidence_to_switch": 999})
    assert decision.chosen_branch == "grr_tail_guard"
    assert decision.fallback_used is True
    assert decision.route_reason == "low_confidence_default"


def test_branch_router_chooses_minrisk_in_crash_mode():
    decision = route_branch_v1(_outputs(), {"risk_off_score": 0.8, "crash_mode": True}, {})
    assert decision.chosen_branch == "legal_minrisk_hardened"
    assert decision.route_reason == "risk_off_minrisk"


def test_branch_router_prefers_trend_when_trend_high_clutter_low():
    outputs = _outputs()
    outputs["trend_uncluttered"] = _frame("t", scores=[2.0, 1.9, 1.8, 1.7, 1.6, 0.1, 0.0], sigma=0.18, amp=0.18, drawdown=0.18)
    decision = route_branch_v1(
        outputs,
        {"risk_off_score": 0.1, "trend_score": 0.9, "clutter_score": 0.1},
        {"min_confidence_to_switch": 0.01, "allow_soft_blend": False},
    )
    assert decision.chosen_branch == "trend_uncluttered"


def test_soft_blend_uses_rank_not_raw_score():
    a = _frame("a", scores=[1000.0, 999.0])
    b = pd.DataFrame(
        {
            "stock_id": ["b000", "a001"],
            "branch_score": [0.2, 0.1],
        }
    )
    b.attrs["score_col"] = "branch_score"
    blended = rank_blend_scores({"a": a, "b": b}, {"a": 0.5, "b": 0.5}, rrf_k=60)
    assert blended.iloc[0]["stock_id"] == "a001"
    assert "rank_blend_score" in blended.columns


def test_hedge_no_future_leakage():
    trace = hedge_weight_trace(
        [
            {"a": -10.0, "b": 10.0},
            {"a": 10.0, "b": -10.0},
        ],
        ["a", "b"],
        eta=0.5,
    )
    assert trace[0]["hedge_weights_before_decision"] == {"a": 0.5, "b": 0.5}
    assert trace[1]["hedge_weights_before_decision"] == trace[0]["hedge_weights_after_update"]
    assert trace[1]["hedge_weights_before_decision"]["b"] > trace[1]["hedge_weights_before_decision"]["a"]


def test_router_debug_fields_present():
    decision = route_branch_v1(_outputs(), {"risk_off_score": 0.2}, {})
    for field in ["utilities", "branch_metrics", "market_state", "available_branches", "illegal_branch_filtered"]:
        assert field in decision.debug_info
    assert isinstance(decision.branch_weights, dict)


def test_branch_snapshot_returns_topk_candidates():
    candidates = build_branch_candidates({"trend_uncluttered": _frame("t", scores=list(range(30, 0, -1)))}, top_k=20)
    assert len(candidates) == 20
    assert {"candidate_stock_id", "rank_in_branch", "post_filter_selected_top5", "final_selected_flag"}.issubset(candidates.columns)
    assert int(candidates["final_selected_flag"].sum()) == 5


def test_branch_margin_not_forced_zero_when_no_raw_score():
    df = pd.DataFrame({"stock_id": [f"x{i:03d}" for i in range(20)]})
    snap = build_branch_snapshots({"rank_only": df}, market_state={"rrf_k": 60})
    assert snap.iloc[0]["score_source"] == "rank_derived"
    assert snap.iloc[0]["branch_score_margin_top5_vs_top10"] > 0


def test_trend_theme_use_hard_risk_cap_not_large_linear_penalty():
    outputs = _outputs()
    outputs["trend_uncluttered"] = _frame("t", scores=[2, 1.9, 1.8, 1.7, 1.6, 0.1, 0.0], sigma=0.08, amp=0.08)
    decision = route_branch_v2a(
        outputs,
        {"trend_score": 0.95, "clutter_score": 0.05, "risk_off_score": 0.1, "market_breadth_5d": 0.8},
        {"trend_override_threshold": 0.45, "trend_soft_blend_band": 0.01, "theme_ai_override_enabled": False},
    )
    assert decision.chosen_branch == "trend_uncluttered"
    assert decision.route_reason == "clean_trend_override"
    assert "risk_penalty" not in decision.debug_info


def test_v2a_does_not_route_to_minrisk_on_aggressive_chaser_blocked():
    outputs = _outputs()
    outputs["current_aggressive"] = _frame("a", sigma=0.95, amp=0.95)
    decision = route_branch_v2a(outputs, {"risk_off_score": 0.55, "crash_mode": False}, {"trend_override_enabled": False, "theme_ai_override_enabled": False})
    assert decision.chosen_branch == "grr_tail_guard"
    assert decision.route_reason == "default_grr_tail_guard"


def test_v2a_crash_mode_can_route_to_minrisk():
    decision = route_branch_v2a(_outputs(), {"risk_off_score": 0.9, "crash_mode": True}, {})
    assert decision.chosen_branch == "legal_minrisk_hardened"
    assert decision.route_reason == "crash_minrisk_rescue"


def test_v2a_soft_blend_uses_rrf_rank():
    outputs = _outputs()
    outputs["trend_uncluttered"] = _frame("t", scores=[2, 1.9, 1.8, 1.7, 1.6, 0.1, 0.0], sigma=0.08, amp=0.08)
    decision = route_branch_v2a(
        outputs,
        {"trend_score": 0.95, "clutter_score": 0.05, "risk_off_score": 0.1, "market_breadth_5d": 0.8},
        {"trend_override_threshold": 0.45, "trend_soft_blend_band": 1.0, "theme_ai_override_enabled": False},
    )
    assert set(decision.branch_weights) == {"grr_tail_guard", "trend_uncluttered"}
    assert decision.debug_info["rrf_k"] == 60
    assert decision.debug_info["post_blend_top5"]


def test_v2a_filters_illegal_baseline_branches():
    outputs = _outputs()
    outputs["reference_baseline_branch"] = _frame("r")
    decision = route_branch_v2a(outputs, {"risk_off_score": 0.1}, {})
    assert decision.debug_info["illegal_branch_filtered"] is True
    assert "reference_baseline_branch" in decision.blocked_branches


def test_v2a_debug_fields_present():
    decision = route_branch_v2a(_outputs(), {"risk_off_score": 0.1}, {})
    for field in ["trend_override_score", "theme_ai_override_score", "trend_hard_cap_ok", "theme_ai_hard_cap_ok", "blend_branches"]:
        assert field in decision.debug_info


def test_recent_oof_strength_no_future_leakage():
    trace = hedge_weight_trace([{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}], ["a", "b"])
    assert trace[0]["hedge_weights_before_decision"] == {"a": 0.5, "b": 0.5}
    assert trace[1]["hedge_weights_before_decision"] == trace[0]["hedge_weights_after_update"]


def _v2b_decision(outputs=None, market_state=None, config=None):
    return route_branch_v2b_overlay(
        outputs or _outputs(),
        market_state or {"trend_score": 0.8, "clutter_score": 0.1, "risk_off_score": 0.1, "crash_mode": False},
        config or {},
    )


def test_v2b_overlay_does_not_full_switch_branch():
    decision = _v2b_decision(config={"trend_window_threshold": 0.0, "theme_ai_overlay_enabled": False, "trend_min_replacement_gap": -1.0})
    assert decision.chosen_branch in {"grr_tail_guard", "grr_tail_guard_overlay"}
    assert decision.chosen_branch != "trend_uncluttered"
    assert len(decision.debug_info["overlay_decision"]["final_top5"]) == 5


def test_v2b_overlay_limits_ai_to_one_swap():
    decision = _v2b_decision(
        config={
            "trend_overlay_enabled": False,
            "theme_ai_window_threshold": 0.0,
            "theme_ai_min_replacement_gap": -1.0,
            "theme_ai_max_swaps": 1,
            "theme_ai_sigma_cap_q": 1.0,
            "theme_ai_amp_cap_q": 1.0,
            "theme_ai_drawdown_cap_q": 1.0,
        }
    )
    swaps = decision.debug_info["overlay_decision"]["swaps"]
    assert len([s for s in swaps if s["source_branch"] == "ai_hardware_mainline_v1"]) <= 1


def test_v2b_overlay_limits_trend_to_two_swaps():
    decision = _v2b_decision(config={"trend_window_threshold": 0.0, "theme_ai_overlay_enabled": False, "trend_min_replacement_gap": -1.0, "trend_max_swaps": 2})
    swaps = decision.debug_info["overlay_decision"]["swaps"]
    assert len([s for s in swaps if s["source_branch"] == "trend_uncluttered"]) <= 2


def test_v2b_overlay_limits_total_swaps_to_two():
    decision = _v2b_decision(
        config={
            "trend_window_threshold": 0.0,
            "theme_ai_window_threshold": 0.0,
            "trend_min_replacement_gap": -1.0,
            "theme_ai_min_replacement_gap": -1.0,
            "theme_ai_sigma_cap_q": 1.0,
            "theme_ai_amp_cap_q": 1.0,
            "theme_ai_drawdown_cap_q": 1.0,
            "max_total_swaps": 2,
        }
    )
    assert decision.debug_info["overlay_decision"]["swap_count"] <= 2


def test_v2b_overlay_uses_default_as_base():
    decision = _v2b_decision(config={"trend_window_threshold": 0.0, "theme_ai_overlay_enabled": False, "trend_min_replacement_gap": -1.0})
    overlay = decision.debug_info["overlay_decision"]
    default_top5 = set(overlay["debug_info"]["default_top5"])
    final_top5 = set(overlay["final_top5"])
    assert len(default_top5 - final_top5) == overlay["swap_count"]


def test_v2b_post_overlay_uses_hard_veto_not_full_tail_rerank():
    decision = _v2b_decision(config={"trend_window_threshold": 0.0, "theme_ai_overlay_enabled": False})
    assert decision.debug_info["full_tail_guard_rerank_used"] is False
    assert decision.debug_info["overlay_decision"]["debug_info"]["post_overlay_hard_veto_only"] is True


def test_v2b_rejects_high_risk_ai_candidate():
    outputs = _outputs()
    outputs["ai_hardware_mainline_v1"] = _frame("h", scores=[9, 8, 7, 6, 5, 4, 3], sigma=9.0, amp=9.0, drawdown=9.0)
    decision = _v2b_decision(
        outputs=outputs,
        config={
            "trend_overlay_enabled": False,
            "theme_ai_window_threshold": 0.0,
            "theme_ai_min_replacement_gap": -1.0,
            "theme_ai_sigma_cap_q": 0.5,
            "theme_ai_amp_cap_q": 0.5,
            "theme_ai_drawdown_cap_q": 0.5,
        },
    )
    overlay = decision.debug_info["overlay_decision"]
    assert not overlay["swaps"]
    assert any(row["reject_reason"] == "post_overlay_hard_veto" for row in overlay["rejected_candidates"])


def test_v2b_does_not_route_minrisk_on_aggressive_chaser_blocked():
    outputs = _outputs()
    outputs["current_aggressive"] = _frame("a", sigma=0.95, amp=0.95, drawdown=0.95)
    decision = _v2b_decision(outputs=outputs, config={"trend_overlay_enabled": False, "theme_ai_overlay_enabled": False})
    assert decision.chosen_branch == "grr_tail_guard"
    assert decision.route_reason == "default_grr_tail_guard"


def test_v2b_effective_override_audit_marks_noop():
    default_top5 = ["a", "b", "c", "d", "e"]
    chosen_top5 = ["x", "b", "c", "d", "e"]
    final_top5 = ["a", "b", "c", "d", "e"]
    effective_override = final_top5 != default_top5
    no_op_reason = "tail_guard_collapsed_to_default" if chosen_top5 != default_top5 and not effective_override else "same_top5_as_default"
    assert effective_override is False
    assert no_op_reason == "tail_guard_collapsed_to_default"


def test_v2b_filters_illegal_baseline_branches():
    outputs = _outputs()
    outputs["reference_baseline_branch"] = _frame("r")
    decision = _v2b_decision(outputs=outputs)
    assert decision.debug_info["illegal_branch_filtered"] is True
    assert "reference_baseline_branch" in decision.blocked_branches


def test_v2b_debug_fields_present():
    decision = _v2b_decision()
    for field in ["overlay_decision", "trend_window_score", "theme_ai_window_score", "branch_metrics", "full_tail_guard_rerank_used"]:
        assert field in decision.debug_info


def test_v2b_trend_dispersion_guard_blocks_trend_overlay():
    decision = _v2b_decision(
        config={
            "trend_window_threshold": 0.0,
            "trend_dispersion_max": 0.0,
            "theme_ai_overlay_enabled": False,
            "trend_min_replacement_gap": -1.0,
        }
    )
    assert decision.debug_info["trend_dispersion_cap_ok"] is False
    assert decision.debug_info["overlay_decision"]["swap_count"] == 0


def test_v2b_ai_independence_guard_blocks_high_consensus_ai():
    decision = _v2b_decision(
        config={
            "trend_overlay_enabled": False,
            "theme_ai_window_threshold": 0.0,
            "theme_ai_consensus_max": -1.0,
            "theme_ai_min_replacement_gap": -1.0,
        }
    )
    assert decision.debug_info["theme_ai_independence_ok"] is False
    assert decision.debug_info["overlay_decision"]["swap_count"] == 0


def test_v2b_accepted_swap_logging_present():
    decision = _v2b_decision(
        config={
            "trend_window_threshold": 0.0,
            "trend_dispersion_max": 1.0,
            "trend_min_replacement_gap": -1.0,
            "theme_ai_overlay_enabled": False,
        }
    )
    records = decision.debug_info["overlay_decision"]["accepted_swap_records"]
    assert records
    assert {"candidate_stock", "replaced_stock", "score_margin", "risk_delta"}.issubset(records[0])


def test_v2b_blocked_candidate_logging_trend_dispersion_reason():
    decision = _v2b_decision(
        config={
            "trend_window_threshold": 0.0,
            "trend_dispersion_max": 0.0,
            "theme_ai_overlay_enabled": False,
        }
    )
    blocked = decision.debug_info["overlay_decision"]["blocked_candidate_records"]
    assert blocked
    assert any("trend_dispersion_too_high" in row["blocked_guard_reasons"] for row in blocked)


def test_v2b_blocked_candidate_logging_ai_consensus_reason():
    decision = _v2b_decision(
        config={
            "trend_overlay_enabled": False,
            "theme_ai_window_threshold": 0.0,
            "theme_ai_consensus_max": -1.0,
        }
    )
    blocked = decision.debug_info["overlay_decision"]["blocked_candidate_records"]
    assert blocked
    assert any("theme_ai_consensus_too_high" in row["blocked_guard_reasons"] for row in blocked)


def test_guard_summary_fields_numeric():
    decision = _v2b_decision(config={"trend_window_threshold": 0.0, "theme_ai_window_threshold": 0.0})
    summary = decision.debug_info["overlay_decision"]["guard_summary"]
    assert summary
    for row in summary:
        assert isinstance(row["candidates_generated"], int)
        assert isinstance(row["accepted"], int)
        assert isinstance(row["accept_rate"], float)


def test_paired_delta_distribution_counts():
    frame = pd.DataFrame(
        [
            {"variant": "default_grr_tail_guard", "window_date": "a", "score": 1.0},
            {"variant": "default_grr_tail_guard", "window_date": "b", "score": 1.0},
            {"variant": "default_grr_tail_guard", "window_date": "c", "score": 1.0},
            {"variant": "x", "window_date": "a", "score": 2.0},
            {"variant": "x", "window_date": "b", "score": 0.5},
            {"variant": "x", "window_date": "c", "score": 1.0},
        ]
    )
    row = paired_delta_distribution(frame)
    x = row[row["variant"] == "x"].iloc[0]
    assert int(x["positive_delta_count"]) == 1
    assert int(x["negative_delta_count"]) == 1
    assert int(x["zero_delta_count"]) == 1


def test_swap_delta_units_raw_and_weighted_math():
    raw_candidate_return = 0.06
    raw_replaced_return = 0.01
    raw_stock_delta = raw_candidate_return - raw_replaced_return
    position_weight = 0.2
    weighted_swap_delta = raw_stock_delta * position_weight
    assert abs(raw_stock_delta - 0.05) < 1e-12
    assert abs(weighted_swap_delta - 0.01) < 1e-12


def test_swap_delta_reconciliation_matches_weighted_swaps():
    ablations = pd.DataFrame(
        [
            {"variant": "default_grr_tail_guard", "window_date": "w1", "score": 0.10},
            {"variant": "v2b_trend_plus_ai_overlay", "window_date": "w1", "score": 0.11},
            {"variant": "default_grr_tail_guard", "window_date": "w2", "score": 0.20},
            {"variant": "v2b_trend_plus_ai_overlay", "window_date": "w2", "score": 0.20},
        ]
    )
    swaps = pd.DataFrame(
        [
            {
                "variant": "v2b_trend_plus_ai_overlay",
                "window": "w1",
                "raw_candidate_return": 0.06,
                "raw_replaced_return": 0.01,
                "raw_stock_delta": 0.05,
                "position_weight": 0.2,
                "weighted_swap_delta": 0.01,
                "delta_realized": 0.01,
            }
        ]
    )
    rec = swap_delta_reconciliation(ablations, swaps)
    row = rec[rec["window"] == "w1"].iloc[0]
    assert abs(row["window_delta_vs_default"] - row["sum_accepted_weighted_swap_delta"]) < 1e-12
    assert abs(row["reconciliation_error"]) < 1e-12


def test_sweep_grid_smoke():
    grid = build_sweep_grid([0.14], [6], [0.70])
    assert len(grid) == 1
    assert grid[0]["crash_minrisk_enabled"] is False


def test_sweep_grid_default_size_is_150():
    grid = build_sweep_grid([0.10, 0.12, 0.14, 0.16, 0.18, 0.20], [4, 5, 6, 7, 8], [0.60, 0.65, 0.70, 0.75, 0.80])
    assert len(grid) == 150


def test_sweep_grid_narrow_size_is_27():
    grid = build_sweep_grid([0.12, 0.14, 0.16], [5, 6, 7], [0.65, 0.70, 0.75])
    assert len(grid) == 27


def test_branch_router_runtime_config_stays_guarded_candidate():
    cfg = PROJECT_CONFIG["branch_router_v2b"]
    assert cfg["crash_minrisk_enabled"] is False
    assert cfg["trend_max_swaps"] == 1
    assert cfg["theme_ai_max_swaps"] == 1
    assert cfg["max_total_swaps"] == 2
    assert cfg["trend_dispersion_max"] == 0.13
    assert cfg["trend_candidate_rank_cap"] == 6
    assert cfg["theme_ai_consensus_max"] == 0.70


def test_riskoff_top60_rerank_branch_does_not_require_future_return():
    df = pd.DataFrame(
        {
            "stock_id": [f"{i:06d}" for i in range(80)],
            "score": [float(i) for i in range(80)],
            "lgb": [float(i % 7) for i in range(80)],
            "ret5": [0.01] * 80,
            "ret20": [-0.08] * 60 + [0.08] * 20,
            "sigma20": [0.03] * 80,
            "amp20": [0.02] * 80,
            "median_amount20": [1_000_000.0 + i for i in range(80)],
        }
    )
    candidates, stats = _riskoff_candidates(df, "shadow-test", top_k=10)
    assert stats["riskoff_triggered"] is True
    assert len(candidates) == 10
    assert "realized_ret" not in candidates.columns
    assert {"riskoff_rerank_score", "candidate_rank", "negative_ret20_penalty"}.issubset(candidates.columns)


def test_v2c_shadow_does_not_change_runtime_config():
    before = dict(PROJECT_CONFIG["branch_router_v2b"])
    _riskoff_candidates(
        pd.DataFrame(
            {
                "stock_id": [f"{i:06d}" for i in range(20)],
                "score": [float(i) for i in range(20)],
                "lgb": [float(i) for i in range(20)],
                "ret5": [0.01] * 20,
                "ret20": [-0.05] * 16 + [0.02] * 4,
                "sigma20": [0.03] * 20,
                "amp20": [0.02] * 20,
                "median_amount20": [1_000_000.0 + i for i in range(20)],
            }
        ),
        "shadow-test",
        top_k=5,
    )
    assert PROJECT_CONFIG["branch_router_v2b"] == before


def _v2d_work_frame(realized_candidate=0.08, realized_base=-0.02):
    return pd.DataFrame(
        {
            "stock_id": ["000001", "000002", "000003", "000004", "000005", "000006"],
            "score": [0.9, 0.8, 0.7, 0.6, 0.1, 0.5],
            "grr_final_score": [0.9, 0.8, 0.7, 0.6, 0.1, 0.5],
            "_risk_value": [0.05, 0.05, 0.05, 0.05, 0.08, 0.07],
            "realized_ret": [0.01, 0.01, 0.01, 0.01, realized_base, realized_candidate],
        }
    )


def _v2d_candidates():
    return pd.DataFrame(
        {
            "stock_id": ["000006"],
            "candidate_rank": [1],
            "candidate_score": [1.2],
            "riskoff_rerank_score": [1.2],
            "_risk_value": [0.07],
        }
    )


def test_v2d_shadow_does_not_change_runtime_config():
    before = dict(PROJECT_CONFIG["v2b_guarded_candidate"])
    assert_runtime_unchanged()
    evaluate_v2d_window(_v2d_work_frame(), ["000001", "000002", "000003", "000004", "000005"], _v2d_candidates())
    assert PROJECT_CONFIG["v2b_guarded_candidate"] == before


def test_v2d_shadow_max_swaps_one_and_no_hard_switch():
    result = evaluate_v2d_window(_v2d_work_frame(), ["000001", "000002", "000003", "000004", "000005"], _v2d_candidates())
    assert result["accepted_swap_count"] == 1
    assert len(set(result["final_top5"]) - {"000001", "000002", "000003", "000004", "000005"}) == 1
    assert result["final_top5"].count("000006") == 1


def test_v2d_shadow_decision_does_not_depend_on_realized_return():
    top5 = ["000001", "000002", "000003", "000004", "000005"]
    good = evaluate_v2d_window(_v2d_work_frame(realized_candidate=0.10, realized_base=-0.10), top5, _v2d_candidates())
    bad = evaluate_v2d_window(_v2d_work_frame(realized_candidate=-0.10, realized_base=0.10), top5, _v2d_candidates())
    assert good["accepted_swap"]["candidate_stock"] == bad["accepted_swap"]["candidate_stock"] == "000006"
    assert good["accepted_swap"]["replaced_stock"] == bad["accepted_swap"]["replaced_stock"] == "000005"


def test_v2d_shadow_output_fields_complete():
    result = evaluate_v2d_window(_v2d_work_frame(), ["000001", "000002", "000003", "000004", "000005"], _v2d_candidates())
    swap = result["accepted_swap"]
    assert {"candidate_stock", "replaced_stock", "candidate_rank", "score_margin", "risk_delta", "weighted_swap_delta"}.issubset(swap)


def test_ai_shadow_guard_does_not_change_actual_swap_decision():
    base = _v2b_decision(
        config={
            "trend_overlay_enabled": False,
            "theme_ai_window_threshold": 0.0,
            "theme_ai_consensus_max": 1.0,
            "theme_ai_min_replacement_gap": -1.0,
        }
    )
    shadow = _v2b_decision(
        config={
            "trend_overlay_enabled": False,
            "theme_ai_window_threshold": 0.0,
            "theme_ai_consensus_max": 1.0,
            "theme_ai_min_replacement_gap": -1.0,
            "shadow_theme_ai_candidate_rank_cap": 0,
        }
    )
    assert base.debug_info["overlay_decision"]["final_top5"] == shadow.debug_info["overlay_decision"]["final_top5"]
    assert "would_block_by_shadow_rank_cap" in shadow.debug_info["overlay_decision"]["accepted_swap_records"][0]
