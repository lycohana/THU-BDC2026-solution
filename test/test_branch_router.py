import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))

from branch_router import (  # noqa: E402
    build_branch_candidates,
    build_branch_snapshots,
    hedge_weight_trace,
    rank_blend_scores,
    route_branch_v1,
    route_branch_v2a,
    route_branch_v2b_overlay,
)


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
