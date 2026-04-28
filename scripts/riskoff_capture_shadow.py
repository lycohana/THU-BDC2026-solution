from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "code" / "src"))

from branch_router_v2c_shadow import (  # noqa: E402
    SOURCE_RUN,
    V2B_DETAIL,
    _bad_counts,
    _realized_for_ids,
    _riskoff_candidates,
    _safe_float,
    build_shadow_inputs,
)


OUT_DIR = ROOT / "temp" / "branch_router_validation" / "riskoff_capture_shadow"


VARIANTS: list[dict[str, Any]] = [
    {
        "variant": "v2d_current_replay",
        "rank_cap": 5,
        "risk_delta_cap": 0.01,
        "target_rule": "lowest_score",
        "strong_keep_guard": True,
        "min_score_margin": 0.0,
    },
    {
        "variant": "rank10_same_risk_cap",
        "rank_cap": 10,
        "risk_delta_cap": 0.01,
        "target_rule": "lowest_score",
        "strong_keep_guard": False,
        "min_score_margin": 0.0,
    },
    {
        "variant": "rank5_no_strong_same_risk_cap",
        "rank_cap": 5,
        "risk_delta_cap": 0.01,
        "target_rule": "lowest_score",
        "strong_keep_guard": False,
        "min_score_margin": 0.0,
    },
    {
        "variant": "rank5_no_strong_relaxed_risk_cap",
        "rank_cap": 5,
        "risk_delta_cap": 0.03,
        "target_rule": "lowest_score",
        "strong_keep_guard": False,
        "min_score_margin": 0.0,
    },
    {
        "variant": "rank5_relaxed_only_when_no_v2b_swap",
        "rank_cap": 5,
        "risk_delta_cap": 0.03,
        "target_rule": "lowest_score",
        "strong_keep_guard": False,
        "min_score_margin": 0.0,
        "v2b_gate": "no_v2b_swap",
        "stack_on_v2b": True,
    },
    {
        "variant": "rank4_relaxed_only_when_no_v2b_swap",
        "rank_cap": 4,
        "risk_delta_cap": 0.03,
        "target_rule": "lowest_score",
        "strong_keep_guard": False,
        "min_score_margin": 0.0,
        "v2b_gate": "no_v2b_swap",
        "stack_on_v2b": True,
    },
    {
        "variant": "rank4_relaxed_highest_risk_no_v2b_swap",
        "rank_cap": 4,
        "risk_delta_cap": 0.03,
        "target_rule": "highest_risk",
        "strong_keep_guard": False,
        "min_score_margin": 0.0,
        "v2b_gate": "no_v2b_swap",
        "stack_on_v2b": True,
    },
    {
        "variant": "rank4_relaxed_blended_weak_no_v2b_swap",
        "rank_cap": 4,
        "risk_delta_cap": 0.03,
        "target_rule": "blended_weak",
        "strong_keep_guard": False,
        "min_score_margin": 0.0,
        "v2b_gate": "no_v2b_swap",
        "stack_on_v2b": True,
    },
    {
        "variant": "rank4_relaxed_dynamic_defensive_target_no_v2b_swap",
        "rank_cap": 4,
        "risk_delta_cap": 0.03,
        "target_rule": "lowest_score_or_defensive_high_risk",
        "strong_keep_guard": False,
        "min_score_margin": 0.0,
        "defensive_candidate_risk_max": 0.04,
        "defensive_risk_gap_min": 0.10,
        "v2b_gate": "no_v2b_swap",
        "stack_on_v2b": True,
    },
    {
        "variant": "rank5_relaxed_when_v2b_runtime_risky",
        "rank_cap": 5,
        "risk_delta_cap": 0.03,
        "target_rule": "lowest_score",
        "strong_keep_guard": False,
        "min_score_margin": 0.0,
        "v2b_gate": "no_v2b_or_runtime_risky",
        "stack_on_v2b": True,
    },
    {
        "variant": "rank5_relaxed_oracle_v2b_nonpositive",
        "rank_cap": 5,
        "risk_delta_cap": 0.03,
        "target_rule": "lowest_score",
        "strong_keep_guard": False,
        "min_score_margin": 0.0,
        "v2b_gate": "no_v2b_or_oracle_nonpositive",
        "stack_on_v2b": True,
    },
    {
        "variant": "rank8_relaxed_risk_cap",
        "rank_cap": 8,
        "risk_delta_cap": 0.03,
        "target_rule": "lowest_score",
        "strong_keep_guard": False,
        "min_score_margin": 0.0,
    },
    {
        "variant": "rank10_relaxed_risk_cap",
        "rank_cap": 10,
        "risk_delta_cap": 0.03,
        "target_rule": "lowest_score",
        "strong_keep_guard": False,
        "min_score_margin": 0.0,
    },
    {
        "variant": "rank10_replace_highest_risk",
        "rank_cap": 10,
        "risk_delta_cap": 0.01,
        "target_rule": "highest_risk",
        "strong_keep_guard": False,
        "min_score_margin": 0.0,
    },
    {
        "variant": "rank10_replace_blended_weak",
        "rank_cap": 10,
        "risk_delta_cap": 0.01,
        "target_rule": "blended_weak",
        "strong_keep_guard": False,
        "min_score_margin": 0.0,
    },
]


def load_v2b_context(detail_dir: Path) -> dict[str, dict[str, Any]]:
    path = detail_dir / "accepted_swaps.csv"
    if not path.exists():
        return {}
    swaps = pd.read_csv(path)
    swaps = swaps[swaps["variant"] == "v2b_trend_plus_ai_overlay"].copy()
    if swaps.empty:
        return {}
    for col in ["risk_delta", "candidate_risk", "weighted_swap_delta"]:
        swaps[col] = pd.to_numeric(swaps[col], errors="coerce").fillna(0.0)
    rows: dict[str, dict[str, Any]] = {}
    for window, sub in swaps.groupby("window"):
        risk_delta_max = float(sub["risk_delta"].max())
        candidate_risk_max = float(sub["candidate_risk"].max())
        weighted_delta_sum = float(sub["weighted_swap_delta"].sum())
        branches = ",".join(sub["branch"].astype(str).tolist())
        # Runtime-safe approximation: high absolute candidate risk plus a positive
        # risk increase. This does not look at realized returns.
        runtime_risky = bool(((sub["candidate_risk"] >= 0.60) & (sub["risk_delta"] > 0.05)).any())
        rows[str(window)] = {
            "v2b_swap_count": int(len(sub)),
            "v2b_branches": branches,
            "v2b_replaced": ",".join(sub["replaced_stock"].astype(str).str.zfill(6).tolist()),
            "v2b_candidates": ",".join(sub["candidate_stock"].astype(str).str.zfill(6).tolist()),
            "v2b_risk_delta_max": risk_delta_max,
            "v2b_candidate_risk_max": candidate_risk_max,
            "v2b_weighted_delta_sum": weighted_delta_sum,
            "v2b_runtime_risky": runtime_risky,
            "v2b_oracle_nonpositive": bool(weighted_delta_sum <= 0.0),
        }
    return rows


def _v2b_gate_allows(spec: dict[str, Any], v2b_info: dict[str, Any]) -> tuple[bool, str]:
    gate = spec.get("v2b_gate", "none")
    if gate == "none":
        return True, ""
    swap_count = int(v2b_info.get("v2b_swap_count", 0) or 0)
    if gate == "no_v2b_swap":
        return (swap_count == 0, "v2b_swap_present")
    if gate == "no_v2b_or_runtime_risky":
        allowed = swap_count == 0 or bool(v2b_info.get("v2b_runtime_risky", False))
        return (allowed, "v2b_not_runtime_risky")
    if gate == "no_v2b_or_oracle_nonpositive":
        allowed = swap_count == 0 or bool(v2b_info.get("v2b_oracle_nonpositive", False))
        return (allowed, "v2b_oracle_positive")
    return True, ""


def _default_frame(work: pd.DataFrame, top5: list[str]) -> pd.DataFrame:
    out = work[work["stock_id"].astype(str).isin(set(top5))].copy()
    out["_rank"] = out["stock_id"].astype(str).map({stock: idx + 1 for idx, stock in enumerate(top5)})
    out["_base_score"] = pd.to_numeric(out.get("grr_final_score", out.get("score")), errors="coerce").fillna(0.0)
    out["_risk"] = pd.to_numeric(out.get("_risk_value", 0.0), errors="coerce").fillna(0.0)
    return out


def _target_row(base_frame: pd.DataFrame, rule: str) -> pd.Series:
    if rule == "highest_risk":
        return base_frame.sort_values(["_risk", "_base_score"], ascending=[False, True]).iloc[0]
    if rule == "blended_weak":
        out = base_frame.copy()
        out["_risk_rank"] = out["_risk"].rank(pct=True, method="average")
        out["_low_score_rank"] = (-out["_base_score"]).rank(pct=True, method="average")
        out["_weakness"] = 0.60 * out["_risk_rank"] + 0.40 * out["_low_score_rank"]
        return out.sort_values(["_weakness", "_risk"], ascending=[False, False]).iloc[0]
    return base_frame.sort_values("_base_score", ascending=True).iloc[0]


def _candidate_frame(work: pd.DataFrame, riskoff_candidates: pd.DataFrame, default_top5: list[str], rank_cap: int) -> pd.DataFrame:
    if riskoff_candidates.empty:
        return pd.DataFrame()
    rows = riskoff_candidates.copy()
    rows["candidate_stock"] = rows["stock_id"].astype(str).str.zfill(6)
    rows["candidate_rank"] = pd.to_numeric(rows["candidate_rank"], errors="coerce").fillna(999).astype(int)
    rows["candidate_score"] = pd.to_numeric(rows["candidate_score"], errors="coerce").fillna(0.0)
    if "_risk_value" in rows.columns:
        rows["_risk"] = pd.to_numeric(rows["_risk_value"], errors="coerce").fillna(0.0)
    else:
        risk_lookup = work.set_index(work["stock_id"].astype(str))["_risk_value"].to_dict() if "_risk_value" in work.columns else {}
        rows["_risk"] = rows["candidate_stock"].map(risk_lookup).fillna(0.0)
    rows = rows[~rows["candidate_stock"].isin(set(default_top5))]
    rows = rows[rows["candidate_rank"] <= int(rank_cap)]
    return rows.sort_values(["candidate_rank", "candidate_score"], ascending=[True, False]).copy()


def _accept_swap(
    work: pd.DataFrame,
    default_top5: list[str],
    riskoff_candidates: pd.DataFrame,
    spec: dict[str, Any],
    v2b_info: dict[str, Any],
) -> dict[str, Any]:
    default_return = _realized_for_ids(work, default_top5)
    default_bad, default_very_bad = _bad_counts(work, default_top5)
    result: dict[str, Any] = {
        "final_top5": list(default_top5),
        "score": default_return,
        "bad_count": default_bad,
        "very_bad_count": default_very_bad,
        "accepted_swap_count": 0,
        "blocked_candidates": 0,
        "blocked_reason_if_any": "",
        "accepted_swap": None,
    }
    gate_allowed, gate_reason = _v2b_gate_allows(spec, v2b_info)
    if not gate_allowed:
        result["blocked_reason_if_any"] = gate_reason
        return result
    if riskoff_candidates.empty:
        result["blocked_reason_if_any"] = "riskoff_not_triggered_or_no_candidates"
        return result
    base_frame = _default_frame(work, default_top5)
    if base_frame.empty:
        result["blocked_reason_if_any"] = "missing_default_top5_frame"
        return result

    base_median_score = float(base_frame["_base_score"].median())
    target_rule = str(spec["target_rule"])

    candidates = _candidate_frame(work, riskoff_candidates, default_top5, int(spec["rank_cap"]))
    if candidates.empty:
        result["blocked_reason_if_any"] = "no_rank_cap_candidate"
        return result

    blocked_reasons: list[str] = []
    accepted: dict[str, Any] | None = None
    for _, cand in candidates.iterrows():
        candidate_stock = str(cand["candidate_stock"]).zfill(6)
        candidate_rank = int(cand["candidate_rank"])
        candidate_score = _safe_float(cand["candidate_score"])
        candidate_risk = _safe_float(cand["_risk"])
        if target_rule == "lowest_score_or_defensive_high_risk":
            default_target = _target_row(base_frame, "lowest_score")
            high_risk_target = _target_row(base_frame, "highest_risk")
            high_risk_gap = _safe_float(high_risk_target["_risk"]) - candidate_risk
            use_high_risk_target = (
                candidate_risk <= float(spec.get("defensive_candidate_risk_max", 0.04))
                and high_risk_gap >= float(spec.get("defensive_risk_gap_min", 0.10))
            )
            target = high_risk_target if use_high_risk_target else default_target
        else:
            target = _target_row(base_frame, target_rule)
        base_stock = str(target["stock_id"]).zfill(6)
        base_rank = int(target["_rank"])
        base_score = _safe_float(target["_base_score"])
        base_risk = _safe_float(target["_risk"])
        risk_delta = candidate_risk - base_risk
        reasons: list[str] = []
        if candidate_rank > int(spec["rank_cap"]):
            reasons.append("candidate_rank_gt_cap")
        if candidate_score <= float(spec["min_score_margin"]):
            reasons.append("score_margin_not_positive")
        if bool(spec["strong_keep_guard"]) and base_rank <= 3 and base_score >= base_median_score:
            reasons.append("default_strong_keep_guard")
        if risk_delta > float(spec["risk_delta_cap"]):
            reasons.append("risk_delta_too_high")
        if reasons:
            blocked_reasons.extend(reasons)
            result["blocked_candidates"] += 1
            continue

        candidate_row = work[work["stock_id"].astype(str) == candidate_stock]
        base_row = work[work["stock_id"].astype(str) == base_stock]
        raw_candidate_return = _safe_float(candidate_row["realized_ret"].iloc[0]) if not candidate_row.empty and "realized_ret" in candidate_row else None
        raw_replaced_return = _safe_float(base_row["realized_ret"].iloc[0]) if not base_row.empty and "realized_ret" in base_row else None
        raw_delta = None if raw_candidate_return is None or raw_replaced_return is None else raw_candidate_return - raw_replaced_return
        accepted = {
            "candidate_stock": candidate_stock,
            "replaced_stock": base_stock,
            "candidate_rank": candidate_rank,
            "replaced_rank": base_rank,
            "candidate_score": candidate_score,
            "replaced_score": base_score,
            "candidate_risk": candidate_risk,
            "replaced_risk": base_risk,
            "risk_delta": risk_delta,
            "raw_candidate_return": raw_candidate_return,
            "raw_replaced_return": raw_replaced_return,
            "raw_stock_delta": raw_delta,
            "position_weight": 0.2,
            "weighted_swap_delta": None if raw_delta is None else 0.2 * raw_delta,
        }
        break

    if accepted is None:
        result["blocked_reason_if_any"] = ";".join(sorted(set(blocked_reasons))) if blocked_reasons else "no_candidate_passed_guards"
        return result

    final_top5 = list(default_top5)
    final_top5[final_top5.index(accepted["replaced_stock"])] = accepted["candidate_stock"]
    score = _realized_for_ids(work, final_top5)
    bad, very_bad = _bad_counts(work, final_top5)
    result.update(
        {
            "final_top5": final_top5,
            "score": score,
            "bad_count": bad,
            "very_bad_count": very_bad,
            "accepted_swap_count": 1,
            "accepted_swap": accepted,
        }
    )
    return result


def run_capture_shadow(inputs: dict[str, Any], variants: list[dict[str, Any]], v2b_context: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window in sorted(inputs["default_top5"]):
        work = inputs["work_by_window"][window]
        default_top5 = inputs["default_top5"][window]
        default_return = _realized_for_ids(work, default_top5)
        candidates, stats = _riskoff_candidates(work, window, top_k=20)
        v2b_info = v2b_context.get(window, {"v2b_swap_count": 0})
        direct_top5 = candidates.head(5)["stock_id"].astype(str).str.zfill(6).tolist() if not candidates.empty else default_top5
        direct_return = _realized_for_ids(work, direct_top5)
        for spec in variants:
            result = _accept_swap(work, default_top5, candidates, spec, v2b_info)
            accepted = result["accepted_swap"] or {}
            rows.append(
                {
                    "window": window,
                    "variant": spec["variant"],
                    "default_return": default_return,
                    "riskoff_direct_return": direct_return,
                    "shadow_return": result["score"],
                    "delta_vs_default": result["score"] - default_return,
                    "delta_vs_direct": result["score"] - direct_return,
                    "accepted_swap_count": result["accepted_swap_count"],
                    "blocked_candidates": result["blocked_candidates"],
                    "blocked_reason_if_any": result["blocked_reason_if_any"],
                    "riskoff_triggered": bool(stats["riskoff_triggered"]),
                    "candidate_stock": accepted.get("candidate_stock"),
                    "replaced_stock": accepted.get("replaced_stock"),
                    "candidate_rank": accepted.get("candidate_rank"),
                    "replaced_rank": accepted.get("replaced_rank"),
                    "candidate_score": accepted.get("candidate_score"),
                    "replaced_score": accepted.get("replaced_score"),
                    "candidate_risk": accepted.get("candidate_risk"),
                    "replaced_risk": accepted.get("replaced_risk"),
                    "risk_delta": accepted.get("risk_delta"),
                    "raw_candidate_return": accepted.get("raw_candidate_return"),
                    "raw_replaced_return": accepted.get("raw_replaced_return"),
                    "raw_stock_delta": accepted.get("raw_stock_delta"),
                    "weighted_swap_delta": accepted.get("weighted_swap_delta"),
                    "final_top5": ",".join(result["final_top5"]),
                    "bad_count": result["bad_count"],
                    "very_bad_count": result["very_bad_count"],
                    "stack_on_v2b": bool(spec.get("stack_on_v2b", False)),
                    "v2b_swap_count": v2b_info.get("v2b_swap_count", 0),
                    "v2b_branches": v2b_info.get("v2b_branches", ""),
                    "v2b_replaced": v2b_info.get("v2b_replaced", ""),
                    "v2b_candidates": v2b_info.get("v2b_candidates", ""),
                    "v2b_risk_delta_max": v2b_info.get("v2b_risk_delta_max", 0.0),
                    "v2b_candidate_risk_max": v2b_info.get("v2b_candidate_risk_max", 0.0),
                    "v2b_weighted_delta_sum": v2b_info.get("v2b_weighted_delta_sum", 0.0),
                    "v2b_runtime_risky": v2b_info.get("v2b_runtime_risky", False),
                    "v2b_oracle_nonpositive": v2b_info.get("v2b_oracle_nonpositive", False),
                }
            )
    return pd.DataFrame(rows)


def _metric(scores: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(scores, errors="coerce").dropna()
    if values.empty:
        return {"mean": 0.0, "q10": 0.0, "worst": 0.0, "hit_rate": 0.0}
    return {
        "mean": float(values.mean()),
        "q10": float(values.quantile(0.10)),
        "worst": float(values.min()),
        "hit_rate": float((values > 0).mean()),
    }


def summarize(results: pd.DataFrame, detail_dir: Path) -> pd.DataFrame:
    ablation = pd.read_csv(detail_dir / "ablation_decisions.csv")
    v2b = ablation[ablation["variant"] == "v2b_trend_plus_ai_overlay"][["window_date", "score"]].rename(
        columns={"window_date": "window", "score": "v2b_return"}
    )
    merged = results.merge(v2b, on="window", how="left")
    merged["delta_vs_v2b"] = pd.to_numeric(merged["shadow_return"], errors="coerce") - pd.to_numeric(merged["v2b_return"], errors="coerce")
    merged["effective_return"] = pd.to_numeric(merged["shadow_return"], errors="coerce")
    stack_mask = merged.get("stack_on_v2b", False).astype(bool) if "stack_on_v2b" in merged.columns else pd.Series(False, index=merged.index)
    no_riskoff_swap = pd.to_numeric(merged["accepted_swap_count"], errors="coerce").fillna(0).eq(0)
    merged.loc[stack_mask & no_riskoff_swap, "effective_return"] = pd.to_numeric(
        merged.loc[stack_mask & no_riskoff_swap, "v2b_return"], errors="coerce"
    )
    merged["effective_delta_vs_default"] = pd.to_numeric(merged["effective_return"], errors="coerce") - pd.to_numeric(merged["default_return"], errors="coerce")
    merged["effective_delta_vs_v2b"] = pd.to_numeric(merged["effective_return"], errors="coerce") - pd.to_numeric(merged["v2b_return"], errors="coerce")
    windows = sorted(merged["window"].astype(str).unique())
    buckets = {"20win": windows[-20:], "40win": windows[-40:], "60win": windows}
    rows: list[dict[str, Any]] = []
    for bucket, keep in buckets.items():
        scoped = merged[merged["window"].astype(str).isin(set(keep))].copy()
        for variant, sub in scoped.groupby("variant"):
            default_delta = pd.to_numeric(sub["effective_delta_vs_default"], errors="coerce").fillna(0.0)
            direct_alpha = pd.to_numeric(sub["riskoff_direct_return"], errors="coerce") - pd.to_numeric(sub["default_return"], errors="coerce")
            captured = float(default_delta.mean() / direct_alpha.mean()) if abs(float(direct_alpha.mean())) > 1e-12 else 0.0
            rows.append(
                {
                    "bucket": bucket,
                    "variant": variant,
                    "window_count": int(len(sub)),
                    "accepted_swaps": int(pd.to_numeric(sub["accepted_swap_count"], errors="coerce").fillna(0).sum()),
                    "accepted_swap_rate": float(pd.to_numeric(sub["accepted_swap_count"], errors="coerce").fillna(0).mean()) if len(sub) else 0.0,
                    "captured_direct_alpha_ratio": captured,
                    "stack_on_v2b": bool(sub.get("stack_on_v2b", pd.Series(False)).astype(bool).any()),
                    **{f"return_{k}": v for k, v in _metric(sub["effective_return"]).items()},
                    **{f"delta_default_{k}": v for k, v in _metric(sub["effective_delta_vs_default"]).items()},
                    **{f"delta_v2b_{k}": v for k, v in _metric(sub["effective_delta_vs_v2b"]).items()},
                    "negative_delta_count": int((default_delta < -1e-12).sum()),
                    "very_bad_mean": float(pd.to_numeric(sub["very_bad_count"], errors="coerce").fillna(0).mean()) if len(sub) else 0.0,
                }
            )
    return pd.DataFrame(rows), merged


def write_conflict_analysis(merged: pd.DataFrame, out_dir: Path, variant: str = "rank5_no_strong_relaxed_risk_cap") -> pd.DataFrame:
    sub = merged[merged["variant"] == variant].copy()
    if sub.empty:
        out = pd.DataFrame()
        out.to_csv(out_dir / "riskoff_v2b_conflict_analysis.csv", index=False)
        return out
    for col in ["accepted_swap_count", "v2b_swap_count", "delta_vs_default", "delta_vs_v2b", "v2b_weighted_delta_sum"]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0.0)
    sub["riskoff_accepts"] = sub["accepted_swap_count"] > 0
    sub["v2b_accepts"] = sub["v2b_swap_count"] > 0
    sub["both_accept"] = sub["riskoff_accepts"] & sub["v2b_accepts"]
    sub["riskoff_beats_v2b"] = sub["delta_vs_v2b"] > 1e-12
    sub["riskoff_loses_to_v2b"] = sub["delta_vs_v2b"] < -1e-12
    sub["same_replaced_as_v2b"] = [
        str(risk_replaced).zfill(6) in set(str(v2b_replaced).split(","))
        if pd.notna(risk_replaced) and str(risk_replaced) not in {"", "nan"}
        else False
        for risk_replaced, v2b_replaced in zip(sub["replaced_stock"], sub["v2b_replaced"])
    ]
    cols = [
        "window",
        "riskoff_triggered",
        "riskoff_accepts",
        "v2b_accepts",
        "both_accept",
        "riskoff_beats_v2b",
        "riskoff_loses_to_v2b",
        "same_replaced_as_v2b",
        "candidate_stock",
        "replaced_stock",
        "candidate_rank",
        "replaced_rank",
        "delta_vs_default",
        "delta_vs_v2b",
        "raw_stock_delta",
        "risk_delta",
        "v2b_branches",
        "v2b_candidates",
        "v2b_replaced",
        "v2b_risk_delta_max",
        "v2b_candidate_risk_max",
        "v2b_runtime_risky",
        "v2b_weighted_delta_sum",
        "blocked_reason_if_any",
    ]
    out = sub[[c for c in cols if c in sub.columns]].copy()
    out.to_csv(out_dir / "riskoff_v2b_conflict_analysis.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for name, mask in {
        "all_windows": pd.Series(True, index=sub.index),
        "both_accept": sub["both_accept"],
        "riskoff_only": sub["riskoff_accepts"] & ~sub["v2b_accepts"],
        "v2b_only": ~sub["riskoff_accepts"] & sub["v2b_accepts"],
        "neither_accept": ~sub["riskoff_accepts"] & ~sub["v2b_accepts"],
        "both_accept_riskoff_beats": sub["both_accept"] & sub["riskoff_beats_v2b"],
        "both_accept_riskoff_loses": sub["both_accept"] & sub["riskoff_loses_to_v2b"],
        "v2b_runtime_risky": sub["v2b_runtime_risky"].astype(bool),
        "v2b_oracle_nonpositive": sub["v2b_oracle_nonpositive"].astype(bool),
    }.items():
        part = sub[mask].copy()
        summary_rows.append(
            {
                "segment": name,
                "window_count": int(len(part)),
                "riskoff_accepts": int(part["riskoff_accepts"].sum()) if len(part) else 0,
                "v2b_accepts": int(part["v2b_accepts"].sum()) if len(part) else 0,
                "delta_default_mean": float(part["delta_vs_default"].mean()) if len(part) else 0.0,
                "delta_v2b_mean": float(part["delta_vs_v2b"].mean()) if len(part) else 0.0,
                "negative_delta_v2b_count": int((part["delta_vs_v2b"] < -1e-12).sum()) if len(part) else 0,
                "v2b_weighted_delta_mean": float(part["v2b_weighted_delta_sum"].mean()) if len(part) else 0.0,
            }
        )
    conflict_summary = pd.DataFrame(summary_rows)
    conflict_summary.to_csv(out_dir / "riskoff_v2b_conflict_summary.csv", index=False)
    return out


def write_report(summary: pd.DataFrame, out_dir: Path) -> None:
    sixty = summary[summary["bucket"] == "60win"].sort_values("delta_default_mean", ascending=False)
    lines = [
        "# Riskoff Capture Shadow",
        "",
        "All variants are shadow-only. Runtime constraints remain max 1 swap and no hard window switch.",
        "",
        "## 60win ranking",
        "```text",
        sixty.to_string(index=False),
        "```",
        "",
        "## Recommendation",
    ]
    if not sixty.empty:
        best = sixty.iloc[0]
        if float(best["delta_default_mean"]) > 0 and float(best["delta_v2b_mean"]) >= 0 and float(best["delta_default_q10"]) >= -0.003:
            lines.append(f"Promote {best['variant']} to a second shadow pass with stricter freeze checks.")
        else:
            lines.append("Keep all variants in shadow. None clears delta/q10/v2b compatibility together.")
    (out_dir / "riskoff_capture_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default=str(SOURCE_RUN.relative_to(ROOT)))
    parser.add_argument("--detail-dir", default=str(V2B_DETAIL.relative_to(ROOT)))
    parser.add_argument("--raw-path", default="data/train_hs300_20260424.csv")
    parser.add_argument("--out-dir", default=str(OUT_DIR.relative_to(ROOT)))
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = build_shadow_inputs(ROOT / args.source_run, ROOT / args.raw_path)
    detail_dir = ROOT / args.detail_dir
    v2b_context = load_v2b_context(detail_dir)
    results = run_capture_shadow(inputs, VARIANTS, v2b_context)
    summary, merged = summarize(results, detail_dir)
    results.to_csv(out_dir / "riskoff_capture_windows.csv", index=False)
    merged.to_csv(out_dir / "riskoff_capture_windows_with_v2b.csv", index=False)
    summary.to_csv(out_dir / "riskoff_capture_summary.csv", index=False)
    write_conflict_analysis(merged, out_dir)
    write_report(summary, out_dir)
    print(json.dumps({"out_dir": str(out_dir), "variants": len(VARIANTS), "rows": len(results)}, ensure_ascii=False), flush=True)
    print(summary[summary["bucket"] == "60win"].sort_values("delta_default_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
