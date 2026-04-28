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
from config import config as PROJECT_CONFIG  # noqa: E402


OUT_DIR = ROOT / "temp" / "branch_router_validation" / "v2d_riskoff_shadow"


def _position_weight() -> float:
    return 0.2


def _default_frame_for_window(work: pd.DataFrame, top5: list[str]) -> pd.DataFrame:
    out = work[work["stock_id"].astype(str).isin(set(top5))].copy()
    out["_rank"] = out["stock_id"].astype(str).map({stock: idx + 1 for idx, stock in enumerate(top5)})
    out["_base_score"] = pd.to_numeric(out.get("grr_final_score", out.get("score")), errors="coerce").fillna(0.0)
    out["_risk"] = pd.to_numeric(out.get("_risk_value", 0.0), errors="coerce").fillna(0.0)
    return out.sort_values("_base_score", ascending=True)


def _candidate_rows(work: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = candidates.copy()
    rows["candidate_stock"] = rows["stock_id"].astype(str).str.zfill(6)
    rows["candidate_score"] = pd.to_numeric(rows["candidate_score"], errors="coerce").fillna(0.0)
    rows["_risk"] = pd.to_numeric(rows.get("_risk_value", 0.0), errors="coerce").fillna(0.0)
    if "_risk_value" not in rows.columns:
        risk_lookup = work.set_index(work["stock_id"].astype(str))["_risk_value"].to_dict() if "_risk_value" in work.columns else {}
        rows["_risk"] = rows["candidate_stock"].map(risk_lookup).fillna(0.0)
    return rows


def evaluate_v2d_window(work: pd.DataFrame, default_top5: list[str], riskoff_candidates: pd.DataFrame) -> dict[str, Any]:
    default_score = _realized_for_ids(work, default_top5)
    default_bad, default_very_bad = _bad_counts(work, default_top5)
    result: dict[str, Any] = {
        "final_top5": list(default_top5),
        "score": default_score,
        "bad_count": default_bad,
        "very_bad_count": default_very_bad,
        "accepted_swap_count": 0,
        "blocked_candidates": 0,
        "accepted_swap": None,
        "blocked_reason_if_any": "",
    }
    if riskoff_candidates.empty:
        result["blocked_reason_if_any"] = "riskoff_not_triggered_or_no_candidates"
        return result

    default_frame = _default_frame_for_window(work, default_top5)
    if default_frame.empty:
        result["blocked_reason_if_any"] = "missing_default_top5_frame"
        return result
    weakest = default_frame.iloc[0]
    base_rank = int(weakest["_rank"])
    base = str(weakest["stock_id"]).zfill(6)
    base_score = _safe_float(weakest["_base_score"])
    base_risk = _safe_float(weakest["_risk"])
    default_score_median = float(default_frame["_base_score"].median())

    candidates = _candidate_rows(work, riskoff_candidates).copy()
    candidates = candidates[~candidates["candidate_stock"].isin(set(default_top5))]
    candidates = candidates[pd.to_numeric(candidates["candidate_rank"], errors="coerce") <= 5]
    if candidates.empty:
        result["blocked_reason_if_any"] = "no_rank_cap_candidate"
        return result

    accepted = None
    blocked_reasons: list[str] = []
    for _, cand in candidates.sort_values(["candidate_rank", "candidate_score"], ascending=[True, False]).iterrows():
        reasons = []
        candidate_stock = str(cand["candidate_stock"]).zfill(6)
        candidate_rank = int(cand["candidate_rank"])
        candidate_score = _safe_float(cand["candidate_score"])
        candidate_risk = _safe_float(cand["_risk"])
        risk_delta = candidate_risk - base_risk
        score_margin = candidate_score
        if candidate_rank > 5:
            reasons.append("candidate_rank_gt_5")
        if score_margin <= 0:
            reasons.append("score_margin_not_positive")
        if base_rank <= 3 and base_score >= default_score_median:
            reasons.append("default_strong_keep_guard")
        if risk_delta > 0.01:
            reasons.append("risk_delta_too_high")
        if reasons:
            blocked_reasons.extend(reasons)
            result["blocked_candidates"] += 1
            continue

        candidate_row = work[work["stock_id"].astype(str) == candidate_stock]
        base_row = work[work["stock_id"].astype(str) == base]
        raw_candidate_return = _safe_float(candidate_row["realized_ret"].iloc[0]) if not candidate_row.empty and "realized_ret" in candidate_row else None
        raw_replaced_return = _safe_float(base_row["realized_ret"].iloc[0]) if not base_row.empty and "realized_ret" in base_row else None
        raw_delta = None if raw_candidate_return is None or raw_replaced_return is None else raw_candidate_return - raw_replaced_return
        weighted_delta = None if raw_delta is None else raw_delta * _position_weight()
        accepted = {
            "candidate_stock": candidate_stock,
            "replaced_stock": base,
            "candidate_rank": candidate_rank,
            "replaced_rank": base_rank,
            "score_margin": score_margin,
            "risk_delta": risk_delta,
            "raw_candidate_return": raw_candidate_return,
            "raw_replaced_return": raw_replaced_return,
            "raw_stock_delta": raw_delta,
            "position_weight": _position_weight(),
            "weighted_swap_delta": weighted_delta,
            "blocked_reason_if_any": "",
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
            "blocked_reason_if_any": "",
        }
    )
    return result


def run_v2d_shadow(inputs: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for window in sorted(inputs["default_top5"]):
        work = inputs["work_by_window"][window]
        candidates, stats = _riskoff_candidates(work, window, top_k=20)
        default_top5 = inputs["default_top5"][window]
        default_return = _realized_for_ids(work, default_top5)
        result = evaluate_v2d_window(work, default_top5, candidates)
        accepted = result["accepted_swap"] or {}
        row = {
            "window": window,
            "default_return": default_return,
            "v2d_shadow_return": result["score"],
            "v2d_delta_vs_default": result["score"] - default_return,
            "accepted_swap_count": result["accepted_swap_count"],
            "candidate_stock": accepted.get("candidate_stock"),
            "replaced_stock": accepted.get("replaced_stock"),
            "candidate_rank": accepted.get("candidate_rank"),
            "replaced_rank": accepted.get("replaced_rank"),
            "score_margin": accepted.get("score_margin"),
            "risk_delta": accepted.get("risk_delta"),
            "raw_candidate_return": accepted.get("raw_candidate_return"),
            "raw_replaced_return": accepted.get("raw_replaced_return"),
            "raw_stock_delta": accepted.get("raw_stock_delta"),
            "position_weight": accepted.get("position_weight", _position_weight() if result["accepted_swap_count"] else None),
            "weighted_swap_delta": accepted.get("weighted_swap_delta"),
            "blocked_reason_if_any": result["blocked_reason_if_any"],
            "riskoff_triggered": stats["riskoff_triggered"],
            "median_ret20": stats["median_ret20"],
            "breadth20": stats["breadth20"],
            "median_sigma20": stats["median_sigma20"],
            "dispersion20": stats["dispersion20"],
            "final_top5": ",".join(result["final_top5"]),
            "bad_count": result["bad_count"],
            "very_bad_count": result["very_bad_count"],
        }
        rows.append(row)
        for _, cand in candidates.iterrows():
            candidate_rows.append(
                {
                    "window": window,
                    "candidate_stock": str(cand["stock_id"]).zfill(6),
                    "candidate_rank": int(cand["candidate_rank"]),
                    "candidate_score": float(cand["candidate_score"]),
                    "riskoff_rerank_score": float(cand["riskoff_rerank_score"]),
                    "riskoff_triggered": bool(stats["riskoff_triggered"]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(candidate_rows)


def _metrics(scores: pd.Series, very_bad: pd.Series | None = None) -> dict[str, float]:
    values = pd.to_numeric(scores, errors="coerce").dropna()
    return {
        "mean": float(values.mean()) if len(values) else 0.0,
        "q10": float(values.quantile(0.10)) if len(values) else 0.0,
        "worst": float(values.min()) if len(values) else 0.0,
        "very_bad": float(pd.to_numeric(very_bad, errors="coerce").fillna(0).mean()) if very_bad is not None and len(very_bad) else 0.0,
    }


def write_summary(v2d: pd.DataFrame, v2b_detail: Path, v2c_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ablation = pd.read_csv(v2b_detail / "ablation_decisions.csv")
    direct = pd.read_csv(v2c_dir / "riskoff_top60_direct_windows.csv")
    windows = sorted(v2d["window"].astype(str).unique())
    bucket_specs = {"20win": windows[-20:], "40win": windows[-40:], "60win": windows}
    rows: list[dict[str, Any]] = []

    def add_row(bucket: str, keep: list[str], variant: str, frame: pd.DataFrame, window_col: str, score_col: str, swap_col: str | None = None) -> None:
        sub = frame[frame[window_col].astype(str).isin(set(keep))].copy()
        default = ablation[(ablation["variant"] == "default_grr_tail_guard") & ablation["window_date"].astype(str).isin(set(keep))].copy()
        v2b = ablation[(ablation["variant"] == "v2b_trend_plus_ai_overlay") & ablation["window_date"].astype(str).isin(set(keep))].copy()
        merged = sub[[window_col, score_col]].rename(columns={window_col: "window", score_col: "score"}).merge(
            default[["window_date", "score"]].rename(columns={"window_date": "window", "score": "default_score"}), on="window", how="inner"
        ).merge(
            v2b[["window_date", "score"]].rename(columns={"window_date": "window", "score": "v2b_score"}), on="window", how="inner"
        )
        delta_default = pd.to_numeric(merged["score"], errors="coerce") - pd.to_numeric(merged["default_score"], errors="coerce")
        delta_v2b = pd.to_numeric(merged["score"], errors="coerce") - pd.to_numeric(merged["v2b_score"], errors="coerce")
        avg_swaps = float(pd.to_numeric(sub[swap_col], errors="coerce").fillna(0).mean()) if swap_col and swap_col in sub else 0.0
        rows.append(
            {
                "window_bucket": bucket,
                "window_count": len(merged),
                "variant": variant,
                **_metrics(merged["score"], sub.get("very_bad_count")),
                "avg_swaps": avg_swaps,
                "delta_vs_default": float(delta_default.mean()) if len(delta_default) else 0.0,
                "delta_vs_v2b": float(delta_v2b.mean()) if len(delta_v2b) else 0.0,
                "positive_delta_count": int((delta_default > 1e-12).sum()),
                "negative_delta_count": int((delta_default < -1e-12).sum()),
                "zero_delta_count": int(delta_default.abs().le(1e-12).sum()),
                "accepted_swaps": int(pd.to_numeric(sub[swap_col], errors="coerce").fillna(0).sum()) if swap_col and swap_col in sub else 0,
                "blocked_candidates": int((sub.get("blocked_reason_if_any", pd.Series(dtype=str)).fillna("") != "").sum()) if variant == "v2d_riskoff_micro_overlay_shadow" else 0,
            }
        )

    for bucket, keep in bucket_specs.items():
        add_row(bucket, keep, "default", ablation[ablation["variant"] == "default_grr_tail_guard"], "window_date", "score")
        add_row(bucket, keep, "v2b_guarded_candidate", ablation[ablation["variant"] == "v2b_trend_plus_ai_overlay"], "window_date", "score", "swap_count")
        add_row(bucket, keep, "riskoff_top60_direct_shadow", direct, "window", "score")
        add_row(bucket, keep, "v2d_riskoff_micro_overlay_shadow", v2d, "window", "v2d_shadow_return", "accepted_swap_count")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "v2d_riskoff_summary.csv", index=False)

    compression_rows = []
    for bucket in ["20win", "40win", "60win"]:
        direct_delta = float(summary[(summary["window_bucket"] == bucket) & (summary["variant"] == "riskoff_top60_direct_shadow")]["delta_vs_default"].iloc[0])
        v2d_delta = float(summary[(summary["window_bucket"] == bucket) & (summary["variant"] == "v2d_riskoff_micro_overlay_shadow")]["delta_vs_default"].iloc[0])
        for variant in ["riskoff_top60_direct_shadow", "v2d_riskoff_micro_overlay_shadow", "v2b_guarded_candidate"]:
            row = summary[(summary["window_bucket"] == bucket) & (summary["variant"] == variant)].iloc[0].to_dict()
            row["captured_direct_alpha_ratio"] = float(v2d_delta / direct_delta) if direct_delta else 0.0
            compression_rows.append(row)
    compression = pd.DataFrame(compression_rows)
    compression.to_csv(out_dir / "riskoff_compression_analysis.csv", index=False)
    return summary, compression


def write_report(summary: pd.DataFrame, compression: pd.DataFrame, out_dir: Path) -> None:
    v2d = summary[summary["variant"] == "v2d_riskoff_micro_overlay_shadow"]
    v2b = summary[summary["variant"] == "v2b_guarded_candidate"]
    merged = v2d[["window_bucket", "delta_vs_default", "worst", "avg_swaps", "negative_delta_count"]].merge(
        v2b[["window_bucket", "delta_vs_default", "worst", "avg_swaps", "negative_delta_count"]],
        on="window_bucket",
        suffixes=("_v2d", "_v2b"),
    )
    passes = bool(
        (merged["delta_vs_default_v2d"] >= merged["delta_vs_default_v2b"] - 0.0005).all()
        and (merged["worst_v2d"] >= merged["worst_v2b"] - 1e-12).all()
        and (merged["avg_swaps_v2d"] <= merged["avg_swaps_v2b"] + 0.05).all()
    )
    final = (
        "B. v2d can become future candidate, but not replace current v2b freeze yet"
        if passes
        else "A. v2d remains shadow only"
    )
    lines = [
        "# v2d_riskoff_micro_overlay_shadow decision report",
        "",
        "## 1. Executive summary",
        final,
        "",
        "## 2. Why v2c failed",
        "v2c union accepted too many trend/AI proposals; riskoff_top60 was positive, but trend/AI weighted deltas dominated negatively.",
        "",
        "## 3. Why riskoff_top60 remains interesting",
        "riskoff_top60_direct_shadow is positive on 20/40/60, but direct use is an offline diagnostic and resembles whole-window replacement.",
        "",
        "## 4. v2d micro overlay design",
        "v2d uses only riskoff_top60 candidates, candidate_rank <= 5, max_swaps = 1, positive riskoff score margin, weak-default replacement, and a small risk-delta cap.",
        "",
        "## 5. 20/40/60 validation",
        "```text\n" + summary.to_string(index=False) + "\n```",
        "",
        "## 6. Compression analysis",
        "```text\n" + compression.to_string(index=False) + "\n```",
        "",
        "## 7. Risk analysis",
        "No runtime hard switch is introduced. Realized returns are used only for offline diagnostics and validation.",
        "",
        "## 8. Final recommendation",
        final,
    ]
    (out_dir / "v2d_riskoff_decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_runtime_unchanged() -> None:
    cfg = PROJECT_CONFIG["v2b_guarded_candidate"]
    assert cfg["crash_minrisk_enabled"] is False
    assert cfg["trend_max_swaps"] == 1
    assert cfg["theme_ai_max_swaps"] == 1
    assert cfg["max_total_swaps"] == 2
    assert cfg["trend_dispersion_max"] == 0.14
    assert cfg["trend_candidate_rank_cap"] == 6
    assert cfg["theme_ai_consensus_max"] == 0.70


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default=str(SOURCE_RUN.relative_to(ROOT)))
    parser.add_argument("--detail-dir", default=str(V2B_DETAIL.relative_to(ROOT)))
    parser.add_argument("--v2c-dir", default="temp/branch_router_validation/v2c_shadow")
    parser.add_argument("--out-dir", default=str(OUT_DIR.relative_to(ROOT)))
    args = parser.parse_args()

    assert_runtime_unchanged()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = build_shadow_inputs(ROOT / args.source_run, ROOT / "data" / "train_hs300_20260424.csv")
    v2d, candidate_frame = run_v2d_shadow(inputs)

    detail_dir = ROOT / args.detail_dir
    ablation = pd.read_csv(detail_dir / "ablation_decisions.csv")
    v2b = ablation[ablation["variant"] == "v2b_trend_plus_ai_overlay"][["window_date", "score"]].rename(
        columns={"window_date": "window", "score": "v2b_return"}
    )
    v2d = v2d.merge(v2b, on="window", how="left")
    v2d["v2d_delta_vs_v2b"] = pd.to_numeric(v2d["v2d_shadow_return"], errors="coerce") - pd.to_numeric(v2d["v2b_return"], errors="coerce")
    v2d.to_csv(out_dir / "v2d_riskoff_micro_overlay_shadow.csv", index=False)
    candidate_frame.to_csv(out_dir / "v2d_riskoff_candidates.csv", index=False)
    summary, compression = write_summary(v2d, detail_dir, ROOT / args.v2c_dir, out_dir)
    write_report(summary, compression, out_dir)
    print(json.dumps({"out_dir": str(out_dir), "accepted_swaps": int(v2d["accepted_swap_count"].sum())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
