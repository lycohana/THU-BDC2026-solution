from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = ROOT / "temp" / "branch_router_validation"
ANALYSIS_20 = ROOT / "temp" / "branch_router_analysis" / "branch_router_v2b_overlay_20win"
LONGER_60_DETAIL = VALIDATION_ROOT / "v2b_guarded_longer_60win_60win"

VARIANT_ALIASES = {
    "default_grr_tail_guard": "default",
    "v2b_trend_plus_ai_overlay": "v2b_guarded_candidate",
    "v2b_trend_overlay_only": "trend_overlay_only",
    "v2b_ai_overlay_only": "ai_overlay_only",
}


def _metric(scores: pd.Series, bad: pd.Series, very_bad: pd.Series) -> dict:
    scores = pd.to_numeric(scores, errors="coerce").dropna()
    if scores.empty:
        return {
            "mean": 0.0,
            "q10": 0.0,
            "worst": 0.0,
            "very_bad": 0.0,
            "win_rate": 0.0,
        }
    return {
        "mean": float(scores.mean()),
        "q10": float(scores.quantile(0.10)),
        "worst": float(scores.min()),
        "very_bad": float(pd.to_numeric(very_bad, errors="coerce").mean()),
        "win_rate": float((scores > 0).mean()),
    }


def _paired_delta(variant: pd.DataFrame, default: pd.DataFrame) -> pd.Series:
    merged = variant[["window_date", "score"]].merge(
        default[["window_date", "score"]].rename(columns={"score": "_default_score"}),
        on="window_date",
        how="inner",
    )
    return pd.to_numeric(merged["score"], errors="coerce") - pd.to_numeric(merged["_default_score"], errors="coerce")


def write_bucket_stability() -> Path:
    out_dir = VALIDATION_ROOT / "v2b_guarded_bucket_stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation = pd.read_csv(LONGER_60_DETAIL / "ablation_decisions.csv")
    swaps = pd.read_csv(LONGER_60_DETAIL / "accepted_swaps.csv")
    windows = sorted(ablation["window_date"].astype(str).unique())
    buckets = {
        "bucket_old_20": windows[:20],
        "bucket_mid_20": windows[20:40],
        "bucket_recent_20": windows[40:60],
        "bucket_all_60": windows,
    }
    rows = []
    for bucket, bucket_windows in buckets.items():
        bucket_set = set(bucket_windows)
        default = ablation[(ablation["variant"] == "default_grr_tail_guard") & ablation["window_date"].astype(str).isin(bucket_set)].copy()
        for source_variant, alias in VARIANT_ALIASES.items():
            df = ablation[(ablation["variant"] == source_variant) & ablation["window_date"].astype(str).isin(bucket_set)].copy()
            delta = _paired_delta(df, default)
            variant_swaps = swaps[(swaps["variant"] == source_variant) & swaps["window"].astype(str).isin(bucket_set)].copy()
            trend_swaps = variant_swaps[variant_swaps["branch"].astype(str) == "trend"]
            ai_swaps = variant_swaps[variant_swaps["branch"].astype(str) == "theme_ai"]
            metric = _metric(df["score"], df["bad_count"], df["very_bad_count"])
            rows.append(
                {
                    "bucket": bucket,
                    "variant": alias,
                    "window_count": int(len(df)),
                    **metric,
                    "avg_swaps": float(pd.to_numeric(df.get("swap_count", pd.Series(0, index=df.index)), errors="coerce").fillna(0).mean()) if len(df) else 0.0,
                    "mean_delta_vs_default": float(delta.mean()) if len(delta) else 0.0,
                    "q10_delta_vs_default": float(delta.quantile(0.10)) if len(delta) else 0.0,
                    "worst_delta_vs_default": float(delta.min()) if len(delta) else 0.0,
                    "positive_delta_count": int((delta > 1e-12).sum()),
                    "negative_delta_count": int((delta < -1e-12).sum()),
                    "zero_delta_count": int(delta.abs().le(1e-12).sum()),
                    "accepted_swaps": int(len(variant_swaps)),
                    "trend_accepted_swaps": int(len(trend_swaps)),
                    "ai_accepted_swaps": int(len(ai_swaps)),
                    "trend_weighted_delta_sum": float(pd.to_numeric(trend_swaps.get("weighted_swap_delta"), errors="coerce").fillna(0).sum()),
                    "ai_weighted_delta_sum": float(pd.to_numeric(ai_swaps.get("weighted_swap_delta"), errors="coerce").fillna(0).sum()),
                    "total_weighted_delta_sum": float(pd.to_numeric(variant_swaps.get("weighted_swap_delta"), errors="coerce").fillna(0).sum()),
                }
            )
    output = out_dir / "bucket_stability.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def write_negative_windows() -> tuple[Path, Path]:
    out_dir = VALIDATION_ROOT / "v2b_guarded_negative_windows_60win"
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation = pd.read_csv(LONGER_60_DETAIL / "ablation_decisions.csv")
    swaps = pd.read_csv(LONGER_60_DETAIL / "accepted_swaps.csv")
    default = ablation[ablation["variant"] == "default_grr_tail_guard"][["window_date", "score"]].rename(columns={"score": "default_return"})
    guarded = ablation[ablation["variant"] == "v2b_trend_plus_ai_overlay"][["window_date", "score"]].rename(columns={"score": "v2b_guarded_return"})
    deltas = guarded.merge(default, on="window_date", how="inner")
    deltas["window_delta_vs_default"] = pd.to_numeric(deltas["v2b_guarded_return"], errors="coerce") - pd.to_numeric(deltas["default_return"], errors="coerce")
    negative = deltas[deltas["window_delta_vs_default"] < -1e-12].copy()
    variant_swaps = swaps[swaps["variant"] == "v2b_trend_plus_ai_overlay"].copy()
    rows = []
    for _, neg in negative.iterrows():
        window = str(neg["window_date"])
        window_swaps = variant_swaps[variant_swaps["window"].astype(str) == window].copy()
        if window_swaps.empty:
            rows.append(
                {
                    "window": window,
                    "default_return": neg["default_return"],
                    "v2b_guarded_return": neg["v2b_guarded_return"],
                    "window_delta_vs_default": neg["window_delta_vs_default"],
                    "notes": "negative_delta_without_accepted_swap_record",
                }
            )
            continue
        for _, swap in window_swaps.iterrows():
            rows.append(
                {
                    "window": window,
                    "default_return": neg["default_return"],
                    "v2b_guarded_return": neg["v2b_guarded_return"],
                    "window_delta_vs_default": neg["window_delta_vs_default"],
                    "branch": swap.get("branch"),
                    "candidate_stock": swap.get("candidate_stock"),
                    "replaced_stock": swap.get("replaced_stock"),
                    "candidate_rank": swap.get("candidate_rank"),
                    "replaced_rank": swap.get("replaced_rank"),
                    "score_margin": swap.get("score_margin"),
                    "risk_delta": swap.get("risk_delta"),
                    "trend_dispersion": swap.get("trend_dispersion"),
                    "theme_ai_consensus": swap.get("theme_ai_consensus"),
                    "raw_candidate_return": swap.get("raw_candidate_return"),
                    "raw_replaced_return": swap.get("raw_replaced_return"),
                    "raw_stock_delta": swap.get("raw_stock_delta"),
                    "position_weight": swap.get("position_weight"),
                    "weighted_swap_delta": swap.get("weighted_swap_delta"),
                    "delta_realized": swap.get("delta_realized"),
                    "accepted_guard_reasons": swap.get("guard_passed_reasons"),
                    "would_be_blocked_by_ai_shadow_guard": bool(
                        swap.get("would_block_by_shadow_score_margin")
                        or swap.get("would_block_by_shadow_risk_increase")
                        or swap.get("would_block_by_shadow_default_keep")
                        or swap.get("would_block_by_shadow_rank_cap")
                    ),
                    "notes": "",
                }
            )
    detail = pd.DataFrame(rows)
    detail_path = out_dir / "negative_delta_windows.csv"
    detail.to_csv(detail_path, index=False)

    if detail.empty or "branch" not in detail:
        summary = pd.DataFrame()
    else:
        numeric_cols = ["weighted_swap_delta", "risk_delta", "score_margin", "candidate_rank", "trend_dispersion", "theme_ai_consensus"]
        for col in numeric_cols:
            detail[col] = pd.to_numeric(detail[col], errors="coerce")
        summary = (
            detail.groupby("branch", dropna=False)
            .agg(
                negative_window_count=("window", "nunique"),
                negative_weighted_delta_sum=("weighted_swap_delta", "sum"),
                avg_risk_delta=("risk_delta", "mean"),
                avg_score_margin=("score_margin", "mean"),
                avg_candidate_rank=("candidate_rank", "mean"),
                avg_trend_dispersion=("trend_dispersion", "mean"),
                avg_ai_consensus=("theme_ai_consensus", "mean"),
            )
            .reset_index()
        )
    summary_path = out_dir / "negative_delta_summary.csv"
    summary.to_csv(summary_path, index=False)
    return detail_path, summary_path


def write_ai_shadow_decision() -> Path:
    shadow = pd.read_csv(ANALYSIS_20 / "ai_shadow_guard.csv")
    focus = shadow[shadow["variant"] == "v2b_trend_plus_ai_overlay"].copy()

    def would_block(window: str) -> bool:
        row = focus[focus["window"].astype(str) == window]
        if row.empty:
            return False
        r = row.iloc[0]
        return bool(
            r.get("would_block_by_shadow_score_margin")
            or r.get("would_block_by_shadow_risk_increase")
            or r.get("would_block_by_shadow_default_keep")
            or r.get("would_block_by_shadow_rank_cap")
        )

    def delta(window: str) -> float:
        row = focus[focus["window"].astype(str) == window]
        if row.empty:
            return 0.0
        return float(pd.to_numeric(row.iloc[0].get("actual_delta"), errors="coerce"))

    saved = -delta("2026-03-30") if would_block("2026-03-30") else 0.0
    killed = sum(delta(window) for window in ["2025-12-17", "2026-02-06", "2026-04-07"] if would_block(window))
    out = pd.DataFrame(
        [
            {
                "guard_name": "shadow_theme_ai_risk_default_keep_rank_margin",
                "would_block_2026_03_30": would_block("2026-03-30"),
                "would_kill_2025_12_17": would_block("2025-12-17"),
                "would_kill_2026_02_06": would_block("2026-02-06"),
                "would_kill_2026_04_07": would_block("2026-04-07"),
                "net_effect_on_known_windows": float(saved - killed),
                "recommend_enable": False,
                "reason": "Blocks a tiny 2026-03-30 negative swap but would kill larger 2026-02-06 and 2026-04-07 positive swaps.",
            }
        ]
    )
    output = VALIDATION_ROOT / "ai_shadow_guard_decision.csv"
    out.to_csv(output, index=False)
    return output


def _read_current_row(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    row = df[
        pd.to_numeric(df["trend_dispersion_max"], errors="coerce").sub(0.14).abs().le(1e-12)
        & (pd.to_numeric(df["trend_candidate_rank_cap"], errors="coerce").astype(int) == 6)
        & pd.to_numeric(df["theme_ai_consensus_max"], errors="coerce").sub(0.70).abs().le(1e-12)
    ]
    return row.iloc[0].to_dict() if not row.empty else {}


def write_decision_report() -> Path:
    report_path = VALIDATION_ROOT / "v2b_guarded_decision_report.md"
    longer40 = pd.read_csv(VALIDATION_ROOT / "v2b_guarded_longer_40win" / "longer_window_validation.csv")
    longer60 = pd.read_csv(VALIDATION_ROOT / "v2b_guarded_longer_60win" / "longer_window_validation.csv")
    bucket = pd.read_csv(VALIDATION_ROOT / "v2b_guarded_bucket_stability" / "bucket_stability.csv")
    negative_summary = pd.read_csv(VALIDATION_ROOT / "v2b_guarded_negative_windows_60win" / "negative_delta_summary.csv")
    shadow_decision = pd.read_csv(VALIDATION_ROOT / "ai_shadow_guard_decision.csv")
    comp0309 = pd.read_csv(ANALYSIS_20 / "comparative_20260309.csv")
    recon = pd.read_csv(ANALYSIS_20 / "swap_delta_reconciliation.csv")
    sweep40_current = _read_current_row(VALIDATION_ROOT / "v2b_guarded_sweep_40win" / "threshold_sweep.csv")
    sweep60_current = _read_current_row(VALIDATION_ROOT / "v2b_guarded_sweep_60win_narrow" / "threshold_sweep.csv")

    guarded60 = longer60[longer60["variant"] == "v2b_guarded_candidate"]
    bucket_guarded = bucket[bucket["variant"] == "v2b_guarded_candidate"][["bucket", "mean_delta_vs_default", "accepted_swaps", "trend_weighted_delta_sum", "ai_weighted_delta_sum"]]
    row0309 = comp0309[comp0309["window"].astype(str) == "2026-03-09"].iloc[0]
    max_recon = float(pd.to_numeric(recon["reconciliation_error"], errors="coerce").abs().max())

    lines = [
        "# v2b_guarded_candidate decision report",
        "",
        "## Executive summary",
        "v2b_guarded_candidate remains a reasonable current active candidate, but the longer-window advantage is clearly weaker than the recent 20-window result.",
        "",
        "## Current frozen config",
        "- branch_router_v2b.enabled = True",
        "- crash_minrisk_enabled = False",
        "- trend_max_swaps = 1, theme_ai_max_swaps = 1, max_total_swaps = 2",
        "- trend_dispersion_max = 0.14, trend_candidate_rank_cap = 6, theme_ai_consensus_max = 0.70",
        "",
        "## 20/40/60 longer validation",
        "```text\n" + guarded60.to_string(index=False) + "\n```",
        "",
        "## Delta semantics audit",
        "raw_stock_delta is single-stock candidate minus replaced realized return. weighted_swap_delta multiplies that value by the top5 position weight. delta_realized uses the portfolio-weighted convention.",
        "",
        "## Reconciliation result",
        f"Maximum absolute reconciliation error: {max_recon:.3e}.",
        "",
        "## 40win full sweep result",
        json.dumps(sweep40_current, ensure_ascii=False, indent=2, default=str) if sweep40_current else "40win full sweep not available.",
        "",
        "## 60win narrow sweep result",
        json.dumps(sweep60_current, ensure_ascii=False, indent=2, default=str) if sweep60_current else "60win narrow sweep not available.",
        "",
        "## Plateau stability judgment",
        "Use sweep_decision_summary.csv for the formal plateau note. Current config should be kept only if its neighborhood remains close on mean while preserving q10/worst and negative_delta_count.",
        "",
        "## Bucket stability: recent/mid/old 20",
        "```text\n" + bucket_guarded.to_string(index=False) + "\n```",
        "",
        "## Negative delta windows analysis",
        ("```text\n" + negative_summary.to_string(index=False) + "\n```") if not negative_summary.empty else "No negative delta swap rows found.",
        "",
        "## AI shadow guard decision",
        "```text\n" + shadow_decision.to_string(index=False) + "\n```",
        "",
        "## 2026-03-09 trend blocked case decision",
        f"raw_stock_delta={float(row0309['raw_stock_delta']):.6f}, position_weight={float(row0309['position_weight']):.2f}, weighted_delta={float(row0309['would_have_weighted_delta']):.6f}, blocked_guard_reasons={row0309['blocked_guard_reasons']}. No runtime exception is recommended.",
        "",
        "## Final recommendation",
        "- Keep v2b_guarded_candidate as current active candidate.",
        "- Keep crash_minrisk_enabled = False.",
        "- Do not enable AI shadow guard.",
        "- Do not add a 2026-03-09 exception.",
        "- Do not use baseline_hybrid at runtime.",
        "- Do not hard switch whole windows.",
        "- Full all_available / 95-anchor validation remains optional future work.",
        "",
        "## Risk note",
        "The v2b advantage weakens from 20win to 60win: +0.006369 -> +0.003062 -> +0.001156. Treat this as recent effective alpha, not a full-history large stable alpha; avoid further threshold overfitting from the 20win slice.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    bucket = write_bucket_stability()
    negative_detail, negative_summary = write_negative_windows()
    shadow = write_ai_shadow_decision()
    report = write_decision_report()
    print(f"bucket_stability={bucket}")
    print(f"negative_delta_windows={negative_detail}")
    print(f"negative_delta_summary={negative_summary}")
    print(f"ai_shadow_guard_decision={shadow}")
    print(f"decision_report={report}")


if __name__ == "__main__":
    main()
