from __future__ import annotations

from typing import Any

import pandas as pd


def paired_delta_distribution(
    rows: pd.DataFrame,
    *,
    variant_col: str = "variant",
    window_col: str = "window_date",
    score_col: str = "score",
    default_variant: str = "default_grr_tail_guard",
    zero_tol: float = 1e-12,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    base = rows[rows[variant_col] == default_variant][[window_col, score_col]].rename(columns={score_col: "_default_score"})
    out = []
    for variant, group in rows.groupby(variant_col, sort=True):
        merged = group[[window_col, score_col]].merge(base, on=window_col, how="inner")
        delta = pd.to_numeric(merged[score_col], errors="coerce") - pd.to_numeric(merged["_default_score"], errors="coerce")
        delta = delta.dropna()
        if delta.empty:
            continue
        out.append(
            {
                "variant": variant,
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "q10_delta": float(delta.quantile(0.10)),
                "q25_delta": float(delta.quantile(0.25)),
                "q75_delta": float(delta.quantile(0.75)),
                "worst_delta": float(delta.min()),
                "best_delta": float(delta.max()),
                "positive_delta_count": int((delta > zero_tol).sum()),
                "negative_delta_count": int((delta < -zero_tol).sum()),
                "zero_delta_count": int(delta.abs().le(zero_tol).sum()),
                "window_count": int(len(delta)),
            }
        )
    return pd.DataFrame(out)


def aggregate_guard_summary(rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    columns = [
        "branch",
        "candidates_generated",
        "accepted",
        "blocked_total",
        "blocked_by_trend_dispersion",
        "blocked_by_candidate_rank",
        "blocked_by_default_strong_keep",
        "blocked_by_risk_increase",
        "blocked_by_ai_consensus",
        "blocked_by_branch_cap",
        "blocked_by_total_cap",
        "accept_rate",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    numeric = [col for col in columns if col not in {"branch", "accept_rate"}]
    grouped = frame.groupby("branch", as_index=False)[numeric].sum()
    grouped["accept_rate"] = grouped["accepted"] / grouped["candidates_generated"].where(grouped["candidates_generated"] != 0, 1)
    return grouped[columns]


def swap_delta_reconciliation(
    ablation_rows: pd.DataFrame,
    accepted_swaps: pd.DataFrame,
    *,
    variant: str = "v2b_trend_plus_ai_overlay",
    default_variant: str = "default_grr_tail_guard",
    window_col: str = "window_date",
    score_col: str = "score",
    zero_tol: float = 1e-10,
) -> pd.DataFrame:
    columns = [
        "window",
        "variant",
        "default_window_return",
        "variant_window_return",
        "window_delta_vs_default",
        "sum_accepted_weighted_swap_delta",
        "reconciliation_error",
        "accepted_swap_count",
        "reconciliation_note",
    ]
    if ablation_rows.empty:
        return pd.DataFrame(columns=columns)

    base = ablation_rows[ablation_rows["variant"] == default_variant][[window_col, score_col]]
    base = base.rename(columns={window_col: "window", score_col: "default_window_return"})
    routed = ablation_rows[ablation_rows["variant"] == variant][[window_col, score_col]]
    routed = routed.rename(columns={window_col: "window", score_col: "variant_window_return"})
    merged = base.merge(routed, on="window", how="inner")

    if accepted_swaps.empty:
        grouped = pd.DataFrame(columns=["window", "sum_accepted_weighted_swap_delta", "accepted_swap_count"])
    else:
        subset = accepted_swaps[accepted_swaps.get("variant", pd.Series(dtype=str)).astype(str) == variant].copy()
        if subset.empty:
            grouped = pd.DataFrame(columns=["window", "sum_accepted_weighted_swap_delta", "accepted_swap_count"])
        else:
            subset["weighted_swap_delta"] = pd.to_numeric(subset.get("weighted_swap_delta"), errors="coerce").fillna(0.0)
            grouped = subset.groupby("window", as_index=False).agg(
                sum_accepted_weighted_swap_delta=("weighted_swap_delta", "sum"),
                accepted_swap_count=("weighted_swap_delta", "size"),
            )

    out = merged.merge(grouped, on="window", how="left")
    out["sum_accepted_weighted_swap_delta"] = pd.to_numeric(out["sum_accepted_weighted_swap_delta"], errors="coerce").fillna(0.0)
    out["accepted_swap_count"] = pd.to_numeric(out["accepted_swap_count"], errors="coerce").fillna(0).astype(int)
    out["window_delta_vs_default"] = (
        pd.to_numeric(out["variant_window_return"], errors="coerce")
        - pd.to_numeric(out["default_window_return"], errors="coerce")
    )
    out["reconciliation_error"] = out["window_delta_vs_default"] - out["sum_accepted_weighted_swap_delta"]
    out["variant"] = variant
    out["reconciliation_note"] = out["reconciliation_error"].abs().gt(zero_tol).map(
        {True: "non_swap_effect_or_missing_return", False: ""}
    )
    return out[columns]
