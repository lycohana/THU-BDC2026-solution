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

from batch_window_analysis import (  # noqa: E402
    add_branch_diagnostic_features,
    filter_branch,
    load_raw,
    normalize_stock_id,
    realized_returns_for_anchor,
)


DEFAULT_SOURCE_RUN = ROOT / "temp" / "batch_window_analysis" / "grr_tail_guard_60win"
DEFAULT_OUT_DIR = ROOT / "temp" / "volume_expansion_shadow"


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _rank_pct(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill = values.median() if values.notna().any() else 0.0
    return values.fillna(fill).rank(method="average", pct=True)


def _z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean = values.mean()
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return ((values - mean) / std).fillna(0.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


def _split_picks(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        return []
    return normalize_stock_id(pd.Series(parts, dtype=str)).astype(str).tolist()


def _window_tag(idx: int, total: int) -> str:
    if idx >= total - 20:
        return "recent_20"
    if idx >= total - 40:
        return "mid_20"
    return "old_20"


def _mean_for_ids(work: pd.DataFrame, stock_ids: list[str], col: str = "future_ret5") -> float:
    if not stock_ids:
        return 0.0
    selected = work[work["stock_id"].astype(str).isin(set(stock_ids))].copy()
    order = {stock_id: i for i, stock_id in enumerate(stock_ids)}
    selected["_order"] = selected["stock_id"].astype(str).map(order)
    selected = selected.sort_values("_order")
    return float(pd.to_numeric(selected.get(col, 0.0), errors="coerce").fillna(0.0).head(5).mean())


def _bad_counts(work: pd.DataFrame, stock_ids: list[str]) -> tuple[int, int]:
    selected = work[work["stock_id"].astype(str).isin(set(stock_ids))].copy()
    rets = pd.to_numeric(selected.get("future_ret5", 0.0), errors="coerce").fillna(0.0)
    return int((rets < -0.03).sum()), int((rets < -0.05).sum())


def _select_top_ids(frame: pd.DataFrame, score_col: str, n: int = 5) -> set[str]:
    if frame.empty or score_col not in frame.columns:
        return set()
    out = frame.copy()
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0)
    return set(out.sort_values(score_col, ascending=False).head(n)["stock_id"].astype(str).tolist())


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / (b.replace(0.0, np.nan) + 1e-12)


def build_raw_feature_panel(raw: pd.DataFrame) -> pd.DataFrame:
    panel = raw.copy()
    panel["stock_id"] = normalize_stock_id(panel["股票代码"])
    panel["date"] = pd.to_datetime(panel["日期"])
    for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel = panel.sort_values(["stock_id", "date"]).reset_index(drop=True)
    group = panel.groupby("stock_id", sort=False)

    amount = panel["成交额"].fillna(0.0)
    turnover = panel["换手率"].fillna(0.0)
    close = panel["收盘"]
    high = panel["最高"]
    low = panel["最低"]

    panel["amount_ma5"] = group["成交额"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    panel["amount_ma20"] = group["成交额"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    panel["amount_ma60"] = group["成交额"].transform(lambda s: s.rolling(60, min_periods=20).mean())
    panel["amount_lag5"] = group["成交额"].shift(5)
    panel["amount_lag20"] = group["成交额"].shift(20)
    panel["turnover_ma20"] = group["换手率"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    panel["close_ma20"] = group["收盘"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    panel["high120"] = group["最高"].transform(lambda s: s.rolling(120, min_periods=30).max())
    panel["low120"] = group["最低"].transform(lambda s: s.rolling(120, min_periods=30).min())
    panel["prior_close_high20"] = group["收盘"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).max())

    prev_close = group["收盘"].shift(1)
    panel["ret1_raw"] = close / (prev_close + 1e-12) - 1.0
    panel["ret3_raw"] = close / (group["收盘"].shift(3) + 1e-12) - 1.0
    panel["ret1_std20"] = group["ret1_raw"].transform(lambda s: s.rolling(20, min_periods=10).std(ddof=0))
    panel["ret1_std60"] = group["ret1_raw"].transform(lambda s: s.rolling(60, min_periods=20).std(ddof=0))

    panel["amount_rel_ma5"] = _safe_div(amount, panel["amount_ma5"])
    panel["amount_rel_ma20"] = _safe_div(amount, panel["amount_ma20"])
    panel["amount_rel_ma60"] = _safe_div(amount, panel["amount_ma60"])
    panel["amount_ma5_div_ma20"] = _safe_div(panel["amount_ma5"], panel["amount_ma20"])
    panel["amount_ma20_div_ma60"] = _safe_div(panel["amount_ma20"], panel["amount_ma60"])
    panel["amount_chg_5d"] = _safe_div(amount, panel["amount_lag5"]) - 1.0
    panel["amount_chg_20d"] = _safe_div(amount, panel["amount_lag20"]) - 1.0
    panel["turnover_expansion"] = _safe_div(turnover, panel["turnover_ma20"])
    panel["amount_ma20_trend5"] = _safe_div(panel["amount_ma20"], group["amount_ma20"].shift(5)) - 1.0
    panel["amount_consistency_5"] = group["成交额"].transform(
        lambda s: (s > s.rolling(20, min_periods=10).mean()).rolling(5, min_periods=1).sum()
    )
    panel["price_position_120d"] = (close - panel["low120"]) / ((panel["high120"] - panel["low120"]) + 1e-12)
    panel["close_div_ma20"] = _safe_div(close, panel["close_ma20"])
    panel["close_breakout_20"] = (close > panel["prior_close_high20"]).astype(float)
    panel["sigma20_div_sigma60"] = _safe_div(panel["ret1_std20"], panel["ret1_std60"])
    high20 = group["最高"].transform(lambda s: s.rolling(20, min_periods=10).max())
    low20 = group["最低"].transform(lambda s: s.rolling(20, min_periods=10).min())
    panel["amp_raw20"] = high20 / (low20 + 1e-12) - 1.0
    panel["drawdown_raw20"] = close / (group["收盘"].transform(lambda s: s.rolling(20, min_periods=10).max()) + 1e-12) - 1.0
    panel["log_liquidity_raw"] = np.log1p(panel["amount_ma20"].clip(lower=0.0))
    panel["log_liquidity_ts_z"] = group["log_liquidity_raw"].transform(lambda s: (s - s.rolling(60, min_periods=20).mean()) / (s.rolling(60, min_periods=20).std(ddof=0) + 1e-12))

    keep = [
        "stock_id",
        "date",
        "amount_rel_ma5",
        "amount_rel_ma20",
        "amount_rel_ma60",
        "amount_ma5_div_ma20",
        "amount_ma20_div_ma60",
        "amount_chg_5d",
        "amount_chg_20d",
        "turnover_expansion",
        "amount_ma20_trend5",
        "amount_consistency_5",
        "price_position_120d",
        "close_div_ma20",
        "close_breakout_20",
        "sigma20_div_sigma60",
        "amp_raw20",
        "drawdown_raw20",
        "log_liquidity_ts_z",
    ]
    return panel[keep].replace([np.inf, -np.inf], np.nan)


def add_volume_event_features(work: pd.DataFrame) -> pd.DataFrame:
    out = work.copy()
    out["amount_rel_ma20"] = _num(out, "amount_rel_ma20")
    out["amount_ma5_div_ma20"] = _num(out, "amount_ma5_div_ma20")
    out["amount_ma20_div_ma60"] = _num(out, "amount_ma20_div_ma60")
    out["amount_chg_5d"] = _num(out, "amount_chg_5d")
    out["turnover_expansion"] = _num(out, "turnover_expansion")
    out["log_liquidity_ts_z"] = _num(out, "log_liquidity_ts_z")
    out["ret5"] = _num(out, "ret5")
    out["ret20"] = _num(out, "ret20")
    out["sigma20"] = _num(out, "sigma20")
    out["amp20"] = _num(out, "amp20")
    out["max_drawdown20"] = _num(out, "max_drawdown20")
    out["downside_beta60"] = _num(out, "downside_beta60")
    out["median_amount20"] = _num(out, "median_amount20")
    out["price_position_120d"] = _num(out, "price_position_120d", default=0.5).clip(0.0, 1.0)
    out["close_breakout_20"] = _num(out, "close_breakout_20")
    out["sigma20_div_sigma60"] = _num(out, "sigma20_div_sigma60", default=1.0)
    out["amount_consistency_5"] = _num(out, "amount_consistency_5")
    out["amount_ma20_trend5"] = _num(out, "amount_ma20_trend5")
    out["ret5_abs"] = out["ret5"].abs()

    out["amount_rel_ma20_rank"] = _rank_pct(out["amount_rel_ma20"])
    out["amount_chg_5d_rank"] = _rank_pct(out["amount_chg_5d"])
    out["turnover_expansion_rank"] = _rank_pct(out["turnover_expansion"])
    out["liquidity_rank"] = _rank_pct(out["median_amount20"])
    out["sigma_rank"] = _rank_pct(out["sigma20"]) if "sigma_rank" not in out.columns else _num(out, "sigma_rank")
    out["amp_rank"] = _rank_pct(out["amp20"]) if "amp_rank" not in out.columns else _num(out, "amp_rank")
    out["drawdown_bad_rank"] = _rank_pct(-out["max_drawdown20"])
    out["downside_beta_rank"] = _rank_pct(out["downside_beta60"])

    out["volume_strength_score"] = (
        0.35 * out["amount_rel_ma20_rank"]
        + 0.25 * out["amount_chg_5d_rank"]
        + 0.20 * out["turnover_expansion_rank"]
        + 0.10 * _rank_pct(out["amount_consistency_5"])
        + 0.10 * _rank_pct(out["log_liquidity_ts_z"])
    )
    risk_penalty = (
        0.30 * out["sigma_rank"]
        + 0.25 * out["amp_rank"]
        + 0.25 * out["drawdown_bad_rank"]
        + 0.20 * out["downside_beta_rank"]
    )
    overheat_penalty = ((out["ret5_abs"] - 0.08) / 0.08).clip(lower=0.0, upper=1.0)
    position_penalty = ((out["price_position_120d"] - 0.75) / 0.25).clip(lower=0.0, upper=1.0)
    out["volume_risk_penalty"] = risk_penalty
    out["health_volume_score"] = (
        out["volume_strength_score"]
        + 0.08 * (out["ret5"] > 0).astype(float)
        + 0.04 * (out["amount_ma20_trend5"] > 0).astype(float)
        - 0.18 * overheat_penalty
        - 0.12 * position_penalty
        - 0.10 * risk_penalty
    )
    out["health_volume_rank"] = _rank_pct(out["health_volume_score"])
    out["flow_persistence_score"] = (
        0.25 * _rank_pct(out["amount_ma5_div_ma20"])
        + 0.25 * _rank_pct(out["amount_ma20_div_ma60"])
        + 0.20 * _rank_pct(out["amount_ma20_trend5"])
        + 0.15 * _rank_pct(out["ret5"])
        + 0.10 * out["liquidity_rank"]
        - 0.15 * out["volume_risk_penalty"]
    )
    out["flow_persistence_rank"] = _rank_pct(out["flow_persistence_score"])
    out["vcp_breakout_score"] = (
        0.25 * (1.0 - _rank_pct(out["sigma20_div_sigma60"]))
        + 0.25 * out["close_breakout_20"]
        + 0.20 * _rank_pct(out["amount_rel_ma20"])
        + 0.15 * _rank_pct(out["ret5"])
        + 0.10 * (1.0 - out["volume_risk_penalty"])
        + 0.05 * (out["price_position_120d"].between(0.30, 0.80)).astype(float)
    )
    out["vcp_breakout_rank"] = _rank_pct(out["vcp_breakout_score"])

    out["pass_risk_hardcap"] = (
        (out["ret5_abs"] < 0.12)
        & (out["sigma20"] < 0.045)
        & (out["amp20"] < 0.09)
        & (out["max_drawdown20"] > -0.12)
        & (out["downside_beta60"] < 1.3)
        & (out["liquidity_rank"] > 0.30)
    )
    out["event_type"] = "None"
    healthy = (
        (out["amount_rel_ma20"].between(1.4, 2.5))
        & (out["health_volume_rank"] >= 0.80)
        & (out["amount_consistency_5"] >= 2)
        & (out["ret5"] > 0)
        & (out["ret20"] < 0.25)
        & (out["price_position_120d"] < 0.75)
        & out["pass_risk_hardcap"]
    )
    liquidity = (
        (out["amount_rel_ma20"].between(1.2, 1.8))
        & (out["turnover_expansion"].between(1.0, 2.2))
        & (out["amount_ma20_trend5"] > 0)
        & (out["sigma_rank"] < 0.70)
        & (out["ret5_abs"] < 0.08)
        & out["pass_risk_hardcap"]
        & ~healthy
    )
    flow = (
        (out["amount_ma5_div_ma20"] > 1.05)
        & (out["amount_ma20_div_ma60"] > 1.02)
        & (out["amount_ma20_trend5"] > 0)
        & (out["flow_persistence_rank"] >= 0.80)
        & (out["ret5"] > 0)
        & (out["ret20"] > 0)
        & (out["ret5_abs"] < 0.12)
        & (out["price_position_120d"] < 0.80)
        & out["pass_risk_hardcap"]
        & ~healthy
        & ~liquidity
    )
    vcp = (
        (out["sigma20_div_sigma60"] < 0.80)
        & (out["close_breakout_20"] > 0)
        & (out["amount_rel_ma20"] > 1.20)
        & (out["ret5"] > 0)
        & (out["ret5"] < 0.08)
        & (out["price_position_120d"].between(0.30, 0.80))
        & (out["vcp_breakout_rank"] >= 0.80)
        & out["pass_risk_hardcap"]
        & ~healthy
        & ~liquidity
        & ~flow
    )
    terminal = (
        ((out["amount_rel_ma20"] > 2.5) | (out["amount_chg_5d"] > 2.0))
        & ((out["ret20"] > 0.25) | (out["price_position_120d"] > 0.80))
        & ((out["sigma_rank"] > 0.75) | (out["amp_rank"] > 0.75) | (out["ret5"] <= 0.01))
    )
    trap = (
        (out["amount_rel_ma20"] > 1.8)
        & ((out["sigma20"] > 0.05) | (out["amp20"] > 0.10) | (out["ret5_abs"] > 0.12))
    )
    out.loc[liquidity, "event_type"] = "LiquidityImprove"
    out.loc[flow, "event_type"] = "FlowPersist"
    out.loc[vcp, "event_type"] = "VCPBreakout"
    out.loc[healthy, "event_type"] = "HealthyVolume"
    out.loc[terminal, "event_type"] = "TerminalSpike"
    out.loc[trap & ~terminal, "event_type"] = "HighVolTrap"
    return out


def _build_compare_sets(work: pd.DataFrame) -> dict[str, set[str]]:
    branch_sets = {
        "in_grr_top5": _select_top_ids(work, "grr_final_score", n=5),
        "in_riskoff_top60": set(filter_branch(work, "stable").sort_values("score_legal_minrisk" if "score_legal_minrisk" in work.columns else "score", ascending=False).head(60)["stock_id"].astype(str).tolist()),
        "in_trend_candidate": set(),
        "in_ai_candidate": set(),
    }
    try:
        trend = filter_branch(work, "portfolio:regime_trend_uncluttered_plus_reversal")
        branch_sets["in_trend_candidate"] = _select_top_ids(trend, "score", n=20)
    except Exception:
        branch_sets["in_trend_candidate"] = set()
    try:
        ai = filter_branch(work, "portfolio:regime_ai_hardware_mainline_v1")
        branch_sets["in_ai_candidate"] = _select_top_ids(ai, "score", n=20)
    except Exception:
        branch_sets["in_ai_candidate"] = set()
    return branch_sets


def build_shadow_tables(source_run: Path, raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = load_raw(raw_path)
    raw_features = build_raw_feature_panel(raw)
    windows = pd.read_csv(source_run / "window_summary.csv")
    windows["anchor_date"] = pd.to_datetime(windows["anchor_date"]).dt.strftime("%Y-%m-%d")
    total = len(windows)
    event_rows: list[pd.DataFrame] = []
    overlay_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    for idx, win in windows.sort_values("anchor_date").reset_index(drop=True).iterrows():
        anchor = pd.Timestamp(win["anchor_date"])
        window = anchor.strftime("%Y-%m-%d")
        score_path = source_run / anchor.strftime("%Y%m%d") / "predict_score_df.csv"
        if not score_path.exists():
            continue
        score_df = pd.read_csv(score_path, dtype={"stock_id": str})
        score_df["stock_id"] = normalize_stock_id(score_df["stock_id"])
        realized, _ = realized_returns_for_anchor(raw, anchor, label_horizon=5)
        realized = realized[["stock_id", "realized_ret"]].rename(columns={"realized_ret": "future_ret5"})
        features = raw_features[raw_features["date"] == anchor].drop(columns=["date"])
        work = score_df.merge(realized, on="stock_id", how="left").merge(features, on="stock_id", how="left")
        work["future_ret5"] = _num(work, "future_ret5")
        work = add_branch_diagnostic_features(work)
        work = add_volume_event_features(work)

        window_tag = _window_tag(idx, total)
        compare_sets = _build_compare_sets(work)
        default_top5 = _split_picks(win.get("selected_picks", ""))
        default_return = _mean_for_ids(work, default_top5)
        default_bad, default_very_bad = _bad_counts(work, default_top5)

        for col, ids in compare_sets.items():
            work[col] = work["stock_id"].astype(str).isin(ids)
        work["trade_date"] = window
        work["window_tag"] = window_tag
        work["base_future_ret5"] = default_return
        work["delta_vs_base"] = work["future_ret5"] - default_return

        keep_cols = [
            "trade_date",
            "window_tag",
            "stock_id",
            "event_type",
            "amount_rel_ma20",
            "amount_rel_ma60",
            "amount_ma5_div_ma20",
            "amount_ma20_div_ma60",
            "amount_chg_5d",
            "amount_chg_20d",
            "turnover_expansion",
            "log_liquidity_ts_z",
            "health_volume_score",
            "health_volume_rank",
            "flow_persistence_score",
            "flow_persistence_rank",
            "vcp_breakout_score",
            "vcp_breakout_rank",
            "close_breakout_20",
            "sigma20_div_sigma60",
            "volume_risk_penalty",
            "ret5",
            "ret20",
            "ret5_abs",
            "sigma20",
            "amp20",
            "max_drawdown20",
            "downside_beta60",
            "price_position_120d",
            "median_amount20",
            "liquidity_rank",
            "pass_risk_hardcap",
            "in_grr_top5",
            "in_riskoff_top60",
            "in_trend_candidate",
            "in_ai_candidate",
            "future_ret5",
            "base_future_ret5",
            "delta_vs_base",
        ]
        event_rows.append(work[[col for col in keep_cols if col in work.columns]].copy())

        overlay_rows.append(
            evaluate_volume_overlay_window(
                work,
                default_top5,
                default_return,
                default_bad,
                default_very_bad,
                window,
                window_tag,
                variant="volume_strict_model_gate",
                min_score_advantage=0.03,
                allowed_event_types=("HealthyVolume", "LiquidityImprove"),
            )
        )
        overlay_rows.append(
            evaluate_volume_overlay_window(
                work,
                default_top5,
                default_return,
                default_bad,
                default_very_bad,
                window,
                window_tag,
                variant="volume_health_only_shadow",
                min_score_advantage=None,
                allowed_event_types=("HealthyVolume",),
            )
        )
        overlay_rows.append(
            evaluate_volume_overlay_window(
                work,
                default_top5,
                default_return,
                default_bad,
                default_very_bad,
                window,
                window_tag,
                variant="flow_persistence_shadow",
                min_score_advantage=None,
                allowed_event_types=("FlowPersist",),
            )
        )
        overlay_rows.append(
            evaluate_volume_overlay_window(
                work,
                default_top5,
                default_return,
                default_bad,
                default_very_bad,
                window,
                window_tag,
                variant="vcp_breakout_shadow",
                min_score_advantage=None,
                allowed_event_types=("VCPBreakout",),
            )
        )
        window_rows.append(
            {
                "trade_date": window,
                "window_tag": window_tag,
                "default_return": default_return,
                "default_bad_count": default_bad,
                "default_very_bad_count": default_very_bad,
                "base_top5": ",".join(default_top5),
                "event_count": int((work["event_type"] != "None").sum()),
                "healthy_count": int((work["event_type"] == "HealthyVolume").sum()),
                "liquidity_count": int((work["event_type"] == "LiquidityImprove").sum()),
                "flow_count": int((work["event_type"] == "FlowPersist").sum()),
                "vcp_count": int((work["event_type"] == "VCPBreakout").sum()),
            }
        )

    events = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()
    overlay_log = pd.DataFrame(overlay_rows)
    windows_out = pd.DataFrame(window_rows)
    if not overlay_log.empty:
        overlay_log = overlay_log.sort_values("trade_date").reset_index(drop=True)
        overlay_log["rolling_20d_cum_delta"] = overlay_log.groupby("overlay_variant")["delta_ret5_offline"].transform(
            lambda s: s.rolling(20, min_periods=1).sum()
        )
        overlay_log["rolling_20d_q10_delta"] = overlay_log.groupby("overlay_variant")["delta_ret5_offline"].transform(
            lambda s: s.rolling(20, min_periods=1).quantile(0.10)
        )
        overlay_log["rolling_20d_negative_delta_count"] = overlay_log.groupby("overlay_variant")["delta_ret5_offline"].transform(
            lambda s: s.lt(-1e-12).rolling(20, min_periods=1).sum()
        )
    return events, overlay_log, windows_out


def evaluate_volume_overlay_window(
    work: pd.DataFrame,
    default_top5: list[str],
    default_return: float,
    default_bad: int,
    default_very_bad: int,
    window: str,
    window_tag: str,
    variant: str,
    min_score_advantage: float | None,
    allowed_event_types: tuple[str, ...],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "overlay_variant": variant,
        "trade_date": window,
        "window_tag": window_tag,
        "base_top5": ",".join(default_top5),
        "weakest_stock": "",
        "weakest_score": np.nan,
        "weakest_tail_penalty": np.nan,
        "candidate_stock": "",
        "candidate_event_type": "",
        "candidate_health_volume_score": np.nan,
        "candidate_risk_score": np.nan,
        "score_advantage": np.nan,
        "swap_accepted": False,
        "blocked_reason": "",
        "post_swap_top5": ",".join(default_top5),
        "future_ret5_base_offline": default_return,
        "future_ret5_post_swap_offline": default_return,
        "delta_ret5_offline": 0.0,
        "base_bad_count": default_bad,
        "base_very_bad_count": default_very_bad,
        "post_bad_count": default_bad,
        "post_very_bad_count": default_very_bad,
        "raw_candidate_return": np.nan,
        "raw_replaced_return": np.nan,
        "raw_stock_delta": np.nan,
        "weighted_swap_delta": 0.0,
    }
    if not default_top5:
        row["blocked_reason"] = "missing_base_top5"
        return row

    selected = work[work["stock_id"].astype(str).isin(set(default_top5))].copy()
    if selected.empty:
        row["blocked_reason"] = "missing_base_rows"
        return row
    selected["_base_order"] = selected["stock_id"].astype(str).map({stock: i + 1 for i, stock in enumerate(default_top5)})
    score_col = "grr_final_score" if "grr_final_score" in selected.columns else "score"
    selected["_base_score"] = _num(selected, score_col)
    selected["_tail_penalty"] = _num(selected, "volume_risk_penalty")
    weakest = selected.sort_values(["_base_score", "_tail_penalty"], ascending=[True, False]).iloc[0]
    weakest_stock = str(weakest["stock_id"])
    weakest_score = _safe_float(weakest["_base_score"])
    weakest_penalty = _safe_float(weakest["_tail_penalty"])
    row.update(
        {
            "weakest_stock": weakest_stock,
            "weakest_score": weakest_score,
            "weakest_tail_penalty": weakest_penalty,
        }
    )

    candidate_pool = work[
        work["event_type"].isin(list(allowed_event_types))
        & work["pass_risk_hardcap"].astype(bool)
        & ~work["stock_id"].astype(str).isin(set(default_top5))
    ].copy()
    if candidate_pool.empty:
        row["blocked_reason"] = "no_volume_candidate"
        return row
    candidate_pool["_score"] = _num(candidate_pool, score_col)
    candidate_pool["_risk_score"] = _num(candidate_pool, "volume_risk_penalty")
    candidate_pool = candidate_pool.sort_values(["health_volume_score", "_score"], ascending=False)
    candidate = candidate_pool.iloc[0]
    candidate_stock = str(candidate["stock_id"])
    candidate_score = _safe_float(candidate["_score"])
    score_advantage = candidate_score - weakest_score
    row.update(
        {
            "candidate_stock": candidate_stock,
            "candidate_event_type": candidate.get("event_type", ""),
            "candidate_health_volume_score": _safe_float(candidate.get("health_volume_score")),
            "candidate_risk_score": _safe_float(candidate.get("_risk_score")),
            "score_advantage": score_advantage,
            "raw_candidate_return": _safe_float(candidate.get("future_ret5")),
            "raw_replaced_return": _safe_float(weakest.get("future_ret5")),
        }
    )
    raw_delta = row["raw_candidate_return"] - row["raw_replaced_return"]
    row["raw_stock_delta"] = raw_delta
    row["weighted_swap_delta"] = 0.2 * raw_delta

    if min_score_advantage is not None and score_advantage <= min_score_advantage:
        row["blocked_reason"] = "score_advantage_too_small"
        return row
    if _safe_float(candidate.get("_risk_score")) > weakest_penalty + 0.10:
        row["blocked_reason"] = "candidate_risk_too_high_vs_weakest"
        return row

    post_top5 = list(default_top5)
    post_top5[post_top5.index(weakest_stock)] = candidate_stock
    post_return = _mean_for_ids(work, post_top5)
    post_bad, post_very_bad = _bad_counts(work, post_top5)
    row.update(
        {
            "swap_accepted": True,
            "blocked_reason": "",
            "post_swap_top5": ",".join(post_top5),
            "future_ret5_post_swap_offline": post_return,
            "delta_ret5_offline": post_return - default_return,
            "post_bad_count": post_bad,
            "post_very_bad_count": post_very_bad,
        }
    )
    return row


def _metric(values: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return {"count": 0, "mean": 0.0, "median": 0.0, "q10": 0.0, "worst": 0.0, "hit_rate": 0.0, "positive_count": 0, "negative_count": 0}
    return {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "q10": float(s.quantile(0.10)),
        "worst": float(s.min()),
        "hit_rate": float((s > 0).mean()),
        "positive_count": int((s > 0).sum()),
        "negative_count": int((s < 0).sum()),
    }


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if events.empty:
        return pd.DataFrame()
    scoped = events[events["event_type"] != "None"].copy()
    for tag in ["recent_20", "mid_20", "old_20", "all_60"]:
        sub_tag = scoped if tag == "all_60" else scoped[scoped["window_tag"] == tag]
        for event_type, sub in sub_tag.groupby("event_type"):
            base = {"window_tag": tag, "event_type": event_type, "event_count": int(len(sub))}
            ret = {f"future_ret5_{k}": v for k, v in _metric(sub["future_ret5"]).items()}
            delta = {f"delta_vs_base_{k}": v for k, v in _metric(sub["delta_vs_base"]).items()}
            base.update(ret)
            base.update(delta)
            base["coverage_dates"] = int(sub["trade_date"].nunique())
            base["riskcap_pass_rate"] = float(sub["pass_risk_hardcap"].mean()) if len(sub) else 0.0
            base["overlap_grr_top5_rate"] = float(sub["in_grr_top5"].mean()) if len(sub) else 0.0
            base["overlap_riskoff_top60_rate"] = float(sub["in_riskoff_top60"].mean()) if len(sub) else 0.0
            base["overlap_trend_rate"] = float(sub["in_trend_candidate"].mean()) if len(sub) else 0.0
            base["overlap_ai_rate"] = float(sub["in_ai_candidate"].mean()) if len(sub) else 0.0
            rows.append(base)
    return pd.DataFrame(rows)


def summarize_overlay(overlay: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if overlay.empty:
        return pd.DataFrame()
    variants = sorted(overlay["overlay_variant"].dropna().unique()) if "overlay_variant" in overlay.columns else ["unknown"]
    for variant in variants:
        var_frame = overlay[overlay["overlay_variant"] == variant] if "overlay_variant" in overlay.columns else overlay
        for tag in ["recent_20", "mid_20", "old_20", "all_60"]:
            sub = var_frame if tag == "all_60" else var_frame[var_frame["window_tag"] == tag]
            metrics = _metric(sub["delta_ret5_offline"])
            rows.append(
                {
                    "overlay_variant": variant,
                    "window_tag": tag,
                    "window_count": int(len(sub)),
                    "accepted_swaps": int(sub["swap_accepted"].sum()) if len(sub) else 0,
                    "accepted_swap_rate": float(sub["swap_accepted"].mean()) if len(sub) else 0.0,
                    "blocked_no_candidate": int((sub["blocked_reason"] == "no_volume_candidate").sum()) if len(sub) else 0,
                    "blocked_score": int((sub["blocked_reason"] == "score_advantage_too_small").sum()) if len(sub) else 0,
                    "blocked_risk": int((sub["blocked_reason"] == "candidate_risk_too_high_vs_weakest").sum()) if len(sub) else 0,
                    "mean_default": float(pd.to_numeric(sub["future_ret5_base_offline"], errors="coerce").mean()) if len(sub) else 0.0,
                    "mean_overlay": float(pd.to_numeric(sub["future_ret5_post_swap_offline"], errors="coerce").mean()) if len(sub) else 0.0,
                    **{f"delta_{k}": v for k, v in metrics.items()},
                    "worst_overlay": float(pd.to_numeric(sub["future_ret5_post_swap_offline"], errors="coerce").min()) if len(sub) else 0.0,
                    "q10_overlay": float(pd.to_numeric(sub["future_ret5_post_swap_offline"], errors="coerce").quantile(0.10)) if len(sub) else 0.0,
                    "negative_delta_count": int((pd.to_numeric(sub["delta_ret5_offline"], errors="coerce").fillna(0.0) < -1e-12).sum()) if len(sub) else 0,
                }
            )
    return pd.DataFrame(rows)


def write_outputs(events: pd.DataFrame, overlay: pd.DataFrame, windows: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_dir / "shadow_volume_event.csv", index=False)
    overlay.to_csv(out_dir / "shadow_volume_overlay_log.csv", index=False)
    windows.to_csv(out_dir / "shadow_volume_windows.csv", index=False)
    event_summary = summarize_events(events)
    overlay_summary = summarize_overlay(overlay)
    event_summary.to_csv(out_dir / "shadow_volume_event_summary.csv", index=False)
    overlay_summary.to_csv(out_dir / "shadow_volume_overlay_summary.csv", index=False)
    summary = {
        "event_rows": int(len(events)),
        "overlay_rows": int(len(overlay)),
        "accepted_swaps": int(overlay["swap_accepted"].sum()) if not overlay.empty else 0,
        "output_dir": str(out_dir),
        "event_summary": event_summary.to_dict(orient="records"),
        "overlay_summary": overlay_summary.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Volume expansion event-study and max-1-swap shadow overlay.")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--raw-path", type=Path, default=ROOT / "data" / "train_hs300_latest.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events, overlay, windows = build_shadow_tables(args.source_run, args.raw_path)
    write_outputs(events, overlay, windows, args.out_dir)
    print(f"[volume-shadow] wrote {len(events)} event rows and {len(overlay)} overlay rows to {args.out_dir}")
    if not overlay.empty:
        print(summarize_overlay(overlay).to_string(index=False))


if __name__ == "__main__":
    main()
