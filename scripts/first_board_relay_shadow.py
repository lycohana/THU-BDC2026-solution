from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "code" / "src"))

from branch_router_v2c_shadow import SOURCE_RUN, V2B_DETAIL, _bad_counts, _realized_for_ids, build_shadow_inputs  # noqa: E402
from batch_window_analysis import load_raw, normalize_stock_id  # noqa: E402


OUT_DIR = ROOT / "temp" / "branch_router_validation" / "first_board_relay_shadow"
warnings.filterwarnings("ignore", category=FutureWarning)


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _rank_pct(series: pd.Series, ascending: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill = values.median() if values.notna().any() else 0.0
    return values.fillna(fill).rank(method="average", pct=True, ascending=ascending)


def _split_ids(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return normalize_stock_id(pd.Series([x.strip() for x in str(value).split(",") if x.strip()], dtype=str)).astype(str).tolist()


def _stock_return(work: pd.DataFrame, stock_id: str) -> float:
    row = work[work["stock_id"].astype(str) == str(stock_id).zfill(6)]
    if row.empty:
        return 0.0
    return float(pd.to_numeric(row["realized_ret"].iloc[0], errors="coerce"))


def _lowest_score_target(work: pd.DataFrame, ids: list[str]) -> tuple[str, float]:
    if not ids:
        return "", 0.0
    score_col = "grr_final_score" if "grr_final_score" in work.columns else "score"
    selected = work[work["stock_id"].astype(str).isin(set(ids))].copy()
    selected["_score"] = pd.to_numeric(selected.get(score_col, 0.0), errors="coerce").fillna(0.0)
    if selected.empty:
        return "", 0.0
    row = selected.sort_values("_score", ascending=True).iloc[0]
    return str(row["stock_id"]).zfill(6), float(row["_score"])


def _limit_threshold(stock_id: str) -> float:
    code = str(stock_id).zfill(6)
    if code.startswith(("300", "301", "688")):
        return 0.195
    return 0.095


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / (b.replace(0.0, np.nan) + 1e-12)


def build_relay_panel(raw: pd.DataFrame) -> pd.DataFrame:
    panel = raw.copy()
    panel["stock_id"] = normalize_stock_id(panel["股票代码"])
    panel["date"] = pd.to_datetime(panel["日期"])
    for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率", "涨跌幅"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel = panel.sort_values(["stock_id", "date"]).reset_index(drop=True)
    group = panel.groupby("stock_id", sort=False)

    pct = panel["涨跌幅"].fillna(0.0) / 100.0
    panel["ret1_raw"] = pct
    panel["is_up_candle"] = (panel["收盘"] > panel["开盘"]) & (pct > 0)
    panel["limit_threshold"] = panel["stock_id"].map(_limit_threshold).astype(float)
    panel["limit_up"] = pct >= panel["limit_threshold"]
    panel["limit_up_lag1"] = group["limit_up"].shift(1).fillna(False).astype(bool)
    panel["limit_up_count20_lag1"] = group["limit_up"].transform(lambda s: s.shift(1).rolling(20, min_periods=1).sum())

    panel["pct_lag1"] = group["ret1_raw"].shift(1)
    panel["pct_lag2"] = group["ret1_raw"].shift(2)
    panel["up_candle_lag1"] = group["is_up_candle"].shift(1).fillna(False).astype(bool)
    panel["up_candle_lag2"] = group["is_up_candle"].shift(2).fillna(False).astype(bool)
    panel["amount_lag1"] = group["成交额"].shift(1)
    panel["amount_lag2"] = group["成交额"].shift(2)
    panel["amount_ma5_lag1"] = group["成交额"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    panel["amount_ma20"] = group["成交额"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    panel["amount_ma20_lag1"] = group["成交额"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    panel["amount_min20_lag1"] = group["成交额"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).min())
    panel["ret20_raw"] = panel["收盘"] / (group["收盘"].shift(20) + 1e-12) - 1.0
    high5 = group["最高"].transform(lambda s: s.rolling(5, min_periods=3).max())
    low5 = group["最低"].transform(lambda s: s.rolling(5, min_periods=3).min())
    high120 = group["最高"].transform(lambda s: s.rolling(120, min_periods=30).max())
    low120 = group["最低"].transform(lambda s: s.rolling(120, min_periods=30).min())
    prior_high120 = group["最高"].transform(lambda s: s.shift(1).rolling(120, min_periods=30).max())

    panel["amp5_raw"] = high5 / (low5 + 1e-12) - 1.0
    panel["price_position_120d_raw"] = ((panel["收盘"] - low120) / (high120 - low120 + 1e-12)).clip(0.0, 1.0)
    panel["new_high120"] = panel["最高"] >= (prior_high120 * 0.999)
    panel["amount_vs_prev"] = _safe_div(panel["成交额"], panel["amount_lag1"])
    panel["amount_vs_ma20"] = _safe_div(panel["成交额"], panel["amount_ma20_lag1"])
    panel["amount_vs_min20"] = _safe_div(panel["成交额"], panel["amount_min20_lag1"])
    turnover_decimal = panel["换手率"] / 100.0
    panel["mcap_proxy"] = panel["成交额"] / (turnover_decimal.replace(0.0, np.nan) + 1e-12)
    panel["prior_two_soft_up"] = (
        panel["up_candle_lag1"]
        & panel["up_candle_lag2"]
        & (panel["pct_lag1"] > 0.0)
        & (panel["pct_lag2"] > 0.0)
        & (panel["pct_lag1"] < 0.05)
        & (panel["pct_lag2"] < 0.05)
    )
    panel["first_board_prior1"] = panel["limit_up"] & (~panel["limit_up_lag1"])
    panel["first_board_20"] = panel["limit_up"] & (panel["limit_up_count20_lag1"].fillna(0.0) <= 0.0)
    for threshold in [0.06, 0.08]:
        key = f"{int(threshold * 100):02d}"
        strong_lag1 = group["ret1_raw"].shift(1) >= threshold
        strong_count20_lag1 = group["ret1_raw"].transform(
            lambda s, t=threshold: (s.shift(1) >= t).rolling(20, min_periods=1).sum()
        )
        panel[f"strong_up_{key}"] = pct >= threshold
        panel[f"first_strong_{key}_prior1"] = panel[f"strong_up_{key}"] & (~strong_lag1.fillna(False).astype(bool))
        panel[f"first_strong_{key}_20"] = panel[f"strong_up_{key}"] & (strong_count20_lag1.fillna(0.0) <= 0.0)
    panel["volume_double"] = panel["amount_vs_prev"] >= 2.0
    panel["volume_warm"] = panel["amount_vs_prev"] >= 1.5
    panel["no_abnormal_burst"] = ~(
        panel["new_high120"]
        & ((panel["amount_vs_ma20"] > 8.0) | (panel["amount_vs_min20"] > 12.0))
    )
    panel["relay_core_daily"] = (
        panel["first_board_prior1"]
        & panel["prior_two_soft_up"]
        & panel["volume_double"]
        & panel["no_abnormal_burst"]
        & (panel["amp5_raw"] <= 0.20)
        & (panel["ret20_raw"] < 0.25)
    )
    for key in ["06", "08"]:
        panel[f"relay_strong_{key}_daily"] = (
            panel[f"first_strong_{key}_prior1"]
            & panel["prior_two_soft_up"]
            & panel["volume_warm"]
            & panel["no_abnormal_burst"]
            & (panel["amp5_raw"] <= 0.20)
            & (panel["ret20_raw"] < 0.25)
        )

    keep = [
        "stock_id",
        "date",
        "limit_up",
        "first_board_prior1",
        "first_board_20",
        "prior_two_soft_up",
        "volume_double",
        "volume_warm",
        "no_abnormal_burst",
        "relay_core_daily",
        "strong_up_06",
        "strong_up_08",
        "first_strong_06_prior1",
        "first_strong_08_prior1",
        "first_strong_06_20",
        "first_strong_08_20",
        "relay_strong_06_daily",
        "relay_strong_08_daily",
        "ret1_raw",
        "pct_lag1",
        "pct_lag2",
        "ret20_raw",
        "amount_vs_prev",
        "amount_vs_ma20",
        "amount_vs_min20",
        "amp5_raw",
        "price_position_120d_raw",
        "new_high120",
        "mcap_proxy",
        "成交额",
    ]
    return panel[keep].replace([np.inf, -np.inf], np.nan)


def add_relay_scores(work: pd.DataFrame) -> pd.DataFrame:
    out = work.copy()
    out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
    ret5 = _num(out, "ret5")
    ret20 = _num(out, "ret20")
    sigma20 = _num(out, "sigma20")
    amp20 = _num(out, "amp20")
    drawdown20 = _num(out, "max_drawdown20")
    downside_beta60 = _num(out, "downside_beta60")
    liquidity = _num(out, "median_amount20")
    model_score = _num(out, "grr_final_score", default=0.0)
    amount_today = _num(out, "成交额")
    amount_vs_prev = _num(out, "amount_vs_prev")
    amount_vs_ma20 = _num(out, "amount_vs_ma20")
    amp5_raw = _num(out, "amp5_raw", default=1.0)
    price_pos = _num(out, "price_position_120d_raw", default=0.5).clip(0.0, 1.0)
    mcap_proxy = _num(out, "mcap_proxy")
    ret20_raw = _num(out, "ret20_raw")

    out["model_rank"] = _rank_pct(model_score)
    out["liq_rank"] = _rank_pct(liquidity)
    out["amount_today"] = amount_today
    out["amount_vs_prev"] = amount_vs_prev
    out["amount_vs_ma20"] = amount_vs_ma20
    out["amp5_raw"] = amp5_raw
    out["price_position_120d_raw"] = price_pos
    out["mcap_proxy"] = mcap_proxy
    out["ret20_raw"] = ret20_raw
    out["relay_risk_pass"] = (
        (sigma20 < 0.055)
        & (amp20 < 0.13)
        & (drawdown20 > -0.14)
        & (downside_beta60 < 1.50)
        & (_rank_pct(liquidity) >= 0.20)
    )
    out["relay_score"] = (
        0.30 * out["model_rank"]
        + 0.18 * _rank_pct(amount_vs_prev.clip(upper=6.0))
        + 0.14 * _rank_pct(amount_vs_ma20.clip(upper=6.0))
        + 0.16 * (1.0 - price_pos)
        + 0.10 * out["liq_rank"]
        + 0.07 * _rank_pct(ret5.clip(lower=-0.05, upper=0.15))
        - 0.08 * _rank_pct(sigma20)
        - 0.07 * _rank_pct(amp20)
    )
    out["relay_low_position"] = (price_pos < 0.75) & (ret20_raw < 0.25)
    out["relay_amount_5_30b"] = (amount_today >= 5e8) & (amount_today <= 3e9)
    out["relay_amount_relaxed"] = (amount_today >= 5e8) & (amount_today <= 1.5e10)
    out["relay_mcap_30_300b"] = (mcap_proxy >= 3e9) & (mcap_proxy <= 3e10)
    out["relay_mcap_relaxed"] = (mcap_proxy >= 3e9) & (mcap_proxy <= 3e11)
    return out


def _candidate_pool(work: pd.DataFrame, base_ids: list[str], mode: str, rank_cap: int) -> pd.DataFrame:
    mode = mode.replace("_target_dbeta2", "")
    out = add_relay_scores(work)
    out = out[~out["stock_id"].isin(set(base_ids))].copy()
    base_cond = out["relay_core_daily"].fillna(False).astype(bool)
    if mode == "daily_proxy_loose":
        cond = base_cond & out["relay_risk_pass"]
    elif mode == "daily_proxy_raw":
        cond = base_cond
    elif mode == "daily_proxy_lowpos_raw":
        cond = base_cond & out["relay_low_position"]
    elif mode == "daily_proxy_lowpos":
        cond = base_cond & out["relay_risk_pass"] & out["relay_low_position"]
    elif mode == "daily_proxy_model":
        cond = base_cond & out["relay_risk_pass"] & (out["model_rank"] >= 0.70)
    elif mode == "daily_proxy_strict_amount":
        cond = base_cond & out["relay_risk_pass"] & out["relay_low_position"] & out["relay_amount_5_30b"]
    elif mode == "daily_proxy_relaxed_amount":
        cond = base_cond & out["relay_risk_pass"] & out["relay_low_position"] & out["relay_amount_relaxed"]
    elif mode == "daily_proxy_relaxed_mcap":
        cond = base_cond & out["relay_risk_pass"] & out["relay_low_position"] & out["relay_mcap_relaxed"]
    elif mode == "daily_proxy_first20":
        cond = (
            out["first_board_20"].fillna(False).astype(bool)
            & out["prior_two_soft_up"].fillna(False).astype(bool)
            & out["volume_double"].fillna(False).astype(bool)
            & out["no_abnormal_burst"].fillna(False).astype(bool)
            & out["relay_risk_pass"]
            & out["relay_low_position"]
        )
    elif mode == "strong6_raw":
        cond = out["relay_strong_06_daily"].fillna(False).astype(bool)
    elif mode == "strong8_raw":
        cond = out["relay_strong_08_daily"].fillna(False).astype(bool)
    elif mode == "strong6_lowpos":
        cond = (
            out["relay_strong_06_daily"].fillna(False).astype(bool)
            & out["relay_low_position"]
            & out["relay_risk_pass"]
        )
    elif mode == "strong8_lowpos":
        cond = (
            out["relay_strong_08_daily"].fillna(False).astype(bool)
            & out["relay_low_position"]
            & out["relay_risk_pass"]
        )
    elif mode == "strong6_model":
        cond = (
            out["relay_strong_06_daily"].fillna(False).astype(bool)
            & out["relay_risk_pass"]
            & (out["model_rank"] >= 0.70)
        )
    elif mode == "strong8_model":
        cond = (
            out["relay_strong_08_daily"].fillna(False).astype(bool)
            & out["relay_risk_pass"]
            & (out["model_rank"] >= 0.70)
        )
    else:
        raise ValueError(f"unknown mode: {mode}")
    out = out[cond].sort_values("relay_score", ascending=False).copy()
    if out.empty:
        return out
    out["candidate_rank"] = np.arange(1, len(out) + 1)
    return out[out["candidate_rank"] <= int(rank_cap)]


def _eval_insert(
    work: pd.DataFrame,
    window: str,
    base_ids: list[str],
    base_name: str,
    mode: str,
    rank_cap: int,
    gate_reason: str = "",
) -> dict[str, Any]:
    base_return = _realized_for_ids(work, base_ids)
    base_bad, base_very_bad = _bad_counts(work, base_ids)
    target, target_score = _lowest_score_target(work, base_ids)
    row: dict[str, Any] = {
        "window": window,
        "variant": f"{base_name}_{mode}_top{rank_cap}",
        "base_name": base_name,
        "mode": mode,
        "rank_cap": rank_cap,
        "base_top5": ",".join(base_ids),
        "base_return": base_return,
        "shadow_return": base_return,
        "delta_vs_base": 0.0,
        "accepted_swap_count": 0,
        "candidate_stock": "",
        "candidate_rank": None,
        "candidate_score": None,
        "candidate_amount_vs_prev": None,
        "candidate_amount_vs_ma20": None,
        "candidate_price_position": None,
        "candidate_mcap_proxy": None,
        "replaced_stock": target,
        "replaced_score": target_score,
        "raw_candidate_return": None,
        "raw_replaced_return": None,
        "raw_stock_delta": None,
        "weighted_swap_delta": 0.0,
        "blocked_reason": gate_reason,
        "final_top5": ",".join(base_ids),
        "bad_count": base_bad,
        "very_bad_count": base_very_bad,
    }
    if gate_reason:
        return row
    if not base_ids or not target:
        row["blocked_reason"] = "missing_base_or_target"
        return row
    if mode.endswith("_target_dbeta2"):
        target_row = work[work["stock_id"].astype(str) == str(target).zfill(6)]
        target_dbeta = float(pd.to_numeric(target_row.get("downside_beta60", pd.Series([0.0])).iloc[0], errors="coerce")) if not target_row.empty else 0.0
        if not np.isfinite(target_dbeta) or target_dbeta < 2.0:
            row["blocked_reason"] = "target_downside_beta_lt_2"
            return row
    pool = _candidate_pool(work, base_ids, mode, rank_cap)
    if mode == "daily_proxy_model" and not pool.empty:
        pool = pool[pd.to_numeric(pool["grr_final_score"], errors="coerce").fillna(0.0) > float(target_score)]
    if pool.empty:
        row["blocked_reason"] = "no_candidate"
        return row
    cand = pool.iloc[0]
    candidate_stock = str(cand["stock_id"]).zfill(6)
    final_ids = list(base_ids)
    final_ids[final_ids.index(target)] = candidate_stock
    shadow_return = _realized_for_ids(work, final_ids)
    bad, very_bad = _bad_counts(work, final_ids)
    raw_candidate_return = _stock_return(work, candidate_stock)
    raw_replaced_return = _stock_return(work, target)
    raw_delta = raw_candidate_return - raw_replaced_return
    row.update(
        {
            "shadow_return": shadow_return,
            "delta_vs_base": shadow_return - base_return,
            "accepted_swap_count": 1,
            "candidate_stock": candidate_stock,
            "candidate_rank": int(cand["candidate_rank"]),
            "candidate_score": float(cand["relay_score"]),
            "candidate_amount_vs_prev": float(cand["amount_vs_prev"]),
            "candidate_amount_vs_ma20": float(cand["amount_vs_ma20"]),
            "candidate_price_position": float(cand["price_position_120d_raw"]),
            "candidate_mcap_proxy": float(cand["mcap_proxy"]),
            "raw_candidate_return": raw_candidate_return,
            "raw_replaced_return": raw_replaced_return,
            "raw_stock_delta": raw_delta,
            "weighted_swap_delta": 0.2 * raw_delta,
            "blocked_reason": "",
            "final_top5": ",".join(final_ids),
            "bad_count": bad,
            "very_bad_count": very_bad,
        }
    )
    return row


def _metrics(values: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return {"mean": 0.0, "q10": 0.0, "worst": 0.0, "hit_rate": 0.0}
    return {
        "mean": float(s.mean()),
        "q10": float(s.quantile(0.10)),
        "worst": float(s.min()),
        "hit_rate": float((s > 0).mean()),
    }


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    windows = sorted(rows["window"].astype(str).unique())
    buckets = {"20win": windows[-20:], "40win": windows[-40:], "60win": windows}
    out_rows: list[dict[str, Any]] = []
    for bucket, keep in buckets.items():
        scoped = rows[rows["window"].astype(str).isin(set(keep))].copy()
        for variant, sub in scoped.groupby("variant"):
            delta = pd.to_numeric(sub["delta_vs_base"], errors="coerce").fillna(0.0)
            out_rows.append(
                {
                    "bucket": bucket,
                    "variant": variant,
                    "window_count": int(len(sub)),
                    "accepted_swaps": int(pd.to_numeric(sub["accepted_swap_count"], errors="coerce").fillna(0).sum()),
                    "accepted_swap_rate": float(pd.to_numeric(sub["accepted_swap_count"], errors="coerce").fillna(0).mean()) if len(sub) else 0.0,
                    **{f"return_{k}": v for k, v in _metrics(sub["shadow_return"]).items()},
                    **{f"delta_{k}": v for k, v in _metrics(sub["delta_vs_base"]).items()},
                    "negative_delta_count": int((delta < -1e-12).sum()),
                    "very_bad_mean": float(pd.to_numeric(sub["very_bad_count"], errors="coerce").fillna(0).mean()) if len(sub) else 0.0,
                }
            )
    return pd.DataFrame(out_rows)


def event_study(inputs: dict[str, Any], relay_panel: pd.DataFrame) -> pd.DataFrame:
    panel = relay_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    events: list[pd.DataFrame] = []
    for window, work in inputs["work_by_window"].items():
        day_panel = panel[panel["date"] == window].copy()
        if day_panel.empty:
            continue
        merged = work.merge(day_panel, on=["stock_id"], how="left", suffixes=("", "_relay"))
        scored = add_relay_scores(merged)
        scored["window"] = window
        events.append(
            scored[
                [
                    "window",
                    "stock_id",
                    "realized_ret",
                    "relay_core_daily",
                    "first_board_prior1",
                    "first_board_20",
                    "relay_strong_06_daily",
                    "relay_strong_08_daily",
                    "first_strong_06_prior1",
                    "first_strong_08_prior1",
                    "prior_two_soft_up",
                    "volume_double",
                    "volume_warm",
                    "relay_low_position",
                    "relay_risk_pass",
                    "relay_amount_5_30b",
                    "relay_amount_relaxed",
                    "relay_mcap_relaxed",
                    "relay_score",
                ]
            ]
        )
    if not events:
        return pd.DataFrame()
    all_events = pd.concat(events, ignore_index=True)
    masks = {
        "first_board": all_events["first_board_prior1"].fillna(False).astype(bool),
        "core_daily": all_events["relay_core_daily"].fillna(False).astype(bool),
        "core_raw_lowpos": all_events["relay_core_daily"].fillna(False).astype(bool)
        & all_events["relay_low_position"].fillna(False).astype(bool),
        "core_risk": all_events["relay_core_daily"].fillna(False).astype(bool) & all_events["relay_risk_pass"].fillna(False).astype(bool),
        "core_lowpos_risk": all_events["relay_core_daily"].fillna(False).astype(bool)
        & all_events["relay_low_position"].fillna(False).astype(bool)
        & all_events["relay_risk_pass"].fillna(False).astype(bool),
        "core_strict_amount": all_events["relay_core_daily"].fillna(False).astype(bool)
        & all_events["relay_low_position"].fillna(False).astype(bool)
        & all_events["relay_risk_pass"].fillna(False).astype(bool)
        & all_events["relay_amount_5_30b"].fillna(False).astype(bool),
        "core_relaxed_amount": all_events["relay_core_daily"].fillna(False).astype(bool)
        & all_events["relay_low_position"].fillna(False).astype(bool)
        & all_events["relay_risk_pass"].fillna(False).astype(bool)
        & all_events["relay_amount_relaxed"].fillna(False).astype(bool),
        "core_first20": all_events["first_board_20"].fillna(False).astype(bool)
        & all_events["prior_two_soft_up"].fillna(False).astype(bool)
        & all_events["volume_double"].fillna(False).astype(bool)
        & all_events["relay_low_position"].fillna(False).astype(bool)
        & all_events["relay_risk_pass"].fillna(False).astype(bool),
        "strong6_daily": all_events["relay_strong_06_daily"].fillna(False).astype(bool),
        "strong6_lowpos_risk": all_events["relay_strong_06_daily"].fillna(False).astype(bool)
        & all_events["relay_low_position"].fillna(False).astype(bool)
        & all_events["relay_risk_pass"].fillna(False).astype(bool),
        "strong8_daily": all_events["relay_strong_08_daily"].fillna(False).astype(bool),
        "strong8_lowpos_risk": all_events["relay_strong_08_daily"].fillna(False).astype(bool)
        & all_events["relay_low_position"].fillna(False).astype(bool)
        & all_events["relay_risk_pass"].fillna(False).astype(bool),
    }
    windows = sorted(all_events["window"].astype(str).unique())
    buckets = {"20win": windows[-20:], "40win": windows[-40:], "60win": windows}
    rows: list[dict[str, Any]] = []
    for bucket, keep in buckets.items():
        in_bucket = all_events["window"].astype(str).isin(set(keep))
        for name, mask in masks.items():
            sub = all_events[in_bucket & mask].copy()
            rets = pd.to_numeric(sub["realized_ret"], errors="coerce").dropna()
            rows.append(
                {
                    "bucket": bucket,
                    "event": name,
                    "count": int(len(rets)),
                    **{f"future_{k}": v for k, v in _metrics(rets).items()},
                    "positive_count": int((rets > 0).sum()),
                    "negative_count": int((rets < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def run_shadow(source_run: Path, detail_dir: Path, raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inputs = build_shadow_inputs(source_run, raw_path)
    raw = load_raw(raw_path)
    relay_panel = build_relay_panel(raw)
    relay_panel["date"] = pd.to_datetime(relay_panel["date"]).dt.strftime("%Y-%m-%d")

    ablation = pd.read_csv(detail_dir / "ablation_decisions.csv")
    v2b = ablation[ablation["variant"] == "v2b_trend_plus_ai_overlay"].set_index("window_date")

    rows: list[dict[str, Any]] = []
    for window in sorted(inputs["default_top5"]):
        day_panel = relay_panel[relay_panel["date"] == window].copy()
        work = inputs["work_by_window"][window].merge(day_panel, on=["stock_id"], how="left", suffixes=("", "_relay"))
        work = add_relay_scores(work)
        default_ids = inputs["default_top5"][window]
        v2b_ids = _split_ids(v2b.loc[window, "selected_stocks"]) if window in v2b.index else default_ids
        v2b_changed = set(v2b_ids) != set(default_ids)
        for base_name, base_ids, gate in [
            ("default", default_ids, ""),
            ("v2b", v2b_ids, ""),
            ("v2b_no_swap_only", v2b_ids, "v2b_already_swapped" if v2b_changed else ""),
        ]:
            for mode in [
                "daily_proxy_raw",
                "daily_proxy_raw_target_dbeta2",
                "daily_proxy_lowpos_raw",
                "daily_proxy_loose",
                "daily_proxy_lowpos",
                "daily_proxy_model",
                "daily_proxy_strict_amount",
                "daily_proxy_relaxed_amount",
                "daily_proxy_relaxed_mcap",
                "daily_proxy_first20",
                "strong6_raw",
                "strong6_raw_target_dbeta2",
                "strong8_raw",
                "strong8_raw_target_dbeta2",
                "strong6_lowpos",
                "strong8_lowpos",
                "strong6_model",
                "strong8_model",
            ]:
                for rank_cap in [1, 3, 5]:
                    rows.append(_eval_insert(work, window, base_ids, base_name, mode, rank_cap, gate_reason=gate))
    result = pd.DataFrame(rows)
    return result, summarize(result), event_study(inputs, relay_panel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default=str(SOURCE_RUN.relative_to(ROOT)))
    parser.add_argument("--detail-dir", default=str(V2B_DETAIL.relative_to(ROOT)))
    parser.add_argument("--raw-path", default="data/train_hs300_20260424.csv")
    parser.add_argument("--out-dir", default=str(OUT_DIR.relative_to(ROOT)))
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, summary, events = run_shadow(ROOT / args.source_run, ROOT / args.detail_dir, ROOT / args.raw_path)
    rows.to_csv(out_dir / "first_board_relay_windows.csv", index=False)
    summary.to_csv(out_dir / "first_board_relay_summary.csv", index=False)
    events.to_csv(out_dir / "first_board_relay_event_study.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "rows": len(rows),
                "out_dir": str(out_dir),
                "top_60win": summary[summary["bucket"] == "60win"].sort_values("delta_mean", ascending=False).head(10).to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows)}, ensure_ascii=False), flush=True)
    print(summary[summary["bucket"] == "60win"].sort_values("delta_mean", ascending=False).head(18).to_string(index=False))
    if not events.empty:
        print("\n[event study]")
        print(events[events["bucket"] == "60win"].sort_values("future_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
