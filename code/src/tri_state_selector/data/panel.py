from __future__ import annotations

from pathlib import Path

import pandas as pd


ALIASES = {
    "date": ["date", "trade_date", "datetime", "dt", "日期"],
    "stock_id": ["stock_id", "asset", "ticker", "symbol", "code", "股票代码"],
    "open": ["open", "开盘"],
    "high": ["high", "最高"],
    "low": ["low", "最低"],
    "close": ["close", "收盘"],
    "volume": ["volume", "vol", "成交量"],
    "amount": ["amount", "turnover_value", "成交额"],
    "turnover": ["turnover", "换手率"],
    "industry": ["industry", "行业"],
    "is_suspended": ["is_suspended", "suspended", "停牌"],
    "up_limit": ["up_limit", "limit_up", "涨停价"],
    "down_limit": ["down_limit", "limit_down", "跌停价"],
    "listed_days": ["listed_days", "上市天数"],
}


def _find_column(df: pd.DataFrame, canonical: str) -> str | None:
    lower_map = {str(col).lower(): col for col in df.columns}
    for candidate in ALIASES.get(canonical, [canonical]):
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def normalize_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for canonical in ALIASES:
        col = _find_column(df, canonical)
        if col is not None:
            out[canonical] = df[col]
    missing = {"date", "stock_id", "close"} - set(out.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")
    out["date"] = pd.to_datetime(out["date"])
    out["stock_id"] = out["stock_id"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    for col in ["open", "high", "low", "close", "volume", "amount", "turnover", "up_limit", "down_limit"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "open" not in out.columns:
        out["open"] = out["close"]
    if "high" not in out.columns:
        out["high"] = out[["open", "close"]].max(axis=1)
    if "low" not in out.columns:
        out["low"] = out[["open", "close"]].min(axis=1)
    if "amount" not in out.columns:
        out["amount"] = 0.0
    if "volume" not in out.columns:
        out["volume"] = 0.0
    if "turnover" not in out.columns:
        out["turnover"] = 0.0
    if "industry" not in out.columns:
        out["industry"] = "UNKNOWN"
    if "is_suspended" not in out.columns:
        out["is_suspended"] = False
    if "listed_days" not in out.columns:
        out["listed_days"] = (
            out.sort_values(["stock_id", "date"]).groupby("stock_id").cumcount() + 1
        )
    return out.sort_values(["date", "stock_id"]).reset_index(drop=True)


def load_panel_csv(path: str | Path) -> pd.DataFrame:
    return normalize_panel(pd.read_csv(path))
