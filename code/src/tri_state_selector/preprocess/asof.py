from __future__ import annotations

import pandas as pd


def align_fundamentals_asof(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    *,
    date_col: str = "date",
    stock_col: str = "stock_id",
    announce_col: str = "announce_date",
) -> pd.DataFrame:
    """Attach only fundamental rows whose announcement date is <= price date."""

    if fundamentals is None or fundamentals.empty:
        return prices.copy()
    if announce_col not in fundamentals.columns:
        raise ValueError("fundamentals must include announce_date for as-of alignment")

    left = prices.copy()
    right = fundamentals.copy()
    left[date_col] = pd.to_datetime(left[date_col])
    right[announce_col] = pd.to_datetime(right[announce_col])
    left[stock_col] = left[stock_col].astype(str).str.zfill(6)
    right[stock_col] = right[stock_col].astype(str).str.zfill(6)

    frames = []
    for stock_id, stock_prices in left.sort_values(date_col).groupby(stock_col, sort=False):
        stock_fund = right[right[stock_col] == stock_id].sort_values(announce_col)
        if stock_fund.empty:
            frames.append(stock_prices)
            continue
        merged = pd.merge_asof(
            stock_prices.sort_values(date_col),
            stock_fund,
            left_on=date_col,
            right_on=announce_col,
            direction="backward",
            suffixes=("", "_fund"),
        )
        if f"{stock_col}_fund" in merged.columns:
            merged = merged.drop(columns=[f"{stock_col}_fund"])
        frames.append(merged)
    return pd.concat(frames, ignore_index=True).sort_values([date_col, stock_col])


def validate_no_lookahead(aligned: pd.DataFrame, *, date_col: str = "date", announce_col: str = "announce_date") -> None:
    if announce_col in aligned.columns:
        bad = aligned[aligned[announce_col].notna() & (pd.to_datetime(aligned[announce_col]) > pd.to_datetime(aligned[date_col]))]
        if not bad.empty:
            raise AssertionError(f"found {len(bad)} rows with future fundamentals")
