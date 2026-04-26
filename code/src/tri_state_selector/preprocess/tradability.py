from __future__ import annotations

import numpy as np
import pandas as pd


def build_tradable_mask(day_frame: pd.DataFrame, prev_weights: pd.Series | None = None) -> pd.DataFrame:
    """Model A-share EOD tradability: up-limit cannot buy, down-limit cannot sell."""

    df = day_frame.copy()
    df["stock_id"] = df["stock_id"].astype(str).str.zfill(6)
    prev = pd.Series(dtype=float) if prev_weights is None else prev_weights.copy()
    prev.index = prev.index.astype(str).str.zfill(6)

    close = pd.to_numeric(df["close"], errors="coerce")
    suspended = df.get("is_suspended", False)
    if not isinstance(suspended, pd.Series):
        suspended = pd.Series(bool(suspended), index=df.index)
    suspended = suspended.fillna(False).astype(bool)

    if "up_limit" in df.columns:
        up_limit = pd.to_numeric(df["up_limit"], errors="coerce")
        at_up_limit = np.isclose(close, up_limit, rtol=0.0, atol=1e-6) | (close >= up_limit - 1e-9)
    else:
        at_up_limit = pd.Series(False, index=df.index)
    if "down_limit" in df.columns:
        down_limit = pd.to_numeric(df["down_limit"], errors="coerce")
        at_down_limit = np.isclose(close, down_limit, rtol=0.0, atol=1e-6) | (close <= down_limit + 1e-9)
    else:
        at_down_limit = pd.Series(False, index=df.index)

    current_pos = df["stock_id"].map(prev).fillna(0.0).to_numpy(dtype=float)
    df["can_buy"] = (~suspended) & (~pd.Series(at_up_limit, index=df.index).astype(bool))
    df["can_sell"] = (~suspended) & (~pd.Series(at_down_limit, index=df.index).astype(bool))
    df["tradable"] = np.where(current_pos > 1e-12, df["can_sell"], df["can_buy"])
    return df
