from __future__ import annotations

import numpy as np
import pandas as pd

from ..preprocess.asof import align_fundamentals_asof, validate_no_lookahead
from ..preprocess.tradability import build_tradable_mask


def robust_zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad <= 1e-12:
        std = x.std(ddof=0)
        denom = std if np.isfinite(std) and std > 1e-12 else 1.0
    else:
        denom = 1.4826 * mad
    return ((x.fillna(med) - med) / denom).clip(-5.0, 5.0)


def _ols_slope_r2(y: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    if y.size < 3 or not np.isfinite(y).all():
        return 0.0, 0.0
    x = np.arange(y.size, dtype=float)
    x = x - x.mean()
    yc = y - y.mean()
    denom = float(np.dot(x, x))
    slope = float(np.dot(x, yc) / (denom + 1e-12))
    fitted = slope * x + y.mean()
    ss_tot = float(np.dot(yc, yc))
    ss_res = float(np.dot(y - fitted, y - fitted))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return slope, float(np.clip(r2, 0.0, 1.0))


class FeatureEngine:
    def __init__(self, min_history: int = 20) -> None:
        self.min_history = min_history

    def compute_stock_features(
        self,
        prices: pd.DataFrame,
        asof: pd.Timestamp,
        fundamentals: pd.DataFrame | None = None,
        prev_weights: pd.Series | None = None,
    ) -> pd.DataFrame:
        asof = pd.Timestamp(asof)
        history = prices[pd.to_datetime(prices["date"]) <= asof].copy()
        if history.empty:
            return pd.DataFrame()
        history = align_fundamentals_asof(history, fundamentals)
        validate_no_lookahead(history)
        history = history.sort_values(["stock_id", "date"])

        market_ret = history.groupby("date")["close"].mean().pct_change(fill_method=None).rename("market_ret")
        history = history.merge(market_ret, left_on="date", right_index=True, how="left")
        history["ret"] = history.groupby("stock_id")["close"].pct_change(fill_method=None)

        latest_rows = []
        for stock_id, group in history.groupby("stock_id", sort=False):
            g = group.tail(260).copy()
            if len(g) < self.min_history:
                continue
            ret = g["ret"].fillna(0.0)
            close = g["close"].astype(float)
            mret = g["market_ret"].fillna(0.0)
            recent = g.iloc[-1].to_dict()

            mom = float(close.iloc[-5] / close.iloc[max(0, len(close) - 60)] - 1.0) if len(close) >= 61 else float(close.iloc[-1] / close.iloc[0] - 1.0)
            rev_5 = -float(close.iloc[-1] / close.iloc[max(0, len(close) - 5)] - 1.0) if len(close) >= 6 else 0.0
            slope, trend_r2 = _ols_slope_r2(np.log(close.tail(60).clip(lower=1e-9).to_numpy()))
            x = mret.tail(60).to_numpy(dtype=float)
            y = ret.tail(60).to_numpy(dtype=float)
            beta = float(np.cov(y, x, ddof=0)[0, 1] / (np.var(x) + 1e-12)) if len(y) >= 20 else 1.0
            residual = y - beta * x
            ivol = float(np.std(residual, ddof=0) * np.sqrt(252.0))
            vol20 = float(ret.tail(20).std(ddof=0) * np.sqrt(252.0))
            tail = ret.tail(60).dropna()
            es5 = float(abs(tail[tail <= tail.quantile(0.05)].mean())) if len(tail) >= 20 else 0.0
            down_limit_cnt = int((g.tail(60).get("down_limit", pd.Series(np.nan, index=g.tail(60).index)).sub(g.tail(60)["close"]).abs() <= 1e-6).sum())
            up_limit_cnt = int((g.tail(20).get("up_limit", pd.Series(np.nan, index=g.tail(20).index)).sub(g.tail(20)["close"]).abs() <= 1e-6).sum())
            blowoff = mom + float(g["turnover"].tail(5).mean() / (g["turnover"].tail(60).mean() + 1e-12)) + up_limit_cnt

            latest_rows.append(
                {
                    "stock_id": stock_id,
                    "industry": recent.get("industry", "UNKNOWN"),
                    "listed_days": recent.get("listed_days", len(g)),
                    "close": recent.get("close"),
                    "RESMOM_60_5": mom - beta * float(mret.tail(60).sum()),
                    "TREND_R2_60": trend_r2,
                    "REV_5": rev_5,
                    "VOL_20": vol20,
                    "LOW_VOL": -vol20,
                    "BETA_60": beta,
                    "LOW_BETA": -beta,
                    "IVOL_60": ivol,
                    "ADV20": float(g["amount"].tail(20).mean()),
                    "ILLIQ20": float((ret.tail(20).abs() / (g["amount"].tail(20).abs() + 1e-12)).mean()),
                    "BLOWOFF": blowoff,
                    "TAIL_RISK": es5 + 0.02 * down_limit_cnt,
                    "SHORT_RESILIENCE": -float(abs(ret.tail(5).clip(upper=0).sum())),
                    "GP_A": recent.get("GP_A", recent.get("gross_profit_to_assets", 0.0)),
                    "CFO_A": recent.get("CFO_A", recent.get("cfo_to_assets", 0.0)),
                    "ACCRUAL": recent.get("ACCRUAL", recent.get("accrual", 0.0)),
                    "BP": recent.get("BP", recent.get("book_to_price", 0.0)),
                    "EP": recent.get("EP", recent.get("earnings_to_price", 0.0)),
                    "LEV": recent.get("LEV", recent.get("leverage", 0.0)),
                }
            )

        feats = pd.DataFrame(latest_rows).set_index("stock_id")
        if feats.empty:
            return feats
        quality_raw = feats["GP_A"].fillna(0.0) + feats["CFO_A"].fillna(0.0) - feats["ACCRUAL"].fillna(0.0)
        feats["QUALITY"] = quality_raw
        z_cols = [
            "RESMOM_60_5",
            "TREND_R2_60",
            "REV_5",
            "VOL_20",
            "LOW_VOL",
            "BETA_60",
            "LOW_BETA",
            "IVOL_60",
            "ADV20",
            "ILLIQ20",
            "BLOWOFF",
            "TAIL_RISK",
            "SHORT_RESILIENCE",
            "GP_A",
            "CFO_A",
            "ACCRUAL",
            "BP",
            "EP",
            "LEV",
            "QUALITY",
        ]
        for col in z_cols:
            feats[col] = robust_zscore(feats[col])

        day = history[history["date"] == history["date"].max()].copy()
        tradable = build_tradable_mask(day, prev_weights).set_index("stock_id")
        feats = feats.join(tradable[["can_buy", "can_sell", "tradable"]], how="left")
        for col in ["can_buy", "can_sell", "tradable"]:
            feats[col] = pd.Series(np.where(feats[col].isna(), False, feats[col]), index=feats.index).astype(bool)
        return feats

    def compute_market_features(self, prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
        hist = prices[pd.to_datetime(prices["date"]) <= pd.Timestamp(asof)].copy()
        hist = hist.sort_values(["date", "stock_id"])
        hist["ret"] = hist.groupby("stock_id")["close"].pct_change(fill_method=None)
        close_wide = hist.pivot(index="date", columns="stock_id", values="close").sort_index()
        ret_wide = close_wide.pct_change(fill_method=None)
        bench = close_wide.mean(axis=1).dropna()
        bench_ret = bench.pct_change(fill_method=None).fillna(0.0)
        ma20 = close_wide.rolling(20, min_periods=5).mean()
        breadth_price = (close_wide > ma20).mean(axis=1)
        mom60 = close_wide / close_wide.shift(60) - 1.0
        breadth_mom = (mom60 > 0.0).mean(axis=1)
        breadth = 0.6 * breadth_price + 0.4 * breadth_mom.fillna(breadth_price)
        slope, _ = _ols_slope_r2(np.log(bench.tail(60).clip(lower=1e-9).to_numpy()))
        rv20 = bench_ret.rolling(20, min_periods=5).std() * np.sqrt(252.0)
        corr20 = ret_wide.tail(20).corr().replace([np.inf, -np.inf], np.nan)
        avg_corr = float(corr20.where(~np.eye(len(corr20), dtype=bool)).stack().mean()) if len(corr20) > 1 else 0.0
        if "down_limit" in hist.columns:
            down_limit_ratio = (
                pd.Series(np.isclose(hist["close"], hist["down_limit"], rtol=0.0, atol=1e-6), index=hist.index)
                .groupby(hist["date"])
                .mean()
                .tail(5)
                .mean()
            )
        else:
            down_limit_ratio = 0.0
        latest = pd.Series(
            {
                "Breadth": float(breadth.dropna().iloc[-1]) if breadth.notna().any() else 0.5,
                "Trend": float(0.5 * bench_ret.tail(20).sum() + 0.3 * bench_ret.tail(60).sum() + 0.2 * slope),
                "Stress": float(
                    0.4 * (rv20.iloc[-1] / (rv20.tail(252).median() + 1e-12) if rv20.notna().any() else 1.0)
                    + 0.3 * avg_corr
                    + 0.2 * down_limit_ratio
                    + 0.1 * abs((bench.tail(60) / bench.tail(60).cummax() - 1.0).min())
                ),
            },
            name=pd.Timestamp(asof),
        )
        return latest

    def compute_market_feature_history(self, prices: pd.DataFrame, asof: pd.Timestamp, lookback: int = 300) -> pd.DataFrame:
        dates = pd.Index(sorted(pd.to_datetime(prices.loc[pd.to_datetime(prices["date"]) <= pd.Timestamp(asof), "date"]).unique()))
        dates = dates[-lookback:]
        rows = [self.compute_market_features(prices, d) for d in dates]
        return pd.DataFrame(rows)
