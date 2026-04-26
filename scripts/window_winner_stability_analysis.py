"""Window-level winner stability analysis for THU-BDC2026.

This is an offline research script.  It uses only train.csv, builds
scorer-equivalent labels for every anchor window, and checks whether "winner"
patterns are stable across time, regimes, simple rules, and walk-forward
validation.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"
OUT_DIR = ROOT / "temp" / "winner_stability"


FEATURE_COLS = [
    "ret1",
    "ret3",
    "ret5",
    "ret10",
    "ret20",
    "ret60",
    "intraday_ret",
    "gap1",
    "vol5",
    "vol10",
    "vol20",
    "vol60",
    "vol_ratio5_20",
    "amp5",
    "amp20",
    "amp_ratio5_20",
    "amount20",
    "log_amount20",
    "amount_ratio5_20",
    "turnover20",
    "turnover_ratio5_20",
    "pos20",
    "drawdown20",
    "dist_low20",
    "rel_ret5_mkt",
    "rel_ret20_mkt",
    "mkt_ret1",
    "mkt_breadth",
    "mkt_sigma",
    "mkt_ret5",
    "mkt_ret20",
]


RANK_FEATURES = [f"{col}_rank" for col in FEATURE_COLS]
MODEL_FEATURES = [
    "ret1_rank",
    "ret3_rank",
    "ret5_rank",
    "ret10_rank",
    "ret20_rank",
    "ret60_rank",
    "intraday_ret_rank",
    "gap1_rank",
    "vol5_rank",
    "vol10_rank",
    "vol20_rank",
    "vol60_rank",
    "vol_ratio5_20_rank",
    "amp5_rank",
    "amp20_rank",
    "amp_ratio5_20_rank",
    "log_amount20_rank",
    "amount_ratio5_20_rank",
    "turnover20_rank",
    "turnover_ratio5_20_rank",
    "pos20_rank",
    "drawdown20_rank",
    "dist_low20_rank",
    "rel_ret5_mkt_rank",
    "rel_ret20_mkt_rank",
    "mkt_ret1_rank",
    "mkt_breadth_rank",
    "mkt_sigma_rank",
    "mkt_ret5_rank",
    "mkt_ret20_rank",
]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    weights: dict[str, float]
    filter_name: str = "none"
    penalty: dict[str, float] | None = None


def normalize_stock_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std()
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / (std + 1e-12)


def add_labels(raw: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    dates = pd.Index(sorted(raw["date"].dropna().unique()))
    date_to_idx = {date: idx for idx, date in enumerate(dates)}
    out = raw.copy()
    out["_date_idx"] = out["date"].map(date_to_idx).astype("int64")

    open_base = out[["stock_id", "_date_idx", "open"]].copy()
    open_t1 = open_base.rename(columns={"open": "open_t1"})
    open_t1["_date_idx"] -= 1
    open_tn = open_base.rename(columns={"open": f"open_t{horizon}"})
    open_tn["_date_idx"] -= horizon

    out = out.merge(open_t1, on=["stock_id", "_date_idx"], how="left")
    out = out.merge(open_tn, on=["stock_id", "_date_idx"], how="left")
    out["fwd_o2o5"] = out[f"open_t{horizon}"] / (out["open_t1"] + 1e-12) - 1.0
    return out.drop(columns=["open_t1", f"open_t{horizon}"])


def load_panel() -> pd.DataFrame:
    raw = pd.read_csv(DATA_PATH, dtype={"股票代码": str})
    raw = raw.rename(
        columns={
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "换手率": "turnover",
            "涨跌幅": "pct_change",
        }
    )
    raw["stock_id"] = normalize_stock_id(raw["股票代码"])
    raw["date"] = pd.to_datetime(raw["日期"])
    raw = raw.sort_values(["stock_id", "date"]).reset_index(drop=True)
    for col in ["open", "close", "high", "low", "volume", "amount", "amplitude", "turnover", "pct_change"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").astype(float)

    df = add_labels(raw)
    grouped = df.groupby("stock_id", sort=False)
    df["prev_close"] = grouped["close"].shift(1)
    df["ret1"] = df["close"] / (df["prev_close"] + 1e-12) - 1.0
    df["intraday_ret"] = df["close"] / (df["open"] + 1e-12) - 1.0
    df["gap1"] = df["open"] / (df["prev_close"] + 1e-12) - 1.0

    for window in [3, 5, 10, 20, 60]:
        df[f"ret{window}"] = grouped["close"].pct_change(window, fill_method=None)

    for window in [5, 10, 20, 60]:
        min_periods = max(3, window // 2)
        df[f"vol{window}"] = grouped["ret1"].transform(
            lambda s, w=window, m=min_periods: s.rolling(w, min_periods=m).std()
        )
        df[f"amp{window}"] = grouped["amplitude"].transform(
            lambda s, w=window, m=min_periods: s.rolling(w, min_periods=m).mean()
        )
        df[f"amount{window}"] = grouped["amount"].transform(
            lambda s, w=window, m=min_periods: s.rolling(w, min_periods=m).mean()
        )
        df[f"turnover{window}"] = grouped["turnover"].transform(
            lambda s, w=window, m=min_periods: s.rolling(w, min_periods=m).mean()
        )

    df["amount_ratio5_20"] = df["amount5"] / (df["amount20"] + 1e-12)
    df["turnover_ratio5_20"] = df["turnover5"] / (df["turnover20"] + 1e-12)
    df["vol_ratio5_20"] = df["vol5"] / (df["vol20"] + 1e-12)
    df["amp_ratio5_20"] = df["amp5"] / (df["amp20"] + 1e-12)
    low20 = grouped["low"].transform(lambda s: s.rolling(20, min_periods=10).min())
    high20 = grouped["high"].transform(lambda s: s.rolling(20, min_periods=10).max())
    df["pos20"] = (df["close"] - low20) / (high20 - low20 + 1e-12)
    df["drawdown20"] = df["close"] / (high20 + 1e-12) - 1.0
    df["dist_low20"] = df["close"] / (low20 + 1e-12) - 1.0
    df["log_amount20"] = np.log1p(df["amount20"].clip(lower=0.0))

    market = df.groupby("date").agg(
        mkt_ret1=("ret1", "mean"),
        mkt_breadth=("ret1", lambda x: np.nanmean(x > 0.0)),
        mkt_sigma=("ret1", "std"),
        mkt_ret5=("ret5", "mean"),
        mkt_ret20=("ret20", "mean"),
    )
    df = df.merge(market.reset_index(), on="date", how="left")
    df["rel_ret5_mkt"] = df["ret5"] - df["mkt_ret5"]
    df["rel_ret20_mkt"] = df["ret20"] - df["mkt_ret20"]

    valid = df.dropna(subset=["fwd_o2o5", "ret60", "vol20", "amount20", "pos20"]).copy()
    valid["daily_rank"] = valid.groupby("date")["fwd_o2o5"].rank(method="first", ascending=False)
    valid["daily_rank_asc"] = valid.groupby("date")["fwd_o2o5"].rank(method="first", ascending=True)
    valid["is_top5"] = valid["daily_rank"] <= 5
    valid["is_top20"] = valid["daily_rank"] <= 20
    valid["is_bottom5"] = valid["daily_rank_asc"] <= 5

    for col in FEATURE_COLS:
        valid[f"{col}_rank"] = valid.groupby("date")[col].rank(pct=True, method="average")

    # Market feature ranks should be date-level constants.  Rank them across dates
    # and broadcast back to rows so logistic coefficients remain meaningful.
    date_market = valid[["date", "mkt_ret1", "mkt_breadth", "mkt_sigma", "mkt_ret5", "mkt_ret20"]].drop_duplicates()
    for col in ["mkt_ret1", "mkt_breadth", "mkt_sigma", "mkt_ret5", "mkt_ret20"]:
        date_market[f"{col}_rank"] = date_market[col].rank(pct=True, method="average")
        valid = valid.drop(columns=[f"{col}_rank"]).merge(date_market[["date", f"{col}_rank"]], on="date", how="left")

    valid["month"] = valid["date"].dt.to_period("M").astype(str)
    return valid.reset_index(drop=True)


def score_selected(day: pd.DataFrame, selected: pd.DataFrame, score_col: str = "fwd_o2o5") -> dict[str, float]:
    if selected.empty:
        return {
            "selected_sum": 0.0,
            "oracle_sum": 0.0,
            "random_sum": 0.0,
            "final_score": 0.0,
            "top5_hits": 0.0,
            "bottom5_hits": 0.0,
        }
    selected_sum = float(selected[score_col].sum())
    oracle_sum = float(day.nlargest(5, score_col)[score_col].sum())
    random_sum = float(5.0 * day[score_col].mean())
    denom = oracle_sum - random_sum
    final_score = (selected_sum - random_sum) / (denom + 1e-12) if abs(denom) > 1e-9 else 0.0
    top_ids = set(day.nsmallest(5, "daily_rank")["stock_id"])
    bottom_ids = set(day.nsmallest(5, "daily_rank_asc")["stock_id"])
    selected_ids = set(selected["stock_id"])
    return {
        "selected_sum": selected_sum,
        "oracle_sum": oracle_sum,
        "random_sum": random_sum,
        "final_score": final_score,
        "top5_hits": float(len(selected_ids & top_ids)),
        "bottom5_hits": float(len(selected_ids & bottom_ids)),
    }


def factor_stability(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in FEATURE_COLS:
        rcol = f"{col}_rank"
        # Daily rank correlation is equivalent to Spearman because rcol is already a rank.
        daily_ic = []
        for _, day in panel.groupby("date", sort=True):
            if day[rcol].nunique(dropna=True) <= 1:
                continue
            ic = day[rcol].corr(day["fwd_o2o5"], method="spearman")
            if np.isfinite(ic):
                daily_ic.append(float(ic))
        daily_ic = np.asarray(daily_ic, dtype=float)
        monthly = panel.groupby("month").apply(
            lambda g: g[rcol].corr(g["fwd_o2o5"], method="spearman"), include_groups=False
        )
        monthly = monthly.replace([np.inf, -np.inf], np.nan).dropna()
        top5 = panel.loc[panel["is_top5"], rcol]
        rest = panel.loc[~panel["is_top5"], rcol]
        bottom5 = panel.loc[panel["is_bottom5"], rcol]
        high = panel.loc[panel[rcol] >= 0.9]
        low = panel.loc[panel[rcol] <= 0.1]
        base_top = float(panel["is_top5"].mean())
        base_bottom = float(panel["is_bottom5"].mean())
        rows.append(
            {
                "feature": col,
                "mean_daily_ic": float(daily_ic.mean()) if daily_ic.size else np.nan,
                "daily_ic_t": float(daily_ic.mean() / (daily_ic.std(ddof=1) / math.sqrt(daily_ic.size)))
                if daily_ic.size > 1 and daily_ic.std(ddof=1) > 1e-12
                else np.nan,
                "daily_ic_pos_frac": float(np.mean(daily_ic > 0.0)) if daily_ic.size else np.nan,
                "monthly_ic_pos_frac": float(np.mean(monthly > 0.0)) if len(monthly) else np.nan,
                "winner_mean_rank": float(top5.mean()),
                "rest_mean_rank": float(rest.mean()),
                "bottom5_mean_rank": float(bottom5.mean()),
                "winner_minus_rest": float(top5.mean() - rest.mean()),
                "winner_minus_bottom5": float(top5.mean() - bottom5.mean()),
                "high_decile_top5_lift": float(high["is_top5"].mean() / base_top),
                "high_decile_bottom5_lift": float(high["is_bottom5"].mean() / base_bottom),
                "low_decile_top5_lift": float(low["is_top5"].mean() / base_top),
                "low_decile_bottom5_lift": float(low["is_bottom5"].mean() / base_bottom),
                "high_decile_mean_return": float(high["fwd_o2o5"].mean()),
                "low_decile_mean_return": float(low["fwd_o2o5"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["winner_minus_rest", "mean_daily_ic"], ascending=[False, False])


def add_regimes(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    date_stats = out[["date", "mkt_ret1", "mkt_breadth", "mkt_sigma", "mkt_ret20"]].drop_duplicates().copy()
    q = date_stats[["mkt_ret1", "mkt_breadth", "mkt_sigma", "mkt_ret20"]].quantile([0.25, 0.75])

    def classify(row: pd.Series) -> str:
        if row["mkt_ret20"] >= q.loc[0.75, "mkt_ret20"] and row["mkt_breadth"] >= q.loc[0.50 if 0.50 in q.index else 0.25, "mkt_breadth"]:
            return "strong_trend"
        if row["mkt_ret20"] <= q.loc[0.25, "mkt_ret20"] and row["mkt_breadth"] <= q.loc[0.25, "mkt_breadth"]:
            return "weak_market"
        if row["mkt_ret1"] <= q.loc[0.25, "mkt_ret1"] and row["mkt_sigma"] >= q.loc[0.75, "mkt_sigma"]:
            return "selloff_high_dispersion"
        if row["mkt_sigma"] >= q.loc[0.75, "mkt_sigma"]:
            return "high_dispersion"
        if row["mkt_breadth"] <= q.loc[0.25, "mkt_breadth"]:
            return "low_breadth"
        return "normal"

    # Add median row manually for breadth threshold in classify.
    q.loc[0.50] = date_stats[["mkt_ret1", "mkt_breadth", "mkt_sigma", "mkt_ret20"]].median()
    date_stats["regime"] = date_stats.apply(classify, axis=1)
    return out.merge(date_stats[["date", "regime"]], on="date", how="left")


def regime_profiles(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, sub in panel.groupby("regime", sort=True):
        for col in ["amp20", "vol20", "turnover20", "dist_low20", "ret20", "ret60", "log_amount20", "pos20"]:
            rcol = f"{col}_rank"
            winners = sub.loc[sub["is_top5"], rcol]
            rest = sub.loc[~sub["is_top5"], rcol]
            rows.append(
                {
                    "regime": regime,
                    "dates": int(sub["date"].nunique()),
                    "feature": col,
                    "winner_mean_rank": float(winners.mean()),
                    "rest_mean_rank": float(rest.mean()),
                    "winner_minus_rest": float(winners.mean() - rest.mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["regime", "winner_minus_rest"], ascending=[True, False])


def condition_masks(panel: pd.DataFrame) -> dict[str, pd.Series]:
    p = panel
    primitives = {
        "amp20_hi70": p["amp20_rank"] >= 0.70,
        "amp20_hi80": p["amp20_rank"] >= 0.80,
        "vol20_hi70": p["vol20_rank"] >= 0.70,
        "vol20_hi80": p["vol20_rank"] >= 0.80,
        "turnover20_hi70": p["turnover20_rank"] >= 0.70,
        "turnover20_hi80": p["turnover20_rank"] >= 0.80,
        "ret20_hi70": p["ret20_rank"] >= 0.70,
        "ret20_hi80": p["ret20_rank"] >= 0.80,
        "ret60_hi70": p["ret60_rank"] >= 0.70,
        "ret60_hi80": p["ret60_rank"] >= 0.80,
        "liquidity_hi70": p["log_amount20_rank"] >= 0.70,
        "pos20_hi70": p["pos20_rank"] >= 0.70,
        "distlow20_hi70": p["dist_low20_rank"] >= 0.70,
        "amount_accel_hi70": p["amount_ratio5_20_rank"] >= 0.70,
        "amount_accel_hi80": p["amount_ratio5_20_rank"] >= 0.80,
        "vol20_mid": p["vol20_rank"].between(0.45, 0.85),
        "amp20_mid": p["amp20_rank"].between(0.45, 0.85),
        "not_top_vol": p["vol20_rank"] <= 0.90,
        "not_top_amp": p["amp20_rank"] <= 0.90,
        "not_deep_drawdown": p["pos20_rank"] >= 0.35,
        "pullback_not_broken": p["pos20_rank"].between(0.35, 0.85),
    }
    masks = dict(primitives)

    # Curated interaction patterns.
    curated = {
        "elastic_trend": ["amp20_hi70", "vol20_hi70", "ret20_hi70"],
        "elastic_turnover_trend": ["amp20_hi70", "vol20_hi70", "turnover20_hi70", "ret20_hi70"],
        "liquid_elastic_trend": ["amp20_hi70", "vol20_hi70", "ret20_hi70", "liquidity_hi70"],
        "elastic_accumulation": ["amp20_hi70", "vol20_hi70", "amount_accel_hi70", "ret20_hi70"],
        "moderate_elastic_trend": ["amp20_mid", "vol20_mid", "ret20_hi70", "turnover20_hi70"],
        "trend_near_high": ["ret20_hi70", "ret60_hi70", "pos20_hi70"],
        "liquid_trend_near_high": ["liquidity_hi70", "ret20_hi70", "pos20_hi70"],
        "high_turnover_breakout": ["turnover20_hi80", "distlow20_hi70", "ret20_hi70"],
        "elastic_but_not_extreme": ["amp20_hi70", "vol20_hi70", "not_top_vol", "not_top_amp", "ret20_hi70"],
        "pullback_elastic_trend": ["amp20_hi70", "vol20_hi70", "pullback_not_broken", "ret60_hi70"],
    }
    for name, parts in curated.items():
        mask = pd.Series(True, index=p.index)
        for part in parts:
            mask &= primitives[part]
        masks[name] = mask

    # Pair scan among a conservative subset.  This gives breadth without going
    # into a huge, brittle combinatorial search.
    pair_keys = [
        "amp20_hi70",
        "vol20_hi70",
        "turnover20_hi70",
        "ret20_hi70",
        "ret60_hi70",
        "liquidity_hi70",
        "pos20_hi70",
        "amount_accel_hi70",
        "not_top_vol",
        "not_top_amp",
    ]
    for left, right in itertools.combinations(pair_keys, 2):
        masks[f"{left}&{right}"] = primitives[left] & primitives[right]
    return masks


def split_dates(dates: list[pd.Timestamp], n_folds: int = 6) -> list[list[pd.Timestamp]]:
    return [list(chunk) for chunk in np.array_split(np.asarray(dates, dtype="datetime64[ns]"), n_folds)]


def condition_stability(panel: pd.DataFrame) -> pd.DataFrame:
    masks = condition_masks(panel)
    dates = sorted(panel["date"].unique())
    folds = split_dates(dates, n_folds=6)
    base_top = float(panel["is_top5"].mean())
    base_bottom = float(panel["is_bottom5"].mean())
    rows = []
    for name, mask in masks.items():
        sub = panel[mask]
        if len(sub) < 1000:
            continue
        fold_lifts = []
        fold_tail_lifts = []
        fold_returns = []
        fold_coverage = []
        for fold_dates in folds:
            fold = panel[panel["date"].isin(fold_dates)]
            fold_mask = mask.loc[fold.index]
            fold_sub = fold[fold_mask]
            if len(fold_sub) < 50:
                continue
            fold_base_top = float(fold["is_top5"].mean())
            fold_base_bottom = float(fold["is_bottom5"].mean())
            fold_lifts.append(float(fold_sub["is_top5"].mean() / (fold_base_top + 1e-12)))
            fold_tail_lifts.append(float(fold_sub["is_bottom5"].mean() / (fold_base_bottom + 1e-12)))
            fold_returns.append(float(fold_sub["fwd_o2o5"].mean()))
            fold_coverage.append(float(len(fold_sub) / len(fold)))
        if not fold_lifts:
            continue
        top_lift = float(sub["is_top5"].mean() / base_top)
        bottom_lift = float(sub["is_bottom5"].mean() / base_bottom)
        rows.append(
            {
                "condition": name,
                "count": int(len(sub)),
                "coverage": float(len(sub) / len(panel)),
                "top5_lift": top_lift,
                "bottom5_lift": bottom_lift,
                "edge_minus_tail": top_lift - bottom_lift,
                "mean_return": float(sub["fwd_o2o5"].mean()),
                "median_return": float(sub["fwd_o2o5"].median()),
                "fold_top5_lift_mean": float(np.mean(fold_lifts)),
                "fold_top5_lift_min": float(np.min(fold_lifts)),
                "fold_top5_lift_pos_frac": float(np.mean(np.asarray(fold_lifts) > 1.0)),
                "fold_bottom5_lift_mean": float(np.mean(fold_tail_lifts)),
                "fold_mean_return_min": float(np.min(fold_returns)),
                "fold_coverage_min": float(np.min(fold_coverage)),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["fold_top5_lift_pos_frac", "fold_top5_lift_min", "edge_minus_tail", "top5_lift"],
        ascending=[False, False, False, False],
    )


def apply_strategy_score(day: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    out = day.copy()
    score = pd.Series(0.0, index=out.index)
    for col, weight in spec.weights.items():
        score += float(weight) * pd.to_numeric(out[col], errors="coerce").fillna(0.5)
    for col, weight in (spec.penalty or {}).items():
        score -= float(weight) * pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["_strategy_score"] = score

    if spec.filter_name == "liq10":
        out = out[out["log_amount20_rank"] >= 0.10]
    elif spec.filter_name == "stable":
        out = out[(out["log_amount20_rank"] >= 0.10) & (out["vol20_rank"] <= 0.85)]
    elif spec.filter_name == "stable_amp":
        out = out[(out["log_amount20_rank"] >= 0.10) & (out["vol20_rank"] <= 0.85) & (out["amp20_rank"] <= 0.85)]
    elif spec.filter_name == "elastic_pool":
        out = out[(out["amp20_rank"] >= 0.55) & (out["vol20_rank"] >= 0.55) & (out["log_amount20_rank"] >= 0.10)]
    elif spec.filter_name == "elastic_not_extreme":
        out = out[
            (out["amp20_rank"] >= 0.55)
            & (out["vol20_rank"] >= 0.55)
            & (out["log_amount20_rank"] >= 0.10)
            & (out["amp20_rank"] <= 0.95)
            & (out["vol20_rank"] <= 0.95)
        ]
    if len(out) < 5:
        out = day.copy()
        out["_strategy_score"] = score
    return out.sort_values("_strategy_score", ascending=False)


def evaluate_strategies(panel: pd.DataFrame) -> pd.DataFrame:
    strategies = [
        StrategySpec(
            "elastic_signature",
            {
                "amp20_rank": 0.22,
                "vol20_rank": 0.22,
                "turnover20_rank": 0.14,
                "dist_low20_rank": 0.13,
                "ret60_rank": 0.11,
                "ret20_rank": 0.10,
                "log_amount20_rank": 0.08,
            },
            filter_name="none",
        ),
        StrategySpec(
            "elastic_liq10",
            {
                "amp20_rank": 0.22,
                "vol20_rank": 0.22,
                "turnover20_rank": 0.14,
                "dist_low20_rank": 0.13,
                "ret60_rank": 0.11,
                "ret20_rank": 0.10,
                "log_amount20_rank": 0.08,
            },
            filter_name="liq10",
        ),
        StrategySpec(
            "elastic_not_extreme",
            {
                "amp20_rank": 0.24,
                "vol20_rank": 0.20,
                "turnover20_rank": 0.16,
                "dist_low20_rank": 0.12,
                "ret60_rank": 0.10,
                "ret20_rank": 0.10,
                "log_amount20_rank": 0.08,
            },
            filter_name="elastic_not_extreme",
        ),
        StrategySpec(
            "trend_position",
            {
                "ret20_rank": 0.25,
                "ret60_rank": 0.20,
                "pos20_rank": 0.20,
                "dist_low20_rank": 0.15,
                "log_amount20_rank": 0.10,
                "turnover20_rank": 0.10,
            },
            filter_name="liq10",
        ),
        StrategySpec(
            "elastic_accumulation",
            {
                "amp20_rank": 0.18,
                "vol20_rank": 0.16,
                "amount_ratio5_20_rank": 0.22,
                "turnover_ratio5_20_rank": 0.12,
                "ret20_rank": 0.16,
                "dist_low20_rank": 0.10,
                "log_amount20_rank": 0.06,
            },
            filter_name="liq10",
        ),
        StrategySpec(
            "conservative_stable",
            {
                "ret20_rank": 0.20,
                "ret60_rank": 0.18,
                "pos20_rank": 0.18,
                "log_amount20_rank": 0.18,
                "turnover20_rank": 0.12,
                "dist_low20_rank": 0.14,
            },
            filter_name="stable",
        ),
        StrategySpec(
            "current_like_stable_amp",
            {
                "ret20_rank": 0.22,
                "ret60_rank": 0.18,
                "pos20_rank": 0.18,
                "log_amount20_rank": 0.18,
                "turnover20_rank": 0.12,
                "dist_low20_rank": 0.12,
            },
            filter_name="stable_amp",
        ),
    ]
    rows = []
    dates = sorted(panel["date"].unique())
    folds = split_dates(dates, n_folds=6)
    for spec in strategies:
        daily_rows = []
        for date, day in panel.groupby("date", sort=True):
            ranked = apply_strategy_score(day, spec)
            selected = ranked.head(5)
            record = score_selected(day, selected)
            record["date"] = date
            daily_rows.append(record)
        daily = pd.DataFrame(daily_rows)
        fold_scores = []
        fold_top_hits = []
        fold_bottom_hits = []
        for fold_dates in folds:
            sub = daily[daily["date"].isin(fold_dates)]
            fold_scores.append(float(sub["final_score"].mean()))
            fold_top_hits.append(float(sub["top5_hits"].mean()))
            fold_bottom_hits.append(float(sub["bottom5_hits"].mean()))
        rows.append(
            {
                "strategy": spec.name,
                "filter": spec.filter_name,
                "mean_final_score": float(daily["final_score"].mean()),
                "median_final_score": float(daily["final_score"].median()),
                "mean_selected_sum": float(daily["selected_sum"].mean()),
                "mean_top5_hits": float(daily["top5_hits"].mean()),
                "mean_bottom5_hits": float(daily["bottom5_hits"].mean()),
                "fold_score_mean": float(np.mean(fold_scores)),
                "fold_score_min": float(np.min(fold_scores)),
                "fold_score_pos_frac": float(np.mean(np.asarray(fold_scores) > 0.0)),
                "fold_top_hits_min": float(np.min(fold_top_hits)),
                "fold_bottom_hits_max": float(np.max(fold_bottom_hits)),
            }
        )
    return pd.DataFrame(rows).sort_values(["fold_score_min", "mean_final_score"], ascending=[False, False])


def walk_forward_logistic(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(panel["date"].unique())
    # Expanding windows: start after enough history, then validate contiguous blocks.
    initial = 180
    block = 45
    rows = []
    coef_rows = []
    for start in range(initial, len(dates) - block + 1, block):
        train_dates = dates[:start]
        valid_dates = dates[start : start + block]
        train = panel[panel["date"].isin(train_dates)].copy()
        valid = panel[panel["date"].isin(valid_dates)].copy()
        x_train = train[MODEL_FEATURES].astype(float).fillna(0.5)
        y_train = train["is_top5"].astype(int)
        x_valid = valid[MODEL_FEATURES].astype(float).fillna(0.5)

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=0.5,
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            ),
        )
        model.fit(x_train, y_train)
        valid = valid.copy()
        valid["_wf_prob"] = model.predict_proba(x_valid)[:, 1]

        daily_rows = []
        for date, day in valid.groupby("date", sort=True):
            selected = day.sort_values("_wf_prob", ascending=False).head(5)
            record = score_selected(day, selected)
            record["date"] = date
            daily_rows.append(record)
        daily = pd.DataFrame(daily_rows)
        rows.append(
            {
                "fold": len(rows) + 1,
                "train_start": str(pd.Timestamp(train_dates[0]).date()),
                "train_end": str(pd.Timestamp(train_dates[-1]).date()),
                "valid_start": str(pd.Timestamp(valid_dates[0]).date()),
                "valid_end": str(pd.Timestamp(valid_dates[-1]).date()),
                "valid_dates": int(len(valid_dates)),
                "mean_final_score": float(daily["final_score"].mean()),
                "mean_selected_sum": float(daily["selected_sum"].mean()),
                "mean_top5_hits": float(daily["top5_hits"].mean()),
                "mean_bottom5_hits": float(daily["bottom5_hits"].mean()),
            }
        )

        clf = model.named_steps["logisticregression"]
        scaler = model.named_steps["standardscaler"]
        # Coefficients are on standardized features; store directly.
        for feature, coef in zip(MODEL_FEATURES, clf.coef_[0]):
            coef_rows.append(
                {
                    "fold": len(rows),
                    "feature": feature,
                    "coef": float(coef),
                    "train_mean": float(scaler.mean_[MODEL_FEATURES.index(feature)]),
                }
            )
    wf = pd.DataFrame(rows)
    coefs = pd.DataFrame(coef_rows)
    if not coefs.empty:
        coef_summary = coefs.groupby("feature").agg(
            coef_mean=("coef", "mean"),
            coef_std=("coef", "std"),
            coef_min=("coef", "min"),
            coef_max=("coef", "max"),
            coef_pos_frac=("coef", lambda s: float(np.mean(s > 0.0))),
        )
        coef_summary["abs_mean"] = coef_summary["coef_mean"].abs()
        coef_summary = coef_summary.reset_index().sort_values(["abs_mean", "coef_pos_frac"], ascending=[False, False])
    else:
        coef_summary = pd.DataFrame()
    return wf, coef_summary


def latest_signature(panel: pd.DataFrame) -> pd.DataFrame:
    latest = pd.read_csv(ROOT / "data" / "train_hs300_latest.csv", dtype={"股票代码": str})
    latest = latest.rename(
        columns={
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交额": "amount",
            "振幅": "amplitude",
            "换手率": "turnover",
        }
    )
    latest["stock_id"] = normalize_stock_id(latest["股票代码"])
    latest["date"] = pd.to_datetime(latest["日期"])
    latest = latest.sort_values(["stock_id", "date"]).reset_index(drop=True)
    for col in ["open", "close", "high", "low", "amount", "amplitude", "turnover"]:
        latest[col] = pd.to_numeric(latest[col], errors="coerce").astype(float)
    g = latest.groupby("stock_id", sort=False)
    latest["prev_close"] = g["close"].shift(1)
    latest["ret1"] = latest["close"] / (latest["prev_close"] + 1e-12) - 1.0
    for window in [5, 10, 20, 60]:
        latest[f"ret{window}"] = g["close"].pct_change(window, fill_method=None)
        min_periods = max(3, window // 2)
        latest[f"vol{window}"] = g["ret1"].transform(lambda s, w=window, m=min_periods: s.rolling(w, min_periods=m).std())
        latest[f"amp{window}"] = g["amplitude"].transform(lambda s, w=window, m=min_periods: s.rolling(w, min_periods=m).mean())
        latest[f"amount{window}"] = g["amount"].transform(lambda s, w=window, m=min_periods: s.rolling(w, min_periods=m).mean())
        latest[f"turnover{window}"] = g["turnover"].transform(lambda s, w=window, m=min_periods: s.rolling(w, min_periods=m).mean())
    latest["amount_ratio5_20"] = latest["amount5"] / (latest["amount20"] + 1e-12)
    latest["turnover_ratio5_20"] = latest["turnover5"] / (latest["turnover20"] + 1e-12)
    latest["vol_ratio5_20"] = latest["vol5"] / (latest["vol20"] + 1e-12)
    latest["amp_ratio5_20"] = latest["amp5"] / (latest["amp20"] + 1e-12)
    low20 = g["low"].transform(lambda s: s.rolling(20, min_periods=10).min())
    high20 = g["high"].transform(lambda s: s.rolling(20, min_periods=10).max())
    latest["pos20"] = (latest["close"] - low20) / (high20 - low20 + 1e-12)
    latest["drawdown20"] = latest["close"] / (high20 + 1e-12) - 1.0
    latest["dist_low20"] = latest["close"] / (low20 + 1e-12) - 1.0
    latest["log_amount20"] = np.log1p(latest["amount20"].clip(lower=0.0))

    latest_day = latest[latest["date"] == latest["date"].max()].copy()
    latest_day = latest_day.dropna(subset=["ret60", "vol20", "amount20", "pos20"])
    for col in [
        "amp20",
        "vol20",
        "turnover20",
        "dist_low20",
        "ret60",
        "ret20",
        "log_amount20",
        "pos20",
        "amount_ratio5_20",
        "drawdown20",
    ]:
        latest_day[f"{col}_rank"] = latest_day[col].rank(pct=True, method="average")
    latest_day["stable_elastic_score"] = (
        0.22 * latest_day["amp20_rank"]
        + 0.20 * latest_day["vol20_rank"]
        + 0.16 * latest_day["turnover20_rank"]
        + 0.12 * latest_day["dist_low20_rank"]
        + 0.10 * latest_day["ret60_rank"]
        + 0.10 * latest_day["ret20_rank"]
        + 0.10 * latest_day["log_amount20_rank"]
    )
    return latest_day.sort_values("stable_elastic_score", ascending=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    panel = add_regimes(panel)
    panel.to_pickle(OUT_DIR / "analysis_panel.pkl")

    factor = factor_stability(panel)
    factor.to_csv(OUT_DIR / "factor_stability.csv", index=False)

    regime = regime_profiles(panel)
    regime.to_csv(OUT_DIR / "regime_profiles.csv", index=False)

    conditions = condition_stability(panel)
    conditions.to_csv(OUT_DIR / "condition_stability.csv", index=False)

    strategies = evaluate_strategies(panel)
    strategies.to_csv(OUT_DIR / "strategy_backtest.csv", index=False)

    wf, coef_summary = walk_forward_logistic(panel)
    wf.to_csv(OUT_DIR / "walk_forward_logistic.csv", index=False)
    coef_summary.to_csv(OUT_DIR / "walk_forward_logistic_coefficients.csv", index=False)

    latest = latest_signature(panel)
    latest_cols = [
        "date",
        "stock_id",
        "stable_elastic_score",
        "amp20_rank",
        "vol20_rank",
        "turnover20_rank",
        "dist_low20_rank",
        "ret60_rank",
        "ret20_rank",
        "log_amount20_rank",
        "pos20_rank",
        "amount_ratio5_20_rank",
        "drawdown20_rank",
    ]
    latest[latest_cols].to_csv(OUT_DIR / "latest_stable_elastic_signature.csv", index=False)

    summary = {
        "valid_rows": int(len(panel)),
        "valid_dates": int(panel["date"].nunique()),
        "date_start": str(panel["date"].min().date()),
        "date_end": str(panel["date"].max().date()),
        "base_top5_rate": float(panel["is_top5"].mean()),
        "base_bottom5_rate": float(panel["is_bottom5"].mean()),
        "top_factors": factor.head(12).to_dict("records"),
        "top_conditions": conditions.head(15).to_dict("records"),
        "strategy_backtest": strategies.to_dict("records"),
        "walk_forward_logistic": wf.to_dict("records"),
        "top_logistic_coefficients": coef_summary.head(15).to_dict("records") if not coef_summary.empty else [],
        "latest_stable_elastic_top20": latest[latest_cols].head(20).to_dict("records"),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"valid_rows={len(panel)} valid_dates={panel['date'].nunique()} range={panel['date'].min().date()}..{panel['date'].max().date()}")
    print("\nTop factor stability:")
    print(
        factor[
            [
                "feature",
                "mean_daily_ic",
                "daily_ic_t",
                "daily_ic_pos_frac",
                "monthly_ic_pos_frac",
                "winner_mean_rank",
                "winner_minus_rest",
                "high_decile_top5_lift",
                "high_decile_bottom5_lift",
            ]
        ]
        .head(15)
        .to_string(index=False, float_format=lambda x: f"{x: .4f}")
    )
    print("\nTop stable conditions:")
    print(
        conditions[
            [
                "condition",
                "coverage",
                "top5_lift",
                "bottom5_lift",
                "edge_minus_tail",
                "fold_top5_lift_min",
                "fold_top5_lift_pos_frac",
                "mean_return",
            ]
        ]
        .head(15)
        .to_string(index=False, float_format=lambda x: f"{x: .4f}")
    )
    print("\nStrategy backtest:")
    print(strategies.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print("\nWalk-forward logistic:")
    print(wf.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print("\nTop logistic coefficients:")
    print(coef_summary.head(15).to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print("\nLatest stable elastic signature top20:")
    print(latest[latest_cols].head(20).to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
