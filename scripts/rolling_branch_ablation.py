"""Evaluate branch scores with the competition Top5 open-to-open scorer.

The input prediction table should contain at least:
date(optional), stock_id, transformer, lgb

If date is missing, the script uses --anchor-date, or the max date from
data/train.csv. Labels are built from the full raw market file with the global
trading calendar:
label_t = open[t+5] / open[t+1] - 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def normalize_stock_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.extract(r"(\d+)")[0].str.zfill(6)


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Missing columns {candidates}; got {df.columns.tolist()}")


def rank_pct_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("date", sort=False)[col].rank(pct=True, method="average")


def build_open_to_open_label(
    raw: pd.DataFrame,
    date_col: str = "日期",
    stock_col: str = "股票代码",
    open_col: str = "开盘",
) -> pd.DataFrame:
    data = raw[[stock_col, date_col, open_col]].copy()
    data[stock_col] = normalize_stock_id(data[stock_col])
    data[date_col] = pd.to_datetime(data[date_col])
    data[open_col] = pd.to_numeric(data[open_col], errors="coerce")

    dates = pd.Index(sorted(data[date_col].dropna().unique()))
    date_map = pd.DataFrame({date_col: dates, "date_idx": np.arange(len(dates))})
    base = data.merge(date_map, on=date_col, how="left")

    open_t1 = base[[stock_col, "date_idx", open_col]].copy()
    open_t1["date_idx"] -= 1
    open_t1 = open_t1.rename(columns={open_col: "open_t1"})

    open_t5 = base[[stock_col, "date_idx", open_col]].copy()
    open_t5["date_idx"] -= 5
    open_t5 = open_t5.rename(columns={open_col: "open_t5"})

    out = base.merge(open_t1, on=[stock_col, "date_idx"], how="left")
    out = out.merge(open_t5, on=[stock_col, "date_idx"], how="left")
    out["label_5d_open"] = out["open_t5"] / (out["open_t1"] + 1e-12) - 1.0
    return out.rename(columns={stock_col: "stock_id", date_col: "date"})[
        ["stock_id", "date", "open_t1", "open_t5", "label_5d_open"]
    ].dropna(subset=["label_5d_open"])


def build_risk_features(raw: pd.DataFrame) -> pd.DataFrame:
    date_col = pick_col(raw, ["日期", "date"])
    stock_col = pick_col(raw, ["股票代码", "stock_id"])
    close_col = pick_col(raw, ["收盘", "close"])
    high_col = pick_col(raw, ["最高", "high"])
    low_col = pick_col(raw, ["最低", "low"])
    amount_col = pick_col(raw, ["成交额", "amount", "turnover"])

    data = raw[[stock_col, date_col, close_col, high_col, low_col, amount_col]].copy()
    data = data.rename(columns={
        stock_col: "stock_id",
        date_col: "date",
        close_col: "close",
        high_col: "high",
        low_col: "low",
        amount_col: "amount",
    })
    data["stock_id"] = normalize_stock_id(data["stock_id"])
    data["date"] = pd.to_datetime(data["date"])
    for col in ["close", "high", "low", "amount"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.sort_values(["stock_id", "date"]).reset_index(drop=True)

    grouped = data.groupby("stock_id", sort=False)
    data["ret1"] = grouped["close"].pct_change(fill_method=None)
    data["ret5"] = grouped["close"].pct_change(5, fill_method=None)
    data["ret20"] = grouped["close"].pct_change(20, fill_method=None)
    data["sigma20"] = grouped["ret1"].rolling(20, min_periods=5).std().reset_index(level=0, drop=True)
    data["median_amount20"] = grouped["amount"].rolling(20, min_periods=5).median().reset_index(level=0, drop=True)
    rolling_high = grouped["high"].rolling(20, min_periods=5).max().reset_index(level=0, drop=True)
    rolling_low = grouped["low"].rolling(20, min_periods=5).min().reset_index(level=0, drop=True)
    data["amp20"] = (rolling_high - rolling_low) / (data["close"] + 1e-12)
    return data[["stock_id", "date", "ret1", "ret5", "ret20", "sigma20", "median_amount20", "amp20"]]


def add_branch_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lgb_norm"] = rank_pct_by_date(out, "lgb")
    out["tf_norm"] = rank_pct_by_date(out, "transformer")
    out["rank_disagreement"] = (out["lgb_norm"] - out["tf_norm"]).abs()
    out["liq_rank"] = rank_pct_by_date(out, "median_amount20")
    out["sigma_rank"] = rank_pct_by_date(out, "sigma20")
    out["amp_rank"] = rank_pct_by_date(out, "amp20")
    out["score_lgb_only"] = out["lgb_norm"]
    out["score_balanced"] = 0.70 * out["lgb_norm"] + 0.30 * out["tf_norm"] - 0.10 * out["rank_disagreement"]
    out["score_conservative"] = 0.30 * out["lgb_norm"] + 0.70 * out["tf_norm"] - 0.20 * out["rank_disagreement"]
    out["score_defensive"] = (
        0.20 * out["lgb_norm"]
        + 0.60 * out["tf_norm"]
        + 0.15 * out["liq_rank"]
        - 0.20 * out["sigma_rank"]
        - 0.15 * out["amp_rank"]
    )
    return out


def infer_regime(g: pd.DataFrame) -> str:
    breadth_1d = g["ret1"].gt(0).mean()
    median_ret5 = g["ret5"].median()
    if breadth_1d < 0.35 or median_ret5 < -0.02:
        return "risk_off"
    if breadth_1d < 0.50:
        return "mixed_defensive"
    if breadth_1d > 0.60 and median_ret5 > 0:
        return "risk_on"
    return "neutral"


def apply_filter(g: pd.DataFrame, filter_name: str) -> pd.DataFrame:
    if filter_name == "none":
        return g
    cond = pd.Series(True, index=g.index)
    if filter_name in {"liq30_sigma70", "defensive"}:
        cond &= g["median_amount20"] >= g["median_amount20"].quantile(0.30)
        cond &= g["sigma20"] <= g["sigma20"].quantile(0.70)
    if filter_name == "defensive":
        cond &= g["amp20"] <= g["amp20"].quantile(0.70)
        cond &= g["ret1"] > -0.035
        cond &= g["ret5"] > -0.08
        cond &= g["ret5"] < g["ret5"].quantile(0.90)
    filtered = g[cond].copy()
    return filtered if len(filtered) >= 30 else g


def score_one_date(g: pd.DataFrame, score_col: str, filter_name: str, k: int = 5, exposure: float = 1.0) -> dict:
    filtered = apply_filter(g, filter_name)
    picks = filtered.sort_values(score_col, ascending=False).head(k).copy()
    if len(picks) == 0:
        return {}
    weight = exposure / len(picks)
    contribution = picks["label_5d_open"] * weight
    return {
        "score": float(contribution.sum()),
        "positive_count": int((picks["label_5d_open"] > 0).sum()),
        "bad_pick_count": int((picks["label_5d_open"] < -0.03).sum()),
        "very_bad_pick_count": int((picks["label_5d_open"] < -0.05).sum()),
        "min_ret": float(picks["label_5d_open"].min()),
        "avg_ret": float(picks["label_5d_open"].mean()),
        "stocks": ",".join(picks["stock_id"].tolist()),
    }


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    def one(group: pd.DataFrame) -> pd.Series:
        mean_score = group["score"].mean()
        p10_score = group["score"].quantile(0.10)
        worst_score = group["score"].min()
        bad_window_rate = group["score"].lt(-0.01).mean()
        mean_bad_pick = group["bad_pick_count"].mean()
        metric = mean_score + 0.50 * p10_score - 0.50 * abs(worst_score) - 0.01 * bad_window_rate - 0.005 * mean_bad_pick
        return pd.Series({
            "dates": len(group),
            "mean_score": mean_score,
            "median_score": group["score"].median(),
            "win_rate": group["score"].gt(0).mean(),
            "bad_window_rate": bad_window_rate,
            "very_bad_window_rate": group["score"].lt(-0.02).mean(),
            "worst_score": worst_score,
            "p10_score": p10_score,
            "mean_positive_count": group["positive_count"].mean(),
            "mean_bad_pick_count": mean_bad_pick,
            "mean_very_bad_pick_count": group["very_bad_pick_count"].mean(),
            "selection_metric": metric,
        })

    return rows.groupby(["branch", "filter"], dropna=False).apply(one).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="temp/predict_score_df.csv")
    parser.add_argument("--raw", default=None)
    parser.add_argument("--train-cutoff", default="data/train.csv")
    parser.add_argument("--anchor-date", default=None)
    parser.add_argument("--out-dir", default="temp/rolling_branch_ablation")
    args = parser.parse_args()

    pred_path = ROOT / args.pred
    raw_path = Path(args.raw) if args.raw else ROOT / "data" / "train_hs300_20260424.csv"
    if not raw_path.exists():
        raw_path = ROOT / "data" / "train.csv"
    train_cutoff_path = ROOT / args.train_cutoff

    pred = pd.read_csv(pred_path, dtype={"stock_id": str})
    pred["stock_id"] = normalize_stock_id(pred["stock_id"])
    if "date" not in pred.columns:
        if args.anchor_date:
            anchor_date = pd.Timestamp(args.anchor_date)
        else:
            cutoff = pd.read_csv(train_cutoff_path, usecols=["日期"])
            anchor_date = pd.to_datetime(cutoff["日期"]).max()
        pred["date"] = anchor_date
    pred["date"] = pd.to_datetime(pred["date"])

    raw = pd.read_csv(raw_path, dtype={"股票代码": str})
    labels = build_open_to_open_label(raw)
    risk = build_risk_features(raw)
    work = pred.merge(labels, on=["stock_id", "date"], how="left")
    work = work.merge(risk, on=["stock_id", "date"], how="left", suffixes=("", "_raw"))
    work = work.dropna(subset=["label_5d_open", "transformer", "lgb"]).copy()
    for col in ["ret1", "ret5", "ret20", "sigma20", "median_amount20", "amp20"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        work[col] = work.groupby("date")[col].transform(lambda s: s.fillna(s.median()))
        work[col] = work[col].fillna(0.0)

    work = add_branch_scores(work)
    branch_defs = {
        "lgb_only": ("score_lgb_only", "none"),
        "balanced_blend": ("score_balanced", "none"),
        "conservative_blend": ("score_conservative", "none"),
        "conservative_liq30_sigma70": ("score_conservative", "liq30_sigma70"),
        "defensive_branch": ("score_defensive", "defensive"),
    }

    rows = []
    for date, group in work.groupby("date", sort=True):
        regime = infer_regime(group)
        for branch, (score_col, filter_name) in branch_defs.items():
            row = score_one_date(group, score_col, filter_name)
            if not row:
                continue
            row.update({
                "date": date,
                "regime": regime,
                "branch": branch,
                "filter": filter_name,
            })
            rows.append(row)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows).sort_values(["date", "branch"])
    summary = summarize(detail).sort_values("selection_metric", ascending=False)
    by_regime = detail.groupby(["regime", "branch", "filter"], dropna=False).apply(
        lambda g: pd.Series({
            "dates": len(g),
            "mean_score": g["score"].mean(),
            "worst_score": g["score"].min(),
            "p10_score": g["score"].quantile(0.10),
            "mean_bad_pick_count": g["bad_pick_count"].mean(),
        })
    ).reset_index()

    detail.to_csv(out_dir / "rolling_detail.csv", index=False)
    summary.to_csv(out_dir / "rolling_summary.csv", index=False)
    by_regime.to_csv(out_dir / "rolling_by_regime.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print("Summary")
    print(summary.to_string(index=False))
    print(f"\nWrote rolling ablation to {out_dir}")


if __name__ == "__main__":
    main()
