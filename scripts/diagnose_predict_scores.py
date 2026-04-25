"""Diagnose one prediction cross-section against the local 5-day scorer.

This script is intentionally post-training only. It reads the latest
predict_score_df.csv and result.csv, joins realized local test returns, and
prints:
- selected contribution table
- target stock ranks across transformer / lgb / blend
- a small no-retrain ablation grid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TARGETS = [
    "601600",
    "603799",
    "600029",
    "600893",
    "600584",
    "600919",
    "601169",
    "601398",
    "601988",
    "601658",
]


def normalize_stock_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    return (values - values.mean()) / (values.std(ddof=0) + 1e-12)


def rank_pct(series: pd.Series) -> pd.Series:
    return series.astype(float).rank(pct=True, method="average")


def load_realized_returns(test_path: Path) -> pd.DataFrame:
    test = pd.read_csv(test_path, dtype={"股票代码": str})
    test["股票代码"] = normalize_stock_id(test["股票代码"])
    test = test.sort_values(["股票代码", "日期"])
    rows = []
    for stock_id, group in test.groupby("股票代码", sort=False):
        start = float(group.iloc[0]["开盘"])
        end = float(group.iloc[-1]["开盘"])
        rows.append((stock_id, end / start - 1.0))
    return pd.DataFrame(rows, columns=["stock_id", "realized_ret"])


def add_filter(df: pd.DataFrame, filter_name: str) -> pd.DataFrame:
    out = df.copy()
    if filter_name == "none":
        return out
    if filter_name == "stable_010_085":
        out = out[out["median_amount20"] >= out["median_amount20"].quantile(0.10)]
        out = out[out["sigma20"] <= out["sigma20"].quantile(0.85)]
        return out
    if filter_name == "stable_030_070":
        out = out[out["median_amount20"] >= out["median_amount20"].quantile(0.30)]
        out = out[out["sigma20"] <= out["sigma20"].quantile(0.70)]
        return out
    if filter_name == "defensive":
        out = out[out["median_amount20"] >= out["median_amount20"].quantile(0.30)]
        out = out[out["sigma20"] <= out["sigma20"].quantile(0.70)]
        out = out[out["amp20"] <= out["amp20"].quantile(0.70)]
        out = out[out["ret5"] <= out["ret5"].quantile(0.90)]
        return out
    raise ValueError(f"Unsupported filter: {filter_name}")


def score_top5(df: pd.DataFrame, score_col: str, filter_name: str, exposure: float) -> dict:
    filtered = add_filter(df, filter_name)
    if len(filtered) < 5:
        return {}
    picks = filtered.sort_values(score_col, ascending=False).head(5).copy()
    weight = exposure / len(picks)
    score = float((picks["realized_ret"] * weight).sum())
    return {
        "variant": score_col,
        "filter": filter_name,
        "exposure": exposure,
        "score": score,
        "positive_count": int((picks["realized_ret"] > 0).sum()),
        "min_ret": float(picks["realized_ret"].min()),
        "picks": ",".join(picks["stock_id"].tolist()),
        "rets": ",".join(f"{x:.3%}" for x in picks["realized_ret"].tolist()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-df", default="app/temp/predict_score_df.csv")
    parser.add_argument("--filtered-df", default="app/temp/predict_filtered_top30.csv")
    parser.add_argument("--result", default="app/output/result.csv")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--out-dir", default="temp/diagnostics")
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score = pd.read_csv(args.score_df, dtype={"stock_id": str})
    score["stock_id"] = normalize_stock_id(score["stock_id"])
    realized = load_realized_returns(Path(args.test))
    score = score.merge(realized, on="stock_id", how="left")

    selected = pd.read_csv(args.result, dtype={"stock_id": str})
    selected["stock_id"] = normalize_stock_id(selected["stock_id"])
    selected_ids = set(selected["stock_id"])
    weight_map = dict(zip(selected["stock_id"], selected["weight"]))

    filtered_ids: set[str] = set()
    filter_rank = {}
    filtered_path = Path(args.filtered_df)
    if filtered_path.exists():
        filtered = pd.read_csv(filtered_path, dtype={"stock_id": str})
        filtered["stock_id"] = normalize_stock_id(filtered["stock_id"])
        filtered_ids = set(filtered["stock_id"])
        filter_rank = {stock_id: i + 1 for i, stock_id in enumerate(filtered["stock_id"].tolist())}

    for col in ["score", "transformer", "lgb"]:
        if col in score.columns:
            score[f"{col}_rank"] = score[col].rank(ascending=False, method="min").astype(int)

    targets = [str(stock_id).zfill(6) for stock_id in args.targets]
    diag = score[score["stock_id"].isin(targets)].copy()
    diag["passed_filter"] = diag["stock_id"].isin(filtered_ids) if filtered_ids else np.nan
    diag["filter_rank"] = diag["stock_id"].map(filter_rank)
    diag["selected"] = diag["stock_id"].isin(selected_ids)
    diag["weight"] = diag["stock_id"].map(weight_map).fillna(0.0)
    diag["contribution"] = diag["realized_ret"] * diag["weight"]

    diag_cols = [
        "stock_id",
        "realized_ret",
        "score",
        "score_rank",
        "transformer",
        "transformer_rank",
        "lgb",
        "lgb_rank",
        "sigma20",
        "ret5",
        "ret20",
        "amp20",
        "passed_filter",
        "filter_rank",
        "selected",
        "weight",
        "contribution",
    ]
    diag = diag[[col for col in diag_cols if col in diag.columns]]
    diag = diag.sort_values(["selected", "score_rank"], ascending=[False, True])
    diag.to_csv(out_dir / "target_stock_diagnostics.csv", index=False)

    selected_table = score[score["stock_id"].isin(selected_ids)].copy()
    selected_table["weight"] = selected_table["stock_id"].map(weight_map).fillna(0.0)
    selected_table["contribution"] = selected_table["realized_ret"] * selected_table["weight"]
    selected_table = selected_table.sort_values("contribution")
    selected_table.to_csv(out_dir / "selected_contributions.csv", index=False)

    variants = {
        "current_score": score["score"],
        "transformer_raw": score["transformer"],
        "lgb_raw": score["lgb"],
        "z_tf10": zscore(score["transformer"]),
        "z_lgb10": zscore(score["lgb"]),
    }
    for tw, lw in [(0.8, 0.2), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]:
        variants[f"z_tf{int(tw * 10)}_lgb{int(lw * 10)}"] = tw * zscore(score["transformer"]) + lw * zscore(score["lgb"])
        variants[f"rank_tf{int(tw * 10)}_lgb{int(lw * 10)}"] = tw * rank_pct(score["transformer"]) + lw * rank_pct(score["lgb"])
        disagreement = (rank_pct(score["transformer"]) - rank_pct(score["lgb"])).abs()
        variants[f"rank_tf{int(tw * 10)}_lgb{int(lw * 10)}_pen02"] = (
            tw * rank_pct(score["transformer"]) + lw * rank_pct(score["lgb"]) - 0.2 * disagreement
        )

    ablation_rows = []
    working = score.copy()
    for name, values in variants.items():
        working[name] = values
        for filter_name in ["none", "stable_010_085", "stable_030_070", "defensive"]:
            for exposure in [1.0, 0.7]:
                row = score_top5(working, name, filter_name, exposure)
                if row:
                    ablation_rows.append(row)

    ablation = pd.DataFrame(ablation_rows).sort_values("score", ascending=False)
    ablation.to_csv(out_dir / "ablation_grid.csv", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print("\nSelected contributions")
    print(selected_table[["stock_id", "realized_ret", "weight", "contribution"]].to_string(index=False))
    print("\nTarget diagnostics")
    print(diag.to_string(index=False))
    print("\nTop 20 ablations")
    print(ablation.head(20).to_string(index=False))
    print(f"\nWrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
