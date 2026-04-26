from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from features import build_history_feature_frame  # noqa: E402
from labels import add_label_o2o_week  # noqa: E402
from reranker import make_rank_labels  # noqa: E402
from batch_window_analysis import load_raw, parse_anchor_args, rank_pct  # noqa: E402


FEATURES = [
    "ret10_pct", "ret20_pct", "pos20_pct", "amount20_pct", "turnover20_pct",
    "amp_mean10_pct", "vol10_pct", "amt_ratio5_pct", "to_ratio5_pct",
    "crowd_penalty", "extreme_vol_penalty",
]


def add_meta_features(frame):
    out = frame.copy()
    for col, source in {
        "ret10_pct": "ret10",
        "ret20_pct": "ret20",
        "pos20_pct": "pos20",
        "amount20_pct": "mean_amount20",
        "turnover20_pct": "turnover20",
        "amp_mean10_pct": "amp_mean10",
        "vol10_pct": "vol10",
        "amt_ratio5_pct": "amt_ratio5",
        "to_ratio5_pct": "to_ratio5",
    }.items():
        if source in out.columns:
            out[col] = out.groupby("日期")[source].transform(lambda s: s.rank(pct=True, method="average"))
        else:
            out[col] = 0.5
    out["crowd_penalty"] = (
        0.65 * ((out["amt_ratio5_pct"] - 0.80) / 0.20).clip(lower=0.0, upper=1.0)
        + 0.35 * ((out["to_ratio5_pct"] - 0.80) / 0.20).clip(lower=0.0, upper=1.0)
    )
    out["extreme_vol_penalty"] = ((out["vol10_pct"] - 0.93) / 0.07).clip(lower=0.0, upper=1.0)
    return out


def build_daily_frame(raw_labeled, date, cache):
    key = pd.Timestamp(date).strftime("%Y-%m-%d")
    if key not in cache:
        feat = build_history_feature_frame(raw_labeled, asof_date=date)
        labels = raw_labeled[raw_labeled["日期"] == pd.Timestamp(date)][["股票代码", "日期", "label_o2o_week"]].copy()
        labels["stock_id"] = labels["股票代码"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
        cache[key] = feat.merge(labels[["stock_id", "日期", "label_o2o_week"]], on="stock_id", how="inner")
    return cache[key].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/train_hs300_20260424.csv")
    parser.add_argument("--out-dir", default="temp/oof_base_scores")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--anchors", default=None)
    parser.add_argument("--start-anchor", default=None)
    parser.add_argument("--end-anchor", default=None)
    parser.add_argument("--last-n", type=int, default=None)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--min-history-days", type=int, default=120)
    parser.add_argument("--label-horizon", type=int, default=5)
    parser.add_argument("--train-window-days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = load_raw(ROOT / args.raw)
    raw_labeled = add_label_o2o_week(raw, horizon=args.label_horizon)
    dates = list(sorted(raw["日期"].dropna().unique()))
    anchors = parse_anchor_args(args, dates)
    run_dir = ROOT / args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cache = {}
    rows = []
    for anchor in anchors:
        anchor = pd.Timestamp(anchor)
        idx = dates.index(anchor)
        train_end = dates[idx - args.label_horizon]
        train_dates = [pd.Timestamp(d) for d in dates if pd.Timestamp(d) <= pd.Timestamp(train_end)][-args.train_window_days:]
        train_parts = [build_daily_frame(raw_labeled, d, cache) for d in train_dates]
        train = pd.concat(train_parts, ignore_index=True)
        train = make_rank_labels(train, label_col="label_o2o_week", date_col="日期")
        train = add_meta_features(train)
        target = train["is_top20"].astype(int)
        if target.nunique() < 2:
            continue
        model = LGBMClassifier(
            objective="binary",
            learning_rate=0.04,
            n_estimators=180,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            random_state=args.seed,
            n_jobs=4,
            verbosity=-1,
        )
        X = train[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        model.fit(X, target)

        pred = build_daily_frame(raw_labeled, anchor, cache)
        pred = add_meta_features(pred)
        proba = model.predict_proba(pred[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0))
        pred["oof_lgb_score"] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        pred["oof_lgb_score_pct"] = rank_pct(pred["oof_lgb_score"])
        pred["oof_lgb_rank_pct"] = rank_pct(pred["oof_lgb_score"])
        pred["anchor_date"] = anchor.strftime("%Y-%m-%d")
        pred["train_end_date"] = pd.Timestamp(train_end).strftime("%Y-%m-%d")
        rows.append(pred[["anchor_date", "train_end_date", "stock_id", "日期", "label_o2o_week", "oof_lgb_score", "oof_lgb_score_pct", "oof_lgb_rank_pct"] + FEATURES])
        print(f"[oof] anchor={anchor:%Y-%m-%d}, train_end={pd.Timestamp(train_end):%Y-%m-%d}, rows={len(train)}")

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(run_dir / "oof_lgb_scores.csv", index=False)
    aggregate = {
        "anchors": [pd.Timestamp(a).strftime("%Y-%m-%d") for a in anchors],
        "rows": int(len(out)),
        "feature_columns": FEATURES,
        "rule": "each anchor score is trained only on dates <= anchor - 5 trading days",
    }
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
