from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from labels import add_label_o2o_week  # noqa: E402
from reranker import predict_rerank_scores, train_lgb_ranker, train_top20_classifier, train_top5_classifier  # noqa: E402
from train_eval_reranker import (  # noqa: E402
    BASE_FEATURES,
    add_reranker_features,
    build_candidate_sets,
    build_training_frame,
)
from batch_window_analysis import load_raw, parse_anchor_args, realized_returns_for_anchor, run_predict_for_anchor  # noqa: E402


def ndcg_at_k(y_true, y_score, k):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    order = np.argsort(-y_score)[:k]
    gains = np.maximum(y_true[order], 0.0)
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(np.maximum(y_true, 0.0))[::-1][:k]
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    return dcg / (idcg + 1e-12)


def precision_at_k(pool, score_col, target_col, k):
    top = pool.sort_values(score_col, ascending=False).head(k)
    return float(top[target_col].mean()) if len(top) else 0.0


def score_top5(pool, score_col):
    return float(pool.sort_values(score_col, ascending=False).head(5)["realized_ret"].mean())


def validate_group_alignment(train_df):
    data = train_df.sort_values(["日期", "stock_id"]).reset_index(drop=True)
    group = data.groupby("日期", sort=False).size()
    checks = {
        "group_sum_equals_len": bool(int(group.sum()) == len(data)),
        "groups_have_unique_date": bool(data.groupby("日期", sort=False)["日期"].nunique().eq(1).all()),
        "sorted_by_date_stock_id": bool(data[["日期", "stock_id"]].equals(data[["日期", "stock_id"]].sort_values(["日期", "stock_id"]).reset_index(drop=True))),
        "rank_label_high_means_higher_return": True,
    }
    means = data.groupby("rank_label")["label_o2o_week"].mean()
    if len(means) >= 2:
        checks["rank_label_high_means_higher_return"] = bool(means.sort_index().is_monotonic_increasing)
    checks["group_count"] = int(len(group))
    checks["group_sum"] = int(group.sum())
    checks["train_rows"] = int(len(data))
    checks["rank_label_mean_by_label"] = {str(k): float(v) for k, v in means.items()}
    return checks


def audit_pool(pool, score_col="rerank_score"):
    out = pool.copy()
    out["is_top20_realized"] = (out["realized_rank"] <= 20).astype(int)
    out["is_top5_realized"] = (out["realized_rank"] <= 5).astype(int)
    metrics = {
        "ranker_ic_spearman": float(out[score_col].corr(out["realized_ret"], method="spearman")) if len(out) > 2 else 0.0,
        "ranker_ic_pearson": float(out[score_col].corr(out["realized_ret"], method="pearson")) if len(out) > 2 else 0.0,
        "top20_auc": 0.0,
        "top20_precision_at_5": precision_at_k(out, score_col, "is_top20_realized", 5),
        "top20_precision_at_10": precision_at_k(out, score_col, "is_top20_realized", 10),
        "top20_precision_at_20": precision_at_k(out, score_col, "is_top20_realized", 20),
        "top5_precision_at_5": precision_at_k(out, score_col, "is_top5_realized", 5),
        "ndcg_at_5": ndcg_at_k(out["realized_ret"], out[score_col], 5),
        "ndcg_at_20": ndcg_at_k(out["realized_ret"], out[score_col], 20),
        "positive_rerank_selected_score": score_top5(out, score_col),
        "negative_rerank_selected_score": score_top5(out.assign(_neg=-out[score_col]), "_neg"),
        "base_selected_score": score_top5(out, "score"),
        "oracle_score_pool": float(out.sort_values("realized_ret", ascending=False).head(5)["realized_ret"].mean()),
    }
    if out["is_top20_realized"].nunique() > 1:
        try:
            metrics["top20_auc"] = float(roc_auc_score(out["is_top20_realized"], out[score_col]))
        except ValueError:
            metrics["top20_auc"] = 0.0
    return metrics


def audit_anchor(raw, raw_labeled, dates, anchor, run_dir, history_cache, args):
    artifacts = run_predict_for_anchor(raw, anchor, run_dir, args.model_dir, not args.no_cache)
    realized, score_window = realized_returns_for_anchor(raw, anchor, label_horizon=args.label_horizon)
    score_df = pd.read_csv(artifacts["score_df_path"], dtype={"stock_id": str})
    score_df["stock_id"] = score_df["stock_id"].astype(str).str.zfill(6)
    work = add_reranker_features(score_df).merge(realized[["stock_id", "realized_ret"]], on="stock_id", how="left")
    work["realized_ret"] = work["realized_ret"].fillna(0.0)
    work["realized_rank"] = work["realized_ret"].rank(method="first", ascending=False)

    anchor_idx = dates.index(anchor)
    cutoff_date = dates[anchor_idx - args.label_horizon]
    train_df = build_training_frame(raw_labeled, dates, cutoff_date, history_cache, max_train_dates=args.max_train_dates)
    group_checks = validate_group_alignment(train_df)

    ranker = train_lgb_ranker(train_df, BASE_FEATURES, seed=args.seed)
    top20 = train_top20_classifier(train_df, BASE_FEATURES, seed=args.seed)
    top5 = train_top5_classifier(train_df, BASE_FEATURES, seed=args.seed)
    reranked = predict_rerank_scores(ranker, top20, top5, work, BASE_FEATURES)
    candidate_sets = build_candidate_sets(reranked)

    rows = []
    for pool_name in ["current_aggressive_top80", "current_aggressive_top120", "current_aggressive_top160", "candidate_union"]:
        pool = candidate_sets[pool_name].merge(
            reranked[["stock_id", "rank_score", "top20_prob", "top5_prob", "rerank_score", "realized_rank"]],
            on="stock_id",
            how="left",
            suffixes=("", "_rerank"),
        )
        if "rerank_score_rerank" in pool.columns:
            pool["rerank_score"] = pool["rerank_score_rerank"].fillna(pool.get("rerank_score", 0.0))
        metrics = audit_pool(pool, "rerank_score")
        rows.append({
            "anchor_date": anchor.strftime("%Y-%m-%d"),
            "score_window": score_window,
            "pool": pool_name,
            **metrics,
            **{f"group_check_{k}": v for k, v in group_checks.items() if k != "rank_label_mean_by_label"},
            "rank_label_mean_by_label": json.dumps(group_checks["rank_label_mean_by_label"], ensure_ascii=False),
            "select_descending": True,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/train_hs300_20260424.csv")
    parser.add_argument("--out-dir", default="temp/reranker_audit")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-run", default=None)
    parser.add_argument("--anchors", default=None)
    parser.add_argument("--start-anchor", default=None)
    parser.add_argument("--end-anchor", default=None)
    parser.add_argument("--last-n", type=int, default=None)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--min-history-days", type=int, default=100)
    parser.add_argument("--label-horizon", type=int, default=5)
    parser.add_argument("--max-train-dates", type=int, default=90)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.source_run and args.last_n is None and args.anchors is None:
        text = str(args.source_run)
        if "last10" in text:
            args.last_n = 10
        elif "last20" in text:
            args.last_n = 20

    raw = load_raw(ROOT / args.raw)
    raw_labeled = add_label_o2o_week(raw, horizon=args.label_horizon)
    dates = list(sorted(raw["日期"].dropna().unique()))
    anchors = parse_anchor_args(args, dates)
    run_dir = ROOT / args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history_cache = {}
    rows = []
    for anchor in anchors:
        anchor = pd.Timestamp(anchor)
        print(f"[audit] running anchor={anchor:%Y-%m-%d}")
        rows.append(audit_anchor(raw, raw_labeled, dates, anchor, run_dir, history_cache, args))

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    result.to_csv(run_dir / "reranker_failure_audit.csv", index=False)
    summary = (
        result.groupby("pool")
        .agg(
            windows=("anchor_date", "nunique"),
            mean_ic_spearman=("ranker_ic_spearman", "mean"),
            mean_ic_pearson=("ranker_ic_pearson", "mean"),
            mean_auc=("top20_auc", "mean"),
            mean_p20_at5=("top20_precision_at_5", "mean"),
            mean_p20_at20=("top20_precision_at_20", "mean"),
            mean_p5_at5=("top5_precision_at_5", "mean"),
            mean_ndcg5=("ndcg_at_5", "mean"),
            mean_pos_score=("positive_rerank_selected_score", "mean"),
            mean_neg_score=("negative_rerank_selected_score", "mean"),
            mean_base_score=("base_selected_score", "mean"),
            mean_oracle_score=("oracle_score_pool", "mean"),
        )
        .reset_index()
        if len(result)
        else pd.DataFrame()
    )
    summary.to_csv(run_dir / "reranker_failure_audit_summary.csv", index=False)
    aggregate = {
        "windows": int(result["anchor_date"].nunique()) if len(result) else 0,
        "summary": json.loads(summary.to_json(orient="records")) if len(summary) else [],
    }
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote reports to {run_dir}")


if __name__ == "__main__":
    main()
