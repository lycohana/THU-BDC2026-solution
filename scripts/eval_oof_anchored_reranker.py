from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from batch_window_analysis import (  # noqa: E402
    add_branch_diagnostic_features,
    load_raw,
    parse_anchor_args,
    rank_pct,
    realized_returns_for_anchor,
    run_predict_for_anchor,
)
from reranker import make_rank_labels  # noqa: E402
from train_eval_reranker import build_candidate_sets  # noqa: E402


META_FEATURES = [
    "oof_lgb_score_pct",
    "oof_lgb_rank_pct",
    "baseline_rank_pct",
    "union_rank_pct",
    "legal_rank_pct",
    "source_count",
    "source_rrf_score",
    "in_baseline_top30",
    "in_union_top30",
    "in_legal_top30",
    "in_current_top120",
    "ret10_pct",
    "ret20_pct",
    "pos20_pct",
    "amount20_pct",
    "turnover20_pct",
    "amp_mean10_pct",
    "vol10_pct",
    "amt_ratio5_pct",
    "to_ratio5_pct",
    "crowd_penalty",
    "extreme_vol_penalty",
]

LAMBDA_GRID = [0.0, 0.05, 0.10, 0.20, 0.30]


def _norm_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.extract(r"(\d+)")[0].str.zfill(6)


def _clean_x(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _rank_map(df: pd.DataFrame, score_col: str) -> dict[str, int]:
    if score_col not in df.columns or df.empty:
        return {}
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    return {str(sid).zfill(6): i + 1 for i, sid in enumerate(ranked["stock_id"])}


def _rrf_from_rank(rank: float | int | None, k: float = 30.0) -> float:
    if pd.isna(rank):
        return 0.0
    return 1.0 / (k + float(rank))


def build_source_features(score_df: pd.DataFrame) -> pd.DataFrame:
    work = score_df.copy()
    work["stock_id"] = _norm_id(work["stock_id"])
    work = add_branch_diagnostic_features(work)
    candidate_sets = build_candidate_sets(work)

    baseline = candidate_sets["reference_baseline_top30"].copy()
    union = candidate_sets["union_topn_rrf_lcb_top30"].copy()
    legal = work.sort_values("score_legal_minrisk", ascending=False).head(30).copy()
    current = candidate_sets["current_aggressive_top120"].copy()

    baseline_rank = _rank_map(work.assign(_baseline_score=work.get("transformer", work["score"])), "_baseline_score")
    union_full = candidate_sets["union_topn_rrf_lcb_top30"]
    union_rank = _rank_map(union_full, "_union_rrf_lcb_score")
    legal_rank = _rank_map(work, "score_legal_minrisk")
    current_rank = _rank_map(current, "score")

    baseline_top = set(baseline["stock_id"])
    union_top = set(union["stock_id"])
    legal_top = set(legal["stock_id"])
    current_top = set(current["stock_id"])

    out = pd.DataFrame({"stock_id": work["stock_id"].astype(str).str.zfill(6)})
    out["baseline_rank"] = out["stock_id"].map(baseline_rank)
    out["union_rank"] = out["stock_id"].map(union_rank)
    out["legal_rank"] = out["stock_id"].map(legal_rank)
    out["current_rank"] = out["stock_id"].map(current_rank)

    out["baseline_rank_pct"] = work.get("transformer", work["score"]).rank(pct=True, method="average").to_numpy()
    union_scores = work["stock_id"].map(
        union_full.set_index("stock_id")["_union_rrf_lcb_score"].to_dict() if "_union_rrf_lcb_score" in union_full.columns else {}
    ).fillna(0.0)
    out["union_rank_pct"] = rank_pct(union_scores).to_numpy()
    out["legal_rank_pct"] = rank_pct(work["score_legal_minrisk"]).to_numpy()

    out["in_baseline_top30"] = out["stock_id"].isin(baseline_top).astype(int)
    out["in_union_top30"] = out["stock_id"].isin(union_top).astype(int)
    out["in_legal_top30"] = out["stock_id"].isin(legal_top).astype(int)
    out["in_current_top120"] = out["stock_id"].isin(current_top).astype(int)
    source_cols = ["in_baseline_top30", "in_union_top30", "in_legal_top30", "in_current_top120"]
    out["source_count"] = out[source_cols].sum(axis=1)
    out["source_rrf_score"] = (
        out["baseline_rank"].map(_rrf_from_rank)
        + out["union_rank"].map(_rrf_from_rank)
        + out["legal_rank"].map(_rrf_from_rank)
        + out["current_rank"].map(_rrf_from_rank)
    )
    out["source_agreement"] = 0.50 * (out["source_count"] / 4.0) + 0.50 * rank_pct(out["source_rrf_score"])
    return out


def train_meta_top20(train_df: pd.DataFrame, seed: int):
    if train_df.empty or train_df["is_top20"].nunique() < 2:
        return None
    model = LGBMClassifier(
        objective="binary",
        learning_rate=0.035,
        n_estimators=120,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=8.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(_clean_x(train_df, META_FEATURES), train_df["is_top20"].astype(int))
    return model


def positive_prob(model, pred_df: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.full(len(pred_df), 0.5, dtype=float)
    proba = model.predict_proba(_clean_x(pred_df, META_FEATURES))
    return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]


def score_selected(selected: pd.DataFrame) -> float:
    return float(selected["label_o2o_week"].fillna(0.0).mean()) if len(selected) else 0.0


def enrich_oof(oof: pd.DataFrame, raw: pd.DataFrame, anchors: list[pd.Timestamp], run_dir: Path, args) -> pd.DataFrame:
    parts = []
    for anchor in anchors:
        anchor_key = anchor.strftime("%Y-%m-%d")
        print(f"[anchored] enrich anchor={anchor_key}")
        day = oof[oof["anchor_date"] == anchor_key].copy()
        if day.empty:
            continue
        artifacts = run_predict_for_anchor(raw, anchor, run_dir, args.model_dir, not args.no_cache)
        score_df = pd.read_csv(artifacts["score_df_path"], dtype={"stock_id": str})
        source = build_source_features(score_df)
        day["stock_id"] = _norm_id(day["stock_id"])
        day = day.merge(source, on="stock_id", how="left")
        day = make_rank_labels(day, label_col="label_o2o_week", date_col="anchor_date")
        day["base_anchor"] = (
            0.60 * pd.to_numeric(day["oof_lgb_score_pct"], errors="coerce").fillna(0.5)
            + 0.40 * pd.to_numeric(day["in_current_top120"], errors="coerce").fillna(0.0)
        )
        day["risk_penalty"] = (
            0.60 * pd.to_numeric(day["crowd_penalty"], errors="coerce").fillna(0.0)
            + 0.40 * pd.to_numeric(day["extreme_vol_penalty"], errors="coerce").fillna(0.0)
        )
        for col in META_FEATURES + ["source_agreement", "base_anchor", "risk_penalty"]:
            if col not in day.columns:
                day[col] = 0.0
            day[col] = pd.to_numeric(day[col], errors="coerce").fillna(0.0)
        parts.append(day)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def evaluate(enriched: pd.DataFrame, anchors: list[pd.Timestamp], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    selections = []
    for anchor in anchors:
        anchor_key = anchor.strftime("%Y-%m-%d")
        pred = enriched[enriched["anchor_date"] == anchor_key].copy()
        train = enriched[enriched["anchor_date"] < anchor_key].copy()
        model = train_meta_top20(train, seed=seed)
        pred["meta_top20"] = positive_prob(model, pred)
        pred["meta_top20_pct"] = rank_pct(pred["meta_top20"])
        true_ranked = pred.sort_values("label_o2o_week", ascending=False).reset_index(drop=True)
        true_top20 = set(true_ranked.head(20)["stock_id"])
        true_top5 = set(true_ranked.head(5)["stock_id"])
        pool = pred[pred["in_current_top120"].eq(1) | pred["in_baseline_top30"].eq(1) | pred["in_union_top30"].eq(1) | pred["in_legal_top30"].eq(1)].copy()
        if pool.empty:
            pool = pred.copy()
        candidate_recall = len(set(pool["stock_id"]) & true_top20) / max(len(true_top20), 1)
        oracle_score = score_selected(pool.sort_values("label_o2o_week", ascending=False).head(5))
        for lambda_meta in LAMBDA_GRID:
            pool_eval = pool.copy()
            pool_eval["final_score"] = (
                0.70 * pool_eval["base_anchor"]
                + float(lambda_meta) * pool_eval["meta_top20_pct"]
                + 0.10 * pool_eval["source_agreement"]
                - 0.10 * pool_eval["risk_penalty"]
            )
            selected = pool_eval.sort_values("final_score", ascending=False).head(5).copy()
            selected_ids = set(selected["stock_id"])
            score = score_selected(selected)
            rows.append({
                "anchor_date": anchor_key,
                "lambda_meta": lambda_meta,
                "selected_score": score,
                "oracle_score_pool": oracle_score,
                "ranking_gap": oracle_score - score,
                "candidate_recall_true_top20": candidate_recall,
                "selected_true_top20_hit_count": len(selected_ids & true_top20),
                "selected_true_top5_hit_count": len(selected_ids & true_top5),
                "selected_ids": ",".join(selected["stock_id"].tolist()),
            })
            keep = selected[["anchor_date", "stock_id", "label_o2o_week", "final_score", "base_anchor", "meta_top20_pct", "source_agreement", "risk_penalty"]].copy()
            keep["lambda_meta"] = lambda_meta
            selections.append(keep)
    return pd.DataFrame(rows), pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby("lambda_meta")
        .agg(
            windows=("anchor_date", "nunique"),
            selected_score_mean=("selected_score", "mean"),
            selected_score_q10=("selected_score", lambda s: s.quantile(0.10)),
            selected_score_worst=("selected_score", "min"),
            ranking_gap_mean=("ranking_gap", "mean"),
            selected_true_top20_hit_count=("selected_true_top20_hit_count", "mean"),
            selected_true_top5_hit_count=("selected_true_top5_hit_count", "mean"),
            candidate_recall_true_top20=("candidate_recall_true_top20", "mean"),
        )
        .reset_index()
        .sort_values(["selected_score_mean", "selected_score_q10"], ascending=[False, False])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/train_hs300_20260424.csv")
    parser.add_argument("--oof-path", default=None)
    parser.add_argument("--oof-run-name", default="exp00305_oof_lgb_last20")
    parser.add_argument("--out-dir", default="temp/oof_anchored_reranker")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--anchors", default=None)
    parser.add_argument("--start-anchor", default=None)
    parser.add_argument("--end-anchor", default=None)
    parser.add_argument("--last-n", type=int, default=None)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--min-history-days", type=int, default=80)
    parser.add_argument("--label-horizon", type=int, default=5)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = load_raw(ROOT / args.raw)
    dates = list(sorted(raw["日期"].dropna().unique()))
    anchors = [pd.Timestamp(a) for a in parse_anchor_args(args, dates)]
    oof_path = Path(args.oof_path) if args.oof_path else ROOT / "temp" / "oof_base_scores" / args.oof_run_name / "oof_lgb_scores.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF score file not found: {oof_path}")
    oof = pd.read_csv(oof_path, dtype={"stock_id": str})
    run_dir = ROOT / args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    enriched = enrich_oof(oof, raw, anchors, run_dir, args)
    rows, selections = evaluate(enriched, anchors, seed=args.seed)
    summary = summarize(rows) if len(rows) else pd.DataFrame()
    enriched.to_csv(run_dir / "anchored_meta_features.csv", index=False)
    rows.to_csv(run_dir / "anchored_reranker_window_results.csv", index=False)
    selections.to_csv(run_dir / "anchored_reranker_selected.csv", index=False)
    summary.to_csv(run_dir / "anchored_reranker_summary.csv", index=False)
    aggregate = {
        "anchors": [a.strftime("%Y-%m-%d") for a in anchors],
        "lambda_grid": LAMBDA_GRID,
        "feature_columns": META_FEATURES,
        "formula": "final_score = 0.70 * base_anchor + lambda_meta * meta_top20_pct + 0.10 * source_agreement - 0.10 * risk_penalty",
        "summary": json.loads(summary.to_json(orient="records")) if len(summary) else [],
    }
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote reports to {run_dir}")


if __name__ == "__main__":
    main()
