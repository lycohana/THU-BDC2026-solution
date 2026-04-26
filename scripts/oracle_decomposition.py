from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from batch_window_analysis import (  # noqa: E402
    load_raw,
    parse_anchor_args,
    realized_returns_for_anchor,
    run_predict_for_anchor,
    score_result,
)


def _ids(df: pd.DataFrame, n: int) -> str:
    return ",".join(df.head(n)["stock_id"].astype(str).tolist())


def _oracle_score(candidates: pd.DataFrame, n: int) -> float:
    pool = candidates.sort_values("score", ascending=False).head(n)
    if len(pool) == 0:
        return 0.0
    return float(pool.sort_values("realized_ret", ascending=False).head(5)["realized_ret"].mean())


def analyze_anchor(raw, anchor, run_dir, model_dir, use_cache, label_horizon):
    anchor_dir = run_dir / pd.Timestamp(anchor).strftime("%Y%m%d")
    artifacts = run_predict_for_anchor(raw, pd.Timestamp(anchor), run_dir, model_dir, use_cache)
    realized, score_window = realized_returns_for_anchor(raw, pd.Timestamp(anchor), label_horizon=label_horizon)
    selected_score, selected_detail = score_result(artifacts["output_path"], realized)

    score_df = pd.read_csv(artifacts["score_df_path"], dtype={"stock_id": str})
    score_df["stock_id"] = score_df["stock_id"].astype(str).str.zfill(6)
    work = score_df.merge(realized[["stock_id", "realized_ret"]], on="stock_id", how="left")
    work["realized_ret"] = work["realized_ret"].fillna(0.0)
    work = work.sort_values("score", ascending=False).reset_index(drop=True)

    true_ranked = work.sort_values("realized_ret", ascending=False).reset_index(drop=True)
    true_top5 = true_ranked.head(5)
    true_top20 = true_ranked.head(20)
    true_top20_set = set(true_top20["stock_id"])
    selected_ids = selected_detail["stock_id"].astype(str).str.zfill(6).tolist()

    row = {
        "anchor_date": pd.Timestamp(anchor).strftime("%Y-%m-%d"),
        "score_window": score_window,
        "true_top5": _ids(true_ranked, 5),
        "true_top20": _ids(true_ranked, 20),
        "selected_ids": ",".join(selected_ids),
        "selected_score": selected_score,
        "true_top5_score": float(true_top5["realized_ret"].mean()),
        "ranking_gap": float(true_top5["realized_ret"].mean() - selected_score),
    }
    for n in [20, 50, 80, 120]:
        topn = work.head(n)
        recall = len(set(topn["stock_id"]) & true_top20_set) / max(len(true_top20_set), 1)
        row[f"recall_true_top20_at_{n}"] = float(recall)
        row[f"oracle_score_top{n}"] = _oracle_score(work, n)
    row["recall_gap"] = float(1.0 - row["recall_true_top20_at_80"])

    detail = work.copy()
    detail["anchor_date"] = row["anchor_date"]
    detail["score_window"] = score_window
    detail.to_csv(anchor_dir / "oracle_rank_detail.csv", index=False)
    return row


def summarize(df: pd.DataFrame) -> dict:
    metric_cols = [
        "selected_score",
        "true_top5_score",
        "ranking_gap",
        "recall_gap",
        "recall_true_top20_at_20",
        "recall_true_top20_at_50",
        "recall_true_top20_at_80",
        "recall_true_top20_at_120",
        "oracle_score_top20",
        "oracle_score_top50",
        "oracle_score_top80",
        "oracle_score_top120",
    ]
    out = {"windows": int(len(df))}
    for col in metric_cols:
        if col in df.columns:
            out[f"mean_{col}"] = float(df[col].mean())
            out[f"median_{col}"] = float(df[col].median())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/train_hs300_20260424.csv")
    parser.add_argument("--out-dir", default="temp/oracle_decomposition")
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
    args = parser.parse_args()

    raw = load_raw(ROOT / args.raw)
    dates = list(sorted(raw["日期"].dropna().unique()))
    anchors = parse_anchor_args(args, dates)
    run_dir = ROOT / args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for anchor in anchors:
        print(f"[oracle] running anchor={pd.Timestamp(anchor):%Y-%m-%d}")
        rows.append(
            analyze_anchor(
                raw=raw,
                anchor=pd.Timestamp(anchor),
                run_dir=run_dir,
                model_dir=args.model_dir,
                use_cache=not args.no_cache,
                label_horizon=args.label_horizon,
            )
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "oracle_decomposition.csv", index=False)
    aggregate = summarize(summary)
    aggregate["anchors"] = [pd.Timestamp(a).strftime("%Y-%m-%d") for a in anchors]
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote reports to {run_dir}")


if __name__ == "__main__":
    main()
