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
    branch_table_for_window,
    load_raw,
    parse_anchor_args,
    realized_returns_for_anchor,
    run_predict_for_anchor,
)


TARGET_BRANCHES = [
    "reference_baseline_branch",
    "ai_hardware_mainline_v1",
    "current_aggressive",
    "union_topn_rrf_lcb",
    "legal_minrisk_hardened",
    "trend_uncluttered",
    "baseline_model_hybrid",
]


def analyze_anchor(raw, anchor, run_dir, model_dir, use_cache, label_horizon):
    artifacts = run_predict_for_anchor(raw, pd.Timestamp(anchor), run_dir, model_dir, use_cache)
    realized, score_window = realized_returns_for_anchor(raw, pd.Timestamp(anchor), label_horizon=label_horizon)
    table = branch_table_for_window(artifacts["score_df_path"], realized)
    table = table[table["branch"].isin(TARGET_BRANCHES)].copy()
    table.insert(0, "anchor_date", pd.Timestamp(anchor).strftime("%Y-%m-%d"))
    table.insert(1, "score_window", score_window)
    return table


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    return (
        rows.groupby(["branch", "filter"], dropna=False)
        .agg(
            windows=("anchor_date", "nunique"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            q10_score=("score", lambda s: s.quantile(0.10)),
            worst_score=("score", "min"),
            win_rate=("score", lambda s: (s > 0).mean()),
            mean_bad_count=("bad_count", "mean"),
            mean_very_bad_count=("very_bad_count", "mean"),
        )
        .reset_index()
        .sort_values(["mean_score", "q10_score"], ascending=[False, False])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/train_hs300_20260424.csv")
    parser.add_argument("--out-dir", default="temp/baseline_compare")
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

    tables = []
    for anchor in anchors:
        print(f"[compare] running anchor={pd.Timestamp(anchor):%Y-%m-%d}")
        tables.append(
            analyze_anchor(
                raw=raw,
                anchor=pd.Timestamp(anchor),
                run_dir=run_dir,
                model_dir=args.model_dir,
                use_cache=not args.no_cache,
                label_horizon=args.label_horizon,
            )
        )

    rows = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    summary = summarize(rows)
    rows.to_csv(run_dir / "branch_compare.csv", index=False)
    summary.to_csv(run_dir / "branch_compare_summary.csv", index=False)
    aggregate = {
        "windows": int(len(set(rows["anchor_date"])) if not rows.empty else 0),
        "branches": json.loads(summary.to_json(orient="records")) if not summary.empty else [],
        "anchors": [pd.Timestamp(a).strftime("%Y-%m-%d") for a in anchors],
    }
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote reports to {run_dir}")


if __name__ == "__main__":
    main()
