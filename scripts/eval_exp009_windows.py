from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))

import replay_tri_state_selector as replay  # noqa: E402
from tri_state_selector import SelectorConfig, TriStateSelector  # noqa: E402
from tri_state_selector.data import normalize_panel  # noqa: E402


BAD_RET = -0.03
VERY_BAD_RET = -0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exp009 tri-state replay over the latest 10 evaluation windows.")
    parser.add_argument("--train", default=str(ROOT / "data" / "train.csv"))
    parser.add_argument("--test", default=str(ROOT / "data" / "test.csv"))
    parser.add_argument(
        "--exp008-summary",
        default=str(ROOT / "temp" / "batch_window_analysis" / "exp008_tri_state_tight_last10_to_20260424" / "window_summary.csv"),
        help="Existing exp008 ten-window summary for side-by-side comparison.",
    )
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "tri_state_replay" / "window_eval"))
    parser.add_argument("--summary", default=str(ROOT / "outputs" / "tri_state_replay" / "window_eval.csv"))
    parser.add_argument("--comparison", default=str(ROOT / "outputs" / "tri_state_replay" / "exp008_exp009_comparison.csv"))
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def read_raw_market(train_path: str | Path, test_path: str | Path) -> pd.DataFrame:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    raw = pd.concat([train, test], ignore_index=True)
    raw["日期"] = pd.to_datetime(raw["日期"])
    raw["股票代码"] = raw["股票代码"].map(lambda x: str(int(x)).zfill(6) if pd.notna(x) else x)
    return raw.sort_values(["日期", "股票代码"]).reset_index(drop=True)


def next_window_dates(raw: pd.DataFrame, asof: pd.Timestamp, days: int = 5) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(sorted(raw.loc[raw["日期"] > asof, "日期"].unique()))
    if len(dates) < days:
        raise ValueError(f"not enough future dates after {asof.date()} for a {days}-day score window")
    return dates[:days]


def score_prediction(result: pd.DataFrame, raw: pd.DataFrame, window_dates: pd.DatetimeIndex) -> tuple[float, dict[str, float]]:
    pred = result.copy()
    pred["stock_id"] = pred["stock_id"].astype(str).str.zfill(6)
    pred["weight"] = pd.to_numeric(pred["weight"], errors="coerce").fillna(0.0)
    future = raw[raw["日期"].isin(window_dates) & raw["股票代码"].isin(pred["stock_id"])].copy()
    rets: dict[str, float] = {}
    for stock_id, group in future.groupby("股票代码", sort=False):
        group = group.sort_values("日期").tail(5)
        if len(group) < 2:
            rets[stock_id] = 0.0
            continue
        start = float(group.iloc[0]["开盘"])
        end = float(group.iloc[-1]["开盘"])
        rets[stock_id] = end / (start + 1e-12) - 1.0
    score = 0.0
    for row in pred.itertuples(index=False):
        score += float(row.weight) * float(rets.get(row.stock_id, 0.0))
    return score, rets


def turnover_overlap(current: pd.DataFrame, previous: pd.DataFrame | None) -> tuple[float, float]:
    if previous is None or previous.empty:
        return 1.0, 0.0
    cur = current.set_index("stock_id")["weight"].astype(float)
    prev = previous.set_index("stock_id")["weight"].astype(float)
    names = sorted(set(cur.index) | set(prev.index))
    turnover = float((cur.reindex(names).fillna(0.0) - prev.reindex(names).fillna(0.0)).abs().sum())
    overlap = len(set(cur.index) & set(prev.index)) / max(len(set(cur.index) | set(prev.index)), 1)
    return turnover, float(overlap)


def run_window(raw: pd.DataFrame, panel: pd.DataFrame, asof: pd.Timestamp, out_dir: Path, top_k: int) -> dict[str, object]:
    window_dir = out_dir / asof.strftime("%Y%m%d")
    window_dir.mkdir(parents=True, exist_ok=True)
    output_path = window_dir / "result.csv"
    risk_path = window_dir / "risk_report.json"
    debug_path = window_dir / "debug_report.json"

    selector = TriStateSelector(
        SelectorConfig(
            mode="competition",
            output_top_k=top_k,
            force_top_k_full_invest=True,
            regime_hysteresis_days=1,
            top_quantile=0.40,
        )
    )
    hist_panel = panel[pd.to_datetime(panel["date"]) <= asof].copy()
    output = selector.rebalance(prices=hist_panel, asof=asof, prev_weights=pd.Series(dtype=float))
    result = replay.to_competition_result(output.target_weights, top_k)
    result.to_csv(output_path, index=False)

    window_dates = next_window_dates(raw, asof, 5)
    selected_score, selected_rets = score_prediction(result, raw, window_dates)
    debug = replay.build_debug_report(asof=asof, output=output, result=result, output_path=output_path, top_k=top_k)
    debug.update(
        {
            "score_window": f"{window_dates[0].date()}~{window_dates[-1].date()}",
            "score_self": selected_score,
            "selected_returns": selected_rets,
        }
    )
    debug_path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
    risk_path.write_text(json.dumps(replay.json_safe(output.risk_report), ensure_ascii=False, indent=2), encoding="utf-8")

    selected = result["stock_id"].astype(str).str.zfill(6).tolist()
    weights = result["weight"].astype(float).tolist()
    stock_rets = [float(selected_rets.get(stock_id, 0.0)) for stock_id in selected]
    return {
        "asof": str(asof.date()),
        "score_window": f"{window_dates[0].date()}~{window_dates[-1].date()}",
        "regime": output.state.value,
        "confidence": float(output.confidence),
        "selected_top5": ",".join(selected),
        "weights": ",".join(f"{w:.6f}" for w in weights),
        "selected_rets": ",".join(f"{r:.2%}" for r in stock_rets),
        "score_self": float(selected_score),
        "mean_selected_score": float(np.mean(stock_rets)) if stock_rets else 0.0,
        "median_selected_score": float(np.median(stock_rets)) if stock_rets else 0.0,
        "q10_selected_score": float(np.quantile(stock_rets, 0.10)) if stock_rets else 0.0,
        "worst_selected_score": float(np.min(stock_rets)) if stock_rets else 0.0,
        "bad_count": int(sum(r <= BAD_RET for r in stock_rets)),
        "very_bad_count": int(sum(r <= VERY_BAD_RET for r in stock_rets)),
        "debug_report": str(debug_path),
        "_result": result,
    }


def build_comparison(exp009: pd.DataFrame, exp008_path: str | Path) -> pd.DataFrame:
    if not Path(exp008_path).exists():
        return pd.DataFrame()
    exp008 = pd.read_csv(exp008_path)
    exp008 = exp008.rename(columns={"anchor_date": "asof", "selected_score": "exp008"})
    merged = exp008[["asof", "exp008", "regime", "chosen_branch", "selected_picks"]].merge(
        exp009[["asof", "score_self", "regime", "selected_top5"]],
        on="asof",
        how="inner",
        suffixes=("_exp008", "_exp009"),
    )
    merged = merged.rename(columns={"score_self": "exp009"})
    merged["delta"] = merged["exp009"] - merged["exp008"]
    return merged[
        [
            "asof",
            "exp008",
            "exp009",
            "delta",
            "regime_exp008",
            "regime_exp009",
            "chosen_branch",
            "selected_picks",
            "selected_top5",
        ]
    ]


def main() -> None:
    args = parse_args()
    raw = read_raw_market(args.train, args.test)
    panel = normalize_panel(raw)
    exp008_summary = pd.read_csv(args.exp008_summary)
    anchors = pd.to_datetime(exp008_summary["anchor_date"]).tail(10).tolist()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    prev_result: pd.DataFrame | None = None
    for asof in anchors:
        row = run_window(raw, panel, pd.Timestamp(asof), out_dir, args.top_k)
        turnover, overlap = turnover_overlap(row["_result"], prev_result)
        row["turnover_vs_prev"] = turnover
        row["overlap_vs_prev"] = overlap
        prev_result = row["_result"]
        row.pop("_result")
        rows.append(row)
        print(f"{row['asof']} {row['regime']} score={row['score_self']:.6f} picks={row['selected_top5']}")

    summary = pd.DataFrame(rows)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    comparison = build_comparison(summary, args.exp008_summary)
    comparison_path = Path(args.comparison)
    if not comparison.empty:
        comparison.to_csv(comparison_path, index=False)

    aggregate = {
        "windows": int(len(summary)),
        "mean_selected_score": float(summary["score_self"].mean()),
        "median_selected_score": float(summary["score_self"].median()),
        "q10_selected_score": float(summary["score_self"].quantile(0.10)),
        "worst_selected_score": float(summary["score_self"].min()),
        "bad_count": int(summary["bad_count"].sum()),
        "very_bad_count": int(summary["very_bad_count"].sum()),
        "mean_turnover_vs_prev": float(summary["turnover_vs_prev"].mean()),
        "mean_overlap_vs_prev": float(summary["overlap_vs_prev"].mean()),
    }
    (summary_path.parent / "window_eval_aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote: {summary_path}")
    if not comparison.empty:
        print(f"comparison: {comparison_path}")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
