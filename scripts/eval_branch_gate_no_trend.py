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

from batch_window_analysis import branch_table_for_window, load_raw, parse_anchor_args, realized_returns_for_anchor, run_predict_for_anchor  # noqa: E402


def quantile_from_history(history, value, lower_is_risk=True):
    if len(history) < 5:
        return 0.5
    s = pd.Series(history, dtype=float)
    q = float((s <= float(value)).mean())
    return q if lower_is_risk else q


def market_features(raw, anchor):
    hist = raw[raw["日期"] <= anchor].copy()
    hist = hist.sort_values(["股票代码", "日期"])
    hist["ret1"] = hist.groupby("股票代码")["收盘"].pct_change()
    recent = hist.groupby("股票代码").tail(10)
    latest = hist.groupby("股票代码").tail(1)
    ret5 = []
    vol10 = []
    for _, group in recent.groupby("股票代码"):
        group = group.sort_values("日期")
        if len(group) >= 6:
            ret5.append(float(group["收盘"].iloc[-1] / (group["收盘"].iloc[-6] + 1e-12) - 1.0))
        if group["ret1"].notna().sum() >= 5:
            vol10.append(float(group["ret1"].std()))
    ret5_s = pd.Series(ret5, dtype=float)
    vol10_s = pd.Series(vol10, dtype=float)
    return {
        "market_ret_5d": float(ret5_s.mean()) if len(ret5_s) else 0.0,
        "breadth_5d": float((ret5_s > 0).mean()) if len(ret5_s) else 0.5,
        "market_vol_10d": float(vol10_s.mean()) if len(vol10_s) else 0.0,
        "market_dispersion_5d": float(ret5_s.std()) if len(ret5_s) else 0.0,
    }


def score_margin(score_df, col="score"):
    scores = score_df.sort_values(col, ascending=False)[col].reset_index(drop=True)
    if len(scores) <= 20:
        return 0.0
    return float(scores.iloc[:5].mean() - scores.iloc[5:20].mean())


def confidence_features(score_df):
    top5 = score_df.sort_values("score", ascending=False).head(5).copy()
    sigma_rank = score_df["sigma20"].rank(pct=True) if "sigma20" in score_df.columns else pd.Series(0.5, index=score_df.index)
    disagreement = (score_df["lgb"].rank(pct=True) - score_df["transformer"].rank(pct=True)).abs() if {"lgb", "transformer"}.issubset(score_df.columns) else pd.Series(0.5, index=score_df.index)
    crowd = pd.Series(0.0, index=score_df.index)
    if {"amt_ratio5", "to_ratio5"}.issubset(score_df.columns):
        amt_q = score_df["amt_ratio5"].rank(pct=True)
        to_q = score_df["to_ratio5"].rank(pct=True)
        crowd = 0.65 * ((amt_q - 0.80) / 0.20).clip(lower=0.0, upper=1.0) + 0.35 * ((to_q - 0.80) / 0.20).clip(lower=0.0, upper=1.0)
    return {
        "top5_score_margin": score_margin(score_df, "score"),
        "top5_sigma_mean": float(sigma_rank.loc[top5.index].mean()),
        "top5_model_disagreement": float(disagreement.loc[top5.index].mean()),
        "top5_crowd_mean": float(crowd.loc[top5.index].mean()),
    }


def choose_branch(flags):
    if flags >= 5:
        return "legal_minrisk_hardened"
    if flags >= 3:
        return "union_topn_rrf_lcb"
    return "current_aggressive"


def evaluate(raw, anchor, run_dir, model_dir, use_cache, history):
    artifacts = run_predict_for_anchor(raw, anchor, run_dir, model_dir, use_cache)
    realized, score_window = realized_returns_for_anchor(raw, anchor)
    branch_table = branch_table_for_window(artifacts["score_df_path"], realized)
    scores = branch_table.set_index("branch")["score"].to_dict()
    picks = branch_table.set_index("branch")["picks"].to_dict()
    score_df = pd.read_csv(artifacts["score_df_path"], dtype={"stock_id": str})

    feats = {}
    feats.update(market_features(raw, anchor))
    feats.update(confidence_features(score_df))

    q = {
        "market_ret_5d_q": quantile_from_history(history["market_ret_5d"], feats["market_ret_5d"]),
        "breadth_5d_q": quantile_from_history(history["breadth_5d"], feats["breadth_5d"]),
        "market_vol_10d_q": quantile_from_history(history["market_vol_10d"], feats["market_vol_10d"]),
        "market_dispersion_5d_q": quantile_from_history(history["market_dispersion_5d"], feats["market_dispersion_5d"]),
        "top5_score_margin_q": quantile_from_history(history["top5_score_margin"], feats["top5_score_margin"]),
        "top5_sigma_mean_q": quantile_from_history(history["top5_sigma_mean"], feats["top5_sigma_mean"]),
        "top5_model_disagreement_q": quantile_from_history(history["top5_model_disagreement"], feats["top5_model_disagreement"]),
        "top5_crowd_mean_q": quantile_from_history(history["top5_crowd_mean"], feats["top5_crowd_mean"]),
    }
    flags = int(q["market_ret_5d_q"] <= 0.30)
    flags += int(q["breadth_5d_q"] <= 0.35)
    flags += int(q["market_vol_10d_q"] >= 0.70)
    flags += int(q["market_dispersion_5d_q"] >= 0.70)
    flags += int(q["top5_score_margin_q"] <= 0.30)
    flags += int(q["top5_sigma_mean_q"] >= 0.75)
    flags += int(q["top5_model_disagreement_q"] >= 0.70)
    flags += int(q["top5_crowd_mean_q"] >= 0.70)
    chosen = choose_branch(flags)

    for k, v in feats.items():
        history[k].append(v)

    return {
        "anchor_date": anchor.strftime("%Y-%m-%d"),
        "score_window": score_window,
        **feats,
        **q,
        "risk_flags": flags,
        "chosen_branch": chosen,
        "chosen_score": float(scores.get(chosen, 0.0)),
        "chosen_picks": picks.get(chosen, ""),
        "score_current_aggressive": scores.get("current_aggressive"),
        "score_union_topn_rrf_lcb": scores.get("union_topn_rrf_lcb"),
        "score_legal_minrisk_hardened": scores.get("legal_minrisk_hardened"),
        "score_reference_baseline_branch": scores.get("reference_baseline_branch"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/train_hs300_20260424.csv")
    parser.add_argument("--out-dir", default="temp/branch_gate")
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
    history = {k: [] for k in [
        "market_ret_5d", "breadth_5d", "market_vol_10d", "market_dispersion_5d",
        "top5_score_margin", "top5_sigma_mean", "top5_model_disagreement", "top5_crowd_mean",
    ]}
    rows = []
    for anchor in anchors:
        anchor = pd.Timestamp(anchor)
        print(f"[gate] running anchor={anchor:%Y-%m-%d}")
        rows.append(evaluate(raw, anchor, run_dir, args.model_dir, not args.no_cache, history))
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "branch_gate_no_trend.csv", index=False)
    aggregate = {
        "windows": int(len(df)),
        "mean_score": float(df["chosen_score"].mean()) if len(df) else 0.0,
        "q10_score": float(df["chosen_score"].quantile(0.10)) if len(df) else 0.0,
        "worst_score": float(df["chosen_score"].min()) if len(df) else 0.0,
        "win_rate": float((df["chosen_score"] > 0).mean()) if len(df) else 0.0,
        "branch_usage": df["chosen_branch"].value_counts().to_dict() if len(df) else {},
    }
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"\nWrote reports to {run_dir}")


if __name__ == "__main__":
    main()
