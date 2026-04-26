from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_aggressive_basket_failures import (  # noqa: E402
    basket_row,
    branch_frames,
    enrich_work,
    load_oof_scores,
    score_selection,
    select_current_top5,
)
from batch_window_analysis import load_raw, parse_anchor_args, realized_returns_for_anchor, run_predict_for_anchor  # noqa: E402
from features import build_history_feature_frame  # noqa: E402
from stock_profile import add_low_upside_flags, build_stock_upside_profile  # noqa: E402
from eval_basket_veto_repair import basket_risk_flags, full_fallback, selected_score, update_history  # noqa: E402


def feature_history(raw: pd.DataFrame, dates: list[pd.Timestamp], cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for date in dates:
        key = pd.Timestamp(date).strftime("%Y-%m-%d")
        if key not in cache:
            feat = build_history_feature_frame(raw, asof_date=pd.Timestamp(date))
            feat["日期"] = pd.Timestamp(date)
            cache[key] = feat
        rows.append(cache[key])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def profile_for_anchor(raw: pd.DataFrame, all_dates: list[pd.Timestamp], anchor: pd.Timestamp, label_horizon: int, feature_cache: dict[str, pd.DataFrame], latest_work: pd.DataFrame):
    idx = all_dates.index(anchor)
    train_end = pd.Timestamp(all_dates[idx - label_horizon])
    train_slice = raw[raw["日期"] <= train_end].copy()
    train_dates = [pd.Timestamp(d) for d in all_dates if pd.Timestamp(d) <= train_end][-90:]
    feat_hist = feature_history(raw, train_dates, feature_cache)
    profile = build_stock_upside_profile(train_slice, feat_hist)
    flags = add_low_upside_flags(profile, latest_work)
    flags.insert(0, "anchor_date", anchor.strftime("%Y-%m-%d"))
    flags.insert(1, "profile_train_end", train_end.strftime("%Y-%m-%d"))
    return flags


def merge_flags(work: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "stock_id",
        "low_upside_drag",
        "very_low_upside_drag",
        "q90_week_return_pct",
        "q95_week_return_pct",
        "top20_hit_rate_pct",
        "top10_hit_rate_pct",
        "avg_amp10_pct_rank",
        "avg_vol10_pct_rank",
        "week_return_q90",
        "week_return_q95",
        "top20_hit_rate",
        "top10_hit_rate",
        "top5_hit_rate",
    ]
    available = [c for c in cols if c in flags.columns]
    out = work.merge(flags[available], on="stock_id", how="left")
    out["low_upside_drag"] = out["low_upside_drag"].fillna(False).astype(bool)
    out["very_low_upside_drag"] = out["very_low_upside_drag"].fillna(False).astype(bool)
    return out


def pick_low_upside_strategy(work: pd.DataFrame, flags_count: int) -> tuple[str, pd.DataFrame, dict]:
    frames = branch_frames(work)
    if flags_count >= 5:
        branch = "legal_minrisk_hardened"
        selected = frames["legal"].head(5).copy()
    elif flags_count >= 3:
        branch = "union_topn_rrf_lcb"
        pool = frames["union"][~frames["union"]["very_low_upside_drag"].astype(bool)].copy()
        if len(pool) < 5:
            pool = frames["union"].copy()
        selected = pool.head(5).copy()
    else:
        branch = "current_aggressive"
        pool = frames["current"][~frames["current"]["low_upside_drag"].astype(bool)].copy()
        if len(pool) < 5:
            pool = frames["current"].copy()
        selected = pool.head(5).copy()
    info = {
        "low_upside_selected_count": int(selected["low_upside_drag"].astype(bool).sum()) if "low_upside_drag" in selected.columns else 0,
        "very_low_upside_selected_count": int(selected["very_low_upside_drag"].astype(bool).sum()) if "very_low_upside_drag" in selected.columns else 0,
    }
    return branch, selected, info


def score_named_selection(strategy: str, anchor: pd.Timestamp, score_window: str, before_score: float, selected: pd.DataFrame, flags_count: int, branch: str, extra: dict):
    score = selected_score(selected)
    row = {
        "strategy": strategy,
        "anchor_date": anchor.strftime("%Y-%m-%d"),
        "score_window": score_window,
        "before_score": before_score,
        "after_score": score,
        "basket_risk_flags": flags_count,
        "chosen_branch": branch,
        **extra,
    }
    detail = selected.copy()
    detail["strategy"] = strategy
    detail["anchor_date"] = anchor.strftime("%Y-%m-%d")
    detail["score_window"] = score_window
    detail["after_score"] = score
    return row, detail


def summarize(windows: pd.DataFrame) -> dict:
    rows = []
    for strategy, group in windows.groupby("strategy", sort=False):
        after = pd.to_numeric(group["after_score"], errors="coerce")
        before = pd.to_numeric(group["before_score"], errors="coerce")
        rows.append({
            "strategy": strategy,
            "windows": int(len(group)),
            "mean": float(after.mean()),
            "q10": float(after.quantile(0.10)),
            "worst": float(after.min()),
            "win_rate": float((after > 0).mean()),
            "score_20260316_before": float(before[group["anchor_date"].eq("2026-03-16")].iloc[0]) if group["anchor_date"].eq("2026-03-16").any() else None,
            "score_20260316_after": float(after[group["anchor_date"].eq("2026-03-16")].iloc[0]) if group["anchor_date"].eq("2026-03-16").any() else None,
            "latest_anchor": str(group["anchor_date"].iloc[-1]) if len(group) else "",
            "latest_score_before": float(before.iloc[-1]) if len(group) else None,
            "latest_score_after": float(after.iloc[-1]) if len(group) else None,
            "mean_low_upside_selected_count": float(pd.to_numeric(group.get("low_upside_selected_count", 0), errors="coerce").fillna(0).mean()),
            "mean_very_low_upside_selected_count": float(pd.to_numeric(group.get("very_low_upside_selected_count", 0), errors="coerce").fillna(0).mean()),
        })
    summary_df = pd.DataFrame(rows).sort_values(["mean", "q10", "worst"], ascending=[False, False, False])
    return {
        "summary": json.loads(summary_df.to_json(orient="records")),
        "best_by_mean": summary_df.iloc[0].to_dict() if len(summary_df) else {},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/train_hs300_20260424.csv")
    parser.add_argument("--oof-path", default="temp/oof_base_scores/exp00305_oof_lgb_last20/oof_lgb_scores.csv")
    parser.add_argument("--out-dir", default="temp/low_upside_filter")
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
    all_dates = list(sorted(raw["日期"].dropna().unique()))
    anchors = [pd.Timestamp(x) for x in parse_anchor_args(args, all_dates)]
    run_dir = ROOT / args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    oof = load_oof_scores(ROOT / args.oof_path if args.oof_path else None)
    feature_cache = {}
    history = {k: [] for k in [
        "top5_score_margin",
        "top5_lgb_transformer_disagreement",
        "top5_meta_top20_prob_min",
        "top5_source_count_mean",
        "top5_source_rrf_mean",
        "top5_vol10_pct_mean",
        "top5_amp10_pct_mean",
        "top5_crowd_penalty_mean",
        "top5_recent_fade_mean",
        "top5_reversal_risk_mean",
        "top5_pairwise_corr_mean",
    ]}

    window_rows = []
    selected_rows = []
    profile_rows = []

    for anchor in anchors:
        print(f"[low-upside] anchor={anchor:%Y-%m-%d}")
        artifacts = run_predict_for_anchor(raw, anchor, run_dir, args.model_dir, not args.no_cache)
        realized, score_window = realized_returns_for_anchor(raw, anchor, label_horizon=args.label_horizon)
        score_df = pd.read_csv(artifacts["score_df_path"], dtype={"stock_id": str})
        work = enrich_work(raw, anchor, score_df, oof)
        work = work.merge(realized[["stock_id", "realized_ret"]], on="stock_id", how="left")
        work["realized_ret"] = work["realized_ret"].fillna(0.0)

        profile = profile_for_anchor(raw, all_dates, anchor, args.label_horizon, feature_cache, work)
        profile_rows.append(profile)
        work = merge_flags(work, profile)

        current = select_current_top5(work)
        before_score, current = score_selection(current, realized)
        basket = basket_row(raw, anchor, score_window, current, before_score, work)
        flags_count, q = basket_risk_flags(basket, history, meta_min_threshold_q=0.20)

        current_row, current_detail = score_named_selection(
            "current_aggressive",
            anchor,
            score_window,
            before_score,
            current,
            flags_count,
            "current_aggressive",
            {
                "low_upside_selected_count": int(current["low_upside_drag"].astype(bool).sum()),
                "very_low_upside_selected_count": int(current["very_low_upside_drag"].astype(bool).sum()),
            },
        )
        window_rows.append({**current_row, **q})
        selected_rows.append(current_detail)

        fallback_branch, fallback_selected = full_fallback(work, flags_count)
        fallback_row, fallback_detail = score_named_selection(
            "full_fallback_fixed",
            anchor,
            score_window,
            before_score,
            fallback_selected,
            flags_count,
            fallback_branch,
            {
                "low_upside_selected_count": int(fallback_selected["low_upside_drag"].astype(bool).sum()) if "low_upside_drag" in fallback_selected.columns else 0,
                "very_low_upside_selected_count": int(fallback_selected["very_low_upside_drag"].astype(bool).sum()) if "very_low_upside_drag" in fallback_selected.columns else 0,
            },
        )
        window_rows.append({**fallback_row, **q})
        selected_rows.append(fallback_detail)

        low_branch, low_selected, low_info = pick_low_upside_strategy(work, flags_count)
        low_row, low_detail = score_named_selection(
            "regime_full_fallback_low_upside_v1",
            anchor,
            score_window,
            before_score,
            low_selected,
            flags_count,
            low_branch,
            low_info,
        )
        window_rows.append({**low_row, **q})
        selected_rows.append(low_detail)
        update_history(history, basket)

    profiles = pd.concat(profile_rows, ignore_index=True) if profile_rows else pd.DataFrame()
    windows = pd.DataFrame(window_rows)
    details = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    profiles.to_csv(run_dir / "low_upside_profile.csv", index=False)
    profiles[profiles["low_upside_drag"] | profiles["very_low_upside_drag"]].to_csv(run_dir / "low_upside_names.csv", index=False)
    windows.to_csv(run_dir / "strategy_window_results.csv", index=False)
    details.to_csv(run_dir / "selected_details.csv", index=False)
    if windows["anchor_date"].eq("2026-03-16").any():
        windows[windows["anchor_date"].eq("2026-03-16")].to_csv(run_dir / "before_after_20260316.csv", index=False)
    latest_anchor = windows["anchor_date"].max() if len(windows) else ""
    if latest_anchor:
        windows[windows["anchor_date"].eq(latest_anchor)].to_csv(run_dir / "before_after_latest.csv", index=False)
    summary = summarize(windows)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\nWrote reports to {run_dir}")


if __name__ == "__main__":
    main()
