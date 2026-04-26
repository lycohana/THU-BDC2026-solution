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

from batch_window_analysis import (  # noqa: E402
    add_branch_diagnostic_features,
    load_raw,
    normalize_stock_id,
    parse_anchor_args,
    rank_pct,
    realized_returns_for_anchor,
    run_predict_for_anchor,
)
from portfolio_utils import apply_filter  # noqa: E402


def _clean_id(series: pd.Series) -> pd.Series:
    return normalize_stock_id(series)


def _rrf(rank_value, k: float = 30.0) -> float:
    if pd.isna(rank_value):
        return 0.0
    return 1.0 / (k + float(rank_value))


def _rank_map(df: pd.DataFrame, score_col: str) -> dict[str, int]:
    if df.empty or score_col not in df.columns:
        return {}
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    return {sid: i + 1 for i, sid in enumerate(ranked["stock_id"].astype(str).str.zfill(6))}


def _safe_mean(df: pd.DataFrame, col: str) -> float:
    return float(pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce").mean()) if len(df) else 0.0


def add_board_aware_features(work: pd.DataFrame) -> pd.DataFrame:
    out = work.copy()
    sid = out["stock_id"].astype(str).str.zfill(6)
    out["board_type"] = np.select(
        [sid.str.startswith("688"), sid.str.startswith("300") | sid.str.startswith("301")],
        ["star", "chinext"],
        default="main",
    )
    out["is_board20"] = out["board_type"].isin(["star", "chinext"]).astype(int)
    out["limit_ratio"] = np.where(out["is_board20"].astype(bool), 0.20, 0.10)
    limit = pd.Series(out["limit_ratio"], index=out.index).replace(0.0, np.nan).fillna(0.10)
    for target, source in {
        "ret1_limit_unit": "ret1",
        "ret3_limit_unit": "ret3",
        "ret5_limit_unit": "ret5",
        "amp10_limit_unit": "amp_mean10",
        "vol10_limit_unit": "vol10",
    }.items():
        raw = pd.to_numeric(out.get(source, pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
        out[target] = (raw.abs() / limit).clip(lower=0.0)

    for col in ["ret10", "ret20", "pos20", "vol10", "amp_mean10", "amt_ratio5", "to_ratio5"]:
        src = pd.to_numeric(out.get(col, pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
        pct_col = "amp10_pct_by_board" if col == "amp_mean10" else f"{col}_pct_by_board"
        out[pct_col] = src.groupby(out["board_type"]).rank(pct=True, method="average").fillna(0.5)
    return out


def add_ret3(raw: pd.DataFrame, anchor: pd.Timestamp) -> pd.DataFrame:
    hist = raw[raw["日期"] <= anchor].copy()
    hist["股票代码"] = _clean_id(hist["股票代码"])
    hist = hist.sort_values(["股票代码", "日期"])
    hist["close_lag3"] = hist.groupby("股票代码")["收盘"].shift(3)
    latest = hist.groupby("股票代码", sort=False).tail(1).copy()
    latest["stock_id"] = latest["股票代码"].astype(str).str.zfill(6)
    latest["ret3"] = pd.to_numeric(latest["收盘"], errors="coerce") / (pd.to_numeric(latest["close_lag3"], errors="coerce") + 1e-12) - 1.0
    return latest[["stock_id", "ret3"]]


def pairwise_corr_mean(raw: pd.DataFrame, anchor: pd.Timestamp, stock_ids: list[str], lookback: int = 40) -> float:
    ids = set(str(x).zfill(6) for x in stock_ids)
    hist = raw[(raw["日期"] <= anchor) & (raw["股票代码"].astype(str).str.zfill(6).isin(ids))].copy()
    if hist.empty or len(ids) <= 1:
        return 0.0
    hist["股票代码"] = _clean_id(hist["股票代码"])
    hist = hist.sort_values(["股票代码", "日期"])
    hist["ret1"] = hist.groupby("股票代码")["收盘"].pct_change(fill_method=None)
    recent_dates = sorted(hist["日期"].dropna().unique())[-int(lookback):]
    piv = hist[hist["日期"].isin(recent_dates)].pivot(index="日期", columns="股票代码", values="ret1")
    corr = piv.corr(min_periods=10)
    if corr.empty:
        return 0.0
    vals = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    return float(vals.mean()) if len(vals) else 0.0


def build_union_branch(work: pd.DataFrame) -> pd.DataFrame:
    source_cols = [
        ("score_lgb_only", 1.0),
        ("transformer", 1.0),
        ("score_balanced", 0.7),
        ("score_defensive_v2", 0.5),
        ("score_conservative_softrisk_v2", 0.5),
    ]
    candidate_ids: set[str] = set()
    for col, _ in source_cols:
        if col in work.columns:
            candidate_ids.update(work.nlargest(min(10, len(work)), col)["stock_id"].tolist())
    candidates = work[work["stock_id"].isin(candidate_ids)].copy()
    if candidates.empty:
        return candidates

    rrf = pd.Series(0.0, index=work.index)
    for col, weight in source_cols:
        if col not in work.columns:
            continue
        ranks = work[col].rank(method="min", ascending=False)
        rrf = rrf + weight / (30.0 + ranks)
    candidates["_rrf_rank"] = rank_pct(rrf.loc[candidates.index])
    top5_lgb_gain_rank = rank_pct(candidates["score_lgb_top5"]) if "score_lgb_top5" in candidates.columns else rank_pct(candidates.get("score_lgb_only", pd.Series(0.0, index=candidates.index)))
    conservative_rank = rank_pct(candidates["score_conservative_softrisk_v2"]) if "score_conservative_softrisk_v2" in candidates.columns else pd.Series(0.0, index=candidates.index)
    defensive_rank = rank_pct(candidates["score_defensive_v2"]) if "score_defensive_v2" in candidates.columns else pd.Series(0.0, index=candidates.index)
    overheat_penalty = candidates["overheat_flag"].astype(float) * (
        0.35 * candidates["ret5_rank"]
        + 0.25 * candidates["amp_rank"]
        + 0.25 * candidates.get("volume_spike_rank", 0.5)
        + 0.15 * candidates.get("turnover_rank", 0.5)
    )
    candidates["_union_rrf_lcb_score"] = (
        0.45 * candidates["_rrf_rank"]
        + 0.20 * top5_lgb_gain_rank
        + 0.15 * candidates.get("tf_norm", 0.0)
        + 0.10 * conservative_rank
        + 0.10 * defensive_rank
        - 0.25 * candidates["tail_risk_score"]
        - 0.20 * candidates["uncertainty_score"]
        - 0.10 * overheat_penalty
    )
    return candidates.sort_values("_union_rrf_lcb_score", ascending=False).copy()


def branch_frames(work: pd.DataFrame) -> dict[str, pd.DataFrame]:
    current = apply_filter(work.copy(), "regime_liquidity_anchor_risk_off", liquidity_quantile=0.10, sigma_quantile=0.85)
    current = current.sort_values("score", ascending=False).copy()
    union = build_union_branch(work)
    legal = apply_filter(work.copy(), "legal_minrisk_hardened", liquidity_quantile=0.10, sigma_quantile=0.85)
    legal = legal.sort_values("score_legal_minrisk", ascending=False).copy()
    baseline = work.sort_values("score_reference_baseline", ascending=False).copy()
    return {
        "current": current,
        "union": union,
        "legal": legal,
        "baseline": baseline,
    }


def load_oof_scores(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["anchor_date", "stock_id", "oof_lgb_score"])
    out = pd.read_csv(path, dtype={"stock_id": str})
    out["stock_id"] = _clean_id(out["stock_id"])
    return out


def enrich_work(raw: pd.DataFrame, anchor: pd.Timestamp, score_df: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    work = score_df.copy()
    work["stock_id"] = _clean_id(work["stock_id"])
    work = add_branch_diagnostic_features(work)
    work = work.merge(add_ret3(raw, anchor), on="stock_id", how="left")

    pct_sources = {
        "ret1_pct": "ret1",
        "ret3_pct": "ret3",
        "ret5_pct": "ret5",
        "ret10_pct": "ret10",
        "ret20_pct": "ret20",
        "pos20_pct": "pos20",
        "amp10_pct": "amp_mean10",
        "vol10_pct": "vol10",
        "amount20_pct": "mean_amount20",
        "turnover20_pct": "turnover20",
        "amt_ratio5_pct": "amt_ratio5",
        "to_ratio5_pct": "to_ratio5",
        "aggressive_score_pct": "score",
    }
    for out_col, src_col in pct_sources.items():
        work[out_col] = rank_pct(work[src_col]) if src_col in work.columns else 0.5
    work["liquidity_score_pct"] = 0.5 * work["amount20_pct"] + 0.5 * work["turnover20_pct"]
    work["crowd_penalty"] = (
        0.65 * ((work["amt_ratio5_pct"] - 0.80) / 0.20).clip(lower=0.0, upper=1.0)
        + 0.35 * ((work["to_ratio5_pct"] - 0.80) / 0.20).clip(lower=0.0, upper=1.0)
    )
    work["extreme_vol_penalty"] = ((work["vol10_pct"] - 0.93) / 0.07).clip(lower=0.0, upper=1.0)
    work["recent_fade"] = (
        0.60 * ((work["ret5_pct"] - 0.70) / 0.30).clip(lower=0.0, upper=1.0) * ((0.45 - work["ret1_pct"]) / 0.45).clip(lower=0.0, upper=1.0)
        + 0.40 * ((work["ret3_pct"] - 0.65) / 0.35).clip(lower=0.0, upper=1.0) * ((0.40 - work["ret1_pct"]) / 0.40).clip(lower=0.0, upper=1.0)
    )
    work["reversal_risk"] = (
        0.35 * work["recent_fade"]
        + 0.25 * work["crowd_penalty"]
        + 0.25 * work["extreme_vol_penalty"]
        + 0.15 * ((work["ret5_pct"] - 0.80) / 0.20).clip(lower=0.0, upper=1.0)
    )
    work = add_board_aware_features(work)

    anchor_key = anchor.strftime("%Y-%m-%d")
    day_oof = oof[oof["anchor_date"] == anchor_key][["stock_id", "oof_lgb_score"]].copy() if not oof.empty else pd.DataFrame(columns=["stock_id", "oof_lgb_score"])
    work = work.merge(day_oof.rename(columns={"oof_lgb_score": "meta_top20_prob"}), on="stock_id", how="left")
    work["meta_top20_prob"] = pd.to_numeric(work["meta_top20_prob"], errors="coerce").fillna(0.5)
    work["meta_top20_prob_pct"] = rank_pct(work["meta_top20_prob"])

    frames = branch_frames(work)
    baseline_rank = _rank_map(frames["baseline"], "score_reference_baseline")
    union_rank = _rank_map(frames["union"], "_union_rrf_lcb_score")
    legal_rank = _rank_map(frames["legal"], "score_legal_minrisk")
    current_rank = _rank_map(frames["current"], "score")
    work["rank_in_baseline"] = work["stock_id"].map(baseline_rank)
    work["rank_in_union"] = work["stock_id"].map(union_rank)
    work["rank_in_legal"] = work["stock_id"].map(legal_rank)
    work["aggressive_rank"] = work["stock_id"].map(current_rank)
    work["in_baseline_top30"] = work["rank_in_baseline"].le(30).fillna(False).astype(int)
    work["in_union_top30"] = work["rank_in_union"].le(30).fillna(False).astype(int)
    work["in_legal_top30"] = work["rank_in_legal"].le(30).fillna(False).astype(int)
    work["in_current_top120"] = work["aggressive_rank"].le(120).fillna(False).astype(int)
    work["source_count"] = work[["in_baseline_top30", "in_union_top30", "in_legal_top30", "in_current_top120"]].sum(axis=1)
    work["source_count_pct"] = work["source_count"] / 4.0
    work["source_rrf_score"] = (
        work["rank_in_baseline"].map(_rrf)
        + work["rank_in_union"].map(_rrf)
        + work["rank_in_legal"].map(_rrf)
        + work["aggressive_rank"].map(_rrf)
    )
    work["source_rrf_pct"] = rank_pct(work["source_rrf_score"])
    return work


def select_current_top5(work: pd.DataFrame) -> pd.DataFrame:
    frames = branch_frames(work)
    return frames["current"].sort_values("score", ascending=False).head(5).copy()


def score_selection(selected: pd.DataFrame, realized: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    detail = selected.copy()
    detail["stock_id"] = _clean_id(detail["stock_id"])
    if "realized_ret" not in detail.columns:
        detail = detail.merge(realized[["stock_id", "realized_ret"]], on="stock_id", how="left")
    detail["realized_ret"] = detail["realized_ret"].fillna(0.0)
    return float(detail["realized_ret"].mean()) if len(detail) else 0.0, detail


def basket_row(raw: pd.DataFrame, anchor: pd.Timestamp, score_window: str, selected: pd.DataFrame, selected_score: float, work: pd.DataFrame) -> dict:
    ids = selected["stock_id"].astype(str).str.zfill(6).tolist()
    top20_score = work.sort_values("score", ascending=False)["score"].reset_index(drop=True)
    margin = float(top20_score.iloc[:5].mean() - top20_score.iloc[5:20].mean()) if len(top20_score) >= 20 else 0.0
    frames = branch_frames(work)
    return {
        "window": score_window,
        "asof_date": anchor.strftime("%Y-%m-%d"),
        "test_start": score_window.split("~")[0] if "~" in score_window else "",
        "test_end": score_window.split("~")[-1] if "~" in score_window else "",
        "selected_score": selected_score,
        "is_crash_window": bool(selected_score <= -0.05),
        "top5_score_margin": margin,
        "top5_lgb_transformer_disagreement": _safe_mean(selected, "rank_disagreement"),
        "top5_meta_top20_prob_mean": _safe_mean(selected, "meta_top20_prob"),
        "top5_meta_top20_prob_min": float(selected["meta_top20_prob"].min()) if len(selected) else 0.0,
        "top5_source_count_mean": _safe_mean(selected, "source_count"),
        "top5_source_rrf_mean": _safe_mean(selected, "source_rrf_score"),
        "top5_overlap_union": len(set(ids) & set(frames["union"].head(20)["stock_id"].astype(str).str.zfill(6))),
        "top5_overlap_legal": len(set(ids) & set(frames["legal"].head(20)["stock_id"].astype(str).str.zfill(6))),
        "top5_overlap_baseline": len(set(ids) & set(frames["baseline"].head(20)["stock_id"].astype(str).str.zfill(6))),
        "top5_vol10_pct_mean": _safe_mean(selected, "vol10_pct"),
        "top5_amp10_pct_mean": _safe_mean(selected, "amp10_pct"),
        "top5_amt_ratio5_pct_mean": _safe_mean(selected, "amt_ratio5_pct"),
        "top5_to_ratio5_pct_mean": _safe_mean(selected, "to_ratio5_pct"),
        "top5_crowd_penalty_mean": _safe_mean(selected, "crowd_penalty"),
        "top5_recent_fade_mean": _safe_mean(selected, "recent_fade"),
        "top5_reversal_risk_mean": _safe_mean(selected, "reversal_risk"),
        "top5_pairwise_corr_mean": pairwise_corr_mean(raw, anchor, ids),
        "board20_count": int(pd.to_numeric(selected.get("is_board20", pd.Series(dtype=int)), errors="coerce").fillna(0).sum()),
        "board20_ret10_pct_by_board_mean": _safe_mean(selected[selected.get("is_board20", 0).astype(bool)] if "is_board20" in selected.columns else selected.iloc[0:0], "ret10_pct_by_board"),
        "board20_pos20_pct_by_board_mean": _safe_mean(selected[selected.get("is_board20", 0).astype(bool)] if "is_board20" in selected.columns else selected.iloc[0:0], "pos20_pct_by_board"),
        "board20_amt_ratio5_pct_by_board_mean": _safe_mean(selected[selected.get("is_board20", 0).astype(bool)] if "is_board20" in selected.columns else selected.iloc[0:0], "amt_ratio5_pct_by_board"),
        "board20_to_ratio5_pct_by_board_mean": _safe_mean(selected[selected.get("is_board20", 0).astype(bool)] if "is_board20" in selected.columns else selected.iloc[0:0], "to_ratio5_pct_by_board"),
        "board20_vol10_limit_unit_mean": _safe_mean(selected[selected.get("is_board20", 0).astype(bool)] if "is_board20" in selected.columns else selected.iloc[0:0], "vol10_limit_unit"),
        "board20_ret5_limit_unit_mean": _safe_mean(selected[selected.get("is_board20", 0).astype(bool)] if "is_board20" in selected.columns else selected.iloc[0:0], "ret5_limit_unit"),
    }


def selected_stock_rows(anchor: pd.Timestamp, score_window: str, selected: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "stock_id", "realized_ret", "aggressive_rank", "aggressive_score_pct", "rank_in_baseline",
        "rank_in_union", "rank_in_legal", "source_count", "source_rrf_score",
        "meta_top20_prob", "meta_top20_prob_pct", "ret1_pct", "ret3_pct", "ret5_pct",
        "ret10_pct", "ret20_pct", "pos20_pct", "amp10_pct", "vol10_pct",
        "amount20_pct", "turnover20_pct", "amt_ratio5_pct", "to_ratio5_pct",
        "crowd_penalty", "recent_fade", "reversal_risk",
        "board_type", "is_board20", "limit_ratio", "ret1_limit_unit",
        "ret3_limit_unit", "ret5_limit_unit", "amp10_limit_unit",
        "vol10_limit_unit", "ret10_pct_by_board", "ret20_pct_by_board",
        "pos20_pct_by_board", "vol10_pct_by_board", "amp10_pct_by_board",
        "amt_ratio5_pct_by_board", "to_ratio5_pct_by_board",
    ]
    out = selected.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    out = out[cols].copy()
    out.insert(0, "window", score_window)
    out.insert(1, "asof_date", anchor.strftime("%Y-%m-%d"))
    return out


def crash_feature_diff(baskets: pd.DataFrame) -> pd.DataFrame:
    numeric = baskets.select_dtypes(include=[np.number, bool]).copy()
    if "is_crash_window" not in baskets.columns or numeric.empty:
        return pd.DataFrame()
    crash = baskets["is_crash_window"].astype(bool)
    rows = []
    for col in numeric.columns:
        if col == "is_crash_window":
            continue
        crash_mean = float(pd.to_numeric(baskets.loc[crash, col], errors="coerce").mean()) if crash.any() else np.nan
        non_mean = float(pd.to_numeric(baskets.loc[~crash, col], errors="coerce").mean()) if (~crash).any() else np.nan
        diff = crash_mean - non_mean
        rows.append({"feature": col, "crash_mean": crash_mean, "non_crash_mean": non_mean, "diff": diff, "abs_diff": abs(diff)})
    out = pd.DataFrame(rows).sort_values("abs_diff", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


def analyze_anchor(raw, oof, anchor, run_dir, args):
    artifacts = run_predict_for_anchor(raw, anchor, run_dir, args.model_dir, not args.no_cache)
    realized, score_window = realized_returns_for_anchor(raw, anchor, label_horizon=args.label_horizon)
    score_df = pd.read_csv(artifacts["score_df_path"], dtype={"stock_id": str})
    work = enrich_work(raw, anchor, score_df, oof)
    selected = select_current_top5(work)
    selected_score, selected = score_selection(selected, realized)
    return basket_row(raw, anchor, score_window, selected, selected_score, work), selected_stock_rows(anchor, score_window, selected), work


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/train_hs300_20260424.csv")
    parser.add_argument("--oof-path", default="temp/oof_base_scores/exp00305_oof_lgb_last20/oof_lgb_scores.csv")
    parser.add_argument("--out-dir", default="temp/aggressive_basket_forensics")
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
    oof = load_oof_scores(ROOT / args.oof_path if args.oof_path else None)

    basket_rows = []
    stock_rows = []
    for anchor in anchors:
        anchor = pd.Timestamp(anchor)
        print(f"[basket] analyze anchor={anchor:%Y-%m-%d}")
        basket, stocks, _ = analyze_anchor(raw, oof, anchor, run_dir, args)
        basket_rows.append(basket)
        stock_rows.append(stocks)

    baskets = pd.DataFrame(basket_rows)
    selected = pd.concat(stock_rows, ignore_index=True) if stock_rows else pd.DataFrame()
    diff = crash_feature_diff(baskets)
    baskets.to_csv(run_dir / "aggressive_basket_features.csv", index=False)
    selected.to_csv(run_dir / "aggressive_selected_stock_features.csv", index=False)
    diff.to_csv(run_dir / "crash_feature_diff.csv", index=False)
    summary = {
        "windows": int(len(baskets)),
        "mean_score": float(baskets["selected_score"].mean()) if len(baskets) else 0.0,
        "q10_score": float(baskets["selected_score"].quantile(0.10)) if len(baskets) else 0.0,
        "worst_score": float(baskets["selected_score"].min()) if len(baskets) else 0.0,
        "crash_windows": int(baskets["is_crash_window"].sum()) if len(baskets) else 0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote reports to {run_dir}")


if __name__ == "__main__":
    main()
