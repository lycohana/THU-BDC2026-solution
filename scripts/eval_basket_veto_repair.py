from __future__ import annotations

import argparse
import json
import sys
from itertools import product
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


REPAIR_K_GRID = [0, 1, 2]
BASKET_RISK_THRESHOLD_GRID = [2, 3, 4, 5]
STOCK_VETO_THRESHOLD_GRID = [0.65, 0.70, 0.75, 0.80]
META_MIN_THRESHOLD_Q_GRID = [0.15, 0.20, 0.25, 0.30]


def q_from_history(history: list[float], value: float) -> float:
    if len(history) < 5:
        return 0.5
    s = pd.Series(history, dtype=float)
    return float((s <= float(value)).mean())


def basket_risk_flags(row: dict, history: dict[str, list[float]], meta_min_threshold_q: float) -> tuple[int, dict]:
    q = {
        "top5_score_margin_q": q_from_history(history["top5_score_margin"], row["top5_score_margin"]),
        "top5_lgb_transformer_disagreement_q": q_from_history(history["top5_lgb_transformer_disagreement"], row["top5_lgb_transformer_disagreement"]),
        "top5_meta_top20_prob_min_q": q_from_history(history["top5_meta_top20_prob_min"], row["top5_meta_top20_prob_min"]),
        "top5_source_count_mean_q": q_from_history(history["top5_source_count_mean"], row["top5_source_count_mean"]),
        "top5_source_rrf_mean_q": q_from_history(history["top5_source_rrf_mean"], row["top5_source_rrf_mean"]),
        "top5_vol10_pct_mean_q": q_from_history(history["top5_vol10_pct_mean"], row["top5_vol10_pct_mean"]),
        "top5_amp10_pct_mean_q": q_from_history(history["top5_amp10_pct_mean"], row["top5_amp10_pct_mean"]),
        "top5_crowd_penalty_mean_q": q_from_history(history["top5_crowd_penalty_mean"], row["top5_crowd_penalty_mean"]),
        "top5_recent_fade_mean_q": q_from_history(history["top5_recent_fade_mean"], row["top5_recent_fade_mean"]),
        "top5_reversal_risk_mean_q": q_from_history(history["top5_reversal_risk_mean"], row["top5_reversal_risk_mean"]),
        "top5_pairwise_corr_mean_q": q_from_history(history["top5_pairwise_corr_mean"], row["top5_pairwise_corr_mean"]),
    }
    flags = 0
    flags += int(q["top5_score_margin_q"] <= 0.30)
    flags += int(q["top5_lgb_transformer_disagreement_q"] >= 0.70)
    flags += int(q["top5_meta_top20_prob_min_q"] <= float(meta_min_threshold_q))
    flags += int(q["top5_source_count_mean_q"] <= 0.30)
    flags += int(q["top5_source_rrf_mean_q"] <= 0.30)
    raw_vol_flag = int(q["top5_vol10_pct_mean_q"] >= 0.75)
    raw_amp_flag = int(q["top5_amp10_pct_mean_q"] >= 0.75)
    flags += raw_vol_flag
    flags += raw_amp_flag
    flags += int(q["top5_crowd_penalty_mean_q"] >= 0.70)
    flags += int(q["top5_recent_fade_mean_q"] >= 0.70)
    flags += int(q["top5_reversal_risk_mean_q"] >= 0.70)
    flags += int(q["top5_pairwise_corr_mean_q"] >= 0.70)
    raw_count = raw_vol_flag + raw_amp_flag
    q["raw_vol_sigma_amp_flag_count"] = raw_count
    q["fallback_reason_mainly_raw_vol_sigma_amp"] = bool(raw_count >= 1 and raw_count >= (flags - raw_count))
    return flags, q


def update_history(history: dict[str, list[float]], row: dict) -> None:
    for key in history:
        history[key].append(float(row.get(key, 0.0)))


def selected_score(selected: pd.DataFrame) -> float:
    return float(pd.to_numeric(selected["realized_ret"], errors="coerce").fillna(0.0).mean()) if len(selected) else 0.0


def branch_top(work: pd.DataFrame, name: str, n: int) -> pd.DataFrame:
    frames = branch_frames(work)
    if name == "current":
        return frames["current"].head(n).copy()
    if name == "union":
        return frames["union"].head(n).copy()
    if name == "legal":
        return frames["legal"].head(n).copy()
    if name == "baseline":
        return frames["baseline"].head(n).copy()
    raise ValueError(name)


def stock_veto_score(selected: pd.DataFrame) -> pd.Series:
    return (
        0.30 * (1.0 - selected["source_count_pct"].fillna(0.0))
        + 0.25 * (1.0 - selected["meta_top20_prob_pct"].fillna(0.5))
        + 0.20 * selected["reversal_risk"].fillna(0.0)
        + 0.15 * selected["recent_fade"].fillna(0.0)
        + 0.10 * selected["extreme_vol_penalty"].fillna(0.0)
    )


def build_repair_pool(work: pd.DataFrame, selected_ids: set[str]) -> pd.DataFrame:
    current20 = branch_top(work, "current", 20)
    current20 = current20[
        (current20["reversal_risk"] <= 0.60)
        & (current20["extreme_vol_penalty"] <= 0.80)
        & (current20["crowd_penalty"] <= 0.80)
    ].copy()
    pool = pd.concat(
        [
            branch_top(work, "union", 20),
            branch_top(work, "legal", 20),
            branch_top(work, "baseline", 20),
            current20,
        ],
        ignore_index=True,
    )
    if pool.empty:
        return pool
    pool["stock_id"] = pool["stock_id"].astype(str).str.zfill(6)
    pool = pool.drop_duplicates("stock_id")
    pool = pool[~pool["stock_id"].isin(selected_ids)].copy()
    pool["repair_score"] = (
        0.40 * pool["source_rrf_pct"].fillna(0.0)
        + 0.25 * pool["aggressive_score_pct"].fillna(0.5)
        + 0.20 * pool["meta_top20_prob_pct"].fillna(0.5)
        + 0.15 * pool["liquidity_score_pct"].fillna(0.5)
        - 0.20 * pool["reversal_risk"].fillna(0.0)
        - 0.10 * pool["extreme_vol_penalty"].fillna(0.0)
    )
    return pool.sort_values("repair_score", ascending=False).copy()


def partial_repair(work: pd.DataFrame, current_selected: pd.DataFrame, need_repair: bool, repair_k: int, veto_threshold: float) -> tuple[pd.DataFrame, int, str]:
    selected = current_selected.copy()
    selected["stock_id"] = selected["stock_id"].astype(str).str.zfill(6)
    selected["stock_veto_score"] = stock_veto_score(selected)
    if not need_repair or repair_k <= 0:
        selected["selection_reason"] = "current_keep"
        return selected, 0, ""

    veto = selected[selected["stock_veto_score"] >= float(veto_threshold)].sort_values("stock_veto_score", ascending=False)
    if veto.empty:
        selected["selection_reason"] = "current_keep"
        return selected, 0, ""

    replace_ids = veto.head(int(repair_k))["stock_id"].tolist()
    keep = selected[~selected["stock_id"].isin(replace_ids)].copy()
    keep["selection_reason"] = "current_keep"
    pool = build_repair_pool(work, set(selected["stock_id"]))
    add = pool.head(max(0, 5 - len(keep))).copy()
    if add.empty:
        selected["selection_reason"] = "current_keep_no_repair_candidate"
        return selected, 0, ""
    add["selection_reason"] = "repair_add"
    repaired = pd.concat([keep, add], ignore_index=True).head(5)
    return repaired, int(len(add)), ",".join(replace_ids)


def full_fallback(work: pd.DataFrame, flags: int) -> tuple[str, pd.DataFrame]:
    if flags >= 5:
        return "legal_minrisk_hardened", branch_top(work, "legal", 5)
    if flags >= 3:
        return "union_topn_rrf_lcb", branch_top(work, "union", 5)
    return "current_aggressive", branch_top(work, "current", 5)


def board20_state(basket: dict) -> tuple[bool, bool]:
    board20_count = int(basket.get("board20_count", 0) or 0)
    trend_ok = (
        board20_count >= 1
        and float(basket.get("board20_ret10_pct_by_board_mean", 0.0) or 0.0) >= 0.65
        and float(basket.get("board20_pos20_pct_by_board_mean", 0.0) or 0.0) >= 0.60
        and float(basket.get("board20_amt_ratio5_pct_by_board_mean", 1.0) or 1.0) <= 0.80
        and float(basket.get("board20_vol10_limit_unit_mean", 1.0) or 1.0) <= 0.65
    )
    overheat = (
        float(basket.get("board20_ret5_limit_unit_mean", 0.0) or 0.0) >= 0.75
        or float(basket.get("board20_amt_ratio5_pct_by_board_mean", 0.0) or 0.0) >= 0.85
        or float(basket.get("board20_to_ratio5_pct_by_board_mean", 0.0) or 0.0) >= 0.85
    )
    return bool(trend_ok), bool(overheat)


def cap_board20_exposure(selected: pd.DataFrame, branch_candidates: pd.DataFrame, cap: int = 2) -> tuple[pd.DataFrame, int]:
    if selected.empty or "is_board20" not in selected.columns:
        return selected, 0
    chosen = selected.copy()
    chosen["stock_id"] = chosen["stock_id"].astype(str).str.zfill(6)
    board_mask = chosen["is_board20"].astype(int).astype(bool)
    board_count = int(board_mask.sum())
    if board_count <= cap:
        return chosen, 0

    board = chosen[board_mask].copy()
    board["_board_keep_score"] = (
        pd.to_numeric(board.get("source_rrf_pct", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(board.get("aggressive_score_pct", 0.0), errors="coerce").fillna(0.0)
    )
    keep_board_ids = set(board.sort_values("_board_keep_score", ascending=False).head(cap)["stock_id"])
    keep = chosen[(~board_mask) | chosen["stock_id"].isin(keep_board_ids)].copy()
    need = max(0, 5 - len(keep))
    candidates = branch_candidates.copy()
    candidates["stock_id"] = candidates["stock_id"].astype(str).str.zfill(6)
    used = set(keep["stock_id"])
    non_board = ~candidates.get("is_board20", pd.Series(0, index=candidates.index)).astype(int).astype(bool)
    filler = candidates[(~candidates["stock_id"].isin(used)) & non_board].head(need).copy()
    repaired = pd.concat([keep, filler], ignore_index=True).head(5)
    final_board_count = int(pd.to_numeric(repaired.get("is_board20", 0), errors="coerce").fillna(0).sum()) if len(repaired) else 0
    return repaired, max(0, board_count - final_board_count)


def regime_full_fallback_board_aware_v1(work: pd.DataFrame, flags: int, q: dict, basket: dict) -> tuple[str, pd.DataFrame, dict]:
    would_fallback = flags >= 3
    trend_ok, overheat = board20_state(basket)
    mainly_raw = bool(q.get("fallback_reason_mainly_raw_vol_sigma_amp", False))
    protection_applied = bool(would_fallback and trend_ok and (not overheat) and mainly_raw)

    if not would_fallback or protection_applied:
        branch = "current_aggressive"
        selected = branch_top(work, "current", 5)
        candidates = branch_top(work, "current", 30)
    elif overheat:
        branch = "legal_minrisk_hardened"
        selected = branch_top(work, "legal", 5)
        candidates = branch_top(work, "legal", 30)
    else:
        branch = "union_topn_rrf_lcb"
        selected = branch_top(work, "union", 5)
        candidates = branch_top(work, "union", 30)

    capped, cap_replacements = cap_board20_exposure(selected, candidates, cap=2)
    info = {
        "board20_trend_ok": trend_ok,
        "board20_overheat": overheat,
        "board20_protection_applied": protection_applied,
        "board20_cap_replacements": int(cap_replacements),
    }
    return branch, capped, info


def prepare_windows(raw: pd.DataFrame, oof: pd.DataFrame, anchors: list[pd.Timestamp], run_dir: Path, args):
    rows = []
    for anchor in anchors:
        print(f"[repair] prepare anchor={anchor:%Y-%m-%d}")
        artifacts = run_predict_for_anchor(raw, anchor, run_dir, args.model_dir, not args.no_cache)
        realized, score_window = realized_returns_for_anchor(raw, anchor, label_horizon=args.label_horizon)
        score_df = pd.read_csv(artifacts["score_df_path"], dtype={"stock_id": str})
        work = enrich_work(raw, anchor, score_df, oof)
        work = work.merge(realized[["stock_id", "realized_ret"]], on="stock_id", how="left")
        work["realized_ret"] = work["realized_ret"].fillna(0.0)
        current = select_current_top5(work)
        before_score, current = score_selection(current, realized)
        basket = basket_row(raw, anchor, score_window, current, before_score, work)
        rows.append({
            "anchor": anchor,
            "score_window": score_window,
            "work": work,
            "current_selected": current,
            "basket": basket,
            "before_score": before_score,
        })
    return rows


def run_eval(prepared: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    window_rows = []
    detail_rows = []
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

    # Fixed full fallback pass with the same expanding basket history.
    for item in prepared:
        flags, q = basket_risk_flags(item["basket"], history, meta_min_threshold_q=0.20)
        branch, selected = full_fallback(item["work"], flags)
        score = selected_score(selected)
        row = {
            "strategy": "full_fallback_fixed",
            "anchor_date": item["anchor"].strftime("%Y-%m-%d"),
            "score_window": item["score_window"],
            "before_score": item["before_score"],
            "after_score": score,
            "chosen_branch": branch,
            "basket_risk_flags": flags,
            "repair_k": np.nan,
            "basket_risk_threshold": np.nan,
            "stock_veto_threshold": np.nan,
            "meta_min_threshold_q": 0.20,
            "need_repair": flags >= 3,
            "replacements": int(branch != "current_aggressive") * 5,
            "replaced_ids": "",
            **q,
        }
        window_rows.append(row)
        detail = selected.copy()
        detail["strategy"] = "full_fallback_fixed"
        detail["anchor_date"] = row["anchor_date"]
        detail["score_window"] = item["score_window"]
        detail["after_score"] = score
        detail_rows.append(detail)
        update_history(history, item["basket"])

    history = {k: [] for k in history}
    for item in prepared:
        flags, q = basket_risk_flags(item["basket"], history, meta_min_threshold_q=0.20)
        branch, selected, board_info = regime_full_fallback_board_aware_v1(item["work"], flags, q, item["basket"])
        score = selected_score(selected)
        row = {
            "strategy": "regime_full_fallback_board_aware_v1",
            "anchor_date": item["anchor"].strftime("%Y-%m-%d"),
            "score_window": item["score_window"],
            "before_score": item["before_score"],
            "after_score": score,
            "chosen_branch": branch,
            "basket_risk_flags": flags,
            "repair_k": np.nan,
            "basket_risk_threshold": np.nan,
            "stock_veto_threshold": np.nan,
            "meta_min_threshold_q": 0.20,
            "need_repair": flags >= 3,
            "replacements": int(branch != "current_aggressive") * 5 + int(board_info["board20_cap_replacements"]),
            "replaced_ids": "",
            **q,
            **board_info,
        }
        window_rows.append(row)
        detail = selected.copy()
        detail["strategy"] = "regime_full_fallback_board_aware_v1"
        detail["anchor_date"] = row["anchor_date"]
        detail["score_window"] = item["score_window"]
        detail["after_score"] = score
        detail_rows.append(detail)
        update_history(history, item["basket"])

    for repair_k, basket_threshold, veto_threshold, meta_q in product(
        REPAIR_K_GRID,
        BASKET_RISK_THRESHOLD_GRID,
        STOCK_VETO_THRESHOLD_GRID,
        META_MIN_THRESHOLD_Q_GRID,
    ):
        history = {k: [] for k in history}
        strategy = f"partial_k{repair_k}_risk{basket_threshold}_veto{veto_threshold:.2f}_meta{meta_q:.2f}"
        for item in prepared:
            flags, q = basket_risk_flags(item["basket"], history, meta_min_threshold_q=meta_q)
            need_repair = flags >= int(basket_threshold)
            selected, replacements, replaced_ids = partial_repair(
                item["work"],
                item["current_selected"],
                need_repair=need_repair,
                repair_k=int(repair_k),
                veto_threshold=float(veto_threshold),
            )
            score = selected_score(selected)
            row = {
                "strategy": strategy,
                "anchor_date": item["anchor"].strftime("%Y-%m-%d"),
                "score_window": item["score_window"],
                "before_score": item["before_score"],
                "after_score": score,
                "chosen_branch": "partial_repair",
                "basket_risk_flags": flags,
                "repair_k": int(repair_k),
                "basket_risk_threshold": int(basket_threshold),
                "stock_veto_threshold": float(veto_threshold),
                "meta_min_threshold_q": float(meta_q),
                "need_repair": bool(need_repair),
                "replacements": int(replacements),
                "replaced_ids": replaced_ids,
                **q,
            }
            window_rows.append(row)
            detail = selected.copy()
            detail["strategy"] = strategy
            detail["anchor_date"] = row["anchor_date"]
            detail["score_window"] = item["score_window"]
            detail["after_score"] = score
            detail_rows.append(detail)
            update_history(history, item["basket"])

    windows = pd.DataFrame(window_rows)
    details = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    summary = summarize(windows)
    return windows, details, summary


def summarize(windows: pd.DataFrame) -> dict:
    rows = []
    for strategy, group in windows.groupby("strategy", sort=False):
        before = pd.to_numeric(group["before_score"], errors="coerce")
        after = pd.to_numeric(group["after_score"], errors="coerce")
        repaired_mask = group["replacements"].fillna(0).astype(float) > 0
        crash_mask = before <= -0.05
        false_positive = (before - after).where((before > after) & repaired_mask, 0.0)
        row = {
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
            "windows_repaired": int(repaired_mask.sum()),
            "avg_replacements_per_repaired_window": float(group.loc[repaired_mask, "replacements"].astype(float).mean()) if repaired_mask.any() else 0.0,
            "false_positive_cost_mean": float(false_positive.mean()),
            "false_positive_cost_sum": float(false_positive.sum()),
            "crash_capture": int(((after > before) & crash_mask).sum()),
        }
        rows.append(row)
    summary_df = pd.DataFrame(rows).sort_values(["mean", "q10", "worst"], ascending=[False, False, False])
    return {
        "summary": json.loads(summary_df.to_json(orient="records")),
        "best_by_mean": summary_df.iloc[0].to_dict() if len(summary_df) else {},
        "best_partial_by_mean": summary_df[summary_df["strategy"].str.startswith("partial_")].iloc[0].to_dict() if summary_df["strategy"].str.startswith("partial_").any() else {},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/train_hs300_20260424.csv")
    parser.add_argument("--oof-path", default="temp/oof_base_scores/exp00305_oof_lgb_last20/oof_lgb_scores.csv")
    parser.add_argument("--out-dir", default="temp/basket_veto_repair")
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
    anchors = [pd.Timestamp(x) for x in parse_anchor_args(args, dates)]
    run_dir = ROOT / args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    oof = load_oof_scores(ROOT / args.oof_path if args.oof_path else None)
    prepared = prepare_windows(raw, oof, anchors, run_dir, args)
    windows, details, summary = run_eval(prepared)

    windows.to_csv(run_dir / "repair_window_results.csv", index=False)
    details.to_csv(run_dir / "repair_selected_details.csv", index=False)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if windows["anchor_date"].eq("2026-03-16").any():
        windows[windows["anchor_date"].eq("2026-03-16")].to_csv(run_dir / "before_after_20260316.csv", index=False)
    latest_anchor = windows["anchor_date"].max() if len(windows) else ""
    if latest_anchor:
        windows[windows["anchor_date"].eq(latest_anchor)].to_csv(run_dir / "before_after_latest.csv", index=False)

    print(json.dumps({
        "best_by_mean": summary.get("best_by_mean", {}),
        "best_partial_by_mean": summary.get("best_partial_by_mean", {}),
        "output": str(run_dir),
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
