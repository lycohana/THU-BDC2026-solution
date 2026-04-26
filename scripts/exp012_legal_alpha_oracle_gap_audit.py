from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP008_DIR = ROOT / "temp" / "batch_window_analysis" / "exp008_tri_state_tight_last10_to_20260424"
DEFAULT_OUT_DIR = ROOT / "outputs" / "exp012_legal_alpha_oracle_gap_audit"
DEFAULT_BASELINE_OFFLINE = ROOT / "temp" / "batch_window_analysis" / "exp008_vs_baseline_last10_to_20260424.csv"

BAD_RET = -0.03
VERY_BAD_RET = -0.05

REQUESTED_LEGAL_BRANCHES = {
    "independent_union_rerank",
    "legal_minrisk_hardened",
    "conservative_softrisk_v2",
    "conservative_softrisk_v2_strict",
    "lgb_only_guarded",
    "balanced_guarded",
    "balanced_blend",
    "defensive_v2_strict",
    "legal_minrisk",
    "stable_top30_rerank_trend",
    "union_topn_rrf_lcb",
    "legal_plus_1alpha",
    "safe_union_2slot",
    "legal_plus_1alpha_shadow",
    "safe_union_2slot_shadow",
}

EXCLUDED_OFFLINE_OR_NONLEGAL_BRANCH_TOKENS = (
    "baseline",
    "reference",
    "current_aggressive",
    "trend_uncluttered",
    "ai_hardware",
)

FOCUS_WINDOWS = {
    "2026-02-13",
    "2026-03-16",
    "2026-03-30",
    "2026-04-07",
    "2026-04-14",
    "2026-04-17",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="exp012 legal alpha oracle gap audit.")
    parser.add_argument("--train", default=str(ROOT / "data" / "train.csv"))
    parser.add_argument("--test", default=str(ROOT / "data" / "test.csv"))
    parser.add_argument("--exp008-dir", default=str(DEFAULT_EXP008_DIR))
    parser.add_argument("--baseline-offline", default=str(DEFAULT_BASELINE_OFFLINE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def normalize_stock_id(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.extract(r"(\d+)")[0]
        .str.zfill(6)
    )


def read_raw_market(train_path: str | Path, test_path: str | Path) -> pd.DataFrame:
    frames = []
    for path in [train_path, test_path]:
        df = pd.read_csv(path, dtype={"股票代码": str})
        df["股票代码"] = normalize_stock_id(df["股票代码"])
        df["日期"] = pd.to_datetime(df["日期"])
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    return raw.sort_values(["日期", "股票代码"]).reset_index(drop=True)


def realized_returns_for_asof(raw: pd.DataFrame, asof: pd.Timestamp, horizon: int = 5) -> tuple[pd.Series, str]:
    future_dates = pd.DatetimeIndex(sorted(raw.loc[raw["日期"] > asof, "日期"].unique()))
    if len(future_dates) < horizon:
        raise ValueError(f"not enough future dates after {asof.date()} for {horizon}-day audit window")
    window_dates = future_dates[:horizon]
    future = raw[raw["日期"].isin(window_dates)].sort_values(["股票代码", "日期"])
    returns: dict[str, float] = {}
    for stock_id, group in future.groupby("股票代码", sort=False):
        if len(group) < 2:
            returns[str(stock_id).zfill(6)] = 0.0
            continue
        start = float(group.iloc[0]["开盘"])
        end = float(group.iloc[-1]["开盘"])
        returns[str(stock_id).zfill(6)] = end / (start + 1e-12) - 1.0
    score_window = f"{window_dates[0].date()}~{window_dates[-1].date()}"
    return pd.Series(returns, dtype=float), score_window


def is_legal_branch(branch: object) -> bool:
    name = str(branch)
    lowered = name.lower()
    if any(token in lowered for token in EXCLUDED_OFFLINE_OR_NONLEGAL_BRANCH_TOKENS):
        return False
    return name in REQUESTED_LEGAL_BRANCHES or not lowered.startswith("_")


def split_stock_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip().zfill(6) for part in str(value).split(",") if part.strip()]


def load_exp008_summary(exp008_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(exp008_dir / "window_summary.csv")
    summary["anchor_date"] = pd.to_datetime(summary["anchor_date"])
    return summary.tail(10).reset_index(drop=True)


def load_branch_rows(exp008_dir: Path) -> pd.DataFrame:
    path = exp008_dir / "branch_diagnostics.csv"
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path)
    if "anchor_date" in rows.columns:
        rows["anchor_date"] = pd.to_datetime(rows["anchor_date"])
    if "branch" in rows.columns:
        rows = rows[rows["branch"].map(is_legal_branch)].copy()
    return rows


def load_selector_rows(window_dir: Path) -> pd.DataFrame:
    path = window_dir / "selector_diagnostics.csv"
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path)
    if "branch" not in rows.columns:
        return pd.DataFrame()
    rows = rows[rows["branch"].map(is_legal_branch)].copy()
    if "available" in rows.columns:
        rows = rows[rows["available"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    return rows


def load_score_frame(window_dir: Path) -> pd.DataFrame:
    for filename in ["predict_score_df.csv", "predict_filtered_top30.csv"]:
        path = window_dir / filename
        if path.exists():
            df = pd.read_csv(path, dtype={"stock_id": str})
            if "stock_id" in df.columns:
                df["stock_id"] = normalize_stock_id(df["stock_id"])
            return df
    return pd.DataFrame(columns=["stock_id"])


def load_exp008_result(window_dir: Path, summary_row: pd.Series) -> list[str]:
    path = window_dir / "result.csv"
    if path.exists():
        result = pd.read_csv(path, dtype={"stock_id": str})
        if "stock_id" in result.columns:
            return normalize_stock_id(result["stock_id"]).head(5).tolist()
    return split_stock_list(summary_row.get("selected_picks", ""))


def load_baseline_offline_reference(path: str | Path) -> dict[str, float]:
    offline_path = Path(path)
    if not offline_path.exists():
        return {}
    df = pd.read_csv(offline_path)
    if "anchor_date" not in df.columns:
        return {}
    score_col = None
    for candidate in ["baseline_score_offline_reference", "baseline", "score", "selected_score"]:
        if candidate in df.columns:
            score_col = candidate
            break
    if score_col is None:
        return {}
    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.strftime("%Y-%m-%d")
    return {
        str(row.anchor_date): float(row._asdict()[score_col])
        for row in df[["anchor_date", score_col]].itertuples(index=False)
        if pd.notna(getattr(row, score_col))
    }


def stock_metadata(score_df: pd.DataFrame, stock_id: str) -> dict[str, float | None]:
    if score_df.empty or "stock_id" not in score_df.columns:
        return {
            "alpha_score_if_available": None,
            "risk_score_if_available": None,
            "consensus_score_if_available": None,
        }
    row = score_df[score_df["stock_id"] == stock_id]
    if row.empty:
        return {
            "alpha_score_if_available": None,
            "risk_score_if_available": None,
            "consensus_score_if_available": None,
        }
    rec = row.iloc[0]

    def first_numeric(cols: Iterable[str]) -> float | None:
        for col in cols:
            if col in rec.index and pd.notna(rec[col]):
                value = pd.to_numeric(pd.Series([rec[col]]), errors="coerce").iloc[0]
                if pd.notna(value):
                    return float(value)
        return None

    return {
        "alpha_score_if_available": first_numeric(["rerank_score", "score", "score_lgb_only", "score_balanced"]),
        "risk_score_if_available": first_numeric(["tail_risk_score", "bad_pick_risk", "uncertainty_score"]),
        "consensus_score_if_available": first_numeric(
            ["consensus_score", "consensus_count", "source_count", "_union_rrf_lcb_score"]
        ),
    }


def add_branch_pick(
    appearances: dict[str, list[tuple[str, int]]],
    branch: str,
    picks: list[str],
    top_k: int,
) -> None:
    for rank, stock_id in enumerate(picks[:top_k], start=1):
        appearances.setdefault(stock_id, []).append((branch, rank))


def collect_branch_appearances(
    exp008_dir: Path,
    asof: pd.Timestamp,
    branch_rows: pd.DataFrame,
    top_k: int,
) -> tuple[dict[str, list[tuple[str, int]]], dict[str, dict[str, object]]]:
    appearances: dict[str, list[tuple[str, int]]] = {}
    branch_scores: dict[str, dict[str, object]] = {}
    date_key = asof.strftime("%Y-%m-%d")

    if not branch_rows.empty:
        rows = branch_rows[branch_rows["anchor_date"].dt.strftime("%Y-%m-%d") == date_key]
        for row in rows.itertuples(index=False):
            branch = str(getattr(row, "branch"))
            picks = split_stock_list(getattr(row, "picks", ""))
            add_branch_pick(appearances, branch, picks, top_k)
            branch_scores[branch] = {
                "score": float(getattr(row, "score")) if pd.notna(getattr(row, "score", np.nan)) else np.nan,
                "picks": picks[:top_k],
            }

    selector_rows = load_selector_rows(exp008_dir / asof.strftime("%Y%m%d"))
    for row in selector_rows.itertuples(index=False):
        branch = str(getattr(row, "branch"))
        picks = split_stock_list(getattr(row, "top5", ""))
        if not picks:
            continue
        add_branch_pick(appearances, branch, picks, top_k)
        branch_scores.setdefault(branch, {"score": np.nan, "picks": picks[:top_k]})

    return appearances, branch_scores


def score_equal_weight(picks: list[str], future_returns: pd.Series, top_k: int) -> float:
    selected = picks[:top_k]
    if not selected:
        return 0.0
    return float(future_returns.reindex(selected).fillna(0.0).mean())


def candidate_oracle_score(pool: list[str], future_returns: pd.Series, top_k: int) -> tuple[float, list[str]]:
    scored = future_returns.reindex(pool).dropna().sort_values(ascending=False)
    picks = scored.head(top_k).index.astype(str).tolist()
    return score_equal_weight(picks, future_returns, top_k), picks


def classify_diagnosis(
    exp008_score: float,
    branch_oracle_score: float,
    candidate_score: float,
    baseline: float | None,
) -> str:
    if baseline is None or pd.isna(baseline):
        if candidate_score <= exp008_score:
            return "legal_candidate_pool_alpha_insufficient_vs_exp008"
        if branch_oracle_score <= exp008_score:
            return "candidate_pool_has_alpha_but_branch_outputs_fail_vs_exp008"
        return "oracle_above_exp008_no_baseline_reference"
    if candidate_score <= baseline:
        return "legal_candidate_pool_alpha_insufficient"
    if candidate_score > baseline and branch_oracle_score <= baseline:
        return "candidate_pool_has_alpha_but_branch_outputs_fail"
    if branch_oracle_score > baseline and exp008_score <= baseline:
        return "branch_gate_failure"
    if abs(exp008_score - branch_oracle_score) <= 0.002 and exp008_score < baseline:
        return "need_new_alpha_source_not_gate"
    if exp008_score >= baseline:
        return "exp008_above_baseline_offline_reference"
    return "oracle_gap_unclassified"


def aggregate_conclusion(exp008_mean: float, branch_mean: float, candidate_mean: float, baseline_mean: float | None) -> str:
    if baseline_mean is None or pd.isna(baseline_mean):
        if candidate_mean <= exp008_mean:
            return "无 baseline 离线对照；当前候选池 oracle 未超过 exp008，优先检查合法候选池 alpha。"
        if branch_mean <= exp008_mean:
            return "无 baseline 离线对照；候选池 oracle 高于 exp008 但 branch oracle 不高，分支输出偏窄。"
        return "无 baseline 离线对照；branch oracle 高于 exp008，branch gate 可能有诊断价值。"
    if candidate_mean <= baseline_mean:
        return "当前合法候选池 alpha 不足，必须新增 alpha 源/模型。"
    if candidate_mean > baseline_mean and branch_mean <= baseline_mean:
        return "分支输出太窄，需要扩大候选池后重新 rerank。"
    if branch_mean > baseline_mean and exp008_mean <= baseline_mean:
        return "branch gate 有价值。"
    if abs(exp008_mean - branch_mean) <= 0.002 and exp008_mean < baseline_mean:
        return "不是 gate 问题，继续调 selector 浪费时间。"
    if exp008_mean >= baseline_mean:
        return "exp008 均值已高于 baseline 离线对照；本轮报告仅保留 oracle gap 诊断。"
    return "oracle 关系不完全落入预设规则，建议逐窗阅读 oracle_gap_summary.csv。"


def build_outputs(
    *,
    train_path: str | Path,
    test_path: str | Path,
    exp008_dir: str | Path,
    baseline_offline_path: str | Path,
    out_dir: str | Path,
    top_k: int = 5,
) -> dict[str, object]:
    exp008_dir = Path(exp008_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_raw_market(train_path, test_path)
    summary = load_exp008_summary(exp008_dir)
    branch_rows = load_branch_rows(exp008_dir)
    baseline_map = load_baseline_offline_reference(baseline_offline_path)

    matrix_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    ranker_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for summary_row in summary.itertuples(index=False):
        asof = pd.Timestamp(getattr(summary_row, "anchor_date"))
        asof_str = asof.strftime("%Y-%m-%d")
        window_dir = exp008_dir / asof.strftime("%Y%m%d")
        future_returns, score_window = realized_returns_for_asof(raw, asof, horizon=top_k)
        score_df = load_score_frame(window_dir)
        exp008_picks = load_exp008_result(window_dir, pd.Series(summary_row._asdict()))
        exp008_score = float(getattr(summary_row, "selected_score"))
        selected_set = set(exp008_picks)

        appearances, branch_scores = collect_branch_appearances(exp008_dir, asof, branch_rows, top_k)
        candidate_pool = sorted(appearances.keys())
        universe_future_top5 = set(future_returns.sort_values(ascending=False).head(top_k).index.astype(str))
        pool_future = future_returns.reindex(candidate_pool).dropna()
        pool_future_top5 = set(pool_future.sort_values(ascending=False).head(top_k).index.astype(str))

        for stock_id in candidate_pool:
            ranks = [rank for _, rank in appearances[stock_id]]
            branches = [branch for branch, _ in appearances[stock_id]]
            meta = stock_metadata(score_df, stock_id)
            future_ret = float(future_returns.get(stock_id, np.nan))
            matrix_rows.append(
                {
                    "asof": asof_str,
                    "score_window": score_window,
                    "stock_id": stock_id,
                    "appeared_in_branches": ";".join(branches),
                    "branch_count": len(set(branches)),
                    "max_branch_rank": int(max(ranks)) if ranks else np.nan,
                    "min_branch_rank": int(min(ranks)) if ranks else np.nan,
                    "mean_branch_rank": float(np.mean(ranks)) if ranks else np.nan,
                    **meta,
                    "selected_by_exp008": stock_id in selected_set,
                    "future_return_5d": future_ret,
                    "is_future_top5_in_pool": stock_id in universe_future_top5,
                    "is_future_positive": bool(future_ret > 0) if pd.notna(future_ret) else False,
                    "is_bad": bool(future_ret <= BAD_RET) if pd.notna(future_ret) else False,
                }
            )

        scored_branches = {}
        for branch, info in branch_scores.items():
            score = info.get("score", np.nan)
            if pd.isna(score):
                score = score_equal_weight(list(info.get("picks", [])), future_returns, top_k)
            scored_branches[branch] = float(score)
        if scored_branches:
            best_branch = max(scored_branches, key=scored_branches.get)
            branch_oracle = float(scored_branches[best_branch])
            exp008_rank = 1 + sum(score > exp008_score for score in scored_branches.values())
        else:
            best_branch = ""
            branch_oracle = exp008_score
            exp008_rank = 1

        candidate_oracle, candidate_picks = candidate_oracle_score(candidate_pool, future_returns, top_k)
        baseline = baseline_map.get(asof_str)
        oracle_rows.append(
            {
                "asof": asof_str,
                "exp008_score": exp008_score,
                "branch_oracle_score": branch_oracle,
                "candidate_oracle_score": candidate_oracle,
                "branch_oracle_gap": branch_oracle - exp008_score,
                "candidate_oracle_gap": candidate_oracle - exp008_score,
                "exp008_rank_vs_branches": int(exp008_rank),
                "candidate_pool_size": int(len(candidate_pool)),
                "future_top5_coverage": float(len(universe_future_top5 & set(candidate_pool)) / max(top_k, 1)),
                "selected_future_return_mean": score_equal_weight(exp008_picks, future_returns, top_k),
                "pool_future_return_q90": float(pool_future.quantile(0.90)) if len(pool_future) else np.nan,
                "pool_future_return_q95": float(pool_future.quantile(0.95)) if len(pool_future) else np.nan,
                "baseline_score_offline_reference": baseline,
                "gap_vs_baseline_offline_reference": None if baseline is None else candidate_oracle - baseline,
                "diagnosis": classify_diagnosis(exp008_score, branch_oracle, candidate_oracle, baseline),
            }
        )

        if not score_df.empty and "stock_id" in score_df.columns:
            ranked = score_df.copy()
            ranked["stock_id"] = normalize_stock_id(ranked["stock_id"])
            ranked["future_return_5d"] = ranked["stock_id"].map(future_returns).astype(float)
            alpha_col = "rerank_score" if "rerank_score" in ranked.columns else "score" if "score" in ranked.columns else None
            if alpha_col is not None:
                ranked["_alpha"] = pd.to_numeric(ranked[alpha_col], errors="coerce")
                valid = ranked[["_alpha", "future_return_5d"]].dropna()
                alpha_sorted = ranked.sort_values("_alpha", ascending=False)
                top = lambda n: alpha_sorted.head(min(n, len(alpha_sorted)))["future_return_5d"].mean()
                bottom20 = alpha_sorted.tail(min(20, len(alpha_sorted)))["future_return_5d"].mean()
                alpha_top20 = set(alpha_sorted.head(min(20, len(alpha_sorted)))["stock_id"])
                ranker_rows.append(
                    {
                        "asof": asof_str,
                        "spearman_ic": float(valid["_alpha"].rank().corr(valid["future_return_5d"].rank()))
                        if len(valid) >= 2
                        else np.nan,
                        "pearson_ic": float(valid["_alpha"].corr(valid["future_return_5d"])) if len(valid) >= 2 else np.nan,
                        "top5_mean_future_return": float(top(5)) if len(alpha_sorted) else np.nan,
                        "top10_mean_future_return": float(top(10)) if len(alpha_sorted) else np.nan,
                        "top20_mean_future_return": float(top(20)) if len(alpha_sorted) else np.nan,
                        "bottom20_mean_future_return": float(bottom20) if len(alpha_sorted) else np.nan,
                        "hit_rate_positive_top5": float(alpha_sorted.head(5)["future_return_5d"].gt(0).mean())
                        if len(alpha_sorted)
                        else np.nan,
                        "hit_rate_future_top5_in_alpha_top20": float(len(universe_future_top5 & alpha_top20) / max(top_k, 1)),
                        "bad_count_top5": int(alpha_sorted.head(5)["future_return_5d"].le(BAD_RET).sum()),
                        "very_bad_count_top5": int(alpha_sorted.head(5)["future_return_5d"].le(VERY_BAD_RET).sum()),
                    }
                )

        if asof_str in FOCUS_WINDOWS:
            missed = [stock for stock in candidate_picks if stock not in selected_set]
            missed_rank = []
            for stock in missed:
                ranks = [rank for _, rank in appearances.get(stock, [])]
                missed_rank.append(f"{stock}:{min(ranks) if ranks else 'NA'}")
            if not missed:
                reason = "exp008_selected_candidate_oracle_top5"
            elif branch_oracle <= exp008_score + 1e-12:
                reason = "candidate_pool_oracle_gap_not_represented_by_branch_top5"
            elif best_branch != getattr(summary_row, "chosen_branch", ""):
                reason = "best_branch_not_chosen_by_exp008_gate"
            else:
                reason = "rerank_missed_higher_future_return_candidates"
            failure_rows.append(
                {
                    "asof": asof_str,
                    "exp008_selected": ",".join(exp008_picks),
                    "exp008_score": exp008_score,
                    "best_branch": best_branch,
                    "best_branch_score": branch_oracle,
                    "candidate_oracle_score": candidate_oracle,
                    "missed_best_stocks": ",".join(missed),
                    "missed_best_stocks_appeared_in_pool": ",".join(
                        f"{stock}:{stock in appearances}" for stock in missed
                    ),
                    "missed_best_stocks_rank": ",".join(missed_rank),
                    "reason": reason,
                }
            )

    matrix = pd.DataFrame(matrix_rows)
    oracle = pd.DataFrame(oracle_rows)
    ranker = pd.DataFrame(ranker_rows)
    failure = pd.DataFrame(failure_rows)

    matrix.to_csv(out_dir / "candidate_pool_matrix.csv", index=False)
    oracle.to_csv(out_dir / "oracle_gap_summary.csv", index=False)
    ranker.to_csv(out_dir / "ranker_quality.csv", index=False)
    failure.to_csv(out_dir / "window_failure_notes.csv", index=False)

    baseline_mean = (
        float(oracle["baseline_score_offline_reference"].dropna().mean())
        if "baseline_score_offline_reference" in oracle.columns and oracle["baseline_score_offline_reference"].notna().any()
        else None
    )
    exp008_mean = float(oracle["exp008_score"].mean())
    branch_mean = float(oracle["branch_oracle_score"].mean())
    candidate_mean = float(oracle["candidate_oracle_score"].mean())
    aggregate = {
        "exp008_mean": exp008_mean,
        "branch_oracle_mean": branch_mean,
        "candidate_oracle_mean": candidate_mean,
        "exp008_vs_baseline_offline_mean_gap": None if baseline_mean is None else exp008_mean - baseline_mean,
        "branch_oracle_vs_baseline_offline_gap": None if baseline_mean is None else branch_mean - baseline_mean,
        "candidate_oracle_vs_baseline_offline_gap": None if baseline_mean is None else candidate_mean - baseline_mean,
        "conclusion": aggregate_conclusion(exp008_mean, branch_mean, candidate_mean, baseline_mean),
    }
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    return aggregate


def main() -> None:
    args = parse_args()
    aggregate = build_outputs(
        train_path=args.train,
        test_path=args.test,
        exp008_dir=args.exp008_dir,
        baseline_offline_path=args.baseline_offline,
        out_dir=args.out_dir,
        top_k=args.top_k,
    )
    print(f"wrote diagnostics to: {Path(args.out_dir)}")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
