"""Batch diagnostic scoring across historical 5-day windows.

This script evaluates only independent model/rule selectors:
- no fixed external portfolio
- no comparison target from another project
- no component-substitution logic

Unless --model-dir points to a strictly historical model for each anchor, this is
a diagnostic replay with the current model, not a leakage-free OOF backtest.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))
from config import config as PROJECT_CONFIG  # noqa: E402


def stable_hash(obj: object) -> str:
    text = json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


SELECTOR_CFG = PROJECT_CONFIG.get("selector", {})
SELECTOR_VERSION = str(SELECTOR_CFG.get("version", "unknown"))
CONFIG_HASH = stable_hash(PROJECT_CONFIG)
GATE_HASH = stable_hash({
    "selector": SELECTOR_CFG,
    "predict_cache_version": PROJECT_CONFIG.get("predict", {}).get("use_cache", True),
})


def normalize_stock_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.extract(r"(\d+)")[0]
        .str.zfill(6)
    )


def load_raw(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={"股票代码": str})
    raw["股票代码"] = normalize_stock_id(raw["股票代码"])
    raw["日期"] = pd.to_datetime(raw["日期"])
    return raw.sort_values(["日期", "股票代码"]).reset_index(drop=True)


def parse_anchor_args(args: argparse.Namespace, dates: list[pd.Timestamp]) -> list[pd.Timestamp]:
    label_horizon = int(args.label_horizon)
    min_idx = int(args.min_history_days) - 1
    max_idx = len(dates) - 1 - label_horizon
    allowed = dates[min_idx : max_idx + 1]

    if args.anchors:
        requested = [pd.Timestamp(x.strip()) for x in args.anchors.split(",") if x.strip()]
        allowed_set = set(allowed)
        anchors = [d for d in requested if d in allowed_set]
        missing = [d.strftime("%Y-%m-%d") for d in requested if d not in allowed_set]
        if missing:
            print(f"[batch] skipped anchors without enough history/future label: {missing}")
        return anchors

    filtered = allowed
    if args.start_anchor:
        filtered = [d for d in filtered if d >= pd.Timestamp(args.start_anchor)]
    if args.end_anchor:
        filtered = [d for d in filtered if d <= pd.Timestamp(args.end_anchor)]
    filtered = filtered[:: int(args.step)]
    if args.last_n:
        filtered = filtered[-int(args.last_n) :]
    return filtered


def realized_returns_for_anchor(raw: pd.DataFrame, anchor: pd.Timestamp, label_horizon: int = 5) -> tuple[pd.DataFrame, str]:
    dates = sorted(raw["日期"].dropna().unique())
    idx = dates.index(anchor)
    d1 = pd.Timestamp(dates[idx + 1])
    d5 = pd.Timestamp(dates[idx + label_horizon])

    open1 = raw[raw["日期"] == d1][["股票代码", "开盘"]].rename(
        columns={"股票代码": "stock_id", "开盘": "open_day1"}
    )
    open5 = raw[raw["日期"] == d5][["股票代码", "开盘"]].rename(
        columns={"股票代码": "stock_id", "开盘": "open_day5"}
    )
    ret = open1.merge(open5, on="stock_id", how="inner")
    ret["realized_ret"] = pd.to_numeric(ret["open_day5"], errors="coerce") / (
        pd.to_numeric(ret["open_day1"], errors="coerce") + 1e-12
    ) - 1.0
    return ret[["stock_id", "open_day1", "open_day5", "realized_ret"]], f"{d1:%Y-%m-%d}~{d5:%Y-%m-%d}"


def score_result(result_path: Path, realized: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    result = pd.read_csv(result_path, dtype={"stock_id": str})
    result["stock_id"] = normalize_stock_id(result["stock_id"])
    result["weight"] = pd.to_numeric(result["weight"], errors="coerce").fillna(0.0)
    detail = result.merge(realized, on="stock_id", how="left")
    detail["realized_ret"] = detail["realized_ret"].fillna(0.0)
    detail["contribution"] = detail["weight"] * detail["realized_ret"]
    return float(detail["contribution"].sum()), detail


def rank_pct(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    fill = values.median() if values.notna().any() else 0.0
    return values.fillna(fill).rank(pct=True, method="average")


def add_branch_diagnostic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "lgb_norm" not in out.columns and "lgb" in out.columns:
        out["lgb_norm"] = rank_pct(out["lgb"])
    if "tf_norm" not in out.columns and "transformer" in out.columns:
        out["tf_norm"] = rank_pct(out["transformer"])
    if "rank_disagreement" not in out.columns and {"lgb_norm", "tf_norm"}.issubset(out.columns):
        out["rank_disagreement"] = (out["lgb_norm"] - out["tf_norm"]).abs()

    rank_sources = {
        "sigma_rank": "sigma20",
        "amp_rank": "amp20",
        "ret1_rank": "ret1",
        "ret5_rank": "ret5",
        "liq_rank": "median_amount20",
        "downside_beta60_rank": "downside_beta60",
        "max_drawdown20_rank": "max_drawdown20",
    }
    for rank_col, source_col in rank_sources.items():
        if rank_col not in out.columns:
            out[rank_col] = rank_pct(out[source_col]) if source_col in out.columns else 0.0

    if "tail_risk_flag" not in out.columns:
        out["tail_risk_flag"] = (
            (out["sigma_rank"] > 0.85)
            | (out["amp_rank"] > 0.85)
            | (out["downside_beta60_rank"] > 0.85)
            | (out["max_drawdown20_rank"] > 0.85)
        )
    if "overheat_flag" not in out.columns:
        out["overheat_flag"] = (out["ret5_rank"] > 0.75) & (out["amp_rank"] > 0.75)
    if "reversal_flag" not in out.columns:
        out["reversal_flag"] = (out["ret5_rank"] > 0.70) & (out["ret1_rank"] < 0.30)

    if "score_conservative_softrisk_v2" not in out.columns and "score_conservative" in out.columns:
        vol_hinge = ((out["sigma_rank"] - 0.80) / 0.20).clip(lower=0.0)
        amp_hinge = ((out["amp_rank"] - 0.80) / 0.20).clip(lower=0.0)
        reversal_penalty = (
            ((out["ret5_rank"] - 0.70) / 0.30).clip(lower=0.0)
            * ((0.35 - out["ret1_rank"]) / 0.35).clip(lower=0.0)
        )
        out["score_conservative_softrisk_v2"] = (
            out["score_conservative"]
            - 0.06 * (0.5 * vol_hinge + 0.5 * amp_hinge) * out["rank_disagreement"]
            - 0.04 * reversal_penalty
        )

    if "score_legal_minrisk" not in out.columns:
        out["score_legal_minrisk"] = (
            0.35 * out.get("tf_norm", 0.0)
            + 0.15 * out.get("lgb_norm", 0.0)
            + 0.20 * out.get("liq_rank", 0.0)
            - 0.15 * out.get("sigma_rank", 0.0)
            - 0.10 * out.get("downside_beta60_rank", 0.0)
            - 0.10 * out.get("max_drawdown20_rank", 0.0)
            - 0.05 * out.get("amp_rank", 0.0)
        )
    if "tail_risk_score" not in out.columns:
        out["tail_risk_score"] = (
            0.20 * out["sigma_rank"]
            + 0.20 * out["amp_rank"]
            + 0.25 * out["downside_beta60_rank"]
            + 0.20 * out["max_drawdown20_rank"]
            + 0.15 * out["max_drawdown20_rank"]
        )
    if "uncertainty_score" not in out.columns:
        out["uncertainty_score"] = (
            0.50 * out.get("rank_disagreement", 0.5)
            + 0.30 * 0.5
            + 0.20 * 0.5
        )
    return out


def infer_regime(work: pd.DataFrame) -> str:
    breadth_1d = work["ret1"].gt(0).mean()
    breadth_5d = work["ret5"].gt(0).mean()
    median_ret1 = work["ret1"].median()
    median_ret5 = work["ret5"].median()
    if breadth_1d >= 0.58 and breadth_5d >= 0.55 and median_ret1 > 0 and median_ret5 > 0.005:
        return "risk_on_strict"
    if breadth_1d >= 0.50 and breadth_5d >= 0.50 and median_ret5 > 0:
        return "neutral_positive"
    if breadth_1d < 0.35 or median_ret1 < -0.005:
        return "risk_off"
    return "mixed_defensive"


def score_gap(work: pd.DataFrame, score_col: str) -> float:
    scores = work.sort_values(score_col, ascending=False)[score_col].reset_index(drop=True)
    if len(scores) <= 5:
        return 0.0
    return float(scores.iloc[:5].mean() - scores.iloc[5:min(20, len(scores))].mean())


def score_gap_5_10(work: pd.DataFrame, score_col: str) -> float:
    scores = work.sort_values(score_col, ascending=False)[score_col].reset_index(drop=True)
    if len(scores) < 10:
        return 0.0
    return float(scores.iloc[:5].mean() - scores.iloc[5:10].mean())


def top20_overlap(work: pd.DataFrame) -> float:
    if not {"lgb_norm", "tf_norm", "stock_id"}.issubset(work.columns) or len(work) == 0:
        return 0.0
    k = min(20, len(work))
    lgb_top = set(work.nlargest(k, "lgb_norm")["stock_id"])
    tf_top = set(work.nlargest(k, "tf_norm")["stock_id"])
    return float(len(lgb_top & tf_top) / max(k, 1))


def branch_diagnostics(work: pd.DataFrame, score_col: str) -> dict:
    top5 = work.sort_values(score_col, ascending=False).head(5).copy()
    defensive_overlap = 0
    balanced_overlap = 0
    if "score_defensive_v2" in work.columns:
        defensive_top10 = set(work.nlargest(min(10, len(work)), "score_defensive_v2")["stock_id"])
        defensive_overlap = len(set(top5["stock_id"]) & defensive_top10)
    if "score_balanced" in work.columns:
        balanced_top10 = set(work.nlargest(min(10, len(work)), "score_balanced")["stock_id"])
        balanced_overlap = len(set(top5["stock_id"]) & balanced_top10)

    top5_mean_sigma = float(top5.get("sigma_rank", pd.Series(0.0, index=top5.index)).mean())
    top5_max_sigma = float(top5.get("sigma_rank", pd.Series(0.0, index=top5.index)).max())
    top5_mean_disagreement = float(top5.get("rank_disagreement", pd.Series(0.0, index=top5.index)).mean())
    top5_max_downside_beta = float(top5.get("downside_beta60_rank", pd.Series(0.0, index=top5.index)).max())
    top5_max_drawdown = float(top5.get("max_drawdown20_rank", pd.Series(0.0, index=top5.index)).max())
    top5_overheat_count = int(top5.get("overheat_flag", pd.Series(False, index=top5.index)).astype(bool).sum())
    top5_reversal_count = int(top5.get("reversal_flag", pd.Series(False, index=top5.index)).astype(bool).sum())
    branch_risk_score = (
        0.18 * top5_mean_sigma
        + 0.15 * top5_max_sigma
        + 0.15 * top5_mean_disagreement
        + 0.15 * top5_max_downside_beta
        + 0.15 * top5_max_drawdown
        + 0.12 * (top5_overheat_count / 5.0)
        + 0.10 * (top5_reversal_count / 5.0)
    )

    return {
        "score_gap": score_gap(work, score_col),
        "score_gap_top5_vs_6_10": score_gap_5_10(work, score_col),
        "top20_overlap": top20_overlap(work),
        "top5_disagreement": top5_mean_disagreement,
        "top5_mean_disagreement": top5_mean_disagreement,
        "top5_max_disagreement": float(top5.get("rank_disagreement", pd.Series(0.0, index=top5.index)).max()),
        "top5_sigma_rank_mean": top5_mean_sigma,
        "top5_mean_sigma": top5_mean_sigma,
        "top5_max_sigma": top5_max_sigma,
        "top5_amp_rank_mean": float(top5.get("amp_rank", pd.Series(0.0, index=top5.index)).mean()),
        "top5_ret1_rank_mean": float(top5.get("ret1_rank", pd.Series(0.0, index=top5.index)).mean()),
        "top5_ret5_rank_mean": float(top5.get("ret5_rank", pd.Series(0.0, index=top5.index)).mean()),
        "top5_min_liq": float(top5.get("liq_rank", pd.Series(0.0, index=top5.index)).min()),
        "top5_max_drawdown": top5_max_drawdown,
        "top5_max_downside_beta": top5_max_downside_beta,
        "top5_tail_risk_count": int(top5.get("tail_risk_flag", pd.Series(False, index=top5.index)).astype(bool).sum()),
        "top5_overheat_count": top5_overheat_count,
        "top5_reversal_count": top5_reversal_count,
        "branch_risk_score": float(branch_risk_score),
        "consensus_with_defensive": defensive_overlap,
        "consensus_with_balanced": balanced_overlap,
        "defensive_overlap": defensive_overlap,
    }


def filter_branch(work: pd.DataFrame, filter_name: str) -> pd.DataFrame:
    if filter_name in {"none", "nofilter"}:
        return work.copy()
    cond = pd.Series(True, index=work.index)
    if filter_name == "stable":
        cond &= work["median_amount20"] >= work["median_amount20"].quantile(0.10)
        cond &= work["sigma20"] <= work["sigma20"].quantile(0.85)
    elif filter_name == "liquidity_q05":
        cond &= work["median_amount20"] >= work["median_amount20"].quantile(0.05)
    elif filter_name in {"legal_minrisk", "legal_minrisk_hardened"}:
        cond &= ~work["tail_risk_flag"].astype(bool)
        cond &= ~work["reversal_flag"].astype(bool)
        cond &= work["liq_rank"] >= (0.15 if filter_name == "legal_minrisk_hardened" else 0.10)
        cond &= work["sigma_rank"] <= 0.75
        cond &= work["downside_beta60_rank"] <= 0.75
        cond &= work["max_drawdown20_rank"] <= 0.75
        if filter_name == "legal_minrisk_hardened":
            cond &= work["amp_rank"] <= 0.85
            cond &= work["ret1_rank"] >= 0.20
    elif filter_name == "stable_top30_rerank_trend":
        cond &= work["median_amount20"] >= work["median_amount20"].quantile(0.10)
        cond &= work["sigma20"] <= work["sigma20"].quantile(0.85)
    else:
        return work.copy()

    filtered = work[cond].copy()
    return filtered if len(filtered) >= 5 else work.copy()


def branch_definitions() -> list[dict]:
    return [
        {"branch": "lgb_only_guarded", "score_col": "score_lgb_only", "filter": "stable"},
        {"branch": "balanced_guarded", "score_col": "score_balanced", "filter": "stable"},
        {"branch": "conservative_softrisk_v2", "score_col": "score_conservative_softrisk_v2", "filter": "liquidity_q05"},
        {"branch": "conservative_softrisk_v2_strict", "score_col": "score_conservative_softrisk_v2", "filter": "legal_minrisk_hardened"},
        {"branch": "defensive_v2_strict", "score_col": "score_defensive_v2", "filter": "legal_minrisk"},
        {"branch": "stable_top30_rerank_trend", "score_col": "score_balanced", "filter": "stable_top30_rerank_trend"},
        {"branch": "union_topn_rrf_lcb", "score_col": "_union_rrf_lcb_score", "filter": "union_topn_rrf_lcb"},
        {"branch": "legal_plus_1alpha_shadow", "score_col": "_legal_plus_score", "filter": "legal_plus_1alpha_shadow"},
        {"branch": "safe_union_2slot_shadow", "score_col": "_safe_union_2slot_score", "filter": "safe_union_2slot_shadow"},
        {"branch": "legal_minrisk_hardened", "score_col": "score_legal_minrisk", "filter": "legal_minrisk_hardened"},
    ]


def build_union_shadow_pool(work: pd.DataFrame) -> pd.DataFrame:
    source_cols = [
        ("score_lgb_only", 1.0),
        ("transformer", 1.0),
        ("score_balanced", 0.7),
        ("score_defensive_v2", 0.5),
        ("score_conservative_softrisk_v2", 0.5),
        ("score_legal_minrisk", 0.4),
    ]
    candidate_ids: set[str] = set()
    for col, _ in source_cols:
        if col in work.columns:
            topn = 20 if col in {"score_defensive_v2", "score_legal_minrisk"} else 15
            candidate_ids.update(work.nlargest(min(topn, len(work)), col)["stock_id"].tolist())
    candidates = work[work["stock_id"].isin(candidate_ids)].copy()
    if len(candidates) == 0:
        return candidates

    rrf = pd.Series(0.0, index=work.index)
    consensus = pd.Series(0.0, index=work.index)
    stable_support = pd.Series(False, index=work.index)
    for col, weight in source_cols:
        if col not in work.columns:
            continue
        ranks = work[col].rank(method="min", ascending=False)
        rrf = rrf + weight / (30.0 + ranks)
        in_top20 = ranks <= min(20, len(work))
        consensus = consensus + in_top20.astype(float)
        if col in {"score_defensive_v2", "score_legal_minrisk"}:
            stable_support = stable_support | in_top20

    candidates["_rrf_rank"] = rank_pct(rrf.loc[candidates.index])
    candidates["consensus_count"] = consensus.loc[candidates.index]
    candidates["stable_support"] = stable_support.loc[candidates.index]

    conservative_rank = rank_pct(candidates["score_conservative_softrisk_v2"]) if "score_conservative_softrisk_v2" in candidates.columns else 0.0
    lgb_rank = rank_pct(candidates["score_lgb_only"]) if "score_lgb_only" in candidates.columns else 0.0
    balanced_rank = rank_pct(candidates["score_balanced"]) if "score_balanced" in candidates.columns else 0.0
    tf_rank = rank_pct(candidates["transformer"]) if "transformer" in candidates.columns else candidates.get("tf_norm", 0.0)
    defensive_rank = rank_pct(candidates["score_defensive_v2"]) if "score_defensive_v2" in candidates.columns else 0.0
    legal_rank = rank_pct(candidates["score_legal_minrisk"]) if "score_legal_minrisk" in candidates.columns else 0.0

    candidates["alpha_score"] = (
        0.35 * conservative_rank
        + 0.25 * lgb_rank
        + 0.20 * balanced_rank
        + 0.10 * tf_rank
        + 0.10 * candidates["_rrf_rank"]
    )
    candidates["stable_score"] = (
        0.40 * legal_rank
        + 0.30 * defensive_rank
        + 0.15 * candidates["liq_rank"]
        - 0.10 * candidates["tail_risk_score"]
        - 0.05 * candidates["uncertainty_score"]
    )
    candidates["risk_score"] = (
        0.35 * candidates["tail_risk_score"]
        + 0.25 * candidates["uncertainty_score"]
        + 0.20 * candidates["tail_risk_flag"].astype(float)
        + 0.10 * candidates.get("overheat_flag", False).astype(float)
        + 0.10 * candidates.get("reversal_flag", False).astype(float)
    )
    alpha_std = candidates["alpha_score"].std(ddof=0) + 1e-9
    candidates["alpha_score_z"] = (candidates["alpha_score"] - candidates["alpha_score"].mean()) / alpha_std
    candidates["branch_only_alpha_flag"] = candidates["consensus_count"] <= 1
    candidates["very_tail_flag"] = (
        (candidates["sigma_rank"] > 0.95)
        | (candidates["amp_rank"] > 0.95)
        | (candidates["downside_beta60_rank"] > 0.95)
        | (candidates["max_drawdown20_rank"] > 0.95)
    )
    candidates["very_high_vol_flag"] = (candidates["sigma_rank"] > 0.90) | (candidates["amp_rank"] > 0.90)
    candidates["high_vol_flag"] = (candidates["sigma_rank"] > 0.80) | (candidates["amp_rank"] > 0.80)
    candidates["clean_alpha"] = (
        (candidates["alpha_score_z"] >= 1.75)
        & (candidates["consensus_count"] >= 3)
        & ~candidates["branch_only_alpha_flag"].astype(bool)
        & ~candidates["tail_risk_flag"].astype(bool)
        & ~candidates["very_tail_flag"].astype(bool)
        & ~candidates["very_high_vol_flag"].astype(bool)
        & ~candidates["high_vol_flag"].astype(bool)
        & (candidates["rank_disagreement"] <= 0.25)
    )
    candidates["stable_candidate"] = (
        candidates["stable_support"].astype(bool)
        & ~candidates["tail_risk_flag"].astype(bool)
        & ~candidates["very_tail_flag"].astype(bool)
        & ~candidates["very_high_vol_flag"].astype(bool)
        & (candidates["risk_score"] <= 0.55)
    )
    return candidates


def score_custom_shadow(work: pd.DataFrame, branch: str) -> tuple[pd.DataFrame, str]:
    pool = build_union_shadow_pool(work)
    if len(pool) == 0:
        return pool, "alpha_score"
    legal = filter_branch(work, "legal_minrisk_hardened").copy()
    if len(legal) == 0:
        return pd.DataFrame(), "alpha_score"
    legal_top = legal.sort_values("score_legal_minrisk", ascending=False).head(5).copy()

    if branch == "legal_plus_1alpha_shadow":
        alpha_pool = pool[pool["clean_alpha"].astype(bool)].sort_values("alpha_score", ascending=False)
        if alpha_pool.empty:
            return pd.DataFrame(), "alpha_score"
        alpha = alpha_pool.iloc[[0]].copy()
        legal_aug = legal_top.copy()
        legal_aug["risk_score"] = (
            0.35 * legal_aug["tail_risk_score"]
            + 0.25 * legal_aug["uncertainty_score"]
            + 0.20 * legal_aug["tail_risk_flag"].astype(float)
        )
        alpha_z_gap = float(alpha["alpha_score_z"].iloc[0] - ((legal_aug["score_legal_minrisk"].rank(pct=True).min() - 0.5) * 2.0))
        risk_margin = float(alpha["risk_score"].iloc[0] - legal_aug["risk_score"].max())
        if alpha_z_gap < 0.90 or risk_margin > 0.10:
            return pd.DataFrame(), "alpha_score"
        drop_idx = legal_aug.sort_values(["score_legal_minrisk", "risk_score"], ascending=[True, False]).index[0]
        selected = pd.concat([legal_aug.drop(index=drop_idx), alpha], ignore_index=True)
        selected["_legal_plus_score"] = selected.get("alpha_score", selected.get("score_legal_minrisk", 0.0))
        return selected, "_legal_plus_score"

    if branch == "safe_union_2slot_shadow":
        alpha_slots = pool[pool["clean_alpha"].astype(bool)].sort_values("alpha_score", ascending=False).head(2).copy()
        if len(alpha_slots) < 1:
            return pd.DataFrame(), "alpha_score"
        stable_pool = pool[pool["stable_candidate"].astype(bool) & ~pool["stock_id"].isin(alpha_slots["stock_id"])].copy()
        stable_slots = stable_pool.sort_values("stable_score", ascending=False).head(max(0, 5 - len(alpha_slots))).copy()
        selected = pd.concat([alpha_slots, stable_slots], ignore_index=True)
        if len(selected) < 5:
            fill = legal_top[~legal_top["stock_id"].isin(selected["stock_id"])].head(5 - len(selected)).copy()
            selected = pd.concat([selected, fill], ignore_index=True)
        if len(selected) < 5:
            return pd.DataFrame(), "alpha_score"
        if int(selected.get("tail_risk_flag", pd.Series(False, index=selected.index)).fillna(False).astype(bool).sum()) > 1:
            return pd.DataFrame(), "alpha_score"
        if int(selected.get("very_tail_flag", pd.Series(False, index=selected.index)).fillna(False).astype(bool).sum()) > 0:
            return pd.DataFrame(), "alpha_score"
        if int(selected.get("very_high_vol_flag", pd.Series(False, index=selected.index)).fillna(False).astype(bool).sum()) > 0:
            return pd.DataFrame(), "alpha_score"
        selected["_safe_union_2slot_score"] = selected.get("alpha_score", selected.get("score_legal_minrisk", 0.0))
        return selected, "_safe_union_2slot_score"

    return pd.DataFrame(), "alpha_score"


def score_branch(work: pd.DataFrame, branch: str, score_col: str, filter_name: str) -> dict:
    candidates = filter_branch(work, filter_name)
    if filter_name in {"legal_plus_1alpha_shadow", "safe_union_2slot_shadow"}:
        candidates, rank_col = score_custom_shadow(work, filter_name)
        if len(candidates) == 0:
            return {}
        top = candidates.head(5).copy()
        score = float(top["realized_ret"].mean())
        best_contribution = float(top["realized_ret"].max() / len(top))
        return {
            "branch_score_col": score_col,
            "branch": branch,
            "filter": filter_name,
            "score": score,
            "bad_count": int((top["realized_ret"] < -0.03).sum()),
            "very_bad_count": int((top["realized_ret"] < -0.05).sum()),
            "positive_count": int((top["realized_ret"] > 0).sum()),
            "without_best": score - best_contribution,
            "picks": ",".join(top["stock_id"].tolist()),
            "rets": ",".join(f"{x:.2%}" for x in top["realized_ret"].tolist()),
            **branch_diagnostics(candidates, rank_col),
        }
    if filter_name == "union_topn_rrf_lcb":
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
        if len(candidates) == 0:
            return {}

        rrf = pd.Series(0.0, index=work.index)
        for col, weight in source_cols:
            if col not in work.columns:
                continue
            ranks = work[col].rank(method="min", ascending=False)
            rrf = rrf + weight / (30.0 + ranks)
        candidates["_rrf_rank"] = rank_pct(rrf.loc[candidates.index])
        top5_lgb_gain_rank = (
            rank_pct(candidates["score_lgb_top5"])
            if "score_lgb_top5" in candidates.columns
            else rank_pct(candidates.get("score_lgb_only", pd.Series(0.0, index=candidates.index)))
        )
        conservative_rank = (
            rank_pct(candidates["score_conservative_softrisk_v2"])
            if "score_conservative_softrisk_v2" in candidates.columns
            else pd.Series(0.0, index=candidates.index)
        )
        defensive_rank = (
            rank_pct(candidates["score_defensive_v2"])
            if "score_defensive_v2" in candidates.columns
            else pd.Series(0.0, index=candidates.index)
        )
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
        rank_col = "_union_rrf_lcb_score"
    elif filter_name == "stable_top30_rerank_trend":
        candidates = candidates.sort_values(score_col, ascending=False).head(30).copy()
        fused = rank_pct(candidates[score_col])
        lgb = rank_pct(candidates["lgb"]) if "lgb" in candidates.columns else 0.0
        ret20 = rank_pct(candidates["ret20"]) if "ret20" in candidates.columns else 0.0
        ret5 = rank_pct(candidates["ret5"]) if "ret5" in candidates.columns else 0.0
        liq = rank_pct(candidates["median_amount20"]) if "median_amount20" in candidates.columns else 0.0
        candidates["_branch_score"] = 0.50 * fused + 0.20 * lgb + 0.15 * ret20 + 0.10 * ret5 + 0.05 * liq - 0.10 * candidates["rank_disagreement"]
        rank_col = "_branch_score"
    else:
        rank_col = score_col

    top = candidates.sort_values(rank_col, ascending=False).head(5).copy()
    if len(top) == 0:
        return {}
    score = float(top["realized_ret"].mean())
    best_contribution = float(top["realized_ret"].max() / len(top))
    return {
        "branch_score_col": score_col,
        "branch": branch,
        "filter": filter_name,
        "score": score,
        "bad_count": int((top["realized_ret"] < -0.03).sum()),
        "very_bad_count": int((top["realized_ret"] < -0.05).sum()),
        "positive_count": int((top["realized_ret"] > 0).sum()),
        "without_best": score - best_contribution,
        "picks": ",".join(top["stock_id"].tolist()),
        "rets": ",".join(f"{x:.2%}" for x in top["realized_ret"].tolist()),
        **branch_diagnostics(candidates, rank_col),
    }


def branch_table_for_window(score_df_path: Path, realized: pd.DataFrame) -> pd.DataFrame:
    score_df = pd.read_csv(score_df_path, dtype={"stock_id": str})
    score_df["stock_id"] = normalize_stock_id(score_df["stock_id"])
    work = score_df.merge(realized[["stock_id", "realized_ret"]], on="stock_id", how="left")
    work["realized_ret"] = work["realized_ret"].fillna(0.0)
    work = add_branch_diagnostic_features(work)

    rows = []
    for branch_def in branch_definitions():
        score_col = branch_def["score_col"]
        if score_col not in work.columns and branch_def["filter"] != "union_topn_rrf_lcb":
            continue
        row = score_branch(work, branch_def["branch"], score_col, branch_def["filter"])
        if row:
            rows.append(row)
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def row_passes(row: pd.Series, params: dict) -> bool:
    branch = row["branch"]
    regime = row["regime"]
    budgets = {
        "risk_on_strict": 0.42,
        "neutral_positive": 0.32,
        "mixed_defensive": 0.24,
        "risk_off": 0.18,
    }
    risk_budget_ok = float(row.get("branch_risk_score", 1.0)) <= budgets.get(regime, 1.0)
    if branch == "legal_minrisk_hardened":
        return True
    if branch == "lgb_only_guarded":
        return (
            regime == "risk_on_strict"
            and risk_budget_ok
            and row["top20_overlap"] >= 0.20
            and row["top5_disagreement"] <= 0.35
            and row["top5_tail_risk_count"] <= 2
            and row["score_gap"] >= 0.05
        )
    if branch in {"conservative_softrisk_v2", "conservative_softrisk_v2_strict"}:
        strict = branch == "conservative_softrisk_v2_strict"
        if regime == "risk_off":
            return (
                row["score_gap"] >= params["risk_off_gap_min"] + (0.01 if strict else 0.0)
                and row["top5_disagreement"] <= params["risk_off_disagree_max"] - (0.02 if strict else 0.0)
                and row["top5_tail_risk_count"] <= min(params["risk_off_risk_count_max"], 1 if strict else 5)
                and row["top5_reversal_count"] <= (1 if strict else 2)
                and row["defensive_overlap"] >= 1
            )
        if regime == "mixed_defensive":
            return (
                risk_budget_ok
                and row["score_gap"] >= params["mixed_gap_min"] + (0.01 if strict else 0.0)
                and row["top5_disagreement"] <= params["mixed_disagree_max"] - (0.02 if strict else 0.0)
                and row["top5_tail_risk_count"] <= min(params["mixed_risk_count_max"], 1 if strict else 5)
                and row["top5_reversal_count"] <= (1 if strict else 2)
            )
        if regime == "neutral_positive":
            return risk_budget_ok and row["score_gap"] >= 0.04 and row["top5_disagreement"] <= 0.30 and row["top5_tail_risk_count"] <= 3
        return risk_budget_ok and row["score_gap"] >= 0.03 and row["top5_disagreement"] <= 0.35
    if branch == "balanced_guarded":
        return (
            regime in {"risk_on_strict", "neutral_positive"}
            and risk_budget_ok
            and row["score_gap"] >= 0.04
            and row["top5_disagreement"] <= 0.30
            and row["top5_tail_risk_count"] <= 3
        )
    if branch == "stable_top30_rerank_trend":
        return regime in {"risk_on_strict", "neutral_positive"} and row["top5_tail_risk_count"] <= 3
    if branch == "defensive_v2_strict":
        return (
            row["score_gap"] >= 0.06
            and row["top5_tail_risk_count"] <= 1
            and row["top5_disagreement"] <= 0.70
        )
    return False


def replay_guarded_selector(diag: pd.DataFrame, params: dict) -> pd.DataFrame:
    orders = {
        "risk_on_strict": ["lgb_only_guarded", "balanced_guarded", "conservative_softrisk_v2_strict", "defensive_v2_strict", "legal_minrisk_hardened"],
        "neutral_positive": ["balanced_guarded", "conservative_softrisk_v2_strict", "defensive_v2_strict", "legal_minrisk_hardened"],
        "mixed_defensive": ["conservative_softrisk_v2_strict", "defensive_v2_strict", "legal_minrisk_hardened"],
        "risk_off": ["defensive_v2_strict", "legal_minrisk_hardened"],
    }
    rows = []
    for anchor, group in diag.groupby("anchor_date", sort=True):
        regime = group["regime"].dropna().iloc[0] if group["regime"].notna().any() else "mixed_defensive"
        chosen = None
        for branch in orders.get(regime, orders["mixed_defensive"]):
            candidates = group[group["branch"] == branch]
            if candidates.empty:
                continue
            row = candidates.iloc[0]
            if row_passes(row, params):
                chosen = row
                break
        if chosen is None:
            chosen = group[group["branch"] == "legal_minrisk_hardened"].iloc[0]
        rows.append(chosen.to_dict())
    return pd.DataFrame(rows)


def selector_utility(selected: pd.DataFrame) -> dict:
    mean_score = float(selected["score"].mean())
    median_score = float(selected["score"].median())
    q10_score = float(selected["score"].quantile(0.10))
    worst_score = float(selected["score"].min())
    mean_bad = float(selected["bad_count"].mean())
    mean_very_bad = float(selected["very_bad_count"].mean())
    utility = mean_score + 0.50 * q10_score - 0.50 * abs(min(worst_score, 0.0)) - 0.006 * mean_bad - 0.010 * mean_very_bad
    return {
        "utility": utility,
        "mean_selected_score": mean_score,
        "median_selected_score": median_score,
        "q10_selected_score": q10_score,
        "worst_selected_score": worst_score,
        "positive_window_rate": float((selected["score"] > 0).mean()),
        "mean_bad_count": mean_bad,
        "mean_very_bad_count": mean_very_bad,
    }


def param_grid() -> list[dict]:
    grid = {
        "risk_off_gap_min": [0.06, 0.08, 0.10],
        "risk_off_disagree_max": [0.12, 0.18, 0.25],
        "risk_off_risk_count_max": [1, 2],
        "mixed_gap_min": [0.04, 0.06, 0.08],
        "mixed_disagree_max": [0.18, 0.25, 0.30],
        "mixed_risk_count_max": [1, 2, 3],
    }
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def search_best_params(diag: pd.DataFrame) -> dict:
    best = None
    for params in param_grid():
        selected = replay_guarded_selector(diag, params)
        metrics = selector_utility(selected)
        hard_ok = (
            metrics["mean_selected_score"] > 0
            and metrics["q10_selected_score"] > -0.02
            and metrics["mean_very_bad_count"] <= 0.5
        )
        candidate = {**params, **metrics, "hard_constraints_ok": bool(hard_ok)}
        if best is None:
            best = candidate
            continue
        if hard_ok and not best["hard_constraints_ok"]:
            best = candidate
        elif hard_ok == best["hard_constraints_ok"] and candidate["utility"] > best["utility"]:
            best = candidate
    return best or {}


def loo_threshold_search(diag: pd.DataFrame) -> pd.DataFrame:
    anchors = sorted(diag["anchor_date"].unique())
    rows = []
    for holdout in anchors:
        train_diag = diag[diag["anchor_date"] != holdout].copy()
        valid_diag = diag[diag["anchor_date"] == holdout].copy()
        best = search_best_params(train_diag)
        param_keys = list(param_grid()[0].keys())
        params = {k: best[k] for k in param_keys}
        valid_selected = replay_guarded_selector(valid_diag, params)
        valid_metrics = selector_utility(valid_selected)
        chosen = valid_selected.iloc[0].to_dict()
        rows.append({
            "holdout": holdout,
            **params,
            "chosen_branch": chosen["branch"],
            "score": chosen["score"],
            "bad_count": chosen["bad_count"],
            "very_bad_count": chosen["very_bad_count"],
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
            "train_utility": best["utility"],
            "train_hard_constraints_ok": best["hard_constraints_ok"],
        })
    return pd.DataFrame(rows)


def run_predict_for_anchor(
    raw: pd.DataFrame,
    anchor: pd.Timestamp,
    run_dir: Path,
    model_dir: str | None,
    use_cache: bool,
) -> dict:
    anchor_tag = anchor.strftime("%Y%m%d")
    anchor_dir = run_dir / anchor_tag
    anchor_dir.mkdir(parents=True, exist_ok=True)

    train_slice = anchor_dir / f"train_until_{anchor_tag}.csv"
    output_path = anchor_dir / "result.csv"
    score_df_path = anchor_dir / "predict_score_df.csv"
    filtered_path = anchor_dir / "predict_filtered_top30.csv"
    selector_json_path = anchor_dir / "selector_diagnostics.json"
    selector_csv_path = anchor_dir / "selector_diagnostics.csv"
    selector_debug_path = anchor_dir / "selector_candidates_debug.csv"
    gated_debug_path = anchor_dir / "gated_selector_debug.csv"
    stdout_path = anchor_dir / "predict_stdout.txt"
    stderr_path = anchor_dir / "predict_stderr.txt"

    raw[raw["日期"] <= anchor].to_csv(train_slice, index=False)

    env = os.environ.copy()
    env["BDC_DATA_FILE"] = str(train_slice)
    env["BDC_OUTPUT_PATH"] = str(output_path)
    env["BDC_SCORE_DF_PATH"] = str(score_df_path)
    env["BDC_FILTERED_DF_PATH"] = str(filtered_path)
    env["BDC_SELECTOR_JSON_PATH"] = str(selector_json_path)
    env["BDC_SELECTOR_CSV_PATH"] = str(selector_csv_path)
    env["BDC_SELECTOR_DEBUG_PATH"] = str(selector_debug_path)
    env["BDC_GATED_SELECTOR_DEBUG_PATH"] = str(gated_debug_path)
    if model_dir:
        env["BDC_MODEL_DIR"] = model_dir
    if not use_cache:
        env["BDC_DISABLE_PREDICT_CACHE"] = "1"

    proc = subprocess.run(
        [sys.executable, str(ROOT / "code" / "src" / "predict.py")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"predict failed for {anchor:%Y-%m-%d}; see {stderr_path}")

    selector_info = {}
    if selector_json_path.exists():
        selector_info = json.loads(selector_json_path.read_text(encoding="utf-8"))

    return {
        "output_path": output_path,
        "score_df_path": score_df_path,
        "selector_info": selector_info,
    }


def analyze_anchor_window(
    raw: pd.DataFrame,
    anchor: pd.Timestamp,
    run_dir: Path,
    model_dir: str | None,
    use_cache: bool,
    label_horizon: int,
) -> dict:
    print(f"[batch] running anchor={anchor:%Y-%m-%d}")
    artifacts = run_predict_for_anchor(
        raw=raw,
        anchor=anchor,
        run_dir=run_dir,
        model_dir=model_dir,
        use_cache=use_cache,
    )
    realized, score_window = realized_returns_for_anchor(raw, anchor, label_horizon=label_horizon)
    selected_score, selected_detail = score_result(artifacts["output_path"], realized)
    branch_table = branch_table_for_window(artifacts["score_df_path"], realized)

    selector_info = artifacts["selector_info"]
    regime = selector_info.get("regime", "")
    regime_stats = selector_info.get("regime_stats", {}) or {}
    shadow_scores = branch_table.set_index("branch")["score"].to_dict() if len(branch_table) else {}
    summary_row = {
        "selector_version": selector_info.get("selector_version", SELECTOR_VERSION),
        "config_hash": CONFIG_HASH,
        "gate_hash": GATE_HASH,
        "anchor_date": anchor.strftime("%Y-%m-%d"),
        "score_window": score_window,
        "selected_score": selected_score,
        "score_full_union_shadow": shadow_scores.get("union_topn_rrf_lcb"),
        "score_legal_shadow": shadow_scores.get("legal_minrisk_hardened"),
        "score_defensive_shadow": shadow_scores.get("defensive_v2_strict"),
        "score_legal_plus_1alpha_shadow": shadow_scores.get("legal_plus_1alpha_shadow"),
        "score_safe_union_2slot_shadow": shadow_scores.get("safe_union_2slot_shadow"),
        "chosen_branch": selector_info.get("chosen_branch", ""),
        "regime": regime,
        "breadth_1d": regime_stats.get("breadth_1d"),
        "breadth_5d": regime_stats.get("breadth_5d"),
        "median_ret1": regime_stats.get("median_ret1"),
        "median_ret5": regime_stats.get("median_ret5"),
        "high_vol_ratio": regime_stats.get("high_vol_ratio"),
        "selected_picks": ",".join(selected_detail["stock_id"].tolist()),
        "selected_rets": ",".join(f"{x:.2%}" for x in selected_detail["realized_ret"].tolist()),
        "bad_count": int((selected_detail["realized_ret"] < -0.03).sum()),
        "very_bad_count": int((selected_detail["realized_ret"] < -0.05).sum()),
        "without_best": selected_score - float(selected_detail["contribution"].max()),
    }

    selected_detail = selected_detail.copy()
    selected_detail.insert(0, "portfolio", "selected")
    selected_detail.insert(0, "anchor_date", anchor.strftime("%Y-%m-%d"))
    selected_detail.insert(1, "score_window", score_window)

    branch_table = branch_table.copy()
    branch_table.insert(0, "anchor_date", anchor.strftime("%Y-%m-%d"))
    branch_table.insert(1, "score_window", score_window)
    branch_table.insert(2, "regime", regime)
    for key in ["breadth_1d", "breadth_5d", "median_ret1", "median_ret5", "high_vol_ratio"]:
        branch_table[key] = regime_stats.get(key)

    print(f"[batch] finished anchor={anchor:%Y-%m-%d}, selected_score={selected_score:.6f}")
    return {
        "anchor": anchor,
        "summary_row": summary_row,
        "branch_table": branch_table,
        "selected_detail": selected_detail,
    }


def resolve_worker_count(requested: int | None, anchor_count: int) -> int:
    if requested is not None:
        if requested < 1:
            raise ValueError("--workers must be >= 1")
        return min(requested, anchor_count)
    return max(1, min(4, anchor_count, os.cpu_count() or 1))


def summarize_windows(summary: pd.DataFrame) -> dict:
    return {
        "windows": int(len(summary)),
        "mean_selected_score": float(summary["selected_score"].mean()),
        "median_selected_score": float(summary["selected_score"].median()),
        "q10_selected_score": float(summary["selected_score"].quantile(0.10)),
        "worst_selected_score": float(summary["selected_score"].min()),
        "selected_win_rate": float((summary["selected_score"] > 0).mean()),
        "mean_bad_count": float(summary["bad_count"].mean()),
        "mean_very_bad_count": float(summary["very_bad_count"].mean()),
        "branch_usage": summary["chosen_branch"].value_counts().to_dict(),
        "note": "diagnostic replay with current model unless historical --model-dir is supplied",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/train_hs300_20260424.csv")
    parser.add_argument("--out-dir", default="temp/batch_window_analysis")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--anchors", default=None, help="Comma-separated anchor dates, e.g. 2026-03-06,2026-04-17")
    parser.add_argument("--start-anchor", default=None)
    parser.add_argument("--end-anchor", default=None)
    parser.add_argument("--last-n", type=int, default=8)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--min-history-days", type=int, default=80)
    parser.add_argument("--label-horizon", type=int, default=5)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of anchors to run in parallel; default=min(4, anchors, cpu_count)",
    )
    args = parser.parse_args()

    raw_path = ROOT / args.raw
    raw = load_raw(raw_path)
    dates = [pd.Timestamp(x) for x in sorted(raw["日期"].dropna().unique())]
    anchors = parse_anchor_args(args, dates)
    if not anchors:
        raise ValueError("No valid anchors selected")

    run_name = args.run_name or pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    workers = resolve_worker_count(args.workers, len(anchors))

    print(f"[batch] raw={raw_path}")
    print(f"[batch] out={run_dir}")
    print(f"[batch] anchors={[d.strftime('%Y-%m-%d') for d in anchors]}")
    print(f"[batch] workers={workers}")

    # 限制每个 predict 子进程的 feature_workers，避免 Windows 资源耗尽
    # Windows multiprocessing IPC 传输大量 DataFrame 时，过多并发进程会触发 OSError 1450
    safe_fw = max(1, min(4, (os.cpu_count() or 1) // workers))
    os.environ["BDC_FEATURE_WORKERS"] = str(safe_fw)
    print(f"[batch] feature_workers_per_predict={safe_fw}")

    if workers == 1:
        results = [
            analyze_anchor_window(
                raw=raw,
                anchor=anchor,
                run_dir=run_dir,
                model_dir=args.model_dir,
                use_cache=not args.no_cache,
                label_horizon=args.label_horizon,
            )
            for anchor in anchors
        ]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_anchor = {
                executor.submit(
                    analyze_anchor_window,
                    raw,
                    anchor,
                    run_dir,
                    args.model_dir,
                    not args.no_cache,
                    args.label_horizon,
                ): anchor
                for anchor in anchors
            }
            for future in as_completed(future_to_anchor):
                anchor = future_to_anchor[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    raise RuntimeError(f"anchor {anchor:%Y-%m-%d} failed") from exc

        anchor_order = {anchor: idx for idx, anchor in enumerate(anchors)}
        results.sort(key=lambda item: anchor_order[item["anchor"]])

    summary_rows = [item["summary_row"] for item in results]
    branch_rows = [item["branch_table"] for item in results]
    detail_rows = [item["selected_detail"] for item in results]

    summary = pd.DataFrame(summary_rows)
    branches = pd.concat(branch_rows, ignore_index=True) if branch_rows else pd.DataFrame()
    details = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()

    summary.to_csv(run_dir / "window_summary.csv", index=False)
    branches.to_csv(run_dir / "branch_diagnostics.csv", index=False)
    details.to_csv(run_dir / "portfolio_details.csv", index=False)

    branch_summary = pd.DataFrame()
    if len(branches):
        total_windows = max(1, branches["anchor_date"].nunique())
        branch_summary = (
            branches.groupby(["branch", "filter"], dropna=False)
            .agg(
                windows=("anchor_date", "nunique"),
                mean_selected_score=("score", "mean"),
                median_selected_score=("score", "median"),
                q10_selected_score=("score", lambda s: s.quantile(0.10)),
                worst_selected_score=("score", "min"),
                mean_bad_count=("bad_count", "mean"),
                mean_very_bad_count=("very_bad_count", "mean"),
                positive_window_rate=("score", lambda s: (s > 0).mean()),
                mean_branch_risk_score=("branch_risk_score", "mean"),
                mean_top5_sigma=("top5_mean_sigma", "mean"),
                mean_top5_disagreement=("top5_mean_disagreement", "mean"),
                mean_top5_tail_count=("top5_tail_risk_count", "mean"),
                mean_top5_reversal_count=("top5_reversal_count", "mean"),
                mean_top5_min_liq=("top5_min_liq", "mean"),
                mean_top5_max_drawdown=("top5_max_drawdown", "mean"),
                mean_top5_max_downside_beta=("top5_max_downside_beta", "mean"),
                mean_score_gap_top5_vs_6_10=("score_gap_top5_vs_6_10", "mean"),
                mean_consensus_with_defensive=("consensus_with_defensive", "mean"),
                mean_consensus_with_balanced=("consensus_with_balanced", "mean"),
            )
            .reset_index()
            .sort_values(["mean_selected_score", "q10_selected_score"], ascending=[False, False])
        )
        chosen_counts = summary["chosen_branch"].value_counts()
        branch_summary["evaluated_rate"] = branch_summary["windows"] / total_windows
        branch_summary["activation_rate"] = (
            branch_summary["branch"].map(chosen_counts).fillna(0).astype(float) / total_windows
        )
        branch_summary.to_csv(run_dir / "branch_summary.csv", index=False)

    guarded_default_params = {
        "risk_off_gap_min": 0.08,
        "risk_off_disagree_max": 0.18,
        "risk_off_risk_count_max": 2,
        "mixed_gap_min": 0.06,
        "mixed_disagree_max": 0.22,
        "mixed_risk_count_max": 2,
    }
    guarded_selected = replay_guarded_selector(branches, guarded_default_params) if len(branches) else pd.DataFrame()
    if len(guarded_selected):
        guarded_selected.to_csv(run_dir / "guarded_selector_replay.csv", index=False)

    loo = pd.DataFrame()
    if len(branches) and branches["anchor_date"].nunique() >= 2:
        loo = loo_threshold_search(branches)
        loo.to_csv(run_dir / "loo_threshold_search.csv", index=False)

    aggregate = summarize_windows(summary)
    aggregate["selector_version"] = SELECTOR_VERSION
    aggregate["config_hash"] = CONFIG_HASH
    aggregate["gate_hash"] = GATE_HASH
    aggregate["anchor_policy"] = {
        "anchors": [d.strftime("%Y-%m-%d") for d in anchors],
        "step": int(args.step),
        "last_n": int(args.last_n) if args.last_n else None,
        "explicit_anchors": bool(args.anchors),
        "label_horizon": int(args.label_horizon),
    }
    if len(guarded_selected):
        aggregate["guarded_default"] = selector_utility(guarded_selected)
    if len(branch_summary):
        aggregate["branch_summary"] = json.loads(branch_summary.to_json(orient="records"))
    if not loo.empty:
        aggregate["loo_guarded"] = {
            "mean_score": float(loo["score"].mean()),
            "median_score": float(loo["score"].median()),
            "q10_score": float(loo["score"].quantile(0.10)),
            "worst_score": float(loo["score"].min()),
            "mean_bad_count": float(loo["bad_count"].mean()),
            "mean_very_bad_count": float(loo["very_bad_count"].mean()),
        }
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print("\nWindow summary")
    print(summary.to_string(index=False))
    print(f"\nWrote reports to {run_dir}")


if __name__ == "__main__":
    main()
