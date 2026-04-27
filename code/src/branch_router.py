from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


LEGAL_BRANCHES = [
    "current_aggressive",
    "trend_uncluttered",
    "legal_minrisk_hardened",
    "ai_hardware_mainline_v1",
    "grr_tail_guard",
]

ILLEGAL_RUNTIME_BRANCHES = [
    "reference_baseline_branch",
    "baseline_model_hybrid",
]

DEFAULT_BRANCH_ROUTER_CONFIG = {
    "enabled": False,
    "legal_branches": LEGAL_BRANCHES,
    "risk_off_high": 0.70,
    "risk_off_mid": 0.45,
    "trend_high": 0.60,
    "clutter_high": 0.65,
    "min_confidence_to_switch": 0.08,
    "max_utility_gap_for_blend": 0.04,
    "default_branch": "grr_tail_guard",
    "allow_soft_blend": True,
    "allow_hard_switch": True,
    "rrf_k": 60,
    "high_risk_chaser_threshold": 2,
    "minrisk_risk_improvement": 0.12,
}

DEFAULT_BRANCH_ROUTER_V2A_CONFIG = {
    "enabled": False,
    "default_branch": "grr_tail_guard",
    "rrf_k": 60,
    "trend_override_enabled": True,
    "theme_ai_override_enabled": True,
    "crash_minrisk_enabled": True,
    "trend_override_threshold": 0.58,
    "trend_clutter_max": 0.65,
    "trend_soft_blend_band": 0.06,
    "trend_sigma_cap_q": 0.90,
    "trend_amp_cap_q": 0.90,
    "trend_high_risk_chaser_cap": 1,
    "theme_ai_override_threshold": 0.58,
    "theme_ai_soft_blend_band": 0.06,
    "theme_ai_sigma_cap_q": 0.90,
    "theme_ai_amp_cap_q": 0.90,
    "theme_ai_high_risk_chaser_cap": 1,
}

DEFAULT_BRANCH_ROUTER_V2B_CONFIG = {
    "enabled": True,
    "default_branch": "grr_tail_guard",
    "rrf_k": 60,
    "trend_overlay_enabled": True,
    "theme_ai_overlay_enabled": True,
    "crash_minrisk_enabled": False,
    "max_total_swaps": 2,
    "trend_window_threshold": 0.55,
    "trend_clutter_max": 0.70,
    "trend_max_swaps": 1,
    "trend_min_replacement_gap": 0.04,
    "trend_sigma_cap_q": 0.90,
    "trend_amp_cap_q": 0.90,
    "trend_drawdown_cap_q": 0.90,
    "trend_current_ret20_max": 0.12,
    "theme_ai_window_threshold": 0.62,
    "theme_ai_max_swaps": 1,
    "theme_ai_min_replacement_gap": 0.08,
    "theme_ai_sigma_cap_q": 0.86,
    "theme_ai_amp_cap_q": 0.86,
    "theme_ai_drawdown_cap_q": 0.88,
    "theme_ai_current_ret20_max": 0.02,
    "liquidity_min_rank": 0.05,
}


@dataclass
class RouterDecision:
    chosen_branch: str
    branch_weights: dict[str, float]
    route_reason: str
    risk_off_score: float
    trend_score: float
    theme_score: float
    clutter_score: float
    confidence: float
    fallback_used: bool
    blocked_branches: list[str]
    debug_info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OverlayCandidate:
    stock_id: str
    source_branch: str
    source_rank: int
    source_score: float
    rank_score: float
    overlay_score: float
    risk_features: dict[str, float]
    liquidity_features: dict[str, float]
    consensus_support: float
    in_default_top5: bool
    replacement_target: str | None
    replacement_gain_estimate: float
    veto_flags: dict[str, bool]
    debug_components: dict[str, float]


@dataclass
class OverlayDecision:
    final_top5: list[str]
    swaps: list[dict[str, Any]]
    swap_count: int
    source_branches_used: list[str]
    overlay_reason: str
    rejected_candidates: list[dict[str, Any]]
    debug_info: dict[str, Any]


def _cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULT_BRANCH_ROUTER_CONFIG)
    if config:
        out.update(config)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(x):
        return default
    return x


def _safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float64)
    values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill = values.median() if values.notna().any() else default
    return values.fillna(fill).astype(np.float64)


def _rank_pct(values: pd.Series, ascending: bool = True) -> pd.Series:
    return values.rank(pct=True, method="average", ascending=ascending).fillna(0.5).astype(np.float64)


def _as_frame(branch_output: Any) -> pd.DataFrame:
    if branch_output is None:
        return pd.DataFrame()
    if isinstance(branch_output, pd.DataFrame):
        return branch_output.copy()
    if isinstance(branch_output, dict):
        for key in ("candidates", "df", "data"):
            if isinstance(branch_output.get(key), pd.DataFrame):
                out = branch_output[key].copy()
                for meta_key, meta_value in branch_output.items():
                    if meta_key not in {key, "candidates", "df", "data"}:
                        out.attrs[meta_key] = meta_value
                return out
        if "selected_stocks" in branch_output:
            stocks = branch_output.get("selected_stocks") or []
            if isinstance(stocks, str):
                stocks = [x.strip() for x in stocks.split(",") if x.strip()]
            return pd.DataFrame({"stock_id": stocks})
    return pd.DataFrame()


def _score_col(df: pd.DataFrame) -> str:
    for attr_key in ("score_col", "branch_score_col"):
        if df.attrs.get(attr_key) in df.columns:
            return str(df.attrs[attr_key])
    for col in ("branch_score", "rank_blend_score", "grr_final_score", "score", "score_legal_minrisk"):
        if col in df.columns:
            return col
    return ""


def _top_candidates(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    col = _score_col(df)
    out = df.copy()
    if col:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        return out.sort_values(col, ascending=False).head(min(top_k, len(out))).copy()
    return out.head(min(top_k, len(out))).copy()


def _score_source_and_values(df: pd.DataFrame, rrf_k: int = 60) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    col = _score_col(out)
    if col:
        out["raw_branch_score"] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        out["_branch_rank_score"] = out["raw_branch_score"]
        out["score_source"] = "raw"
        out = out.sort_values("raw_branch_score", ascending=False).copy()
    else:
        out["rank_in_branch"] = np.arange(1, len(out) + 1, dtype=np.float64)
        out["raw_branch_score"] = np.nan
        out["_branch_rank_score"] = 1.0 / (float(rrf_k) + out["rank_in_branch"])
        out["score_source"] = "rank_derived"
        return out, "rank_derived"
    out["rank_in_branch"] = out["raw_branch_score"].rank(method="first", ascending=False).astype(np.float64)
    if len(out):
        out = out.sort_values("rank_in_branch").copy()
    return out, "raw"


def _with_risk_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sigma_rank" in out.columns:
        out["_sigma_rank_for_router"] = _safe_series(out, "sigma_rank", 0.5)
    elif "grr_sigma_rank" in out.columns:
        out["_sigma_rank_for_router"] = _safe_series(out, "grr_sigma_rank", 0.5)
    else:
        out["_sigma_rank_for_router"] = _rank_pct(_safe_series(out, "sigma20", 0.0), ascending=True)
    if "amp_rank" in out.columns:
        out["_amp_rank_for_router"] = _safe_series(out, "amp_rank", 0.5)
    elif "grr_amp_rank" in out.columns:
        out["_amp_rank_for_router"] = _safe_series(out, "grr_amp_rank", 0.5)
    else:
        out["_amp_rank_for_router"] = _rank_pct(_safe_series(out, "amp20", 0.0), ascending=True)
    if "max_drawdown20_rank" in out.columns:
        out["_drawdown_rank_for_router"] = _safe_series(out, "max_drawdown20_rank", 0.5)
    elif "grr_drawdown_rank" in out.columns:
        out["_drawdown_rank_for_router"] = _safe_series(out, "grr_drawdown_rank", 0.5)
    else:
        out["_drawdown_rank_for_router"] = _rank_pct(_safe_series(out, "max_drawdown20", 0.0), ascending=True)
    if "ret5_rank" in out.columns:
        out["_ret5_rank_for_router"] = _safe_series(out, "ret5_rank", 0.5)
    elif "grr_ret5_rank" in out.columns:
        out["_ret5_rank_for_router"] = _safe_series(out, "grr_ret5_rank", 0.5)
    else:
        out["_ret5_rank_for_router"] = _rank_pct(_safe_series(out, "ret5", 0.0), ascending=True)
    high_risk = (
        (out["_sigma_rank_for_router"] >= 0.75)
        & (out["_amp_rank_for_router"] >= 0.75)
        & (out["_ret5_rank_for_router"] >= 0.70)
    )
    if "high_risk_chaser_flag" not in out.columns:
        out["high_risk_chaser_flag"] = high_risk.astype(bool)
    return out


def build_branch_candidates(
    branch_outputs: dict[str, Any],
    top_k: int = 20,
    rrf_k: int = 60,
    final_top_n: int = 5,
) -> pd.DataFrame:
    rows = []
    for branch_name, output in (branch_outputs or {}).items():
        if branch_name in ILLEGAL_RUNTIME_BRANCHES:
            continue
        frame = _as_frame(output)
        if frame.empty or "stock_id" not in frame.columns:
            continue
        scored, source = _score_source_and_values(frame, rrf_k=rrf_k)
        scored = _with_risk_feature_columns(scored).head(min(top_k, len(scored))).copy()
        selected = set(scored.head(min(final_top_n, len(scored)))["stock_id"].astype(str))
        for _, row in scored.iterrows():
            stock_id = str(row["stock_id"])
            rows.append(
                {
                    "branch_name": branch_name,
                    "candidate_stock_id": stock_id,
                    "raw_branch_score": _safe_float(row.get("raw_branch_score"), np.nan),
                    "rank_score": _safe_float(row.get("_branch_rank_score"), 0.0),
                    "rank_in_branch": int(_safe_float(row.get("rank_in_branch"), 0.0)),
                    "post_filter_selected_top5": stock_id in selected,
                    "final_selected_flag": stock_id in selected,
                    "score_source": source,
                    "sigma_rank": _safe_float(row.get("_sigma_rank_for_router"), 0.5),
                    "amp_rank": _safe_float(row.get("_amp_rank_for_router"), 0.5),
                    "drawdown_rank": _safe_float(row.get("_drawdown_rank_for_router"), 0.5),
                    "risk_rank": _safe_float(
                        0.4 * _safe_float(row.get("_sigma_rank_for_router"), 0.5)
                        + 0.3 * _safe_float(row.get("_amp_rank_for_router"), 0.5)
                        + 0.3 * _safe_float(row.get("_drawdown_rank_for_router"), 0.5),
                        0.5,
                    ),
                    "median_amount20": _safe_float(row.get("median_amount20"), 0.0),
                    "high_risk_chaser_flag": bool(row.get("high_risk_chaser_flag", False)),
                    "veto_flag": bool(row.get("grr_high_risk_chaser_veto", False)),
                }
            )
    return pd.DataFrame(rows)


def _rank_or_raw_mean(df: pd.DataFrame, raw_col: str, rank_col: str) -> float:
    if raw_col in df.columns:
        return _safe_float(_safe_series(df, raw_col).mean(), 0.0)
    if rank_col in df.columns:
        return _safe_float(_safe_series(df, rank_col).mean(), 0.5)
    return 0.0


def _branch_snapshot_from_frame(
    window_date: str,
    branch_name: str,
    df: pd.DataFrame,
    market_state: dict[str, Any] | None = None,
    realized_selected_score: float | None = None,
    top_sets: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    market_state = market_state or {}
    top_sets = top_sets or {}
    scored, score_source = _score_source_and_values(df, rrf_k=int(market_state.get("rrf_k", 60)))
    scored = _with_risk_feature_columns(scored)
    top20 = scored.head(min(20, len(scored))).copy()
    top10 = top20.head(min(10, len(top20))).copy()
    top5 = top10.head(min(5, len(top10))).copy()
    scores = _safe_series(top10, "_branch_rank_score") if "_branch_rank_score" in top10.columns else pd.Series(0.0, index=top10.index)
    top5_scores = scores.iloc[: min(5, len(scores))]
    next_scores = scores.iloc[5: min(10, len(scores))]
    margin = float(top5_scores.mean() - next_scores.mean()) if len(next_scores) else 0.0
    dispersion = float(scores.std(ddof=0)) if len(scores) else 0.0
    top20_scores = _safe_series(top20, "_branch_rank_score") if "_branch_rank_score" in top20.columns else pd.Series(0.0, index=top20.index)
    p = top20_scores - top20_scores.min()
    if float(p.sum()) <= 1e-12:
        p = pd.Series(1.0 / max(len(top20_scores), 1), index=top20_scores.index)
    else:
        p = p / p.sum()
    entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(max(len(p), 2))) if len(p) else 0.0
    concentration = float(top5_scores.sum() / (top20_scores.sum() + 1e-12)) if len(top20_scores) else 0.0

    stock_set = set(top5.get("stock_id", pd.Series(dtype=str)).astype(str).tolist())
    sigma_rank = _safe_series(top5, "_sigma_rank_for_router", 0.5)
    amp_rank = _safe_series(top5, "_amp_rank_for_router", 0.5)
    drawdown_rank = _safe_series(top5, "_drawdown_rank_for_router", 0.5)
    ret5_rank = _safe_series(top5, "_ret5_rank_for_router", 0.5)
    risk_rank_top5 = 0.4 * sigma_rank + 0.3 * amp_rank + 0.3 * drawdown_rank
    risk_rank_top10 = (
        0.4 * _safe_series(top10, "_sigma_rank_for_router", 0.5)
        + 0.3 * _safe_series(top10, "_amp_rank_for_router", 0.5)
        + 0.3 * _safe_series(top10, "_drawdown_rank_for_router", 0.5)
    )
    high_risk = (sigma_rank >= 0.75) & (amp_rank >= 0.75) & (ret5_rank >= 0.70)

    return {
        "window_date": window_date,
        "branch_name": branch_name,
        "selected_stocks": ",".join(top5.get("stock_id", pd.Series(dtype=str)).astype(str).tolist()),
        "branch_score": float(top5_scores.mean()) if len(top5_scores) else 0.0,
        "score_source": score_source,
        "candidate_depth": int(len(top20)),
        "top5_score_mean": float(top5_scores.mean()) if len(top5_scores) else 0.0,
        "top6_10_score_mean": float(next_scores.mean()) if len(next_scores) else np.nan,
        "realized_selected_score": realized_selected_score,
        "mean_sigma20": _rank_or_raw_mean(top5, "sigma20", "grr_sigma_rank"),
        "mean_amp20": _rank_or_raw_mean(top5, "amp20", "grr_amp_rank"),
        "mean_drawdown20": _rank_or_raw_mean(top5, "max_drawdown20", "grr_drawdown_rank"),
        "mean_ret5": _safe_float(_safe_series(top5, "ret5", 0.0).mean(), 0.0),
        "mean_abs_ret5": _safe_float(_safe_series(top5, "ret5", 0.0).abs().mean(), 0.0),
        "selected_ret20_strength": _safe_float(_safe_series(top5, "ret20", 0.0).mean(), 0.0),
        "mean_median_amount20": _safe_float(_safe_series(top5, "median_amount20", 0.0).mean(), 0.0),
        "mean_consensus_rank": _safe_float(_safe_series(top5, "grr_consensus_count", 0.0).mean(), 0.0),
        "mean_consensus_support": _safe_float(_safe_series(top5, "grr_consensus_norm", 0.0).mean(), 0.0),
        "overlap_with_grr_top5": len(stock_set & top_sets.get("grr_tail_guard", set())),
        "overlap_with_minrisk_top5": len(stock_set & top_sets.get("legal_minrisk_hardened", set())),
        "overlap_with_trend_top5": len(stock_set & top_sets.get("trend_uncluttered", set())),
        "overlap_with_ai_hardware_top5": len(stock_set & top_sets.get("ai_hardware_mainline_v1", set())),
        "branch_score_margin_top5_vs_top10": margin,
        "branch_score_dispersion_top10": dispersion,
        "branch_rank_entropy_top20": entropy,
        "top5_vs_top20_score_concentration": concentration,
        "risk_rank_mean_top5": float(risk_rank_top5.mean()) if len(risk_rank_top5) else 0.5,
        "risk_rank_mean_top10": float(risk_rank_top10.mean()) if len(risk_rank_top10) else 0.5,
        "amp_rank_mean_top5": float(amp_rank.mean()) if len(amp_rank) else 0.5,
        "sigma_rank_mean_top5": float(sigma_rank.mean()) if len(sigma_rank) else 0.5,
        "drawdown_rank_mean_top5": float(drawdown_rank.mean()) if len(drawdown_rank) else 0.5,
        "high_risk_chaser_count": int(high_risk.sum()),
        "veto_count": int(top5.get("grr_high_risk_chaser_veto", pd.Series(False, index=top5.index)).astype(bool).sum()) if len(top5) else 0,
        "risk_off_score": _safe_float(market_state.get("risk_off_score"), 0.0),
        "crash_mode": bool(market_state.get("crash_mode", False)),
        "market_ret5": _safe_float(market_state.get("market_ret5"), 0.0),
        "market_ret20": _safe_float(market_state.get("market_ret20"), 0.0),
        "market_sigma20_median": _safe_float(market_state.get("market_sigma20_median"), 0.0),
        "market_breadth_5d": _safe_float(market_state.get("market_breadth_5d"), 0.5),
        "ret5_dispersion": _safe_float(market_state.get("ret5_dispersion"), 0.0),
        "amount_chg_5d": _safe_float(market_state.get("amount_chg_5d"), 0.0),
        "trend_strength": _safe_float(market_state.get("trend_strength"), 0.0),
        "clutter_score": _safe_float(market_state.get("clutter_score"), 0.0),
    }


def build_branch_snapshots(
    branch_outputs: dict[str, Any],
    window_date: str = "",
    market_state: dict[str, Any] | None = None,
    realized_scores: dict[str, float] | None = None,
) -> pd.DataFrame:
    realized_scores = realized_scores or {}
    frames = {name: _as_frame(output) for name, output in branch_outputs.items()}
    top_sets = {
        name: set(_top_candidates(frame, 5).get("stock_id", pd.Series(dtype=str)).astype(str).tolist())
        for name, frame in frames.items()
        if not frame.empty
    }
    rows = []
    for branch_name, frame in frames.items():
        if frame.empty:
            continue
        rows.append(
            _branch_snapshot_from_frame(
                window_date=window_date,
                branch_name=branch_name,
                df=frame,
                market_state=market_state,
                realized_selected_score=realized_scores.get(branch_name),
                top_sets=top_sets,
            )
        )
    return pd.DataFrame(rows)


def compute_branch_state_features(
    market_df: pd.DataFrame,
    candidate_outputs: dict[str, Any] | None = None,
    risk_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    risk_features = risk_features or {}
    if market_df is None or market_df.empty:
        state = {
            "risk_off_score": _safe_float(risk_features.get("risk_off_score"), 0.0),
            "crash_mode": bool(risk_features.get("crash_mode", False)),
            "trend_score": 0.0,
            "theme_score": 0.0,
            "clutter_score": 0.0,
            "market_ret5": 0.0,
            "market_ret20": 0.0,
            "market_sigma20_median": 0.0,
            "market_breadth_5d": 0.5,
            "ret5_dispersion": 0.0,
            "amount_chg_5d": 0.0,
            "trend_strength": 0.0,
        }
        state.update(risk_features)
        return state

    ret5 = _safe_series(market_df, "ret5", 0.0)
    ret20 = _safe_series(market_df, "ret20", 0.0)
    sigma20 = _safe_series(market_df, "sigma20", 0.0)
    amount_chg = _safe_series(market_df, "amt_ratio5", 1.0) - 1.0
    breadth5 = float(ret5.gt(0.0).mean())
    median_ret5 = float(ret5.median())
    median_ret20 = float(ret20.median())
    ret5_dispersion = float(ret5.quantile(0.80) - ret5.quantile(0.20))
    high_vol_ratio = float(_rank_pct(sigma20).gt(0.75).mean())
    trend_strength = float(np.clip(0.45 * breadth5 + 0.35 * (median_ret5 / 0.04) + 0.20 * (median_ret20 / 0.08), 0.0, 1.0))

    disagreement = _safe_series(market_df, "rank_disagreement", 0.35)
    if "rank_disagreement" not in market_df.columns and {"lgb", "transformer"}.issubset(market_df.columns):
        disagreement = (_rank_pct(_safe_series(market_df, "lgb")) - _rank_pct(_safe_series(market_df, "transformer"))).abs()
    clutter_score = float(np.clip(0.60 * disagreement.median() + 0.40 * np.clip(ret5_dispersion / 0.12, 0.0, 1.0), 0.0, 1.0))
    risk_off_score = float(
        np.clip(
            0.30 * (1.0 - breadth5)
            + 0.25 * np.clip(-median_ret5 / 0.035, 0.0, 1.0)
            + 0.20 * high_vol_ratio
            + 0.15 * np.clip(ret5_dispersion / 0.14, 0.0, 1.0)
            + 0.10 * np.clip(-median_ret20 / 0.08, 0.0, 1.0),
            0.0,
            1.0,
        )
    )

    state = {
        "risk_off_score": _safe_float(risk_features.get("risk_off_score"), risk_off_score),
        "crash_mode": bool(risk_features.get("crash_mode", risk_off_score >= 0.70)),
        "trend_score": _safe_float(risk_features.get("trend_score"), trend_strength),
        "theme_score": _safe_float(risk_features.get("theme_score"), 0.0),
        "clutter_score": _safe_float(risk_features.get("clutter_score"), clutter_score),
        "market_ret5": median_ret5,
        "market_ret20": median_ret20,
        "market_sigma20_median": float(sigma20.median()),
        "market_breadth_5d": breadth5,
        "ret5_dispersion": ret5_dispersion,
        "amount_chg_5d": float(amount_chg.median()),
        "trend_strength": trend_strength,
    }

    if candidate_outputs:
        frames = {k: _top_candidates(_as_frame(v), 5) for k, v in candidate_outputs.items()}
        if frames.get("ai_hardware_mainline_v1") is not None and not frames["ai_hardware_mainline_v1"].empty:
            ai_ids = set(frames["ai_hardware_mainline_v1"].get("stock_id", pd.Series(dtype=str)).astype(str))
            overlap = 0
            for peer in ("grr_tail_guard", "trend_uncluttered"):
                if peer in frames and not frames[peer].empty:
                    overlap += len(ai_ids & set(frames[peer].get("stock_id", pd.Series(dtype=str)).astype(str)))
            state["theme_score"] = max(state["theme_score"], float(np.clip(overlap / 5.0, 0.0, 1.0)))
    state.update(risk_features)
    return state


def _branch_metrics(branch_outputs: dict[str, Any], market_state: dict[str, Any]) -> dict[str, dict[str, float]]:
    snapshots = build_branch_snapshots(branch_outputs, market_state=market_state)
    metrics = {}
    for _, row in snapshots.iterrows():
        branch = str(row["branch_name"])
        metrics[branch] = {
            "score_strength": np.tanh(max(_safe_float(row.get("branch_score_margin_top5_vs_top10")), 0.0) * 4.0),
            "dispersion": _safe_float(row.get("branch_score_dispersion_top10")),
            "branch_score_margin_top5_vs_top10": _safe_float(row.get("branch_score_margin_top5_vs_top10")),
            "branch_score_dispersion_top10": _safe_float(row.get("branch_score_dispersion_top10")),
            "branch_rank_entropy_top20": _safe_float(row.get("branch_rank_entropy_top20")),
            "top5_vs_top20_score_concentration": _safe_float(row.get("top5_vs_top20_score_concentration")),
            "consensus_support": _safe_float(row.get("mean_consensus_support")),
            "mean_sigma20": _safe_float(row.get("mean_sigma20")),
            "mean_amp20": _safe_float(row.get("mean_amp20")),
            "mean_drawdown20": _safe_float(row.get("mean_drawdown20")),
            "risk_rank_mean_top5": _safe_float(row.get("risk_rank_mean_top5"), 0.5),
            "risk_rank_mean_top10": _safe_float(row.get("risk_rank_mean_top10"), 0.5),
            "amp_rank_mean_top5": _safe_float(row.get("amp_rank_mean_top5"), 0.5),
            "sigma_rank_mean_top5": _safe_float(row.get("sigma_rank_mean_top5"), 0.5),
            "drawdown_rank_mean_top5": _safe_float(row.get("drawdown_rank_mean_top5"), 0.5),
            "mean_ret5": _safe_float(row.get("mean_ret5")),
            "mean_abs_ret5": _safe_float(row.get("mean_abs_ret5")),
            "selected_ret20_strength": _safe_float(row.get("selected_ret20_strength")),
            "candidate_depth": _safe_float(row.get("candidate_depth")),
            "high_risk_chaser_count": _safe_float(row.get("high_risk_chaser_count")),
            "veto_count": _safe_float(row.get("veto_count")),
            "overlap_with_grr_top5": _safe_float(row.get("overlap_with_grr_top5")),
            "overlap_with_trend_top5": _safe_float(row.get("overlap_with_trend_top5")),
            "overlap_with_ai_hardware_top5": _safe_float(row.get("overlap_with_ai_hardware_top5")),
        }
    return metrics


def _rank_risks(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {k: dict(v) for k, v in metrics.items()}
    for field in ("mean_sigma20", "mean_amp20", "mean_drawdown20"):
        values = pd.Series({k: _safe_float(v.get(field), 0.0) for k, v in metrics.items()})
        if values.empty:
            continue
        ranked = values.rank(pct=True, method="average", ascending=True).to_dict()
        for branch, value in ranked.items():
            out[branch][field + "_rank"] = float(value)
    return out


def route_branch_v1(
    branch_outputs: dict[str, Any],
    market_state: dict[str, Any],
    config: dict[str, Any] | None,
) -> RouterDecision:
    cfg = _cfg(config)
    legal = list(cfg.get("legal_branches", LEGAL_BRANCHES))
    default_branch = str(cfg.get("default_branch", "grr_tail_guard"))
    debug: dict[str, Any] = {}
    blocked: list[str] = []

    filtered_outputs: dict[str, Any] = {}
    illegal_filtered = False
    for branch, output in (branch_outputs or {}).items():
        if branch in ILLEGAL_RUNTIME_BRANCHES:
            illegal_filtered = True
            blocked.append(branch)
            continue
        if branch in legal:
            filtered_outputs[branch] = output
    debug["illegal_branch_filtered"] = illegal_filtered
    debug["available_branches"] = sorted(filtered_outputs)
    debug["blocked_branches"] = blocked.copy()

    if not filtered_outputs:
        return RouterDecision(
            chosen_branch=default_branch,
            branch_weights={default_branch: 1.0},
            route_reason="no_legal_branch_available",
            risk_off_score=0.0,
            trend_score=0.0,
            theme_score=0.0,
            clutter_score=0.0,
            confidence=0.0,
            fallback_used=True,
            blocked_branches=blocked,
            debug_info=debug,
        )

    if default_branch not in filtered_outputs:
        default_branch = "legal_minrisk_hardened" if "legal_minrisk_hardened" in filtered_outputs else sorted(filtered_outputs)[0]
        debug["default_branch_repaired"] = default_branch

    state = compute_branch_state_features(pd.DataFrame(), filtered_outputs, market_state)
    risk_off_score = _safe_float(state.get("risk_off_score"), 0.0)
    trend_score = _safe_float(state.get("trend_score"), _safe_float(state.get("trend_strength"), 0.0))
    theme_score = _safe_float(state.get("theme_score"), 0.0)
    clutter_score = _safe_float(state.get("clutter_score"), 0.0)
    crash_mode = bool(state.get("crash_mode", False))

    if crash_mode or risk_off_score >= float(cfg.get("risk_off_high", 0.70)):
        chosen = "legal_minrisk_hardened" if "legal_minrisk_hardened" in filtered_outputs else default_branch
        return RouterDecision(
            chosen_branch=chosen,
            branch_weights={chosen: 1.0},
            route_reason="risk_off_minrisk",
            risk_off_score=risk_off_score,
            trend_score=trend_score,
            theme_score=theme_score,
            clutter_score=clutter_score,
            confidence=1.0,
            fallback_used=chosen == default_branch,
            blocked_branches=blocked,
            debug_info={**debug, "utilities": {}, "market_state": state},
        )

    metrics = _rank_risks(_branch_metrics(filtered_outputs, state))
    current = metrics.get("current_aggressive", {})
    minrisk = metrics.get("legal_minrisk_hardened", {})
    if (
        "current_aggressive" in filtered_outputs
        and _safe_float(current.get("high_risk_chaser_count")) >= float(cfg.get("high_risk_chaser_threshold", 2))
        and "legal_minrisk_hardened" in filtered_outputs
    ):
        cur_risk = 0.5 * _safe_float(current.get("mean_sigma20_rank"), 0.5) + 0.5 * _safe_float(current.get("mean_amp20_rank"), 0.5)
        min_risk = 0.5 * _safe_float(minrisk.get("mean_sigma20_rank"), 0.5) + 0.5 * _safe_float(minrisk.get("mean_amp20_rank"), 0.5)
        if cur_risk - min_risk >= float(cfg.get("minrisk_risk_improvement", 0.12)):
            chosen = default_branch if default_branch in filtered_outputs else "legal_minrisk_hardened"
            if risk_off_score >= float(cfg.get("risk_off_mid", 0.45)):
                chosen = "legal_minrisk_hardened"
            blocked.append("current_aggressive")
            return RouterDecision(
                chosen_branch=chosen,
                branch_weights={chosen: 1.0},
                route_reason="aggressive_chaser_blocked",
                risk_off_score=risk_off_score,
                trend_score=trend_score,
                theme_score=theme_score,
                clutter_score=clutter_score,
                confidence=float(cur_risk - min_risk),
                fallback_used=chosen == default_branch,
                blocked_branches=blocked,
                debug_info={**debug, "utilities": {}, "branch_metrics": metrics, "market_state": state},
            )

    recent = state.get("recent_branch_oof_strength", {}) or {}
    utilities: dict[str, float] = {}
    for branch in filtered_outputs:
        m = metrics.get(branch, {})
        utility = (
            0.28 * _safe_float(m.get("score_strength"))
            + 0.18 * _safe_float(m.get("consensus_support"))
            + 0.16 * trend_score
            + 0.16 * _safe_float(recent.get(branch), 0.0)
            - 0.12 * _safe_float(m.get("mean_sigma20_rank"), 0.5)
            - 0.10 * _safe_float(m.get("mean_amp20_rank"), 0.5)
            - 0.10 * _safe_float(m.get("mean_drawdown20_rank"), 0.5)
            - 0.06 * _safe_float(m.get("high_risk_chaser_count"), 0.0)
            - 0.10 * clutter_score
        )
        if branch == "trend_uncluttered":
            utility += 0.24 * max(0.0, trend_score - float(cfg.get("trend_high", 0.60)))
            utility += 0.16 * max(0.0, float(cfg.get("clutter_high", 0.65)) - clutter_score)
        elif branch == "ai_hardware_mainline_v1":
            overlap_bonus = min(_safe_float(m.get("overlap_with_grr_top5")) + _safe_float(m.get("overlap_with_trend_top5")), 5.0) / 5.0
            risk_penalty = max(0.0, _safe_float(m.get("mean_sigma20_rank"), 0.5) - 0.65)
            utility += 0.20 * theme_score + 0.12 * overlap_bonus - 0.18 * risk_penalty
        elif branch == "current_aggressive":
            utility += 0.20 * max(0.0, float(cfg.get("risk_off_mid", 0.45)) - risk_off_score)
            utility -= 0.08 * _safe_float(m.get("veto_count"), 0.0)
        elif branch == "legal_minrisk_hardened":
            utility += 0.20 * max(0.0, risk_off_score - 0.30)
            utility += 0.10 * max(0.0, 0.50 - _safe_float(state.get("market_breadth_5d"), 0.5))
            utility += 0.08 * np.clip(_safe_float(state.get("ret5_dispersion"), 0.0) / 0.12, 0.0, 1.0)
        elif branch == "grr_tail_guard":
            utility += 0.03
        utilities[branch] = float(utility)

    debug["utilities"] = utilities
    debug["branch_metrics"] = metrics
    debug["market_state"] = state

    ordered = sorted(utilities.items(), key=lambda item: item[1], reverse=True)
    best_branch, best_utility = ordered[0]
    default_utility = utilities.get(default_branch, best_utility)
    confidence = float(best_utility - default_utility)
    if best_branch == default_branch or confidence < float(cfg.get("min_confidence_to_switch", 0.08)):
        return RouterDecision(
            chosen_branch=default_branch,
            branch_weights={default_branch: 1.0},
            route_reason="low_confidence_default",
            risk_off_score=risk_off_score,
            trend_score=trend_score,
            theme_score=theme_score,
            clutter_score=clutter_score,
            confidence=max(confidence, 0.0),
            fallback_used=True,
            blocked_branches=blocked,
            debug_info=debug,
        )

    if not bool(cfg.get("allow_hard_switch", True)):
        return RouterDecision(
            chosen_branch=default_branch,
            branch_weights={default_branch: 1.0},
            route_reason="hard_switch_disabled",
            risk_off_score=risk_off_score,
            trend_score=trend_score,
            theme_score=theme_score,
            clutter_score=clutter_score,
            confidence=confidence,
            fallback_used=True,
            blocked_branches=blocked,
            debug_info=debug,
        )

    if bool(cfg.get("allow_soft_blend", True)) and len(ordered) >= 2:
        second_branch, second_utility = ordered[1]
        if best_utility - second_utility <= float(cfg.get("max_utility_gap_for_blend", 0.04)):
            raw = np.asarray([max(best_utility, -5.0), max(second_utility, -5.0)], dtype=np.float64)
            raw = np.exp(raw - raw.max())
            weights = raw / raw.sum()
            return RouterDecision(
                chosen_branch=best_branch,
                branch_weights={best_branch: float(weights[0]), second_branch: float(weights[1])},
                route_reason="soft_blend_rank_rrf",
                risk_off_score=risk_off_score,
                trend_score=trend_score,
                theme_score=theme_score,
                clutter_score=clutter_score,
                confidence=confidence,
                fallback_used=False,
                blocked_branches=blocked,
                debug_info=debug,
            )

    return RouterDecision(
        chosen_branch=best_branch,
        branch_weights={best_branch: 1.0},
        route_reason="utility_switch",
        risk_off_score=risk_off_score,
        trend_score=trend_score,
        theme_score=theme_score,
        clutter_score=clutter_score,
        confidence=confidence,
        fallback_used=False,
        blocked_branches=blocked,
        debug_info=debug,
    )


def _clip01(value: Any) -> float:
    return float(np.clip(_safe_float(value), 0.0, 1.0))


def _positive_norm(value: Any, scale: float) -> float:
    return float(np.tanh(max(_safe_float(value), 0.0) / max(float(scale), 1e-9)))


def _v2a_cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULT_BRANCH_ROUTER_V2A_CONFIG)
    if config:
        out.update(config)
    return out


def _v2b_cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULT_BRANCH_ROUTER_V2B_CONFIG)
    if config:
        out.update(config)
    return out


def route_branch_v2a(
    branch_outputs: dict[str, Any],
    market_state: dict[str, Any],
    config: dict[str, Any] | None,
) -> RouterDecision:
    cfg = _v2a_cfg(config)
    default_branch = str(cfg.get("default_branch", "grr_tail_guard"))
    debug: dict[str, Any] = {}
    blocked: list[str] = []
    filtered_outputs: dict[str, Any] = {}
    illegal_filtered = False
    for branch, output in (branch_outputs or {}).items():
        if branch in ILLEGAL_RUNTIME_BRANCHES:
            illegal_filtered = True
            blocked.append(branch)
            continue
        if branch in LEGAL_BRANCHES:
            filtered_outputs[branch] = output
    debug["illegal_branch_filtered"] = illegal_filtered
    debug["blocked_branches"] = blocked.copy()
    debug["available_branches"] = sorted(filtered_outputs)

    if default_branch not in filtered_outputs:
        default_branch = "grr_tail_guard" if "grr_tail_guard" in filtered_outputs else sorted(filtered_outputs)[0] if filtered_outputs else "grr_tail_guard"
        debug["default_branch_repaired"] = default_branch

    state = compute_branch_state_features(pd.DataFrame(), filtered_outputs, market_state)
    risk_off_score = _safe_float(state.get("risk_off_score"), 0.0)
    trend_score = _safe_float(state.get("trend_score"), _safe_float(state.get("trend_strength"), 0.0))
    theme_score = _safe_float(state.get("theme_score"), 0.0)
    clutter_score = _safe_float(state.get("clutter_score"), 0.0)
    crash_mode = bool(state.get("crash_mode", False))
    metrics = _rank_risks(_branch_metrics(filtered_outputs, {**state, "rrf_k": cfg.get("rrf_k", 60)}))
    debug["branch_metrics"] = metrics
    debug["market_state"] = state

    if crash_mode and bool(cfg.get("crash_minrisk_enabled", True)) and "legal_minrisk_hardened" in filtered_outputs:
        return RouterDecision(
            chosen_branch="legal_minrisk_hardened",
            branch_weights={"legal_minrisk_hardened": 1.0},
            route_reason="crash_minrisk_rescue",
            risk_off_score=risk_off_score,
            trend_score=trend_score,
            theme_score=theme_score,
            clutter_score=clutter_score,
            confidence=1.0,
            fallback_used=False,
            blocked_branches=blocked,
            debug_info={**debug, "trend_override_score": 0.0, "theme_ai_override_score": 0.0, "blend_branches": [], "blend_weights": {}, "rrf_k": int(cfg.get("rrf_k", 60))},
        )

    def finish(branch: str, reason: str, score: float, band: float) -> RouterDecision:
        hard = score >= float(cfg.get(reason + "_threshold", 999.0)) + band
        if hard:
            weights = {branch: 1.0}
        else:
            weights = {default_branch: 0.50, branch: 0.50}
        blended = rank_blend_scores(filtered_outputs, weights, rrf_k=int(cfg.get("rrf_k", 60)))
        return RouterDecision(
            chosen_branch=branch,
            branch_weights=weights,
            route_reason="clean_trend_override" if branch == "trend_uncluttered" else "theme_ai_override",
            risk_off_score=risk_off_score,
            trend_score=trend_score,
            theme_score=theme_score,
            clutter_score=clutter_score,
            confidence=score,
            fallback_used=False,
            blocked_branches=blocked,
            debug_info={
                **debug,
                "trend_override_score": trend_override_score,
                "theme_ai_override_score": theme_ai_override_score,
                "trend_hard_cap_ok": trend_hard_cap_ok,
                "theme_ai_hard_cap_ok": theme_ai_hard_cap_ok,
                "blend_branches": list(weights),
                "blend_weights": weights,
                "rrf_k": int(cfg.get("rrf_k", 60)),
                "post_blend_top5": blended.head(5)["stock_id"].astype(str).tolist() if "stock_id" in blended.columns else [],
            },
        )

    trend = metrics.get("trend_uncluttered", {})
    trend_hard_cap_ok = bool(
        "trend_uncluttered" in filtered_outputs
        and _safe_float(trend.get("sigma_rank_mean_top5"), _safe_float(trend.get("mean_sigma20_rank"), 0.5)) <= float(cfg.get("trend_sigma_cap_q", 0.90))
        and _safe_float(trend.get("amp_rank_mean_top5"), _safe_float(trend.get("mean_amp20_rank"), 0.5)) <= float(cfg.get("trend_amp_cap_q", 0.90))
        and _safe_float(trend.get("high_risk_chaser_count"), 0.0) <= float(cfg.get("trend_high_risk_chaser_cap", 1))
        and not crash_mode
    )
    breadth = _clip01(state.get("market_breadth_5d", 0.5))
    market_ret20_component = _clip01(0.5 + _safe_float(state.get("market_ret20"), 0.0) / 0.12)
    trend_override_score = float(
        0.28 * _clip01(trend_score)
        + 0.18 * _positive_norm(trend.get("branch_score_margin_top5_vs_top10"), 0.05)
        + 0.10 * _positive_norm(trend.get("branch_score_dispersion_top10"), 0.05)
        + 0.14 * _clip01(trend.get("consensus_support"))
        + 0.12 * breadth
        + 0.08 * market_ret20_component
        + 0.14 * (1.0 - _clip01(clutter_score))
        - 0.04 * _safe_float(trend.get("high_risk_chaser_count"), 0.0)
    )

    ai = metrics.get("ai_hardware_mainline_v1", {})
    overlap = min(_safe_float(ai.get("overlap_with_grr_top5")) + _safe_float(ai.get("overlap_with_trend_top5")), 5.0) / 5.0
    recent = state.get("recent_branch_oof_strength", {}) or {}
    theme_ai_hard_cap_ok = bool(
        "ai_hardware_mainline_v1" in filtered_outputs
        and _safe_float(ai.get("sigma_rank_mean_top5"), _safe_float(ai.get("mean_sigma20_rank"), 0.5)) <= float(cfg.get("theme_ai_sigma_cap_q", 0.90))
        and _safe_float(ai.get("amp_rank_mean_top5"), _safe_float(ai.get("mean_amp20_rank"), 0.5)) <= float(cfg.get("theme_ai_amp_cap_q", 0.90))
        and _safe_float(ai.get("high_risk_chaser_count"), 0.0) <= float(cfg.get("theme_ai_high_risk_chaser_cap", 1))
        and not crash_mode
    )
    theme_ai_override_score = float(
        0.20 * _positive_norm(ai.get("branch_score_margin_top5_vs_top10"), 0.05)
        + 0.14 * _positive_norm(ai.get("branch_score_dispersion_top10"), 0.05)
        + 0.14 * _clip01(ai.get("consensus_support"))
        + 0.14 * _clip01(overlap)
        + 0.12 * _clip01(0.5 + _safe_float(ai.get("selected_ret20_strength"), 0.0) / 0.12)
        + 0.08 * _clip01(0.5 + _safe_float(state.get("amount_chg_5d"), 0.0))
        + 0.10 * _clip01((1.0 + _safe_float(recent.get("ai_hardware_mainline_v1"), 0.0)) / 2.0)
        + 0.08 * _clip01(theme_score)
        - 0.04 * _safe_float(ai.get("high_risk_chaser_count"), 0.0)
    )

    debug["trend_override_score"] = trend_override_score
    debug["theme_ai_override_score"] = theme_ai_override_score
    debug["trend_hard_cap_ok"] = trend_hard_cap_ok
    debug["theme_ai_hard_cap_ok"] = theme_ai_hard_cap_ok

    if (
        bool(cfg.get("trend_override_enabled", True))
        and trend_hard_cap_ok
        and clutter_score <= float(cfg.get("trend_clutter_max", 0.65))
        and trend_override_score >= float(cfg.get("trend_override_threshold", 0.58))
    ):
        return finish("trend_uncluttered", "trend_override", trend_override_score, float(cfg.get("trend_soft_blend_band", 0.06)))

    if (
        bool(cfg.get("theme_ai_override_enabled", True))
        and theme_ai_hard_cap_ok
        and theme_ai_override_score >= float(cfg.get("theme_ai_override_threshold", 0.58))
    ):
        return finish("ai_hardware_mainline_v1", "theme_ai_override", theme_ai_override_score, float(cfg.get("theme_ai_soft_blend_band", 0.06)))

    return RouterDecision(
        chosen_branch=default_branch,
        branch_weights={default_branch: 1.0},
        route_reason="default_grr_tail_guard",
        risk_off_score=risk_off_score,
        trend_score=trend_score,
        theme_score=theme_score,
        clutter_score=clutter_score,
        confidence=0.0,
        fallback_used=True,
        blocked_branches=blocked,
        debug_info={**debug, "blend_branches": [], "blend_weights": {}, "rrf_k": int(cfg.get("rrf_k", 60)), "post_blend_top5": []},
    )


def _prepared_ranked_frame(output: Any, rrf_k: int = 60, top_k: int = 20) -> pd.DataFrame:
    frame = _as_frame(output)
    if frame.empty or "stock_id" not in frame.columns:
        return pd.DataFrame()
    scored, _ = _score_source_and_values(frame, rrf_k=rrf_k)
    scored = _with_risk_feature_columns(scored).head(min(top_k, len(scored))).copy()
    scored["rank_score"] = 1.0 / (float(rrf_k) + scored["rank_in_branch"].astype(float))
    raw = _safe_series(scored, "_branch_rank_score", 0.0)
    denom = float(raw.max() - raw.min())
    scored["candidate_score_norm"] = 0.5 if denom <= 1e-12 else (raw - raw.min()) / denom
    if "grr_consensus_norm" in scored.columns:
        scored["consensus_support"] = _safe_series(scored, "grr_consensus_norm", 0.0)
    else:
        scored["consensus_support"] = 0.0
    if "liq_rank" in scored.columns:
        scored["_liq_rank_for_router"] = _safe_series(scored, "liq_rank", 0.5)
    else:
        scored["_liq_rank_for_router"] = _rank_pct(_safe_series(scored, "median_amount20", 0.0), ascending=True)
    return scored


def _default_keep_scores(default_output: Any, rrf_k: int = 60) -> tuple[list[str], dict[str, float], dict[str, Any]]:
    default_raw = _as_frame(default_output)
    default = _prepared_ranked_frame(default_output, rrf_k=rrf_k, top_k=max(30, len(default_raw)))
    if "final_selected_flag" in default.columns and pd.Series(default["final_selected_flag"]).astype(bool).any():
        selected = default[pd.Series(default["final_selected_flag"]).astype(bool)].copy()
        if "final_selected_order" in selected.columns:
            selected["final_selected_order"] = pd.to_numeric(selected["final_selected_order"], errors="coerce").fillna(999)
            selected = selected.sort_values("final_selected_order")
        top5 = selected.head(min(5, len(selected))).copy()
    elif "post_filter_selected_top5" in default.columns and pd.Series(default["post_filter_selected_top5"]).astype(bool).any():
        top5 = default[pd.Series(default["post_filter_selected_top5"]).astype(bool)].head(min(5, len(default))).copy()
    else:
        top5 = default.head(min(5, len(default))).copy()
    keep = {}
    debug = {}
    for _, row in top5.iterrows():
        stock_id = str(row["stock_id"])
        rank_score = 1.0 / (float(rrf_k) + float(row.get("rank_in_branch", 5.0)))
        raw_norm = _safe_float(row.get("candidate_score_norm"), 0.5)
        consensus = _safe_float(row.get("consensus_support"), 0.0)
        risk_rank = (
            0.4 * _safe_float(row.get("_sigma_rank_for_router"), 0.5)
            + 0.3 * _safe_float(row.get("_amp_rank_for_router"), 0.5)
            + 0.3 * _safe_float(row.get("_drawdown_rank_for_router"), 0.5)
        )
        score = (
            1.00 * rank_score
            + 0.08 * raw_norm
            + 0.08 * consensus
            - 0.03 * risk_rank
            - 0.02 * _safe_float(row.get("_drawdown_rank_for_router"), 0.5)
        )
        keep[stock_id] = float(score)
        debug[stock_id] = {
            "rank_score": float(rank_score),
            "raw_norm": raw_norm,
            "consensus": consensus,
            "risk_rank": float(risk_rank),
            "keep_score": float(score),
        }
    return top5.get("stock_id", pd.Series(dtype=str)).astype(str).tolist(), keep, debug


def _window_snapshot_metrics(branch_outputs: dict[str, Any], market_state: dict[str, Any], rrf_k: int) -> dict[str, dict[str, float]]:
    return _rank_risks(_branch_metrics(branch_outputs, {**market_state, "rrf_k": rrf_k}))


def _trend_window_score(metrics: dict[str, dict[str, float]], market_state: dict[str, Any]) -> float:
    current = metrics.get("current_aggressive", {})
    trend = metrics.get("trend_uncluttered", {})
    return float(
        0.20 * _clip01(_safe_float(current.get("risk_rank_mean_top5"), 0.5))
        + 0.15 * _clip01(_safe_float(current.get("sigma_rank_mean_top5"), 0.5))
        + 0.15 * _clip01(_safe_float(current.get("amp_rank_mean_top5"), 0.5))
        + 0.18 * _positive_norm(trend.get("branch_score_margin_top5_vs_top10"), 0.12)
        + 0.12 * _positive_norm(trend.get("branch_score_dispersion_top10"), 0.10)
        + 0.14 * (1.0 - _clip01(market_state.get("clutter_score", 0.0)))
        - 0.20 * float(bool(market_state.get("crash_mode", False)))
    )


def _theme_ai_window_score(metrics: dict[str, dict[str, float]], market_state: dict[str, Any]) -> float:
    ai = metrics.get("ai_hardware_mainline_v1", {})
    grr = metrics.get("grr_tail_guard", {})
    current = metrics.get("current_aggressive", {})
    return float(
        0.20 * _clip01(grr.get("branch_rank_entropy_top20"))
        + 0.18 * _positive_norm(ai.get("branch_score_margin_top5_vs_top10"), 4.0)
        + 0.14 * _positive_norm(ai.get("branch_score_dispersion_top10"), 2.5)
        + 0.14 * (1.0 - _clip01(ai.get("consensus_support")))
        + 0.12 * _clip01(0.5 - _safe_float(current.get("selected_ret20_strength"), 0.0) / 0.12)
        + 0.10 * _clip01(1.0 - _safe_float(ai.get("high_risk_chaser_count"), 0.0) / 2.0)
        - 0.25 * float(bool(market_state.get("crash_mode", False)))
    )


def _hard_veto(row: pd.Series, prefix: str, cfg: dict[str, Any]) -> dict[str, bool]:
    return {
        "sigma_cap": _safe_float(row.get("_sigma_rank_for_router"), 0.5) > float(cfg.get(f"{prefix}_sigma_cap_q", 0.90)),
        "amp_cap": _safe_float(row.get("_amp_rank_for_router"), 0.5) > float(cfg.get(f"{prefix}_amp_cap_q", 0.90)),
        "drawdown_cap": _safe_float(row.get("_drawdown_rank_for_router"), 0.5) > float(cfg.get(f"{prefix}_drawdown_cap_q", 0.90)),
        "liquidity_floor": _safe_float(row.get("_liq_rank_for_router"), 0.5) < float(cfg.get("liquidity_min_rank", 0.05)),
        "high_risk_chaser": bool(row.get("high_risk_chaser_flag", False)),
    }


def build_overlay_candidates(
    default_output: Any,
    specialist_output: Any,
    specialist_name: str,
    market_state: dict[str, Any],
    config: dict[str, Any],
) -> list[OverlayCandidate]:
    cfg = _v2b_cfg(config)
    rrf_k = int(cfg.get("rrf_k", 60))
    default_top5, keep_scores, _ = _default_keep_scores(default_output, rrf_k=rrf_k)
    default_set = set(default_top5)
    specialist = _prepared_ranked_frame(specialist_output, rrf_k=rrf_k, top_k=20)
    if specialist.empty:
        return []
    branch_outputs = {
        cfg.get("default_branch", "grr_tail_guard"): default_output,
        specialist_name: specialist_output,
    }
    metrics = _window_snapshot_metrics(branch_outputs, market_state, rrf_k)
    branch_metric = metrics.get(specialist_name, {})
    prefix = "trend" if specialist_name == "trend_uncluttered" else "theme_ai"
    margin_norm = _positive_norm(branch_metric.get("branch_score_margin_top5_vs_top10"), 0.12 if prefix == "trend" else 4.0)
    dispersion_norm = _positive_norm(branch_metric.get("branch_score_dispersion_top10"), 0.10 if prefix == "trend" else 2.5)
    candidates: list[OverlayCandidate] = []
    weakest = min(keep_scores, key=keep_scores.get) if keep_scores else None
    weakest_score = keep_scores.get(weakest, 0.0) if weakest else 0.0
    for _, row in specialist.iterrows():
        stock_id = str(row["stock_id"])
        in_default = stock_id in default_set
        veto = _hard_veto(row, prefix, cfg)
        risk_rank = (
            0.4 * _safe_float(row.get("_sigma_rank_for_router"), 0.5)
            + 0.3 * _safe_float(row.get("_amp_rank_for_router"), 0.5)
            + 0.3 * _safe_float(row.get("_drawdown_rank_for_router"), 0.5)
        )
        independence = 0.10 if specialist_name == "ai_hardware_mainline_v1" and _safe_float(row.get("consensus_support"), 0.0) <= 0.20 else 0.0
        overlay_score = (
            1.00 * _safe_float(row.get("rank_score"), 0.0)
            + 0.16 * _safe_float(row.get("candidate_score_norm"), 0.5)
            + 0.12 * margin_norm
            + 0.08 * dispersion_norm
            + 0.08 * _safe_float(row.get("consensus_support"), 0.0)
            + independence
            - 0.04 * risk_rank
            - 0.08 * float(bool(row.get("high_risk_chaser_flag", False)))
        )
        candidates.append(
            OverlayCandidate(
                stock_id=stock_id,
                source_branch=specialist_name,
                source_rank=int(_safe_float(row.get("rank_in_branch"), 0.0)),
                source_score=_safe_float(row.get("_branch_rank_score"), 0.0),
                rank_score=_safe_float(row.get("rank_score"), 0.0),
                overlay_score=float(overlay_score),
                risk_features={
                    "sigma_rank": _safe_float(row.get("_sigma_rank_for_router"), 0.5),
                    "amp_rank": _safe_float(row.get("_amp_rank_for_router"), 0.5),
                    "drawdown_rank": _safe_float(row.get("_drawdown_rank_for_router"), 0.5),
                    "risk_rank": float(risk_rank),
                },
                liquidity_features={
                    "liq_rank": _safe_float(row.get("_liq_rank_for_router"), 0.5),
                    "median_amount20": _safe_float(row.get("median_amount20"), 0.0),
                },
                consensus_support=_safe_float(row.get("consensus_support"), 0.0),
                in_default_top5=in_default,
                replacement_target=weakest,
                replacement_gain_estimate=float(overlay_score - weakest_score),
                veto_flags=veto,
                debug_components={
                    "candidate_score_norm": _safe_float(row.get("candidate_score_norm"), 0.5),
                    "margin_norm": margin_norm,
                    "dispersion_norm": dispersion_norm,
                    "independence_bonus": independence,
                    "risk_penalty_small": 0.04 * risk_rank,
                },
            )
        )
    return candidates


def apply_candidate_overlay(
    default_output: Any,
    overlay_candidates: list[OverlayCandidate],
    market_state: dict[str, Any],
    config: dict[str, Any],
) -> OverlayDecision:
    cfg = _v2b_cfg(config)
    rrf_k = int(cfg.get("rrf_k", 60))
    default_top5, keep_scores, keep_debug = _default_keep_scores(default_output, rrf_k=rrf_k)
    final = list(default_top5)
    swaps: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    sorted_candidates = sorted(
        [c for c in overlay_candidates if not c.in_default_top5],
        key=lambda c: c.overlay_score,
        reverse=True,
    )
    for candidate in sorted_candidates:
        if len(swaps) >= int(cfg.get("max_total_swaps", 2)):
            rejected.append({**asdict(candidate), "reject_reason": "total_swap_limit"})
            continue
        prefix = "trend" if candidate.source_branch == "trend_uncluttered" else "theme_ai"
        max_swaps = int(cfg.get(f"{prefix}_max_swaps", 1))
        min_gap = float(cfg.get(f"{prefix}_min_replacement_gap", 0.08))
        if source_counts.get(candidate.source_branch, 0) >= max_swaps:
            rejected.append({**asdict(candidate), "reject_reason": "source_swap_limit"})
            continue
        if any(candidate.veto_flags.values()):
            rejected.append({**asdict(candidate), "reject_reason": "post_overlay_hard_veto"})
            continue
        available_targets = [stock for stock in final if stock not in {swap["added_stock"] for swap in swaps}]
        if not available_targets:
            rejected.append({**asdict(candidate), "reject_reason": "no_replacement_target"})
            continue
        target = min(available_targets, key=lambda stock: keep_scores.get(stock, 0.0))
        gap = candidate.overlay_score - keep_scores.get(target, 0.0)
        if gap < min_gap:
            rejected.append({**asdict(candidate), "reject_reason": "replacement_gap_below_threshold", "replacement_gap": float(gap)})
            continue
        idx = final.index(target)
        final[idx] = candidate.stock_id
        source_counts[candidate.source_branch] = source_counts.get(candidate.source_branch, 0) + 1
        swaps.append(
            {
                "removed_stock": target,
                "added_stock": candidate.stock_id,
                "source_branch": candidate.source_branch,
                "replacement_gap": float(gap),
                "candidate_overlay_score": candidate.overlay_score,
                "removed_keep_score": keep_scores.get(target, 0.0),
                "veto_flags": candidate.veto_flags,
                "replacement_target_reason": "weakest_default_keep_score",
            }
        )
    return OverlayDecision(
        final_top5=final,
        swaps=swaps,
        swap_count=len(swaps),
        source_branches_used=sorted(source_counts),
        overlay_reason="candidate_overlay" if swaps else "no_overlay_swap",
        rejected_candidates=rejected[:20],
        debug_info={
            "default_top5": default_top5,
            "default_keep_scores": keep_scores,
            "default_keep_score_components": keep_debug,
            "weakest_default_stock": min(keep_scores, key=keep_scores.get) if keep_scores else None,
            "replacement_target_reason": "weakest_default_keep_score",
            "post_overlay_hard_veto_only": True,
        },
    )


def route_branch_v2b_overlay(
    branch_outputs: dict[str, Any],
    market_state: dict[str, Any],
    config: dict[str, Any] | None,
) -> RouterDecision:
    cfg = _v2b_cfg(config)
    default_branch = str(cfg.get("default_branch", "grr_tail_guard"))
    blocked: list[str] = []
    filtered_outputs = {}
    illegal_filtered = False
    for branch, output in (branch_outputs or {}).items():
        if branch in ILLEGAL_RUNTIME_BRANCHES:
            illegal_filtered = True
            blocked.append(branch)
            continue
        if branch in LEGAL_BRANCHES:
            filtered_outputs[branch] = output
    if default_branch not in filtered_outputs:
        default_branch = "grr_tail_guard" if "grr_tail_guard" in filtered_outputs else sorted(filtered_outputs)[0] if filtered_outputs else "grr_tail_guard"
    state = compute_branch_state_features(pd.DataFrame(), filtered_outputs, market_state)
    rrf_k = int(cfg.get("rrf_k", 60))
    metrics = _window_snapshot_metrics(filtered_outputs, state, rrf_k)
    trend_window_score = _trend_window_score(metrics, state)
    theme_ai_window_score = _theme_ai_window_score(metrics, state)
    current_ret20_strength = _safe_float(metrics.get("current_aggressive", {}).get("selected_ret20_strength"), 0.0)
    trend_window_cap_ok = current_ret20_strength <= float(cfg.get("trend_current_ret20_max", 0.12))
    theme_ai_window_cap_ok = current_ret20_strength <= float(cfg.get("theme_ai_current_ret20_max", 0.02))
    debug = {
        "illegal_branch_filtered": illegal_filtered,
        "blocked_branches": blocked.copy(),
        "available_branches": sorted(filtered_outputs),
        "market_state": state,
        "branch_metrics": metrics,
        "trend_window_score": trend_window_score,
        "theme_ai_window_score": theme_ai_window_score,
        "current_aggressive_ret20_strength": current_ret20_strength,
        "trend_window_cap_ok": trend_window_cap_ok,
        "theme_ai_window_cap_ok": theme_ai_window_cap_ok,
        "full_tail_guard_rerank_used": False,
    }

    if bool(state.get("crash_mode", False)) and bool(cfg.get("crash_minrisk_enabled", True)) and "legal_minrisk_hardened" in filtered_outputs:
        return RouterDecision(
            chosen_branch="legal_minrisk_hardened",
            branch_weights={"legal_minrisk_hardened": 1.0},
            route_reason="crash_minrisk_rescue",
            risk_off_score=_safe_float(state.get("risk_off_score")),
            trend_score=_safe_float(state.get("trend_score")),
            theme_score=_safe_float(state.get("theme_score")),
            clutter_score=_safe_float(state.get("clutter_score")),
            confidence=1.0,
            fallback_used=False,
            blocked_branches=blocked,
            debug_info={**debug, "overlay_decision": {"final_top5": _top_candidates(_as_frame(filtered_outputs["legal_minrisk_hardened"]), 5)["stock_id"].astype(str).tolist(), "swaps": [], "swap_count": 0, "source_branches_used": []}},
        )

    overlay_candidates: list[OverlayCandidate] = []
    if (
        bool(cfg.get("trend_overlay_enabled", True))
        and "trend_uncluttered" in filtered_outputs
        and trend_window_score >= float(cfg.get("trend_window_threshold", 0.55))
        and _safe_float(state.get("clutter_score"), 0.0) <= float(cfg.get("trend_clutter_max", 0.70))
        and trend_window_cap_ok
        and not bool(state.get("crash_mode", False))
    ):
        overlay_candidates.extend(build_overlay_candidates(filtered_outputs[default_branch], filtered_outputs["trend_uncluttered"], "trend_uncluttered", state, cfg))
    if (
        bool(cfg.get("theme_ai_overlay_enabled", True))
        and "ai_hardware_mainline_v1" in filtered_outputs
        and theme_ai_window_score >= float(cfg.get("theme_ai_window_threshold", 0.62))
        and theme_ai_window_cap_ok
        and not bool(state.get("crash_mode", False))
    ):
        overlay_candidates.extend(build_overlay_candidates(filtered_outputs[default_branch], filtered_outputs["ai_hardware_mainline_v1"], "ai_hardware_mainline_v1", state, cfg))

    overlay = apply_candidate_overlay(filtered_outputs.get(default_branch), overlay_candidates, state, cfg)
    route_reason = "candidate_overlay" if overlay.swap_count else "default_grr_tail_guard"
    used = overlay.source_branches_used
    return RouterDecision(
        chosen_branch="grr_tail_guard_overlay" if overlay.swap_count else default_branch,
        branch_weights={default_branch: 1.0},
        route_reason=route_reason,
        risk_off_score=_safe_float(state.get("risk_off_score")),
        trend_score=_safe_float(state.get("trend_score")),
        theme_score=_safe_float(state.get("theme_score")),
        clutter_score=_safe_float(state.get("clutter_score")),
        confidence=max(trend_window_score, theme_ai_window_score),
        fallback_used=overlay.swap_count == 0,
        blocked_branches=blocked,
        debug_info={
            **debug,
            "overlay_decision": {
                "final_top5": overlay.final_top5,
                "swaps": overlay.swaps,
                "swap_count": overlay.swap_count,
                "source_branches_used": used,
                "overlay_reason": overlay.overlay_reason,
                "rejected_candidates": overlay.rejected_candidates,
                "debug_info": overlay.debug_info,
            },
        },
    )


def rank_blend_scores(branch_outputs: dict[str, Any], branch_weights: dict[str, float], rrf_k: int = 60) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    score = {}
    for branch, weight in branch_weights.items():
        if branch in ILLEGAL_RUNTIME_BRANCHES or weight <= 0:
            continue
        frame = _as_frame(branch_outputs.get(branch))
        if frame.empty or "stock_id" not in frame.columns:
            continue
        col = _score_col(frame)
        ranked = frame.copy()
        if col:
            ranked[col] = pd.to_numeric(ranked[col], errors="coerce").fillna(0.0)
            ranked["_rank"] = ranked[col].rank(method="first", ascending=False)
        else:
            ranked["_rank"] = np.arange(1, len(ranked) + 1, dtype=np.float64)
        for stock_id, rank in zip(ranked["stock_id"].astype(str), ranked["_rank"]):
            score[stock_id] = score.get(stock_id, 0.0) + float(weight) * (1.0 / (float(rrf_k) + float(rank)))
        rows.append(ranked.drop(columns=["_rank"], errors="ignore"))

    if not rows:
        return pd.DataFrame(columns=["stock_id", "rank_blend_score"])
    union = pd.concat(rows, ignore_index=True).drop_duplicates("stock_id", keep="first")
    union["rank_blend_score"] = union["stock_id"].astype(str).map(score).fillna(0.0)
    union.attrs["score_col"] = "rank_blend_score"
    return union.sort_values("rank_blend_score", ascending=False).reset_index(drop=True)


def hedge_weight_trace(
    window_branch_scores: list[dict[str, float]],
    branches: list[str],
    eta: float = 0.5,
    prior: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    if prior:
        weights = np.asarray([max(_safe_float(prior.get(branch), 0.0), 0.0) for branch in branches], dtype=np.float64)
        weights = weights / weights.sum() if weights.sum() > 1e-12 else np.full(len(branches), 1.0 / max(len(branches), 1))
    else:
        weights = np.full(len(branches), 1.0 / max(len(branches), 1), dtype=np.float64)

    trace = []
    for idx, scores in enumerate(window_branch_scores):
        before = {branch: float(weight) for branch, weight in zip(branches, weights)}
        losses = np.asarray([-_safe_float(scores.get(branch), 0.0) for branch in branches], dtype=np.float64)
        updated = np.maximum(weights, 1e-6) * np.exp(-float(eta) * losses)
        weights = updated / updated.sum() if updated.sum() > 1e-12 else np.full(len(branches), 1.0 / max(len(branches), 1))
        after = {branch: float(weight) for branch, weight in zip(branches, weights)}
        trace.append(
            {
                "window_index": idx,
                "hedge_weights_before_decision": before,
                "hedge_weights_after_update": after,
            }
        )
    return trace
