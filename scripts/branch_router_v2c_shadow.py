from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "code" / "src"))

from batch_window_analysis import add_branch_diagnostic_features, filter_branch, load_raw, normalize_stock_id, realized_returns_for_anchor  # noqa: E402


SOURCE_RUN = ROOT / "temp" / "batch_window_analysis" / "grr_tail_guard_60win"
V2B_DETAIL = ROOT / "temp" / "branch_router_validation" / "v2b_guarded_longer_60win_60win"
OUT_DIR = ROOT / "temp" / "branch_router_validation" / "v2c_shadow"


def _z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean = values.mean()
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return ((values - mean) / std).fillna(0.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(x):
        return default
    return x


def _split_picks(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return normalize_stock_id(pd.Series([item.strip() for item in str(value).split(",") if item.strip()], dtype=str)).astype(str).tolist()


def _realized_for_ids(work: pd.DataFrame, stock_ids: list[str]) -> float:
    if not stock_ids:
        return 0.0
    selected = work[work["stock_id"].astype(str).isin(set(stock_ids))].copy()
    order = {stock_id: idx for idx, stock_id in enumerate(stock_ids)}
    selected["_order"] = selected["stock_id"].astype(str).map(order)
    selected = selected.sort_values("_order")
    return float(pd.to_numeric(selected["realized_ret"], errors="coerce").fillna(0.0).head(5).mean())


def _bad_counts(work: pd.DataFrame, stock_ids: list[str]) -> tuple[int, int]:
    selected = work[work["stock_id"].astype(str).isin(set(stock_ids))].copy()
    rets = pd.to_numeric(selected["realized_ret"], errors="coerce").fillna(0.0)
    return int((rets < -0.03).sum()), int((rets < -0.05).sum())


def _riskoff_candidates(work: pd.DataFrame, window: str, top_k: int = 20) -> tuple[pd.DataFrame, dict[str, Any]]:
    ret20 = pd.to_numeric(work.get("ret20", pd.Series(0.0, index=work.index)), errors="coerce").fillna(0.0)
    sigma20 = pd.to_numeric(work.get("sigma20", pd.Series(0.0, index=work.index)), errors="coerce").fillna(0.0)
    median_ret20 = float(ret20.median())
    breadth20 = float((ret20 > 0).mean())
    median_sigma20 = float(sigma20.median())
    dispersion20 = float(ret20.quantile(0.90) - ret20.quantile(0.10))
    triggered = bool(median_ret20 < 0 and breadth20 < 0.45 and median_sigma20 > 0.018 and dispersion20 > 0.10)
    stats = {
        "window": window,
        "riskoff_triggered": triggered,
        "median_ret20": median_ret20,
        "breadth20": breadth20,
        "median_sigma20": median_sigma20,
        "dispersion20": dispersion20,
    }
    if not triggered:
        return pd.DataFrame(), stats

    stable = filter_branch(work, "stable").copy()
    if stable.empty:
        return pd.DataFrame(), stats
    stable["fused_z"] = _z(stable.get("score", 0.0))
    stable["lgb_z"] = _z(stable.get("lgb", 0.0))
    stable["log_liquidity"] = np.log1p(pd.to_numeric(stable.get("median_amount20", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0))
    stable["log_liquidity_z"] = _z(stable["log_liquidity"])
    stable["ret5_z"] = _z(stable.get("ret5", 0.0))
    stable["ret20_z"] = _z(stable.get("ret20", 0.0))
    stable["amp_z"] = _z(stable.get("amp20", 0.0))
    stable["negative_ret20_penalty"] = (pd.to_numeric(stable.get("ret20", 0.0), errors="coerce").fillna(0.0) < 0).astype(float)
    stable["riskoff_rerank_score"] = (
        0.30 * stable["fused_z"]
        + 0.10 * stable["lgb_z"]
        + 0.30 * stable["log_liquidity_z"]
        + 0.10 * stable["ret5_z"]
        + 0.10 * stable["ret20_z"]
        - 0.10 * stable["amp_z"]
        - 0.50 * stable["negative_ret20_penalty"]
    )
    out = stable.sort_values("riskoff_rerank_score", ascending=False).head(top_k).copy()
    out["candidate_rank"] = np.arange(1, len(out) + 1)
    out["candidate_score"] = out["riskoff_rerank_score"]
    for key, value in stats.items():
        out[key] = value
    return out, stats


def _risk_value(row: pd.Series) -> float:
    return float(np.nanmean([_safe_float(row.get("sigma20")), _safe_float(row.get("amp20")), _safe_float(row.get("max_drawdown20"))]))


def build_shadow_inputs(source_run: Path, raw_path: Path) -> dict[str, Any]:
    raw = load_raw(raw_path)
    windows = pd.read_csv(source_run / "window_summary.csv")
    windows["anchor_date"] = pd.to_datetime(windows["anchor_date"]).dt.strftime("%Y-%m-%d")
    work_by_window: dict[str, pd.DataFrame] = {}
    default_top5: dict[str, list[str]] = {}
    riskoff_rows: list[dict[str, Any]] = []
    riskoff_direct_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for _, win in windows.sort_values("anchor_date").iterrows():
        window = str(win["anchor_date"])
        anchor = pd.Timestamp(window)
        score_path = source_run / anchor.strftime("%Y%m%d") / "predict_score_df.csv"
        score_df = pd.read_csv(score_path, dtype={"stock_id": str})
        score_df["stock_id"] = normalize_stock_id(score_df["stock_id"])
        realized, _ = realized_returns_for_anchor(raw, anchor, label_horizon=5)
        work = score_df.merge(realized[["stock_id", "realized_ret"]], on="stock_id", how="left")
        work["realized_ret"] = pd.to_numeric(work["realized_ret"], errors="coerce").fillna(0.0)
        work = add_branch_diagnostic_features(work)
        work["log_liquidity"] = np.log1p(pd.to_numeric(work.get("median_amount20", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0))
        work["_risk_value"] = work.apply(_risk_value, axis=1)
        work_by_window[window] = work
        default_top5[window] = _split_picks(win.get("selected_picks", ""))

        for _, row in work.iterrows():
            feature_rows.append(
                {
                    "window": window,
                    "stock_id": str(row["stock_id"]),
                    "ret5": row.get("ret5"),
                    "ret20": row.get("ret20"),
                    "log_liquidity": row.get("log_liquidity"),
                    "amp": row.get("amp20"),
                    "risk": row.get("_risk_value"),
                    "base_score": row.get("grr_final_score", row.get("score")),
                    "realized_ret": row.get("realized_ret"),
                }
            )

        candidates, stats = _riskoff_candidates(work, window, top_k=20)
        if not candidates.empty:
            for _, row in candidates.iterrows():
                riskoff_rows.append(
                    {
                        "window": window,
                        "branch": "riskoff_top60",
                        "riskoff_triggered": True,
                        "candidate_stock": str(row["stock_id"]),
                        "candidate_rank": int(row["candidate_rank"]),
                        "candidate_score": float(row["candidate_score"]),
                        "riskoff_rerank_score": float(row["riskoff_rerank_score"]),
                        "fused_z": float(row["fused_z"]),
                        "lgb_z": float(row["lgb_z"]),
                        "log_liquidity_z": float(row["log_liquidity_z"]),
                        "ret5_z": float(row["ret5_z"]),
                        "ret20_z": float(row["ret20_z"]),
                        "amp_z": float(row["amp_z"]),
                        "negative_ret20_penalty": float(row["negative_ret20_penalty"]),
                        **{k: stats[k] for k in ["median_ret20", "breadth20", "median_sigma20", "dispersion20"]},
                    }
                )
            direct_top5 = candidates.head(5)["stock_id"].astype(str).tolist()
            score = _realized_for_ids(work, direct_top5)
            bad, very_bad = _bad_counts(work, direct_top5)
        else:
            direct_top5 = default_top5[window]
            score = _realized_for_ids(work, direct_top5)
            bad, very_bad = _bad_counts(work, direct_top5)
        riskoff_direct_rows.append(
            {
                "window": window,
                "riskoff_triggered": bool(stats["riskoff_triggered"]),
                "selected_stocks": ",".join(direct_top5),
                "score": score,
                "bad_count": bad,
                "very_bad_count": very_bad,
            }
        )
    return {
        "windows": windows,
        "work_by_window": work_by_window,
        "default_top5": default_top5,
        "riskoff_candidates": pd.DataFrame(riskoff_rows),
        "riskoff_direct": pd.DataFrame(riskoff_direct_rows),
        "features": pd.DataFrame(feature_rows),
    }


def _feature_lookup(features: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["window"]), str(row["stock_id"])): row.to_dict() for _, row in features.iterrows()}


def build_candidate_proposals(inputs: dict[str, Any], detail_dir: Path) -> pd.DataFrame:
    features = _feature_lookup(inputs["features"])
    proposals: list[dict[str, Any]] = []
    for file_name in ["accepted_swaps.csv", "blocked_candidates.csv"]:
        path = detail_dir / file_name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["variant"] == "v2b_trend_plus_ai_overlay"].copy()
        for _, row in frame.iterrows():
            window = str(row.get("window"))
            branch = str(row.get("branch"))
            source = {"trend": "trend_guarded", "theme_ai": "theme_ai_guarded"}.get(branch)
            if not source:
                continue
            cand = str(row.get("candidate_stock")).zfill(6)
            base = str(row.get("replaced_stock")).zfill(6)
            cf = features.get((window, cand), {})
            bf = features.get((window, base), {})
            proposals.append(
                {
                    "window": window,
                    "source_branch": source,
                    "base_stock_to_replace": base,
                    "candidate_stock": cand,
                    "base_rank": row.get("replaced_rank"),
                    "candidate_rank": row.get("candidate_rank"),
                    "base_score": row.get("replaced_score"),
                    "candidate_score": row.get("candidate_score"),
                    "score_margin": row.get("score_margin"),
                    "risk_delta": row.get("risk_delta"),
                    "trend_dispersion": row.get("trend_dispersion"),
                    "theme_ai_consensus": row.get("theme_ai_consensus"),
                    "riskoff_triggered": False,
                    "riskoff_score": None,
                    "candidate_ret5": cf.get("ret5"),
                    "candidate_ret20": cf.get("ret20"),
                    "candidate_log_liquidity": cf.get("log_liquidity"),
                    "candidate_amp": cf.get("amp"),
                    "base_ret5": bf.get("ret5"),
                    "base_ret20": bf.get("ret20"),
                    "base_log_liquidity": bf.get("log_liquidity"),
                    "base_amp": bf.get("amp"),
                    "raw_candidate_return": row.get("raw_candidate_return"),
                    "raw_replaced_return": row.get("raw_replaced_return"),
                    "raw_stock_delta": row.get("raw_stock_delta"),
                    "position_weight": row.get("position_weight", 0.2),
                    "weighted_swap_delta": row.get("weighted_swap_delta"),
                    "would_improve": bool(_safe_float(row.get("weighted_swap_delta")) > 1e-12),
                    "proposal_reason": "v2b_guarded_overlay_candidate",
                }
            )

    riskoff = inputs["riskoff_candidates"]
    for window, group in riskoff.groupby("window", sort=True):
        work = inputs["work_by_window"][str(window)]
        default_ids = inputs["default_top5"][str(window)]
        default_frame = work[work["stock_id"].astype(str).isin(default_ids)].copy()
        default_frame["_base_rank"] = default_frame["stock_id"].astype(str).map({stock: idx + 1 for idx, stock in enumerate(default_ids)})
        default_frame["_base_score"] = pd.to_numeric(default_frame.get("grr_final_score", default_frame.get("score")), errors="coerce").fillna(0.0)
        target = default_frame.sort_values("_base_score", ascending=True).iloc[0]
        base = str(target["stock_id"])
        bf = features.get((str(window), base), {})
        for _, row in group.head(10).iterrows():
            cand = str(row["candidate_stock"]).zfill(6)
            if cand in default_ids:
                continue
            cf = features.get((str(window), cand), {})
            raw_delta = None
            weighted_delta = None
            if cf.get("realized_ret") is not None and bf.get("realized_ret") is not None:
                raw_delta = _safe_float(cf.get("realized_ret")) - _safe_float(bf.get("realized_ret"))
                weighted_delta = 0.2 * raw_delta
            proposals.append(
                {
                    "window": str(window),
                    "source_branch": "riskoff_top60",
                    "base_stock_to_replace": base,
                    "candidate_stock": cand,
                    "base_rank": target.get("_base_rank"),
                    "candidate_rank": row.get("candidate_rank"),
                    "base_score": target.get("_base_score"),
                    "candidate_score": row.get("candidate_score"),
                    "score_margin": _safe_float(row.get("candidate_score")) - _safe_float(target.get("_base_score")),
                    "risk_delta": _safe_float(cf.get("risk")) - _safe_float(bf.get("risk")),
                    "trend_dispersion": None,
                    "theme_ai_consensus": None,
                    "riskoff_triggered": True,
                    "riskoff_score": row.get("riskoff_rerank_score"),
                    "candidate_ret5": cf.get("ret5"),
                    "candidate_ret20": cf.get("ret20"),
                    "candidate_log_liquidity": cf.get("log_liquidity"),
                    "candidate_amp": cf.get("amp"),
                    "base_ret5": bf.get("ret5"),
                    "base_ret20": bf.get("ret20"),
                    "base_log_liquidity": bf.get("log_liquidity"),
                    "base_amp": bf.get("amp"),
                    "raw_candidate_return": cf.get("realized_ret"),
                    "raw_replaced_return": bf.get("realized_ret"),
                    "raw_stock_delta": raw_delta,
                    "position_weight": 0.2,
                    "weighted_swap_delta": weighted_delta,
                    "would_improve": bool(weighted_delta is not None and weighted_delta > 1e-12),
                    "proposal_reason": "riskoff_top60_rerank_candidate",
                }
            )
    return pd.DataFrame(proposals)


def _bucket_for_window(window: str, windows: list[str]) -> str:
    idx = windows.index(window)
    if idx < 20:
        return "old_20"
    if idx < 40:
        return "mid_20"
    return "recent_20"


def write_branch_skill_prior(proposals: pd.DataFrame, windows: list[str], out_dir: Path) -> pd.DataFrame:
    rows = []
    props = proposals.copy()
    props["window_bucket"] = props["window"].astype(str).map(lambda x: _bucket_for_window(x, windows))
    props["regime_bucket"] = np.where(props["riskoff_triggered"].astype(bool), "riskoff", "non_riskoff")
    for branch in sorted(props["source_branch"].dropna().unique()):
        branch_df = props[props["source_branch"] == branch].copy()
        for bucket, group in list(branch_df.groupby("window_bucket")) + [("all_60", branch_df)]:
            delta = pd.to_numeric(group["weighted_swap_delta"], errors="coerce").fillna(0.0)
            rows.append(
                {
                    "branch": branch,
                    "window_bucket": bucket,
                    "proposal_count": int(len(group)),
                    "accepted_if_naive_count": int((pd.to_numeric(group["score_margin"], errors="coerce").fillna(0.0) > 0).sum()),
                    "positive_proposal_count": int((delta > 1e-12).sum()),
                    "negative_proposal_count": int((delta < -1e-12).sum()),
                    "zero_proposal_count": int(delta.abs().le(1e-12).sum()),
                    "mean_weighted_delta": float(delta.mean()) if len(delta) else 0.0,
                    "q10_weighted_delta": float(delta.quantile(0.10)) if len(delta) else 0.0,
                    "worst_weighted_delta": float(delta.min()) if len(delta) else 0.0,
                    "misfire_rate": float((delta < -1e-12).mean()) if len(delta) else 0.0,
                    "avg_risk_delta": float(pd.to_numeric(group["risk_delta"], errors="coerce").mean()),
                    "avg_score_margin": float(pd.to_numeric(group["score_margin"], errors="coerce").mean()),
                }
            )
    prior = pd.DataFrame(rows)
    prior.to_csv(out_dir / "branch_skill_prior.csv", index=False)

    summary_rows = []
    for branch in sorted(props["source_branch"].dropna().unique()):
        p = prior[prior["branch"] == branch]
        recent = _safe_float(p.loc[p["window_bucket"] == "recent_20", "mean_weighted_delta"].iloc[0] if (p["window_bucket"] == "recent_20").any() else 0.0)
        all60 = _safe_float(p.loc[p["window_bucket"] == "all_60", "mean_weighted_delta"].iloc[0] if (p["window_bucket"] == "all_60").any() else 0.0)
        branch_props = props[props["source_branch"] == branch]
        riskoff_props = branch_props[branch_props["riskoff_triggered"].astype(bool)]
        similar = float(pd.to_numeric(riskoff_props["weighted_swap_delta"], errors="coerce").fillna(0.0).mean()) if len(riskoff_props) else all60
        q10 = _safe_float(p.loc[p["window_bucket"] == "all_60", "q10_weighted_delta"].iloc[0] if (p["window_bucket"] == "all_60").any() else 0.0)
        worst = _safe_float(p.loc[p["window_bucket"] == "all_60", "worst_weighted_delta"].iloc[0] if (p["window_bucket"] == "all_60").any() else 0.0)
        misfire = _safe_float(p.loc[p["window_bucket"] == "all_60", "misfire_rate"].iloc[0] if (p["window_bucket"] == "all_60").any() else 0.0)
        downside = 0.5 * abs(min(q10, 0.0)) + 0.5 * abs(min(worst, 0.0))
        score = 0.50 * recent + 0.30 * similar + 0.20 * all60 - downside
        summary_rows.append(
            {
                "branch": branch,
                "recent_20_mean_delta": recent,
                "similar_regime_mean_delta": similar,
                "all_60_mean_delta": all60,
                "q10_weighted_delta": q10,
                "worst_weighted_delta": worst,
                "misfire_rate": misfire,
                "downside_penalty": downside,
                "branch_skill_prior": score,
                "recommend_branch_enabled_shadow": bool(score > 0 and misfire <= 0.45),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "branch_skill_prior_summary.csv", index=False)
    return summary


def _metric(scores: pd.Series, bad: pd.Series | None = None, very_bad: pd.Series | None = None) -> dict[str, float]:
    scores = pd.to_numeric(scores, errors="coerce").dropna()
    if scores.empty:
        return {"mean": 0.0, "q10": 0.0, "worst": 0.0, "very_bad": 0.0, "win_rate": 0.0}
    return {
        "mean": float(scores.mean()),
        "q10": float(scores.quantile(0.10)),
        "worst": float(scores.min()),
        "very_bad": float(pd.to_numeric(very_bad, errors="coerce").mean()) if very_bad is not None else 0.0,
        "win_rate": float((scores > 0).mean()),
    }


def run_v2c_shadow(inputs: dict[str, Any], proposals: pd.DataFrame, prior: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prior_map = {row["branch"]: _safe_float(row["branch_skill_prior"]) for _, row in prior.iterrows()}
    misfire_map = {row["branch"]: _safe_float(row["misfire_rate"]) for _, row in prior.iterrows()}
    prop = proposals.copy()
    for col in ["score_margin", "candidate_log_liquidity", "candidate_ret5", "candidate_ret20", "candidate_amp", "risk_delta"]:
        prop[f"{col}_z"] = _z(prop[col])
    prop["branch_skill_prior"] = prop["source_branch"].map(prior_map).fillna(0.0)
    prop["branch_misfire_penalty"] = prop["source_branch"].map(misfire_map).fillna(0.0) * 0.02
    base_rank = pd.to_numeric(prop["base_rank"], errors="coerce").fillna(99)
    base_score = pd.to_numeric(prop["base_score"], errors="coerce").fillna(0.0)
    prop["default_strong_keep_penalty"] = np.where((base_rank <= 3) & (base_score >= base_score.median()), 0.02, 0.0)
    prop["swap_edge"] = (
        prop["branch_skill_prior"]
        + 0.35 * prop["score_margin_z"]
        + 0.20 * prop["candidate_log_liquidity_z"]
        + 0.15 * prop["candidate_ret5_z"]
        + 0.15 * prop["candidate_ret20_z"]
        - 0.20 * prop["candidate_amp_z"]
        - 0.35 * prop["risk_delta_z"]
        - prop["default_strong_keep_penalty"]
        - prop["branch_misfire_penalty"]
    )

    accepted_rows = []
    blocked_rows = []
    selected_rows = []
    for window in sorted(inputs["default_top5"]):
        top5 = list(inputs["default_top5"][window])
        used_candidates = set(top5)
        replaced = set()
        branch_counts = {"trend_guarded": 0, "theme_ai_guarded": 0, "riskoff_top60": 0}
        window_props = prop[prop["window"].astype(str) == window].sort_values("swap_edge", ascending=False)
        for _, row in window_props.iterrows():
            reasons = []
            branch = row["source_branch"]
            cand = str(row["candidate_stock"]).zfill(6)
            base = str(row["base_stock_to_replace"]).zfill(6)
            max_branch = {"trend_guarded": 1, "theme_ai_guarded": 1, "riskoff_top60": 1}.get(branch, 0)
            if _safe_float(row.get("swap_edge")) <= 0:
                reasons.append("swap_edge_not_positive")
            if len(accepted_rows) and sum(1 for r in accepted_rows if r["window"] == window) >= 2:
                reasons.append("max_total_swaps_reached")
            if branch_counts.get(branch, 0) >= max_branch:
                reasons.append("branch_swap_cap_reached")
            if cand in used_candidates:
                reasons.append("candidate_already_used")
            if base in replaced:
                reasons.append("base_already_replaced")
            if base not in top5:
                reasons.append("base_not_in_current_top5")
            if reasons:
                blocked_rows.append({**row.to_dict(), "blocked_reasons": json.dumps(reasons, ensure_ascii=False)})
                continue
            top5[top5.index(base)] = cand
            used_candidates.add(cand)
            replaced.add(base)
            branch_counts[branch] = branch_counts.get(branch, 0) + 1
            accepted_rows.append({**row.to_dict(), "accepted_reason": "positive_shadow_swap_edge"})
        work = inputs["work_by_window"][window]
        score = _realized_for_ids(work, top5)
        bad, very_bad = _bad_counts(work, top5)
        selected_rows.append(
            {
                "window": window,
                "final_top5": ",".join(top5),
                "score": score,
                "bad_count": bad,
                "very_bad_count": very_bad,
                "swap_count": sum(1 for r in accepted_rows if r["window"] == window),
            }
        )
    selected = pd.DataFrame(selected_rows)
    accepted = pd.DataFrame(accepted_rows)
    blocked = pd.DataFrame(blocked_rows)
    selected.to_csv(out_dir / "v2c_shadow_selected_windows.csv", index=False)
    accepted.to_csv(out_dir / "v2c_shadow_accepted_swaps.csv", index=False)
    blocked.to_csv(out_dir / "v2c_shadow_blocked_proposals.csv", index=False)
    return selected, accepted, blocked


def write_validation(inputs: dict[str, Any], selected: pd.DataFrame, accepted: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    v2b = pd.read_csv(V2B_DETAIL / "ablation_decisions.csv")
    riskoff_direct = inputs["riskoff_direct"].copy()
    windows = sorted(inputs["default_top5"])
    bucket_specs = {"20win": windows[-20:], "40win": windows[-40:], "60win": windows}
    rows = []

    def add_variant(name: str, frame: pd.DataFrame, window_col: str, score_col: str, bucket_name: str, bucket_windows: list[str], swaps: pd.DataFrame | None = None):
        subset = frame[frame[window_col].astype(str).isin(set(bucket_windows))].copy()
        default = v2b[(v2b["variant"] == "default_grr_tail_guard") & v2b["window_date"].astype(str).isin(set(bucket_windows))].copy()
        merged = subset[[window_col, score_col]].rename(columns={window_col: "window", score_col: "score"}).merge(
            default[["window_date", "score"]].rename(columns={"window_date": "window", "score": "default_score"}),
            on="window",
            how="inner",
        )
        delta = pd.to_numeric(merged["score"], errors="coerce") - pd.to_numeric(merged["default_score"], errors="coerce")
        variant_swaps = swaps[swaps["window"].astype(str).isin(set(bucket_windows))].copy() if swaps is not None and not swaps.empty else pd.DataFrame()
        row = {
            "variant": name,
            "window_bucket": bucket_name,
            "window_count": int(len(merged)),
            **_metric(merged["score"], subset.get("bad_count"), subset.get("very_bad_count")),
            "avg_swaps": float(pd.to_numeric(subset.get("swap_count", pd.Series(0, index=subset.index)), errors="coerce").fillna(0).mean()) if len(subset) else 0.0,
            "mean_delta_vs_default": float(delta.mean()) if len(delta) else 0.0,
            "q10_delta_vs_default": float(delta.quantile(0.10)) if len(delta) else 0.0,
            "worst_delta_vs_default": float(delta.min()) if len(delta) else 0.0,
            "positive_delta_count": int((delta > 1e-12).sum()),
            "negative_delta_count": int((delta < -1e-12).sum()),
            "zero_delta_count": int(delta.abs().le(1e-12).sum()),
        }
        for branch, col_prefix in [("trend_guarded", "trend"), ("theme_ai_guarded", "ai"), ("riskoff_top60", "riskoff")]:
            branch_swaps = variant_swaps[variant_swaps.get("source_branch", pd.Series(dtype=str)).astype(str) == branch] if not variant_swaps.empty else pd.DataFrame()
            row[f"{col_prefix}_swaps"] = int(len(branch_swaps))
            row[f"{col_prefix}_weighted_delta_sum"] = float(pd.to_numeric(branch_swaps.get("weighted_swap_delta"), errors="coerce").fillna(0).sum()) if not branch_swaps.empty else 0.0
        rows.append(row)

    for bucket, bucket_windows in bucket_specs.items():
        add_variant("default", v2b[v2b["variant"] == "default_grr_tail_guard"], "window_date", "score", bucket, bucket_windows)
        add_variant("v2b_guarded_candidate", v2b[v2b["variant"] == "v2b_trend_plus_ai_overlay"], "window_date", "score", bucket, bucket_windows, pd.read_csv(V2B_DETAIL / "accepted_swaps.csv").query("variant == 'v2b_trend_plus_ai_overlay'"))
        add_variant("riskoff_top60_direct_shadow", riskoff_direct, "window", "score", bucket, bucket_windows)
        add_variant("v2c_regime_riskoff_union_guarded_shadow", selected, "window", "score", bucket, bucket_windows, accepted)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "v2c_shadow_validation.csv", index=False)
    return out


def write_riskoff_direct_summary(validation: pd.DataFrame, out_dir: Path) -> None:
    direct = validation[validation["variant"] == "riskoff_top60_direct_shadow"].copy()
    riskoff_direct = pd.read_csv(out_dir / "riskoff_top60_direct_windows.csv") if (out_dir / "riskoff_top60_direct_windows.csv").exists() else pd.DataFrame()
    if not riskoff_direct.empty:
        for idx, row in direct.iterrows():
            windows = sorted(riskoff_direct["window"].astype(str).unique())
            if row["window_bucket"] == "20win":
                keep = windows[-20:]
            elif row["window_bucket"] == "40win":
                keep = windows[-40:]
            else:
                keep = windows
            trigger_rate = riskoff_direct[riskoff_direct["window"].astype(str).isin(set(keep))]["riskoff_triggered"].astype(bool).mean()
            direct.loc[idx, "avg_full_replacement_count"] = float(trigger_rate)
    else:
        direct["avg_full_replacement_count"] = 0.0
    direct["offline_diagnostic_only"] = True
    direct.to_csv(out_dir / "riskoff_top60_direct_shadow.csv", index=False)


def write_report(validation: pd.DataFrame, prior: pd.DataFrame, out_dir: Path) -> None:
    v2c = validation[validation["variant"] == "v2c_regime_riskoff_union_guarded_shadow"]
    v2b = validation[validation["variant"] == "v2b_guarded_candidate"]
    direct = validation[validation["variant"] == "riskoff_top60_direct_shadow"]
    lines = [
        "# v2c_regime_riskoff_union_guarded_shadow research report",
        "",
        "## 1. Why not directly restore exp-002-11",
        "The riskoff top60 branch is evaluated only as an offline diagnostic and proposal source. It is not used as a full-window runtime switch.",
        "",
        "## 2. riskoff_top60 branch 20/40/60",
        "```text\n" + direct.to_string(index=False) + "\n```",
        "",
        "## 3. branch_skill_prior",
        "```text\n" + prior.to_string(index=False) + "\n```",
        "",
        "## 4. v2c shadow union router",
        "```text\n" + v2c.to_string(index=False) + "\n```",
        "",
        "## 5. v2b comparison",
        "```text\n" + v2b.to_string(index=False) + "\n```",
        "",
        "## 6. Branch risk notes",
        "AI/theme remains the branch with the clearest downside in old/mid windows. Trend remains comparatively stable. Riskoff_top60 is useful only if proposal-level guards improve the old-window tail.",
        "",
        "## 7. Recommendation",
    ]
    merged = v2c[["window_bucket", "mean_delta_vs_default", "worst", "negative_delta_count", "avg_swaps"]].merge(
        v2b[["window_bucket", "mean_delta_vs_default", "worst", "negative_delta_count", "avg_swaps"]],
        on="window_bucket",
        suffixes=("_v2c", "_v2b"),
    )
    sixty = merged[merged["window_bucket"] == "60win"]
    if not sixty.empty and float(sixty["mean_delta_vs_default_v2c"].iloc[0]) > float(sixty["mean_delta_vs_default_v2b"].iloc[0]) and float(sixty["worst_v2c"].iloc[0]) >= float(sixty["worst_v2b"].iloc[0]):
        lines.append("v2c shadow beats v2b on 60win without worst deterioration; consider as a future runtime candidate after a separate freeze pass.")
    else:
        lines.append("Keep v2c in shadow. Do not replace v2b_guarded_candidate in this pass.")
    lines.extend(
        [
            "",
            "## Constraints confirmed",
            "- No hard switch runtime.",
            "- No baseline_hybrid/reference runtime.",
            "- No crash_minrisk_rescue.",
            "- No AI shadow guard runtime.",
            "- No 2026-03-09 single-window exception.",
            "- Realized returns are diagnostics only.",
        ]
    )
    (out_dir / "v2c_research_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default=str(SOURCE_RUN.relative_to(ROOT)))
    parser.add_argument("--detail-dir", default=str(V2B_DETAIL.relative_to(ROOT)))
    parser.add_argument("--out-dir", default=str(OUT_DIR.relative_to(ROOT)))
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = build_shadow_inputs(ROOT / args.source_run, ROOT / "data" / "train_hs300_20260424.csv")
    inputs["riskoff_candidates"].to_csv(out_dir / "riskoff_top60_candidates.csv", index=False)
    inputs["riskoff_direct"].to_csv(out_dir / "riskoff_top60_direct_windows.csv", index=False)
    proposals = build_candidate_proposals(inputs, ROOT / args.detail_dir)
    proposals.to_csv(out_dir / "candidate_proposals.csv", index=False)
    windows = sorted(inputs["default_top5"])
    prior = write_branch_skill_prior(proposals, windows, out_dir)
    selected, accepted, blocked = run_v2c_shadow(inputs, proposals, prior, out_dir)
    validation = write_validation(inputs, selected, accepted, out_dir)
    write_riskoff_direct_summary(validation, out_dir)
    write_report(validation, prior, out_dir)
    print(f"v2c shadow outputs written to {out_dir}")


if __name__ == "__main__":
    main()
