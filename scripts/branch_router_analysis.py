"""Offline branch router analysis for legal oracle gap distillation.

The script consumes replay artifacts from scripts/batch_window_analysis.py and
builds a leakage-aware branch router report. Baseline/reference branches are
kept only for offline oracle comparison and are deliberately filtered before
runtime routing decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from branch_router import (  # noqa: E402
    ILLEGAL_RUNTIME_BRANCHES,
    LEGAL_BRANCHES,
    build_branch_candidates,
    build_branch_snapshots,
    compute_branch_state_features,
    hedge_weight_trace,
    rank_blend_scores,
    route_branch_v1,
    route_branch_v2a,
    route_branch_v2b_overlay,
)
from config import config as PROJECT_CONFIG  # noqa: E402
from batch_window_analysis import (  # noqa: E402
    add_branch_diagnostic_features,
    branch_definitions,
    filter_branch,
    load_raw,
    normalize_stock_id,
    realized_returns_for_anchor,
)
from portfolio_utils import add_trend_uncluttered_scores  # noqa: E402


FOCUS_WINDOWS = [
    "2025-12-03",
    "2026-01-09",
    "2026-01-30",
    "2026-02-13",
    "2026-03-02",
    "2026-03-16",
    "2026-03-30",
    "2026-04-07",
    "2026-04-14",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _metric(scores: pd.Series, bad_counts: pd.Series | None = None, very_bad_counts: pd.Series | None = None) -> dict[str, float]:
    scores = pd.to_numeric(scores, errors="coerce").dropna()
    if scores.empty:
        return {
            "mean": 0.0,
            "median": 0.0,
            "q10": 0.0,
            "worst": 0.0,
            "win_rate": 0.0,
            "mean_bad_count": 0.0,
            "mean_very_bad_count": 0.0,
        }
    return {
        "mean": float(scores.mean()),
        "median": float(scores.median()),
        "q10": float(scores.quantile(0.10)),
        "worst": float(scores.min()),
        "win_rate": float((scores > 0).mean()),
        "mean_bad_count": float(pd.to_numeric(bad_counts, errors="coerce").mean()) if bad_counts is not None else 0.0,
        "mean_very_bad_count": float(pd.to_numeric(very_bad_counts, errors="coerce").mean()) if very_bad_counts is not None else 0.0,
    }


def _branch_def_map() -> dict[str, dict[str, str]]:
    out = {item["branch"]: item for item in branch_definitions()}
    out["grr_tail_guard"] = {"branch": "grr_tail_guard", "score_col": "grr_final_score", "filter": "nofilter"}
    return out


def _build_branch_frame(work: pd.DataFrame, branch: str, branch_def: dict[str, str], top_k: int = 30) -> pd.DataFrame:
    score_col = branch_def["score_col"]
    if score_col not in work.columns:
        if branch == "grr_tail_guard" and "score" in work.columns:
            score_col = "score"
        else:
            return pd.DataFrame()
    candidates = filter_branch(work, branch_def["filter"]).copy()
    if candidates.empty:
        return candidates
    candidates[score_col] = pd.to_numeric(candidates[score_col], errors="coerce").fillna(0.0)
    candidates = candidates.sort_values(score_col, ascending=False).head(min(top_k, len(candidates))).copy()
    candidates["branch_score"] = candidates[score_col]
    candidates.attrs["score_col"] = "branch_score"
    candidates.attrs["branch_score_col"] = score_col
    return candidates


def _rank_pct(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    fill = values.median() if values.notna().any() else 0.0
    return values.fillna(fill).rank(pct=True, method="average")


def _selected_ids(work: pd.DataFrame, branch: str, branch_def: dict[str, str]) -> set[str]:
    selected = _build_branch_frame(work, branch, branch_def, top_k=5)
    return set(selected.get("stock_id", pd.Series(dtype=str)).astype(str).tolist())


def _top_ids(frame: pd.DataFrame | None, n: int = 5) -> list[str]:
    if frame is None or frame.empty or "stock_id" not in frame.columns:
        return []
    score_col = frame.attrs.get("score_col", "branch_score")
    out = frame.copy()
    if score_col in out.columns:
        out[score_col] = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0)
        out = out.sort_values(score_col, ascending=False)
    return out.head(n)["stock_id"].astype(str).tolist()


def _split_picks(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    picks = [str(item).strip() for item in str(value).split(",") if str(item).strip()]
    if not picks:
        return []
    return normalize_stock_id(pd.Series(picks, dtype=str)).astype(str).tolist()


def _inject_final_top5(frame: pd.DataFrame, work: pd.DataFrame, selected_ids: list[str]) -> pd.DataFrame:
    if frame.empty or not selected_ids:
        return frame
    out = frame.copy()
    selected_set = set(selected_ids)
    missing_ids = [stock_id for stock_id in selected_ids if stock_id not in set(out["stock_id"].astype(str))]
    if missing_ids:
        missing = work[work["stock_id"].astype(str).isin(missing_ids)].copy()
        if not missing.empty:
            score_col = out.attrs.get("branch_score_col", "grr_final_score")
            missing["branch_score"] = pd.to_numeric(missing.get(score_col, missing.get("score", 0.0)), errors="coerce").fillna(0.0)
            out = pd.concat([out, missing], ignore_index=True, sort=False)
    out["final_selected_flag"] = out["stock_id"].astype(str).isin(selected_set)
    out["post_filter_selected_top5"] = out["final_selected_flag"]
    order = {stock_id: idx + 1 for idx, stock_id in enumerate(selected_ids)}
    out["final_selected_order"] = out["stock_id"].astype(str).map(order)
    out.attrs["score_col"] = "branch_score"
    out.attrs["branch_score_col"] = "branch_score"
    return out


def _build_prefilter_branch_frame(work: pd.DataFrame, branch: str, branch_def: dict[str, str], top_k: int = 20) -> pd.DataFrame:
    selected = _selected_ids(work, branch, branch_def)
    if branch == "trend_uncluttered":
        out = add_trend_uncluttered_scores(work.copy())
        trend_score = pd.to_numeric(out.get("trend_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
        reversal_score = pd.to_numeric(out.get("reversal_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
        base_score = pd.to_numeric(out.get("base_score", out.get("score", pd.Series(0.0, index=out.index))), errors="coerce").fillna(0.0)
        out["branch_score"] = trend_score.where(out.get("is_trend_pool", False).astype(bool), 0.65 * base_score + 0.35 * reversal_score)
        pool = out.sort_values("branch_score", ascending=False).head(min(top_k, len(out))).copy()
    elif branch == "ai_hardware_mainline_v1":
        out = work.copy()
        mainline = {"000977": 5, "688256": 4, "300408": 3, "601138": 2, "002463": 1}
        out["_mainline_boost"] = out["stock_id"].astype(str).map(mainline).fillna(0.0)
        out["branch_score"] = (
            1.50 * out["_mainline_boost"]
            + 0.40 * _rank_pct(out.get("score", pd.Series(0.0, index=out.index)))
            + 0.25 * _rank_pct(out.get("ret20", pd.Series(0.0, index=out.index)))
            + 0.20 * _rank_pct(out.get("lgb", pd.Series(0.0, index=out.index)))
            + 0.15 * _rank_pct(out.get("median_amount20", pd.Series(0.0, index=out.index)))
        )
        selected_frame = out[out["stock_id"].astype(str).isin(selected)].copy()
        filler = out[~out["stock_id"].astype(str).isin(selected)].sort_values("branch_score", ascending=False).head(max(0, top_k - len(selected_frame))).copy()
        pool = pd.concat([selected_frame, filler], ignore_index=True).sort_values("branch_score", ascending=False).head(top_k).copy()
    else:
        pool = _build_branch_frame(work, branch, branch_def, top_k=max(top_k, 20))
        if "branch_score" not in pool.columns:
            score_col = branch_def.get("score_col", "score")
            pool["branch_score"] = pd.to_numeric(pool.get(score_col, pool.get("score", 0.0)), errors="coerce").fillna(0.0)

    if pool.empty:
        return pool
    pool = pool.copy()
    pool["post_filter_selected_top5"] = pool["stock_id"].astype(str).isin(selected)
    pool["final_selected_flag"] = pool["post_filter_selected_top5"]
    pool.attrs["score_col"] = "branch_score"
    pool.attrs["branch_score_col"] = "branch_score"
    return pool


def _state_bucket(row: pd.Series) -> str:
    if bool(row.get("crash_mode", False)) or float(row.get("risk_off_score", 0.0)) >= 0.45:
        return "risk_off"
    if float(row.get("trend_score", 0.0)) >= 0.60 and float(row.get("clutter_score", 0.0)) < 0.65:
        return "trend"
    if float(row.get("clutter_score", 0.0)) >= 0.65:
        return "cluttered"
    return "normal"


def _realized_for_selection(selection: pd.DataFrame) -> dict[str, Any]:
    top = selection.head(min(5, len(selection))).copy()
    if top.empty:
        return {"score": 0.0, "bad_count": 0, "very_bad_count": 0, "selected_stocks": ""}
    rets = pd.to_numeric(top.get("realized_ret", pd.Series(0.0, index=top.index)), errors="coerce").fillna(0.0)
    return {
        "score": float(rets.mean()),
        "bad_count": int((rets < -0.03).sum()),
        "very_bad_count": int((rets < -0.05).sum()),
        "selected_stocks": ",".join(top["stock_id"].astype(str).tolist()),
        "selected_rets": ",".join(f"{x:.2%}" for x in rets.tolist()),
    }


def _realized_for_ids(work: pd.DataFrame, stock_ids: list[str]) -> dict[str, Any]:
    if not stock_ids:
        return {"score": 0.0, "bad_count": 0, "very_bad_count": 0, "selected_stocks": "", "selected_rets": ""}
    order = {str(stock_id): idx for idx, stock_id in enumerate(stock_ids)}
    selected = work[work["stock_id"].astype(str).isin(order)].copy()
    selected["_order"] = selected["stock_id"].astype(str).map(order)
    selected = selected.sort_values("_order")
    return _realized_for_selection(selected)


def _oracle_scores(score_map: dict[str, float], allowed: list[str]) -> tuple[str, float]:
    candidates = {branch: score for branch, score in score_map.items() if branch in allowed and pd.notna(score)}
    if not candidates:
        return "", 0.0
    branch = max(candidates, key=candidates.get)
    return branch, float(candidates[branch])


def _recent_strength(past_scores: list[dict[str, float]]) -> dict[str, float]:
    if not past_scores:
        return {branch: 0.0 for branch in LEGAL_BRANCHES}
    frame = pd.DataFrame(past_scores).reindex(columns=LEGAL_BRANCHES).fillna(0.0)
    means = frame.tail(8).mean()
    centered = means - means.mean()
    scale = float(centered.abs().max()) or 1.0
    return {branch: float(centered.get(branch, 0.0) / scale) for branch in LEGAL_BRANCHES}


def run_analysis(args: argparse.Namespace) -> Path:
    source_run = ROOT / args.source_run
    if not source_run.exists():
        raise FileNotFoundError(f"source replay run not found: {source_run}")
    out_dir = ROOT / args.out_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(ROOT / args.raw)
    window_summary = pd.read_csv(source_run / "window_summary.csv")
    branch_diag = pd.read_csv(source_run / "branch_diagnostics.csv")
    window_summary["anchor_date"] = pd.to_datetime(window_summary["anchor_date"]).dt.strftime("%Y-%m-%d")
    branch_diag["anchor_date"] = pd.to_datetime(branch_diag["anchor_date"]).dt.strftime("%Y-%m-%d")
    if args.last_n:
        keep = window_summary["anchor_date"].tail(args.last_n).tolist()
        window_summary = window_summary[window_summary["anchor_date"].isin(keep)].copy()
        branch_diag = branch_diag[branch_diag["anchor_date"].isin(keep)].copy()

    branch_defs = _branch_def_map()
    router_cfg = dict(PROJECT_CONFIG.get("branch_router_v1", {}))
    router_cfg["enabled"] = True
    router_cfg.update(json.loads(args.router_config_json) if args.router_config_json else {})

    snapshots = []
    candidate_rows = []
    decisions = []
    detail_rows = []
    ablation_rows: dict[str, list[dict[str, Any]]] = {
        "default_grr_tail_guard": [],
        "v1_branch_router": [],
        "v2a_trend_only": [],
        "v2a_theme_only": [],
        "v2a_trend_plus_theme": [],
        "v2a_trend_plus_theme_plus_crash_minrisk_only": [],
        "v2b_trend_overlay_only": [],
        "v2b_ai_overlay_only": [],
        "v2b_trend_plus_ai_overlay": [],
        "v2b_trend_plus_ai_overlay_plus_crash_minrisk_rescue": [],
    }
    effective_audit_rows: list[dict[str, Any]] = []
    focus_details: list[dict[str, Any]] = []
    past_legal_scores: list[dict[str, float]] = []
    branch_score_windows: list[dict[str, float]] = []

    for _, win in window_summary.sort_values("anchor_date").iterrows():
        anchor = pd.Timestamp(win["anchor_date"])
        anchor_tag = anchor.strftime("%Y%m%d")
        score_path = source_run / anchor_tag / "predict_score_df.csv"
        if not score_path.exists():
            raise FileNotFoundError(f"missing score df for {anchor:%Y-%m-%d}: {score_path}")

        realized, _ = realized_returns_for_anchor(raw, anchor, label_horizon=int(args.label_horizon))
        score_df = pd.read_csv(score_path, dtype={"stock_id": str})
        score_df["stock_id"] = normalize_stock_id(score_df["stock_id"])
        work = score_df.merge(realized[["stock_id", "realized_ret"]], on="stock_id", how="left")
        work["realized_ret"] = pd.to_numeric(work["realized_ret"], errors="coerce").fillna(0.0)
        work = add_branch_diagnostic_features(work)

        legal_outputs = {
            branch: _build_prefilter_branch_frame(work, branch, branch_defs[branch], top_k=20)
            for branch in LEGAL_BRANCHES
            if branch in branch_defs
        }
        legal_outputs = {branch: frame for branch, frame in legal_outputs.items() if not frame.empty}
        if "grr_tail_guard" in legal_outputs:
            legal_outputs["grr_tail_guard"] = _inject_final_top5(
                legal_outputs["grr_tail_guard"],
                work,
                _split_picks(win.get("selected_picks", "")),
            )
        illegal_outputs = {
            branch: _build_branch_frame(work, branch, branch_defs[branch])
            for branch in ILLEGAL_RUNTIME_BRANCHES
            if branch in branch_defs
        }
        illegal_outputs = {branch: frame for branch, frame in illegal_outputs.items() if not frame.empty}

        grr_risk_off = float(pd.to_numeric(work.get("grr_risk_off_score", pd.Series(0.0, index=work.index)), errors="coerce").fillna(0.0).max())
        market_breadth_5d = float(pd.to_numeric(work.get("ret5", pd.Series(0.0, index=work.index)), errors="coerce").fillna(0.0).gt(0.0).mean())
        market_ret5 = float(pd.to_numeric(work.get("ret5", pd.Series(0.0, index=work.index)), errors="coerce").fillna(0.0).median())
        risk_features = {
            "risk_off_score": grr_risk_off,
            "crash_mode": bool(grr_risk_off >= float(router_cfg.get("risk_off_high", 0.70)) or (market_breadth_5d <= 0.15 and market_ret5 < -0.03)),
            "recent_branch_oof_strength": _recent_strength(past_legal_scores),
        }
        market_state = compute_branch_state_features(work, legal_outputs, risk_features)
        market_state["rrf_k"] = int(PROJECT_CONFIG.get("branch_router_v2a", {}).get("rrf_k", 60))

        diag_rows = branch_diag[branch_diag["anchor_date"] == win["anchor_date"]]
        score_map = {row["branch"]: float(row["score"]) for _, row in diag_rows.iterrows() if pd.notna(row.get("score"))}
        score_map["grr_tail_guard"] = float(win["selected_score"])

        branch_score_windows.append({branch: score_map.get(branch, 0.0) for branch in LEGAL_BRANCHES})
        snap = build_branch_snapshots(
            legal_outputs,
            window_date=win["anchor_date"],
            market_state=market_state,
            realized_scores={branch: score_map.get(branch) for branch in LEGAL_BRANCHES},
        )
        snap["market_state_bucket"] = snap.apply(_state_bucket, axis=1)
        snapshots.append(snap)
        cand = build_branch_candidates(legal_outputs, top_k=20, rrf_k=int(market_state["rrf_k"]))
        if not cand.empty:
            cand.insert(0, "window_date", win["anchor_date"])
            candidate_rows.append(cand)

        route_inputs = {**legal_outputs, **illegal_outputs}
        decision = route_branch_v1(route_inputs, market_state, router_cfg)

        if decision.route_reason == "soft_blend_rank_rrf":
            blended = rank_blend_scores(legal_outputs, decision.branch_weights, rrf_k=int(router_cfg.get("rrf_k", 60)))
            final = _realized_for_selection(blended)
            chosen_score = final["score"]
            chosen_bad = final["bad_count"]
            chosen_very_bad = final["very_bad_count"]
            chosen_stocks = final["selected_stocks"]
            chosen_rets = final["selected_rets"]
        else:
            chosen_score = float(score_map.get(decision.chosen_branch, score_map.get("grr_tail_guard", 0.0)))
            chosen_bad = int(diag_rows.loc[diag_rows["branch"] == decision.chosen_branch, "bad_count"].iloc[0]) if (diag_rows["branch"] == decision.chosen_branch).any() else int(win.get("bad_count", 0))
            chosen_very_bad = int(diag_rows.loc[diag_rows["branch"] == decision.chosen_branch, "very_bad_count"].iloc[0]) if (diag_rows["branch"] == decision.chosen_branch).any() else int(win.get("very_bad_count", 0))
            if decision.chosen_branch == "grr_tail_guard":
                chosen_stocks = str(win.get("selected_picks", ""))
                chosen_rets = str(win.get("selected_rets", ""))
            else:
                row = diag_rows[diag_rows["branch"] == decision.chosen_branch]
                chosen_stocks = str(row["picks"].iloc[0]) if not row.empty else ""
                chosen_rets = str(row["rets"].iloc[0]) if not row.empty else ""

        legal_oracle_branch, legal_oracle_score = _oracle_scores(score_map, LEGAL_BRANCHES)
        illegal_oracle_branch, illegal_oracle_score = _oracle_scores(score_map, LEGAL_BRANCHES + ILLEGAL_RUNTIME_BRANCHES)

        decision_row = {
            "window_date": win["anchor_date"],
            "chosen_branch": decision.chosen_branch,
            "branch_weights": json.dumps(_jsonable(decision.branch_weights), ensure_ascii=False, sort_keys=True),
            "route_reason": decision.route_reason,
            "risk_off_score": decision.risk_off_score,
            "trend_score": decision.trend_score,
            "theme_score": decision.theme_score,
            "clutter_score": decision.clutter_score,
            "confidence": decision.confidence,
            "fallback_used": decision.fallback_used,
            "blocked_branches": ",".join(decision.blocked_branches),
            "illegal_branch_filtered": bool(decision.debug_info.get("illegal_branch_filtered")),
            "selected_score": chosen_score,
            "bad_count": chosen_bad,
            "very_bad_count": chosen_very_bad,
            "selected_stocks": chosen_stocks,
            "selected_rets": chosen_rets,
            "default_score": float(score_map.get("grr_tail_guard", 0.0)),
            "legal_oracle_branch": legal_oracle_branch,
            "legal_oracle_score": legal_oracle_score,
            "illegal_oracle_branch": illegal_oracle_branch,
            "illegal_oracle_score": illegal_oracle_score,
            "utilities_json": json.dumps(_jsonable(decision.debug_info.get("utilities", {})), ensure_ascii=False, sort_keys=True),
            "market_state_json": json.dumps(_jsonable(market_state), ensure_ascii=False, sort_keys=True),
            "debug_json": json.dumps(_jsonable(decision.debug_info), ensure_ascii=False, sort_keys=True),
        }
        decisions.append(decision_row)
        detail_rows.append({**decision_row, **{f"score_{branch}": score_map.get(branch) for branch in LEGAL_BRANCHES + ILLEGAL_RUNTIME_BRANCHES}})

        def realize_decision(decision_obj) -> dict[str, Any]:
            overlay_debug = (decision_obj.debug_info or {}).get("overlay_decision", {})
            final_top5 = overlay_debug.get("final_top5") if isinstance(overlay_debug, dict) else None
            if final_top5:
                final = _realized_for_ids(work, [str(stock_id) for stock_id in final_top5])
                score = final["score"]
                bad_count = final["bad_count"]
                very_bad_count = final["very_bad_count"]
                selected_stocks = final["selected_stocks"]
                selected_rets = final["selected_rets"]
            elif len(decision_obj.branch_weights) > 1:
                blended = rank_blend_scores(legal_outputs, decision_obj.branch_weights, rrf_k=int(market_state["rrf_k"]))
                final = _realized_for_selection(blended)
                score = final["score"]
                bad_count = final["bad_count"]
                very_bad_count = final["very_bad_count"]
                selected_stocks = final["selected_stocks"]
                selected_rets = final["selected_rets"]
            else:
                branch = next(iter(decision_obj.branch_weights), decision_obj.chosen_branch)
                score = float(score_map.get(branch, score_map.get("grr_tail_guard", 0.0)))
                match = diag_rows[diag_rows["branch"] == branch]
                bad_count = int(match["bad_count"].iloc[0]) if not match.empty else int(win.get("bad_count", 0))
                very_bad_count = int(match["very_bad_count"].iloc[0]) if not match.empty else int(win.get("very_bad_count", 0))
                if branch == "grr_tail_guard":
                    selected_stocks = str(win.get("selected_picks", ""))
                    selected_rets = str(win.get("selected_rets", ""))
                else:
                    selected_stocks = str(match["picks"].iloc[0]) if not match.empty and "picks" in match else ""
                    selected_rets = str(match["rets"].iloc[0]) if not match.empty and "rets" in match else ""
            swaps = overlay_debug.get("swaps", []) if isinstance(overlay_debug, dict) else []
            source_branches = overlay_debug.get("source_branches_used", []) if isinstance(overlay_debug, dict) else []
            return {
                "window_date": win["anchor_date"],
                "chosen_branch": decision_obj.chosen_branch,
                "route_reason": decision_obj.route_reason,
                "score": score,
                "bad_count": bad_count,
                "very_bad_count": very_bad_count,
                "selected_stocks": selected_stocks,
                "selected_rets": selected_rets,
                "legal_oracle_branch": legal_oracle_branch,
                "legal_oracle_score": legal_oracle_score,
                "swap_count": int(overlay_debug.get("swap_count", 0)) if isinstance(overlay_debug, dict) else 0,
                "source_branches_used": ",".join(source_branches) if isinstance(source_branches, list) else str(source_branches),
                "swaps_json": json.dumps(_jsonable(swaps), ensure_ascii=False, sort_keys=True),
                "debug": decision_obj.debug_info,
            }

        default_decision = type("D", (), {
            "chosen_branch": "grr_tail_guard",
            "branch_weights": {"grr_tail_guard": 1.0},
            "route_reason": "default_grr_tail_guard",
            "debug_info": {},
        })()
        ablation_rows["default_grr_tail_guard"].append(realize_decision(default_decision))
        ablation_rows["v1_branch_router"].append(realize_decision(decision))
        v2a_base = dict(PROJECT_CONFIG.get("branch_router_v2a", {}))
        v2a_base["enabled"] = True
        v2a_configs = {
            "v2a_trend_only": {**v2a_base, "trend_override_enabled": True, "theme_ai_override_enabled": False, "crash_minrisk_enabled": False},
            "v2a_theme_only": {**v2a_base, "trend_override_enabled": False, "theme_ai_override_enabled": True, "crash_minrisk_enabled": False},
            "v2a_trend_plus_theme": {**v2a_base, "trend_override_enabled": True, "theme_ai_override_enabled": True, "crash_minrisk_enabled": False},
            "v2a_trend_plus_theme_plus_crash_minrisk_only": {**v2a_base, "trend_override_enabled": True, "theme_ai_override_enabled": True, "crash_minrisk_enabled": True},
        }
        for name, cfg in v2a_configs.items():
            v2a_decision = route_branch_v2a(route_inputs, market_state, cfg)
            realized_v2a = realize_decision(v2a_decision)
            ablation_rows[name].append(realized_v2a)
            if v2a_decision.chosen_branch != "grr_tail_guard":
                default_top5 = _top_ids(legal_outputs.get("grr_tail_guard"))
                chosen_top5 = _top_ids(legal_outputs.get(v2a_decision.chosen_branch))
                final_top5 = v2a_decision.debug_info.get("post_blend_top5") or chosen_top5
                final_realized = _realized_for_ids(work, [str(stock_id) for stock_id in final_top5])
                effective = list(final_top5) != list(default_top5)
                if effective:
                    no_op_reason = ""
                elif list(chosen_top5) == list(default_top5):
                    no_op_reason = "same_top5_as_default"
                elif list(final_top5) == list(default_top5):
                    no_op_reason = "tail_guard_collapsed_to_default"
                else:
                    no_op_reason = "unknown"
                effective_audit_rows.append(
                    {
                        "variant": name,
                        "window_date": win["anchor_date"],
                        "chosen_branch": v2a_decision.chosen_branch,
                        "route_reason": v2a_decision.route_reason,
                        "default_top5": ",".join(default_top5),
                        "chosen_branch_top5": ",".join(chosen_top5),
                        "final_output_top5": ",".join([str(x) for x in final_top5]),
                        "overlap_default_vs_chosen_branch": len(set(default_top5) & set(chosen_top5)),
                        "overlap_default_vs_final": len(set(default_top5) & set(final_top5)),
                        "overlap_chosen_branch_vs_final": len(set(chosen_top5) & set(final_top5)),
                        "default_score": float(score_map.get("grr_tail_guard", 0.0)),
                        "chosen_branch_raw_score_realized": float(score_map.get(v2a_decision.chosen_branch, 0.0)),
                        "final_score_realized": float(final_realized["score"]),
                        "effective_override": bool(effective),
                        "no_op_reason": no_op_reason,
                    }
                )

        v2b_base = dict(PROJECT_CONFIG.get("branch_router_v2b", {}))
        v2b_base["enabled"] = True
        v2b_configs = {
            "v2b_trend_overlay_only": {**v2b_base, "trend_overlay_enabled": True, "theme_ai_overlay_enabled": False, "crash_minrisk_enabled": False},
            "v2b_ai_overlay_only": {**v2b_base, "trend_overlay_enabled": False, "theme_ai_overlay_enabled": True, "crash_minrisk_enabled": False},
            "v2b_trend_plus_ai_overlay": {**v2b_base, "trend_overlay_enabled": True, "theme_ai_overlay_enabled": True, "crash_minrisk_enabled": False},
            "v2b_trend_plus_ai_overlay_plus_crash_minrisk_rescue": {**v2b_base, "trend_overlay_enabled": True, "theme_ai_overlay_enabled": True, "crash_minrisk_enabled": True},
        }
        for name, cfg in v2b_configs.items():
            ablation_rows[name].append(realize_decision(route_branch_v2b_overlay(route_inputs, market_state, cfg)))

        if win["anchor_date"] in FOCUS_WINDOWS:
            focus_details.append({
                "window_date": win["anchor_date"],
                "market_state": market_state,
                "legal_oracle_branch": legal_oracle_branch,
                "legal_oracle_score": legal_oracle_score,
                "snapshots": snap.to_dict(orient="records"),
                "default_score": float(score_map.get("grr_tail_guard", 0.0)),
                "v2a": {name: ablation_rows[name][-1] for name in v2a_configs},
                "v2b": {name: ablation_rows[name][-1] for name in v2b_configs},
            })
        past_legal_scores.append({branch: score_map.get(branch, 0.0) for branch in LEGAL_BRANCHES})

    snapshots_df = pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame()
    candidates_df = pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
    decisions_df = pd.DataFrame(decisions)
    detail_df = pd.DataFrame(detail_rows)

    hedge_trace = hedge_weight_trace(branch_score_windows, LEGAL_BRANCHES, eta=float(PROJECT_CONFIG.get("grr_top5", {}).get("router", {}).get("hedge_eta", 0.50)))
    hedge_rows = []
    for decision, hedge in zip(decisions, hedge_trace):
        hedge_rows.append(
            {
                "window_date": decision["window_date"],
                "hedge_weights_before_decision": json.dumps(hedge["hedge_weights_before_decision"], sort_keys=True),
                "hedge_weights_after_update": json.dumps(hedge["hedge_weights_after_update"], sort_keys=True),
            }
        )
    hedge_df = pd.DataFrame(hedge_rows)
    decisions_df = decisions_df.merge(hedge_df, on="window_date", how="left")

    snapshots_df.to_csv(out_dir / "branch_snapshots.csv", index=False)
    candidates_df.to_csv(out_dir / "branch_candidates.csv", index=False)
    decisions_df.to_csv(out_dir / "router_decisions.csv", index=False)
    effective_audit_df = pd.DataFrame(effective_audit_rows)
    effective_audit_df.to_csv(out_dir / "effective_override_audit.csv", index=False)
    ablation_decision_frames = []
    for name, rows in ablation_rows.items():
        frame = pd.DataFrame([{k: v for k, v in row.items() if k != "debug"} for row in rows])
        frame.insert(0, "variant", name)
        ablation_decision_frames.append(frame)
    if ablation_decision_frames:
        pd.concat(ablation_decision_frames, ignore_index=True).to_csv(out_dir / "ablation_decisions.csv", index=False)

    overlay_diag_rows = []
    if not snapshots_df.empty:
        margin_lookup = {
            (str(row["window_date"]), str(row["branch_name"])): row
            for _, row in snapshots_df.iterrows()
        }
        default_by_date = {row["window_date"]: row["score"] for row in ablation_rows["default_grr_tail_guard"]}
        for row in ablation_rows["v2b_trend_plus_ai_overlay"]:
            date = row["window_date"]
            trend_snap = margin_lookup.get((date, "trend_uncluttered"), {})
            ai_snap = margin_lookup.get((date, "ai_hardware_mainline_v1"), {})
            overlay_diag_rows.append(
                {
                    "window_date": date,
                    "legal_oracle_branch": row["legal_oracle_branch"],
                    "default_score": default_by_date.get(date, 0.0),
                    "final_score": row["score"],
                    "delta": row["score"] - default_by_date.get(date, 0.0),
                    "swap_count": row.get("swap_count", 0),
                    "source_branches_used": row.get("source_branches_used", ""),
                    "trend_margin": float(trend_snap.get("branch_score_margin_top5_vs_top10", 0.0)),
                    "trend_dispersion": float(trend_snap.get("branch_score_dispersion_top10", 0.0)),
                    "ai_margin": float(ai_snap.get("branch_score_margin_top5_vs_top10", 0.0)),
                    "ai_dispersion": float(ai_snap.get("branch_score_dispersion_top10", 0.0)),
                    "selected_stocks": row.get("selected_stocks", ""),
                    "swaps_json": row.get("swaps_json", "[]"),
                }
            )
    pd.DataFrame(overlay_diag_rows).to_csv(out_dir / "overlay_diagnostics.csv", index=False)

    legal_oracle = decisions_df["legal_oracle_score"]
    illegal_oracle = decisions_df["illegal_oracle_score"]
    default_scores = decisions_df["default_score"]
    router_scores = decisions_df["selected_score"]

    branch_perf = {}
    for branch in LEGAL_BRANCHES:
        scores = detail_df[f"score_{branch}"].dropna() if f"score_{branch}" in detail_df else pd.Series(dtype=float)
        branch_perf[branch] = _metric(scores)

    by_state = {}
    if not snapshots_df.empty:
        for (branch, state), group in snapshots_df.groupby(["branch_name", "market_state_bucket"], dropna=False):
            by_state.setdefault(branch, {})[state] = _metric(group["realized_selected_score"])

    aggregate = {
        "windows": int(len(decisions_df)),
        "legal_branches": LEGAL_BRANCHES,
        "illegal_runtime_branches": ILLEGAL_RUNTIME_BRANCHES,
        "per_branch_performance": branch_perf,
        "per_branch_state_performance": by_state,
        "legal_oracle_upper_bound": _metric(legal_oracle),
        "illegal_oracle_upper_bound": _metric(illegal_oracle),
        "current_default": _metric(default_scores, window_summary["bad_count"], window_summary["very_bad_count"]),
        "branch_router_v1": _metric(router_scores, decisions_df["bad_count"], decisions_df["very_bad_count"]),
        "legal_oracle_gap": float(legal_oracle.mean() - default_scores.mean()),
        "illegal_oracle_gap": float(illegal_oracle.mean() - default_scores.mean()),
        "route_usage_count": decisions_df["chosen_branch"].value_counts().to_dict(),
        "route_reason_count": decisions_df["route_reason"].value_counts().to_dict(),
        "illegal_branch_filtered_count": int(decisions_df["illegal_branch_filtered"].astype(bool).sum()),
        "effective_override_audit": {
            "override_count": int(len(effective_audit_df)),
            "effective_override_count": int(effective_audit_df["effective_override"].astype(bool).sum()) if not effective_audit_df.empty else 0,
            "no_op_override_count": int((~effective_audit_df["effective_override"].astype(bool)).sum()) if not effective_audit_df.empty else 0,
            "no_op_override_windows": effective_audit_df.loc[~effective_audit_df["effective_override"].astype(bool), "window_date"].astype(str).tolist() if not effective_audit_df.empty else [],
        },
        "metrics_10win": {
            "current_default": _metric(default_scores.tail(10)),
            "branch_router_v1": _metric(router_scores.tail(10), decisions_df["bad_count"].tail(10), decisions_df["very_bad_count"].tail(10)),
            "legal_oracle_upper_bound": _metric(legal_oracle.tail(10)),
        },
        "metrics_20win": {
            "current_default": _metric(default_scores),
            "branch_router_v1": _metric(router_scores, decisions_df["bad_count"], decisions_df["very_bad_count"]),
            "legal_oracle_upper_bound": _metric(legal_oracle),
        },
        "no_future_leakage_note": "Router OOF strength and hedge_weights_before_decision use only prior windows; current realized score is appended after the decision row.",
    }
    aggregate["branch_margin_unusable"] = bool(
        snapshots_df.empty
        or snapshots_df.groupby("branch_name")["branch_score_margin_top5_vs_top10"].apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0.0).abs().sum() <= 1e-12).any()
    )

    def summarize_ablation(rows: list[dict[str, Any]]) -> dict[str, Any]:
        df = pd.DataFrame(rows)
        usage = df["chosen_branch"].value_counts().to_dict()
        reasons = df["route_reason"].value_counts().to_dict()
        source_series = df.get("source_branches_used", pd.Series("", index=df.index)).astype(str)
        swap_count = pd.to_numeric(df.get("swap_count", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        source_counts: dict[str, int] = {}
        for swaps_json in df.get("swaps_json", pd.Series("[]", index=df.index)).fillna("[]"):
            try:
                swaps = json.loads(swaps_json)
            except (TypeError, json.JSONDecodeError):
                swaps = []
            for swap in swaps:
                source = str(swap.get("source_branch", ""))
                if source:
                    source_counts[source] = source_counts.get(source, 0) + 1
        has_swap = swap_count > 0
        false_override = df[(df["chosen_branch"] != "grr_tail_guard") & (df["chosen_branch"] != "grr_tail_guard_overlay") & (df["chosen_branch"] != df["legal_oracle_branch"])]["window_date"].tolist()
        false_swap = df[
            has_swap
            & ~(
                ((df["legal_oracle_branch"] == "trend_uncluttered") & source_series.str.contains("trend_uncluttered", regex=False))
                | ((df["legal_oracle_branch"] == "ai_hardware_mainline_v1") & source_series.str.contains("ai_hardware_mainline_v1", regex=False))
            )
        ]["window_date"].tolist()
        missed_override = df[(df["chosen_branch"] == "grr_tail_guard") & (df["legal_oracle_branch"].isin(["trend_uncluttered", "ai_hardware_mainline_v1"]))]["window_date"].tolist()
        missed_swap = df[
            (df["legal_oracle_branch"].isin(["trend_uncluttered", "ai_hardware_mainline_v1"]))
            & ~(
                ((df["legal_oracle_branch"] == "trend_uncluttered") & source_series.str.contains("trend_uncluttered", regex=False))
                | ((df["legal_oracle_branch"] == "ai_hardware_mainline_v1") & source_series.str.contains("ai_hardware_mainline_v1", regex=False))
            )
        ]["window_date"].tolist()
        no_op = df[(df["chosen_branch"] != "grr_tail_guard") & (df.get("selected_stocks", "") == "")]["window_date"].tolist()
        return {
            **_metric(df["score"], df["bad_count"], df["very_bad_count"]),
            "route_usage_count": usage,
            "route_reason_count": reasons,
            "swap_count_mean": float(swap_count.mean()) if len(swap_count) else 0.0,
            "swap_count_by_source": source_counts,
            "false_override_count": int(len(false_override)),
            "missed_override_count": int(len(missed_override)),
            "false_swap_count": int(len(false_swap)),
            "missed_swap_count": int(len(missed_swap)),
            "no_op_override_count": int(len(no_op)),
            "false_override_windows": false_override,
            "missed_override_windows": missed_override,
            "false_swap_windows": false_swap,
            "missed_swap_windows": missed_swap,
        }

    aggregate["ablations"] = {name: summarize_ablation(rows) for name, rows in ablation_rows.items()}
    aggregate["ablations_10win"] = {name: summarize_ablation(rows[-10:]) for name, rows in ablation_rows.items()}

    probe_hits = []
    for row in focus_details:
        snaps_by_branch = {snap["branch_name"]: snap for snap in row["snapshots"]}
        default = snaps_by_branch.get("grr_tail_guard", {})
        minrisk = snaps_by_branch.get("legal_minrisk_hardened", {})
        risk_gap = float(default.get("risk_rank_mean_top5", 0.5)) - float(minrisk.get("risk_rank_mean_top5", 0.5))
        trend_score_probe = row["v2a"]["v2a_trend_plus_theme"]["debug"].get("trend_override_score", 0.0)
        theme_score_probe = row["v2a"]["v2a_trend_plus_theme"]["debug"].get("theme_ai_override_score", 0.0)
        hit = (
            risk_gap >= 0.25
            and trend_score_probe < float(PROJECT_CONFIG.get("branch_router_v2a", {}).get("trend_override_threshold", 0.58))
            and theme_score_probe < float(PROJECT_CONFIG.get("branch_router_v2a", {}).get("theme_ai_override_threshold", 0.58))
            and float(default.get("mean_consensus_support", 1.0)) <= 0.60
            and float(default.get("risk_rank_mean_top5", 0.0)) >= 0.45
        )
        if hit:
            probe_hits.append(row["window_date"])
    aggregate["minrisk_rescue_probe"] = {
        "hit_windows": probe_hits,
        "hit_2026_03_16": "2026-03-16" in probe_hits,
        "false_hit_2026_02_13": "2026-02-13" in probe_hits,
        "enabled": False,
    }
    (out_dir / "aggregate.json").write_text(json.dumps(_jsonable(aggregate), ensure_ascii=False, indent=2), encoding="utf-8")

    margin_lines = ["# margin diagnostics", ""]
    for branch, group in snapshots_df.groupby("branch_name", sort=True):
        depths = pd.to_numeric(group["candidate_depth"], errors="coerce")
        margins = pd.to_numeric(group["branch_score_margin_top5_vs_top10"], errors="coerce").fillna(0.0)
        sources = sorted(set(group["score_source"].astype(str)))
        margin_lines.append(
            f"- {branch}: min_candidate_depth={int(depths.min())}, mean_candidate_depth={depths.mean():.2f}, "
            f"score_source={','.join(sources)}, margin_nonzero={bool(margins.abs().sum() > 1e-12)}"
        )
    margin_lines.extend(["", "## focus margins"])
    for branch, dates in {
        "trend_uncluttered": ["2025-12-03", "2026-02-13"],
        "ai_hardware_mainline_v1": ["2026-03-30", "2026-04-07", "2026-04-14"],
    }.items():
        for date in dates:
            row = snapshots_df[(snapshots_df["window_date"] == date) & (snapshots_df["branch_name"] == branch)]
            if row.empty:
                margin_lines.append(f"- {date} {branch}: missing TopK candidates")
            else:
                r = row.iloc[0]
                margin_lines.append(
                    f"- {date} {branch}: margin={float(r['branch_score_margin_top5_vs_top10']):.6f}, "
                    f"dispersion={float(r['branch_score_dispersion_top10']):.6f}, depth={int(r['candidate_depth'])}"
                )
    if aggregate["branch_margin_unusable"]:
        bad = snapshots_df.groupby("branch_name")["branch_score_margin_top5_vs_top10"].apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0.0).abs().sum() <= 1e-12)
        margin_lines.extend(["", f"WARNING branch_margin_unusable=true for {bad[bad].index.tolist()}"])
    (out_dir / "margin_diagnostics.md").write_text("\n".join(margin_lines) + "\n", encoding="utf-8")

    overlay_lines = ["# v2b overlay detail", ""]
    all_details = []
    ablation_lookup = {
        name: {rec["window_date"]: rec for rec in rows}
        for name, rows in ablation_rows.items()
    }
    for _, win in window_summary.sort_values("anchor_date").iterrows():
        date = win["anchor_date"]
        if date not in ablation_lookup.get("v2b_trend_plus_ai_overlay", {}):
            continue
        all_details.append(
            {
                "window_date": date,
                "legal_oracle_branch": ablation_lookup["default_grr_tail_guard"][date]["legal_oracle_branch"],
                "default_score": ablation_lookup["default_grr_tail_guard"][date]["score"],
                "v2b": {name: ablation_lookup[name][date] for name in [
                    "v2b_trend_overlay_only",
                    "v2b_ai_overlay_only",
                    "v2b_trend_plus_ai_overlay",
                    "v2b_trend_plus_ai_overlay_plus_crash_minrisk_rescue",
                ]},
            }
        )
    for row in all_details:
        date = row["window_date"]
        overlay_lines.extend(["", f"## {date}", f"- oracle_branch: {row['legal_oracle_branch']}"])
        for variant in [
            "v2b_trend_overlay_only",
            "v2b_ai_overlay_only",
            "v2b_trend_plus_ai_overlay",
            "v2b_trend_plus_ai_overlay_plus_crash_minrisk_rescue",
        ]:
            rec = row["v2b"][variant]
            swaps = json.loads(rec.get("swaps_json", "[]")) if rec.get("swaps_json") else []
            default_top5 = []
            debug = rec.get("debug", {}) or {}
            overlay_debug = debug.get("overlay_decision", {}) if isinstance(debug, dict) else {}
            inner_debug = overlay_debug.get("debug_info", {}) if isinstance(overlay_debug, dict) else {}
            if isinstance(inner_debug, dict):
                default_top5 = inner_debug.get("default_top5", [])
            if swaps:
                swap_text = "; ".join(
                    f"{s.get('removed_stock')}->{s.get('added_stock')}({s.get('source_branch')}, gap={float(s.get('replacement_gap', 0.0)):.4f})"
                    for s in swaps
                )
            else:
                rejected = overlay_debug.get("rejected_candidates", []) if isinstance(overlay_debug, dict) else []
                best_failed = rejected[0].get("reject_reason", "no_candidate") if rejected else "threshold_or_window_gate"
                swap_text = f"none; best_failed={best_failed}"
            overlay_lines.extend(
                [
                    f"- {variant}: default_top5={','.join(default_top5)}, final_top5={rec.get('selected_stocks', '')}, "
                    f"source={rec.get('source_branches_used', '') or 'none'}, swap_count={rec.get('swap_count', 0)}, "
                    f"default_score={row.get('default_score', 0.0):.6f}, "
                    f"final_score={rec.get('score', 0.0):.6f}, delta={rec.get('score', 0.0) - row.get('default_score', rec.get('score', 0.0)):.6f}, swaps={swap_text}",
                ]
            )
    (out_dir / "overlay_detail.md").write_text("\n".join(overlay_lines) + "\n", encoding="utf-8")

    detail_lines = [
        "# branch_router_v1 window detail",
        "",
        f"- legal_oracle_mean: {aggregate['legal_oracle_upper_bound']['mean']:.6f}",
        f"- current_default_mean: {aggregate['current_default']['mean']:.6f}",
        f"- branch_router_v1_mean: {aggregate['branch_router_v1']['mean']:.6f}",
        f"- branch_router_v1_q10: {aggregate['branch_router_v1']['q10']:.6f}",
        f"- branch_router_v1_worst: {aggregate['branch_router_v1']['worst']:.6f}",
        "",
        "## Focus windows",
    ]
    for date in FOCUS_WINDOWS:
        rows = decisions_df[decisions_df["window_date"] == date]
        if rows.empty:
            continue
        row = rows.iloc[0]
        utilities = json.loads(row["utilities_json"]) if isinstance(row["utilities_json"], str) and row["utilities_json"] else {}
        oracle = row["legal_oracle_branch"]
        chosen = row["chosen_branch"]
        if chosen == oracle:
            miss_reason = "selected legal oracle"
        elif not utilities:
            miss_reason = f"hard guard chose {chosen}; oracle {oracle} was bypassed by risk rule"
        else:
            miss_reason = (
                f"oracle utility={utilities.get(oracle, 0.0):.4f}, "
                f"chosen utility={utilities.get(chosen, 0.0):.4f}; check risk/clutter/recent strength thresholds"
            )
        detail_lines.extend(
            [
                "",
                f"### {date}",
                f"- market_state: risk_off={row['risk_off_score']:.4f}, trend={row['trend_score']:.4f}, clutter={row['clutter_score']:.4f}",
                f"- chosen_branch: {chosen}, reason={row['route_reason']}, confidence={row['confidence']:.4f}",
                f"- legal_oracle_branch: {oracle}, legal_oracle_score={row['legal_oracle_score']:.6f}",
                f"- utilities: {json.dumps(utilities, ensure_ascii=False, sort_keys=True)}",
                f"- why_not_oracle: {miss_reason}",
            ]
        )
    (out_dir / "window_detail.md").write_text("\n".join(detail_lines) + "\n", encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default="temp/batch_window_analysis/grr_tail_guard_20win")
    parser.add_argument("--out-dir", default="temp/branch_router_analysis")
    parser.add_argument("--run-name", default="branch_router_v1_20win")
    parser.add_argument("--raw", default="data/train_hs300_20260424.csv")
    parser.add_argument("--last-n", type=int, default=20)
    parser.add_argument("--label-horizon", type=int, default=5)
    parser.add_argument("--router-config-json", default="")
    args = parser.parse_args()
    out_dir = run_analysis(args)
    print(f"Wrote branch router analysis to {out_dir}")


if __name__ == "__main__":
    main()
