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

from batch_window_analysis import load_raw, normalize_stock_id  # noqa: E402
from branch_router_v2c_shadow import SOURCE_RUN, V2B_DETAIL, _bad_counts, _realized_for_ids, build_shadow_inputs  # noqa: E402


OUT_DIR = ROOT / "temp" / "branch_router_validation" / "confidence_anchor_shadow"


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _rank_pct(series: pd.Series, ascending: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill = values.median() if values.notna().any() else 0.0
    return values.fillna(fill).rank(method="average", pct=True, ascending=ascending)


def _split_ids(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [str(x).strip().zfill(6) for x in str(value).split(",") if str(x).strip()]


def build_anchor_panel(raw_path: Path) -> pd.DataFrame:
    raw = load_raw(raw_path)
    panel = raw.copy()
    panel["stock_id"] = normalize_stock_id(panel["股票代码"])
    panel["date"] = pd.to_datetime(panel["日期"]).dt.strftime("%Y-%m-%d")
    for col in ["收盘", "最高", "最低"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel = panel.sort_values(["stock_id", "date"]).reset_index(drop=True)
    group = panel.groupby("stock_id", sort=False)
    high120 = group["最高"].transform(lambda s: s.rolling(120, min_periods=40).max())
    high252 = group["最高"].transform(lambda s: s.rolling(252, min_periods=80).max())
    low120 = group["最低"].transform(lambda s: s.rolling(120, min_periods=40).min())
    panel["pth120"] = panel["收盘"] / (high120 + 1e-12)
    panel["pth252"] = panel["收盘"] / (high252 + 1e-12)
    panel["range_pos120"] = (panel["收盘"] - low120) / ((high120 - low120) + 1e-12)
    return panel[["stock_id", "date", "pth120", "pth252", "range_pos120"]].replace([np.inf, -np.inf], np.nan)


def build_score_history(source_run: Path) -> pd.DataFrame:
    windows = pd.read_csv(source_run / "window_summary.csv")
    windows["anchor_date"] = pd.to_datetime(windows["anchor_date"]).dt.strftime("%Y-%m-%d")
    rows: list[pd.DataFrame] = []
    for _, win in windows.sort_values("anchor_date").iterrows():
        window = str(win["anchor_date"])
        path = source_run / pd.Timestamp(window).strftime("%Y%m%d") / "predict_score_df.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"stock_id": str})
        frame["stock_id"] = normalize_stock_id(frame["stock_id"])
        score_col = "grr_final_score" if "grr_final_score" in frame.columns else "score"
        out = frame[["stock_id", score_col]].copy()
        out["window"] = window
        out["hist_model_score"] = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0)
        out["hist_model_rank"] = _rank_pct(out["hist_model_score"])
        rows.append(out[["window", "stock_id", "hist_model_score", "hist_model_rank"]])
    hist = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["window", "stock_id", "hist_model_score", "hist_model_rank"])
    hist = hist.sort_values(["stock_id", "window"]).reset_index(drop=True)
    hist["prev_model_rank"] = hist.groupby("stock_id")["hist_model_rank"].shift(1)
    hist["prev2_model_rank"] = hist.groupby("stock_id")["hist_model_rank"].shift(2)
    hist["rank_change_1"] = hist["hist_model_rank"] - hist["prev_model_rank"]
    hist["rank_stability_3"] = hist[["hist_model_rank", "prev_model_rank", "prev2_model_rank"]].mean(axis=1)
    return hist[["window", "stock_id", "prev_model_rank", "prev2_model_rank", "rank_change_1", "rank_stability_3"]]


def _target(work: pd.DataFrame, ids: list[str], rule: str) -> tuple[str, float, float]:
    selected = work[work["stock_id"].astype(str).isin(set(ids))].copy()
    if selected.empty:
        return "", 0.0, 0.0
    selected["_score"] = _num(selected, "grr_final_score")
    selected["_risk"] = _num(selected, "_risk_value", default=0.5)
    selected["_uncertainty"] = _rank_pct(_num(selected, "uncertainty_score")) + _rank_pct(_num(selected, "rank_disagreement"))
    if rule == "highest_risk":
        row = selected.sort_values(["_risk", "_score"], ascending=[False, True]).iloc[0]
    elif rule == "highest_uncertainty":
        row = selected.sort_values(["_uncertainty", "_risk"], ascending=[False, False]).iloc[0]
    else:
        row = selected.sort_values(["_score", "_risk"], ascending=[True, False]).iloc[0]
    return str(row["stock_id"]).zfill(6), float(row["_score"]), float(row["_risk"])


def _stock_return(work: pd.DataFrame, stock_id: str) -> float:
    row = work[work["stock_id"].astype(str) == str(stock_id).zfill(6)]
    if row.empty:
        return 0.0
    return float(pd.to_numeric(row["realized_ret"].iloc[0], errors="coerce"))


def _add_scores(work: pd.DataFrame) -> pd.DataFrame:
    out = work.copy()
    out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
    ret5 = _num(out, "ret5")
    ret20 = _num(out, "ret20")
    sigma20 = _num(out, "sigma20")
    amp20 = _num(out, "amp20")
    drawdown20 = _num(out, "max_drawdown20")
    downside_beta60 = _num(out, "downside_beta60")
    median_amount20 = _num(out, "median_amount20")
    model = _num(out, "grr_final_score")
    consensus = _num(out, "grr_consensus_norm")
    disagreement = _num(out, "rank_disagreement", default=0.5)
    uncertainty = _num(out, "uncertainty_score", default=0.5)
    risk = _num(out, "_risk_value", default=0.5)
    pth120 = _num(out, "pth120", default=0.5)
    pth252 = _num(out, "pth252", default=0.5)
    range_pos120 = _num(out, "range_pos120", default=0.5)
    prev_rank = _num(out, "prev_model_rank", default=0.5)
    rank_change = _num(out, "rank_change_1", default=0.0)
    stability = _num(out, "rank_stability_3", default=0.5)

    out["ret5"] = ret5
    out["ret20"] = ret20
    out["sigma20"] = sigma20
    out["amp20"] = amp20
    out["max_drawdown20"] = drawdown20
    out["downside_beta60"] = downside_beta60
    out["pth120"] = pth120
    out["pth252"] = pth252
    out["range_pos120"] = range_pos120
    out["model_score"] = model
    out["model_rank"] = _rank_pct(model)
    out["prev_model_rank_filled"] = prev_rank
    out["rank_change_1"] = rank_change
    out["rank_stability_3"] = stability
    out["consensus_rank"] = _rank_pct(consensus)
    out["liquidity_rank"] = _rank_pct(median_amount20)
    out["risk_rank"] = _rank_pct(risk)
    out["disagreement_rank"] = _rank_pct(disagreement)
    out["uncertainty_rank"] = _rank_pct(uncertainty)
    out["anchor_rank"] = 0.65 * _rank_pct(pth120) + 0.35 * _rank_pct(pth252)
    out["anchor_score"] = (
        0.28 * out["model_rank"]
        + 0.24 * out["anchor_rank"]
        + 0.16 * out["liquidity_rank"]
        + 0.12 * out["consensus_rank"]
        - 0.14 * out["risk_rank"]
        - 0.08 * ((ret5 - 0.09) / 0.08).clip(lower=0.0, upper=1.0)
    )
    out["confidence_score"] = (
        0.34 * out["model_rank"]
        + 0.20 * out["consensus_rank"]
        + 0.16 * (1.0 - out["disagreement_rank"])
        + 0.14 * (1.0 - out["uncertainty_rank"])
        + 0.10 * out["liquidity_rank"]
        - 0.14 * out["risk_rank"]
    )
    out["persistence_score"] = (
        0.30 * out["model_rank"]
        + 0.24 * out["rank_stability_3"]
        + 0.16 * _rank_pct(rank_change)
        + 0.12 * out["consensus_rank"]
        + 0.10 * out["liquidity_rank"]
        - 0.14 * out["risk_rank"]
    )
    out["anchor_confidence_score"] = 0.45 * out["anchor_score"] + 0.35 * out["confidence_score"] + 0.20 * out["persistence_score"]
    out["pass_common"] = (
        (out["liquidity_rank"] >= 0.30)
        & (out["model_rank"] >= 0.70)
        & (sigma20 < 0.050)
        & (amp20 < 0.10)
        & (drawdown20 > -0.13)
        & (downside_beta60 < 1.40)
        & (ret5 > -0.04)
        & (ret5 < 0.10)
        & (ret20 < 0.35)
    )
    out["pass_anchor52"] = out["pass_common"] & (pth120 >= 0.86) & (range_pos120 < 0.98) & (ret20 > -0.03)
    out["pass_confidence"] = (
        out["pass_common"]
        & (out["consensus_rank"] >= 0.60)
        & (out["disagreement_rank"] <= 0.50)
        & (out["uncertainty_rank"] <= 0.60)
    )
    out["pass_persistence"] = (
        out["pass_common"]
        & (prev_rank >= 0.65)
        & (stability >= 0.65)
        & (rank_change > -0.20)
    )
    out["pass_anchor_confidence"] = out["pass_anchor52"] & out["pass_confidence"]
    return out


def _pool(work: pd.DataFrame, base_ids: list[str], mode: str, rank_cap: int) -> pd.DataFrame:
    out = _add_scores(work)
    out = out[~out["stock_id"].isin(set(base_ids))].copy()
    mapping = {
        "anchor52": ("pass_anchor52", "anchor_score"),
        "confidence": ("pass_confidence", "confidence_score"),
        "persistence": ("pass_persistence", "persistence_score"),
        "anchor_confidence": ("pass_anchor_confidence", "anchor_confidence_score"),
    }
    pass_col, score_col = mapping[mode]
    out = out[out[pass_col]].copy()
    if out.empty:
        return out
    out = out.sort_values(score_col, ascending=False).copy()
    out["candidate_rank"] = np.arange(1, len(out) + 1)
    out = out[out["candidate_rank"] <= int(rank_cap)]
    out["candidate_score"] = out[score_col]
    return out.sort_values(["candidate_rank", score_col], ascending=[True, False])


def _eval_insert(
    work: pd.DataFrame,
    window: str,
    base_ids: list[str],
    base_name: str,
    mode: str,
    rank_cap: int,
    target_rule: str,
    gate_reason: str = "",
) -> dict[str, Any]:
    base_return = _realized_for_ids(work, base_ids)
    base_bad, base_very_bad = _bad_counts(work, base_ids)
    target, target_score, target_risk = _target(work, base_ids, target_rule)
    variant = f"{base_name}_{mode}_{target_rule}_top{rank_cap}"
    row: dict[str, Any] = {
        "window": window,
        "variant": variant,
        "base_name": base_name,
        "mode": mode,
        "rank_cap": rank_cap,
        "target_rule": target_rule,
        "base_return": base_return,
        "shadow_return": base_return,
        "delta_vs_base": 0.0,
        "accepted_swap_count": 0,
        "candidate_stock": "",
        "candidate_rank": None,
        "candidate_score": None,
        "candidate_pth120": None,
        "candidate_model_rank": None,
        "candidate_prev_rank": None,
        "candidate_rank_change": None,
        "candidate_risk": None,
        "replaced_stock": target,
        "replaced_score": target_score,
        "replaced_risk": target_risk,
        "raw_candidate_return": None,
        "raw_replaced_return": None,
        "raw_stock_delta": None,
        "weighted_swap_delta": 0.0,
        "blocked_reason": gate_reason,
        "final_top5": ",".join(base_ids),
        "bad_count": base_bad,
        "very_bad_count": base_very_bad,
    }
    if gate_reason:
        return row
    if not base_ids or not target:
        row["blocked_reason"] = "missing_base_or_target"
        return row
    candidates = _pool(work, base_ids, mode, rank_cap)
    if not candidates.empty:
        candidates = candidates[candidates["model_score"] > float(target_score)]
    if candidates.empty:
        row["blocked_reason"] = "no_candidate"
        return row
    cand = candidates.iloc[0]
    candidate_stock = str(cand["stock_id"]).zfill(6)
    final_ids = list(base_ids)
    final_ids[final_ids.index(target)] = candidate_stock
    shadow_return = _realized_for_ids(work, final_ids)
    bad, very_bad = _bad_counts(work, final_ids)
    raw_candidate_return = _stock_return(work, candidate_stock)
    raw_replaced_return = _stock_return(work, target)
    raw_delta = raw_candidate_return - raw_replaced_return
    row.update(
        {
            "shadow_return": shadow_return,
            "delta_vs_base": shadow_return - base_return,
            "accepted_swap_count": 1,
            "candidate_stock": candidate_stock,
            "candidate_rank": int(cand["candidate_rank"]),
            "candidate_score": float(cand["candidate_score"]),
            "candidate_pth120": float(cand["pth120"]),
            "candidate_model_rank": float(cand["model_rank"]),
            "candidate_prev_rank": float(cand["prev_model_rank_filled"]),
            "candidate_rank_change": float(cand["rank_change_1"]),
            "candidate_risk": float(cand.get("_risk_value", 0.0)),
            "raw_candidate_return": raw_candidate_return,
            "raw_replaced_return": raw_replaced_return,
            "raw_stock_delta": raw_delta,
            "weighted_swap_delta": 0.2 * raw_delta,
            "blocked_reason": "",
            "final_top5": ",".join(final_ids),
            "bad_count": bad,
            "very_bad_count": very_bad,
        }
    )
    return row


def _metrics(values: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return {"mean": 0.0, "q10": 0.0, "worst": 0.0, "hit_rate": 0.0}
    return {"mean": float(s.mean()), "q10": float(s.quantile(0.10)), "worst": float(s.min()), "hit_rate": float((s > 0).mean())}


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    windows = sorted(rows["window"].astype(str).unique())
    buckets = {"20win": windows[-20:], "40win": windows[-40:], "60win": windows}
    out_rows: list[dict[str, Any]] = []
    for bucket, keep in buckets.items():
        scoped = rows[rows["window"].astype(str).isin(set(keep))].copy()
        for variant, sub in scoped.groupby("variant"):
            delta = pd.to_numeric(sub["delta_vs_base"], errors="coerce").fillna(0.0)
            out_rows.append(
                {
                    "bucket": bucket,
                    "variant": variant,
                    "window_count": int(len(sub)),
                    "accepted_swaps": int(pd.to_numeric(sub["accepted_swap_count"], errors="coerce").fillna(0).sum()),
                    "accepted_swap_rate": float(pd.to_numeric(sub["accepted_swap_count"], errors="coerce").fillna(0).mean()) if len(sub) else 0.0,
                    **{f"return_{k}": v for k, v in _metrics(sub["shadow_return"]).items()},
                    **{f"delta_{k}": v for k, v in _metrics(sub["delta_vs_base"]).items()},
                    "negative_delta_count": int((delta < -1e-12).sum()),
                    "very_bad_mean": float(pd.to_numeric(sub["very_bad_count"], errors="coerce").fillna(0).mean()) if len(sub) else 0.0,
                }
            )
    return pd.DataFrame(out_rows)


def run_shadow(source_run: Path, detail_dir: Path, raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = build_shadow_inputs(source_run, raw_path)
    anchor = build_anchor_panel(raw_path)
    anchor_by_date = {str(date): group.drop(columns=["date"]).copy() for date, group in anchor.groupby("date", sort=False)}
    score_history = build_score_history(source_run)
    hist_by_window = {str(window): group.drop(columns=["window"]).copy() for window, group in score_history.groupby("window", sort=False)}
    ablation = pd.read_csv(detail_dir / "ablation_decisions.csv")
    v2b = ablation[ablation["variant"] == "v2b_trend_plus_ai_overlay"].set_index("window_date")
    rows: list[dict[str, Any]] = []
    for window in sorted(inputs["default_top5"]):
        work = inputs["work_by_window"][window].copy()
        if window in anchor_by_date:
            work = work.merge(anchor_by_date[window], on="stock_id", how="left")
        if window in hist_by_window:
            work = work.merge(hist_by_window[window], on="stock_id", how="left")
        default_ids = inputs["default_top5"][window]
        v2b_ids = _split_ids(v2b.loc[window, "selected_stocks"]) if window in v2b.index else default_ids
        v2b_changed = set(v2b_ids) != set(default_ids)
        bases = [
            ("default", default_ids, ""),
            ("v2b", v2b_ids, ""),
            ("v2b_no_swap_only", v2b_ids, "v2b_already_swapped" if v2b_changed else ""),
        ]
        for base_name, base_ids, gate in bases:
            for mode in ["anchor52", "confidence", "persistence", "anchor_confidence"]:
                for target_rule in ["lowest_score", "highest_risk", "highest_uncertainty"]:
                    for rank_cap in [1, 3, 5]:
                        rows.append(_eval_insert(work, window, base_ids, base_name, mode, rank_cap, target_rule, gate_reason=gate))
    result = pd.DataFrame(rows)
    return result, summarize(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default=str(SOURCE_RUN.relative_to(ROOT)))
    parser.add_argument("--detail-dir", default=str(V2B_DETAIL.relative_to(ROOT)))
    parser.add_argument("--raw-path", default="data/train_hs300_20260424.csv")
    parser.add_argument("--out-dir", default=str(OUT_DIR.relative_to(ROOT)))
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = run_shadow(ROOT / args.source_run, ROOT / args.detail_dir, ROOT / args.raw_path)
    rows.to_csv(out_dir / "confidence_anchor_windows.csv", index=False)
    summary.to_csv(out_dir / "confidence_anchor_summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps({"rows": len(rows), "out_dir": str(out_dir), "summary": summary.to_dict(orient="records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows)}, ensure_ascii=False), flush=True)
    print(summary[summary["bucket"] == "60win"].sort_values("delta_mean", ascending=False).head(24).to_string(index=False))


if __name__ == "__main__":
    main()
