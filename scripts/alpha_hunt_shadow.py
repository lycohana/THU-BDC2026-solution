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

from branch_router_v2c_shadow import SOURCE_RUN, V2B_DETAIL, _bad_counts, _realized_for_ids, build_shadow_inputs  # noqa: E402


OUT_DIR = ROOT / "temp" / "branch_router_validation" / "alpha_hunt_shadow"


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


def _target(work: pd.DataFrame, ids: list[str], rule: str) -> tuple[str, float, float]:
    if not ids:
        return "", 0.0, 0.0
    selected = work[work["stock_id"].astype(str).isin(set(ids))].copy()
    if selected.empty:
        return "", 0.0, 0.0
    selected["_score"] = _num(selected, "grr_final_score")
    selected["_risk"] = _num(selected, "_risk_value", default=0.5)
    if rule == "highest_risk":
        row = selected.sort_values(["_risk", "_score"], ascending=[False, True]).iloc[0]
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
    ret1 = _num(out, "ret1")
    ret5 = _num(out, "ret5")
    ret10 = _num(out, "ret10")
    ret20 = _num(out, "ret20")
    sigma20 = _num(out, "sigma20")
    amp20 = _num(out, "amp20")
    drawdown20 = _num(out, "max_drawdown20")
    downside_beta60 = _num(out, "downside_beta60")
    beta60 = _num(out, "beta60")
    idio_vol60 = _num(out, "idio_vol60")
    median_amount20 = _num(out, "median_amount20")
    risk = _num(out, "_risk_value", default=0.5)
    model = _num(out, "grr_final_score")
    consensus = _num(out, "grr_consensus_norm")
    market_ret20 = float(ret20.median())
    market_ret10 = float(ret10.median())

    out["ret1"] = ret1
    out["ret5"] = ret5
    out["ret10"] = ret10
    out["ret20"] = ret20
    out["sigma20"] = sigma20
    out["amp20"] = amp20
    out["max_drawdown20"] = drawdown20
    out["downside_beta60"] = downside_beta60
    out["beta60"] = beta60
    out["idio_vol60"] = idio_vol60
    out["liquidity_rank"] = _rank_pct(median_amount20)
    out["model_score"] = model
    out["model_rank"] = _rank_pct(model)
    out["consensus_rank"] = _rank_pct(consensus)
    out["risk_rank"] = _rank_pct(risk)
    out["sigma_rank_local"] = _rank_pct(sigma20)
    out["idio_rank"] = _rank_pct(idio_vol60)
    out["drawdown_bad_rank"] = _rank_pct(-drawdown20)
    out["resid_ret20"] = ret20 - beta60 * market_ret20
    out["resid_ret10"] = ret10 - beta60 * market_ret10
    out["resid20_rank"] = _rank_pct(out["resid_ret20"])
    out["resid10_rank"] = _rank_pct(out["resid_ret10"])
    out["dip_rank"] = _rank_pct(-ret5)
    out["rebound_rank"] = _rank_pct(ret1)

    out["pullback_score"] = (
        0.30 * out["model_rank"]
        + 0.20 * out["resid20_rank"]
        + 0.20 * out["dip_rank"]
        + 0.15 * out["liquidity_rank"]
        + 0.10 * out["consensus_rank"]
        - 0.20 * out["risk_rank"]
    )
    out["pullback_rebound_score"] = out["pullback_score"] + 0.15 * out["rebound_rank"]
    out["low_vol_quality_score"] = (
        0.35 * out["model_rank"]
        + 0.20 * (1.0 - out["sigma_rank_local"])
        + 0.18 * (1.0 - out["idio_rank"])
        + 0.12 * out["liquidity_rank"]
        + 0.10 * out["consensus_rank"]
        - 0.15 * out["drawdown_bad_rank"]
    )
    out["residual_momentum_score"] = (
        0.30 * out["resid20_rank"]
        + 0.20 * out["resid10_rank"]
        + 0.25 * out["model_rank"]
        + 0.12 * out["liquidity_rank"]
        - 0.17 * out["risk_rank"]
        - 0.08 * ((ret5 - 0.10) / 0.08).clip(lower=0.0, upper=1.0)
    )

    out["pass_common_liq"] = out["liquidity_rank"] >= 0.30
    out["pass_common_risk"] = (
        out["pass_common_liq"]
        & (sigma20 < 0.050)
        & (amp20 < 0.10)
        & (drawdown20 > -0.13)
        & (downside_beta60 < 1.40)
    )
    out["pass_pullback"] = (
        out["pass_common_risk"]
        & (out["model_rank"] >= 0.70)
        & (ret20 > 0.02)
        & (ret20 < 0.32)
        & (ret5 > -0.08)
        & (ret5 < 0.015)
        & (ret1 > -0.025)
    )
    out["pass_pullback_rebound"] = out["pass_pullback"] & (ret1 > 0.0)
    out["pass_low_vol_quality"] = (
        out["pass_common_liq"]
        & (out["model_rank"] >= 0.70)
        & (out["sigma_rank_local"] <= 0.35)
        & (out["idio_rank"] <= 0.45)
        & (downside_beta60 < 1.15)
        & (drawdown20 > -0.09)
        & (ret5 > -0.06)
        & (ret20 < 0.25)
    )
    out["pass_residual_momentum"] = (
        out["pass_common_risk"]
        & (out["model_rank"] >= 0.70)
        & (out["resid_ret20"] > 0.0)
        & (out["resid_ret10"] > -0.02)
        & (ret5 > -0.03)
        & (ret5 < 0.10)
        & (ret20 < 0.35)
    )
    return out


def _pool(work: pd.DataFrame, base_ids: list[str], mode: str, rank_cap: int) -> pd.DataFrame:
    out = _add_scores(work)
    out = out[~out["stock_id"].isin(set(base_ids))].copy()
    score_col_by_mode = {
        "pullback": ("pass_pullback", "pullback_score"),
        "pullback_rebound": ("pass_pullback_rebound", "pullback_rebound_score"),
        "low_vol_quality": ("pass_low_vol_quality", "low_vol_quality_score"),
        "residual_momentum": ("pass_residual_momentum", "residual_momentum_score"),
    }
    pass_col, score_col = score_col_by_mode[mode]
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
        "base_top5": ",".join(base_ids),
        "base_return": base_return,
        "shadow_return": base_return,
        "delta_vs_base": 0.0,
        "accepted_swap_count": 0,
        "candidate_stock": "",
        "candidate_rank": None,
        "candidate_score": None,
        "candidate_model_rank": None,
        "candidate_ret1": None,
        "candidate_ret5": None,
        "candidate_ret20": None,
        "candidate_sigma20": None,
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
            "candidate_model_rank": float(cand["model_rank"]),
            "candidate_ret1": float(cand["ret1"]),
            "candidate_ret5": float(cand["ret5"]),
            "candidate_ret20": float(cand["ret20"]),
            "candidate_sigma20": float(cand["sigma20"]),
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
    return {
        "mean": float(s.mean()),
        "q10": float(s.quantile(0.10)),
        "worst": float(s.min()),
        "hit_rate": float((s > 0).mean()),
    }


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
    ablation = pd.read_csv(detail_dir / "ablation_decisions.csv")
    v2b = ablation[ablation["variant"] == "v2b_trend_plus_ai_overlay"].set_index("window_date")
    rows: list[dict[str, Any]] = []
    for window in sorted(inputs["default_top5"]):
        work = inputs["work_by_window"][window]
        default_ids = inputs["default_top5"][window]
        v2b_ids = _split_ids(v2b.loc[window, "selected_stocks"]) if window in v2b.index else default_ids
        v2b_changed = set(v2b_ids) != set(default_ids)
        bases = [
            ("default", default_ids, ""),
            ("v2b", v2b_ids, ""),
            ("v2b_no_swap_only", v2b_ids, "v2b_already_swapped" if v2b_changed else ""),
        ]
        for base_name, base_ids, gate in bases:
            for mode in ["pullback", "pullback_rebound", "low_vol_quality", "residual_momentum"]:
                for target_rule in ["lowest_score", "highest_risk"]:
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
    rows.to_csv(out_dir / "alpha_hunt_windows.csv", index=False)
    summary.to_csv(out_dir / "alpha_hunt_summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps({"rows": len(rows), "out_dir": str(out_dir), "summary": summary.to_dict(orient="records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows)}, ensure_ascii=False), flush=True)
    print(summary[summary["bucket"] == "60win"].sort_values("delta_mean", ascending=False).head(24).to_string(index=False))


if __name__ == "__main__":
    main()
