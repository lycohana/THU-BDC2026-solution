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


OUT_DIR = ROOT / "temp" / "branch_router_validation" / "chase_momentum_shadow"


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


def _lowest_score_target(work: pd.DataFrame, ids: list[str]) -> tuple[str, float]:
    if not ids:
        return "", 0.0
    score_col = "grr_final_score" if "grr_final_score" in work.columns else "score"
    selected = work[work["stock_id"].astype(str).isin(set(ids))].copy()
    selected["_score"] = pd.to_numeric(selected.get(score_col, 0.0), errors="coerce").fillna(0.0)
    if selected.empty:
        return "", 0.0
    row = selected.sort_values("_score", ascending=True).iloc[0]
    return str(row["stock_id"]).zfill(6), float(row["_score"])


def _stock_return(work: pd.DataFrame, stock_id: str) -> float:
    row = work[work["stock_id"].astype(str) == str(stock_id).zfill(6)]
    if row.empty:
        return 0.0
    return float(pd.to_numeric(row["realized_ret"].iloc[0], errors="coerce"))


def _add_chase_scores(work: pd.DataFrame) -> pd.DataFrame:
    out = work.copy()
    out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
    ret5 = _num(out, "ret5")
    ret20 = _num(out, "ret20")
    sigma20 = _num(out, "sigma20")
    amp20 = _num(out, "amp20")
    drawdown20 = _num(out, "max_drawdown20")
    downside_beta60 = _num(out, "downside_beta60")
    median_amount20 = _num(out, "median_amount20")
    risk_value = _num(out, "_risk_value", default=0.5)
    model_score = _num(out, "grr_final_score", default=0.0)

    out["ret5"] = ret5
    out["ret20"] = ret20
    out["sigma20"] = sigma20
    out["amp20"] = amp20
    out["max_drawdown20"] = drawdown20
    out["downside_beta60"] = downside_beta60
    out["liquidity_rank"] = _rank_pct(median_amount20)
    out["ret5_rank"] = _rank_pct(ret5)
    out["ret20_rank"] = _rank_pct(ret20)
    out["ret5_abs"] = ret5.abs()
    out["momo_efficiency"] = ret5 / (sigma20.abs() + 1e-6)
    out["momo_eff_rank"] = _rank_pct(out["momo_efficiency"])
    out["risk_rank_local"] = _rank_pct(risk_value)
    out["model_score"] = model_score
    out["model_score_rank"] = _rank_pct(model_score)
    out["drawdown_bad_rank"] = _rank_pct(-drawdown20)
    out["late_heat"] = ((ret5 - 0.12) / 0.08).clip(lower=0.0, upper=1.0) + ((ret20 - 0.35) / 0.20).clip(lower=0.0, upper=1.0)

    out["naive_chase_score"] = 0.65 * out["ret5_rank"] + 0.35 * out["ret20_rank"]
    out["risk_managed_chase_score"] = (
        0.35 * out["momo_eff_rank"]
        + 0.25 * out["ret5_rank"]
        + 0.15 * out["ret20_rank"]
        + 0.15 * out["liquidity_rank"]
        - 0.20 * out["risk_rank_local"]
        - 0.15 * out["late_heat"]
    )
    out["smooth_chase_score"] = (
        0.30 * out["ret20_rank"]
        + 0.25 * out["momo_eff_rank"]
        + 0.20 * out["liquidity_rank"]
        - 0.20 * out["drawdown_bad_rank"]
        - 0.15 * out["risk_rank_local"]
    )
    out["confirmed_chase_score"] = (
        0.30 * out["model_score_rank"]
        + 0.25 * out["momo_eff_rank"]
        + 0.20 * out["ret5_rank"]
        + 0.15 * out["liquidity_rank"]
        - 0.15 * out["risk_rank_local"]
        - 0.10 * out["late_heat"]
    )
    out["pass_basic_liquidity"] = out["liquidity_rank"] >= 0.30
    out["pass_chase_riskcap"] = (
        out["pass_basic_liquidity"]
        & (ret5 > 0.015)
        & (ret5 < 0.12)
        & (ret20 > 0.0)
        & (ret20 < 0.35)
        & (sigma20 < 0.045)
        & (amp20 < 0.09)
        & (drawdown20 > -0.12)
        & (downside_beta60 < 1.35)
    )
    out["pass_smooth_chase"] = (
        out["pass_basic_liquidity"]
        & (ret5 > 0.0)
        & (ret5 < 0.08)
        & (ret20 > 0.03)
        & (ret20 < 0.28)
        & (sigma20 < 0.040)
        & (drawdown20 > -0.10)
        & (downside_beta60 < 1.25)
    )
    return out


def _candidate_pool(work: pd.DataFrame, base_ids: list[str], mode: str, rank_cap: int) -> pd.DataFrame:
    out = _add_chase_scores(work)
    out = out[~out["stock_id"].isin(set(base_ids))].copy()
    if mode == "naive_ret5":
        out = out[(out["ret5"] > 0.0) & out["pass_basic_liquidity"]]
        score_col = "naive_chase_score"
    elif mode == "naive_hot":
        out = out[(out["ret5"] > 0.08) & out["pass_basic_liquidity"]]
        score_col = "naive_chase_score"
    elif mode == "risk_managed":
        out = out[out["pass_chase_riskcap"]]
        score_col = "risk_managed_chase_score"
    elif mode == "smooth_trend":
        out = out[out["pass_smooth_chase"]]
        score_col = "smooth_chase_score"
    elif mode == "model_confirmed":
        out = out[out["pass_chase_riskcap"] & (out["model_score_rank"] >= 0.70)]
        score_col = "confirmed_chase_score"
    elif mode == "model_confirmed_strict":
        out = out[
            out["pass_smooth_chase"]
            & (out["model_score_rank"] >= 0.80)
            & (out["momo_eff_rank"] >= 0.60)
        ]
        score_col = "confirmed_chase_score"
    else:
        raise ValueError(f"unknown chase mode: {mode}")
    if out.empty:
        return out
    out = out.sort_values(score_col, ascending=False).copy()
    out["candidate_rank"] = np.arange(1, len(out) + 1)
    out = out[out["candidate_rank"] <= int(rank_cap)]
    return out.sort_values(["candidate_rank", score_col], ascending=[True, False])


def _eval_insert(
    work: pd.DataFrame,
    window: str,
    base_ids: list[str],
    base_name: str,
    mode: str,
    rank_cap: int,
    gate_reason: str = "",
) -> dict[str, Any]:
    base_return = _realized_for_ids(work, base_ids)
    base_bad, base_very_bad = _bad_counts(work, base_ids)
    target, target_score = _lowest_score_target(work, base_ids)
    row: dict[str, Any] = {
        "window": window,
        "variant": f"{base_name}_{mode}_top{rank_cap}",
        "base_name": base_name,
        "mode": mode,
        "rank_cap": rank_cap,
        "base_top5": ",".join(base_ids),
        "base_return": base_return,
        "shadow_return": base_return,
        "delta_vs_base": 0.0,
        "accepted_swap_count": 0,
        "candidate_stock": "",
        "candidate_rank": None,
        "candidate_score": None,
        "candidate_ret5": None,
        "candidate_ret20": None,
        "candidate_sigma20": None,
        "candidate_risk": None,
        "replaced_stock": target,
        "replaced_score": target_score,
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
    pool = _candidate_pool(work, base_ids, mode, rank_cap)
    if mode.startswith("model_confirmed") and not pool.empty:
        pool = pool[pool["model_score"] > float(target_score)]
    if pool.empty:
        row["blocked_reason"] = "no_candidate"
        return row
    cand = pool.iloc[0]
    candidate_stock = str(cand["stock_id"]).zfill(6)
    final_ids = list(base_ids)
    final_ids[final_ids.index(target)] = candidate_stock
    shadow_return = _realized_for_ids(work, final_ids)
    bad, very_bad = _bad_counts(work, final_ids)
    raw_candidate_return = _stock_return(work, candidate_stock)
    raw_replaced_return = _stock_return(work, target)
    raw_delta = raw_candidate_return - raw_replaced_return
    score_col = {
        "naive_ret5": "naive_chase_score",
        "naive_hot": "naive_chase_score",
        "risk_managed": "risk_managed_chase_score",
        "smooth_trend": "smooth_chase_score",
        "model_confirmed": "confirmed_chase_score",
        "model_confirmed_strict": "confirmed_chase_score",
    }[mode]
    row.update(
        {
            "shadow_return": shadow_return,
            "delta_vs_base": shadow_return - base_return,
            "accepted_swap_count": 1,
            "candidate_stock": candidate_stock,
            "candidate_rank": int(cand["candidate_rank"]),
            "candidate_score": float(cand[score_col]),
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
        for base_name, base_ids, gate in [
            ("default", default_ids, ""),
            ("v2b", v2b_ids, ""),
            ("v2b_no_swap_only", v2b_ids, "v2b_already_swapped" if v2b_changed else ""),
        ]:
            for mode in ["naive_ret5", "naive_hot", "risk_managed", "smooth_trend", "model_confirmed", "model_confirmed_strict"]:
                for rank_cap in [1, 3, 5]:
                    rows.append(_eval_insert(work, window, base_ids, base_name, mode, rank_cap, gate_reason=gate))
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
    rows.to_csv(out_dir / "chase_momentum_windows.csv", index=False)
    summary.to_csv(out_dir / "chase_momentum_summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps({"rows": len(rows), "out_dir": str(out_dir), "summary": summary.to_dict(orient="records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows)}, ensure_ascii=False), flush=True)
    print(summary[summary["bucket"] == "60win"].sort_values("delta_mean", ascending=False).head(18).to_string(index=False))


if __name__ == "__main__":
    main()
