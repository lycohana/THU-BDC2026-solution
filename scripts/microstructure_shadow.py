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


OUT_DIR = ROOT / "temp" / "branch_router_validation" / "microstructure_shadow"


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


def build_raw_micro_panel(raw_path: Path) -> pd.DataFrame:
    raw = load_raw(raw_path)
    panel = raw.copy()
    panel["stock_id"] = normalize_stock_id(panel["股票代码"])
    panel["date"] = pd.to_datetime(panel["日期"]).dt.strftime("%Y-%m-%d")
    for col in ["开盘", "收盘", "最高", "最低", "成交额", "换手率"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel = panel.sort_values(["stock_id", "date"]).reset_index(drop=True)
    group = panel.groupby("stock_id", sort=False)
    prev_close = group["收盘"].shift(1)
    daily_ret = panel["收盘"] / (prev_close + 1e-12) - 1.0
    intraday_ret = panel["收盘"] / (panel["开盘"] + 1e-12) - 1.0
    overnight_ret = panel["开盘"] / (prev_close + 1e-12) - 1.0
    high_jump = panel["最高"] / (prev_close + 1e-12) - 1.0
    low_drop = panel["最低"] / (prev_close + 1e-12) - 1.0
    panel["daily_ret_raw"] = daily_ret
    panel["intraday_ret"] = intraday_ret
    panel["overnight_ret"] = overnight_ret
    panel["high_jump"] = high_jump
    panel["low_drop"] = low_drop
    panel["max_ret20_raw"] = group["daily_ret_raw"].transform(lambda s: s.rolling(20, min_periods=10).max())
    panel["max_high_jump20"] = group["high_jump"].transform(lambda s: s.rolling(20, min_periods=10).max())
    panel["turnover_ma20_raw"] = group["换手率"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    panel["turnover_ma5_raw"] = group["换手率"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    panel["turnover_rel20_raw"] = panel["换手率"] / (panel["turnover_ma20_raw"] + 1e-12)
    panel["turnover_ma5_div20_raw"] = panel["turnover_ma5_raw"] / (panel["turnover_ma20_raw"] + 1e-12)
    panel["amount_ma20_raw"] = group["成交额"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    keep = [
        "stock_id",
        "date",
        "daily_ret_raw",
        "intraday_ret",
        "overnight_ret",
        "high_jump",
        "low_drop",
        "max_ret20_raw",
        "max_high_jump20",
        "turnover_rel20_raw",
        "turnover_ma5_div20_raw",
        "amount_ma20_raw",
    ]
    return panel[keep].replace([np.inf, -np.inf], np.nan)


def _target(work: pd.DataFrame, ids: list[str], rule: str) -> tuple[str, float, float]:
    selected = work[work["stock_id"].astype(str).isin(set(ids))].copy()
    if selected.empty:
        return "", 0.0, 0.0
    selected["_score"] = _num(selected, "grr_final_score")
    selected["_risk"] = _num(selected, "_risk_value", default=0.5)
    selected["_lottery"] = 0.50 * _rank_pct(_num(selected, "max_ret20_raw")) + 0.25 * _rank_pct(_num(selected, "max_high_jump20")) + 0.25 * selected["_risk"]
    if rule == "highest_risk":
        row = selected.sort_values(["_risk", "_score"], ascending=[False, True]).iloc[0]
    elif rule == "highest_lottery":
        row = selected.sort_values(["_lottery", "_risk"], ascending=[False, False]).iloc[0]
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
    ret20 = _num(out, "ret20")
    sigma20 = _num(out, "sigma20")
    amp20 = _num(out, "amp20")
    drawdown20 = _num(out, "max_drawdown20")
    downside_beta60 = _num(out, "downside_beta60")
    median_amount20 = _num(out, "median_amount20")
    model = _num(out, "grr_final_score")
    risk = _num(out, "_risk_value", default=0.5)
    max_ret20 = _num(out, "max_ret20_raw")
    max_jump20 = _num(out, "max_high_jump20")
    turnover_rel20 = _num(out, "turnover_rel20_raw", default=1.0)
    turnover_ma5_div20 = _num(out, "turnover_ma5_div20_raw", default=1.0)
    intraday = _num(out, "intraday_ret")
    overnight = _num(out, "overnight_ret")

    out["ret1"] = ret1
    out["ret5"] = ret5
    out["ret20"] = ret20
    out["sigma20"] = sigma20
    out["amp20"] = amp20
    out["max_drawdown20"] = drawdown20
    out["downside_beta60"] = downside_beta60
    out["model_score"] = model
    out["model_rank"] = _rank_pct(model)
    out["liquidity_rank"] = _rank_pct(median_amount20)
    out["risk_rank"] = _rank_pct(risk)
    out["max_ret_rank"] = _rank_pct(max_ret20)
    out["max_jump_rank"] = _rank_pct(max_jump20)
    out["turnover_rank"] = _rank_pct(turnover_rel20)
    out["turnover_trend_rank"] = _rank_pct(turnover_ma5_div20)
    out["intraday_rank"] = _rank_pct(intraday)
    out["overnight_drop_rank"] = _rank_pct(-overnight)
    out["anti_lottery_score"] = (
        0.35 * out["model_rank"]
        + 0.20 * out["liquidity_rank"]
        + 0.20 * (1.0 - out["max_ret_rank"])
        + 0.15 * (1.0 - out["max_jump_rank"])
        - 0.15 * out["risk_rank"]
    )
    out["turnover_momentum_score"] = (
        0.30 * out["model_rank"]
        + 0.22 * _rank_pct(ret5)
        + 0.18 * out["turnover_rank"]
        + 0.12 * out["turnover_trend_rank"]
        + 0.10 * out["liquidity_rank"]
        - 0.15 * out["risk_rank"]
    )
    out["low_turnover_reversal_score"] = (
        0.35 * out["model_rank"]
        + 0.22 * _rank_pct(-ret5)
        + 0.15 * (1.0 - out["turnover_rank"])
        + 0.12 * out["intraday_rank"]
        + 0.10 * out["liquidity_rank"]
        - 0.15 * out["risk_rank"]
    )
    out["gap_rebound_score"] = (
        0.30 * out["model_rank"]
        + 0.20 * out["overnight_drop_rank"]
        + 0.20 * out["intraday_rank"]
        + 0.12 * out["liquidity_rank"]
        - 0.15 * out["risk_rank"]
        - 0.10 * out["max_ret_rank"]
    )
    out["pass_common"] = (
        (out["liquidity_rank"] >= 0.30)
        & (out["model_rank"] >= 0.70)
        & (sigma20 < 0.050)
        & (amp20 < 0.10)
        & (drawdown20 > -0.13)
        & (downside_beta60 < 1.40)
    )
    out["pass_anti_lottery"] = (
        out["pass_common"]
        & (out["max_ret_rank"] <= 0.55)
        & (out["max_jump_rank"] <= 0.60)
        & (ret5 > -0.04)
        & (ret20 < 0.25)
    )
    out["pass_turnover_momentum"] = (
        out["pass_common"]
        & (out["turnover_rank"] >= 0.65)
        & (ret5 > 0.015)
        & (ret5 < 0.10)
        & (ret20 > -0.02)
        & (ret20 < 0.32)
    )
    out["pass_low_turnover_reversal"] = (
        out["pass_common"]
        & (out["turnover_rank"] <= 0.55)
        & (ret5 > -0.08)
        & (ret5 < 0.005)
        & (ret20 > -0.05)
        & (intraday > -0.015)
    )
    out["pass_gap_rebound"] = (
        out["pass_common"]
        & (overnight < -0.005)
        & (intraday > 0.0)
        & (ret5 > -0.08)
        & (ret5 < 0.04)
        & (ret20 < 0.25)
    )
    return out


def _pool(work: pd.DataFrame, base_ids: list[str], mode: str, rank_cap: int) -> pd.DataFrame:
    out = _add_scores(work)
    out = out[~out["stock_id"].isin(set(base_ids))].copy()
    mapping = {
        "anti_lottery": ("pass_anti_lottery", "anti_lottery_score"),
        "turnover_momentum": ("pass_turnover_momentum", "turnover_momentum_score"),
        "low_turnover_reversal": ("pass_low_turnover_reversal", "low_turnover_reversal_score"),
        "gap_rebound": ("pass_gap_rebound", "gap_rebound_score"),
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
        "candidate_ret5": None,
        "candidate_ret20": None,
        "candidate_max_ret20": None,
        "candidate_turnover_rel20": None,
        "candidate_overnight": None,
        "candidate_intraday": None,
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
            "candidate_ret5": float(cand["ret5"]),
            "candidate_ret20": float(cand["ret20"]),
            "candidate_max_ret20": float(cand["max_ret20_raw"]),
            "candidate_turnover_rel20": float(cand["turnover_rel20_raw"]),
            "candidate_overnight": float(cand["overnight_ret"]),
            "candidate_intraday": float(cand["intraday_ret"]),
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
    micro = build_raw_micro_panel(raw_path)
    micro_by_date = {str(date): group.drop(columns=["date"]).copy() for date, group in micro.groupby("date", sort=False)}
    ablation = pd.read_csv(detail_dir / "ablation_decisions.csv")
    v2b = ablation[ablation["variant"] == "v2b_trend_plus_ai_overlay"].set_index("window_date")
    rows: list[dict[str, Any]] = []
    for window in sorted(inputs["default_top5"]):
        work = inputs["work_by_window"][window].copy()
        day_micro = micro_by_date.get(window)
        if day_micro is not None:
            work = work.merge(day_micro, on="stock_id", how="left")
        default_ids = inputs["default_top5"][window]
        v2b_ids = _split_ids(v2b.loc[window, "selected_stocks"]) if window in v2b.index else default_ids
        v2b_changed = set(v2b_ids) != set(default_ids)
        bases = [
            ("default", default_ids, ""),
            ("v2b", v2b_ids, ""),
            ("v2b_no_swap_only", v2b_ids, "v2b_already_swapped" if v2b_changed else ""),
        ]
        for base_name, base_ids, gate in bases:
            for mode in ["anti_lottery", "turnover_momentum", "low_turnover_reversal", "gap_rebound"]:
                for target_rule in ["lowest_score", "highest_risk", "highest_lottery"]:
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
    rows.to_csv(out_dir / "microstructure_windows.csv", index=False)
    summary.to_csv(out_dir / "microstructure_summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps({"rows": len(rows), "out_dir": str(out_dir), "summary": summary.to_dict(orient="records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows)}, ensure_ascii=False), flush=True)
    print(summary[summary["bucket"] == "60win"].sort_values("delta_mean", ascending=False).head(24).to_string(index=False))


if __name__ == "__main__":
    main()
