from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "code" / "src"))

from branch_router_v2c_shadow import SOURCE_RUN, V2B_DETAIL, _bad_counts, _realized_for_ids, build_shadow_inputs  # noqa: E402
from batch_window_analysis import load_raw, normalize_stock_id  # noqa: E402


OUT_DIR = ROOT / "temp" / "branch_router_validation" / "post_announcement_reaction_shadow"
FREEZE_WINDOWS = ROOT / "temp" / "submission_freeze" / "post_guard_overlay_queue_freeze" / "post_guard_overlay_windows.csv"

warnings.filterwarnings("ignore", category=FutureWarning)


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
    return normalize_stock_id(pd.Series([x.strip() for x in str(value).split(",") if x.strip()], dtype=str)).astype(str).tolist()


def _stock_return(work: pd.DataFrame, stock_id: str) -> float:
    row = work[work["stock_id"].astype(str) == str(stock_id).zfill(6)]
    if row.empty:
        return 0.0
    return float(pd.to_numeric(row["realized_ret"].iloc[0], errors="coerce"))


def _replace(ids: list[str], out_id: Any, in_id: Any) -> list[str]:
    out = list(ids)
    out_id = str(out_id).replace(".0", "").zfill(6)
    in_id = str(in_id).replace(".0", "").zfill(6)
    if out_id in out and in_id not in out:
        out[out.index(out_id)] = in_id
    return out


def _target(work: pd.DataFrame, ids: list[str], rule: str) -> tuple[str, float, float, float]:
    if not ids:
        return "", 0.0, 0.0, 0.0
    selected = work[work["stock_id"].astype(str).isin(set(ids))].copy()
    if selected.empty:
        return "", 0.0, 0.0, 0.0
    selected["_score"] = _num(selected, "grr_final_score")
    selected["_risk"] = _num(selected, "_risk_value", default=0.5)
    selected["_dbeta"] = _num(selected, "downside_beta60")
    if rule == "highest_risk":
        row = selected.sort_values(["_risk", "_score"], ascending=[False, True]).iloc[0]
    elif rule == "highest_dbeta":
        row = selected.sort_values(["_dbeta", "_risk", "_score"], ascending=[False, False, True]).iloc[0]
    else:
        row = selected.sort_values(["_score", "_risk"], ascending=[True, False]).iloc[0]
    return str(row["stock_id"]).zfill(6), float(row["_score"]), float(row["_risk"]), float(row["_dbeta"])


def build_reaction_panel(raw: pd.DataFrame) -> pd.DataFrame:
    panel = raw.copy()
    panel["stock_id"] = normalize_stock_id(panel["股票代码"])
    panel["date"] = pd.to_datetime(panel["日期"])
    for col in ["开盘", "收盘", "最高", "最低", "成交额", "换手率", "涨跌幅"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel = panel.sort_values(["stock_id", "date"]).reset_index(drop=True)
    group = panel.groupby("stock_id", sort=False)
    prev_close = group["收盘"].shift(1)

    panel["gap_open_ret"] = panel["开盘"] / (prev_close + 1e-12) - 1.0
    panel["intraday_ret"] = panel["收盘"] / (panel["开盘"] + 1e-12) - 1.0
    panel["close_to_high_pos"] = (panel["收盘"] - panel["最低"]) / ((panel["最高"] - panel["最低"]) + 1e-12)
    panel["ret5_raw"] = panel["收盘"] / (group["收盘"].shift(5) + 1e-12) - 1.0
    panel["ret20_raw"] = panel["收盘"] / (group["收盘"].shift(20) + 1e-12) - 1.0
    panel["amount_ma20_raw"] = group["成交额"].transform(lambda s: s.rolling(20, min_periods=10).median())
    panel["amount_rel20_raw"] = panel["成交额"] / (panel["amount_ma20_raw"] + 1e-12)
    panel["turnover_ma20_raw"] = group["换手率"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    high20 = group["最高"].transform(lambda s: s.rolling(20, min_periods=10).max())
    low20 = group["最低"].transform(lambda s: s.rolling(20, min_periods=10).min())
    panel["amp20_raw"] = high20 / (low20 + 1e-12) - 1.0
    panel["drawdown20_raw"] = panel["收盘"] / (group["收盘"].transform(lambda s: s.rolling(20, min_periods=10).max()) + 1e-12) - 1.0
    panel["gap_green"] = (panel["gap_open_ret"] >= 0.03) & (panel["intraday_ret"] > 0.0)
    panel["gap_green_sane"] = (
        panel["gap_green"]
        & (panel["gap_open_ret"] <= 0.075)
        & (panel["intraday_ret"] <= 0.07)
        & (panel["ret5_raw"] < 0.16)
        & (panel["ret20_raw"] < 0.35)
    )
    panel["gap_green_close_strong"] = panel["gap_green_sane"] & (panel["close_to_high_pos"] >= 0.65)

    keep = [
        "stock_id",
        "date",
        "gap_open_ret",
        "intraday_ret",
        "close_to_high_pos",
        "ret5_raw",
        "ret20_raw",
        "amount_rel20_raw",
        "turnover_ma20_raw",
        "amp20_raw",
        "drawdown20_raw",
        "gap_green",
        "gap_green_sane",
        "gap_green_close_strong",
    ]
    return panel[keep].replace([np.inf, -np.inf], np.nan)


def add_reaction_scores(work: pd.DataFrame) -> pd.DataFrame:
    out = work.copy()
    out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
    model = _num(out, "grr_final_score")
    ret5 = _num(out, "ret5")
    ret20 = _num(out, "ret20")
    sigma20 = _num(out, "sigma20")
    amp20 = _num(out, "amp20")
    drawdown20 = _num(out, "max_drawdown20")
    downside_beta60 = _num(out, "downside_beta60")
    liquidity = _num(out, "median_amount20")
    gap = _num(out, "gap_open_ret")
    intraday = _num(out, "intraday_ret")
    close_pos = _num(out, "close_to_high_pos", default=0.5).clip(0.0, 1.0)
    amount_rel = _num(out, "amount_rel20_raw", default=1.0)
    risk = _num(out, "_risk_value", default=0.5)

    out["model_rank"] = _rank_pct(model)
    out["liq_rank"] = _rank_pct(liquidity)
    out["gap_rank"] = _rank_pct(gap.clip(lower=-0.05, upper=0.10))
    out["intraday_rank"] = _rank_pct(intraday.clip(lower=-0.05, upper=0.10))
    out["close_pos_rank"] = _rank_pct(close_pos)
    out["amount_rel_rank"] = _rank_pct(amount_rel.clip(lower=0.0, upper=5.0))
    out["risk_rank"] = _rank_pct(risk)
    out["sigma_rank"] = _rank_pct(sigma20)
    out["amp_rank"] = _rank_pct(amp20)
    out["dbeta_rank"] = _rank_pct(downside_beta60)
    out["reaction_score"] = (
        0.24 * out["model_rank"]
        + 0.20 * out["gap_rank"]
        + 0.18 * out["intraday_rank"]
        + 0.14 * out["close_pos_rank"]
        + 0.10 * out["amount_rel_rank"]
        + 0.08 * out["liq_rank"]
        - 0.10 * out["risk_rank"]
        - 0.08 * out["amp_rank"]
    )
    out["pass_common_risk"] = (
        (sigma20 < 0.055)
        & (amp20 < 0.13)
        & (drawdown20 > -0.14)
        & (downside_beta60 < 1.60)
        & (out["liq_rank"] >= 0.25)
    )
    out["pass_timing_clean_gap"] = out.get("gap_green", False).fillna(False).astype(bool)
    out["pass_sane_gap"] = out.get("gap_green_sane", False).fillna(False).astype(bool)
    out["pass_close_strong"] = out.get("gap_green_close_strong", False).fillna(False).astype(bool)
    out["pass_model_confirmed"] = out["pass_sane_gap"] & out["pass_common_risk"] & (out["model_rank"] >= 0.70)
    out["pass_strong_confirmed"] = out["pass_close_strong"] & out["pass_common_risk"] & (out["model_rank"] >= 0.70)
    out["pass_repair_candidate"] = out["pass_sane_gap"] & out["pass_common_risk"] & (out["model_rank"] >= 0.55)
    return out


def _pool(work: pd.DataFrame, base_ids: list[str], mode: str, rank_cap: int) -> pd.DataFrame:
    out = add_reaction_scores(work)
    out = out[~out["stock_id"].isin(set(base_ids))].copy()
    pass_col_by_mode = {
        "gap_green_raw": "pass_timing_clean_gap",
        "gap_green_sane": "pass_sane_gap",
        "model_confirmed": "pass_model_confirmed",
        "strong_confirmed": "pass_strong_confirmed",
        "repair_candidate": "pass_repair_candidate",
    }
    pass_col = pass_col_by_mode[mode]
    out = out[out[pass_col]].sort_values("reaction_score", ascending=False).copy()
    if out.empty:
        return out
    out["candidate_rank"] = np.arange(1, len(out) + 1)
    out["candidate_score"] = out["reaction_score"]
    return out[out["candidate_rank"] <= int(rank_cap)]


def _eval_insert(
    work: pd.DataFrame,
    window: str,
    base_ids: list[str],
    base_name: str,
    mode: str,
    rank_cap: int,
    target_rule: str,
    target_dbeta_min: float | None = None,
    gate_reason: str = "",
) -> dict[str, Any]:
    base_return = _realized_for_ids(work, base_ids)
    base_bad, base_very_bad = _bad_counts(work, base_ids)
    target, target_score, target_risk, target_dbeta = _target(work, base_ids, target_rule)
    suffix = "" if target_dbeta_min is None else f"_dbeta{str(target_dbeta_min).replace('.', 'p')}"
    row: dict[str, Any] = {
        "window": window,
        "variant": f"{base_name}_{mode}_{target_rule}{suffix}_top{rank_cap}",
        "base_name": base_name,
        "mode": mode,
        "rank_cap": rank_cap,
        "target_rule": target_rule,
        "target_dbeta_min": target_dbeta_min,
        "base_top5": ",".join(base_ids),
        "base_return": base_return,
        "shadow_return": base_return,
        "delta_vs_base": 0.0,
        "accepted_swap_count": 0,
        "candidate_stock": "",
        "candidate_rank": None,
        "candidate_score": None,
        "candidate_gap": None,
        "candidate_intraday": None,
        "candidate_close_pos": None,
        "candidate_model_rank": None,
        "replaced_stock": target,
        "replaced_score": target_score,
        "replaced_risk": target_risk,
        "replaced_downside_beta60": target_dbeta,
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
    if target_dbeta_min is not None and target_dbeta < target_dbeta_min:
        row["blocked_reason"] = "target_dbeta_below_gate"
        return row
    candidates = _pool(work, base_ids, mode, rank_cap)
    if mode in {"model_confirmed", "strong_confirmed", "repair_candidate"} and not candidates.empty:
        candidates = candidates[_num(candidates, "grr_final_score") > float(target_score)]
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
            "candidate_gap": float(cand["gap_open_ret"]),
            "candidate_intraday": float(cand["intraday_ret"]),
            "candidate_close_pos": float(cand["close_to_high_pos"]),
            "candidate_model_rank": float(cand["model_rank"]),
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


def _frozen_ids(window: str, default_ids: list[str], v2b_ids: list[str], freeze: pd.DataFrame) -> list[str]:
    ids = list(v2b_ids or default_ids)
    if window not in freeze.index:
        return ids
    row = freeze.loc[window]
    if int(row.get("riskoff_accepted", 0)) == 1:
        ids = _replace(ids, row.get("riskoff_replaced"), row.get("riskoff_candidate"))
    if int(row.get("pullback_accepted", 0)) == 1:
        ids = _replace(ids, row.get("pullback_replaced"), row.get("pullback_candidate"))
    return ids


def event_study(inputs: dict[str, Any], panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    rows: list[pd.DataFrame] = []
    for window, work0 in inputs["work_by_window"].items():
        work = work0.merge(panel[panel["date"] == window], on="stock_id", how="left")
        work = add_reaction_scores(work)
        work["window"] = window
        rows.append(
            work[
                [
                    "window",
                    "stock_id",
                    "realized_ret",
                    "pass_timing_clean_gap",
                    "pass_sane_gap",
                    "pass_close_strong",
                    "pass_model_confirmed",
                    "pass_strong_confirmed",
                    "pass_repair_candidate",
                ]
            ]
        )
    all_events = pd.concat(rows, ignore_index=True)
    masks = {
        "gap_green_raw": all_events["pass_timing_clean_gap"].fillna(False).astype(bool),
        "gap_green_sane": all_events["pass_sane_gap"].fillna(False).astype(bool),
        "gap_green_close_strong": all_events["pass_close_strong"].fillna(False).astype(bool),
        "model_confirmed": all_events["pass_model_confirmed"].fillna(False).astype(bool),
        "strong_confirmed": all_events["pass_strong_confirmed"].fillna(False).astype(bool),
        "repair_candidate": all_events["pass_repair_candidate"].fillna(False).astype(bool),
    }
    windows = sorted(all_events["window"].astype(str).unique())
    buckets = {"20win": windows[-20:], "40win": windows[-40:], "60win": windows}
    out: list[dict[str, Any]] = []
    for bucket, keep in buckets.items():
        in_bucket = all_events["window"].astype(str).isin(set(keep))
        for name, mask in masks.items():
            sub = all_events[in_bucket & mask]
            rets = pd.to_numeric(sub["realized_ret"], errors="coerce").dropna()
            out.append(
                {
                    "bucket": bucket,
                    "event": name,
                    "count": int(len(rets)),
                    **{f"future_{k}": v for k, v in _metrics(rets).items()},
                    "positive_count": int((rets > 0).sum()),
                    "negative_count": int((rets < 0).sum()),
                }
            )
    return pd.DataFrame(out)


def run_shadow(source_run: Path, detail_dir: Path, raw_path: Path, freeze_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inputs = build_shadow_inputs(source_run, raw_path)
    raw = load_raw(raw_path)
    panel = build_reaction_panel(raw)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    ablation = pd.read_csv(detail_dir / "ablation_decisions.csv")
    v2b = ablation[ablation["variant"] == "v2b_trend_plus_ai_overlay"].set_index("window_date")
    freeze = pd.read_csv(freeze_path).set_index("window") if freeze_path.exists() else pd.DataFrame()

    rows: list[dict[str, Any]] = []
    modes = ["gap_green_raw", "gap_green_sane", "model_confirmed", "strong_confirmed", "repair_candidate"]
    target_rules = ["highest_dbeta", "highest_risk", "lowest_score"]
    for window in sorted(inputs["default_top5"]):
        day_panel = panel[panel["date"] == window].copy()
        work = inputs["work_by_window"][window].merge(day_panel, on="stock_id", how="left")
        work = add_reaction_scores(work)
        default_ids = inputs["default_top5"][window]
        v2b_ids = _split_ids(v2b.loc[window, "selected_stocks"]) if window in v2b.index else default_ids
        frozen_ids = _frozen_ids(window, default_ids, v2b_ids, freeze)
        v2b_changed = set(v2b_ids) != set(default_ids)
        for base_name, base_ids, gate in [
            ("default", default_ids, ""),
            ("v2b", v2b_ids, ""),
            ("v2b_no_swap_only", v2b_ids, "v2b_already_swapped" if v2b_changed else ""),
            ("frozen_queue", frozen_ids, ""),
        ]:
            for mode in modes:
                for target_rule in target_rules:
                    for dbeta_gate in [None, 1.2]:
                        for rank_cap in [1, 3]:
                            rows.append(_eval_insert(work, window, base_ids, base_name, mode, rank_cap, target_rule, dbeta_gate, gate_reason=gate))
    result = pd.DataFrame(rows)
    return result, summarize(result), event_study(inputs, panel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default=str(SOURCE_RUN.relative_to(ROOT)))
    parser.add_argument("--detail-dir", default=str(V2B_DETAIL.relative_to(ROOT)))
    parser.add_argument("--raw-path", default="data/train_hs300_20260424.csv")
    parser.add_argument("--freeze-path", default=str(FREEZE_WINDOWS.relative_to(ROOT)))
    parser.add_argument("--out-dir", default=str(OUT_DIR.relative_to(ROOT)))
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary, events = run_shadow(ROOT / args.source_run, ROOT / args.detail_dir, ROOT / args.raw_path, ROOT / args.freeze_path)
    rows.to_csv(out_dir / "post_announcement_reaction_windows.csv", index=False)
    summary.to_csv(out_dir / "post_announcement_reaction_summary.csv", index=False)
    events.to_csv(out_dir / "post_announcement_reaction_event_study.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "rows": len(rows),
                "top_60win": summary[summary["bucket"] == "60win"].sort_values("delta_mean", ascending=False).head(15).to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows)}, ensure_ascii=False), flush=True)
    print("\n[overlay 60win]")
    print(summary[summary["bucket"] == "60win"].sort_values("delta_mean", ascending=False).head(24).to_string(index=False))
    print("\n[event study 60win]")
    print(events[events["bucket"] == "60win"].sort_values("future_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
