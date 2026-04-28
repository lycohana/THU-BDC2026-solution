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


OUT_DIR = ROOT / "temp" / "branch_router_validation" / "smartstable_minicap_shadow"
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


def _target(work: pd.DataFrame, ids: list[str], rule: str) -> tuple[str, float, float]:
    if not ids:
        return "", 0.0, 0.0
    selected = work[work["stock_id"].astype(str).isin(set(ids))].copy()
    if selected.empty:
        return "", 0.0, 0.0
    selected["_score"] = _num(selected, "grr_final_score")
    selected["_risk"] = _num(selected, "_risk_value", default=0.5)
    selected["_downside_beta"] = _num(selected, "downside_beta60")
    if rule == "highest_risk":
        row = selected.sort_values(["_risk", "_score"], ascending=[False, True]).iloc[0]
    elif rule == "highest_dbeta":
        row = selected.sort_values(["_downside_beta", "_risk", "_score"], ascending=[False, False, True]).iloc[0]
    else:
        row = selected.sort_values(["_score", "_risk"], ascending=[True, False]).iloc[0]
    return str(row["stock_id"]).zfill(6), float(row["_score"]), float(row["_risk"])


def build_minicap_panel(raw: pd.DataFrame) -> pd.DataFrame:
    panel = raw.copy()
    panel["stock_id"] = normalize_stock_id(panel["股票代码"])
    panel["date"] = pd.to_datetime(panel["日期"])
    for col in ["开盘", "收盘", "最高", "最低", "成交额", "换手率"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel = panel.sort_values(["stock_id", "date"]).reset_index(drop=True)
    group = panel.groupby("stock_id", sort=False)
    close = panel["收盘"]
    amount = panel["成交额"]
    turnover_decimal = panel["换手率"] / 100.0

    ma3 = group["收盘"].transform(lambda s: s.rolling(3, min_periods=3).mean())
    ma6 = group["收盘"].transform(lambda s: s.rolling(6, min_periods=4).mean())
    ma12 = group["收盘"].transform(lambda s: s.rolling(12, min_periods=8).mean())
    ma24 = group["收盘"].transform(lambda s: s.rolling(24, min_periods=16).mean())
    panel["bbi"] = (ma3 + ma6 + ma12 + ma24) / 4.0
    panel["close_div_bbi"] = close / (panel["bbi"] + 1e-12)
    panel["bbi_up"] = close > panel["bbi"]
    panel["mcap_proxy"] = amount / (turnover_decimal.replace(0.0, np.nan) + 1e-12)
    panel["amount_ma20_raw"] = group["成交额"].transform(lambda s: s.rolling(20, min_periods=10).median())
    panel["turnover_ma20_raw"] = group["换手率"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    panel["ret20_raw"] = close / (group["收盘"].shift(20) + 1e-12) - 1.0
    panel["ret5_raw"] = close / (group["收盘"].shift(5) + 1e-12) - 1.0
    high20 = group["最高"].transform(lambda s: s.rolling(20, min_periods=10).max())
    low20 = group["最低"].transform(lambda s: s.rolling(20, min_periods=10).min())
    panel["amp20_raw"] = high20 / (low20 + 1e-12) - 1.0
    panel["drawdown20_raw"] = close / (group["收盘"].transform(lambda s: s.rolling(20, min_periods=10).max()) + 1e-12) - 1.0

    keep = [
        "stock_id",
        "date",
        "收盘",
        "bbi",
        "close_div_bbi",
        "bbi_up",
        "mcap_proxy",
        "amount_ma20_raw",
        "turnover_ma20_raw",
        "ret5_raw",
        "ret20_raw",
        "amp20_raw",
        "drawdown20_raw",
    ]
    return panel[keep].replace([np.inf, -np.inf], np.nan)


def add_minicap_scores(work: pd.DataFrame) -> pd.DataFrame:
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
    risk = _num(out, "_risk_value", default=0.5)
    mcap_proxy = _num(out, "mcap_proxy")
    close_div_bbi = _num(out, "close_div_bbi", default=1.0)
    close_price = _num(out, "收盘")

    out["model_rank"] = _rank_pct(model)
    out["smallcap_rank"] = _rank_pct(mcap_proxy, ascending=False)
    out["liq_rank"] = _rank_pct(liquidity)
    out["bbi_strength_rank"] = _rank_pct(close_div_bbi.clip(lower=0.90, upper=1.20))
    out["risk_rank"] = _rank_pct(risk)
    out["sigma_rank"] = _rank_pct(sigma20)
    out["amp_rank"] = _rank_pct(amp20)
    out["dbeta_rank"] = _rank_pct(downside_beta60)
    out["mcap_proxy"] = mcap_proxy
    out["close_price"] = close_price
    out["bbi_up"] = out.get("bbi_up", False).fillna(False).astype(bool)

    out["smartstable_score"] = (
        0.32 * out["smallcap_rank"]
        + 0.22 * out["bbi_strength_rank"]
        + 0.20 * out["model_rank"]
        + 0.10 * out["liq_rank"]
        - 0.10 * out["risk_rank"]
        - 0.08 * out["amp_rank"]
        - 0.06 * out["dbeta_rank"]
    )
    out["smartstable_nomodel_score"] = (
        0.46 * out["smallcap_rank"]
        + 0.26 * out["bbi_strength_rank"]
        + 0.12 * out["liq_rank"]
        - 0.10 * out["risk_rank"]
        - 0.08 * out["amp_rank"]
    )
    out["mcap_5_100b"] = (mcap_proxy >= 5e9) & (mcap_proxy <= 1e11)
    out["mcap_5_300b"] = (mcap_proxy >= 5e9) & (mcap_proxy <= 3e11)
    out["pass_quality_risk"] = (
        (sigma20 < 0.050)
        & (amp20 < 0.110)
        & (drawdown20 > -0.13)
        & (downside_beta60 < 1.40)
        & (out["liq_rank"] >= 0.25)
    )
    out["pass_defensive_risk"] = (
        (sigma20 < 0.040)
        & (amp20 < 0.090)
        & (drawdown20 > -0.10)
        & (downside_beta60 < 1.10)
        & (out["liq_rank"] >= 0.30)
    )
    out["pass_trend_sane"] = (ret5 > -0.06) & (ret5 < 0.12) & (ret20 > -0.08) & (ret20 < 0.30)
    out["pass_not_overheated"] = (ret5 < 0.08) & (ret20 < 0.25)
    return out


def _pool(work: pd.DataFrame, base_ids: list[str], mode: str, rank_cap: int) -> pd.DataFrame:
    out = add_minicap_scores(work)
    out = out[~out["stock_id"].isin(set(base_ids))].copy()
    if mode == "exact_5_100b":
        cond = out["bbi_up"] & out["mcap_5_100b"] & out["pass_quality_risk"] & out["pass_trend_sane"]
        score_col = "smartstable_nomodel_score"
    elif mode == "relaxed_5_300b":
        cond = out["bbi_up"] & out["mcap_5_300b"] & (out["smallcap_rank"] >= 0.45) & out["pass_quality_risk"] & out["pass_trend_sane"]
        score_col = "smartstable_score"
    elif mode == "model_confirmed":
        cond = (
            out["bbi_up"]
            & (out["smallcap_rank"] >= 0.50)
            & (out["model_rank"] >= 0.70)
            & out["pass_quality_risk"]
            & out["pass_trend_sane"]
        )
        score_col = "smartstable_score"
    elif mode == "defensive_bbi":
        cond = (
            out["bbi_up"]
            & (out["smallcap_rank"] >= 0.45)
            & out["pass_defensive_risk"]
            & out["pass_trend_sane"]
        )
        score_col = "smartstable_score"
    elif mode == "not_overheated":
        cond = (
            out["bbi_up"]
            & (out["smallcap_rank"] >= 0.50)
            & out["pass_quality_risk"]
            & out["pass_not_overheated"]
        )
        score_col = "smartstable_score"
    elif mode == "smallest_bbi_raw":
        cond = out["bbi_up"] & (out["smallcap_rank"] >= 0.60)
        score_col = "smartstable_nomodel_score"
    else:
        raise ValueError(f"unknown mode: {mode}")
    out = out[cond].sort_values(score_col, ascending=False).copy()
    if out.empty:
        return out
    out["candidate_rank"] = np.arange(1, len(out) + 1)
    out["candidate_score"] = out[score_col]
    return out[out["candidate_rank"] <= int(rank_cap)]


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
    row: dict[str, Any] = {
        "window": window,
        "variant": f"{base_name}_{mode}_{target_rule}_top{rank_cap}",
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
        "candidate_mcap_proxy": None,
        "candidate_smallcap_rank": None,
        "candidate_close_div_bbi": None,
        "candidate_model_rank": None,
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
    if mode in {"model_confirmed", "relaxed_5_300b", "defensive_bbi", "not_overheated"} and not candidates.empty:
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
            "candidate_mcap_proxy": float(cand["mcap_proxy"]),
            "candidate_smallcap_rank": float(cand["smallcap_rank"]),
            "candidate_close_div_bbi": float(cand["close_div_bbi"]),
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


def summarize(rows: pd.DataFrame, value_col: str = "delta_vs_base") -> pd.DataFrame:
    windows = sorted(rows["window"].astype(str).unique())
    buckets = {"20win": windows[-20:], "40win": windows[-40:], "60win": windows}
    out_rows: list[dict[str, Any]] = []
    for bucket, keep in buckets.items():
        scoped = rows[rows["window"].astype(str).isin(set(keep))].copy()
        for variant, sub in scoped.groupby("variant"):
            delta = pd.to_numeric(sub[value_col], errors="coerce").fillna(0.0)
            out_rows.append(
                {
                    "bucket": bucket,
                    "variant": variant,
                    "window_count": int(len(sub)),
                    "accepted_swaps": int(pd.to_numeric(sub.get("accepted_swap_count", 0), errors="coerce").fillna(0).sum()),
                    "accepted_swap_rate": float(pd.to_numeric(sub.get("accepted_swap_count", 0), errors="coerce").fillna(0).mean()) if len(sub) else 0.0,
                    **{f"return_{k}": v for k, v in _metrics(sub["shadow_return"]).items()},
                    **{f"delta_{k}": v for k, v in _metrics(sub[value_col]).items()},
                    "negative_delta_count": int((delta < -1e-12).sum()),
                    "very_bad_mean": float(pd.to_numeric(sub.get("very_bad_count", 0), errors="coerce").fillna(0).mean()) if len(sub) else 0.0,
                }
            )
    return pd.DataFrame(out_rows)


def standalone_rows(inputs: dict[str, Any], panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    rows: list[dict[str, Any]] = []
    for window in sorted(inputs["default_top5"]):
        day_panel = panel[panel["date"] == window].copy()
        work = inputs["work_by_window"][window].merge(day_panel, on=["stock_id"], how="left", suffixes=("", "_mini"))
        work = add_minicap_scores(work)
        default_ids = inputs["default_top5"][window]
        default_return = _realized_for_ids(work, default_ids)
        for mode in ["exact_5_100b", "relaxed_5_300b", "model_confirmed", "defensive_bbi", "not_overheated", "smallest_bbi_raw"]:
            pool = _pool(work, [], mode, 8)
            selected = pool.head(5)["stock_id"].astype(str).str.zfill(6).tolist()
            if not selected:
                shadow_return = default_return
                selected_text = ""
                blocked = "no_candidate"
            else:
                shadow_return = _realized_for_ids(work, selected)
                selected_text = ",".join(selected)
                blocked = ""
            bad, very_bad = _bad_counts(work, selected if selected else default_ids)
            rows.append(
                {
                    "window": window,
                    "variant": f"standalone_{mode}_top5",
                    "selected_stocks": selected_text,
                    "base_return": default_return,
                    "shadow_return": shadow_return,
                    "delta_vs_base": shadow_return - default_return,
                    "accepted_swap_count": int(bool(selected)),
                    "blocked_reason": blocked,
                    "bad_count": bad,
                    "very_bad_count": very_bad,
                }
            )
    return pd.DataFrame(rows)


def event_study(inputs: dict[str, Any], panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    rows: list[pd.DataFrame] = []
    for window, work0 in inputs["work_by_window"].items():
        work = work0.merge(panel[panel["date"] == window], on=["stock_id"], how="left", suffixes=("", "_mini"))
        work = add_minicap_scores(work)
        work["window"] = window
        rows.append(
            work[
                [
                    "window",
                    "stock_id",
                    "realized_ret",
                    "bbi_up",
                    "mcap_5_100b",
                    "mcap_5_300b",
                    "smallcap_rank",
                    "pass_quality_risk",
                    "pass_defensive_risk",
                    "pass_trend_sane",
                    "pass_not_overheated",
                    "model_rank",
                ]
            ]
        )
    all_events = pd.concat(rows, ignore_index=True)
    masks = {
        "bbi_up": all_events["bbi_up"].fillna(False).astype(bool),
        "exact_5_100b": all_events["bbi_up"].fillna(False).astype(bool)
        & all_events["mcap_5_100b"].fillna(False).astype(bool)
        & all_events["pass_quality_risk"].fillna(False).astype(bool)
        & all_events["pass_trend_sane"].fillna(False).astype(bool),
        "relaxed_5_300b": all_events["bbi_up"].fillna(False).astype(bool)
        & all_events["mcap_5_300b"].fillna(False).astype(bool)
        & (all_events["smallcap_rank"] >= 0.45)
        & all_events["pass_quality_risk"].fillna(False).astype(bool)
        & all_events["pass_trend_sane"].fillna(False).astype(bool),
        "model_confirmed": all_events["bbi_up"].fillna(False).astype(bool)
        & (all_events["smallcap_rank"] >= 0.50)
        & (all_events["model_rank"] >= 0.70)
        & all_events["pass_quality_risk"].fillna(False).astype(bool)
        & all_events["pass_trend_sane"].fillna(False).astype(bool),
        "defensive_bbi": all_events["bbi_up"].fillna(False).astype(bool)
        & (all_events["smallcap_rank"] >= 0.45)
        & all_events["pass_defensive_risk"].fillna(False).astype(bool)
        & all_events["pass_trend_sane"].fillna(False).astype(bool),
        "not_overheated": all_events["bbi_up"].fillna(False).astype(bool)
        & (all_events["smallcap_rank"] >= 0.50)
        & all_events["pass_quality_risk"].fillna(False).astype(bool)
        & all_events["pass_not_overheated"].fillna(False).astype(bool),
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


def run_shadow(source_run: Path, detail_dir: Path, raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inputs = build_shadow_inputs(source_run, raw_path)
    raw = load_raw(raw_path)
    panel = build_minicap_panel(raw)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")

    ablation = pd.read_csv(detail_dir / "ablation_decisions.csv")
    v2b = ablation[ablation["variant"] == "v2b_trend_plus_ai_overlay"].set_index("window_date")

    rows: list[dict[str, Any]] = []
    modes = ["exact_5_100b", "relaxed_5_300b", "model_confirmed", "defensive_bbi", "not_overheated", "smallest_bbi_raw"]
    for window in sorted(inputs["default_top5"]):
        day_panel = panel[panel["date"] == window].copy()
        work = inputs["work_by_window"][window].merge(day_panel, on=["stock_id"], how="left", suffixes=("", "_mini"))
        work = add_minicap_scores(work)
        default_ids = inputs["default_top5"][window]
        v2b_ids = _split_ids(v2b.loc[window, "selected_stocks"]) if window in v2b.index else default_ids
        v2b_changed = set(v2b_ids) != set(default_ids)
        for base_name, base_ids, gate in [
            ("default", default_ids, ""),
            ("v2b", v2b_ids, ""),
            ("v2b_no_swap_only", v2b_ids, "v2b_already_swapped" if v2b_changed else ""),
        ]:
            for mode in modes:
                for target_rule in ["lowest_score", "highest_risk", "highest_dbeta"]:
                    for rank_cap in [1, 3, 5]:
                        rows.append(_eval_insert(work, window, base_ids, base_name, mode, rank_cap, target_rule, gate_reason=gate))
    overlay = pd.DataFrame(rows)
    standalone = standalone_rows(inputs, panel)
    return overlay, summarize(overlay), standalone, summarize(standalone), event_study(inputs, panel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default=str(SOURCE_RUN.relative_to(ROOT)))
    parser.add_argument("--detail-dir", default=str(V2B_DETAIL.relative_to(ROOT)))
    parser.add_argument("--raw-path", default="data/train_hs300_20260424.csv")
    parser.add_argument("--out-dir", default=str(OUT_DIR.relative_to(ROOT)))
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay, overlay_summary, standalone, standalone_summary, events = run_shadow(ROOT / args.source_run, ROOT / args.detail_dir, ROOT / args.raw_path)
    overlay.to_csv(out_dir / "smartstable_overlay_windows.csv", index=False)
    overlay_summary.to_csv(out_dir / "smartstable_overlay_summary.csv", index=False)
    standalone.to_csv(out_dir / "smartstable_standalone_windows.csv", index=False)
    standalone_summary.to_csv(out_dir / "smartstable_standalone_summary.csv", index=False)
    events.to_csv(out_dir / "smartstable_event_study.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "overlay_rows": len(overlay),
                "standalone_rows": len(standalone),
                "top_overlay_60win": overlay_summary[overlay_summary["bucket"] == "60win"].sort_values("delta_mean", ascending=False).head(10).to_dict(orient="records"),
                "top_standalone_60win": standalone_summary[standalone_summary["bucket"] == "60win"].sort_values("delta_mean", ascending=False).head(10).to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "overlay_rows": len(overlay), "standalone_rows": len(standalone)}, ensure_ascii=False), flush=True)
    print("\n[overlay 60win]")
    print(overlay_summary[overlay_summary["bucket"] == "60win"].sort_values("delta_mean", ascending=False).head(18).to_string(index=False))
    print("\n[standalone 60win]")
    print(standalone_summary[standalone_summary["bucket"] == "60win"].sort_values("delta_mean", ascending=False).head(12).to_string(index=False))
    print("\n[event study 60win]")
    print(events[events["bucket"] == "60win"].sort_values("future_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
