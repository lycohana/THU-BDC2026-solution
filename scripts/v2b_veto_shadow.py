from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "code" / "src"))

from batch_window_analysis import normalize_stock_id  # noqa: E402
from branch_router_v2c_shadow import SOURCE_RUN, V2B_DETAIL, _bad_counts, _realized_for_ids, build_shadow_inputs  # noqa: E402


OUT_DIR = ROOT / "temp" / "branch_router_validation" / "v2b_veto_shadow"


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
    return normalize_stock_id(pd.Series([item.strip() for item in str(value).split(",") if item.strip()], dtype=str)).astype(str).tolist()


def _stock_return(work: pd.DataFrame, stock_id: str) -> float:
    row = work[work["stock_id"].astype(str) == str(stock_id).zfill(6)]
    if row.empty:
        return 0.0
    return float(pd.to_numeric(row["realized_ret"].iloc[0], errors="coerce"))


def _enriched_work(work: pd.DataFrame) -> pd.DataFrame:
    out = work.copy()
    out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
    model = _num(out, "grr_final_score")
    lgb = _num(out, "lgb")
    transformer = _num(out, "transformer")
    sigma20 = _num(out, "sigma20")
    amp20 = _num(out, "amp20")
    drawdown20 = _num(out, "max_drawdown20")
    idio_vol60 = _num(out, "idio_vol60")
    median_amount20 = _num(out, "median_amount20")
    ret1 = _num(out, "ret1")
    ret5 = _num(out, "ret5")
    ret10 = _num(out, "ret10")
    ret20 = _num(out, "ret20")
    amt_ratio5 = _num(out, "amt_ratio5", 1.0)
    to_ratio5 = _num(out, "to_ratio5", 1.0)
    risk = _num(out, "_risk_value", 0.5)
    out["model_rank"] = _rank_pct(model)
    out["lgb_rank"] = _rank_pct(lgb)
    out["transformer_rank"] = _rank_pct(transformer)
    out["disagreement"] = (out["lgb_rank"] - out["transformer_rank"]).abs()
    out["sigma_rank"] = _rank_pct(sigma20)
    out["amp_rank"] = _rank_pct(amp20)
    out["drawdown_bad_rank"] = _rank_pct(drawdown20)
    out["idio_rank"] = _rank_pct(idio_vol60)
    out["liquidity_rank"] = _rank_pct(median_amount20)
    out["ret1_rank"] = _rank_pct(ret1)
    out["ret5_rank"] = _rank_pct(ret5)
    out["ret10_rank"] = _rank_pct(ret10)
    out["ret20_rank"] = _rank_pct(ret20)
    out["amount_rank"] = _rank_pct(amt_ratio5)
    out["turnover_rank"] = _rank_pct(to_ratio5)
    out["risk_rank"] = _rank_pct(risk)
    return out


def _row_features(work: pd.DataFrame, stock_id: str, prefix: str) -> dict[str, float]:
    row = work[work["stock_id"].astype(str) == str(stock_id).zfill(6)]
    if row.empty:
        return {}
    r = row.iloc[0]
    cols = [
        "model_rank",
        "lgb_rank",
        "transformer_rank",
        "disagreement",
        "sigma_rank",
        "amp_rank",
        "drawdown_bad_rank",
        "idio_rank",
        "liquidity_rank",
        "ret1",
        "ret5",
        "ret10",
        "ret20",
        "amt_ratio5",
        "to_ratio5",
        "amount_rank",
        "turnover_rank",
        "risk_rank",
        "grr_consensus_norm",
    ]
    return {f"{prefix}_{col}": float(pd.to_numeric(pd.Series([r.get(col)]), errors="coerce").fillna(0.0).iloc[0]) for col in cols}


def _accepted_by_window(detail_dir: Path) -> dict[str, pd.DataFrame]:
    path = detail_dir / "accepted_swaps.csv"
    swaps = pd.read_csv(path, dtype={"candidate_stock": str, "replaced_stock": str})
    swaps = swaps[swaps["variant"] == "v2b_trend_plus_ai_overlay"].copy()
    swaps["candidate_stock"] = normalize_stock_id(swaps["candidate_stock"])
    swaps["replaced_stock"] = normalize_stock_id(swaps["replaced_stock"])
    swaps["swap_index_in_window"] = pd.to_numeric(swaps["swap_index_in_window"], errors="coerce").fillna(0).astype(int)
    return {str(window): group.sort_values("swap_index_in_window").copy() for window, group in swaps.groupby("window")}


def _default_and_v2b(detail_dir: Path) -> pd.DataFrame:
    ablation = pd.read_csv(detail_dir / "ablation_decisions.csv")
    keep = ablation[ablation["variant"].isin(["default_grr_tail_guard", "v2b_trend_plus_ai_overlay"])].copy()
    return keep.pivot(index="window_date", columns="variant", values="selected_stocks")


def _apply_veto(base_ids: list[str], swaps: pd.DataFrame, veto: Callable[[pd.Series], bool]) -> tuple[list[str], list[dict[str, Any]]]:
    final = list(base_ids)
    records: list[dict[str, Any]] = []
    for _, swap in swaps.iterrows():
        replaced = str(swap["replaced_stock"]).zfill(6)
        candidate = str(swap["candidate_stock"]).zfill(6)
        blocked = bool(veto(swap))
        if not blocked and replaced in final:
            final[final.index(replaced)] = candidate
        rec = swap.to_dict()
        rec["vetoed"] = blocked
        rec["kept_candidate"] = not blocked
        records.append(rec)
    return final, records


def _veto_rules() -> dict[str, Callable[[pd.Series], bool]]:
    def is_theme(row: pd.Series) -> bool:
        return str(row.get("branch", "")) == "theme_ai"

    def is_trend(row: pd.Series) -> bool:
        return str(row.get("branch", "")) == "trend"

    return {
        "current_v2b": lambda row: False,
        "theme_rank_cap3": lambda row: is_theme(row) and float(row.get("candidate_rank", 99)) > 3,
        "theme_rank_cap3_or_margin12": lambda row: is_theme(row)
        and (float(row.get("candidate_rank", 99)) > 3 or float(row.get("score_margin", 0.0)) < 0.12),
        "theme_rank_cap3_or_neg_ret5": lambda row: is_theme(row)
        and (float(row.get("candidate_rank", 99)) > 3 or float(row.get("cand_ret5", 0.0)) < 0.0),
        "theme_rank_cap3_or_hot_ret10": lambda row: is_theme(row)
        and (float(row.get("candidate_rank", 99)) > 3 or float(row.get("cand_ret10", 0.0)) > 0.18),
        "theme_rank_cap3_or_lottery": lambda row: is_theme(row)
        and (
            float(row.get("candidate_rank", 99)) > 3
            or float(row.get("cand_sigma_rank", 0.5)) > 0.82
            or float(row.get("cand_amp_rank", 0.5)) > 0.82
        ),
        "theme_rank_cap3_or_cooling_weak": lambda row: is_theme(row)
        and (
            float(row.get("candidate_rank", 99)) > 3
            or (
                float(row.get("cand_model_rank", 1.0)) < 0.55
                and float(row.get("cand_grr_consensus_norm", 1.0)) <= 0.01
                and float(row.get("cand_ret20", 0.0)) < 0.0
                and float(row.get("cand_amt_ratio5", 1.0)) < 0.90
                and float(row.get("cand_to_ratio5", 1.0)) < 0.90
            )
        ),
        "theme_cap_cooling_riskdelta35": lambda row: is_theme(row)
        and (
            float(row.get("candidate_rank", 99)) > 3
            or (
                float(row.get("cand_model_rank", 1.0)) < 0.55
                and float(row.get("cand_ret20", 0.0)) < 0.0
                and float(row.get("cand_amt_ratio5", 1.0)) < 0.90
                and float(row.get("cand_to_ratio5", 1.0)) < 0.90
            )
            or (
                float(row.get("candidate_rank", 99)) >= 3
                and float(row.get("risk_delta", 0.0)) > 0.35
            )
        ),
        "theme_cap_cooling_riskdelta35_trend_amp80": lambda row: (
            is_theme(row)
            and (
                float(row.get("candidate_rank", 99)) > 3
                or (
                    float(row.get("cand_model_rank", 1.0)) < 0.55
                    and float(row.get("cand_ret20", 0.0)) < 0.0
                    and float(row.get("cand_amt_ratio5", 1.0)) < 0.90
                    and float(row.get("cand_to_ratio5", 1.0)) < 0.90
                )
                or (
                    float(row.get("candidate_rank", 99)) >= 3
                    and float(row.get("risk_delta", 0.0)) > 0.35
                )
            )
        )
        or (
            is_trend(row)
            and float(row.get("candidate_rank", 99)) > 1
            and float(row.get("cand_amp_rank", 0.0)) > 0.80
        ),
        "theme_rank_cap3_trend_risk_guard": lambda row: (
            is_theme(row)
            and float(row.get("candidate_rank", 99)) > 3
        )
        or (is_trend(row) and float(row.get("risk_delta", 0.0)) > 0.12 and float(row.get("trend_dispersion", 0.0)) > 0.12),
        "all_negative_model_gap": lambda row: float(row.get("score_margin", 0.0)) < 0.10,
        "all_candidate_high_risk": lambda row: float(row.get("cand_risk_rank", 0.5)) > 0.88,
        "all_disagreement_high": lambda row: float(row.get("cand_disagreement", 0.0)) > 0.70,
    }


def summarize(rows: pd.DataFrame, default_rows: pd.DataFrame) -> pd.DataFrame:
    windows = sorted(rows["window"].astype(str).unique())
    buckets = {"20win": windows[-20:], "40win": windows[-40:], "60win": windows}
    default_lookup = default_rows.set_index("window")["default_return"].to_dict()
    v2b_lookup = default_rows.set_index("window")["v2b_return"].to_dict()
    out: list[dict[str, Any]] = []
    for bucket, keep in buckets.items():
        scoped = rows[rows["window"].astype(str).isin(set(keep))].copy()
        for rule, sub in scoped.groupby("rule"):
            returns = pd.to_numeric(sub["shadow_return"], errors="coerce").fillna(0.0)
            delta_default = sub["shadow_return"] - sub["window"].map(default_lookup).astype(float)
            delta_v2b = sub["shadow_return"] - sub["window"].map(v2b_lookup).astype(float)
            out.append(
                {
                    "bucket": bucket,
                    "rule": rule,
                    "window_count": int(len(sub)),
                    "return_mean": float(returns.mean()) if len(sub) else 0.0,
                    "return_q10": float(returns.quantile(0.10)) if len(sub) else 0.0,
                    "delta_default_mean": float(delta_default.mean()) if len(sub) else 0.0,
                    "delta_default_q10": float(delta_default.quantile(0.10)) if len(sub) else 0.0,
                    "delta_default_worst": float(delta_default.min()) if len(sub) else 0.0,
                    "delta_v2b_mean": float(delta_v2b.mean()) if len(sub) else 0.0,
                    "delta_v2b_q10": float(delta_v2b.quantile(0.10)) if len(sub) else 0.0,
                    "delta_v2b_worst": float(delta_v2b.min()) if len(sub) else 0.0,
                    "vetoed_swaps": int(pd.to_numeric(sub["vetoed_swaps"], errors="coerce").fillna(0).sum()),
                    "kept_swaps": int(pd.to_numeric(sub["kept_swaps"], errors="coerce").fillna(0).sum()),
                    "negative_delta_v2b_count": int((delta_v2b < -1e-12).sum()),
                }
            )
    return pd.DataFrame(out)


def run_shadow(source_run: Path, detail_dir: Path, raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inputs = build_shadow_inputs(source_run, raw_path)
    selected = _default_and_v2b(detail_dir)
    swaps_by_window = _accepted_by_window(detail_dir)
    rules = _veto_rules()
    rows: list[dict[str, Any]] = []
    swap_records: list[dict[str, Any]] = []
    default_rows: list[dict[str, Any]] = []
    for window in sorted(inputs["default_top5"]):
        work = _enriched_work(inputs["work_by_window"][window])
        default_ids = _split_ids(selected.loc[window, "default_grr_tail_guard"]) if window in selected.index else inputs["default_top5"][window]
        v2b_ids = _split_ids(selected.loc[window, "v2b_trend_plus_ai_overlay"]) if window in selected.index else default_ids
        swaps = swaps_by_window.get(window, pd.DataFrame()).copy()
        if not swaps.empty:
            enriched = []
            for _, swap in swaps.iterrows():
                rec = swap.to_dict()
                rec.update(_row_features(work, rec["candidate_stock"], "cand"))
                rec.update(_row_features(work, rec["replaced_stock"], "repl"))
                enriched.append(rec)
            swaps = pd.DataFrame(enriched)
        default_return = _realized_for_ids(work, default_ids)
        v2b_return = _realized_for_ids(work, v2b_ids)
        default_rows.append({"window": window, "default_return": default_return, "v2b_return": v2b_return})
        for rule_name, veto in rules.items():
            if rule_name == "current_v2b":
                final_ids = list(v2b_ids)
                records = []
            else:
                final_ids, records = _apply_veto(default_ids, swaps, veto)
            shadow_return = _realized_for_ids(work, final_ids)
            bad, very_bad = _bad_counts(work, final_ids)
            rows.append(
                {
                    "window": window,
                    "rule": rule_name,
                    "default_top5": ",".join(default_ids),
                    "v2b_top5": ",".join(v2b_ids),
                    "shadow_top5": ",".join(final_ids),
                    "default_return": default_return,
                    "v2b_return": v2b_return,
                    "shadow_return": shadow_return,
                    "delta_vs_default": shadow_return - default_return,
                    "delta_vs_v2b": shadow_return - v2b_return,
                    "accepted_swaps": int(len(swaps)),
                    "vetoed_swaps": int(sum(1 for rec in records if rec.get("vetoed"))),
                    "kept_swaps": int(sum(1 for rec in records if rec.get("kept_candidate"))),
                    "bad_count": bad,
                    "very_bad_count": very_bad,
                }
            )
            for rec in records:
                rec["window"] = window
                rec["rule"] = rule_name
                swap_records.append(rec)
    windows = pd.DataFrame(rows)
    base = pd.DataFrame(default_rows)
    return windows, summarize(windows, base), pd.DataFrame(swap_records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default=str(SOURCE_RUN.relative_to(ROOT)))
    parser.add_argument("--detail-dir", default=str(V2B_DETAIL.relative_to(ROOT)))
    parser.add_argument("--raw-path", default="data/train_hs300_20260424.csv")
    parser.add_argument("--out-dir", default=str(OUT_DIR.relative_to(ROOT)))
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    windows, summary, swaps = run_shadow(ROOT / args.source_run, ROOT / args.detail_dir, ROOT / args.raw_path)
    windows.to_csv(out_dir / "v2b_veto_windows.csv", index=False)
    summary.to_csv(out_dir / "v2b_veto_summary.csv", index=False)
    swaps.to_csv(out_dir / "v2b_veto_swap_audit.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps({"rows": len(windows), "summary": summary.to_dict(orient="records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "rows": len(windows)}, ensure_ascii=False), flush=True)
    print(summary[summary["bucket"] == "60win"].sort_values("delta_v2b_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
