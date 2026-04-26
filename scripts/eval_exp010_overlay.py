from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_exp009_windows as win_eval  # noqa: E402
import replay_tri_state_selector as replay  # noqa: E402
from tri_state_selector import SelectorConfig, TriStateSelector  # noqa: E402
from tri_state_selector.data import normalize_panel  # noqa: E402
from tri_state_selector.selector import _return_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exp010: exp008 core with exp009 veto-only diagnostics.")
    parser.add_argument("--train", default=str(ROOT / "data" / "train.csv"))
    parser.add_argument("--test", default=str(ROOT / "data" / "test.csv"))
    parser.add_argument(
        "--exp008-dir",
        default=str(ROOT / "temp" / "batch_window_analysis" / "exp008_tri_state_tight_last10_to_20260424"),
    )
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "exp010_exp008_core_with_exp009_veto"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    return parser.parse_args()


def load_core_result(exp008_dir: Path, asof: pd.Timestamp, summary_row: pd.Series) -> pd.DataFrame:
    result_path = exp008_dir / asof.strftime("%Y%m%d") / "result.csv"
    if result_path.exists():
        result = pd.read_csv(result_path, dtype={"stock_id": str})
        result["stock_id"] = result["stock_id"].astype(str).str.zfill(6)
        result["weight"] = pd.to_numeric(result["weight"], errors="coerce").fillna(0.0)
        return result[["stock_id", "weight"]].head(5)
    picks = str(summary_row["selected_picks"]).split(",")
    picks = [p.strip().zfill(6) for p in picks if p.strip()]
    return pd.DataFrame({"stock_id": picks[:5], "weight": np.full(min(5, len(picks)), 1.0 / min(5, len(picks)))})


def load_candidate_frame(exp008_dir: Path, asof: pd.Timestamp) -> pd.DataFrame:
    window_dir = exp008_dir / asof.strftime("%Y%m%d")
    for name in ["predict_filtered_top30.csv", "predict_score_df.csv"]:
        path = window_dir / name
        if path.exists():
            df = pd.read_csv(path, dtype={"stock_id": str})
            df["stock_id"] = df["stock_id"].astype(str).str.zfill(6)
            return df
    return pd.DataFrame(columns=["stock_id"])


def exp009_diagnostics(panel: pd.DataFrame, asof: pd.Timestamp) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    selector = TriStateSelector(
        SelectorConfig(
            mode="competition",
            output_top_k=5,
            force_top_k_full_invest=True,
            regime_hysteresis_days=1,
            top_quantile=0.40,
        )
    )
    hist_panel = panel[pd.to_datetime(panel["date"]) <= asof].copy()
    output = selector.rebalance(prices=hist_panel, asof=asof, prev_weights=pd.Series(dtype=float))
    returns = _return_matrix(hist_panel, asof)
    return output, output.ranked_candidates.copy(), returns


def core_corr_flag(core: pd.DataFrame, returns: pd.DataFrame, cap: float) -> tuple[bool, float]:
    names = core["stock_id"].astype(str).str.zfill(6).tolist()
    aligned = returns.reindex(columns=names).dropna(how="all")
    if aligned.shape[1] < 2:
        return False, 0.0
    corr = aligned.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mask = ~np.eye(len(corr), dtype=bool)
    max_pair = float(corr.where(mask).max().max())
    return max_pair >= cap, max_pair


def add_risk_columns(candidates: pd.DataFrame, exp009_ranked: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    if out.empty:
        return out
    out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
    diag_cols = ["TAIL_RISK", "VOL_20", "BETA_60", "IVOL_60"]
    diag = exp009_ranked.reindex(out["stock_id"])[[c for c in diag_cols if c in exp009_ranked.columns]]
    diag = diag.reset_index().rename(columns={"stock_id": "stock_id", "index": "stock_id"})
    if "stock_id" not in diag.columns:
        diag["stock_id"] = out["stock_id"].to_numpy()
    out = out.merge(diag, on="stock_id", how="left", suffixes=("", "_exp009"))
    risk_parts = []
    for col in ["risk_score", "effective_risk_score", "tail_risk_score", "sigma_rank", "beta60_rank", "downside_beta60_rank", "max_drawdown20_rank"]:
        if col in out.columns:
            risk_parts.append(pd.to_numeric(out[col], errors="coerce").fillna(0.0))
    for col in ["TAIL_RISK", "VOL_20", "BETA_60", "IVOL_60"]:
        if col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
            risk_parts.append((values.rank(pct=True) if len(values) > 1 else values * 0.0).fillna(0.0))
    for col in ["high_vol_flag", "very_high_vol_flag", "tail_risk_flag", "very_tail_flag", "extreme_momo_flag"]:
        if col in out.columns:
            risk_parts.append(out[col].fillna(False).astype(bool).astype(float))
    out["exp010_risk_score"] = pd.concat(risk_parts, axis=1).mean(axis=1) if risk_parts else 0.0
    return out


def apply_veto(
    core: pd.DataFrame,
    candidates: pd.DataFrame,
    exp009_output,
    exp009_ranked: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    confidence_threshold: float,
    pair_corr_cap: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    core = core.copy()
    core["stock_id"] = core["stock_id"].astype(str).str.zfill(6)
    core["weight"] = 1.0 / len(core)
    scored = add_risk_columns(candidates, exp009_ranked)
    core_risk = scored[scored["stock_id"].isin(core["stock_id"])].copy()
    if core_risk.empty:
        core_risk = core.assign(exp010_risk_score=0.0)

    tail_cols = [c for c in ["tail_risk_flag", "very_tail_flag", "high_vol_flag", "very_high_vol_flag"] if c in core_risk.columns]
    tailrisk_flag = bool(core_risk[tail_cols].fillna(False).astype(bool).any(axis=None)) if tail_cols else bool(core_risk["exp010_risk_score"].max() >= 0.70)
    corr_cluster_flag, max_core_corr = core_corr_flag(core, returns, pair_corr_cap)
    unknown_industry_flag = exp009_output.risk_report.get("unknown_industry_policy_effective") == "stock_as_industry"
    high_risk = bool((tailrisk_flag or corr_cluster_flag) and exp009_output.confidence >= confidence_threshold)

    details: dict[str, object] = {
        "regime": exp009_output.state.value,
        "confidence": float(exp009_output.confidence),
        "tailrisk_flag": tailrisk_flag,
        "corr_cluster_flag": corr_cluster_flag,
        "unknown_industry_flag": unknown_industry_flag,
        "max_core_corr": max_core_corr,
        "high_risk": high_risk,
        "veto_triggered": False,
        "replaced_out": None,
        "replaced_in": None,
        "reason": "no_high_confidence_risk_flag",
    }
    if not high_risk:
        return core, details

    current = set(core["stock_id"])
    pool = scored[~scored["stock_id"].isin(current)].copy()
    if pool.empty:
        details["reason"] = "no_replacement_pool"
        return core, details

    worst = core_risk.sort_values("exp010_risk_score", ascending=False).iloc[0]
    pool = pool.sort_values(["exp010_risk_score", "score"], ascending=[True, False])
    replacement = pool.iloc[0]
    if float(replacement["exp010_risk_score"]) >= float(worst["exp010_risk_score"]):
        details["reason"] = "no_lower_risk_replacement"
        return core, details

    out = core.copy()
    idx = out.index[out["stock_id"] == str(worst["stock_id"]).zfill(6)][0]
    out.loc[idx, "stock_id"] = str(replacement["stock_id"]).zfill(6)
    out["weight"] = 1.0 / len(out)
    details.update(
        {
            "veto_triggered": True,
            "replaced_out": str(worst["stock_id"]).zfill(6),
            "replaced_in": str(replacement["stock_id"]).zfill(6),
            "reason": "high_confidence_exp009_risk_veto_replaced_one",
            "replaced_out_risk": float(worst["exp010_risk_score"]),
            "replaced_in_risk": float(replacement["exp010_risk_score"]),
        }
    )
    return out, details


def main() -> None:
    args = parse_args()
    exp008_dir = Path(args.exp008_dir)
    summary = pd.read_csv(exp008_dir / "window_summary.csv")
    raw = win_eval.read_raw_market(args.train, args.test)
    panel = normalize_panel(raw)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    prev_result: pd.DataFrame | None = None
    for _, summary_row in summary.tail(10).iterrows():
        asof = pd.Timestamp(summary_row["anchor_date"])
        window_dir = out_dir / asof.strftime("%Y%m%d")
        window_dir.mkdir(parents=True, exist_ok=True)
        core = load_core_result(exp008_dir, asof, summary_row)
        candidates = load_candidate_frame(exp008_dir, asof)
        exp009_output, exp009_ranked, returns = exp009_diagnostics(panel, asof)
        result, veto = apply_veto(
            core,
            candidates,
            exp009_output,
            exp009_ranked,
            returns,
            confidence_threshold=args.confidence_threshold,
            pair_corr_cap=SelectorConfig().pair_corr_cap,
        )
        result_path = window_dir / "result.csv"
        result.to_csv(result_path, index=False)
        window_dates = win_eval.next_window_dates(raw, asof, 5)
        score_self, selected_rets = win_eval.score_prediction(result, raw, window_dates)
        turnover, overlap = win_eval.turnover_overlap(result, prev_result)
        prev_result = result
        selected = result["stock_id"].tolist()
        stock_rets = [float(selected_rets.get(stock_id, 0.0)) for stock_id in selected]
        debug = {
            "asof": str(asof.date()),
            "core": "exp008",
            "overlay": "exp009_veto_only",
            "regime": veto["regime"],
            "confidence": veto["confidence"],
            "tailrisk_flag": veto["tailrisk_flag"],
            "corr_cluster_flag": veto["corr_cluster_flag"],
            "unknown_industry_flag": veto["unknown_industry_flag"],
            "high_risk": veto["high_risk"],
            "veto": veto,
            "core_top5": core["stock_id"].tolist(),
            "final_top5": selected,
            "score_self": score_self,
        }
        debug_path = window_dir / "debug_report.json"
        debug_path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(
            {
                "asof": str(asof.date()),
                "score_window": f"{window_dates[0].date()}~{window_dates[-1].date()}",
                "regime": veto["regime"],
                "confidence": veto["confidence"],
                "tailrisk_flag": veto["tailrisk_flag"],
                "corr_cluster_flag": veto["corr_cluster_flag"],
                "unknown_industry_flag": veto["unknown_industry_flag"],
                "high_risk": veto["high_risk"],
                "veto_triggered": veto["veto_triggered"],
                "replaced_out": veto["replaced_out"],
                "replaced_in": veto["replaced_in"],
                "selected_top5": ",".join(selected),
                "weights": ",".join(f"{w:.6f}" for w in result["weight"]),
                "selected_rets": ",".join(f"{r:.2%}" for r in stock_rets),
                "score_self": float(score_self),
                "bad_count": int(sum(r <= win_eval.BAD_RET for r in stock_rets)),
                "very_bad_count": int(sum(r <= win_eval.VERY_BAD_RET for r in stock_rets)),
                "turnover_vs_prev": turnover,
                "overlap_vs_prev": overlap,
                "debug_report": str(debug_path),
            }
        )
        print(f"{asof.date()} score={score_self:.6f} veto={veto['veto_triggered']} picks={','.join(selected)}")

    eval_df = pd.DataFrame(rows)
    eval_path = out_dir / "window_eval.csv"
    eval_df.to_csv(eval_path, index=False)
    exp008 = summary.rename(columns={"anchor_date": "asof", "selected_score": "exp008"})
    comparison = exp008[["asof", "exp008", "regime", "chosen_branch", "selected_picks"]].merge(
        eval_df[["asof", "score_self", "regime", "selected_top5", "veto_triggered"]],
        on="asof",
        how="inner",
        suffixes=("_exp008", "_exp010"),
    )
    comparison = comparison.rename(columns={"score_self": "exp010"})
    comparison["delta"] = comparison["exp010"] - comparison["exp008"]
    comparison_path = out_dir / "exp008_exp010_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    aggregate = {
        "windows": int(len(eval_df)),
        "mean_selected_score": float(eval_df["score_self"].mean()),
        "median_selected_score": float(eval_df["score_self"].median()),
        "q10_selected_score": float(eval_df["score_self"].quantile(0.10)),
        "worst_selected_score": float(eval_df["score_self"].min()),
        "bad_count": int(eval_df["bad_count"].sum()),
        "very_bad_count": int(eval_df["very_bad_count"].sum()),
        "veto_count": int(eval_df["veto_triggered"].sum()),
        "mean_turnover_vs_prev": float(eval_df["turnover_vs_prev"].mean()),
        "mean_overlap_vs_prev": float(eval_df["overlap_vs_prev"].mean()),
    }
    aggregate_path = out_dir / "window_eval_aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote: {eval_path}")
    print(f"comparison: {comparison_path}")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
