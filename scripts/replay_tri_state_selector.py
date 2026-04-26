from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))

from tri_state_selector import SelectorConfig, TriStateSelector  # noqa: E402
from tri_state_selector.data import normalize_panel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the tri-state selector on competition train/test CSVs without replacing the legacy runner."
    )
    parser.add_argument("--train", default=str(ROOT / "data" / "train.csv"), help="Path to competition train.csv")
    parser.add_argument("--test", default=str(ROOT / "data" / "test.csv"), help="Path to competition test.csv")
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "tri_state_replay" / "result.csv"),
        help="Competition-format output path. Defaults outside output/result.csv to avoid replacing the old main flow.",
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "outputs" / "tri_state_replay" / "risk_report.json"),
        help="Risk/debug report path.",
    )
    parser.add_argument(
        "--debug-report",
        default=str(ROOT / "outputs" / "tri_state_replay" / "debug_report.json"),
        help="Detailed stage-by-stage replay diagnostic JSON path.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Maximum number of stocks to write.")
    parser.add_argument("--asof", default=None, help="Optional as-of date. Defaults to latest date in test.csv.")
    parser.add_argument("--mode", choices=["research", "competition"], default="competition", help="Selector mode for replay.")
    parser.add_argument(
        "--full-invest-top5",
        action="store_true",
        default=True,
        help="Normalize competition top-k output to full investment. Enabled by default for replay.",
    )
    parser.add_argument(
        "--no-full-invest-top5",
        action="store_false",
        dest="full_invest_top5",
        help="Disable full-invest top-k normalization.",
    )
    return parser.parse_args()


def read_competition_panel(train_path: str | Path, test_path: str | Path) -> tuple[pd.DataFrame, pd.Timestamp]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if test.empty:
        raise ValueError("test.csv is empty; cannot infer replay as-of date")
    raw = pd.concat([train, test], ignore_index=True)
    panel = normalize_panel(raw)
    asof = pd.Timestamp(normalize_panel(test)["date"].max())
    return panel, asof


def to_competition_result(weights: pd.Series, top_k: int) -> pd.DataFrame:
    stock_weights = weights.drop(index="CASH", errors="ignore")
    stock_weights = stock_weights[stock_weights > 1e-12].sort_values(ascending=False).head(top_k)
    return pd.DataFrame(
        {
            "stock_id": stock_weights.index.astype(str).str.zfill(6),
            "weight": stock_weights.round(6).to_numpy(dtype=float),
        }
    )


def series_to_float_dict(value: object, limit: int | None = None) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, pd.DataFrame):
        if "score" in value.columns:
            value = value["score"]
        elif "weight" in value.columns:
            value = value["weight"]
        else:
            return {}
    if isinstance(value, pd.Series):
        s = value.copy()
    elif isinstance(value, dict):
        s = pd.Series(value)
    else:
        return {}
    s = pd.to_numeric(s, errors="coerce").dropna()
    if limit is not None:
        s = s.head(limit)
    return {str(k).zfill(6) if str(k) != "CASH" else "CASH": float(v) for k, v in s.items()}


def json_safe(value: object) -> object:
    if isinstance(value, pd.Series):
        return series_to_float_dict(value)
    if isinstance(value, pd.DataFrame):
        return value.reset_index().to_dict(orient="records")
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def dropped_between(before: dict[str, float], after: dict[str, float]) -> list[str]:
    return sorted([name for name, weight in before.items() if name != "CASH" and weight > 1e-12 and after.get(name, 0.0) <= 1e-12])


def build_debug_report(
    *,
    asof: pd.Timestamp,
    output,
    result: pd.DataFrame,
    output_path: Path,
    top_k: int,
) -> dict[str, object]:
    rr = output.risk_report
    stage_names = [
        "raw_weights",
        "weights_after_single_cap",
        "weights_after_industry_cap",
        "weights_after_corr_cap",
        "weights_after_vol_target",
        "weights_after_es_constraint",
        "weights_after_drawdown_governor",
        "weights_after_turnover_penalty",
    ]
    stages = {name: series_to_float_dict(rr.get(name)) for name in stage_names}
    final_result_weights = {
        str(row.stock_id).zfill(6): float(row.weight)
        for row in result.itertuples(index=False)
    }
    full_target_weights = series_to_float_dict(output.target_weights)

    totals = {name: float(sum(weights.values())) for name, weights in stages.items()}
    totals["full_target_weights"] = float(sum(full_target_weights.values()))
    totals["competition_result_top_k"] = float(sum(final_result_weights.values()))

    dropped_by_stage = {}
    previous_name = stage_names[0]
    for name in stage_names[1:]:
        dropped_by_stage[name] = dropped_between(stages.get(previous_name, {}), stages.get(name, {}))
        previous_name = name
    dropped_by_stage["competition_top_k_truncation"] = dropped_between(full_target_weights, final_result_weights)

    reason_codes = list(rr.get("reason_codes", []))
    if output.state.value != "defensive" and rr.get("cash_weight", 0.0) > 0:
        reason_codes.append("cash_is_residual_from_risk_constraints_not_defensive_cash_floor")
    if totals["competition_result_top_k"] + 1e-12 < totals["full_target_weights"]:
        reason_codes.append(f"result_csv_contains_top_{top_k}_only_full_target_has_more_names")
    if rr.get("drawdown_multiplier", 1.0) == 1.0:
        reason_codes.append("drawdown_governor_noop")
    if rr.get("es_scale", 1.0) == 1.0:
        reason_codes.append("portfolio_es_constraint_noop")
    if rr.get("gross_exposure", 0.0) < 0.99 and output.state.value == "neutral":
        reason_codes.append("neutral_gross_exposure_below_full_investment_after_caps_or_cluster_constraints")

    return {
        "asof": str(asof.date()),
        "mode": rr.get("mode", "research"),
        "regime_state": output.state.value,
        "confidence": float(output.confidence),
        "market_features": json_safe(rr.get("market_features", {})),
        "unknown_industry_policy_effective": rr.get("unknown_industry_policy_effective"),
        "output_top_k": int(rr.get("output_top_k", top_k)),
        "force_top_k_full_invest": bool(rr.get("force_top_k_full_invest", False)),
        "top_raw_scores": series_to_float_dict(rr.get("top_raw_scores"), limit=30),
        "top_shaped_scores": series_to_float_dict(rr.get("top_shaped_scores"), limit=30),
        "selected_before_risk": series_to_float_dict(rr.get("selected_before_risk")),
        "selected_after_risk": series_to_float_dict(rr.get("selected_after_risk")),
        "selected_for_competition_output": series_to_float_dict(rr.get("selected_for_competition_output")),
        "weight_sum_before_topk": float(rr.get("weight_sum_before_topk", 0.0)),
        "weight_sum_after_topk_normalize": float(rr.get("weight_sum_after_topk_normalize", 0.0)),
        "raw_weights": stages["raw_weights"],
        "weights_after_single_cap": stages["weights_after_single_cap"],
        "weights_after_industry_cap": stages["weights_after_industry_cap"],
        "weights_after_corr_cap": stages["weights_after_corr_cap"],
        "weights_after_vol_target": stages["weights_after_vol_target"],
        "weights_after_es_constraint": stages["weights_after_es_constraint"],
        "weights_after_drawdown_governor": stages["weights_after_drawdown_governor"],
        "final_weights": final_result_weights,
        "full_target_weights": full_target_weights,
        "total_weight_before_each_stage": totals,
        "dropped_by_stage": dropped_by_stage,
        "reason_codes": sorted(set(reason_codes)),
        "result_path": str(output_path),
        "risk_report": json_safe(rr),
    }


def main() -> None:
    args = parse_args()
    panel, default_asof = read_competition_panel(args.train, args.test)
    asof = pd.Timestamp(args.asof) if args.asof else default_asof

    selector = TriStateSelector(
        SelectorConfig(
            # The replay keeps the selector conservative but allows enough gross
            # exposure for a five-name competition file.
            mode=args.mode,
            output_top_k=args.top_k,
            force_top_k_full_invest=args.full_invest_top5,
            regime_hysteresis_days=1,
            top_quantile=0.40,
        )
    )
    output = selector.rebalance(prices=panel, asof=asof, prev_weights=pd.Series(dtype=float))
    result = to_competition_result(output.target_weights, args.top_k)

    output_path = Path(args.output)
    report_path = Path(args.report)
    debug_path = Path(args.debug_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    report = {
        "asof": str(asof.date()),
        "state": output.state.value,
        "confidence": output.confidence,
        "result_path": str(output_path),
        "target_weights": {str(k): float(v) for k, v in output.target_weights.items()},
        "risk_report": json_safe(output.risk_report),
        "top_candidates": output.ranked_candidates.head(20)["score"].to_dict(),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    debug = build_debug_report(asof=asof, output=output, result=result, output_path=output_path, top_k=args.top_k)
    debug_path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"asof: {asof.date()}")
    print(f"state: {output.state.value}")
    print(f"confidence: {output.confidence:.4f}")
    print(f"wrote: {output_path}")
    print(f"debug: {debug_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
