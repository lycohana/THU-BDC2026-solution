from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "code" / "src"))

from branch_router import LEGAL_BRANCHES, compute_branch_state_features, route_branch_v2b_overlay  # noqa: E402
from branch_router_analysis import (  # noqa: E402
    _branch_def_map,
    _build_prefilter_branch_frame,
    _inject_final_top5,
    _metric,
    _realized_for_ids,
    _split_picks,
    run_analysis,
)
from batch_window_analysis import add_branch_diagnostic_features, load_raw, normalize_stock_id, realized_returns_for_anchor  # noqa: E402
from config import config as PROJECT_CONFIG  # noqa: E402


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_grid(value: str, cast=float) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def build_sweep_grid(
    trend_dispersion_values: list[float],
    trend_rank_values: list[int],
    ai_consensus_values: list[float],
) -> list[dict]:
    return [
        {
            "trend_dispersion_max": float(trend_dispersion),
            "trend_candidate_rank_cap": int(trend_rank),
            "theme_ai_consensus_max": float(ai_consensus),
            "crash_minrisk_enabled": False,
            "trend_max_swaps": 1,
            "theme_ai_max_swaps": 1,
            "max_total_swaps": 2,
            "default_strong_keep_guard": True,
            "trend_risk_increase_guard": True,
        }
        for trend_dispersion, trend_rank, ai_consensus in product(trend_dispersion_values, trend_rank_values, ai_consensus_values)
    ]


def _cfg_key(cfg: dict) -> tuple[float, int, float]:
    return (
        float(cfg["trend_dispersion_max"]),
        int(cfg["trend_candidate_rank_cap"]),
        float(cfg["theme_ai_consensus_max"]),
    )


def _write_sweep_decision_summary(sweep_df: pd.DataFrame, out_dir: Path) -> None:
    if sweep_df.empty:
        return
    current = {
        "trend_dispersion_max": 0.14,
        "trend_candidate_rank_cap": 6,
        "theme_ai_consensus_max": 0.70,
    }
    df = sweep_df.copy()
    df["rank_by_mean"] = pd.to_numeric(df["mean"], errors="coerce").rank(method="min", ascending=False).astype(int)
    df["is_current_config"] = (
        pd.to_numeric(df["trend_dispersion_max"], errors="coerce").sub(current["trend_dispersion_max"]).abs().le(1e-12)
        & (pd.to_numeric(df["trend_candidate_rank_cap"], errors="coerce").astype(int) == current["trend_candidate_rank_cap"])
        & pd.to_numeric(df["theme_ai_consensus_max"], errors="coerce").sub(current["theme_ai_consensus_max"]).abs().le(1e-12)
    )
    df["is_plateau_neighbor_of_current"] = (
        pd.to_numeric(df["trend_dispersion_max"], errors="coerce").isin([0.12, 0.14, 0.16])
        & pd.to_numeric(df["trend_candidate_rank_cap"], errors="coerce").astype(int).isin([5, 6, 7])
        & pd.to_numeric(df["theme_ai_consensus_max"], errors="coerce").isin([0.65, 0.70, 0.75])
    )
    best_mean = float(pd.to_numeric(df["mean"], errors="coerce").max())
    current_mean = pd.to_numeric(df.loc[df["is_current_config"], "mean"], errors="coerce")
    current_mean_value = float(current_mean.iloc[0]) if len(current_mean) else None
    neighbor = df[df["is_plateau_neighbor_of_current"]].copy()
    if current_mean_value is None:
        plateau_note = "current_config_missing_from_grid"
    elif neighbor.empty:
        plateau_note = "no_current_neighbors_in_grid"
    else:
        neighbor_mean = pd.to_numeric(neighbor["mean"], errors="coerce")
        spread = float(neighbor_mean.max() - neighbor_mean.min())
        gap_to_best = float(best_mean - current_mean_value)
        if spread <= 0.0015 and gap_to_best <= 0.0015:
            plateau_note = "current_config_inside_stable_plateau"
        elif gap_to_best <= 0.0015:
            plateau_note = "current_config_near_best_but_neighbor_spread_visible"
        else:
            plateau_note = "current_config_lags_best_review_before_freeze"
    df["decision_note"] = ""
    df.loc[df["is_current_config"], "decision_note"] = plateau_note
    df.loc[df["is_plateau_neighbor_of_current"] & ~df["is_current_config"], "decision_note"] = "current_plateau_neighbor"
    columns = [
        "rank_by_mean",
        "trend_dispersion_max",
        "trend_candidate_rank_cap",
        "theme_ai_consensus_max",
        "mean",
        "q10",
        "worst",
        "very_bad",
        "avg_swaps",
        "mean_delta_vs_default",
        "q10_delta_vs_default",
        "worst_delta_vs_default",
        "positive_delta_count",
        "negative_delta_count",
        "zero_delta_count",
        "accepted_swaps",
        "blocked_swaps",
        "guard_accept_rate",
        "is_current_config",
        "is_plateau_neighbor_of_current",
        "decision_note",
    ]
    df = df.sort_values(["rank_by_mean", "q10", "negative_delta_count"], ascending=[True, False, True])
    df[columns].to_csv(out_dir / "sweep_decision_summary.csv", index=False)


def _analysis_args(source_run: str, run_name: str, last_n: int | None, v2b_config: dict, out_dir: str) -> SimpleNamespace:
    return SimpleNamespace(
        source_run=source_run,
        out_dir=out_dir,
        run_name=run_name,
        raw="data/train_hs300_20260424.csv",
        last_n=last_n,
        label_horizon=5,
        router_config_json="",
        v2b_config_json=json.dumps(v2b_config, sort_keys=True),
    )


def run_sweep(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    out_dir = ROOT / args.out_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    trend_rank_grid = getattr(args, "candidate_rank_cap_grid", None) or args.trend_rank_grid
    grid = build_sweep_grid(
        _parse_grid(args.trend_dispersion_grid, float),
        _parse_grid(trend_rank_grid, int),
        _parse_grid(args.ai_consensus_grid, float),
    )
    if args.max_configs is not None:
        grid = grid[: max(0, int(args.max_configs))]
    _log(
        "[sweep] start "
        f"run_name={args.run_name} source_run={args.source_run} last_n={args.last_n} "
        f"configs={len(grid)} dry_run={bool(args.dry_run)}"
    )
    rows = []
    if args.dry_run:
        rows = grid
    else:
        source_run = ROOT / args.source_run
        _log(f"[sweep] loading raw data and window summary from {source_run}")
        raw = load_raw(ROOT / "data/train_hs300_20260424.csv")
        window_summary = pd.read_csv(source_run / "window_summary.csv")
        window_summary["anchor_date"] = pd.to_datetime(window_summary["anchor_date"]).dt.strftime("%Y-%m-%d")
        if args.last_n:
            keep = window_summary["anchor_date"].tail(args.last_n).tolist()
            window_summary = window_summary[window_summary["anchor_date"].isin(keep)].copy()
        total_windows = int(len(window_summary))
        default_scores = pd.to_numeric(window_summary["selected_score"], errors="coerce").fillna(0.0)
        default_metric = _metric(default_scores)
        _log(
            "[sweep] preparing cached per-window router inputs "
            f"windows={total_windows} first={window_summary['anchor_date'].iloc[0] if total_windows else 'NA'} "
            f"last={window_summary['anchor_date'].iloc[-1] if total_windows else 'NA'}"
        )

        branch_defs = _branch_def_map()
        states = []
        sorted_windows = list(window_summary.sort_values("anchor_date").iterrows())
        for win_idx, (_, win) in enumerate(sorted_windows, start=1):
            anchor = pd.Timestamp(win["anchor_date"])
            score_path = source_run / anchor.strftime("%Y%m%d") / "predict_score_df.csv"
            if not score_path.exists():
                raise FileNotFoundError(f"missing score df for {anchor:%Y-%m-%d}: {score_path}")
            realized, _ = realized_returns_for_anchor(raw, anchor, label_horizon=5)
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
            grr_risk_off = float(pd.to_numeric(work.get("grr_risk_off_score", pd.Series(0.0, index=work.index)), errors="coerce").fillna(0.0).max())
            market_breadth_5d = float(pd.to_numeric(work.get("ret5", pd.Series(0.0, index=work.index)), errors="coerce").fillna(0.0).gt(0.0).mean())
            market_ret5 = float(pd.to_numeric(work.get("ret5", pd.Series(0.0, index=work.index)), errors="coerce").fillna(0.0).median())
            market_state = compute_branch_state_features(
                work,
                legal_outputs,
                {
                    "risk_off_score": grr_risk_off,
                    "crash_mode": bool(grr_risk_off >= 0.70 or (market_breadth_5d <= 0.15 and market_ret5 < -0.03)),
                    "recent_branch_oof_strength": {branch: 0.0 for branch in LEGAL_BRANCHES},
                },
            )
            states.append(
                {
                    "window": win["anchor_date"],
                    "work": work,
                    "legal_outputs": legal_outputs,
                    "market_state": market_state,
                    "default_score": float(win["selected_score"]),
                }
            )
            if win_idx == 1 or win_idx == total_windows or win_idx % max(1, int(args.progress_every)) == 0:
                _log(f"[sweep] prepared window {win_idx}/{total_windows}: {win['anchor_date']}")

        checkpoint = out_dir / "threshold_sweep.partial.csv"
        total_configs = len(grid)
        completed: dict[tuple[float, int, float], dict] = {}
        if checkpoint.exists() and not args.no_resume:
            partial = pd.read_csv(checkpoint)
            for _, row in partial.iterrows():
                cfg_key = (
                    float(row["trend_dispersion_max"]),
                    int(row["trend_candidate_rank_cap"]),
                    float(row["theme_ai_consensus_max"]),
                )
                completed[cfg_key] = row.to_dict()
            rows.extend(completed.values())
            _log(f"[sweep] resume checkpoint={checkpoint} completed_configs={len(completed)}")
        for cfg_idx, cfg in enumerate(grid, start=1):
            key = _cfg_key(cfg)
            if key in completed:
                _log(f"[sweep] skip completed config {cfg_idx}/{total_configs}: {key}")
                continue
            cfg_started = time.monotonic()
            _log(
                "[sweep] config "
                f"{cfg_idx}/{total_configs}: trend_dispersion_max={cfg['trend_dispersion_max']} "
                f"trend_candidate_rank_cap={cfg['trend_candidate_rank_cap']} "
                f"theme_ai_consensus_max={cfg['theme_ai_consensus_max']}"
            )
            scores = []
            bad_counts = []
            very_bad_counts = []
            deltas = []
            swap_counts = []
            accepted = 0
            blocked = 0
            candidates_generated = 0
            route_cfg = dict(PROJECT_CONFIG.get("branch_router_v2b", {}))
            route_cfg["enabled"] = True
            route_cfg.update(cfg)
            for state in states:
                decision = route_branch_v2b_overlay(state["legal_outputs"], state["market_state"], route_cfg)
                overlay = decision.debug_info.get("overlay_decision", {}) if isinstance(decision.debug_info, dict) else {}
                final_top5 = overlay.get("final_top5") or _split_picks("")
                realized = _realized_for_ids(state["work"], [str(stock_id) for stock_id in final_top5])
                score = float(realized["score"])
                scores.append(score)
                bad_counts.append(int(realized["bad_count"]))
                very_bad_counts.append(int(realized["very_bad_count"]))
                deltas.append(score - state["default_score"])
                swap_counts.append(int(overlay.get("swap_count", 0)))
                accepted += len(overlay.get("accepted_swap_records", []) or [])
                blocked += len(overlay.get("blocked_candidate_records", []) or [])
                for item in overlay.get("guard_summary", []) or []:
                    candidates_generated += int(item.get("candidates_generated", 0))
            metrics = _metric(pd.Series(scores), pd.Series(bad_counts), pd.Series(very_bad_counts))
            delta = pd.Series(deltas, dtype=float)
            rows.append(
                {
                    **cfg,
                    "mean": metrics["mean"],
                    "q10": metrics["q10"],
                    "worst": metrics["worst"],
                    "very_bad": metrics["mean_very_bad_count"],
                    "avg_swaps": float(pd.Series(swap_counts, dtype=float).mean()) if swap_counts else 0.0,
                    "mean_delta_vs_default": metrics["mean"] - default_metric["mean"],
                    "q10_delta_vs_default": metrics["q10"] - default_metric["q10"],
                    "worst_delta_vs_default": metrics["worst"] - default_metric["worst"],
                    "positive_delta_count": int((delta > 1e-12).sum()),
                    "negative_delta_count": int((delta < -1e-12).sum()),
                    "zero_delta_count": int(delta.abs().le(1e-12).sum()),
                    "accepted_swaps": int(accepted),
                    "blocked_swaps": int(blocked),
                    "guard_accept_rate": float(accepted / max(candidates_generated, 1)),
                }
            )
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            elapsed = time.monotonic() - cfg_started
            total_elapsed = time.monotonic() - started
            _log(
                "[sweep] finished config "
                f"{cfg_idx}/{total_configs} in {elapsed:.1f}s; "
                f"mean={rows[-1]['mean']:.6f} q10={rows[-1]['q10']:.6f} "
                f"worst={rows[-1]['worst']:.6f} avg_swaps={rows[-1]['avg_swaps']:.3f}; "
                f"checkpoint={checkpoint} total_elapsed={total_elapsed:.1f}s"
            )
    output = out_dir / "threshold_sweep.csv"
    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(output, index=False)
    if not sweep_df.empty and not args.dry_run:
        plateau_rows = []
        metrics = [
            "mean",
            "q10",
            "worst",
            "very_bad",
            "avg_swaps",
            "positive_delta_count",
            "negative_delta_count",
            "zero_delta_count",
            "accepted_swaps",
            "blocked_swaps",
            "guard_accept_rate",
        ]
        for field in ["trend_dispersion_max", "trend_candidate_rank_cap", "theme_ai_consensus_max"]:
            field_rows = []
            for value, group in sweep_df.groupby(field, sort=True):
                row = {"group_field": field, "group_value": value, "config_count": int(len(group))}
                for metric in metrics:
                    row[f"{metric}_mean"] = float(pd.to_numeric(group[metric], errors="coerce").mean())
                    row[f"{metric}_max"] = float(pd.to_numeric(group[metric], errors="coerce").max())
                    row[f"{metric}_min"] = float(pd.to_numeric(group[metric], errors="coerce").min())
                plateau_rows.append(row)
                field_rows.append(row)
            field_name = {
                "trend_dispersion_max": "trend_dispersion",
                "trend_candidate_rank_cap": "candidate_rank_cap",
                "theme_ai_consensus_max": "ai_consensus",
            }[field]
            pd.DataFrame(field_rows).to_csv(out_dir / f"plateau_by_{field_name}.csv", index=False)
        pd.DataFrame(plateau_rows).to_csv(out_dir / "plateau_summary.csv", index=False)
        _write_sweep_decision_summary(sweep_df, out_dir)
    _log(f"[sweep] wrote {output} rows={len(sweep_df)} elapsed={time.monotonic() - started:.1f}s")
    return output


def run_longer(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    out_dir = ROOT / args.out_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cfg = {"crash_minrisk_enabled": False}
    specs = [item.strip() for item in args.window_counts.split(",") if item.strip()]
    _log(f"[longer] start run_name={args.run_name} source_run={args.source_run} specs={specs}")
    for spec_idx, window_spec in enumerate(specs, start=1):
        last_n = None if window_spec == "all" else int(window_spec)
        child_name = f"{args.run_name}_{window_spec}win"
        _log(f"[longer] spec {spec_idx}/{len(specs)} window_spec={window_spec} child_run={child_name}")
        child_dir = run_analysis(_analysis_args(args.source_run, child_name, last_n, cfg, args.out_dir))
        aggregate = json.loads((child_dir / "aggregate.json").read_text(encoding="utf-8"))
        _log(f"[longer] loaded aggregate windows={aggregate['windows']} child_dir={child_dir}")
        paired = {row["variant"]: row for row in aggregate.get("paired_delta_distribution", [])}
        variant_aliases = [
            ("default_grr_tail_guard", "default"),
            ("v2b_trend_plus_ai_overlay", "v2b_guarded_candidate"),
            ("v2b_trend_overlay_only", "trend_overlay_only"),
            ("v2b_ai_overlay_only", "ai_overlay_only"),
            ("v2b_trend_plus_ai_overlay", "trend_plus_ai_overlay"),
            ("v2b_trend_plus_ai_overlay_plus_crash_minrisk_rescue", "crash_minrisk_rescue_offline_diagnostic"),
        ]
        for source_variant, output_variant in variant_aliases:
            metrics = aggregate["ablations"][source_variant]
            delta = paired.get(source_variant, {})
            rows.append(
                {
                    "variant": output_variant,
                    "source_variant": source_variant,
                    "window_count": aggregate["windows"],
                    "window_spec": window_spec,
                    "mean": metrics["mean"],
                    "q10": metrics["q10"],
                    "worst": metrics["worst"],
                    "very_bad": metrics["mean_very_bad_count"],
                    "win_rate": metrics["win_rate"],
                    "avg_swaps": metrics["swap_count_mean"],
                    "mean_delta_vs_default": delta.get("mean_delta", 0.0),
                    "q10_delta_vs_default": delta.get("q10_delta", 0.0),
                    "worst_delta_vs_default": delta.get("worst_delta", 0.0),
                    "positive_delta_count": delta.get("positive_delta_count", 0),
                    "negative_delta_count": delta.get("negative_delta_count", 0),
                    "zero_delta_count": delta.get("zero_delta_count", 0),
                    "offline_diagnostic_only": output_variant.endswith("offline_diagnostic"),
                }
            )
    output = out_dir / "longer_window_validation.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    _log(f"[longer] wrote {output} rows={len(rows)} elapsed={time.monotonic() - started:.1f}s")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sweep", "longer"], required=True)
    parser.add_argument("--source-run", default="temp/batch_window_analysis/grr_tail_guard_20win")
    parser.add_argument("--out-dir", default="temp/branch_router_validation")
    parser.add_argument("--run-name", default="v2b_guarded_validation")
    parser.add_argument("--last-n", type=int, default=20)
    parser.add_argument("--trend-dispersion-grid", default="0.10,0.12,0.14,0.16,0.18,0.20")
    parser.add_argument("--trend-rank-grid", default="4,5,6,7,8")
    parser.add_argument("--candidate-rank-cap-grid", default=None, help="Alias for --trend-rank-grid")
    parser.add_argument("--ai-consensus-grid", default="0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--window-counts", default="20,40,60,100,all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5, help="Print one preparation progress line every N windows during sweep")
    parser.add_argument("--max-configs", type=int, default=None, help="Optional cap for controlled sweep smoke runs; does not change runtime config")
    parser.add_argument("--no-resume", action="store_true", help="Ignore threshold_sweep.partial.csv and start the sweep from scratch")
    args = parser.parse_args()
    output = run_sweep(args) if args.mode == "sweep" else run_longer(args)
    print(output)


if __name__ == "__main__":
    main()
