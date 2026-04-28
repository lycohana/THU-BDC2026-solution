from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "temp" / "branch_router_validation"
NEW_V2B_DIR = VALIDATION_DIR / "v2b_deeper_veto_disp013_60win"
OLD_V2B_DIR = VALIDATION_DIR / "v2b_guarded_longer_60win_60win"
RISKOFF_DIR = VALIDATION_DIR / "riskoff_capture_shadow_dynamic_target_after_deeper_disp013"
PULLBACK_DIR = VALIDATION_DIR / "alpha_hunt_shadow_after_deeper_disp013"
TENWIN_RUN_DIR = ROOT / "temp" / "batch_window_analysis" / "current_10win_stress_veto"
TENWIN_COMPARE_CSV = ROOT / "temp" / "10win_stress_veto_vs_baseline.csv"
TENWIN_COMPARE_JSON = ROOT / "temp" / "10win_stress_veto_vs_baseline_summary.json"
FREEZE_DIR = ROOT / "temp" / "submission_freeze" / "post_guard_overlay_queue_freeze"
EVIDENCE_DIR = FREEZE_DIR / "evidence"

QUEUE = {
    "name": "riskoff_rank4_dynamic_pullback_stress_veto_v2",
    "base": "v2b_guarded_candidate_with_deeper_veto_disp013",
    "runtime_enabled": True,
    "shadow_only": False,
    "max_alpha_swaps": 1,
    "final_risk_veto_max_swaps": 1,
    "priority": [
        "riskoff_fill_rank4_dynamic_defensive_target_no_v2b_swap",
        "pullback_rebound_highest_risk",
        "stress_chaser_veto_final",
    ],
    "stress_chaser_veto": {
        "role": "final runtime risk repair, not a new alpha sleeve",
        "market_gate": "median_ret20 < 0, breadth20 <= 0.50, median_sigma20 > 0.018, dispersion20 > 0.10",
        "target_gate": "panic single-name beta shock OR hot ret5/amp/downside_beta chaser",
        "replacement_pool": "runtime minrisk candidate outside current Top5",
    },
}


def _read_v2b_returns(detail_dir: Path, col: str) -> pd.DataFrame:
    path = detail_dir / "ablation_decisions.csv"
    frame = pd.read_csv(path)
    out = frame[frame["variant"] == "v2b_trend_plus_ai_overlay"][["window_date", "score"]].copy()
    out["window"] = pd.to_datetime(out["window_date"]).dt.strftime("%Y-%m-%d")
    out[col] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0)
    return out[["window", col]]


def _read_riskoff() -> pd.DataFrame:
    frame = pd.read_csv(RISKOFF_DIR / "riskoff_capture_windows.csv")
    out = frame[frame["variant"] == "rank4_relaxed_dynamic_defensive_target_no_v2b_swap"].copy()
    out["window"] = pd.to_datetime(out["window"]).dt.strftime("%Y-%m-%d")
    out["riskoff_accepted"] = pd.to_numeric(out["accepted_swap_count"], errors="coerce").fillna(0).astype(int)
    out["riskoff_return"] = pd.to_numeric(out["shadow_return"], errors="coerce").fillna(0.0)
    out["riskoff_candidate"] = out["candidate_stock"].fillna("").astype(str)
    out["riskoff_replaced"] = out["replaced_stock"].fillna("").astype(str)
    return out[["window", "riskoff_accepted", "riskoff_return", "riskoff_candidate", "riskoff_replaced"]]


def _read_pullback() -> pd.DataFrame:
    frame = pd.read_csv(PULLBACK_DIR / "alpha_hunt_windows.csv")
    out = frame[frame["variant"] == "v2b_pullback_rebound_highest_risk_top1"].copy()
    out["window"] = pd.to_datetime(out["window"]).dt.strftime("%Y-%m-%d")
    out["pullback_accepted"] = pd.to_numeric(out["accepted_swap_count"], errors="coerce").fillna(0).astype(int)
    out["pullback_return"] = pd.to_numeric(out["shadow_return"], errors="coerce").fillna(0.0)
    out["pullback_candidate"] = out["candidate_stock"].fillna("").astype(str)
    out["pullback_replaced"] = out["replaced_stock"].fillna("").astype(str)
    return out[["window", "pullback_accepted", "pullback_return", "pullback_candidate", "pullback_replaced"]]


def _build_windows() -> pd.DataFrame:
    new_v2b = _read_v2b_returns(NEW_V2B_DIR, "new_v2b_return")
    old_v2b = _read_v2b_returns(OLD_V2B_DIR, "old_v2b_return")
    out = new_v2b.merge(old_v2b, on="window", how="left")
    out = out.merge(_read_riskoff(), on="window", how="left")
    out = out.merge(_read_pullback(), on="window", how="left")
    for col in ["riskoff_accepted", "pullback_accepted"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    out["queue_source"] = "v2b_guarded"
    out["queue_return"] = out["new_v2b_return"]
    riskoff_mask = out["riskoff_accepted"] > 0
    pullback_mask = (~riskoff_mask) & (out["pullback_accepted"] > 0)
    out.loc[riskoff_mask, "queue_source"] = "riskoff_rank4"
    out.loc[riskoff_mask, "queue_return"] = out.loc[riskoff_mask, "riskoff_return"]
    out.loc[pullback_mask, "queue_source"] = "pullback_rebound"
    out.loc[pullback_mask, "queue_return"] = out.loc[pullback_mask, "pullback_return"]
    out["delta_vs_new_v2b"] = out["queue_return"] - out["new_v2b_return"]
    out["delta_vs_old_v2b"] = out["queue_return"] - out["old_v2b_return"]
    return out.sort_values("window").reset_index(drop=True)


def _summary(windows: pd.DataFrame) -> pd.DataFrame:
    all_windows = sorted(windows["window"].astype(str).unique())
    buckets = {"20win": all_windows[-20:], "40win": all_windows[-40:], "60win": all_windows}
    rows: list[dict[str, Any]] = []
    for bucket, keep in buckets.items():
        sub = windows[windows["window"].isin(set(keep))].copy()
        delta_new = pd.to_numeric(sub["delta_vs_new_v2b"], errors="coerce").fillna(0.0)
        delta_old = pd.to_numeric(sub["delta_vs_old_v2b"], errors="coerce").fillna(0.0)
        returns = pd.to_numeric(sub["queue_return"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "bucket": bucket,
                "queue_name": QUEUE["name"],
                "window_count": int(len(sub)),
                "queue_return_mean": float(returns.mean()),
                "queue_return_q10": float(returns.quantile(0.10)),
                "queue_return_worst": float(returns.min()),
                "delta_new_v2b_mean": float(delta_new.mean()),
                "delta_new_v2b_q10": float(delta_new.quantile(0.10)),
                "delta_new_v2b_worst": float(delta_new.min()),
                "delta_old_v2b_mean": float(delta_old.mean()),
                "delta_old_v2b_q10": float(delta_old.quantile(0.10)),
                "delta_old_v2b_worst": float(delta_old.min()),
                "negative_delta_new_v2b_count": int((delta_new < -1e-12).sum()),
                "riskoff_swaps": int((sub["queue_source"] == "riskoff_rank4").sum()),
                "pullback_swaps": int((sub["queue_source"] == "pullback_rebound").sum()),
            }
        )
    return pd.DataFrame(rows)


def _copy_evidence() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    sources = [
        VALIDATION_DIR / "v2b_deeper_veto_disp013" / "longer_window_validation.csv",
        NEW_V2B_DIR / "accepted_swaps.csv",
        NEW_V2B_DIR / "guard_summary_by_window.csv",
        RISKOFF_DIR / "riskoff_capture_summary.csv",
        RISKOFF_DIR / "riskoff_capture_windows.csv",
        PULLBACK_DIR / "alpha_hunt_summary.csv",
        PULLBACK_DIR / "alpha_hunt_windows.csv",
        TENWIN_RUN_DIR / "window_summary.csv",
        TENWIN_RUN_DIR / "portfolio_details.csv",
        TENWIN_COMPARE_CSV,
        TENWIN_COMPARE_JSON,
    ]
    for path in sources:
        if path.exists():
            shutil.copy2(path, EVIDENCE_DIR / path.name)


def _read_tenwin_smoke() -> dict[str, Any]:
    if TENWIN_COMPARE_JSON.exists():
        return json.loads(TENWIN_COMPARE_JSON.read_text(encoding="utf-8"))
    if not TENWIN_COMPARE_CSV.exists():
        return {}
    frame = pd.read_csv(TENWIN_COMPARE_CSV)
    current = pd.to_numeric(frame["selected_score"], errors="coerce").fillna(0.0)
    baseline = pd.to_numeric(frame["baseline_score"], errors="coerce").fillna(0.0)
    return {
        "windows": int(len(frame)),
        "current": {
            "mean": float(current.mean()),
            "q10": float(current.quantile(0.10)),
            "worst": float(current.min()),
            "win_rate": float((current > 0).mean()),
        },
        "baseline": {
            "mean": float(baseline.mean()),
            "q10": float(baseline.quantile(0.10)),
            "worst": float(baseline.min()),
            "win_rate": float((baseline > 0).mean()),
        },
    }


def _write_report(summary: pd.DataFrame, checks: list[str]) -> None:
    lines = [
        "# Post-Guard Supplemental Overlay Freeze",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Decision",
        "",
        "Freeze the post-guard supplemental queue as live runtime material.",
        "The live prediction path applies one alpha overlay at most, then may apply one final stress-chaser risk veto.",
        "",
        "## Queue",
        "",
        f"Name: `{QUEUE['name']}`",
        f"Priority: `{' -> '.join(QUEUE['priority'])}`",
        "",
        summary.to_markdown(index=False),
        "",
        "## Stress Veto 10-Window Smoke",
        "",
        "The stress veto is frozen as a tail-repair guard. It must stay runtime-safe: no realized returns, no baseline branch, no score_self feedback.",
        "",
        json.dumps(_read_tenwin_smoke(), ensure_ascii=False, indent=2),
        "",
        "## Safety Checks",
        "",
        "\n".join(f"- {check}" for check in checks),
        "",
    ]
    (FREEZE_DIR / "decision_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    windows = _build_windows()
    summary = _summary(windows)
    tenwin = _read_tenwin_smoke()
    tenwin_current = tenwin.get("current", {}) if tenwin else {}
    tenwin_baseline = tenwin.get("baseline", {}) if tenwin else {}
    checks = [
        f"PASS delta_new_v2b_mean positive in all buckets: {bool((summary['delta_new_v2b_mean'] > 0).all())}",
        f"PASS delta_new_v2b_q10 non-negative in all buckets: {bool((summary['delta_new_v2b_q10'] >= -1e-12).all())}",
        f"PASS stress veto 10win mean beats baseline: {bool(tenwin_current.get('mean', -999) > tenwin_baseline.get('mean', 999))}",
        f"PASS stress veto 10win q10 beats baseline: {bool(tenwin_current.get('q10', -999) > tenwin_baseline.get('q10', 999))}",
        f"PASS stress veto 10win worst beats baseline: {bool(tenwin_current.get('worst', -999) > tenwin_baseline.get('worst', 999))}",
        f"PASS runtime enablement: {bool(QUEUE['runtime_enabled']) and not bool(QUEUE['shadow_only'])}",
    ]
    if not (
        (summary["delta_new_v2b_mean"] > 0).all()
        and (summary["delta_new_v2b_q10"] >= -1e-12).all()
        and tenwin_current.get("mean", -999) > tenwin_baseline.get("mean", 999)
        and tenwin_current.get("q10", -999) > tenwin_baseline.get("q10", 999)
        and tenwin_current.get("worst", -999) > tenwin_baseline.get("worst", 999)
    ):
        raise AssertionError("\n".join(checks))
    windows.to_csv(FREEZE_DIR / "post_guard_overlay_windows.csv", index=False)
    summary.to_csv(FREEZE_DIR / "post_guard_overlay_summary.csv", index=False)
    config = {
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "queue": QUEUE,
        "summary": summary.to_dict(orient="records"),
        "stress_veto_10win_smoke": tenwin,
        "checks": checks,
    }
    (FREEZE_DIR / "frozen_post_guard_overlay_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (FREEZE_DIR / "runtime_safety_check.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
    _copy_evidence()
    _write_report(summary, checks)
    print(f"Wrote post-guard freeze artifacts to {FREEZE_DIR}")


if __name__ == "__main__":
    main()
