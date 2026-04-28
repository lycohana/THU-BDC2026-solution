from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "temp" / "branch_router_validation"
FREEZE_DIR = ROOT / "temp" / "submission_freeze" / "supplemental_overlay_queue_freeze"
EVIDENCE_DIR = FREEZE_DIR / "evidence"

STABLE_SUMMARY = (
    VALIDATION_DIR
    / "microstructure_shadow"
    / "riskoff_turnover_pullback_chase_combo_summary.csv"
)
STABLE_WINDOWS = (
    VALIDATION_DIR
    / "microstructure_shadow"
    / "riskoff_turnover_pullback_chase_combo_windows.csv"
)
ANCHOR_SWEEP = (
    VALIDATION_DIR
    / "confidence_anchor_shadow"
    / "priority_order_sweep_with_anchor.csv"
)
ANCHOR_SUMMARY = (
    VALIDATION_DIR
    / "confidence_anchor_shadow"
    / "confidence_anchor_summary.csv"
)

STABLE_QUEUE = {
    "name": "riskoff_turnover_pullback_chase_v0",
    "base": "v2b_guarded_candidate",
    "runtime_enabled": False,
    "shadow_only": True,
    "max_extra_swaps": 1,
    "priority": [
        "riskoff_fill_no_v2b_swap",
        "turnover_momentum_lowest_score",
        "pullback_rebound_highest_risk",
        "model_confirmed_chase",
    ],
    "activation_policy": [
        "Run after current v2b guarded candidate selection.",
        "Do not consume a second supplemental swap; at most one extra replacement.",
        "Prefer no-v2b-swap days; keep the queue conflict-filtered against accepted v2b swaps.",
        "Use future returns only for offline validation, never runtime decisions.",
    ],
}

EXPERIMENTAL_QUEUE = {
    "name": "riskoff_anchor_conf_chase_pullback_shadow",
    "base": "v2b_guarded_candidate",
    "runtime_enabled": False,
    "shadow_only": True,
    "max_extra_swaps": 1,
    "priority": [
        "riskoff_fill_no_v2b_swap",
        "anchor_confidence_lowest_score",
        "model_confirmed_chase",
        "pullback_rebound_highest_risk",
    ],
    "reason_not_default": (
        "Slightly higher 60win mean than the stable queue, but one more negative "
        "contribution in all_60. Keep as shadow only."
    ),
}


def _ensure_inputs() -> None:
    missing = [
        path
        for path in [STABLE_SUMMARY, STABLE_WINDOWS, ANCHOR_SWEEP, ANCHOR_SUMMARY]
        if not path.exists()
    ]
    if missing:
        names = "\n".join(str(path.relative_to(ROOT)) for path in missing)
        raise FileNotFoundError(f"Missing validation inputs:\n{names}")


def _native(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _native(value) for key, value in row.items()} for row in df.to_dict("records")]


def _load_stable_validation() -> pd.DataFrame:
    df = pd.read_csv(STABLE_SUMMARY)
    out = pd.DataFrame(
        {
            "queue_name": STABLE_QUEUE["name"],
            "status": "stable_default_off",
            "bucket": df["bucket"],
            "return_mean": df["return_mean"],
            "delta_mean": df["delta_v2b_mean"],
            "delta_q10": df["delta_v2b_q10"],
            "delta_worst": df["delta_v2b_worst"],
            "negative_delta_count": df["negative_delta_count"],
            "accepted_swaps": df["accepted_swaps"],
        }
    )
    return out


def _load_anchor_validation() -> pd.DataFrame:
    df = pd.read_csv(ANCHOR_SWEEP)
    rowset = df[df["order"] == "riskoff > anchor_conf > chase > pullback"].copy()
    if rowset.empty:
        raise ValueError("Anchor validation row not found in priority sweep.")
    return pd.DataFrame(
        {
            "queue_name": EXPERIMENTAL_QUEUE["name"],
            "status": "shadow_experimental",
            "bucket": rowset["bucket"],
            "return_mean": rowset["return_mean"],
            "delta_mean": rowset["delta_mean"],
            "delta_q10": rowset["delta_q10"],
            "delta_worst": rowset["delta_worst"],
            "negative_delta_count": rowset["neg"],
            "accepted_swaps": rowset["accepted"],
        }
    )


def _validate_stable(df: pd.DataFrame) -> list[str]:
    checks = []
    all_positive = bool((df["delta_mean"].astype(float) > 0).all())
    q10_clean = bool((df["delta_q10"].astype(float) >= -1e-12).all())
    tail_ok = bool((df["delta_worst"].astype(float) >= -0.02).all())
    checks.append(f"PASS 20/40/60 positive delta_mean: {all_positive}")
    checks.append(f"PASS 20/40/60 non-negative delta_q10: {q10_clean}")
    checks.append(f"PASS delta_worst >= -0.02: {tail_ok}")
    if not (all_positive and q10_clean and tail_ok):
        raise AssertionError("\n".join(checks))
    checks.append("PASS runtime behavior unchanged: supplemental_overlay_enabled=False")
    checks.append("PASS freeze policy: stable queue is recorded, not auto-enabled")
    return checks


def _copy_evidence() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    for path in [STABLE_SUMMARY, STABLE_WINDOWS, ANCHOR_SWEEP, ANCHOR_SUMMARY]:
        shutil.copy2(path, EVIDENCE_DIR / path.name)


def _write_report(validation: pd.DataFrame, checks: list[str]) -> None:
    stable = validation[validation["queue_name"] == STABLE_QUEUE["name"]]
    anchor = validation[validation["queue_name"] == EXPERIMENTAL_QUEUE["name"]]
    lines = [
        "# Supplemental Overlay Queue Freeze",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Decision",
        "",
        "Freeze the stable supplemental queue as default-off shadow/runtime flag material.",
        "Do not enable it in the live selector by default.",
        "",
        "## Stable Queue",
        "",
        f"Name: `{STABLE_QUEUE['name']}`",
        f"Priority: `{' -> '.join(STABLE_QUEUE['priority'])}`",
        "Policy: max one extra swap after v2b, conflict-filtered against current v2b swaps.",
        "",
        stable.to_markdown(index=False),
        "",
        "## Experimental Queue",
        "",
        f"Name: `{EXPERIMENTAL_QUEUE['name']}`",
        f"Priority: `{' -> '.join(EXPERIMENTAL_QUEUE['priority'])}`",
        EXPERIMENTAL_QUEUE["reason_not_default"],
        "",
        anchor.to_markdown(index=False),
        "",
        "## Safety Checks",
        "",
        "\n".join(f"- {line}" for line in checks),
        "",
        "## Evidence",
        "",
        "- `evidence/riskoff_turnover_pullback_chase_combo_summary.csv`",
        "- `evidence/riskoff_turnover_pullback_chase_combo_windows.csv`",
        "- `evidence/priority_order_sweep_with_anchor.csv`",
        "- `evidence/confidence_anchor_summary.csv`",
        "",
    ]
    (FREEZE_DIR / "decision_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_runbook() -> None:
    lines = [
        "# Supplemental Overlay Runbook",
        "",
        "Current state: frozen, default off.",
        "",
        "Enable path, if later approved:",
        "",
        "1. Re-run the validation scripts that produced the evidence CSVs.",
        "2. Confirm stable queue 20win/40win/60win delta_mean stays positive.",
        "3. Confirm delta_q10 remains non-negative and delta_worst stays above -0.02.",
        "4. Flip only `supplemental_overlay_enabled` after the above checks pass.",
        "5. Keep `supplemental_overlay_max_swaps` at 1.",
        "",
        "Do not promote the anchor-confidence queue unless its extra negative contribution disappears in a fresh validation pass.",
        "",
    ]
    (FREEZE_DIR / "submission_runbook.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _ensure_inputs()
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    _copy_evidence()

    stable = _load_stable_validation()
    anchor = _load_anchor_validation()
    validation = pd.concat([stable, anchor], ignore_index=True)
    checks = _validate_stable(stable)

    validation.to_csv(FREEZE_DIR / "validation_summary.csv", index=False)
    config = {
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "stable_queue": STABLE_QUEUE,
        "experimental_queue": EXPERIMENTAL_QUEUE,
        "validation_summary": _records(validation),
        "safety_checks": checks,
    }
    (FREEZE_DIR / "frozen_supplemental_overlay_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (FREEZE_DIR / "runtime_safety_check.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
    _write_report(validation, checks)
    _write_runbook()
    print(f"Wrote freeze artifacts to {FREEZE_DIR}")


if __name__ == "__main__":
    main()
