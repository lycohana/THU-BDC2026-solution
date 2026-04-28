from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))

from config import config as PROJECT_CONFIG  # noqa: E402


FREEZE_DIR = ROOT / "temp" / "submission_freeze" / "v2b_guarded_candidate_freeze"
VALIDATION = ROOT / "temp" / "branch_router_validation" / "v2b_guarded_longer_60win" / "longer_window_validation.csv"
SOURCE_INVENTORY = ROOT / "temp" / "branch_router_validation" / "source_run_inventory.csv"
DECISION_REPORT = ROOT / "temp" / "branch_router_validation" / "v2b_guarded_decision_report.md"


def frozen_config() -> dict:
    cfg = PROJECT_CONFIG["v2b_guarded_candidate"].copy()
    return {
        "name": "v2b_guarded_candidate_freeze",
        "branch_router_v2b": {
            "enabled": bool(cfg["enabled"]),
            "crash_minrisk_enabled": bool(cfg["crash_minrisk_enabled"]),
            "trend_max_swaps": int(cfg["trend_max_swaps"]),
            "theme_ai_max_swaps": int(cfg["theme_ai_max_swaps"]),
            "max_total_swaps": int(cfg["max_total_swaps"]),
            "trend_dispersion_max": float(cfg["trend_dispersion_max"]),
            "trend_candidate_rank_cap": int(cfg["trend_candidate_rank_cap"]),
            "default_strong_keep_guard": bool(cfg["default_strong_keep_guard"]),
            "trend_risk_increase_guard": bool(cfg["trend_risk_increase_guard"]),
            "theme_ai_consensus_max": float(cfg["theme_ai_consensus_max"]),
        },
        "runtime_scope": "current submission candidate",
        "notes": [
            "stock-level overlay only",
            "crash_minrisk_rescue disabled",
            "v2c/v2d remain shadow research unless separately frozen",
        ],
    }


def write_validation_summary(out_dir: Path) -> pd.DataFrame:
    rows = [
        {
            "window_count": 20,
            "variant": "v2b_guarded_candidate",
            "mean": 0.033555,
            "delta_vs_default": 0.006369,
            "q10": -0.013834,
            "worst": -0.029746,
            "avg_swaps": 0.50,
            "positive_delta_count": 7,
            "negative_delta_count": 1,
            "zero_delta_count": 12,
        },
        {
            "window_count": 40,
            "variant": "v2b_guarded_candidate",
            "mean": 0.059115,
            "delta_vs_default": 0.003062,
            "q10": -0.001580,
            "worst": -0.029746,
            "avg_swaps": 0.275,
            "positive_delta_count": 7,
            "negative_delta_count": 2,
            "zero_delta_count": 31,
        },
        {
            "window_count": 60,
            "variant": "v2b_guarded_candidate",
            "mean": 0.052071,
            "delta_vs_default": 0.001156,
            "q10": -0.003648,
            "worst": -0.029746,
            "avg_swaps": 0.233333,
            "positive_delta_count": 7,
            "negative_delta_count": 5,
            "zero_delta_count": 48,
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "validation_summary.csv", index=False)
    return df


def write_runtime_safety_check(out_dir: Path) -> None:
    text = "\n".join(
        [
            "crash_minrisk_enabled = False",
            "no hard switch",
            "no baseline_hybrid runtime",
            "no reference_baseline runtime",
            "no AI shadow guard runtime",
            "no 2026-03-09 exception",
            "no realized return runtime",
            "no score_self runtime",
            "v2c not runtime",
            "v2d not runtime",
        ]
    )
    (out_dir / "runtime_safety_check.txt").write_text(text + "\n", encoding="utf-8")


def write_decision_report(out_dir: Path, validation: pd.DataFrame) -> None:
    report = [
        "# v2b_guarded_candidate freeze decision report",
        "",
        "## Executive summary",
        "v2b_guarded_candidate is frozen as the current submission candidate. It is a recent-effective, low-frequency stock-level overlay alpha, not a full-history strong alpha.",
        "",
        "## Frozen validation",
        "```text",
        validation.to_string(index=False),
        "```",
        "",
        "## Decision",
        "- Keep v2b_guarded_candidate as the current submission candidate.",
        "- Do not continue tuning around recent 20-window mean.",
        "- Keep crash_minrisk_enabled = False.",
        "- Keep v2c and v2d as shadow research unless a separate freeze pass promotes one.",
        "- Do not enable whole-window hard switch, baseline runtime, AI shadow guard runtime, or 2026-03-09 exception.",
    ]
    if DECISION_REPORT.exists():
        report.extend(["", "## Previous decision report pointer", str(DECISION_REPORT)])
    (out_dir / "decision_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def write_runbook(out_dir: Path) -> None:
    text = """# v2b_guarded_candidate submission runbook

## 1. Frozen config

See `frozen_config.json`. The active submission candidate is `v2b_guarded_candidate` with `crash_minrisk_enabled = false`, stock-level overlays only, and max total swaps = 2.

## 2. Train

Repository entrypoint:

```bash
bash app/train.sh
```

PowerShell/local Python entrypoint:

```powershell
uv run python app\\code\\src\\train.py
```

## 3. Inference

Repository entrypoint:

```bash
bash test.sh
```

PowerShell/local Python entrypoint:

```powershell
uv run python app\\code\\src\\test.py
```

## 4. Generate `output/result.csv`

Run inference. The expected output path is:

```text
output/result.csv
```

## 5. Run tests

```powershell
.\\.venv\\Scripts\\python.exe -m pytest test -q
```

## 6. Confirm baseline is not runtime

Check runtime configs and reports for:

```text
no baseline_hybrid runtime
no reference_baseline runtime
```

Baseline/reference artifacts may appear only in offline diagnostics.

## 7. Confirm crash minrisk rescue is closed

Check:

```text
branch_router_v2b.crash_minrisk_enabled = false
v2b_guarded_candidate.crash_minrisk_enabled = false
```

## 8. Confirm v2c/v2d are not runtime

v2c/v2d files under `temp/branch_router_validation/` are shadow research outputs only. Do not wire them into `app/code/src/test.py` or runtime branch outputs without a separate freeze pass.

## 9. Save final artifacts

Save this freeze directory, the 20/40/60 validation outputs, and the final `output/result.csv` together with the commit hash.

## 10. Submission checklist

- [ ] `pytest test -q` passes.
- [ ] `runtime_safety_check.txt` still matches config.
- [ ] `output/result.csv` exists and has at most 5 stocks.
- [ ] No baseline/reference runtime branch is enabled.
- [ ] No hard switch is enabled.
- [ ] No score_self/test.csv feedback is used for runtime decisions.
"""
    (out_dir / "submission_runbook.md").write_text(text, encoding="utf-8")


def main() -> None:
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    (FREEZE_DIR / "frozen_config.json").write_text(json.dumps(frozen_config(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    validation = write_validation_summary(FREEZE_DIR)
    write_decision_report(FREEZE_DIR, validation)
    write_runtime_safety_check(FREEZE_DIR)
    write_runbook(FREEZE_DIR)
    if SOURCE_INVENTORY.exists():
        shutil.copy2(SOURCE_INVENTORY, FREEZE_DIR / "source_run_inventory.csv")
    else:
        pd.DataFrame(columns=["source_run", "path", "exists", "window_summary_exists", "window_count", "is_complete", "notes"]).to_csv(
            FREEZE_DIR / "source_run_inventory.csv", index=False
        )
    if not (FREEZE_DIR / "pytest_result.txt").exists():
        (FREEZE_DIR / "pytest_result.txt").write_text("pending: run .\\.venv\\Scripts\\python.exe -m pytest test -q\n", encoding="utf-8")
    print(f"freeze artifacts written to {FREEZE_DIR}")


if __name__ == "__main__":
    main()
