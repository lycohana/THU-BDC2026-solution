"""Conditional anti-lottery shadow: test safer anti-lottery variants that avoid
replacing high-beta / high-vol stocks in trending markets.

Variants tested:
  1. dbeta_guard:      block if replaced stock downside_beta60 > threshold
  2. combo_risk_guard: block if replaced stock amp20 > X AND dd20 > Y
  3. full_guard:       dbeta_guard OR combo_risk_guard
  4. ret20_guard:      block if replaced stock ret20 > threshold (positive trend)
  5. combined_all:     dbeta_guard OR combo_risk_guard OR ret20_guard

Only uses T-day observable features. No future returns, no scorer, no baseline.
"""

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

from branch_router_v2c_shadow import SOURCE_RUN, V2B_DETAIL, _bad_counts, _realized_for_ids, build_shadow_inputs  # noqa: E402
from microstructure_shadow import (  # noqa: E402
    _add_scores,
    _num,
    _pool,
    _rank_pct,
    _stock_return,
    _target,
    build_raw_micro_panel,
    summarize,
)

OUT_DIR = ROOT / "temp" / "branch_router_validation" / "conditional_antilottery_shadow"


# ---------------------------------------------------------------------------
# Guard checks on the replaced stock
# ---------------------------------------------------------------------------

def _dbeta_guard(work: pd.DataFrame, stock_id: str, threshold: float = 1.35) -> bool:
    """Return True if stock should be PROTECTED (not replaced)."""
    row = work[work["stock_id"].astype(str) == str(stock_id).zfill(6)]
    if row.empty:
        return False
    db = float(pd.to_numeric(row.iloc[0].get("downside_beta60", 0), errors="coerce"))
    return db > threshold


def _combo_risk_guard(
    work: pd.DataFrame, stock_id: str,
    amp_threshold: float = 0.19, dd_threshold: float = 0.11,
) -> bool:
    """Block if amp20 > amp_threshold AND dd20 > dd_threshold."""
    row = work[work["stock_id"].astype(str) == str(stock_id).zfill(6)]
    if row.empty:
        return False
    amp = float(pd.to_numeric(row.iloc[0].get("amp20", 0), errors="coerce"))
    dd = float(pd.to_numeric(row.iloc[0].get("max_drawdown20", 0), errors="coerce"))
    return amp > amp_threshold and dd > dd_threshold


def _ret20_guard(work: pd.DataFrame, stock_id: str, threshold: float = 0.08) -> bool:
    """Block if replaced stock ret20 > threshold (strong positive trend)."""
    row = work[work["stock_id"].astype(str) == str(stock_id).zfill(6)]
    if row.empty:
        return False
    ret20 = float(pd.to_numeric(row.iloc[0].get("ret20", 0), errors="coerce"))
    return ret20 > threshold


def _consensus_guard(work: pd.DataFrame, stock_id: str, min_count: int = 2) -> bool:
    """Block if replaced stock has consensus_count >= min_count (when available)."""
    row = work[work["stock_id"].astype(str) == str(stock_id).zfill(6)]
    if row.empty:
        return False
    cc = pd.to_numeric(row.iloc[0].get("grr_consensus_count", float("nan")), errors="coerce")
    if pd.isna(cc):
        return False  # NaN => no consensus data => don't block
    return float(cc) >= min_count


# ---------------------------------------------------------------------------
# Variant definition
# ---------------------------------------------------------------------------

GUARD_VARIANTS: dict[str, dict[str, Any]] = {
    "raw_antilottery": {
        "description": "No guards (baseline replication)",
        "guards": [],
    },
    "dbeta_guard_135": {
        "description": "Block if downside_beta60 > 1.35",
        "guards": [("dbeta", {"threshold": 1.35})],
    },
    "dbeta_guard_120": {
        "description": "Block if downside_beta60 > 1.20 (stricter)",
        "guards": [("dbeta", {"threshold": 1.20})],
    },
    "combo_risk_guard": {
        "description": "Block if amp20 > 0.19 AND dd20 > 0.11",
        "guards": [("combo_risk", {"amp_threshold": 0.19, "dd_threshold": 0.11})],
    },
    "full_guard": {
        "description": "dbeta>1.35 OR (amp20>0.19 AND dd20>0.11)",
        "guards": [
            ("dbeta", {"threshold": 1.35}),
            ("combo_risk", {"amp_threshold": 0.19, "dd_threshold": 0.11}),
        ],
    },
    "ret20_guard_08": {
        "description": "Block if replaced stock ret20 > 0.08 (positive trend protection)",
        "guards": [("ret20", {"threshold": 0.08})],
    },
    "combined_all": {
        "description": "dbeta>1.35 OR combo_risk OR ret20>0.08",
        "guards": [
            ("dbeta", {"threshold": 1.35}),
            ("combo_risk", {"amp_threshold": 0.19, "dd_threshold": 0.11}),
            ("ret20", {"threshold": 0.08}),
        ],
    },
    "dbeta_combo": {
        "description": "dbeta>1.35 OR (amp20>0.19 AND dd20>0.11) — lean version",
        "guards": [
            ("dbeta", {"threshold": 1.35}),
            ("combo_risk", {"amp_threshold": 0.19, "dd_threshold": 0.11}),
        ],
    },
}


def _apply_guards(
    work: pd.DataFrame, stock_id: str,
    guards: list[tuple[str, dict[str, float]]],
) -> tuple[bool, str]:
    """Return (blocked, reason). blocked=True means stock is PROTECTED."""
    for guard_type, params in guards:
        if guard_type == "dbeta" and _dbeta_guard(work, stock_id, **params):
            return True, f"dbeta>{params['threshold']}"
        if guard_type == "combo_risk" and _combo_risk_guard(work, stock_id, **params):
            return True, f"combo_risk(amp>{params['amp_threshold']},dd>{params['dd_threshold']})"
        if guard_type == "ret20" and _ret20_guard(work, stock_id, **params):
            return True, f"ret20>{params['threshold']}"
        if guard_type == "consensus" and _consensus_guard(work, stock_id, **params):
            return True, f"consensus>={params['min_count']}"
    return False, ""


# ---------------------------------------------------------------------------
# Single-window evaluation
# ---------------------------------------------------------------------------

def _eval_insert(
    work: pd.DataFrame,
    window: str,
    base_ids: list[str],
    base_name: str,
    variant: str,
    guard_spec: list[tuple[str, dict[str, float]]],
    gate_reason: str = "",
) -> dict[str, Any]:
    base_return = _realized_for_ids(work, base_ids)
    base_bad, base_very_bad = _bad_counts(work, base_ids)
    target, target_score, target_risk = _target(work, base_ids, "lowest_score")
    row: dict[str, Any] = {
        "window": window,
        "variant": variant,
        "base_name": base_name,
        "base_return": base_return,
        "shadow_return": base_return,
        "delta_vs_base": 0.0,
        "accepted_swap_count": 0,
        "candidate_stock": "",
        "candidate_score": None,
        "candidate_risk": None,
        "replaced_stock": target,
        "replaced_score": target_score,
        "replaced_risk": target_risk,
        "replaced_dbeta": None,
        "replaced_amp20": None,
        "replaced_dd20": None,
        "replaced_ret20": None,
        "guard_blocked": False,
        "guard_reason": "",
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

    # Record replaced stock features
    t_row = work[work["stock_id"].astype(str) == target]
    if not t_row.empty:
        tr = t_row.iloc[0]
        row["replaced_dbeta"] = float(pd.to_numeric(tr.get("downside_beta60", 0), errors="coerce"))
        row["replaced_amp20"] = float(pd.to_numeric(tr.get("amp20", 0), errors="coerce"))
        row["replaced_dd20"] = float(pd.to_numeric(tr.get("max_drawdown20", 0), errors="coerce"))
        row["replaced_ret20"] = float(pd.to_numeric(tr.get("ret20", 0), errors="coerce"))

    # Check guard
    blocked, reason = _apply_guards(work, target, guard_spec)
    if blocked:
        row["guard_blocked"] = True
        row["guard_reason"] = reason
        row["blocked_reason"] = f"guard:{reason}"
        return row

    # Find candidate
    candidates = _pool(work, base_ids, "anti_lottery", 1)
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
            "candidate_score": float(cand["candidate_score"]),
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_shadow(
    source_run: Path, raw_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = build_shadow_inputs(source_run, raw_path)
    micro = build_raw_micro_panel(raw_path)
    micro_by_date = {str(d): g.drop(columns=["date"]).copy() for d, g in micro.groupby("date", sort=False)}

    rows: list[dict[str, Any]] = []
    for window in sorted(inputs["default_top5"]):
        work = inputs["work_by_window"][window].copy()
        day_micro = micro_by_date.get(window)
        if day_micro is not None:
            work = work.merge(day_micro, on="stock_id", how="left")
        default_ids = inputs["default_top5"][window]

        for variant, spec in GUARD_VARIANTS.items():
            rows.append(
                _eval_insert(
                    work, window, default_ids, "default",
                    variant, spec["guards"],
                )
            )

    result = pd.DataFrame(rows)
    summary = summarize(result)
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default=str(SOURCE_RUN.relative_to(ROOT)))
    parser.add_argument("--raw-path", default="data/train_hs300_20260424.csv")
    parser.add_argument("--out-dir", default=str(OUT_DIR.relative_to(ROOT)))
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, summary = run_shadow(ROOT / args.source_run, ROOT / args.raw_path)
    rows.to_csv(out_dir / "conditional_windows.csv", index=False)
    summary.to_csv(out_dir / "conditional_summary.csv", index=False)

    # Build comparison table
    merged = _build_comparison(summary)
    merged.to_csv(out_dir / "comparison_table.csv", index=False)

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "rows": len(rows),
                "out_dir": str(out_dir),
                "variants": list(GUARD_VARIANTS.keys()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows)}, ensure_ascii=False), flush=True)
    print("\n=== 20/40/60 COMPARISON ===")
    print(merged.to_string(index=False))

    # Print guard trigger details
    _print_guard_details(rows)


def _build_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Build a side-by-side comparison table of all variants across buckets."""
    rows: list[dict[str, Any]] = []
    for variant in GUARD_VARIANTS:
        entry: dict[str, Any] = {"variant": variant}
        for bucket in ["20win", "40win", "60win"]:
            sub = summary[(summary["bucket"] == bucket) & (summary["variant"] == variant)]
            if sub.empty:
                for k in ["delta_mean", "delta_q10", "delta_worst", "negative_delta_count", "accepted_swaps"]:
                    entry[f"{bucket}_{k}"] = None
                continue
            r = sub.iloc[0]
            entry[f"{bucket}_delta_mean"] = float(r["delta_mean"])
            entry[f"{bucket}_delta_q10"] = float(r["delta_q10"])
            entry[f"{bucket}_delta_worst"] = float(r["delta_worst"])
            entry[f"{bucket}_neg_count"] = int(r["negative_delta_count"])
            entry[f"{bucket}_swaps"] = int(r["accepted_swaps"])
        rows.append(entry)
    return pd.DataFrame(rows)


def _print_guard_details(rows: pd.DataFrame) -> None:
    """Print per-window guard decisions."""
    guarded = rows[
        (rows["variant"] == "dbeta_guard_135")
        & (rows["guard_blocked"] == True)
    ]
    if not guarded.empty:
        print(f"\n=== GUARD BLOCKED WINDOWS (dbeta_guard_135) ===")
        for _, r in guarded.iterrows():
            print(
                f"  {r['window']}: {r['replaced_stock']} dbeta={r['replaced_dbeta']:.3f} "
                f"amp20={r['replaced_amp20']:.4f} dd20={r['replaced_dd20']:.4f} "
                f"ret20={r['replaced_ret20']:.4f} -> BLOCKED by {r['guard_reason']}"
            )

    swaps = rows[
        (rows["variant"] == "dbeta_guard_135")
        & (rows["accepted_swap_count"] == 1)
    ]
    if not swaps.empty:
        print(f"\n=== ACCEPTED SWAPS (dbeta_guard_135) ===")
        for _, r in swaps.iterrows():
            print(
                f"  {r['window']}: {r['replaced_stock']} (score={r['replaced_score']:.3f}) -> "
                f"{r['candidate_stock']} (score={r['candidate_score']:.3f}) "
                f"delta={r['delta_vs_base']:.4f} raw_delta={r['raw_stock_delta']:.4f}"
            )

    # Show raw_antilottery windows for comparison
    raw_swaps = rows[
        (rows["variant"] == "raw_antilottery")
        & (rows["accepted_swap_count"] == 1)
    ]
    if not raw_swaps.empty:
        print(f"\n=== ALL RAW ANTI-LOTTERY SWAPS ===")
        for _, r in raw_swaps.iterrows():
            print(
                f"  {r['window']}: {r['replaced_stock']} (score={r['replaced_score']:.3f}, "
                f"dbeta={r['replaced_dbeta']:.3f}) -> "
                f"{r['candidate_stock']} delta={r['delta_vs_base']:.4f}"
            )


if __name__ == "__main__":
    main()
