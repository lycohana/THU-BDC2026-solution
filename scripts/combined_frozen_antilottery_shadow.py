"""Simulate frozen queue (riskoff_rank4_dynamic_pullback_stress_veto_v2) +
conditional anti-lottery overlay (dbeta_guard_135).

Produces 20/40/60-window comparison and 10-window smoke test.
No future returns used in decisions. No runtime code changed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "code" / "src"))

from branch_router_v2c_shadow import SOURCE_RUN, _bad_counts, _realized_for_ids, build_shadow_inputs  # noqa: E402
from microstructure_shadow import _add_scores, _num, _pool, _rank_pct, _stock_return, _target, build_raw_micro_panel  # noqa: E402

FROZEN_WINDOWS = ROOT / "temp" / "submission_freeze" / "post_guard_overlay_queue_freeze" / "post_guard_overlay_windows.csv"
OUT_DIR = ROOT / "temp" / "branch_router_validation" / "combined_frozen_antilottery"


def _dbeta_guard(work: pd.DataFrame, stock_id: str, threshold: float = 1.35) -> bool:
    """Return True if stock should be PROTECTED (not replaced)."""
    row = work[work["stock_id"].astype(str) == str(stock_id).zfill(6)]
    if row.empty:
        return False
    db = float(pd.to_numeric(row.iloc[0].get("downside_beta60", 0), errors="coerce"))
    return db > threshold


def run_combined() -> tuple[pd.DataFrame, dict]:
    """Run combined frozen queue + anti-lottery overlay."""
    inputs = build_shadow_inputs(SOURCE_RUN, ROOT / "data" / "train_hs300_20260424.csv")
    micro = build_raw_micro_panel(ROOT / "data" / "train_hs300_20260424.csv")
    micro_by_date = {str(d): g.drop(columns=["date"]).copy() for d, g in micro.groupby("date", sort=False)}

    frozen = pd.read_csv(FROZEN_WINDOWS)
    frozen_by_window = {str(r["window"]): r for _, r in frozen.iterrows()}

    rows: list[dict] = []
    for window in sorted(inputs["default_top5"]):
        work = inputs["work_by_window"][window].copy()
        day_micro = micro_by_date.get(window)
        if day_micro is not None:
            work = work.merge(day_micro, on="stock_id", how="left")

        default_ids = inputs["default_top5"][window]
        frozen_row = frozen_by_window.get(window)

        # Determine frozen queue return (includes riskoff/pullback if applied)
        if frozen_row is not None:
            frozen_return = float(frozen_row["queue_return"])
            frozen_source = str(frozen_row["queue_source"])
            riskoff_applied = int(frozen_row["riskoff_accepted"])
            pullback_applied = int(frozen_row["pullback_accepted"])
            # Use frozen queue stock list (we don't have it directly, so use default_ids
            # and compute frozen_return as the benchmark)
        else:
            frozen_return = _realized_for_ids(work, default_ids)
            frozen_source = "default"
            riskoff_applied = 0
            pullback_applied = 0

        # Try anti-lottery overlay on default_ids
        target, target_score, target_risk = _target(work, default_ids, "lowest_score")
        al_applied = False
        al_candidate = ""
        al_replaced = ""
        guard_blocked = False
        guard_reason = ""

        if target:
            # Check dbeta guard
            if _dbeta_guard(work, target, threshold=1.35):
                guard_blocked = True
                guard_reason = f"dbeta>{1.35}"
            else:
                # Find candidate
                candidates = _pool(work, default_ids, "anti_lottery", 1)
                if not candidates.empty:
                    candidates = candidates[candidates["model_score"] > float(target_score)]
                if not candidates.empty:
                    cand = candidates.iloc[0]
                    candidate_stock = str(cand["stock_id"]).zfill(6)
                    final_ids = list(default_ids)
                    final_ids[final_ids.index(target)] = candidate_stock
                    al_applied = True
                    al_candidate = candidate_stock
                    al_replaced = target

        # Compute combined return:
        # If anti-lottery would swap, apply it on top of frozen queue result
        if al_applied:
            # Compute anti-lottery delta on default base
            al_ids = list(default_ids)
            al_ids[al_ids.index(target)] = candidate_stock
            al_default_return = _realized_for_ids(work, default_ids)
            al_shadow_return = _realized_for_ids(work, al_ids)
            al_delta = al_shadow_return - al_default_return
            # Combined: frozen_return + al_delta (overlay on top)
            combined_return = frozen_return + al_delta
        else:
            al_delta = 0.0
            combined_return = frozen_return

        base_bad, base_very_bad = _bad_counts(work, default_ids)

        rows.append({
            "window": window,
            "frozen_return": frozen_return,
            "frozen_source": frozen_source,
            "riskoff_applied": riskoff_applied,
            "pullback_applied": pullback_applied,
            "al_applied": int(al_applied),
            "al_replaced": al_replaced,
            "al_candidate": al_candidate,
            "al_delta": al_delta,
            "guard_blocked": guard_blocked,
            "guard_reason": guard_reason,
            "combined_return": combined_return,
            "delta_vs_frozen": combined_return - frozen_return,
            "default_return": _realized_for_ids(work, default_ids),
            "default_top5": ",".join(default_ids),
            "bad_count": base_bad,
        })

    result = pd.DataFrame(rows)
    summary = _summarize_combined(result)
    return result, summary


def _summarize_combined(result: pd.DataFrame) -> dict:
    """Compute 20/40/60-window and 10-window metrics."""
    windows = sorted(result["window"].unique())
    buckets = {
        "20win": windows[-20:],
        "40win": windows[-40:],
        "60win": windows,
    }
    smoke_windows = windows[-10:]

    bucket_metrics: list[dict] = []
    for bucket, keep in buckets.items():
        scoped = result[result["window"].isin(set(keep))]
        frozen_ret = scoped["frozen_return"]
        combined_ret = scoped["combined_return"]
        delta = scoped["delta_vs_frozen"]

        bucket_metrics.append({
            "bucket": bucket,
            "frozen_mean": float(frozen_ret.mean()),
            "frozen_q10": float(frozen_ret.quantile(0.10)),
            "frozen_worst": float(frozen_ret.min()),
            "combined_mean": float(combined_ret.mean()),
            "combined_q10": float(combined_ret.quantile(0.10)),
            "combined_worst": float(combined_ret.min()),
            "delta_mean": float(delta.mean()),
            "delta_q10": float(delta.quantile(0.10)),
            "delta_worst": float(delta.min()),
            "negative_delta_count": int((delta < -1e-12).sum()),
            "al_swaps": int(scoped["al_applied"].sum()),
            "guard_blocks": int(scoped["guard_blocked"].sum()),
        })

    # 10-window smoke test
    smoke = result[result["window"].isin(set(smoke_windows))]
    smoke_metrics = {
        "frozen": {
            "mean": float(smoke["frozen_return"].mean()),
            "q10": float(smoke["frozen_return"].quantile(0.10)),
            "worst": float(smoke["frozen_return"].min()),
        },
        "combined": {
            "mean": float(smoke["combined_return"].mean()),
            "q10": float(smoke["combined_return"].quantile(0.10)),
            "worst": float(smoke["combined_return"].min()),
        },
        "delta": {
            "mean": float(smoke["delta_vs_frozen"].mean()),
            "q10": float(smoke["delta_vs_frozen"].quantile(0.10)),
            "worst": float(smoke["delta_vs_frozen"].min()),
        },
        "frozen_baseline_ref": {
            "mean": 0.0018668252911232218,
            "q10": -0.005652222062380649,
            "worst": -0.0141071555817288,
        },
        "frozen_current_ref": {
            "mean": 0.014543919874455918,
            "q10": -0.00405070431535826,
            "worst": -0.0057767848778843,
        },
    }

    return {
        "buckets": bucket_metrics,
        "smoke_10win": smoke_metrics,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result, summary = run_combined()
    result.to_csv(OUT_DIR / "combined_windows.csv", index=False)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Print comparison
    print("\n=== 20/40/60 WINDOW COMPARISON: FROZEN + ANTI-LOTTERY ===")
    for b in summary["buckets"]:
        print(f"\n{b['bucket']}:")
        print(f"  frozen:   mean={b['frozen_mean']:.6f}  q10={b['frozen_q10']:.6f}  worst={b['frozen_worst']:.6f}")
        print(f"  combined: mean={b['combined_mean']:.6f}  q10={b['combined_q10']:.6f}  worst={b['combined_worst']:.6f}")
        print(f"  delta:    mean={b['delta_mean']:.6f}  q10={b['delta_q10']:.6f}  worst={b['delta_worst']:.6f}")
        print(f"  neg_count={b['negative_delta_count']}  al_swaps={b['al_swaps']}  guard_blocks={b['guard_blocks']}")

    print("\n=== 10-WINDOW SMOKE TEST ===")
    s = summary["smoke_10win"]
    print(f"  frozen_current_ref: mean={s['frozen_current_ref']['mean']:.6f}  q10={s['frozen_current_ref']['q10']:.6f}  worst={s['frozen_current_ref']['worst']:.6f}")
    print(f"  frozen:             mean={s['frozen']['mean']:.6f}  q10={s['frozen']['q10']:.6f}  worst={s['frozen']['worst']:.6f}")
    print(f"  combined:           mean={s['combined']['mean']:.6f}  q10={s['combined']['q10']:.6f}  worst={s['combined']['worst']:.6f}")
    print(f"  delta:              mean={s['delta']['mean']:.6f}  q10={s['delta']['q10']:.6f}  worst={s['delta']['worst']:.6f}")

    # Check 2026-04-24
    last = result[result["window"] == "2026-04-24"]
    if not last.empty:
        r = last.iloc[0]
        print(f"\n2026-04-24: frozen_ret={r['frozen_return']:.4f}, combined_ret={r['combined_return']:.4f}, al_applied={r['al_applied']}, guard_blocked={r['guard_blocked']}")

    # Print anti-lottery overlay windows
    al_windows = result[result["al_applied"] == 1]
    if not al_windows.empty:
        print(f"\n=== ANTI-LOTTERY OVERLAY APPLIED ({len(al_windows)} windows) ===")
        for _, r in al_windows.iterrows():
            print(f"  {r['window']}: replace {r['al_replaced']} -> {r['al_candidate']}, al_delta={r['al_delta']:.4f}, frozen={r['frozen_return']:.4f}, combined={r['combined_return']:.4f}")


if __name__ == "__main__":
    main()
