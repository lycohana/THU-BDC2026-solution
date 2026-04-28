from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "code" / "src"))

from branch_router_v2c_shadow import SOURCE_RUN, V2B_DETAIL, _bad_counts, _realized_for_ids, build_shadow_inputs  # noqa: E402


OUT_DIR = ROOT / "temp" / "branch_router_validation" / "aggressive_insert_shadow"


def _split_ids(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [str(x).strip().zfill(6) for x in str(value).split(",") if str(x).strip()]


def _lowest_score_target(work: pd.DataFrame, ids: list[str]) -> tuple[str, float]:
    if not ids:
        return "", 0.0
    score_col = "grr_final_score" if "grr_final_score" in work.columns else "score"
    selected = work[work["stock_id"].astype(str).isin(set(ids))].copy()
    selected["_score"] = pd.to_numeric(selected.get(score_col, 0.0), errors="coerce").fillna(0.0)
    if selected.empty:
        return "", 0.0
    row = selected.sort_values("_score", ascending=True).iloc[0]
    return str(row["stock_id"]).zfill(6), float(row["_score"])


def _stock_return(work: pd.DataFrame, stock_id: str) -> float:
    row = work[work["stock_id"].astype(str) == str(stock_id).zfill(6)]
    if row.empty:
        return 0.0
    return float(pd.to_numeric(row["realized_ret"].iloc[0], errors="coerce"))


def _candidate_pool(branch_candidates: pd.DataFrame, window: str, branch: str, base_ids: list[str]) -> pd.DataFrame:
    sub = branch_candidates[
        (branch_candidates["window_date"].astype(str) == str(window))
        & (branch_candidates["branch_name"].astype(str) == branch)
    ].copy()
    if sub.empty:
        return sub
    sub["candidate_stock"] = sub["candidate_stock_id"].astype(str).str.zfill(6)
    sub["raw_branch_score"] = pd.to_numeric(sub["raw_branch_score"], errors="coerce").fillna(0.0)
    sub["rank_in_branch"] = pd.to_numeric(sub["rank_in_branch"], errors="coerce").fillna(999).astype(int)
    sub = sub[~sub["candidate_stock"].isin(set(base_ids))]
    return sub.sort_values(["rank_in_branch", "raw_branch_score"], ascending=[True, False])


def _eval_insert(
    work: pd.DataFrame,
    branch_candidates: pd.DataFrame,
    window: str,
    base_ids: list[str],
    base_name: str,
    source_branch: str,
    rank_cap: int,
    risk_cap: float | None,
    gate_reason: str = "",
) -> dict[str, Any]:
    base_return = _realized_for_ids(work, base_ids)
    base_bad, base_very_bad = _bad_counts(work, base_ids)
    target, target_score = _lowest_score_target(work, base_ids)
    row: dict[str, Any] = {
        "window": window,
        "variant": f"{base_name}_{source_branch}_top{rank_cap}" + ("_riskcap" if risk_cap is not None else "_noguard"),
        "base_name": base_name,
        "source_branch": source_branch,
        "rank_cap": rank_cap,
        "risk_cap": risk_cap,
        "base_top5": ",".join(base_ids),
        "base_return": base_return,
        "shadow_return": base_return,
        "delta_vs_base": 0.0,
        "accepted_swap_count": 0,
        "candidate_stock": "",
        "candidate_rank": None,
        "candidate_score": None,
        "candidate_risk": None,
        "replaced_stock": target,
        "replaced_score": target_score,
        "raw_candidate_return": None,
        "raw_replaced_return": None,
        "raw_stock_delta": None,
        "weighted_swap_delta": 0.0,
        "blocked_reason": "",
        "final_top5": ",".join(base_ids),
        "bad_count": base_bad,
        "very_bad_count": base_very_bad,
    }
    if gate_reason:
        row["blocked_reason"] = gate_reason
        return row
    if not base_ids or not target:
        row["blocked_reason"] = "missing_base_or_target"
        return row
    pool = _candidate_pool(branch_candidates, window, source_branch, base_ids)
    pool = pool[pool["rank_in_branch"] <= int(rank_cap)]
    if risk_cap is not None and "risk_rank" in pool.columns:
        pool["risk_rank"] = pd.to_numeric(pool["risk_rank"], errors="coerce").fillna(1.0)
        pool = pool[pool["risk_rank"] <= float(risk_cap)]
    if pool.empty:
        row["blocked_reason"] = "no_candidate"
        return row
    cand = pool.iloc[0]
    candidate_stock = str(cand["candidate_stock"]).zfill(6)
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
            "candidate_rank": int(cand["rank_in_branch"]),
            "candidate_score": float(cand["raw_branch_score"]),
            "candidate_risk": float(pd.to_numeric(cand.get("risk_rank", 0.0), errors="coerce")),
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


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    windows = sorted(rows["window"].astype(str).unique())
    buckets = {"20win": windows[-20:], "40win": windows[-40:], "60win": windows}
    out_rows: list[dict[str, Any]] = []
    for bucket, keep in buckets.items():
        scoped = rows[rows["window"].astype(str).isin(set(keep))].copy()
        for variant, sub in scoped.groupby("variant"):
            delta = pd.to_numeric(sub["delta_vs_base"], errors="coerce").fillna(0.0)
            out_rows.append(
                {
                    "bucket": bucket,
                    "variant": variant,
                    "window_count": int(len(sub)),
                    "accepted_swaps": int(pd.to_numeric(sub["accepted_swap_count"], errors="coerce").fillna(0).sum()),
                    "accepted_swap_rate": float(pd.to_numeric(sub["accepted_swap_count"], errors="coerce").fillna(0).mean()) if len(sub) else 0.0,
                    **{f"return_{k}": v for k, v in _metrics(sub["shadow_return"]).items()},
                    **{f"delta_{k}": v for k, v in _metrics(sub["delta_vs_base"]).items()},
                    "negative_delta_count": int((delta < -1e-12).sum()),
                    "very_bad_mean": float(pd.to_numeric(sub["very_bad_count"], errors="coerce").fillna(0).mean()) if len(sub) else 0.0,
                }
            )
    return pd.DataFrame(out_rows)


def run_shadow(source_run: Path, detail_dir: Path, raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = build_shadow_inputs(source_run, raw_path)
    branch_candidates = pd.read_csv(detail_dir / "branch_candidates.csv", dtype={"candidate_stock_id": str})
    ablation = pd.read_csv(detail_dir / "ablation_decisions.csv")
    v2b = ablation[ablation["variant"] == "v2b_trend_plus_ai_overlay"].set_index("window_date")
    rows: list[dict[str, Any]] = []
    for window in sorted(inputs["default_top5"]):
        work = inputs["work_by_window"][window]
        default_ids = inputs["default_top5"][window]
        v2b_ids = _split_ids(v2b.loc[window, "selected_stocks"]) if window in v2b.index else default_ids
        v2b_changed = set(v2b_ids) != set(default_ids)
        for base_name, base_ids in [("default", default_ids), ("v2b", v2b_ids)]:
            for source_branch in ["current_aggressive", "trend_uncluttered"]:
                for rank_cap in [1, 3]:
                    rows.append(
                        _eval_insert(
                            work,
                            branch_candidates,
                            window,
                            base_ids,
                            base_name,
                            source_branch,
                            rank_cap,
                            risk_cap=None,
                        )
                    )
                    rows.append(
                        _eval_insert(
                            work,
                            branch_candidates,
                            window,
                            base_ids,
                            base_name,
                            source_branch,
                            rank_cap,
                            risk_cap=0.90,
                        )
                    )
        for source_branch in ["current_aggressive", "trend_uncluttered"]:
            for rank_cap in [1, 3]:
                for risk_cap in [None, 0.90]:
                    rows.append(
                        _eval_insert(
                            work,
                            branch_candidates,
                            window,
                            v2b_ids,
                            "v2b_no_swap_only",
                            source_branch,
                            rank_cap,
                            risk_cap=risk_cap,
                            gate_reason="v2b_already_swapped" if v2b_changed else "",
                        )
                    )
    result = pd.DataFrame(rows)
    return result, summarize(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default=str(SOURCE_RUN.relative_to(ROOT)))
    parser.add_argument("--detail-dir", default=str(V2B_DETAIL.relative_to(ROOT)))
    parser.add_argument("--raw-path", default="data/train_hs300_20260424.csv")
    parser.add_argument("--out-dir", default=str(OUT_DIR.relative_to(ROOT)))
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = run_shadow(ROOT / args.source_run, ROOT / args.detail_dir, ROOT / args.raw_path)
    rows.to_csv(out_dir / "aggressive_insert_windows.csv", index=False)
    summary.to_csv(out_dir / "aggressive_insert_summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps({"rows": len(rows), "out_dir": str(out_dir), "summary": summary.to_dict(orient="records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows)}, ensure_ascii=False), flush=True)
    print(summary[summary["bucket"] == "60win"].sort_values("delta_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
