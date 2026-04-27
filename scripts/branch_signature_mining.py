from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGET_BRANCHES = ["trend_uncluttered", "ai_hardware_mainline_v1", "legal_minrisk_hardened"]


def _metric_rows(df: pd.DataFrame, target_branch: str, features: list[str]) -> pd.DataFrame:
    rows = []
    target = df["legal_oracle_branch"].eq(target_branch)
    for feature in features:
        values = pd.to_numeric(df.get(feature), errors="coerce")
        if values.notna().sum() < 3 or target.sum() == 0 or (~target).sum() == 0:
            continue
        oracle_values = values[target].dropna()
        other_values = values[~target].dropna()
        if oracle_values.empty or other_values.empty:
            continue
        diff = float(oracle_values.mean() - other_values.mean())
        direction = 1 if diff >= 0 else -1
        ranked = values.rank(pct=True, method="average")
        rank_sep = float(ranked[target].mean() - ranked[~target].mean())
        candidates = sorted(values.dropna().unique())
        best = None
        for threshold in candidates:
            pred = values >= threshold if direction >= 0 else values <= threshold
            false_pos = int((pred & ~target).sum())
            missed = int((~pred & target).sum())
            score = missed * 3 + false_pos
            if best is None or score < best["score"]:
                best = {
                    "threshold": float(threshold),
                    "false_positive_count": false_pos,
                    "missed_target_count": missed,
                    "score": score,
                }
        rows.append(
            {
                "target_branch": target_branch,
                "feature": feature,
                "oracle_mean": float(oracle_values.mean()),
                "non_oracle_mean": float(other_values.mean()),
                "difference": diff,
                "rank_separation": rank_sep,
                "direction": ">=" if direction >= 0 else "<=",
                "simple_threshold_candidate": best["threshold"] if best else None,
                "false_positive_count": best["false_positive_count"] if best else None,
                "missed_target_count": best["missed_target_count"] if best else None,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_rank_separation"] = out["rank_separation"].abs()
    out["abs_difference"] = out["difference"].abs()
    return out.sort_values(["abs_rank_separation", "abs_difference"], ascending=[False, False])


def build_signature_table(source_run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshots = pd.read_csv(source_run / "branch_snapshots.csv")
    decisions = pd.read_csv(source_run / "router_decisions.csv")
    keep_cols = ["window_date", "legal_oracle_branch", "legal_oracle_score", "default_score"]
    base = decisions[keep_cols].drop_duplicates("window_date").copy()
    wide_rows = []
    for date, group in snapshots.groupby("window_date", sort=True):
        row = {"window_date": date}
        for _, snap in group.iterrows():
            branch = snap["branch_name"]
            for col in [
                "branch_score_margin_top5_vs_top10",
                "branch_score_dispersion_top10",
                "mean_consensus_support",
                "overlap_with_grr_top5",
                "overlap_with_trend_top5",
                "overlap_with_ai_hardware_top5",
                "high_risk_chaser_count",
                "mean_sigma20",
                "mean_amp20",
                "mean_drawdown20",
                "selected_ret20_strength",
                "branch_rank_entropy_top20",
                "risk_rank_mean_top5",
            ]:
                row[f"{branch}.{col}"] = snap.get(col)
        default = group[group["branch_name"] == "grr_tail_guard"]
        minrisk = group[group["branch_name"] == "legal_minrisk_hardened"]
        if not default.empty and not minrisk.empty:
            row["default_vs_minrisk_risk_gap"] = float(default["risk_rank_mean_top5"].iloc[0]) - float(minrisk["risk_rank_mean_top5"].iloc[0])
        first = group.iloc[0]
        row.update(
            {
                "trend_score": first.get("trend_strength"),
                "clutter_score": first.get("clutter_score"),
                "market_ret5": first.get("market_ret5"),
                "market_ret20": first.get("market_ret20"),
                "breadth_5d": first.get("market_breadth_5d"),
                "risk_off_score": first.get("risk_off_score"),
                "crash_mode": int(bool(first.get("crash_mode"))),
                "amount_chg_5d": first.get("amount_chg_5d"),
            }
        )
        wide_rows.append(row)
    features = base.merge(pd.DataFrame(wide_rows), on="window_date", how="left")
    feature_cols = [col for col in features.columns if col not in keep_cols]
    mined = pd.concat([_metric_rows(features, branch, feature_cols).head(20) for branch in TARGET_BRANCHES], ignore_index=True)
    return features, mined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default="temp/branch_router_analysis/branch_snapshot_repair_20win")
    parser.add_argument("--out-dir", default="temp/branch_router_analysis/signature_mining_20win")
    args = parser.parse_args()
    source_run = ROOT / args.source_run
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    features, mined = build_signature_table(source_run)
    features.to_csv(out_dir / "signature_features.csv", index=False)
    mined.to_csv(out_dir / "signature_rankings.csv", index=False)
    summary = {}
    lines = ["# branch signature mining", ""]
    for branch in TARGET_BRANCHES:
        windows = features.loc[features["legal_oracle_branch"].eq(branch), "window_date"].tolist()
        top = mined[mined["target_branch"].eq(branch)].head(8)
        summary[branch] = {
            "oracle_windows": windows,
            "top_features": json.loads(top.to_json(orient="records")),
        }
        lines.append(f"## {branch}")
        lines.append(f"- oracle_windows: {windows}")
        for _, row in top.iterrows():
            lines.append(
                f"- {row['feature']}: oracle_mean={row['oracle_mean']:.6f}, non_oracle_mean={row['non_oracle_mean']:.6f}, "
                f"diff={row['difference']:.6f}, rank_sep={row['rank_separation']:.6f}, "
                f"threshold {row['direction']} {row['simple_threshold_candidate']:.6f}, "
                f"fp={int(row['false_positive_count'])}, missed={int(row['missed_target_count'])}"
            )
        lines.append("")
    (out_dir / "aggregate.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "signature_mining.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote signature mining to {out_dir}")


if __name__ == "__main__":
    main()
