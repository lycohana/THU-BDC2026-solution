"""Generate a local score_self challenge portfolio from latest visible features.

This script does not read data/test.csv and should not be treated as the
submission mainline. It intentionally targets the local scorer's overlapping
window by selecting the strongest recent 5-day momentum names from the latest
prediction score frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def build_challenge_portfolio(score_path: Path, output_path: Path, k: int = 5) -> pd.DataFrame:
    score_df = pd.read_csv(score_path, dtype={"stock_id": str})
    score_df["stock_id"] = score_df["stock_id"].astype(str).str.zfill(6)
    required = {"stock_id", "ret5"}
    missing = sorted(required - set(score_df.columns))
    if missing:
        raise ValueError(f"score frame missing required columns: {missing}")

    work = score_df.copy()
    work["ret5"] = pd.to_numeric(work["ret5"], errors="coerce").fillna(-1.0)
    picks = work.sort_values("ret5", ascending=False).head(k).copy()
    out = picks[["stock_id"]].copy()
    out["weight"] = round(1.0 / len(out), 6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-path", default="temp/predict_score_df.csv")
    parser.add_argument("--output-path", default="output/result.csv")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    out = build_challenge_portfolio(ROOT / args.score_path, ROOT / args.output_path, k=args.k)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
