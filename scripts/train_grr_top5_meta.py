from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "code" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exp009_meta import run as train_exp009_meta  # noqa: E402
from exp009_oof_builder import build_oof  # noqa: E402


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build strict OOF base scores, then train the GRR-aware exp009 Top5 meta reranker."
    )
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--fold-window-months", type=int, default=2)
    parser.add_argument("--gap-months", type=int, default=1)
    parser.add_argument("--purge-trading-days", type=int, default=5)
    parser.add_argument("--oof-output", default="./temp/grr_top5_oof_scores.csv")
    parser.add_argument("--fold-model-root", default="./temp/grr_top5_fold_models")
    parser.add_argument("--reuse-existing-fold-models", type=int, default=1)
    parser.add_argument("--artifact-dir", default="./model/grr_top5_meta")
    parser.add_argument("--report-dir", default="./temp/grr_top5_meta")
    parser.add_argument("--min-dates", type=int, default=20)
    parser.add_argument("--save-debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    oof_args = Args(
        n_folds=args.n_folds,
        fold_window_months=args.fold_window_months,
        gap_months=args.gap_months,
        purge_trading_days=args.purge_trading_days,
        output=args.oof_output,
        fold_model_root=args.fold_model_root,
        reuse_existing_fold_models=args.reuse_existing_fold_models,
        train_transformer=0,
    )
    build_oof(oof_args)

    meta_args = Args(
        oof_path=args.oof_output,
        output_dir=args.artifact_dir,
        report_dir=args.report_dir,
        min_dates=args.min_dates,
        save_debug=args.save_debug,
    )
    report = train_exp009_meta(meta_args)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
