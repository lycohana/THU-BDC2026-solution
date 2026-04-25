"""Train an independent branch-level meta gate from replay diagnostics.

Input is one or more branch_diagnostics.csv files produced by
scripts/batch_window_analysis.py. Labels come only from each legal branch's own
future replay score; no external benchmark or fixed portfolio is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import mean_absolute_error, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "code" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import config  # noqa: E402
from predict import META_GATE_NUMERIC_FEATURES  # noqa: E402


DEFAULT_DIAGNOSTICS = ROOT / "temp" / "batch_window_analysis" / "exp005_single_20260213_v2" / "branch_diagnostics.csv"


def normalize_input_paths(values: list[str] | None) -> list[Path]:
    if not values:
        return [DEFAULT_DIAGNOSTICS]

    paths: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.rglob("branch_diagnostics.csv")))
        else:
            paths.append(path)
    return [p if p.is_absolute() else ROOT / p for p in paths]


def load_diagnostics(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            print(f"[meta_gate] skip_missing={path}")
            continue
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No branch_diagnostics.csv files found")
    data = pd.concat(frames, ignore_index=True)
    required = {"branch", "regime", "score", "bad_count", "very_bad_count"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"diagnostics missing required columns: {missing}")
    return data


def build_features(data: pd.DataFrame, branches: list[str], regimes: list[str]) -> tuple[pd.DataFrame, list[str]]:
    X = pd.DataFrame(index=data.index)
    for col in META_GATE_NUMERIC_FEATURES:
        X[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0) if col in data.columns else 0.0
    for branch in branches:
        X[f"branch={branch}"] = (data["branch"].astype(str) == branch).astype(float)
    for regime in regimes:
        X[f"regime={regime}"] = (data["regime"].astype(str) == regime).astype(float)
    return X.astype(np.float64), list(X.columns)


def make_models(random_state: int):
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        safe_model = LGBMClassifier(
            objective="binary",
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=8,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=3.0,
            random_state=random_state,
            n_jobs=4,
            verbosity=-1,
        )
        score_model = LGBMRegressor(
            objective="regression",
            n_estimators=250,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=8,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=3.0,
            random_state=random_state,
            n_jobs=4,
            verbosity=-1,
        )
        return safe_model, score_model
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier

        safe_model = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=4,
        )
        score_model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.03,
            random_state=random_state,
        )
        return safe_model, score_model


def train_gate(data: pd.DataFrame, safe_floor: float, output_path: Path) -> dict:
    work = data.copy()
    work["score"] = pd.to_numeric(work["score"], errors="coerce").fillna(0.0)
    work["very_bad_count"] = pd.to_numeric(work["very_bad_count"], errors="coerce").fillna(0.0)
    work["safe_label"] = ((work["score"] > safe_floor) & (work["very_bad_count"] <= 0)).astype(int)

    branches = sorted(work["branch"].astype(str).unique().tolist())
    regimes = sorted(work["regime"].astype(str).unique().tolist())
    X, feature_columns = build_features(work, branches, regimes)
    y_safe = work["safe_label"].to_numpy(dtype=np.int64)
    y_score = work["score"].to_numpy(dtype=np.float64)

    random_state = int(config.get("seed", 42))
    if len(np.unique(y_safe)) < 2:
        safe_model = DummyClassifier(strategy="constant", constant=int(y_safe[0] if len(y_safe) else 0))
        safe_model.fit(X, y_safe)
    else:
        safe_model, _ = make_models(random_state)
        safe_model.fit(X, y_safe)

    _, score_model = make_models(random_state)
    score_model.fit(X, y_score)

    if hasattr(safe_model, "predict_proba"):
        safe_prob = safe_model.predict_proba(X)
        safe_prob = safe_prob[:, 1] if safe_prob.shape[1] > 1 else safe_prob[:, 0]
    else:
        safe_prob = safe_model.predict(X)
    score_pred = score_model.predict(X)

    metrics = {
        "rows": int(len(work)),
        "anchors": int(work["anchor_date"].nunique()) if "anchor_date" in work.columns else None,
        "branches": branches,
        "regimes": regimes,
        "safe_floor": float(safe_floor),
        "safe_rate": float(y_safe.mean()) if len(y_safe) else 0.0,
        "score_mae_in_sample": float(mean_absolute_error(y_score, score_pred)) if len(y_score) else 0.0,
    }
    if len(np.unique(y_safe)) >= 2:
        metrics["safe_auc_in_sample"] = float(roc_auc_score(y_safe, safe_prob))

    artifact = {
        "safe_model": safe_model,
        "score_model": score_model,
        "feature_columns": feature_columns,
        "branches": branches,
        "regimes": regimes,
        "metrics": metrics,
        "note": "Train on purged rolling OOF diagnostics before enabling in production.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    (output_path.with_suffix(".json")).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "diagnostics",
        nargs="*",
        help="branch_diagnostics.csv files or directories containing them",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / config["output_dir"] / "branch_meta_gate.pkl"),
    )
    parser.add_argument("--safe-floor", type=float, default=-0.01)
    args = parser.parse_args()

    paths = normalize_input_paths(args.diagnostics)
    data = load_diagnostics(paths)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    metrics = train_gate(data, safe_floor=args.safe_floor, output_path=output_path)
    print("[meta_gate] wrote", output_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
