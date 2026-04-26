import json
import os

import joblib
import numpy as np
import pandas as pd

from config import config
from exp009_meta import build_exp009_features


LOG_PREFIX = "[BDC][exp009]"


def _log(message):
    print(f"{LOG_PREFIX} {message}", flush=True)


def _cfg():
    return config.get("selector", {}).get("exp009_meta", {})


def _artifact_path(model_dir, key, default_name):
    cfg = _cfg()
    name = cfg.get(key, default_name)
    return os.path.join(model_dir, name)


def load_exp009_artifacts(model_dir):
    artifact_dir = model_dir
    ranker_path = _artifact_path(artifact_dir, "ranker_path", "exp009_meta_ranker.pkl")
    feature_cols_path = _artifact_path(artifact_dir, "feature_cols_path", "exp009_feature_cols.json")
    meta_config_path = _artifact_path(artifact_dir, "meta_config_path", "exp009_meta_config.json")

    if not (os.path.exists(ranker_path) and os.path.exists(feature_cols_path) and os.path.exists(meta_config_path)):
        return None

    try:
        with open(feature_cols_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
        with open(meta_config_path, "r", encoding="utf-8") as f:
            meta_config = json.load(f)
        artifacts = {
            "ranker": joblib.load(ranker_path),
            "feature_cols": list(feature_cols),
            "meta_config": meta_config,
            "bad_1pct": None,
            "bad_2pct": None,
            "artifact_dir": artifact_dir,
        }
        bad_1pct_path = _artifact_path(artifact_dir, "bad_1pct_path", "exp009_bad_1pct.pkl")
        bad_2pct_path = _artifact_path(artifact_dir, "bad_2pct_path", "exp009_bad_2pct.pkl")
        if os.path.exists(bad_1pct_path):
            artifacts["bad_1pct"] = joblib.load(bad_1pct_path)
        if os.path.exists(bad_2pct_path):
            artifacts["bad_2pct"] = joblib.load(bad_2pct_path)

        runtime_cfg = _cfg()
        for key in ["lambda_bad_1pct", "lambda_bad_2pct", "lambda_uncertainty", "lambda_disagreement"]:
            if key in runtime_cfg:
                artifacts["meta_config"][key] = runtime_cfg[key]
        return artifacts
    except Exception as exc:
        _log(f"artifact load failed, fallback to existing score: {exc}")
        return None


def build_exp009_live_features(score_df):
    out = score_df.copy()
    if "date" not in out.columns:
        out["date"] = pd.Timestamp("1970-01-01")
    return build_exp009_features(out)


def _predict_bad(model, X):
    if model is None:
        return np.zeros(len(X), dtype=np.float64)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1].astype(np.float64)
    return np.asarray(model.predict(X), dtype=np.float64)


def _live_zscore(values, index):
    series = pd.Series(values, index=index, dtype=np.float64).replace([np.inf, -np.inf], np.nan)
    series = series.fillna(series.median() if series.notna().any() else 0.0)
    std = series.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(0.0, index=index, dtype=np.float64)
    return (series - series.mean()) / (std + 1e-9)


def _lambda(meta_config, key, default):
    try:
        return float(meta_config.get(key, default))
    except (TypeError, ValueError):
        return default


def apply_exp009_meta(score_df, artifacts):
    if not artifacts:
        return score_df

    try:
        out = build_exp009_live_features(score_df)
        feature_cols = list(artifacts.get("feature_cols", []))
        missing = [col for col in feature_cols if col not in out.columns]
        if missing:
            _log(f"warning: missing live feature cols filled with 0: {missing}")
            for col in missing:
                out[col] = 0.0

        X = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["exp009_meta_raw"] = np.asarray(artifacts["ranker"].predict(X), dtype=np.float64)
        out["exp009_p_bad_1pct"] = _predict_bad(artifacts.get("bad_1pct"), X)
        out["exp009_p_bad_2pct"] = _predict_bad(artifacts.get("bad_2pct"), X)

        meta_config = artifacts.get("meta_config", {})
        out["exp009_final_score"] = (
            _live_zscore(out["exp009_meta_raw"], out.index)
            - _lambda(meta_config, "lambda_bad_1pct", 0.25) * out["exp009_p_bad_1pct"]
            - _lambda(meta_config, "lambda_bad_2pct", 0.45) * out["exp009_p_bad_2pct"]
            - _lambda(meta_config, "lambda_uncertainty", 0.15) * out["uncertainty_score"]
            - _lambda(meta_config, "lambda_disagreement", 0.10) * out["abs_score_gap"]
        )
        return out
    except Exception as exc:
        _log(f"apply failed, fallback to existing score: {exc}")
        return score_df
