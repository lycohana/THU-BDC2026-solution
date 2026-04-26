import argparse
import json
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from config import config


LOG_PREFIX = "[BDC][exp009]"
VERSION = "exp009_oof_meta_badaware_selector"

CORE_SCORE_COLS = ["transformer", "lgb", "score"]
EXPERT_SCORE_COLS = [
    "transformer",
    "lgb",
    "score",
    "lgb_rank_score",
    "lgb_reg_score",
    "lgb_top5_rank_score",
    "lgb_top5_score",
]
RISK_DEFAULTS = {
    "sigma20": 0.0,
    "median_amount20": 0.0,
    "ret1": 0.0,
    "ret5": 0.0,
    "ret20": 0.0,
    "amp20": 0.0,
    "beta60": 1.0,
    "downside_beta60": 1.0,
    "max_drawdown20": 0.0,
}

EXP009_FEATURE_COLS = [
    "transformer_z",
    "lgb_z",
    "blend_z",
    "transformer_rank_pct",
    "lgb_rank_pct",
    "blend_rank_pct",
    "lgb_rank_score_z",
    "lgb_reg_score_z",
    "lgb_top5_z",
    "lgb_rank_score_rank_pct",
    "lgb_reg_score_rank_pct",
    "lgb_top5_rank_pct",
    "grr_rrf_score",
    "grr_rrf_rank_pct",
    "grr_candidate_flag",
    "grr_router_score",
    "grr_router_rank_pct",
    "rank_mean",
    "rank_std",
    "score_gap",
    "abs_score_gap",
    "consensus_top5",
    "consensus_top10",
    "consensus_top20",
    "liq_rank",
    "sigma_rank",
    "amp_rank",
    "ret1_rank",
    "ret5_rank",
    "ret20_rank",
    "beta60_rank",
    "downside_beta60_rank",
    "max_drawdown20_rank",
    "high_vol_flag",
    "extreme_momo_flag",
    "overheat_flag",
    "reversal_flag",
    "tail_risk_flag",
    "bad_pick_risk",
    "tail_risk_score",
    "uncertainty_score",
]


def _log(message):
    print(f"{LOG_PREFIX} {message}", flush=True)


def _as_numeric(series, default=0.0):
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill = values.median() if values.notna().any() else default
    return values.fillna(fill).astype(np.float64)


def _ensure_numeric_col(df, col, default=0.0):
    if col not in df.columns:
        df[col] = default
    df[col] = _as_numeric(df[col], default=default)
    return df


def _zscore_series(series):
    values = pd.to_numeric(series, errors="coerce").astype(np.float64)
    mean = values.mean()
    std = values.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(0.0, index=series.index, dtype=np.float64)
    return ((values - mean) / (std + 1e-9)).astype(np.float64)


def _group_zscore(df, col):
    return df.groupby("date", sort=False)[col].transform(_zscore_series).astype(np.float64)


def _group_rank_pct(df, col):
    return (
        df.groupby("date", sort=False)[col]
        .transform(lambda x: pd.to_numeric(x, errors="coerce").rank(pct=True, method="average"))
        .fillna(0.5)
        .astype(np.float64)
    )


def _group_topn_flag(df, col, n):
    rank = df.groupby("date", sort=False)[col].rank(ascending=False, method="first")
    return (rank <= n).astype(np.float64)


def _group_rrf(df, score_cols, k=60):
    valid_cols = [col for col in score_cols if col in df.columns]
    if not valid_cols:
        return pd.Series(0.0, index=df.index, dtype=np.float64)

    out = pd.Series(0.0, index=df.index, dtype=np.float64)
    for col in valid_cols:
        ranks = df.groupby("date", sort=False)[col].rank(ascending=False, method="first")
        out = out.add(1.0 / (float(k) + ranks), fill_value=0.0)
    return out.astype(np.float64)


def _group_union_topk_flag(df, score_cols, k=24):
    valid_cols = [col for col in score_cols if col in df.columns]
    if not valid_cols:
        return pd.Series(0.0, index=df.index, dtype=np.float64)

    flag = pd.Series(False, index=df.index)
    grouped = df.groupby("date", sort=False)
    for col in valid_cols:
        ranks = grouped[col].rank(ascending=False, method="first")
        flag = flag | (ranks <= int(k))
    return flag.astype(np.float64)


def _ensure_core_scores(df):
    present = [col for col in CORE_SCORE_COLS if col in df.columns]
    if len(present) < 2:
        raise ValueError("OOF score 文件中 transformer/lgb/score 至少需要存在两个。")

    for col in present:
        df[col] = _as_numeric(df[col], default=0.0)

    if "score" not in df.columns:
        df["score"] = np.mean([df[col].to_numpy(dtype=np.float64) for col in present], axis=0)
    if "transformer" not in df.columns:
        df["transformer"] = df["score"]
    if "lgb" not in df.columns:
        df["lgb"] = df["score"]

    for col in CORE_SCORE_COLS:
        df[col] = _as_numeric(df[col], default=0.0)
    if "lgb_top5_rank_score" not in df.columns and "lgb_top5_score" in df.columns:
        df["lgb_top5_rank_score"] = df["lgb_top5_score"]
    if "lgb_top5_score" not in df.columns and "lgb_top5_rank_score" in df.columns:
        df["lgb_top5_score"] = df["lgb_top5_rank_score"]
    for col in EXPERT_SCORE_COLS:
        if col not in df.columns:
            df[col] = df["lgb"] if col.startswith("lgb_") else df["score"]
        df[col] = _as_numeric(df[col], default=0.0)
    return df


def validate_oof_schema(df, min_dates=20):
    required = {"date", "stock_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"OOF score 文件缺少必要列: {missing}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("OOF score 文件中的 date 存在无法转为 datetime 的值。")

    if "target" not in out.columns:
        if "forward_open_return" not in out.columns:
            raise ValueError("OOF score 文件必须包含 target 或 forward_open_return。")
        out = out.rename(columns={"forward_open_return": "target"})
    out["target"] = _as_numeric(out["target"], default=0.0)

    score_cols = [col for col in CORE_SCORE_COLS if col in out.columns]
    if len(score_cols) < 2:
        raise ValueError("OOF score 文件中 transformer/lgb/score 至少需要存在两个。")

    n_dates = out["date"].nunique()
    if n_dates < min_dates:
        raise ValueError(f"OOF score 文件交易日不足: {n_dates} < min_dates={min_dates}")

    small_days = out.groupby("date")["stock_id"].size()
    small_days = small_days[small_days < 20]
    if len(small_days) > 0:
        _log(f"warning: {len(small_days)} dates have fewer than 20 stocks")

    return out


def build_exp009_features(df):
    out = df.copy()
    if "date" not in out.columns:
        out["date"] = pd.Timestamp("1970-01-01")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").fillna(pd.Timestamp("1970-01-01"))
    if "stock_id" not in out.columns:
        raise ValueError("build_exp009_features requires stock_id")

    out = _ensure_core_scores(out)
    for col, default in RISK_DEFAULTS.items():
        out = _ensure_numeric_col(out, col, default=default)

    out["transformer_z"] = _group_zscore(out, "transformer")
    out["lgb_z"] = _group_zscore(out, "lgb")
    out["blend_z"] = _group_zscore(out, "score")

    out["transformer_rank_pct"] = _group_rank_pct(out, "transformer")
    out["lgb_rank_pct"] = _group_rank_pct(out, "lgb")
    out["blend_rank_pct"] = _group_rank_pct(out, "score")
    out["lgb_rank_score_z"] = _group_zscore(out, "lgb_rank_score")
    out["lgb_reg_score_z"] = _group_zscore(out, "lgb_reg_score")
    out["lgb_top5_z"] = _group_zscore(out, "lgb_top5_rank_score")
    out["lgb_rank_score_rank_pct"] = _group_rank_pct(out, "lgb_rank_score")
    out["lgb_reg_score_rank_pct"] = _group_rank_pct(out, "lgb_reg_score")
    out["lgb_top5_rank_pct"] = _group_rank_pct(out, "lgb_top5_rank_score")
    rank_cols = ["transformer_rank_pct", "lgb_rank_pct", "blend_rank_pct"]
    out["rank_mean"] = out[rank_cols].mean(axis=1).astype(np.float64)
    out["rank_std"] = out[rank_cols].std(axis=1, ddof=0).fillna(0.0).astype(np.float64)
    out["score_gap"] = (out["transformer_z"] - out["lgb_z"]).astype(np.float64)
    out["abs_score_gap"] = out["score_gap"].abs().astype(np.float64)

    grr_cfg = config.get("grr_top5", {})
    rrf_cols = [col for col in grr_cfg.get("expert_cols", EXPERT_SCORE_COLS) if col in out.columns]
    out["grr_rrf_score"] = _group_rrf(out, rrf_cols, k=grr_cfg.get("rrf_k", 60))
    out["grr_rrf_rank_pct"] = _group_rank_pct(out, "grr_rrf_score")
    out["grr_candidate_flag"] = _group_union_topk_flag(out, rrf_cols, k=grr_cfg.get("candidate_k", 24))
    out["grr_router_score"] = (
        0.35 * out["lgb_rank_pct"]
        + 0.20 * out["lgb_top5_rank_pct"]
        + 0.25 * out["transformer_rank_pct"]
        + 0.20 * out["grr_rrf_rank_pct"]
    ).astype(np.float64)
    out["grr_router_rank_pct"] = _group_rank_pct(out, "grr_router_score")

    tf_top5 = _group_topn_flag(out, "transformer", 5)
    lgb_top5 = _group_topn_flag(out, "lgb", 5)
    tf_top10 = _group_topn_flag(out, "transformer", 10)
    lgb_top10 = _group_topn_flag(out, "lgb", 10)
    tf_top20 = _group_topn_flag(out, "transformer", 20)
    lgb_top20 = _group_topn_flag(out, "lgb", 20)
    out["consensus_top5"] = ((tf_top5 > 0) & (lgb_top5 > 0)).astype(np.float64)
    out["consensus_top10"] = ((tf_top10 > 0) & (lgb_top10 > 0)).astype(np.float64)
    out["consensus_top20"] = ((tf_top20 > 0) & (lgb_top20 > 0)).astype(np.float64)

    out["liq_rank"] = _group_rank_pct(out, "median_amount20")
    out["sigma_rank"] = _group_rank_pct(out, "sigma20")
    out["amp_rank"] = _group_rank_pct(out, "amp20")
    out["ret1_rank"] = _group_rank_pct(out, "ret1")
    out["ret5_rank"] = _group_rank_pct(out, "ret5")
    out["ret20_rank"] = _group_rank_pct(out, "ret20")
    out["beta60_rank"] = _group_rank_pct(out, "beta60")
    out["downside_beta60_rank"] = _group_rank_pct(out, "downside_beta60")
    out["max_drawdown20_rank"] = _group_rank_pct(out, "max_drawdown20")

    out["high_vol_flag"] = ((out["sigma_rank"] > 0.85) | (out["amp_rank"] > 0.85)).astype(np.float64)
    out["extreme_momo_flag"] = (
        (out["ret1_rank"] > 0.95) | (out["ret5_rank"] > 0.90) | (out["ret20_rank"] > 0.90)
    ).astype(np.float64)
    out["overheat_flag"] = (
        (out["ret5_rank"] > 0.75) & (out["ret20_rank"] > 0.70) & (out["amp_rank"] > 0.75)
    ).astype(np.float64)
    out["reversal_flag"] = (
        ((out["ret5_rank"] < 0.15) | (out["ret20_rank"] < 0.15)) & (out["ret1_rank"] > 0.70)
    ).astype(np.float64)
    out["tail_risk_flag"] = (
        (out["max_drawdown20_rank"] < 0.15) | (out["downside_beta60_rank"] > 0.85) | (out["sigma_rank"] > 0.90)
    ).astype(np.float64)

    out["tail_risk_score"] = (
        0.25 * out["sigma_rank"]
        + 0.20 * out["amp_rank"]
        + 0.20 * out["downside_beta60_rank"]
        + 0.20 * (1.0 - out["max_drawdown20_rank"])
        + 0.15 * out["high_vol_flag"]
    ).clip(0.0, 2.0)
    out["uncertainty_score"] = (
        0.45 * out["rank_std"]
        + 0.25 * out["abs_score_gap"].clip(0.0, 3.0) / 3.0
        + 0.15 * (1.0 - out["consensus_top20"])
        + 0.15 * out["tail_risk_flag"]
    ).clip(0.0, 2.0)
    out["bad_pick_risk"] = (
        0.35 * out["tail_risk_score"]
        + 0.25 * out["uncertainty_score"]
        + 0.15 * out["overheat_flag"]
        + 0.15 * out["extreme_momo_flag"]
        + 0.10 * (1.0 - out["liq_rank"])
    ).clip(0.0, 2.0)

    for col in EXP009_FEATURE_COLS:
        out[col] = _as_numeric(out[col], default=0.0)
    return out


def make_relevance_labels(df):
    out = df.copy()
    if "target" not in out.columns and "forward_open_return" in out.columns:
        out = out.rename(columns={"forward_open_return": "target"})
    if "target" not in out.columns:
        raise ValueError("make_relevance_labels requires target or forward_open_return")
    if "blend_rank_pct" not in out.columns:
        out = build_exp009_features(out)

    out["target"] = _as_numeric(out["target"], default=0.0)
    pct = (
        out.groupby("date", sort=False)["target"]
        .transform(lambda x: pd.to_numeric(x, errors="coerce").rank(pct=True, method="average"))
        .fillna(0.0)
    )
    rel = np.zeros(len(out), dtype=np.int32)
    rel[pct >= 0.60] = 1
    rel[pct >= 0.80] = 2
    rel[pct >= 0.90] = 3
    rel[pct >= 0.95] = 4
    out["relevance"] = rel
    out["bad_1pct"] = (out["target"] < -0.01).astype(np.int32)
    out["bad_2pct"] = (out["target"] < -0.02).astype(np.int32)
    out["very_bad"] = (out["target"] < -0.03).astype(np.int32)
    out["false_positive"] = ((out["blend_rank_pct"] >= 0.90) & (out["target"] < 0.0)).astype(np.int32)
    return out


def _sort_for_ranker(df):
    return df.sort_values(["date", "stock_id"]).reset_index(drop=True)


def _groups_by_date(df):
    return df.groupby("date", sort=True).size().astype(int).tolist()


def train_meta_ranker(train_df, valid_df, feature_cols):
    import lightgbm as lgb

    train_df = _sort_for_ranker(train_df)
    valid_df = _sort_for_ranker(valid_df)
    seed = int(config.get("seed", 42))
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[5],
        label_gain=config.get("label_gain", [0, 1, 2, 4, 8]),
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=int(config.get("lgb", {}).get("n_jobs", 8)),
        verbosity=-1,
    )
    ranker.fit(
        train_df[feature_cols],
        train_df["relevance"],
        group=_groups_by_date(train_df),
        eval_set=[(valid_df[feature_cols], valid_df["relevance"])],
        eval_group=[_groups_by_date(valid_df)],
        eval_at=[5],
        callbacks=[lgb.log_evaluation(period=100)],
    )
    return ranker


def _fit_bad_model(train_df, valid_df, feature_cols, label_col):
    import lightgbm as lgb

    y_train = train_df[label_col].astype(int)
    counts = y_train.value_counts()
    if len(counts) < 2 or counts.min() < 5:
        _log(f"warning: skip {label_col}, insufficient positive/negative samples: {counts.to_dict()}")
        return None

    seed = int(config.get("seed", 42))
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=int(config.get("lgb", {}).get("n_jobs", 8)),
        scale_pos_weight=max(1.0, neg / max(pos, 1)),
        verbosity=-1,
    )
    eval_set = None
    if valid_df[label_col].nunique() == 2:
        eval_set = [(valid_df[feature_cols], valid_df[label_col].astype(int))]
    model.fit(
        train_df[feature_cols],
        y_train,
        eval_set=eval_set,
        callbacks=[lgb.log_evaluation(period=100)] if eval_set else None,
    )
    return model


def train_bad_models(train_df, valid_df, feature_cols):
    return {
        "bad_1pct": _fit_bad_model(train_df, valid_df, feature_cols, "bad_1pct"),
        "bad_2pct": _fit_bad_model(train_df, valid_df, feature_cols, "bad_2pct"),
    }


def _predict_bad(model, X):
    if model is None:
        return np.zeros(len(X), dtype=np.float64)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1].astype(np.float64)
    return np.asarray(model.predict(X), dtype=np.float64)


def _cfg_value(cfg, key, default):
    if cfg is None:
        return default
    return float(cfg.get(key, default))


def _apply_final_score(df, cfg):
    out = df.copy()
    out["final_score"] = (
        _group_zscore(out, "meta_raw_score")
        - _cfg_value(cfg, "lambda_bad_1pct", 0.25) * out["p_bad_1pct"]
        - _cfg_value(cfg, "lambda_bad_2pct", 0.45) * out["p_bad_2pct"]
        - _cfg_value(cfg, "lambda_uncertainty", 0.15) * out["uncertainty_score"]
        - _cfg_value(cfg, "lambda_disagreement", 0.10) * out["abs_score_gap"]
    )
    return out


def _daily_top5_metrics(df, score_col, prefix):
    rows = []
    for date, day in df.groupby("date", sort=True):
        top = day.sort_values(score_col, ascending=False).head(5)
        if top.empty:
            continue
        selected_score = float(top["target"].mean())
        rows.append(
            {
                "date": date,
                f"{prefix}_selected_score": selected_score,
                f"{prefix}_bad_count": int((top["target"] < -0.01).sum()),
                f"{prefix}_very_bad_count": int((top["target"] < -0.02).sum()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_daily(daily, prefix):
    scores = daily[f"{prefix}_selected_score"].astype(float)
    bad = daily[f"{prefix}_bad_count"].astype(float)
    very_bad = daily[f"{prefix}_very_bad_count"].astype(float)
    mean_score = float(scores.mean()) if len(scores) else 0.0
    std_score = float(scores.std(ddof=0)) if len(scores) else 0.0
    worst_score = float(scores.min()) if len(scores) else 0.0
    mean_bad = float(bad.mean()) if len(bad) else 0.0
    mean_very_bad = float(very_bad.mean()) if len(very_bad) else 0.0
    return {
        "mean_selected_score": mean_score,
        "median_selected_score": float(scores.median()) if len(scores) else 0.0,
        "q10_selected_score": float(scores.quantile(0.10)) if len(scores) else 0.0,
        "worst_selected_score": worst_score,
        "std_selected_score": std_score,
        "mean_bad_count": mean_bad,
        "mean_very_bad_count": mean_very_bad,
        "win_rate_positive": float((scores > 0.0).mean()) if len(scores) else 0.0,
        "robust_score": (
            mean_score
            - 0.50 * std_score
            - 1.00 * abs(min(0.0, worst_score))
            - 0.50 * mean_bad
            - 1.00 * mean_very_bad
        ),
        "n_dates": int(len(scores)),
    }


def evaluate_exp009(valid_df, feature_cols, ranker, bad_models, cfg):
    out = valid_df.copy()
    X = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["meta_raw_score"] = np.asarray(ranker.predict(X), dtype=np.float64)
    out["p_bad_1pct"] = _predict_bad(bad_models.get("bad_1pct"), X)
    out["p_bad_2pct"] = _predict_bad(bad_models.get("bad_2pct"), X)
    out = _apply_final_score(out, cfg)

    exp_daily = _daily_top5_metrics(out, "final_score", "exp009")
    ref_daily = _daily_top5_metrics(out, "score", "internal_reference")
    daily = exp_daily.merge(ref_daily, on="date", how="outer").sort_values("date")
    report = {
        "version": VERSION,
        "exp009": _summarize_daily(daily.dropna(subset=["exp009_selected_score"]), "exp009"),
        "internal_reference": _summarize_daily(
            daily.dropna(subset=["internal_reference_selected_score"]),
            "internal_reference",
        ),
    }
    return report, daily, out


def _make_meta_config(feature_cols):
    selector_cfg = config.get("selector", {}).get("exp009_meta", {})
    return {
        "version": VERSION,
        "feature_cols": list(feature_cols),
        "lambda_bad_1pct": float(selector_cfg.get("lambda_bad_1pct", 0.25)),
        "lambda_bad_2pct": float(selector_cfg.get("lambda_bad_2pct", 0.45)),
        "lambda_uncertainty": float(selector_cfg.get("lambda_uncertainty", 0.15)),
        "lambda_disagreement": float(selector_cfg.get("lambda_disagreement", 0.10)),
        "created_from": "train_oof_only",
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def _split_train_valid(df):
    dates = sorted(pd.to_datetime(df["date"]).dropna().unique())
    if len(dates) < 4:
        raise ValueError("OOF score 文件可用交易日太少，无法切分 train/valid。")
    valid_n = max(3, int(round(len(dates) * 0.20)))
    valid_n = min(valid_n, len(dates) - 1)
    valid_dates = set(dates[-valid_n:])
    train_df = df[~df["date"].isin(valid_dates)].copy()
    valid_df = df[df["date"].isin(valid_dates)].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError("OOF train/valid 切分为空，请检查 date 分布。")
    return train_df, valid_df


def save_artifacts(output_dir, report_dir, ranker, bad_models, feature_cols, meta_config, report, daily):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    joblib.dump(ranker, os.path.join(output_dir, "exp009_meta_ranker.pkl"))
    if bad_models.get("bad_1pct") is not None:
        joblib.dump(bad_models["bad_1pct"], os.path.join(output_dir, "exp009_bad_1pct.pkl"))
    if bad_models.get("bad_2pct") is not None:
        joblib.dump(bad_models["bad_2pct"], os.path.join(output_dir, "exp009_bad_2pct.pkl"))

    with open(os.path.join(output_dir, "exp009_feature_cols.json"), "w", encoding="utf-8") as f:
        json.dump(list(feature_cols), f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "exp009_meta_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta_config, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "exp009_validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    daily.to_csv(os.path.join(output_dir, "exp009_validation_daily.csv"), index=False)

    with open(os.path.join(report_dir, "exp009_validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    daily.to_csv(os.path.join(report_dir, "exp009_validation_daily.csv"), index=False)


def run(args):
    if not os.path.exists(args.oof_path):
        raise FileNotFoundError("缺少 OOF score 文件，请先运行 experiment_blend.py 或 OOF 构造脚本生成 train-only OOF 预测。")

    warnings.filterwarnings("ignore", category=UserWarning)
    raw = pd.read_csv(args.oof_path, dtype={"stock_id": str})
    raw = validate_oof_schema(raw, min_dates=args.min_dates)
    feat = build_exp009_features(raw)
    labeled = make_relevance_labels(feat)
    feature_cols = [col for col in EXP009_FEATURE_COLS if col in labeled.columns]
    labeled[feature_cols] = labeled[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    train_df, valid_df = _split_train_valid(labeled)
    _log(f"train_dates={train_df['date'].nunique()} valid_dates={valid_df['date'].nunique()} rows={len(labeled)}")
    ranker = train_meta_ranker(train_df, valid_df, feature_cols)
    bad_models = train_bad_models(train_df, valid_df, feature_cols)
    meta_config = _make_meta_config(feature_cols)
    report, daily, scored_valid = evaluate_exp009(valid_df, feature_cols, ranker, bad_models, meta_config)
    save_artifacts(args.output_dir, args.report_dir, ranker, bad_models, feature_cols, meta_config, report, daily)
    if args.save_debug:
        os.makedirs(args.report_dir, exist_ok=True)
        scored_valid.to_csv(os.path.join(args.report_dir, "exp009_valid_scored_debug.csv"), index=False)
    _log(f"saved artifacts to {args.output_dir}")
    _log(f"validation_report={os.path.join(args.output_dir, 'exp009_validation_report.json')}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Train exp009 OOF meta reranker and bad-pick filters.")
    parser.add_argument("--oof-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-dir", default="./temp/exp009")
    parser.add_argument("--min-dates", type=int, default=20)
    parser.add_argument("--save-debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
