import numpy as np
import pandas as pd

from labels import add_label_o2o_week, normalize_stock_id


def _rank_pct(series):
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill = values.median() if values.notna().any() else 0.0
    return values.fillna(fill).rank(pct=True, method="average")


def _ensure_stock_date(df):
    out = df.copy()
    if "stock_id" not in out.columns:
        stock_col = "股票代码" if "股票代码" in out.columns else None
        if stock_col is None:
            raise ValueError("DataFrame must contain stock_id or 股票代码")
        out["stock_id"] = out[stock_col].map(normalize_stock_id)
    else:
        out["stock_id"] = out["stock_id"].map(normalize_stock_id)
    if "日期" in out.columns:
        out["日期"] = pd.to_datetime(out["日期"])
    elif "date" in out.columns:
        out["日期"] = pd.to_datetime(out["date"])
    return out


def _feature_pct_frame(feature_df):
    feat = _ensure_stock_date(feature_df)
    aliases = {
        "amp10_source": ["amp10_pct", "amp_mean10", "amp10"],
        "vol10_source": ["vol10_pct", "vol10"],
        "turnover20_source": ["turnover20_pct", "turnover20"],
        "amount20_source": ["amount20_pct", "mean_amount20", "amount20", "median_amount20"],
    }
    for target, candidates in aliases.items():
        found = next((col for col in candidates if col in feat.columns), None)
        feat[target] = pd.to_numeric(feat[found], errors="coerce") if found else 0.0

    if "日期" in feat.columns:
        group = feat.groupby("日期")
        feat["amp10_pct_profile"] = group["amp10_source"].transform(lambda s: _rank_pct(s))
        feat["vol10_pct_profile"] = group["vol10_source"].transform(lambda s: _rank_pct(s))
        feat["turnover20_pct_profile"] = group["turnover20_source"].transform(lambda s: _rank_pct(s))
        feat["amount20_pct_profile"] = group["amount20_source"].transform(lambda s: _rank_pct(s))
    else:
        feat["amp10_pct_profile"] = _rank_pct(feat["amp10_source"])
        feat["vol10_pct_profile"] = _rank_pct(feat["vol10_source"])
        feat["turnover20_pct_profile"] = _rank_pct(feat["turnover20_source"])
        feat["amount20_pct_profile"] = _rank_pct(feat["amount20_source"])
    return feat


def build_stock_upside_profile(train_df, feature_df):
    """Build historical upside profile using scorer-equivalent weekly labels."""
    labels = _ensure_stock_date(train_df)
    if "label_o2o_week" not in labels.columns:
        labels = add_label_o2o_week(labels)
        labels = _ensure_stock_date(labels)
    labels = labels.dropna(subset=["label_o2o_week"]).copy()
    labels["future_rank"] = labels.groupby("日期")["label_o2o_week"].rank(method="first", ascending=False)
    labels["is_top20"] = (labels["future_rank"] <= 20).astype(float)
    labels["is_top10"] = (labels["future_rank"] <= 10).astype(float)
    labels["is_top5"] = (labels["future_rank"] <= 5).astype(float)

    def q(x, p):
        return float(pd.Series(x).quantile(p)) if len(x) else np.nan

    ret_profile = (
        labels.groupby("stock_id")
        .agg(
            week_return_mean=("label_o2o_week", "mean"),
            week_return_median=("label_o2o_week", "median"),
            week_return_q80=("label_o2o_week", lambda s: q(s, 0.80)),
            week_return_q90=("label_o2o_week", lambda s: q(s, 0.90)),
            week_return_q95=("label_o2o_week", lambda s: q(s, 0.95)),
            week_return_max=("label_o2o_week", "max"),
            week_return_std=("label_o2o_week", "std"),
            top20_hit_rate=("is_top20", "mean"),
            top10_hit_rate=("is_top10", "mean"),
            top5_hit_rate=("is_top5", "mean"),
            profile_obs=("label_o2o_week", "count"),
        )
        .reset_index()
    )

    feat = _feature_pct_frame(feature_df)
    feat_profile = (
        feat.groupby("stock_id")
        .agg(
            avg_amp10_pct=("amp10_pct_profile", "mean"),
            avg_vol10_pct=("vol10_pct_profile", "mean"),
            avg_turnover20_pct=("turnover20_pct_profile", "mean"),
            avg_amount20_pct=("amount20_pct_profile", "mean"),
        )
        .reset_index()
    )
    profile = ret_profile.merge(feat_profile, on="stock_id", how="left")
    for col in ["week_return_std", "avg_amp10_pct", "avg_vol10_pct", "avg_turnover20_pct", "avg_amount20_pct"]:
        profile[col] = pd.to_numeric(profile[col], errors="coerce").fillna(0.0)
    return profile


def add_low_upside_flags(profile_df, latest_feature_df):
    profile = profile_df.copy()
    profile["stock_id"] = profile["stock_id"].map(normalize_stock_id)
    latest = _ensure_stock_date(latest_feature_df)

    for col in ["week_return_q90", "week_return_q95", "top20_hit_rate", "top10_hit_rate", "avg_amp10_pct", "avg_vol10_pct"]:
        profile[col] = pd.to_numeric(profile[col], errors="coerce").fillna(0.0)

    profile["q90_week_return_pct"] = _rank_pct(profile["week_return_q90"])
    profile["q95_week_return_pct"] = _rank_pct(profile["week_return_q95"])
    profile["top20_hit_rate_pct"] = _rank_pct(profile["top20_hit_rate"])
    profile["top10_hit_rate_pct"] = _rank_pct(profile["top10_hit_rate"])
    profile["avg_amp10_pct_rank"] = _rank_pct(profile["avg_amp10_pct"])
    profile["avg_vol10_pct_rank"] = _rank_pct(profile["avg_vol10_pct"])

    latest_cols = ["stock_id"]
    for col in ["ret20_pct", "pos20_pct"]:
        if col not in latest.columns:
            source = col.replace("_pct", "")
            latest[col] = _rank_pct(latest[source]) if source in latest.columns else 0.5
        latest_cols.append(col)
    latest = latest[latest_cols].rename(columns={"ret20_pct": "recent_ret20_pct", "pos20_pct": "recent_pos20_pct"})
    out = profile.merge(latest, on="stock_id", how="left")
    out["recent_ret20_pct"] = pd.to_numeric(out["recent_ret20_pct"], errors="coerce").fillna(0.5)
    out["recent_pos20_pct"] = pd.to_numeric(out["recent_pos20_pct"], errors="coerce").fillna(0.5)

    out["low_upside_drag"] = (
        (out["q90_week_return_pct"] <= 0.35)
        & (out["top20_hit_rate_pct"] <= 0.35)
        & (out["avg_amp10_pct_rank"] <= 0.45)
        & (out["avg_vol10_pct_rank"] <= 0.45)
        & (out["recent_ret20_pct"] < 0.70)
        & (out["recent_pos20_pct"] < 0.75)
    )
    out["very_low_upside_drag"] = (
        (out["q95_week_return_pct"] <= 0.25)
        & (out["top20_hit_rate_pct"] <= 0.25)
        & (out["avg_amp10_pct_rank"] <= 0.35)
        & (out["avg_vol10_pct_rank"] <= 0.35)
        & (out["recent_ret20_pct"] < 0.60)
        & (out["recent_pos20_pct"] < 0.65)
    )
    return out
