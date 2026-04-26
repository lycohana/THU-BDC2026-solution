import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import config
from labels import add_label_o2o_week, normalize_stock_id
from lgb_branch import (
    LGB_FEATURES_FILE,
    LGB_RANKER_FILE,
    LGB_REGRESSOR_FILE,
    LGB_TOP5_RANKER_FILE,
    fit_lgb_branches,
    load_lgb_branches,
    predict_lgb_components,
)
from train import preprocess_data, preprocess_val_data, set_seed


LOG_PREFIX = "[BDC][exp009-oof]"
HORIZON = 5
RISK_COLS = [
    "sigma20",
    "median_amount20",
    "ret1",
    "ret5",
    "ret20",
    "amp20",
    "beta60",
    "downside_beta60",
    "max_drawdown20",
]
REQUIRED_OUTPUT_COLS = [
    "date",
    "stock_id",
    "target",
    "forward_open_return",
    "transformer",
    "lgb",
    "score",
    "lgb_rank_score",
    "lgb_reg_score",
    "lgb_top5_rank_score",
    "lgb_top5_score",
    *RISK_COLS,
]


@dataclass
class FoldSpec:
    fold: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str


def _log(message):
    print(f"{LOG_PREFIX} {message}", flush=True)


def _zscore(values):
    values = np.asarray(values, dtype=np.float64)
    std = values.std()
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - values.mean()) / (std + 1e-9)


def _group_zscore(df, col):
    return (
        df.groupby("date", sort=False)[col]
        .transform(lambda x: pd.Series(_zscore(x.to_numpy(dtype=np.float64)), index=x.index))
        .astype(np.float64)
    )


def _trading_on_or_before(dates, cutoff):
    eligible = dates[dates <= pd.Timestamp(cutoff)]
    return pd.Timestamp(eligible[-1]) if len(eligible) else None


def _previous_trading_day(dates, date):
    eligible = dates[dates < pd.Timestamp(date)]
    return pd.Timestamp(eligible[-1]) if len(eligible) else None


def build_fold_specs(raw_dates, n_folds, fold_window_months, gap_months, purge_trading_days=HORIZON):
    raw_dates = pd.Index(sorted(pd.to_datetime(raw_dates).dropna().unique()))
    if len(raw_dates) <= HORIZON + 1:
        raise ValueError("交易日数量不足，无法构造 OOF fold。")

    labelable_dates = pd.Index(raw_dates[:-HORIZON])
    cursor_end = pd.Timestamp(labelable_dates[-1])
    specs_newest_first = []

    for _ in range(int(n_folds)):
        val_start_cut = cursor_end - pd.DateOffset(months=int(fold_window_months))
        val_start = _trading_on_or_before(labelable_dates, val_start_cut)
        if val_start is None:
            break
        val_dates = labelable_dates[(labelable_dates >= val_start) & (labelable_dates <= cursor_end)]
        if len(val_dates) == 0:
            break

        train_end_cut = pd.Timestamp(val_start) - pd.DateOffset(months=int(gap_months))
        train_end_by_embargo = _trading_on_or_before(raw_dates, train_end_cut)
        if train_end_by_embargo is None:
            break
        val_start_pos = list(raw_dates).index(pd.Timestamp(val_start))
        purge_pos = max(0, val_start_pos - int(purge_trading_days))
        train_end_by_purge = pd.Timestamp(raw_dates[purge_pos])
        train_end = min(pd.Timestamp(train_end_by_embargo), train_end_by_purge)

        train_dates = raw_dates[raw_dates <= train_end]
        if len(train_dates) <= max(30, config["sequence_length"] + HORIZON):
            _log(
                "stop building older folds: "
                f"train_days={len(train_dates)} train_end={train_end:%Y-%m-%d}"
            )
            break

        specs_newest_first.append(
            FoldSpec(
                fold=-1,
                train_start=f"{pd.Timestamp(train_dates[0]):%Y-%m-%d}",
                train_end=f"{train_end:%Y-%m-%d}",
                val_start=f"{pd.Timestamp(val_dates[0]):%Y-%m-%d}",
                val_end=f"{pd.Timestamp(val_dates[-1]):%Y-%m-%d}",
            )
        )
        prev_end = _previous_trading_day(labelable_dates, val_dates[0])
        if prev_end is None:
            break
        cursor_end = prev_end

    specs = list(reversed(specs_newest_first))
    if len(specs) != int(n_folds):
        raise ValueError(f"只构造出 {len(specs)} 个 fold，少于请求的 n_folds={n_folds}。")
    for idx, spec in enumerate(specs):
        spec.fold = idx
    return specs


def _clean_raw(raw):
    out = raw.copy()
    out["股票代码"] = out["股票代码"].map(normalize_stock_id)
    out["日期"] = pd.to_datetime(out["日期"])
    out = out.sort_values(["日期", "股票代码"]).reset_index(drop=True)
    return out


def build_label_frame(raw):
    labeled = add_label_o2o_week(raw, horizon=HORIZON, label_col="target")
    out = labeled[["日期", "股票代码", "target"]].copy()
    out = out.rename(columns={"日期": "date", "股票代码": "stock_id"})
    out["date"] = pd.to_datetime(out["date"])
    out["stock_id"] = out["stock_id"].map(normalize_stock_id)
    out["forward_open_return"] = pd.to_numeric(out["target"], errors="coerce")
    return out


def _rolling_max_drawdown(close):
    values = np.asarray(close, dtype=np.float64)
    if values.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(values)
    drawdown = values / (running_max + 1e-12) - 1.0
    return float(abs(np.nanmin(drawdown)))


def _add_stock_risk(group):
    group = group.sort_values("日期").copy()
    close = pd.to_numeric(group["收盘"], errors="coerce")
    high = pd.to_numeric(group["最高"], errors="coerce")
    low = pd.to_numeric(group["最低"], errors="coerce")
    amount = pd.to_numeric(group["成交额"], errors="coerce")
    market = pd.to_numeric(group["market_ret1"], errors="coerce")

    group["ret1"] = close.pct_change(fill_method=None)
    group["ret5"] = close / (close.shift(5) + 1e-12) - 1.0
    group["ret20"] = close / (close.shift(20) + 1e-12) - 1.0
    group["sigma20"] = group["ret1"].rolling(20, min_periods=5).std()
    group["median_amount20"] = amount.rolling(20, min_periods=5).median()
    group["amp20"] = (high.rolling(20, min_periods=5).max() - low.rolling(20, min_periods=5).min()) / (
        close.abs() + 1e-12
    )

    cov = group["ret1"].rolling(60, min_periods=20).cov(market)
    var = market.rolling(60, min_periods=20).var()
    beta = cov / (var + 1e-12)
    down_x = market.where(market < 0.0)
    down_y = group["ret1"].where(market < 0.0)
    down_cov = down_y.rolling(60, min_periods=8).cov(down_x)
    down_var = down_x.rolling(60, min_periods=8).var()
    group["beta60"] = beta
    group["downside_beta60"] = (down_cov / (down_var + 1e-12)).fillna(beta)
    group["max_drawdown20"] = close.rolling(20, min_periods=5).apply(_rolling_max_drawdown, raw=False)
    return group


def build_risk_frame(raw):
    data = raw.copy()
    for col in ["收盘", "最高", "最低", "成交额"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.sort_values(["股票代码", "日期"]).reset_index(drop=True)
    data["ret1_tmp"] = data.groupby("股票代码", sort=False)["收盘"].pct_change(fill_method=None)
    data["market_ret1"] = data.groupby("日期", sort=False)["ret1_tmp"].transform("mean")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message="DataFrameGroupBy.apply operated.*")
        risk = data.groupby("股票代码", group_keys=False, sort=False).apply(_add_stock_risk)
    risk = risk.rename(columns={"日期": "date", "股票代码": "stock_id"})
    risk["stock_id"] = risk["stock_id"].map(normalize_stock_id)
    risk["date"] = pd.to_datetime(risk["date"])
    for col in RISK_COLS:
        risk[col] = pd.to_numeric(risk[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return risk[["date", "stock_id", *RISK_COLS]]


def _artifact_ready(fold_dir, train_transformer):
    required = [
        "scaler.pkl",
        "stockid2idx.json",
        LGB_FEATURES_FILE,
        LGB_RANKER_FILE,
        LGB_REGRESSOR_FILE,
    ]
    if train_transformer:
        required.append("best_model.pth")
    return all(os.path.exists(os.path.join(fold_dir, name)) for name in required)


def _load_stock_mapping(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): int(v) for k, v in raw.items()}


def _save_stock_mapping(path, mapping):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def _eligible_stock_ids(frame, min_history):
    counts = frame.groupby("股票代码", sort=False)["日期"].nunique()
    return set(counts[counts >= int(min_history)].index.map(normalize_stock_id))


def _ensure_feature_columns(df, feature_cols):
    out = df.copy()
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0.0
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _fit_and_apply_scaler(train_data, val_data, feature_cols, fold_dir):
    scaler = StandardScaler()
    train_data = _ensure_feature_columns(train_data, feature_cols)
    val_data = _ensure_feature_columns(val_data, feature_cols)
    train_data = train_data.dropna(subset=feature_cols).copy()
    val_data = val_data.dropna(subset=feature_cols).copy()
    train_data[feature_cols] = train_data[feature_cols].astype(np.float64)
    val_data[feature_cols] = val_data[feature_cols].astype(np.float64)
    train_data.loc[:, feature_cols] = pd.DataFrame(
        scaler.fit_transform(train_data[feature_cols]),
        columns=feature_cols,
        index=train_data.index,
    )
    val_data.loc[:, feature_cols] = pd.DataFrame(
        scaler.transform(val_data[feature_cols]),
        columns=feature_cols,
        index=val_data.index,
    )
    joblib.dump(scaler, os.path.join(fold_dir, "scaler.pkl"))
    return train_data, val_data


def _apply_existing_scaler(data, feature_cols, fold_dir):
    scaler = joblib.load(os.path.join(fold_dir, "scaler.pkl"))
    scaler_features = list(getattr(scaler, "feature_names_in_", feature_cols))
    data = _ensure_feature_columns(data, scaler_features)
    data = data.dropna(subset=scaler_features).copy()
    data[scaler_features] = data[scaler_features].astype(np.float64)
    data.loc[:, scaler_features] = pd.DataFrame(
        scaler.transform(data[scaler_features]),
        columns=scaler_features,
        index=data.index,
    )
    return data


def _split_lgb_train_valid(train_data):
    dates = sorted(pd.to_datetime(train_data["日期"]).dropna().unique())
    if len(dates) < 10:
        raise ValueError("fold 内训练日期太少，无法为 LGB early stopping 切分验证集。")
    valid_n = max(5, int(round(len(dates) * 0.15)))
    valid_n = min(valid_n, len(dates) - 1)
    valid_dates = set(dates[-valid_n:])
    lgb_train = train_data[~train_data["日期"].isin(valid_dates)].copy()
    lgb_valid = train_data[train_data["日期"].isin(valid_dates)].copy()
    if lgb_train.empty or lgb_valid.empty:
        raise ValueError("fold 内 LGB train/valid 切分为空。")
    return lgb_train, lgb_valid


def _predict_lgb_oof(fold_dir, val_data):
    bundle = load_lgb_branches(fold_dir)
    if bundle is None:
        raise FileNotFoundError(f"fold LGB artifact 不完整: {fold_dir}")
    val_data = _ensure_feature_columns(val_data, bundle["features"])
    components = predict_lgb_components(bundle, val_data)
    if components is None:
        raise RuntimeError("LGB components prediction returned None")

    out = val_data[["日期", "股票代码", "label"]].copy()
    out = out.rename(columns={"日期": "date", "股票代码": "stock_id", "label": "target"})
    out["date"] = pd.to_datetime(out["date"])
    out["stock_id"] = out["stock_id"].map(normalize_stock_id)
    out["lgb_rank_score"] = components["rank_score"]
    out["lgb_reg_score"] = components["reg_score"]
    out["lgb_top5_rank_score"] = components.get("top5_rank_score", np.zeros(len(out), dtype=np.float64))
    out["lgb_top5_score"] = out["lgb_top5_rank_score"]

    lgb_cfg = config.get("lgb", {})
    rank_w = float(lgb_cfg.get("rank_weight", 0.65))
    reg_w = float(lgb_cfg.get("reg_weight", 0.35))
    top5_w = float(lgb_cfg.get("top5_rank_weight", config.get("lgb_top5", {}).get("blend_weight", 0.0)))

    chunks = []
    for _, day in out.groupby("date", sort=False):
        rank_z = _zscore(day["lgb_rank_score"].to_numpy(dtype=np.float64))
        reg_z = _zscore(day["lgb_reg_score"].to_numpy(dtype=np.float64))
        lgb_score = rank_w * rank_z + reg_w * reg_z
        if top5_w > 0:
            lgb_score = lgb_score + top5_w * _zscore(day["lgb_top5_rank_score"].to_numpy(dtype=np.float64))
        day = day.copy()
        day["lgb"] = lgb_score
        chunks.append(day)
    out = pd.concat(chunks, ignore_index=True)
    out["transformer"] = 0.0
    out["score"] = _group_zscore(out, "lgb")
    out["forward_open_return"] = pd.to_numeric(out["target"], errors="coerce")
    return out


def _train_fold(raw, spec, fold_dir, train_transformer):
    if train_transformer:
        raise NotImplementedError("第一版 exp009_oof_builder 仅支持 --train-transformer 0。")

    os.makedirs(fold_dir, exist_ok=True)
    train_end = pd.Timestamp(spec.train_end)
    val_end = pd.Timestamp(spec.val_end)
    val_label_end = raw.loc[raw["日期"] <= val_end, "日期"].max()
    label_end_idx = list(sorted(raw["日期"].unique())).index(val_end) + HORIZON
    all_dates = pd.Index(sorted(raw["日期"].unique()))
    val_context_end = pd.Timestamp(all_dates[min(label_end_idx, len(all_dates) - 1)])

    train_raw = raw[raw["日期"] <= train_end].copy()
    eligible_ids = _eligible_stock_ids(train_raw, config["sequence_length"])
    train_raw = train_raw[train_raw["股票代码"].isin(eligible_ids)].copy()
    val_raw = raw[raw["日期"] <= val_context_end].copy()
    stock_ids = sorted(eligible_ids)
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    _save_stock_mapping(os.path.join(fold_dir, "stockid2idx.json"), stockid2idx)
    val_raw = val_raw[val_raw["股票代码"].isin(stockid2idx)].copy()

    _log(
        f"fold_{spec.fold:02d} train preprocess raw_rows={len(train_raw)} "
        f"train_end={spec.train_end}"
    )
    train_data, feature_cols = preprocess_data(train_raw, is_train=True, stockid2idx=stockid2idx)

    _log(
        f"fold_{spec.fold:02d} val preprocess raw_rows={len(val_raw)} "
        f"val={spec.val_start}~{spec.val_end}"
    )
    val_data, _ = preprocess_val_data(val_raw, stockid2idx=stockid2idx)
    val_mask = (pd.to_datetime(val_data["日期"]) >= pd.Timestamp(spec.val_start)) & (
        pd.to_datetime(val_data["日期"]) <= pd.Timestamp(spec.val_end)
    )
    val_data = val_data[val_mask].copy()

    train_data, val_data = _fit_and_apply_scaler(train_data, val_data, feature_cols, fold_dir)
    lgb_train, lgb_valid = _split_lgb_train_valid(train_data)
    _log(
        f"fold_{spec.fold:02d} train LGB rows={len(lgb_train)} "
        f"early_stop_rows={len(lgb_valid)} val_rows={len(val_data)}"
    )
    fit_lgb_branches(lgb_train, lgb_valid, feature_cols, fold_dir, config)
    return val_data


def _load_or_build_val_data(raw, spec, fold_dir, train_transformer):
    if train_transformer:
        raise NotImplementedError("第一版 exp009_oof_builder 仅支持 --train-transformer 0。")

    stockid2idx = _load_stock_mapping(os.path.join(fold_dir, "stockid2idx.json"))
    all_dates = pd.Index(sorted(raw["日期"].unique()))
    val_end = pd.Timestamp(spec.val_end)
    label_end_idx = list(all_dates).index(val_end) + HORIZON
    val_context_end = pd.Timestamp(all_dates[min(label_end_idx, len(all_dates) - 1)])
    val_raw = raw[raw["日期"] <= val_context_end].copy()
    val_raw = val_raw[val_raw["股票代码"].isin(stockid2idx)].copy()

    _log(f"fold_{spec.fold:02d} reuse artifacts; preprocess validation only")
    val_data, _ = preprocess_val_data(val_raw, stockid2idx=stockid2idx)
    val_mask = (pd.to_datetime(val_data["日期"]) >= pd.Timestamp(spec.val_start)) & (
        pd.to_datetime(val_data["日期"]) <= pd.Timestamp(spec.val_end)
    )
    val_data = val_data[val_mask].copy()

    bundle = load_lgb_branches(fold_dir)
    if bundle is None:
        raise FileNotFoundError(f"fold LGB artifact 不完整: {fold_dir}")
    return _apply_existing_scaler(val_data, bundle["features"], fold_dir)


def build_oof(args):
    set_seed(config.get("seed", 42))
    if args.train_transformer:
        raise NotImplementedError("第一版优先实现 LGB-only true OOF，请使用 --train-transformer 0。")

    data_path = Path(config.get("data_path", "./data")) / "train.csv"
    _log(f"load train data from {data_path}")
    raw = pd.read_csv(data_path, dtype={"股票代码": str})
    raw = _clean_raw(raw)

    specs = build_fold_specs(
        raw["日期"],
        n_folds=args.n_folds,
        fold_window_months=args.fold_window_months,
        gap_months=args.gap_months,
        purge_trading_days=args.purge_trading_days,
    )
    for spec in specs:
        _log(
            f"fold_{spec.fold:02d} train={spec.train_start}~{spec.train_end} "
            f"val={spec.val_start}~{spec.val_end}"
        )

    risk = build_risk_frame(raw)
    all_oof = []
    report_folds = []
    os.makedirs(args.fold_model_root, exist_ok=True)

    for spec in specs:
        fold_dir = os.path.join(args.fold_model_root, f"fold_{spec.fold:02d}")
        can_reuse = bool(args.reuse_existing_fold_models) and _artifact_ready(fold_dir, args.train_transformer)
        if can_reuse:
            val_data = _load_or_build_val_data(raw, spec, fold_dir, args.train_transformer)
        else:
            _log(f"fold_{spec.fold:02d} artifacts missing or reuse disabled; train fold")
            val_data = _train_fold(raw, spec, fold_dir, args.train_transformer)

        oof = _predict_lgb_oof(fold_dir, val_data)
        oof["fold"] = spec.fold
        oof = oof.merge(risk, on=["date", "stock_id"], how="left")
        for col in RISK_COLS:
            oof[col] = pd.to_numeric(oof[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        all_oof.append(oof)

        report_folds.append(
            {
                **asdict(spec),
                "rows": int(len(oof)),
                "stocks": int(oof["stock_id"].nunique()),
                "dates": int(oof["date"].nunique()),
            }
        )
        _log(
            f"fold_{spec.fold:02d} predicted rows={len(oof)} "
            f"dates={oof['date'].nunique()} stocks={oof['stock_id'].nunique()}"
        )

    out = pd.concat(all_oof, ignore_index=True)
    out = out.sort_values(["date", "stock_id"]).reset_index(drop=True)
    for col in REQUIRED_OUTPUT_COLS:
        if col not in out.columns:
            out[col] = 0.0
    out = out[REQUIRED_OUTPUT_COLS + [col for col in ["fold"] if col in out.columns]]
    validate_output_schema(out)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    report = {
        "version": "exp009_true_lgb_oof_v1",
        "train_transformer": int(args.train_transformer),
        "n_folds": int(args.n_folds),
        "fold_window_months": int(args.fold_window_months),
        "gap_months": int(args.gap_months),
        "purge_trading_days": int(args.purge_trading_days),
        "output": args.output,
        "fold_model_root": args.fold_model_root,
        "rows": int(len(out)),
        "stocks": int(out["stock_id"].nunique()),
        "dates": int(out["date"].nunique()),
        "folds": report_folds,
    }
    report_path = "./temp/exp009_oof_builder_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _log(f"saved OOF to {args.output}")
    _log(f"saved report to {report_path}")
    return out, report


def validate_output_schema(df):
    missing = [col for col in REQUIRED_OUTPUT_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"OOF 输出缺少列: {missing}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("OOF 输出 date 存在无法解析的值。")

    n_dates = int(out["date"].nunique())
    if n_dates < 20:
        raise ValueError(f"OOF 输出交易日不足: {n_dates} < 20")

    stocks_by_date = out.groupby("date")["stock_id"].nunique()
    bad_dates = stocks_by_date[stocks_by_date < 20]
    if len(bad_dates):
        raise ValueError(f"OOF 输出存在每日至少 20 只股票检查失败的日期数: {len(bad_dates)}")

    target_ok = pd.to_numeric(out["target"], errors="coerce").notna().mean()
    score_ok = pd.to_numeric(out["score"], errors="coerce").notna().mean()
    if target_ok <= 0.90:
        raise ValueError(f"target 非空比例过低: {target_ok:.4f}")
    if score_ok <= 0.90:
        raise ValueError(f"score 非空比例过低: {score_ok:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build strict train-only true OOF scores for exp009.")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--fold-window-months", type=int, default=2)
    parser.add_argument("--gap-months", type=int, default=1)
    parser.add_argument("--purge-trading-days", type=int, default=HORIZON)
    parser.add_argument("--output", default="./temp/exp009_oof_scores.csv")
    parser.add_argument("--fold-model-root", default="./temp/exp009_fold_models")
    parser.add_argument("--reuse-existing-fold-models", type=int, default=1)
    parser.add_argument("--train-transformer", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    build_oof(parse_args())
