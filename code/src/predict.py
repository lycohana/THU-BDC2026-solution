import hashlib
import itertools
import json
import os
import multiprocessing as mp
from contextlib import nullcontext

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from exp009_runtime import apply_exp009_meta, load_exp009_artifacts
from features import build_history_feature_frame
from feature_registry import finalize_feature_frame, get_feature_columns, get_feature_engineer
from lgb_branch import load_lgb_branches, predict_lgb_components, predict_lgb_score
from model import StockTransformer
from portfolio_utils import apply_supplemental_overlay, build_weight_portfolio, select_candidates as shared_select_candidates
from reranker import apply_grr_top5


PREDICT_CACHE_VERSION = 4


def _file_fingerprint(path):
    stat = os.stat(path)
    return {
        'path': os.path.abspath(path),
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
    }


def _build_predict_cache_path(data_file, scaler_path):
    predict_cfg = config.get('predict', {})
    if os.environ.get('BDC_DISABLE_PREDICT_CACHE') == '1':
        return None
    if not predict_cfg.get('use_cache', True):
        return None

    cache_dir = predict_cfg.get('cache_dir', './temp')
    os.makedirs(cache_dir, exist_ok=True)

    cache_key = {
        'version': PREDICT_CACHE_VERSION,
        'data_file': _file_fingerprint(data_file),
        'scaler_file': _file_fingerprint(scaler_path),
        'feature_num': config['feature_num'],
        'sequence_length': config['sequence_length'],
        'feature_enhance': config.get('feature_enhance', {}),
    }
    digest = hashlib.md5(
        json.dumps(cache_key, sort_keys=True, ensure_ascii=False).encode('utf-8')
    ).hexdigest()[:16]
    return os.path.join(cache_dir, f'predict_artifacts_{digest}.pkl')


def _resolve_model_dir():
    override_dir = os.environ.get('BDC_MODEL_DIR')
    if override_dir:
        return override_dir

    preferred_dir = config['output_dir']
    required_files = ['best_model.pth', 'scaler.pkl']
    if all(os.path.exists(os.path.join(preferred_dir, name)) for name in required_files):
        return preferred_dir

    legacy_dir = os.path.join('./model', f"{config['sequence_length']}_{config['feature_num']}")
    if all(os.path.exists(os.path.join(legacy_dir, name)) for name in required_files):
        print(f'[BDC][predict] fallback_model_dir={legacy_dir}')
        return legacy_dir

    return preferred_dir


def _resolve_feature_workers(num_groups):
    if num_groups <= 0:
        return 1

    predict_cfg = config.get('predict', {})
    requested = int(predict_cfg.get('feature_workers', min(6, mp.cpu_count())))
    requested = max(1, requested)

    # 外部环境变量显式限制 worker 数（批处理等并发场景设置以避免 Windows 资源耗尽）
    env_cap = os.environ.get("BDC_FEATURE_WORKERS")
    if env_cap is not None:
        try:
            requested = min(requested, int(env_cap))
        except ValueError:
            pass

    # Windows 下多进程启动成本较高，小规模任务时主动收缩 worker 数。
    if num_groups < 64:
        requested = min(requested, 4)

    return max(1, min(requested, num_groups, mp.cpu_count()))


def _sanitize_feature_frame(df, features):
    cleaned = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df.loc[:, features] = cleaned
    return df


def _align_features_to_scaler(processed, feature_columns, scaler):
    scaler_features = getattr(scaler, 'feature_names_in_', None)
    if scaler_features is None:
        return processed, list(feature_columns)

    scaler_features = list(scaler_features)
    missing_features = [col for col in scaler_features if col not in processed.columns]
    if missing_features:
        print(f'[BDC][predict] add_missing_features_for_scaler={len(missing_features)}')
        for col in missing_features:
            processed[col] = 0.0

    return processed, scaler_features


def preprocess_predict_data(df, stockid2idx):
    feature_engineer = get_feature_engineer(config['feature_num'])
    feature_columns = get_feature_columns(config['feature_num'])

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError('输入数据为空，无法预测')

    num_processes = _resolve_feature_workers(len(groups))
    print(f'[BDC][predict] cpu_count={mp.cpu_count()}, feature_workers={num_processes}')

    if num_processes == 1:
        processed_list = [
            feature_engineer(group)
            for group in tqdm(groups, total=len(groups), desc='预测集特征工程')
        ]
    else:
        chunksize = max(1, len(groups) // (num_processes * 4))
        with mp.Pool(processes=num_processes) as pool:
            processed_list = list(
                tqdm(
                    pool.imap(feature_engineer, groups, chunksize=chunksize),
                    total=len(groups),
                    desc='预测集特征工程',
                )
            )

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])

    enhance_cfg = config.get('feature_enhance', {})
    if any(enhance_cfg.values()):
        print(f'[BDC][feature_enhance] 启用预测特征增强：{enhance_cfg}')
    processed, feature_columns = finalize_feature_frame(
        processed,
        feature_columns,
        enhance_cfg=enhance_cfg,
        date_col='日期',
    )

    return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, latest_date):
    tail_df = (
        data[data['日期'] <= latest_date]
        .groupby('股票代码', sort=False)
        .tail(sequence_length)
        .copy()
    )
    if tail_df.empty:
        raise ValueError('没有可用于预测的股票序列，请检查数据与 sequence_length')

    tail_df['_seq_len'] = tail_df.groupby('股票代码', sort=False)['股票代码'].transform('size')
    tail_df = tail_df[tail_df['_seq_len'] == sequence_length].drop(columns=['_seq_len'])
    tail_df = tail_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    if tail_df.empty:
        raise ValueError('没有满足 sequence_length 的股票序列')

    sequence_stock_ids = (
        tail_df.groupby('股票代码', sort=False)
        .head(1)['股票代码']
        .tolist()
    )
    feature_matrix = tail_df[features].to_numpy(dtype=np.float32, copy=False)
    sequences = np.ascontiguousarray(
        feature_matrix.reshape(len(sequence_stock_ids), sequence_length, len(features))
    )
    return sequences, sequence_stock_ids


def build_risk_frame(raw_df):
    return build_history_feature_frame(raw_df)

    hist = raw_df.groupby('股票代码', sort=False).tail(80).copy()
    if hist.empty:
        return pd.DataFrame(columns=[
            'stock_id',
            'sigma20',
            'median_amount20',
            'ret1',
            'ret5',
            'ret20',
            'amp20',
            'beta60',
            'downside_beta60',
            'idio_vol60',
            'max_drawdown20',
        ])

    hist = hist.sort_values(['日期', '股票代码']).copy()
    hist['收盘'] = hist['收盘'].astype(float)
    hist['最高'] = hist['最高'].astype(float)
    hist['最低'] = hist['最低'].astype(float)
    hist['成交额'] = hist['成交额'].astype(float)
    if '换手率' in hist.columns:
        hist['换手率'] = hist['换手率'].astype(float)
    else:
        hist['换手率'] = 0.0

    hist['ret1'] = hist.groupby('股票代码', sort=False)['收盘'].pct_change(fill_method=None)
    hist['close_lag5'] = hist.groupby('股票代码', sort=False)['收盘'].shift(5)
    hist['close_lag10'] = hist.groupby('股票代码', sort=False)['收盘'].shift(10)
    hist['market_ret1'] = hist.groupby('日期', sort=False)['ret1'].transform('mean')
    hist['daily_amp'] = (hist['最高'] - hist['最低']) / (hist['收盘'].abs() + 1e-12)

    grouped = hist.groupby('股票代码', sort=False)
    recent20 = grouped.tail(21).copy()
    grouped20 = recent20.groupby('股票代码', sort=False)
    agg = grouped20.agg(
        sigma20=('ret1', 'std'),
        median_amount20=('成交额', 'median'),
        mean_amount20=('成交额', 'mean'),
        turnover20=('换手率', 'mean'),
        first_close=('收盘', 'first'),
        max_high=('最高', 'max'),
        min_low=('最低', 'min'),
        row_count=('收盘', 'size'),
    )
    recent10 = grouped.tail(10).copy()
    grouped10 = recent10.groupby('股票代码', sort=False)
    agg10 = grouped10.agg(
        vol10=('ret1', 'std'),
        amp_mean10=('daily_amp', 'mean'),
    )
    recent5 = grouped.tail(5).copy()
    grouped5 = recent5.groupby('股票代码', sort=False)
    agg5 = grouped5.agg(
        mean_amount5=('成交额', 'mean'),
        turnover5=('换手率', 'mean'),
    )
    latest = grouped.tail(1).set_index('股票代码').reindex(agg.index)

    last_close = latest['收盘'].to_numpy(dtype=np.float64)
    close_lag5 = latest['close_lag5'].fillna(0.0).to_numpy(dtype=np.float64)
    close_lag10 = latest['close_lag10'].fillna(0.0).to_numpy(dtype=np.float64)
    first_close = agg['first_close'].to_numpy(dtype=np.float64)
    row_count = agg['row_count'].to_numpy(dtype=np.int64)
    high20 = agg['max_high'].to_numpy(dtype=np.float64)
    low20 = agg['min_low'].to_numpy(dtype=np.float64)

    risk = pd.DataFrame(index=agg.index)
    risk['sigma20'] = agg['sigma20'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['median_amount20'] = agg['median_amount20'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['mean_amount20'] = agg['mean_amount20'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['turnover20'] = agg['turnover20'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['vol10'] = agg10.reindex(agg.index)['vol10'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['amp_mean10'] = agg10.reindex(agg.index)['amp_mean10'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['ret1'] = latest['ret1'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['ret5'] = np.where(row_count >= 6, last_close / (close_lag5 + 1e-12) - 1.0, 0.0)
    risk['ret10'] = np.where(row_count >= 11, last_close / (close_lag10 + 1e-12) - 1.0, 0.0)
    risk['ret20'] = np.where(row_count >= 2, last_close / (first_close + 1e-12) - 1.0, 0.0)
    risk['amp20'] = (
        (high20 - low20)
        / (last_close + 1e-12)
    )
    risk['pos20'] = (last_close - low20) / (high20 - low20 + 1e-12)
    amount5 = agg5.reindex(agg.index)['mean_amount5'].fillna(0.0).to_numpy(dtype=np.float64)
    turnover5 = agg5.reindex(agg.index)['turnover5'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['amt_ratio5'] = amount5 / (risk['mean_amount20'].to_numpy(dtype=np.float64) + 1e-12)
    risk['to_ratio5'] = turnover5 / (risk['turnover20'].to_numpy(dtype=np.float64) + 1e-12)

    beta_rows = []
    for stock_id, group in grouped:
        group = group.tail(61).dropna(subset=['ret1', 'market_ret1']).copy()
        if len(group) < 20:
            beta_rows.append((stock_id, 0.0, 0.0, 0.0, 0.0))
            continue

        x = group['market_ret1'].to_numpy(dtype=np.float64)
        y = group['ret1'].to_numpy(dtype=np.float64)
        x_centered = x - x.mean()
        var_x = float(np.dot(x_centered, x_centered) / max(len(x_centered), 1))
        beta = float(np.cov(y, x, ddof=0)[0, 1] / (var_x + 1e-12))

        downside = group[group['market_ret1'] < 0]
        if len(downside) >= 8:
            dx = downside['market_ret1'].to_numpy(dtype=np.float64)
            dy = downside['ret1'].to_numpy(dtype=np.float64)
            dx_centered = dx - dx.mean()
            var_dx = float(np.dot(dx_centered, dx_centered) / max(len(dx_centered), 1))
            downside_beta = float(np.cov(dy, dx, ddof=0)[0, 1] / (var_dx + 1e-12))
        else:
            downside_beta = beta

        residual = y - beta * x
        idio_vol = float(np.std(residual))

        close20 = group['收盘'].tail(20).to_numpy(dtype=np.float64)
        if close20.size == 0:
            max_drawdown = 0.0
        else:
            running_max = np.maximum.accumulate(close20)
            drawdowns = close20 / (running_max + 1e-12) - 1.0
            max_drawdown = float(abs(np.min(drawdowns)))

        beta_rows.append((stock_id, beta, downside_beta, idio_vol, max_drawdown))

    beta_df = pd.DataFrame(
        beta_rows,
        columns=['stock_id', 'beta60', 'downside_beta60', 'idio_vol60', 'max_drawdown20'],
    ).set_index('stock_id')
    risk = risk.join(beta_df, how='left')

    risk = risk.reset_index().rename(columns={'股票代码': 'stock_id'})
    risk['amount20'] = risk['median_amount20']
    risk['return_1'] = risk['ret1']
    risk['return_5'] = risk['ret5']
    return risk


def build_latest_inference_frame(processed, sequence_stock_ids, latest_date):
    latest_rows = processed[processed['日期'] == latest_date].copy()
    latest_rows = latest_rows.rename(columns={'股票代码': 'stock_id'})
    latest_rows = latest_rows.set_index('stock_id').reindex(sequence_stock_ids).reset_index()
    return latest_rows


def prepare_inference_artifacts(raw_df, stockid2idx, scaler, data_file, scaler_path, latest_date):
    cache_path = _build_predict_cache_path(data_file, scaler_path)
    if cache_path and os.path.exists(cache_path):
        print(f'[BDC][predict] load_cached_artifacts={cache_path}')
        return joblib.load(cache_path)

    processed, features = preprocess_predict_data(raw_df, stockid2idx)
    processed, features = _align_features_to_scaler(processed, features, scaler)
    processed = _sanitize_feature_frame(processed, features)
    processed = processed.astype({feature: np.float32 for feature in features}, copy=False)
    processed.loc[:, features] = scaler.transform(processed[features]).astype(np.float32, copy=False)

    sequences_np, sequence_stock_ids = build_inference_sequences(
        processed,
        features,
        config['sequence_length'],
        latest_date,
    )
    inference_df = build_latest_inference_frame(processed, sequence_stock_ids, latest_date)
    risk_df = build_risk_frame(raw_df)

    breadth = float(risk_df['ret1'].gt(0).mean()) if 'ret1' in risk_df.columns and len(risk_df) else 1.0

    artifacts = {
        'latest_date': pd.Timestamp(latest_date),
        'features': features,
        'sequence_stock_ids': sequence_stock_ids,
        'sequences_np': sequences_np,
        'inference_df': inference_df,
        'risk_df': risk_df,
        'breadth': breadth,
    }

    if cache_path:
        compress = int(config.get('predict', {}).get('cache_compress', 3))
        joblib.dump(artifacts, cache_path, compress=compress)
        print(f'[BDC][predict] saved_cached_artifacts={cache_path}')

    return artifacts


def optimize_weights(candidates, score_col='score', risk_col='sigma20', tau=0.6, cap=0.50, exposure_cap=1.0):
    """
    修复版权重优化：提高 cap 到 0.50，调整 tau 让权重分布更分散
    避免全部退化成等权 0.2
    """
    raw_score = candidates[score_col].to_numpy(dtype=np.float64)
    score = (raw_score - raw_score.mean()) / (raw_score.std() + 1e-9)
    risk = candidates[risk_col].to_numpy(dtype=np.float64)
    risk = np.clip(risk, 1e-4, None)
    raw_weight = np.exp(score / tau) / risk
    raw_weight = np.minimum(raw_weight, cap)
    weights = raw_weight / (raw_weight.sum() + 1e-12) * exposure_cap
    out = candidates[['stock_id']].copy()
    out['weight'] = np.round(weights, 6)
    return out


def inv_vol_weights(candidates, k=5, max_weight=0.40, min_weight=0.10, exposure_cap=1.0):
    """逆波动率权重：低波动率股票获得更高权重"""
    top = candidates.sort_values('score', ascending=False).head(k).copy()
    risk = np.clip(top['sigma20'].to_numpy(dtype=np.float64), 1e-4, None)
    raw_weight = 1.0 / risk
    weights = raw_weight / (raw_weight.sum() + 1e-12)
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / (weights.sum() + 1e-12) * exposure_cap
    out = top[['stock_id']].copy()
    out['weight'] = np.round(weights, 6)
    return out


def score_soft_weights(candidates, k=5, tau=1.0, max_weight=0.50, min_weight=0.05, exposure_cap=1.0):
    """纯分数软权重"""
    top = candidates.sort_values('score', ascending=False).head(k).copy()
    score = top['score'].to_numpy(dtype=np.float64)
    score = (score - score.mean()) / (score.std() + 1e-9)
    raw_weight = np.exp(score / tau)
    weights = raw_weight / (raw_weight.sum() + 1e-12)
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / (weights.sum() + 1e-12) * exposure_cap
    out = top[['stock_id']].copy()
    out['weight'] = np.round(weights, 6)
    return out


def select_candidates(score_df, post_cfg=None, history_df=None, asof_date=None):
    return shared_select_candidates(
        score_df,
        post_cfg=post_cfg or config.get('postprocess', {}),
        history_df=history_df,
        asof_date=asof_date,
    )


def select_weighting_func(weighting_name):
    """根据配置选择权重函数"""
    weighting_map = {
        'equal': lambda x: equal_weights(x, k=5),
        'risk_soft': lambda x: optimize_weights(x.head(5)),
        'score_soft': lambda x: score_soft_weights(x.head(5)),
        'inv_vol': lambda x: inv_vol_weights(x.head(5)),
    }
    return weighting_map.get(weighting_name, weighting_map['equal'])


def equal_weights(candidates, exposure_cap=1.0, k=5):
    top = candidates.head(k).copy()
    out = top[['stock_id']].copy()
    out['weight'] = np.round(exposure_cap / len(top), 6)
    return out


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-9)


def rank_pct_score(x):
    return pd.Series(np.asarray(x, dtype=np.float64)).rank(pct=True, method='average').to_numpy(dtype=np.float64)


def normalize_score(x, mode):
    if mode == 'rank_pct':
        return rank_pct_score(x)
    if mode == 'none':
        return np.asarray(x, dtype=np.float64)
    return zscore(x)


def _env_bool(name, default=None):
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _rank_pct(series):
    return series.astype(float).rank(pct=True, method='average')


def _safe_numeric(score_df, col, default=0.0):
    if col not in score_df.columns:
        return pd.Series(default, index=score_df.index, dtype=np.float64)
    values = pd.to_numeric(score_df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    fill = values.median() if values.notna().any() else default
    return values.fillna(fill).astype(np.float64)


def add_selector_scores(score_df):
    out = score_df.copy()
    out['lgb_norm'] = _rank_pct(_safe_numeric(out, 'lgb'))
    out['tf_norm'] = _rank_pct(_safe_numeric(out, 'transformer'))
    out['rank_disagreement'] = (out['lgb_norm'] - out['tf_norm']).abs()
    out['disagreement'] = out['rank_disagreement']
    out['liq_rank'] = _rank_pct(_safe_numeric(out, 'median_amount20'))
    out['sigma_rank'] = _rank_pct(_safe_numeric(out, 'sigma20'))
    out['amp_rank'] = _rank_pct(_safe_numeric(out, 'amp20'))
    out['ret1_rank'] = _rank_pct(_safe_numeric(out, 'ret1'))
    out['ret5_rank'] = _rank_pct(_safe_numeric(out, 'ret5'))
    out['beta60_rank'] = _rank_pct(_safe_numeric(out, 'beta60'))
    out['downside_beta60_rank'] = _rank_pct(_safe_numeric(out, 'downside_beta60'))
    out['max_drawdown20_rank'] = _rank_pct(_safe_numeric(out, 'max_drawdown20'))

    out['high_vol_flag'] = (out['sigma_rank'] > 0.85) | (out['amp_rank'] > 0.85)
    out['extreme_momo_flag'] = out['ret5_rank'] > 0.90
    out['overheat_flag'] = (out['ret5_rank'] > 0.75) & (out['amp_rank'] > 0.75)
    out['reversal_flag'] = (out['ret5_rank'] > 0.70) & (out['ret1_rank'] < 0.30)
    out['tail_risk_flag'] = (
        out['high_vol_flag']
        | (out['downside_beta60_rank'] > 0.85)
        | (out['max_drawdown20_rank'] > 0.85)
    )

    out['sigma_penalty'] = ((out['sigma_rank'] - 0.75) / 0.25).clip(lower=0.0)
    out['amp_penalty'] = ((out['amp_rank'] - 0.75) / 0.25).clip(lower=0.0)
    out['ret5_high_penalty'] = ((out['ret5_rank'] - 0.90) / 0.10).clip(lower=0.0)
    out['ret5_low_penalty'] = ((0.10 - out['ret5_rank']) / 0.10).clip(lower=0.0)
    out['bad_pick_risk'] = (
        0.25 * out['sigma_penalty']
        + 0.25 * out['amp_penalty']
        + 0.20 * out['ret5_high_penalty']
        + 0.15 * out['ret5_low_penalty']
        + 0.15 * out['rank_disagreement']
    )

    out['score_lgb_only'] = out['lgb_norm']
    out['score_balanced'] = (
        0.70 * out['lgb_norm']
        + 0.30 * out['tf_norm']
        - 0.10 * out['rank_disagreement']
    )
    out['score_conservative'] = (
        0.30 * out['lgb_norm']
        + 0.70 * out['tf_norm']
        - 0.20 * out['rank_disagreement']
    )
    out['score_conservative_softrisk'] = (
        out['score_conservative']
        - 0.05 * out['sigma_penalty']
        - 0.05 * out['amp_penalty']
        - 0.03 * out['ret5_high_penalty']
        - 0.03 * out['ret5_low_penalty']
    )
    out['vol_hinge'] = ((out['sigma_rank'] - 0.80) / 0.20).clip(lower=0.0)
    out['amp_hinge'] = ((out['amp_rank'] - 0.80) / 0.20).clip(lower=0.0)
    out['reversal_penalty'] = (
        ((out['ret5_rank'] - 0.70) / 0.30).clip(lower=0.0)
        * ((0.35 - out['ret1_rank']) / 0.35).clip(lower=0.0)
    )
    out['risk_disagreement_penalty'] = (
        0.5 * out['vol_hinge']
        + 0.5 * out['amp_hinge']
    ) * out['rank_disagreement']
    out['score_conservative_softrisk_v2'] = (
        out['score_conservative']
        - 0.06 * out['risk_disagreement_penalty']
        - 0.04 * out['reversal_penalty']
    )
    out['score_defensive'] = (
        0.20 * out['lgb_norm']
        + 0.60 * out['tf_norm']
        + 0.15 * out['liq_rank']
        - 0.20 * out['sigma_rank']
        - 0.15 * out['amp_rank']
    )
    out['score_defensive_v2'] = (
        0.75 * out['tf_norm']
        + 0.10 * out['lgb_norm']
        + 0.10 * out['liq_rank']
        - 0.10 * out['sigma_rank']
        - 0.10 * out['beta60_rank']
        - 0.10 * out['downside_beta60_rank']
        - 0.05 * out['max_drawdown20_rank']
    )
    out['score_legal_minrisk'] = (
        0.35 * out['tf_norm']
        + 0.15 * out['lgb_norm']
        + 0.20 * out['liq_rank']
        - 0.15 * out['sigma_rank']
        - 0.10 * out['downside_beta60_rank']
        - 0.10 * out['max_drawdown20_rank']
        - 0.05 * out['amp_rank']
    )
    cvar20_rank = out['cvar20_rank'] if 'cvar20_rank' in out.columns else out['max_drawdown20_rank']
    out['tail_risk_score'] = (
        0.25 * out['sigma_rank']
        + 0.20 * out['amp_rank']
        + 0.20 * out['downside_beta60_rank']
        + 0.20 * out['max_drawdown20_rank']
        + 0.15 * cvar20_rank
    )
    out['uncertainty_score'] = (
        0.50 * out['disagreement']
        + 0.30 * _safe_numeric(out, 'oof_error_bucket_rank', default=0.5)
        + 0.20 * _safe_numeric(out, 'regime_error_rank', default=0.5)
    )
    return out


def infer_regime(score_df):
    if len(score_df) == 0:
        return 'mixed_defensive', {}

    ret1 = _safe_numeric(score_df, 'ret1')
    ret5 = _safe_numeric(score_df, 'ret5')
    sigma20 = _safe_numeric(score_df, 'sigma20')
    breadth_1d = float(ret1.gt(0).mean())
    breadth_5d = float(ret5.gt(0).mean())
    median_ret1 = float(ret1.median())
    median_ret5 = float(ret5.median())
    high_vol_ratio = float(sigma20.gt(sigma20.quantile(0.75)).mean())

    if (
        breadth_1d >= 0.58
        and breadth_5d >= 0.55
        and median_ret1 > 0
        and median_ret5 > 0.005
    ):
        regime = 'risk_on_strict'
    elif breadth_1d >= 0.50 and breadth_5d >= 0.50 and median_ret5 > 0:
        regime = 'neutral_positive'
    elif breadth_1d < 0.35 or median_ret1 < -0.005:
        regime = 'risk_off'
    else:
        regime = 'mixed_defensive'

    return regime, {
        'breadth_1d': breadth_1d,
        'breadth_5d': breadth_5d,
        'median_ret1': median_ret1,
        'median_ret5': median_ret5,
        'high_vol_ratio': high_vol_ratio,
    }


def _top_overlap(score_df, col_a='lgb_norm', col_b='tf_norm', n=20):
    if len(score_df) == 0 or col_a not in score_df.columns or col_b not in score_df.columns:
        return 0.0
    k = min(n, len(score_df))
    a = set(score_df.nlargest(k, col_a)['stock_id'])
    b = set(score_df.nlargest(k, col_b)['stock_id'])
    return float(len(a & b) / max(k, 1))


def _score_gap(candidates, score_col='score'):
    if len(candidates) <= 5:
        return 0.0
    scores = candidates.sort_values(score_col, ascending=False)[score_col].reset_index(drop=True)
    next_end = min(20, len(scores))
    return float(scores.iloc[:5].mean() - scores.iloc[5:next_end].mean())


def _branch_candidates(score_df, selector_cfg, branch_name, regime=None):
    if branch_name == 'independent_union_rerank':
        return _union_rerank_candidates(score_df, selector_cfg, regime=regime)
    if branch_name == 'safe_union_2slot':
        return _safe_union_2slot_candidates(score_df, selector_cfg, regime=regime)
    if branch_name == 'legal_plus_1alpha':
        return _legal_plus_1alpha_candidates(score_df, selector_cfg, regime=regime)

    branch_cfg = selector_cfg.get('branches', {}).get(branch_name)
    if branch_cfg is None:
        return pd.DataFrame(columns=score_df.columns), {}

    score_col = branch_cfg.get('score_col', 'score_lgb_only')
    if score_col not in score_df.columns:
        return pd.DataFrame(columns=score_df.columns), branch_cfg

    branch_df = score_df.copy()
    branch_df['score'] = branch_df[score_col]
    filter_name = branch_cfg.get('filter', 'nofilter')
    filtered = shared_select_candidates(
        branch_df,
        post_cfg={
            'filter': filter_name,
            'liquidity_quantile': branch_cfg.get('liquidity_quantile', 0.30),
            'sigma_quantile': branch_cfg.get('sigma_quantile', 0.70),
        },
    )
    return filtered, branch_cfg


def _full_rank_quality(score_df, score_col):
    if score_col not in score_df.columns or len(score_df) == 0:
        return pd.Series(0.0, index=score_df.index, dtype=np.float64)
    values = pd.to_numeric(score_df[score_col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median() if values.notna().any() else 0.0)
    return values.rank(pct=True, method='average').astype(np.float64)


def _union_risk_budget(selector_cfg, regime):
    budget_cfg = selector_cfg.get('union_rerank', {}).get('risk_budget', {})
    default = {
        'max_tail_risk_count': 2,
        'max_high_vol_count': 2,
        'max_very_high_vol_count': 1,
        'max_very_tail_count': 1,
        'max_alpha_exception_count': 2,
        'max_branch_only_alpha_count': 1,
    }
    out = default.copy()
    out.update(budget_cfg.get(regime, {}))
    return out


def _candidate_flag(row, col):
    return bool(row.get(col, False))


def _selected_counts(rows):
    non_exception = [row for row in rows if not _candidate_flag(row, 'alpha_exception')]
    return {
        'tail_risk_count': sum(_candidate_flag(row, 'tail_risk_flag') for row in non_exception),
        'high_vol_count': sum(_candidate_flag(row, 'high_vol_flag') for row in non_exception),
        'very_high_vol_count': sum(_candidate_flag(row, 'very_high_vol_flag') for row in non_exception),
        'very_tail_count': sum(_candidate_flag(row, 'very_tail_flag') for row in non_exception),
        'alpha_exception_count': sum(_candidate_flag(row, 'alpha_exception') for row in rows),
        'branch_only_alpha_count': sum(_candidate_flag(row, 'branch_only_alpha_flag') for row in non_exception),
        'stable_count': sum(_candidate_flag(row, 'stable_candidate') for row in rows),
        'consensus_count': sum(float(row.get('consensus_count', 0.0)) >= 2.0 for row in rows),
    }


def _within_union_budget(rows, budget):
    counts = _selected_counts(rows)
    return (
        counts['tail_risk_count'] <= int(budget.get('max_tail_risk_count', 5))
        and counts['high_vol_count'] <= int(budget.get('max_high_vol_count', 5))
        and counts['very_high_vol_count'] <= int(budget.get('max_very_high_vol_count', 5))
        and counts['very_tail_count'] <= int(budget.get('max_very_tail_count', 5))
        and counts['alpha_exception_count'] <= int(budget.get('max_alpha_exception_count', 5))
        and counts['branch_only_alpha_count'] <= int(budget.get('max_branch_only_alpha_count', 5))
    )


def _row_float(row, col, default=0.0):
    value = row.get(col, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return value


def _combo_constraints(selector_cfg, regime):
    combo_cfg = selector_cfg.get('union_rerank', {}).get('combo_search', {})
    configured = combo_cfg.get('constraints', {})
    out = {
        'min_stable_count': 0,
        'min_consensus_count': 0,
        'max_tail_risk_count': 5,
        'max_high_vol_count': 5,
        'max_very_tail_count': 5,
        'max_very_high_vol_count': 5,
        'max_branch_only_alpha_count': 5,
    }
    out.update(configured.get('default', {}))
    out.update(configured.get(regime, {}))
    return out


def _combo_objective_weights(selector_cfg, regime):
    combo_cfg = selector_cfg.get('union_rerank', {}).get('combo_search', {})
    weights = dict(combo_cfg.get('objective_weights', {}))
    weights.update(combo_cfg.get('objective_weights_by_regime', {}).get(regime, {}))
    return weights


def _combo_constraints_pass(rows, budget, constraints):
    if len({str(row.get('stock_id')) for row in rows}) != len(rows):
        return False
    if not _within_union_budget(rows, budget):
        return False

    counts = _selected_counts(rows)
    limits = {
        'max_tail_risk_count': 'tail_risk_count',
        'max_high_vol_count': 'high_vol_count',
        'max_very_tail_count': 'very_tail_count',
        'max_very_high_vol_count': 'very_high_vol_count',
        'max_branch_only_alpha_count': 'branch_only_alpha_count',
    }
    for limit_key, count_key in limits.items():
        if counts[count_key] > int(constraints.get(limit_key, 5)):
            return False
    if counts['stable_count'] < int(constraints.get('min_stable_count', 0)):
        return False
    if counts['consensus_count'] < int(constraints.get('min_consensus_count', 0)):
        return False
    max_alpha_slots = constraints.get('max_alpha_slots')
    if max_alpha_slots is not None and counts['alpha_exception_count'] > int(max_alpha_slots):
        return False
    return True


def _combo_objective(rows, weights):
    k = max(len(rows), 1)
    counts = _selected_counts(rows)
    alpha_lcb = np.mean([
        _row_float(row, 'alpha_lcb', _row_float(row, 'alpha_score', 0.0))
        for row in rows
    ])
    consensus = np.mean([
        min(_row_float(row, 'consensus_count', 0.0), 3.0) / 3.0
        for row in rows
    ])
    stable_support = 0.70 * np.mean([
        _row_float(row, 'stable_fill_score', _row_float(row, 'score_legal_minrisk', 0.0))
        for row in rows
    ]) + 0.30 * (counts['stable_count'] / k)
    risk = np.mean([
        _row_float(row, 'effective_risk_score', _row_float(row, 'risk_score', 0.0))
        for row in rows
    ])
    disagreement = np.mean([
        _row_float(row, 'rank_disagreement', 0.0)
        for row in rows
    ])
    tail_count = counts['tail_risk_count'] / k
    high_vol_count = counts['high_vol_count'] / k
    branch_only_count = counts['branch_only_alpha_count'] / k

    return float(
        weights.get('alpha_lcb', 1.0) * alpha_lcb
        + weights.get('consensus', 0.20) * consensus
        + weights.get('stable_support', 0.15) * stable_support
        - weights.get('risk', 0.40) * risk
        - weights.get('disagreement', 0.30) * disagreement
        - weights.get('tail_count', 0.20) * tail_count
        - weights.get('high_vol_count', 0.12) * high_vol_count
        - weights.get('branch_only_count', 0.10) * branch_only_count
    )


def _combo_search_union_pick(pool, selector_cfg, regime, k=5):
    combo_cfg = selector_cfg.get('union_rerank', {}).get('combo_search', {})
    if not combo_cfg.get('enabled', False) or len(pool) < k:
        return None, {'combo_search_enabled': False}
    if regime in set(combo_cfg.get('disabled_regimes', [])):
        return None, {
            'combo_search_enabled': False,
            'combo_search_disabled_reason': f'disabled_for_{regime}',
        }

    topn = min(int(combo_cfg.get('topn', 25)), len(pool))
    sort_cols = [col for col in ['alpha_lcb', 'rerank_score', 'stable_fill_score'] if col in pool.columns]
    if not sort_cols:
        sort_cols = ['score']
    work = pool.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(topn).copy()
    records = work.to_dict('records')
    budget = _union_risk_budget(selector_cfg, regime)
    constraints = _combo_constraints(selector_cfg, regime)
    weights = _combo_objective_weights(selector_cfg, regime)

    best_rows = None
    best_score = -np.inf
    evaluated = 0
    feasible = 0
    for combo in itertools.combinations(records, k):
        evaluated += 1
        rows = list(combo)
        if not _combo_constraints_pass(rows, budget, constraints):
            continue
        feasible += 1
        score = _combo_objective(rows, weights)
        if score > best_score:
            best_score = score
            best_rows = rows

    if best_rows is None:
        return None, {
            'combo_search_enabled': True,
            'combo_search_topn': topn,
            'combo_search_evaluated': evaluated,
            'combo_search_feasible': feasible,
            'combo_search_score': np.nan,
        }

    best_rows = sorted(
        best_rows,
        key=lambda row: (
            _row_float(row, 'alpha_lcb', _row_float(row, 'alpha_score', 0.0)),
            _row_float(row, 'stable_fill_score', 0.0),
            _row_float(row, 'rerank_score', 0.0),
        ),
        reverse=True,
    )
    for row in best_rows:
        row['combo_search_score'] = best_score
        row['selection_method'] = 'combo_search'
    return best_rows, {
        'combo_search_enabled': True,
        'combo_search_topn': topn,
        'combo_search_evaluated': evaluated,
        'combo_search_feasible': feasible,
        'combo_search_score': float(best_score),
    }


def _constrained_union_pick(pool, selector_cfg, regime, k=5):
    budget = _union_risk_budget(selector_cfg, regime)
    rows = pool.sort_values('rerank_score', ascending=False).to_dict('records')
    selected = []
    selected_ids = set()

    shape_cfg = selector_cfg.get('union_rerank', {}).get('portfolio_shape', {}).get(regime, {})
    max_alpha_slots = int(shape_cfg.get('max_pure_alpha_slots', 0))
    if max_alpha_slots > 0:
        alpha_rows = (
            pool[pool['alpha_exception'].astype(bool)]
            .sort_values(['alpha_score', 'rerank_score'], ascending=False)
            .to_dict('records')
        )
        for row in alpha_rows:
            stock_id = row.get('stock_id')
            if stock_id in selected_ids:
                continue
            proposal = selected + [row]
            if _within_union_budget(proposal, budget):
                selected.append(row)
                selected_ids.add(stock_id)
            if len(selected) >= min(max_alpha_slots, k):
                break

    stable_slots = int(shape_cfg.get('min_stable_slots', 0))
    if regime == 'risk_off' and max_alpha_slots > 0:
        stable_slots = max(stable_slots, k - len(selected))
    if stable_slots > 0 and 'stable_fill_score' in pool.columns:
        stable_rows = (
            pool[pool['stable_candidate'].astype(bool)]
            .sort_values(['stable_fill_score', 'rerank_score'], ascending=False)
            .to_dict('records')
        )
        for row in stable_rows:
            stock_id = row.get('stock_id')
            if stock_id in selected_ids:
                continue
            proposal = selected + [row]
            if _within_union_budget(proposal, budget):
                selected.append(row)
                selected_ids.add(stock_id)
            if len(selected) >= k or _selected_counts(selected)['stable_count'] >= stable_slots:
                break

    if len(selected) >= k:
        return selected[:k]

    for row in rows:
        stock_id = row.get('stock_id')
        if stock_id in selected_ids:
            continue
        proposal = selected + [row]
        if _within_union_budget(proposal, budget):
            selected.append(row)
            selected_ids.add(stock_id)
            if len(selected) >= k:
                break

    if len(selected) < k:
        relaxed = budget.copy()
        relaxed['max_tail_risk_count'] = int(relaxed.get('max_tail_risk_count', 5)) + 1
        relaxed['max_high_vol_count'] = int(relaxed.get('max_high_vol_count', 5)) + 1
        relaxed['max_very_tail_count'] = int(relaxed.get('max_very_tail_count', 5)) + 1
        for row in rows:
            stock_id = row.get('stock_id')
            if stock_id in selected_ids:
                continue
            proposal = selected + [row]
            if _within_union_budget(proposal, relaxed):
                selected.append(row)
                selected_ids.add(stock_id)
            if len(selected) >= k:
                break

    if len(selected) < k:
        for row in rows:
            stock_id = row.get('stock_id')
            if stock_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(stock_id)
            if len(selected) >= k:
                break
    return selected


def _is_bad_tail_candidate(row):
    if _candidate_flag(row, 'alpha_exception'):
        return False
    return (
        _candidate_flag(row, 'very_tail_flag')
        or _candidate_flag(row, 'very_high_vol_flag')
        or (
            _candidate_flag(row, 'tail_risk_flag')
            and _candidate_flag(row, 'high_vol_flag')
            and float(row.get('consensus_count', 0.0)) <= 1.0
        )
        or (
            float(row.get('rank_disagreement', 0.0)) >= 0.25
            and _candidate_flag(row, 'branch_only_alpha_flag')
        )
    )


def _replace_bad_tail_candidates(selected, pool, selector_cfg, regime):
    replace_cfg = selector_cfg.get('union_rerank', {}).get('replacement', {})
    if not replace_cfg.get('enabled', True):
        return selected

    alpha_tolerance = float(replace_cfg.get('alpha_tolerance', 0.15))
    min_risk_improvement = float(replace_cfg.get('min_risk_improvement', 0.20))
    require_safe = bool(replace_cfg.get('require_safe_branch_support', True))
    budget = _union_risk_budget(selector_cfg, regime)
    selected = [dict(row) for row in selected]

    candidates = pool.sort_values('rerank_score', ascending=False).to_dict('records')
    for idx, row in enumerate(list(selected)):
        if not _is_bad_tail_candidate(row):
            continue

        selected_ids = {item['stock_id'] for pos, item in enumerate(selected) if pos != idx}
        replacement = None
        for cand in candidates:
            stock_id = cand.get('stock_id')
            if stock_id in selected_ids or stock_id == row.get('stock_id'):
                continue
            if require_safe and not _candidate_flag(cand, 'safe_branch_support'):
                continue
            if float(cand.get('risk_score', 1.0)) > float(row.get('risk_score', 1.0)) - min_risk_improvement:
                continue
            if float(cand.get('alpha_score', 0.0)) < float(row.get('alpha_score', 0.0)) - alpha_tolerance:
                continue
            proposal = selected[:idx] + [cand] + selected[idx + 1:]
            if not _within_union_budget(proposal, budget):
                continue
            replacement = cand
            break

        if replacement is not None:
            replacement = dict(replacement)
            replacement['replacement_reason'] = f"replace_bad_tail:{row.get('stock_id')}"
            row['replacement_reason'] = 'removed_bad_tail'
            selected[idx] = replacement

    return selected


def _union_rerank_candidates(score_df, selector_cfg, regime=None):
    branch_cfg = selector_cfg.get('branches', {}).get('independent_union_rerank', {})
    union_cfg = selector_cfg.get('union_rerank', {})
    if not union_cfg.get('enabled', False):
        return pd.DataFrame(columns=score_df.columns), branch_cfg

    pool_cfg = union_cfg.get('candidate_pool', {})
    if not pool_cfg:
        return pd.DataFrame(columns=score_df.columns), branch_cfg

    branch_sets = {}
    pool_ids = set()
    for source_branch, topn in pool_cfg.items():
        if source_branch == 'independent_union_rerank':
            continue
        candidates, source_cfg = _branch_candidates(score_df, selector_cfg, source_branch)
        if len(candidates) == 0:
            branch_sets[source_branch] = set()
            continue
        ids = candidates.head(int(topn))['stock_id'].astype(str).tolist()
        branch_sets[source_branch] = set(ids)
        pool_ids.update(ids)

    if not pool_ids:
        return pd.DataFrame(columns=score_df.columns), branch_cfg

    pool = score_df[score_df['stock_id'].astype(str).isin(pool_ids)].copy()
    if len(pool) < 5:
        return pool.sort_values('score', ascending=False).reset_index(drop=True), branch_cfg

    branch_score_cols = {
        'conservative_softrisk_v2': 'score_conservative_softrisk_v2',
        'conservative_softrisk_v2_strict': 'score_conservative_softrisk_v2',
        'lgb_only_guarded': 'score_lgb_only',
        'balanced_guarded': 'score_balanced',
        'defensive_v2_strict': 'score_defensive_v2',
        'legal_minrisk_hardened': 'score_legal_minrisk',
    }
    for source_branch, score_col in branch_score_cols.items():
        rank_col = f'{source_branch}_rank_quality'
        rank_map = dict(zip(score_df['stock_id'].astype(str), _full_rank_quality(score_df, score_col)))
        pool[rank_col] = pool['stock_id'].astype(str).map(rank_map).fillna(0.0).astype(np.float64)

    source_names = list(pool_cfg.keys())
    for source_branch in source_names:
        pool[f'in_{source_branch}'] = pool['stock_id'].astype(str).isin(branch_sets.get(source_branch, set()))
    pool['consensus_count'] = pool[[f'in_{name}' for name in source_names]].sum(axis=1).astype(float)
    pool['branch_sources'] = pool.apply(
        lambda row: '|'.join([name for name in source_names if bool(row.get(f'in_{name}', False))]),
        axis=1,
    )

    branch_rank_bonus = (
        0.50 * (pool['consensus_count'] / max(float(len(source_names)), 1.0))
        + 0.50 * pool[[f'{name}_rank_quality' for name in branch_score_cols if f'{name}_rank_quality' in pool.columns]].max(axis=1)
    )
    pool['alpha_score'] = (
        0.35 * _safe_numeric(pool, 'score_conservative_softrisk_v2')
        + 0.25 * _safe_numeric(pool, 'lgb_norm')
        + 0.20 * _safe_numeric(pool, 'score_balanced')
        + 0.10 * _safe_numeric(pool, 'tf_norm')
        + 0.10 * branch_rank_bonus
    )
    pool['alpha_rank_num'] = pool['alpha_score'].rank(ascending=False, method='first')

    full_conservative_rank = _safe_numeric(score_df, 'score_conservative_softrisk_v2').rank(ascending=False, method='first')
    full_lgb_rank = _safe_numeric(score_df, 'score_lgb_only').rank(ascending=False, method='first')
    conservative_rank_map = dict(zip(score_df['stock_id'].astype(str), full_conservative_rank))
    lgb_rank_map = dict(zip(score_df['stock_id'].astype(str), full_lgb_rank))
    pool['conservative_rank_num'] = pool['stock_id'].astype(str).map(conservative_rank_map).fillna(9999).astype(float)
    pool['lgb_rank_num'] = pool['stock_id'].astype(str).map(lgb_rank_map).fillna(9999).astype(float)

    pool['very_high_vol_flag'] = (
        (_safe_numeric(pool, 'sigma_rank') > 0.95)
        | (_safe_numeric(pool, 'amp_rank') > 0.95)
    )
    pool['very_tail_flag'] = (
        (_safe_numeric(pool, 'downside_beta60_rank') > 0.90)
        | (_safe_numeric(pool, 'max_drawdown20_rank') > 0.90)
    )
    pool['alpha_exception'] = (
        ((pool['conservative_rank_num'] <= 3) & (pool['lgb_rank_num'] <= 25))
        | ((pool['conservative_rank_num'] <= 2) & (_safe_numeric(pool, 'tf_norm') >= 0.95))
    )
    pool['branch_only_alpha_flag'] = (
        (pool['consensus_count'] <= 1)
        & (pool['alpha_rank_num'] <= 15)
        & ~pool['alpha_exception'].astype(bool)
    )
    pool['safe_branch_support'] = (
        pool.get('in_defensive_v2_strict', False).astype(bool)
        | pool.get('in_legal_minrisk_hardened', False).astype(bool)
        | (_safe_numeric(pool, 'score_defensive_v2') >= _safe_numeric(score_df, 'score_defensive_v2').quantile(0.85))
    )
    pool['stable_candidate'] = (
        pool['safe_branch_support'].astype(bool)
        & ~pool['tail_risk_flag'].astype(bool)
        & ~pool['reversal_flag'].astype(bool)
    )

    pool['risk_score'] = (
        0.30 * pool['tail_risk_flag'].astype(float)
        + 0.25 * pool['very_tail_flag'].astype(float)
        + 0.20 * pool['high_vol_flag'].astype(float)
        + 0.15 * pool['very_high_vol_flag'].astype(float)
        + 0.10 * _safe_numeric(pool, 'rank_disagreement')
    )
    pool['stable_fill_score'] = (
        0.50 * _safe_numeric(pool, 'lgb_norm')
        + 0.20 * _safe_numeric(pool, 'score_balanced')
        + 0.08 * _safe_numeric(pool, 'liq_rank')
        + 0.15 * _safe_numeric(pool, 'ret1_rank')
        + 0.08 * pool.get('in_lgb_only_guarded', False).astype(float)
        + 0.05 * pool.get('in_legal_minrisk_hardened', False).astype(float)
        + 0.03 * pool.get('in_balanced_guarded', False).astype(float)
        - 0.10 * _safe_numeric(pool, 'risk_score')
    )
    pool['effective_risk_score'] = pool['risk_score'] * np.where(pool['alpha_exception'].astype(bool), 0.45, 1.0)

    lcb_cfg = union_cfg.get('alpha_lcb', {})
    if lcb_cfg.get('enabled', False):
        oof_col = lcb_cfg.get('oof_error_col', 'alpha_error_q80')
        oof_penalty = _safe_numeric(pool, oof_col, default=0.0).clip(lower=0.0)
        pool['alpha_uncertainty_penalty'] = (
            float(lcb_cfg.get('base_penalty', 0.015))
            + float(lcb_cfg.get('uncertainty_weight', 0.055)) * _safe_numeric(pool, 'uncertainty_score', default=0.5)
            + float(lcb_cfg.get('disagreement_weight', 0.025)) * _safe_numeric(pool, 'rank_disagreement')
            + float(lcb_cfg.get('risk_weight', 0.020)) * _safe_numeric(pool, 'effective_risk_score')
            + float(lcb_cfg.get('tail_risk_weight', 0.035)) * pool['tail_risk_flag'].astype(float)
            + float(lcb_cfg.get('branch_only_weight', 0.025)) * pool['branch_only_alpha_flag'].astype(float)
            + oof_penalty
        )
        pool['alpha_lcb'] = pool['alpha_score'] - pool['alpha_uncertainty_penalty']
    else:
        pool['alpha_uncertainty_penalty'] = 0.0
        pool['alpha_lcb'] = pool['alpha_score']
    pool['alpha_lcb_z'] = (
        (pool['alpha_lcb'] - pool['alpha_lcb'].mean())
        / (pool['alpha_lcb'].std(ddof=0) + 1e-9)
    )

    risk_lambda_cfg = union_cfg.get('risk_lambda', {})
    risk_lambda = float(risk_lambda_cfg.get(regime, risk_lambda_cfg.get('risk_off', 0.75)))
    pool['consensus_bonus'] = 0.03 * np.minimum(pool['consensus_count'], 3.0)
    pool['liquidity_bonus'] = 0.06 * _safe_numeric(pool, 'liq_rank')
    pool['defensive_support_bonus'] = 0.06 * pool['safe_branch_support'].astype(float)
    pool['score_disagreement_penalty'] = 0.08 * _safe_numeric(pool, 'rank_disagreement')
    pool['rerank_score'] = (
        pool['alpha_lcb']
        - risk_lambda * pool['effective_risk_score']
        + pool['consensus_bonus']
        + pool['liquidity_bonus']
        + pool['defensive_support_bonus']
        - pool['score_disagreement_penalty']
    )
    pool['score'] = pool['rerank_score']
    pool['replacement_reason'] = ''

    selected, combo_info = _combo_search_union_pick(pool, selector_cfg, regime or 'risk_off', k=5)
    if selected is None:
        selected = _constrained_union_pick(pool, selector_cfg, regime or 'risk_off', k=5)
        selected = _replace_bad_tail_candidates(selected, pool, selector_cfg, regime or 'risk_off')
        combo_info.setdefault('selection_method', 'greedy_budget')
    else:
        combo_info.setdefault('selection_method', 'combo_search')
    selected_ids = [row['stock_id'] for row in selected]
    selected_score_map = {row['stock_id']: row['rerank_score'] for row in selected}
    selected_reason_map = {row['stock_id']: row.get('replacement_reason', '') for row in selected}

    pool['final_selected'] = pool['stock_id'].isin(selected_ids)
    pool['replacement_reason'] = pool['stock_id'].map(selected_reason_map).fillna(pool['replacement_reason'])
    pool['union_debug_rank'] = pool['rerank_score'].rank(ascending=False, method='first')
    for key, value in combo_info.items():
        pool.attrs[key] = value
    pool.attrs['candidate_debug'] = pool.sort_values('rerank_score', ascending=False).copy()

    selected_df = pool[pool['stock_id'].isin(selected_ids)].copy()
    selected_df['_selected_order'] = selected_df['stock_id'].map({stock_id: idx for idx, stock_id in enumerate(selected_ids)})
    selected_df['score'] = selected_df['stock_id'].map(selected_score_map).fillna(selected_df['rerank_score'])
    selected_df = selected_df.sort_values('_selected_order').drop(columns=['_selected_order']).reset_index(drop=True)
    selected_df.attrs['candidate_debug'] = pool.sort_values('rerank_score', ascending=False).copy()
    selected_df.attrs['candidate_pool'] = pool.copy()
    selected_df.attrs['combo_info'] = combo_info
    return selected_df, branch_cfg


def _safe_union_2slot_candidates(score_df, selector_cfg, regime=None):
    branch_cfg = selector_cfg.get('branches', {}).get('safe_union_2slot', {})
    gate_cfg = selector_cfg.get('gated_union_rerank', {}).get('safe_union_2slot', {})
    if not gate_cfg.get('enabled', True):
        return pd.DataFrame(columns=score_df.columns), branch_cfg

    regime = regime or infer_regime(score_df)[0]
    _, regime_stats = infer_regime(score_df)
    veto, veto_reason = _market_hard_veto(regime, regime_stats)
    if veto:
        empty = pd.DataFrame(columns=score_df.columns)
        empty.attrs['safe_union_info'] = {'safe_union_reason': veto_reason}
        return empty, branch_cfg
    market_ok, market_reason = _market_enable_gate(regime, regime_stats)
    if gate_cfg.get('require_market_enable', True) and not market_ok:
        empty = pd.DataFrame(columns=score_df.columns)
        empty.attrs['safe_union_info'] = {'safe_union_reason': f'safe_union_{market_reason}'}
        return empty, branch_cfg

    union_candidates, _ = _union_rerank_candidates(score_df, selector_cfg, regime=regime)
    pool = union_candidates.attrs.get('candidate_pool') if hasattr(union_candidates, 'attrs') else None
    if pool is None or len(pool) < 5:
        empty = pd.DataFrame(columns=score_df.columns)
        empty.attrs['safe_union_info'] = {'safe_union_reason': 'missing_union_pool'}
        return empty, branch_cfg

    pool = pool.copy()
    min_clean = int(gate_cfg.get('min_clean_alpha_count', 2))
    min_lcb_z = float(gate_cfg.get('min_clean_alpha_lcb_z', 1.25))
    max_alpha_slots = int(gate_cfg.get('max_alpha_slots', 2))
    max_disagreement = float(gate_cfg.get('max_best_alpha_disagreement', 0.25))

    clean_alpha_mask = (
        (pool.get('alpha_exception', pd.Series(False, index=pool.index)).astype(bool) | (_safe_numeric(pool, 'alpha_lcb_z') >= min_lcb_z + 0.35))
        & (_safe_numeric(pool, 'alpha_lcb_z') >= min_lcb_z)
        & (pd.to_numeric(pool.get('consensus_count', pd.Series(0.0, index=pool.index)), errors='coerce').fillna(0.0) >= 2.0)
        & ~pool.get('branch_only_alpha_flag', pd.Series(False, index=pool.index)).astype(bool)
        & ~pool.get('tail_risk_flag', pd.Series(False, index=pool.index)).astype(bool)
        & ~pool.get('very_tail_flag', pd.Series(False, index=pool.index)).astype(bool)
        & ~pool.get('very_high_vol_flag', pd.Series(False, index=pool.index)).astype(bool)
        & (pd.to_numeric(pool.get('rank_disagreement', pd.Series(1.0, index=pool.index)), errors='coerce').fillna(1.0) <= max_disagreement)
    )
    clean_alpha_ids = set(pool.loc[clean_alpha_mask, 'stock_id'].astype(str))
    if len(clean_alpha_ids) < min_clean:
        empty = pd.DataFrame(columns=score_df.columns)
        empty.attrs['safe_union_info'] = {
            'safe_union_reason': 'insufficient_clean_alpha_lcb',
            'safe_union_clean_alpha_count': len(clean_alpha_ids),
        }
        return empty, branch_cfg

    eligible = pool[
        clean_alpha_mask
        | pool.get('stable_candidate', pd.Series(False, index=pool.index)).astype(bool)
        | pool.get('safe_branch_support', pd.Series(False, index=pool.index)).astype(bool)
    ].copy()
    if len(eligible) < 5:
        empty = pd.DataFrame(columns=score_df.columns)
        empty.attrs['safe_union_info'] = {'safe_union_reason': 'insufficient_safe_pool'}
        return empty, branch_cfg

    constraints = {
        'min_stable_count': int(gate_cfg.get('min_stable_slots', 3)),
        'min_consensus_count': 2,
        'max_tail_risk_count': int(gate_cfg.get('max_tail_risk_count', 1)),
        'max_high_vol_count': int(gate_cfg.get('max_high_vol_count', 1)),
        'max_very_tail_count': int(gate_cfg.get('max_very_tail_count', 0)),
        'max_very_high_vol_count': int(gate_cfg.get('max_very_high_vol_count', 0)),
        'max_branch_only_alpha_count': int(gate_cfg.get('max_branch_only_alpha_count', 0)),
        'max_alpha_slots': max_alpha_slots,
    }
    budget = _union_risk_budget(selector_cfg, regime)
    budget.update({
        'max_tail_risk_count': constraints['max_tail_risk_count'],
        'max_high_vol_count': constraints['max_high_vol_count'],
        'max_very_tail_count': constraints['max_very_tail_count'],
        'max_very_high_vol_count': constraints['max_very_high_vol_count'],
        'max_branch_only_alpha_count': constraints['max_branch_only_alpha_count'],
    })
    weights = _combo_objective_weights(selector_cfg, regime)
    topn = min(int(selector_cfg.get('union_rerank', {}).get('combo_search', {}).get('topn', 25)), len(eligible))
    eligible = eligible.sort_values(['alpha_lcb', 'stable_fill_score', 'rerank_score'], ascending=False).head(topn)

    best_rows = None
    best_score = -np.inf
    evaluated = 0
    feasible = 0
    for combo in itertools.combinations(eligible.to_dict('records'), 5):
        evaluated += 1
        rows = list(combo)
        clean_count = sum(str(row.get('stock_id')) in clean_alpha_ids for row in rows)
        if clean_count < min_clean or clean_count > max_alpha_slots:
            continue
        if not _combo_constraints_pass(rows, budget, constraints):
            continue
        feasible += 1
        score = _combo_objective(rows, weights)
        if score > best_score:
            best_score = score
            best_rows = rows

    if best_rows is None:
        empty = pd.DataFrame(columns=score_df.columns)
        empty.attrs['safe_union_info'] = {
            'safe_union_reason': 'no_feasible_safe_combo',
            'safe_union_evaluated': evaluated,
            'safe_union_feasible': feasible,
        }
        return empty, branch_cfg

    selected_ids = [row['stock_id'] for row in sorted(
        best_rows,
        key=lambda row: (
            str(row.get('stock_id')) in clean_alpha_ids,
            _row_float(row, 'alpha_lcb', 0.0),
            _row_float(row, 'stable_fill_score', 0.0),
        ),
        reverse=True,
    )]
    selected = pool[pool['stock_id'].isin(selected_ids)].copy()
    selected['_selected_order'] = selected['stock_id'].map({stock_id: idx for idx, stock_id in enumerate(selected_ids)})
    selected['safe_union_2slot_score'] = (
        0.70 * _safe_numeric(selected, 'alpha_lcb', default=0.0)
        + 0.20 * _safe_numeric(selected, 'stable_fill_score', default=0.0)
        + 0.10 * np.minimum(_safe_numeric(selected, 'consensus_count', default=0.0), 3.0) / 3.0
        - 0.20 * _safe_numeric(selected, 'effective_risk_score', default=0.0)
    )
    selected['score'] = selected['safe_union_2slot_score']
    selected = selected.sort_values('_selected_order').drop(columns=['_selected_order']).reset_index(drop=True)
    safe_info = {
        'safe_union_reason': 'enabled_safe_union_2slot',
        'safe_union_clean_alpha_count': int(sum(str(stock_id) in clean_alpha_ids for stock_id in selected_ids)),
        'safe_union_evaluated': evaluated,
        'safe_union_feasible': feasible,
        'safe_union_score': float(best_score),
    }
    selected.attrs['safe_union_info'] = safe_info
    selected.attrs['candidate_pool'] = pool.copy()
    selected.attrs['candidate_debug'] = pool.sort_values('rerank_score', ascending=False).copy()
    return selected, branch_cfg


def _best_clean_alpha_candidate(union_candidates, selector_cfg, regime, regime_stats):
    gate_cfg = selector_cfg.get('gated_union_rerank', {}).get('legal_plus_1alpha', {})
    if not gate_cfg.get('enabled', True):
        return None, {}

    veto, veto_reason = _market_hard_veto(regime, regime_stats)
    if veto:
        return None, {'legal_plus_reason': veto_reason}

    market_ok, market_reason = _market_enable_gate(regime, regime_stats)
    if not market_ok:
        return None, {'legal_plus_reason': f'legal_plus_{market_reason}'}

    pool = union_candidates.attrs.get('candidate_pool') if hasattr(union_candidates, 'attrs') else None
    if pool is None or len(pool) == 0:
        return None, {'legal_plus_reason': 'missing_union_pool'}

    pool = pool.copy()
    alpha_col = 'alpha_lcb' if 'alpha_lcb' in pool.columns else 'alpha_score'
    full_alpha = _safe_numeric(pool, alpha_col)
    alpha_z = (full_alpha - full_alpha.mean()) / (full_alpha.std(ddof=0) + 1e-9)
    pool['alpha_score_z'] = alpha_z

    min_z = float(gate_cfg.get('min_best_alpha_z', 1.75))
    min_consensus = float(gate_cfg.get('min_best_alpha_consensus', 2.0))
    max_disagreement = float(gate_cfg.get('max_best_alpha_disagreement', 0.25))

    candidates = pool[
        pool.get('alpha_exception', pd.Series(False, index=pool.index)).astype(bool)
        & (pool['alpha_score_z'] >= min_z)
        & (pd.to_numeric(pool.get('consensus_count', pd.Series(0.0, index=pool.index)), errors='coerce').fillna(0.0) >= min_consensus)
        & ~pool.get('branch_only_alpha_flag', pd.Series(False, index=pool.index)).astype(bool)
        & ~pool.get('tail_risk_flag', pd.Series(False, index=pool.index)).astype(bool)
        & ~pool.get('very_tail_flag', pd.Series(False, index=pool.index)).astype(bool)
        & ~pool.get('very_high_vol_flag', pd.Series(False, index=pool.index)).astype(bool)
        & (pd.to_numeric(pool.get('rank_disagreement', pd.Series(1.0, index=pool.index)), errors='coerce').fillna(1.0) <= max_disagreement)
    ].copy()

    if candidates.empty:
        return None, {
            'legal_plus_reason': 'no_clean_single_alpha',
            'best_alpha_score_z': float(pool['alpha_score_z'].max()) if len(pool) else 0.0,
        }

    market_not_toxic = (
        float(regime_stats.get('median_ret1', 0.0)) > -0.009
        and not (
            float(regime_stats.get('breadth_5d', 0.0)) < 0.40
            and float(regime_stats.get('median_ret5', 0.0)) < 0.0
        )
    )
    if not market_not_toxic:
        return None, {'legal_plus_reason': 'market_toxic_for_single_alpha'}

    stable_fill_count = int(
        union_candidates.get('stable_candidate', pd.Series(False, index=union_candidates.index)).astype(bool).sum()
    )
    if regime in {'risk_off', 'mixed_defensive'} and stable_fill_count < 1:
        return None, {'legal_plus_reason': 'insufficient_stable_support'}

    sort_cols = ['alpha_score_z', alpha_col, 'rerank_score']
    best = candidates.sort_values(sort_cols, ascending=False).iloc[0].copy()
    return best, {
        'legal_plus_reason': 'enabled_legal_plus_1alpha',
        'best_alpha_stock': str(best.get('stock_id')),
        'best_alpha_score_z': float(best.get('alpha_score_z', 0.0)),
        'best_alpha_lcb': float(best.get('alpha_lcb', best.get('alpha_score', 0.0))),
        'best_alpha_consensus_count': float(best.get('consensus_count', 0.0)),
    }


def _legal_plus_1alpha_candidates(score_df, selector_cfg, regime=None):
    branch_cfg = selector_cfg.get('branches', {}).get('legal_plus_1alpha', {})
    regime = regime or infer_regime(score_df)[0]
    _, regime_stats = infer_regime(score_df)

    union_candidates, _ = _union_rerank_candidates(score_df, selector_cfg, regime=regime)
    best_alpha, plus_info = _best_clean_alpha_candidate(union_candidates, selector_cfg, regime, regime_stats)
    if best_alpha is None:
        empty = pd.DataFrame(columns=score_df.columns)
        empty.attrs['legal_plus_info'] = plus_info
        return empty, branch_cfg

    legal_candidates, _ = _branch_candidates(score_df, selector_cfg, 'legal_minrisk_hardened', regime=regime)
    legal_top = legal_candidates.head(5).copy()
    if len(legal_top) < 5:
        empty = pd.DataFrame(columns=score_df.columns)
        empty.attrs['legal_plus_info'] = {'legal_plus_reason': 'insufficient_legal_candidates'}
        return empty, branch_cfg

    alpha_id = str(best_alpha['stock_id'])
    if alpha_id in set(legal_top['stock_id'].astype(str)):
        out = legal_top.copy()
        out['legal_plus_score'] = out.get('score_legal_minrisk', out.get('score', 0.0))
        out = out.reset_index(drop=True)
        out.attrs['legal_plus_info'] = {**plus_info, 'legal_plus_replaced_stock': ''}
        if hasattr(union_candidates, 'attrs') and union_candidates.attrs.get('candidate_debug') is not None:
            out.attrs['candidate_debug'] = union_candidates.attrs['candidate_debug']
        return out, branch_cfg

    pool = union_candidates.attrs.get('candidate_pool') if hasattr(union_candidates, 'attrs') else pd.DataFrame()
    alpha_map = {}
    consensus_map = {}
    if len(pool):
        alpha_map = dict(zip(pool['stock_id'].astype(str), _safe_numeric(pool, 'alpha_score')))
        consensus_map = dict(zip(pool['stock_id'].astype(str), _safe_numeric(pool, 'consensus_count')))

    legal_top['_alpha_for_drop'] = legal_top['stock_id'].astype(str).map(alpha_map).fillna(_safe_numeric(legal_top, 'score_legal_minrisk'))
    legal_top['_consensus_for_drop'] = legal_top['stock_id'].astype(str).map(consensus_map).fillna(0.0)
    legal_top['_drop_rank'] = (
        legal_top['_alpha_for_drop'].rank(method='first', ascending=True)
        + 0.15 * legal_top['_consensus_for_drop'].rank(method='first', ascending=True)
        - 0.10 * _safe_numeric(legal_top, 'score_legal_minrisk').rank(method='first', ascending=False)
    )
    drop_idx = legal_top.sort_values('_drop_rank', ascending=True).index[0]
    dropped_stock = str(legal_top.loc[drop_idx, 'stock_id'])
    kept = legal_top.drop(index=drop_idx).copy()

    alpha_row = score_df[score_df['stock_id'].astype(str) == alpha_id].copy()
    if alpha_row.empty:
        alpha_row = pd.DataFrame([best_alpha.to_dict()])
    alpha_row['legal_plus_score'] = float(best_alpha.get('alpha_score', best_alpha.get('score', 0.0)))
    kept['legal_plus_score'] = kept.get('score_legal_minrisk', kept.get('score', 0.0))
    out = pd.concat([alpha_row.head(1), kept], ignore_index=True, sort=False)
    out['score'] = out['legal_plus_score']
    out = out.drop(columns=[col for col in ['_alpha_for_drop', '_consensus_for_drop', '_drop_rank'] if col in out.columns], errors='ignore').reset_index(drop=True)
    out.attrs['legal_plus_info'] = {**plus_info, 'legal_plus_replaced_stock': dropped_stock}
    if hasattr(union_candidates, 'attrs') and union_candidates.attrs.get('candidate_debug') is not None:
        out.attrs['candidate_debug'] = union_candidates.attrs['candidate_debug']
    return out, branch_cfg


def _branch_diag(score_df, candidates, branch_name, score_col='score'):
    if len(candidates) == 0:
        return {
            'branch': branch_name,
            'available': False,
            'score_gap': 0.0,
            'top5_disagreement': 1.0,
            'top20_overlap': _top_overlap(score_df),
            'top5_sigma_rank_mean': 1.0,
            'top5_amp_rank_mean': 1.0,
            'top5_ret1_rank_mean': 0.0,
            'top5_ret5_rank_mean': 0.0,
            'top5_tail_risk_count': 5,
            'top5_extreme_momo_count': 5,
            'top5_overheat_count': 5,
            'top5_reversal_count': 5,
            'branch_risk_score': 1.0,
            'defensive_overlap': 0,
            'risk_screen_ok': False,
        }

    top = candidates.sort_values(score_col, ascending=False).head(5).copy()
    rank_cols = [
        'sigma_rank',
        'amp_rank',
        'ret1_rank',
        'ret5_rank',
        'liq_rank',
        'downside_beta60_rank',
        'max_drawdown20_rank',
        'tail_risk_flag',
        'extreme_momo_flag',
        'overheat_flag',
        'reversal_flag',
    ]
    rank_maps = {col: dict(zip(score_df['stock_id'], score_df[col])) for col in rank_cols if col in score_df.columns}

    def mapped_mean(col, default=0.0):
        if col in top.columns:
            return float(pd.to_numeric(top[col], errors='coerce').fillna(default).mean())
        if col in rank_maps:
            return float(pd.to_numeric(top['stock_id'].map(rank_maps[col]), errors='coerce').fillna(default).mean())
        return float(default)

    def mapped_sum(col):
        if col in top.columns:
            return int(pd.to_numeric(top[col], errors='coerce').fillna(0).astype(bool).sum())
        if col in rank_maps:
            return int(pd.to_numeric(top['stock_id'].map(rank_maps[col]), errors='coerce').fillna(0).astype(bool).sum())
        return 0

    defensive_overlap = 0
    if 'score_defensive_v2' in score_df.columns:
        defensive_top20 = set(score_df.nlargest(min(20, len(score_df)), 'score_defensive_v2')['stock_id'])
        defensive_overlap = len(set(top['stock_id']) & defensive_top20)

    balanced_top10_overlap = 0
    conservative_top10_overlap = 0
    if 'score_balanced' in score_df.columns:
        balanced_top10 = set(score_df.nlargest(min(10, len(score_df)), 'score_balanced')['stock_id'])
        balanced_top10_overlap = len(set(top['stock_id']) & balanced_top10)
    if 'score_conservative_softrisk_v2' in score_df.columns:
        conservative_top10 = set(score_df.nlargest(min(10, len(score_df)), 'score_conservative_softrisk_v2')['stock_id'])
        conservative_top10_overlap = len(set(top['stock_id']) & conservative_top10)

    score_gap_5_10 = 0.0
    if len(candidates) >= 10:
        sorted_scores = candidates.sort_values(score_col, ascending=False)[score_col].reset_index(drop=True)
        score_gap_5_10 = float(sorted_scores.iloc[:5].mean() - sorted_scores.iloc[5:10].mean())

    top5_disagreement = float(top.get('rank_disagreement', pd.Series(0.0, index=top.index)).mean())
    top5_max_sigma_rank = float(pd.to_numeric(top.get('sigma_rank', pd.Series(1.0, index=top.index)), errors='coerce').fillna(1.0).max())
    top5_max_downside_beta60_rank = float(pd.to_numeric(top.get('downside_beta60_rank', pd.Series(1.0, index=top.index)), errors='coerce').fillna(1.0).max())
    top5_max_drawdown20_rank = float(pd.to_numeric(top.get('max_drawdown20_rank', pd.Series(1.0, index=top.index)), errors='coerce').fillna(1.0).max())
    top5_overheat_count = mapped_sum('overheat_flag')
    top5_reversal_count = mapped_sum('reversal_flag')
    branch_risk_score = (
        0.18 * mapped_mean('sigma_rank', default=1.0)
        + 0.15 * top5_max_sigma_rank
        + 0.15 * top5_disagreement
        + 0.15 * top5_max_downside_beta60_rank
        + 0.15 * top5_max_drawdown20_rank
        + 0.12 * (top5_overheat_count / 5.0)
        + 0.10 * (top5_reversal_count / 5.0)
    )

    return {
        'branch': branch_name,
        'available': len(top) >= 5,
        'score_gap': _score_gap(candidates, score_col=score_col),
        'score_gap_5_10': score_gap_5_10,
        'top5_disagreement': top5_disagreement,
        'top5_max_disagreement': mapped_mean('rank_disagreement', default=1.0) if 'rank_disagreement' not in top.columns else float(pd.to_numeric(top['rank_disagreement'], errors='coerce').fillna(1.0).max()),
        'top20_overlap': _top_overlap(score_df),
        'top5_sigma_rank_mean': mapped_mean('sigma_rank', default=1.0),
        'top5_max_sigma_rank': top5_max_sigma_rank,
        'top5_amp_rank_mean': mapped_mean('amp_rank', default=1.0),
        'top5_max_amp_rank': float(pd.to_numeric(top.get('amp_rank', pd.Series(1.0, index=top.index)), errors='coerce').fillna(1.0).max()),
        'top5_ret1_rank_mean': mapped_mean('ret1_rank', default=0.0),
        'top5_ret5_rank_mean': mapped_mean('ret5_rank', default=0.0),
        'top5_min_liq_rank': float(pd.to_numeric(top.get('liq_rank', pd.Series(0.0, index=top.index)), errors='coerce').fillna(0.0).min()),
        'top5_max_downside_beta60_rank': top5_max_downside_beta60_rank,
        'top5_max_drawdown20_rank': top5_max_drawdown20_rank,
        'top5_tail_risk_count': mapped_sum('tail_risk_flag'),
        'top5_high_vol_count': int(((pd.to_numeric(top.get('sigma_rank', pd.Series(0.0, index=top.index)), errors='coerce').fillna(0.0) > 0.85) | (pd.to_numeric(top.get('amp_rank', pd.Series(0.0, index=top.index)), errors='coerce').fillna(0.0) > 0.85)).sum()),
        'top5_very_high_vol_count': int(((pd.to_numeric(top.get('sigma_rank', pd.Series(0.0, index=top.index)), errors='coerce').fillna(0.0) > 0.95) | (pd.to_numeric(top.get('amp_rank', pd.Series(0.0, index=top.index)), errors='coerce').fillna(0.0) > 0.95)).sum()),
        'top5_very_tail_count': int(((pd.to_numeric(top.get('downside_beta60_rank', pd.Series(0.0, index=top.index)), errors='coerce').fillna(0.0) > 0.90) | (pd.to_numeric(top.get('max_drawdown20_rank', pd.Series(0.0, index=top.index)), errors='coerce').fillna(0.0) > 0.90)).sum()),
        'top5_extreme_momo_count': mapped_sum('extreme_momo_flag'),
        'top5_overheat_count': top5_overheat_count,
        'top5_reversal_count': top5_reversal_count,
        'branch_risk_score': float(branch_risk_score),
        'defensive_overlap': defensive_overlap,
        'balanced_top10_overlap': balanced_top10_overlap,
        'conservative_top10_overlap': conservative_top10_overlap,
        'risk_screen_ok': True,
        'top5': ','.join(top['stock_id'].astype(str).tolist()),
    }


def _market_hard_veto(regime, regime_stats):
    b5 = float(regime_stats.get('breadth_5d', 0.0))
    m1 = float(regime_stats.get('median_ret1', 0.0))
    m5 = float(regime_stats.get('median_ret5', 0.0))
    hvr = float(regime_stats.get('high_vol_ratio', 0.0))

    if regime == 'risk_off':
        if b5 < 0.45 and m5 < 0.0:
            return True, 'veto_risk_off_weak_5d'
        if m1 <= -0.0085 and m5 <= 0.0:
            return True, 'veto_risk_off_falling_knife'

    if regime == 'mixed_defensive':
        if b5 >= 0.60 and m5 >= 0.010 and m1 < 0.001:
            return True, 'veto_mixed_rebound_exhaustion'

    if hvr >= 0.40 and m1 < 0.0 and m5 <= 0.005:
        return True, 'veto_high_vol_weak_market'

    return False, ''


def _market_enable_gate(regime, regime_stats):
    b1 = float(regime_stats.get('breadth_1d', 0.0))
    b5 = float(regime_stats.get('breadth_5d', 0.0))
    m1 = float(regime_stats.get('median_ret1', 0.0))
    m5 = float(regime_stats.get('median_ret5', 0.0))
    hvr = float(regime_stats.get('high_vol_ratio', 0.0))

    if (
        regime == 'risk_on_strict'
        and b1 >= 0.60
        and b5 >= 0.60
        and m1 >= 0.0
        and m5 >= 0.010
    ):
        return True, 'enable_risk_on_trend'

    if (
        regime == 'risk_off'
        and b5 >= 0.50
        and m5 >= 0.0
        and m1 > -0.008
        and b1 <= 0.45
        and hvr <= 0.35
    ):
        return True, 'enable_risk_off_rebound'

    if (
        regime == 'mixed_defensive'
        and b5 <= 0.35
        and m5 < 0.0
        and b1 >= 0.35
        and m1 > -0.006
        and hvr <= 0.40
    ):
        return True, 'enable_mixed_washout_rebound'

    if (
        regime == 'neutral_positive'
        and b5 >= 0.55
        and m5 >= 0.005
        and m1 > -0.003
        and hvr <= 0.35
    ):
        return True, 'enable_neutral_quality'

    return False, 'market_not_supported'


def _union_selected_rows(candidates):
    if len(candidates) == 0:
        return candidates.copy()
    return candidates.sort_values('score', ascending=False).head(5).copy()


def _union_gate_diag(candidates, branch_diag):
    top = _union_selected_rows(candidates)
    full_pool = candidates.attrs.get('candidate_pool', candidates) if hasattr(candidates, 'attrs') else candidates
    if len(top) == 0:
        out = dict(branch_diag)
        out.update({
            'alpha_exception_count': 0,
            'stable_fill_count': 0,
            'max_alpha_score_z': 0.0,
            'selected_branch_count': 0,
            'avg_consensus_count': 0.0,
            'top5_branch_only_alpha_count': 5,
        })
        return out

    alpha_score = _safe_numeric(top, 'alpha_score')
    full_alpha = _safe_numeric(full_pool, 'alpha_score')
    max_alpha_z = float((alpha_score.max() - full_alpha.mean()) / (full_alpha.std(ddof=0) + 1e-9))
    alpha_lcb = _safe_numeric(top, 'alpha_lcb', default=float(alpha_score.mean()) if len(alpha_score) else 0.0)
    full_alpha_lcb = _safe_numeric(full_pool, 'alpha_lcb', default=float(full_alpha.mean()) if len(full_alpha) else 0.0)
    max_alpha_lcb_z = float((alpha_lcb.max() - full_alpha_lcb.mean()) / (full_alpha_lcb.std(ddof=0) + 1e-9))
    alpha_lcb_z = _safe_numeric(top, 'alpha_lcb_z')
    alpha_lcb_count = int((alpha_lcb_z >= 1.25).sum())
    score_gap = float(branch_diag.get('score_gap', 0.0))
    score_gap_5_10 = float(branch_diag.get('score_gap_5_10', 0.0))
    if len(full_pool) >= 10 and 'rerank_score' in full_pool.columns:
        sorted_scores = pd.to_numeric(full_pool.sort_values('rerank_score', ascending=False)['rerank_score'], errors='coerce').fillna(0.0).reset_index(drop=True)
        score_gap = float(sorted_scores.iloc[:5].mean() - sorted_scores.iloc[5:min(20, len(sorted_scores))].mean())
        score_gap_5_10 = float(sorted_scores.iloc[:5].mean() - sorted_scores.iloc[5:10].mean())
    branch_sources = set()
    if 'branch_sources' in top.columns:
        for text in top['branch_sources'].fillna('').astype(str):
            branch_sources.update([item for item in text.split('|') if item])

    alpha_flags = top.get('alpha_exception', pd.Series(False, index=top.index)).astype(bool)
    risk_top = top[~alpha_flags].copy()
    if len(risk_top) == 0:
        risk_top = top.copy()
    risk_disagreement = float(pd.to_numeric(risk_top.get('rank_disagreement', pd.Series(0.0, index=risk_top.index)), errors='coerce').fillna(0.0).mean())
    risk_max_sigma = float(pd.to_numeric(risk_top.get('sigma_rank', pd.Series(0.0, index=risk_top.index)), errors='coerce').fillna(0.0).max())
    risk_max_downside = float(pd.to_numeric(risk_top.get('downside_beta60_rank', pd.Series(0.0, index=risk_top.index)), errors='coerce').fillna(0.0).max())
    risk_max_drawdown = float(pd.to_numeric(risk_top.get('max_drawdown20_rank', pd.Series(0.0, index=risk_top.index)), errors='coerce').fillna(0.0).max())
    risk_overheat_count = int(risk_top.get('overheat_flag', pd.Series(False, index=risk_top.index)).astype(bool).sum())
    risk_reversal_count = int(risk_top.get('reversal_flag', pd.Series(False, index=risk_top.index)).astype(bool).sum())
    branch_risk_score = (
        0.18 * float(pd.to_numeric(risk_top.get('sigma_rank', pd.Series(0.0, index=risk_top.index)), errors='coerce').fillna(0.0).mean())
        + 0.15 * risk_max_sigma
        + 0.15 * risk_disagreement
        + 0.15 * risk_max_downside
        + 0.15 * risk_max_drawdown
        + 0.12 * (risk_overheat_count / 5.0)
        + 0.10 * (risk_reversal_count / 5.0)
    )

    out = dict(branch_diag)
    out.update({
        'alpha_exception_count': int(alpha_flags.sum()),
        'alpha_lcb_count': alpha_lcb_count,
        'stable_fill_count': int(top.get('stable_candidate', pd.Series(False, index=top.index)).astype(bool).sum()),
        'max_alpha_score_z': max_alpha_z,
        'max_alpha_lcb_z': max_alpha_lcb_z,
        'mean_alpha_lcb': float(alpha_lcb.mean()),
        'score_gap': score_gap,
        'score_gap_5_10': score_gap_5_10,
        'selected_branch_count': int(len(branch_sources)),
        'avg_consensus_count': float(pd.to_numeric(top.get('consensus_count', pd.Series(0.0, index=top.index)), errors='coerce').fillna(0.0).mean()),
        'top5_tail_risk_count': int(risk_top.get('tail_risk_flag', pd.Series(False, index=risk_top.index)).astype(bool).sum()),
        'top5_high_vol_count': int(risk_top.get('high_vol_flag', pd.Series(False, index=risk_top.index)).astype(bool).sum()),
        'top5_very_high_vol_count': int(risk_top.get('very_high_vol_flag', pd.Series(False, index=risk_top.index)).astype(bool).sum()),
        'top5_very_tail_count': int(risk_top.get('very_tail_flag', pd.Series(False, index=risk_top.index)).astype(bool).sum()),
        'top5_branch_only_alpha_count': int(risk_top.get('branch_only_alpha_flag', pd.Series(False, index=risk_top.index)).astype(bool).sum()),
        'top5_disagreement': risk_disagreement,
        'branch_risk_score': float(branch_risk_score),
    })
    if hasattr(candidates, 'attrs') and candidates.attrs.get('combo_info'):
        out.update(candidates.attrs.get('combo_info', {}))
    return out


def _alpha_quality_gate(union_diag, regime):
    alpha_exception_count = int(union_diag.get('alpha_exception_count', 0))
    alpha_lcb_count = int(union_diag.get('alpha_lcb_count', alpha_exception_count))
    clean_alpha_count = max(alpha_exception_count, alpha_lcb_count)
    stable_fill_count = int(union_diag.get('stable_fill_count', 0))
    max_alpha_z = float(union_diag.get('max_alpha_score_z', 0.0))
    max_alpha_lcb_z = float(union_diag.get('max_alpha_lcb_z', max_alpha_z))
    score_gap = float(union_diag.get('score_gap', 0.0))
    score_gap_5_10 = float(union_diag.get('score_gap_5_10', 0.0))
    selected_branch_count = int(union_diag.get('selected_branch_count', 0))
    avg_consensus_count = float(union_diag.get('avg_consensus_count', 0.0))

    alpha_strong = max(max_alpha_z, max_alpha_lcb_z) >= 1.50 or score_gap >= 0.080 or score_gap_5_10 >= 0.040
    consensus_ok = selected_branch_count >= 3 or avg_consensus_count >= 2.0

    if regime == 'risk_off':
        return (
            alpha_exception_count >= 2
            and stable_fill_count >= 2
            and alpha_strong
            and consensus_ok
        )
    if regime == 'mixed_defensive':
        return (
            clean_alpha_count >= 1
            and stable_fill_count >= 1
            and alpha_strong
            and consensus_ok
        )
    if regime == 'risk_on_strict':
        return clean_alpha_count >= 1 and alpha_strong
    if regime == 'neutral_positive':
        return (
            clean_alpha_count >= 1
            and stable_fill_count >= 1
            and alpha_strong
            and consensus_ok
        )
    return False


def _portfolio_risk_gate(union_diag, regime):
    tail = int(union_diag.get('top5_tail_risk_count', 5))
    very_tail = int(union_diag.get('top5_very_tail_count', 5))
    high_vol = int(union_diag.get('top5_high_vol_count', 5))
    very_high_vol = int(union_diag.get('top5_very_high_vol_count', 5))
    branch_only = int(union_diag.get('top5_branch_only_alpha_count', 5))
    disagreement = float(union_diag.get('top5_disagreement', 1.0))
    branch_risk_score = float(union_diag.get('branch_risk_score', 1.0))

    if regime == 'risk_off':
        return (
            tail <= 2
            and very_tail <= 1
            and high_vol <= 2
            and very_high_vol <= 1
            and branch_only <= 1
            and disagreement <= 0.18
            and branch_risk_score <= 0.72
        )
    if regime == 'mixed_defensive':
        return (
            tail <= 2
            and very_tail <= 1
            and high_vol <= 2
            and very_high_vol <= 1
            and branch_only <= 1
            and disagreement <= 0.12
            and branch_risk_score <= 0.75
        )
    if regime == 'risk_on_strict':
        return (
            tail <= 3
            and very_tail <= 1
            and very_high_vol <= 2
            and branch_only <= 2
            and branch_risk_score <= 0.82
        )
    if regime == 'neutral_positive':
        return (
            tail <= 2
            and very_tail <= 1
            and high_vol <= 2
            and very_high_vol <= 1
            and branch_risk_score <= 0.75
        )
    return False


def _fallback_branch_for_gate(selector_cfg, regime, reason):
    gate_cfg = selector_cfg.get('gated_union_rerank', {})
    if regime == 'risk_off' and reason in {'veto_risk_off_weak_5d', 'veto_risk_off_falling_knife'}:
        return gate_cfg.get('secondary_fallback_branch', 'defensive_v2_strict')
    if regime == 'risk_on_strict' and reason == 'alpha_quality_failed_risk_on':
        return gate_cfg.get('secondary_fallback_branch', 'defensive_v2_strict')
    return gate_cfg.get('default_fallback_branch', 'legal_minrisk_hardened')


def _route_fallback_branch(chosen_name, diagnostics, selector_cfg, regime, regime_stats):
    router_cfg = selector_cfg.get('fallback_router', {})
    if not router_cfg.get('enabled', False):
        return chosen_name, {}

    default_branch = router_cfg.get('default_branch', selector_cfg.get('fallback_branch', 'legal_minrisk_hardened'))
    secondary_branch = router_cfg.get('secondary_branch', 'defensive_v2_strict')
    if chosen_name not in {default_branch, secondary_branch}:
        return chosen_name, {'fallback_router_reason': 'not_fallback_branch'}

    by_branch = {row.get('branch'): row for row in diagnostics if row.get('available', False)}
    default_diag = by_branch.get(default_branch)
    secondary_diag = by_branch.get(secondary_branch)
    if default_diag is None or secondary_diag is None:
        return chosen_name, {'fallback_router_reason': 'missing_fallback_diag'}

    market_toxic = (
        regime == 'risk_off'
        and (
            float(regime_stats.get('median_ret1', 0.0)) <= -0.006
            or (
                float(regime_stats.get('breadth_5d', 1.0)) < 0.45
                and float(regime_stats.get('median_ret5', 0.0)) < 0.0
            )
            or (
                float(regime_stats.get('high_vol_ratio', 0.0)) >= 0.40
                and float(regime_stats.get('median_ret1', 0.0)) < 0.0
            )
        )
    )
    if router_cfg.get('require_market_toxic', True) and not market_toxic:
        return chosen_name, {'fallback_router_reason': 'market_not_toxic'}

    default_risk = float(default_diag.get('branch_risk_score', 1.0))
    secondary_risk = float(secondary_diag.get('branch_risk_score', 1.0))
    min_improvement = float(router_cfg.get('min_risk_improvement', 0.08))
    secondary_cleaner = (
        secondary_risk + min_improvement < default_risk
        and int(secondary_diag.get('top5_tail_risk_count', 5)) <= int(default_diag.get('top5_tail_risk_count', 5))
        and int(secondary_diag.get('top5_very_tail_count', 5)) <= int(default_diag.get('top5_very_tail_count', 5))
        and float(secondary_diag.get('top5_disagreement', 1.0)) <= float(default_diag.get('top5_disagreement', 1.0)) + 0.08
    )
    if chosen_name == default_branch and secondary_cleaner:
        return secondary_branch, {
            'fallback_router_reason': 'secondary_cleaner_in_toxic_market',
            'fallback_router_default_risk': default_risk,
            'fallback_router_secondary_risk': secondary_risk,
        }

    return chosen_name, {
        'fallback_router_reason': 'keep_default',
        'fallback_router_default_risk': default_risk,
        'fallback_router_secondary_risk': secondary_risk,
    }


def _evaluate_gated_union(union_candidates, union_diag, regime, regime_stats, selector_cfg):
    gate_cfg = selector_cfg.get('gated_union_rerank', {})
    if not gate_cfg.get('enabled', False):
        return True, 'gate_disabled', 'independent_union_rerank', {}

    veto, veto_reason = _market_hard_veto(regime, regime_stats)
    gate_debug = {
        'market_veto': bool(veto),
        'market_veto_reason': veto_reason,
    }
    if veto:
        fallback = _fallback_branch_for_gate(selector_cfg, regime, veto_reason)
        gate_debug.update({
            'market_enable': False,
            'market_enable_reason': veto_reason,
            'alpha_quality_ok': False,
            'portfolio_risk_ok': False,
            'chosen_after_gate': fallback,
            'fallback_reason': veto_reason,
        })
        return False, veto_reason, fallback, gate_debug

    market_ok, market_reason = _market_enable_gate(regime, regime_stats)
    gate_debug.update({
        'market_enable': bool(market_ok),
        'market_enable_reason': market_reason,
    })
    if not market_ok:
        fallback = _fallback_branch_for_gate(selector_cfg, regime, market_reason)
        gate_debug.update({
            'alpha_quality_ok': False,
            'portfolio_risk_ok': False,
            'chosen_after_gate': fallback,
            'fallback_reason': market_reason,
        })
        return False, market_reason, fallback, gate_debug

    gate_diag = _union_gate_diag(union_candidates, union_diag)
    alpha_ok = _alpha_quality_gate(gate_diag, regime)
    gate_debug.update({
        'alpha_quality_ok': bool(alpha_ok),
        'alpha_exception_count': gate_diag.get('alpha_exception_count', 0),
        'alpha_lcb_count': gate_diag.get('alpha_lcb_count', gate_diag.get('alpha_exception_count', 0)),
        'stable_fill_count': gate_diag.get('stable_fill_count', 0),
        'max_alpha_score_z': gate_diag.get('max_alpha_score_z', 0.0),
        'max_alpha_lcb_z': gate_diag.get('max_alpha_lcb_z', gate_diag.get('max_alpha_score_z', 0.0)),
        'mean_alpha_lcb': gate_diag.get('mean_alpha_lcb', 0.0),
        'selected_branch_count': gate_diag.get('selected_branch_count', 0),
        'avg_consensus_count': gate_diag.get('avg_consensus_count', 0.0),
        'combo_search_score': gate_diag.get('combo_search_score', np.nan),
        'combo_search_feasible': gate_diag.get('combo_search_feasible', 0),
    })
    if not alpha_ok:
        reason = 'alpha_quality_failed_risk_on' if regime == 'risk_on_strict' else 'alpha_quality_failed'
        fallback = _fallback_branch_for_gate(selector_cfg, regime, reason)
        gate_debug.update({
            'portfolio_risk_ok': False,
            'chosen_after_gate': fallback,
            'fallback_reason': reason,
        })
        return False, reason, fallback, gate_debug

    risk_ok = _portfolio_risk_gate(gate_diag, regime)
    gate_debug.update({
        'portfolio_risk_ok': bool(risk_ok),
        'top5_tail_risk_count': gate_diag.get('top5_tail_risk_count', 0),
        'top5_very_tail_count': gate_diag.get('top5_very_tail_count', 0),
        'top5_high_vol_count': gate_diag.get('top5_high_vol_count', 0),
        'top5_very_high_vol_count': gate_diag.get('top5_very_high_vol_count', 0),
        'top5_branch_only_alpha_count': gate_diag.get('top5_branch_only_alpha_count', 0),
        'top5_disagreement': gate_diag.get('top5_disagreement', 0.0),
        'branch_risk_score': gate_diag.get('branch_risk_score', 0.0),
    })
    if not risk_ok:
        fallback = _fallback_branch_for_gate(selector_cfg, regime, 'portfolio_risk_budget_failed')
        gate_debug.update({
            'chosen_after_gate': fallback,
            'fallback_reason': 'portfolio_risk_budget_failed',
        })
        return False, 'portfolio_risk_budget_failed', fallback, gate_debug

    gate_debug.update({
        'chosen_after_gate': 'independent_union_rerank',
        'fallback_reason': '',
    })
    return True, market_reason, 'independent_union_rerank', gate_debug


def _diag_passes(diag, regime, regime_stats, selector_cfg, branch_name):
    if not diag['available']:
        return False

    if branch_name == 'independent_union_rerank':
        return True

    if branch_name == 'legal_plus_1alpha':
        return True

    if branch_name == 'safe_union_2slot':
        return True

    if branch_name in {'legal_minrisk', 'legal_minrisk_hardened'}:
        if branch_name == 'legal_minrisk_hardened' and regime == 'risk_off':
            market_5d_weak = float(regime_stats.get('median_ret5', 0.0)) < 0.0
            market_1d_toxic = float(regime_stats.get('median_ret1', 0.0)) <= -0.010
            legal_risk = float(diag.get('branch_risk_score', 0.0))
            if (market_5d_weak and legal_risk >= 0.30) or (market_1d_toxic and legal_risk >= 0.18):
                return False
        return True

    budget_cfg = selector_cfg.get('branch_risk_budget', {})
    risk_budget_ok = (
        not budget_cfg.get('enabled', False)
        or float(diag.get('branch_risk_score', 1.0)) <= float(budget_cfg.get(regime, 1.0))
    )

    if branch_name in {'lgb_only', 'lgb_only_guarded'}:
        return (
            regime == 'risk_on_strict'
            and risk_budget_ok
            and regime_stats.get('breadth_1d', 0.0) >= 0.62
            and regime_stats.get('breadth_5d', 0.0) >= 0.58
            and regime_stats.get('median_ret1', 0.0) > 0.0
            and regime_stats.get('median_ret5', 0.0) > 0.008
            and diag['top5_tail_risk_count'] == 0
            and diag['top5_reversal_count'] == 0
            and diag['top5_high_vol_count'] <= 1
            and diag['top5_very_high_vol_count'] == 0
            and diag['top5_disagreement'] <= 0.18
            and diag['top5_max_disagreement'] <= 0.28
            and diag['top5_min_liq_rank'] >= 0.15
            and diag['top5_max_downside_beta60_rank'] <= 0.80
            and diag['top5_max_drawdown20_rank'] <= 0.80
            and max(diag['balanced_top10_overlap'], diag['conservative_top10_overlap']) >= 3
            and diag['score_gap_5_10'] >= 0.04
        )

    if branch_name in {'conservative_blend', 'conservative_softrisk', 'conservative_softrisk_v2', 'conservative_softrisk_v2_strict'}:
        strict = branch_name == 'conservative_softrisk_v2_strict'
        if regime == 'risk_off':
            return (
                not strict
                and regime_stats.get('breadth_5d', 0.0) >= 0.50
                and regime_stats.get('median_ret5', 0.0) >= 0.0
                and float(diag.get('branch_risk_score', 1.0)) <= 0.70
                and diag['top5_tail_risk_count'] <= 4
                and diag['top5_reversal_count'] <= 1
                and diag['top5_disagreement'] <= 0.08
                and diag['top5_min_liq_rank'] >= 0.10
            )
        if regime == 'mixed_defensive':
            return (
                strict
                and risk_budget_ok
                and diag['top5_tail_risk_count'] == 0
                and diag['top5_reversal_count'] == 0
                and diag['top5_disagreement'] <= 0.16
                and diag['top5_max_disagreement'] <= 0.24
                and diag['top5_max_sigma_rank'] <= 0.80
                and diag['top5_max_downside_beta60_rank'] <= 0.78
                and diag['top5_max_drawdown20_rank'] <= 0.78
                and diag['top5_min_liq_rank'] >= 0.15
            )
        if regime == 'neutral_positive':
            return (
                strict
                and risk_budget_ok
                and diag['top5_tail_risk_count'] <= 1
                and diag['top5_reversal_count'] <= 1
                and diag['top5_disagreement'] <= 0.20
                and diag['top5_max_disagreement'] <= 0.30
                and diag['top5_min_liq_rank'] >= 0.10
            )
        return (
            strict
            and risk_budget_ok
            and diag['top5_tail_risk_count'] <= 1
            and diag['top5_reversal_count'] <= 1
            and diag['top5_disagreement'] <= 0.20
            and diag['top5_max_disagreement'] <= 0.30
            and diag['top5_min_liq_rank'] >= 0.10
        )

    if branch_name in {'balanced_blend', 'balanced_guarded'}:
        if regime not in {'risk_on_strict', 'neutral_positive'}:
            return False
        base_ok = (
            risk_budget_ok
            and diag['top5_tail_risk_count'] <= 1
            and diag['top5_reversal_count'] <= 1
            and diag['top5_very_tail_count'] == 0
            and diag['top5_disagreement'] <= 0.22
            and diag['top5_max_disagreement'] <= 0.32
            and diag['top5_min_liq_rank'] >= 0.12
            and diag['top5_max_downside_beta60_rank'] <= 0.85
            and diag['top5_max_drawdown20_rank'] <= 0.85
        )
        if regime == 'neutral_positive':
            base_ok = base_ok and (
                diag['top5_tail_risk_count'] == 0
                or diag['top5_max_disagreement'] <= 0.25
            )
        return base_ok

    if branch_name in {'stable_top30_rerank_trend'}:
        return (
            regime in {'risk_on_strict', 'neutral_positive'}
            and diag['top5_tail_risk_count'] <= 3
            and diag['top5_reversal_count'] <= 2
        )

    if branch_name in {'defensive_v2', 'defensive_branch', 'defensive_v2_strict'}:
        return (
            regime in {'risk_off', 'mixed_defensive'}
            and diag['score_gap'] >= 0.06
            and diag['top5_tail_risk_count'] <= 1
            and diag['top5_disagreement'] <= (0.80 if regime == 'risk_off' else 0.70)
        )

    return False


META_GATE_NUMERIC_FEATURES = [
    'score_gap',
    'score_gap_5_10',
    'top20_overlap',
    'top5_disagreement',
    'top5_max_disagreement',
    'top5_sigma_rank_mean',
    'top5_max_sigma_rank',
    'top5_amp_rank_mean',
    'top5_max_amp_rank',
    'top5_ret1_rank_mean',
    'top5_ret5_rank_mean',
    'top5_min_liq_rank',
    'top5_max_downside_beta60_rank',
    'top5_max_drawdown20_rank',
    'top5_tail_risk_count',
    'top5_high_vol_count',
    'top5_very_high_vol_count',
    'top5_very_tail_count',
    'top5_extreme_momo_count',
    'top5_overheat_count',
    'top5_reversal_count',
    'branch_risk_score',
    'defensive_overlap',
    'balanced_top10_overlap',
    'conservative_top10_overlap',
    'breadth_1d',
    'breadth_5d',
    'median_ret1',
    'median_ret5',
    'high_vol_ratio',
]


def _load_meta_gate(selector_cfg):
    gate_cfg = selector_cfg.get('meta_gate', {})
    if not gate_cfg.get('enabled', False):
        return None

    artifact_path = gate_cfg.get('artifact_path')
    if not artifact_path:
        artifact_path = os.path.join(config['output_dir'], 'branch_meta_gate.pkl')
    if not os.path.exists(artifact_path):
        print(f'[BDC][selector] meta_gate_missing={artifact_path}')
        return None

    try:
        artifact = joblib.load(artifact_path)
        artifact['artifact_path'] = artifact_path
        return artifact
    except Exception as exc:
        print(f'[BDC][selector] meta_gate_load_failed={artifact_path}: {exc}')
        return None


def _build_meta_gate_features(rows, artifact):
    raw = pd.DataFrame(rows).copy()
    features = pd.DataFrame(index=raw.index)
    for col in META_GATE_NUMERIC_FEATURES:
        if col in raw.columns:
            features[col] = pd.to_numeric(raw[col], errors='coerce').fillna(0.0)
        else:
            features[col] = 0.0

    for branch in artifact.get('branches', []):
        features[f'branch={branch}'] = (raw.get('branch', '') == branch).astype(float)
    for regime in artifact.get('regimes', []):
        features[f'regime={regime}'] = (raw.get('regime', '') == regime).astype(float)

    feature_columns = artifact.get('feature_columns', list(features.columns))
    for col in feature_columns:
        if col not in features.columns:
            features[col] = 0.0
    return features[feature_columns].astype(np.float64)


def _score_meta_gate(rows, artifact, selector_cfg, fallback_branch):
    if artifact is None or not rows:
        return []

    X = _build_meta_gate_features(rows, artifact)
    safe_model = artifact.get('safe_model')
    score_model = artifact.get('score_model')
    if safe_model is None or score_model is None:
        return []

    if hasattr(safe_model, 'predict_proba'):
        proba = safe_model.predict_proba(X)
        safe_prob = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    else:
        safe_prob = np.asarray(safe_model.predict(X), dtype=np.float64)
    pred_score = np.asarray(score_model.predict(X), dtype=np.float64)

    gate_cfg = selector_cfg.get('meta_gate', {})
    safe_threshold = float(gate_cfg.get('safe_threshold', 0.70))
    unsafe_penalty = float(gate_cfg.get('unsafe_penalty', 0.02))
    risk_penalty = float(gate_cfg.get('risk_penalty', 0.0))
    require_rule_pass = bool(gate_cfg.get('require_rule_pass', False))

    scored = []
    for row, p_safe, p_score in zip(rows, safe_prob, pred_score):
        branch_name = row.get('branch')
        gate_allowed = float(p_safe) >= safe_threshold
        if require_rule_pass:
            gate_allowed = gate_allowed and bool(row.get('rule_passes', False))
        if branch_name == fallback_branch:
            gate_allowed = True

        utility = (
            float(p_score)
            - unsafe_penalty * (1.0 - float(p_safe))
            - risk_penalty * float(row.get('branch_risk_score', 0.0))
        )
        scored.append({
            'branch': branch_name,
            'meta_safe_prob': float(p_safe),
            'meta_pred_score': float(p_score),
            'meta_utility': float(utility),
            'meta_allowed': bool(gate_allowed),
        })
    return scored


def choose_selector_branch(score_df, selector_cfg):
    regime, regime_stats = infer_regime(score_df)
    forced = selector_cfg.get('force_branch')
    fallback_branch = selector_cfg.get('fallback_branch', 'legal_minrisk')
    branch_order = selector_cfg.get('regime_branch_order', {}).get(regime, ['lgb_only_guarded', fallback_branch])
    if forced:
        branch_order = [forced]
    elif fallback_branch not in branch_order:
        branch_order = list(branch_order) + [fallback_branch]

    diagnostics = []
    chosen_name = None
    chosen_candidates = None
    chosen_cfg = {}
    branch_payloads = {}
    gate_debug_rows = []
    fallback_router_info = {}

    for branch_name in branch_order:
        candidates, branch_cfg = _branch_candidates(score_df, selector_cfg, branch_name, regime=regime)
        diag = _branch_diag(score_df, candidates, branch_name)
        diag['filter'] = branch_cfg.get('filter', 'unavailable')
        diag['score_col'] = branch_cfg.get('score_col', 'unavailable')
        diag.update(regime_stats)
        diag['regime'] = regime
        if hasattr(candidates, 'attrs') and candidates.attrs.get('legal_plus_info'):
            diag.update(candidates.attrs.get('legal_plus_info', {}))
        if hasattr(candidates, 'attrs') and candidates.attrs.get('safe_union_info'):
            diag.update(candidates.attrs.get('safe_union_info', {}))
        diag['rule_passes'] = _diag_passes(diag, regime, regime_stats, selector_cfg, branch_name)
        diag['passes'] = diag['rule_passes']

        if branch_name == 'independent_union_rerank' and diag['rule_passes']:
            gate_ok, gate_reason, gate_branch, gate_debug = _evaluate_gated_union(
                candidates,
                diag,
                regime,
                regime_stats,
                selector_cfg,
            )
            diag.update(gate_debug)
            diag['gate_reason'] = gate_reason
            diag['passes'] = bool(gate_ok)
            diag['rule_passes'] = bool(gate_ok)
            gate_debug_row = dict(gate_debug)
            gate_debug_row.update(regime_stats)
            gate_debug_row['regime'] = regime
            gate_debug_row['branch'] = branch_name
            gate_debug_row['final_picks'] = diag.get('top5', '')
            gate_debug_rows.append(gate_debug_row)
            if not gate_ok and gate_branch not in branch_order:
                branch_order = list(branch_order) + [gate_branch]

        diagnostics.append(diag)
        branch_payloads[branch_name] = (candidates, branch_cfg)

        if diag['passes'] and chosen_name is None:
            chosen_name = branch_name
            chosen_candidates = candidates
            chosen_cfg = branch_cfg
            if forced:
                break

    gate_artifact = None if forced else _load_meta_gate(selector_cfg)
    gate_scores = _score_meta_gate(diagnostics, gate_artifact, selector_cfg, fallback_branch)
    if gate_scores:
        gate_by_branch = {row['branch']: row for row in gate_scores}
        for diag in diagnostics:
            gate_row = gate_by_branch.get(diag['branch'])
            if gate_row:
                diag.update(gate_row)
                diag['passes'] = bool(gate_row['meta_allowed'])

        allowed = [
            diag for diag in diagnostics
            if diag.get('available', False) and diag.get('meta_allowed', False)
        ]
        if allowed:
            best = max(allowed, key=lambda row: row.get('meta_utility', -np.inf))
            chosen_name = best['branch']
            chosen_candidates, chosen_cfg = branch_payloads.get(chosen_name, (None, {}))

    if not forced and chosen_name is not None:
        routed_name, fallback_router_info = _route_fallback_branch(
            chosen_name,
            diagnostics,
            selector_cfg,
            regime,
            regime_stats,
        )
        if routed_name != chosen_name:
            routed_candidates, routed_cfg = branch_payloads.get(routed_name, (None, {}))
            if routed_candidates is None or len(routed_candidates) < 5:
                routed_candidates, routed_cfg = _branch_candidates(score_df, selector_cfg, routed_name, regime=regime)
                branch_payloads[routed_name] = (routed_candidates, routed_cfg)
            if routed_candidates is not None and len(routed_candidates) >= 5:
                chosen_name = routed_name
                chosen_candidates = routed_candidates
                chosen_cfg = routed_cfg

    if chosen_candidates is None or len(chosen_candidates) < 5:
        chosen_candidates, chosen_cfg = _branch_candidates(score_df, selector_cfg, fallback_branch, regime=regime)
        chosen_name = fallback_branch
        if chosen_candidates is None or len(chosen_candidates) < 5:
            emergency = selector_cfg.get('emergency_fallback_branch', 'legal_minrisk_hardened')
            chosen_candidates, chosen_cfg = _branch_candidates(score_df, selector_cfg, emergency, regime=regime)
            chosen_name = emergency
        diag = _branch_diag(score_df, chosen_candidates, chosen_name)
        diag['filter'] = chosen_cfg.get('filter', 'legal_minrisk_hardened')
        diag['score_col'] = chosen_cfg.get('score_col', 'score_legal_minrisk')
        diag['passes'] = len(chosen_candidates) >= 5
        diagnostics.append(diag)

    selector_info = {
        'selector_version': selector_cfg.get('version', 'unknown'),
        'regime': regime,
        'regime_stats': regime_stats,
        'chosen_branch': chosen_name,
        'diagnostics': diagnostics,
        'exposure_cap': float(chosen_cfg.get('exposure', 1.0)),
        'meta_gate_enabled': gate_artifact is not None,
        'meta_gate_artifact': gate_artifact.get('artifact_path') if gate_artifact else None,
        'gate_debug': gate_debug_rows,
        'fallback_router': fallback_router_info,
    }
    candidate_debug = chosen_candidates.attrs.get('candidate_debug') if hasattr(chosen_candidates, 'attrs') else None
    if candidate_debug is not None:
        selector_info['candidate_debug'] = candidate_debug.to_dict('records')
    return chosen_candidates, selector_info


def _resolve_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _get_autocast_context(device):
    amp_enabled = bool(config.get('predict', {}).get('amp', True)) and device.type == 'cuda'
    if not amp_enabled:
        return nullcontext(), False
    return torch.autocast(device_type='cuda', dtype=torch.float16), True


def _resolve_data_file():
    candidates = []
    env_data_file = os.environ.get('BDC_DATA_FILE')
    if env_data_file:
        candidates.append(env_data_file)

    candidates.extend([
        os.path.join('./model', 'input', 'train_hs300_latest.csv'),
        os.path.join(config['data_path'], 'train_hs300_latest.csv'),
        os.path.join(config['data_path'], 'train.csv'),
    ])

    for path in candidates:
        if path and os.path.exists(path):
            return path

    raise FileNotFoundError(
        '未找到推理数据文件，已检查: ' + ', '.join(candidates)
    )


def main():
    data_file = _resolve_data_file()
    model_dir = _resolve_model_dir()
    model_path = os.path.join(model_dir, 'best_model.pth')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    output_path = os.environ.get('BDC_OUTPUT_PATH', os.path.join('./output/', 'result.csv'))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'未找到模型文件: {model_path}')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')

    raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    raw_df = raw_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    latest_date = raw_df['日期'].max()

    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    scaler = joblib.load(scaler_path)
    artifacts = prepare_inference_artifacts(
        raw_df=raw_df,
        stockid2idx=stockid2idx,
        scaler=scaler,
        data_file=data_file,
        scaler_path=scaler_path,
        latest_date=latest_date,
    )

    features = artifacts['features']
    sequences_np = artifacts['sequences_np']
    sequence_stock_ids = artifacts['sequence_stock_ids']

    device = _resolve_device()
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    autocast_ctx, amp_enabled = _get_autocast_context(device)
    with torch.inference_mode():
        with autocast_ctx:
            x = torch.from_numpy(sequences_np).unsqueeze(0).to(device, non_blocking=device.type == 'cuda')
            transformer_scores = model(x).squeeze(0).float().cpu().numpy()

    inference_df = artifacts['inference_df']
    lgb_bundle = load_lgb_branches(model_dir)
    lgb_scores = predict_lgb_score(lgb_bundle, inference_df, config)
    lgb_components = predict_lgb_components(lgb_bundle, inference_df)
    blend_cfg = config.get('blend', {})
    score_mode = blend_cfg.get('score_mode', 'blend')
    norm_mode = blend_cfg.get('normalize', 'zscore')
    if lgb_scores is None:
        scores = transformer_scores
        score_source = 'transformer'
    elif score_mode in ('lgb', 'lgb_only'):
        scores = np.asarray(lgb_scores, dtype=np.float64)
        score_source = 'lgb_only'
    elif score_mode in ('transformer', 'transformer_only'):
        scores = np.asarray(transformer_scores, dtype=np.float64)
        score_source = 'transformer_only'
    elif score_mode == 'selector':
        scores = np.asarray(lgb_scores, dtype=np.float64)
        score_source = 'selector_pending'
    else:
        t_w = blend_cfg.get('transformer_weight', 0.55)
        lgb_w = blend_cfg.get('lgb_weight', 0.45)
        transformer_z = normalize_score(transformer_scores, norm_mode)
        lgb_z = normalize_score(lgb_scores, norm_mode)
        scores = t_w * transformer_z + lgb_w * lgb_z
        agreement_penalty = blend_cfg.get('agreement_penalty', 0.0)
        if agreement_penalty > 0:
            scores = scores - agreement_penalty * np.abs(transformer_z - lgb_z)
        score_source = f'transformer+lgb({t_w:.2f}/{lgb_w:.2f}, norm={norm_mode}, penalty={agreement_penalty:.2f})'

    score_df = pd.DataFrame({
        'stock_id': sequence_stock_ids,
        'score': scores,
        'transformer': transformer_scores,
        'lgb': np.asarray(lgb_scores, dtype=np.float64) if lgb_scores is not None else np.full(len(sequence_stock_ids), np.nan),
    })
    if lgb_components is not None:
        component_name_map = {
            'rank_score': 'lgb_rank_score',
            'reg_score': 'lgb_reg_score',
            'top5_rank_score': 'lgb_top5_score',
        }
        for raw_name, out_name in component_name_map.items():
            if raw_name in lgb_components:
                score_df[out_name] = np.asarray(lgb_components[raw_name], dtype=np.float64)
    score_df = score_df.merge(artifacts['risk_df'], on='stock_id', how='left')
    score_df['sigma20'] = score_df['sigma20'].fillna(score_df['sigma20'].median()).clip(lower=1e-4)
    score_df['median_amount20'] = score_df['median_amount20'].fillna(0.0)
    for col in [
        'ret1',
        'ret5',
        'ret10',
        'ret20',
        'amp20',
        'amp_mean10',
        'vol10',
        'pos20',
        'amount20',
        'mean_amount20',
        'turnover20',
        'amt_ratio5',
        'to_ratio5',
        'return_1',
        'return_5',
        'beta60',
        'downside_beta60',
        'idio_vol60',
        'max_drawdown20',
    ]:
        score_df[col] = score_df[col].fillna(score_df[col].median() if score_df[col].notna().any() else 0.0)

    post_cfg = config.get('postprocess', {})
    selector_cfg = config.get('selector', {})
    exp009_cfg = selector_cfg.get('exp009_meta', {})
    exp009_enabled = _env_bool('BDC_EXP009_META_ENABLED', exp009_cfg.get('enabled', False))
    grr_cfg = dict(config.get('grr_top5', {}))
    grr_enabled = _env_bool('BDC_GRR_TOP5_ENABLED', grr_cfg.get('enabled', False))
    grr_cfg['enabled'] = grr_enabled
    print(f'[BDC][predict] data_file={os.path.abspath(data_file)}')
    print(f'[BDC][predict] postprocess_filter={post_cfg.get("filter", "stable")}')
    print(f'[BDC][predict] selector_enabled={selector_cfg.get("enabled", False)}')
    print(f'[BDC][predict] output_path={os.path.abspath(output_path)}')
    print(f'[BDC][predict] grr_top5_enabled={grr_enabled}')
    print(f'[BDC][predict] exp009_meta_enabled={exp009_enabled}')
    if grr_cfg.get('enabled', False):
        grr_config = dict(config)
        grr_config['grr_top5'] = grr_cfg
        score_df = apply_grr_top5(score_df, grr_config)
        if 'grr_final_score' in score_df.columns:
            score_source = 'grr_top5:rrf+router'
    if exp009_enabled:
        exp009_dir = exp009_cfg.get('artifact_dir') or model_dir
        exp009_artifacts = load_exp009_artifacts(exp009_dir)
        if exp009_artifacts is None:
            print('[BDC][exp009] artifacts missing, fallback to existing score')
        else:
            before_cols = set(score_df.columns)
            score_df = apply_exp009_meta(score_df, exp009_artifacts)
            if 'exp009_final_score' in score_df.columns:
                score_df['score'] = score_df['exp009_final_score']
                score_source = 'exp009_meta_badaware'
            elif before_cols == set(score_df.columns):
                print('[BDC][exp009] apply skipped, fallback to existing score')
    selector_info = None
    if selector_cfg.get('enabled', False) and score_mode == 'selector' and lgb_scores is not None:
        score_df = add_selector_scores(score_df)
        filtered, selector_info = choose_selector_branch(score_df, selector_cfg)
        chosen_branch = selector_info['chosen_branch']
        branch_cfg = selector_cfg.get('branches', {}).get(chosen_branch, {})
        chosen_score_col = branch_cfg.get('score_col')
        if chosen_score_col in score_df.columns:
            score_df['score'] = score_df[chosen_score_col]
        exposure_cap = selector_info['exposure_cap']
        score_source = f'selector:{chosen_branch}, regime={selector_info["regime"]}'
    else:
        filtered = select_candidates(
            score_df,
            post_cfg,
            history_df=raw_df,
            asof_date=latest_date,
        )
        filtered = apply_supplemental_overlay(
            score_df,
            filtered,
            config.get('branch_router_v2b', {}),
        )
        breadth = artifacts['breadth']
        if 'exposure_cap' in post_cfg:
            exposure_cap = float(post_cfg['exposure_cap'])
        else:
            exposure_cap = 0.7 if breadth < 0.30 else 1.0

    if len(filtered) < 5:
        raise ValueError(f'可预测股票不足5只，当前仅有 {len(filtered)} 只')

    weighting_name = post_cfg.get('weighting', 'equal')
    output_df = build_weight_portfolio(filtered, weighting_name, exposure_cap=exposure_cap)
    output_df.to_csv(output_path, index=False)
    score_df_path = os.environ.get('BDC_SCORE_DF_PATH', './temp/predict_score_df.csv')
    exp009_live_path = os.environ.get('BDC_EXP009_LIVE_SCORE_DF_PATH', './temp/exp009_live_score_df.csv')
    filtered_path = os.environ.get('BDC_FILTERED_DF_PATH', './temp/predict_filtered_top30.csv')
    selector_json_path = os.environ.get('BDC_SELECTOR_JSON_PATH', './temp/selector_diagnostics.json')
    selector_csv_path = os.environ.get('BDC_SELECTOR_CSV_PATH', './temp/selector_diagnostics.csv')
    selector_debug_path = os.environ.get('BDC_SELECTOR_DEBUG_PATH', './temp/selector_candidates_debug.csv')
    gated_debug_path = os.environ.get('BDC_GATED_SELECTOR_DEBUG_PATH', './temp/gated_selector_debug.csv')
    for path in [score_df_path, exp009_live_path, filtered_path, selector_json_path, selector_csv_path, selector_debug_path, gated_debug_path]:
        path_dir = os.path.dirname(path)
        if path_dir:
            os.makedirs(path_dir, exist_ok=True)

    score_df.sort_values('score', ascending=False).to_csv(score_df_path, index=False)
    if exp009_enabled or 'exp009_final_score' in score_df.columns:
        exp009_cols = [
            'stock_id',
            'score',
            'transformer',
            'lgb',
            'exp009_meta_raw',
            'exp009_p_bad_1pct',
            'exp009_p_bad_2pct',
            'exp009_final_score',
            'sigma20',
            'ret5',
            'ret20',
            'amp20',
        ]
        live_cols = [col for col in exp009_cols if col in score_df.columns]
        score_df.sort_values('score', ascending=False)[live_cols].to_csv(exp009_live_path, index=False)
    filtered.head(30).to_csv(filtered_path, index=False)
    if selector_info is not None:
        with open(selector_json_path, 'w', encoding='utf-8') as f:
            json.dump(selector_info, f, ensure_ascii=False, indent=2, default=str)
        pd.DataFrame(selector_info['diagnostics']).to_csv(selector_csv_path, index=False)
        if selector_info.get('candidate_debug'):
            pd.DataFrame(selector_info['candidate_debug']).to_csv(selector_debug_path, index=False)
        if selector_info.get('gate_debug'):
            pd.DataFrame(selector_info['gate_debug']).to_csv(gated_debug_path, index=False)

    print(f'[BDC][predict] date={latest_date.date()}')
    print(f'[BDC][predict] ranked_stocks={len(score_df)}')
    print(f'[BDC][predict] score_source={score_source}')
    print(f'[BDC][predict] amp={amp_enabled}')
    print(f'[BDC][predict] postprocess=filter:{post_cfg.get("filter", "stable")}, weighting:{post_cfg.get("weighting", "equal")}')
    if selector_info is not None:
        print(f'[BDC][selector] regime={selector_info["regime"]}, stats={selector_info["regime_stats"]}')
        print(f'[BDC][selector] chosen_branch={selector_info["chosen_branch"]}')
    print(f'[BDC][predict] exposure={output_df["weight"].sum():.4f}')
    print(f'[BDC][predict] output={output_path}')
    print(f'[BDC][predict] selected={",".join(output_df["stock_id"].astype(str).tolist())}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
