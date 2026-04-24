import hashlib
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
from feature_registry import finalize_feature_frame, get_feature_columns, get_feature_engineer
from lgb_branch import load_lgb_branches, predict_lgb_score
from model import StockTransformer
from portfolio_utils import build_weight_portfolio, select_candidates as shared_select_candidates


PREDICT_CACHE_VERSION = 1


def _file_fingerprint(path):
    stat = os.stat(path)
    return {
        'path': os.path.abspath(path),
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
    }


def _build_predict_cache_path(data_file, scaler_path):
    predict_cfg = config.get('predict', {})
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
    hist = raw_df.groupby('股票代码', sort=False).tail(21).copy()
    if hist.empty:
        return pd.DataFrame(columns=['stock_id', 'sigma20', 'median_amount20', 'ret5', 'ret20', 'amp20'])

    hist['收盘'] = hist['收盘'].astype(float)
    hist['最高'] = hist['最高'].astype(float)
    hist['最低'] = hist['最低'].astype(float)
    hist['成交额'] = hist['成交额'].astype(float)

    hist['ret1'] = hist.groupby('股票代码', sort=False)['收盘'].pct_change(fill_method=None)
    hist['close_lag5'] = hist.groupby('股票代码', sort=False)['收盘'].shift(5)

    grouped = hist.groupby('股票代码', sort=False)
    agg = grouped.agg(
        sigma20=('ret1', 'std'),
        median_amount20=('成交额', 'median'),
        first_close=('收盘', 'first'),
        max_high=('最高', 'max'),
        min_low=('最低', 'min'),
        row_count=('收盘', 'size'),
    )
    latest = grouped.tail(1).set_index('股票代码').reindex(agg.index)

    last_close = latest['收盘'].to_numpy(dtype=np.float64)
    close_lag5 = latest['close_lag5'].fillna(0.0).to_numpy(dtype=np.float64)
    first_close = agg['first_close'].to_numpy(dtype=np.float64)
    row_count = agg['row_count'].to_numpy(dtype=np.int64)

    risk = pd.DataFrame(index=agg.index)
    risk['sigma20'] = agg['sigma20'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['median_amount20'] = agg['median_amount20'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['ret5'] = np.where(row_count >= 6, last_close / (close_lag5 + 1e-12) - 1.0, 0.0)
    risk['ret20'] = np.where(row_count >= 2, last_close / (first_close + 1e-12) - 1.0, 0.0)
    risk['amp20'] = (
        (agg['max_high'].to_numpy(dtype=np.float64) - agg['min_low'].to_numpy(dtype=np.float64))
        / (last_close + 1e-12)
    )

    risk = risk.reset_index().rename(columns={'股票代码': 'stock_id'})
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

    latest_processed = processed[processed['日期'] == latest_date]
    breadth = float(latest_processed['return_1'].gt(0).mean()) if 'return_1' in latest_processed.columns else 1.0

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


def select_candidates(score_df):
    return shared_select_candidates(score_df, post_cfg=config.get('postprocess', {}))


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


def main():
    data_file = os.path.join(config['data_path'], 'train.csv')
    model_dir = _resolve_model_dir()
    model_path = os.path.join(model_dir, 'best_model.pth')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    output_path = os.path.join('./output/', 'result.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
    if lgb_scores is None:
        scores = transformer_scores
        score_source = 'transformer'
    else:
        blend_cfg = config.get('blend', {})
        t_w = blend_cfg.get('transformer_weight', 0.55)
        lgb_w = blend_cfg.get('lgb_weight', 0.45)
        transformer_z = zscore(transformer_scores)
        lgb_z = zscore(lgb_scores)
        scores = t_w * transformer_z + lgb_w * lgb_z
        agreement_penalty = blend_cfg.get('agreement_penalty', 0.0)
        if agreement_penalty > 0:
            scores = scores - agreement_penalty * np.abs(transformer_z - lgb_z)
        score_source = f'transformer+lgb({t_w:.2f}/{lgb_w:.2f}, penalty={agreement_penalty:.2f})'

    score_df = pd.DataFrame({
        'stock_id': sequence_stock_ids,
        'score': scores,
        'transformer': transformer_scores,
        'lgb': np.asarray(lgb_scores, dtype=np.float64) if lgb_scores is not None else np.full(len(sequence_stock_ids), np.nan),
    })
    score_df = score_df.merge(artifacts['risk_df'], on='stock_id', how='left')
    score_df['sigma20'] = score_df['sigma20'].fillna(score_df['sigma20'].median()).clip(lower=1e-4)
    score_df['median_amount20'] = score_df['median_amount20'].fillna(0.0)
    for col in ['ret5', 'ret20', 'amp20']:
        score_df[col] = score_df[col].fillna(score_df[col].median() if score_df[col].notna().any() else 0.0)

    filtered = select_candidates(score_df)
    if len(filtered) < 5:
        raise ValueError(f'可预测股票不足5只，当前仅有 {len(filtered)} 只')

    breadth = artifacts['breadth']
    exposure_cap = 0.7 if breadth < 0.30 else 1.0
    post_cfg = config.get('postprocess', {})
    weighting_name = post_cfg.get('weighting', 'equal')
    output_df = build_weight_portfolio(filtered, weighting_name, exposure_cap=exposure_cap)
    output_df.to_csv(output_path, index=False)

    print(f'[BDC][predict] date={latest_date.date()}')
    print(f'[BDC][predict] ranked_stocks={len(score_df)}')
    print(f'[BDC][predict] score_source={score_source}')
    print(f'[BDC][predict] amp={amp_enabled}')
    print(f'[BDC][predict] postprocess=filter:{post_cfg.get("filter", "stable")}, weighting:{post_cfg.get("weighting", "equal")}')
    print(f'[BDC][predict] exposure={output_df["weight"].sum():.4f}')
    print(f'[BDC][predict] output={output_path}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
