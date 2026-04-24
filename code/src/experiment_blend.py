"""
exp-002-04: Walk-Forward OOF + 修复权重优化 + 扩展过滤策略
"""
import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'code' / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'code' / 'src'))

from config import config
from feature_registry import finalize_feature_frame, get_feature_columns, get_feature_engineer
from lgb_branch import load_lgb_branches, predict_lgb_score
from model import StockTransformer
from portfolio_utils import (
    add_forward_open_returns,
    apply_filter,
    build_weight_portfolio,
    normalize_stock_id,
    portfolio_metrics,
    score_portfolio_like_scorer,
)
from train import preprocess_val_data, split_train_val_by_last_month
from predict import zscore


# ==================== 过滤策略 ====================

def current_filter(score_df, liquidity_quantile=0.20):
    """流动性过滤：保留成交额中位数前 80%"""
    out = score_df.copy()
    liquidity_floor = out['median_amount20'].quantile(liquidity_quantile)
    filtered = out[out['median_amount20'] >= liquidity_floor].copy()
    if len(filtered) < 5:
        filtered = out
    return filtered


def stable_filter(score_df, liquidity_quantile=0.20, sigma_quantile=0.85):
    """稳定过滤：流动性 + 低波动率"""
    out = current_filter(score_df, liquidity_quantile)
    sigma_cap = out['sigma20'].quantile(sigma_quantile)
    filtered = out[out['sigma20'] <= sigma_cap].copy()
    return filtered if len(filtered) >= 5 else out


def no_extreme_momentum_filter(score_df):
    """排除极端动量股票"""
    out = stable_filter(score_df)
    ret5_low = out['ret5'].quantile(0.10)
    ret5_high = out['ret5'].quantile(0.90)
    amp_cap = out['amp20'].quantile(0.90)
    filtered = out[
        (out['ret5'] >= ret5_low)
        & (out['ret5'] <= ret5_high)
        & (out['amp20'] <= amp_cap)
    ].copy()
    return filtered if len(filtered) >= 5 else out


def consensus_filter(score_df, cutoffs=(30, 50, 80, 120)):
    """两模型共识过滤"""
    out = score_df.copy()
    out['transformer_rank'] = out['transformer'].rank(ascending=False, method='first')
    out['lgb_rank'] = out['lgb'].rank(ascending=False, method='first')
    for cutoff in cutoffs:
        filtered = out[(out['transformer_rank'] <= cutoff) & (out['lgb_rank'] <= cutoff)].copy()
        if len(filtered) >= 5:
            return filtered
    return out


def consensus_stable_filter(score_df):
    """共识 + 稳定过滤"""
    return stable_filter(consensus_filter(score_df))


def topk_filter(score_df, k=10):
    """只保留分数最高的 top-k 股票"""
    out = score_df.sort_values('score', ascending=False).head(k).copy()
    return out if len(out) >= 5 else score_df


# ==================== 权重选择策略 ====================

def equal_topk(score_df, k=5):
    """等权 Top5"""
    top = score_df.sort_values('score', ascending=False).head(k)
    return [(sid, 1.0 / len(top)) for sid in top['stock_id'].tolist()]


def risk_soft_topk(score_df, k=5, tau=0.6, cap=0.50):
    """
    风险调整软权重 (修复版)
    - 提高 cap 到 0.50，避免全部退化成等权
    - 调整 tau 让权重分布更分散
    """
    top = score_df.sort_values('score', ascending=False).head(k).copy()
    raw_score = top['score'].to_numpy(dtype=np.float64)
    score = (raw_score - raw_score.mean()) / (raw_score.std() + 1e-9)
    risk = top['sigma20'].to_numpy(dtype=np.float64)
    risk = np.clip(risk, 1e-4, None)
    raw_weight = np.exp(score / tau) / risk
    raw_weight = np.minimum(raw_weight, cap)
    weights = raw_weight / (raw_weight.sum() + 1e-12)
    return list(zip(top['stock_id'].tolist(), weights.tolist()))


def score_soft_topk(score_df, k=5, tau=1.0, max_weight=0.50, min_weight=0.05):
    """纯分数软权重"""
    top = score_df.sort_values('score', ascending=False).head(k).copy()
    score = top['score'].to_numpy(dtype=np.float64)
    score = (score - score.mean()) / (score.std() + 1e-9)
    raw_weight = np.exp(score / tau)
    weights = raw_weight / (raw_weight.sum() + 1e-12)
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / (weights.sum() + 1e-12)
    return list(zip(top['stock_id'].tolist(), weights.tolist()))


def score_risk_soft_topk(score_df, k=5, tau=1.0, risk_power=0.5, max_weight=0.50, min_weight=0.05):
    """分数 - 风险联合软权重"""
    top = score_df.sort_values('score', ascending=False).head(k).copy()
    score = top['score'].to_numpy(dtype=np.float64)
    score = (score - score.mean()) / (score.std() + 1e-9)
    risk = np.clip(top['sigma20'].to_numpy(dtype=np.float64), 1e-4, None)
    raw_weight = np.exp(score / tau) / np.power(risk, risk_power)
    weights = raw_weight / (raw_weight.sum() + 1e-12)
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / (weights.sum() + 1e-12)
    return list(zip(top['stock_id'].tolist(), weights.tolist()))


def inv_vol_topk(score_df, k=5, max_weight=0.40, min_weight=0.10):
    """
    逆波动率权重 (新增)
    低波动率股票获得更高权重
    """
    top = score_df.sort_values('score', ascending=False).head(k).copy()
    risk = np.clip(top['sigma20'].to_numpy(dtype=np.float64), 1e-4, None)
    raw_weight = 1.0 / risk
    weights = raw_weight / (raw_weight.sum() + 1e-12)
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / (weights.sum() + 1e-12)
    return list(zip(top['stock_id'].tolist(), weights.tolist()))


# ==================== 投资组合评分 ====================

def score_portfolio(day_df, weights_df):
    """Score-equivalent portfolio return via explicit stock_id/date join."""
    date = pd.Timestamp(day_df['date'].iloc[0])
    return score_portfolio_like_scorer(
        weights_df,
        day_df[['stock_id', 'date', 'forward_open_return']],
        date,
    )


def _code_ids_from_instrument_ids(instrument_ids, idx2stock):
    return [normalize_stock_id(idx2stock[int(idx)]) for idx in instrument_ids]


def _apply_blend_scores(score_df, transformer_weight, lgb_weight, penalty):
    """Match predict.py: z-score each prediction cross-section, not the whole fold."""
    out = score_df.copy()
    score_chunks = []
    for _, day in out.groupby('date', sort=False):
        transformer_z = zscore(day['transformer'].to_numpy(dtype=np.float64))
        lgb_z = zscore(day['lgb'].to_numpy(dtype=np.float64))
        score = transformer_weight * transformer_z + lgb_weight * lgb_z
        if penalty > 0:
            score = score - penalty * np.abs(transformer_z - lgb_z)
        score_chunks.append(pd.Series(score, index=day.index))
    out['score'] = pd.concat(score_chunks).sort_index()
    return out


def _write_calibration_diagnostics(score_df, output_path):
    """Bucket OOF scores and check whether score quantiles map to future returns."""
    if score_df.empty:
        return None

    blend_cfg = config.get('blend', {})
    diag = _apply_blend_scores(
        score_df,
        float(blend_cfg.get('transformer_weight', 0.30)),
        float(blend_cfg.get('lgb_weight', 0.70)),
        float(blend_cfg.get('agreement_penalty', 0.0)),
    )
    rows = []
    for name, col in [('transformer', 'transformer'), ('lgb', 'lgb'), ('blend_current', 'score')]:
        tmp = diag[[col, 'forward_open_return']].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if tmp[col].nunique() < 3:
            continue
        tmp['bucket'] = pd.qcut(tmp[col], q=10, duplicates='drop')
        grouped = tmp.groupby('bucket', observed=True)
        for bucket, group in grouped:
            rows.append({
                'score_name': name,
                'bucket': str(bucket),
                'count': int(len(group)),
                'score_mean': float(group[col].mean()),
                'future_return_mean': float(group['forward_open_return'].mean()),
                'future_return_median': float(group['forward_open_return'].median()),
            })

    calibration = pd.DataFrame(rows)
    calibration_path = Path(output_path).with_name(Path(output_path).stem + '_calibration.csv')
    calibration.to_csv(calibration_path, index=False)
    print(f'[BDC][OOF] calibration={calibration_path}')
    return calibration_path


# ==================== Walk-Forward OOF 框架 ====================

def preprocess_oof_data(df, stockid2idx):
    """OOF 专用单进程特征工程，避免 Windows 多进程重复加载 torch DLL。"""
    feature_engineer = get_feature_engineer(config['feature_num'])
    feature_columns = get_feature_columns(config['feature_num'])

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    min_history = config['sequence_length']
    groups = [
        group
        for _, group in df.groupby('股票代码', sort=False)
        if len(group) >= min_history
    ]
    if not groups:
        raise ValueError('OOF 输入为空，无法继续')

    processed_list = [
        feature_engineer(group)
        for group in tqdm(groups, total=len(groups), desc='OOF验证集特征工程')
    ]
    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)
    processed = processed[processed['open_t1'] > 1e-4].copy()
    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label']).copy()
    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)

    enhance_cfg = config.get('feature_enhance', {})
    if any(enhance_cfg.values()):
        print(f'[BDC][feature_enhance] 启用 OOF 特征增强：{enhance_cfg}')
    processed, feature_columns = finalize_feature_frame(
        processed,
        feature_columns,
        enhance_cfg=enhance_cfg,
        date_col='日期',
    )
    return processed, feature_columns

def build_walk_forward_folds(full_df, n_folds=4, fold_window_months=2, gap_months=1):
    """
    构建 walk-forward OOF 折次

    参数:
    - n_folds: 折数
    - fold_window_months: 每个验证窗口月数
    - gap_months: 训练集和验证集之间的间隙月数 (防止数据泄漏)

    返回:
    list of dict: [{'train_end': ..., 'val_start': ..., 'val_end': ...}, ...]
    """
    full_df = full_df.copy()
    full_df['日期'] = pd.to_datetime(full_df['日期'])

    all_dates = sorted(pd.to_datetime(full_df['日期'].unique()))
    min_date = min(all_dates)
    max_date = max(all_dates)

    folds = []

    for fold_idx in range(n_folds):
        val_end = max_date - pd.DateOffset(months=fold_idx * (fold_window_months + gap_months))
        val_start = val_end - pd.DateOffset(months=fold_window_months)
        train_end = val_start - pd.DateOffset(months=gap_months)

        if train_end <= min_date:
            break

        folds.append({
            'fold': fold_idx,
            'train_end': pd.Timestamp(train_end),
            'val_start': pd.Timestamp(val_start),
            'val_end': pd.Timestamp(val_end),
        })

    return list(reversed(folds))


def build_oof_samples_for_fold(full_df, fold_info, features, sequence_length, stockid2idx, scaler):
    """
    为单个 OOF 折次构建验证样本

    参数:
    - full_df: 完整数据
    - fold_info: 折次信息
    - features: 特征列名
    - sequence_length: 序列长度
    - stockid2idx: 股票 ID 到索引的映射
    - scaler: 标准化器
    """
    val_start = fold_info['val_start']
    val_end = fold_info['val_end']

    context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)
    context_df = full_df[
        (full_df['日期'] >= context_start)
        & (full_df['日期'] <= val_end)
    ].copy()

    if len(context_df) == 0:
        return None, None, None

    val_df, generated_features = preprocess_oof_data(context_df, stockid2idx)
    val_df['instrument_id'] = val_df['instrument'].astype(int)
    for feature in features:
        if feature not in val_df.columns:
            val_df[feature] = 0.0

    risk_df = _build_risk_features(val_df)
    val_df[features] = val_df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    val_df[features] = scaler.transform(val_df[features])

    samples = [
        sample
        for sample in build_validation_samples(val_df, features, sequence_length, val_start)
        if pd.Timestamp(sample['date']) <= val_end
    ]

    return samples, risk_df, val_df


def _build_risk_features(data):
    """构建风险特征"""
    data = data.copy()
    data['日期'] = pd.to_datetime(data['日期'])
    rows = []
    for stock_id, group in data.groupby('股票代码', sort=False):
        group = group.sort_values('日期')
        close = group['收盘'].astype(float)
        high = group['最高'].astype(float)
        low = group['最低'].astype(float)
        amount = group['成交额'].astype(float)
        for idx in range(len(group)):
            hist = group.iloc[max(0, idx - 20):idx + 1]
            if len(hist) < 6:
                continue
            hist_close = hist['收盘'].astype(float)
            hist_high = hist['最高'].astype(float)
            hist_low = hist['最低'].astype(float)
            hist_ret = hist_close.pct_change(fill_method=None).dropna()
            rows.append({
                'date': pd.to_datetime(group.iloc[idx]['日期']),
                'stock_id': normalize_stock_id(stock_id),
                'sigma20': float(hist_ret.std()) if len(hist_ret) > 1 else 0.0,
                'median_amount20': float(hist['成交额'].astype(float).median()),
                'ret5': float(hist_close.iloc[-1] / hist_close.iloc[-6] - 1.0) if len(hist_close) >= 6 else 0.0,
                'ret20': float(hist_close.iloc[-1] / hist_close.iloc[0] - 1.0) if len(hist_close) >= 2 else 0.0,
                'amp20': float((hist_high.max() - hist_low.min()) / (hist_close.iloc[-1] + 1e-12)),
            })
    return pd.DataFrame(rows)


def run_oof_grid(
    n_folds=4,
    weights=None,
    penalties=None,
    output_path='temp/oof_combo_grid.csv',
    fold_window_months=2,
    gap_months=1,
):
    """
    Walk-Forward OOF 模式：在多个验证折上运行网格搜索

    参数:
    - n_folds: OOF 折数
    - weights: Transformer 权重列表
    - penalties: 分歧惩罚列表
    - output_path: 输出路径
    """
    if weights is None:
        weights = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    if penalties is None:
        penalties = [0.0, 0.1, 0.2]

    print(f'[BDC][OOF] 加载数据...')
    data_file = os.path.join(config['data_path'], 'train.csv')
    full_df = pd.read_csv(data_file, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])

    # 加载 scaler 获取特征列
    scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
    scaler = joblib.load(scaler_path)
    features = list(scaler.feature_names_in_)
    print(f'[BDC][OOF] 特征维度={len(features)}')

    # 获取所有股票 ID
    stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    idx2stock = {idx: sid for sid, idx in stockid2idx.items()}
    forward_return_df = add_forward_open_returns(full_df)

    # 加载模型
    model_path = os.path.join(config['output_dir'], 'best_model.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建 OOF 折次
    folds = build_walk_forward_folds(
        full_df,
        n_folds=n_folds,
        fold_window_months=fold_window_months,
        gap_months=gap_months,
    )
    print(f'[BDC][OOF] 构建了 {len(folds)} 个验证折')

    # 加载 LGBM
    lgb_bundle = load_lgb_branches(config['output_dir'])

    # 加载 Transformer 模型
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 为每个折次计算分数
    all_fold_results = []
    all_scored_frames = []

    for fold_info in folds:
        fold_idx = fold_info['fold']
        print(
            f'[BDC][OOF] 处理折次 fold={fold_idx}, '
            f'val_period={fold_info["val_start"].date()} ~ {fold_info["val_end"].date()}'
        )

        # 构建该折的样本
        samples, risk_df, val_df = build_oof_samples_for_fold(
            full_df, fold_info, features, config['sequence_length'], stockid2idx, scaler
        )

        if samples is None or len(samples) == 0:
            print(f'[BDC][OOF] 折次 {fold_idx} 无有效样本，跳过')
            continue

        # 为该折的每个样本评分
        scored_days = []
        for sample in samples:
            with torch.no_grad():
                x = torch.from_numpy(sample['sequences']).unsqueeze(0).to(device)
                transformer_scores = model(x).squeeze(0).detach().cpu().numpy()

            day_features = val_df[
                (pd.to_datetime(val_df['日期']) == sample['date'])
                & (val_df['instrument_id'].isin(sample['stock_ids']))
            ].copy()

            if len(day_features) == 0:
                continue

            order_map = {sid: idx for idx, sid in enumerate(sample['stock_ids'])}
            day_features['_order'] = day_features['instrument_id'].map(order_map)
            day_features = day_features.sort_values('_order')

            lgb_scores = predict_lgb_score(lgb_bundle, day_features, config)
            sample_stock_codes = _code_ids_from_instrument_ids(sample['stock_ids'], idx2stock)

            day = pd.DataFrame({
                'date': sample['date'],
                'stock_id': sample_stock_codes,
                'target': sample['targets'],
                'transformer': transformer_scores[:len(sample['stock_ids'])],
                'lgb': lgb_scores[:len(sample['stock_ids'])] if lgb_scores is not None else 0.0,
            }).merge(risk_df, on=['date', 'stock_id'], how='left')
            day = day.merge(forward_return_df, on=['date', 'stock_id'], how='left')
            day = day.dropna(subset=['forward_open_return']).copy()
            if len(day) < 10:
                continue
            scored_days.append(day)

        if not scored_days:
            print(f'[BDC][OOF] 折次 {fold_idx} 无可评分日期，跳过')
            continue

        fold_score_df = pd.concat(scored_days, ignore_index=True)
        fold_score_df['sigma20'] = fold_score_df['sigma20'].fillna(fold_score_df['sigma20'].median()).clip(lower=1e-4)
        fold_score_df['median_amount20'] = fold_score_df['median_amount20'].fillna(0.0)
        for col in ['ret5', 'ret20', 'amp20']:
            fold_score_df[col] = fold_score_df[col].fillna(fold_score_df[col].median())
        fold_score_df['fold'] = fold_idx
        all_scored_frames.append(fold_score_df)

        # 在该折上运行网格搜索
        fold_results = _run_grid_on_fold(fold_score_df, weights, penalties, fold_idx)
        all_fold_results.append(fold_results)
        print(f'[BDC][OOF] 折次 {fold_idx} 完成，{len(fold_results)} 条结果')

    if not all_fold_results:
        raise RuntimeError('OOF 没有产生任何有效折次结果，请检查数据窗口设置。')

    combined = pd.concat(all_fold_results, ignore_index=True)

    aggregated = combined.groupby(
        ['transformer_weight', 'lgb_weight', 'agreement_penalty', 'filter', 'weighting'],
        as_index=False,
    ).agg(
        oof_mean_return=('validation_mean_return', 'mean'),
        oof_median_return=('validation_median_return', 'mean'),
        oof_std_return=('validation_mean_return', 'std'),
        oof_q10_return=('validation_q10_return', 'mean'),
        oof_min_return=('validation_min_return', 'min'),
        oof_max_return=('validation_mean_return', 'max'),
        oof_last_fold_return=('last_fold_return', 'last'),
        win_rate_vs_equal=('win_rate_vs_equal', 'mean'),
        avg_max_weight=('avg_max_weight', 'mean'),
        avg_top2_weight=('avg_top2_weight', 'mean'),
        avg_herfindahl=('avg_herfindahl', 'mean'),
        avg_entropy_effective_n=('avg_entropy_effective_n', 'mean'),
        constraint_pass_rate=('constraint_pass_rate', 'mean'),
        oof_n_folds=('fold', 'count'),
        folds=('fold', list),
    )
    aggregated['oof_std_return'] = aggregated['oof_std_return'].fillna(0.0)
    aggregated = aggregated.sort_values(
        ['oof_mean_return', 'oof_median_return', 'win_rate_vs_equal', 'constraint_pass_rate'],
        ascending=[False, False, False, False],
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path = output_path.with_name(output_path.stem + '_detail.csv')
    score_path = output_path.with_name(output_path.stem + '_scores.csv')
    combined.to_csv(detail_path, index=False)
    aggregated.to_csv(output_path, index=False)
    if all_scored_frames:
        scored = pd.concat(all_scored_frames, ignore_index=True)
        scored.to_csv(score_path, index=False)
        _write_calibration_diagnostics(scored, output_path)

    print(f'\n[BDC][OOF] Top-20 配置:')
    print(aggregated.head(20).to_string(index=False))
    print(f'\n[BDC][OOF] 详细结果：{detail_path}')
    print(f'[BDC][OOF] 逐股票分数：{score_path}')
    print(f'[BDC][OOF] 汇总结果：{output_path}')

    return aggregated


def _run_grid_on_fold(score_df, weights, penalties, fold_idx):
    """在单个折上运行网格搜索"""
    rows = []
    post_cfg = config.get('postprocess', {})
    filter_names = ['nofilter', 'liquidity80', 'stable', 'no_extreme_momentum', 'consensus', 'consensus_stable', 'topk10']
    weighting_names = [
        'equal',
        'risk_soft',
        'score_soft',
        'score_risk_soft',
        'inv_vol',
        'shrunk_t2_rho10_cap30_min05',
        'shrunk_t3_rho20_cap35_min05',
        'shrunk_t5_rho30_cap35_min08',
    ]

    for transformer_weight in weights:
        lgb_weight = 1.0 - transformer_weight

        for penalty in penalties:
            base = _apply_blend_scores(score_df, transformer_weight, lgb_weight, penalty)

            daily_scores = []
            for date, day_base in base.groupby('date', sort=True):
                for filter_name in filter_names:
                    filtered = apply_filter(
                        day_base,
                        filter_name,
                        liquidity_quantile=post_cfg.get('liquidity_quantile', 0.20),
                        sigma_quantile=post_cfg.get('sigma_quantile', 0.85),
                    )
                    equal_df = build_weight_portfolio(filtered, 'equal')
                    equal_score, _ = score_portfolio(day_base, equal_df)
                    for weight_name in weighting_names:
                        try:
                            weights_df = build_weight_portfolio(filtered, weight_name)
                            score, _ = score_portfolio(day_base, weights_df)
                            metrics = portfolio_metrics(weights_df)
                            daily_scores.append({
                                'key': (filter_name, weight_name),
                                'date': date,
                                'score': score,
                                'win_vs_equal': float(score > equal_score),
                                'max_weight': metrics['max_weight'],
                                'top2_weight': metrics['top2_weight'],
                                'herfindahl': metrics['herfindahl'],
                                'entropy_effective_n': metrics['entropy_effective_n'],
                                'constraint_ok': float(
                                    metrics['top2_weight'] <= 0.70
                                    and metrics['herfindahl'] <= 0.28
                                    and metrics['entropy_effective_n'] >= 4.0
                                ),
                            })
                        except Exception:
                            continue

            for (filter_name, weight_name), grouped in pd.DataFrame(daily_scores).groupby('key', sort=False):
                grouped = grouped.sort_values('date')
                rows.append({
                    'fold': fold_idx,
                    'experiment': f'fold{fold_idx}_t{transformer_weight:.2f}_l{lgb_weight:.2f}_p{penalty:.2f}_{filter_name}_{weight_name}',
                    'transformer_weight': transformer_weight,
                    'lgb_weight': lgb_weight,
                    'agreement_penalty': penalty,
                    'filter': filter_name,
                    'weighting': weight_name,
                    'validation_mean_return': float(grouped['score'].mean()),
                    'validation_median_return': float(grouped['score'].median()),
                    'validation_std_return': float(grouped['score'].std(ddof=0)),
                    'validation_q10_return': float(grouped['score'].quantile(0.10)),
                    'validation_min_return': float(grouped['score'].min()),
                    'last_fold_return': float(grouped['score'].iloc[-1]),
                    'win_rate_vs_equal': float(grouped['win_vs_equal'].mean()),
                    'avg_max_weight': float(grouped['max_weight'].mean()),
                    'avg_top2_weight': float(grouped['top2_weight'].mean()),
                    'avg_herfindahl': float(grouped['herfindahl'].mean()),
                    'avg_entropy_effective_n': float(grouped['entropy_effective_n'].mean()),
                    'constraint_pass_rate': float(grouped['constraint_ok'].mean()),
                    'n_days': int(len(grouped)),
                })

    return pd.DataFrame(rows)


# ==================== 单验证集模式 (兼容 exp-002-03) ====================

def load_validation_scores():
    """加载单个验证集的分数 (向后兼容)"""
    data_file = os.path.join(config['data_path'], 'train.csv')
    full_df = pd.read_csv(data_file, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    train_df, val_df, val_start = split_train_val_by_last_month(full_df, config['sequence_length'])

    stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    idx2stock = {idx: sid for sid, idx in stockid2idx.items()}
    forward_return_df = add_forward_open_returns(full_df)
    val_data, features = preprocess_val_data(val_df, stockid2idx)
    val_data['instrument_id'] = val_data['instrument'].astype(int)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    risk_df = add_validation_risk_features(val_data)

    scaler = joblib.load(os.path.join(config['output_dir'], 'scaler.pkl'))
    val_data[features] = scaler.transform(val_data[features])
    samples = build_validation_samples(val_data, features, config['sequence_length'], val_start)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
    model.load_state_dict(torch.load(os.path.join(config['output_dir'], 'best_model.pth'), map_location=device))
    model.to(device)
    model.eval()

    lgb_bundle = load_lgb_branches(config['output_dir'])
    if lgb_bundle is None:
        raise FileNotFoundError('LGBM branch artifacts were not found.')

    scored_days = []
    feature_cols = lgb_bundle['features']
    for sample in samples:
        with torch.no_grad():
            x = torch.from_numpy(sample['sequences']).unsqueeze(0).to(device)
            transformer_scores = model(x).squeeze(0).detach().cpu().numpy()

        day_features = val_data[
            (pd.to_datetime(val_data['日期']) == sample['date'])
            & (val_data['instrument_id'].isin(sample['stock_ids']))
        ].copy()
        day_features['_order'] = day_features['instrument_id'].map({sid: i for i, sid in enumerate(sample['stock_ids'])})
        day_features = day_features.sort_values('_order')
        lgb_scores = predict_lgb_score(lgb_bundle, day_features, config)
        sample_stock_codes = _code_ids_from_instrument_ids(sample['stock_ids'], idx2stock)

        day = pd.DataFrame({
            'date': sample['date'],
            'stock_id': sample_stock_codes,
            'target': sample['targets'],
            'transformer': transformer_scores,
            'lgb': lgb_scores,
        }).merge(risk_df, on=['date', 'stock_id'], how='left')
        day = day.merge(forward_return_df, on=['date', 'stock_id'], how='left')
        day = day.dropna(subset=['forward_open_return']).copy()
        if len(day) < 10:
            continue
        scored_days.append(day)

    score_df = pd.concat(scored_days, ignore_index=True)
    score_df['sigma20'] = score_df['sigma20'].fillna(score_df['sigma20'].median()).clip(lower=1e-4)
    score_df['median_amount20'] = score_df['median_amount20'].fillna(0.0)
    for col in ['ret5', 'ret20', 'amp20']:
        score_df[col] = score_df[col].fillna(score_df[col].median())
    return score_df


def build_validation_samples(val_data, features, sequence_length, min_window_end_date):
    """构建验证样本 (向后兼容)"""
    data = val_data.copy()
    data['日期'] = pd.to_datetime(data['日期'])
    min_window_end_date = pd.to_datetime(min_window_end_date)
    data = data.sort_values(['instrument_id', '日期']).reset_index(drop=True)
    data = data.dropna(subset=['label'])

    windows = []
    for stock_code, group in data.groupby('instrument_id', sort=False):
        if len(group) < sequence_length:
            continue
        feature_values = group[features].values.astype(np.float32)
        labels = group['label'].values.astype(np.float32)
        dates = group['日期'].values
        for i in range(len(group) - sequence_length + 1):
            end_idx = i + sequence_length - 1
            end_date = pd.to_datetime(dates[end_idx])
            if end_date < min_window_end_date:
                continue
            windows.append({
                'date': end_date,
                'stock_id': int(stock_code),
                'seq': feature_values[i:i + sequence_length],
                'target': float(labels[end_idx]),
            })

    window_df = pd.DataFrame(windows)
    samples = []
    for date, day in window_df.groupby('date', sort=True):
        if len(day) < 10:
            continue
        samples.append({
            'date': pd.to_datetime(date),
            'stock_ids': day['stock_id'].tolist(),
            'sequences': np.stack(day['seq'].values),
            'targets': day['target'].to_numpy(dtype=np.float64),
        })
    return samples


def add_validation_risk_features(val_data):
    """添加验证集风险特征 (向后兼容)"""
    data = val_data.copy()
    data['日期'] = pd.to_datetime(data['日期'])
    rows = []
    group_col = '股票代码' if '股票代码' in data.columns else 'instrument_id'
    for stock_id, group in data.groupby(group_col, sort=False):
        group = group.sort_values('日期')
        close = group['收盘'].astype(float)
        high = group['最高'].astype(float)
        low = group['最低'].astype(float)
        amount = group['成交额'].astype(float)
        ret1 = close.pct_change(fill_method=None)
        for idx in range(len(group)):
            hist = group.iloc[max(0, idx - 20):idx + 1]
            if len(hist) < 6:
                continue
            hist_close = hist['收盘'].astype(float)
            hist_high = hist['最高'].astype(float)
            hist_low = hist['最低'].astype(float)
            hist_ret = hist_close.pct_change(fill_method=None).dropna()
            rows.append({
                'date': pd.to_datetime(group.iloc[idx]['日期']),
                'stock_id': normalize_stock_id(stock_id),
                'sigma20': float(hist_ret.std()) if len(hist_ret) > 1 else 0.0,
                'median_amount20': float(hist['成交额'].astype(float).median()),
                'ret5': float(hist_close.iloc[-1] / hist_close.iloc[-6] - 1.0),
                'ret20': float(hist_close.iloc[-1] / hist_close.iloc[0] - 1.0),
                'amp20': float((hist_high.max() - hist_low.min()) / (hist_close.iloc[-1] + 1e-12)),
            })
    return pd.DataFrame(rows)


def _get_feature_columns(feature_num):
    """获取特征列"""
    if feature_num == '39':
        return [
            'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
            'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
            'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
            'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
            'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
        ]
    elif feature_num == '158+39':
        return [
            'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
            'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
            'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
            'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
            'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
            'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
            'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
            'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
            'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
            'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
            'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
            'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
            'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
            'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
            'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
            'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
            'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
            'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
            'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
            'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
        ]
    else:
        raise ValueError(f'Unsupported feature_num: {feature_num}')


def run_single_val_grid(weights, output_path, penalties):
    """单验证集网格搜索 (向后兼容)"""
    score_df = load_validation_scores()
    rows = []
    details = []
    post_cfg = config.get('postprocess', {})
    filter_names = ['nofilter', 'liquidity80', 'stable', 'no_extreme_momentum', 'consensus', 'consensus_stable']
    weighting_names = [
        'equal',
        'risk_soft',
        'score_soft',
        'score_risk_soft',
        'inv_vol',
        'shrunk_t2_rho10_cap30_min05',
        'shrunk_t3_rho20_cap35_min05',
        'shrunk_t5_rho30_cap35_min08',
    ]

    for transformer_weight in weights:
        lgb_weight = 1.0 - transformer_weight

        for penalty in penalties:
            base = _apply_blend_scores(score_df, transformer_weight, lgb_weight, penalty)

            daily_scores = []
            daily_details = []
            for date, day_base in base.groupby('date', sort=True):
                for filter_name in filter_names:
                    filtered = apply_filter(
                        day_base,
                        filter_name,
                        liquidity_quantile=post_cfg.get('liquidity_quantile', 0.20),
                        sigma_quantile=post_cfg.get('sigma_quantile', 0.85),
                    )
                    equal_df = build_weight_portfolio(filtered, 'equal')
                    equal_score, _ = score_portfolio(day_base, equal_df)
                    for weight_name in weighting_names:
                        key = (filter_name, weight_name)
                        weights_df = build_weight_portfolio(filtered, weight_name)
                        score, breakdown = score_portfolio(day_base, weights_df)
                        metrics = portfolio_metrics(weights_df)
                        daily_scores.append({
                            'key': key,
                            'date': date,
                            'score': score,
                            'breakdown': breakdown,
                            'win_vs_equal': float(score > equal_score),
                            'max_weight': metrics['max_weight'],
                            'top2_weight': metrics['top2_weight'],
                            'herfindahl': metrics['herfindahl'],
                            'entropy_effective_n': metrics['entropy_effective_n'],
                            'constraint_ok': float(
                                metrics['top2_weight'] <= 0.70
                                and metrics['herfindahl'] <= 0.28
                                and metrics['entropy_effective_n'] >= 4.0
                            ),
                        })

            for (filter_name, weight_name), grouped in pd.DataFrame(daily_scores).groupby('key', sort=False):
                grouped = grouped.sort_values('date')
                first_day = grouped.iloc[-1]
                breakdown = first_day['breakdown'].copy()
                name = f't{transformer_weight:.2f}_l{lgb_weight:.2f}_p{penalty:.2f}_{filter_name}_{weight_name}'
                rows.append({
                    'experiment': name,
                    'transformer_weight': transformer_weight,
                    'lgb_weight': lgb_weight,
                    'agreement_penalty': penalty,
                    'filter': filter_name,
                    'weighting': weight_name,
                    'validation_mean_return': float(grouped['score'].mean()),
                    'validation_median_return': float(grouped['score'].median()),
                    'validation_std_return': float(grouped['score'].std(ddof=0)),
                    'validation_q10_return': float(grouped['score'].quantile(0.10)),
                    'validation_min_return': float(grouped['score'].min()),
                    'last_fold_return': float(grouped['score'].iloc[-1]),
                    'win_rate_vs_equal': float(grouped['win_vs_equal'].mean()),
                    'avg_max_weight': float(grouped['max_weight'].mean()),
                    'avg_top2_weight': float(grouped['top2_weight'].mean()),
                    'avg_herfindahl': float(grouped['herfindahl'].mean()),
                    'avg_entropy_effective_n': float(grouped['entropy_effective_n'].mean()),
                    'constraint_pass_rate': float(grouped['constraint_ok'].mean()),
                    'last_val_date': pd.to_datetime(first_day['date']).strftime('%Y-%m-%d'),
                    'last_val_stocks': '/'.join(str(s) for s in breakdown['stock_id'].tolist()),
                    'last_val_weights': '/'.join(f'{w:.4f}' for w in breakdown['weight'].tolist()),
                    'last_val_targets': '/'.join(f'{r:.4f}' for r in breakdown['forward_open_return'].tolist()),
                })
                breakdown.insert(0, 'experiment', name)
                breakdown.insert(1, 'date', first_day['date'])
                details.append(breakdown)

    result = pd.DataFrame(rows).sort_values('validation_mean_return', ascending=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    pd.concat(details, ignore_index=True).to_csv(output_path.with_name(output_path.stem + '_details.csv'), index=False)
    print('[BDC][experiment_blend] top_results')
    print(result.head(20).to_string(index=False))
    print(f'\n[BDC][experiment_blend] saved={output_path}')
    print(f'[BDC][experiment_blend] saved_details={output_path.with_name(output_path.stem + "_details.csv")}')


# ==================== 入口 ====================

def parse_args():
    parser = argparse.ArgumentParser(description='Run blend and postprocess grid search.')
    parser.add_argument('--mode', default='oof', choices=['oof', 'validation'],
                        help='oof=Walk-Forward OOF, validation=单验证集 (exp-002-03 兼容)')
    parser.add_argument('--n_folds', type=int, default=4, help='OOF 折数')
    parser.add_argument('--weights', default='0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0',
                        help='Comma-separated transformer weights.')
    parser.add_argument('--penalties', default='0,0.1,0.2',
                        help='Comma-separated agreement penalties.')
    parser.add_argument('--output', default='temp/oof_combo_grid.csv', help='Output path')
    parser.add_argument('--fold_window_months', type=int, default=2, help='每个 OOF 验证窗口的月份数')
    parser.add_argument('--gap_months', type=int, default=1, help='相邻 OOF 验证窗口之间的月份间隔')
    return parser.parse_args()


def main():
    args = parse_args()
    weights = [float(x) for x in args.weights.split(',') if x.strip()]
    penalties = [float(x) for x in args.penalties.split(',') if x.strip()]

    if args.mode == 'oof':
        run_oof_grid(
            n_folds=args.n_folds,
            weights=weights,
            penalties=penalties,
            output_path=args.output,
            fold_window_months=args.fold_window_months,
            gap_months=args.gap_months,
        )
    else:
        run_single_val_grid(weights=weights, penalties=penalties, output_path=args.output)


if __name__ == '__main__':
    main()
