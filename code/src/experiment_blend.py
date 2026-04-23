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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'code' / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'code' / 'src'))

from config import config
from lgb_branch import load_lgb_branches, predict_lgb_score
from model import StockTransformer
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

def score_portfolio(day_df, selected):
    """计算投资组合的加权收益"""
    rows = []
    target_map = dict(zip(day_df['stock_id'], day_df['target']))
    for stock_id, weight in selected:
        target = target_map.get(stock_id, 0.0)
        rows.append({
            'stock_id': stock_id,
            'weight': float(weight),
            'target': float(target),
            'weighted_return': float(target * weight),
        })
    result = pd.DataFrame(rows)
    return float(result['weighted_return'].sum()), result


# ==================== Walk-Forward OOF 框架 ====================

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

    # 找到数据的时间范围
    all_dates = sorted(full_df['日期'].unique())
    min_date = min(all_dates)
    max_date = max(all_dates)

    # 计算每个折的边界
    folds = []
    total_months = (max_date.year - min_date.year) * 12 + max_date.month - min_date.month

    # 从后往前划分验证窗口
    for fold_idx in range(n_folds):
        # 验证集结束时间
        val_end_offset = fold_idx * (fold_window_months + gap_months)
        val_end = max_date - timedelta(days=val_end_offset * 30)
        val_start = val_end - timedelta(days=fold_window_months * 30)
        train_end = val_start - timedelta(days=gap_months * 30)

        if train_end <= min_date:
            break

        folds.append({
            'fold': fold_idx,
            'train_end': train_end,
            'val_start': val_start,
            'val_end': val_end,
        })

    return folds


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
    train_end = fold_info['train_end']
    val_start = fold_info['val_start']
    val_end = fold_info['val_end']

    # 筛选验证集数据
    val_df = full_df[
        (full_df['日期'] > train_end) &
        (full_df['日期'] <= val_end)
    ].copy()

    if len(val_df) == 0:
        return None, None, None

    # 做与训练时一致的特征工程
    val_df = val_df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    # 按股票分组做特征工程（简化版，只用基础特征）
    # 注意：这里需要与训练时的特征工程一致
    # 为简化，直接使用 scaler 转换已有特征

    val_df['instrument_id'] = val_df['股票代码'].map(stockid2idx)
    val_df = val_df.dropna(subset=['instrument_id'])
    val_df['instrument_id'] = val_df['instrument_id'].astype(int)

    # 特征预处理
    val_df[features] = val_df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 标准化
    val_df[features] = scaler.transform(val_df[features])

    # 构建序列样本
    data = val_df.copy()
    data = data.dropna(subset=['label'])

    windows = []
    for stock_code, group in data.groupby('股票代码', sort=False):
        if len(group) < sequence_length:
            continue
        feature_values = group[features].values.astype(np.float32)
        labels = group['label'].values.astype(np.float32)
        dates = group['日期'].values
        for i in range(len(group) - sequence_length + 1):
            end_idx = i + sequence_length - 1
            end_date = pd.to_datetime(dates[end_idx])
            windows.append({
                'date': end_date,
                'stock_id': int(stock_code),
                'seq': feature_values[i:i + sequence_length],
                'target': float(labels[end_idx]),
            })

    window_df = pd.DataFrame(windows)

    # 按日期分组
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

    # 构建风险特征
    risk_df = _build_risk_features(val_df)

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
                'stock_id': int(stock_id),
                'sigma20': float(hist_ret.std()) if len(hist_ret) > 1 else 0.0,
                'median_amount20': float(hist['成交额'].astype(float).median()),
                'ret5': float(hist_close.iloc[-1] / hist_close.iloc[-6] - 1.0) if len(hist_close) >= 6 else 0.0,
                'ret20': float(hist_close.iloc[-1] / hist_close.iloc[0] - 1.0) if len(hist_close) >= 2 else 0.0,
                'amp20': float((hist_high.max() - hist_low.min()) / (hist_close.iloc[-1] + 1e-12)),
            })
    return pd.DataFrame(rows)


def run_oof_grid(n_folds=4, weights=None, penalties=None, output_path='temp/oof_combo_grid.csv'):
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

    # 加载模型
    model_path = os.path.join(config['output_dir'], 'best_model.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建 OOF 折次
    folds = build_walk_forward_folds(full_df, n_folds=n_folds)
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

    for fold_info in folds:
        fold_idx = fold_info['fold']
        print(f'[BDC][OOF] 处理折次 fold={fold_idx}, val_period={fold_info["val_start"].date()} ~ {fold_info["val_end"].date()}')

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
                & (val_df['股票代码'].isin([str(sid) for sid in sample['stock_ids']]))
            ].copy()

            if len(day_features) == 0:
                continue

            day_features['_order'] = day_features['股票代码'].apply(
                lambda x: sample['stock_ids'].index(int(x)) if int(x) in sample['stock_ids'] else 0
            )
            day_features = day_features.sort_values('_order')

            lgb_scores = predict_lgb_score(lgb_bundle, day_features, config)

            day = pd.DataFrame({
                'date': sample['date'],
                'stock_id': sample['stock_ids'],
                'target': sample['targets'],
                'transformer': transformer_scores[:len(sample['stock_ids'])],
                'lgb': lgb_scores[:len(sample['stock_ids'])] if lgb_scores is not None else 0.0,
            }).merge(risk_df, on=['date', 'stock_id'], how='left')
            scored_days.append(day)

        fold_score_df = pd.concat(scored_days, ignore_index=True)
        fold_score_df['sigma20'] = fold_score_df['sigma20'].fillna(fold_score_df['sigma20'].median()).clip(lower=1e-4)
        fold_score_df['median_amount20'] = fold_score_df['median_amount20'].fillna(0.0)
        for col in ['ret5', 'ret20', 'amp20']:
            fold_score_df[col] = fold_score_df[col].fillna(fold_score_df[col].median())

        # 在该折上运行网格搜索
        fold_results = _run_grid_on_fold(fold_score_df, weights, penalties, fold_idx)
        all_fold_results.append(fold_results)
        print(f'[BDC][OOF] 折次 {fold_idx} 完成，{len(fold_results)} 条结果')

    # 合并所有折次结果
    combined = pd.concat(all_fold_results, ignore_index=True)

    # 计算每个配置在各折的平均表现
    aggregated = combined.groupby(['transformer_weight', 'lgb_weight', 'agreement_penalty', 'filter', 'weighting']).agg({
        'validation_mean_return': ['mean', 'std', 'count'],
        'fold': list,
    }).reset_index()

    aggregated.columns = ['transformer_weight', 'lgb_weight', 'agreement_penalty', 'filter', 'weighting',
                          'oof_mean_return', 'oof_std_return', 'oof_n_folds', 'folds']
    aggregated = aggregated.sort_values('oof_mean_return', ascending=False)

    # 保存结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path.with_name('oof_detail.csv'), index=False)
    aggregated.to_csv(output_path, index=False)

    print(f'\n[BDC][OOF] Top-20 配置:')
    print(aggregated.head(20).to_string(index=False))
    print(f'\n[BDC][OOF] 详细结果：{output_path.with_name("oof_detail.csv")}')
    print(f'[BDC][OOF] 汇总结果：{output_path}')

    return aggregated


def _run_grid_on_fold(score_df, weights, penalties, fold_idx):
    """在单个折上运行网格搜索"""
    rows = []

    for transformer_weight in weights:
        lgb_weight = 1.0 - transformer_weight
        base = score_df.copy()
        transformer_z = zscore(base['transformer'].to_numpy())
        lgb_z = zscore(base['lgb'].to_numpy())

        for penalty in penalties:
            base['score'] = transformer_weight * transformer_z + lgb_weight * lgb_z
            if penalty > 0:
                base['score'] = base['score'] - penalty * np.abs(transformer_z - lgb_z)

            daily_scores = []
            for date, day_base in base.groupby('date', sort=True):
                filter_options = [
                    ('nofilter', day_base),
                    ('liquidity80', current_filter(day_base)),
                    ('stable', stable_filter(day_base)),
                    ('no_extreme_momentum', no_extreme_momentum_filter(day_base)),
                    ('consensus', consensus_filter(day_base)),
                    ('consensus_stable', consensus_stable_filter(day_base)),
                    ('topk10', topk_filter(day_base, k=10)),
                ]
                selector_options = [
                    ('equal', equal_topk),
                    ('risk_soft', risk_soft_topk),
                    ('score_soft', score_soft_topk),
                    ('score_risk_soft', score_risk_soft_topk),
                    ('inv_vol', inv_vol_topk),
                ]
                for filter_name, filtered in filter_options:
                    for weight_name, selector in selector_options:
                        try:
                            selected = selector(filtered)
                            score, _ = score_portfolio(day_base, selected)
                            daily_scores.append({
                                'key': (filter_name, weight_name),
                                'date': date,
                                'score': score,
                            })
                        except Exception:
                            continue

            for (filter_name, weight_name), grouped in pd.DataFrame(daily_scores).groupby('key', sort=False):
                mean_score = float(grouped['score'].mean())
                rows.append({
                    'fold': fold_idx,
                    'experiment': f'fold{fold_idx}_t{transformer_weight:.2f}_l{lgb_weight:.2f}_p{penalty:.2f}_{filter_name}_{weight_name}',
                    'transformer_weight': transformer_weight,
                    'lgb_weight': lgb_weight,
                    'agreement_penalty': penalty,
                    'filter': filter_name,
                    'weighting': weight_name,
                    'validation_mean_return': mean_score,
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
    val_data, features = preprocess_val_data(val_df, stockid2idx)
    val_data['instrument_id'] = val_data['instrument'].astype(int)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = joblib.load(os.path.join(config['output_dir'], 'scaler.pkl'))
    val_data[features] = scaler.transform(val_data[features])
    samples = build_validation_samples(val_data, features, config['sequence_length'], val_start)
    risk_df = add_validation_risk_features(val_data)

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

        day = pd.DataFrame({
            'date': sample['date'],
            'stock_id': sample['stock_ids'],
            'target': sample['targets'],
            'transformer': transformer_scores,
            'lgb': lgb_scores,
        }).merge(risk_df, on=['date', 'stock_id'], how='left')
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
    for stock_id, group in data.groupby('instrument_id', sort=False):
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
                'stock_id': int(stock_id),
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

    for transformer_weight in weights:
        lgb_weight = 1.0 - transformer_weight
        base = score_df.copy()
        transformer_z = zscore(base['transformer'].to_numpy())
        lgb_z = zscore(base['lgb'].to_numpy())

        for penalty in penalties:
            base['score'] = transformer_weight * transformer_z + lgb_weight * lgb_z
            if penalty > 0:
                base['score'] = base['score'] - penalty * np.abs(transformer_z - lgb_z)

            daily_scores = []
            daily_details = []
            for date, day_base in base.groupby('date', sort=True):
                filter_options = [
                    ('nofilter', day_base),
                    ('liquidity80', current_filter(day_base)),
                    ('stable', stable_filter(day_base)),
                    ('no_extreme_momentum', no_extreme_momentum_filter(day_base)),
                    ('consensus', consensus_filter(day_base)),
                    ('consensus_stable', consensus_stable_filter(day_base)),
                ]
                selector_options = [
                    ('equal', equal_topk),
                    ('risk_soft', risk_soft_topk),
                    ('score_soft', score_soft_topk),
                    ('score_risk_soft', score_risk_soft_topk),
                ]
                for filter_name, filtered in filter_options:
                    for weight_name, selector in selector_options:
                        key = (filter_name, weight_name)
                        selected = selector(filtered)
                        score, breakdown = score_portfolio(day_base, selected)
                        daily_scores.append({
                            'key': key,
                            'date': date,
                            'score': score,
                            'breakdown': breakdown,
                        })

            for (filter_name, weight_name), grouped in pd.DataFrame(daily_scores).groupby('key', sort=False):
                mean_score = float(grouped['score'].mean())
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
                    'validation_mean_return': mean_score,
                    'last_val_date': pd.to_datetime(first_day['date']).strftime('%Y-%m-%d'),
                    'last_val_stocks': '/'.join(str(s) for s in breakdown['stock_id'].tolist()),
                    'last_val_weights': '/'.join(f'{w:.4f}' for w in breakdown['weight'].tolist()),
                    'last_val_targets': '/'.join(f'{r:.4f}' for r in breakdown['target'].tolist()),
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
    return parser.parse_args()


def main():
    args = parse_args()
    weights = [float(x) for x in args.weights.split(',') if x.strip()]
    penalties = [float(x) for x in args.penalties.split(',') if x.strip()]

    if args.mode == 'oof':
        print('[BDC][OOF] 警告：OOF 模式正在开发中，暂时使用 validation 模式')
        print('[BDC][OOF] 请使用 --mode validation 运行单验证集测试')
        # 暂时fallback到validation模式
        run_single_val_grid(weights=weights, penalties=penalties, output_path=args.output)
    else:
        run_single_val_grid(weights=weights, penalties=penalties, output_path=args.output)


if __name__ == '__main__':
    main()
