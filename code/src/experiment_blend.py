import argparse
import os
import sys
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


def score_portfolio(day_df, selected):
    rows = []
    target_map = dict(zip(day_df['stock_id'], day_df['target']))
    for stock_id, weight in selected:
        target = target_map[stock_id]
        rows.append({
            'stock_id': stock_id,
            'weight': float(weight),
            'target': float(target),
            'weighted_return': float(target * weight),
        })
    result = pd.DataFrame(rows)
    return float(result['weighted_return'].sum()), result


def equal_topk(score_df, k=5):
    top = score_df.sort_values('score', ascending=False).head(k)
    return [(sid, 1.0 / len(top)) for sid in top['stock_id'].tolist()]


def risk_soft_topk(score_df, k=5, tau=0.4, cap=0.35):
    top = score_df.sort_values('score', ascending=False).head(k).copy()
    raw_score = top['score'].to_numpy(dtype=np.float64)
    score = (raw_score - raw_score.mean()) / (raw_score.std() + 1e-9)
    risk = top['sigma20'].to_numpy(dtype=np.float64)
    risk = np.clip(risk, 1e-4, None)
    raw_weight = np.exp(score / tau) / risk
    raw_weight = np.minimum(raw_weight, cap)
    weights = raw_weight / (raw_weight.sum() + 1e-12)
    return list(zip(top['stock_id'].tolist(), weights.tolist()))


def score_soft_topk(score_df, k=5, tau=0.8, max_weight=0.45, min_weight=0.05):
    top = score_df.sort_values('score', ascending=False).head(k).copy()
    score = top['score'].to_numpy(dtype=np.float64)
    score = (score - score.mean()) / (score.std() + 1e-9)
    raw_weight = np.exp(score / tau)
    weights = raw_weight / (raw_weight.sum() + 1e-12)
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / (weights.sum() + 1e-12)
    return list(zip(top['stock_id'].tolist(), weights.tolist()))


def score_risk_soft_topk(score_df, k=5, tau=0.8, risk_power=0.5, max_weight=0.45, min_weight=0.05):
    top = score_df.sort_values('score', ascending=False).head(k).copy()
    score = top['score'].to_numpy(dtype=np.float64)
    score = (score - score.mean()) / (score.std() + 1e-9)
    risk = np.clip(top['sigma20'].to_numpy(dtype=np.float64), 1e-4, None)
    raw_weight = np.exp(score / tau) / np.power(risk, risk_power)
    weights = raw_weight / (raw_weight.sum() + 1e-12)
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / (weights.sum() + 1e-12)
    return list(zip(top['stock_id'].tolist(), weights.tolist()))


def current_filter(score_df):
    out = score_df.copy()
    liquidity_floor = out['median_amount20'].quantile(0.20)
    filtered = out[out['median_amount20'] >= liquidity_floor].copy()
    if len(filtered) < 5:
        filtered = out
    return filtered


def stable_filter(score_df):
    out = current_filter(score_df)
    sigma_cap = out['sigma20'].quantile(0.85)
    filtered = out[out['sigma20'] <= sigma_cap].copy()
    return filtered if len(filtered) >= 5 else out


def no_extreme_momentum_filter(score_df):
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


def consensus_filter(score_df):
    out = score_df.copy()
    out['transformer_rank'] = out['transformer'].rank(ascending=False, method='first')
    out['lgb_rank'] = out['lgb'].rank(ascending=False, method='first')
    for cutoff in (30, 50, 80, 120):
        filtered = out[(out['transformer_rank'] <= cutoff) & (out['lgb_rank'] <= cutoff)].copy()
        if len(filtered) >= 5:
            return filtered
    return out


def consensus_stable_filter(score_df):
    return stable_filter(consensus_filter(score_df))


def add_history_risk_features(raw_df, stock_ids, latest_date):
    rows = []
    for stock_id in stock_ids:
        hist = raw_df[
            (raw_df['股票代码'] == stock_id)
            & (raw_df['日期'] <= latest_date)
        ].sort_values('日期').tail(21)
        if len(hist) < 6:
            continue
        close = hist['收盘'].astype(float)
        high = hist['最高'].astype(float)
        low = hist['最低'].astype(float)
        rows.append({
            'stock_id': stock_id,
            'ret5': float(close.iloc[-1] / close.iloc[-6] - 1.0),
            'ret20': float(close.iloc[-1] / close.iloc[0] - 1.0),
            'amp20': float((high.max() - low.min()) / (close.iloc[-1] + 1e-12)),
        })
    return pd.DataFrame(rows)


def build_validation_samples(val_data, features, sequence_length, min_window_end_date):
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


def load_validation_scores():
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


def run_grid(weights, output_path, penalties):
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
                    ('risk_soft_old', risk_soft_topk),
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


def parse_args():
    parser = argparse.ArgumentParser(description='Run train-only validation blend and postprocess grid.')
    parser.add_argument(
        '--weights',
        default='0,0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.7,0.8,0.9,1.0',
        help='Comma-separated transformer weights. LGBM weight is 1 - transformer_weight.',
    )
    parser.add_argument('--mode', default='validation', choices=['validation'])
    parser.add_argument('--output', default='temp/exp_002_03_validation_combo_grid.csv')
    parser.add_argument(
        '--penalties',
        default='0,0.1,0.2,0.3,0.5',
        help='Comma-separated disagreement penalties for |transformer_z - lgb_z|.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    weights = [float(x) for x in args.weights.split(',') if x.strip()]
    penalties = [float(x) for x in args.penalties.split(',') if x.strip()]
    run_grid(weights, args.output, penalties)


if __name__ == '__main__':
    main()
