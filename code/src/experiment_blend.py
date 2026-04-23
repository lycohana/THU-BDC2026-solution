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
from predict import build_inference_sequences, build_risk_frame, preprocess_predict_data, zscore


def calculate_stock_return(test_df, stock_id):
    stock_id = str(stock_id).zfill(6)
    group = test_df[test_df['股票代码'].astype(str).str.zfill(6) == stock_id].sort_values('日期').tail(5)
    if len(group) < 2:
        return np.nan
    return (group.iloc[-1]['开盘'] - group.iloc[0]['开盘']) / group.iloc[0]['开盘']


def score_portfolio(test_df, selected):
    rows = []
    for stock_id, weight in selected:
        ret = calculate_stock_return(test_df, stock_id)
        rows.append({
            'stock_id': str(stock_id).zfill(6),
            'weight': float(weight),
            'return': float(ret),
            'weighted_return': float(ret * weight),
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


def current_filter(score_df):
    out = score_df.copy()
    liquidity_floor = out['median_amount20'].quantile(0.20)
    filtered = out[out['median_amount20'] >= liquidity_floor].copy()
    if len(filtered) < 5:
        filtered = out
    return filtered


def load_scores():
    data_file = os.path.join(config['data_path'], 'train.csv')
    raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    latest_date = raw_df['日期'].max()

    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    processed, features = preprocess_predict_data(raw_df, stockid2idx)
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = joblib.load(os.path.join(config['output_dir'], 'scaler.pkl'))
    processed[features] = scaler.transform(processed[features])
    sequences_np, sequence_stock_ids = build_inference_sequences(
        processed,
        features,
        config['sequence_length'],
        stock_ids,
        latest_date,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
    model.load_state_dict(torch.load(os.path.join(config['output_dir'], 'best_model.pth'), map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
        transformer_scores = model(x).squeeze(0).detach().cpu().numpy()

    inference_df = pd.DataFrame({'stock_id': sequence_stock_ids})
    latest_rows = processed[processed['日期'] == latest_date].copy()
    latest_rows = latest_rows.rename(columns={'股票代码': 'stock_id'})
    inference_df = inference_df.merge(latest_rows, on='stock_id', how='left')

    lgb_bundle = load_lgb_branches(config['output_dir'])
    lgb_scores = predict_lgb_score(lgb_bundle, inference_df, config)
    if lgb_scores is None:
        raise FileNotFoundError('LGBM branch artifacts were not found.')

    risk_df = build_risk_frame(raw_df, sequence_stock_ids, latest_date)
    score_df = pd.DataFrame({
        'stock_id': sequence_stock_ids,
        'transformer': transformer_scores,
        'lgb': lgb_scores,
    }).merge(risk_df, on='stock_id', how='left')
    score_df['sigma20'] = score_df['sigma20'].fillna(score_df['sigma20'].median()).clip(lower=1e-4)
    score_df['median_amount20'] = score_df['median_amount20'].fillna(0.0)
    return score_df


def run_grid(weights, output_path):
    test_df = pd.read_csv(os.path.join(config['data_path'], 'test.csv'), dtype={'股票代码': str})
    score_df = load_scores()
    rows = []
    details = []

    for transformer_weight in weights:
        lgb_weight = 1.0 - transformer_weight
        base = score_df.copy()
        base['score'] = (
            transformer_weight * zscore(base['transformer'].to_numpy())
            + lgb_weight * zscore(base['lgb'].to_numpy())
        )

        for filter_name, filtered in [('nofilter', base), ('liquidity80', current_filter(base))]:
            for weight_name, selector in [('equal', equal_topk), ('risk_soft', risk_soft_topk)]:
                selected = selector(filtered)
                score, breakdown = score_portfolio(test_df, selected)
                name = f't{transformer_weight:.2f}_l{lgb_weight:.2f}_{filter_name}_{weight_name}'
                rows.append({
                    'experiment': name,
                    'transformer_weight': transformer_weight,
                    'lgb_weight': lgb_weight,
                    'filter': filter_name,
                    'weighting': weight_name,
                    'score_self': score,
                    'stocks': '/'.join(breakdown['stock_id'].tolist()),
                    'weights': '/'.join(f'{w:.4f}' for w in breakdown['weight'].tolist()),
                    'returns': '/'.join(f'{r:.4f}' for r in breakdown['return'].tolist()),
                })
                breakdown.insert(0, 'experiment', name)
                details.append(breakdown)

    result = pd.DataFrame(rows).sort_values('score_self', ascending=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    pd.concat(details, ignore_index=True).to_csv(output_path.with_name(output_path.stem + '_details.csv'), index=False)
    print(result.head(20).to_string(index=False))
    print(f'\nSaved: {output_path}')
    print(f'Saved: {output_path.with_name(output_path.stem + "_details.csv")}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run exp-002-02 blend and postprocess grid.')
    parser.add_argument(
        '--weights',
        default='0,0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.7,0.8,0.9,1.0',
        help='Comma-separated transformer weights. LGBM weight is 1 - transformer_weight.',
    )
    parser.add_argument('--output', default='temp/exp_002_02_blend_grid.csv')
    return parser.parse_args()


def main():
    args = parse_args()
    weights = [float(x) for x in args.weights.split(',') if x.strip()]
    run_grid(weights, args.output)


if __name__ == '__main__':
    main()
