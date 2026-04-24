import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'code' / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import config
from lgb_branch import fit_lgb_top5_ranker
from train import preprocess_data, preprocess_val_data, split_train_val_by_last_month, set_seed


GAIN_VARIANTS = {
    'v1': {'top1_gain': None, 'top5_gain': 10, 'top10_gain': 4, 'top20_gain': 1, 'negative_cap': 1},
    'v2': {'top1_gain': 12, 'top5_gain': 10, 'top10_gain': 4, 'top20_gain': 1, 'negative_cap': 1},
    'v3': {'top1_gain': None, 'top5_gain': 10, 'top10_gain': 3, 'top20_gain': 1, 'negative_cap': 1},
}


def _sanitize_feature_frame(df, features):
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=features).copy()


def _align_to_scaler_features(train_data, val_data, features, scaler):
    scaler_features = getattr(scaler, 'feature_names_in_', None)
    if scaler_features is None:
        return train_data, val_data, list(features)

    scaler_features = list(scaler_features)
    for frame in (train_data, val_data):
        missing = [col for col in scaler_features if col not in frame.columns]
        for col in missing:
            frame[col] = 0.0
    return train_data, val_data, scaler_features


def main():
    parser = argparse.ArgumentParser(description='Train Top5-heavy LightGBM ranker branch only.')
    parser.add_argument('--variant', choices=sorted(GAIN_VARIANTS), default='v1')
    parser.add_argument('--output_dir', default=config['output_dir'])
    parser.add_argument('--data_path', default=config['data_path'])
    parser.add_argument('--refit_scaler', action='store_true',
                        help='Refit a temporary scaler on the train split instead of reusing model/scaler.pkl.')
    args = parser.parse_args()

    set_seed(config.get('seed', 42))
    output_dir = Path(args.output_dir)
    data_file = Path(args.data_path) / 'train.csv'
    scaler_path = output_dir / 'scaler.pkl'
    if not data_file.exists():
        raise FileNotFoundError(f'未找到训练数据: {data_file}')

    config.setdefault('lgb_top5', {}).update(GAIN_VARIANTS[args.variant])
    config['lgb_top5']['train'] = True

    print(f'[BDC][lgb_top5] variant={args.variant}, output_dir={output_dir}', flush=True)
    full_df = pd.read_csv(data_file, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)

    train_df, val_df, _ = split_train_val_by_last_month(full_df, config['sequence_length'])
    stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    print('[BDC][lgb_top5] preprocessing train/valid', flush=True)
    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)
    train_data = _sanitize_feature_frame(train_data, features)
    val_data = _sanitize_feature_frame(val_data, features)

    if args.refit_scaler or not scaler_path.exists():
        print('[BDC][lgb_top5] fitting temporary scaler', flush=True)
        scaler = StandardScaler()
        train_data[features] = scaler.fit_transform(train_data[features])
        val_data[features] = scaler.transform(val_data[features])
    else:
        print(f'[BDC][lgb_top5] loading scaler={scaler_path}', flush=True)
        scaler = joblib.load(scaler_path)
        train_data, val_data, features = _align_to_scaler_features(train_data, val_data, features, scaler)
        train_data = _sanitize_feature_frame(train_data, features)
        val_data = _sanitize_feature_frame(val_data, features)
        train_data[features] = scaler.transform(train_data[features])
        val_data[features] = scaler.transform(val_data[features])

    report = fit_lgb_top5_ranker(train_data, val_data, features, str(output_dir), config)
    print('[BDC][lgb_top5] report:')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
