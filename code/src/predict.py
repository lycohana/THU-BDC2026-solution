import os
import multiprocessing as mp

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from lgb_branch import load_lgb_branches, predict_lgb_score
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39
from utils import add_cross_section_features, extend_feature_columns_with_cross_section


feature_cloums_map = {
	'39': [
		'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	],
	'158+39': [
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
}

feature_engineer_func_map = {
	'39': engineer_features_39,
	'158+39': engineer_features_158plus39,
}


def preprocess_predict_data(df, stockid2idx):
	assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
	feature_engineer = feature_engineer_func_map[config['feature_num']]
	feature_columns = feature_cloums_map[config['feature_num']]

	df = df.copy()
	df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
	groups = [group for _, group in df.groupby('股票代码', sort=False)]
	if len(groups) == 0:
		raise ValueError('输入数据为空，无法预测')

	num_processes = min(10, mp.cpu_count())
	print('cpus!!!!!!!!!!!!!!!!!!',mp.cpu_count())
	with mp.Pool(processes=num_processes) as pool:
		processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='预测集特征工程'))

	processed = pd.concat(processed_list).reset_index(drop=True)
	processed['instrument'] = processed['股票代码'].map(stockid2idx)
	processed = processed.dropna(subset=['instrument']).copy()
	processed['instrument'] = processed['instrument'].astype(np.int64)
	processed['日期'] = pd.to_datetime(processed['日期'])
	processed = add_cross_section_features(processed, date_col='日期')
	feature_columns = extend_feature_columns_with_cross_section(feature_columns, processed)

	return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
	sequences, sequence_stock_ids = [], []
	for stock_id in stock_ids:
		stock_history = data[
			(data['股票代码'] == stock_id) &
			(data['日期'] <= latest_date)
		].sort_values('日期').tail(sequence_length)

		if len(stock_history) == sequence_length:
			sequences.append(stock_history[features].values.astype(np.float32))
			sequence_stock_ids.append(stock_id)

	if len(sequences) == 0:
		raise ValueError('没有可用于预测的股票序列，请检查数据与 sequence_length')

	return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def build_risk_frame(raw_df, stock_ids, latest_date):
	rows = []
	for stock_id in stock_ids:
		hist = raw_df[
			(raw_df['股票代码'] == stock_id) &
			(raw_df['日期'] <= latest_date)
		].sort_values('日期').tail(21)
		if len(hist) < 2:
			continue
		ret = hist['收盘'].astype(float).pct_change().dropna()
		amount = hist['成交额'].astype(float)
		rows.append({
			'stock_id': stock_id,
			'sigma20': float(ret.std()) if len(ret) > 1 else 0.0,
			'median_amount20': float(amount.median()) if len(amount) else 0.0,
		})
	return pd.DataFrame(rows)


def optimize_weights(candidates, score_col='score', risk_col='sigma20', tau=0.4, cap=0.35, exposure_cap=1.0):
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


def zscore(x):
	x = np.asarray(x, dtype=np.float64)
	return (x - x.mean()) / (x.std() + 1e-9)


def main():
	data_file = os.path.join(config['data_path'], 'train.csv')
	model_path = os.path.join(config['output_dir'], 'best_model.pth')
	scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
	output_path = os.path.join('./output/', 'result.csv')

	if not os.path.exists(model_path):
		raise FileNotFoundError(f'未找到模型文件: {model_path}')
	if not os.path.exists(scaler_path):
		raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')

	raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
	raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
	raw_df['日期'] = pd.to_datetime(raw_df['日期'])
	latest_date = raw_df['日期'].max()

	stock_ids = sorted(raw_df['股票代码'].unique())
	stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

	processed, features = preprocess_predict_data(raw_df, stockid2idx)
	processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

	scaler = joblib.load(scaler_path)
	processed[features] = scaler.transform(processed[features])

	sequence_length = config['sequence_length']
	sequences_np, sequence_stock_ids = build_inference_sequences(
		processed,
		features,
		sequence_length,
		stock_ids,
		latest_date,
	)

	if torch.cuda.is_available():
		device = torch.device('cuda')
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')

	model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.to(device)
	model.eval()

	with torch.no_grad():
		x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)  # [1, N, L, F]
		transformer_scores = model(x).squeeze(0).detach().cpu().numpy()         # [N]

	inference_df = pd.DataFrame({'stock_id': sequence_stock_ids})
	latest_rows = processed[processed['日期'] == latest_date].copy()
	latest_rows = latest_rows.rename(columns={'股票代码': 'stock_id'})
	inference_df = inference_df.merge(latest_rows, on='stock_id', how='left')

	lgb_bundle = load_lgb_branches(config['output_dir'])
	lgb_scores = predict_lgb_score(lgb_bundle, inference_df, config)
	if lgb_scores is None:
		scores = transformer_scores
		score_source = 'transformer'
	else:
		blend_cfg = config.get('blend', {})
		t_w = blend_cfg.get('transformer_weight', 0.55)
		lgb_w = blend_cfg.get('lgb_weight', 0.45)
		scores = t_w * zscore(transformer_scores) + lgb_w * zscore(lgb_scores)
		score_source = f'transformer+lgb({t_w:.2f}/{lgb_w:.2f})'

	score_df = pd.DataFrame({
		'stock_id': sequence_stock_ids,
		'score': scores,
	})
	risk_df = build_risk_frame(raw_df, sequence_stock_ids, latest_date)
	score_df = score_df.merge(risk_df, on='stock_id', how='left')
	score_df['sigma20'] = score_df['sigma20'].fillna(score_df['sigma20'].median()).clip(lower=1e-4)
	score_df['median_amount20'] = score_df['median_amount20'].fillna(0.0)

	liquidity_floor = score_df['median_amount20'].quantile(0.20)
	filtered = score_df[score_df['median_amount20'] >= liquidity_floor].copy()
	if len(filtered) < 5:
		filtered = score_df.copy()
	filtered = filtered.sort_values('score', ascending=False).reset_index(drop=True)

	if len(filtered) < 5:
		raise ValueError(f'可预测股票不足5只，当前仅有 {len(filtered)} 只')

	latest_processed = processed[processed['日期'] == latest_date]
	breadth = latest_processed['return_1'].gt(0).mean() if 'return_1' in latest_processed.columns else 1.0
	exposure_cap = 0.7 if breadth < 0.30 else 1.0
	output_df = optimize_weights(filtered.head(5), exposure_cap=exposure_cap)
	output_df.to_csv(output_path, index=False)

	print(f'预测日期: {latest_date.date()}')
	print(f'参与排序股票数: {len(score_df)}')
	print(f'分数来源: {score_source}')
	print(f'最终仓位: {output_df["weight"].sum():.4f}')
	print(f'结果已写入: {output_path}')


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	main()
