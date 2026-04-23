import os
import multiprocessing as mp

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


def preprocess_predict_data(df, stockid2idx):
	feature_engineer = get_feature_engineer(config['feature_num'])
	feature_columns = get_feature_columns(config['feature_num'])

	df = df.copy()
	df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
	groups = [group for _, group in df.groupby('股票代码', sort=False)]
	if len(groups) == 0:
		raise ValueError('输入数据为空，无法预测')

	num_processes = min(10, mp.cpu_count())
	print(f'[BDC][predict] cpu_count={mp.cpu_count()}, feature_workers={num_processes}')
	with mp.Pool(processes=num_processes) as pool:
		processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='预测集特征工程'))

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
		close = hist['收盘'].astype(float)
		high = hist['最高'].astype(float)
		low = hist['最低'].astype(float)
		ret = close.pct_change(fill_method=None).dropna()
		amount = hist['成交额'].astype(float)
		rows.append({
			'stock_id': stock_id,
			'sigma20': float(ret.std()) if len(ret) > 1 else 0.0,
			'median_amount20': float(amount.median()) if len(amount) else 0.0,
			'ret5': float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else 0.0,
			'ret20': float(close.iloc[-1] / close.iloc[0] - 1.0) if len(close) >= 2 else 0.0,
			'amp20': float((high.max() - low.min()) / (close.iloc[-1] + 1e-12)),
		})
	return pd.DataFrame(rows)


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
		'transformer': transformer_scores,
		'lgb': np.asarray(lgb_scores, dtype=np.float64) if lgb_scores is not None else np.full(len(sequence_stock_ids), np.nan),
	})
	risk_df = build_risk_frame(raw_df, sequence_stock_ids, latest_date)
	score_df = score_df.merge(risk_df, on='stock_id', how='left')
	score_df['sigma20'] = score_df['sigma20'].fillna(score_df['sigma20'].median()).clip(lower=1e-4)
	score_df['median_amount20'] = score_df['median_amount20'].fillna(0.0)
	for col in ['ret5', 'ret20', 'amp20']:
		score_df[col] = score_df[col].fillna(score_df[col].median() if score_df[col].notna().any() else 0.0)

	filtered = select_candidates(score_df)

	if len(filtered) < 5:
		raise ValueError(f'可预测股票不足5只，当前仅有 {len(filtered)} 只')

	latest_processed = processed[processed['日期'] == latest_date]
	breadth = latest_processed['return_1'].gt(0).mean() if 'return_1' in latest_processed.columns else 1.0
	exposure_cap = 0.7 if breadth < 0.30 else 1.0
	post_cfg = config.get('postprocess', {})
	weighting_name = post_cfg.get('weighting', 'equal')
	output_df = build_weight_portfolio(filtered, weighting_name, exposure_cap=exposure_cap)
	output_df.to_csv(output_path, index=False)

	print(f'[BDC][predict] date={latest_date.date()}')
	print(f'[BDC][predict] ranked_stocks={len(score_df)}')
	print(f'[BDC][predict] score_source={score_source}')
	print(f'[BDC][predict] postprocess=filter:{post_cfg.get("filter", "stable")}, weighting:{post_cfg.get("weighting", "equal")}')
	print(f'[BDC][predict] exposure={output_df["weight"].sum():.4f}')
	print(f'[BDC][predict] output={output_path}')


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	main()
