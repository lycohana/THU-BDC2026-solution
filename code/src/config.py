# 配置参数
sequence_length = 60
feature_num = '158+39'
experiment_name = 'exp-002-05'
config = {
    'experiment_name': experiment_name,
    'sequence_length': sequence_length,   # 使用过去 60 个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 256,          # Transformer 输入维度
    'nhead': 4,             # 注意力头数量
    'num_layers': 3,        # Transformer 层数
    'dim_feedforward': 512, # 前馈网络维度
    'batch_size': 4,        # 排序任务 batch_size 可以小一些，因为每个 batch 包含更多股票
    'num_epochs': 50,       # 排序任务可能需要更多 epochs
    'learning_rate': 1e-5,  # 稍微降低学习率
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    'pairwise_weight': 1, # 配对损失权重
    'base_weight': 1.0, # 非 top-k 样本权重
    'top5_weight': 2.0, # top-5 样本权重（应大于 base_weight）

    'output_dir': f'./model/{experiment_name}_{sequence_length}_{feature_num}',
    'data_path': './data',
    'train_ranking_data_path': f'./temp/{experiment_name}_train_ranking_{sequence_length}_{feature_num}_cs.pkl',
    'val_ranking_data_path': f'./temp/{experiment_name}_val_ranking_{sequence_length}_{feature_num}_cs.pkl',
    'train_lgb': True,
    'blend': {
        'transformer_weight': 0.30,
        'lgb_weight': 0.70,
        'agreement_penalty': 0.00,
    },
    'postprocess': {
        'filter': 'regime_liquidity_risk_off',
        'weighting': 'equal',
        'liquidity_quantile': 0.10,
        'sigma_quantile': 0.85,
    },
    'lgb': {
        'rank_weight': 0.65,
        'reg_weight': 0.35,
        'top5_rank_weight': 0.0,  # 默认不启用 Top5-heavy 分支，保护线保持不变
        'label_clip': 0.20,
        'rank_learning_rate': 0.03,
        'reg_learning_rate': 0.03,
        'rank_n_estimators': 1500,
        'reg_n_estimators': 1200,
        'num_leaves': 63,
        'min_child_samples': 64,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_lambda': 2.0,
        'early_stopping_rounds': 100,
        'log_period': 100,
        'n_jobs': 8,
    },
    'lgb_top5': {
        'train': False,
        'blend_weight': 0.0,
        'top1_gain': None,
        'top5_gain': 10,
        'top10_gain': 4,
        'top20_gain': 1,
        'negative_cap': 1,
        'learning_rate': 0.03,
        'n_estimators': 1500,
        'num_leaves': 31,
        'min_child_samples': 64,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_lambda': 5.0,
        'early_stopping_rounds': 100,
        'log_period': 100,
    },
    # exp-002-04 新增：LGBM 超参搜索空间
    'lgb_search': {
        'num_leaves': [31, 63, 127],
        'min_child_samples': [32, 64, 128],
        'learning_rate': [0.02, 0.03, 0.05],
    },
    # exp-002-04 新增：特征增强选项
    'feature_enhance': {
        'add_industry_neutral': False,  # 行业中性化
        'add_more_momentum': True,      # 更多动量特征 (ret3, ret7, ret15)
        'add_amount_change': True,      # 成交额相对变化
    },
    'predict': {
        'feature_workers': 6,
        'use_cache': True,
        'cache_dir': './temp',
        'cache_compress': 3,
        'amp': True,
    },
    'seed': 42,
}
