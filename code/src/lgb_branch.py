import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd


LGB_RANKER_FILE = 'lgb_ranker.pkl'
LGB_REGRESSOR_FILE = 'lgb_regressor.pkl'
LGB_TOP5_RANKER_FILE = 'lgb_ranker_top5.pkl'
LGB_FEATURES_FILE = 'lgb_features.json'
LGB_REPORT_FILE = 'lgb_report.json'
LGB_TOP5_REPORT_FILE = 'lgb_top5_report.json'


def _clean_feature_frame(df, feature_cols):
    out = df[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    return out.fillna(0.0)


def _make_gain_by_date(df, label_col='label', max_gain=20):
    pct_rank = df.groupby('日期')[label_col].rank(method='first', pct=True)
    return np.floor(pct_rank * max_gain).astype(int).clip(0, max_gain)


def _make_top5_heavy_gain_by_date(df, label_col='label', gain_cfg=None):
    gain_cfg = gain_cfg or {}
    top1_gain = gain_cfg.get('top1_gain')
    top5_gain = int(gain_cfg.get('top5_gain', 10))
    top10_gain = int(gain_cfg.get('top10_gain', 4))
    top20_gain = int(gain_cfg.get('top20_gain', 1))
    negative_cap = gain_cfg.get('negative_cap', 1)

    rank = df.groupby('日期')[label_col].rank(method='first', ascending=False)
    gain = np.zeros(len(df), dtype=np.int32)
    if top1_gain is not None:
        gain[rank.to_numpy() == 1] = int(top1_gain)
        gain[(rank.to_numpy() >= 2) & (rank.to_numpy() <= 5)] = top5_gain
    else:
        gain[rank.to_numpy() <= 5] = top5_gain
    gain[(rank.to_numpy() >= 6) & (rank.to_numpy() <= 10)] = top10_gain
    gain[(rank.to_numpy() >= 11) & (rank.to_numpy() <= 20)] = top20_gain

    if negative_cap is not None:
        negative_mask = df[label_col].to_numpy(dtype=np.float64) < 0
        gain[negative_mask] = np.minimum(gain[negative_mask], int(negative_cap))
    return pd.Series(gain, index=df.index, dtype='int32')


def build_lgb_rank_data(df, feature_cols, label_col='label'):
    data = df.sort_values(['日期', '股票代码']).reset_index(drop=True).copy()
    data['gain'] = _make_gain_by_date(data, label_col=label_col)
    group = data.groupby('日期', sort=False).size().to_numpy()
    return _clean_feature_frame(data, feature_cols), data['gain'].astype(int), group


def build_lgb_top5_rank_data(df, feature_cols, label_col='label', gain_cfg=None):
    data = df.sort_values(['日期', '股票代码']).reset_index(drop=True).copy()
    data['gain'] = _make_top5_heavy_gain_by_date(data, label_col=label_col, gain_cfg=gain_cfg)
    group = data.groupby('日期', sort=False).size().to_numpy()
    return _clean_feature_frame(data, feature_cols), data['gain'].astype(int), group


def build_lgb_reg_data(df, feature_cols, label_col='label', label_clip=0.20):
    data = df.sort_values(['日期', '股票代码']).reset_index(drop=True).copy()
    y = data[label_col].astype(float).clip(-label_clip, label_clip)
    return _clean_feature_frame(data, feature_cols), y


def _zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-9)


def _topk_final_score_for_frame(valid_df, score_col='score', label_col='label', k=5):
    scores = []
    for _, day in valid_df.groupby('日期', sort=True):
        if len(day) < k:
            continue
        pred_sum = day.nlargest(k, score_col)[label_col].sum()
        max_sum = day.nlargest(k, label_col)[label_col].sum()
        random_sum = k * day[label_col].mean()
        denom = max_sum - random_sum
        if abs(denom) <= 1e-9:
            continue
        scores.append((pred_sum - random_sum) / denom)
    return float(np.mean(scores)) if scores else 0.0


def _get_top5_cfg(cfg):
    top5_cfg = dict(cfg.get('lgb_top5', {}))
    top5_cfg.setdefault('top5_gain', 10)
    top5_cfg.setdefault('top10_gain', 4)
    top5_cfg.setdefault('top20_gain', 1)
    top5_cfg.setdefault('negative_cap', 1)
    return top5_cfg


def _top5_ranker_params(cfg):
    seed = cfg.get('seed', 42)
    lgb_cfg = cfg.get('lgb', {})
    top5_cfg = _get_top5_cfg(cfg)
    return {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5],
        'learning_rate': top5_cfg.get('learning_rate', lgb_cfg.get('rank_learning_rate', 0.03)),
        'n_estimators': top5_cfg.get('n_estimators', lgb_cfg.get('rank_n_estimators', 1500)),
        'num_leaves': top5_cfg.get('num_leaves', min(int(lgb_cfg.get('num_leaves', 63)), 31)),
        'min_child_samples': top5_cfg.get('min_child_samples', lgb_cfg.get('min_child_samples', 64)),
        'subsample': top5_cfg.get('subsample', lgb_cfg.get('subsample', 0.8)),
        'subsample_freq': 1,
        'colsample_bytree': top5_cfg.get('colsample_bytree', lgb_cfg.get('colsample_bytree', 0.7)),
        'reg_lambda': top5_cfg.get('reg_lambda', max(float(lgb_cfg.get('reg_lambda', 2.0)), 5.0)),
        'random_state': seed,
        'n_jobs': top5_cfg.get('n_jobs', lgb_cfg.get('n_jobs', 8)),
        'verbosity': -1,
    }


def fit_lgb_top5_ranker(train_df, valid_df, feature_cols, output_dir, cfg):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        print(f"LightGBM 未安装，跳过 Top5-heavy LGBM 分支训练: {exc}")
        return None

    os.makedirs(output_dir, exist_ok=True)
    lgb_cfg = cfg.get('lgb', {})
    top5_cfg = _get_top5_cfg(cfg)

    X_rank_tr, y_rank_tr, g_rank_tr = build_lgb_top5_rank_data(
        train_df,
        feature_cols,
        gain_cfg=top5_cfg,
    )
    X_rank_va, y_rank_va, g_rank_va = build_lgb_top5_rank_data(
        valid_df,
        feature_cols,
        gain_cfg=top5_cfg,
    )

    params = _top5_ranker_params(cfg)
    print(
        '[BDC][lgb_top5] training '
        f'rows={len(train_df)}, valid_rows={len(valid_df)}, '
        f'gains=top5:{top5_cfg.get("top5_gain")}/top10:{top5_cfg.get("top10_gain")}/'
        f'top20:{top5_cfg.get("top20_gain")}/neg_cap:{top5_cfg.get("negative_cap")}, '
        f'num_leaves={params["num_leaves"]}, min_child_samples={params["min_child_samples"]}',
        flush=True,
    )

    ranker = lgb.LGBMRanker(**params)
    ranker.fit(
        X_rank_tr,
        y_rank_tr,
        group=g_rank_tr,
        eval_set=[(X_rank_va, y_rank_va)],
        eval_group=[g_rank_va],
        callbacks=[
            lgb.early_stopping(
                top5_cfg.get('early_stopping_rounds', lgb_cfg.get('early_stopping_rounds', 100)),
                verbose=False,
            ),
            lgb.log_evaluation(top5_cfg.get('log_period', lgb_cfg.get('log_period', 100))),
        ],
    )

    valid_scored = valid_df.sort_values(['日期', '股票代码']).reset_index(drop=True).copy()
    top5_score = ranker.predict(_clean_feature_frame(valid_scored, feature_cols))
    valid_scored['score'] = top5_score

    report = {
        'top5_rank_best_iteration': int(getattr(ranker, 'best_iteration_', 0) or 0),
        'top5_valid_final_score': _topk_final_score_for_frame(valid_scored),
        'num_features': len(feature_cols),
        'num_train_rows': int(len(train_df)),
        'num_valid_rows': int(len(valid_df)),
        'gain_config': {
            'top1_gain': top5_cfg.get('top1_gain'),
            'top5_gain': top5_cfg.get('top5_gain'),
            'top10_gain': top5_cfg.get('top10_gain'),
            'top20_gain': top5_cfg.get('top20_gain'),
            'negative_cap': top5_cfg.get('negative_cap'),
        },
        'params': {
            'learning_rate': params['learning_rate'],
            'n_estimators': params['n_estimators'],
            'num_leaves': params['num_leaves'],
            'min_child_samples': params['min_child_samples'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'reg_lambda': params['reg_lambda'],
        },
    }

    joblib.dump(ranker, os.path.join(output_dir, LGB_TOP5_RANKER_FILE))
    with open(os.path.join(output_dir, LGB_FEATURES_FILE), 'w', encoding='utf-8') as f:
        json.dump({'features': list(feature_cols)}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, LGB_TOP5_REPORT_FILE), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[BDC][lgb_top5] 训练完成，valid_final_score={report['top5_valid_final_score']:.6f}")
    return report


def fit_lgb_branches_with_search(train_df, valid_df, feature_cols, output_dir, cfg):
    """
    exp-002-04: LGBM 超参网格搜索（两阶段 + 进度条）
    第一阶段：粗搜索 3x3=9 种组合（固定 lr=0.03）
    第二阶段：在最优 num_leaves/min_child_samples 附近搜索 3 种 lr
    """
    try:
        import lightgbm as lgb
        from tqdm import tqdm
    except ImportError as exc:
        print(f"LightGBM 或 tqdm 未安装：{exc}")
        return None

    os.makedirs(output_dir, exist_ok=True)
    seed = cfg.get('seed', 42)
    lgb_cfg = cfg.get('lgb', {})
    search_cfg = cfg.get('lgb_search', {})
    label_clip = lgb_cfg.get('label_clip', 0.20)

    # 准备数据
    X_rank_tr, y_rank_tr, g_rank_tr = build_lgb_rank_data(train_df, feature_cols)
    X_rank_va, y_rank_va, g_rank_va = build_lgb_rank_data(valid_df, feature_cols)
    X_reg_tr, y_reg_tr = build_lgb_reg_data(train_df, feature_cols, label_clip=label_clip)
    X_reg_va, y_reg_va = build_lgb_reg_data(valid_df, feature_cols, label_clip=label_clip)

    # 超参网格（简化版）
    num_leaves_list = search_cfg.get('num_leaves', [31, 63, 127])
    min_child_samples_list = search_cfg.get('min_child_samples', [32, 64, 128])
    learning_rate_list = search_cfg.get('learning_rate', [0.02, 0.03, 0.05])

    total_combinations = len(num_leaves_list) * len(min_child_samples_list) * len(learning_rate_list)
    print(f'[BDC][lgb_search] 超参网格：num_leaves={num_leaves_list}, min_child_samples={min_child_samples_list}, lr={learning_rate_list}')
    print(f'[BDC][lgb_search] 总组合数：{total_combinations}')

    best_score = -1
    best_params = None
    best_ranker = None
    best_regressor = None

    # 两阶段搜索
    # 阶段 1: 固定 lr=0.03，搜索 num_leaves 和 min_child_samples
    print('[BDC][lgb_search] === 阶段 1: 粗搜索 (固定 lr=0.03) ===')
    lr_stage1 = 0.03
    stage1_results = []

    progress_bar = tqdm(total=len(num_leaves_list) * len(min_child_samples_list), desc='阶段 1')
    for num_leaves in num_leaves_list:
        for min_child_samples in min_child_samples_list:
            params_str = f'nl{num_leaves}_mcs{min_child_samples}_lr{lr_stage1}'
            score = _train_one_config(
                X_rank_tr, y_rank_tr, g_rank_tr, X_rank_va, y_rank_va, g_rank_va,
                X_reg_tr, y_reg_tr, X_reg_va, y_reg_va,
                num_leaves, min_child_samples, lr_stage1,
                lgb_cfg, seed, feature_cols, valid_df, cfg
            )
            stage1_results.append({'params': {'num_leaves': num_leaves, 'min_child_samples': min_child_samples, 'learning_rate': lr_stage1}, 'score': score})
            progress_bar.update(1)
    progress_bar.close()

    # 选出阶段 1 最优的 2 个配置
    stage1_results.sort(key=lambda x: x['score'], reverse=True)
    top2_stage1 = stage1_results[:2]
    print(f'[BDC][lgb_search] 阶段 1 Top-2:')
    for r in top2_stage1:
        print(f'  - {r["params"]} => {r["score"]:.6f}')

    # 阶段 2: 在最优配置附近搜索不同的 lr
    print('[BDC][lgb_search] === 阶段 2: 细搜索 (不同 lr) ===')
    all_configs = []
    for r in top2_stage1:
        for lr in learning_rate_list:
            config = r['params'].copy()
            config['learning_rate'] = lr
            all_configs.append(config)

    # 去重
    seen = set()
    unique_configs = []
    for c in all_configs:
        key = (c['num_leaves'], c['min_child_samples'], c['learning_rate'])
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)

    progress_bar2 = tqdm(total=len(unique_configs), desc='阶段 2')
    for config_params in unique_configs:
        params_str = f"nl{config_params['num_leaves']}_mcs{config_params['min_child_samples']}_lr{config_params['learning_rate']}"
        score = _train_one_config(
            X_rank_tr, y_rank_tr, g_rank_tr, X_rank_va, y_rank_va, g_rank_va,
            X_reg_tr, y_reg_tr, X_reg_va, y_reg_va,
            config_params['num_leaves'], config_params['min_child_samples'], config_params['learning_rate'],
            lgb_cfg, seed, feature_cols, valid_df, cfg
        )
        if score > best_score:
            best_score = score
            best_params = config_params
        progress_bar2.update(1)
    progress_bar2.close()

    print(f'[BDC][lgb_search] 最优配置：{best_params}, valid_score={best_score:.6f}')

    # 用最优配置重新训练最终模型
    ranker = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        ndcg_eval_at=[5],
        learning_rate=best_params['learning_rate'],
        n_estimators=lgb_cfg.get('rank_n_estimators', 1500),
        num_leaves=best_params['num_leaves'],
        min_child_samples=best_params['min_child_samples'],
        subsample=lgb_cfg.get('subsample', 0.8),
        subsample_freq=1,
        colsample_bytree=lgb_cfg.get('colsample_bytree', 0.7),
        reg_lambda=lgb_cfg.get('reg_lambda', 2.0),
        random_state=seed,
        n_jobs=lgb_cfg.get('n_jobs', 8),
        verbosity=-1,
    )
    ranker.fit(
        X_rank_tr,
        y_rank_tr,
        group=g_rank_tr,
        eval_set=[(X_rank_va, y_rank_va)],
        eval_group=[g_rank_va],
        callbacks=[
            lgb.early_stopping(lgb_cfg.get('early_stopping_rounds', 100), verbose=False),
            lgb.log_evaluation(lgb_cfg.get('log_period', 100)),
        ],
    )

    regressor = lgb.LGBMRegressor(
        objective='regression',
        metric='l2',
        learning_rate=best_params['learning_rate'],
        n_estimators=lgb_cfg.get('reg_n_estimators', 1200),
        num_leaves=best_params['num_leaves'],
        min_child_samples=best_params['min_child_samples'],
        subsample=lgb_cfg.get('subsample', 0.8),
        subsample_freq=1,
        colsample_bytree=lgb_cfg.get('colsample_bytree', 0.7),
        reg_lambda=lgb_cfg.get('reg_lambda', 2.0),
        random_state=seed,
        n_jobs=lgb_cfg.get('n_jobs', 8),
        verbosity=-1,
    )
    regressor.fit(
        X_reg_tr,
        y_reg_tr,
        eval_set=[(X_reg_va, y_reg_va)],
        callbacks=[
            lgb.early_stopping(lgb_cfg.get('early_stopping_rounds', 100), verbose=False),
            lgb.log_evaluation(lgb_cfg.get('log_period', 100)),
        ],
    )

    valid_scored = valid_df.sort_values(['日期', '股票代码']).reset_index(drop=True).copy()
    rank_score = ranker.predict(_clean_feature_frame(valid_scored, feature_cols))
    reg_score = regressor.predict(_clean_feature_frame(valid_scored, feature_cols))
    valid_scored['lgb_rank_score'] = rank_score
    valid_scored['lgb_reg_score'] = reg_score
    valid_scored['score'] = (
        lgb_cfg.get('rank_weight', 0.65) * _zscore(rank_score)
        + lgb_cfg.get('reg_weight', 0.35) * _zscore(reg_score)
    )

    report = {
        'rank_best_iteration': int(getattr(ranker, 'best_iteration_', 0) or 0),
        'reg_best_iteration': int(getattr(regressor, 'best_iteration_', 0) or 0),
        'valid_final_score': _topk_final_score_for_frame(valid_scored),
        'num_features': len(feature_cols),
        'num_train_rows': int(len(train_df)),
        'num_valid_rows': int(len(valid_df)),
        'search_best_params': best_params,
        'search_best_score': best_score,
    }

    joblib.dump(ranker, os.path.join(output_dir, LGB_RANKER_FILE))
    joblib.dump(regressor, os.path.join(output_dir, LGB_REGRESSOR_FILE))
    with open(os.path.join(output_dir, LGB_FEATURES_FILE), 'w', encoding='utf-8') as f:
        json.dump({'features': list(feature_cols)}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, LGB_REPORT_FILE), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if cfg.get('lgb_top5', {}).get('train', False):
        top5_report = fit_lgb_top5_ranker(train_df, valid_df, feature_cols, output_dir, cfg)
        if top5_report is not None:
            report['top5_report'] = top5_report

    print(f"[BDC][lgb_search] 最优模型保存，valid_final_score={report['valid_final_score']:.6f}")
    return report


def _train_one_config(X_rank_tr, y_rank_tr, g_rank_tr, X_rank_va, y_rank_va, g_rank_va,
                      X_reg_tr, y_reg_tr, X_reg_va, y_reg_va,
                      num_leaves, min_child_samples, learning_rate,
                      lgb_cfg, seed, feature_cols, valid_df, cfg):
    """训练单个超参配置并返回验证分数"""
    import lightgbm as lgb

    ranker = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        ndcg_eval_at=[5],
        learning_rate=learning_rate,
        n_estimators=lgb_cfg.get('rank_n_estimators', 1500),
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=lgb_cfg.get('subsample', 0.8),
        subsample_freq=1,
        colsample_bytree=lgb_cfg.get('colsample_bytree', 0.7),
        reg_lambda=lgb_cfg.get('reg_lambda', 2.0),
        random_state=seed,
        n_jobs=lgb_cfg.get('n_jobs', 8),
        verbosity=-1,
    )
    ranker.fit(
        X_rank_tr,
        y_rank_tr,
        group=g_rank_tr,
        eval_set=[(X_rank_va, y_rank_va)],
        eval_group=[g_rank_va],
        callbacks=[
            lgb.early_stopping(lgb_cfg.get('early_stopping_rounds', 100), verbose=False),
            lgb.log_evaluation(0),  # 静音
        ],
    )

    regressor = lgb.LGBMRegressor(
        objective='regression',
        metric='l2',
        learning_rate=learning_rate,
        n_estimators=lgb_cfg.get('reg_n_estimators', 1200),
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=lgb_cfg.get('subsample', 0.8),
        subsample_freq=1,
        colsample_bytree=lgb_cfg.get('colsample_bytree', 0.7),
        reg_lambda=lgb_cfg.get('reg_lambda', 2.0),
        random_state=seed,
        n_jobs=lgb_cfg.get('n_jobs', 8),
        verbosity=-1,
    )
    regressor.fit(
        X_reg_tr,
        y_reg_tr,
        eval_set=[(X_reg_va, y_reg_va)],
        callbacks=[
            lgb.early_stopping(lgb_cfg.get('early_stopping_rounds', 100), verbose=False),
            lgb.log_evaluation(0),  # 静音
        ],
    )

    valid_scored = valid_df.sort_values(['日期', '股票代码']).reset_index(drop=True).copy()
    rank_score = ranker.predict(_clean_feature_frame(valid_scored, feature_cols))
    reg_score = regressor.predict(_clean_feature_frame(valid_scored, feature_cols))
    valid_scored['lgb_rank_score'] = rank_score
    valid_scored['lgb_reg_score'] = reg_score
    valid_scored['score'] = (
        lgb_cfg.get('rank_weight', 0.65) * _zscore(rank_score)
        + lgb_cfg.get('reg_weight', 0.35) * _zscore(reg_score)
    )
    return _topk_final_score_for_frame(valid_scored)


def fit_lgb_branches(train_df, valid_df, feature_cols, output_dir, cfg):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        print(f"LightGBM 未安装，跳过 LGBM 分支训练: {exc}")
        return None

    os.makedirs(output_dir, exist_ok=True)
    seed = cfg.get('seed', 42)
    lgb_cfg = cfg.get('lgb', {})
    label_clip = lgb_cfg.get('label_clip', 0.20)

    X_rank_tr, y_rank_tr, g_rank_tr = build_lgb_rank_data(train_df, feature_cols)
    X_rank_va, y_rank_va, g_rank_va = build_lgb_rank_data(valid_df, feature_cols)
    X_reg_tr, y_reg_tr = build_lgb_reg_data(train_df, feature_cols, label_clip=label_clip)
    X_reg_va, y_reg_va = build_lgb_reg_data(valid_df, feature_cols, label_clip=label_clip)

    ranker = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        ndcg_eval_at=[5],
        learning_rate=lgb_cfg.get('rank_learning_rate', 0.03),
        n_estimators=lgb_cfg.get('rank_n_estimators', 1500),
        num_leaves=lgb_cfg.get('num_leaves', 63),
        min_child_samples=lgb_cfg.get('min_child_samples', 64),
        subsample=lgb_cfg.get('subsample', 0.8),
        subsample_freq=1,
        colsample_bytree=lgb_cfg.get('colsample_bytree', 0.7),
        reg_lambda=lgb_cfg.get('reg_lambda', 2.0),
        random_state=seed,
        n_jobs=lgb_cfg.get('n_jobs', 8),
        verbosity=-1,
    )
    ranker.fit(
        X_rank_tr,
        y_rank_tr,
        group=g_rank_tr,
        eval_set=[(X_rank_va, y_rank_va)],
        eval_group=[g_rank_va],
        callbacks=[
            lgb.early_stopping(lgb_cfg.get('early_stopping_rounds', 100), verbose=False),
            lgb.log_evaluation(lgb_cfg.get('log_period', 100)),
        ],
    )

    regressor = lgb.LGBMRegressor(
        objective='regression',
        metric='l2',
        learning_rate=lgb_cfg.get('reg_learning_rate', 0.03),
        n_estimators=lgb_cfg.get('reg_n_estimators', 1200),
        num_leaves=lgb_cfg.get('num_leaves', 63),
        min_child_samples=lgb_cfg.get('min_child_samples', 64),
        subsample=lgb_cfg.get('subsample', 0.8),
        subsample_freq=1,
        colsample_bytree=lgb_cfg.get('colsample_bytree', 0.7),
        reg_lambda=lgb_cfg.get('reg_lambda', 2.0),
        random_state=seed,
        n_jobs=lgb_cfg.get('n_jobs', 8),
        verbosity=-1,
    )
    regressor.fit(
        X_reg_tr,
        y_reg_tr,
        eval_set=[(X_reg_va, y_reg_va)],
        callbacks=[
            lgb.early_stopping(lgb_cfg.get('early_stopping_rounds', 100), verbose=False),
            lgb.log_evaluation(lgb_cfg.get('log_period', 100)),
        ],
    )

    valid_scored = valid_df.sort_values(['日期', '股票代码']).reset_index(drop=True).copy()
    rank_score = ranker.predict(_clean_feature_frame(valid_scored, feature_cols))
    reg_score = regressor.predict(_clean_feature_frame(valid_scored, feature_cols))
    valid_scored['lgb_rank_score'] = rank_score
    valid_scored['lgb_reg_score'] = reg_score
    valid_scored['score'] = (
        lgb_cfg.get('rank_weight', 0.65) * _zscore(rank_score)
        + lgb_cfg.get('reg_weight', 0.35) * _zscore(reg_score)
    )

    report = {
        'rank_best_iteration': int(getattr(ranker, 'best_iteration_', 0) or 0),
        'reg_best_iteration': int(getattr(regressor, 'best_iteration_', 0) or 0),
        'valid_final_score': _topk_final_score_for_frame(valid_scored),
        'num_features': len(feature_cols),
        'num_train_rows': int(len(train_df)),
        'num_valid_rows': int(len(valid_df)),
    }

    joblib.dump(ranker, os.path.join(output_dir, LGB_RANKER_FILE))
    joblib.dump(regressor, os.path.join(output_dir, LGB_REGRESSOR_FILE))
    with open(os.path.join(output_dir, LGB_FEATURES_FILE), 'w', encoding='utf-8') as f:
        json.dump({'features': list(feature_cols)}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, LGB_REPORT_FILE), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if cfg.get('lgb_top5', {}).get('train', False):
        top5_report = fit_lgb_top5_ranker(train_df, valid_df, feature_cols, output_dir, cfg)
        if top5_report is not None:
            report['top5_report'] = top5_report

    print(f"LGBM 分支训练完成，valid_final_score={report['valid_final_score']:.6f}")
    return report


def load_lgb_branches(model_dir):
    ranker_path = os.path.join(model_dir, LGB_RANKER_FILE)
    regressor_path = os.path.join(model_dir, LGB_REGRESSOR_FILE)
    top5_ranker_path = os.path.join(model_dir, LGB_TOP5_RANKER_FILE)
    features_path = os.path.join(model_dir, LGB_FEATURES_FILE)
    if not (os.path.exists(ranker_path) and os.path.exists(regressor_path) and os.path.exists(features_path)):
        return None

    with open(features_path, 'r', encoding='utf-8') as f:
        feature_cols = json.load(f)['features']
    bundle = {
        'ranker': joblib.load(ranker_path),
        'regressor': joblib.load(regressor_path),
        'features': feature_cols,
    }
    if os.path.exists(top5_ranker_path):
        bundle['top5_ranker'] = joblib.load(top5_ranker_path)
    return bundle


def predict_lgb_score(lgb_bundle, feature_df, cfg):
    components = predict_lgb_components(lgb_bundle, feature_df)
    if components is None:
        return None
    lgb_cfg = cfg.get('lgb', {})
    top5_weight = float(lgb_cfg.get('top5_rank_weight', cfg.get('lgb_top5', {}).get('blend_weight', 0.0)))
    if top5_weight > 0 and 'top5_rank_score' in components:
        rank_weight = float(lgb_cfg.get('rank_weight', 0.65))
        reg_weight = float(lgb_cfg.get('reg_weight', 0.35))
        total = rank_weight + reg_weight + top5_weight
        if total <= 1e-12:
            total = 1.0
        return (
            (rank_weight / total) * _zscore(components['rank_score'])
            + (reg_weight / total) * _zscore(components['reg_score'])
            + (top5_weight / total) * _zscore(components['top5_rank_score'])
        )
    return (
        lgb_cfg.get('rank_weight', 0.65) * _zscore(components['rank_score'])
        + lgb_cfg.get('reg_weight', 0.35) * _zscore(components['reg_score'])
    )


def predict_lgb_components(lgb_bundle, feature_df):
    if lgb_bundle is None:
        return None
    feature_cols = lgb_bundle['features']
    X = _clean_feature_frame(feature_df, feature_cols)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="Found 'eval_at' in params.*")
        warnings.filterwarnings('ignore', message="Found 'ndcg_eval_at' in params.*")
        rank_score = lgb_bundle['ranker'].predict(X)
        reg_score = lgb_bundle['regressor'].predict(X)
        top5_ranker = lgb_bundle.get('top5_ranker')
        top5_rank_score = top5_ranker.predict(X) if top5_ranker is not None else None
    components = {
        'rank_score': np.asarray(rank_score, dtype=np.float64),
        'reg_score': np.asarray(reg_score, dtype=np.float64),
    }
    if top5_rank_score is not None:
        components['top5_rank_score'] = np.asarray(top5_rank_score, dtype=np.float64)
    return components
