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
        'score_mode': 'blend',
        'normalize': 'zscore',
        'transformer_weight': 0.30,
        'lgb_weight': 0.70,
        'agreement_penalty': 0.00,
    },
    'selector': {
        'enabled': False,
        'version': 'exp-009_combo_lcb_selector',
        'force_branch': None,
        'fallback_branch': 'legal_minrisk_hardened',
        'emergency_fallback_branch': 'legal_minrisk_hardened',
        'branches': {
            'independent_union_rerank': {
                'score_col': 'rerank_score',
                'filter': 'union_rerank',
                'exposure': 1.0,
            },
            'safe_union_2slot': {
                'score_col': 'safe_union_2slot_score',
                'filter': 'safe_union_2slot',
                'exposure': 1.0,
            },
            'legal_plus_1alpha': {
                'score_col': 'legal_plus_score',
                'filter': 'legal_plus_1alpha',
                'exposure': 1.0,
            },
            'lgb_only_guarded': {
                'score_col': 'score_lgb_only',
                'filter': 'stable',
                'liquidity_quantile': 0.10,
                'sigma_quantile': 0.85,
                'exposure': 1.0,
            },
            'balanced_guarded': {
                'score_col': 'score_balanced',
                'filter': 'stable',
                'liquidity_quantile': 0.10,
                'sigma_quantile': 0.85,
                'exposure': 1.0,
            },
            'conservative_softrisk_v2': {
                'score_col': 'score_conservative_softrisk_v2',
                'filter': 'liquidity_q05',
                'exposure': 1.0,
            },
            'conservative_softrisk_v2_strict': {
                'score_col': 'score_conservative_softrisk_v2',
                'filter': 'legal_minrisk_hardened',
                'exposure': 1.0,
            },
            'defensive_v2_strict': {
                'score_col': 'score_defensive_v2',
                'filter': 'legal_minrisk',
                'exposure': 1.0,
            },
            'legal_minrisk_hardened': {
                'score_col': 'score_legal_minrisk',
                'filter': 'legal_minrisk_hardened',
                'exposure': 1.0,
            },
        },
        'regime_branch_order': {
            'risk_on_strict': ['independent_union_rerank', 'safe_union_2slot', 'legal_plus_1alpha', 'defensive_v2_strict', 'legal_minrisk_hardened'],
            'neutral_positive': ['independent_union_rerank', 'safe_union_2slot', 'legal_plus_1alpha', 'legal_minrisk_hardened', 'defensive_v2_strict'],
            'mixed_defensive': ['independent_union_rerank', 'safe_union_2slot', 'legal_plus_1alpha', 'legal_minrisk_hardened', 'defensive_v2_strict'],
            'risk_off': ['independent_union_rerank', 'safe_union_2slot', 'legal_plus_1alpha', 'legal_minrisk_hardened', 'defensive_v2_strict'],
        },
        'gated_union_rerank': {
            'enabled': True,
            'default_fallback_branch': 'legal_minrisk_hardened',
            'secondary_fallback_branch': 'defensive_v2_strict',
            'emergency_fallback_branch': 'legal_minrisk_hardened',
            'legal_plus_1alpha': {
                'enabled': True,
                'min_best_alpha_z': 1.75,
                'min_best_alpha_consensus': 2,
                'max_best_alpha_disagreement': 0.25,
            },
            'safe_union_2slot': {
                'enabled': True,
                'min_clean_alpha_count': 2,
                'min_clean_alpha_lcb_z': 1.25,
                'max_alpha_slots': 2,
                'min_stable_slots': 3,
                'max_tail_risk_count': 1,
                'max_high_vol_count': 1,
                'max_very_tail_count': 0,
                'max_very_high_vol_count': 0,
                'max_branch_only_alpha_count': 0,
                'require_market_enable': True,
            },
        },
        'union_rerank': {
            'enabled': True,
            'candidate_pool': {
                'conservative_softrisk_v2': 30,
                'lgb_only_guarded': 30,
                'balanced_guarded': 30,
                'defensive_v2_strict': 40,
                'legal_minrisk_hardened': 40,
            },
            'alpha_lcb': {
                'enabled': True,
                'base_penalty': 0.015,
                'uncertainty_weight': 0.055,
                'disagreement_weight': 0.025,
                'risk_weight': 0.020,
                'tail_risk_weight': 0.035,
                'branch_only_weight': 0.025,
                'oof_error_col': 'alpha_error_q80',
            },
            'combo_search': {
                'enabled': True,
                'disabled_regimes': ['risk_off'],
                'topn': 25,
                'objective_weights': {
                    'alpha_lcb': 1.00,
                    'consensus': 0.20,
                    'stable_support': 0.15,
                    'risk': 0.40,
                    'disagreement': 0.30,
                    'tail_count': 0.20,
                    'high_vol_count': 0.12,
                    'branch_only_count': 0.10,
                },
                'objective_weights_by_regime': {
                    'risk_on_strict': {
                        'alpha_lcb': 1.10,
                        'consensus': 0.12,
                        'stable_support': 0.08,
                        'risk': 0.16,
                        'disagreement': 0.18,
                        'tail_count': 0.08,
                        'high_vol_count': 0.04,
                        'branch_only_count': 0.05,
                    },
                    'neutral_positive': {
                        'risk': 0.24,
                        'disagreement': 0.22,
                        'tail_count': 0.12,
                        'high_vol_count': 0.08,
                    },
                },
                'constraints': {
                    'default': {
                        'min_stable_count': 2,
                        'min_consensus_count': 2,
                        'max_tail_risk_count': 2,
                        'max_high_vol_count': 2,
                        'max_very_tail_count': 1,
                        'max_very_high_vol_count': 1,
                        'max_branch_only_alpha_count': 1,
                    },
                    'risk_off': {
                        'min_stable_count': 3,
                        'min_consensus_count': 2,
                        'max_tail_risk_count': 1,
                        'max_high_vol_count': 1,
                        'max_very_tail_count': 0,
                        'max_very_high_vol_count': 0,
                        'max_branch_only_alpha_count': 0,
                    },
                },
            },
            'risk_lambda': {
                'risk_on_strict': 0.25,
                'neutral_positive': 0.45,
                'mixed_defensive': 0.60,
                'risk_off': 0.75,
            },
            'risk_budget': {
                'risk_on_strict': {
                    'max_tail_risk_count': 3,
                    'max_high_vol_count': 3,
                    'max_very_high_vol_count': 2,
                    'max_very_tail_count': 2,
                    'max_alpha_exception_count': 3,
                    'max_branch_only_alpha_count': 2,
                },
                'neutral_positive': {
                    'max_tail_risk_count': 2,
                    'max_high_vol_count': 2,
                    'max_very_high_vol_count': 1,
                    'max_very_tail_count': 1,
                    'max_alpha_exception_count': 2,
                    'max_branch_only_alpha_count': 1,
                },
                'mixed_defensive': {
                    'max_tail_risk_count': 2,
                    'max_high_vol_count': 2,
                    'max_very_high_vol_count': 1,
                    'max_very_tail_count': 1,
                    'max_alpha_exception_count': 2,
                    'max_branch_only_alpha_count': 1,
                },
                'risk_off': {
                    'max_tail_risk_count': 2,
                    'max_high_vol_count': 2,
                    'max_very_high_vol_count': 1,
                    'max_very_tail_count': 1,
                    'max_alpha_exception_count': 2,
                    'max_branch_only_alpha_count': 1,
                },
            },
            'portfolio_shape': {
                'risk_off': {
                    'max_pure_alpha_slots': 2,
                    'min_stable_slots': 2,
                    'min_consensus_slots': 2,
                },
            },
            'replacement': {
                'enabled': True,
                'alpha_tolerance': 0.15,
                'min_risk_improvement': 0.20,
                'require_safe_branch_support': True,
            },
        },
        'branch_risk_budget': {
            'enabled': True,
            'risk_on_strict': 0.42,
            'neutral_positive': 0.32,
            'mixed_defensive': 0.24,
            'risk_off': 0.18,
        },
        'fallback_router': {
            'enabled': True,
            'default_branch': 'legal_minrisk_hardened',
            'secondary_branch': 'defensive_v2_strict',
            'min_risk_improvement': 0.08,
            'require_market_toxic': True,
        },
        'meta_gate': {
            'enabled': False,
            'artifact_path': f'./model/{experiment_name}_{sequence_length}_{feature_num}/branch_meta_gate.pkl',
            'safe_threshold': 0.70,
            'unsafe_penalty': 0.02,
            'risk_penalty': 0.00,
            'require_rule_pass': False,
        },
        'confidence': {
            'min_score_gap': 0.015,
            'max_disagreement': {
                'risk_on': 0.55,
                'neutral': 0.50,
                'mixed_defensive': 0.48,
                'risk_off': 0.45,
            },
            'min_top20_overlap': {
                'risk_on': 0.10,
                'neutral': 0.10,
                'mixed_defensive': 0.00,
                'risk_off': 0.00,
            },
            'max_top5_sigma_quantile': 1.0,
            'max_top5_amp_quantile': 1.0,
        },
    },
    'postprocess': {
        'filter': 'regime_liquidity_risk_off',
        'weighting': 'equal',
        'liquidity_quantile': 0.10,
        'sigma_quantile': 0.85,
        'exposure_cap': 1.0,
    },
    'lgb': {
        'rank_weight': 0.65,
        'reg_weight': 0.35,
        'top5_rank_weight': 0.0,  # 默认不启用 Top5-heavy 分支
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
