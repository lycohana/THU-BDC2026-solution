import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMClassifier, LGBMRanker
except ImportError:
    LGBMClassifier = None
    LGBMRanker = None


def _clean_X(df, feature_cols):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    return X.fillna(0.0).astype(np.float32)


def _zscore(values):
    x = np.asarray(values, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-9)


def _safe_series(df, col, default=0.0):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float64)
    values = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    fill = values.median() if values.notna().any() else default
    return values.fillna(fill).astype(np.float64)


def _rank_pct(values, ascending=True):
    return pd.Series(values).rank(pct=True, method='average', ascending=ascending).fillna(0.5).astype(np.float64)


def reciprocal_rank_fusion(df, score_cols, k=60, output_col='rrf_score'):
    """Aggregate heterogeneous expert ranks without assuming score calibration."""
    out = df.copy()
    valid_cols = [col for col in score_cols if col in out.columns]
    if not valid_cols:
        out[output_col] = 0.0
        return out

    score = np.zeros(len(out), dtype=np.float64)
    for col in valid_cols:
        values = pd.to_numeric(out[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        values = values.fillna(values.median() if values.notna().any() else 0.0)
        ranks = values.rank(ascending=False, method='first').to_numpy(dtype=np.float64)
        score += 1.0 / (float(k) + ranks)
    out[output_col] = score
    return out


def union_topk_candidates(df, score_cols, candidate_k=24):
    """Build a candidate pool from each expert's topK names."""
    ids = set()
    valid_cols = [col for col in score_cols if col in df.columns]
    for col in valid_cols:
        values = pd.to_numeric(df[col], errors='coerce')
        top = df.assign(_tmp_score=values).nlargest(min(int(candidate_k), len(df)), '_tmp_score')
        ids.update(top['stock_id'].astype(str).tolist())
    if not ids:
        return df.head(0).copy()
    return df[df['stock_id'].astype(str).isin(ids)].copy()


def hedge_update(weights, losses, eta=0.5, floor=1e-6):
    """One-step Hedge expert update: w_i <- w_i * exp(-eta * loss_i)."""
    w = np.asarray(weights, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    if w.size != losses.size:
        raise ValueError('weights and losses must have the same length')
    updated = np.maximum(w, float(floor)) * np.exp(-float(eta) * losses)
    total = updated.sum()
    if total <= 1e-12:
        return np.full_like(updated, 1.0 / max(len(updated), 1))
    return updated / total


def _router_weights(score_df, expert_cols, cfg):
    base = np.asarray(cfg.get('base_weights', [1.0 / max(len(expert_cols), 1)] * len(expert_cols)), dtype=np.float64)
    if base.size != len(expert_cols):
        base = np.full(len(expert_cols), 1.0 / max(len(expert_cols), 1), dtype=np.float64)

    sigma = pd.to_numeric(score_df.get('sigma20', pd.Series(0.0, index=score_df.index)), errors='coerce').fillna(0.0)
    ret5 = pd.to_numeric(score_df.get('ret5', pd.Series(0.0, index=score_df.index)), errors='coerce').fillna(0.0)
    high_vol = float(sigma.rank(pct=True).gt(0.75).mean()) if len(sigma) else 0.0
    breadth = float(ret5.gt(0).mean()) if len(ret5) else 0.5

    adjusted = base.copy()
    for idx, col in enumerate(expert_cols):
        if col in {'lgb', 'lgb_rank_score', 'lgb_top5_score'}:
            adjusted[idx] *= 1.0 + cfg.get('lgb_risk_off_boost', 0.20) * max(0.0, high_vol - 0.25)
        if col == 'transformer':
            adjusted[idx] *= 1.0 + cfg.get('transformer_risk_on_boost', 0.15) * max(0.0, breadth - 0.5)
    total = adjusted.sum()
    return adjusted / total if total > 1e-12 else np.full(len(expert_cols), 1.0 / max(len(expert_cols), 1))


def _add_tail_guard_features(df, expert_cols, cfg):
    out = df.copy()
    out['grr_sigma_rank'] = _rank_pct(_safe_series(out, 'sigma20'), ascending=True).to_numpy()
    out['grr_amp_rank'] = _rank_pct(_safe_series(out, 'amp20'), ascending=True).to_numpy()
    out['grr_drawdown_rank'] = _rank_pct(_safe_series(out, 'max_drawdown20'), ascending=True).to_numpy()
    out['grr_ret5_rank'] = _rank_pct(_safe_series(out, 'ret5'), ascending=True).to_numpy()
    out['grr_liq_rank'] = _rank_pct(_safe_series(out, 'median_amount20'), ascending=True).to_numpy()

    consensus_topk = int(cfg.get('consensus_topk', cfg.get('candidate_k', 24)))
    consensus = np.zeros(len(out), dtype=np.float64)
    for col in expert_cols:
        if col not in out.columns:
            continue
        ranks = _safe_series(out, col).rank(ascending=False, method='first')
        consensus += (ranks <= consensus_topk).to_numpy(dtype=np.float64)
    out['grr_consensus_count'] = consensus
    out['grr_consensus_norm'] = consensus / max(len(expert_cols), 1)
    out['grr_minrisk_score'] = (
        0.30 * (1.0 - out['grr_sigma_rank'])
        + 0.25 * (1.0 - out['grr_amp_rank'])
        + 0.25 * (1.0 - out['grr_drawdown_rank'])
        + 0.20 * out['grr_liq_rank']
    ).astype(np.float64)
    ret5_rank = out['grr_ret5_rank'].to_numpy(dtype=np.float64)
    out['grr_ret5_extreme'] = np.maximum(ret5_rank - 0.80, 0.0) / 0.20 + np.maximum(0.20 - ret5_rank, 0.0) / 0.20
    return out


def compute_market_crash_state(score_df, cfg=None, score_col='grr_final_score'):
    """Infer tail-risk state from legal same-day historical features only."""
    cfg = cfg or {}
    if len(score_df) == 0:
        return {
            'crash_mode': False,
            'risk_off_score': 0.0,
            'breadth_1d': 0.5,
            'breadth_5d': 0.5,
            'top_fragility_score': 0.0,
        }

    work = score_df.copy()
    if 'grr_sigma_rank' not in work.columns:
        work = _add_tail_guard_features(work, [col for col in ['lgb', 'transformer', 'score'] if col in work.columns], cfg)

    ret1 = _safe_series(work, 'ret1')
    ret5 = _safe_series(work, 'ret5')
    breadth_1d = float(ret1.gt(0.0).mean())
    breadth_5d = float(ret5.gt(0.0).mean())
    median_ret1 = float(ret1.median())
    median_ret5 = float(ret5.median())
    high_vol_ratio = float(work['grr_sigma_rank'].gt(0.75).mean())
    high_amp_ratio = float(work['grr_amp_rank'].gt(0.75).mean())
    high_dd_ratio = float(work['grr_drawdown_rank'].gt(0.75).mean())

    market_score = (
        0.24 * (1.0 - breadth_1d)
        + 0.22 * (1.0 - breadth_5d)
        + 0.18 * np.clip(-median_ret1 / 0.015, 0.0, 1.0)
        + 0.16 * np.clip(-median_ret5 / 0.035, 0.0, 1.0)
        + 0.10 * high_vol_ratio
        + 0.05 * high_amp_ratio
        + 0.05 * high_dd_ratio
    )

    top_col = score_col if score_col in work.columns else 'score' if 'score' in work.columns else None
    if top_col is not None:
        top = work.sort_values(top_col, ascending=False).head(5).copy()
    else:
        top = work.head(min(5, len(work))).copy()
    top_fragility = (
        0.30 * float(top['grr_sigma_rank'].mean())
        + 0.25 * float(top['grr_amp_rank'].mean())
        + 0.20 * float(top['grr_drawdown_rank'].mean())
        + 0.15 * float(top['grr_ret5_extreme'].mean())
        + 0.10 * float((1.0 - top['grr_consensus_norm']).mean())
    )
    high_risk_top_count = int(
        (
            (top['grr_sigma_rank'] >= cfg.get('high_risk_sigma_q', 0.75))
            & (top['grr_amp_rank'] >= cfg.get('high_risk_amp_q', 0.75))
        ).sum()
    )

    risk_off_score = float(0.62 * market_score + 0.38 * top_fragility)
    crash_mode = bool(
        risk_off_score >= float(cfg.get('crash_threshold', 0.50))
        or (
            top_fragility >= float(cfg.get('top_fragility_threshold', 0.62))
            and breadth_1d <= float(cfg.get('fragility_breadth_1d_max', 0.48))
        )
        or high_risk_top_count >= int(cfg.get('high_risk_top_count_trigger', 3))
    )
    return {
        'crash_mode': crash_mode,
        'risk_off_score': risk_off_score,
        'market_score': float(market_score),
        'top_fragility_score': float(top_fragility),
        'high_risk_top_count': high_risk_top_count,
        'breadth_1d': breadth_1d,
        'breadth_5d': breadth_5d,
        'median_ret1': median_ret1,
        'median_ret5': median_ret5,
        'high_vol_ratio': high_vol_ratio,
    }


def tail_guard_rerank(score_df, expert_cols, cfg):
    guard_cfg = cfg.get('tail_guard', {})
    out = _add_tail_guard_features(score_df, expert_cols, {**cfg, **guard_cfg})
    state = compute_market_crash_state(out, {**cfg, **guard_cfg}, score_col='grr_final_score')
    if not cfg.get('crash_guard_enabled', True):
        state['crash_mode'] = False
    out['grr_risk_off_score'] = state['risk_off_score']
    out['grr_crash_mode'] = bool(state['crash_mode'])
    out['grr_tail_guard_triggered'] = bool(state['crash_mode'])

    veto_enabled = bool(cfg.get('high_risk_chaser_veto', True))
    consensus_min = float(guard_cfg.get('veto_min_consensus_norm', 0.50))
    high_risk_veto = (
        veto_enabled
        & out['grr_crash_mode'].astype(bool)
        & (out['grr_sigma_rank'] >= float(guard_cfg.get('veto_sigma_q', 0.75)))
        & (out['grr_amp_rank'] >= float(guard_cfg.get('veto_amp_q', 0.75)))
        & (out['grr_ret5_rank'] >= float(guard_cfg.get('veto_ret5_q', 0.70)))
        & (out['grr_consensus_norm'] < consensus_min)
    )
    out['grr_high_risk_chaser_veto'] = high_risk_veto.astype(bool)

    if not state['crash_mode']:
        out['grr_tail_guard_score'] = out['grr_final_score']
        return out, state

    risk_penalty = (
        float(guard_cfg.get('sigma_penalty', 0.55)) * out['grr_sigma_rank']
        + float(guard_cfg.get('amp_penalty', 0.45)) * out['grr_amp_rank']
        + float(guard_cfg.get('drawdown_penalty', 0.45)) * out['grr_drawdown_rank']
        + float(guard_cfg.get('ret5_extreme_penalty', 0.30)) * out['grr_ret5_extreme']
        + float(guard_cfg.get('low_consensus_penalty', 0.55)) * (1.0 - out['grr_consensus_norm'])
    )
    consensus_quality = out['grr_consensus_norm'] * (
        1.0
        - 0.45 * out['grr_sigma_rank']
        - 0.35 * out['grr_amp_rank']
        - 0.20 * out['grr_drawdown_rank']
    ).clip(lower=0.0, upper=1.0)
    guard_score = (
        float(guard_cfg.get('final_score_weight_crash', 0.40)) * out['grr_final_score']
        + float(guard_cfg.get('minrisk_bonus', 1.10)) * out['grr_minrisk_score']
        + float(guard_cfg.get('consensus_bonus', 0.45)) * consensus_quality
        - float(guard_cfg.get('risk_penalty_scale', 1.0)) * risk_penalty
    )
    guard_score = guard_score.mask(high_risk_veto, guard_score.min() - float(guard_cfg.get('veto_score_gap', 2.0)))
    out['grr_tail_guard_score'] = guard_score.astype(np.float64)
    out['grr_final_score'] = out['grr_tail_guard_score']
    return out, state


def apply_grr_top5(score_df, cfg):
    """Generator + RRF candidate rerank + lightweight online-router score."""
    grr_cfg = cfg.get('grr_top5', cfg)
    expert_cols = [col for col in grr_cfg.get('expert_cols', ['lgb_top5_score', 'lgb', 'transformer', 'score']) if col in score_df.columns]
    out = score_df.copy()
    if len(out) == 0 or not expert_cols:
        return out

    candidate_k = int(grr_cfg.get('candidate_k', 24))
    pool = union_topk_candidates(out, expert_cols, candidate_k=candidate_k)
    if len(pool) == 0:
        return out

    pool = reciprocal_rank_fusion(pool, expert_cols, k=grr_cfg.get('rrf_k', 60), output_col='grr_rrf_score')
    weights = _router_weights(out, expert_cols, grr_cfg.get('router', {}))
    routed = np.zeros(len(pool), dtype=np.float64)
    for weight, col in zip(weights, expert_cols):
        routed += float(weight) * _zscore(pd.to_numeric(pool[col], errors='coerce').fillna(0.0))
    pool['grr_router_score'] = routed

    risk_penalty = np.zeros(len(pool), dtype=np.float64)
    for col, weight in grr_cfg.get('risk_penalty_cols', {'sigma20': 0.05, 'amp20': 0.03, 'max_drawdown20': 0.03}).items():
        if col in pool.columns:
            risk_penalty += float(weight) * pd.to_numeric(pool[col], errors='coerce').rank(pct=True).fillna(0.5).to_numpy()

    pool['grr_final_score'] = (
        float(grr_cfg.get('rrf_weight', 0.45)) * _zscore(pool['grr_rrf_score'])
        + float(grr_cfg.get('router_weight', 0.55)) * pool['grr_router_score']
        - risk_penalty
    )
    pool, guard_state = tail_guard_rerank(pool, expert_cols, grr_cfg) if grr_cfg.get('tail_guard_enabled', True) else (
        pool.assign(
            grr_tail_guard_score=pool['grr_final_score'],
            grr_tail_guard_triggered=False,
            grr_crash_mode=False,
            grr_risk_off_score=0.0,
            grr_high_risk_chaser_veto=False,
        ),
        {'crash_mode': False, 'risk_off_score': 0.0},
    )
    out = out.merge(
        pool[
            [
                'stock_id',
                'grr_rrf_score',
                'grr_router_score',
                'grr_final_score',
                'grr_tail_guard_score',
                'grr_tail_guard_triggered',
                'grr_crash_mode',
                'grr_risk_off_score',
                'grr_high_risk_chaser_veto',
                'grr_consensus_count',
                'grr_consensus_norm',
                'grr_minrisk_score',
            ]
        ],
        on='stock_id',
        how='left',
    )
    fallback = pd.to_numeric(out.get('score', pd.Series(0.0, index=out.index)), errors='coerce').fillna(0.0)
    out['grr_final_score'] = out['grr_final_score'].fillna(_zscore(fallback).min() - 1.0)
    out['grr_tail_guard_score'] = out['grr_tail_guard_score'].fillna(out['grr_final_score'])
    out['grr_tail_guard_triggered'] = out['grr_tail_guard_triggered'].astype('boolean').fillna(False).astype(bool)
    out['grr_crash_mode'] = out['grr_crash_mode'].astype('boolean').fillna(bool(guard_state.get('crash_mode', False))).astype(bool)
    out['grr_risk_off_score'] = out['grr_risk_off_score'].fillna(float(guard_state.get('risk_off_score', 0.0)))
    out['grr_high_risk_chaser_veto'] = out['grr_high_risk_chaser_veto'].astype('boolean').fillna(False).astype(bool)
    if grr_cfg.get('replace_score', True):
        out['score'] = out['grr_final_score']
    return out


def make_rank_labels(df, label_col='label_o2o_week', date_col='日期'):
    out = df.copy()
    out['future_rank'] = out.groupby(date_col)[label_col].rank(method='first', ascending=False)
    out['rank_label'] = 0
    out.loc[out['future_rank'] <= 120, 'rank_label'] = 1
    out.loc[out['future_rank'] <= 60, 'rank_label'] = 2
    out.loc[out['future_rank'] <= 20, 'rank_label'] = 3
    out.loc[out['future_rank'] <= 5, 'rank_label'] = 4
    out['is_top20'] = (out['future_rank'] <= 20).astype(int)
    out['is_top5'] = (out['future_rank'] <= 5).astype(int)
    return out


def train_lgb_ranker(train_df, feature_cols, date_col='日期', label_col='rank_label', seed=42):
    if LGBMRanker is None:
        raise ImportError('lightgbm is required to train the candidate ranker')
    data = train_df.sort_values([date_col, 'stock_id']).reset_index(drop=True)
    group = data.groupby(date_col, sort=False).size().to_numpy()
    model = LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        ndcg_eval_at=[5, 20],
        learning_rate=0.04,
        n_estimators=220,
        num_leaves=31,
        min_child_samples=48,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(_clean_X(data, feature_cols), data[label_col].astype(int), group=group)
    return model


def train_top20_classifier(train_df, feature_cols, label_col='is_top20', seed=42):
    if LGBMClassifier is None:
        raise ImportError('lightgbm is required to train the top20 classifier')
    model = LGBMClassifier(
        objective='binary',
        learning_rate=0.035,
        n_estimators=180,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(_clean_X(train_df, feature_cols), train_df[label_col].astype(int))
    return model


def train_top5_classifier(train_df, feature_cols, label_col='is_top5', seed=42):
    if LGBMClassifier is None:
        raise ImportError('lightgbm is required to train the top5 classifier')
    model = LGBMClassifier(
        objective='binary',
        learning_rate=0.03,
        n_estimators=180,
        num_leaves=15,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=8.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(_clean_X(train_df, feature_cols), train_df[label_col].astype(int))
    return model


def _positive_prob(model, X):
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        return proba[:, 0]
    return proba[:, 1]


def predict_rerank_scores(ranker, top20_clf, top5_clf, pred_df, feature_cols):
    out = pred_df.copy()
    X = _clean_X(out, feature_cols)
    out['rank_score'] = np.asarray(ranker.predict(X), dtype=np.float64)
    out['top20_prob'] = _positive_prob(top20_clf, X)
    out['top5_prob'] = _positive_prob(top5_clf, X)
    out['rerank_score'] = (
        0.50 * _zscore(out['rank_score'])
        + 0.40 * _zscore(out['top20_prob'])
        + 0.10 * _zscore(out['top5_prob'])
    )
    return out


def select_by_reranker(pred_df, candidate_ids=None, k=5):
    out = pred_df.copy()
    if candidate_ids is not None:
        ids = set(str(x).zfill(6) for x in candidate_ids)
        out = out[out['stock_id'].astype(str).str.zfill(6).isin(ids)].copy()
    if len(out) < k:
        out = pred_df.copy()
    return out.sort_values('rerank_score', ascending=False).head(k).copy()
