import numpy as np
import pandas as pd
import os


def normalize_stock_id(value):
    """Return scorer-compatible 6 digit stock id strings."""
    if pd.isna(value):
        return value
    text = str(value).strip()
    if text.endswith('.0'):
        text = text[:-2]
    return text.zfill(6)


def _zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-9)


def _project_with_bounds(raw_weight, exposure_cap=1.0, min_weight=0.0, max_weight=1.0):
    raw = np.clip(np.asarray(raw_weight, dtype=np.float64), 0.0, None)
    n = raw.size
    if n == 0:
        return raw

    exposure_cap = float(max(exposure_cap, 0.0))
    min_weight = float(max(min_weight, 0.0))
    max_weight = float(max(max_weight, min_weight))

    if n * min_weight > exposure_cap + 1e-12:
        min_weight = exposure_cap / n
    if n * max_weight < exposure_cap - 1e-12:
        max_weight = exposure_cap / n

    result = np.full(n, min_weight, dtype=np.float64)
    remaining = exposure_cap - result.sum()
    if remaining <= 1e-12:
        return result

    active = np.ones(n, dtype=bool)
    while remaining > 1e-12 and active.any():
        active_idx = np.where(active)[0]
        active_raw = raw[active_idx]
        if active_raw.sum() <= 1e-12:
            proposal = np.full(active_idx.size, remaining / active_idx.size, dtype=np.float64)
        else:
            proposal = remaining * active_raw / active_raw.sum()

        room = max_weight - result[active_idx]
        overflow = proposal > room + 1e-12
        if not overflow.any():
            result[active_idx] += proposal
            remaining = 0.0
            break

        saturated_idx = active_idx[overflow]
        result[saturated_idx] = max_weight
        active[saturated_idx] = False
        remaining = exposure_cap - result.sum()

    if remaining > 1e-10:
        slack_idx = np.where(result < max_weight - 1e-12)[0]
        if slack_idx.size > 0:
            result[slack_idx] += remaining / slack_idx.size

    return np.clip(result, min_weight, max_weight)


def _to_weight_frame(top, weights):
    out = top[['stock_id']].copy()
    out['weight'] = np.round(weights, 6)
    return out


def _topk(score_df, k=5):
    return score_df.sort_values('score', ascending=False).head(k).copy()


def _available_zscore(out, col, default=0.0):
    if col not in out.columns:
        return np.full(len(out), default, dtype=np.float64)
    values = out[col].replace([np.inf, -np.inf], np.nan).fillna(out[col].median() if out[col].notna().any() else 0.0)
    return _zscore(values.to_numpy(dtype=np.float64))


def _available_log_zscore(out, col, default=0.0):
    if col not in out.columns:
        return np.full(len(out), default, dtype=np.float64)
    values = out[col].replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median() if values.notna().any() else 0.0)
    return _zscore(np.log1p(values.clip(lower=0).to_numpy(dtype=np.float64)))


def _num_col(frame, col, default=0.0):
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=np.float64)
    return pd.to_numeric(frame[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(default)


def _z(values):
    series = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)
    fill = series.median() if series.notna().any() else 0.0
    arr = series.fillna(fill).to_numpy(dtype=np.float64)
    return _zscore(arr)


def _rank_pct(series, ascending=True):
    values = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)
    fill = values.median() if values.notna().any() else 0.0
    return values.fillna(fill).rank(method='average', pct=True, ascending=ascending)


def _risk_value_frame(frame):
    return pd.concat(
        [
            _num_col(frame, 'sigma20'),
            _num_col(frame, 'amp20'),
            _num_col(frame, 'max_drawdown20'),
        ],
        axis=1,
    ).mean(axis=1)


def _riskoff_live_candidates(score_df, top_k=20):
    work = score_df.copy()
    work['stock_id'] = work['stock_id'].map(normalize_stock_id)
    ret20 = _num_col(work, 'ret20')
    sigma20 = _num_col(work, 'sigma20')
    stats = {
        'riskoff_triggered': bool(
            float(ret20.median()) < 0.0
            and float((ret20 > 0).mean()) < 0.45
            and float(sigma20.median()) > 0.018
            and float(ret20.quantile(0.90) - ret20.quantile(0.10)) > 0.10
        ),
        'median_ret20': float(ret20.median()),
        'breadth20': float((ret20 > 0).mean()),
        'median_sigma20': float(sigma20.median()),
        'dispersion20': float(ret20.quantile(0.90) - ret20.quantile(0.10)),
    }
    if not stats['riskoff_triggered']:
        return pd.DataFrame(), stats

    stable = stable_filter(work, liquidity_quantile=0.20, sigma_quantile=0.85).copy()
    if stable.empty:
        return pd.DataFrame(), stats
    stable['fused_z'] = _z(_num_col(stable, 'score'))
    stable['lgb_z'] = _z(_num_col(stable, 'lgb'))
    stable['log_liquidity_z'] = _z(np.log1p(_num_col(stable, 'median_amount20').clip(lower=0.0)))
    stable['ret5_z'] = _z(_num_col(stable, 'ret5'))
    stable['ret20_z'] = _z(_num_col(stable, 'ret20'))
    stable['amp_z'] = _z(_num_col(stable, 'amp20'))
    stable['negative_ret20_penalty'] = (_num_col(stable, 'ret20') < 0.0).astype(float)
    stable['_risk_value'] = _risk_value_frame(stable)
    stable['riskoff_rerank_score'] = (
        0.30 * stable['fused_z']
        + 0.10 * stable['lgb_z']
        + 0.30 * stable['log_liquidity_z']
        + 0.10 * stable['ret5_z']
        + 0.10 * stable['ret20_z']
        - 0.10 * stable['amp_z']
        - 0.50 * stable['negative_ret20_penalty']
    )
    out = stable.sort_values('riskoff_rerank_score', ascending=False).head(top_k).copy()
    out['candidate_rank'] = np.arange(1, len(out) + 1)
    out['candidate_score'] = out['riskoff_rerank_score']
    return out, stats


def _top5_frame(selected):
    out = selected.copy()
    out['stock_id'] = out['stock_id'].map(normalize_stock_id)
    return out.sort_values('score', ascending=False).head(5).copy()


def _risk_target_row(top5, rule='lowest_score'):
    work = top5.copy()
    work['_risk_value'] = _risk_value_frame(work)
    if rule == 'highest_risk':
        return work.sort_values(['_risk_value', 'score'], ascending=[False, True]).iloc[0]
    return work.sort_values(['score', '_risk_value'], ascending=[True, False]).iloc[0]


def _force_selected_top5(final_rows, reason):
    out = final_rows.copy().reset_index(drop=True)
    # build_weight_portfolio sorts by score, so freeze the overlay order with
    # synthetic local scores after a runtime-safe swap decision has been made.
    out['score'] = np.linspace(float(len(out)), 1.0, len(out))
    out['supplemental_overlay_reason'] = reason
    return out


def _apply_riskoff_rank4_dynamic_overlay(score_df, selected, cfg):
    top5 = _top5_frame(selected)
    if len(top5) < 5:
        return selected, None
    rank_cap = int(cfg.get('riskoff_rank_cap', 4))
    risk_delta_cap = float(cfg.get('riskoff_risk_delta_cap', 0.03))
    defensive_risk_max = float(cfg.get('riskoff_defensive_candidate_risk_max', 0.04))
    defensive_gap_min = float(cfg.get('riskoff_defensive_risk_gap_min', 0.10))
    candidates, stats = _riskoff_live_candidates(score_df, top_k=20)
    if candidates.empty:
        return selected, {'accepted': False, 'overlay': 'riskoff_rank4_dynamic', **stats}
    selected_ids = set(top5['stock_id'].astype(str))
    candidates = candidates[
        (~candidates['stock_id'].astype(str).isin(selected_ids))
        & (pd.to_numeric(candidates['candidate_rank'], errors='coerce').fillna(999) <= rank_cap)
    ].copy()
    if candidates.empty:
        return selected, {'accepted': False, 'overlay': 'riskoff_rank4_dynamic', 'blocked_reason': 'no_rank4_candidate', **stats}

    default_target = _risk_target_row(top5, 'lowest_score')
    high_risk_target = _risk_target_row(top5, 'highest_risk')
    for _, candidate in candidates.iterrows():
        candidate_risk = float(candidate.get('_risk_value', 0.0))
        high_risk_gap = float(high_risk_target['_risk_value']) - candidate_risk
        target = high_risk_target if candidate_risk <= defensive_risk_max and high_risk_gap >= defensive_gap_min else default_target
        risk_delta = candidate_risk - float(target['_risk_value'])
        if risk_delta > risk_delta_cap:
            continue
        final = top5[top5['stock_id'] != target['stock_id']].copy()
        final = pd.concat([final, candidate.to_frame().T], ignore_index=True)
        final = final.head(5)
        info = {
            'accepted': True,
            'overlay': 'riskoff_rank4_dynamic',
            'candidate_stock': str(candidate['stock_id']),
            'replaced_stock': str(target['stock_id']),
            'candidate_rank': int(candidate['candidate_rank']),
            'risk_delta': risk_delta,
            **stats,
        }
        return _force_selected_top5(final, 'riskoff_rank4_dynamic'), info
    return selected, {'accepted': False, 'overlay': 'riskoff_rank4_dynamic', 'blocked_reason': 'risk_delta_cap', **stats}


def _pullback_rebound_candidates(score_df, selected_ids, rank_cap=1):
    out = score_df.copy()
    out['stock_id'] = out['stock_id'].map(normalize_stock_id)
    out = out[~out['stock_id'].astype(str).isin(set(selected_ids))].copy()
    if out.empty:
        return out
    ret1 = _num_col(out, 'ret1')
    ret5 = _num_col(out, 'ret5')
    ret10 = _num_col(out, 'ret10')
    ret20 = _num_col(out, 'ret20')
    sigma20 = _num_col(out, 'sigma20')
    amp20 = _num_col(out, 'amp20')
    drawdown20 = _num_col(out, 'max_drawdown20')
    downside_beta60 = _num_col(out, 'downside_beta60')
    beta60 = _num_col(out, 'beta60')
    model = _num_col(out, 'grr_final_score', default=0.0)
    consensus = _num_col(out, 'grr_consensus_norm', default=0.0)
    risk = _risk_value_frame(out)
    market_ret20 = float(ret20.median())
    market_ret10 = float(ret10.median())
    out['liquidity_rank'] = _rank_pct(_num_col(out, 'median_amount20'))
    out['model_rank'] = _rank_pct(model)
    out['consensus_rank'] = _rank_pct(consensus)
    out['risk_rank'] = _rank_pct(risk)
    out['resid_ret20'] = ret20 - beta60 * market_ret20
    out['resid_ret10'] = ret10 - beta60 * market_ret10
    out['pullback_rebound_score'] = (
        0.30 * out['model_rank']
        + 0.20 * _rank_pct(out['resid_ret20'])
        + 0.20 * _rank_pct(-ret5)
        + 0.15 * out['liquidity_rank']
        + 0.10 * out['consensus_rank']
        - 0.20 * out['risk_rank']
        + 0.15 * _rank_pct(ret1)
    )
    mask = (
        (out['liquidity_rank'] >= 0.30)
        & (sigma20 < 0.050)
        & (amp20 < 0.10)
        & (drawdown20 > -0.13)
        & (downside_beta60 < 1.40)
        & (out['model_rank'] >= 0.70)
        & (ret20 > 0.02)
        & (ret20 < 0.32)
        & (ret5 > -0.08)
        & (ret5 < 0.015)
        & (ret1 > 0.0)
    )
    candidates = out[mask].sort_values('pullback_rebound_score', ascending=False).copy()
    candidates['candidate_rank'] = np.arange(1, len(candidates) + 1)
    return candidates[candidates['candidate_rank'] <= int(rank_cap)].copy()


def _apply_pullback_rebound_overlay(score_df, selected, cfg):
    top5 = _top5_frame(selected)
    if len(top5) < 5:
        return selected, None
    candidates = _pullback_rebound_candidates(score_df, top5['stock_id'].astype(str).tolist(), rank_cap=int(cfg.get('pullback_rank_cap', 1)))
    if candidates.empty:
        return selected, {'accepted': False, 'overlay': 'pullback_rebound', 'blocked_reason': 'no_candidate'}
    target = _risk_target_row(top5, 'highest_risk')
    candidate = candidates.iloc[0]
    final = top5[top5['stock_id'] != target['stock_id']].copy()
    final = pd.concat([final, candidate.to_frame().T], ignore_index=True).head(5)
    return _force_selected_top5(final, 'pullback_rebound'), {
        'accepted': True,
        'overlay': 'pullback_rebound',
        'candidate_stock': str(candidate['stock_id']),
        'replaced_stock': str(target['stock_id']),
        'candidate_rank': int(candidate['candidate_rank']),
    }


def _pullback_stable_candidates(score_df, selected_ids, cfg):
    out = score_df.copy()
    out['stock_id'] = out['stock_id'].map(normalize_stock_id)
    if out.empty:
        return out

    ret5 = _num_col(out, 'ret5')
    ret20 = _num_col(out, 'ret20')
    intraday_ret = _num_col(out, 'intraday_ret', default=0.0)
    sigma20 = _num_col(out, 'sigma20')
    amp20 = _num_col(out, 'amp20')
    drawdown20 = _num_col(out, 'max_drawdown20')
    downside_beta60 = _num_col(out, 'downside_beta60')
    lgb = _num_col(out, 'lgb')
    model = _model_score(out)
    risk = _risk_value_frame(out)

    out['model_score'] = model
    out['model_rank'] = _rank_pct(model)
    out['lgb_rank'] = _rank_pct(lgb)
    out['liquidity_rank'] = _rank_pct(_num_col(out, 'median_amount20'))
    out['ret5_rank'] = _rank_pct(ret5)
    out['ret20_rank'] = _rank_pct(ret20)
    out['intraday_rank'] = _rank_pct(intraday_ret)
    out['sigma_rank'] = _rank_pct(sigma20)
    out['amp_rank'] = _rank_pct(amp20)
    out['risk_rank'] = _rank_pct(risk)
    out['dbeta_rank'] = _rank_pct(downside_beta60)
    out['pullback_stable_score'] = (
        0.35 * (1.0 - out['ret5_rank'])
        + 0.25 * out['ret20_rank']
        + 0.20 * out['intraday_rank']
        + 0.15 * out['lgb_rank']
        + 0.10 * out['liquidity_rank']
        + 0.05 * out['model_rank']
        - 0.15 * out['sigma_rank']
        - 0.10 * out['amp_rank']
    )
    mask = (
        (out['liquidity_rank'] >= float(cfg.get('pullback_stable_liquidity_rank_min', 0.20)))
        & (ret20 >= float(cfg.get('pullback_stable_ret20_min', 0.0)))
        & (ret20 <= float(cfg.get('pullback_stable_ret20_max', 0.35)))
        & (out['sigma_rank'] <= float(cfg.get('pullback_stable_sigma_rank_max', 0.85)))
        & (out['amp_rank'] <= float(cfg.get('pullback_stable_amp_rank_max', 0.90)))
        & (out['dbeta_rank'] <= float(cfg.get('pullback_stable_dbeta_rank_max', 0.90)))
        & (drawdown20 <= float(cfg.get('pullback_stable_drawdown_max', 0.16)))
        & (
            (out['lgb_rank'] >= float(cfg.get('pullback_stable_lgb_rank_min', 0.50)))
            | (out['model_rank'] >= float(cfg.get('pullback_stable_model_rank_min', 0.45)))
        )
    )
    selected_ids = {str(x) for x in selected_ids}
    candidates = out[mask & ~out['stock_id'].astype(str).isin(selected_ids)].sort_values('pullback_stable_score', ascending=False).copy()
    candidates['candidate_rank'] = np.arange(1, len(candidates) + 1)
    rank_cap = int(cfg.get('pullback_stable_rank_cap', 1))
    candidates = candidates[candidates['candidate_rank'] <= rank_cap].copy()
    candidates['candidate_score'] = candidates['pullback_stable_score']
    return candidates


def _apply_pullback_stable_overlay(score_df, selected, cfg):
    top5 = _top5_frame(selected)
    if len(top5) < 5:
        return selected, None
    candidates = _pullback_stable_candidates(score_df, top5['stock_id'].astype(str).tolist(), cfg)
    if candidates.empty:
        return selected, {'accepted': False, 'overlay': 'pullback_stable_booster', 'blocked_reason': 'no_candidate'}

    candidate = candidates.iloc[0]
    target = _risk_target_row(top5, 'highest_risk')
    candidate_risk = float(_risk_value_frame(candidate.to_frame().T).iloc[0])
    target_risk = float(target.get('_risk_value', _risk_value_frame(target.to_frame().T).iloc[0]))
    risk_delta = candidate_risk - target_risk
    if risk_delta > float(cfg.get('pullback_stable_risk_delta_cap', 0.12)):
        return selected, {
            'accepted': False,
            'overlay': 'pullback_stable_booster',
            'blocked_reason': 'risk_delta_cap',
            'candidate_stock': str(candidate['stock_id']),
            'replaced_stock': str(target['stock_id']),
            'risk_delta': risk_delta,
        }

    final = top5[top5['stock_id'] != target['stock_id']].copy()
    final = pd.concat([final, candidate.to_frame().T], ignore_index=True).head(5)
    return _force_selected_top5(final, 'pullback_stable_booster'), {
        'accepted': True,
        'overlay': 'pullback_stable_booster',
        'candidate_stock': str(candidate['stock_id']),
        'replaced_stock': str(target['stock_id']),
        'candidate_rank': int(candidate['candidate_rank']),
        'candidate_score': float(candidate['candidate_score']),
        'risk_delta': risk_delta,
    }


def _ret5_guarded_candidates(score_df, selected_ids, cfg):
    out = score_df.copy()
    out['stock_id'] = out['stock_id'].map(normalize_stock_id)
    if out.empty:
        return out

    ret5 = _num_col(out, 'ret5')
    ret20 = _num_col(out, 'ret20')
    sigma20 = _num_col(out, 'sigma20')
    amp20 = _num_col(out, 'amp20')
    drawdown20 = _num_col(out, 'max_drawdown20')
    downside_beta60 = _num_col(out, 'downside_beta60')
    lgb = _num_col(out, 'lgb')

    out['liquidity_rank'] = _rank_pct(_num_col(out, 'median_amount20'))
    out['lgb_rank'] = _rank_pct(lgb)
    out['ret5_rank'] = _rank_pct(ret5)
    out['sigma_rank'] = _rank_pct(sigma20)
    out['amp_rank'] = _rank_pct(amp20)
    out['dbeta_rank'] = _rank_pct(downside_beta60)
    out['ret5_guarded_score'] = (
        0.85 * out['ret5_rank']
        + 0.05 * out['lgb_rank']
        + 0.05 * out['liquidity_rank']
        - 0.03 * out['sigma_rank']
        - 0.03 * out['amp_rank']
    )
    mask = (
        (ret5 >= float(cfg.get('ret5_guarded_ret5_min', 0.08)))
        & (ret20 >= float(cfg.get('ret5_guarded_ret20_min', 0.08)))
        & (ret20 <= float(cfg.get('ret5_guarded_ret20_max', 0.45)))
        & (sigma20 <= float(cfg.get('ret5_guarded_sigma20_max', 0.050)))
        & (amp20 <= float(cfg.get('ret5_guarded_amp20_max', 0.30)))
        & (drawdown20 <= float(cfg.get('ret5_guarded_drawdown_max', 0.16)))
        & (downside_beta60 <= float(cfg.get('ret5_guarded_downside_beta_max', 1.50)))
        & (out['liquidity_rank'] >= float(cfg.get('ret5_guarded_liquidity_rank_min', 0.20)))
        & (out['lgb_rank'] >= float(cfg.get('ret5_guarded_lgb_rank_min', 0.45)))
    )
    selected_ids = {str(x) for x in selected_ids}
    candidates = out[mask & ~out['stock_id'].astype(str).isin(selected_ids)].sort_values('ret5_guarded_score', ascending=False).copy()
    candidates['candidate_rank'] = np.arange(1, len(candidates) + 1)
    rank_cap = int(cfg.get('ret5_guarded_rank_cap', 3))
    candidates = candidates[candidates['candidate_rank'] <= rank_cap].copy()
    candidates['candidate_score'] = candidates['ret5_guarded_score']
    return candidates


def _apply_ret5_guarded_overlay(score_df, selected, cfg):
    top5 = _top5_frame(selected)
    if len(top5) < 5:
        return selected, None
    max_swaps = int(cfg.get('ret5_guarded_max_swaps', 3))
    if max_swaps <= 0:
        return selected, None
    candidates = _ret5_guarded_candidates(score_df, top5['stock_id'].astype(str).tolist(), cfg)
    if candidates.empty:
        return selected, {'accepted': False, 'overlay': 'ret5_guarded_booster', 'blocked_reason': 'no_candidate'}

    target_order = top5.sort_values(['ret5', 'score'], ascending=[True, True]).copy()
    swap_count = min(max_swaps, len(candidates), len(target_order))
    targets = target_order.head(swap_count)
    chosen = candidates.head(swap_count)
    final = top5[~top5['stock_id'].astype(str).isin(set(targets['stock_id'].astype(str)))].copy()
    final = pd.concat([final, chosen], ignore_index=True).head(5)
    return _force_selected_top5(final, 'ret5_guarded_booster'), {
        'accepted': True,
        'overlay': 'ret5_guarded_booster',
        'candidate_stock': ','.join(chosen['stock_id'].astype(str).tolist()),
        'replaced_stock': ','.join(targets['stock_id'].astype(str).tolist()),
        'candidate_rank': ','.join(chosen['candidate_rank'].astype(int).astype(str).tolist()),
        'candidate_score': float(chosen['candidate_score'].mean()),
        'swap_count': int(swap_count),
    }


def _cooldown_minrisk_state(score_df, cfg):
    ret20 = _num_col(score_df, 'ret20')
    ret5 = _num_col(score_df, 'ret5')
    ret1 = _num_col(score_df, 'ret1')
    sigma20 = _num_col(score_df, 'sigma20')
    stats = {
        'median_ret20': float(ret20.median()),
        'breadth20': float((ret20 > 0).mean()),
        'breadth5': float((ret5 > 0).mean()),
        'breadth1': float((ret1 > 0).mean()),
        'median_sigma20': float(sigma20.median()),
        'dispersion20': float(ret20.quantile(0.90) - ret20.quantile(0.10)),
    }
    cold_pullback = (
        stats['median_ret20'] >= float(cfg.get('cooldown_minrisk_cold_median_ret20_min', -0.015))
        and stats['median_ret20'] <= float(cfg.get('cooldown_minrisk_cold_median_ret20_max', 0.020))
        and stats['breadth20'] <= float(cfg.get('cooldown_minrisk_cold_breadth20_max', 0.58))
        and stats['breadth5'] <= float(cfg.get('cooldown_minrisk_cold_breadth5_max', 0.50))
        and stats['dispersion20'] >= float(cfg.get('cooldown_minrisk_cold_dispersion20_min', 0.20))
    )
    weak_broad_rebound = (
        stats['median_ret20'] >= float(cfg.get('cooldown_minrisk_rebound_median_ret20_min', -0.015))
        and stats['median_ret20'] <= float(cfg.get('cooldown_minrisk_rebound_median_ret20_max', 0.0))
        and stats['breadth20'] <= float(cfg.get('cooldown_minrisk_rebound_breadth20_max', 0.48))
        and stats['breadth5'] >= float(cfg.get('cooldown_minrisk_rebound_breadth5_min', 0.55))
        and stats['breadth1'] <= float(cfg.get('cooldown_minrisk_rebound_breadth1_max', 0.50))
        and stats['dispersion20'] >= float(cfg.get('cooldown_minrisk_rebound_dispersion20_min', 0.22))
    )
    stats['cooldown_mode'] = 'cold_pullback' if cold_pullback else ('weak_broad_rebound' if weak_broad_rebound else '')
    stats['cooldown_minrisk_triggered'] = bool(cold_pullback or weak_broad_rebound)
    return stats


def _cooldown_minrisk_candidates(score_df):
    out = _add_runtime_minrisk_scores(score_df)
    if 'score_legal_minrisk' not in out.columns:
        out['score_legal_minrisk'] = (
            0.35 * _num_col(out, 'tf_norm')
            + 0.15 * _num_col(out, 'lgb_norm')
            + 0.20 * out['liq_rank']
            - 0.15 * out['sigma_rank']
            - 0.10 * out['downside_beta60_rank']
            - 0.10 * out['max_drawdown20_rank']
            - 0.05 * out['amp_rank']
        )
    candidates = legal_minrisk_hardened_filter(out).copy()
    if len(candidates) >= 5:
        return candidates.sort_values('score_legal_minrisk', ascending=False)
    return out.sort_values('score_legal_minrisk', ascending=False)


def _apply_cooldown_minrisk_overlay(score_df, selected, cfg):
    if not bool(cfg.get('cooldown_minrisk_enabled', True)):
        return selected, None
    stats = _cooldown_minrisk_state(score_df, cfg)
    if not stats['cooldown_minrisk_triggered']:
        return selected, {'accepted': False, 'overlay': 'cooldown_minrisk_repair', **stats}
    candidates = _cooldown_minrisk_candidates(score_df)
    if len(candidates) < 5:
        return selected, {'accepted': False, 'overlay': 'cooldown_minrisk_repair', 'blocked_reason': 'not_enough_candidates', **stats}
    current = _top5_frame(selected)
    chosen = candidates.head(5).copy()
    score_col = 'score_legal_minrisk' if 'score_legal_minrisk' in chosen.columns else 'runtime_minrisk_score'
    return _force_selected_top5(chosen, 'cooldown_minrisk_repair'), {
        'accepted': True,
        'overlay': 'cooldown_minrisk_repair',
        'candidate_stock': ','.join(chosen['stock_id'].astype(str).tolist()),
        'replaced_stock': ','.join(current['stock_id'].astype(str).tolist()),
        'candidate_score': float(_num_col(chosen, score_col).mean()),
        'swap_count': 5,
        **stats,
    }


def _deep_rebound_state(score_df, cfg):
    ret20 = _num_col(score_df, 'ret20')
    amp20 = _num_col(score_df, 'amp20')
    stats = {
        'median_ret20': float(ret20.median()),
        'breadth20': float((ret20 > 0).mean()),
        'median_amp20': float(amp20.median()),
    }
    stats['deep_rebound_triggered'] = bool(
        stats['median_ret20'] <= float(cfg.get('deep_rebound_median_ret20_max', -0.055))
        and stats['breadth20'] <= float(cfg.get('deep_rebound_breadth20_max', 0.30))
        and stats['median_amp20'] >= float(cfg.get('deep_rebound_median_amp20_min', 0.16))
    )
    return stats


def _deep_rebound_candidates(score_df, cfg):
    out = score_df.copy()
    out['stock_id'] = out['stock_id'].map(normalize_stock_id)
    if out.empty:
        return out

    ret20 = _num_col(out, 'ret20')
    amp20 = _num_col(out, 'amp20')
    sigma20 = _num_col(out, 'sigma20')
    out['tf_rank'] = _rank_pct(_num_col(out, 'transformer', default=0.0))
    out['lgb_rank'] = _rank_pct(_num_col(out, 'lgb'))
    out['ret20_rank'] = _rank_pct(ret20)
    out['amp_rank'] = _rank_pct(amp20)
    out['deep_rebound_score'] = (
        1.00 * out['tf_rank']
        + 0.05 * out['lgb_rank']
        + 0.15 * out['ret20_rank']
        + 0.05 * out['amp_rank']
    )
    mask = (
        (ret20 <= float(cfg.get('deep_rebound_ret20_max', 0.30)))
        & (amp20 >= float(cfg.get('deep_rebound_amp20_min', 0.05)))
        & (sigma20 <= float(cfg.get('deep_rebound_sigma20_max', 0.060)))
    )
    candidates = out[mask].sort_values('deep_rebound_score', ascending=False).head(5).copy()
    candidates['candidate_rank'] = np.arange(1, len(candidates) + 1)
    candidates['candidate_score'] = candidates['deep_rebound_score']
    return candidates


def _apply_deep_rebound_overlay(score_df, selected, cfg):
    stats = _deep_rebound_state(score_df, cfg)
    if not stats['deep_rebound_triggered']:
        return selected, {'accepted': False, 'overlay': 'deep_rebound_repair', **stats}
    candidates = _deep_rebound_candidates(score_df, cfg)
    if len(candidates) < 5:
        return selected, {'accepted': False, 'overlay': 'deep_rebound_repair', 'blocked_reason': 'not_enough_candidates', **stats}
    current = _top5_frame(selected)
    return _force_selected_top5(candidates.head(5), 'deep_rebound_repair'), {
        'accepted': True,
        'overlay': 'deep_rebound_repair',
        'candidate_stock': ','.join(candidates['stock_id'].astype(str).head(5).tolist()),
        'replaced_stock': ','.join(current['stock_id'].astype(str).head(5).tolist()),
        'candidate_rank': ','.join(candidates['candidate_rank'].astype(int).astype(str).head(5).tolist()),
        'candidate_score': float(candidates['candidate_score'].head(5).mean()),
        'swap_count': 5,
        **stats,
    }


def _model_score(frame):
    if 'grr_final_score' in frame.columns:
        return _num_col(frame, 'grr_final_score')
    return _num_col(frame, 'score')


def _add_runtime_anti_lottery_scores(score_df):
    out = score_df.copy()
    out['stock_id'] = out['stock_id'].map(normalize_stock_id)
    ret5 = _num_col(out, 'ret5')
    ret20 = _num_col(out, 'ret20')
    sigma20 = _num_col(out, 'sigma20')
    amp20 = _num_col(out, 'amp20')
    drawdown20 = _num_col(out, 'max_drawdown20')
    downside_beta60 = _num_col(out, 'downside_beta60')
    risk = _risk_value_frame(out)
    max_ret20 = _num_col(out, 'max_ret20_raw')
    max_jump20 = _num_col(out, 'max_high_jump20')

    out['model_score'] = _model_score(out)
    out['model_rank'] = _rank_pct(out['model_score'])
    out['liquidity_rank'] = _rank_pct(_num_col(out, 'median_amount20'))
    out['risk_rank'] = _rank_pct(risk)
    out['max_ret_rank'] = _rank_pct(max_ret20)
    out['max_jump_rank'] = _rank_pct(max_jump20)
    out['anti_lottery_score'] = (
        0.35 * out['model_rank']
        + 0.20 * out['liquidity_rank']
        + 0.20 * (1.0 - out['max_ret_rank'])
        + 0.15 * (1.0 - out['max_jump_rank'])
        - 0.15 * out['risk_rank']
    )
    out['pass_anti_lottery'] = (
        (out['liquidity_rank'] >= 0.30)
        & (out['model_rank'] >= 0.70)
        & (sigma20 < 0.050)
        & (amp20 < 0.10)
        & (drawdown20 > -0.13)
        & (downside_beta60 < 1.40)
        & (out['max_ret_rank'] <= 0.55)
        & (out['max_jump_rank'] <= 0.60)
        & (ret5 > -0.04)
        & (ret20 < 0.25)
    )
    return out


def _anti_lottery_candidates(score_df, selected_ids, cfg):
    out = _add_runtime_anti_lottery_scores(score_df)
    selected_ids = {str(x) for x in selected_ids}
    candidates = out[
        ~out['stock_id'].astype(str).isin(selected_ids)
        & out['pass_anti_lottery'].astype(bool)
    ].copy()
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values('anti_lottery_score', ascending=False).copy()
    candidates['candidate_rank'] = np.arange(1, len(candidates) + 1)
    rank_cap = int(cfg.get('anti_lottery_rank_cap', 1))
    candidates = candidates[candidates['candidate_rank'] <= rank_cap].copy()
    candidates['candidate_score'] = candidates['anti_lottery_score']
    return candidates


def _anti_lottery_target(top5):
    work = top5.copy()
    work['_risk_value'] = _risk_value_frame(work)
    work['_model_score'] = _model_score(work)
    return work.sort_values(['_model_score', '_risk_value'], ascending=[True, False]).iloc[0]


def _apply_conditional_anti_lottery_overlay(score_df, selected, cfg):
    if not bool(cfg.get('anti_lottery_overlay_enabled', False)):
        return selected, None
    top5 = _top5_frame(selected)
    if len(top5) < 5:
        return selected, None
    target = _anti_lottery_target(top5)
    dbeta_guard = float(cfg.get('anti_lottery_dbeta_guard_max', 1.35))
    target_dbeta = float(pd.to_numeric(target.get('downside_beta60', 0.0), errors='coerce'))
    if target_dbeta > dbeta_guard:
        return selected, {
            'accepted': False,
            'overlay': 'conditional_anti_lottery_dbeta_guard',
            'blocked_reason': 'target_dbeta_guard',
            'replaced_stock': str(target['stock_id']),
            'target_downside_beta60': target_dbeta,
            'guard_max': dbeta_guard,
        }

    selected_ids = set(top5['stock_id'].astype(str))
    candidates = _anti_lottery_candidates(score_df, selected_ids, cfg)
    target_score = float(target.get('_model_score', 0.0))
    if not candidates.empty:
        candidates = candidates[candidates['model_score'] > target_score].copy()
    if candidates.empty:
        return selected, {
            'accepted': False,
            'overlay': 'conditional_anti_lottery_dbeta_guard',
            'blocked_reason': 'no_candidate',
            'replaced_stock': str(target['stock_id']),
            'target_model_score': target_score,
        }

    candidate = candidates.iloc[0]
    final = top5[top5['stock_id'] != target['stock_id']].copy()
    final = pd.concat([final, candidate.to_frame().T], ignore_index=True).head(5)
    return _force_selected_top5(final, 'conditional_anti_lottery_dbeta_guard'), {
        'accepted': True,
        'overlay': 'conditional_anti_lottery_dbeta_guard',
        'candidate_stock': str(candidate['stock_id']),
        'replaced_stock': str(target['stock_id']),
        'candidate_rank': int(candidate['candidate_rank']),
        'candidate_score': float(candidate['candidate_score']),
        'target_model_score': target_score,
        'target_downside_beta60': target_dbeta,
    }


def _stress_state(score_df, cfg):
    ret20 = _num_col(score_df, 'ret20')
    sigma20 = _num_col(score_df, 'sigma20')
    stats = {
        'median_ret20': float(ret20.median()),
        'breadth20': float((ret20 > 0).mean()),
        'median_sigma20': float(sigma20.median()),
        'dispersion20': float(ret20.quantile(0.90) - ret20.quantile(0.10)),
    }
    stats['stress_triggered'] = bool(
        stats['median_ret20'] < float(cfg.get('stress_chaser_median_ret20_max', 0.0))
        and stats['breadth20'] <= float(cfg.get('stress_chaser_breadth20_max', 0.50))
        and stats['median_sigma20'] > float(cfg.get('stress_chaser_median_sigma20_min', 0.018))
        and stats['dispersion20'] > float(cfg.get('stress_chaser_dispersion20_min', 0.10))
    )
    return stats


def _add_runtime_minrisk_scores(score_df):
    out = score_df.copy()
    out['stock_id'] = out['stock_id'].map(normalize_stock_id)
    out['liq_rank'] = _rank_pct(_num_col(out, 'median_amount20'))
    out['sigma_rank'] = _rank_pct(_num_col(out, 'sigma20'))
    out['amp_rank'] = _rank_pct(_num_col(out, 'amp20'))
    out['ret1_rank'] = _rank_pct(_num_col(out, 'ret1'))
    out['ret5_rank'] = _rank_pct(_num_col(out, 'ret5'))
    out['downside_beta60_rank'] = _rank_pct(_num_col(out, 'downside_beta60'))
    out['max_drawdown20_rank'] = _rank_pct(_num_col(out, 'max_drawdown20'))
    out['tail_risk_flag'] = (
        (out['sigma_rank'] > 0.85)
        | (out['amp_rank'] > 0.85)
        | (out['downside_beta60_rank'] > 0.85)
        | (out['max_drawdown20_rank'] > 0.85)
    )
    out['reversal_flag'] = (out['ret5_rank'] > 0.70) & (out['ret1_rank'] < 0.30)
    out['runtime_minrisk_score'] = (
        0.35 * _rank_pct(_num_col(out, 'score'))
        + 0.15 * _rank_pct(_num_col(out, 'lgb'))
        + 0.20 * out['liq_rank']
        - 0.15 * out['sigma_rank']
        - 0.10 * out['downside_beta60_rank']
        - 0.10 * out['max_drawdown20_rank']
        - 0.05 * out['amp_rank']
    )
    return out


def _runtime_minrisk_candidates(score_df, selected_ids):
    out = _add_runtime_minrisk_scores(score_df)
    selected_ids = {str(x) for x in selected_ids}
    cond = (
        ~out['stock_id'].astype(str).isin(selected_ids)
        & ~out['tail_risk_flag'].astype(bool)
        & ~out['reversal_flag'].astype(bool)
        & (out['liq_rank'] >= 0.15)
        & (out['sigma_rank'] <= 0.75)
        & (out['downside_beta60_rank'] <= 0.75)
        & (out['max_drawdown20_rank'] <= 0.75)
        & (out['amp_rank'] <= 0.85)
        & (out['ret1_rank'] >= 0.20)
    )
    candidates = out[cond].copy()
    if len(candidates) < 5:
        relaxed = out[
            ~out['stock_id'].astype(str).isin(selected_ids)
            & (out['liq_rank'] >= 0.10)
            & (out['sigma_rank'] <= 0.85)
            & ~out['reversal_flag'].astype(bool)
        ].copy()
        if len(relaxed) >= 1:
            candidates = relaxed
    return candidates.sort_values('runtime_minrisk_score', ascending=False)


def _stress_chaser_target(top5, cfg):
    work = top5.copy()
    ret1 = _num_col(work, 'ret1')
    ret5 = _num_col(work, 'ret5')
    ret20 = _num_col(work, 'ret20')
    amp20 = _num_col(work, 'amp20')
    downside_beta60 = _num_col(work, 'downside_beta60')
    panic = (
        (ret1 < float(cfg.get('stress_panic_ret1_max', -0.05)))
        & (downside_beta60 > float(cfg.get('stress_panic_downside_beta_min', 1.50)))
        & (ret20 > 0.0)
    )
    hot = (
        (ret20 > 0.0)
        & (ret20 < float(cfg.get('stress_hot_ret20_max', 0.25)))
        & (ret5 > float(cfg.get('stress_hot_ret5_min', 0.04)))
        & (amp20 > float(cfg.get('stress_hot_amp20_min', 0.12)))
        & (downside_beta60 > float(cfg.get('stress_hot_downside_beta_min', 0.80)))
    )
    work['_stress_chaser_score'] = (
        panic.astype(float) * 100.0
        + hot.astype(float) * (1.0 + amp20 + downside_beta60 / 10.0)
    )
    targets = work[work['_stress_chaser_score'] > 0.0].sort_values('_stress_chaser_score', ascending=False)
    if targets.empty:
        return None
    return targets.iloc[0]


def _apply_stress_chaser_veto_overlay(score_df, selected, cfg):
    if not bool(cfg.get('stress_chaser_veto_enabled', False)):
        return selected, None
    top5 = _top5_frame(selected)
    if len(top5) < 5:
        return selected, None
    stats = _stress_state(score_df, cfg)
    if not stats['stress_triggered']:
        return selected, {'accepted': False, 'overlay': 'stress_chaser_veto', **stats}
    target = _stress_chaser_target(top5, cfg)
    if target is None:
        return selected, {'accepted': False, 'overlay': 'stress_chaser_veto', 'blocked_reason': 'no_stress_chaser_target', **stats}
    selected_ids = set(top5['stock_id'].astype(str))
    candidates = _runtime_minrisk_candidates(score_df, selected_ids)
    if candidates.empty:
        return selected, {'accepted': False, 'overlay': 'stress_chaser_veto', 'blocked_reason': 'no_minrisk_candidate', **stats}
    candidate = candidates.iloc[0]
    final = top5[top5['stock_id'] != target['stock_id']].copy()
    final = pd.concat([final, candidate.to_frame().T], ignore_index=True).head(5)
    info = {
        'accepted': True,
        'overlay': 'stress_chaser_veto',
        'candidate_stock': str(candidate['stock_id']),
        'replaced_stock': str(target['stock_id']),
        'target_score': float(target['_stress_chaser_score']),
        **stats,
    }
    return _force_selected_top5(final, 'stress_chaser_veto'), info


def apply_supplemental_overlay(score_df, selected, cfg=None):
    cfg = cfg or {}
    if not bool(cfg.get('supplemental_overlay_enabled', False)):
        return selected
    if bool(cfg.get('supplemental_overlay_shadow_only', True)):
        return selected
    priority = cfg.get('supplemental_overlay_priority', [])
    current = selected
    diagnostics = []
    for name in priority:
        if name == 'riskoff_fill_rank4_dynamic_defensive_target_no_v2b_swap':
            current, info = _apply_riskoff_rank4_dynamic_overlay(score_df, current, cfg)
        elif name == 'pullback_rebound_highest_risk':
            current, info = _apply_pullback_rebound_overlay(score_df, current, cfg)
        elif name == 'deep_rebound_repair':
            current, info = _apply_deep_rebound_overlay(score_df, current, cfg)
        elif name == 'cooldown_minrisk_repair':
            current, info = _apply_cooldown_minrisk_overlay(score_df, current, cfg)
        elif name == 'ret5_guarded_booster':
            current, info = _apply_ret5_guarded_overlay(score_df, current, cfg)
        elif name == 'pullback_stable_booster':
            current, info = _apply_pullback_stable_overlay(score_df, current, cfg)
        else:
            info = {'accepted': False, 'overlay': name, 'blocked_reason': 'unsupported'}
        if info:
            diagnostics.append(info)
            if info.get('accepted'):
                break

    max_swaps = int(cfg.get('max_total_swaps', cfg.get('supplemental_overlay_max_swaps', 1)))
    accepted_count = sum(int(info.get('swap_count', 1)) for info in diagnostics if info.get('accepted'))
    skip_stress = any(
        info.get('accepted') and info.get('overlay') in {'deep_rebound_repair', 'cooldown_minrisk_repair'}
        for info in diagnostics
    )
    if not skip_stress:
        current, veto_info = _apply_stress_chaser_veto_overlay(score_df, current, cfg)
        if veto_info:
            diagnostics.append(veto_info)
        accepted_count = sum(int(info.get('swap_count', 1)) for info in diagnostics if info.get('accepted'))
    if accepted_count < max_swaps:
        current, anti_lottery_info = _apply_conditional_anti_lottery_overlay(score_df, current, cfg)
        if anti_lottery_info:
            diagnostics.append(anti_lottery_info)

    current.attrs['supplemental_overlay_info'] = diagnostics
    accepted = [info for info in diagnostics if info.get('accepted')]
    if accepted:
        print(f"[BDC][supplemental_overlay] accepted={accepted}")
    else:
        print(f"[BDC][supplemental_overlay] no_swap diagnostics={diagnostics}")
    return current


def equal_weight_portfolio(score_df, k=5, exposure_cap=1.0):
    top = _topk(score_df, k=k)
    if len(top) == 0:
        return top.assign(weight=pd.Series(dtype=float))
    weights = np.full(len(top), exposure_cap / len(top), dtype=np.float64)
    return _to_weight_frame(top, weights)


def fixed_descending_weight_portfolio(score_df, k=5, exposure_cap=1.0):
    top = _topk(score_df, k=k)
    base = np.array([0.24, 0.22, 0.20, 0.18, 0.16], dtype=np.float64)[:len(top)]
    if len(top) == 0:
        return top.assign(weight=pd.Series(dtype=float))
    weights = base / (base.sum() + 1e-12) * exposure_cap
    return _to_weight_frame(top, weights)


def risk_soft_weight_portfolio(score_df, k=5, tau=0.6, max_weight=0.50, exposure_cap=1.0):
    top = _topk(score_df, k=k)
    score = _zscore(top['score'].to_numpy(dtype=np.float64))
    risk = np.clip(top['sigma20'].to_numpy(dtype=np.float64), 1e-4, None)
    raw_weight = np.exp(score / tau) / risk
    weights = _project_with_bounds(raw_weight, exposure_cap=exposure_cap, min_weight=0.0, max_weight=max_weight)
    return _to_weight_frame(top, weights)


def score_soft_weight_portfolio(score_df, k=5, tau=1.0, min_weight=0.05, max_weight=0.50, exposure_cap=1.0):
    top = _topk(score_df, k=k)
    score = _zscore(top['score'].to_numpy(dtype=np.float64))
    raw_weight = np.exp(score / tau)
    weights = _project_with_bounds(raw_weight, exposure_cap=exposure_cap, min_weight=min_weight, max_weight=max_weight)
    return _to_weight_frame(top, weights)


def shrunk_score_soft_weight_portfolio(
    score_df,
    k=5,
    temperature=3.0,
    rho=0.20,
    min_weight=0.05,
    max_weight=0.35,
    exposure_cap=1.0,
):
    """Softmax weights shrunk toward equal weights to avoid over-trusting score scale."""
    top = _topk(score_df, k=k)
    if len(top) == 0:
        return top.assign(weight=pd.Series(dtype=float))

    score = _zscore(top['score'].to_numpy(dtype=np.float64))
    raw = np.exp(score / max(float(temperature), 1e-6))
    soft = raw / (raw.sum() + 1e-12) * exposure_cap
    equal = np.full(len(top), exposure_cap / len(top), dtype=np.float64)
    mixed = (1.0 - float(rho)) * equal + float(rho) * soft
    weights = _project_with_bounds(mixed, exposure_cap=exposure_cap, min_weight=min_weight, max_weight=max_weight)
    return _to_weight_frame(top, weights)


def score_risk_soft_weight_portfolio(score_df, k=5, tau=1.0, risk_power=0.5, min_weight=0.05, max_weight=0.50, exposure_cap=1.0):
    top = _topk(score_df, k=k)
    score = _zscore(top['score'].to_numpy(dtype=np.float64))
    risk = np.clip(top['sigma20'].to_numpy(dtype=np.float64), 1e-4, None)
    raw_weight = np.exp(score / tau) / np.power(risk, risk_power)
    weights = _project_with_bounds(raw_weight, exposure_cap=exposure_cap, min_weight=min_weight, max_weight=max_weight)
    return _to_weight_frame(top, weights)


def inv_vol_weight_portfolio(score_df, k=5, min_weight=0.10, max_weight=0.40, exposure_cap=1.0):
    top = _topk(score_df, k=k)
    risk = np.clip(top['sigma20'].to_numpy(dtype=np.float64), 1e-4, None)
    raw_weight = 1.0 / risk
    weights = _project_with_bounds(raw_weight, exposure_cap=exposure_cap, min_weight=min_weight, max_weight=max_weight)
    return _to_weight_frame(top, weights)


def current_filter(score_df, liquidity_quantile=0.20):
    out = score_df.copy()
    if 'median_amount20' not in out.columns:
        return out
    liquidity_floor = out['median_amount20'].quantile(liquidity_quantile)
    filtered = out[out['median_amount20'] >= liquidity_floor].copy()
    return filtered if len(filtered) >= 5 else out


def liquidity_floor_filter(score_df, liquidity_quantile=0.05):
    out = score_df.copy()
    if 'median_amount20' not in out.columns:
        return out
    filtered = out[out['median_amount20'] >= out['median_amount20'].quantile(liquidity_quantile)].copy()
    return filtered if len(filtered) >= 5 else out


def stable_filter(score_df, liquidity_quantile=0.20, sigma_quantile=0.85):
    out = current_filter(score_df, liquidity_quantile=liquidity_quantile)
    if 'sigma20' not in out.columns:
        return out
    sigma_cap = out['sigma20'].quantile(sigma_quantile)
    filtered = out[out['sigma20'] <= sigma_cap].copy()
    return filtered if len(filtered) >= 5 else out


def no_extreme_momentum_filter(score_df, liquidity_quantile=0.20, sigma_quantile=0.85):
    out = stable_filter(score_df, liquidity_quantile=liquidity_quantile, sigma_quantile=sigma_quantile)
    required_cols = {'ret5', 'amp20'}
    if not required_cols.issubset(out.columns):
        return out
    ret5_low = out['ret5'].quantile(0.10)
    ret5_high = out['ret5'].quantile(0.90)
    amp_cap = out['amp20'].quantile(0.90)
    filtered = out[
        (out['ret5'] >= ret5_low)
        & (out['ret5'] <= ret5_high)
        & (out['amp20'] <= amp_cap)
    ].copy()
    return filtered if len(filtered) >= 5 else out


def liq_sigma_filter(score_df, liquidity_quantile=0.30, sigma_quantile=0.70):
    out = stable_filter(
        score_df,
        liquidity_quantile=liquidity_quantile,
        sigma_quantile=sigma_quantile,
    )
    return out if len(out) >= 5 else score_df


def defensive_filter(
    score_df,
    liquidity_quantile=0.30,
    sigma_quantile=0.70,
    amp_quantile=0.70,
):
    out = score_df.copy()
    cond = pd.Series(True, index=out.index)

    if 'median_amount20' in out.columns:
        cond &= out['median_amount20'] >= out['median_amount20'].quantile(liquidity_quantile)
    if 'sigma20' in out.columns:
        cond &= out['sigma20'] <= out['sigma20'].quantile(sigma_quantile)
    if 'amp20' in out.columns:
        cond &= out['amp20'] <= out['amp20'].quantile(amp_quantile)

    ret1_col = 'ret1' if 'ret1' in out.columns else 'return_1' if 'return_1' in out.columns else None
    if ret1_col is not None:
        cond &= out[ret1_col] > -0.035

    ret5_col = 'ret5' if 'ret5' in out.columns else 'return_5' if 'return_5' in out.columns else None
    if ret5_col is not None:
        cond &= out[ret5_col] > -0.08
        cond &= out[ret5_col] < out[ret5_col].quantile(0.90)

    filtered = out[cond].copy()
    return filtered if len(filtered) >= 30 else out


def consensus_filter(score_df, cutoffs=(30, 50, 80, 120)):
    out = score_df.copy()
    required_cols = {'transformer', 'lgb'}
    if not required_cols.issubset(out.columns):
        return out
    out['transformer_rank'] = out['transformer'].rank(ascending=False, method='first')
    out['lgb_rank'] = out['lgb'].rank(ascending=False, method='first')
    for cutoff in cutoffs:
        filtered = out[(out['transformer_rank'] <= cutoff) & (out['lgb_rank'] <= cutoff)].copy()
        if len(filtered) >= 5:
            return filtered
    return score_df.copy()


def consensus_stable_filter(score_df, liquidity_quantile=0.20, sigma_quantile=0.85):
    return stable_filter(
        consensus_filter(score_df),
        liquidity_quantile=liquidity_quantile,
        sigma_quantile=sigma_quantile,
    )


def topk_filter(score_df, k=10):
    out = score_df.sort_values('score', ascending=False).head(k).copy()
    return out if len(out) >= 5 else score_df


def stable_topk_filter(score_df, k=30, liquidity_quantile=0.20, sigma_quantile=0.85):
    out = stable_filter(score_df, liquidity_quantile=liquidity_quantile, sigma_quantile=sigma_quantile)
    filtered = out.sort_values('score', ascending=False).head(k).copy()
    return filtered if len(filtered) >= 5 else out


def stable_topk_rerank_filter(
    score_df,
    k=30,
    liquidity_quantile=0.20,
    sigma_quantile=0.85,
    variant='balanced',
):
    """Select stable top-k candidates, then rerank with train-time observable signals."""
    out = stable_topk_filter(
        score_df,
        k=k,
        liquidity_quantile=liquidity_quantile,
        sigma_quantile=sigma_quantile,
    ).copy()
    if len(out) < 5:
        return out.sort_values('score', ascending=False).reset_index(drop=True)

    fused = _available_zscore(out, 'score')
    lgb = _available_zscore(out, 'lgb')
    transformer = _available_zscore(out, 'transformer')
    ret20 = _available_zscore(out, 'ret20')
    ret5 = _available_zscore(out, 'ret5')
    liquidity = _available_zscore(out, 'median_amount20')
    log_liquidity = _available_log_zscore(out, 'median_amount20')
    sigma = _available_zscore(out, 'sigma20')
    amp = _available_zscore(out, 'amp20')

    disagreement = np.abs(_zscore(transformer) - _zscore(lgb))

    if variant == 'trend':
        rerank_score = 0.50 * fused + 0.20 * lgb + 0.15 * ret20 + 0.10 * ret5 + 0.05 * liquidity - 0.10 * disagreement
    elif variant == 'defensive':
        rerank_score = 0.45 * fused + 0.20 * lgb + 0.15 * liquidity - 0.15 * sigma - 0.10 * amp - 0.10 * disagreement
    elif variant == 'lgb_anchor':
        rerank_score = 0.45 * lgb + 0.25 * fused + 0.10 * transformer + 0.10 * ret20 + 0.05 * liquidity - 0.10 * sigma
    elif variant == 'liquidity_risk_off':
        ret20_raw = out.get('ret20', pd.Series(0.0, index=out.index)).astype(float)
        negative_ret20_penalty = (ret20_raw < 0.0).astype(float).to_numpy(dtype=np.float64)
        rerank_score = (
            0.30 * fused
            + 0.10 * lgb
            + 0.30 * log_liquidity
            + 0.10 * ret5
            + 0.10 * ret20
            - 0.10 * amp
            - 0.50 * negative_ret20_penalty
        )
    elif variant == 'liquidity_anchor_risk_off':
        ret20_raw = out.get('ret20', pd.Series(0.0, index=out.index)).astype(float)
        negative_ret20_penalty = (ret20_raw < 0.0).astype(float).to_numpy(dtype=np.float64)
        rerank_score = (
            0.15 * fused
            + 0.05 * lgb
            + 0.50 * log_liquidity
            + 0.15 * ret5
            + 0.10 * ret20
            - 0.05 * amp
            - 0.50 * negative_ret20_penalty
        )
    else:
        rerank_score = 0.55 * fused + 0.15 * lgb + 0.10 * transformer + 0.10 * ret20 + 0.05 * liquidity - 0.10 * sigma - 0.05 * disagreement

    out['base_score'] = out['score']
    out['rerank_score'] = rerank_score
    out['score'] = out['rerank_score']
    return out.sort_values('score', ascending=False).reset_index(drop=True)


def _rank_pct_fill(df, col, default=0.5, fallback_col=None):
    source_col = col if col in df.columns else fallback_col
    if source_col is None or source_col not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float64)
    values = pd.to_numeric(df[source_col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if not values.notna().any():
        return pd.Series(default, index=df.index, dtype=np.float64)
    return values.fillna(values.median()).rank(pct=True, method='average').astype(np.float64)


def _clip01(series):
    return pd.Series(np.clip(series, 0.0, 1.0), index=series.index, dtype=np.float64)


def add_trend_uncluttered_scores(score_df):
    out = score_df.copy()
    out['base_score'] = pd.to_numeric(out.get('score', 0.0), errors='coerce').fillna(0.0)
    out['base_score_pct'] = out['base_score'].rank(pct=True, method='average').astype(np.float64)

    out['ret5_pct'] = _rank_pct_fill(out, 'ret5')
    out['ret10_pct'] = _rank_pct_fill(out, 'ret10', fallback_col='ret5')
    out['ret20_pct'] = _rank_pct_fill(out, 'ret20')
    out['pos20_pct'] = _rank_pct_fill(out, 'pos20')
    out['amp_mean10_pct'] = _rank_pct_fill(out, 'amp_mean10', fallback_col='amp20')
    out['vol10_pct'] = _rank_pct_fill(out, 'vol10', fallback_col='sigma20')
    out['amount20_pct'] = _rank_pct_fill(out, 'mean_amount20', fallback_col='median_amount20')
    out['turnover20_pct'] = _rank_pct_fill(out, 'turnover20')
    out['amt_ratio5_pct'] = _rank_pct_fill(out, 'amt_ratio5')
    out['to_ratio5_pct'] = _rank_pct_fill(out, 'to_ratio5')

    out['crowd_penalty'] = (
        0.65 * _clip01((out['amt_ratio5_pct'] - 0.80) / 0.20)
        + 0.35 * _clip01((out['to_ratio5_pct'] - 0.80) / 0.20)
    )
    out['extreme_vol_penalty'] = _clip01((out['vol10_pct'] - 0.93) / 0.07)
    out['deep_down_penalty'] = _clip01((0.20 - out['pos20_pct']) / 0.20)

    out['trend_raw'] = (
        0.28 * out['ret10_pct']
        + 0.18 * out['ret20_pct']
        + 0.20 * out['pos20_pct']
        + 0.18 * out['amp_mean10_pct']
        + 0.10 * out['amount20_pct']
        + 0.06 * out['turnover20_pct']
        - 0.35 * out['crowd_penalty']
        - 0.15 * out['extreme_vol_penalty']
    )
    out['trend_score'] = (
        0.55 * out['base_score_pct']
        + 0.45 * out['trend_raw'].rank(pct=True, method='average').astype(np.float64)
    )

    out['reversal_raw'] = (
        0.24 * (1.0 - out['ret10_pct'])
        + 0.22 * out['amp_mean10_pct']
        + 0.22 * out['vol10_pct']
        + 0.12 * out['amount20_pct']
        + 0.12 * (1.0 - out['amt_ratio5_pct'])
        + 0.08 * out['ret5_pct']
        - 0.15 * out['deep_down_penalty']
    )
    out['reversal_score'] = (
        0.65 * out['base_score_pct']
        + 0.35 * out['reversal_raw'].rank(pct=True, method='average').astype(np.float64)
    )

    out['is_trend_pool'] = (
        (out['ret10_pct'] >= 0.75)
        & (out['ret20_pct'] >= 0.60)
        & (out['pos20_pct'] >= 0.65)
        & (out['amp_mean10_pct'] >= 0.70)
        & (out['amount20_pct'] >= 0.40)
        & (out['amt_ratio5_pct'] <= 0.95)
        & (out['to_ratio5_pct'] <= 0.95)
    )
    out['is_reversal_pool'] = (
        (out['ret10_pct'] <= 0.35)
        & (out['amp_mean10_pct'] >= 0.60)
        & (out['vol10_pct'] >= 0.60)
        & (out['amt_ratio5_pct'] <= 0.65)
        & (out['amount20_pct'] >= 0.35)
        & (out['base_score_pct'] >= 0.45)
        & (out['pos20_pct'] >= 0.20)
    )
    return out


def trend_uncluttered_plus_reversal_filter(score_df, max_names=5):
    out = add_trend_uncluttered_scores(score_df)
    base_top = out.sort_values('base_score', ascending=False).head(80).copy()
    trend_pool = out[out['is_trend_pool']].sort_values('trend_score', ascending=False).copy()
    reversal_pool = out[out['is_reversal_pool']].sort_values('reversal_score', ascending=False).copy()

    if len(trend_pool) >= max_names:
        selected = trend_pool.head(max_names).copy()
        selected['selector_reason'] = 'trend_uncluttered'
        selected['selector_rank_score'] = selected['trend_score']

        if len(reversal_pool) > 0:
            best_rev = reversal_pool.iloc[0]
            if str(best_rev['stock_id']) not in set(selected['stock_id'].astype(str)):
                worst_idx = selected['selector_rank_score'].idxmin()
                worst_score = float(selected.loc[worst_idx, 'selector_rank_score'])
                if float(best_rev['reversal_score']) >= worst_score - 0.05:
                    selected = selected.drop(index=worst_idx)
                    best_rev_df = best_rev.to_frame().T
                    best_rev_df['selector_reason'] = 'high_vol_reversal'
                    best_rev_df['selector_rank_score'] = best_rev_df['reversal_score']
                    selected = pd.concat([selected, best_rev_df], axis=0)
    else:
        selected = pd.DataFrame(columns=out.columns)

    if len(selected) < max_names:
        selected_ids = set(selected.get('stock_id', pd.Series(dtype=str)).astype(str))
        filler = base_top[~base_top['stock_id'].astype(str).isin(selected_ids)].copy()
        non_crowded = filler[
            (filler['amt_ratio5_pct'] <= 0.90)
            & (filler['to_ratio5_pct'] <= 0.90)
        ].copy()
        if len(non_crowded) >= max_names - len(selected):
            filler = non_crowded
        filler['selector_reason'] = 'base_fill_non_crowded'
        filler['selector_rank_score'] = 0.60 * filler['base_score_pct'] + 0.40 * filler['trend_score']
        selected = pd.concat([selected, filler.head(max_names - len(selected))], axis=0)

    if len(selected) < max_names:
        selected_ids = set(selected.get('stock_id', pd.Series(dtype=str)).astype(str))
        fallback = base_top[~base_top['stock_id'].astype(str).isin(selected_ids)].copy()
        fallback['selector_reason'] = 'base_fill'
        fallback['selector_rank_score'] = fallback['base_score_pct']
        selected = pd.concat([selected, fallback.head(max_names - len(selected))], axis=0)

    selected = selected.sort_values('selector_rank_score', ascending=False).head(max_names).copy()
    selected['score'] = selected['selector_rank_score'].astype(float)
    return selected.reset_index(drop=True)


def legal_minrisk_filter(score_df):
    out = score_df.copy()
    required = {
        'tail_risk_flag',
        'reversal_flag',
        'liq_rank',
        'sigma_rank',
        'downside_beta60_rank',
        'max_drawdown20_rank',
    }
    if not required.issubset(out.columns):
        return stable_filter(out, liquidity_quantile=0.10, sigma_quantile=0.75).sort_values('score', ascending=False).reset_index(drop=True)

    cond = (
        ~out['tail_risk_flag'].astype(bool)
        & ~out['reversal_flag'].astype(bool)
        & (out['liq_rank'].astype(float) >= 0.10)
        & (out['sigma_rank'].astype(float) <= 0.75)
        & (out['downside_beta60_rank'].astype(float) <= 0.75)
        & (out['max_drawdown20_rank'].astype(float) <= 0.75)
    )
    filtered = out[cond].copy()
    if len(filtered) >= 5:
        return filtered.sort_values('score', ascending=False).reset_index(drop=True)

    relaxed = out[
        (out['liq_rank'].astype(float) >= 0.10)
        & (out['sigma_rank'].astype(float) <= 0.85)
        & ~out['reversal_flag'].astype(bool)
    ].copy()
    if len(relaxed) >= 5:
        return relaxed.sort_values('score', ascending=False).reset_index(drop=True)

    return stable_filter(out, liquidity_quantile=0.10, sigma_quantile=0.85).sort_values('score', ascending=False).reset_index(drop=True)


def legal_minrisk_hardened_filter(score_df):
    out = score_df.copy()
    required = {
        'tail_risk_flag',
        'reversal_flag',
        'liq_rank',
        'sigma_rank',
        'downside_beta60_rank',
        'max_drawdown20_rank',
        'amp_rank',
        'ret1_rank',
    }
    if not required.issubset(out.columns):
        return legal_minrisk_filter(out)

    cond = (
        ~out['tail_risk_flag'].astype(bool)
        & ~out['reversal_flag'].astype(bool)
        & (out['liq_rank'].astype(float) >= 0.15)
        & (out['sigma_rank'].astype(float) <= 0.75)
        & (out['downside_beta60_rank'].astype(float) <= 0.75)
        & (out['max_drawdown20_rank'].astype(float) <= 0.75)
        & (out['amp_rank'].astype(float) <= 0.85)
        & (out['ret1_rank'].astype(float) >= 0.20)
    )
    filtered = out[cond].copy()
    if len(filtered) >= 5:
        return filtered.sort_values('score', ascending=False).reset_index(drop=True)
    return legal_minrisk_filter(out)


def _is_extreme_risk_off(score_df):
    required = {'ret20', 'sigma20'}
    if not required.issubset(score_df.columns) or len(score_df) == 0:
        return False
    ret20 = score_df['ret20'].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    sigma = score_df['sigma20'].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if ret20.empty or sigma.empty:
        return False
    breadth20 = float((ret20 > 0).mean())
    median_ret20 = float(ret20.median())
    dispersion20 = float(ret20.std(ddof=0))
    median_sigma20 = float(sigma.median())
    return (
        median_ret20 < 0.0
        and breadth20 < 0.45
        and median_sigma20 > 0.018
        and dispersion20 > 0.10
    )


def ai_hardware_mainline_filter(score_df):
    """Manual 2026-04-24 AI/hardware mainline selection."""
    mainline_ids = ['000977', '688256', '300408', '601138', '002463']
    out = score_df.copy()
    out['stock_id'] = out['stock_id'].map(normalize_stock_id)
    selected = out[out['stock_id'].isin(mainline_ids)].copy()
    selected['_mainline_order'] = selected['stock_id'].map({sid: i for i, sid in enumerate(mainline_ids)})
    selected = selected.sort_values('_mainline_order').drop(columns=['_mainline_order'])
    if len(selected) != len(mainline_ids):
        missing = [sid for sid in mainline_ids if sid not in set(selected['stock_id'].astype(str))]
        raise ValueError(f'regime_ai_hardware_mainline_v1 missing stocks in predictions: {missing}')
    return selected.reset_index(drop=True)


def build_candidate_correlation_clusters(history_df, candidate_ids, asof_date, lookback=60):
    """
    Build candidate-only correlation clusters using data available through asof_date.
    Returns one row per candidate with cluster_id, cluster_size, and cluster_strength.
    """
    candidate_ids = [normalize_stock_id(sid) for sid in candidate_ids]
    if history_df is None or len(candidate_ids) == 0:
        return pd.DataFrame({
            'stock_id': candidate_ids,
            'cluster_id': np.arange(len(candidate_ids), dtype=int),
            'cluster_size': np.ones(len(candidate_ids), dtype=int),
            'cluster_strength': np.zeros(len(candidate_ids), dtype=np.float64),
        })

    required = {'股票代码', '日期', '收盘'}
    if not required.issubset(history_df.columns):
        raise ValueError(f'history_df missing required columns: {sorted(required - set(history_df.columns))}')

    hist = history_df[['股票代码', '日期', '收盘']].copy()
    hist['stock_id'] = hist['股票代码'].map(normalize_stock_id)
    hist['日期'] = pd.to_datetime(hist['日期'])
    hist = hist[(hist['stock_id'].isin(candidate_ids)) & (hist['日期'] <= pd.Timestamp(asof_date))]
    hist = hist.sort_values(['stock_id', '日期']).groupby('stock_id', sort=False).tail(lookback + 1)
    hist['daily_return'] = hist.groupby('stock_id', sort=False)['收盘'].pct_change(fill_method=None)

    pivot = (
        hist.dropna(subset=['daily_return'])
        .pivot_table(index='日期', columns='stock_id', values='daily_return', aggfunc='last')
        .reindex(columns=candidate_ids)
    )
    corr = pivot.corr(min_periods=10).reindex(index=candidate_ids, columns=candidate_ids)
    corr = corr.fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)

    adjacency = corr.to_numpy(dtype=np.float64) >= 0.35
    visited = set()
    components = []
    for i, sid in enumerate(candidate_ids):
        if sid in visited:
            continue
        stack = [i]
        comp_idx = []
        visited.add(sid)
        while stack:
            current = stack.pop()
            comp_idx.append(current)
            for nxt in np.where(adjacency[current])[0]:
                nxt_sid = candidate_ids[int(nxt)]
                if nxt_sid not in visited:
                    visited.add(nxt_sid)
                    stack.append(int(nxt))
        components.append(comp_idx)

    rows = []
    for cluster_id, comp_idx in enumerate(components):
        comp_ids = [candidate_ids[i] for i in comp_idx]
        if len(comp_idx) > 1:
            block = corr.loc[comp_ids, comp_ids].to_numpy(dtype=np.float64)
            upper = block[np.triu_indices_from(block, k=1)]
            strength = float(np.nanmean(upper)) if upper.size else 0.0
        else:
            strength = 0.0
        for sid in comp_ids:
            rows.append({
                'stock_id': sid,
                'cluster_id': cluster_id,
                'cluster_size': len(comp_idx),
                'cluster_strength': strength,
            })

    clusters = pd.DataFrame(rows)
    clusters.attrs['correlation_matrix'] = corr
    return clusters


def _candidate_return_corr(history_df, candidate_ids, asof_date, lookback=60):
    candidate_ids = [normalize_stock_id(sid) for sid in candidate_ids]
    required = {'股票代码', '日期', '收盘'}
    if history_df is None or not required.issubset(history_df.columns):
        corr = pd.DataFrame(0.0, index=candidate_ids, columns=candidate_ids)
        np.fill_diagonal(corr.values, 1.0)
        return corr

    hist = history_df[['股票代码', '日期', '收盘']].copy()
    hist['stock_id'] = hist['股票代码'].map(normalize_stock_id)
    hist['日期'] = pd.to_datetime(hist['日期'])
    hist = hist[(hist['stock_id'].isin(candidate_ids)) & (hist['日期'] <= pd.Timestamp(asof_date))]
    hist = hist.sort_values(['stock_id', '日期']).groupby('stock_id', sort=False).tail(lookback + 1)
    hist['daily_return'] = hist.groupby('stock_id', sort=False)['收盘'].pct_change(fill_method=None)
    pivot = (
        hist.dropna(subset=['daily_return'])
        .pivot_table(index='日期', columns='stock_id', values='daily_return', aggfunc='last')
        .reindex(columns=candidate_ids)
    )
    corr = pivot.corr(min_periods=10).reindex(index=candidate_ids, columns=candidate_ids)
    corr = corr.fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    return corr


def _theme_pct(df, col, default=0.5, fallback_col=None):
    source = col if col in df.columns else fallback_col
    if source is None or source not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float64)
    values = pd.to_numeric(df[source], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if not values.notna().any():
        return pd.Series(default, index=df.index, dtype=np.float64)
    return values.fillna(values.median()).rank(pct=True, method='average').astype(np.float64)


def _theme_log_liquidity_pct(df):
    source = 'amount20' if 'amount20' in df.columns else 'median_amount20'
    if source not in df.columns:
        return pd.Series(0.5, index=df.index, dtype=np.float64)
    values = pd.to_numeric(df[source], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if not values.notna().any():
        return pd.Series(0.5, index=df.index, dtype=np.float64)
    values = values.fillna(values.median()).clip(lower=0.0)
    return np.log1p(values).rank(pct=True, method='average').astype(np.float64)


def select_theme_consensus_top5(candidate_top20, clusters, max_names=5):
    """
    Select up to max_names from the dominant correlation cluster, with isolated-theme vetoes.
    """
    out = candidate_top20.copy()
    out['stock_id'] = out['stock_id'].map(normalize_stock_id)
    clusters = clusters.copy()
    clusters['stock_id'] = clusters['stock_id'].map(normalize_stock_id)
    out = out.merge(clusters, on='stock_id', how='left')
    out['cluster_id'] = out['cluster_id'].fillna(-1).astype(int)
    out['cluster_size'] = out['cluster_size'].fillna(1).astype(int)
    out['cluster_strength'] = out['cluster_strength'].fillna(0.0).astype(float)

    out['score_pct'] = _theme_pct(out, 'score')
    out['ret20_pct_within_top20'] = _theme_pct(out, 'ret20')
    out['ret10_pct_within_top20'] = _theme_pct(out, 'ret10')
    out['liquidity_pct_within_top20'] = _theme_log_liquidity_pct(out)
    out['short_term_overheat_penalty'] = np.maximum(
        0.0,
        (pd.to_numeric(out.get('ret5', 0.0), errors='coerce').fillna(0.0) - 0.15) / 0.10,
    )
    out['final_theme_score'] = (
        0.55 * out['score_pct']
        + 0.20 * out['ret20_pct_within_top20']
        + 0.15 * out['ret10_pct_within_top20']
        + 0.10 * out['liquidity_pct_within_top20']
        - 0.15 * out['short_term_overheat_penalty']
    )

    cluster_stats = (
        out.groupby('cluster_id', sort=False)
        .agg(
            cluster_size=('stock_id', 'size'),
            cluster_score_sum=('score_pct', 'sum'),
            cluster_score_mean=('score_pct', 'mean'),
            cluster_ret20_mean=('ret20', 'mean'),
            cluster_ret10_mean=('ret10', 'mean'),
            cluster_liquidity_mean=('liquidity_pct_within_top20', 'mean'),
        )
        .reset_index()
    )
    dominant_options = cluster_stats[cluster_stats['cluster_size'] >= 3].copy()
    dominant_cluster = None
    if len(dominant_options) > 0:
        dominant_cluster = int(
            dominant_options.sort_values(
                ['cluster_score_sum', 'cluster_size'],
                ascending=[False, False],
            ).iloc[0]['cluster_id']
        )

    out = out.drop(columns=['cluster_size']).merge(
        cluster_stats[['cluster_id', 'cluster_size', 'cluster_score_sum']],
        on='cluster_id',
        how='left',
    )
    out['cluster_size'] = out['cluster_size'].fillna(1).astype(int)
    out['cluster_score_sum'] = out['cluster_score_sum'].fillna(out['score_pct']).astype(float)

    corr = clusters.attrs.get('correlation_matrix')
    if corr is not None and dominant_cluster is not None:
        dominant_ids = out.loc[out['cluster_id'] == dominant_cluster, 'stock_id'].tolist()
        corr_to_dom = {}
        for sid in out['stock_id']:
            available = [dom_sid for dom_sid in dominant_ids if dom_sid in corr.columns and dom_sid != sid]
            if sid in corr.index and available:
                corr_to_dom[sid] = float(corr.loc[sid, available].mean())
            elif sid in dominant_ids:
                corr_to_dom[sid] = 1.0
            else:
                corr_to_dom[sid] = 0.0
        out['corr_to_dominant_mean'] = out['stock_id'].map(corr_to_dom).fillna(0.0).astype(float)
    else:
        out['corr_to_dominant_mean'] = 0.0

    out['is_isolated'] = out['cluster_size'] <= 1
    out['weak_momentum'] = (
        (pd.to_numeric(out.get('ret10', 0.0), errors='coerce').fillna(0.0) < 0.0)
        & (pd.to_numeric(out.get('ret20', 0.0), errors='coerce').fillna(0.0) < 0.05)
    )
    out['model_disagree_bad'] = (
        (pd.to_numeric(out.get('lgb', 0.0), errors='coerce').fillna(0.0) < 1.0)
        & (pd.to_numeric(out.get('transformer', 0.0), errors='coerce').fillna(0.0) > 1.5)
    )
    out['veto'] = (
        (out['is_isolated'] & out['weak_momentum'])
        | (out['is_isolated'] & out['model_disagree_bad'])
    )

    out.attrs['cluster_stats'] = cluster_stats
    out.attrs['dominant_cluster'] = dominant_cluster
    if dominant_cluster is None:
        out['selected'] = False
        return out.iloc[0:0].copy(), out

    in_dom = out['cluster_id'] == dominant_cluster
    primary = out[
        in_dom
        & ~out['veto']
        & (pd.to_numeric(out['ret20'], errors='coerce').fillna(0.0) > 0.10)
        & (
            (pd.to_numeric(out['ret10'], errors='coerce').fillna(0.0) > 0.0)
            | (pd.to_numeric(out['ret20'], errors='coerce').fillna(0.0) > 0.20)
        )
    ].copy()
    primary = primary.sort_values('final_theme_score', ascending=False)

    selected = primary.head(max_names).copy()
    if len(selected) < max_names:
        selected_ids = set(selected['stock_id'].astype(str))
        filler = out[
            ~in_dom
            & ~out['stock_id'].astype(str).isin(selected_ids)
            & ~out['veto']
            & (pd.to_numeric(out['ret20'], errors='coerce').fillna(0.0) > 0.10)
            & (pd.to_numeric(out['ret10'], errors='coerce').fillna(0.0) > 0.0)
            & (out['corr_to_dominant_mean'] >= 0.20)
        ].copy()
        filler = filler.sort_values('final_theme_score', ascending=False)
        selected = pd.concat([selected, filler.head(max_names - len(selected))], axis=0)

    selected = selected.sort_values('final_theme_score', ascending=False).head(max_names).copy()
    selected['score'] = selected['final_theme_score'].astype(float)
    out['selected'] = out['stock_id'].astype(str).isin(set(selected['stock_id'].astype(str)))
    return selected.reset_index(drop=True), out


def select_regime_theme_consensus_top20(pred_df, history_df, asof_date, max_names=5):
    candidate_top20 = pred_df.sort_values('score', ascending=False).head(20).copy()
    candidate_top20['stock_id'] = candidate_top20['stock_id'].map(normalize_stock_id)
    candidate_ids = candidate_top20['stock_id'].tolist()
    print(f'[theme_consensus] candidate_top20 = {",".join(candidate_ids)}')

    clusters = build_candidate_correlation_clusters(
        history_df=history_df,
        candidate_ids=candidate_ids,
        asof_date=asof_date,
        lookback=60,
    )
    selected, debug_df = select_theme_consensus_top5(candidate_top20, clusters, max_names=max_names)
    cluster_stats = debug_df.attrs.get('cluster_stats', pd.DataFrame())
    dominant_cluster = debug_df.attrs.get('dominant_cluster')
    print(f'[theme_consensus] clusters = {cluster_stats.to_dict("records") if len(cluster_stats) else []}')
    print(f'[theme_consensus] dominant_cluster = {dominant_cluster}')

    if dominant_cluster is None:
        fallback = apply_filter(
            pred_df,
            'regime_liquidity_anchor_risk_off',
            liquidity_quantile=0.10,
            sigma_quantile=0.85,
        )
        debug_df['selected'] = debug_df['stock_id'].astype(str).isin(
            set(fallback.head(max_names)['stock_id'].astype(str))
        )
        selected = fallback.head(max_names).copy()

    vetoed = debug_df.loc[debug_df.get('veto', False).astype(bool), 'stock_id'].astype(str).tolist()
    selected_ids = selected.head(max_names)['stock_id'].astype(str).tolist()
    print(f'[theme_consensus] vetoed = {",".join(vetoed)}')
    print(f'[theme_consensus] selected = {",".join(selected_ids)}')

    csv_cols = [
        'stock_id',
        'score',
        'lgb',
        'transformer',
        'ret5',
        'ret10',
        'ret20',
        'cluster_id',
        'cluster_size',
        'cluster_score_sum',
        'corr_to_dominant_mean',
        'is_isolated',
        'weak_momentum',
        'model_disagree_bad',
        'veto',
        'final_theme_score',
        'selected',
    ]
    for col in csv_cols:
        if col not in debug_df.columns:
            debug_df[col] = np.nan
    out_path = os.path.join('temp', 'theme_consensus', 'latest_theme_consensus_candidates.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    debug_df[csv_cols].to_csv(out_path, index=False)

    if len(selected) < max_names:
        raise ValueError(f'theme_consensus selected fewer than {max_names} names: {selected_ids}')
    return selected.reset_index(drop=True)


def build_anchor_theme_groups(
    history_df,
    candidate_top20,
    asof_date,
    lookback=60,
    corr_min=0.22,
    avg_corr_min=0.18,
    max_group_size=10,
):
    """
    Build local theme groups around each anchor without connected-component chaining.
    """
    candidates = candidate_top20.copy()
    candidates['stock_id'] = candidates['stock_id'].map(normalize_stock_id)
    candidate_ids = candidates['stock_id'].tolist()
    corr = _candidate_return_corr(history_df, candidate_ids, asof_date, lookback=lookback)

    candidates['score_pct'] = _theme_pct(candidates, 'score')
    candidates['ret20_pct_within_top20'] = _theme_pct(candidates, 'ret20')
    candidates['ret10_pct_within_top20'] = _theme_pct(candidates, 'ret10')
    candidates['liquidity_pct_within_top20'] = _theme_log_liquidity_pct(candidates)
    candidates['weak_momentum'] = (
        (pd.to_numeric(candidates.get('ret10', 0.0), errors='coerce').fillna(0.0) < 0.0)
        & (pd.to_numeric(candidates.get('ret20', 0.0), errors='coerce').fillna(0.0) < 0.05)
    )
    candidates['model_disagree_bad'] = (
        (pd.to_numeric(candidates.get('lgb', 0.0), errors='coerce').fillna(0.0) < 1.0)
        & (pd.to_numeric(candidates.get('transformer', 0.0), errors='coerce').fillna(0.0) > 1.5)
    )
    candidates['hard_veto'] = candidates['weak_momentum'] | candidates['model_disagree_bad']

    by_id = candidates.set_index('stock_id', drop=False)
    groups = []
    for anchor in candidate_ids:
        members = [anchor]
        if not bool(by_id.loc[anchor, 'hard_veto']):
            direct = []
            for sid in candidate_ids:
                if sid == anchor or bool(by_id.loc[sid, 'hard_veto']):
                    continue
                anchor_corr = float(corr.loc[anchor, sid]) if anchor in corr.index and sid in corr.columns else 0.0
                if anchor_corr >= corr_min:
                    direct.append((sid, anchor_corr * float(by_id.loc[sid, 'score_pct']), anchor_corr))
            direct.sort(key=lambda item: item[1], reverse=True)
            for sid, _, anchor_corr in direct:
                if len(members) >= max_group_size:
                    break
                group_corrs = [
                    float(corr.loc[sid, existing])
                    for existing in members
                    if sid in corr.index and existing in corr.columns
                ]
                avg_corr = float(np.mean(group_corrs)) if group_corrs else 0.0
                if anchor_corr >= corr_min and avg_corr >= avg_corr_min:
                    members.append(sid)

        valid_members = [sid for sid in members if not bool(by_id.loc[sid, 'hard_veto'])]
        metric_members = valid_members if valid_members else members
        group_df = by_id.loc[metric_members]
        group_size = len(valid_members)
        if len(metric_members) > 1:
            block = corr.loc[metric_members, metric_members].to_numpy(dtype=np.float64)
            upper = block[np.triu_indices_from(block, k=1)]
            cohesion = float(np.nanmean(upper)) if upper.size else 0.0
        else:
            cohesion = 1.0 if group_size == 1 else 0.0
        groups.append({
            'anchor_id': anchor,
            'members': valid_members,
            'raw_members': members,
            'group_size': group_size,
            'group_size_score': min(group_size, 8) / 8.0,
            'group_score_mass': float(group_df['score_pct'].sum() / max(min(group_size, 8), 1)),
            'positive_trend_frac': float(((group_df['ret10'] > 0.0) & (group_df['ret20'] > 0.10)).mean()) if len(group_df) else 0.0,
            'group_ret20_mean': float(group_df['ret20'].mean()) if len(group_df) else 0.0,
            'group_liquidity_mean': float(group_df['liquidity_pct_within_top20'].mean()) if len(group_df) else 0.0,
            'group_cohesion': cohesion,
            'anchor_hard_veto': bool(by_id.loc[anchor, 'hard_veto']),
        })

    groups_df = pd.DataFrame(groups)
    if len(groups_df) > 0:
        groups_df['group_ret20_mean_pct'] = groups_df['group_ret20_mean'].rank(pct=True, method='average')
        groups_df['group_liquidity_mean_pct'] = groups_df['group_liquidity_mean'].rank(pct=True, method='average')
        groups_df['group_theme_score'] = (
            0.30 * groups_df['group_size_score']
            + 0.25 * groups_df['group_score_mass']
            + 0.20 * groups_df['positive_trend_frac']
            + 0.15 * groups_df['group_ret20_mean_pct']
            + 0.10 * groups_df['group_cohesion']
        )
    groups_df.attrs['correlation_matrix'] = corr
    groups_df.attrs['candidate_metrics'] = candidates
    return groups_df


def select_regime_theme_consensus_top20_v2(pred_df, history_df, asof_date, max_names=5):
    candidate_top20 = pred_df.sort_values('score', ascending=False).head(20).copy()
    candidate_top20['stock_id'] = candidate_top20['stock_id'].map(normalize_stock_id)
    groups = build_anchor_theme_groups(history_df, candidate_top20, asof_date)
    corr = groups.attrs.get('correlation_matrix')
    metrics = groups.attrs.get('candidate_metrics', candidate_top20.copy()).copy()
    metrics['stock_id'] = metrics['stock_id'].map(normalize_stock_id)

    print('[theme_v2] groups by anchor:')
    for row in groups.sort_values('group_theme_score', ascending=False).to_dict('records'):
        print(
            f"[theme_v2] anchor={row['anchor_id']}, "
            f"group_size={int(row['group_size'])}, "
            f"group_theme_score={float(row['group_theme_score']):.6f}, "
            f"members={','.join(row['members'])}"
        )

    eligible = groups[(groups['group_size'] >= 5) & (~groups['anchor_hard_veto'])].copy()
    dominant = None
    dominant_members = []
    if len(eligible) == 0:
        selected = apply_filter(
            pred_df,
            'regime_liquidity_anchor_risk_off',
            liquidity_quantile=0.10,
            sigma_quantile=0.85,
        ).head(max_names).copy()
    else:
        dominant = eligible.sort_values(
            ['group_theme_score', 'group_size', 'group_score_mass'],
            ascending=[False, False, False],
        ).iloc[0].to_dict()
        dominant_anchor = dominant['anchor_id']
        dominant_members = list(dominant['members'])
        dominant_set = set(dominant_members)

        metrics['anchor_id'] = dominant_anchor
        metrics['anchor_group_id'] = dominant_anchor
        metrics['anchor_group_size'] = int(dominant['group_size'])
        metrics['anchor_group_theme_score'] = float(dominant['group_theme_score'])
        metrics['corr_to_anchor'] = metrics['stock_id'].map(
            lambda sid: float(corr.loc[dominant_anchor, sid])
            if corr is not None and dominant_anchor in corr.index and sid in corr.columns
            else 0.0
        )
        metrics['corr_to_group_mean'] = metrics['stock_id'].map(
            lambda sid: float(corr.loc[sid, [m for m in dominant_members if m != sid]].mean())
            if corr is not None and sid in corr.index and len([m for m in dominant_members if m != sid]) > 0
            else (1.0 if sid in dominant_set else 0.0)
        )
        metrics['corr_to_anchor_pct'] = metrics['corr_to_anchor'].rank(pct=True, method='average')
        metrics['group_selected_as_dominant'] = metrics['stock_id'].isin(dominant_set)
        metrics['sidecar_candidate'] = (
            ~metrics['group_selected_as_dominant']
            & ~metrics['hard_veto']
            & (metrics['corr_to_group_mean'] >= 0.18)
            & (pd.to_numeric(metrics['ret20'], errors='coerce').fillna(0.0) > 0.10)
            & (pd.to_numeric(metrics['ret10'], errors='coerce').fillna(0.0) > 0.0)
        )
        overheat_penalty = np.maximum(
            0.0,
            (pd.to_numeric(metrics.get('ret5', 0.0), errors='coerce').fillna(0.0) - 0.15) / 0.10,
        )
        metrics['final_theme_score'] = (
            0.50 * metrics['score_pct']
            + 0.20 * metrics['ret20_pct_within_top20']
            + 0.15 * metrics['ret10_pct_within_top20']
            + 0.10 * metrics['liquidity_pct_within_top20']
            + 0.05 * metrics['corr_to_anchor_pct']
            - 0.15 * overheat_penalty
        )

        primary = metrics[
            metrics['group_selected_as_dominant']
            & ~metrics['hard_veto']
        ].copy().sort_values('final_theme_score', ascending=False)
        selected = primary.head(max_names).copy()
        if len(primary) >= 4 and len(selected) < max_names:
            sidecar = metrics[metrics['sidecar_candidate']].copy()
            sidecar = sidecar.sort_values('final_theme_score', ascending=False)
            selected = pd.concat([selected, sidecar.head(max_names - len(selected))], axis=0)
        if len(selected) < max_names:
            selected_ids = set(selected['stock_id'].astype(str))
            selected = pd.concat([
                selected,
                primary[~primary['stock_id'].astype(str).isin(selected_ids)].head(max_names - len(selected)),
            ], axis=0)
        selected = selected.sort_values('final_theme_score', ascending=False).head(max_names).copy()
        selected['score'] = selected['final_theme_score'].astype(float)

    if 'anchor_id' not in metrics.columns:
        metrics['anchor_id'] = np.nan
        metrics['anchor_group_id'] = np.nan
        metrics['anchor_group_size'] = np.nan
        metrics['anchor_group_theme_score'] = np.nan
        metrics['corr_to_anchor'] = np.nan
        metrics['corr_to_group_mean'] = np.nan
        metrics['group_selected_as_dominant'] = False
        metrics['sidecar_candidate'] = False
        metrics['final_theme_score'] = np.nan
    metrics['selected'] = metrics['stock_id'].astype(str).isin(set(selected['stock_id'].astype(str)))
    metrics['veto'] = metrics['hard_veto']

    dominant_anchor = dominant['anchor_id'] if dominant is not None else None
    hard_veto_ids = metrics.loc[metrics['hard_veto'].astype(bool), 'stock_id'].astype(str).tolist()
    selected_ids = selected.head(max_names)['stock_id'].astype(str).tolist()
    print(f'[theme_v2] dominant_anchor: {dominant_anchor}')
    print(f'[theme_v2] dominant_group_members: {",".join(dominant_members)}')
    print(f'[theme_v2] hard_veto: {",".join(hard_veto_ids)}')
    print(f'[theme_v2] selected: {",".join(selected_ids)}')

    csv_cols = [
        'stock_id',
        'score',
        'lgb',
        'transformer',
        'ret5',
        'ret10',
        'ret20',
        'anchor_id',
        'anchor_group_id',
        'anchor_group_size',
        'anchor_group_theme_score',
        'corr_to_anchor',
        'corr_to_group_mean',
        'group_selected_as_dominant',
        'weak_momentum',
        'model_disagree_bad',
        'hard_veto',
        'sidecar_candidate',
        'final_theme_score',
        'selected',
    ]
    for col in csv_cols:
        if col not in metrics.columns:
            metrics[col] = np.nan
    out_path = os.path.join('temp', 'theme_consensus', 'latest_theme_consensus_candidates.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    metrics[csv_cols].to_csv(out_path, index=False)

    if len(selected) < max_names:
        raise ValueError(f'theme_consensus_v2 selected fewer than {max_names} names: {selected_ids}')
    return selected.reset_index(drop=True)


def apply_filter(score_df, filter_name, liquidity_quantile=0.20, sigma_quantile=0.85, history_df=None, asof_date=None):
    if filter_name == 'nofilter':
        return score_df.sort_values('score', ascending=False).reset_index(drop=True)
    if filter_name == 'liquidity_q05':
        return liquidity_floor_filter(score_df, liquidity_quantile=0.05).sort_values('score', ascending=False).reset_index(drop=True)
    if filter_name == 'liquidity_q10':
        return liquidity_floor_filter(score_df, liquidity_quantile=0.10).sort_values('score', ascending=False).reset_index(drop=True)
    if filter_name == 'liquidity80':
        return current_filter(score_df, liquidity_quantile=liquidity_quantile).sort_values('score', ascending=False).reset_index(drop=True)
    if filter_name == 'stable':
        return stable_filter(score_df, liquidity_quantile=liquidity_quantile, sigma_quantile=sigma_quantile).sort_values('score', ascending=False).reset_index(drop=True)
    if filter_name == 'liq30_sigma70':
        return liq_sigma_filter(score_df, liquidity_quantile=0.30, sigma_quantile=0.70).sort_values('score', ascending=False).reset_index(drop=True)
    if filter_name == 'defensive':
        return defensive_filter(score_df, liquidity_quantile=0.30, sigma_quantile=0.70, amp_quantile=0.70).sort_values('score', ascending=False).reset_index(drop=True)
    if filter_name == 'no_extreme_momentum':
        return no_extreme_momentum_filter(score_df, liquidity_quantile=liquidity_quantile, sigma_quantile=sigma_quantile).sort_values('score', ascending=False).reset_index(drop=True)
    if filter_name == 'consensus':
        return consensus_filter(score_df).sort_values('score', ascending=False).reset_index(drop=True)
    if filter_name == 'consensus_stable':
        return consensus_stable_filter(score_df, liquidity_quantile=liquidity_quantile, sigma_quantile=sigma_quantile).sort_values('score', ascending=False).reset_index(drop=True)
    if filter_name == 'topk10':
        return topk_filter(score_df, k=10).sort_values('score', ascending=False).reset_index(drop=True)
    if filter_name == 'stable_top30':
        return stable_topk_filter(score_df, k=30, liquidity_quantile=liquidity_quantile, sigma_quantile=sigma_quantile).sort_values('score', ascending=False).reset_index(drop=True)
    if filter_name == 'stable_top30_rerank':
        return stable_topk_rerank_filter(score_df, k=30, liquidity_quantile=liquidity_quantile, sigma_quantile=sigma_quantile)
    if filter_name == 'stable_top30_rerank_trend':
        return stable_topk_rerank_filter(score_df, k=30, liquidity_quantile=liquidity_quantile, sigma_quantile=sigma_quantile, variant='trend')
    if filter_name == 'stable_top30_rerank_defensive':
        return stable_topk_rerank_filter(score_df, k=30, liquidity_quantile=liquidity_quantile, sigma_quantile=sigma_quantile, variant='defensive')
    if filter_name == 'stable_top30_rerank_lgb_anchor':
        return stable_topk_rerank_filter(score_df, k=30, liquidity_quantile=liquidity_quantile, sigma_quantile=sigma_quantile, variant='lgb_anchor')
    if filter_name == 'legal_minrisk':
        return legal_minrisk_filter(score_df)
    if filter_name == 'legal_minrisk_hardened':
        return legal_minrisk_hardened_filter(score_df)
    if filter_name == 'regime_trend_uncluttered_plus_reversal':
        return trend_uncluttered_plus_reversal_filter(score_df)
    if filter_name == 'regime_liquidity_risk_off':
        if _is_extreme_risk_off(score_df):
            return stable_topk_rerank_filter(
                score_df,
                k=60,
                liquidity_quantile=liquidity_quantile,
                sigma_quantile=sigma_quantile,
                variant='liquidity_risk_off',
            )
        return stable_filter(
            score_df,
            liquidity_quantile=liquidity_quantile,
            sigma_quantile=sigma_quantile,
        ).sort_values('score', ascending=False).reset_index(drop=True)
    if filter_name == 'regime_liquidity_anchor_risk_off':
        if _is_extreme_risk_off(score_df):
            return stable_topk_rerank_filter(
                score_df,
                k=60,
                liquidity_quantile=liquidity_quantile,
                sigma_quantile=sigma_quantile,
                variant='liquidity_anchor_risk_off',
            )
        return stable_filter(
            score_df,
            liquidity_quantile=liquidity_quantile,
            sigma_quantile=sigma_quantile,
        ).sort_values('score', ascending=False).reset_index(drop=True)
    if filter_name == 'regime_ai_hardware_mainline_v1':
        return ai_hardware_mainline_filter(score_df)
    if filter_name == 'regime_theme_consensus_top20_v1':
        if history_df is None or asof_date is None:
            raise ValueError('regime_theme_consensus_top20_v1 requires history_df and asof_date')
        return select_regime_theme_consensus_top20(score_df, history_df, asof_date)
    if filter_name == 'regime_theme_consensus_top20_v2':
        if history_df is None or asof_date is None:
            raise ValueError('regime_theme_consensus_top20_v2 requires history_df and asof_date')
        return select_regime_theme_consensus_top20_v2(score_df, history_df, asof_date)
    raise ValueError(f'Unsupported filter_name: {filter_name}')


def select_candidates(score_df, post_cfg=None, history_df=None, asof_date=None):
    post_cfg = post_cfg or {}
    filter_name = post_cfg.get('filter', 'stable')
    liquidity_q = post_cfg.get('liquidity_quantile', 0.20)
    sigma_q = post_cfg.get('sigma_quantile', 0.85)
    return apply_filter(
        score_df,
        filter_name,
        liquidity_quantile=liquidity_q,
        sigma_quantile=sigma_q,
        history_df=history_df,
        asof_date=asof_date,
    )


def build_weight_portfolio(score_df, weighting_name, k=5, exposure_cap=1.0):
    weighting_map = {
        'equal': equal_weight_portfolio,
        'fixed_descending': fixed_descending_weight_portfolio,
        'risk_soft': risk_soft_weight_portfolio,
        'score_soft': score_soft_weight_portfolio,
        'score_risk_soft': score_risk_soft_weight_portfolio,
        'inv_vol': inv_vol_weight_portfolio,
        'shrunk_softmax': shrunk_score_soft_weight_portfolio,
        'shrunk_t2_rho10_cap30_min05': lambda df, k=5, exposure_cap=1.0: shrunk_score_soft_weight_portfolio(
            df, k=k, exposure_cap=exposure_cap, temperature=2.0, rho=0.10, max_weight=0.30, min_weight=0.05
        ),
        'shrunk_t3_rho20_cap35_min05': lambda df, k=5, exposure_cap=1.0: shrunk_score_soft_weight_portfolio(
            df, k=k, exposure_cap=exposure_cap, temperature=3.0, rho=0.20, max_weight=0.35, min_weight=0.05
        ),
        'shrunk_t5_rho30_cap35_min08': lambda df, k=5, exposure_cap=1.0: shrunk_score_soft_weight_portfolio(
            df, k=k, exposure_cap=exposure_cap, temperature=5.0, rho=0.30, max_weight=0.35, min_weight=0.08
        ),
    }
    if weighting_name not in weighting_map:
        raise ValueError(f'Unsupported weighting_name: {weighting_name}')
    return weighting_map[weighting_name](score_df, k=k, exposure_cap=exposure_cap)


def weights_df_to_selection(weights_df):
    return list(zip(weights_df['stock_id'].tolist(), weights_df['weight'].tolist()))


def portfolio_metrics(weights_df):
    weights = weights_df['weight'].to_numpy(dtype=np.float64) if len(weights_df) else np.array([], dtype=np.float64)
    if weights.size == 0:
        return {
            'max_weight': 0.0,
            'top2_weight': 0.0,
            'herfindahl': 0.0,
            'entropy_effective_n': 0.0,
        }
    sorted_weights = np.sort(weights)[::-1]
    herfindahl = float(np.sum(np.square(weights)))
    positive = weights[weights > 1e-12]
    entropy = -float(np.sum(positive * np.log(positive)))
    return {
        'max_weight': float(sorted_weights[0]),
        'top2_weight': float(sorted_weights[:2].sum()) if sorted_weights.size >= 2 else float(sorted_weights.sum()),
        'herfindahl': herfindahl,
        'entropy_effective_n': float(np.exp(entropy)),
    }


def add_forward_open_returns(raw_df, horizon=5, date_col='日期', stock_col='股票代码', open_col='开盘'):
    """Build scorer-equivalent future return: open[t+horizon] / open[t+1] - 1."""
    data = raw_df[[stock_col, date_col, open_col]].copy()
    data[stock_col] = data[stock_col].map(normalize_stock_id)
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values([stock_col, date_col]).reset_index(drop=True)
    grouped_open = data.groupby(stock_col, sort=False)[open_col]
    data['open_t1'] = grouped_open.shift(-1)
    data[f'open_t{horizon}'] = grouped_open.shift(-horizon)
    data['forward_open_return'] = (
        data[f'open_t{horizon}'] - data['open_t1']
    ) / (data['open_t1'] + 1e-12)
    return data.rename(columns={stock_col: 'stock_id', date_col: 'date'})[
        ['stock_id', 'date', 'forward_open_return']
    ].dropna(subset=['forward_open_return'])


def score_portfolio_like_scorer(weights_df, forward_return_df, date):
    """Score a portfolio by explicit stock_id/date join instead of positional arrays."""
    if len(weights_df) == 0:
        empty = weights_df.copy()
        empty['forward_open_return'] = pd.Series(dtype=float)
        empty['weighted_return'] = pd.Series(dtype=float)
        return 0.0, empty

    weights = weights_df.copy()
    weights['stock_id'] = weights['stock_id'].map(normalize_stock_id)
    date = pd.Timestamp(date)
    returns = forward_return_df[forward_return_df['date'] == date][['stock_id', 'forward_open_return']].copy()
    returns['stock_id'] = returns['stock_id'].map(normalize_stock_id)
    merged = weights.merge(returns, on='stock_id', how='left')
    merged['forward_open_return'] = merged['forward_open_return'].fillna(0.0)
    merged['weighted_return'] = merged['weight'].astype(float) * merged['forward_open_return'].astype(float)
    return float(merged['weighted_return'].sum()), merged
