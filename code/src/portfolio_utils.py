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
