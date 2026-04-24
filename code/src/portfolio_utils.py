import numpy as np
import pandas as pd


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
    else:
        rerank_score = 0.55 * fused + 0.15 * lgb + 0.10 * transformer + 0.10 * ret20 + 0.05 * liquidity - 0.10 * sigma - 0.05 * disagreement

    out['base_score'] = out['score']
    out['rerank_score'] = rerank_score
    out['score'] = out['rerank_score']
    return out.sort_values('score', ascending=False).reset_index(drop=True)


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


def apply_filter(score_df, filter_name, liquidity_quantile=0.20, sigma_quantile=0.85):
    if filter_name == 'nofilter':
        return score_df.sort_values('score', ascending=False).reset_index(drop=True)
    if filter_name == 'liquidity80':
        return current_filter(score_df, liquidity_quantile=liquidity_quantile).sort_values('score', ascending=False).reset_index(drop=True)
    if filter_name == 'stable':
        return stable_filter(score_df, liquidity_quantile=liquidity_quantile, sigma_quantile=sigma_quantile).sort_values('score', ascending=False).reset_index(drop=True)
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
    raise ValueError(f'Unsupported filter_name: {filter_name}')


def select_candidates(score_df, post_cfg=None):
    post_cfg = post_cfg or {}
    filter_name = post_cfg.get('filter', 'stable')
    liquidity_q = post_cfg.get('liquidity_quantile', 0.20)
    sigma_q = post_cfg.get('sigma_quantile', 0.85)
    return apply_filter(score_df, filter_name, liquidity_quantile=liquidity_q, sigma_quantile=sigma_q)


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
