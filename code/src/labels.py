import pandas as pd
import numpy as np


def normalize_stock_id(value):
    if pd.isna(value):
        return value
    text = str(value).strip()
    if text.endswith('.0'):
        text = text[:-2]
    return text.zfill(6)


def add_label_o2o_week(
    df,
    horizon=5,
    stock_col='股票代码',
    date_col='日期',
    open_col='开盘',
    label_col='label_o2o_week',
):
    """Add scorer-equivalent open-to-open weekly label.

    For an anchor date t, the competition scorer uses the first and last rows
    in the following 5-row test window. With a complete trading calendar this is:

        open[t + horizon] / open[t + 1] - 1

    The function uses the global trading-date index, not per-stock row offsets,
    so missing stock rows do not silently shift the target.
    """
    out = df.copy()
    out[stock_col] = out[stock_col].map(normalize_stock_id)
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values([stock_col, date_col]).reset_index(drop=True)

    dates = pd.Index(sorted(out[date_col].dropna().unique()))
    date_to_idx = {date: idx for idx, date in enumerate(dates)}
    out['_date_idx'] = out[date_col].map(date_to_idx).astype('int64')

    open_base = out[[stock_col, '_date_idx', open_col]].copy()

    entry_col = 'open_entry_t1'
    exit_col = f'open_exit_t{horizon}'

    open_t1 = open_base.rename(columns={open_col: entry_col})
    open_t1['_date_idx'] -= 1

    open_tn = open_base.rename(columns={open_col: exit_col})
    open_tn['_date_idx'] -= horizon

    out = out.merge(open_t1[[stock_col, '_date_idx', entry_col]], on=[stock_col, '_date_idx'], how='left')
    out = out.merge(open_tn[[stock_col, '_date_idx', exit_col]], on=[stock_col, '_date_idx'], how='left')

    out[label_col] = (
        pd.to_numeric(out[exit_col], errors='coerce')
        / (pd.to_numeric(out[entry_col], errors='coerce') + 1e-12)
        - 1.0
    )
    return out.drop(columns=['_date_idx', entry_col, exit_col])


def build_quality_label(
    df,
    raw_label_col='label',
    output_col='quality5',
    stock_col='股票代码',
    date_col='日期',
    high_col='最高',
    low_col='最低',
    close_col='收盘',
    tradable_col=None,
    fee=0.0,
    slippage=0.0,
    lambda_vol=0.0,
    lambda_dd=0.0,
    horizon=5,
):
    """Build a Top5-oriented trading-quality label from scorer-aligned return.

    quality = tradable * (raw_return - fee - slippage)
              - lambda_vol * future_realized_vol
              - lambda_dd * future_path_drawdown
    """
    out = df.copy()
    out[stock_col] = out[stock_col].map(normalize_stock_id)
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values([stock_col, date_col]).reset_index(drop=True)

    if raw_label_col not in out.columns:
        out = add_label_o2o_week(
            out,
            horizon=horizon,
            stock_col=stock_col,
            date_col=date_col,
            open_col='开盘',
            label_col=raw_label_col,
        )

    for col in [high_col, low_col, close_col, raw_label_col]:
        out[col] = pd.to_numeric(out[col], errors='coerce')

    grouped = out.groupby(stock_col, sort=False)
    future_returns = []
    future_drawdowns = []
    for step in range(1, horizon + 1):
        future_close = grouped[close_col].shift(-step)
        future_ret = future_close / (out[close_col] + 1e-12) - 1.0
        future_returns.append(future_ret)

        future_low = grouped[low_col].shift(-step)
        entry_close = grouped[close_col].shift(-1)
        future_drawdowns.append((future_low / (entry_close + 1e-12) - 1.0).clip(upper=0.0).abs())

    if future_returns:
        ret_frame = pd.concat(future_returns, axis=1)
        dd_frame = pd.concat(future_drawdowns, axis=1)
        out[f'{output_col}_realized_vol'] = ret_frame.std(axis=1).fillna(0.0)
        out[f'{output_col}_path_drawdown'] = dd_frame.max(axis=1).fillna(0.0)
    else:
        out[f'{output_col}_realized_vol'] = 0.0
        out[f'{output_col}_path_drawdown'] = 0.0

    tradable = 1.0
    if tradable_col and tradable_col in out.columns:
        tradable = pd.to_numeric(out[tradable_col], errors='coerce').fillna(0.0).clip(0.0, 1.0)

    out[output_col] = (
        tradable * (out[raw_label_col] - float(fee) - float(slippage))
        - float(lambda_vol) * out[f'{output_col}_realized_vol']
        - float(lambda_dd) * out[f'{output_col}_path_drawdown']
    )
    return out


def build_relevance_bins(
    df,
    quality_col='quality5',
    output_col='relevance5',
    date_col='日期',
    n_bins=5,
):
    """Map same-day cross-sectional quality into integer relevance bins."""
    if n_bins < 2:
        raise ValueError('n_bins must be at least 2')

    out = df.copy()
    quality = pd.to_numeric(out[quality_col], errors='coerce')
    pct = quality.groupby(out[date_col]).rank(method='first', pct=True, ascending=True)
    relevance = np.floor(pct.to_numpy(dtype=np.float64) * int(n_bins)).astype(float)
    relevance = np.clip(relevance, 0, int(n_bins) - 1)
    relevance[pct.isna().to_numpy()] = np.nan
    out[output_col] = pd.Series(relevance, index=out.index).astype('Int64')
    return out


def build_aux_horizon_labels(
    df,
    horizons=(1, 3),
    stock_col='股票代码',
    date_col='日期',
    open_col='开盘',
    prefix='aux',
):
    """Add scorer-style auxiliary open-to-open labels for shorter horizons."""
    out = df.copy()
    for horizon in horizons:
        out = add_label_o2o_week(
            out,
            horizon=int(horizon),
            stock_col=stock_col,
            date_col=date_col,
            open_col=open_col,
            label_col=f'{prefix}{int(horizon)}',
        )
    return out


def realized_o2o_week_for_anchor(
    raw,
    anchor,
    horizon=5,
    stock_col='股票代码',
    date_col='日期',
    open_col='开盘',
):
    """Return score_self-compatible realized returns for one anchor date."""
    data = raw[[stock_col, date_col, open_col]].copy()
    data[stock_col] = data[stock_col].map(normalize_stock_id)
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values([date_col, stock_col]).reset_index(drop=True)

    dates = list(sorted(data[date_col].dropna().unique()))
    anchor = pd.Timestamp(anchor)
    idx = dates.index(anchor)
    d1 = pd.Timestamp(dates[idx + 1])
    dn = pd.Timestamp(dates[idx + horizon])

    open1 = data[data[date_col] == d1][[stock_col, open_col]].rename(
        columns={stock_col: 'stock_id', open_col: 'open_day1'}
    )
    openn = data[data[date_col] == dn][[stock_col, open_col]].rename(
        columns={stock_col: 'stock_id', open_col: f'open_day{horizon}'}
    )
    realized = open1.merge(openn, on='stock_id', how='inner')
    realized['realized_ret'] = (
        pd.to_numeric(realized[f'open_day{horizon}'], errors='coerce')
        / (pd.to_numeric(realized['open_day1'], errors='coerce') + 1e-12)
        - 1.0
    )
    return realized, f'{d1:%Y-%m-%d}~{dn:%Y-%m-%d}'
