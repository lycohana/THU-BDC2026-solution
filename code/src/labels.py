import pandas as pd


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

    open_t1 = open_base.rename(columns={open_col: 'open_t1'})
    open_t1['_date_idx'] -= 1

    open_tn = open_base.rename(columns={open_col: f'open_t{horizon}'})
    open_tn['_date_idx'] -= horizon

    out = out.merge(open_t1[[stock_col, '_date_idx', 'open_t1']], on=[stock_col, '_date_idx'], how='left')
    out = out.merge(open_tn[[stock_col, '_date_idx', f'open_t{horizon}']], on=[stock_col, '_date_idx'], how='left')

    out[label_col] = (
        pd.to_numeric(out[f'open_t{horizon}'], errors='coerce')
        / (pd.to_numeric(out['open_t1'], errors='coerce') + 1e-12)
        - 1.0
    )
    return out.drop(columns=['_date_idx', 'open_t1', f'open_t{horizon}'])


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
