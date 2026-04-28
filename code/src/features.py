import numpy as np
import pandas as pd

from labels import normalize_stock_id


HISTORY_FEATURE_COLUMNS = [
    'stock_id',
    'sigma20',
    'median_amount20',
    'mean_amount20',
    'turnover20',
    'vol10',
    'amp_mean10',
    'ret1',
    'ret5',
    'ret10',
    'ret20',
    'intraday_ret',
    'amp20',
    'pos20',
    'amt_ratio5',
    'to_ratio5',
    'beta60',
    'downside_beta60',
    'idio_vol60',
    'max_drawdown20',
    'max_ret20_raw',
    'max_high_jump20',
    'amount20',
    'return_1',
    'return_5',
]


def build_history_feature_frame(raw_df, asof_date=None, lookback=80):
    """Build historical-only risk/shape features for the latest available date.

    All rolling windows are computed from rows with 日期 <= asof_date. The row at
    asof_date is allowed because predictions are assumed to run after that day's
    official data is available.
    """
    data = raw_df.copy()
    data['股票代码'] = data['股票代码'].map(normalize_stock_id)
    data['日期'] = pd.to_datetime(data['日期'])
    if asof_date is not None:
        data = data[data['日期'] <= pd.Timestamp(asof_date)].copy()
    data = data.sort_values(['股票代码', '日期']).reset_index(drop=True)

    hist = data.groupby('股票代码', sort=False).tail(int(lookback)).copy()
    if hist.empty:
        return pd.DataFrame(columns=HISTORY_FEATURE_COLUMNS)

    for col in ['开盘', '收盘', '最高', '最低', '成交额']:
        hist[col] = pd.to_numeric(hist[col], errors='coerce').astype(float)
    if '换手率' in hist.columns:
        hist['换手率'] = pd.to_numeric(hist['换手率'], errors='coerce').fillna(0.0).astype(float)
    else:
        hist['换手率'] = 0.0

    grouped = hist.groupby('股票代码', sort=False)
    hist['ret1'] = grouped['收盘'].pct_change(fill_method=None)
    hist['intraday_ret'] = hist['收盘'] / (hist['开盘'] + 1e-12) - 1.0
    prev_close = grouped['收盘'].shift(1)
    hist['daily_ret_raw'] = hist['收盘'] / (prev_close + 1e-12) - 1.0
    hist['high_jump'] = hist['最高'] / (prev_close + 1e-12) - 1.0
    hist['close_lag5'] = grouped['收盘'].shift(5)
    hist['close_lag10'] = grouped['收盘'].shift(10)
    hist['market_ret1'] = hist.groupby('日期', sort=False)['ret1'].transform('mean')
    hist['daily_amp'] = (hist['最高'] - hist['最低']) / (hist['收盘'].abs() + 1e-12)

    recent20 = grouped.tail(21).copy()
    grouped20 = recent20.groupby('股票代码', sort=False)
    agg = grouped20.agg(
        sigma20=('ret1', 'std'),
        median_amount20=('成交额', 'median'),
        mean_amount20=('成交额', 'mean'),
        turnover20=('换手率', 'mean'),
        first_close=('收盘', 'first'),
        max_high=('最高', 'max'),
        min_low=('最低', 'min'),
        max_ret20_raw=('daily_ret_raw', 'max'),
        max_high_jump20=('high_jump', 'max'),
        row_count=('收盘', 'size'),
    )

    recent10 = grouped.tail(10).copy()
    agg10 = recent10.groupby('股票代码', sort=False).agg(
        vol10=('ret1', 'std'),
        amp_mean10=('daily_amp', 'mean'),
    )

    recent5 = grouped.tail(5).copy()
    agg5 = recent5.groupby('股票代码', sort=False).agg(
        mean_amount5=('成交额', 'mean'),
        turnover5=('换手率', 'mean'),
    )

    latest = grouped.tail(1).set_index('股票代码').reindex(agg.index)

    last_close = latest['收盘'].to_numpy(dtype=np.float64)
    close_lag5 = latest['close_lag5'].fillna(0.0).to_numpy(dtype=np.float64)
    close_lag10 = latest['close_lag10'].fillna(0.0).to_numpy(dtype=np.float64)
    first_close = agg['first_close'].to_numpy(dtype=np.float64)
    row_count = agg['row_count'].to_numpy(dtype=np.int64)
    high20 = agg['max_high'].to_numpy(dtype=np.float64)
    low20 = agg['min_low'].to_numpy(dtype=np.float64)

    risk = pd.DataFrame(index=agg.index)
    risk['sigma20'] = agg['sigma20'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['median_amount20'] = agg['median_amount20'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['mean_amount20'] = agg['mean_amount20'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['turnover20'] = agg['turnover20'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['vol10'] = agg10.reindex(agg.index)['vol10'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['amp_mean10'] = agg10.reindex(agg.index)['amp_mean10'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['ret1'] = latest['ret1'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['ret5'] = np.where(row_count >= 6, last_close / (close_lag5 + 1e-12) - 1.0, 0.0)
    risk['ret10'] = np.where(row_count >= 11, last_close / (close_lag10 + 1e-12) - 1.0, 0.0)
    risk['ret20'] = np.where(row_count >= 2, last_close / (first_close + 1e-12) - 1.0, 0.0)
    risk['intraday_ret'] = latest['intraday_ret'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['amp20'] = (high20 - low20) / (last_close + 1e-12)
    risk['pos20'] = (last_close - low20) / (high20 - low20 + 1e-12)
    risk['max_ret20_raw'] = agg['max_ret20_raw'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['max_high_jump20'] = agg['max_high_jump20'].fillna(0.0).to_numpy(dtype=np.float64)

    amount5 = agg5.reindex(agg.index)['mean_amount5'].fillna(0.0).to_numpy(dtype=np.float64)
    turnover5 = agg5.reindex(agg.index)['turnover5'].fillna(0.0).to_numpy(dtype=np.float64)
    risk['amt_ratio5'] = amount5 / (risk['mean_amount20'].to_numpy(dtype=np.float64) + 1e-12)
    risk['to_ratio5'] = turnover5 / (risk['turnover20'].to_numpy(dtype=np.float64) + 1e-12)

    beta_rows = []
    for stock_id, group in grouped:
        group = group.tail(61).dropna(subset=['ret1', 'market_ret1']).copy()
        if len(group) < 20:
            beta_rows.append((stock_id, 0.0, 0.0, 0.0, 0.0))
            continue

        x = group['market_ret1'].to_numpy(dtype=np.float64)
        y = group['ret1'].to_numpy(dtype=np.float64)
        x_centered = x - x.mean()
        var_x = float(np.dot(x_centered, x_centered) / max(len(x_centered), 1))
        beta = float(np.cov(y, x, ddof=0)[0, 1] / (var_x + 1e-12))

        downside = group[group['market_ret1'] < 0]
        if len(downside) >= 8:
            dx = downside['market_ret1'].to_numpy(dtype=np.float64)
            dy = downside['ret1'].to_numpy(dtype=np.float64)
            dx_centered = dx - dx.mean()
            var_dx = float(np.dot(dx_centered, dx_centered) / max(len(dx_centered), 1))
            downside_beta = float(np.cov(dy, dx, ddof=0)[0, 1] / (var_dx + 1e-12))
        else:
            downside_beta = beta

        residual = y - beta * x
        idio_vol = float(np.std(residual))

        close20 = group['收盘'].tail(20).to_numpy(dtype=np.float64)
        if close20.size == 0:
            max_drawdown = 0.0
        else:
            running_max = np.maximum.accumulate(close20)
            max_drawdown = float(abs(np.min(close20 / (running_max + 1e-12) - 1.0)))

        beta_rows.append((stock_id, beta, downside_beta, idio_vol, max_drawdown))

    beta_df = pd.DataFrame(
        beta_rows,
        columns=['stock_id', 'beta60', 'downside_beta60', 'idio_vol60', 'max_drawdown20'],
    ).set_index('stock_id')
    risk = risk.join(beta_df, how='left')

    risk = risk.reset_index().rename(columns={'股票代码': 'stock_id'})
    risk['amount20'] = risk['median_amount20']
    risk['return_1'] = risk['ret1']
    risk['return_5'] = risk['ret5']
    return risk[HISTORY_FEATURE_COLUMNS]
