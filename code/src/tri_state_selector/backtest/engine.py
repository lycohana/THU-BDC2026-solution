from __future__ import annotations

import pandas as pd


class BacktestEngine:
    """Minimal EOD skeleton for walk-forward integration."""

    def __init__(self, selector) -> None:
        self.selector = selector

    def run_rebalance_dates(self, prices: pd.DataFrame, rebalance_dates: list[pd.Timestamp], **kwargs) -> list[object]:
        outputs = []
        for date in rebalance_dates:
            outputs.append(self.selector.rebalance(prices=prices, asof=pd.Timestamp(date), **kwargs))
        return outputs
