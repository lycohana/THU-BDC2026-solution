from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Order:
    stock_id: str
    side: str
    target_weight: float
    delta_weight: float


def empty_weights() -> pd.Series:
    return pd.Series(dtype=float, name="weight")
