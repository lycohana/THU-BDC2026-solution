from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    test: pd.DatetimeIndex


def walk_forward_splits(
    dates: pd.Index,
    *,
    train_days: int = 252 * 5,
    validation_days: int = 252,
    test_days: int = 126,
    step_days: int = 63,
) -> list[WalkForwardSplit]:
    unique = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    splits: list[WalkForwardSplit] = []
    start = 0
    total = train_days + validation_days + test_days
    while start + total <= len(unique):
        train = unique[start : start + train_days]
        val = unique[start + train_days : start + train_days + validation_days]
        test = unique[start + train_days + validation_days : start + total]
        if train.max() >= val.min() or val.max() >= test.min():
            raise AssertionError("walk-forward windows overlap or leak forward")
        splits.append(WalkForwardSplit(train=train, validation=val, test=test))
        start += step_days
    return splits


def stationary_bootstrap(x: pd.Series | np.ndarray, n_boot: int = 1000, p: float = 0.10, seed: int = 42) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(arr)
    if n == 0:
        return np.empty((0, 0))
    samples = np.empty((n_boot, n), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n)
        for t in range(n):
            samples[b, t] = arr[idx]
            if rng.random() < p:
                idx = rng.integers(0, n)
            else:
                idx = (idx + 1) % n
    return samples


def white_reality_check(*args, **kwargs) -> dict[str, object]:
    return {"implemented": False, "method": "White Reality Check", "reason": "interface placeholder"}


def hansen_spa(*args, **kwargs) -> dict[str, object]:
    return {"implemented": False, "method": "Hansen SPA", "reason": "interface placeholder"}


def deflated_sharpe_ratio(*args, **kwargs) -> dict[str, object]:
    return {"implemented": False, "method": "Deflated Sharpe Ratio", "reason": "interface placeholder"}
