import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))

from utils import engineer_features  # noqa: E402


def _price_frame(length: int, *, include_nan: bool = False) -> pd.DataFrame:
    index = pd.Index([f"row_{i}" for i in range(length)], name="row_id")
    base = np.linspace(10.0, 10.0 + max(length - 1, 0), length)
    if include_nan and length >= 3:
        base[1] = np.nan
    close = pd.Series(base, index=index).ffill().bfill().fillna(10.0)
    return pd.DataFrame(
        {
            "开盘": close + 0.1,
            "收盘": close,
            "最高": close + 0.3,
            "最低": close - 0.3,
            "成交量": np.linspace(1000.0, 1000.0 + 10.0 * max(length - 1, 0), length),
            "成交额": np.linspace(10000.0, 10000.0 + 100.0 * max(length - 1, 0), length),
        },
        index=index,
    )


def _assert_engineer_features_index_safe(frame: pd.DataFrame) -> pd.DataFrame:
    out = engineer_features(frame)
    assert len(out) == len(frame)
    assert out.index.equals(frame.index)
    for col in ["RSQR5", "RSQR10", "RSQR20", "RSQR30", "RSQR60"]:
        assert col in out.columns
        assert len(out[col]) == len(frame)
    return out


def test_engineer_features_rsqr_handles_extremely_short_series():
    _assert_engineer_features_index_safe(_price_frame(3))


def test_engineer_features_rsqr_handles_series_shorter_than_regression_window():
    _assert_engineer_features_index_safe(_price_frame(9))


def test_engineer_features_rsqr_handles_series_equal_to_regression_window():
    _assert_engineer_features_index_safe(_price_frame(10))


def test_engineer_features_rsqr_handles_nan_input_without_length_mismatch():
    out = _assert_engineer_features_index_safe(_price_frame(9, include_nan=True))
    assert not out.replace([np.inf, -np.inf], np.nan).isna().any().any()
