from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))

from tri_state_selector import SelectorConfig, TriStateSelector  # noqa: E402
from tri_state_selector.backtest import walk_forward_splits  # noqa: E402
from tri_state_selector.optimizer import PortfolioConstructor, covariance_is_pathological  # noqa: E402
from tri_state_selector.preprocess import align_fundamentals_asof, build_tradable_mask  # noqa: E402
from tri_state_selector.regime import Regime, TriStateRegimeClassifier  # noqa: E402
from tri_state_selector.risk import RiskShaper  # noqa: E402


def make_panel(n_stocks: int = 12, n_days: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    rows = []
    for i in range(n_stocks):
        close = 10.0 * np.cumprod(1.0 + rng.normal(0.001, 0.015 + i * 0.0005, size=n_days))
        stock_id = f"{i + 1:06d}"
        for j, d in enumerate(dates):
            c = close[j]
            rows.append(
                {
                    "date": d,
                    "stock_id": stock_id,
                    "open": c,
                    "high": c * 1.01,
                    "low": c * 0.99,
                    "close": c,
                    "amount": 80_000_000 + i * 2_000_000,
                    "turnover": 1.0,
                    "industry": f"IND{i % 4}",
                    "up_limit": c * 1.10,
                    "down_limit": c * 0.90,
                    "listed_days": 120 + j,
                    "is_suspended": False,
                }
            )
    return pd.DataFrame(rows)


def test_asof_join_has_no_lookahead():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-10", "2025-01-20"]),
            "stock_id": ["000001", "000001"],
            "close": [10.0, 11.0],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "stock_id": ["000001", "000001"],
            "announce_date": pd.to_datetime(["2025-01-15", "2025-01-25"]),
            "BP": [1.0, 2.0],
        }
    )
    aligned = align_fundamentals_asof(prices, fundamentals)
    assert pd.isna(aligned.loc[aligned["date"] == pd.Timestamp("2025-01-10"), "BP"]).all()
    assert aligned.loc[aligned["date"] == pd.Timestamp("2025-01-20"), "BP"].iloc[0] == 1.0


def test_state_hysteresis_requires_confirmation():
    hist = pd.DataFrame({"Breadth": [0.50] * 30, "Trend": [0.0] * 30, "Stress": [1.0] * 30})
    clf = TriStateRegimeClassifier(hysteresis_days=2).fit(hist)
    s1, _, d1 = clf.predict(pd.Series({"Breadth": 0.90, "Trend": 0.10, "Stress": 0.50}), Regime.NEUTRAL)
    s2, _, d2 = clf.predict(pd.Series({"Breadth": 0.91, "Trend": 0.11, "Stress": 0.50}), Regime.NEUTRAL)
    assert s1 == Regime.NEUTRAL
    assert s2 == Regime.TREND
    assert d1["pending_count"] == 1
    assert d2["raw_state"] == "trend"


def test_limit_up_down_tradability():
    day = pd.DataFrame(
        {
            "stock_id": ["000001", "000002", "000003"],
            "close": [11.0, 9.0, 10.0],
            "up_limit": [11.0, 11.0, 11.0],
            "down_limit": [9.0, 9.0, 9.0],
            "is_suspended": [False, False, True],
        }
    )
    mask = build_tradable_mask(day, pd.Series({"000002": 0.1}))
    assert not bool(mask.loc[mask["stock_id"] == "000001", "can_buy"].iloc[0])
    assert not bool(mask.loc[mask["stock_id"] == "000002", "can_sell"].iloc[0])
    assert not bool(mask.loc[mask["stock_id"] == "000003", "tradable"].iloc[0])


def test_constraints_are_satisfied():
    panel = make_panel()
    selector = TriStateSelector(SelectorConfig(regime_hysteresis_days=1, top_quantile=1.0))
    out = selector.rebalance(prices=panel, asof=panel["date"].max(), prev_weights=pd.Series(dtype=float))
    stock_weights = out.target_weights.drop(index="CASH", errors="ignore")
    cap = selector.cfg.single_name_cap[out.state.value]
    assert float(stock_weights.max()) <= cap + 1e-8
    assert float(out.target_weights.sum()) <= 1.0 + 1e-8
    assert float(out.risk_report["portfolio_es95"]) <= selector.cfg.portfolio_es_limit + 1e-8
    assert {"state", "confidence", "portfolio_es95", "fallback"}.issubset(out.risk_report.keys())


def test_drawdown_governor_reduces_exposure():
    cfg = SelectorConfig()
    nav = pd.Series([1.0, 0.98, 0.95, 0.87])
    multiplier, state = RiskShaper(cfg).drawdown_multiplier(nav)
    assert state == "hard"
    assert multiplier == 0.50


def test_pathological_covariance_fallback():
    cov = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]], index=["a", "b"], columns=["a", "b"])
    assert covariance_is_pathological(cov)
    feats = pd.DataFrame(
        {
            "VOL_20": [0.2, 0.3],
            "industry": ["A", "B"],
            "tradable": [True, True],
            "can_buy": [True, True],
            "listed_days": [100, 100],
            "ADV20": [1.0, 1.0],
        },
        index=["000001", "000002"],
    )
    scores = pd.Series([1.0, 0.5], index=feats.index)
    returns = pd.DataFrame({"000001": [0.01, 0.01, 0.01], "000002": [0.02, 0.02, 0.02]})
    result = PortfolioConstructor(SelectorConfig()).build(Regime.NEUTRAL, scores, feats, returns=returns)
    assert result.risk_report["fallback"] == "capped_inverse_vol"


def test_walk_forward_splits_do_not_leak():
    dates = pd.bdate_range("2020-01-01", periods=100)
    splits = walk_forward_splits(dates, train_days=40, validation_days=20, test_days=10, step_days=10)
    assert splits
    for split in splits:
        assert split.train.max() < split.validation.min()
        assert split.validation.max() < split.test.min()


def test_minimal_selector_outputs_required_surfaces():
    panel = make_panel(n_stocks=20, n_days=120)
    out = TriStateSelector(SelectorConfig(regime_hysteresis_days=1)).rebalance(
        prices=panel,
        asof=panel["date"].max(),
        prev_weights=pd.Series(dtype=float),
    )
    assert out.state in {Regime.TREND, Regime.NEUTRAL, Regime.DEFENSIVE}
    assert 0.0 <= out.confidence <= 1.0
    assert not out.ranked_candidates.empty
    assert not out.target_weights.empty
    assert set(out.order_list.columns) == {"stock_id", "side", "target_weight", "delta_weight"}
