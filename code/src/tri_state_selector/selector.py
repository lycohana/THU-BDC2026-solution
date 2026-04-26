from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .branches import DefensiveStrategy, NeutralStrategy, TrendStrategy
from .config import SelectorConfig
from .data import normalize_panel
from .features import FeatureEngine
from .monitor import build_order_list
from .optimizer import PortfolioConstructor
from .regime import Regime, TriStateRegimeClassifier
from .risk import RiskShaper


@dataclass
class RebalanceInput:
    asof: pd.Timestamp
    prices: pd.DataFrame
    fundamentals: pd.DataFrame | None = None
    prev_weights: pd.Series | None = None
    nav: pd.Series | None = None


@dataclass
class RebalanceOutput:
    state: Regime
    confidence: float
    ranked_candidates: pd.DataFrame
    target_weights: pd.Series
    risk_report: dict[str, object]
    order_list: pd.DataFrame


class TriStateSelector:
    def __init__(self, cfg: SelectorConfig | None = None) -> None:
        self.cfg = cfg or SelectorConfig()
        self.features = FeatureEngine()
        self.regime = TriStateRegimeClassifier(
            lookback=self.cfg.regime_lookback,
            hysteresis_days=self.cfg.regime_hysteresis_days,
            use_hmm_hard_override=self.cfg.use_hmm_hard_override,
        )
        self.risk = RiskShaper(self.cfg)
        self.portfolio = PortfolioConstructor(self.cfg)
        self.branches = {
            Regime.TREND: TrendStrategy(),
            Regime.NEUTRAL: NeutralStrategy(),
            Regime.DEFENSIVE: DefensiveStrategy(),
        }
        self.prev_state: Regime | None = None

    def fit_regime(self, prices: pd.DataFrame, asof: pd.Timestamp) -> "TriStateSelector":
        panel = normalize_panel(prices)
        hist = self.features.compute_market_feature_history(panel, asof, self.cfg.regime_lookback)
        self.regime.fit(hist)
        return self

    def rebalance(
        self,
        *,
        prices: pd.DataFrame,
        asof: pd.Timestamp,
        fundamentals: pd.DataFrame | None = None,
        prev_weights: pd.Series | None = None,
        nav: pd.Series | None = None,
    ) -> RebalanceOutput:
        rb = RebalanceInput(asof=asof, prices=prices, fundamentals=fundamentals, prev_weights=prev_weights, nav=nav)
        return self.rebalance_input(rb)

    def rebalance_input(self, rb: RebalanceInput) -> RebalanceOutput:
        panel = normalize_panel(rb.prices)
        asof = pd.Timestamp(rb.asof)
        stock_feats = self.features.compute_stock_features(panel, asof, rb.fundamentals, rb.prev_weights)
        market_feats = self.features.compute_market_features(panel, asof)
        if self.regime.history.empty:
            self.fit_regime(panel, asof)
        state, confidence, regime_debug = self.regime.predict(market_feats, self.prev_state)

        raw_scores = self.branches[state].score(stock_feats)
        scores = self.risk.apply_stock_penalties(raw_scores, stock_feats)
        returns = _return_matrix(panel, asof)
        result = self.portfolio.build(
            state,
            scores,
            stock_feats,
            returns=returns,
            prev_weights=rb.prev_weights,
            nav=rb.nav,
        )
        orders = build_order_list(result.weights, rb.prev_weights)
        ranked = stock_feats.join(scores.rename("score"), how="inner").sort_values("score", ascending=False)
        risk_report = {
            "mode": self.cfg.mode,
            "state": state.value,
            "confidence": confidence,
            "market_features": market_feats.to_dict(),
            "regime_debug": regime_debug,
            "top_raw_scores": raw_scores.sort_values(ascending=False).head(30),
            "top_shaped_scores": scores.sort_values(ascending=False).head(30),
            **result.risk_report,
        }
        self.prev_state = state
        return RebalanceOutput(
            state=state,
            confidence=confidence,
            ranked_candidates=ranked,
            target_weights=result.weights,
            risk_report=risk_report,
            order_list=orders,
        )


def _return_matrix(panel: pd.DataFrame, asof: pd.Timestamp, lookback: int = 252) -> pd.DataFrame:
    hist = panel[pd.to_datetime(panel["date"]) <= pd.Timestamp(asof)].copy()
    wide = hist.pivot(index="date", columns="stock_id", values="close").sort_index()
    return wide.pct_change(fill_method=None).tail(lookback)
