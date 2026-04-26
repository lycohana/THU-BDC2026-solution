from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Regime(str, Enum):
    TREND = "trend"
    NEUTRAL = "neutral"
    DEFENSIVE = "defensive"


@dataclass
class HMMShadowModel:
    enabled: bool = False

    def fit(self, observations: pd.DataFrame) -> "HMMShadowModel":
        return self

    def predict_proba(self, observation: pd.Series) -> dict[Regime, float]:
        return {Regime.TREND: 1.0 / 3.0, Regime.NEUTRAL: 1.0 / 3.0, Regime.DEFENSIVE: 1.0 / 3.0}


class TriStateRegimeClassifier:
    def __init__(
        self,
        lookback: int = 252,
        hysteresis_days: int = 2,
        margin: float = 0.02,
        hmm_shadow: HMMShadowModel | None = None,
        use_hmm_hard_override: bool = False,
    ) -> None:
        self.lookback = lookback
        self.hysteresis_days = hysteresis_days
        self.margin = margin
        self.hmm_shadow = hmm_shadow or HMMShadowModel()
        self.use_hmm_hard_override = use_hmm_hard_override
        self.history = pd.DataFrame(columns=["Breadth", "Trend", "Stress"])
        self.current_state = Regime.NEUTRAL
        self.pending_state: Regime | None = None
        self.pending_count = 0

    def fit(self, hist_market_features: pd.DataFrame) -> "TriStateRegimeClassifier":
        self.history = hist_market_features[["Breadth", "Trend", "Stress"]].dropna().tail(self.lookback).copy()
        self.hmm_shadow.fit(self.history)
        return self

    def _thresholds(self) -> dict[str, float]:
        hist = self.history.tail(self.lookback)
        if len(hist) < 20:
            return {"trend_q60": 0.0, "breadth_q65": 0.55, "stress_q70": 1.2, "stress_q80": 1.4, "breadth_q35": 0.45, "trend_q40": 0.0}
        return {
            "trend_q60": float(hist["Trend"].quantile(0.60)),
            "breadth_q65": float(hist["Breadth"].quantile(0.65)),
            "stress_q70": float(hist["Stress"].quantile(0.70)),
            "stress_q80": float(hist["Stress"].quantile(0.80)),
            "breadth_q35": float(hist["Breadth"].quantile(0.35)),
            "trend_q40": float(hist["Trend"].quantile(0.40)),
        }

    def _raw_state(self, obs: pd.Series) -> tuple[Regime, float]:
        q = self._thresholds()
        trend_ok = obs["Trend"] >= q["trend_q60"] + self.margin and obs["Breadth"] >= q["breadth_q65"] and obs["Stress"] <= q["stress_q70"]
        defensive = obs["Stress"] >= q["stress_q80"] or (obs["Breadth"] <= q["breadth_q35"] and obs["Trend"] <= q["trend_q40"] - self.margin)
        if trend_ok:
            state = Regime.TREND
            distance = min(obs["Trend"] - q["trend_q60"], obs["Breadth"] - q["breadth_q65"], q["stress_q70"] - obs["Stress"])
        elif defensive:
            state = Regime.DEFENSIVE
            distance = max(obs["Stress"] - q["stress_q80"], q["breadth_q35"] - obs["Breadth"], q["trend_q40"] - obs["Trend"])
        else:
            state = Regime.NEUTRAL
            distance = self.margin
        confidence = float(np.clip(0.50 + abs(distance) / (abs(distance) + 0.10), 0.50, 0.99))
        return state, confidence

    def predict(self, observation: pd.Series, prev_regime: Regime | None = None) -> tuple[Regime, float, dict[str, object]]:
        obs = pd.Series(observation, dtype=float)
        raw, confidence = self._raw_state(obs)
        old_state = prev_regime or self.current_state

        if raw != old_state:
            if self.pending_state == raw:
                self.pending_count += 1
            else:
                self.pending_state = raw
                self.pending_count = 1
            state = raw if self.pending_count >= self.hysteresis_days else old_state
        else:
            self.pending_state = None
            self.pending_count = 0
            state = old_state

        hmm_probs = self.hmm_shadow.predict_proba(obs)
        if self.use_hmm_hard_override and self.hmm_shadow.enabled:
            hmm_state = max(hmm_probs, key=hmm_probs.get)
            if hmm_probs[hmm_state] >= 0.80:
                state = hmm_state

        self.current_state = state
        self.history = pd.concat([self.history, obs.to_frame().T], ignore_index=True).tail(self.lookback)
        debug = {"raw_state": raw.value, "pending_state": self.pending_state.value if self.pending_state else None, "pending_count": self.pending_count, "hmm_probs": {k.value: v for k, v in hmm_probs.items()}}
        return state, confidence, debug
