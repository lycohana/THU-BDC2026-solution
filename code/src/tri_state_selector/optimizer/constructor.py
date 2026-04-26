from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import SelectorConfig
from ..regime.classifier import Regime
from ..risk.shaper import RiskShaper


@dataclass
class PortfolioResult:
    weights: pd.Series
    risk_report: dict[str, object]
    candidates: pd.DataFrame


def covariance_is_pathological(cov: pd.DataFrame | None, condition_limit: float = 1.0e8) -> bool:
    if cov is None or cov.empty or cov.shape[0] != cov.shape[1]:
        return True
    arr = cov.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        return True
    eigvals = np.linalg.eigvalsh((arr + arr.T) / 2.0)
    if eigvals.min() <= 1e-12:
        return True
    return bool(eigvals.max() / eigvals.min() > condition_limit)


class PortfolioConstructor:
    def __init__(self, cfg: SelectorConfig) -> None:
        self.cfg = cfg
        self.risk = RiskShaper(cfg)

    def build(
        self,
        regime: Regime,
        scores: pd.Series,
        feats: pd.DataFrame,
        *,
        returns: pd.DataFrame | None = None,
        prev_weights: pd.Series | None = None,
        nav: pd.Series | None = None,
    ) -> PortfolioResult:
        frame = feats.join(scores.rename("score"), how="inner")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
        frame = frame[(frame.get("tradable", True) == True) & (frame.get("can_buy", True) == True)]
        frame = frame[frame.get("listed_days", self.cfg.min_listed_days) >= self.cfg.min_listed_days]
        if "ADV20" in frame.columns:
            frame = frame[frame["ADV20"] >= frame["ADV20"].quantile(0.10)]
        if frame.empty:
            cash = pd.Series({"CASH": 1.0}, name="weight")
            return PortfolioResult(cash, {"fallback": "cash_no_candidates"}, pd.DataFrame())

        if self.cfg.mode == "competition":
            n_top = min(int(self.cfg.output_top_k), len(frame))
        else:
            n_top = min(self.cfg.max_names[regime.value], max(1, int(np.ceil(len(frame) * self.cfg.top_quantile))))
        candidates = frame.sort_values("score", ascending=False).head(n_top).copy()
        cov, corr = self._cov_corr(returns, candidates.index)
        fallback = "none"
        if covariance_is_pathological(cov, self.cfg.cov_condition_limit):
            fallback = "capped_inverse_vol"
            raw = self._capped_inverse_vol(candidates)
        else:
            raw = self._score_tilted_inverse_vol(candidates)

        exposure = 1.0
        allow_cash = self.cfg.mode == "research" or self.cfg.allow_cash
        if regime == Regime.DEFENSIVE and allow_cash:
            exposure = 1.0 - self.cfg.cash_floor_defensive
        raw = raw / (raw.sum() + 1e-12) * exposure
        shaped, risk_report = self.risk.shape_portfolio(
            raw,
            feats,
            regime,
            returns=returns,
            corr=corr,
            prev_weights=prev_weights,
            nav=nav,
        )
        if regime == Regime.DEFENSIVE and allow_cash:
            shaped = shaped * min(1.0, exposure / (shaped.sum() + 1e-12))
        weight_sum_before_topk = float(shaped.sum())
        weight_sum_after_topk_normalize = weight_sum_before_topk
        if self.cfg.mode == "competition" and self.cfg.force_top_k_full_invest:
            shaped = self._force_full_investment(shaped, candidates, regime)
            weight_sum_after_topk_normalize = float(shaped.sum())
            risk_report.setdefault("reason_codes", []).append("force_top_k_full_invest_normalized_after_risk_stages")
        cash = max(0.0, 1.0 - float(shaped.sum()))
        if cash > 1e-10 and (self.cfg.mode == "research" or allow_cash):
            shaped.loc["CASH"] = cash
        shaped.name = "weight"
        risk_report.update(
            {
                "fallback": fallback,
                "gross_exposure": float(shaped.drop(index="CASH", errors="ignore").sum()),
                "cash_weight": float(shaped.get("CASH", 0.0)),
                "name_count": int(len(shaped.drop(index="CASH", errors="ignore"))),
                "selected_before_risk": candidates["score"].copy(),
                "selected_after_risk": shaped.drop(index="CASH", errors="ignore").copy(),
                "selected_for_competition_output": shaped.drop(index="CASH", errors="ignore").copy()
                if self.cfg.mode == "competition"
                else pd.Series(dtype=float),
                "output_top_k": int(self.cfg.output_top_k),
                "force_top_k_full_invest": bool(self.cfg.force_top_k_full_invest),
                "weight_sum_before_topk": weight_sum_before_topk,
                "weight_sum_after_topk_normalize": weight_sum_after_topk_normalize,
            }
        )
        return PortfolioResult(shaped.sort_values(ascending=False), risk_report, candidates)

    def _force_full_investment(self, shaped: pd.Series, candidates: pd.DataFrame, regime: Regime) -> pd.Series:
        stock_weights = shaped.drop(index="CASH", errors="ignore").reindex(candidates.index).fillna(0.0)
        if stock_weights.sum() <= 1e-12:
            stock_weights = pd.Series(1.0, index=candidates.index, dtype=float)
        normalized = stock_weights / (stock_weights.sum() + 1e-12)
        capped, _ = self.risk.cap_weights(normalized, candidates, regime)
        if capped.sum() > 1e-12 and capped.sum() < 1.0 - 1e-10:
            capped = capped / capped.sum()
            capped, _ = self.risk.cap_weights(capped, candidates, regime)
        return capped

    def _score_tilted_inverse_vol(self, candidates: pd.DataFrame) -> pd.Series:
        score = candidates["score"].astype(float)
        score_pos = (score - score.min()).clip(lower=0.0) + 1e-6
        vol = candidates.get("VOL_20", pd.Series(1.0, index=candidates.index)).abs().replace(0.0, np.nan).fillna(1.0)
        raw = np.power(score_pos, self.cfg.score_tilt) / (vol + 1e-6)
        return pd.Series(raw, index=candidates.index, dtype=float)

    def _capped_inverse_vol(self, candidates: pd.DataFrame) -> pd.Series:
        vol = candidates.get("VOL_20", pd.Series(1.0, index=candidates.index)).abs().replace(0.0, np.nan).fillna(1.0)
        return pd.Series(1.0 / (vol + 1e-6), index=candidates.index, dtype=float)

    def _cov_corr(self, returns: pd.DataFrame | None, names: pd.Index) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        if returns is None or returns.empty:
            return None, None
        aligned = returns.reindex(columns=names).dropna(how="all", axis=1).tail(252)
        if aligned.shape[1] < 2:
            return None, None
        cov = aligned.cov() * 252.0
        corr = aligned.corr()
        return cov, corr
