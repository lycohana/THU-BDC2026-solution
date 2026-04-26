from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SelectorConfig:
    """Runtime defaults for the long-only A-share selector."""

    mode: Literal["research", "competition"] = "research"
    unknown_industry_policy: Literal["cap", "ignore", "stock_as_industry"] = "cap"
    output_top_k: int = 5
    force_top_k_full_invest: bool = False
    allow_cash: bool = False
    rebalance_every: int = 5
    min_listed_days: int = 60
    min_adv20: float = 30_000_000.0
    top_quantile: float = 0.20
    max_names: dict[str, int] = field(
        default_factory=lambda: {"trend": 15, "neutral": 24, "defensive": 18}
    )
    target_vol: dict[str, float] = field(
        default_factory=lambda: {"trend": 0.18, "neutral": 0.12, "defensive": 0.08}
    )
    single_name_cap: dict[str, float] = field(
        default_factory=lambda: {"trend": 0.08, "neutral": 0.05, "defensive": 0.04}
    )
    industry_cap: dict[str, float] = field(
        default_factory=lambda: {"trend": 0.25, "neutral": 0.20, "defensive": 0.15}
    )
    cluster_cap: float = 0.35
    pair_corr_cap: float = 0.55
    portfolio_es_limit: float = 0.035
    dd_soft: float = 0.08
    dd_hard: float = 0.12
    cash_floor_defensive: float = 0.30
    score_tilt: float = 1.5
    turnover_penalty: float = 0.05
    regime_lookback: int = 252
    regime_hysteresis_days: int = 2
    use_hmm_hard_override: bool = False
    cov_condition_limit: float = 1.0e8
    random_seed: int = 42
