from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SelectorConfig
from ..regime.classifier import Regime


def expected_shortfall(returns: pd.Series | np.ndarray, alpha: float = 0.95) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    cutoff = np.quantile(arr, 1.0 - alpha)
    tail = arr[arr <= cutoff]
    return float(abs(tail.mean())) if tail.size else 0.0


def max_drawdown(nav: pd.Series | np.ndarray) -> float:
    arr = np.asarray(nav, dtype=float)
    if arr.size == 0:
        return 0.0
    running = np.maximum.accumulate(arr)
    dd = arr / (running + 1e-12) - 1.0
    return float(abs(np.min(dd)))


class RiskShaper:
    def __init__(self, cfg: SelectorConfig) -> None:
        self.cfg = cfg

    def apply_stock_penalties(self, scores: pd.Series, feats: pd.DataFrame) -> pd.Series:
        tail = feats.get("TAIL_RISK", pd.Series(0.0, index=scores.index)).reindex(scores.index).fillna(0.0)
        illiq = feats.get("ILLIQ20", pd.Series(0.0, index=scores.index)).reindex(scores.index).fillna(0.0)
        return (scores - 0.08 * tail - 0.02 * illiq).rename("score")

    def drawdown_multiplier(self, nav: pd.Series | None) -> tuple[float, str]:
        if nav is None or len(nav) < 2:
            return 1.0, "none"
        dd63 = max_drawdown(nav.tail(63))
        dd126 = max_drawdown(nav.tail(126))
        if dd126 >= self.cfg.dd_hard:
            return 0.50, "hard"
        if dd63 >= self.cfg.dd_soft:
            return 0.75, "soft"
        return 1.0, "none"

    def cap_weights(
        self,
        weights: pd.Series,
        feats: pd.DataFrame,
        regime: Regime,
    ) -> tuple[pd.Series, dict[str, object]]:
        stock_cap = self.effective_single_name_cap(regime)
        industries, industry_cap, policy = self.effective_industries(feats, weights.index, regime)
        out = weights.clip(lower=0.0, upper=stock_cap)
        out = _redistribute_to_cap(out, target_sum=weights.sum(), cap=stock_cap)
        after_single_cap = out.copy()

        if policy != "ignore":
            for _ in range(20):
                changed = False
                for industry, names in industries.groupby(industries).groups.items():
                    total = float(out.loc[list(names)].sum())
                    if total > industry_cap + 1e-10:
                        scale = industry_cap / total
                        out.loc[list(names)] *= scale
                        changed = True
                out = _redistribute_with_industry_caps(out, weights.sum(), stock_cap, industry_cap, industries)
                if not changed:
                    break
        report = {
            "max_name_weight": float(out.max()) if len(out) else 0.0,
            "max_industry_weight": float(out.groupby(industries).sum().max()) if len(out) else 0.0,
            "single_name_cap_effective": float(stock_cap),
            "industry_cap_effective": float(industry_cap),
            "unknown_industry_policy_effective": policy,
            "weights_after_single_cap": after_single_cap,
            "weights_after_industry_cap": out.copy(),
        }
        return out, report

    def effective_single_name_cap(self, regime: Regime) -> float:
        cap = float(self.cfg.single_name_cap[regime.value])
        if self.cfg.mode == "competition":
            cap = max(cap, 0.25)
        return cap

    def effective_industries(
        self,
        feats: pd.DataFrame,
        names: pd.Index,
        regime: Regime,
    ) -> tuple[pd.Series, float, str]:
        industry_cap = float(self.cfg.industry_cap[regime.value])
        if "industry" in feats.columns:
            industries = feats["industry"].reindex(names).fillna("UNKNOWN").astype(str)
        else:
            industries = pd.Series("UNKNOWN", index=names, dtype=str)

        values = industries.fillna("UNKNOWN").astype(str)
        all_unknown = values.empty or values.str.upper().isin({"UNKNOWN", "", "NAN", "NONE"}).all()
        policy = self.cfg.unknown_industry_policy
        if self.cfg.mode == "competition" and all_unknown:
            policy = "stock_as_industry"
        if policy == "stock_as_industry":
            industries = pd.Series(names.astype(str), index=names, dtype=str)
            if self.cfg.mode == "competition" and self.cfg.force_top_k_full_invest:
                industry_cap = max(industry_cap, 1.0 / max(int(self.cfg.output_top_k), 1))
        elif policy == "ignore":
            industries = pd.Series("ALL_IGNORED", index=names, dtype=str)
            industry_cap = 1.0
        return industries, industry_cap, policy

    def apply_corr_cluster_caps(
        self,
        weights: pd.Series,
        corr: pd.DataFrame | None,
    ) -> tuple[pd.Series, dict[str, object]]:
        if corr is None or corr.empty or weights.empty:
            return weights, {"max_pair_corr": 0.0, "fallback": "no_corr"}
        names = [n for n in weights.index if n in corr.index and n in corr.columns]
        if len(names) < 2:
            return weights, {"max_pair_corr": 0.0, "fallback": "too_few_names"}

        out = weights.copy()
        sub = corr.loc[names, names].fillna(0.0)
        mask = ~np.eye(len(sub), dtype=bool)
        max_pair = float(sub.where(mask).max().max())
        clusters: list[list[str]] = []
        unused = set(names)
        while unused:
            seed = unused.pop()
            cluster = {seed}
            linked = set(sub.index[sub.loc[seed] >= self.cfg.pair_corr_cap])
            cluster |= linked
            unused -= cluster
            clusters.append(sorted(cluster))
        for cluster in clusters:
            total = float(out.reindex(cluster).fillna(0.0).sum())
            if total > self.cfg.cluster_cap:
                out.loc[cluster] *= self.cfg.cluster_cap / total
        return out, {"max_pair_corr": max_pair, "cluster_count": len(clusters)}

    def apply_es_constraint(
        self,
        weights: pd.Series,
        returns: pd.DataFrame | None,
    ) -> tuple[pd.Series, dict[str, object]]:
        if returns is None or returns.empty or weights.empty:
            return weights, {"portfolio_es95": 0.0, "es_scale": 1.0}
        aligned = returns.reindex(columns=weights.index).fillna(0.0)
        port_ret = aligned.to_numpy(dtype=float) @ weights.to_numpy(dtype=float)
        es95 = expected_shortfall(port_ret, 0.95)
        scale = min(1.0, self.cfg.portfolio_es_limit / (es95 + 1e-12)) if es95 > 0 else 1.0
        return weights * scale, {"raw_portfolio_es95": es95, "portfolio_es95": float(es95 * scale), "es_scale": float(scale)}

    def apply_turnover_penalty(self, weights: pd.Series, prev_weights: pd.Series | None) -> tuple[pd.Series, dict[str, object]]:
        if prev_weights is None or prev_weights.empty or weights.empty:
            return weights, {"turnover": float(weights.abs().sum())}
        prev = prev_weights.reindex(weights.index).fillna(0.0)
        lam = float(np.clip(self.cfg.turnover_penalty, 0.0, 0.95))
        blended = (1.0 - lam) * weights + lam * prev
        turnover = float((blended - prev).abs().sum())
        return blended, {"turnover": turnover}

    def shape_portfolio(
        self,
        weights: pd.Series,
        feats: pd.DataFrame,
        regime: Regime,
        *,
        returns: pd.DataFrame | None = None,
        corr: pd.DataFrame | None = None,
        prev_weights: pd.Series | None = None,
        nav: pd.Series | None = None,
    ) -> tuple[pd.Series, dict[str, object]]:
        report: dict[str, object] = {
            "raw_weights": weights.copy(),
            "reason_codes": [],
        }
        out, item = self.cap_weights(weights, feats, regime)
        report.update(item)
        if abs(float(report["weights_after_single_cap"].sum()) - float(weights.sum())) > 1e-10:
            report["reason_codes"].append("single_name_cap_reduced_total_weight")
        if abs(float(report["weights_after_industry_cap"].sum()) - float(report["weights_after_single_cap"].sum())) > 1e-10:
            report["reason_codes"].append("industry_cap_reduced_total_weight")
        out, item = self.apply_corr_cluster_caps(out, corr)
        report.update(item)
        report["weights_after_corr_cap"] = out.copy()
        if abs(float(out.sum()) - float(report["weights_after_industry_cap"].sum())) > 1e-10:
            report["reason_codes"].append("corr_cluster_cap_reduced_total_weight")
        report["weights_after_vol_target"] = out.copy()
        report["reason_codes"].append("vol_target_stage_noop_currently_not_applied")
        out, item = self.apply_es_constraint(out, returns)
        report.update(item)
        report["weights_after_es_constraint"] = out.copy()
        if item.get("es_scale", 1.0) < 1.0:
            report["reason_codes"].append("portfolio_es_constraint_scaled_exposure")
        dd_mult, dd_state = self.drawdown_multiplier(nav)
        out = out * dd_mult
        report["weights_after_drawdown_governor"] = out.copy()
        report.update({"drawdown_governor": dd_state, "drawdown_multiplier": dd_mult})
        if dd_mult < 1.0:
            report["reason_codes"].append(f"drawdown_governor_{dd_state}_scaled_exposure")
        out, item = self.apply_turnover_penalty(out, prev_weights)
        report.update(item)
        report["weights_after_turnover_penalty"] = out.copy()
        if prev_weights is not None and not prev_weights.empty and self.cfg.turnover_penalty > 0:
            report["reason_codes"].append("turnover_penalty_blended_with_previous_weights")
        return out.clip(lower=0.0), report


def _redistribute_to_cap(weights: pd.Series, target_sum: float, cap: float) -> pd.Series:
    if weights.empty or cap <= 0:
        return weights * 0.0
    out = weights.clip(lower=0.0, upper=cap).copy()
    target_sum = min(float(target_sum), cap * len(out))
    for _ in range(50):
        shortfall = target_sum - float(out.sum())
        if abs(shortfall) <= 1e-10:
            break
        room = cap - out
        eligible = room > 1e-12
        if not eligible.any() or shortfall < 0:
            if out.sum() > 0:
                out *= target_sum / out.sum()
            break
        out.loc[eligible] += shortfall * room.loc[eligible] / room.loc[eligible].sum()
        out = out.clip(upper=cap)
    return out


def _redistribute_with_industry_caps(
    weights: pd.Series,
    target_sum: float,
    stock_cap: float,
    industry_cap: float,
    industries: pd.Series,
) -> pd.Series:
    out = weights.clip(lower=0.0, upper=stock_cap).copy()
    target_sum = min(target_sum, stock_cap * len(out), industry_cap * industries.nunique())
    for _ in range(30):
        shortfall = target_sum - float(out.sum())
        if shortfall <= 1e-10:
            return out
        room = stock_cap - out
        ind_room = industries.map(lambda ind: industry_cap - out.groupby(industries).sum().get(ind, 0.0))
        eligible = (room > 1e-12) & (ind_room > 1e-12)
        if not eligible.any():
            return out
        add_room = np.minimum(room.loc[eligible], ind_room.loc[eligible])
        out.loc[eligible] += shortfall * add_room / (add_room.sum() + 1e-12)
        out = out.clip(upper=stock_cap)
    return out
