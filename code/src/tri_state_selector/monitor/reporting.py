from __future__ import annotations

import pandas as pd


def build_order_list(target_weights: pd.Series, prev_weights: pd.Series | None = None) -> pd.DataFrame:
    prev = pd.Series(dtype=float) if prev_weights is None else prev_weights.copy()
    prev.index = prev.index.astype(str)
    tgt = target_weights.copy()
    tgt.index = tgt.index.astype(str)
    names = sorted(set(prev.index) | set(tgt.index))
    rows = []
    for name in names:
        if name == "CASH":
            continue
        delta = float(tgt.get(name, 0.0) - prev.get(name, 0.0))
        if abs(delta) <= 1e-10:
            continue
        rows.append(
            {
                "stock_id": name,
                "side": "BUY" if delta > 0 else "SELL",
                "target_weight": float(tgt.get(name, 0.0)),
                "delta_weight": delta,
            }
        )
    return pd.DataFrame(rows, columns=["stock_id", "side", "target_weight", "delta_weight"])


def risk_report_summary(report: dict[str, object]) -> dict[str, object]:
    keys = ["portfolio_es95", "drawdown_governor", "max_name_weight", "max_industry_weight", "cash_weight", "fallback", "turnover"]
    return {key: report.get(key) for key in keys if key in report}
