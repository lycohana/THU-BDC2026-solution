from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))

from tri_state_selector import SelectorConfig, TriStateSelector  # noqa: E402
from tri_state_selector.data import load_panel_csv  # noqa: E402


def make_demo_panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-02", periods=180)
    rows = []
    for i in range(30):
        stock_id = f"{i + 1:06d}"
        drift = 0.0002 + i / 200000
        rets = rng.normal(drift, 0.018 + (i % 5) * 0.002, size=len(dates))
        close = 20.0 * np.cumprod(1.0 + rets)
        industry = f"IND{i % 6}"
        for d, c in zip(dates, close):
            rows.append(
                {
                    "date": d,
                    "stock_id": stock_id,
                    "open": c * (1.0 + rng.normal(0, 0.002)),
                    "high": c * 1.02,
                    "low": c * 0.98,
                    "close": c,
                    "amount": 50_000_000 + i * 1_000_000,
                    "turnover": 1.0 + (i % 5) * 0.1,
                    "industry": industry,
                    "up_limit": c * 1.10,
                    "down_limit": c * 0.90,
                    "listed_days": 200,
                    "is_suspended": False,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    panel_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    panel = load_panel_csv(panel_path) if panel_path else make_demo_panel()
    asof = pd.Timestamp(panel["date"].max())
    selector = TriStateSelector(SelectorConfig())
    out = selector.rebalance(prices=panel, asof=asof, prev_weights=pd.Series(dtype=float))
    print("state:", out.state.value)
    print("confidence:", round(out.confidence, 4))
    print(out.target_weights.rename("weight").to_frame())


if __name__ == "__main__":
    main()
