import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))

from portfolio_utils import apply_supplemental_overlay  # noqa: E402


def _row(
    stock_id,
    score,
    *,
    lgb=0.0,
    ret1=0.0,
    ret5=0.0,
    ret20=-0.04,
    sigma20=0.02,
    amp20=0.06,
    drawdown20=0.03,
    downside_beta60=0.2,
    amount=1_000_000_000,
):
    return {
        "stock_id": stock_id,
        "score": score,
        "lgb": lgb,
        "ret1": ret1,
        "ret5": ret5,
        "ret20": ret20,
        "sigma20": sigma20,
        "amp20": amp20,
        "max_drawdown20": drawdown20,
        "downside_beta60": downside_beta60,
        "median_amount20": amount,
    }


def _cfg():
    return {
        "supplemental_overlay_enabled": True,
        "supplemental_overlay_shadow_only": False,
        "supplemental_overlay_priority": [],
        "stress_chaser_veto_enabled": True,
        "stress_chaser_median_ret20_max": 0.0,
        "stress_chaser_breadth20_max": 0.50,
        "stress_chaser_median_sigma20_min": 0.018,
        "stress_chaser_dispersion20_min": 0.10,
        "stress_panic_ret1_max": -0.05,
        "stress_panic_downside_beta_min": 1.50,
        "stress_hot_ret5_min": 0.04,
        "stress_hot_ret20_max": 0.25,
        "stress_hot_amp20_min": 0.12,
        "stress_hot_downside_beta_min": 0.80,
    }


def test_stress_chaser_veto_replaces_single_panic_beta_name():
    rows = [
        _row("000001", 1.00, ret20=-0.08, amount=2_000_000_000),
        _row("000002", 0.90, ret20=-0.06, amount=1_900_000_000),
        _row(
            "000003",
            0.80,
            ret1=-0.061,
            ret5=-0.02,
            ret20=0.02,
            sigma20=0.035,
            amp20=0.18,
            downside_beta60=2.1,
            amount=1_800_000_000,
        ),
        _row("000004", 0.70, ret20=-0.10, amount=1_700_000_000),
        _row("000005", 0.60, ret20=-0.03, amount=1_600_000_000),
        _row("000006", 0.55, lgb=0.8, ret1=0.01, ret20=0.08, sigma20=0.006, amp20=0.03, amount=3_000_000_000),
        _row("000007", 0.30, ret20=0.10, sigma20=0.025, amount=900_000_000),
        _row("000008", 0.20, ret20=-0.12, sigma20=0.022, amount=800_000_000),
    ]
    score_df = pd.DataFrame(rows)
    selected = score_df.iloc[:5].copy()

    out = apply_supplemental_overlay(score_df, selected, _cfg())
    ids = out.head(5)["stock_id"].astype(str).tolist()

    assert "000003" not in ids
    assert "000006" in ids
    assert any(info.get("overlay") == "stress_chaser_veto" and info.get("accepted") for info in out.attrs["supplemental_overlay_info"])


def test_stress_chaser_veto_stays_idle_outside_stress_state():
    rows = [
        _row("000001", 1.00, ret20=0.08, amount=2_000_000_000),
        _row("000002", 0.90, ret20=0.06, amount=1_900_000_000),
        _row(
            "000003",
            0.80,
            ret1=-0.061,
            ret5=-0.02,
            ret20=0.02,
            sigma20=0.035,
            amp20=0.18,
            downside_beta60=2.1,
            amount=1_800_000_000,
        ),
        _row("000004", 0.70, ret20=0.04, amount=1_700_000_000),
        _row("000005", 0.60, ret20=0.03, amount=1_600_000_000),
        _row("000006", 0.55, lgb=0.8, ret1=0.01, ret20=0.08, sigma20=0.006, amp20=0.03, amount=3_000_000_000),
    ]
    score_df = pd.DataFrame(rows)
    selected = score_df.iloc[:5].copy()

    out = apply_supplemental_overlay(score_df, selected, _cfg())
    ids = out.head(5)["stock_id"].astype(str).tolist()

    assert ids == ["000001", "000002", "000003", "000004", "000005"]
    assert not any(info.get("accepted") for info in out.attrs["supplemental_overlay_info"])
