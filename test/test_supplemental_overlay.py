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
    intraday_ret=0.0,
    sigma20=0.02,
    amp20=0.06,
    drawdown20=0.03,
    downside_beta60=0.2,
    amount=1_000_000_000,
    grr_final_score=None,
    max_ret20_raw=0.08,
    max_high_jump20=0.10,
):
    row = {
        "stock_id": stock_id,
        "score": score,
        "lgb": lgb,
        "ret1": ret1,
        "ret5": ret5,
        "ret20": ret20,
        "intraday_ret": intraday_ret,
        "sigma20": sigma20,
        "amp20": amp20,
        "max_drawdown20": drawdown20,
        "downside_beta60": downside_beta60,
        "median_amount20": amount,
        "max_ret20_raw": max_ret20_raw,
        "max_high_jump20": max_high_jump20,
    }
    if grr_final_score is not None:
        row["grr_final_score"] = grr_final_score
    return row


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


def test_conditional_anti_lottery_replaces_low_score_non_beta_target():
    cfg = _cfg()
    cfg.update(
        {
            "stress_chaser_veto_enabled": False,
            "anti_lottery_overlay_enabled": True,
            "anti_lottery_rank_cap": 1,
            "anti_lottery_dbeta_guard_max": 1.35,
            "max_total_swaps": 2,
        }
    )
    rows = [
        _row("000001", 1.00, grr_final_score=1.00, ret20=0.04, max_ret20_raw=0.12, max_high_jump20=0.14),
        _row("000002", 0.90, grr_final_score=0.90, ret20=0.03, max_ret20_raw=0.11, max_high_jump20=0.13),
        _row("000003", 0.80, grr_final_score=0.80, ret20=0.02, max_ret20_raw=0.10, max_high_jump20=0.12),
        _row("000004", 0.70, grr_final_score=0.70, ret20=0.01, max_ret20_raw=0.09, max_high_jump20=0.11),
        _row("000005", 0.60, grr_final_score=0.10, ret20=0.00, downside_beta60=1.0, max_ret20_raw=0.08, max_high_jump20=0.10),
        _row("000006", 0.55, grr_final_score=0.95, ret5=0.01, ret20=0.08, sigma20=0.015, amp20=0.05, downside_beta60=0.4, amount=3_000_000_000, max_ret20_raw=0.00, max_high_jump20=0.01),
    ]
    score_df = pd.DataFrame(rows)
    selected = score_df.iloc[:5].copy()

    out = apply_supplemental_overlay(score_df, selected, cfg)
    ids = out.head(5)["stock_id"].astype(str).tolist()

    assert "000005" not in ids
    assert "000006" in ids
    assert any(info.get("overlay") == "conditional_anti_lottery_dbeta_guard" and info.get("accepted") for info in out.attrs["supplemental_overlay_info"])


def test_conditional_anti_lottery_protects_high_downside_beta_target():
    cfg = _cfg()
    cfg.update(
        {
            "stress_chaser_veto_enabled": False,
            "anti_lottery_overlay_enabled": True,
            "anti_lottery_rank_cap": 1,
            "anti_lottery_dbeta_guard_max": 1.35,
            "max_total_swaps": 2,
        }
    )
    rows = [
        _row("000001", 1.00, grr_final_score=1.00, ret20=0.04, max_ret20_raw=0.12, max_high_jump20=0.14),
        _row("000002", 0.90, grr_final_score=0.90, ret20=0.03, max_ret20_raw=0.11, max_high_jump20=0.13),
        _row("000003", 0.80, grr_final_score=0.80, ret20=0.02, max_ret20_raw=0.10, max_high_jump20=0.12),
        _row("000004", 0.70, grr_final_score=0.70, ret20=0.01, max_ret20_raw=0.09, max_high_jump20=0.11),
        _row("000005", 0.60, grr_final_score=0.10, ret20=0.00, downside_beta60=1.6, max_ret20_raw=0.08, max_high_jump20=0.10),
        _row("000006", 0.55, grr_final_score=0.95, ret5=0.01, ret20=0.08, sigma20=0.015, amp20=0.05, downside_beta60=0.4, amount=3_000_000_000, max_ret20_raw=0.00, max_high_jump20=0.01),
    ]
    score_df = pd.DataFrame(rows)
    selected = score_df.iloc[:5].copy()

    out = apply_supplemental_overlay(score_df, selected, cfg)
    ids = out.head(5)["stock_id"].astype(str).tolist()

    assert ids == ["000001", "000002", "000003", "000004", "000005"]
    assert any(
        info.get("overlay") == "conditional_anti_lottery_dbeta_guard"
        and info.get("blocked_reason") == "target_dbeta_guard"
        for info in out.attrs["supplemental_overlay_info"]
    )


def test_pullback_stable_booster_replaces_highest_risk_name():
    cfg = _cfg()
    cfg.update(
        {
            "stress_chaser_veto_enabled": False,
            "anti_lottery_overlay_enabled": False,
            "supplemental_overlay_priority": ["pullback_stable_booster"],
            "pullback_stable_rank_cap": 1,
            "pullback_stable_lgb_rank_min": 0.50,
        }
    )
    rows = [
        _row("000001", 1.00, ret20=0.04, sigma20=0.020, amp20=0.06, downside_beta60=0.5),
        _row("000002", 0.90, ret20=0.03, sigma20=0.021, amp20=0.07, downside_beta60=0.6),
        _row("000003", 0.80, ret20=0.02, sigma20=0.024, amp20=0.08, downside_beta60=0.7),
        _row("000004", 0.70, ret20=0.01, sigma20=0.022, amp20=0.08, downside_beta60=0.6),
        _row("000005", 0.60, ret20=0.05, sigma20=0.060, amp20=0.16, downside_beta60=1.2),
        _row(
            "000006",
            0.55,
            lgb=1.2,
            ret5=-0.01,
            ret20=0.12,
            intraday_ret=0.025,
            sigma20=0.018,
            amp20=0.07,
            drawdown20=0.05,
            downside_beta60=0.4,
            amount=3_000_000_000,
        ),
    ]
    score_df = pd.DataFrame(rows)
    selected = score_df.iloc[:5].copy()

    out = apply_supplemental_overlay(score_df, selected, cfg)
    ids = out.head(5)["stock_id"].astype(str).tolist()

    assert "000005" not in ids
    assert "000006" in ids
    assert any(info.get("overlay") == "pullback_stable_booster" and info.get("accepted") for info in out.attrs["supplemental_overlay_info"])


def test_ret5_guarded_booster_replaces_weak_short_term_names():
    cfg = _cfg()
    cfg.update(
        {
            "stress_chaser_veto_enabled": False,
            "anti_lottery_overlay_enabled": True,
            "supplemental_overlay_priority": ["ret5_guarded_booster"],
            "max_total_swaps": 3,
            "ret5_guarded_max_swaps": 3,
            "ret5_guarded_rank_cap": 3,
        }
    )
    rows = [
        _row("000001", 1.00, ret5=0.03, ret20=0.12, sigma20=0.020, amp20=0.06, downside_beta60=0.5),
        _row("000002", 0.90, ret5=-0.01, ret20=0.03, sigma20=0.021, amp20=0.07, downside_beta60=0.6),
        _row("000003", 0.80, ret5=0.10, ret20=0.20, sigma20=0.024, amp20=0.08, downside_beta60=0.7),
        _row("000004", 0.70, ret5=-0.03, ret20=0.01, sigma20=0.022, amp20=0.08, downside_beta60=0.6),
        _row("000005", 0.60, ret5=-0.04, ret20=-0.02, sigma20=0.026, amp20=0.09, downside_beta60=0.7),
        _row("000006", 0.50, lgb=1.4, ret5=0.18, ret20=0.24, sigma20=0.030, amp20=0.25, drawdown20=0.08, downside_beta60=1.3, amount=3_000_000_000),
        _row("000007", 0.40, lgb=1.1, ret5=0.16, ret20=0.26, sigma20=0.025, amp20=0.22, drawdown20=0.04, downside_beta60=1.2, amount=2_800_000_000),
        _row("000008", 0.30, lgb=0.9, ret5=0.12, ret20=0.18, sigma20=0.021, amp20=0.18, drawdown20=0.03, downside_beta60=1.1, amount=2_600_000_000),
        _row("000009", 0.20, lgb=1.5, ret5=0.20, ret20=0.70, sigma20=0.030, amp20=0.40, drawdown20=0.02, downside_beta60=1.1, amount=3_200_000_000),
    ]
    score_df = pd.DataFrame(rows)
    selected = score_df.iloc[:5].copy()

    out = apply_supplemental_overlay(score_df, selected, cfg)
    ids = out.head(5)["stock_id"].astype(str).tolist()

    assert {"000006", "000007", "000008"}.issubset(set(ids))
    assert "000005" not in ids
    assert "000004" not in ids
    assert "000002" not in ids
    assert "000009" not in ids
    assert any(
        info.get("overlay") == "ret5_guarded_booster"
        and info.get("accepted")
        and info.get("swap_count") == 3
        for info in out.attrs["supplemental_overlay_info"]
    )


def test_cooldown_minrisk_repair_preempts_ret5_chase_in_cold_pullback():
    cfg = _cfg()
    cfg.update(
        {
            "stress_chaser_veto_enabled": False,
            "supplemental_overlay_priority": ["cooldown_minrisk_repair", "ret5_guarded_booster"],
            "cooldown_minrisk_enabled": True,
        }
    )
    rows = [
        _row("000001", 1.00, ret1=-0.02, ret5=-0.03, ret20=0.24, sigma20=0.070, amp20=0.40, downside_beta60=1.9, amount=800_000_000),
        _row("000002", 0.95, ret1=-0.01, ret5=-0.02, ret20=0.12, sigma20=0.065, amp20=0.36, downside_beta60=1.8, amount=850_000_000),
        _row("000003", 0.90, ret1=-0.02, ret5=-0.04, ret20=-0.10, sigma20=0.060, amp20=0.34, downside_beta60=1.7, amount=900_000_000),
        _row("000004", 0.85, ret1=0.00, ret5=-0.01, ret20=-0.07, sigma20=0.058, amp20=0.32, downside_beta60=1.6, amount=950_000_000),
        _row("000005", 0.80, ret1=0.00, ret5=0.01, ret20=0.01, sigma20=0.055, amp20=0.31, downside_beta60=1.5, amount=1_000_000_000),
        _row("000006", 0.55, lgb=1.5, ret1=0.01, ret5=0.02, ret20=0.005, sigma20=0.012, amp20=0.05, downside_beta60=0.3, amount=4_000_000_000),
        _row("000007", 0.50, lgb=1.4, ret1=0.01, ret5=0.01, ret20=-0.004, sigma20=0.013, amp20=0.05, downside_beta60=0.3, amount=3_800_000_000),
        _row("000008", 0.45, lgb=1.3, ret1=0.00, ret5=-0.01, ret20=-0.006, sigma20=0.014, amp20=0.05, downside_beta60=0.4, amount=3_600_000_000),
        _row("000009", 0.40, lgb=1.2, ret1=0.01, ret5=0.00, ret20=0.002, sigma20=0.015, amp20=0.06, downside_beta60=0.4, amount=3_400_000_000),
        _row("000010", 0.35, lgb=1.1, ret1=0.01, ret5=-0.01, ret20=-0.003, sigma20=0.016, amp20=0.06, downside_beta60=0.5, amount=3_200_000_000),
        _row("000011", 0.30, lgb=1.8, ret1=0.02, ret5=0.18, ret20=0.18, sigma20=0.025, amp20=0.20, downside_beta60=1.0, amount=5_000_000_000),
    ]
    score_df = pd.DataFrame(rows)
    selected = score_df.iloc[:5].copy()

    out = apply_supplemental_overlay(score_df, selected, cfg)
    ids = out.head(5)["stock_id"].astype(str).tolist()

    assert not set(ids) & {"000001", "000002", "000003", "000004", "000005"}
    assert {"000006", "000007", "000008", "000009"}.issubset(set(ids))
    info = out.attrs["supplemental_overlay_info"]
    assert any(item.get("overlay") == "cooldown_minrisk_repair" and item.get("accepted") for item in info)
    assert not any(item.get("overlay") == "ret5_guarded_booster" and item.get("accepted") for item in info)


def test_growth_rrf_repair_uses_hybrid_when_risk_appetite_improves():
    cfg = _cfg()
    cfg.update(
        {
            "stress_chaser_veto_enabled": False,
            "supplemental_overlay_priority": ["growth_rrf_repair", "cooldown_minrisk_repair"],
            "growth_rrf_repair_enabled": True,
        }
    )
    rows = [
        _row("000001", 0.10, lgb=0.1, ret1=0.012, ret5=0.02, ret20=0.02),
        _row("000002", 0.09, lgb=0.1, ret1=0.012, ret5=0.02, ret20=0.02),
        _row("000003", 0.08, lgb=0.1, ret1=0.012, ret5=0.02, ret20=0.02),
        _row("000004", 0.07, lgb=0.1, ret1=0.012, ret5=0.02, ret20=0.02),
        _row("000005", 0.06, lgb=0.1, ret1=0.012, ret5=0.02, ret20=0.02),
        _row("300006", 1.00, lgb=1.4, ret1=0.015, ret5=0.03, ret20=0.04, amount=3_000_000_000),
        _row("002007", 0.95, lgb=1.3, ret1=0.015, ret5=0.03, ret20=0.04, amount=2_900_000_000),
        _row("688008", 0.90, lgb=1.2, ret1=0.015, ret5=0.03, ret20=0.04, amount=2_800_000_000),
        _row("300009", 0.85, lgb=1.1, ret1=0.015, ret5=0.03, ret20=0.04, amount=2_700_000_000),
        _row("002010", 0.80, lgb=1.0, ret1=0.015, ret5=0.03, ret20=0.04, amount=2_600_000_000),
    ]
    for i, row in enumerate(rows):
        row["transformer"] = 0.1 if i < 5 else 1.5 - i * 0.02
    score_df = pd.DataFrame(rows)
    selected = score_df.iloc[:5].copy()

    out = apply_supplemental_overlay(score_df, selected, cfg)
    ids = out.head(5)["stock_id"].astype(str).tolist()

    assert set(ids) == {"300006", "002007", "688008", "300009", "002010"}
    info = out.attrs["supplemental_overlay_info"]
    assert any(item.get("overlay") == "growth_rrf_repair" and item.get("accepted") for item in info)
    assert not any(item.get("overlay") == "cooldown_minrisk_repair" and item.get("accepted") for item in info)


def test_deep_rebound_repair_full_replaces_and_skips_later_overlays():
    cfg = _cfg()
    cfg.update(
        {
            "supplemental_overlay_priority": ["deep_rebound_repair", "ret5_guarded_booster"],
            "deep_rebound_median_ret20_max": -0.055,
            "deep_rebound_breadth20_max": 0.30,
            "deep_rebound_median_amp20_min": 0.16,
            "anti_lottery_overlay_enabled": True,
        }
    )
    rows = [
        _row("000001", 1.0, ret20=-0.08, amp20=0.20, sigma20=0.02),
        _row("000002", 0.9, ret20=-0.09, amp20=0.19, sigma20=0.02),
        _row("000003", 0.8, ret20=-0.07, amp20=0.18, sigma20=0.02),
        _row("000004", 0.7, ret20=-0.10, amp20=0.21, sigma20=0.02),
        _row("000005", 0.6, ret20=-0.06, amp20=0.17, sigma20=0.02),
        _row("000006", 0.5, lgb=0.4, ret20=-0.04, amp20=0.22, sigma20=0.02),
        _row("000007", 0.4, lgb=0.3, ret20=-0.05, amp20=0.20, sigma20=0.02),
        _row("000008", 0.3, lgb=0.2, ret20=-0.06, amp20=0.19, sigma20=0.02),
        _row("000009", 0.2, lgb=0.1, ret20=-0.07, amp20=0.18, sigma20=0.02),
        _row("000010", 0.1, lgb=0.0, ret20=-0.08, amp20=0.17, sigma20=0.02),
    ]
    for i, row in enumerate(rows):
        row["transformer"] = i + 1
    score_df = pd.DataFrame(rows)
    selected = score_df.iloc[:5].copy()

    out = apply_supplemental_overlay(score_df, selected, cfg)
    ids = out.head(5)["stock_id"].astype(str).tolist()

    assert ids == ["000010", "000009", "000008", "000007", "000006"]
    info = out.attrs["supplemental_overlay_info"]
    assert any(item.get("overlay") == "deep_rebound_repair" and item.get("accepted") and item.get("swap_count") == 5 for item in info)
    assert not any(item.get("overlay") == "stress_chaser_veto" for item in info)
    assert not any(item.get("overlay") == "conditional_anti_lottery_dbeta_guard" for item in info)
