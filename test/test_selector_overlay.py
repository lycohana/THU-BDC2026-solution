import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))

from predict import choose_selector_branch  # noqa: E402


def make_row(stock_id, score, *, tail=False, reversal=False, disagreement=0.10, ret1=0.60, ret5=0.60):
    return {
        "stock_id": stock_id,
        "score": score,
        "score_lgb_only": score,
        "score_balanced": score,
        "score_conservative_softrisk_v2": score,
        "score_defensive_v2": score,
        "score_legal_minrisk": score,
        "lgb_norm": score,
        "tf_norm": score,
        "rank_disagreement": disagreement,
        "disagreement": disagreement,
        "liq_rank": 0.50,
        "sigma_rank": 0.40,
        "amp_rank": 0.40,
        "beta60_rank": 0.40,
        "downside_beta60_rank": 0.40,
        "max_drawdown20_rank": 0.40,
        "tail_risk_score": 0.40,
        "uncertainty_score": 0.20,
        "ret1_rank": ret1,
        "ret5_rank": ret5,
        "tail_risk_flag": tail,
        "reversal_flag": reversal,
        "extreme_momo_flag": False,
        "ret1": 0.01,
        "ret5": 0.02,
        "median_amount20": 1.0 + score,
        "sigma20": 0.02,
        "amp20": 0.10,
    }


def make_selector_cfg():
    return {
        "fallback_branch": "defensive_v2_strict",
        "emergency_fallback_branch": "legal_minrisk_hardened",
        "branches": {
            "balanced_guarded": {
                "score_col": "score_balanced",
                "filter": "stable",
                "liquidity_quantile": 0.10,
                "sigma_quantile": 0.85,
                "exposure": 1.0,
            },
            "defensive_v2_strict": {
                "score_col": "score_defensive_v2",
                "filter": "legal_minrisk",
                "exposure": 1.0,
            },
            "legal_minrisk_hardened": {
                "score_col": "score_legal_minrisk",
                "filter": "legal_minrisk_hardened",
                "exposure": 1.0,
            },
        },
        "regime_branch_order": {
            "risk_on_strict": ["balanced_guarded"],
            "neutral_positive": ["balanced_guarded"],
            "mixed_defensive": [],
            "risk_off": ["defensive_v2_strict"],
        },
    }


class LegalSelectorTest(unittest.TestCase):
    def test_selector_uses_independent_branch_pool(self):
        rows = [make_row(f"00000{i}", 0.50 + i / 100.0) for i in range(1, 8)]
        selected, info = choose_selector_branch(pd.DataFrame(rows), make_selector_cfg())

        self.assertIn(info["chosen_branch"], {"balanced_guarded", "defensive_v2_strict"})
        self.assertEqual(len(selected.head(5)), 5)
        self.assertTrue(all(d["branch"] in {"balanced_guarded", "defensive_v2_strict"} for d in info["diagnostics"]))

    def test_selector_falls_back_to_defensive_branch(self):
        rows = [
            make_row("000001", 0.90, tail=True, disagreement=0.60),
            make_row("000002", 0.88, reversal=True, disagreement=0.60),
            make_row("000003", 0.86, disagreement=0.60),
            make_row("000004", 0.84, disagreement=0.60),
            make_row("000005", 0.82, disagreement=0.60),
            make_row("000006", 0.80, disagreement=0.60),
            make_row("000007", 0.78, disagreement=0.60),
            make_row("000008", 0.76, disagreement=0.60),
        ]
        selected, info = choose_selector_branch(pd.DataFrame(rows), make_selector_cfg())

        self.assertEqual(info["chosen_branch"], "defensive_v2_strict")
        self.assertEqual(len(selected.head(5)), 5)


if __name__ == "__main__":
    unittest.main()
