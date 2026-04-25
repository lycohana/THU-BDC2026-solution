import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))

from predict import (  # noqa: E402
    _combo_search_union_pick,
    _safe_union_2slot_candidates,
    choose_selector_branch,
)


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
        "high_vol_flag": False,
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


class ComboLcbSelectorTest(unittest.TestCase):
    def test_combo_search_replaces_single_risky_rank_winner(self):
        rows = [
            {
                "stock_id": "000001",
                "alpha_lcb": 1.00,
                "alpha_score": 1.00,
                "rerank_score": 1.00,
                "stable_fill_score": 0.80,
                "risk_score": 0.05,
                "effective_risk_score": 0.05,
                "rank_disagreement": 0.05,
                "consensus_count": 3,
                "stable_candidate": True,
                "tail_risk_flag": False,
                "high_vol_flag": False,
                "very_tail_flag": False,
                "very_high_vol_flag": False,
                "alpha_exception": True,
                "branch_only_alpha_flag": False,
            },
            {
                "stock_id": "000002",
                "alpha_lcb": 0.98,
                "alpha_score": 0.98,
                "rerank_score": 0.98,
                "stable_fill_score": 0.78,
                "risk_score": 0.05,
                "effective_risk_score": 0.05,
                "rank_disagreement": 0.05,
                "consensus_count": 3,
                "stable_candidate": True,
                "tail_risk_flag": False,
                "high_vol_flag": False,
                "very_tail_flag": False,
                "very_high_vol_flag": False,
                "alpha_exception": True,
                "branch_only_alpha_flag": False,
            },
            {
                "stock_id": "000003",
                "alpha_lcb": 0.95,
                "alpha_score": 0.95,
                "rerank_score": 0.95,
                "stable_fill_score": 0.75,
                "risk_score": 0.05,
                "effective_risk_score": 0.05,
                "rank_disagreement": 0.05,
                "consensus_count": 3,
                "stable_candidate": True,
                "tail_risk_flag": False,
                "high_vol_flag": False,
                "very_tail_flag": False,
                "very_high_vol_flag": False,
                "alpha_exception": False,
                "branch_only_alpha_flag": False,
            },
            {
                "stock_id": "000004",
                "alpha_lcb": 0.92,
                "alpha_score": 0.92,
                "rerank_score": 0.92,
                "stable_fill_score": 0.72,
                "risk_score": 0.05,
                "effective_risk_score": 0.05,
                "rank_disagreement": 0.05,
                "consensus_count": 2,
                "stable_candidate": True,
                "tail_risk_flag": False,
                "high_vol_flag": False,
                "very_tail_flag": False,
                "very_high_vol_flag": False,
                "alpha_exception": False,
                "branch_only_alpha_flag": False,
            },
            {
                "stock_id": "000005",
                "alpha_lcb": 0.91,
                "alpha_score": 0.91,
                "rerank_score": 0.91,
                "stable_fill_score": 0.20,
                "risk_score": 0.95,
                "effective_risk_score": 0.95,
                "rank_disagreement": 0.35,
                "consensus_count": 1,
                "stable_candidate": False,
                "tail_risk_flag": True,
                "high_vol_flag": True,
                "very_tail_flag": True,
                "very_high_vol_flag": True,
                "alpha_exception": False,
                "branch_only_alpha_flag": True,
            },
            {
                "stock_id": "000006",
                "alpha_lcb": 0.70,
                "alpha_score": 0.70,
                "rerank_score": 0.70,
                "stable_fill_score": 0.88,
                "risk_score": 0.03,
                "effective_risk_score": 0.03,
                "rank_disagreement": 0.04,
                "consensus_count": 3,
                "stable_candidate": True,
                "tail_risk_flag": False,
                "high_vol_flag": False,
                "very_tail_flag": False,
                "very_high_vol_flag": False,
                "alpha_exception": False,
                "branch_only_alpha_flag": False,
            },
        ]
        cfg = {
            "union_rerank": {
                "combo_search": {
                    "enabled": True,
                    "topn": 6,
                    "constraints": {
                        "risk_off": {
                            "min_stable_count": 4,
                            "min_consensus_count": 4,
                            "max_tail_risk_count": 0,
                            "max_high_vol_count": 0,
                            "max_very_tail_count": 0,
                            "max_very_high_vol_count": 0,
                            "max_branch_only_alpha_count": 0,
                        }
                    },
                },
                "risk_budget": {
                    "risk_off": {
                        "max_tail_risk_count": 0,
                        "max_high_vol_count": 0,
                        "max_very_high_vol_count": 0,
                        "max_very_tail_count": 0,
                        "max_alpha_exception_count": 2,
                        "max_branch_only_alpha_count": 0,
                    }
                },
            }
        }

        selected, info = _combo_search_union_pick(pd.DataFrame(rows), cfg, "risk_off")
        selected_ids = {row["stock_id"] for row in selected}

        self.assertGreater(info["combo_search_feasible"], 0)
        self.assertNotIn("000005", selected_ids)
        self.assertIn("000006", selected_ids)

    def test_safe_union_2slot_builds_two_alpha_three_stable(self):
        rows = [make_row(f"0000{i:02d}", 1.00 - i * 0.05) for i in range(10)]
        cfg = {
            "branches": {
                "safe_union_2slot": {
                    "score_col": "safe_union_2slot_score",
                    "filter": "safe_union_2slot",
                    "exposure": 1.0,
                },
                "conservative_softrisk_v2": {
                    "score_col": "score_conservative_softrisk_v2",
                    "filter": "liquidity_q05",
                },
                "lgb_only_guarded": {
                    "score_col": "score_lgb_only",
                    "filter": "stable",
                    "liquidity_quantile": 0.10,
                    "sigma_quantile": 0.85,
                },
                "balanced_guarded": {
                    "score_col": "score_balanced",
                    "filter": "stable",
                    "liquidity_quantile": 0.10,
                    "sigma_quantile": 0.85,
                },
                "defensive_v2_strict": {
                    "score_col": "score_defensive_v2",
                    "filter": "legal_minrisk",
                },
                "legal_minrisk_hardened": {
                    "score_col": "score_legal_minrisk",
                    "filter": "legal_minrisk_hardened",
                },
            },
            "gated_union_rerank": {
                "safe_union_2slot": {
                    "enabled": True,
                    "min_clean_alpha_count": 2,
                    "min_clean_alpha_lcb_z": 1.00,
                    "max_alpha_slots": 2,
                    "min_stable_slots": 3,
                    "max_tail_risk_count": 1,
                    "max_high_vol_count": 1,
                    "max_very_tail_count": 0,
                    "max_very_high_vol_count": 0,
                    "max_branch_only_alpha_count": 0,
                }
            },
            "union_rerank": {
                "enabled": True,
                "candidate_pool": {
                    "conservative_softrisk_v2": 10,
                    "lgb_only_guarded": 10,
                    "balanced_guarded": 10,
                    "defensive_v2_strict": 10,
                    "legal_minrisk_hardened": 10,
                },
                "alpha_lcb": {"enabled": True, "base_penalty": 0.0},
                "combo_search": {
                    "enabled": True,
                    "topn": 10,
                    "constraints": {
                        "default": {
                            "min_stable_count": 2,
                            "min_consensus_count": 2,
                            "max_tail_risk_count": 1,
                            "max_high_vol_count": 1,
                            "max_very_tail_count": 0,
                            "max_very_high_vol_count": 0,
                            "max_branch_only_alpha_count": 0,
                        }
                    },
                },
                "risk_budget": {},
                "risk_lambda": {"risk_on_strict": 0.0},
            },
        }

        selected, _ = _safe_union_2slot_candidates(pd.DataFrame(rows), cfg, regime="risk_on_strict")

        self.assertEqual(len(selected), 5)
        self.assertEqual(selected.attrs["safe_union_info"]["safe_union_reason"], "enabled_safe_union_2slot")
        self.assertLessEqual(int(selected["alpha_exception"].astype(bool).sum()), 2)
        self.assertGreaterEqual(int(selected["stable_candidate"].astype(bool).sum()), 3)


if __name__ == "__main__":
    unittest.main()
