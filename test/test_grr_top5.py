import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))

from labels import add_label_o2o_week, build_aux_horizon_labels, build_quality_label, build_relevance_bins  # noqa: E402
from exp009_oof_builder import build_fold_specs  # noqa: E402
from lgb_branch import build_lgb_rank_data  # noqa: E402
from reranker import apply_grr_top5, compute_market_crash_state, hedge_update, reciprocal_rank_fusion, union_topk_candidates  # noqa: E402


def make_price_panel():
    rows = []
    for stock_idx in range(8):
        stock = f"{stock_idx:06d}"
        for day in range(8):
            price = 10.0 + stock_idx + day * (0.1 + stock_idx * 0.02)
            rows.append(
                {
                    "股票代码": stock,
                    "日期": f"2024-01-{day + 1:02d}",
                    "开盘": price,
                    "收盘": price * 1.01,
                    "最高": price * 1.03,
                    "最低": price * 0.98,
                    "feat": stock_idx + day / 10.0,
                }
            )
    return pd.DataFrame(rows)


class GRRTop5Test(unittest.TestCase):
    def test_quality_and_relevance_labels_are_integer_bins(self):
        df = add_label_o2o_week(make_price_panel(), horizon=5, label_col="raw5")
        df = build_quality_label(df, raw_label_col="raw5", lambda_vol=0.1, lambda_dd=0.1)
        df = build_relevance_bins(df, n_bins=5)
        df = build_aux_horizon_labels(df, horizons=[1, 3])
        clean = df.dropna(subset=["quality5", "relevance5"])

        self.assertTrue(set(clean["relevance5"].astype(int).unique()).issubset({0, 1, 2, 3, 4}))
        self.assertIn("aux1", clean.columns)
        self.assertIn("aux3", clean.columns)

        X, y, group = build_lgb_rank_data(clean.rename(columns={"feat": "feature"}), ["feature"])
        self.assertEqual(len(X), len(y))
        self.assertEqual(int(y.max()), int(clean["relevance5"].max()))
        self.assertGreater(len(group), 0)

    def test_rrf_and_grr_pool_are_scale_robust(self):
        score_df = pd.DataFrame(
            {
                "stock_id": [f"{i:06d}" for i in range(10)],
                "lgb": np.linspace(1000, 1, 10),
                "transformer": np.linspace(0.1, 1.0, 10),
                "score": np.linspace(0.5, 0.2, 10),
                "sigma20": np.linspace(0.01, 0.05, 10),
                "amp20": np.linspace(0.02, 0.10, 10),
                "max_drawdown20": np.linspace(0.01, 0.20, 10),
                "ret5": np.linspace(-0.01, 0.02, 10),
            }
        )
        pool = union_topk_candidates(score_df, ["lgb", "transformer"], candidate_k=3)
        self.assertGreaterEqual(len(pool), 5)

        fused = reciprocal_rank_fusion(pool, ["lgb", "transformer"])
        self.assertIn("grr_score", fused.rename(columns={"rrf_score": "grr_score"}).columns)

        out = apply_grr_top5(score_df, {"grr_top5": {"enabled": True, "candidate_k": 3}})
        self.assertIn("grr_final_score", out.columns)
        self.assertTrue(np.isfinite(out["score"]).all())

    def test_hedge_update_penalizes_high_loss_expert(self):
        updated = hedge_update([0.5, 0.5], [0.0, 2.0], eta=0.5)
        self.assertGreater(updated[0], updated[1])
        self.assertAlmostEqual(float(updated.sum()), 1.0)

    def test_oof_fold_specs_apply_trading_day_purge(self):
        dates = pd.bdate_range("2024-01-01", periods=180)
        specs = build_fold_specs(dates, n_folds=1, fold_window_months=1, gap_months=0, purge_trading_days=5)
        spec = specs[0]
        date_list = list(pd.Index(dates))
        train_end_pos = date_list.index(pd.Timestamp(spec.train_end))
        val_start_pos = date_list.index(pd.Timestamp(spec.val_start))
        self.assertLessEqual(train_end_pos, val_start_pos - 5)

    def test_tail_guard_flags_fragile_top5(self):
        df = pd.DataFrame(
            {
                "stock_id": [f"{i:06d}" for i in range(12)],
                "score": np.r_[np.linspace(3.0, 2.0, 5), np.linspace(1.0, 0.0, 7)],
                "lgb": np.r_[np.linspace(3.0, 2.0, 5), np.linspace(1.0, 0.0, 7)],
                "transformer": np.r_[np.linspace(3.0, 2.0, 5), np.linspace(0.0, 1.0, 7)],
                "sigma20": np.r_[np.linspace(0.08, 0.06, 5), np.linspace(0.01, 0.02, 7)],
                "amp20": np.r_[np.linspace(0.50, 0.35, 5), np.linspace(0.05, 0.12, 7)],
                "max_drawdown20": np.r_[np.linspace(0.22, 0.16, 5), np.linspace(0.02, 0.08, 7)],
                "ret1": np.r_[np.full(5, -0.03), np.linspace(-0.01, 0.01, 7)],
                "ret5": np.r_[np.linspace(0.15, 0.08, 5), np.linspace(-0.01, 0.02, 7)],
                "median_amount20": np.linspace(1.0, 2.0, 12),
            }
        )
        state = compute_market_crash_state(df, {"high_risk_top_count_trigger": 2}, score_col="score")
        self.assertTrue(state["crash_mode"])
        out = apply_grr_top5(
            df,
            {
                "grr_top5": {
                    "enabled": True,
                    "tail_guard_enabled": True,
                    "crash_guard_enabled": True,
                    "high_risk_chaser_veto": True,
                    "candidate_k": 6,
                    "expert_cols": ["lgb", "transformer", "score"],
                    "tail_guard": {"high_risk_top_count_trigger": 2},
                }
            },
        )
        self.assertIn("grr_tail_guard_triggered", out.columns)
        self.assertTrue(out["grr_tail_guard_triggered"].any())


if __name__ == "__main__":
    unittest.main()
