import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))

from exp009_meta import build_exp009_features, make_relevance_labels  # noqa: E402
from exp009_runtime import apply_exp009_meta  # noqa: E402


def make_toy_oof():
    rows = []
    for day_idx, date in enumerate(["2024-01-02", "2024-01-03"]):
        offset = 100.0 * day_idx
        for i in range(30):
            rows.append(
                {
                    "date": date,
                    "stock_id": f"{i:06d}",
                    "target": (i - 15) / 1000.0,
                    "transformer": offset + i,
                    "lgb": offset + 29 - i,
                    "score": offset + i * 0.5,
                    "sigma20": 0.01 + i / 1000.0,
                    "median_amount20": 1000000 + i,
                    "ret1": (i - 15) / 1000.0,
                    "ret5": (i - 10) / 1000.0,
                    "ret20": (i - 5) / 1000.0,
                    "amp20": 0.05 + i / 1000.0,
                    "beta60": 1.0 + i / 100.0,
                    "downside_beta60": 1.0 + i / 100.0,
                    "max_drawdown20": -0.10 + i / 1000.0,
                }
            )
    return pd.DataFrame(rows)


class DummyRanker:
    def predict(self, X):
        return X.sum(axis=1).to_numpy(dtype=np.float64)


class DummyBadModel:
    def predict_proba(self, X):
        p = np.full(len(X), 0.10, dtype=np.float64)
        return np.column_stack([1.0 - p, p])


class Exp009MetaTest(unittest.TestCase):
    def test_features_are_cross_sectional_by_date(self):
        df = make_toy_oof()
        feat = build_exp009_features(df)
        means = feat.groupby("date")["transformer_z"].mean().abs()
        self.assertTrue((means < 1e-9).all())
        day1_first = feat.loc[feat["date"].eq(pd.Timestamp("2024-01-02")), "transformer_rank_pct"].iloc[0]
        day2_first = feat.loc[feat["date"].eq(pd.Timestamp("2024-01-03")), "transformer_rank_pct"].iloc[0]
        self.assertAlmostEqual(day1_first, 1.0 / 30.0)
        self.assertAlmostEqual(day2_first, 1.0 / 30.0)
        self.assertIn("grr_rrf_score", feat.columns)
        self.assertIn("grr_candidate_flag", feat.columns)
        self.assertTrue(feat["grr_candidate_flag"].between(0.0, 1.0).all())

    def test_relevance_labels_range(self):
        labeled = make_relevance_labels(build_exp009_features(make_toy_oof()))
        self.assertTrue(set(labeled["relevance"].unique()).issubset({0, 1, 2, 3, 4}))
        self.assertIn("bad_1pct", labeled.columns)
        self.assertIn("false_positive", labeled.columns)

    def test_apply_missing_artifacts_returns_original_score(self):
        live = make_toy_oof().drop(columns=["date", "target"]).head(10)
        out = apply_exp009_meta(live, None)
        pd.testing.assert_frame_equal(live, out)

    def test_live_missing_columns_are_filled(self):
        live = make_toy_oof().drop(
            columns=["date", "target", "sigma20", "ret20", "amp20", "beta60", "downside_beta60", "max_drawdown20"]
        ).head(10)
        artifacts = {
            "ranker": DummyRanker(),
            "bad_1pct": DummyBadModel(),
            "bad_2pct": None,
            "feature_cols": ["transformer_z", "lgb_z", "blend_z", "missing_feature_for_test"],
            "meta_config": {
                "lambda_bad_1pct": 0.25,
                "lambda_bad_2pct": 0.45,
                "lambda_uncertainty": 0.15,
                "lambda_disagreement": 0.10,
            },
        }
        out = apply_exp009_meta(live, artifacts)
        self.assertIn("exp009_final_score", out.columns)
        self.assertEqual(len(out), len(live))
        self.assertTrue(np.isfinite(out["exp009_final_score"]).all())


if __name__ == "__main__":
    unittest.main()
