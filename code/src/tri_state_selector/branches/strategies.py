from __future__ import annotations

import pandas as pd


class BranchStrategy:
    name = "base"

    def score(self, feats: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    @staticmethod
    def _col(feats: pd.DataFrame, name: str) -> pd.Series:
        return feats[name] if name in feats.columns else pd.Series(0.0, index=feats.index)


class TrendStrategy(BranchStrategy):
    name = "trend"

    def score(self, feats: pd.DataFrame) -> pd.Series:
        c = self._col
        return (
            0.30 * c(feats, "RESMOM_60_5")
            + 0.15 * c(feats, "TREND_R2_60")
            + 0.15 * c(feats, "GP_A")
            + 0.10 * c(feats, "CFO_A")
            + 0.10 * c(feats, "ADV20")
            + 0.05 * c(feats, "REV_5")
            - 0.10 * c(feats, "IVOL_60")
            - 0.10 * c(feats, "BLOWOFF")
            - 0.05 * c(feats, "TAIL_RISK")
        ).rename("score")


class NeutralStrategy(BranchStrategy):
    name = "neutral"

    def score(self, feats: pd.DataFrame) -> pd.Series:
        c = self._col
        return (
            0.20 * c(feats, "BP")
            + 0.20 * c(feats, "GP_A")
            + 0.10 * c(feats, "CFO_A")
            - 0.10 * c(feats, "ACCRUAL")
            + 0.15 * c(feats, "LOW_BETA")
            + 0.10 * c(feats, "LOW_VOL")
            + 0.10 * c(feats, "RESMOM_60_5")
            + 0.05 * c(feats, "ADV20")
        ).rename("score")


class DefensiveStrategy(BranchStrategy):
    name = "defensive"

    def score(self, feats: pd.DataFrame) -> pd.Series:
        c = self._col
        return (
            0.25 * c(feats, "LOW_VOL")
            + 0.20 * c(feats, "LOW_BETA")
            + 0.20 * c(feats, "QUALITY")
            + 0.15 * c(feats, "CFO_A")
            - 0.10 * c(feats, "LEV")
            + 0.05 * c(feats, "EP")
            + 0.05 * c(feats, "SHORT_RESILIENCE")
        ).rename("score")
