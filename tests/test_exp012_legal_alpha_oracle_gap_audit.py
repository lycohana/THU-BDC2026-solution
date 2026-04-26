from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import exp012_legal_alpha_oracle_gap_audit as audit  # noqa: E402


def _market_csv(path: Path, dates: pd.DatetimeIndex, stock_ids: list[str]) -> None:
    rows = []
    for i, stock_id in enumerate(stock_ids):
        price = 10.0 + i
        for j, date in enumerate(dates):
            price *= 1.0 + 0.001 * (i + 1) + 0.0005 * np.sin(j + i)
            rows.append(
                {
                    "股票代码": stock_id,
                    "日期": date.strftime("%Y-%m-%d"),
                    "开盘": price,
                    "收盘": price,
                    "最高": price * 1.01,
                    "最低": price * 0.99,
                    "成交量": 1_000_000,
                    "成交额": 100_000_000,
                    "振幅": 2.0,
                    "涨跌额": 0.0,
                    "换手率": 1.0,
                    "涨跌幅": 0.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_window(exp008_dir: Path, asof: str, picks: list[str], stock_ids: list[str]) -> None:
    wdir = exp008_dir / asof.replace("-", "")
    wdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"stock_id": picks, "weight": [0.2] * 5}).to_csv(wdir / "result.csv", index=False)
    score_rows = []
    for i, stock_id in enumerate(stock_ids):
        score_rows.append(
            {
                "stock_id": stock_id,
                "score": float(20 - i),
                "score_lgb_only": float(i),
                "score_balanced": float(10 - abs(5 - i)),
                "score_conservative_softrisk_v2": float(15 - i),
                "score_defensive_v2": float(8 - i / 2),
                "score_legal_minrisk": float(12 - i / 3),
                "tail_risk_score": float(i / 20),
                "uncertainty_score": float(i / 30),
            }
        )
    pd.DataFrame(score_rows).to_csv(wdir / "predict_score_df.csv", index=False)
    pd.DataFrame(
        [
            {
                "branch": "independent_union_rerank",
                "available": True,
                "top5": ",".join(picks),
                "filter": "union_rerank",
                "score_col": "rerank_score",
            },
            {
                "branch": "reference_baseline_branch",
                "available": True,
                "top5": ",".join(stock_ids[:5]),
                "filter": "nofilter",
                "score_col": "score_reference_baseline",
            },
        ]
    ).to_csv(wdir / "selector_diagnostics.csv", index=False)


def _exp008_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    stock_ids = [f"{i:06d}" for i in range(1, 13)]
    dates = pd.bdate_range("2026-01-02", periods=35)
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    _market_csv(train, dates[:24], stock_ids)
    _market_csv(test, dates[24:], stock_ids)

    exp008 = tmp_path / "exp008"
    exp008.mkdir()
    anchors = [dates[20].strftime("%Y-%m-%d"), dates[21].strftime("%Y-%m-%d")]
    summary_rows = []
    branch_rows = []
    for n, asof in enumerate(anchors):
        picks = stock_ids[n : n + 5]
        _write_window(exp008, asof, picks, stock_ids)
        summary_rows.append(
            {
                "anchor_date": asof,
                "score_window": "future",
                "selected_score": 0.01 + n * 0.001,
                "chosen_branch": "legal_minrisk_hardened",
                "regime": "risk_off",
                "selected_picks": ",".join(picks),
            }
        )
        branch_rows.extend(
            [
                {
                    "anchor_date": asof,
                    "score_window": "future",
                    "regime": "risk_off",
                    "branch_score_col": "score_lgb_only",
                    "branch": "lgb_only_guarded",
                    "filter": "stable",
                    "score": 0.02 + n * 0.001,
                    "picks": ",".join(stock_ids[5:10]),
                    "rets": "",
                },
                {
                    "anchor_date": asof,
                    "score_window": "future",
                    "regime": "risk_off",
                    "branch_score_col": "score_legal_minrisk",
                    "branch": "legal_minrisk_hardened",
                    "filter": "legal_minrisk_hardened",
                    "score": 0.01 + n * 0.001,
                    "picks": ",".join(picks),
                    "rets": "",
                },
                {
                    "anchor_date": asof,
                    "score_window": "future",
                    "regime": "risk_off",
                    "branch_score_col": "score_reference_baseline",
                    "branch": "reference_baseline_branch",
                    "filter": "nofilter",
                    "score": 0.99,
                    "picks": ",".join(stock_ids[:5]),
                    "rets": "",
                },
            ]
        )
    pd.DataFrame(summary_rows).to_csv(exp008 / "window_summary.csv", index=False)
    pd.DataFrame(branch_rows).to_csv(exp008 / "branch_diagnostics.csv", index=False)

    baseline = tmp_path / "baseline.csv"
    pd.DataFrame(
        {
            "anchor_date": anchors,
            "baseline": [0.015, 0.016],
        }
    ).to_csv(baseline, index=False)
    return train, test, exp008, baseline


def test_exp012_outputs_are_generated(tmp_path: Path) -> None:
    train, test, exp008, baseline = _exp008_fixture(tmp_path)
    out = tmp_path / "out"
    audit.build_outputs(
        train_path=train,
        test_path=test,
        exp008_dir=exp008,
        baseline_offline_path=baseline,
        out_dir=out,
    )

    assert (out / "candidate_pool_matrix.csv").exists()
    assert (out / "oracle_gap_summary.csv").exists()
    assert (out / "ranker_quality.csv").exists()
    assert (out / "window_failure_notes.csv").exists()
    assert (out / "aggregate.json").exists()


def test_baseline_is_offline_reference_only(tmp_path: Path) -> None:
    train, test, exp008, baseline = _exp008_fixture(tmp_path)
    out = tmp_path / "out"
    audit.build_outputs(
        train_path=train,
        test_path=test,
        exp008_dir=exp008,
        baseline_offline_path=baseline,
        out_dir=out,
    )

    oracle = pd.read_csv(out / "oracle_gap_summary.csv")
    matrix = pd.read_csv(out / "candidate_pool_matrix.csv")
    assert "baseline_score_offline_reference" in oracle.columns
    assert "gap_vs_baseline_offline_reference" in oracle.columns
    assert "baseline" not in ",".join(matrix.columns).lower()
    assert not matrix["appeared_in_branches"].str.contains("baseline|reference", case=False).any()


def test_exp012_does_not_overwrite_result_csv_or_old_flow(tmp_path: Path) -> None:
    train, test, exp008, baseline = _exp008_fixture(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    sentinel = output_dir / "result.csv"
    sentinel.write_text("stock_id,weight\n000001,1.0\n", encoding="utf-8")

    out = tmp_path / "out"
    audit.build_outputs(
        train_path=train,
        test_path=test,
        exp008_dir=exp008,
        baseline_offline_path=baseline,
        out_dir=out,
    )

    assert sentinel.read_text(encoding="utf-8") == "stock_id,weight\n000001,1.0\n"
    assert not (out / "result.csv").exists()


def test_aggregate_contains_offline_gaps_and_conclusion(tmp_path: Path) -> None:
    train, test, exp008, baseline = _exp008_fixture(tmp_path)
    out = tmp_path / "out"
    audit.build_outputs(
        train_path=train,
        test_path=test,
        exp008_dir=exp008,
        baseline_offline_path=baseline,
        out_dir=out,
    )

    aggregate = json.loads((out / "aggregate.json").read_text(encoding="utf-8"))
    assert "exp008_vs_baseline_offline_mean_gap" in aggregate
    assert "branch_oracle_vs_baseline_offline_gap" in aggregate
    assert "candidate_oracle_vs_baseline_offline_gap" in aggregate
    assert isinstance(aggregate["conclusion"], str)
