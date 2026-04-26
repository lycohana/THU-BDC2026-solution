from __future__ import annotations

import runpy
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _make_competition_csv(path: Path, start: str, periods: int, n_stocks: int = 8) -> None:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range(start, periods=periods)
    rows = []
    for i in range(n_stocks):
        close = 10.0 * np.cumprod(1.0 + rng.normal(0.001, 0.012 + i * 0.001, size=periods))
        for d, c in zip(dates, close):
            rows.append(
                {
                    "股票代码": i + 1,
                    "日期": d.strftime("%Y-%m-%d"),
                    "开盘": c,
                    "收盘": c,
                    "最高": c * 1.01,
                    "最低": c * 0.99,
                    "成交量": 1_000_000 + i,
                    "成交额": 80_000_000 + i * 1_000_000,
                    "振幅": 2.0,
                    "涨跌额": 0.0,
                    "换手率": 1.0,
                    "涨跌幅": 0.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_replay_runner_writes_competition_format(tmp_path: Path, monkeypatch) -> None:
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    out = tmp_path / "result.csv"
    report = tmp_path / "risk_report.json"
    debug_report = tmp_path / "debug_report.json"
    _make_competition_csv(train, "2025-01-02", 80)
    _make_competition_csv(test, "2025-04-24", 8)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay_tri_state_selector.py",
            "--train",
            str(train),
            "--test",
            str(test),
            "--output",
            str(out),
            "--report",
            str(report),
            "--debug-report",
            str(debug_report),
        ],
    )
    runpy.run_path(str(ROOT / "scripts" / "replay_tri_state_selector.py"), run_name="__main__")

    result = pd.read_csv(out, dtype={"stock_id": str})
    assert list(result.columns) == ["stock_id", "weight"]
    assert len(result) <= 5
    assert result["stock_id"].str.fullmatch(r"\d{6}").all()
    assert result["weight"].ge(0).all()
    assert 0.999 <= result["weight"].sum() <= 1.0 + 1e-8
    assert report.exists()
    assert debug_report.exists()

    debug = json.loads(debug_report.read_text(encoding="utf-8"))
    required = {
        "asof",
        "regime_state",
        "confidence",
        "mode",
        "market_features",
        "unknown_industry_policy_effective",
        "output_top_k",
        "force_top_k_full_invest",
        "top_raw_scores",
        "top_shaped_scores",
        "selected_before_risk",
        "selected_after_risk",
        "selected_for_competition_output",
        "weight_sum_before_topk",
        "weight_sum_after_topk_normalize",
        "raw_weights",
        "weights_after_single_cap",
        "weights_after_industry_cap",
        "weights_after_corr_cap",
        "weights_after_vol_target",
        "weights_after_drawdown_governor",
        "final_weights",
        "total_weight_before_each_stage",
        "dropped_by_stage",
        "reason_codes",
    }
    assert required.issubset(debug.keys())
    expected_final = {row.stock_id: float(row.weight) for row in result.itertuples(index=False)}
    assert debug["final_weights"] == expected_final
    assert debug["mode"] == "competition"
    assert debug["unknown_industry_policy_effective"] == "stock_as_industry"
    assert debug["output_top_k"] == 5
    assert debug["force_top_k_full_invest"] is True
    assert debug["total_weight_before_each_stage"]["weights_after_industry_cap"] >= 0.999
    assert 0.999 <= debug["weight_sum_after_topk_normalize"] <= 1.0 + 1e-8


def test_research_mode_keeps_unknown_industry_cap_behavior(tmp_path: Path, monkeypatch) -> None:
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    out = tmp_path / "research_result.csv"
    report = tmp_path / "research_risk_report.json"
    debug_report = tmp_path / "research_debug_report.json"
    _make_competition_csv(train, "2025-01-02", 80)
    _make_competition_csv(test, "2025-04-24", 8)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay_tri_state_selector.py",
            "--mode",
            "research",
            "--no-full-invest-top5",
            "--train",
            str(train),
            "--test",
            str(test),
            "--output",
            str(out),
            "--report",
            str(report),
            "--debug-report",
            str(debug_report),
        ],
    )
    runpy.run_path(str(ROOT / "scripts" / "replay_tri_state_selector.py"), run_name="__main__")

    result = pd.read_csv(out, dtype={"stock_id": str})
    debug = json.loads(debug_report.read_text(encoding="utf-8"))
    assert debug["mode"] == "research"
    assert debug["unknown_industry_policy_effective"] == "cap"
    assert debug["total_weight_before_each_stage"]["weights_after_industry_cap"] <= 0.20000001
    assert result["weight"].sum() < 0.999
