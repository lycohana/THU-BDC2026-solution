# EXP-002-13 Report Triage

Date: 2026-04-26

## Verdict

The report's main diagnosis is useful: the current strong line is not proven stable, and the next work should focus on branch-level robustness instead of more tuning of `risk_off/anchor` around one public window.

However, the report's strongest assumption is now invalid for the latest split. The `0.13198580335505333` score is a single historical scoring window, not evidence that the same postprocess generalizes to the week starting 2026-04-20. On the strict latest setup, `data/train.csv` ends at 2026-04-17 and `data/test.csv` covers 2026-04-20 through 2026-04-24. The old active line selected:

```text
002493,603501,000792,601868,603799
```

and scored `-0.021446068839702515`.

## What Was Accepted

Keep `LightGBM 0.70 + Transformer 0.30`, z-score blend, equal weight, and `selector.enabled=False` as the main reproducible line. Do not turn on the old heavy selector by default.

Keep `union_topn_rrf_lcb` and `legal_minrisk_hardened` as shadow/fallback branches. They are lower mean but materially better in the left tail.

Use branch matrix style diagnostics before changing defaults. The local script already writes this information through:

```powershell
uv run python scripts\batch_window_analysis.py --last-n 20 --workers 1 --run-name exp00213_report_last20
```

## What Was Rejected Or Downgraded

Do not treat `regime_liquidity_anchor_risk_off` as stable solely because it reaches `0.1319858` on one window.

Do not make a hand-built `trend_uncluttered_plus_reversal` branch the default yet. A first implementation was added as `regime_trend_uncluttered_plus_reversal`, but replay showed it is not robust enough.

## New Branch Trial

Implemented:

```text
regime_trend_uncluttered_plus_reversal
```

Files changed:

```text
code/src/predict.py
code/src/portfolio_utils.py
```

The branch builds historical-only features:

```text
ret10
vol10
amp_mean10
pos20
mean_amount20
turnover20
amt_ratio5
to_ratio5
```

It then builds:

```text
trend pool: established 10/20 day trend, high position, high amplitude, enough liquidity
reversal pool: low ret10, high amplitude/volatility, non-crowded, model score not too weak
```

Prediction logging was also added to reduce result-file confusion:

```text
data_file
postprocess_filter
selector_enabled
output_path
selected stock ids
```

## Latest-Week Trial Result

With `regime_trend_uncluttered_plus_reversal` on the strict latest split:

```text
selected = 300782,601872,601600,002709,002466
score_self = -0.015826438564792773
```

This is only slightly better than the old latest-week miss and still fails the actual goal.

## Last-10 Replay Result

Command:

```powershell
uv run python scripts\batch_window_analysis.py --last-n 10 --workers 1 --run-name exp00213_trend_uncluttered_last10
```

New branch selected-score aggregate:

```text
mean   = -0.0003205974485553204
median = -0.010067173955181318
q10    = -0.04884758168815389
worst  = -0.05147003108591086
win_rate = 0.4
mean_very_bad_count = 1.3
```

It improved one strong trend window:

```text
2026-04-07 anchor: score = 0.142771
```

but it damaged multiple other windows. It should remain an experiment branch, not the default.

For comparison, the 20-window replay of the current aggressive line was:

```text
mean   = 0.03298195171718406
q10    = -0.02266737493340353
worst  = -0.09845397611610363
win_rate = 0.8
```

Shadow branches over the same 20 windows:

```text
union_topn_rrf_lcb:
  mean  = 0.0125144638
  q10   = -0.0276479902
  worst = -0.0341852793

legal_minrisk_hardened:
  mean  = 0.0050842786
  q10   = -0.0095541721
  worst = -0.0355425077
```

## Current Decision

Default config was restored to:

```text
postprocess.filter = regime_liquidity_anchor_risk_off
```

The new branch stays available for targeted gated use, but it is not stable enough to replace the default.

## Next Step

The useful path is not more single-branch hand tuning. The next experiment should build a small gate that only activates the trend-uncluttered branch under confirmed trend-regime conditions, while using `legal_minrisk_hardened` or `union_topn_rrf_lcb` when the risk of a trend reversal is high.

Minimum gate inputs should be historical-only:

```text
market_ret_5d
breadth_5d
market_vol_10d
top5_score_margin
top5_sigma_mean
top5_model_disagreement
trend_pool_size
trend_pool_mean_crowd_penalty
```

Success criteria:

```text
latest 2026-04-20 week: improve over -0.021446
last10 mean >= union_topn_rrf_lcb
last10 worst >= -0.04
last10 q10 >= -0.02
```
