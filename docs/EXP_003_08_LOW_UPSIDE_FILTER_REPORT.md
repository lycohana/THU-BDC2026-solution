# EXP-003-08 Low Upside Drag Filter

## Scope

Implemented a history-derived low-upside profile without external industry data or hand-written bank/gold lists.

Added:

- `code/src/stock_profile.py`
- `scripts/eval_low_upside_filter.py`

Default submission config was not changed.

```text
selector.enabled = False
postprocess.filter = regime_liquidity_anchor_risk_off
```

## Profile Definition

`build_stock_upside_profile(train_df, feature_df)` uses scorer-equivalent `label_o2o_week` and computes per-stock:

- `week_return_mean`
- `week_return_median`
- `week_return_q80`
- `week_return_q90`
- `week_return_q95`
- `week_return_max`
- `week_return_std`
- `top20_hit_rate`
- `top10_hit_rate`
- `top5_hit_rate`
- `avg_amp10_pct`
- `avg_vol10_pct`
- `avg_turnover20_pct`
- `avg_amount20_pct`

`add_low_upside_flags(profile_df, latest_feature_df)` adds:

- `low_upside_drag`
- `very_low_upside_drag`

Recent filters use only prediction-date-visible `recent_ret20_pct` and `recent_pos20_pct`.

## Strategy

Compared:

- `current_aggressive`
- `full_fallback_fixed`
- `regime_full_fallback_low_upside_v1`

Rules for low-upside v1:

- current branch excludes `low_upside_drag`
- union fallback excludes `very_low_upside_drag`
- legal fallback does not exclude, only records selected low-upside counts
- no board-aware protection
- no board exposure cap
- no partial repair
- no reranker
- no trend_uncluttered

## Run

```powershell
uv run python scripts\eval_low_upside_filter.py --last-n 10 --run-name exp00308_low_upside_last10
```

Output:

```text
temp/low_upside_filter/exp00308_low_upside_last10/
```

Required files produced:

- `low_upside_profile.csv`
- `low_upside_names.csv`
- `strategy_window_results.csv`
- `selected_details.csv`
- `before_after_20260316.csv`
- `before_after_latest.csv`

## Result

| strategy | mean | q10 | worst | win_rate | 2026-03-16 | latest 2026-04-14 |
|---|---:|---:|---:|---:|---:|---:|
| full_fallback_fixed | 0.022905 | -0.022667 | -0.033126 | 0.70 | 0.001818 | 0.006080 |
| regime_full_fallback_low_upside_v1 | 0.022905 | -0.022667 | -0.033126 | 0.70 | 0.001818 | 0.006080 |
| current_aggressive | 0.011661 | -0.039659 | -0.098454 | 0.60 | -0.098454 | 0.011732 |

`low_upside_names.csv` contains 535 flagged anchor-stock rows and 81 unique stocks.

## Interpretation

The profile works and identifies a meaningful low-upside universe, but the current thresholds do not change the selected last10 portfolios versus `full_fallback_fixed`.

Reason:

- current aggressive Top5 had zero low-upside selections in last10
- union fallback had only one low-upside selection and no very-low-upside selection
- legal fallback selected some low-upside names, but the experiment intentionally records legal exposure without excluding it

Decision: keep the infrastructure, but do not enable `regime_full_fallback_low_upside_v1` yet.

