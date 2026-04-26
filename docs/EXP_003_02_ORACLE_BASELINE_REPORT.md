# EXP-003-02 Oracle And Baseline Report

Date: 2026-04-26

## Scope

Implemented EXP-003-00 through EXP-003-02 infrastructure without changing the default submission branch.

Current default remains:

```text
postprocess.filter = regime_liquidity_anchor_risk_off
selector.enabled = False
```

No further hand tuning was made to `regime_trend_uncluttered_plus_reversal`.

## Implemented

### EXP-003-00 Label Module

Added:

```text
code/src/labels.py
```

Main label:

```text
label_o2o_week = open[t+5] / open[t+1] - 1
```

This matches `test/score_self.py`, which scores the first and last rows of the 5-row test window:

```text
(open_day5 - open_day1) / open_day1
```

Sanity check at anchor `2026-04-17`:

```text
002938 0.207903
002384 0.193855
600522 0.176471
```

These are the same top names produced by direct `score_self`-style calculation for `2026-04-20~2026-04-24`.

### EXP-003-01 Historical Feature Module

Added:

```text
code/src/features.py
```

Centralized historical-only features:

```text
ret1, ret5, ret10, ret20
sigma20, vol10
amp20, amp_mean10
pos20
median_amount20, mean_amount20, turnover20
amt_ratio5, to_ratio5
beta60, downside_beta60, idio_vol60, max_drawdown20
```

`predict.py` now calls `build_history_feature_frame(raw_df)`.

All rolling features are computed from rows with:

```text
日期 <= asof_date
```

so prediction-window future data is not used.

### EXP-003-02 Oracle Decomposition

Added:

```text
scripts/oracle_decomposition.py
```

Command run:

```powershell
uv run python scripts\oracle_decomposition.py --last-n 20 --run-name exp00302_oracle_last20
```

Output:

```text
temp/oracle_decomposition/exp00302_oracle_last20/oracle_decomposition.csv
temp/oracle_decomposition/exp00302_oracle_last20/aggregate.json
```

20-window aggregate:

```text
mean_selected_score          = 0.03298195171718406
mean_true_top5_score         = 0.15430363431167501
mean_ranking_gap             = 0.12132168259449097
mean_recall_true_top20_at_20 = 0.255
mean_recall_true_top20_at_50 = 0.445
mean_recall_true_top20_at_80 = 0.595
mean_recall_true_top20_at_120 = 0.7425
mean_oracle_score_top20      = 0.10085175004135902
mean_oracle_score_top50      = 0.12842495017643688
mean_oracle_score_top80      = 0.139338649807814
mean_oracle_score_top120     = 0.1453985191437994
```

Interpretation:

The model often places enough true winners somewhere in the top 80/120 candidate pool, but the final Top5 ranking is weak. This points more toward a Top-k reranker/classifier problem than a pure alpha-discovery problem.

## Baseline Branch Matrix

Added Sherlock-style branch:

```text
reference_baseline_branch
```

Definition:

```text
Transformer score only, whole universe Top5, equal weight
```

This is the closest local reproduction of the Sherlock1956 baseline behavior: `StockTransformer -> Top5 -> equal 0.2`.

Also added to branch matrix:

```text
current_aggressive
trend_uncluttered
baseline_model_hybrid
```

Added comparison script:

```text
scripts/compare_with_baseline.py
```

Command run:

```powershell
uv run python scripts\compare_with_baseline.py --last-n 20 --run-name exp00302_baseline_compare_last20
```

Output:

```text
temp/baseline_compare/exp00302_baseline_compare_last20/branch_compare.csv
temp/baseline_compare/exp00302_baseline_compare_last20/branch_compare_summary.csv
temp/baseline_compare/exp00302_baseline_compare_last20/aggregate.json
```

20-window branch summary:

| branch | mean | q10 | worst | win_rate |
|---|---:|---:|---:|---:|
| current_aggressive | 0.032982 | -0.022667 | -0.098454 | 0.80 |
| baseline_model_hybrid | 0.018286 | -0.034491 | -0.074436 | 0.70 |
| trend_uncluttered | 0.015883 | -0.048848 | -0.053765 | 0.60 |
| reference_baseline_branch | 0.014352 | -0.027536 | -0.042340 | 0.60 |
| union_topn_rrf_lcb | 0.012514 | -0.027648 | -0.034185 | 0.65 |
| legal_minrisk_hardened | 0.005084 | -0.009554 | -0.035543 | 0.75 |

## Conclusions

1. `current_aggressive` has the best mean but unacceptable worst case.
2. `reference_baseline_branch` is weaker than current aggressive, but its tail is much better than the aggressive branch.
3. `union_topn_rrf_lcb` is the best low-complexity middle branch by tail control.
4. `legal_minrisk_hardened` is still the cleanest defensive fallback.
5. `trend_uncluttered` should not be default. It is a regime-specific candidate only.
6. Oracle decomposition shows the main bottleneck is final ranking within a candidate pool, not complete absence of winners.

## Recommendation

Do not start a new ranker/classifier until one more diagnostic is done:

```text
For each window, train/evaluate a simple Top20-in-candidate reranker target:
  label = true_top20 membership inside model top80/top120
```

If a simple classifier can improve `recall_true_top20_at_20` from 0.255 without destroying tail risk, then proceed to a small ranker/classifier. Otherwise prioritize branch gating.
