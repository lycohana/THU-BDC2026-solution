# EXP-003-03 Candidate Reranker V1 Report

Date: 2026-04-26

## Scope

Implemented `candidate_reranker_v1` as an isolated experiment. The default submission branch was not changed.

Default remains:

```text
selector.enabled = False
postprocess.filter = regime_liquidity_anchor_risk_off
```

## Implemented

Added:

```text
code/src/reranker.py
scripts/train_eval_reranker.py
```

`code/src/reranker.py` includes:

```text
make_rank_labels()
train_lgb_ranker()
train_top20_classifier()
train_top5_classifier()
predict_rerank_scores()
select_by_reranker()
```

Training labels use `label_o2o_week` from `labels.py`.

For each `asof_date`, training rows are restricted to dates with known labels:

```text
date <= asof_date - 5 trading days
```

Rank label:

```text
future_rank <= 5   -> 4
future_rank <= 20  -> 3
future_rank <= 60  -> 2
future_rank <= 120 -> 1
else               -> 0
```

Models trained per anchor:

```text
LGBMRanker
LGBMClassifier is_top20
LGBMClassifier is_top5
```

Rerank score:

```text
0.50 * z(rank_score)
+ 0.40 * z(top20_prob)
+ 0.10 * z(top5_prob)
```

Candidate pools evaluated:

```text
current_aggressive_top80
current_aggressive_top120
current_aggressive_top160
candidate_union
```

`winner_ladder.csv` is emitted with true Top20 ranks across:

```text
current_aggressive
reference_baseline
union
trend_uncluttered
candidate_reranker
```

and feature snapshots.

## Commands Run

```powershell
uv run python scripts\train_eval_reranker.py --last-n 20 --run-name exp00303_reranker_v1_last20
uv run python scripts\train_eval_reranker.py --last-n 10 --run-name exp00303_reranker_v1_last10
```

Outputs:

```text
temp/reranker_eval/exp00303_reranker_v1_last20/
temp/reranker_eval/exp00303_reranker_v1_last10/
```

Each output directory contains:

```text
aggregate.json
reranker_summary.csv
reranker_window_results.csv
reranker_selected_details.csv
winner_ladder.csv
```

## Last20 Result

| pool | mean | q10 | worst | old_gap | new_gap | gain_ratio | top20_hits | top5_hits | candidate_recall_top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_aggressive_top80 | 0.007079 | -0.056389 | -0.100970 | 0.086625 | 0.112528 | -1.322889 | 0.85 | 0.20 | 0.4375 |
| current_aggressive_top120 | 0.006427 | -0.056389 | -0.100970 | 0.086625 | 0.113180 | -1.439817 | 0.85 | 0.20 | 0.5125 |
| current_aggressive_top160 | 0.005957 | -0.066851 | -0.100970 | 0.086625 | 0.113650 | -1.537225 | 0.85 | 0.20 | 0.5625 |
| candidate_union | 0.001098 | -0.069544 | -0.100970 | 0.086625 | 0.118509 | -1.448555 | 0.80 | 0.25 | 0.6175 |

## Last10 Result

| pool | mean | q10 | worst | old_gap | new_gap | gain_ratio | top20_hits | top5_hits | candidate_recall_top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_aggressive_top160 | 0.000311 | -0.088673 | -0.100970 | 0.107925 | 0.119275 | -0.128677 | 0.90 | 0.20 | 0.470 |
| current_aggressive_top120 | -0.000039 | -0.088673 | -0.100970 | 0.107925 | 0.119625 | -0.132464 | 0.90 | 0.20 | 0.410 |
| current_aggressive_top80 | -0.000189 | -0.088673 | -0.100970 | 0.107925 | 0.119775 | -0.134100 | 0.90 | 0.20 | 0.350 |
| candidate_union | -0.009013 | -0.079481 | -0.100970 | 0.107925 | 0.128599 | -0.223523 | 0.60 | 0.20 | 0.485 |

## Conclusion

`candidate_reranker_v1` is a clean negative result.

It does not shrink the Top120 ranking gap. It increases the gap in both last20 and last10 tests and produces worse selected scores than the current aggressive branch and the earlier middle/defensive branches.

The failure mode is informative:

```text
Candidate recall exists, but the supervised reranker is not learning the useful within-candidate order.
```

The likely reason is that historical training rows only have price-derived feature snapshots, while live candidate rows also depend on model-score context. The current v1 does not have historical model-score features for each past date, so it cannot learn how to interpret model ranks consistently.

## Decision

Do not route default submissions through `candidate_reranker_v1`.

Do not proceed to a heavier classifier/ranker until the training feature mismatch is fixed.

Recommended next diagnostic:

```text
Build historical model-score OOF features per anchor/window, then retry a much smaller reranker.
```

Without historical model-score features, branch gating remains a better next direction than another larger reranker.
