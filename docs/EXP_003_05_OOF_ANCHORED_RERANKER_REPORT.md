# EXP-003-05 OOF Score Anchored Reranker

## Scope

Implemented:

- `scripts/generate_oof_base_scores.py`
- `scripts/eval_oof_anchored_reranker.py`

The OOF rule is:

```text
each anchor score is trained only on dates <= anchor - 5 trading days
```

This script trains only a LightGBM top20 model for OOF base scores. It does not retrain Transformer.

## OOF Features

The anchored reranker includes:

- `oof_lgb_score_pct`
- `oof_lgb_rank_pct`
- `baseline_rank_pct`
- `union_rank_pct`
- `legal_rank_pct`
- `source_count`
- `source_rrf_score`
- `in_baseline_top30`
- `in_union_top30`
- `in_legal_top30`
- `in_current_top120`
- `ret10_pct`, `ret20_pct`, `pos20_pct`
- `amount20_pct`, `turnover20_pct`
- `amp_mean10_pct`, `vol10_pct`
- `amt_ratio5_pct`, `to_ratio5_pct`
- `crowd_penalty`, `extreme_vol_penalty`

Final score:

```text
final_score =
  0.70 * base_anchor
+ lambda_meta * meta_top20_pct
+ 0.10 * source_agreement
- 0.10 * risk_penalty
```

Grid:

```text
lambda_meta = 0, 0.05, 0.10, 0.20, 0.30
```

## Runs

```powershell
uv run python scripts\generate_oof_base_scores.py --last-n 20 --run-name exp00305_oof_lgb_last20
uv run python scripts\eval_oof_anchored_reranker.py --last-n 20 --run-name exp00305_oof_anchored_last20 --oof-run-name exp00305_oof_lgb_last20
uv run python scripts\eval_oof_anchored_reranker.py --last-n 10 --run-name exp00305_oof_anchored_last10 --oof-run-name exp00305_oof_lgb_last20
```

Outputs:

```text
temp/oof_base_scores/exp00305_oof_lgb_last20/
temp/oof_anchored_reranker/exp00305_oof_anchored_last20/
temp/oof_anchored_reranker/exp00305_oof_anchored_last10/
```

## Last20 Result

| lambda_meta | mean | q10 | worst | gap mean | top20 hits |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.01331 | -0.03286 | -0.11350 | 0.11443 | 1.00 |
| 0.05 | 0.01149 | -0.04239 | -0.10684 | 0.11625 | 1.00 |
| 0.10 | 0.01115 | -0.04869 | -0.10684 | 0.11659 | 0.95 |
| 0.20 | 0.01063 | -0.03585 | -0.10582 | 0.11711 | 0.95 |
| 0.00 | 0.00903 | -0.05739 | -0.10684 | 0.11870 | 1.00 |

## Last10 Result

| lambda_meta | mean | q10 | worst | gap mean | top20 hits |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.00936 | -0.03826 | -0.10684 | 0.11750 | 1.00 |
| 0.00 | -0.00001 | -0.07174 | -0.10684 | 0.12687 | 0.90 |
| 0.05 | -0.00060 | -0.04745 | -0.10684 | 0.12747 | 0.90 |
| 0.30 | -0.00168 | -0.04962 | -0.07009 | 0.12854 | 0.70 |
| 0.20 | -0.00404 | -0.05329 | -0.10684 | 0.13090 | 0.70 |

## Conclusion

The OOF anchored reranker is cleaner than `candidate_reranker_v1`, but it is still not strong enough to replace the default branch. It does not close the Top120 ranking gap enough, and its mean score remains far below `current_aggressive` last20.

Decision: keep the OOF infrastructure for future meta work. Do not change default config.

