# EXP-003-03F Reranker Failure Audit

## Scope

This audit keeps `candidate_reranker_v1` in the repository, but checks whether its negative result is a score direction bug or a structural failure.

Implemented script:

```powershell
uv run python scripts\audit_reranker_failure.py --last-n 20 --run-name exp00303F_audit_last20
uv run python scripts\audit_reranker_failure.py --last-n 10 --run-name exp00303F_audit_last10
```

Outputs:

```text
temp/reranker_audit/exp00303F_audit_last20/
temp/reranker_audit/exp00303F_audit_last10/
```

## Group Alignment Checks

The audit validates the LGBMRanker setup for every anchor:

- training rows are sorted by `date, stock_id`
- group sizes sum to `len(train_df)`
- each group contains a single date
- larger `rank_label` corresponds to higher future `label_o2o_week`
- selection uses descending `ranker_score`

No group alignment failure was found.

## Last20 Summary

| pool | mean spearman | mean auc | p20@5 | pos score | neg score | base score | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| candidate_union | -0.0643 | 0.6210 | 0.160 | 0.0011 | 0.0031 | 0.0269 | 0.1313 |
| current_top120 | -0.0397 | 0.6119 | 0.170 | 0.0064 | 0.0016 | 0.0330 | 0.1196 |
| current_top160 | -0.0308 | 0.6374 | 0.170 | 0.0060 | 0.0004 | 0.0330 | 0.1225 |
| current_top80 | -0.0368 | 0.5954 | 0.170 | 0.0071 | 0.0065 | 0.0330 | 0.1161 |

## Last10 Summary

| pool | mean spearman | mean auc | p20@5 | pos score | neg score | base score | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| candidate_union | -0.0827 | 0.6534 | 0.120 | -0.0090 | 0.0042 | -0.0036 | 0.1265 |
| current_top120 | -0.0237 | 0.6416 | 0.180 | -0.0000 | -0.0031 | 0.0117 | 0.1196 |
| current_top160 | -0.0232 | 0.6599 | 0.180 | 0.0003 | -0.0041 | 0.0117 | 0.1254 |
| current_top80 | -0.0027 | 0.6296 | 0.180 | -0.0002 | -0.0035 | 0.0117 | 0.1132 |

## Conclusion

This is not primarily a direction bug. In `candidate_union`, negative rerank beats positive rerank, but both are far below the base/oracle spread. In current aggressive pools, positive/negative direction is mixed and still below base selection.

Decision: stop the feature-only reranker route for now. Keep the implementation for auditability, but do not route default submission through it.

