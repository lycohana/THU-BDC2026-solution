# GPTPro Handoff: THU-BDC2026 Experiment Reports

## Current Default Submission State

Default submission branch has not been changed.

```text
selector.enabled = False
postprocess.filter = regime_liquidity_anchor_risk_off
postprocess.weighting = equal
```

Current conclusion: keep the existing strong default branch for submission unless a later experiment proves a stable replacement. EXP-003 reranker/gate work is diagnostic and experimental only.

## Report Index

1. `docs/EXP_002_11_FINAL_REPORT.md`
   - Final known strong line before EXP-003.
   - Reproduced full pipeline score around `0.12018139687305522`.

2. `docs/EXP_002_12_STABLE_RESTORE.md`
   - Stable restore / default branch status.
   - Confirms current code should preserve the working strong branch.

3. `docs/EXP_002_13_REPORT_TRIAGE.md`
   - Report triage after external analysis.
   - Main decision: do not keep hand-tuning `risk_off/anchor`; move to oracle decomposition and structural diagnostics.

4. `docs/EXP_003_02_ORACLE_BASELINE_REPORT.md`
   - Oracle decomposition, reference baseline branch, branch matrix.
   - Key finding: current aggressive has alpha, but Top120 ranking gap remains large.

5. `docs/EXP_003_03_RERANKER_V1_REPORT.md`
   - First candidate reranker.
   - Result: negative. Do not delete, but do not use in default path.

6. `docs/EXP_003_03F_RERANKER_AUDIT_REPORT.md`
   - Failure audit for reranker_v1.
   - Checks LGBMRanker group alignment and positive/negative rerank direction.
   - Key conclusion: not a simple direction bug; feature-only reranker route should stop for now.

7. `docs/EXP_003_04_BRANCH_GATE_NO_TREND_REPORT.md`
   - No-trend branch gate using only `current_aggressive`, `union_topn_rrf_lcb`, `legal_minrisk_hardened`.
   - Last20 tail improves, but last10 fails to catch the 2026-03-16 crash window.
   - Key conclusion: keep as diagnostic, not default fallback.

8. `docs/EXP_003_05_OOF_ANCHORED_RERANKER_REPORT.md`
   - OOF LGB base scores and anchored meta reranker.
   - Strict OOF rule: each anchor score uses only dates `<= anchor - 5 trading days`.
   - Key conclusion: cleaner than reranker_v1, but still not strong enough to replace default.

## Key Experiment Results

### EXP-003-02 Oracle / Baseline

`oracle_decomposition --last-n 20`

```text
mean_selected_score = 0.03298195171718406
mean_true_top5_score = 0.15430363431167501
mean_ranking_gap = 0.12132168259449097
recall_true_top20_at_20 = 0.255
recall_true_top20_at_50 = 0.445
recall_true_top20_at_80 = 0.595
recall_true_top20_at_120 = 0.7425
oracle_score_top120 = 0.1453985191437994
```

Branch comparison last20:

```text
current_aggressive:        mean 0.032982, q10 -0.022667, worst -0.098454
baseline_model_hybrid:     mean 0.018286, q10 -0.034491, worst -0.074436
trend_uncluttered:         mean 0.015883, q10 -0.048848, worst -0.053765
reference_baseline_branch: mean 0.014352, q10 -0.027536, worst -0.042340
union_topn_rrf_lcb:        mean 0.012514, q10 -0.027648, worst -0.034185
legal_minrisk_hardened:    mean 0.005084, q10 -0.009554, worst -0.035543
```

Interpretation: current aggressive is still the best mean branch, but tail risk is real.

### EXP-003-03 Reranker v1

Last20 best pool:

```text
current_aggressive_top80:
mean = 0.007079
q10 = -0.056389
worst = -0.100970
rerank_gain_ratio = -1.322889
```

Last10 best pool:

```text
current_aggressive_top160:
mean = 0.000311
q10 = -0.088673
worst = -0.100970
rerank_gain_ratio = -0.128677
```

Interpretation: reranker_v1 is a clean negative result.

### EXP-003-03F Reranker Failure Audit

Group alignment checks passed:

```text
train_df sorted by date, stock_id
group sum == len(train_df)
each group date unique
larger rank_label means higher future return
selection sorts ranker_score descending
```

Last20:

```text
candidate_union:
  mean_ic_spearman = -0.0643
  mean_auc = 0.6210
  positive_rerank_selected_score = 0.0011
  negative_rerank_selected_score = 0.0031
  base_selected_score = 0.0269
  oracle_score_pool = 0.1313

current_aggressive_top120:
  mean_ic_spearman = -0.0397
  mean_auc = 0.6119
  positive_rerank_selected_score = 0.0064
  negative_rerank_selected_score = 0.0016
  base_selected_score = 0.0330
  oracle_score_pool = 0.1196
```

Interpretation: not simply a score direction bug. Negative rerank is sometimes better, but both directions are too weak.

### EXP-003-04 Branch Gate No Trend

Rule:

```text
risk_flags >= 5 -> legal_minrisk_hardened
risk_flags >= 3 -> union_topn_rrf_lcb
otherwise       -> current_aggressive
```

Last20:

```text
mean_score = 0.024596
q10_score = -0.019735
worst_score = -0.035543
win_rate = 0.65
branch_usage:
  current_aggressive = 8
  union_topn_rrf_lcb = 7
  legal_minrisk_hardened = 5
```

Last10:

```text
mean_score = 0.008663
q10_score = -0.039659
worst_score = -0.098454
win_rate = 0.50
branch_usage:
  current_aggressive = 9
  union_topn_rrf_lcb = 1
```

Interpretation: last20 tail improves, but the gate fails on the known 2026-03-16 crash window because it only raises `risk_flags=1`.

### EXP-003-05 OOF Anchored Reranker

OOF rule:

```text
For every anchor date, oof_lgb_score is trained only using dates <= anchor - 5 trading days.
No Transformer retrain.
No in-sample base_score for meta training.
```

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

Last20 best:

```text
lambda_meta = 0.30
mean = 0.013308
q10 = -0.032861
worst = -0.113498
ranking_gap_mean = 0.114430
candidate_recall_true_top20 = 0.6125
```

Last10 best:

```text
lambda_meta = 0.10
mean = 0.009361
q10 = -0.038262
worst = -0.106841
ranking_gap_mean = 0.117501
candidate_recall_true_top20 = 0.495
```

Interpretation: infrastructure is useful, but this anchored reranker is not yet a replacement for current aggressive.

## Current Technical Artifacts

Important scripts:

```text
scripts/oracle_decomposition.py
scripts/compare_with_baseline.py
scripts/train_eval_reranker.py
scripts/audit_reranker_failure.py
scripts/eval_branch_gate_no_trend.py
scripts/generate_oof_base_scores.py
scripts/eval_oof_anchored_reranker.py
```

Core modules:

```text
code/src/labels.py
code/src/features.py
code/src/reranker.py
```

Important output folders:

```text
temp/oracle_decomposition/exp00302_oracle_last20/
temp/baseline_compare/exp00302_baseline_compare_last20/
temp/reranker_eval/exp00303_reranker_v1_last20/
temp/reranker_eval/exp00303_reranker_v1_last10/
temp/reranker_audit/exp00303F_audit_last20/
temp/reranker_audit/exp00303F_audit_last10/
temp/branch_gate/exp00304_gate_no_trend_last20/
temp/branch_gate/exp00304_gate_no_trend_last10/
temp/oof_base_scores/exp00305_oof_lgb_last20/
temp/oof_anchored_reranker/exp00305_oof_anchored_last20/
temp/oof_anchored_reranker/exp00305_oof_anchored_last10/
```

## Recommended Next Decision

Do not change the default branch yet.

Recommended next direction:

```text
1. Treat current_aggressive as the live submission branch.
2. Keep EXP-003 reranker/gate outputs as diagnostics.
3. Do not continue feature-only reranker.
4. If continuing, focus on why 2026-03-16 was not flagged:
   - risk flag design
   - candidate crowding proxy
   - branch disagreement features
   - regime features computed over longer history
5. Do not add trend_uncluttered back into default until a gate can distinguish its good/bad regimes.
```

