# Single-Round Optimization Report — 2026-04-28

## 1. Baseline Status

| Item | Value |
|---|---|
| pytest | 66 passed, 5.86s |
| score_self.py | 0.0097956898761716 |
| test.py output (2026-04-24) | 603019, 601318, 000977, 002601, 601336 |
| Postprocess filter | regime_liquidity_anchor_risk_off |
| GRR Top5 enabled | True |
| Score source | grr_top5:rrf+router |
| Exposure | 1.0000 (breadth_1d > 0.30) |

### Frozen Queue Baseline (`riskoff_rank4_dynamic_pullback_stress_veto_v2`)

| Bucket | delta_new_v2b_mean | delta_new_v2b_q10 | delta_new_v2b_worst | neg_count |
|---|---|---|---|---|
| 20win | +0.00165 | 0.0 | 0.0 | 0 |
| 40win | +0.00083 | 0.0 | 0.0 | 0 |
| 60win | +0.00117 | 0.0 | 0.0 | 0 |

### 10-Window Smoke Test

| Metric | Current (frozen) | Baseline (no overlay) | Delta |
|---|---|---|---|
| mean | 0.01454 | 0.00187 | +0.01268 |
| q10 | -0.00405 | -0.00565 | +0.00260 |
| worst | -0.00578 | -0.01411 | +0.00833 |
| win_rate | 0.70 | 0.50 | +0.20 |

---

## 2. Experiment Design

### Direction: Anti-Lottery Microstructure Filter

Based on prior shadow experiments (shadow_next_candidates.csv), the `anti_lottery_lowest_score` direction had the highest composite score (0.000535) with all-positive 20/40/60 deltas and all-q10 non-negative.

**Anti-lottery score formula:**
```
anti_lottery_score = 0.35 * model_rank + 0.20 * liquidity_rank
                   + 0.20 * (1 - max_ret_rank) + 0.15 * (1 - max_jump_rank)
                   - 0.15 * risk_rank
```

**Common filter gates:**
- `model_rank >= 0.70` (top 30% by model score)
- `liquidity_rank >= 0.30` (above median liquidity)
- `sigma20 < 0.050`, `amp20 < 0.10`, `drawdown20 > -0.13`, `downside_beta60 < 1.40`

**Anti-lottery specific gates:**
- `max_ret_rank <= 0.55` (not in top 45% by max 20d return)
- `max_jump_rank <= 0.60` (not in top 40% by max 20d high jump)
- `ret5 > -0.04`, `ret20 < 0.25`

**Replacement rule:** Replace the lowest-scoring stock in Top-5 with the best anti-lottery candidate that has a higher model score.

### Scripts and Commands

```powershell
# Re-run microstructure shadow with full 60-window data
$env:UV_CACHE_DIR='.uv-cache'
uv run python scripts/microstructure_shadow.py `
  --source-run temp/batch_window_analysis/grr_tail_guard_60win `
  --detail-dir temp/branch_router_validation/v2b_guarded_longer_60win_60win `
  --raw-path data/train_hs300_20260424.csv `
  --out-dir temp/branch_router_validation/microstructure_shadow_60win
```

---

## 3. Shadow Results — 20/40/60 Window Comparison

### Best Variant: `default_anti_lottery_lowest_score_top1`

| Bucket | delta_mean | delta_q10 | delta_worst | neg_count | swaps |
|---|---|---|---|---|---|
| 20win | +0.00215 | 0.0 | 0.0 | 0 | 4/20 |
| 40win | +0.00061 | 0.0 | -0.01867 | 1 | 5/40 |
| 60win | +0.00035 | 0.0 | -0.01867 | 2 | 7/60 |

### Swap Detail (7 triggers in 60 windows)

| Window | Replaced | Candidate | Delta | Outcome |
|---|---|---|---|---|
| 2025-04-09 | 2714 (score=-2.90) | 600919 (score=0.81) | +0.0073 | WIN |
| 2025-04-16 | 2594 (score=-2.70) | 600919 (score=0.81) | **-0.0107** | LOSS |
| 2025-09-23 | 2371 (score=-2.75) | 600941 (score=0.78) | **-0.0187** | LOSS |
| 2026-02-13 | 2050 (score=-3.13) | 600900 (score=0.80) | +0.0185 | WIN |
| 2026-03-02 | 792 (score=-1.70) | 601398 (score=0.80) | +0.0122 | WIN |
| 2026-03-09 | 2594 (score=-3.07) | 601166 (score=0.76) | +0.0010 | NEAR-BREAK-EVEN |
| 2026-03-23 | 2594 (score=-2.68) | 600036 (score=0.82) | +0.0113 | WIN |

Win rate: 4/7 (57%). Mean positive: +0.0121. Mean negative: -0.0147. **Negative deltas are larger than positive deltas.**

### Full Variant Ranking (60win, top 10 by delta_mean)

| Variant | d20 | d40 | d60 | neg60 | q60 |
|---|---|---|---|---|---|
| default_anti_lottery_lowest_score | +0.00215 | +0.00061 | +0.00035 | 2 | 0.0 |
| v2b_low_turnover_reversal_lowest_score | +0.00031 | +0.00030 | +0.00031 | 5 | 0.0 |
| v2b_turnover_momentum_lowest_score | +0.00024 | +0.00024 | +0.00024 | 2 | 0.0 |
| default_low_turnover_reversal_lowest_score | +0.00123 | +0.00018 | +0.00021 | 2 | 0.0 |
| default_anti_lottery_highest_lottery | +0.00014 | +0.00052 | +0.00009 | 2 | 0.0 |
| default_gap_rebound_highest_risk | +0.00068 | +0.00034 | +0.00007 | 1 | 0.0 |
| default_turnover_momentum_highest_risk | +0.00069 | +0.00034 | +0.00005 | 1 | 0.0 |
| v2b_gap_rebound_highest_risk | +0.00068 | +0.00034 | +0.00007 | 1 | 0.0 |
| v2b_turnover_momentum_highest_risk | +0.00069 | +0.00034 | +0.00005 | 1 | 0.0 |
| v2b_no_swap_only_gap_rebound_highest_risk | +0.00068 | +0.00034 | +0.00007 | 1 | 0.0 |

---

## 4. 2026-04-24 Live Output Check

Anti-lottery overlay **did NOT trigger** on 2026-04-24. The selected stocks remain:

```
603019, 601318, 000977, 002601, 601336
```

**Output unchanged.** This passes the 2026-04-24 live output requirement.

---

## 5. Required Check Results

| Check | Result |
|---|---|
| 20win delta > 0 | PASS (+0.00215) |
| 40win delta >= 0 | PASS (+0.00061) |
| 60win delta >= 0 | PASS (+0.00035) |
| 20win q10 >= 0 | PASS (0.0) |
| 40win q10 >= 0 | PASS (0.0) |
| 60win q10 >= 0 | PASS (0.0) |
| q10/worst not worsen vs frozen | BORDERLINE (q10 same, but worst -0.0187 exists) |
| 10-window smoke beats baseline | NOT TESTED (not integrated) |
| 2026-04-24 output unchanged | PASS (no trigger) |
| neg60 = 0 | **FAIL** (2 negative windows in 60win) |

---

## 6. Integration Decision

### **NOT integrated into runtime.**

### Justification:

1. **Negative delta windows exist at 60win scale.** Two windows (2025-04-16, 2025-09-23) show anti-lottery replacing "lottery-like" stocks that actually had strong realized returns (+7.73%, +8.52%). The filter's assumption — low model score + high max_ret20 = should be removed — is wrong in these cases because the stocks had genuine momentum that the model under-scored.

2. **Negative deltas are larger than positive deltas.** Mean loss in negative windows: -0.0147 vs mean gain in positive windows: +0.0121. The overlay has negative expected skew.

3. **Low trigger rate (12%) limits robustness.** Only 7 out of 60 windows had any swap, making it difficult to statistically distinguish signal from noise.

4. **Frozen queue already handles the same risk.** The existing `riskoff_rank4_dynamic_pullback_stress_veto_v2` overlay already addresses high-risk replacement through riskoff/pullback/stress veto mechanisms with better tail protection.

5. **Incremental gain is marginal.** 60win delta_mean = +0.00035, which is within the noise band of the 60-window analysis.

---

## 7. Files Changed

| File | Action |
|---|---|
| `temp/branch_router_validation/microstructure_shadow_60win/` | NEW — 60-window shadow results |
| `temp/branch_router_validation/microstructure_shadow_60win/microstructure_summary.csv` | NEW — aggregated 20/40/60 comparison |
| `temp/branch_router_validation/microstructure_shadow_60win/microstructure_windows.csv` | NEW — per-window swap details |
| `docs/singleround_optimization_report_20260428.md` | NEW — this report |

**No runtime code changed.** No config.py, predict.py, portfolio_utils.py, or branch_router.py modifications.

---

## 8. Recommendation for Codex Review

**Recommendation: Hold. Keep frozen queue unchanged.**

The anti-lottery direction shows positive 20-window momentum but degrades at longer horizons. The two negative windows (April and September 2025) demonstrate a structural weakness: the filter penalizes stocks with extreme recent returns, but in trending markets these stocks can have genuine momentum that the model under-rates.

### Future research directions (shadow only):
1. **Conditional anti-lottery**: Only apply when market breadth is low (breadth20 < 0.50) and dispersion is high (dispersion20 > 0.12), filtering out the trending-market false negatives.
2. **Model score floor**: Require replaced stock score to be below a stricter threshold (e.g., grr_final_score < -2.5) and candidate to be in top-20% by model score, tightening the replacement quality.
3. **Consensus-weighted anti-lottery**: Add consensus count from GRR Top5 as a filter — only replace stocks with 0 or 1 expert votes, not stocks with 2+ votes even if they look "lottery-like".
