# EXP-003-04 Branch Gate No Trend

## Scope

Implemented `scripts/eval_branch_gate_no_trend.py`.

Allowed branches:

- `current_aggressive`
- `union_topn_rrf_lcb`
- `legal_minrisk_hardened`

`reference_baseline_branch` is recorded, but is not used as fallback.

Gate rule:

```text
risk_flags >= 5 -> legal_minrisk_hardened
risk_flags >= 3 -> union_topn_rrf_lcb
otherwise       -> current_aggressive
```

Risk flags use only history-visible quantities:

- `market_ret_5d_q <= 0.30`
- `breadth_5d_q <= 0.35`
- `market_vol_10d_q >= 0.70`
- `market_dispersion_5d_q >= 0.70`
- `top5_score_margin_q <= 0.30`
- `top5_sigma_mean_q >= 0.75`
- `top5_model_disagreement_q >= 0.70`
- `top5_crowd_mean_q >= 0.70`

## Runs

```powershell
uv run python scripts\eval_branch_gate_no_trend.py --last-n 20 --run-name exp00304_gate_no_trend_last20
uv run python scripts\eval_branch_gate_no_trend.py --last-n 10 --run-name exp00304_gate_no_trend_last10
```

## Last20 Result

```json
{
  "windows": 20,
  "mean_score": 0.02459587530002005,
  "q10_score": -0.019735120791683297,
  "worst_score": -0.035542507677504956,
  "win_rate": 0.65,
  "branch_usage": {
    "current_aggressive": 8,
    "union_topn_rrf_lcb": 7,
    "legal_minrisk_hardened": 5
  }
}
```

Compared with `current_aggressive` last20 from EXP-003-02:

```text
current_aggressive mean = 0.03298, q10 = -0.02267, worst = -0.09845
gate_no_trend      mean = 0.02460, q10 = -0.01974, worst = -0.03554
```

The gate sacrifices mean return but improves worst-case loss sharply on last20.

## Last10 Result

```json
{
  "windows": 10,
  "mean_score": 0.008662630579904431,
  "q10_score": -0.03965924340465543,
  "worst_score": -0.09845397611610363,
  "win_rate": 0.5,
  "branch_usage": {
    "current_aggressive": 9,
    "union_topn_rrf_lcb": 1
  }
}
```

The gate failed to catch the 2026-03-16 crash window. That window only produced `risk_flags=1`, so the rule stayed in `current_aggressive`.

## Conclusion

`branch_gate_no_trend` is useful as a diagnostic branch matrix, but not ready as default fallback. The risk flags are too conservative for recent windows and do not identify the known 2026-03-16 tail event.

Decision: keep the script and outputs; do not change default config.

