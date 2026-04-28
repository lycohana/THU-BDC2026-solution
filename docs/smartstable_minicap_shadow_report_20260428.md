# SmartStable MiniCap Shadow Review - 2026-04-28

## Scope

The source idea is a JoinQuant-style full-market small-cap strategy:

- universe: `399101.XSHE` / small and mid cap names;
- market cap: 5-100 billion CNY;
- BBI trend filter: close above `(MA3 + MA6 + MA12 + MA24) / 4`;
- weekly rebalance, 8 equal holdings;
- intraday stop loss, limit-up open-break monitoring, ST/new-stock/paused filters.

The current project data is only a 300-stock HS300-style daily universe. It does not include true market cap, ST status, listing age, intraday price path, or limit-up open-break state. This shadow therefore tests only a daily, no-leak proxy:

- `mcap_proxy = amount / turnover`;
- BBI trend from daily close;
- small-cap rank within the available 300 names;
- existing risk features: `sigma20`, `amp20`, `max_drawdown20`, `downside_beta60`;
- optional model confirmation via `grr_final_score` rank.

## Artifacts

- Script: `scripts/smartstable_minicap_shadow.py`
- Overlay windows: `temp/branch_router_validation/smartstable_minicap_shadow/smartstable_overlay_windows.csv`
- Overlay summary: `temp/branch_router_validation/smartstable_minicap_shadow/smartstable_overlay_summary.csv`
- Standalone windows: `temp/branch_router_validation/smartstable_minicap_shadow/smartstable_standalone_windows.csv`
- Standalone summary: `temp/branch_router_validation/smartstable_minicap_shadow/smartstable_standalone_summary.csv`
- Frozen-queue strict check:
  - `smartstable_after_frozen_windows.csv`
  - `smartstable_after_frozen_summary.csv`
  - `smartstable_after_frozen_dbeta12_windows.csv`
  - `smartstable_after_frozen_dbeta12_summary.csv`

## Standalone Result

As an independent selector, the proxy strategy loses badly to the current main line.

| Variant | 60win Return Mean | Delta Mean | Delta Q10 | Delta Worst | Negative Delta |
|---|---:|---:|---:|---:|---:|
| standalone_model_confirmed_top5 | 0.03235 | -0.01857 | -0.07442 | -0.17275 | 24 |
| standalone_not_overheated_top5 | 0.00749 | -0.04342 | -0.09666 | -0.18699 | 46 |
| standalone_relaxed_5_300b_top5 | 0.00615 | -0.04476 | -0.09909 | -0.16059 | 50 |
| standalone_exact_5_100b_top5 | 0.00608 | -0.04483 | -0.09817 | -0.16059 | 45 |
| standalone_smallest_bbi_raw_top5 | 0.00110 | -0.04982 | -0.10538 | -0.15476 | 51 |

Conclusion: do not replace the current Top5 selector with a small-cap BBI selector.

## Event Study

| Event | Count | Mean | Q10 | Worst | Hit Rate |
|---|---:|---:|---:|---:|---:|
| model_confirmed | 48 | +0.02723 | -0.01857 | -0.10643 | 0.708 |
| not_overheated | 541 | +0.00482 | -0.03731 | -0.25078 | 0.512 |
| exact_5_100b | 720 | +0.00441 | -0.03595 | -0.25078 | 0.504 |
| bbi_up | 8599 | +0.00255 | -0.04815 | -0.30451 | 0.469 |

The only useful shape is `small-cap proxy + BBI + model confirmation`. Pure BBI or pure small-cap is too noisy.

## Overlay vs v2b

Best v2b-side variants were clean but sparse:

| Variant | 60win Delta | Q10 | Worst | Negative Delta | Swaps |
|---|---:|---:|---:|---:|---:|
| v2b_model_confirmed_highest_dbeta_top1 | +0.00074 | 0.00000 | 0.00000 | 0 | 3 |
| v2b_not_overheated_highest_dbeta_top1 | +0.00074 | 0.00000 | 0.00000 | 0 | 3 |
| v2b_relaxed_5_300b_highest_dbeta_top1 | +0.00074 | 0.00000 | 0.00000 | 0 | 3 |

These variants mainly work by replacing the highest downside-beta holding, not by selecting small caps generically.

## Strict Check After Frozen Queue

Applying the same idea after the current frozen queue is weaker:

| Variant | 60win Delta | Q10 | Worst | Negative Delta | Swaps |
|---|---:|---:|---:|---:|---:|
| frozen_queue_model_confirmed_highest_dbeta_top1 | +0.00047 | 0.00000 | -0.00202 | 1 | 3 |

The negative window was 2026-03-09, where the replaced stock `002594` had `downside_beta60 = 0.8078`, so it was not a real high-beta risk source.

Adding target-side hard gate `target downside_beta60 >= 1.2` fixes the tail:

| Variant | 20win Delta | 40win Delta | 60win Delta | Q10 | Worst | Negative Delta | Swaps |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen_queue_dbeta12_model_confirmed_highest_dbeta_top1 | +0.00015 | +0.00008 | +0.00051 | 0.00000 | 0.00000 | 0 | 2 |

Accepted windows:

| Window | In | Out | Target Downside Beta | Weighted Delta |
|---|---|---|---:|---:|
| 2025-04-01 | 000999 | 300408 | 1.2654 | +0.02742 |
| 2026-03-02 | 601939 | 601117 | 2.8342 | +0.00299 |

## Decision

Do not connect to runtime.

Reason:

- The original strategy needs full-market small caps and intraday risk handling; the current scorer does not have those inputs.
- Standalone small-cap BBI selection strongly underperforms the current main line.
- The only safe overlay after the frozen queue has just 2 swaps over 60 windows, too sparse for a new runtime flag.

Keep the useful part as a research note:

> `BBI + model-confirmed small-cap proxy` can be a replacement candidate only when the current holding is a genuine high-downside-beta risk source.

This is closer to a risk repair rule than a standalone SmartStable MiniCap alpha.
