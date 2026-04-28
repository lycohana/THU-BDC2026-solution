# First Board Relay Shadow Review - 2026-04-28

## Idea

Community idea: two soft up candles, a low-position first limit-up day with warm volume expansion, then buy next open after auction confirmation.

The original version depends on two data sources that are not available in the current runtime scorer:

- 09:27 auction volume and auction return.
- Minute-level dynamic take-profit / stop-loss.

Therefore this shadow only tests the daily-visible part:

- first limit-up or strong-up day on anchor date;
- previous two days are positive candles and each daily return is below 5%;
- anchor-day amount expands versus the previous day;
- recent 5-day amplitude is not above 20%;
- no future label, auction, or next-day open gap is used for runtime-like candidate selection.

## Artifacts

- Script: `scripts/first_board_relay_shadow.py`
- Window output: `temp/branch_router_validation/first_board_relay_shadow/first_board_relay_windows.csv`
- Summary: `temp/branch_router_validation/first_board_relay_shadow/first_board_relay_summary.csv`
- Event study: `temp/branch_router_validation/first_board_relay_shadow/first_board_relay_event_study.csv`

## Core Findings

Strict first-board transfer does not fit the current HS300 daily universe.

| Event | 60win Count | Mean | Q10 | Worst | Hit Rate |
|---|---:|---:|---:|---:|---:|
| first_board | 81 | -0.00352 | -0.09314 | -0.22907 | 0.444 |
| core_daily | 5 | +0.06046 | -0.07323 | -0.09125 | 0.600 |
| strong6_daily | 15 | +0.03219 | -0.07490 | -0.09125 | 0.533 |
| strong8_daily | 9 | +0.02848 | -0.07272 | -0.09125 | 0.556 |

Raw first-board continuation has bad left tail. The attractive mean is carried by a few large winners.

## Overlay Simulation

The raw strong-up insert is not safe enough:

| Variant | 60win Delta Mean | Q10 | Worst | Neg Count | Swaps |
|---|---:|---:|---:|---:|---:|
| v2b_no_swap_only_strong6_raw_top1 | +0.00075 | 0.00000 | -0.02772 | 5 | 7 |
| v2b_daily_proxy_raw_top1 | +0.00035 | 0.00000 | -0.02233 | 3 | 5 |
| default_daily_proxy_raw_top1 | +0.00046 | 0.00000 | -0.02628 | 3 | 5 |

The safer rescue version only allows insertion when the replaced weakest holding has `downside_beta60 >= 2.0`:

| Variant | 20win Delta | 40win Delta | 60win Delta | Q10 | Worst | Neg Count | Swaps |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2b_no_swap_only_strong6_raw_target_dbeta2_top1 | +0.00634 | +0.00317 | +0.00211 | 0.00000 | 0.00000 | 0 | 2 |
| default_strong6_raw_target_dbeta2_top1 | +0.00634 | +0.00317 | +0.00211 | 0.00000 | 0.00000 | 0 | 2 |
| v2b_daily_proxy_raw_target_dbeta2_top1 | +0.00432 | +0.00216 | +0.00144 | 0.00000 | 0.00000 | 0 | 2 |

Accepted windows for the best rescue variant:

| Window | In | Out | Weighted Delta |
|---|---|---|---:|
| 2025-12-03 | 300394 | 688082 | +0.07028 |
| 2026-01-23 | 000630 | 600875 | +0.05642 |

## Decision

Do not connect to runtime yet.

Reason:

- The full first-board idea is not supported by the available runtime data.
- In HS300 daily data, raw first-board / strong-up continuation has a bad tail.
- The only clean variant has just 2 swaps over 60 windows, which is too sparse to justify a new runtime flag.

Keep as shadow evidence. If revisited, treat it as a high-downside-beta rescue rule, not as a generic chase/relay alpha.
