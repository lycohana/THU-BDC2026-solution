# Post-Announcement Reaction Shadow Review - 2026-04-28

## Research Notes

The external strategy claims to buy at the announcement-day open while using the same day's bullish candle as a filter. That is a timing leak:

- `T open gap >= 3%` is only known around auction/open.
- `T close > T open` is only known after the close.
- Therefore the earliest clean entry after confirming `gap + bullish candle` is `T+1 open`.

Useful references:

- JoinQuant stock data docs: `get_fundamentals(..., date=...)` searches data published before or on the query date, while `statDate` is not point-in-time safe by itself.
  <https://joinquant.com/help/data/stock>
- JoinQuant future-data restrictions describe that daily `close/high/low/volume/money` are not available before the market close and that same-day data access is time-dependent.
  <https://www.joinquant.com/community/post/detailMobile?postId=23804>
- RiceQuant RQData fundamentals docs expose `info_date` as announcement date, which is the correct field to use for point-in-time financial data alignment.
  <https://www.ricequant.com/doc/rqdata/python/stock-mod.html>
- China PEAD literature supports the existence of post-earnings announcement drift, but the implementation must respect announcement time and trade-time availability.
  <https://www.sciencedirect.com/science/article/pii/S1042443111000205>
- Recent China earnings-announcement research emphasizes overnight/intraday differences around announcements, reinforcing that open/auction timing is not a small detail.
  <https://www.sciencedirect.com/science/article/pii/S0929119923001207>

## Local Test Design

The current project has no earnings announcement table, forecast table, or point-in-time `info_date` field. So this shadow tests the clean price-reaction proxy only:

- `gap_open_ret >= 3%`
- `intraday_ret = close / open - 1 > 0`
- signal is formed after `T close`
- entry label remains `T+1 open -> T+5 open`
- no same-day close condition is used for same-day open entry

Artifacts:

- Script: `scripts/post_announcement_reaction_shadow.py`
- Main windows: `temp/branch_router_validation/post_announcement_reaction_shadow/post_announcement_reaction_windows.csv`
- Main summary: `temp/branch_router_validation/post_announcement_reaction_shadow/post_announcement_reaction_summary.csv`
- Event study: `temp/branch_router_validation/post_announcement_reaction_shadow/post_announcement_reaction_event_study.csv`
- Veto probe:
  - `post_announcement_reaction_veto_windows.csv`
  - `post_announcement_reaction_veto_summary.csv`

## Event Study Result

Clean T+1-entry high-open bullish-candle proxy is negative in the current HS300 weekly label.

| Event | 60win Count | Mean | Q10 | Worst | Hit Rate |
|---|---:|---:|---:|---:|---:|
| gap_green_raw | 74 | -0.00654 | -0.09653 | -0.13179 | 0.365 |
| gap_green_sane | 44 | -0.03038 | -0.09653 | -0.12592 | 0.250 |
| gap_green_close_strong | 31 | -0.03512 | -0.10314 | -0.12592 | 0.226 |

The original strategy's large return is consistent with the timing leak: once we move entry to `T+1 open`, the signal becomes a short-term overreaction, not a continuation alpha.

## Overlay Result

Adding high-open bullish-candle candidates after the current frozen queue hurts.

Best non-zero frozen-queue variants:

| Variant | Swaps | 60win Delta | Q10 | Worst | Negative Delta |
|---|---:|---:|---:|---:|---:|
| frozen_queue_gap_green_raw_highest_risk_dbeta1p2_top1 | 15 | -0.00122 | -0.01037 | -0.03305 | 10 |
| frozen_queue_gap_green_sane_lowest_score_dbeta1p2_top1 | 12 | -0.00150 | -0.00358 | -0.04813 | 7 |
| frozen_queue_gap_green_sane_highest_risk_dbeta1p2_top1 | 10 | -0.00177 | -0.00516 | -0.03305 | 8 |

No model-confirmed candidate existed after the risk gates. That is a useful guardrail: the existing model and risk filters already reject these events.

## Veto Probe

I also tested reversing the idea: if the current frozen Top5 contains a high-open bullish-candle overreaction stock, replace it with a min-risk candidate.

Result: zero accepted swaps in 60 windows. The current frozen queue does not meaningfully hold this specific overreaction pattern, so there is nothing to veto at runtime.

## Decision

Do not connect to runtime.

Keep the script as a timing-leak regression harness:

- If future fundamental announcement data is added, require `announcement_info_date <= T-1` for an open-entry strategy.
- If the signal uses `T close > T open`, entry must be `T+1 open`, never `T open`.
- Treat `gap + green candle` as a potential overreaction warning, not a buy signal, until real point-in-time earnings data proves otherwise.
