# Recent Deep Rebound Optimization Report - 2026-04-28

## Result

本轮并入主线的是 `deep_rebound_repair`，优先级提前到 riskoff 之前。它只在深回撤、高振幅、弱广度环境触发：

- `median_ret20 <= -0.055`
- `breadth20 <= 0.30`
- `median_amp20 >= 0.16`

触发后选取 `Transformer rank + ret20 rank + 轻量 LGB/amp rank` 的修复篮子，并限制 `ret20 <= 0.30`、`amp20 >= 0.05`、`sigma20 <= 0.060`。该规则不会触发最新 2026-04-24 主线输出，最新仍走 `ret5_guarded_booster`。

## Evidence Sources

- A 股短周期动量与风险状态、流动性、投资者情绪有关：[weekly idiosyncratic momentum in China](https://arxiv.org/abs/1910.13115)。
- 中国市场短期动量容易失效，短期反转和换手效应更明显：[Does short-term momentum exist in China?](https://www.sciencedirect.com/science/article/pii/S0927538X22002153)。
- 投资者关注冲击会带来短期过度反应：[Information shocks and short-term market overreaction](https://www.sciencedirect.com/science/article/pii/S1057521924001510)。
- 股民经验侧重点相同：弱市追涨要谨慎，强势股要看位置、换手和回撤；弱势末端反而偏向做“前期强势股回调/修复”。参考：[雪球弱市策略](https://xueqiu.com/2066307904/115805041)、[雪球追涨换手心得](https://xueqiu.com/4796350440/218984836)。

## Latest Self Score

Latest live portfolio:

| stock_id | weight |
|---|---:|
| 603019 | 0.2 |
| 000977 | 0.2 |
| 600522 | 0.2 |
| 600150 | 0.2 |
| 300015 | 0.2 |

`score_self = 0.10337386404823427`

## Recent Windows

Run dir: `temp/batch_window_analysis/recent_deeprebound_smoke_20260428`

| Anchor | Window | Program | Baseline | Delta |
|---|---|---:|---:|---:|
| 2026-03-23 | 2026-03-24~2026-03-30 | +0.064042 | +0.058341 | +0.005701 |
| 2026-03-30 | 2026-03-31~2026-04-07 | +0.033994 | +0.011545 | +0.022449 |
| 2026-04-07 | 2026-04-08~2026-04-14 | +0.081859 | +0.043585 | +0.038274 |
| 2026-04-14 | 2026-04-15~2026-04-21 | +0.012761 | -0.002824 | +0.015585 |

Summary:

| Metric | Value |
|---|---:|
| Program mean | +0.048164 |
| Baseline mean | +0.027662 |
| Mean delta | +0.020502 |
| Program > 0 | 4 / 4 |
| Program > baseline | 4 / 4 |

Chart: `docs/recent_deeprebound_vs_baseline_chart.png`

## Verification

- `uv run pytest test/test_supplemental_overlay.py test/test_branch_router.py -q`: `54 passed`
- `uv run pytest test/ -q`: `71 passed`
- `uv run python app/code/src/test.py`: latest portfolio unchanged vs ret5 guarded high-score basket
- `uv run python test/score_self.py`: `0.10337386404823427`
- `uv run python scripts/batch_window_analysis.py --run-name recent_deeprebound_smoke_20260428 --anchors 2026-03-23,2026-03-30,2026-04-07,2026-04-14 --workers 1 --no-cache`: recent windows all positive and all above baseline
