# Runtime Overlay Flow

本文件固化当前 live 预测路径中的 post-guard supplemental overlay 流程。目标不是继续堆新 alpha，而是在保留 v2b / GRR 主线的前提下，提高 riskoff micro 捕获率，并用 stress-chaser veto 修复高风险追涨或单票 beta 冲击。

## Core Decision

当前冻结队列：

`riskoff_rank4_dynamic_pullback_stress_veto_v2`

运行原则：

1. Base 仍然是 `v2b_guarded_candidate_with_deeper_veto_disp013` 产出的候选顺序。
2. Alpha overlay 每天最多接受 1 次，优先级为：
   `riskoff_fill_rank4_dynamic_defensive_target_no_v2b_swap -> pullback_rebound_highest_risk`
3. Stress-chaser veto 是最后的风险修复，不算新 alpha sleeve；它只在市场 stress gate 触发后，最多替换 1 个高风险持仓。
4. Runtime 决策只使用 T 日可见特征，禁止使用 scorer、future return、baseline 分支或窗口回测结果。

## Live Path

`predict.py` 的关键顺序：

1. 读取最新训练/预测数据，构造 `score_df` 和风险特征。
2. 若启用 GRR，先运行 `apply_grr_top5`。
3. 通过 `select_candidates` 生成 v2b / guarded candidate 列表。
4. 调用 `apply_supplemental_overlay(score_df, filtered, config["branch_router_v2b"])`。
5. 对 overlay 后的 Top5 调用 `build_weight_portfolio` 生成最终持仓。

因此 overlay 的输入是“已被主线筛过的 Top5 + 全市场当日特征”，输出仍然是 Top5，权重逻辑不变。

## Runtime Queue

### 1. Riskoff Rank4 Dynamic Defensive

用途：只在 v2b 未换或 v2b 换入偏负风险源时，尝试捕获 `riskoff_top60_direct` 的一部分收益。

触发边界：

- 不能抢 v2b 的强票。
- 只能替换排名靠后的弱风险源。
- 候选必须过防御/流动性/低风险过滤。

### 2. Pullback Rebound Highest Risk

用途：保留一条追涨/回踩后再放量的轻量补丁，但优先级低于 riskoff defensive。

触发边界：

- 只在 riskoff overlay 未接受时生效。
- 只替换当前 Top5 内最高风险的一只。
- 通过 `pullback_rank_cap` 限制激进程度。

### 3. Stress-Chaser Veto Final

用途：在市场已经处于 stress 状态时，剔除 Top5 里的局部风险爆点，而不是寻找收益最高的新票。

市场 stress gate：

- `median_ret20 < stress_chaser_median_ret20_max`
- `breadth20 <= stress_chaser_breadth20_max`
- `median_sigma20 > stress_chaser_median_sigma20_min`
- `dispersion20 > stress_chaser_dispersion20_min`

可替换目标：

- Panic beta shock：`ret1 < stress_panic_ret1_max` 且 `downside_beta60 > stress_panic_downside_beta_min`
- Hot chaser risk：`ret5 > stress_hot_ret5_min` 且 `ret20 < stress_hot_ret20_max` 且 `amp20 > stress_hot_amp20_min` 且 `downside_beta60 > stress_hot_downside_beta_min`

替换池：

- 当前 Top5 以外。
- 通过 runtime min-risk 打分。
- 优先低 `sigma20`、低 `amp20`、低 `max_drawdown20`、低 `downside_beta60`，同时要求足够流动性。

## Frozen Evidence

冻结脚本：

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python scripts/post_guard_overlay_freeze.py
```

输出目录：

`temp/submission_freeze/post_guard_overlay_queue_freeze`

主要产物：

- `frozen_post_guard_overlay_config.json`
- `post_guard_overlay_windows.csv`
- `post_guard_overlay_summary.csv`
- `decision_report.md`
- `runtime_safety_check.txt`
- `evidence/`

## Required Checks

固化前必须同时满足：

1. `20win / 40win / 60win` 的 `delta_new_v2b_mean > 0`
2. `20win / 40win / 60win` 的 `delta_new_v2b_q10 >= 0`
3. Stress veto 10-window smoke 的 current mean / q10 / worst 均优于 baseline
4. Latest live 2026-04-24 若无 riskoff/pullback/stress gate，最终选股应保持不变
5. 单测和全量 pytest 必须通过

当前 10-window smoke：

- Current mean: `0.014543919874455918`
- Current q10: `-0.00405070431535826`
- Current worst: `-0.0057767848778843`
- Baseline mean: `0.0018668252911232218`
- Baseline q10: `-0.005652222062380649`
- Baseline worst: `-0.0141071555817288`

## Why This Is Frozen First

外部研究和本地 shadow 都指向同一个风险：单纯追逐放量/成交量均线差容易在短窗里过拟合，且在高波动股票中会放大尾部。因此这版先固化的是“风险 veto + riskoff 捕获”，不是新的量价 overlay。后续发散实验可以继续做，但必须先以 shadow 表证明不会破坏当前 frozen queue 的 q10 / worst。

## Next Research Queue

下一轮只做 shadow，不直接接 runtime：

1. Turnover persistence：研究长期换手改善是否和现有 v2b 低重叠。
2. Low liquidity-beta / downside-beta defensive：检查 stress veto 替换池是否可以加入 liquidity beta proxy。
3. 52-week-high anchor + turnover：把追涨拆成“新闻型高换手动量”和“低锚定反转陷阱”两类。
4. Conflict filter v2：逐日分析 riskoff / pullback / stress veto 与 v2b accepted swaps 的替换目标冲突。

## Post-Freeze Search Notes

2026-04-28 追加跑了三组 shadow 搜索，全部只写诊断表，没有接 runtime：

- `scripts/chase_momentum_shadow.py`
- `scripts/microstructure_shadow.py`
- `scripts/confidence_anchor_shadow.py`

统一候选筛选表：

`temp/branch_router_validation/shadow_next_candidates.csv`

当前结论：

1. 追涨方向最强的是 `default_model_confirmed_strict_top3/top5`，但负 delta 窗口仍偏多，只能继续 shadow。
2. 微结构方向最稳的是 `default_anti_lottery_lowest_score_top1/top3/top5`，20/40/60 均值为正且 q10 不伤，但收益厚度不足，暂不接 runtime。
3. Anchor/confidence 方向更干净但样本稀疏，适合作为下一轮过滤器，不适合作为独立 overlay。

外部研究给这轮搜索的启发：

- 高换手股票更容易从短期反转切换到短期动量，因此追涨必须区分“高换手新闻型延续”和“低换手流动性反转”。
- 中国市场低 liquidity-beta 股票存在风险调整后优势，因此 stress veto 的替换池后续可以加入 liquidity beta proxy，而不是只看普通 downside beta。
- 成交量/换手变化可以是 alpha，但单纯用量能扩张很容易变成过拟合；因此当前 runtime 先固化风险 veto，而不是继续堆量价 overlay。
