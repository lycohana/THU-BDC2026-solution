# Runtime Overlay Flow

本文件固化当前 live 预测路径中的 post-guard supplemental overlay 流程。目标不是继续堆新 alpha，而是在保留 v2b / GRR 主线的前提下，提高 riskoff micro 捕获率，并用 stress-chaser veto 修复高风险追涨或单票 beta 冲击。

## Core Decision

当前冻结队列：

`riskoff_rank4_dynamic_ret5guarded_stress_veto_antilottery_v1`

运行原则：

1. Base 仍然是 `v2b_guarded_candidate_with_deeper_veto_disp013` 产出的候选顺序。
2. Alpha overlay 每天最多接受 1 次，优先级为：
   `riskoff_fill_rank4_dynamic_defensive_target_no_v2b_swap -> pullback_rebound_highest_risk -> ret5_guarded_booster`
3. `ret5_guarded_booster` 是冲高 sleeve：最多替换 3 个短期最弱持仓，但候选必须先通过 `ret20`、`sigma20`、`amp20`、`downside_beta60`、`max_drawdown20` 和流动性硬约束。
4. Stress-chaser veto 是最后的风险修复，不算新 alpha sleeve；它只在市场 stress gate 触发后，最多替换 1 个高风险持仓。
5. Conditional anti-lottery 是 stress veto 后的轻量补充，只在总替换次数未达到 `max_total_swaps=2` 时尝试；被替换标的必须满足 `downside_beta60 <= 1.35`，避免误删 stress 中可能反弹的高下行 beta 股票。
6. Runtime 决策只使用 T 日可见特征，禁止使用 scorer、future return、baseline 分支或窗口回测结果。

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

### 3. Ret5 Guarded Booster

用途：把本地高分来源从“裸追 5 日涨幅”改成“5 日强势 + 风控硬门槛”。它保留冲高能力，同时用波动、振幅、下行 beta、回撤和流动性过滤掉最不稳的强势票。

触发边界：

- 只在 riskoff 和 strict pullback 都未接受时生效。
- 候选必须在 Top5 外，且 `ret5 >= 0.08`、`0.08 <= ret20 <= 0.45`、`sigma20 <= 0.050`、`amp20 <= 0.30`、`downside_beta60 <= 1.50`、`max_drawdown20 <= 0.16`。
- 候选按 `ret5` 排名为主，`lgb` 和流动性只作轻量加分，高波动和高振幅作轻量扣分。
- 最多替换当前 Top5 中 `ret5` 最弱的 3 只，并在诊断中记录 `swap_count`，避免后置 anti-lottery 再把刚换入的冲高篮子冲掉。

### 4. Stress-Chaser Veto Final

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

### 5. Pullback Stable Booster

用途：保留为 shadow/实验补位；当前 runtime priority 不启用。它在原 `pullback_rebound` 没有候选时，给中期仍为正、短期不极端、T 日收盘强度确认、同时流动性和波动可控的票一次低优先级补位机会。

触发边界：

- 只在 riskoff 和 strict pullback 都未接受时生效。
- 候选必须在 Top5 外，并通过 `ret20`、`intraday_ret`、`sigma20`、`amp20`、`downside_beta60`、`max_drawdown20` 和流动性约束。
- 打分偏好低 `ret5` 排名、较高 `ret20` 排名、较高 `intraday_ret` 排名、较好 `lgb` 支持，同时惩罚高波动和高振幅。
- 只替换当前 Top5 中最高风险的一只，并受 `pullback_stable_risk_delta_cap` 约束。

### 6. Conditional Anti-Lottery

用途：在 frozen queue 处理后，替换 Top5 中模型分最低、且并非高下行 beta 保护对象的“彩票型”弱票。

触发边界：

- 仅当当前累计 accepted swaps 少于 `max_total_swaps` 时尝试。
- 替换目标按 `grr_final_score` 最低优先；若目标 `downside_beta60 > 1.35`，直接保护不换。
- 候选必须在 Top5 外，模型排名不低于前 30%，流动性排名不低于前 70%，`sigma20 < 0.050`、`amp20 < 0.10`、`downside_beta60 < 1.40`。
- 候选不能是近 20 日极端涨幅或高点跳升靠前的高彩票票：`max_ret_rank <= 0.55` 且 `max_jump_rank <= 0.60`。
- 候选 `grr_final_score` 必须高于被替换目标。

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

## Timing Leak Research Notes

2026-04-28 追加研究了“业绩公告 + 高开阳线”类策略。关键结论：

- 若策略在 `T` 日开盘买入，不能使用 `T` 日 `close > open` 作为筛选条件；这是同日未来函数。
- 在当前日频框架里，`gap >= 3% 且 close > open` 只能在 `T` 日收盘后确认，因此最早只能按 `T+1` 开盘进入。
- `scripts/post_announcement_reaction_shadow.py` 按这个无未来函数时点重测后，`gap_green_raw / sane / close_strong` 在 60win 事件研究中均为负均值、负 q10。
- 将该信号作为 frozen queue 后置候选也会伤害 delta；反向 veto 没有触发，因为当前 frozen queue 基本没有持有这类过热形态。

因此这类信号暂不接 runtime；只保留为未来公告数据接入时的时点校验模板。

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
