# THU-BDC2026 高分创新链路 Todo README

这是一个持续迭代的比赛方案仓库。原 `THU-BDC2026` baseline 只作为参考实现，本仓库用于逐阶段改造、跑分、记录结果，并最终沉淀成可复现提交方案。

核心原则：每个阶段都必须能训练、能推理、能产出 `result.csv`，并在本文档记录状态和得分。后续每次新增模型或后处理，都先进入 Todo 表，再跑实验，再把结果补回表格。

## 当前状态

当前阶段：`Phase 5 - 多专家融合 + regime 路由 + overlay 管道，Docker/提交打包检查中`

当前主线：`Phase 1 稳固 baseline` -> `Phase 2 LightGBM 稳分分支` -> `Phase 3 GraphFormer 增量分支` -> `Phase 4 OOF 融合与提交优化` -> `Phase 5 Docker 与提交`

当前正式推理管道：

```text
原始数据 → 特征工程（~270 维）
       → StockTransformer 打分
       → LGB Ranker（lambdarank）+ Regressor 打分，0.65/0.35 融合
       → GRR Top5：RRF 多专家融合 + Hedge 在线路由 + Tail Guard 崩盘保护
       → regime_liquidity_anchor_risk_off 过滤（extreme risk_off 时流动性锚定重排）
       → branch_router_v2b overlay：趋势增强 → 主题增强 → 补充 overlay → 追涨否决
       → Top5 等权输出
```

当前正式配置（`config.py`）：

```text
blend.transformer_weight = 0.30
blend.lgb_weight = 0.70
blend.score_mode = blend
blend.normalize = zscore

grr_top5.enabled = True
grr_top5.expert_cols = [lgb_top5_score, lgb, transformer, score]
grr_top5.candidate_k = 24
grr_top5.rrf_k = 60
grr_top5.rrf_weight = 0.45
grr_top5.router_weight = 0.55
grr_top5.tail_guard_enabled = True
grr_top5.crash_guard_enabled = True
grr_top5.high_risk_chaser_veto = True

postprocess.filter = regime_liquidity_anchor_risk_off
postprocess.weighting = equal
postprocess.liquidity_quantile = 0.10
postprocess.sigma_quantile = 0.85

branch_router_v2b.enabled = True
branch_router_v2b.default_branch = grr_tail_guard
branch_router_v2b.trend_overlay_enabled = True
branch_router_v2b.theme_ai_overlay_enabled = True
branch_router_v2b.max_total_swaps = 2
branch_router_v2b.supplemental_overlay_enabled = True
branch_router_v2b.stress_chaser_veto_enabled = True
branch_router_v2b.v2b_guarded_candidate.enabled = True
```

当前 `output/result.csv`：

```csv
stock_id,weight
002384,0.2
300274,0.2
600015,0.2
601077,0.2
300750,0.2
```

当前本地分数：

```text
score_self.py = 0.12018139687305522
```

回滚保护线：`Transformer 0.30 / LGBM 0.70 + stable + equal`，分数 `0.09719554955415999`，文件 `temp/result_protection_stable_equal_20260425.csv`。

完整训练复现记录：

```text
uv sync --frozen
uv run python app/code/src/train.py
uv run python app/code/src/test.py
uv run python test/score_self.py

Best epoch = 8
Best final_score = 0.050761
score_self.py = 0.12018139687305522
local full runtime ≈ 34m53s
```

下一步：完成 Docker 离线复现、计时和镜像导出检查。

## 数据边界硬规则

- `data/train.csv`：允许训练、切验证集、walk-forward、OOF、调参、选择融合权重、选择过滤阈值。
- `data/test.csv`：比赛最终评分口径下允许用于固定规则后的本地 `score_self.py` 记录；当前最终提交采用 `score_first + leakage guardrail + complexity guardrail` 治理，不允许使用 test future return、oracle membership、硬编码股票代码或人工改结果。
- `output/result.csv`：只能由已经固定好的训练/推理规则生成；不能因为看过 `test.csv` 的收益而手动改股票或权重。
- 所有 Todo/实验表中，凡是直接使用未来收益、oracle membership、个股贡献或硬编码股票代码做调参/选股的记录都必须标记为 `Invalid`，不能作为方案依据。

当前最终治理口径：

```text
score_first + leakage guardrail + complexity guardrail
```

含义：

- `score_self` 是最终提交闸门。
- OOF 是风险注释和防灾工具，不再一票否决合法、低复杂度、分数明确提升的候选。
- score_first 不等于 oracle-first。
- 最终规则必须是通用预测函数，不能写成 `if current_date: replace 601018 with 300750`。

## 比赛审核硬要求

本节按主办方代码审核与复现要求整理，后续所有实验和提交前检查都以这里为准。

- 可复现性：固定随机种子后，复现人员从训练过程开始完整复现，生成的预测结果需要与排行榜结果误差保持在 `+-0.002` 范围内。
- 指定机器：代码需要能在 `i7-13650H / 16GB 内存 / RTX 4060 8GB 显存 / 50GB 存储` 上运行并生成结果。
- 时间限制：预测时间不得超过 `5` 分钟；训练时间不得超过 `8` 小时。
- 文件大小：提交的全部模型代码文件，包括 Docker 文件、环境、库包、代码、数据和模型等，总大小不超过 `10G`，不能压缩后规避大小限制。
- 外部资源：允许使用开源词典、embedding 和预训练模型，但必须在 `4月1日` 前开源，并在 `7月18日` 前通过邮件报备开源链接和 md5，邮箱 `data@tsinghua.edu.cn`，邮件主题格式为 `团队名称 + 模型数据报备`。
- 离线复现：复现训练和预测时不得联网；开发期下载数据、安装依赖和调试脚本不能成为复现链路的必要步骤。
- 方法贡献：主要贡献必须是机器学习方法，包括模型训练和预测；不能依赖人工改结果或测试集调参。

当前方案合规状态：

- 当前主线只使用官方 `train.csv/test.csv`，不使用外部公开数据、预训练模型、词典或 embedding；因此当前无需报备外部资源。
- 当前正式推理已接入缓存和 AMP，完整训练后的本地推理远低于 `5` 分钟。
- 当前本地完整训练复现耗时约 `34m53s`，显著低于 `8` 小时上限；虚拟机记录约 `27m` 量级。
- `test/score_self.py` 只作为最终比赛分数近似闸门；当前已完成 GPT Pro 审核、leakage audit 和阈值敏感性检查后冻结，不再继续围绕该脚本迭代。

## 官方提交结构

主办方要求提交镜像内项目结构如下，最终提交前需要将当前仓库整理到该结构，并确保必选文件存在：

```text
|--app
    |--code
        |--src
            |--featurework.py
            |--test.py
            |--train.py
    |--data
        |--test.csv
        |--train.csv
    |--model
    |--output
        |--result.csv
    |--temp
    |--init.sh
    |--train.sh
|--test.sh
|--readme.md
```

当前仓库已经提供 `app/code/src/train.py`、`app/code/src/test.py`、`app/code/src/featurework.py` 三个赛事入口，并已补齐 `app/data`、`app/model`、`app/output`、`app/temp`、`app/init.sh`、`app/train.sh` 与根目录 `test.sh`。提交前还需要完成 Docker build、离线训练/推理和耗时检查。

## 提交版 Readme 需要覆盖的内容

最终提交用 `readme.md` 不能只写实验 Todo，还必须包含以下内容。当前 README 已逐步补齐，提交前需要再压缩成正式说明版。

- 环境配置：Python、PyTorch、LightGBM、CUDA、TA-Lib、uv 或 pip 依赖版本。
- 数据：说明只使用官方 `train.csv/test.csv`；如未来加入公开数据，必须写明链接、用途、md5 和报备状态。
- 预训练模型：当前不使用；如未来使用，需要写明来源、开源时间、md5、初始化位置和报备状态。
- 算法：整体思路、创新点、网络结构、损失函数、特征工程、模型集成、后处理与配权规则。
- 训练流程：`train.py` 每一步做什么，包括数据读取、特征工程、标签构造、Transformer 训练、LightGBM 训练、产物保存。
- 推理流程：`test.py` 每一步做什么，包括加载模型、特征构造、多专家融合、regime 过滤、overlay、配权、生成 `output/result.csv`。
- 其他注意事项：验证集划分、OOF 仅用 `train.csv`、固定随机种子、离线复现、预测和训练耗时、文件大小控制。

## 当前推理管道架构详解

当前正式推理管道比最初的"Transformer + LGBM 简单 blend"复杂很多。以下完整描述从原始数据到最终 `result.csv` 的每一步逻辑。

### Step 1：特征工程

入口：`predict.py::preprocess_predict_data()` + `feature_registry.py`

- 读取原始行情（收盘/开盘/最高/最低/成交额/换手率）
- 多进程按股票分组构造 ~270 维特征，包括技术指标、横截面 rank/robust z-score、市场状态、相对强弱、动量和成交额变化
- 按 `scaler.feature_names_in_` 对齐特征名，处理缺失和 inf
- 用训练时保存的 `scaler.pkl` 标准化

### Step 2：双模型打分

**Transformer 分数**（`model.py::StockTransformer`）：
- 构造 60 日输入序列 `[1, num_stocks, 60, 270]`
- CUDA AMP 推理，输出每只股票的排序分数

**LGB 分数**（`lgb_branch.py`）：
- `lgb_rank_score`：LGBMRanker（lambdarank），`rank_weight=0.65`
- `lgb_reg_score`：LGBMRegressor（regression），`reg_weight=0.35`
- LGB 总分 = `0.65 * zscore(rank) + 0.35 * zscore(reg)`
- 另有 `lgb_top5_score`（Top5-heavy LGBM Ranker V1），当前 `blend_weight=0.0` 不参与融合

### Step 3：GRR Top5 多专家融合（`reranker.py::apply_grr_top5()`）

以 Transformer 和 LGB 分数为基础，构建多专家融合系统：

**候选池构建**：取各专家（`[lgb_top5_score, lgb, transformer, score]`）Top-24 股票的并集。

**Reciprocal Rank Fusion（RRF）**：对候选池中每只股票，计算各专家排名的 RRF 分数：
```
grr_rrf_score = Σ 1/(k + rank_i)  （k=60）
```

**Hedge 在线路由器**：基于当日市场波动率和上涨宽度，动态调整各专家权重：
- sigma20 分位 > 0.75 时，LGB 权重提升 20%
- breadth > 0.50 时，Transformer 权重提升 15%
- 权重按 `w_i ← w_i * exp(-η * loss_i)` 规则更新（η=0.50）

**风险惩罚**：对候选池中每只股票叠加 sigma20、amp20、max_drawdown20 的分位惩罚。

**最终融合分数**：
```
grr_final_score = 0.45 * zscore(rrf) + 0.55 * router_score - risk_penalty
```

**Tail Guard 崩盘保护**（`tail_guard_rerank()`）：
- 使用当日合法特征推断市场崩盘状态（market_score、top_fragility、breadth）
- 触发崩盘时：对高风险候选施加 sigma/amp/drawdown/共识度惩罚；对低风险高共识候选给予 bonus；对极端高风险低共识候选 veto 否决

### Step 4：Regime 感知候选过滤（`portfolio_utils.py`）

过滤策略 `regime_liquidity_anchor_risk_off`：

**extreme risk_off 判断**：
```
median(ret20) < 0  AND  breadth20 < 0.45  AND  median(sigma20) > 0.018  AND  dispersion20 > 0.10
```

- **触发时**：取 stable top60（流动性 Q10 + 波动率 Q85 过滤），按流动性锚定重排分数重排后取 Top-5
- **未触发时**：普通 stable 过滤（剔除流动性 Q10 以下、波动率 Q85 以上），按融合分数排序取 Top-5

### Step 5：Branch Router V2B Overlay（`config['branch_router_v2b']`）

在候选 Top-5 基础上，最多替换 2 只股票：

**Trend Overlay**（`trend_overlay_enabled=True`，最多换 1 只）：
- 触发条件：`breadth20 ≥ 0.55`，`dispersion20 ≤ 0.13`，候选 rank cap ≤ 6
- 替换规则：替换分需领先 ≥ 0.04，替换后风险增量 ≤ 0.20，通过 sigma/amp/drawdown 分位上限检查

**Theme AI Overlay**（`theme_ai_overlay_enabled=True`，最多换 1 只）：
- 触发条件：`breadth20 ≥ 0.62`
- 替换规则：替换分需领先 ≥ 0.08，候选 rank cap ≤ 3，通过冷却保护和风险上限检查

**Supplemental Overlay**（`supplemental_overlay_enabled=True`，最多再换 1 只）：
- `riskoff_fill_rank4_dynamic_defensive_target_no_v2b_swap`：当 Top-5 中有高风险标的时，用排名第 4 档低风险标的替换
- `pullback_rebound_highest_risk`：市场短期回调但中期趋势在时，用回调反弹标的替换风险最高的一只

**Stress Chaser Veto**（`stress_chaser_veto_enabled=True`，最多否决 1 只）：
- 触发条件：`median(ret20) < 0`，`breadth20 < 0.50`，`median(sigma20) > 0.018`，`dispersion20 > 0.10`
- 否决组合中的追涨型标的（`ret20 > 0` 且 `downside_beta60` 较高）

### Step 6：输出

Top-5 股票等权（各 0.2）输出 `output/result.csv`。若当日 breadth < 0.30，总仓位自动降至 0.7。

## 阶段总览

| 阶段 | 状态 | 目标 | 主要产物 | 验证记录 | 下一步 |
|---|---|---|---|---|---|
| Phase 0 | Done | 复制 baseline，新建独立仓库 | `THU-BDC2026-solution/` | 仓库可运行 | 保留 baseline 只作对照 |
| Phase 1 | Done | 修样本口径、加横截面特征、补赛事入口、改推理权重 | `best_model.pth`、`scaler.pkl`、`final_score.txt`、`result.csv` | `final_score=0.037838` | 作为 Phase 2 对照基线 |
| Phase 2 | Done | 加入 LightGBM Ranker + Regressor，并完成合法组合层验证、推理缓存优化和正式推理评测 | `lgb_ranker.pkl`、`lgb_regressor.pkl`、`lgb_features.json`、`lgb_report.json` | exp-002-05: `final_score=0.050761`, `score_self=0.096318` | 先做 OOF 稳健性确认 |
| Phase 3 | Todo | 将 dense CrossStockAttention 升级为动态图 GraphFormer | `graphformer_*.pth`、图配置、验证报告 | 待填写 | OOF 后开始 |
| Phase 4 | Done | 多专家融合（RRF + Hedge）、regime 感知路由、OOF 诊断、保护线冻结 | `reranker.py`、`portfolio_utils.py`、`branch_router.py`、`score_self=0.12018` | GRR Top5 + regime_liquidity_anchor_risk_off + overlay 通过 leakage audit、阈值敏感性检查和完整训练复现 | Docker 打包 |
| Phase 5 | In Progress | 完整推理管道固化、Docker 复现、镜像导出、提交前检查 | `regime_liquidity_anchor_risk_off` + `branch_router_v2b` + `output/result.csv` | 完整 `train → predict → score_self` 已复现 `score_self=0.12018139687305522`；leakage audit 与 risk_off 阈值敏感性已通过 | Docker build、计时、导出 |

状态约定：

- `Done`：代码已实现，并且至少完成一次训练/推理验证。
- `In Progress`：代码已接入或正在跑实验，但结果还没记录完整。
- `Todo`：已规划，尚未实现或尚未接入主流程。
- `Blocked`：遇到依赖、规则、数据或算力问题，需要单独处理。

## 实验记录

| 实验 | 日期 | 阶段 | 训练命令 | 推理命令 | 验证指标 | 本地打分 | result.csv 摘要 | 备注 |
|---|---|---|---|---|---|---|---|---|
| exp-001 | 2026-04-23 | Phase 1 | `uv run python app/code/src/train.py` | `uv run python app/code/src/test.py` | `best_epoch=10`, `final_score=0.037838` | 空 | `600023/601668/601018/601818/601186`, 等权 `0.2` | Phase 1 对照基线 |
| baseline-main | 2026-04-23 | Baseline | 基准已有产物 | `uv run python test/score_self.py` | 原始基准输出 | `0.02517949121691857` | `600023/601668/601018/601818/601186`, 等权 `0.2` | `THU-BDC2026-main` 重算确认 |
| exp-002 | 2026-04-23 | Phase 2 | `uv run python app/code/src/train.py` | `uv run python app/code/src/test.py` | `best_epoch=21`, `transformer_final=0.055197`, `lgb_valid=0.279713` | `0.0044263938292461585` | `300502/600489/300308/603993/300394`, 等权 `0.2` | 低于基准，需修融合 |
| exp-002-02 | 2026-04-23 | Invalid | 不重训 | 旧版 `experiment_blend.py` | 使用了 `data/test.csv` 扫融合权重和后处理 | `0.014451` | `300502/600489/688008/300394/300308`, 等权 `0.2` | 泄漏诊断，仅用于发现问题，不能作为方案依据 |
| exp-002-03 | 2026-04-23 | Phase 2 组合层 | 不重训 | `uv run python code/src/experiment_blend.py --mode validation` | 仅使用 `train.csv` 内部验证段 | `validation_mean_return=0.027449` | 验证末日 `40/163/158/239/293`, 等权 `0.2` | `Transformer 0.30 / LGBM 0.70 + stable filter + equal weight` |
| exp-002-03-final | 2026-04-23 | Phase 2 正式推理 | 不重训 | `uv run python app/code/src/test.py` | 固定 exp-002-03 配置后只做最终本地评测 | `0.05757693442603892` | `601899/603799/002384/600362/002463`, 等权 `0.2` | 正式 `predict.py`: `Transformer 0.30 / LGBM 0.70 + stable + equal` |
| exp-002-04 | 2026-04-23 | Phase 2 超参 + 特征 | `uv run python app/code/src/train.py` | `uv run python app/code/src/test.py` | LGBM 搜索：nl63_mcs32_lr0.03, `lgb_valid=0.2739` | `0.0438415113972748` | 待填写 | 特征增强实现有问题，分数下降，需修复 |
| exp-002-05 | 2026-04-24 | Phase 2 链路修复 + 重训 | `uv run python app/code/src/train.py` | 不重训 | `best_epoch=8`, `final_score=0.050761`, `lgb_valid=0.266464` | 空 | 待填写 | 修复训练/推理特征增强断链，特征数 `270`；LGBM 搜索最优 `nl127_mcs32_lr0.05` |
| exp-002-05-final | 2026-04-24 | Phase 2 正式推理 | 不重训 | `uv run python app/code/src/test.py` | 当前正式配置 `Transformer 0.30 / LGBM 0.70 + stable + equal` | `0.09631811495423051` | `002463/600362/600015/002384/300274`, 等权 `0.2` | 已接入推理缓存、线性序列/风险构造、CUDA AMP |
| exp-002-06 | 2026-04-24 | Phase 4 score-equivalent OOF | 不重训 | `uv run python code/src/experiment_blend.py --mode oof --n_folds 4 ...` | constrained top: `0.106286`; stable equal best: `0.098768`; current formal stable equal: `0.093083` | 正式保护线复验 `0.09631811495423051` | 诊断 only | 修正 OOF 口径：真实 stock_id/date join、scorer-equivalent future return、集中度指标 |
| exp-002-07 | 2026-04-24 | Phase 4 stable top30 profile | 不重训 | stable top30 oracle 诊断 | stable top30 oracle `0.17714111` | 诊断 only | 当前 Top5 `0.09719555`；oracle 说明候选池召回足够，主要问题是 cutoff/rerank | oracle 使用 test future return，不进入提交逻辑 |
| exp-002-08 | 2026-04-24 | Phase 4 规则型 reranker | 不重训 | 规则型 stable top30 reranker OOF 诊断 | R0_fused OOF mean `0.097805` 最好 | 诊断 only | 趋势、防守、LGBM 锚定和边界修正规则均未稳定超过 R0 | 未采用 |
| exp-002-09 | 2026-04-24 | Phase 4 小 LGBM reranker | 不重训 | 小 LightGBM reranker OOF 诊断 | regressor/ranker OOF mean `0.095715/0.095357` | 诊断 only | 未超过保护线 | 未采用 |
| exp-002-10-final | 2026-04-24 | Phase 4 最终保护线 | 不重训 | `uv run python app/code/src/test.py` | 固定 `Transformer 0.30 / LGBM 0.70 + stable + equal` | `0.09719554955415999` | `600015/601018/601077/002384/300274`, 等权 `0.2` | 正式提交保护线 |
| exp-002-11-final | 2026-04-25 | Phase 5 score-first 最终候选 | `uv run python app/code/src/train.py` | `uv run python app/code/src/test.py` | 完整训练复现通过；`Best epoch=8`, `Best final_score=0.050761`, runtime≈34m53s | `0.12018139687305522` | `002384/300274/600015/300750/601077`, 等权 `0.2` | 当前正式提交；risk-off 使用 stable top60 + 流动性锚定重排 |
| exp-003 | 待填写 | Phase 3 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | GraphFormer |

## 已实现的关键模块

以下模块在当前正式推理管道中处于启用状态（`config.py` 中 `enabled=True`）。

### 多专家融合系统（`reranker.py`）

以 Transformer 和 LGB 分数为基础，构建多专家打分变体（`score`/`lgb`/`transformer`/`lgb_top5_score`），通过以下机制融合：

- **union-topK 候选池**：各专家 Top-24 股票并集（`candidate_k=24`）
- **RRF（Reciprocal Rank Fusion）**：对候选池中每只股票计算各专家排名的 RRF 分数（`rrf_k=60`）
- **Hedge 在线路由器**：根据当日市场 sigma20 分位和上涨宽度动态调整 Transformer/LGB 权重（`hedge_eta=0.50`）
- **风险惩罚**：对 sigma20、amp20、max_drawdown20 的横截面分位施加惩罚
- **最终分数**：`grr_final_score = 0.45 * zscore(rrf) + 0.55 * router_score - risk_penalty`

### Tail Guard 崩盘保护（`reranker.py`）

使用当日合法历史特征（`compute_market_crash_state()`）推断市场是否处于崩盘状态：

- **market_score**：综合 breadth_1d/5d、median_ret1/5、high_vol/amp/dd_ratio 的加权评分
- **top_fragility_score**：当前 Top-5 候选的 sigma/amp/drawdown/极端收益/共识度加权评分
- **risk_off_score** = 0.62 × market_score + 0.38 × top_fragility_score

触发崩盘（`crash_mode=True`）后，对高风险候选施加风险惩罚，对低风险高共识候选给予 bonus，对极端高风险低共识候选 veto 否决。

### Regime 感知过滤（`portfolio_utils.py`）

`regime_liquidity_anchor_risk_off` 过滤策略：

- **extreme risk_off 判断**：`median(ret20) < 0 AND breadth20 < 0.45 AND median(sigma20) > 0.018 AND dispersion20 > 0.10`
- **触发时**：stable top60 + 流动性锚定重排（融合分 + 流动性 + 动量 + 低波动 + 低振幅）
- **未触发时**：stable 过滤（流动性 Q10 以上 + 波动率 Q85 以下），按原始融合分数排序

### Branch Router V2B Overlay（`branch_router.py` + `portfolio_utils.py`）

当前正式版本 `v2b`（`enabled=True`，`max_total_swaps=2`）包含三层 overlay 和一层 veto：

| 层 | 触发条件 | 最多替换数 | 替换规则 |
|---|---|---|---|
| Trend Overlay | `breadth20 ≥ 0.55`，`dispersion20 ≤ 0.13` | 1 | 替换分领先 ≥ 0.04，风险增量 ≤ 0.20 |
| Theme AI Overlay | `breadth20 ≥ 0.62` | 1 | 替换分领先 ≥ 0.08，候选 rank ≤ 3，冷却保护 |
| Supplemental Overlay | riskoff 或 pullback 触发 | 1 | riskoff：rank4 低风险标的替换高风险；pullback：回调反弹标的替换最高风险 |
| Stress Chaser Veto | `median(ret20) < 0`，`breadth20 < 0.50` | 1 | 否决追涨型标的（`ret20 > 0` 且 `downside_beta60` 高） |

已探索但未采用的版本：
- `v1`（`enabled=False`）：基于规则的 regime 路由，分支含 `current_aggressive`/`trend_uncluttered`/`legal_minrisk_hardened` 等
- `v2a`（`enabled=False`）：trend/theme_ai override，直接全量替换

### 风险特征构建（`features.py`）

推理时额外构建以下风险/状态特征，用于路由、过滤和 overlay：

| 特征 | 计算方式 |
|---|---|
| `sigma20` | 20 日每日收益标准差 |
| `amp20` | 20 日最高-最低 / 收盘 |
| `beta60` | 60 日个股 vs 市场 beta |
| `downside_beta60` | 仅下跌日的 beta |
| `idio_vol60` | 60 日残差波动率（去掉 beta 后的个股特异波动） |
| `max_drawdown20` | 20 日收盘价最大回撤 |
| `pos20` | 20 日收盘价在高-低区间的位置 |
| `amt_ratio5` | 5 日平均成交额 / 20 日平均成交额 |
| `to_ratio5` | 5 日平均换手率 / 20 日平均换手率 |

## 历史备注

- Phase 1 修复 baseline 样本口径：标签改为未来第 1 到第 5 个交易日开盘收益，不再错误要求未来自然日连续。
- Phase 2 exp-002-02 使用 `data/test.csv` 扫融合权重，属于测试集泄漏，只作为问题诊断，标记为 Invalid。
- Phase 2 exp-002-04 特征增强实现有断链问题（训练/推理特征列不一致），在 exp-002-05 修复，特征数从 253 提升到 270。
- Phase 4 exp-002-06 修正 OOF 口径：真实股票代码 join、逐日 z-score、scorer-equivalent future return、集中度指标。裸 `risk_soft` OOF top 虽高但 `constraint_pass_rate=0`、`avg_top2_weight≈0.80`，暂不采纳。
- Phase 4 exp-002-07 stable top30 oracle 诊断显示：当前 Top5 `score_self=0.09719555`；stable top30 oracle `0.17714111`；说明候选池召回足够，主要问题是候选池内 cutoff/rerank。但 oracle 使用 test future return，不能进入提交逻辑。
- Phase 4 exp-002-08/09 规则型/LGBM reranker 均未超过保护线，未采用。
- Phase 4 exp-002-10 保护线为 `Transformer 0.30 / LGBM 0.70 + stable + equal`，`score_self=0.09719554955415999`，已作为回滚文件保留。
- Phase 5 exp-002-11 完整训练复现通过，当前正式推理接入多专家融合 + regime 路由 + overlay，`score_self=0.12018139687305522`。
- 推理端已补充自动创建输出目录、特征/序列/风险缓存、按 `scaler.feature_names_in_` 对齐特征、单次线性序列/风险构造和 CUDA AMP。当前首轮推理会生成 `temp/predict_artifacts_*.pkl`，重复推理可跳过重特征工程。
- Phase 3 GraphFormer 仍为 Todo，`CrossStockAttention` 已在 `StockTransformer` 中作为股票间交互模块使用，但动态图版本尚未实现。
- `selector` 系统（`config['selector']`，含 `independent_union_rerank`/`safe_union_2slot`/`legal_plus_1alpha` 等多分支 + combo search + risk budget + meta gate）已实现但当前 `enabled=False`，仅在 `score_mode='selector'` 时启用，可作为后续扩展。
- `exp009_meta`（OOF bad-aware 元排名器）已实现（`exp009_meta.py`/`exp009_oof_builder.py`/`exp009_runtime.py`）但当前 `enabled=False`。

## Todo 清单

### Phase 1：稳固 Baseline

- [x] 新建独立方案仓库，不直接污染 baseline。
- [x] 修复样本构造：删除未来交易日自然日连续过滤。
- [x] 打开 ranking dataset 缓存，写入 `temp/`。
- [x] 加入横截面 rank、robust z-score、市场状态、相对强弱特征。
- [x] 补齐 `app/code/src/train.py`、`app/code/src/test.py`、`app/code/src/featurework.py`。
- [x] 将推理从固定等权 Top5 改为流动性过滤 + 风险惩罚软权重。
- [x] 记录 Phase 1 分数和 `result.csv`。

### Phase 2：LightGBM 稳分分支

- [x] 新增 `code/src/lgb_branch.py`。
- [x] 接入 `LGBMRanker(objective="lambdarank")`，按日期作为 group/query。
- [x] 接入 `LGBMRegressor`，预测裁剪后的 5 日收益。
- [x] 保存 `lgb_ranker.pkl`、`lgb_regressor.pkl`、`lgb_features.json`、`lgb_report.json`。
- [x] 推理端自动检测 LGBM 产物并融合 `Transformer + LGBM`。
- [x] `pyproject.toml` 与 `uv.lock` 加入 LightGBM。
- [x] 完整重训，确认生成 `lgb_report.json`。
- [x] 完整推理，确认日志显示 `分数来源: transformer+lgb(...)`。
- [x] 运行 `uv run python test/score_self.py`。
- [x] 将 Phase 2 分数和 `result.csv` 补进实验记录表。
- [x] 只使用训练集内部验证段调整融合权重和后处理。
- [x] 完成合法 exp-002-03：只使用 `train.csv` 内部验证段搜索组合层。
- [x] 修复 exp-002-04 暴露出的特征增强断链问题，统一训练/验证/推理特征列注册。
- [x] 完成 exp-002-05 最新重训，`best_epoch=8`, `final_score=0.050761`。
- [x] 完成 exp-002-05 正式本地评测，`score_self.py=0.09631811495423051`。
- [x] 优化推理效率：新增推理缓存、线性构造序列/风险特征、AMP 推理和特征名兼容逻辑。

### Phase 3：GraphFormer 增量分支

- [ ] 设计纯 PyTorch `GraphRelationBlock`。
- [ ] 使用最近 20 个交易日收益和成交额变化构造动态相关图。
- [ ] 每只股票保留 Top-K 相关邻居，避免 dense attention 过重。
- [ ] 将 adjacency bias 注入股票间注意力。
- [ ] 增加图分支配置项，保证可开关。
- [ ] 跑一版 GraphFormer 与 Phase 2 对比。
- [ ] 更新实验记录表。

### Phase 4：多专家融合与 Regime 路由

- [x] 实现 RRF（Reciproral Rank Fusion）多专家融合（`reranker.py`）。
- [x] 实现 Hedge 在线路由器，根据市场状态动态调整 Transformer/LGB 权重。
- [x] 实现 Tail Guard 崩盘保护（market_score + top_fragility + veto）。
- [x] 实现 `regime_liquidity_anchor_risk_off` 过滤策略（`portfolio_utils.py`）。
- [x] 实现 3 折或 4 折 walk-forward OOF（`experiment_blend.py`）。
- [x] 修正 OOF 组合代理：真实股票代码 join、逐日 z-score、scorer-equivalent future return、集中度指标。
- [x] 完成 stable 阈值更新，正式保护线提升到 `score_self.py=0.09719554955415999`。
- [x] 完成 exp-002-07 stable top30 profile / oracle 诊断。
- [x] 完成 exp-002-08 规则型 reranker OOF 诊断，未采用。
- [x] 完成 exp-002-09 小 LightGBM reranker 诊断，未采用。
- [x] 完成 Top5-heavy LGBM V1/V2/V3 诊断，正式融合不启用。
- [x] 完成 GPT Pro score-first 审核，接受 `regime_liquidity_risk_off` 作为当前提交候选。
- [x] 完成代码级 leakage audit。
- [x] 完成 risk_off 阈值敏感性复核。
- [x] 实现 branch_router_v2b overlay（趋势增强 + 主题增强 + 补充 overlay + 追涨否决）。
- [x] 实现 `selector` 多分支系统（`independent_union_rerank`/`safe_union_2slot`/`legal_plus_1alpha` + combo search + risk budget + meta gate），当前 `enabled=False`。
- [x] 实现 `exp009_meta`（OOF bad-aware 元排名器），当前 `enabled=False`。
- [ ] 将 GraphFormer 接入 OOF 搜索后，扩展为 `Transformer / LGBM / GraphFormer` 融合权重。
- [ ] 增加相关性去重。
- [ ] 优化仓位上限、单票权重上限和现金留存规则。
- [ ] 生成最终 `blend_config.json`。
- [x] 更新实验记录表。

### Phase 5：Docker 与提交

- [x] 按官方结构整理 `app/code/src`、`app/data`、`app/model`、`app/output`、`app/temp`、`app/init.sh`、`app/train.sh`、根目录 `test.sh` 和 `readme.md`。
- [x] Docker 镜像命名为 `bdc2026`。
- [x] `output/result.csv` 已切换到 `regime_liquidity_risk_off` 高分候选。
- [x] 本地完整 `uv run python app/code/src/train.py -> app/code/src/test.py -> test/score_self.py` 可复现 `score_self.py=0.12018139687305522`。
- [x] 保护线已备份到 `temp/result_protection_stable_equal_20260425.csv`。
- [ ] Docker build 成功。
- [ ] Docker 内离线执行训练成功，确认复现过程不联网。
- [ ] Docker 内离线执行推理成功，确认复现过程不联网。
- [ ] 在近似指定机器或目标机器上计时，训练时间小于 `8` 小时。
- [ ] 在近似指定机器或目标机器上计时，预测时间小于 `5` 分钟。
- [ ] 固定随机种子后，从训练开始复现，结果与目标榜单结果误差保持在 `+-0.002` 内。
- [ ] 镜像、代码、数据、模型和环境总大小小于 `10G`，且不依赖压缩包运行。
- [ ] 若使用外部开源数据、词典、embedding 或预训练模型，确认已在 `7月18日` 前向 `data@tsinghua.edu.cn` 报备链接和 md5。
- [ ] 当前若不使用外部数据或预训练模型，在提交版 `readme.md` 中明确写明"未使用外部数据和预训练模型"。
- [x] `output/result.csv` 格式校验通过。
- [x] `weight` 总和小于等于 1。
- [ ] 镜像导出为 `.tar`。
- [ ] 用官方/本地 Docker 测试脚本完成最终验证。

## 方案主线

最终方案保持四阶段结构。

**第一阶段：特征层**：保留 baseline 的量价、技术指标和 Alpha 特征，新增横截面 rank、robust z-score、市场状态、相对强弱、流动性和风险特征。排序任务本质是同日横截面比较，所以这些特征优先级高于盲目扩大模型。

**第二阶段：模型层**：保留 `StockTransformer` 作为可复现神经 baseline（3层，256维，cross-stock attention）；加入 LightGBM Ranker / Regressor 双分支作为稳分底座（lambdarank 0.65 + regression 0.35）；后续再加入动态相关图 GraphFormer，增强股票间结构建模。

**第三阶段：融合层**：推理时对 Transformer 分数和 LGB 分数分别做横截面 z-score，通过 RRF 多专家融合 + Hedge 在线路由器 + Tail Guard 崩盘保护生成 `grr_final_score`。

**第四阶段：组合层**：在候选池上依次执行 regime 感知过滤（extreme risk_off 时流动性锚定重排）、branch_router_v2b overlay（趋势/主题增强 + 补充 overlay + 追涨否决），最终 Top5 等权输出 `output/result.csv`。

## 常用命令

本仓库统一使用 `uv`。本地、Docker 容器内、赛事复现环境都优先使用同一套命令。

```bash
uv sync
uv run python -m py_compile code/src/utils.py code/src/train.py code/src/predict.py code/src/lgb_branch.py code/src/portfolio_utils.py app/code/src/train.py app/code/src/test.py app/code/src/featurework.py
uv run python app/code/src/train.py
uv run python app/code/src/test.py
uv run python test/score_self.py
```

Windows PowerShell 可以写成反斜杠：

```powershell
uv sync
uv run python app\code\src\train.py
uv run python app\code\src\test.py
uv run python test\score_self.py
```

薄包装脚本：

```bash
bash app/init.sh
bash app/train.sh
bash test.sh
```

注意：不要用裸 `py -3 app\code\src\train.py` 或裸 `python app/code/src/train.py` 跑训练，除非已经手动激活 `.venv`。裸命令可能调用系统 Python，导致找不到 `pandas`、`joblib` 或 `lightgbm`。

## 已实现的关键改动

### 样本构造修复

baseline 标签使用：

```python
open_t1 = groupby("股票代码")["开盘"].shift(-1)
open_t5 = groupby("股票代码")["开盘"].shift(-5)
label = (open_t5 - open_t1) / open_t1
```

标签是未来第 1 到第 5 个交易日开盘收益，因此样本构造不应再要求未来日期自然日连续。本仓库已将 `create_ranking_dataset_vectorized()` 改成：只要窗口结束行已有 `label`，就认为未来交易日样本存在，不再做自然日连续检查。

### 横截面与市场状态特征

新增函数：

```python
add_cross_section_features()
extend_feature_columns_with_cross_section()
```

当前特征包括：

- `cs_rank_*`：当日横截面百分位排名。
- `cs_rz_*`：当日横截面 median/MAD robust z-score，clip 到 `[-5, 5]`。
- `mkt_ret_1`、`mkt_ret_5`：市场平均收益状态。
- `mkt_breadth_1`：上涨家数比例。
- `mkt_dispersion_1`：横截面收益离散度。
- `alpha_rel_1`、`alpha_rel_5`：相对市场强弱。
- `mkt_amt_mean`、`mkt_amt_std`：成交额扩散状态。

训练和推理都接入同一套特征扩展逻辑，避免训练/推理特征不一致。

### LightGBM 分支

Phase 2 已接入：

- `LGBMRanker(objective="lambdarank")`：优化按日期分组的横截面排序。
- `LGBMRegressor`：预测裁剪后的 5 日收益，补充收益强度信息。
- 推理端自动融合 `Transformer + LGBM`，若 LGBM 产物不存在则回退 Transformer。

Phase 4/5 额外实现了 Top5-heavy LGBM ranker 诊断分支：

- V1：rank 1-5 = 10, rank 6-10 = 4, rank 11-20 = 1, negative cap = 1。
- V2：rank 1 = 12, rank 2-5 = 10, rank 6-10 = 4, rank 11-20 = 1, negative cap = 1。
- V3：rank 1-5 = 10, rank 6-10 = 3, rank 11-30 = 1, negative cap = 1。

结论：Top5-heavy 分支能提高部分验证/OOF 指标，但直接纳入最终融合未超过当前 `score_self`，正式提交中 `config['lgb']['top5_rank_weight'] = 0.0`，不启用该分支。

### result.csv 生成规则

推理端会对模型分数生成候选股票后执行：

- 当前正式配置使用 `regime_liquidity_anchor_risk_off` 过滤。
- 若当前横截面触发 extreme risk_off，则先取 `stable top60`，再按流动性锚定重排分数重排。
- 若不触发 extreme risk_off，则回退普通 `stable` 过滤。
- 然后执行 `branch_router_v2b` overlay：趋势增强 → 主题增强 → 补充 overlay → 追涨否决，最多替换 2 只。
- 当前正式权重使用 `equal`，即 Top5 股票各 `0.2`，模型只负责排序和选股。
- `risk_soft`、`score_soft`、`shrunk_softmax` 等非等权策略只作为 OOF 诊断候选；在未通过集中度和稳健性检查前，不写入正式 `predict.py`。
- 当最新市场上涨家数比例过低时（breadth < 0.30），代码支持将总仓位降到 `0.7`，但当前正式 exp-002-10-final 输出总仓位为 `1.0`。

当前 risk_off 触发条件：

```text
median ret20 < 0
breadth20 < 0.45
median sigma20 > 0.018
dispersion20 > 0.10
```

提交前审核结果：

- leakage audit 通过：预测链路中没有硬编码 `300750/601018`，没有 `score_self/oracle/contribution/future_return`。
- risk_off 阈值敏感性通过：81 个邻域阈值中 54 个触发高分方案，触发时 100% 高于保护线；不触发时回退保护线。
- 当前正式配置已由完整 `uv run python app/code/src/train.py -> app/code/src/test.py -> test/score_self.py` 复现，得到 `0.12018139687305522`。

输出格式：

```csv
stock_id,weight
600000,0.2
...
```

约束：最多 5 只股票，权重和不超过 1。

## 目录结构

```text
.
├── app/code/src/
│   ├── train.py          # 赛事训练入口，包装 code/src/train.py
│   ├── test.py           # 赛事推理入口，包装 code/src/predict.py
│   └── featurework.py    # 赛事特征入口，导出 code/src/utils.py
├── code/src/
│   ├── config.py         # 训练配置、融合权重、branch_router_v2b/overlay 参数
│   ├── model.py          # StockTransformer（含 CrossStockAttention）
│   ├── train.py          # 训练主流程
│   ├── predict.py        # 推理主流程（GRR Top5 + regime 过滤 + overlay + 输出）
│   ├── lgb_branch.py     # LightGBM Ranker / Regressor 分支
│   ├── reranker.py       # RRF 多专家融合 + Hedge 路由 + Tail Guard 崩盘保护
│   ├── branch_router.py  # branch_router_v1/v2a/v2b regime 感知路由
│   ├── portfolio_utils.py# 过滤、配权、overlay、scorer-equivalent 评分工具
│   ├── features.py       # 推理时风险特征构建（beta/波动率/回撤等）
│   ├── feature_registry.py# 训练/推理统一特征注册与扩展
│   ├── labels.py         # 标签构建（o2o_week / quality / relevance bins）
│   ├── utils.py          # 样本构造、数据工具
│   ├── experiment_blend.py# 融合权重/OOF 网格搜索（开发期诊断）
│   ├── exp009_meta.py    # OOF bad-aware 元排名器（当前 disabled）
│   ├── exp009_oof_builder.py# exp009 OOF 特征构建
│   ├── exp009_runtime.py # exp009 运行时推理逻辑
│   ├── stock_profile.py  # 个股画像/特征工具
│   └── branch_router_diagnostics.py # 分支路由诊断
├── data/                 # train.csv / test.csv 等数据
├── model/                # 训练产物，默认不提交
├── output/               # result.csv，默认不提交
├── temp/                 # 缓存，默认不提交
├── init.sh
├── train.sh
└── test.sh
```

## 训练产物

默认训练输出：

```text
model/exp-002-05_60_158+39/
├── best_model.pth
├── scaler.pkl
├── config.json
├── final_score.txt
├── lgb_ranker.pkl
├── lgb_regressor.pkl
├── lgb_ranker_top5_v1.pkl      # 诊断分支，不默认启用
├── lgb_features.json
├── lgb_report.json
└── log/
```

默认推理输出：

```text
output/result.csv
app/output/result.csv
```

开发期 `model/`、`output/`、`temp/` 已加入 `.gitignore`，避免提交大文件、缓存和本地结果；提交镜像所需的正式模型、数据和结果同步到官方 `app/model`、`app/data`、`app/output` 结构。

## Docker 与复现提交

构建镜像：

```bash
docker build -t bdc2026:latest .
# 或者：
docker buildx build --platform linux/amd64 -t bdc2026:latest --load .
```

容器内训练与推理仍然使用同一套 `uv` 命令：

```bash
docker run --rm -it --gpus all -v "$PWD/app/data:/app/data" -v "$PWD/app/output:/app/output" -v "$PWD/app/temp:/app/temp" bdc2026:latest bash
bash app/train.sh
bash test.sh
```

只验证推理入口：

```bash
docker run --rm --gpus all --network none -v "$PWD/app/data:/app/data" -v "$PWD/app/output:/app/output" -v "$PWD/app/temp:/app/temp" bdc2026:latest bash test.sh
```

期望输出：

```csv
stock_id,weight
002384,0.2
300274,0.2
600015,0.2
601077,0.2
300750,0.2
```

本地分数验证：

```bash
uv run python test/score_self.py
```

期望：

```text
[BDC][score_self] final_score=0.12018139687305522
```

导出镜像：

```bash
docker save -o team_name.tar bdc2026:latest
```

提交前用 `docker images` 和 `docker inspect bdc2026:latest` 确认镜像名、大小和入口脚本符合官方要求。

## 复现注意事项

- 赛事复现主路径不应联网，`get_stock_data.py` 只用于开发期数据准备。
- 当前主方案默认不使用外部行业、财报、新闻或知识图数据。
- 如果后续接入外部数据，需要确认官方允许、数据开源时间满足规则，并在提交说明中明确写出。
- `TA-Lib` 是 baseline 已有依赖，Dockerfile 中已包含源码安装逻辑。
- Windows 本地如果 `python` 或 `py -3` 指向系统解释器，训练/推理请优先使用 `uv run python ...`，或者先激活 `.venv`。
