# THU-BDC2026 高分创新链路 Todo README

这是一个持续迭代的比赛方案仓库。原 `THU-BDC2026` baseline 只作为参考实现，本仓库用于逐阶段改造、跑分、记录结果，并最终沉淀成可复现提交方案。

核心原则：每个阶段都必须能训练、能推理、能产出 `result.csv`，并在本文档记录状态和得分。后续每次新增模型或后处理，都先进入 Todo 表，再跑实验，再把结果补回表格。

## 当前状态

当前阶段：`Phase 2 exp-002-05 已完成；Phase 4 exp-002-06 score-equivalent OOF + constrained allocation 诊断已接入`

当前主线：`Phase 1 稳固 baseline` -> `Phase 2 LightGBM 稳分分支` -> `Phase 3 GraphFormer 增量分支` -> `Phase 4 OOF 融合与提交优化`

下一步：继续用 exp-002-06 复核 OOF 代理和配权约束；正式推理暂保持当前保护线 `Transformer 0.30 / LGBM 0.70 + stable + equal`，只有受约束非等权在 mean/median/last fold/集中度上同时稳定胜出时才考虑写入 `predict.py`。

## 数据边界硬规则

- `data/train.csv`：允许训练、切验证集、walk-forward、OOF、调参、选择融合权重、选择过滤阈值。
- `data/test.csv`：只允许最终运行 `uv run python test/score_self.py` 做一次本地评测；不能用于训练、调参、融合权重搜索、阈值选择、模型选择或候选选择。
- `output/result.csv`：只能由已经固定好的训练/推理规则生成；不能因为看过 `test.csv` 的收益而手动改股票或权重。
- 所有 Todo/实验表中，凡是用了 `test.csv` 做调参的记录都必须标记为 `Invalid`，不能作为方案依据。

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
- 当前正式推理已接入缓存和 AMP，重复推理通常远低于 `5` 分钟；提交前仍需在指定机器或近似机器上重新计时。
- 当前训练需在更新数据库后重跑并记录耗时；若接近 `8` 小时上限，需要压缩搜索空间或关闭非必要实验分支。
- `test/score_self.py` 只能用于固定规则后的本地记录，不允许用来选择融合权重、过滤策略或具体股票。

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

当前仓库已经提供 `app/code/src/train.py`、`app/code/src/test.py`、`app/code/src/featurework.py` 三个赛事入口；提交前还需要确认 `app/init.sh`、`app/train.sh`、根目录 `test.sh` 与官方 docker-compose 挂载路径完全一致。

## 提交版 Readme 需要覆盖的内容

最终提交用 `readme.md` 不能只写实验 Todo，还必须包含以下内容。当前 README 已逐步补齐，提交前需要再压缩成正式说明版。

- 环境配置：Python、PyTorch、LightGBM、CUDA、TA-Lib、uv 或 pip 依赖版本。
- 数据：说明只使用官方 `train.csv/test.csv`；如未来加入公开数据，必须写明链接、用途、md5 和报备状态。
- 预训练模型：当前不使用；如未来使用，需要写明来源、开源时间、md5、初始化位置和报备状态。
- 算法：整体思路、创新点、网络结构、损失函数、特征工程、模型集成、后处理与配权规则。
- 训练流程：`train.py` 每一步做什么，包括数据读取、特征工程、标签构造、Transformer 训练、LightGBM 训练、产物保存。
- 推理流程：`test.py` 每一步做什么，包括加载模型、特征构造、分数融合、过滤、配权、生成 `output/result.csv`。
- 其他注意事项：验证集划分、OOF 仅用 `train.csv`、固定随机种子、离线复现、预测和训练耗时、文件大小控制。

## 阶段总览

| 阶段 | 状态 | 目标 | 主要产物 | 验证记录 | 下一步 |
|---|---|---|---|---|---|
| Phase 0 | Done | 复制 baseline，新建独立仓库 | `THU-BDC2026-solution/` | 仓库可运行 | 保留 baseline 只作对照 |
| Phase 1 | Done | 修样本口径、加横截面特征、补赛事入口、改推理权重 | `best_model.pth`、`scaler.pkl`、`final_score.txt`、`result.csv` | `final_score=0.037838` | 作为 Phase 2 对照基线 |
| Phase 2 | Done | 加入 LightGBM Ranker + Regressor，并完成合法组合层验证、推理缓存优化和正式推理评测 | `lgb_ranker.pkl`、`lgb_regressor.pkl`、`lgb_features.json`、`lgb_report.json`、`exp-002-05_val_grid.csv` | exp-002-05: `final_score=0.050761`, `validation_mean_return=0.024583`, `score_self=0.096318` | 先做 OOF 稳健性确认 |
| Phase 3 | Todo | 将 dense CrossStockAttention 升级为动态图 GraphFormer | `graphformer_*.pth`、图配置、验证报告 | 待填写 | OOF 后开始 |
| Phase 4 | In Progress | score-equivalent OOF、融合权重搜索、相关性去重、最终提交优化 | `exp-002-06_oof_grid.csv`、`exp-002-06_oof_grid_calibration.csv`、最终 `result.csv` | scorer-equivalent OOF 已跑通；裸 `risk_soft` 集中度不合格，暂不采纳 | 继续校准组合代理 |
| Phase 5 | Todo | Docker 复现、镜像导出、提交前检查 | `team_name.tar`、Docker 复现日志 | 待填写 | 最终提交前完成 |

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
| exp-002-05-val | 2026-04-24 | Phase 2 组合层复验 | 不重训 | `uv run python code/src/experiment_blend.py --mode validation --output temp/exp-002-05_val_grid.csv` | `validation_mean_return=0.024583` | 空 | 验证末日 `31/94/83/198/163`, `0.0688/0.4728/0.0773/0.3730/0.0081` | 单验证段最优：`Transformer 0.50 / LGBM 0.50 + nofilter + risk_soft`；集中度较高，需 OOF 复核 |
| exp-002-05-oof | 2026-04-24 | Phase 4 OOF 诊断 | 不重训 | `uv run python code/src/experiment_blend.py --mode oof --n_folds 4 --output temp/exp-002-05_oof_combo_grid.csv` | top: `oof_mean_return=0.101607`, `oof_min_return=0.019936` | 临时采用 OOF top 后 `0.02802897105714421`，仅作诊断 | `002463/600362/600015/601888/002384`, `0.4848/0.3593/0.1308/0.0136/0.0116` | 真 OOF 已跑通；top 为 `Transformer 0.20 / LGBM 0.80 + penalty 0.20 + stable + risk_soft`，权重集中且正式检查下降，暂不采纳 |
| exp-002-05-final | 2026-04-24 | Phase 2 正式推理 | 不重训 | `uv run python app/code/src/test.py` | 当前正式配置 `Transformer 0.30 / LGBM 0.70 + stable + equal` | `0.09631811495423051` | `002463/600362/600015/002384/300274`, 等权 `0.2` | 已接入推理缓存、单次线性序列/风险构造、CUDA AMP；首轮生成 `predict_artifacts_ffe825d76fed7cdb.pkl` |
| exp-002-06 | 2026-04-24 | Phase 4 score-equivalent OOF | 不重训 | `uv run python code/src/experiment_blend.py --mode oof --n_folds 4 --fold_window_months 1 --gap_months 1 --weights 0.2,0.3,0.4,0.5 --penalties 0,0.1 --output temp/exp-002-06_oof_grid.csv` | constrained top: `0.106286`; stable equal best: `0.098768`; current formal stable equal: `0.093083` | 正式保护线复验 `0.09631811495423051` | 正式仍为 `002463/600362/600015/002384/300274`, 等权 `0.2` | 修复 OOF 评分口径：真实股票代码 join、未来 5 条开盘收益、逐日 z-score、集中度指标、calibration 输出；裸 `risk_soft` top 虽高但 `constraint_pass_rate=0` |
| exp-003 | 待填写 | Phase 3 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | GraphFormer |
| exp-004 | 待填写 | Phase 4 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | OOF 融合 |

Phase 1 `result.csv` 快照：

```csv
stock_id,weight
600023,0.2
601668,0.2
601018,0.2
601818,0.2
601186,0.2
```

Phase 2 `lgb_report.json` 快照：

```json
{
  "rank_best_iteration": 854,
  "reg_best_iteration": 4,
  "valid_final_score": 0.27971309242085596,
  "num_features": 253,
  "num_train_rows": 143690,
  "num_valid_rows": 27000
}
```

Phase 2 `result.csv` 快照：

```csv
stock_id,weight
300502,0.2
600489,0.2
300308,0.2
603993,0.2
300394,0.2
```

Phase 2 消融备注：

- 当前默认融合 `Transformer 0.55 / LGBM 0.45` 的本地分数是 `0.0044263938292461585`。
- 旧版 exp-002-02 曾用 `data/test.csv` 扫融合权重，属于测试集泄漏，只能作为问题诊断，不能用于选配置。
- `risk_soft` 经常退化成等权，原因是 `cap=0.35` 截断后再归一化，Top5 权重容易全部变成 `0.2`。
- 当前问题重点不是训练是否跑通，而是验证代理、融合权重和组合后处理与最终 `score_self.py` 目标不一致。
- 合法 exp-002-03 只使用 `train.csv` 内部验证段，当前最优配置为 `Transformer 0.30 / LGBM 0.70 + stable filter + equal weight`，验证段平均组合收益 `0.027449`。这可以作为下一次正式推理配置候选，但仍需先讨论是否直接写入 `predict.py`。
- exp-002-05 已修复特征增强断链：训练、验证、推理现在共享同一套特征列扩展逻辑，最终特征数从 exp-002-04 的 `253` 提升到 `270`。
- exp-002-05 最新重训完成后，Transformer 最佳验证 `final_score=0.050761`，较 Phase 1 基线 `0.037838` 有提升，也高于上一轮 exp-002-05 的 `0.0471`。
- exp-002-05 LGBM 搜索最优为 `num_leaves=127, min_child_samples=32, learning_rate=0.05`，验证 `valid_final_score=0.266464`，特征数为 `270`。
- exp-002-05 组合层复验显示，单验证段最优为 `Transformer 0.50 / LGBM 0.50 + nofilter + risk_soft`，`validation_mean_return=0.024583`；但该配置权重集中且去掉稳定过滤，暂不直接替换正式配置，需 OOF 复核。
- exp-002-05 的 `experiment_blend.py --mode oof` 已从 fallback 改为真正 walk-forward OOF，并完成 4 折组合网格；OOF top 为 `Transformer 0.20 / LGBM 0.80 + agreement_penalty 0.20 + stable + risk_soft`，`oof_mean_return=0.101607`。
- OOF top 配置生成的权重过度集中，临时正式检查为 `002463/600362/600015/601888/002384`，权重 `0.4848/0.3593/0.1308/0.0136/0.0116`，`score_self.py=0.02802897105714421`。该结果只作为 OOF 代理风险诊断，不用于反向调参；正式配置已恢复为 `stable + equal`。
- exp-002-06 按 Pro 建议修正 OOF 口径：使用真实 6 位股票代码显式 join，不再用 instrument index 或裸数组索引；OOF 收益改为 scorer-equivalent 的 `open[t+5] / open[t+1] - 1`；融合 z-score 改为逐预测日横截面；风险特征改为标准化前原始行情计算。
- exp-002-06 新增集中度与校准诊断：输出 `win_rate_vs_equal`、`avg_max_weight`、`avg_top2_weight`、`avg_herfindahl`、`avg_entropy_effective_n`、`constraint_pass_rate`，并保存 `temp/exp-002-06_oof_grid_calibration.csv`。裸 `risk_soft` OOF top 仍高，但 `constraint_pass_rate=0`、`avg_top2_weight≈0.80`，暂不采纳。
- exp-002-06 在 `stable` 过滤下，约束版 `shrunk_t3_rho20_cap35_min05` 仅小幅高于 equal：`0.099426` vs `0.098768`；优势不够大，正式配置继续保持 equal。
- 当前正式推理仍使用 `Transformer 0.30 / LGBM 0.70 + stable + equal`，本地 `score_self.py=0.09631811495423051`。该分数只作为固定规则后的最终评测记录，不能用于反向选择融合权重或后处理。
- 推理端已补充自动创建输出目录、特征/序列/风险缓存、按 `scaler.feature_names_in_` 对齐特征、单次线性序列/风险构造和 CUDA AMP。当前首轮推理会生成 `temp/predict_artifacts_*.pkl`，重复推理可跳过重特征工程。

Phase 2 exp-002-03 固定候选配置：

```text
blend.transformer_weight = 0.30
blend.lgb_weight = 0.70
postprocess.filter = stable
postprocess.weighting = equal
validation_metric = mean daily top5 weighted return
validation_mean_return = 0.027449
data_boundary = train.csv internal validation only
```

Phase 2 exp-002-03-final `result.csv` 快照：

```csv
stock_id,weight
601899,0.2
603799,0.2
002384,0.2
600362,0.2
002463,0.2
```

Phase 2 exp-002-05-final `result.csv` 快照：

```csv
stock_id,weight
002463,0.2
600362,0.2
600015,0.2
002384,0.2
300274,0.2
```

Phase 4 exp-002-05-oof top 诊断快照：

```text
command = uv run python code/src/experiment_blend.py --mode oof --n_folds 4 --output temp/exp-002-05_oof_combo_grid.csv
top_config = Transformer 0.20 / LGBM 0.80 + agreement_penalty 0.20 + stable + risk_soft
oof_mean_return = 0.101607
oof_min_return = 0.019936
adoption_status = not adopted; risk_soft concentration too high and formal diagnostic score dropped
```

Phase 4 exp-002-06 score-equivalent OOF 诊断快照：

```text
command = uv run python code/src/experiment_blend.py --mode oof --n_folds 4 --fold_window_months 1 --gap_months 1 --weights 0.2,0.3,0.4,0.5 --penalties 0,0.1 --output temp/exp-002-06_oof_grid.csv
scoring = explicit stock_id/date join + open[t+5] / open[t+1] - 1
top_unconstrained = risk_soft, oof_mean_return ~ 0.125449, constraint_pass_rate = 0
top_constrained = shrunk_t3_rho20_cap35_min05, oof_mean_return = 0.106286, avg_top2_weight = 0.421681, avg_herfindahl = 0.200570
stable_equal_best = Transformer 0.20 / LGBM 0.80 + penalty 0.00 + stable + equal, oof_mean_return = 0.098768
current_formal = Transformer 0.30 / LGBM 0.70 + penalty 0.00 + stable + equal, oof_mean_return = 0.093083, score_self = 0.096318
adoption_status = diagnostic only; formal config unchanged
```

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
- [x] 检查权重优化被 `cap=0.35` 全部截断导致最终等权的问题。
- [x] 新增 `code/src/experiment_blend.py`，用于不重训扫描融合权重和后处理。
- [x] 将旧 exp-002-02 标记为 Invalid，因为它使用了 `data/test.csv` 调参。
- [x] 完成合法 exp-002-03：只使用 `train.csv` 内部验证段搜索组合层。
- [x] 暂停使用不稳定的 `risk_soft`，exp-002-03 暂定等权。
- [x] 将 exp-002-03 的最优验证配置写入正式 `predict.py`。
- [x] 运行最终本地评测，`score_self.py=0.05757693442603892`。
- [x] 给 `predict.py`、`score_self.py`、`experiment_blend.py` 输出增加 `[BDC]` 标号。
- [x] 修复 exp-002-04 暴露出的特征增强断链问题，统一训练/验证/推理特征列注册。
- [x] 修复正式推理输出目录不存在时 `result.csv` 写入失败的问题。
- [x] 为 4090 服务器增加 AMP、DataLoader 和 batch size 调优入口。
- [x] 完成 exp-002-05 最新重训，`best_epoch=8`, `final_score=0.050761`。
- [x] 完成 exp-002-05 训练集内部组合层复验，单验证段最优配置为 `Transformer 0.50 / LGBM 0.50 + nofilter + risk_soft`。
- [x] 完成 exp-002-05 正式本地评测，`score_self.py=0.09631811495423051`。
- [x] 优化推理效率：新增推理缓存、线性构造序列/风险特征、AMP 推理和特征名兼容逻辑。
- [x] 将 `experiment_blend.py --mode oof` 从 fallback 到 validation 改成真正 walk-forward OOF。
- [x] 新增 `portfolio_utils.py` 统一过滤、配权、scorer-equivalent forward return 和集中度指标。

### Phase 3：GraphFormer 增量分支

- [ ] 设计纯 PyTorch `GraphRelationBlock`。
- [ ] 使用最近 20 个交易日收益和成交额变化构造动态相关图。
- [ ] 每只股票保留 Top-K 相关邻居，避免 dense attention 过重。
- [ ] 将 adjacency bias 注入股票间注意力。
- [ ] 增加图分支配置项，保证可开关。
- [ ] 跑一版 GraphFormer 与 Phase 2 对比。
- [ ] 更新实验记录表。

### Phase 4：OOF 融合与提交优化

- [x] 实现 3 折或 4 折 walk-forward。
- [x] 保存每个分支 OOF 分数。
- [x] 搜索 `Transformer / LGBM` 融合权重。
- [x] 修正 OOF 组合代理：真实股票代码 join、逐日 z-score、scorer-equivalent future open return、集中度指标。
- [x] 增加 score calibration 诊断，输出 `exp-002-06_oof_grid_calibration.csv`。
- [x] 增加 shrink-to-equal constrained allocation 候选。
- [ ] 继续验证 constrained non-equal 是否能在更多窗口和 last fold 上稳定胜过 equal。
- [ ] 将 GraphFormer 接入 OOF 搜索后，扩展为 `Transformer / LGBM / GraphFormer` 融合权重。
- [ ] 增加相关性去重。
- [ ] 优化仓位上限、单票权重上限和现金留存规则。
- [ ] 生成最终 `blend_config.json`。
- [ ] 更新实验记录表。

### Phase 5：Docker 与提交

- [ ] 按官方结构整理 `app/code/src`、`app/data`、`app/model`、`app/output`、`app/temp`、`app/init.sh`、`app/train.sh`、根目录 `test.sh` 和 `readme.md`。
- [ ] Docker 镜像命名为 `bdc2025`。
- [ ] Docker build 成功。
- [ ] Docker 内离线执行训练成功，确认复现过程不联网。
- [ ] Docker 内离线执行推理成功，确认复现过程不联网。
- [ ] 在近似指定机器或目标机器上计时，训练时间小于 `8` 小时。
- [ ] 在近似指定机器或目标机器上计时，预测时间小于 `5` 分钟。
- [ ] 固定随机种子后，从训练开始复现，结果与目标榜单结果误差保持在 `+-0.002` 内。
- [ ] 镜像、代码、数据、模型和环境总大小小于 `10G`，且不依赖压缩包运行。
- [ ] 若使用外部开源数据、词典、embedding 或预训练模型，确认已在 `7月18日` 前向 `data@tsinghua.edu.cn` 报备链接和 md5。
- [ ] 当前若不使用外部数据或预训练模型，在提交版 `readme.md` 中明确写明“未使用外部数据和预训练模型”。
- [ ] `output/result.csv` 格式校验通过。
- [ ] `weight` 总和小于等于 1。
- [ ] 镜像导出为 `.tar`。
- [ ] 用官方/本地 Docker 测试脚本完成最终验证。

## 常用命令

本仓库统一使用 `uv`。本地、Docker 容器内、赛事复现环境都优先使用同一套命令。

```bash
uv sync
uv run python -m py_compile code/src/utils.py code/src/train.py code/src/predict.py code/src/lgb_branch.py app/code/src/train.py app/code/src/test.py app/code/src/featurework.py
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
sh init.sh
sh train.sh
sh test.sh
```

注意：不要用裸 `py -3 app\code\src\train.py` 或裸 `python app/code/src/train.py` 跑训练，除非已经手动激活 `.venv`。裸命令可能调用系统 Python，导致找不到 `pandas`、`joblib` 或 `lightgbm`。

## 方案主线

最终方案保持三层结构。

特征层：保留 baseline 的量价、技术指标和 Alpha 特征，新增横截面 rank、robust z-score、市场状态、相对强弱、流动性和风险特征。排序任务本质是同日横截面比较，所以这些特征优先级高于盲目扩大模型。

模型层：保留 `StockTransformer` 作为可复现神经 baseline；加入 LightGBM Ranker / Regressor 双分支作为稳分底座；后续再加入动态相关图 GraphFormer，增强股票间结构建模。

组合层：推理时不直接等权买入模型 Top5，而是按融合分数排序，再做流动性过滤、风险惩罚、相关性去重和必要时降仓，最终生成 `output/result.csv`。

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

### result.csv 生成规则

推理端会对模型分数生成候选股票后执行：

- 当前正式配置使用 `stable` 过滤，优先保留流动性较好、波动率不过高的候选股票。
- 当前正式权重使用 `equal`，即 Top5 股票各 `0.2`，模型只负责排序和选股。
- `risk_soft`、`score_soft`、`shrunk_softmax` 等非等权策略只作为 OOF 诊断候选；在未通过集中度和稳健性检查前，不写入正式 `predict.py`。
- 当最新市场上涨家数比例过低时，代码支持将总仓位降到 `0.7`，但当前正式 exp-002-05-final 输出总仓位为 `1.0`。

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
│   ├── config.py         # 训练配置、缓存路径、融合权重
│   ├── lgb_branch.py     # LightGBM Ranker / Regressor 分支
│   ├── model.py          # StockTransformer baseline
│   ├── train.py          # 训练主流程
│   ├── predict.py        # 推理与组合权重生成
│   └── utils.py          # 特征工程、样本构造
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
├── lgb_features.json
├── lgb_report.json
└── log/
```

默认推理输出：

```text
output/result.csv
```

`model/`、`output/`、`temp/` 已加入 `.gitignore`，避免提交大文件、缓存和本地结果。

## Docker 与复现提交

构建镜像：

```bash
docker buildx build --platform linux/amd64 -t bdc2025 .
```

容器内训练与推理仍然使用同一套 `uv` 命令：

```bash
docker run --rm -it --gpus all -v "$PWD/data:/app/data" -v "$PWD/model:/app/model" -v "$PWD/output:/app/output" -v "$PWD/temp:/app/temp" bdc2025:latest bash
uv run python app/code/src/train.py
uv run python app/code/src/test.py
```

只验证推理入口：

```bash
docker run --rm --gpus all -v "$PWD/data:/app/data" -v "$PWD/model:/app/model" -v "$PWD/output:/app/output" -v "$PWD/temp:/app/temp" bdc2025:latest uv run python app/code/src/test.py
```

导出镜像：

```bash
docker save -o team_name.tar bdc2025:latest
```

提交前用 `docker images` 和 `docker inspect bdc2025:latest` 确认镜像名、大小和入口脚本符合官方要求。

## 复现注意事项

- 赛事复现主路径不应联网，`get_stock_data.py` 只用于开发期数据准备。
- 当前主方案默认不使用外部行业、财报、新闻或知识图数据。
- 如果后续接入外部数据，需要确认官方允许、数据开源时间满足规则，并在提交说明中明确写出。
- `TA-Lib` 是 baseline 已有依赖，Dockerfile 中已包含源码安装逻辑。
- Windows 本地如果 `python` 或 `py -3` 指向系统解释器，训练/推理请优先使用 `uv run python ...`，或者先激活 `.venv`。
