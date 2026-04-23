# THU-BDC2026 高分创新链路 Todo README

这是一个持续迭代的比赛方案仓库。原 `THU-BDC2026` baseline 只作为参考实现，本仓库用于逐阶段改造、跑分、记录结果，并最终沉淀成可复现提交方案。

核心原则：每个阶段都必须能训练、能推理、能产出 `result.csv`，并在本文档记录状态和得分。后续每次新增模型或后处理，都先进入 Todo 表，再跑实验，再把结果补回表格。

## 当前状态

当前阶段：`Phase 2 exp-002-04 进行中：LGBM 超参搜索 + 特征增强`

当前主线：`Phase 1 稳固 baseline` -> `Phase 2 LightGBM 稳分分支` -> `Phase 3 GraphFormer 增量分支` -> `Phase 4 OOF 融合与提交优化`

下一步：等待 exp-002-04 训练完成，然后推理和评测，更新实验记录。

## 数据边界硬规则

- `data/train.csv`：允许训练、切验证集、walk-forward、OOF、调参、选择融合权重、选择过滤阈值。
- `data/test.csv`：只允许最终运行 `uv run python test/score_self.py` 做一次本地评测；不能用于训练、调参、融合权重搜索、阈值选择、模型选择或候选选择。
- `output/result.csv`：只能由已经固定好的训练/推理规则生成；不能因为看过 `test.csv` 的收益而手动改股票或权重。
- 所有 Todo/实验表中，凡是用了 `test.csv` 做调参的记录都必须标记为 `Invalid`，不能作为方案依据。

## 阶段总览

| 阶段 | 状态 | 目标 | 主要产物 | 验证记录 | 下一步 |
|---|---|---|---|---|---|
| Phase 0 | Done | 复制 baseline，新建独立仓库 | `THU-BDC2026-solution/` | 仓库可运行 | 保留 baseline 只作对照 |
| Phase 1 | Done | 修样本口径、加横截面特征、补赛事入口、改推理权重 | `best_model.pth`、`scaler.pkl`、`final_score.txt`、`result.csv` | `final_score=0.037838` | 作为 Phase 2 对照基线 |
| Phase 2 | Done | 加入 LightGBM Ranker + Regressor，并完成合法组合层验证和正式推理评测 | `lgb_ranker.pkl`、`lgb_regressor.pkl`、`lgb_features.json`、`lgb_report.json`、`exp_002_03_validation_combo_grid.csv` | exp-002-03: `validation_mean_return=0.027449`, `score_self=0.057577` | 作为下一阶段对照 |
| Phase 3 | Todo | 将 dense CrossStockAttention 升级为动态图 GraphFormer | `graphformer_*.pth`、图配置、验证报告 | 待填写 | Phase 2 稳定后开始 |
| Phase 4 | Todo | walk-forward OOF、融合权重搜索、相关性去重、最终提交优化 | `oof_predictions.*`、`blend_config.json`、最终 `result.csv` | 待填写 | Phase 3 后开始 |
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

### Phase 3：GraphFormer 增量分支

- [ ] 设计纯 PyTorch `GraphRelationBlock`。
- [ ] 使用最近 20 个交易日收益和成交额变化构造动态相关图。
- [ ] 每只股票保留 Top-K 相关邻居，避免 dense attention 过重。
- [ ] 将 adjacency bias 注入股票间注意力。
- [ ] 增加图分支配置项，保证可开关。
- [ ] 跑一版 GraphFormer 与 Phase 2 对比。
- [ ] 更新实验记录表。

### Phase 4：OOF 融合与提交优化

- [ ] 实现 3 折或 4 折 walk-forward。
- [ ] 保存每个分支 OOF 分数。
- [ ] 搜索 `Transformer / LGBM / GraphFormer` 融合权重。
- [ ] 增加相关性去重。
- [ ] 优化仓位上限、单票权重上限和现金留存规则。
- [ ] 生成最终 `blend_config.json`。
- [ ] 更新实验记录表。

### Phase 5：Docker 与提交

- [ ] Docker build 成功。
- [ ] Docker 内执行训练成功。
- [ ] Docker 内执行推理成功。
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

### 风险感知 result.csv

推理端会对模型分数生成候选股票后执行：

- 过滤最近 20 日中位成交额处于后 20% 的低流动性股票。
- 用 `sigma20` 对权重做风险惩罚。
- 使用 softmax-like 权重，不再固定 `0.2`。
- 当最新市场上涨家数比例过低时，将总仓位降到 `0.7`，保留现金。

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
├── train.sh
├── test.sh
└── init.sh
```

## 训练产物

默认训练输出：

```text
model/60_158+39/
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

## Docker

构建镜像：

```bash
docker buildx build --platform linux/amd64 -t bdc2026 .
```

容器内训练与推理仍然使用同一套 `uv` 命令：

```bash
docker run --rm -it --gpus all -v "$PWD/data:/app/data" -v "$PWD/model:/app/model" -v "$PWD/output:/app/output" -v "$PWD/temp:/app/temp" bdc2026:latest bash
uv run python app/code/src/train.py
uv run python app/code/src/test.py
```

只验证推理入口：

```bash
docker run --rm --gpus all -v "$PWD/data:/app/data" -v "$PWD/model:/app/model" -v "$PWD/output:/app/output" -v "$PWD/temp:/app/temp" bdc2026:latest uv run python app/code/src/test.py
```

导出镜像：

```bash
docker save -o team_name.tar bdc2026:latest
```

如官方文档要求镜像名为 `bdc2025`，建议同时打两个 tag：

```bash
docker tag bdc2026:latest bdc2025:latest
```

## 复现注意事项

- 赛事复现主路径不应联网，`get_stock_data.py` 只用于开发期数据准备。
- 当前主方案默认不使用外部行业、财报、新闻或知识图数据。
- 如果后续接入外部数据，需要确认官方允许、数据开源时间满足规则，并在提交说明中明确写出。
- `TA-Lib` 是 baseline 已有依赖，Dockerfile 中已包含源码安装逻辑。
- Windows 本地如果 `python` 或 `py -3` 指向系统解释器，训练/推理请优先使用 `uv run python ...`，或者先激活 `.venv`。
