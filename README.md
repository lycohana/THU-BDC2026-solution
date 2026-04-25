# THU-BDC2026 代码说明

本项目为 THU-BDC2026 股票 Top5 选股/配权方案。最终推理入口生成 `output/result.csv`，格式为 `stock_id,weight`，最多 5 只股票，总权重不超过 1。

当前根目录 `readme.md` 面向代码审核、Docker 复现和最终提交准备。研发过程、实验流水账和 Todo 记录已移至 [docs/DEV_README.md](docs/DEV_README.md)。

## 环境配置

建议使用 Docker 复现。当前本地开发使用 `uv` 管理 Python 环境。

- Python: `>=3.10,<3.13`
- PyTorch: `>=2.6.0`
- LightGBM: `>=4.3.0`
- pandas: `>=2.3.2`
- scikit-learn: `>=1.7.2`
- TA-Lib: `>=0.6.8`
- joblib / tqdm / tensorboard 等依赖见 `pyproject.toml` 与 `uv.lock`

复现训练和预测时不得联网。提交镜像应包含完整环境、代码、数据、模型和运行脚本。

## 数据

当前方案只使用主办方提供的数据：

- `data/train.csv`：用于训练、验证、walk-forward OOF、融合权重和后处理策略选择。
- `data/test.csv`：只用于固定规则后的最终推理和本地 `score_self.py` 记录，不用于调参、选模型、选股票或改权重。

当前不使用任何外部公开数据、词典、embedding 或预训练模型，因此当前版本无需外部资源报备。若后续加入外部资源，必须满足开源时间和邮件报备要求，并在本文档补充链接、md5 和用途。

## 预训练模型

当前不使用预训练模型。所有模型均由 `train.csv` 从头训练得到。

## 算法

### 整体思路

方案采用“特征增强 + 双分支排序模型 + 固定组合后处理”的结构。

1. 特征层：在 baseline 的 Alpha 与技术指标基础上，补充横截面 rank、robust z-score、市场状态、相对强弱、动量和成交额变化等特征。
2. Transformer 分支：用过去 60 个交易日的股票序列特征学习时序排序信号。
3. LightGBM 分支：使用 `LGBMRanker(objective="lambdarank")` 学习同日横截面排序，并用 `LGBMRegressor` 补充收益强度。
4. 融合层：推理时对 Transformer 分数和 LightGBM 分数分别做横截面 z-score，再按当前正式配置 `Transformer 0.30 / LGBM 0.70` 线性融合。
5. 组合层：当前正式提交保护线为 `stable filter + equal weight`，即模型负责选 Top5，入选 5 只股票等权 `0.2`。

### 方法创新点

- 修复 baseline 样本口径：标签是未来第 1 到第 5 个交易日开盘收益，不再错误要求未来自然日连续。
- 增加横截面特征和市场状态特征，使模型更适合同日排序任务。
- 引入 Transformer 与 LightGBM 双分支融合，兼顾时序表达与稳健横截面排序。
- 推理端支持缓存、特征名对齐和 CUDA AMP，加快复现推理。
- OOF 诊断采用 scorer-equivalent 评价：显式按真实 `stock_id/date` join，并使用 `open[t+5] / open[t+1] - 1` 计算未来 5 条开盘收益。

### 网络结构

神经网络主干为 `StockTransformer`：

- 输入形状：`[num_stocks, sequence_length, num_features]`
- `sequence_length = 60`
- 当前特征数约 `270`
- Transformer 编码股票时序特征，并输出每只股票的排序分数

LightGBM 分支包含：

- `LGBMRanker`：按交易日作为 group/query，优化横截面排序。
- `LGBMRegressor`：预测裁剪后的未来 5 日开盘收益，补充收益强度。

### 损失函数

Transformer 使用排序相关损失，主要优化同日股票之间的相对顺序和 Top5 识别能力。LightGBM Ranker 使用 lambdarank 排序目标，Regressor 使用回归目标拟合裁剪后的收益标签。

标签定义：

```python
open_t1 = groupby("股票代码")["开盘"].shift(-1)
open_t5 = groupby("股票代码")["开盘"].shift(-5)
label = (open_t5 - open_t1) / open_t1
```

### 数据扩增

当前没有使用传统图像式数据扩增，也没有引入外部增强数据。主要增强来自特征工程：

- 横截面 rank 和 robust z-score
- 市场上涨家数比例、市场平均收益、横截面离散度
- 相对市场强弱
- 更多动量和成交额变化特征

### 模型集成

当前正式融合配置：

```text
transformer_weight = 0.30
lgb_weight = 0.70
agreement_penalty = 0.00
postprocess.filter = stable
postprocess.weighting = equal
postprocess.liquidity_quantile = 0.10
postprocess.sigma_quantile = 0.85
```

`risk_soft`、`score_soft`、`shrunk_softmax` 等非等权策略仅作为 OOF 诊断候选。当前正式推理保持等权。`nofilter`、非等权和 stable top30 二阶段重排均做过诊断，未进入正式提交逻辑。

## 训练流程

训练入口：

```bash
uv run python app/code/src/train.py
```

主要步骤：

1. 读取 `data/train.csv`，统一股票代码格式和日期格式。
2. 构造技术指标、Alpha 特征、横截面特征和特征增强列。
3. 构造未来 5 条开盘收益标签。
4. 按时间切分训练集和验证集。
5. 标准化特征并保存 `scaler.pkl`。
6. 训练 `StockTransformer`，保存最佳 `best_model.pth`。
7. 训练 LightGBM Ranker / Regressor，保存 `lgb_ranker.pkl`、`lgb_regressor.pkl`、`lgb_features.json` 和 `lgb_report.json`。
8. 保存训练配置和验证结果。

默认训练产物目录：

```text
model/exp-002-05_60_158+39/
├── best_model.pth
├── scaler.pkl
├── config.json
├── final_score.txt
├── lgb_ranker.pkl
├── lgb_regressor.pkl
├── lgb_features.json
└── lgb_report.json
```

## 推理流程

推理入口：

```bash
uv run python app/code/src/test.py
```

主要步骤：

1. 读取 `data/train_hs300_20260424.csv`，取最新交易日 2026-04-24 作为预测日。
2. 构造与训练一致的特征，并按 `scaler.feature_names_in_` 对齐。
3. 构造每只股票最近 60 个交易日的输入序列。
4. 加载 Transformer 和 LightGBM 产物。
5. 分别计算 Transformer 分数和 LightGBM 分数。
6. 对两个分支分数做横截面 z-score，并按 `0.30 / 0.70` 融合。
7. 使用 `stable` 过滤候选股票，当前阈值为流动性分位 `0.10`、波动率分位 `0.85`。
8. 选 Top5 并等权输出。
9. 写入 `output/result.csv`。

当前正式输出示例：

```csv
stock_id,weight
600015,0.2
601018,0.2
601077,0.2
002384,0.2
300274,0.2
```

本地固定规则后记录分数：

```text
score_self.py = 0.09719554955415999
```

该分数为当前固定提交规则后的本地记录。`data/test.csv` 的未来收益只用于本地评分脚本记录，不进入模型训练、推理特征或提交逻辑。

## 常用命令

本地环境：

```bash
uv sync
uv run python -m py_compile code/src/config.py code/src/train.py code/src/predict.py code/src/lgb_branch.py code/src/portfolio_utils.py app/code/src/train.py app/code/src/test.py app/code/src/featurework.py
uv run python app/code/src/train.py
uv run python app/code/src/test.py
uv run python test/score_self.py
```

Windows PowerShell：

```powershell
uv sync
uv run python app\code\src\train.py
uv run python app\code\src\test.py
uv run python test\score_self.py
```

脚本入口：

```bash
bash app/init.sh
bash app/train.sh
bash test.sh
```

## Docker 与提交准备

镜像名使用 `bdc2026`：

```bash
docker build -t bdc2026:latest .
# 或者：
docker buildx build --platform linux/amd64 -t bdc2026:latest --load .
docker save -o team_name.tar bdc2026:latest
```

本地 Docker Compose 已使用镜像名 `bdc2026:latest`，并按官方结构挂载 `app/data`、`app/output`、`app/temp`：

```bash
docker compose up --no-build app
```

容器内训练与推理：

```bash
bash app/init.sh
bash app/train.sh
bash test.sh
```

提交前需要确认：

- 镜像内训练和推理均可离线运行。
- 预测时间小于 5 分钟。
- 训练时间小于 8 小时。
- 镜像、代码、数据、模型和环境总大小小于 10G。
- 固定随机种子后，从训练开始复现的结果与目标结果误差在 `+-0.002` 内。
- `output/result.csv` 最多 5 行，`weight` 总和不超过 1。

## 官方提交结构

最终提交需按主办方结构整理：

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

当前仓库已经提供赛事入口 `app/code/src/train.py`、`app/code/src/test.py`、`app/code/src/featurework.py`，并补齐 `app/data`、`app/model`、`app/output`、`app/temp`、`app/init.sh`、`app/train.sh` 和根目录 `test.sh`。Dockerfile 会将 `app/model` 同步到容器运行目录 `/app/model`，确保只执行推理时也能加载当前模型产物；从零训练时 `app/train.sh` 会重新生成 `model/` 产物。

## 其他注意事项

- 固定随机种子逻辑在训练流程中执行，提交前需再次从零训练复现确认。
- `model/`、`output/`、`temp/` 为运行产物目录，最终提交镜像需要包含复现所需模型和配置，但不要包含无关实验缓存。
- 当前不使用外部数据和预训练模型；若后续变更，必须更新本文档并按官方要求报备。
- 研发日志和实验记录见 [docs/DEV_README.md](docs/DEV_README.md)。
