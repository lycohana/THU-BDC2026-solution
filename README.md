# THU-BDC2026 代码说明

本项目为 THU-BDC2026 股票 Top5 选股/配权方案。最终推理入口生成 `output/result.csv`，格式为 `stock_id,weight`，最多 5 只股票，总权重不超过 1。当前正式推理数据为 `model/input/train_hs300_latest.csv`，历史行情截至 `2026-04-24`，用于给出下一交易日 `2026-04-27` 的 Top5 组合。该文件随模型一起放在非挂载目录中，避免官方复现时 `data/temp/output` 挂载覆盖。

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
- `model/input/train_hs300_latest.csv`：正式推理输入，覆盖 `2024-01-02` 至 `2026-04-24` 的沪深300历史行情，用于生成 `2026-04-27` 组合。该文件放在非挂载目录，避免官方复现时 `data/temp/output` 挂载覆盖。

当前不使用任何外部公开数据、词典、embedding 或预训练模型，因此当前版本无需外部资源报备。若后续加入外部资源，必须满足开源时间和邮件报备要求，并在本文档补充链接、md5 和用途。

## 预训练模型

当前不使用预训练模型。所有模型均由 `train.csv` 从头训练得到。

## 算法

### 整体思路

方案采用 **"双模型排序打分 + 多专家融合 + 市场 regime 感知风险路由"** 的四阶段结构，完整推理链路为：

```
原始数据 → 特征工程 → Transformer/LGBM 双模型打分
       → RRF 多专家融合 + Hedge 在线路由 → GRR Top5 评分
       → regime_liquidity_anchor_risk_off 候选过滤
       → branch_router_v2b 主题/趋势 overlay（最多换 2 只）
       → supplemental_overlay（riskoff/pullback/追涨否决）
       → 最终 Top5 等权输出
```

### 第一阶段：特征工程与模型训练

#### 特征层

在 baseline 的量价与技术指标基础上，补充横截面 rank、robust z-score、市场状态、相对强弱、动量和成交额变化等特征。当前特征数约 270，入口为 `feature_registry.py`，训练和推理共用同一套扩展逻辑，避免特征不一致。

主要特征类别：

- **基础量价特征**：收盘价变动、成交额、换手率、振幅、K 线形态等。
- **动量特征**：1/3/5/10/20 日收益率。
- **横截面特征**：`cs_rank_*`（当日百分位排名）、`cs_rz_*`（median/MAD robust z-score，clip 到 [-5, 5]）。
- **市场状态特征**：`mkt_ret_1/5`（市场平均收益）、`mkt_breadth_1`（上涨家数比例）、`mkt_dispersion_1`（收益离散度）、`mkt_amt_mean/std`（成交额扩散）。
- **相对强弱特征**：`alpha_rel_1/5`（相对市场强弱）。
- **风险特征**（推理时额外构造，用于路由和过滤）：`sigma20`（20 日波动率）、`amp20`（20 日振幅）、`beta60`（60 日 beta）、`downside_beta60`（下行 beta）、`idio_vol60`（特质波动率）、`max_drawdown20`（20 日最大回撤）、`amt_ratio5/to_ratio5`（成交额/换手率短期变化）。

#### 标签定义

标签是未来第 1 到第 5 个交易日的开盘收益，与赛事 scorer 口径完全对齐：

```python
open_t1 = groupby("股票代码")["开盘"].shift(-1)
open_t5 = groupby("股票代码")["开盘"].shift(-5)
label = (open_t5 - open_t1) / open_t1
```

同时构建质量标签 `quality5`（可选扣除手续费、波动率和回撤惩罚）和离散化相关性分桶 `relevance5`（5 个分位），以及辅助短期标签 `aux1`、`aux3`。

#### Transformer 分支（StockTransformer）

神经网络主干为 `StockTransformer`（`model.py`）：

- 输入形状：`[batch, num_stocks, sequence_length=60, num_features]`
- 时序编码器：标准 TransformerEncoder，3 层，d_model=256，4 头注意力，前馈维度 512
- 特征注意力：对时序输出做加权聚合（Tanh → Linear → Softmax）
- 股票间交互注意力：CrossStockAttention，让同日股票之间相互关注
- 排序分数头：两层 MLP + LayerNorm → 标量分数

损失函数为 `WeightedRankingLoss`，组合 listwise KL 散度损失和 pairwise sigmoid 损失，对 Top-5 样本施加更高权重（`top5_weight=2.0`）。

#### LightGBM 分支

LightGBM 分支包含两个模型（`lgb_branch.py`）：

- **LGBMRanker**：`objective="lambdarank"`，按交易日作为 group/query，优化同日横截面排序。`rank_weight=0.65`。
- **LGBMRegressor**：`objective="regression"`，预测裁剪后的 5 日收益，补充收益强度。`reg_weight=0.35`。

两者分数分别 z-score 后按 0.65/0.35 加权得到 LGB 总分。训练时支持两阶段超参网格搜索（`lgb_search`），先固定 lr 搜索 `num_leaves` 和 `min_child_samples`，再在最优配置附近搜索学习率。

### 第二阶段：多专家融合与 GRR Top5 评分

模型融合采用 Reciprocal Rank Fusion（RRF）+ Hedge 在线路由的多专家系统（`reranker.py`），而非简单的固定权重线性混合。

#### 多专家打分

以 Transformer 分数和 LGB 分数为基础，派生出多种打分变体：

| 变体名 | 组成 |
|---|---|
| `score`（主分） | Transformer 0.30 / LGB 0.70 融合 |
| `lgb` | LGB Ranker 0.65 + Regressor 0.35 |
| `transformer` | Transformer 原始分数 |
| `lgb_top5_score` | Top5-heavy LGBM Ranker（V1，诊断用，`blend_weight=0.0` 不参与融合） |

每个变体作为一路"专家"，通过 union-topK 候选池（各专家 Top-24 并集）和 RRF（k=60）聚合排序，再叠加在线路由加权和风险惩罚得到 `grr_final_score`。

#### Hedge 在线路由器

基于近期市场波动率和上涨宽度，动态调整各专家权重：

- 高波动率（sigma20 分位 > 0.75）：LGB 权重提升（`lgb_risk_off_boost=0.20`）
- 高上涨宽度（breadth > 0.50）：Transformer 权重提升（`transformer_risk_on_boost=0.15`）

路由权重按 Hedge 更新规则衰减：`w_i ← w_i * exp(-η * loss_i)`（`hedge_eta=0.50`）。

#### Tail Guard 崩盘保护

GRR Top5 内置 `tail_guard` 模块，使用当日合法历史特征推断市场崩盘状态：

**崩盘状态检测指标**：
- `market_score`：综合 `breadth_1d`、`breadth_5d`、`median_ret1/5`、`high_vol/amp/dd_ratio` 的加权评分
- `top_fragility_score`：当前 Top-5 候选的波动率、振幅、回撤、极端收益和共识度加权评分
- `risk_off_score = 0.62 * market_score + 0.38 * top_fragility_score`

**崩盘触发条件**（满足任一）：
- `risk_off_score ≥ crash_threshold(0.36)`
- `top_fragility_score ≥ 0.50` 且 `breadth_1d ≤ 0.48`
- Top-5 中高风险股票数 ≥ 2

**崩盘模式下的重排逻辑**：
- 对高风险股票施加风险惩罚：sigma/amp/drawdown 排名 + 低共识度 + 极端收益惩罚
- 对低风险、高共识度股票给予 minrisk_bonus 和 consensus_bonus
- 对同时满足高 sigma、高 amp、高 ret5 且低共识度的股票，通过 veto 直接否决

### 第三阶段：Regime 感知候选过滤

主过滤策略为 `regime_liquidity_anchor_risk_off`（`portfolio_utils.py`），根据当日横截面指标判断市场状态并采用不同过滤路径：

**extreme risk_off 触发条件**（满足全部）：
- `median ret20 < 0`
- `breadth20 < 0.45`
- `median sigma20 > 0.018`
- `dispersion20 > 0.10`

**过滤路径**：
- **触发 extreme risk_off**：取 `stable top60`，再按融合分数 + 流动性 + 动量 + 低波动 + 低振幅的锚定重排分数重排，最后取 Top-5
- **未触发**：回退普通 `stable` 过滤（剔除流动性分位 < 10% 和波动率分位 > 85% 的股票），按原始融合分数排序取 Top-5

### 第四阶段：Overlay 与 Veto（branch_router_v2b）

在候选 Top-5 基础上，通过三层 overlay 机制进行精细化调整（`config['branch_router_v2b']`，`enabled=True`，`max_total_swaps=2`）：

#### Trend Overlay（趋势增强）

当近 20 日市场上涨宽度 ≥ `trend_window_threshold(0.55)`、离散度 ≤ `trend_dispersion_max(0.13)` 且最大振幅分位合理时：
- 从候选池 Top-6 中寻找高分替代者（替换分需领先 ≥ `trend_min_replacement_gap(0.04)`）
- 限制替换后风险增量 ≤ `trend_risk_increase_max(0.20)`
- 最多替换 1 只，且需通过波动率、振幅、回撤分位上限检查

#### Theme AI Overlay（主题增强）

当近 20 日市场上涨宽度 ≥ `theme_ai_window_threshold(0.62)` 时：
- 从候选池 Top-3 中寻找高分替代者（替换分需领先 ≥ `theme_ai_min_replacement_gap(0.08)`）
- 施加冷却保护（`cooling_weak_guard`）：禁止替换共识度过高或近期收益过热的标的
- 最多替换 1 只，且需通过共识度、波动率、回撤等多重风险上限检查

#### Supplemental Overlay（补充层）

在 Trend/Theme overlay 之后，按优先级依次尝试两种补充策略（最多再替换 1 只，`supplemental_overlay_max_swaps=1`）：

1. **riskoff_rank4_dynamic_defensive_target**：当 Top-5 中存在排名靠后的高风险标的时，用排名第 4 档的低风险防御型标的替换
2. **pullback_rebound_highest_risk**：当市场处于短期回调但中期趋势仍在时，用回调反弹标的替换组合中风险最高的一只

#### Stress Chaser Veto（追涨否决）

在所有 overlay 完成后，检查市场是否处于高压力状态，若满足以下条件则否决组合中的追涨型标的：
- `median_ret20 < 0` 且 `breadth20 < 0.50`
- `median sigma20 > 0.018` 且 `dispersion20 > 0.10`
- 最多替换 1 只被否决的追涨标的

### 最终输出

所有 overlay/veto 处理完成后，Top-5 股票等权（各 0.2）输出 `output/result.csv`。若当日市场上涨家数比例过低（breadth < 0.30），总仓位自动降至 0.7。

## 训练流程

训练入口：

```bash
bash app/train.sh
```

主要步骤：

1. 读取 `data/train.csv`，统一股票代码格式和日期格式。
2. 构造技术指标、Alpha 特征、横截面特征和特征增强列（约 270 维）。
3. 构造未来 5 日开盘收益标签、质量标签和离散化相关性分桶。
4. 按最后 2 个月切分训练集和验证集。
5. 标准化特征并保存 `scaler.pkl`。
6. 训练 `StockTransformer`（50 epoch，AdamW，线性学习率衰减），保存最佳 `best_model.pth`。
7. 训练 LightGBM Ranker / Regressor（含两阶段超参搜索），保存 `lgb_ranker.pkl`、`lgb_regressor.pkl`、`lgb_features.json` 和 `lgb_report.json`。
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
bash test.sh
```

主要步骤：

1. 读取 `model/input/train_hs300_latest.csv`，取最新交易日 `2026-04-24` 作为推理基准日，输出下一交易日 `2026-04-27` 组合。
2. 构造与训练一致的特征（约 270 维），并按 `scaler.feature_names_in_` 对齐；构造 60 日输入序列和风险特征帧。
3. 加载 Transformer 产物，计算 Transformer 分数。
4. 加载 LightGBM Ranker / Regressor 产物，计算 LGB 分数（rank 0.65 + reg 0.35）。
5. **GRR Top5 多专家融合**：
   - 取各专家 Top-24 并集作为候选池
   - RRF（k=60）聚合多专家排名
   - Hedge 在线路由器根据当日市场状态动态调整 Transformer/LGB 权重
   - 叠加 sigma/amp/drawdown 风险惩罚
   - Tail Guard 崩盘保护（检测→风险惩罚→高风险否决）
6. **regime_liquidity_anchor_risk_off 过滤**：判断 extreme risk_off 状态，触发时用流动性锚定重排 Top-60，未触发时用 stable 过滤。
7. **branch_router_v2b overlay**（最多换 2 只）：趋势增强 → 主题增强 → 补充 overlay（riskoff/pullback）→ 追涨否决。
8. 选 Top5 等权输出。
9. 写入 `output/result.csv`。

推理端支持缓存和 CUDA AMP：首轮生成 `temp/predict_artifacts_*.pkl`，重复推理可跳过重特征工程。

当前正式输出示例：

```csv
stock_id,weight
002384,0.2
300274,0.2
600015,0.2
601077,0.2
300750,0.2
```

本地固定规则后记录分数：

```text
score_self.py = 0.12018139687305522
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
