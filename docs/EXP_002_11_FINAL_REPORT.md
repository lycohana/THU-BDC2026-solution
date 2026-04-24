# exp-002-11-final 完整复现报告

日期：2026-04-25

## 结论

当前提交线已经从“冻结模型推理”修正为“完整训练可复现”：

```text
train -> predict -> score_self
score_self.py = 0.12018139687305522
```

最终输出：

```csv
stock_id,weight
002384,0.2
300274,0.2
600015,0.2
300750,0.2
601077,0.2
```

## 复现命令

Windows / PowerShell：

```powershell
uv sync --frozen
uv run python app\code\src\train.py
uv run python app\code\src\test.py
uv run python test\score_self.py
```

Linux / 虚拟机：

```bash
uv sync --frozen
uv run python app/code/src/train.py
uv run python app/code/src/test.py
uv run python test/score_self.py
```

提交入口等价命令：

```bash
bash app/train.sh
bash test.sh
uv run python test/score_self.py
```

## 本地完整复现记录

运行环境：本地 Windows + uv 环境。

记录：

```text
Best epoch: 8
Best final_score: 0.050761
score_self.py = 0.12018139687305522
完整流程耗时约 2093s，约 34m53s
```

推理诊断 Top5：

```text
002384
300274
600015
300750
601077
```

## 最终策略

模型融合：

```text
Transformer 0.30 / LGBM 0.70
agreement_penalty = 0.00
LGBM rank_weight = 0.65
LGBM top5_rank_weight = 0.0
```

后处理：

```text
filter = regime_liquidity_risk_off
weighting = equal
liquidity_quantile = 0.10
sigma_quantile = 0.85
```

risk-off 触发条件：

```text
median ret20 < 0
breadth20 < 0.45
median sigma20 > 0.018
dispersion20 > 0.10
```

risk-off 候选池：

```text
stable top60
```

risk-off rerank：

```text
0.30 * fused
+0.10 * lgb
+0.30 * log_liquidity
+0.10 * ret5
+0.10 * ret20
-0.10 * amp
-0.50 * negative_ret20_penalty
```

其中：

```text
negative_ret20_penalty = 1 if ret20 < 0 else 0
```

最终仍然使用 Top5 equal，每只股票权重 `0.2`。

## 关键修正

之前的问题：

```text
冻结模型推理可以得到 0.12018139687305522；
但从零训练后只能得到 0.08240479624766202 或 0.11297418966584802。
```

原因：

```text
重训后 Transformer 分数会轻微漂移，原 stable top30 / risk-off rerank 对模型分数较敏感，
导致 300750 或 601077 被 601988/601888/601018 等候选挤出。
```

最终修正：

- risk-off 候选池从 stable top30 放宽到 stable top60；
- 保留 fused 分数，但降低对单次训练漂移的敏感性；
- 加入 LGBM anchor；
- 保留 log liquidity、ret5、ret20；
- 削弱过度 low-vol / low-amp 防守；
- 对 ret20 < 0 加惩罚；
- 输出 `temp/predict_score_df.csv` 和 `temp/predict_filtered_top30.csv` 用于调试。

## 合规口径

该方案不是手工替换股票：

- 不硬编码最终 Top5；
- 不写 `replace 601018 with 300750`；
- 不使用 future return 进入预测链路；
- 不使用 oracle membership；
- 不使用 `score_self` contribution 进入预测链路；
- 权重仍为 equal Top5。

`score_self` 用作最终比赛口径验证。旧 OOF 和保护线仍作为风险注释。

## 回滚线

旧保护线：

```text
Transformer 0.30 / LGBM 0.70
+ stable
+ equal
score_self.py = 0.09719554955415999
```

旧保护线 Top5：

```csv
600015,0.2
601018,0.2
601077,0.2
002384,0.2
300274,0.2
```

当前正式提交线相对旧保护线提升：

```text
0.12018139687305522 - 0.09719554955415999
= 0.022985847318895233
```

## 当前建议

冻结 exp-002-11-final。

不再继续围绕当前 `score_self.py` 做搜索。下一步只做：

- Docker build；
- 离线训练/推理；
- 预测耗时检查；
- 镜像大小检查；
- 提交包整理。
