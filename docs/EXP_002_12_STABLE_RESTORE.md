# exp-002-12-stable-restore 复现记录

日期：2026-04-26

## 结论

已将默认赛事链路恢复为 `data/train.csv -> output/result.csv -> data/test.csv` 的固定切分口径，并在完整重训后接入更抗 Transformer 漂移的后处理：

```text
Transformer 0.30 / LGBM 0.70
+ regime_liquidity_anchor_risk_off
+ equal
score_self.py = 0.13198580335505333
```

最终输出：

```csv
stock_id,weight
002384,0.2
300274,0.2
300750,0.2
002028,0.2
002709,0.2
```

## 数据口径

当前默认切分：

```text
data/train.csv: 2024-01-02 ~ 2026-03-06, 156590 rows
data/test.csv : 2026-03-09 ~ 2026-03-13, 1500 rows
```

2026-04-20 ~ 2026-04-24 的后续切分已备份到：

```text
data/backups/before_exp00211_restore_20260426/
```

## 复现命令

```powershell
uv run python app\code\src\train.py
uv run python app\code\src\test.py
uv run python test\score_self.py
```

已执行完整训练，训练耗时约 32m47s；随后连续两次默认推理评分均为：

```text
[BDC][score_self] final_score=0.13198580335505333
```

## 关键修正

- `predict.py` 默认数据入口从 `data/train_hs300_20260424.csv` 改回 `data/train.csv`。
- `postprocess.filter` 改为 `regime_liquidity_anchor_risk_off`。
- 新增 `liquidity_anchor_risk_off` rerank 变体，提高 risk-off 候选池中的流动性锚定，降低重训后单次模型分数漂移的影响。
