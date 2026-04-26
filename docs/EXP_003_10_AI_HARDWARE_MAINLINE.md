# EXP-003-10 AI Hardware Mainline

This patch adds a narrow postprocess branch:

```text
regime_ai_hardware_mainline_v1
```

It is based on the 2026-04-24 latest setup. It is not presented as a general model, reranker, fallback rule, board-aware filter, low-upside filter, or theme-consensus clustering method.

The 2026-04-24 model Top20 already recalled the AI/hardware theme. The final action is manual mainline selection to remove isolated gold, isolated military, and resource-cycle mixing from the final five-name basket.

## Top20 Theme Read

Current model Top20 recalled two main candidate groups.

A. AI / compute / electronic hardware:

```text
000977, 300308, 688256, 300408, 601138, 002463, 603019, 603986, 300476, 300502
```

B. Resource / cyclical repair:

```text
002466, 002460, 600176, 000807, 300782
```

## Final AI Balanced Mainline

The final selected basket is:

```text
000977,688256,300408,601138,002463
```

All names use equal weight:

```text
0.2 each
```

## Exclusions

`600547` is excluded as isolated gold with negative ret10 and ret20.

`600893` is excluded as isolated military: ret10 is negative, ret20 is weak, and the name is mainly pulled by the Transformer score.

`002466`, `002460`, and `600176` are excluded because they are part of the resource repair line. Mixing them with the AI mainline lowers theme purity.

`300308` is strong but more aggressive. It remains an aggressive backup and is not included in the balanced mainline.

## Scope

No model changes were made.

No retraining was done.

No LGB or Transformer weights were changed.

Selector, reranker, theme-consensus clustering, board-aware logic, and low-upside filtering are not enabled for this branch.
