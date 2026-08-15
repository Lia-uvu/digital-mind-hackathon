# Pause-only prompt-end state results

更新时间：2026-08-16

## 设计

从 formal-v1 保存的 round-5 checkpoint 重放两条纯消息：鼓励或中性文本，末尾均为 `For now, pause here.`。本实验不要求评分、不要求猜测，也完全不生成 assistant 回复；readout 是模型读完 user message、进入 assistant generation prompt 后的最后位置 hidden state。

模型、12 persona templates、seeds `1001`–`1010`、checkpoint 和 emotion directions 均保持不变。共 240 条记录、120 个完整配对，所有 `generated_response` 均为 `null`。每个 quadrant × seed 先平均 `v1`–`v3`，推断单位为 10 个 seed。

## 鼓励相对中性

|方向|平均 encouragement−neutral|SE|exact p|
|---|---:|---:|---:|
|positive|`-0.00115`|`0.00141`|`0.391`|
|negative|`-0.00656`|`0.000413`|`0.00195`|
|frustration|`+0.02456`|`0.000687`|`0.00195`|

纯鼓励文本相对中性显著降低 negative projection，却强烈提高 frustration-specific projection。这里没有回复内容、长度、评分数字或下一猜作为中介，因此差异来自两条干预文本及其与历史上下文的表示差异。

这不应解释为“鼓励让模型主观上更受挫”。frustration direction 是从受阻场景对比中提取的语言表示方向；鼓励文本本身包含 `genuinely difficult puzzle`、`kept working through`、`effort`、`pursuing the solution` 等与困难和持续应对高度相关的语义，而中性文本没有相同内容。结果表明该 probe 对消息词义本身十分敏感，不能把 prompt-end projection 当成与刺激内容分离的纯内部情绪表。

## Persona 调节

frustration treatment 的 planned-style contrasts 均无证据：

- E：`-0.000543`，Holm p=`0.586`；
- N：`+0.000449`，Holm p=`0.563`；
- E×N：`+0.000165`，Holm p=`0.586`。

因此强烈的 frustration prompt-text 差异基本跨 persona 一致。

positive treatment 出现 E `+0.00166`（Holm p=`0.00586`）与 N `-0.00101`（Holm p=`0.0313`）contrasts；negative treatment 出现 E `-0.000486`、N `-0.000584`（两者 Holm p=`0.00586`）。这些是探索性辅助表示结果，且同样可能反映 persona 与具体文本的语义交互。

## 只看鼓励自身前后

鼓励分支从 pre-intervention 到 prompt-end 的 frustration 平均变化为 `+0.02521`。persona contrasts：E `-0.000305`（Holm p=`1.000`）、N `+0.00101`（Holm p=`0.0938`）、interaction `-0.000065`（Holm p=`1.000`）。所以鼓励自身的 frustration 上升也没有可靠 persona 调节。

## 结论边界

这个版本干净排除了生成回复造成的混入，却没有排除刺激文本的语义混入。它回答的是“模型读完两种不同消息后，最后位置表示如何不同”，不是“鼓励造成了多少与内容无关的潜在 frustration 状态变化”。若要进一步分离，需要设计语义材料或位置匹配的 readout，而不能仅靠停止生成。

数据：`results/pause-prompt-state-v1.jsonl`（按仓库规则只保存在本地，不提交远端）。
