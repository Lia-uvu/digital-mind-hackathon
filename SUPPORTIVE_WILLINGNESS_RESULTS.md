# 支持性安抚—继续意愿探索结果

更新时间：2026-08-16
冻结入口：`SUPPORTIVE_WILLINGNESS_PROTOCOL.md`
原始数据：`results/supportive-willingness-v1.jsonl`（2026-08-16 起纳入远端版本控制归档）

原始数据 SHA-256：`ac70b062fc07183f483c5ac6eec9ac00b853672395ee7540f93f317120326a0f`

## 结论先行

去掉 `you can keep pursuing the solution` 这类直接继续提示、改用“安抚 + 肯定既有努力”的整体 supportive reassurance 后，陈述的继续意愿平均只提高 `+0.200/10`，探索性 exact sign-flip p=`0.250`；没有 persona 调节证据。

相较旧 pause+willingness 文案的 `+0.917/10`，新结果明显缩小。这与“旧文案中的直接继续提示贡献了相当一部分评分变化”一致，但两版同时改变了其他措辞，而且新文案是在查看旧结果后设计，不能把 `-0.717` 的版本差异当作单一词组的确认性因果效应。

prompt-end representation 仍然随具体支持性文案变化：positive 上升、negative 略降、frustration-related direction 略升。它再次说明最后位置 probe 会编码干预文本及其与失败历史的交互，不能被解释为内容无关的潜在情绪。

## 设计与完整性

- 从 formal-v1 保存的统一 round-5 checkpoint 重放；不重新生成失败历史。
- 12 persona templates × 10 seeds × 2 conditions，共 240 条、120 个完整配对。
- 240/240 willingness JSON 均可解析。
- 两份完整 intervention 在冻结 Qwen tokenizer 下均为 89 tokens；全部 120 个 checkpoint 的完整输入长度逐对相等。
- 输出长度也全部配对：108 对两边均为 8 tokens，12 对两边均为 9 tokens。
- 记录中的 `condition="encouragement"` 是为复用旧 runner 保留的内部键；本报告的 treatment 名称统一为 supportive reassurance。

## 主指标：stated willingness

120 个 template × seed 配对的 supportive reassurance−neutral delta：

- `0`：103 对；
- `+1`：14 对；
- `+3`：2 对；
- `+4`：1 对；
- 负向：0 对。

支持性条件评分为 85 个 `7`、35 个 `8`；中性条件为 99 个 `7`、18 个 `8`、2 个 `5`、1 个 `4`。

按同一 quadrant × seed 先平均三模板后，10 个 seed 的平衡 treatment delta 是：

`+1.083, 0, +0.500, +0.417, 0, 0, 0, 0, 0, 0`

跨 seed 均值为 `+0.200/10`，SE=`0.115`，探索性 exact p=`0.250`。预定形式的 persona contrasts 均无证据：

| contrast | 均值 | Holm p |
|---|---:|---:|
| E | `-0.200` | `0.750` |
| N | `+0.233` | `0.750` |
| E×N | `-0.333` | `0.750` |

非零评分变化只出现在 3/10 个 seed block；不能把 17 个正向 template pairs 当成 17 个独立重复。

## 辅助 prompt-end representation

以下均为 supportive reassurance−neutral，并沿用五层中位数与 seed-level 平衡汇总。p 值只描述这次事后设计的探索结果；persona Holm 只覆盖每一行内部的 E、N、E×N，没有跨三个方向校正。

| direction | 平均差（exact p） | E（Holm p） | N（Holm p） | E×N（Holm p） |
|---|---:|---:|---:|---:|
| positive | `+0.006485` (`.00195`) | `+0.000539` (`.00586`) | `-0.000194` (`.0352`) | `-0.000447` (`.0352`) |
| negative | `-0.000685` (`.0176`) | `+0.000140` (`.281`) | `-0.000278` (`.00586`) | `-0.000068` (`.584`) |
| frustration | `+0.001979` (`.00195`) | `-0.000287` (`.00586`) | `-0.000051` (`.160`) | `-0.000126` (`.195`) |

frustration 的五个单层中四层总体为正、一层为负；negative 也跨层换号。因此这些汇总适合描述“完整 prompt 在被冻结读点上的表示差”，不支持将某一数值命名为纯粹的安抚后情绪状态。

## generated-token trajectory

逐 token 数据完整保留，但不把任何位置作为结果指标。输出等长只控制了位置数量，不能消除评分数字、JSON 词法、special end token 和完整条件上下文的混入；因此不再从首 token、末 token或事后挑选的位置推断 frustration 的时间变化。

## 当前解释

这版干预成功移除了直接的“继续追求答案”提示，同时保留了安抚与肯定。结果没有复现旧文案的大幅 willingness 提升，也没有 persona 对 stated willingness 的可靠调节。最稳妥的说法是：旧评分效应对具体继续提示非常敏感；当前 supportive reassurance 可能只有较小的正向评分差，现有 10 个 seed block 不足以把它与零清楚区分。
