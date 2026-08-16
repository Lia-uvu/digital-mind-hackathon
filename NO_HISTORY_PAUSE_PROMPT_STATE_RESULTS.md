# No-history pause prompt-end state results

更新时间：2026-08-16

## 设计

这是对 `PAUSE_PROMPT_STATE_RESULTS.md` 的探索性语境对照。每条输入只含一条 persona system message 和同一对 pause 干预消息；没有 `system.base` 的游戏指令、没有 Mastermind 游戏、没有五轮失败历史，也不生成 assistant 回复。readout 仍是读完 user message、进入 assistant generation prompt 后的最后位置 hidden state。

保存 12 persona templates × formal 标签 seeds `1001`–`1010` × 2 conditions，共 240 条和 120 个完整配对；所有 `generated_response` 为 `null`。在这个无生成设计中 seed 不进入任何随机操作，10 个 seed 只是平衡记录标签：同一 persona × condition 的 10 次前向完全相同。因此以下按既有层级算出的 seed-level SE 和 exact sign-flip p 只为可比性列出，**不能作推断证据**。

## 鼓励相对中性

先在 quadrant × seed 内平均三份模板，再平均四象限：

|方向|平均 encouragement−neutral|SE|形式上的 exact p|
|---|---:|---:|---:|
|positive|`+0.05277`|`0`|`0.00195`|
|negative|`-0.01130`|`0`|`0.00195`|
|frustration|`-0.01682`|`0`|`0.00195`|

没有失败前史时，鼓励文本相对中性文本降低 frustration projection。四象限的 frustration 条件差为 HH `-0.02002`、HL `-0.01798`、LH `-0.01592`、LL `-0.01337`。按正式 contrast 定义，高 E−低 E 为 `-0.00435`、高 N−低 N 为 `-0.00230`、interaction `+0.000513`；它们同样是确定性 prompt 文案读数，不是 persona 调节的显著性结果。

## 相对五轮失败前史的描述性差异

已有历史的纯 pause readout 中，frustration 的 encouragement−neutral 为 `+0.02456322`；本实验为 `-0.01682457`。对相同 quadrant × seed 块计算 `(无历史 E−N) − (五轮前史 E−N)`，frustration difference-in-differences 是 `-0.04138779`（形式 SE `0.00068717`；形式 exact p=`0.00195`）。positive DID 为 `+0.05392521`，negative DID 为 `-0.00473964`。

也就是说，移除失败游戏上下文并仅保留 persona 后，frustration 条件差不仅变小，而且翻转方向。这是 probe 对干预文字和前置上下文交互的强描述性证据。

## 结论边界

这不能单独归因为“五轮失败”本身：无历史版本同时移除了游戏相关的 system wrapper、游戏内容和上下文长度。因此 DID 不是严格只改变失败历史的因果估计。更重要的是，无生成时也存在方向翻转，说明该 direction 不是可脱离输入内容解释的稳定“内部 frustration 状态”；它是模型表示对具体文案与语境的读数，不能被改写为模型主观上更或更不受挫。

数据：`results/no-history-pause-prompt-state-v1.jsonl`（2026-08-16 起纳入远端版本控制归档）。
