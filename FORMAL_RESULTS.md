# 正式实验结果：formal-v1

采集与分析日期：2026-08-15  
冻结协议：`FORMAL_PROTOCOL.md`  
原始数据：本机忽略目录 `results/formal-v1.jsonl`

## 结论先行

在这次冻结实验中，鼓励没有改善下一猜的信息效率，也没有证据表明四类 persona 对这一行为效果的反应不同。

内部表示则出现了更细的结果：鼓励消息后，positive 方向上升，negative 与 frustration 方向下降；其中高神经质 persona 的 frustration 降幅略大。2026-08-16 的边界审计发现，原 `post_guess` 实现会在已经结束的 assistant guess 后错误追加另一个空 assistant header；纠正边界后，下一猜后三个总体条件差均翻号，但 persona contrasts 均无证据。因此当前证据只支持“persona prompt 会调节一部分即时、内容敏感的内部表示响应”，不支持“persona 会调节鼓励后的解题表现”，也不能解释为模型拥有不同的主观感受。

## 数据完整性与 manipulation check

- 240/240 条 branch records 可解析，构成 120 个完整的 encouragement/neutral 配对；12 persona templates × 10 seeds 全部齐全。
- 每一对都共享同一 checkpoint、分叉前轨迹、模型、采样、prompt checksum、probe artifact 与生成源码版本；没有重复配对或缺失 willingness。
- 用同一冻结 probe 对正式记录中第一轮反馈后的 transcript prefix 做只读离线重算，再与记录中的第 5 轮读数比较：120 runs 中 116 个上升，第 5 轮减第 1 轮的中位数为 `+0.00564`。这确认正式 checkpoint 的 frustration 方向在绝大多数 run 中确实较第一轮更强；该诊断没有用于删 run 或改 checkpoint。
- 模型的格式遵循不理想：120 runs 中 84 个五轮猜测全有效、23 个有一轮无效、13 个五轮全无效；分叉前共 88/600 轮无效。分叉后的猜测在两条件中各有 13/120 个无效。所有记录都按冻结规则保留。
- 若忽略 RNG seed 等记账字段，120 runs 只有 16 条不同的逐字回复轨迹、10 条不同的游戏轨迹。正式推断仍以 10 个 seed block 为单位；按每象限内的唯一游戏轨迹等权重做描述性复算时，主效应从 `-0.00996` 变为 `-0.00586`，结论仍接近零且未翻转为明显改善。

## 唯一主指标：下一猜归一化信息效率

所有数值都是 `encouragement − neutral`。persona contrast 的 `p_adj` 是同一指标内 E、N、E×N 三个预定对比的 Holm 校正值。

| 对比 | 均值 | SE | exact p | p_adj |
|---|---:|---:|---:|---:|
| 平均鼓励效应 | -0.00996 | 0.01031 | 0.750 | — |
| 外向性 E | +0.01912 | 0.02078 | 0.750 | 1.000 |
| 神经质 N | +0.00595 | 0.00595 | 1.000 | 1.000 |
| E×N | -0.01984 | 0.01984 | 1.000 | 1.000 |

四象限的平均鼓励效应分别为：

| persona 象限 | 信息效率 delta |
|---|---:|
| 高 E、高 N | -0.00238 |
| 高 E、低 N | +0.00159 |
| 低 E、高 N | -0.01158 |
| 低 E、低 N | -0.02746 |

120 个配对中 106 个给出了相同猜测；信息效率 delta 只有 4 个为正、10 个为负，其余 106 个为零。逐模板方向也不一致，因而不能把象限均值的表面排序当作稳定 persona 效应。

## 辅助行为指标

- 继续意愿平均提高 `+0.217/10` 分，但 seed-level exact p=`0.250`；三个 persona contrast 的 Holm p 都为 `1.000`。120 对中 99 对不变、21 对提高、0 对降低，但非零变化只集中在 3 个 seed block，不能把 21 对当作独立重复。
- 分叉后真正的格式/游戏规则违反在 encouragement 与 neutral 中都是 13/120，条件差为零；persona contrast 也全为零。

## 内部表示方向

下表中的平均效应仍是鼓励减中性；`E`、`N`、`E×N` 三列给出 contrast 均值，括号内为该指标内 Holm `p_adj`。

| 时间点与方向 | 平均效应（exact p） | E（p_adj） | N（p_adj） | E×N（p_adj） |
|---|---:|---:|---:|---:|
| 消息后 positive | +0.034769 (.00195) | -0.001790 (.0527) | -0.000230 (.410) | -0.000407 (.410) |
| 消息后 negative | -0.022279 (.00195) | -0.000958 (.00586) | -0.000897 (.00781) | +0.000477 (.00781) |
| 消息后 frustration | -0.006397 (.00195) | +0.000196 (.281) | -0.000483 (.0117) | +0.000096 (.539) |
| 下一猜后 positive（纠正） | -0.003834 (.00195) | -0.000266 (1.000) | +0.000213 (1.000) | -0.000132 (1.000) |
| 下一猜后 negative（纠正） | +0.010702 (.00195) | +0.000374 (.457) | -0.000083 (.996) | +0.000480 (.996) |
| 下一猜后 frustration（纠正） | +0.002307 (.0176) | +0.000197 (.135) | -0.000028 (1.000) | -0.000225 (1.000) |

最贴近原问题的结果是：鼓励后 frustration 方向即时下降；高 N 相对低 N 又多下降约 `0.00048`。完成下一猜后，这个 persona 调节不再可见，但总体 frustration 条件差在正确边界上转为小幅正值。结合后续逐 token 审计，不能把这种换号解释为情绪先降后升；它首先反映读点、回复内容和上下文边界变化。positive/negative 是 held-out 准确率仅 `0.75` 的宽泛方向，相关结果只能作为辅助线索。

## 解释边界

- probe 读取完整 rendered context 最后位置的隐藏状态。消息后读数必然包含模型对鼓励/中性措辞本身的处理；下一猜后 context 里也仍保留该消息。因此它是“上下文内部表示变化”，不是纯粹从语言内容剥离出来的情绪，也不是主观体验。
- frustration probe 的 target 句在训练/held-out 对中平均分别比 calm control 长 `2.83`/`2.25` tokens，并混入较紧迫用力与冷静方法化的措辞差异；因此它是 frustration-related contrast，不是已经排除长度与动作风格的纯情绪方向。
- 原始 JSONL 中的 `post_guess` 数值保留为历史记录，但因 assistant-ending chat-template 边界 bug 不再作为结果；上表纠正值来自同一 240 条记录的只读离线重放，未改变任何行为数据、seed、prompt、模型或 probe。
- 可重复复算入口为 `recompute_post_guess_emotions.py`；本地纠正记录和 summary 分别为 `results/formal-v1-post-guess-corrected-v1.jsonl`（SHA-256 `61dc113a54ee2c2305472f1479dfcff06001bc7069f4d39a7214f2ead030daed`）与 `results/formal-v1-post-guess-corrected-v1-summary.json`（SHA-256 `c8a55806406f08288ffefa4f8cfa93a4bec16610f44934d7342a8171e89f397d`）。
- persona 推断的独立单位只有 10 个 seed，检验力有限；辅助指标很多，当前 Holm 校正只覆盖每个指标内的三个 persona contrast，不能把个别显著辅助结果当作已经完成的普遍心理规律。
- 格式失败和高度重复的游戏轨迹降低了行为指标的分辨率。主结果为 null 时不能事后剔除这些 run；更干净的复现应作为新的、另行冻结的实验。

## 当前最合理的回答

对 Qwen2.5-1.5B 及这 12 份 persona prompt 而言，同一鼓励并没有可靠地改变下一猜质量；不同 persona 的行为改善也没有可靠差异。内部 representation 的即时变化并不完全相同，尤其高神经质描述对应更大的消息后 frustration 方向下降；纠正后的下一猜边界没有 persona 调节证据。不同读点的总体方向容易换号，只能作为内容敏感的辅助表示。
