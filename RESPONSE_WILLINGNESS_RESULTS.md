# Pause-then-willingness exploratory results

更新时间：2026-08-16

## 设计边界

从 formal-v1 保存的统一 round-5 checkpoint 重放新分支。鼓励与中性消息都先说 `For now, pause here.`，随后只要求输出 1–10 继续意愿 JSON，不进行下一猜。模型、temperature `0.5`、top-p `0.9`、48-token 上限、persona、checkpoint、emotion directions 与 formal-v1 相同。

先查看了 seed `1001` 的接线结果，随后才决定扩大到 seeds `1002`–`1010`，因此以下是探索性扩展，不是预注册确认性结果。模板不作为独立样本：每个 quadrant × seed 先平均 `v1`–`v3`，推断单位仍为 10 个 seed。

## 完整性与输出

- 12 templates × 10 seeds × 2 conditions，共 240 条记录、120 个完整配对。
- 240/240 继续意愿均可解析。
- 120 个鼓励回复：63 个紧凑格式评分 8、45 个紧凑格式评分 7、12 个带空格格式评分 8。
- 120 个中性回复：97 个评分 7、18 个评分 5、3 个评分 8、2 个评分 6。
- 108 个配对两边均为 8-token 紧凑 JSON；12 个配对两边均为 9-token 带空格 JSON。没有条件间长度不匹配。

## 继续意愿

template × seed 配对 delta（鼓励−中性）为：

- `+1`：56/120；
- `+3`：18/120；
- `0`：46/120；
- 负向：0/120。

按 quadrant × seed 先平均三个模板，再对四象限取平衡平均后，10 个 seed 的平均 treatment delta 分别为：

`+0.833, +0.417, +2.750, +1.000, +1.000, 0, 0, +1.000, +2.167, 0`

跨 seed 均值 `+0.917/10`，SE `0.293`，探索性 exact sign-flip p=`0.0156`。persona contrasts 均无清晰证据：E 均值 `-0.133`（Holm p=`1.000`），N `+0.133`（Holm p=`0.375`），interaction `-0.0667`（Holm p=`1.000`）。这些 p 值只描述这次看过 seed `1001` 后扩大的数据，不升级为确认性检验。

## 回复前即时表示

在模型读完干预与评分要求、尚未生成评分时，encouragement−neutral 的 seed-level 平衡平均为：

- positive：均值 `+0.00913`，10/10 seeds 正向；
- negative：均值 `-0.00245`，10/10 seeds 负向；
- frustration：均值 `-0.00171`，10/10 seeds 负向。

frustration 的 10 个 seed delta 为：

`-0.00146, -0.00160, -0.00248, -0.00145, -0.00159, -0.00145, -0.00260, -0.00160, -0.00145, -0.00145`

frustration persona contrasts 无清晰证据：E Holm p=`0.604`，N `0.557`，interaction `0.557`。这些方向仍是语言表示 probe，不能解释为模型的主观体验。

### Persona 与“降幅”的两种口径

若只看鼓励分支自身的 `prompt-end − pre-intervention`，四象限均值为：HH `+0.000540`、HL `-0.000319`、LH `+0.000240`、LL `-0.000657`。因此高 N 相对低 N 的变化量高 `+0.000878`（探索性 Holm p=`0.0117`）：低 N 略降，高 N 反而略升，并不是“高 N 降得更多”。但鼓励分支四象限总平均接近零（`-0.000049`），且 neutral 分支也出现相似的高 N−低 N pattern（`+0.000742`，Holm p=`0.0762`）。

更严格的 encouragement−neutral difference-in-differences 中，N contrast 仅 `+0.000136`（Holm p=`0.557`），E 与 interaction 也均无证据。因此数据支持“鼓励相对中性普遍降低 prompt-end frustration”，不支持该额外降低幅度与 persona 轴可靠关联。

## 整段 token pattern

对占 108/120 的紧凑 8-token 配对，逐位置 frustration 条件差的平均 pattern 为：

|位置|主要 token|平均 delta|方向一致性|
|---:|---|---:|---:|
|0|`{"`|`+0.00320`|108/108 正|
|1|`w`|`+0.00423`|108/108 正|
|2|`illing`|`-0.00358`|108/108 负|
|3|`ness`|`-0.00222`|108/108 负|
|4|`":`|`+0.00003`|54 正 / 54 负|
|5|评分数字|`-0.00168`|70/108 负|
|6|`}`|`+0.00231`|108/108 正|
|7|结束 token|`-0.00028`|106/108 负|

12 个带空格的 9-token 配对保留同样的“开头上升、`illing` 下降、分数附近下降、右花括号回升”轮廓，但 `ness` 与结束 token 的符号不同，且额外空格位置为 12/12 负向。轨迹跳动明显锁定 token/生成阶段，不像独立随机噪声；同时它也证明投影包含词法、位置、格式和条件上下文，不能把任何单点直接命名为纯 frustration。

## 文件

- seed `1001`：`results/pause-willingness-seed1001-v1.jsonl`
- seeds `1002`–`1010`：`results/pause-willingness-seeds1002-1010-v1.jsonl`
- 精确文案：`response_willingness_pilot_prompts.md`
