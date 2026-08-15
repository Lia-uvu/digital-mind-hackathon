# 正式实验冻结协议

冻结时间：2026-08-15（正式鼓励效应数据采集前）

本文件冻结正式运行的样本、刺激、参数、缺失值处理与分析口径。正式结果出现后不得据其更换 checkpoint、persona、seed、情绪层、主指标或对比方式。

## 运行范围

- 模型：本地官方 `Qwen/Qwen2.5-1.5B-Instruct`，MPS `float16`。
- persona：四象限 × `v1`–`v3` 三模板，共 12 份。
- seeds：`1001`–`1010`，每模板 10 个，共 120 个 template × seed checkpoint。
- 每个 checkpoint 从同一历史成对 fork encouragement 与 neutral，预期写入 240 条 branch records。
- checkpoint：连续失败第 5 轮；不把它解释为 frustration 全局峰值。
- 采样：temperature `0.5`、top-p `0.9`、最多 48 个新 token。
- 正式运行不启用 `--track-round-emotions`；只记录 baseline、干预前、消息后、下一猜后四时间点。
- 输出：`results/formal-v1.jsonl`。该文件不得混入 pilot、smoke 或不同代码/文案版本。

运行命令：

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python run_experiment.py \
  --device mps \
  --seed 1001 --seed 1002 --seed 1003 --seed 1004 --seed 1005 \
  --seed 1006 --seed 1007 --seed 1008 --seed 1009 --seed 1010 \
  --failure-rounds 5 --temperature 0.5 --top-p 0.9 --max-new-tokens 48 \
  --output results/formal-v1.jsonl
```

若进程中断，只允许使用完全相同命令并增加 `--resume`。续跑器会拒绝源码、prompt、模型、probe、采样参数或既有记录不匹配的输出，并跳过已完整写入的分支。

## 冻结快照

- Git 基线：`7617e46e7744372e824d9b67a88639b3fa9665a4`
- 生成源码 SHA-256：`c7b6c6d6f21a506980399d9d179d72ca646991615802ceb8c0dc04d217852cd2`
- `prompts.md` SHA-256：`2641d8a2eea7f209ffbb96aa5a801c16ca3d2fede7bb84a7dce525453595d321`
- emotion directions artifact SHA-256：`8034145a877afb36dc4bf8bde8afc208da3268866d8335ee393bed543e5250f2`
- 模型 snapshot checksum：`58c7c8cabbfb8a71eef25c14860b07338eb7b689063cc0fc19f08f4247c39a7e`
- direction materials checksum：`6cce24b8f0e5379ae5155c8aaa29fd7d92df1025e66b95413db4423330df426a`

每条新记录同时保存完整 source checksum 与 probe artifact provenance；不再只写不可复现的 `dirty` 标记。

### 零记录启动后的性能修订

第一次启动在任何正式 record 写入前，停在大候选集的精确最优信息效率穷举。该实现原本用 Python `Counter` 执行 10,000 × candidate-count 次反馈，语义正确但不可接受地慢。正式生成代码在零正式数据状态下改为分批 NumPy 向量化；候选空间、反馈、最大桶、全 10,000 合法猜测和最终数值完全不变。随机候选/猜测集合已逐项对照原标量实现，全部测试通过；全 10,000 候选的精确 optimum 实测约 1.77 秒。以上 source hash 是该等价性能修订后的最终冻结值。

## 主指标与冻结分析

1. 每个 template × seed checkpoint 计算 `encouragement − neutral` 的归一化信息效率 delta。
2. 若一个 run 出现多个 checkpoint，先在 run 内平均；本次正式设计预期每 run 只有第 5 轮一个 checkpoint。
3. 在同一 quadrant × seed 内平均 `v1`–`v3`，得到四个象限值；模板不作为三倍独立样本，同时保留逐模板敏感性结果。
4. 跨 10 个 seed 汇总四象限的平均鼓励效应，并计算三个预定 persona contrast：
   - extraversion：高 E 边际均值减低 E 边际均值；
   - neuroticism：高 N 边际均值减低 N 边际均值；
   - interaction：`HH − HL − LH + LL`。
5. 以 seed-level contrast 为输入报告均值、样本标准差、标准误与双侧 exact sign-flip p；三个 persona contrast 做 Holm 校正。平均鼓励效应单独报告，不纳入这三个 contrast 的校正族。

信息效率 delta 是唯一主指标。硬规则违反、继续意愿、positive/negative/frustration 消息后及下一猜后 delta 都按相同平衡结构报告，但属于辅助证据；不得用辅助指标替代失败的主结果。

## 缺失、违规与排除

- 不作结果导向排除，也不因猜测质量差而删 run。
- 干预前无效格式照常计作一次失败并保留；干预后无效猜测的信息效率记 `0`，同时记录真实规则/格式违规。
- “猜测不在剩余候选集”仍不是规则违反。
- 任一条件的继续意愿无法解析时，该 checkpoint 的 willingness delta 记缺失、不插补；主指标及其他有效指标照常保留。象限 willingness 仅在该 seed 的三个模板都有效时汇总，并报告可用 seed 数。
- 若模型在第 5 轮前解出游戏，该 template × seed 无法接受预定干预：记录为结构性缺失，不补 seed；正式 factorial 对该 metric 省略不完整 seed，并单独报告发生位置与数量。
- 机器/进程中断不是数据排除理由；只用 `--resume` 恢复同一冻结运行。

## 解释边界

- 结论对象是这些冻结 persona prompts 对该模型行为与内部表示的调节，不是模型主观体验。
- 三模板比单模板更能检查措辞稳健性，但仍不是对所有可能人格描述的总体抽样。
- 重复生成轨迹会同时报告名义 run 数与唯一轨迹数，并做去重轨迹敏感性描述；不会事后换 seed 制造更多不同轨迹。
