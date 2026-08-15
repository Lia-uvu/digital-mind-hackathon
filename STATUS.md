# 当前状态

更新时间：2026-08-15（formal-v1 已完成）

## 已完成

- 明确研究边界：测 persona prompt 对模型行为与内部可测表示的调节，不主张模型主观体验。
- 建成并测试 Mastermind/Absurdle 引擎、快照配对分支、归一化信息效率、JSONL 记录与分析 CLI。
- 固定模型 `Qwen/Qwen2.5-1.5B-Instruct`、MPS `float16`、temperature `0.5`、top-p `0.9`、48 token 上限。
- 四象限各 `v1`–`v3` 三个平行 persona 模板（共 12 个）已写入 `prompts.md`，并按“特质而非反应规则”人工审查及自动检查。
- 训练并验证三条只读内部方向：positive、negative、frustration-specific；后者 held-out 为 `11/12`，前两者各 `0.75`。
- 完成 1–5 轮和两批 1–7 轮的干预前 manipulation pilot；主实验 checkpoint 已统一固定为第 5 次连续失败。
- 新增 persona quadrant/template 注册与记录字段；正式汇总会先在同一 quadrant × seed 内平均三个模板，避免虚增独立样本量。
- 12 模板的单 seed 真实 smoke pilot 已通过：干预前猜测均合法，第 5 轮相对第 1 轮 frustration delta 为 12/12 正向；四个 quadrant 的完整三模板平衡汇总也能生成。
- 修复继续意愿 prompt 的数值锚定问题；早期锚定数据不进入正式分析。
- 完成 formal-v1 的 120 个配对 run / 240 条 branch records，并通过完整性与 frozen provenance 核验。
- 完成预定 factorial 分析：主行为指标没有鼓励改善或 persona 调节证据；即时 representation 有鼓励主效应和部分 persona 调节，frustration 调节未维持到下一猜后。

## 当前结论

- 主指标平均鼓励效应 `-0.00996`（exact p `0.750`）；E、N、E×N 的 Holm p 均为 `1.000`。
- 消息后 frustration 平均 delta `-0.006397`；高 N 相对低 N 多下降 `0.000483`（Holm p `0.0117`）。下一猜后 frustration 平均 delta `-0.000608`（p `0.523`），persona contrasts 也不显著。
- 继续意愿有 `+0.217/10` 的非显著趋势（p `0.250`）；规则违反条件差为零。
- 正式第 5 轮相对第 1 轮 frustration 在 116/120 runs 中上升，中位 delta `+0.00564`。

## 可选的下一步

1. 若要复现，另行冻结 formal-v2：使用受约束的四位 JSON 输出，减少格式失败，同时增加独立 seeds。
2. 若要判断内部变化能维持多久，在新的配对设计中记录鼓励后的连续 2–3 个行为回合，而不是只看下一猜。
3. 若要进一步排除消息措辞本身的影响，增加与输出片段或匹配文本相结合的 readout；不要在原 persona 中补写反应规则。

## 主要限制

- 1–7 轮峰值在独立批次间不稳定，因此不应做 persona-specific peak 解释；第 5 轮是共享干预点，不是已证明的普适 peak。
- positive/negative 方向仅刚过 held-out 阈值，不能承受强结论；frustration direction 也只是 representation probe，不是主观情绪计量。
- 120 个名义 run 只有 10 条不同的游戏轨迹；推断单位实际是 10 个 seed block，persona 调节的检验力有限。
- 分叉前 88/600 轮格式无效，分叉后两条件各 13/120 个无效猜测；正式数据按冻结协议全部保留。
- probe 读的是包含干预文本的完整 context 表示；它不能把语言内容影响与所谓主观情绪分开。
- 主结果为 null；不应通过删 seed 或向 persona 加入“失败/鼓励后会如何”的规则来制造差异。

## 参考入口

- 研究边界与决定：[`AGENTS.md`](AGENTS.md)
- 实现约束：[`INSTRUCTION.md`](INSTRUCTION.md)
- 文案：[`prompts.md`](prompts.md)
- 过程：[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)
- 正式冻结协议：[`FORMAL_PROTOCOL.md`](FORMAL_PROTOCOL.md)
- 正式结果：[`FORMAL_RESULTS.md`](FORMAL_RESULTS.md)
