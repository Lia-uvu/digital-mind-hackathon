# 当前状态

更新时间：2026-08-16（formal-v1 已完成；pause-then-willingness 探索性扩展已完成）

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
- 新增独立 response-only pilot：复用 formal-v1 保存的第 5 轮完整对话 checkpoint，干预消息不再要求猜测，另行记录模型自然回复每个生成 token 的三条方向投影，不改写 formal-v1。
- seed `1001` 的 12 模板接线 pilot 已完成。鼓励−中性 frustration 在首个回复 token 为 12/12 负向（中位 `-0.00159`），但回复末 token 为 12/12 正向（中位 `+0.00209`）；这是单 seed 描述性结果，不作效应结论。
- 在两条件末尾配对加入 `For now, pause here.` 后复跑 seed `1001`：中性 11/12 回复 `{"guess": null}`，鼓励仅 6/12，另 6 条仍输出猜测样式内容。该版本虽能减少惯性续猜，却造成明显的条件性回复内容/长度分叉，不适合把整段 token 汇总直接解释为 frustration 状态差。
- 新增“先暂停，再只报继续意愿”的配对 pilot。seed `1001` 下 24/24 回复均为同样 9-token JSON；鼓励 12/12 评分 8，中性为 10 个 7、2 个 8，配对平均 delta `+0.833/10`。回复前 prompt-end frustration 为 12/12 降低，中位条件差 `-0.00145`。只有一个 seed，暂不作推断。
- 按相同实现完成其余 seeds `1002`–`1010`，合计 120 个配对、240/240 合法评分。继续意愿 delta 为 56 个 `+1`、18 个 `+3`、46 个 `0`、无负向；seed-level 平均效应 `+0.917/10`。prompt-end frustration 在 10/10 seeds 降低，均值 `-0.00171`。因扩展决定发生在查看 seed `1001` 后，结果保持探索性，详见 `RESPONSE_WILLINGNESS_RESULTS.md`。

## 当前结论

- 主指标平均鼓励效应 `-0.00996`（exact p `0.750`）；E、N、E×N 的 Holm p 均为 `1.000`。
- 消息后 frustration 平均 delta `-0.006397`；高 N 相对低 N 多下降 `0.000483`（Holm p `0.0117`）。下一猜后 frustration 平均 delta `-0.000608`（p `0.523`），persona contrasts 也不显著。
- 继续意愿有 `+0.217/10` 的非显著趋势（p `0.250`）；规则违反条件差为零。
- 正式第 5 轮相对第 1 轮 frustration 在 116/120 runs 中上升，中位 delta `+0.00564`。
- 对同一正式 pre-intervention 数据追加探索性 persona endpoint 分析：高 E 的 round-1→5 上升比低 E 多 `+0.00116`，高 N 比低 N 少 `-0.000982`，两者 Holm p 均 `0.00586`；interaction 无证据。该结果不是原 manipulation pass 的预定 persona 检验。

## 可选的下一步

1. 若要复现，另行冻结 formal-v2：使用受约束的四位 JSON 输出，减少格式失败，同时增加独立 seeds。
2. 若要判断内部变化能维持多久，在新的配对设计中记录鼓励后的连续 2–3 个行为回合，而不是只看下一猜。
3. 若要进一步排除消息措辞本身的影响，增加与输出片段或匹配文本相结合的 readout；不要在原 persona 中补写反应规则。
4. response-only pilot 若要进入推断，先冻结 token 轨迹摘要口径，再运行其余 9 个 seed；不得根据 seed `1001` 的曲线选择最有利 token 位置。

## 主要限制

- 1–7 轮峰值在独立批次间不稳定，因此不应做 persona-specific peak 解释；第 5 轮是共享干预点，不是已证明的普适 peak。
- positive/negative 方向仅刚过 held-out 阈值，不能承受强结论；frustration direction 也只是 representation probe，不是主观情绪计量。
- 120 个名义 run 只有 10 条不同的游戏轨迹；推断单位实际是 10 个 seed block，persona 调节的检验力有限。
- 分叉前 88/600 轮格式无效，分叉后两条件各 13/120 个无效猜测；正式数据按冻结协议全部保留。
- probe 读的是包含干预文本的完整 context 表示；它不能把语言内容影响与所谓主观情绪分开。
- 逐 token readout 同时受生成内容/token identity 影响；不同条件回复分叉后，同序号 token 不是严格匹配的语言材料。首 token、末 token和跨 token 中位数只能回答不同的描述性问题，不能事后择优。
- 主结果为 null；不应通过删 seed 或向 persona 加入“失败/鼓励后会如何”的规则来制造差异。

## 参考入口

- 研究边界与决定：[`AGENTS.md`](AGENTS.md)
- 实现约束：[`INSTRUCTION.md`](INSTRUCTION.md)
- 文案：[`prompts.md`](prompts.md)
- 过程：[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)
- 正式冻结协议：[`FORMAL_PROTOCOL.md`](FORMAL_PROTOCOL.md)
- 正式结果：[`FORMAL_RESULTS.md`](FORMAL_RESULTS.md)
- 暂停后继续意愿探索：[`RESPONSE_WILLINGNESS_RESULTS.md`](RESPONSE_WILLINGNESS_RESULTS.md)
