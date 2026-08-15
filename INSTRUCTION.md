# Implementation Instruction

本文件是当前黑客松实验的可执行施工说明。研究意图与指标定义以 `AGENTS.md` 为准；实现若需要改变其中的长期实验边界，先停下与 Lia 对齐。

## 目标

在本地开源小模型上运行四象限、每象限三个平行模板的 Mastermind/Absurdle 实验，从同一 checkpoint 成对 fork 鼓励与中性条件，记录：

1. 下一猜的归一化信息效率 delta；
2. 真正的游戏规则违反；
3. 下一猜之后的继续意愿；
4. 同一模型内部 positive、negative 与 frustration-specific 激活方向的时间点 delta。

结论只描述 persona prompt 对模型行为和内部可测表示的调节，不声称模型具有主观情绪。

## 不可破坏的实验边界

### Persona

- 两条轴为外向性高/低与神经质（情绪敏感度）高/低，共四个象限；每象限固定 `v1`–`v3` 三个平行措辞模板，共 12 份 persona 文本。
- 十二份文本只改变这两条轴和等价措辞；姓名、人口统计、知识、能力和游戏经验等其他信息保持相同或完全省略。
- persona 只能写一般特质，让模型自行产生“特质 → 当下反应”的映射。
- 允许：`容易担忧`、`情绪通常平稳`、`安静内敛`、`活跃外向`。
- 禁止：`失败后会沮丧`、`连续犯错会降低表现`、`被鼓励后会振作`、`会因为支持而愿意继续`，以及任何包含本实验任务、失败轮次、鼓励、中性条件、下一猜或继续意愿的反应预测。
- 自动检查只能拦截明显泄漏，不能代替人工语义审查。

### Game

- 候选空间为 `0000`–`9999`，允许重复数字和前导零。
- 反馈为 `(exact, misplaced)`，多重数字按标准 Mastermind 规则计数。
- 每轮按反馈分桶并保留最大桶；并列时依次选择总匹配数更少、exact 更少、tuple 字典序更小者。
- 引擎、反馈和信息效率必须是纯函数或可复制状态；同一 checkpoint 的两分支不得共享可变状态。

### Branching

- 每个独立 run 先生成受挫轨迹，到配置的失败轮数后保存完整对话、候选集与 RNG 信息。
- 鼓励和中性分支从相同 checkpoint 启动，并使用相同 branch seed。
- 两种干预均不得包含解题策略；长度和关注程度尽量匹配。
- 干预后先生成并计分下一猜，再询问继续意愿。

### Emotion representation

- 主测量必须来自被试模型自身 hidden states，不得用外部文本情绪分类器冒充内部情绪向量。
- 当前首选为固定 Git commit 的 `repeng` 与 `Qwen/Qwen2.5-1.5B-Instruct`，PyTorch `float16`、MPS。
- 只读取激活，不向模型注入 steering vector。
- 实验前用冻结的成对材料分别训练 `positive_vs_neutral`、`negative_vs_neutral` 和 `frustration_vs_calm-response` 方向；训练材料不得包含 Mastermind、鼓励或 persona 文本。
- frustration 材料的每一对面对相同的受阻事件，只改变是否出现紧绷、急躁、粗暴重试等反应；场景不得直接写出 frustration 标签。该方向用于读出 frustration-related representation，不解释为主观感受。
- 用独立 held-out 文本验证与主读数相同的“五层 cosine 中位数”聚合分数，目标情绪相对中性的分离准确率至少为 0.75；验证失败的方向不得进入主结果。逐层验证结果也完整保存，但不按单层结果事后挑层。
- 测量游戏前 baseline、干预前、干预消息后、下一猜后；Qwen 预数据 pilot 后固定使用倒数第 5–9 层这一段连续的五层，主分数使用逐层 cosine 投影的中位数，同时保存逐层值。正式数据采集后不得按结果重新挑层。
- `--track-round-emotions` 只用于干预前 manipulation pilot：每轮反馈后读取三条方向；分析器只读取共享 checkpoint 前轨迹，即使 runner 为记录完整性生成了下游分支，也不得查看其结果再决定 checkpoint。原始预注册 check 至少使用 20 个独立 run，要求第 5 轮相对第 1 轮的 pooled median delta > 0、逐轮 slope 中位数 > 0、至少 70% runs 的 delta > 0。
- 本实现是轻量 contrastive RepE probe，不是 Anthropic 171-emotion pipeline 的完整复刻；继续意愿是下游行为评分，不参与任何方向训练。

## 文件与职责

- `prompts.md`：所有可编辑实验文案和 12 个 persona 模板；代码不得另藏 prompt 文本。
- `encouragement_lab/personas.py`：12 个 prompt key 的象限、因子水平与模板 id 注册表；不存放 persona 文案。
- `encouragement_lab/mastermind.py`：引擎、反馈、候选集和效率计算。
- `encouragement_lab/prompt_loader.py`：从 `prompts.md` 读取具名文本，并执行明显泄漏检查。
- `encouragement_lab/model.py`：Qwen 加载、确定性生成和 checkpoint 所需 RNG 管理。
- `encouragement_lab/emotion_probe.py`：三条方向训练、held-out 验证和只读投影。
- `encouragement_lab/manipulation.py`：严格去重成对分支后汇总干预前逐轮 frustration 轨迹与预注册判定。
- `encouragement_lab/factorial.py`：正式数据的三模板 seed-block 汇总、2 × 2 planned contrasts 与 exact sign-flip/Holm 统计。
- `encouragement_lab/records.py`：版本化 JSONL record 与安全追加写入。
- `encouragement_lab/experiment.py`：受挫轨迹、checkpoint、配对分支和指标组装。
- `run_experiment.py`：薄 CLI；参数化模型、失败轮数、seed、输出路径和 dry-run。
- `tests/`：组件测试；边界测试跟随拥有该边界的模块。

保持实现小而直白；不引入数据库、任务队列、Web UI、插件系统或多层配置框架。

当前情绪方向提取使用 PCA，不需要聚类。若后续确实需要对 run 或情绪轨迹分群，先按院子规则阅读 `../bedrock/neroli/AGENTS.md`，检查 Neroli 的聚类算法是否能通过小而明确的公开 contract 复用；没有实际需求时不提前建立跨组件依赖。

## 输出记录最低字段

每条分支记录至少包含：schema version、run/seed/checkpoint、模型与采样配置、persona id、quadrant、template id 与 prompt checksum、condition、完整 transcript、干预前每轮的 guess/feedback/候选数/原始效率/规则违规轨迹、失败轮数、分叉时候选集规模、guess、feedback、原始/最优/归一化效率、规则违规、四时间点逐层情绪投影与汇总值、继续意愿、代码与依赖版本。

正式分析先在每个 template × seed run 内汇总配对 delta，再在同一 quadrant × seed 内平均三个完整模板，最后跨 seed 汇总；模板不作为三倍独立样本。每个模板的敏感性结果同时保留。

正式运行使用 `FORMAL_PROTOCOL.md` 冻结的 seeds 与参数。记录必须保存生成源码 SHA-256 和 emotion direction artifact provenance；已有非空输出默认拒绝追加，只有 metadata 完全一致时才允许 `--resume`。无法解析的继续意愿仅令 willingness delta 缺失，不得导致主信息效率或情绪指标整条丢失。

## 验收

- 对一组覆盖前导零、重复数字和全异数字的代表性 guesses，穷举全部 10,000 个秘密验证反馈合法性；另用精确案例验证重复数字的计数语义。
- 同一状态复制后运行两个分支不会互相改变候选集。
- 对固定状态，最大桶、并列规则、`I`、`I*` 和 `I_norm` 结果确定。
- 12 个 persona 模板按四象限各三份保持平衡，通过结构检查，且人工阅读不含实验反应规则。
- 所有运行时 prompt 均可追溯到 `prompts.md`。
- dry-run 不加载模型即可完成引擎、分支与 JSONL 全流程。
- 模型 smoke test 能生成合法猜测；三条 emotion probe 的 held-out 分离检查及逐轮 frustration manipulation check 通过后，才允许正式采集。

## 分工原则

可委派给 Terra：机械、边界清楚、可用测试完全验收的模块，例如 Mastermind 引擎、穷举测试、JSONL 序列化与文件级单元测试。

主代理保留：persona 与实验文案、干预语义、分叉因果边界、指标口径、模型随机性、内部激活方向训练与验证、最终集成及结果解释。
