# 实验过程记录

本记录按时间保存预数据与正式实验的关键决定和结果。

## 2026-08-15：收窄研究问题与实验边界

**事实**

- 研究问题收窄为：在同一个开源小模型中，persona prompt 是否调节鼓励文本对下一猜质量、继续意愿和内部可测表示的影响。
- 明确不把模型输出或向量读数解释为模型的主观体验。
- 采用四位数字 Mastermind 的 Absurdle 式反馈引擎：每轮保留最大合法反馈桶；不同 persona 可以走到不同候选集，鼓励与中性只要求从同一 run 的同一快照配对 fork。
- 主行为指标改为相对当前局面最优猜测归一化的信息效率；不再把“猜测不在剩余候选集”误记为规则违反，因为它可能是有效探测猜测。

**修正/转向**

- 放弃“所有 persona 必须有完全相同失败轨迹”的设想。Absurdle 只保证持续对抗性，不会令不同猜测后的局面相同。
- persona 文本只保留一般特质；禁止写入“失败后会沮丧”“被鼓励后会振作”等待测反应规则，避免把答案写进刺激。

**暂定决定**

- 核心设计为外向性高/低 × 神经质（情绪敏感度）高/低的四个 prompt；其他人口学、能力和游戏经验信息不变或省略。

## 2026-08-15：实现并冻结基础运行条件

**事实**

- 实现了可复制的 Mastermind/Absurdle 引擎、配对分支、JSONL 记录、结果分析和组件测试。
- 被试模型选为官方 `Qwen/Qwen2.5-1.5B-Instruct` 本地快照；MPS `float16` 推理。
- 中性前轨迹的采样 pilot 中，temperature `0.3` 的 3 个 seed 轨迹相同；`0.5` 和 `0.7` 各在 5 个 seed 中产生 4 条不同轨迹，且 25/25 次猜测合法。因此冻结 temperature `0.5`、top-p `0.9`、最多 48 个新 token。
- 早期继续意愿 JSON 示例写了数字 `7`，模型出现固定答 7 的锚定；正式 prompt 已删去该数字并加检查，旧数据不用作正式结果。

**限制**

- 不同 seed 仍可能复现相同生成轨迹；seed 是独立运行单位，但不自动保证行为轨迹唯一。

## 2026-08-15：从宽泛情绪词转向 frustration-specific 方向

**事实**

- 以被冻结、与游戏/persona/鼓励无关的文本训练模型自身 hidden-state 方向；固定读倒数第 5–9 层，以五层 cosine 投影中位数汇总，只读不 steering。
- 保留 wide positive 和 negative 相对中性方向：各自 held-out 分离准确率 `0.75`，刚过预设阈值，只作为谨慎的辅助读数。
- 新增 frustration-specific direction：12 对训练场景和 12 对 held-out 场景中，每对共享同一受阻事件，只改变为紧绷急躁重试或冷静处理；场景不直接出现目标情绪词。
- frustration direction 的 held-out 分离为 `11/12 = 0.9167`，五个固定层均为 `0.9167`。

**修正/转向**

- 不把“做得好愿意继续—做得不好不想继续”的行为/价值轴当作 frustration 方向；继续意愿保留为下一猜后的下游行为评分。
- 当前方法是轻量的成对 contrastive RepE probe，不是 Anthropic 171-emotion pipeline 的完整复刻。

## 2026-08-15：验证连续失败前的 frustration 读数

**事实：1–5 轮 manipulation pilot**

- 在四个核心 persona、seeds 601–605 上运行 20 条独立游戏；每次只读取干预前每轮的 frustration 投影，分析时不使用鼓励/中性分支结果。
- 第 5 轮相对第 1 轮的 run 内变化 20/20 为正；pooled median delta 为 `+0.00843`，逐轮斜率中位数为 `+0.00169`。第 5 轮是这一批 1–5 轮 pooled median 的最高点。
- 因而通过了当时预先写下的 round-5 manipulation 判定：至少 20 runs、端点 median 和 slope 为正、至少 70% 的 run 为正。

**限制**

- 该读数是训练出的内部方向投影，没有天然的心理量纲；`+0.00843` 表示方向一致的相对增加，不能单独称为“大量”或主观强度。
- 曲线并非逐轮单调：端点增加稳定，不等于每一轮都持续上升。

## 2026-08-15：扩展到 1–7 轮并检查峰值稳定性

**事实：第一批 extended diagnostic（seeds 601–605，20 runs）**

- 四个 persona 的 pooled 曲线都在第 5 轮达到最大值；pooled 第 5 轮为 `0.020425`，第 6 轮降至 `0.015512`，第 7 轮为 `0.017463`。

**事实：独立复测（seeds 611–615，20 runs）**

- 四个 persona 的 pooled 曲线都在第 7 轮达到最大值；pooled 第 7 轮为 `0.019369`，第 2 轮为 `0.019262`，差距很小。

**事实：两批合并后的描述性图形（40 runs）**

- 合并曲线的 pooled 最高点为第 7 轮（`0.018617`）；高外向两个 prompt 的最高点为第 7 轮，低外向两个 prompt 的最高点为第 2 轮。
- 这不是稳定的 persona-specific 峰值证据：第一批全部第 5 轮最高，独立第二批全部第 7 轮最高；合并后的位置会随批次组合和中位数的微小差异改变。
- 到第 5 轮相对第 1 轮，40/40 run 仍为正；逐 run delta 的中位数为 `+0.00801`。另一个不同口径——第 5 轮 pooled median 减第 1 轮 pooled median——为 `+0.00549`，不把两者混用。

**修正/转向**

- 原本考虑按 persona 各自的峰值分叉；因峰值在独立批次间不稳定而放弃。按各自峰值还会让失败轮数、候选集难度与对话长度一起变化，混淆“同一失败剂量下 persona 是否调节鼓励效果”的主问题。

**当前决定**

- 主实验统一在连续失败第 5 轮分叉。它在最初的预数据判定中通过、位于扩展曲线的中段，且避开更长轨迹中候选集可能坍缩和峰值选择不稳定的问题。
- 不以“第 5 轮是全局真实峰值”作主张；只主张它是统一、预先固定且已有 manipulation 支持的 checkpoint。

## 当前探索性发现与边界

**事实**

- 两批合并时，高外向两个 prompt 的 run 内 round-1–7 斜率中位数约为 `0.00082`、`0.00092`；低外向两个 prompt 约为 `0.00062`、`0.00074`。

**限制**

- 这只是当时四份具体 persona 文案的描述性差异，且存在重复轨迹；当时每象限仅一份模板，不能归因为“外向性造成更快 frustration 增加”，更不能当作心理学规律。
- 正式鼓励效应、persona 调节、继续意愿和 positive/negative/frustration 的条件 delta 都尚未做推断。

## 2026-08-15：把每象限扩展为三个 persona 模板

**决定与实现**

- 为减少单一句子措辞效应，将四象限从各 1 份扩为各 `v1`–`v3` 三份平行模板，共 12 份。新增文本只重述外向性与神经质的一般特质，不加入失败、鼓励、坚持、下一猜或任务表现的反应规则。
- 新增显式 persona 注册表，运行记录写入 quadrant 与 template id；正式分析先算每个 template × seed 的鼓励减中性 delta，再在同一 quadrant × seed 内平均三个模板，最后跨 seed 汇总。模板不会被当成三倍独立样本，同时保留逐模板敏感性结果。
- 单独运行一个完整第 5 轮 persona 配对约 14.3 秒，其中包含模型加载；同一批会复用模型。固定每模板 seed 数时，12 模板的总推理量约为原四模板设计的 3 倍，情绪方向不需重训。
- 人工审查时发现原 `v1` 的低神经质描述比高神经质描述多约 9 个模型 token；正式采集前已改写为语义平行、同象限仅差约 1 token 的一般特质表述。全部 12 个模板在对应象限内的长度差控制在约 1–3 token。
- 扩展并平衡后的完整 `prompts.md` SHA-256 为 `2641d8a2eea7f209ffbb96aa5a801c16ca3d2fede7bb84a7dce525453595d321`。方向 artifact SHA-256 为 `8034145a877afb36dc4bf8bde8afc208da3268866d8335ee393bed543e5250f2`，其记录的模型 snapshot checksum 为 `58c7c8cabbfb8a71eef25c14860b07338eb7b689063cc0fc19f08f4247c39a7e`、材料 checksum 为 `6cce24b8f0e5379ae5155c8aaa29fd7d92df1025e66b95413db4423330df426a`。

**接线 smoke pilot**

- 对全部 12 份模板使用 seed 701 做第 5 轮真实运行；干预前猜测均合法，第 5 轮 frustration 投影相对第 1 轮为 12/12 正向，中位 delta `+0.00937`。
- 该单 seed 只验证新增模板、记录字段、分支和 probe 接线没有明显破坏；其鼓励/中性分支结果不进入正式分析，也不用于重新选择 checkpoint。

## 续写约定

- 正式采集开始后，按日期追加模型/方向 artifact checksum、persona template id、run 数、剔除规则和预先确定的汇总方式。
- 将 checkpoint 选择数据与正式鼓励效应数据明确分开；不得用正式条件结果重新选择轮数、层或 persona 文案。

## 2026-08-15：冻结正式采集协议

- 正式 seeds 固定为 `1001`–`1010`；12 templates × 10 seeds，共 120 个配对 checkpoint、预期 240 条 branch records。
- 固定唯一主指标为 encouragement − neutral 的归一化信息效率 delta；同一 quadrant × seed 先平均三模板，再计算 extraversion、neuroticism、interaction 三个 planned contrasts。三个 contrast 使用 seed-level exact sign-flip p 并作 Holm 校正。
- 继续意愿解析失败只记该副指标缺失，不删除主指标；无效猜测保留并按既定规则记效率与违规。正式数据不因结果难看而排除或追加 seed。
- 生成记录新增精确 source SHA-256 与 emotion-probe artifact provenance；非空输出默认禁止继续写入，只有完全一致的冻结 metadata 才能使用 `--resume`。
- 完整冻结参数、checksum、命令与解释边界见 `FORMAL_PROTOCOL.md`。此时尚未查看任何正式鼓励效应结果。

**零记录性能修订**

- 首次正式命令启动后，在写入任何 record 之前卡在大候选集的 `I*` 精确穷举；确认输出文件尚不存在后停止。
- 仅把同一 10,000-guess 最大反馈桶计算从 Python `Counter` 内循环改为有界 NumPy 批次，指标定义、候选、反馈与取最优规则均不变。随机集合与旧标量算法逐项对照通过；全 10,000 候选的精确 optimum 从不可接受的长等待降为约 1.77 秒。
- 生成源码重新冻结为 SHA-256 `c7b6c6d6f21a506980399d9d179d72ca646991615802ceb8c0dc04d217852cd2`；修订发生时正式数据仍为零。

## 2026-08-15：完成 formal-v1

- 按冻结命令采集 12 templates × 10 seeds 的 120 个 checkpoint，共 240 条完整配对分支；没有追加、替换或结果导向排除 seed。
- 正式记录的 checkpoint、模型、采样、prompt、probe 和生成源码 provenance 全部一致。120 runs 有 10 条不同的游戏轨迹；格式失败按冻结规则保留。
- 用冻结 probe 对正式 transcript 第一轮反馈后的前缀做离线只读重算：第 5 轮 frustration 相对第 1 轮在 116/120 runs 中上升，中位 delta `+0.00564`，正式 manipulation 仍成立。
- 唯一主指标归一化信息效率的平均鼓励效应为 `-0.00996`（SE `0.01031`，exact p `0.750`）；E、N 与 E×N 的 Holm p 均为 `1.000`。因此没有行为改善或 persona 行为调节证据。
- 鼓励后的即时内部表示表现为 positive 上升、negative 与 frustration 下降。frustration 的高 N−低 N contrast 为 `-0.000483`（Holm p `0.0117`），但下一猜后 frustration 平均效应和 persona contrasts 均不再可见。
- 继续意愿平均 delta 为 `+0.217/10`（exact p `0.250`）；鼓励与中性分支各有 13/120 个无效猜测，规则违反条件差为零。
- 详细表格、数据质量和解释边界见 `FORMAL_RESULTS.md`。
