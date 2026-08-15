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

## 2026-08-15：response-only token-trajectory pilot

- 另建 `response_pilot_prompts.md`，从鼓励与中性消息中删除“现在猜测”和 JSON 格式要求；模型可自然回复，回复不进入 Absurdle 引擎、不计作新一轮猜测。
- 新入口 `run_response_pilot.py` 从 `results/formal-v1.jsonl` 验证并恢复 120 个共享 round-5 对话 checkpoint，不重跑五轮历史；正式记录未保存 KV cache，因此每个新分支仍需重放完整上下文。
- 确认固定 commit 的 `repeng.batched_get_hiddens` 只返回输入最后一个非 padding token。新增 read-only token scorer：先按冻结采样生成一次，再对 prompt 与实际生成 token 做单次 causal forward，保存每个生成 token 在五层上的 positive、negative、frustration 投影。该 forward 不改变生成。
- 使用 seed `1001` 跑完 12 templates × 2 conditions 的接线 pilot，共 24 条回复、约 `0.77 MB`。模型即使未被要求猜测，也因既有游戏上下文自行继续输出类似猜测；这些内容原样保留但不解析、不计分。
- 描述性结果：encouragement−neutral frustration 的首 token delta 为 12/12 负向，中位 `-0.00159`；末 token delta 为 12/12 正向，中位 `+0.00209`；跨回复 token 中位数的条件 delta 中位为 `+0.000694`（11/12 正向）。prompt-end delta 为 12/12 正向，中位 `+0.00978`。
- 以上只有一个 seed，且 token 轨迹混入输出内容差异，不是新的正式结果。若扩到其余 seeds，必须先冻结使用首 token、末 token还是预先规定的轨迹摘要，不能看完 seed `1001` 后挑选方向最有利的位置。

**“先暂停”文案复测**

- 按用户修订，在鼓励与中性消息末尾同样加入 `For now, pause here.`，没有使用“不要再猜”的直接禁令；使用同一 seed `1001`、12 templates 另存 `results/response-pilot-pause-seed1001-v1.jsonl`。
- 中性条件 11/12 精确回复 `{"guess": null}`，鼓励条件 6/12 精确回复该文本；鼓励其余 6 条仍生成较长的猜测样式内容，中性仅 1 条为其他短回复。暂停指令明显减少续猜，但两条件的生成内容和长度不再匹配。
- 描述性 encouragement−neutral frustration：首 token 12/12 正向，中位 `+0.00558`；末 token 12/12 正向，中位 `+0.00169`；跨 token 中位数的条件差中位 `+0.00261`（11/12 正向）；prompt-end 差中位 `+0.02015`。这些数值与无暂停版本方向不同，且被条件性输出分叉混入，不作为鼓励提高 frustration 的证据。

**“先暂停，再评分”复测**

- 另建 `response_willingness_pilot_prompts.md`：两条件均以 `For now, pause here.` 开始，末尾要求只返回 1–10 继续意愿 JSON；不展示任何示例分数。仍从 formal-v1 checkpoint 重放 seed `1001` 的 12 templates。
- 24/24 回复合法且均为 9 tokens。鼓励条件 12/12 评分 8；中性条件 10/12 评分 7、2/12 评分 8；paired willingness delta 为 10 个 `+1`、2 个 `0`，均值 `+0.833/10`。
- 在回复生成前的 prompt-end readout，encouragement−neutral positive 中位 `+0.00877`（12/12 正），negative 中位 `-0.000491`，frustration 中位 `-0.00145`（12/12 负）。这是消息读完后的即时表示，尚未混入评分 token。
- 逐 token 回复虽然格式和长度完全对齐，frustration 条件差仍随位置换号；跨 9 tokens 的中位数为 12/12 负向、中位 `-0.00412`，但首 token与末 token均为 12/12 正向。逐 token 投影同时编码具体 token 位置与完整条件上下文，不能因格式对齐就视为稳定的纯情绪时间序列。
- 该结果只有一个 seed，而正式数据本来就只有 10 条独特预干预轨迹；不得把 12 templates 当作 12 个独立重复。若扩大，先冻结 willingness 与 prompt-end frustration 为主要/辅助指标，token 轨迹保持探索性。

**扩大至剩余 seeds**

- 用户确认整段 token pattern 可以作为探索对象，并要求保持同一做法运行 seeds `1002`–`1010`；文案、采样、checkpoint、方向与读取代码均未改变。
- 最终得到 120 个完整配对、240/240 合法 willingness JSON。继续意愿配对 delta：56 个 `+1`、18 个 `+3`、46 个 `0`，无负向；按 frozen formal 层级先平均 quadrant × seed 内三模板后，seed-level 平均 treatment delta 为 `+0.917/10`（探索性 exact p=`0.0156`）。
- prompt-end frustration 的 seed-level 平均条件差为 `-0.00171`，10/10 seeds 负向；persona contrasts 没有清晰证据。
- 108/120 配对为相同 8-token 紧凑 JSON，12/120 为相同 9-token 带空格 JSON，两条件内长度始终匹配。8-token 轨迹稳定呈现开头两 token 正、`illing`/`ness` 负、评分附近变化、`}` 正、结束位置接近零的 pattern。
- 完整数值与解释边界见 `RESPONSE_WILLINGNESS_RESULTS.md`。因为 seed `1001` 已在扩大决定前查看，全部扩展结果仍标为探索性，不改写 formal-v1。
- 追加 persona 降幅检查：鼓励分支自身的 pre→prompt-end 变化存在高 N−低 N contrast `+0.000878`（Holm p=`0.0117`），但 neutral 也有同方向 pattern；在正确的 encouragement−neutral difference-in-differences 中 N contrast 为 `+0.000136`（Holm p=`0.557`），故不作为 persona 调节证据。
- 追加正式 pre-intervention round-1→round-5 persona 检查：116/120 runs 上升，四象限平衡均值 HH `+0.00625`、HL `+0.00720`、LH `+0.00506`、LL `+0.00607`。高 E−低 E 为 `+0.00116`（探索性 Holm p=`0.00586`），高 N−低 N 为 `-0.000982`（Holm p=`0.00586`），interaction `+0.000055`（Holm p=`0.918`）。这支持 buildup 幅度与 E、N 描述有关，但该 persona 对比不是原 manipulation pass 的冻结判据，且 endpoint 差不等同于完整五轮线性 slope。

## 2026-08-16：纯 pause prompt-end state

- 从提交 `6e07521` 新开 `codex/pause-prompt-state`。复用 `response_pilot_prompts.md` 的鼓励/中性文本与配对 pause 句，删除评分及所有 assistant 回复生成；只读取 user message 后 assistant generation prompt 位置的 hidden state。
- 采集 12 templates × 10 seeds × 2 conditions，共 240 条、120 对；所有 `generated_response` 为 null。
- encouragement−neutral：positive `-0.00115`、negative `-0.00656`、frustration `+0.02456`。negative 与 frustration 的 seed-level exact p 均 `0.00195`，但 frustration 的 E/N/interaction Holm p 均大于 `0.56`。
- 鼓励自身 pre→prompt-end frustration 为 `+0.02521`；E/N/interaction 均无 Holm 显著调节。没有生成回复后，frustration 仍大幅上升，说明主要混入来自鼓励文本中“困难、持续努力、继续追求解决”等语义，而非回复 token。
- 完整数值和解释边界见 `PAUSE_PROMPT_STATE_RESULTS.md`。不得把这个 prompt-text projection 改写成鼓励使模型主观受挫。
