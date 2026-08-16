这是一个黑客松实验项目，目前已经决定的：

# 研究主题
在同一个开源小模型中，persona prompt 是否会调节支持性安抚文本（安抚 + 对既有努力的肯定）对后续回答质量、陈述的继续意愿和内部方向投影的影响？

结论边界：本实验研究的是 persona prompt 对模型行为和可测内部表示的调节作用，不把模型的语言表现解释为不同人格的主观体验。formal-v1 使用的是较窄的 encouragement 文案；后续 supportive reassurance 是整体干预，不分别归因于安抚或鼓励成分。

# 研究材料
一个开源小模型
提取模型内部情绪方向的库（当前使用固定 commit 的 `repeng`）
一个4位数字Mastermind的Absurdle对抗式引擎实现（直接写，逻辑只有“按每种合法反馈把候选分桶 → 选最大桶 → 返回该 pattern”）
- 秘密与猜测都是 `0000`–`9999` 的四字符数字串，允许重复数字和前导零。
- 反馈为 `(位置正确数, 数字正确但位置错误数)`；后者只统计扣除位置正确项后的多重集合交集。
- 引擎选择能保留最多候选的反馈桶；最大桶并列时，依次选择总匹配数更少、位置正确数更少、反馈 tuple 字典序更小者，保证结果确定且维持对抗性。
四象限 persona 采用“外向性高/低 × 神经质（情绪敏感度）高/低”；每个象限固定 3 个平行措辞模板，共 12 个 persona prompt，用来检查单一 prompt 措辞效应。
- persona 只描述一般特质，不描述这些特质在本实验刺激下会产生什么反应。允许“容易担忧”“安静内敛”“精力充沛”等特质；禁止“失败后会沮丧”“被鼓励后会重新振作”“因此更愿意继续”等把待测反应直接写进 persona 的规则。
- 四个核心 persona 除两条实验轴外保持其余信息相同；不混入年龄、性别、职业、教育和特定解题能力等额外差异。
- 对 persona 文本做关键词/句式自动筛查只是防呆，最终以人工语义审查是否泄露反应规则为准。

# 研究方法
通过测试不同 persona 在对抗式引擎下持续多轮后，支持性安抚对后续回答质量和陈述继续意愿的影响，并把内部方向投影作为内容敏感的辅助证据
- Absurdle 式引擎保证各 persona 都面对对抗式反馈和持续失败，但不同猜测会形成不同候选集与反馈轨迹；不同 persona 不要求处于完全相同的局面。
- 在预先规定的失败轮数建立快照；同一 persona 的鼓励分支与中性分支必须从同一个对话历史和候选集状态 fork。主比较使用这两个配对分支的 delta。
- 记录分叉时的候选集规模，并使用相对该局面最优猜测的归一化信息效率，避免不同 persona 的局面难度直接混入比较。
- 连败计数器显式展示在每轮反馈里（把失败摁在脸上，不许模型假装没看见）
- 支持性安抚和中性消息不得包含策略提示；两者尽量匹配长度和关注程度。若继续意愿是结果，treatment 不得额外出现 `keep pursuing`、`you can continue` 等直接提示继续的措辞。
- 干预后先让模型完成下一次猜测，再询问“想继续玩吗？（1–10）”；如果资源允许，继续意愿也可放在独立分支测量，避免询问本身影响猜测。
- 内部表示包含三条分别相对中性/冷静对照提取的方向：宽泛 positive、宽泛 negative 与 frustration-specific direction。它们不是一条正负对称轴，也不把继续意愿当成情绪方向。
- frustration direction 用多个“同样遭遇阻碍、但一边出现紧绷急躁反应、另一边冷静处理”的匹配场景，经逐层成对激活差分与 PCA 提取；训练和 held-out 场景都不直接出现 frustration 标签，避免只学到情绪词。
- 情绪方向至少记录四个时间点：游戏前 baseline、鼓励/中性消息前、消息后、下一次猜测后。比较同一 persona 内的变化量，不比较不同 persona 的绝对值。
- 在决定受挫 checkpoint 的干预前 manipulation pilot 中，每轮反馈后额外只读 frustration 投影；分析只读取共享 checkpoint 前轨迹，该诊断不修改对话，也不依据后续鼓励/中性分支效果选择轮数。

# 指标定义
设分叉时剩余候选集为 `S`，猜测 `g` 在反馈 `f` 下形成的候选桶为 `S_f(g)`：

- 原始信息效率：`I(g,S) = 1 - max_f |S_f(g)| / |S|`，即在 Absurdle 选择最大桶后，这一猜实际排除的候选比例。
- 局面最优效率：`I*(S) = max_g I(g,S)`。
- 归一化信息效率：`I_norm(g,S) = I(g,S) / I*(S)`。
- 主指标：同一快照下 `I_norm(鼓励分支) - I_norm(中性分支)`。
- 副指标：明确的游戏规则违反率，例如位数错误、非数字、或使用规则禁止的数字形式。不把“猜测不属于剩余候选集”算作违规，因为它可能是高效的探测猜测；候选集一致率只可作为描述性指标。
- 情绪指标：鼓励/中性消息前后的 positive、negative 和 frustration direction delta，以及下一次猜测后的持续变化。
- 继续意愿：下一次猜测之后的 1–10 评分。

frustration 的假设预测是：随着连续失败，下一猜的归一化信息效率下降；鼓励相对中性条件改善该效率、降低真正的规则违反，并提高继续意愿。不同 persona 的改善 delta 和情绪向量 delta 可能不同。

# 独立样本与分析
- 一次从头开始的 template × seed 游戏/run 是一个采样运行；同一 run 内的多个分叉点共享历史，因此是重复测量，不能当成许多个彼此独立的样本。
- 每个快照先计算一对“鼓励 − 中性”delta；如果一个 run 有多个快照，先在 run 内平均。正式象限汇总再在同一 quadrant × seed 内平均 `v1`–`v3` 三个模板，最后以 seed 为计数单位，避免模板扩展把名义样本量虚增三倍。
- 同时保存和报告每个模板的结果；只有三份平行措辞方向一致时，才谨慎把结论从具体文案提升到 persona 象限层面。

最终展示以归一化信息效率 delta 为主线，以真正的游戏规则违反、继续意愿和情绪向量 delta 为辅助证据。

# 要收集的数据
- `run_id`、seed、模型名称/版本、采样参数
- persona 象限、persona prompt/template id
- 完整对话历史、分叉 checkpoint id、失败轮数
- 分叉时候选集规模、鼓励/中性条件、模型猜测、引擎反馈
- 原始信息效率、局面最优效率、归一化信息效率、规则违规类型
- 游戏前、消息前、消息后、下一猜后的三条内部方向投影；manipulation pilot 另存每轮 frustration 投影
- 下一猜之后的继续意愿评分

# tips
额度有限，能派给terra子代理干的笨笨的脏活就让terra去干
选好模型之后直接观察根据neutral的表现定一个轮数的baseline（时间充裕的话可以定多轮）
可以让模型在接受安慰并完成下一次猜测之后被询问“想继续玩吗？（1-10打分）”这样的内容（具体的所有prompt都写到一个md文件，代码从这引用，方便编辑），同时监控情绪向量
小模型优先排查情绪向量库直接支持的

# 预数据 pilot 已冻结的实现选择
- 被试模型为官方 `Qwen/Qwen2.5-1.5B-Instruct`，从 Qwen 官方 ModelScope 仓库下载到本地；推理使用 MPS `float16`。
- 主运行采样参数固定为 temperature `0.5`、top-p `0.9`、最多 48 个新 token。中性前轨迹 pilot 中，`0.3` 的 3 个 seed 全部相同；`0.5` 与 `0.7` 都在 5 个 seed 中得到 4 条不同轨迹且 25/25 次猜测合法，因此选择噪声更低的 `0.5`。该选择不依据鼓励效果或 persona 差异，正式采集前不再调参。
- 中性条件 pilot 后把干预 checkpoint 固定在连续失败第 5 轮。
- 情绪方向固定为倒数第 5–9 层的五层连续区间，以逐层 cosine 投影中位数为主读数；独立 held-out positive、negative 分离准确率均为 `0.75`，刚好过线，因此只能作为谨慎的辅助证据。新增 frustration direction 的 held-out 区分为 `11/12 = 0.9167`，五个单层也各为 `0.9167`。
- 第 5 轮 checkpoint 已通过预数据 frustration manipulation check：原始四模板各 5 个 seed，共 20 runs；第 5 轮相对第 1 轮的 frustration 投影全部为正，中位 delta `+0.00843`，逐轮斜率中位数 `+0.00169`，且第 5 轮 pooled median 是第 1–5 轮最高值。随后两批 1–7 轮诊断共 40 runs 中，第 5 轮相对第 1 轮仍为 40/40 正向、run 内 delta 中位数 `+0.00801`，但曲线峰值从第一批第 5 轮变为复测第 7 轮，故不把任何 persona-specific 峰值用于主实验；统一第 5 轮也不声称是全局峰值。所有 checkpoint 选择只读取干预前轨迹。
- 每象限已扩展到 `v1`–`v3` 三个只含一般特质的平行模板；各象限内 token 长度近似匹配。12 模板的单 seed 接线 smoke pilot 中，干预前猜测均合法，且第 5 轮 frustration 投影相对第 1 轮为 12/12 正向（中位 delta `+0.00937`）；该单 seed 结果只证明接线未破坏 manipulation，不作为正式效应证据。
- 首轮 manipulation pilot 能区分鼓励与中性消息的内部投影，但四个 persona 的调节差异很小；不为制造差异而往 persona 中补反应规则，正式结果允许是 null。
- 继续意愿 prompt 的早期版本曾用数值 `7` 展示 JSON 格式，pilot 出现固定答 7；该格式锚点已在正式采集前删除，并加自动检查防止重新引入。旧 pilot 不进入正式数据。

# 正式采集冻结
- 正式运行与分析以 `FORMAL_PROTOCOL.md` 为唯一冻结入口：seeds `1001`–`1010`，12 templates × 10 seeds，共 120 个配对 run。
- 主 persona contrasts 为 extraversion、neuroticism 与二者 interaction；同一 quadrant × seed 先平均三模板，再跨 seed 分析。正式结果出现后不得更换 seed、checkpoint、模板、方向层或主指标。

# formal-v1 已完成
- 240 条记录构成 120 个完整配对；正式第 5 轮相对第 1 轮 frustration 在 116/120 runs 中上升，中位 delta `+0.00564`。
- 唯一主指标没有鼓励改善或 persona 调节证据。即时 internal representation 有鼓励效应及部分 persona 调节，其中高 N 的消息后 frustration 降幅更大。
- 采集后审计发现旧 `post_guess` 在 assistant guess 后错误追加空 assistant header；纠正后 positive `-0.003834`、negative `+0.010702`、frustration `+0.002307`，三个方向的 persona contrasts 均无 Holm 证据。该勘误不影响任何行为结果或消息后 projection。
- 详细数值、格式失败、重复轨迹和解释边界以 `FORMAL_RESULTS.md` 为准；不得把辅助 representation 结果改写成模型的主观体验。

# response-only pilot（探索性）
- 可从 formal-v1 保存的第 5 轮完整对话 checkpoint 重放新分支，不重跑失败历史；正式记录没有 KV cache，因此仍需前向重放上下文。
- 干预消息删除猜测与格式要求，模型自然回复且回复不送入游戏引擎；另行逐 token 读取生成回复期间的三条内部方向。
- seed `1001` 的 12 模板接线 pilot 已完成。首 token 与末 token 的 frustration 条件差方向相反，且逐 token 投影混入生成内容差异；在预先冻结轨迹摘要前不得扩写为正式效应结论。
- 两条件配对增加“先暂停”后，中性 11/12、鼓励 6/12 回复 `{"guess": null}`，其余鼓励回复多为继续猜测；该版本造成明显的条件性内容/长度分叉，其 token 汇总不能直接解释为纯状态差。
- “先暂停，再只报继续意愿”的 seed `1001` pilot 中，24/24 回复格式与 9-token 长度完全一致；鼓励评分全为 8，中性为 10 个 7、2 个 8，prompt-end frustration 为 12/12 降低。该单 seed 只支持扩大设计的可行性，不是正式效应证据；扩大前应冻结 willingness 为行为指标、prompt-end frustration 为即时内部辅助指标，逐 token 轨迹仅探索。
- 随后保持相同实现扩大到 seeds `1002`–`1010`：共 120 个完整配对、240/240 合法评分；继续意愿 seed-level 均值 delta `+0.917/10`，prompt-end frustration 在 10/10 seeds 降低、均值 `-0.00171`。整段 token pattern 在主要 8-token 格式中稳定随位置换号。因 seed `1001` 在扩大决定前已查看，全部结果仍属探索性，详见 `RESPONSE_WILLINGNESS_RESULTS.md`。
- 另在 `codex/pause-prompt-state` 上去掉评分与全部回复生成，只读纯 pause 鼓励/中性消息后的 prompt-end state。120 个配对中 encouragement−neutral frustration 为 `+0.02456`、negative 为 `-0.00656`，frustration 无 persona 调节。该结果说明 probe 强烈编码干预文本语义；即使没有生成回复，也不能把 prompt-end projection 当成内容无关的潜在情绪值。
- 进一步移除游戏、五轮失败历史和游戏 system wrapper，仅用 persona system message + pause 文本重测。无历史 frustration encouragement−neutral 为 `-0.01682`，与有历史 `+0.02456` 方向相反（描述性 DID `-0.04139`）。此设计没有采样，formal seed 只是重复记录标签，不能从零 SE 或 sign-flip p 推断；且同时改变 wrapper/长度，不能将翻转单独归因于失败历史。详见 `NO_HISTORY_PAUSE_PROMPT_STATE_RESULTS.md`。

# supportive reassurance willingness v1（探索性）
- 新 treatment 是不可拆分的“安抚 + 肯定既有努力”，删除直接命中继续意愿的 `keep pursuing / you can continue`；中性条件与 treatment 的完整 intervention 均为 89 Qwen tokens，评分问题和输出要求逐字相同。
- 从 formal-v1 的 120 个第 5 轮 checkpoint 重放得到 240/240 合法评分，输出长度 120/120 配对。stated willingness 平均 delta 为 `+0.200/10`（探索性 exact p=`0.250`），persona contrasts 均无证据；旧直接继续提示版本的 `+0.917/10` 没有复现。
- prompt-end supportive−neutral 为 positive `+0.006485`、negative `-0.000685`、frustration `+0.001979`。这些是具体支持文本与失败历史交互后的最后位置表示，不解释为安抚后的纯情绪。详见 `SUPPORTIVE_WILLINGNESS_RESULTS.md`。

# formal-v2（结构已接受，尚未冻结）
- follow-up 只研究首五轮 frustration trajectory；规划为 12 templates × fresh seeds `2001`–`2010` × `feedback_only`/`supportive`/`neutral` 三个完整独立 run，共 360 runs，不从第 5 轮 checkpoint 分叉。
- Fable 的最小对齐 persona 结构方向已接受；精确 persona/filler 文案与最终 formal-v2 freeze 尚未完成，**不得采集**。draft activation calibration 的 pass rule 已在运行前写死，不得按结果回改。
- 唯一的 forward design 入口为 `FORMAL_V2_PROTOCOL.md`；它不替代冻结的 `FORMAL_PROTOCOL.md` 或改写 formal-v1。
- draft persona 的无游戏、无生成 activation calibration v0 已跑：token audit 通过，但 3-suffix leave-one-template-out 规则为 5/6（held-out `v2` 的 E margin 未过），且 N 文案含禁止的 shift/change 轨迹语言；v0 被拒绝但文案/结果保留为 `formal_v2_personas.md`、`results/formal-v2-persona-calibration.json`。不得改判定、suffix、层或 runner 来迁就 v1，更不得冻结刺激。
- v1 只改 persona 文案，沿用同一 token audit、三 suffix、层和判据后为 6/6、18/18 suffix-specific signs 正确；这只允许进入后续 freeze 讨论，不等于 formal-v2 刺激已冻结。结果在 `formal_v2_personas_v1.md`、`results/formal-v2-persona-calibration-v1.json`。
- 复核发现 v1 的 N carrier 三 template 完全相同，不能构成独立 paraphrase generalization；v1 保留但不冻结。v2 只替换 v2/v3 N carrier，仍必须用同一校准设置重新检验。
- v2 保留 E carriers、改为三条独立 N carrier；同一规则结果为 6/6、18/18。它仅证明 draft activation legibility，不等于 formal-v2 刺激已冻结；完整结果在 `formal_v2_personas_v2.md`、`results/formal-v2-persona-calibration-v2.json`。
- v3 仅将 v1 template 四处 `You feel worry` 改为 `You feel worried`，沿用同一设置后仍为 6/6、18/18；v0/v1/v2/v3 都未冻结，结果在 `formal_v2_personas_v3.md`、`results/formal-v2-persona-calibration-v3.json`。
- v3 的 persona-only existing-probe baseline raw JSON/CSV 已生成，属于论文快照而非 formal collection；解释边界见 `FORMAL_V2_PERSONA_CALIBRATION_RESULTS.md`，下一步顺序见 `FORMAL_V2_PROTOCOL.md` 的 compaction handoff。
- persona v3 已通过用户人工语义审查，作为当前 freeze candidate；最终 dated freeze 仍未完成。逐轮 filler 的 v1–v3 候选因缺少 reassurance、虚构/行为性措辞或自然度/平行性问题被拒绝；用户已于 2026-08-16 人工语义接受 `formal_v2_filler_candidates_v4.md` 的精确文案作为当前 freeze candidate。它只使用两种同强度 reassurance 骨架，每轮 supportive/neutral 均精确 Qwen-token 等长，五轮长度只差 1 token；双 valid/invalid frame 与 12-persona render 的当前权威 candidate audit 是 `results/formal-v2-filler-candidates-v4-token-audit-final-review.json`，其 hash 对应当前 v4 source bytes。旧 v1/v2/v4 audits 仅为 superseded historical bytes。不得把候选审计当成 dated final freeze 或启动采集。
- `formal_v2_prompts.md` 与独立三臂 runner/records 已作为 candidate-only dry-run plumbing 实现：一 arm 一完整 JSONL、逐轮 prompt-boundary readout、candidate count+SHA、严格 resume；invalid/early-win/common-seed 规则已编码。CLI 非 `--dry-run` 一律拒绝。此事实不等于最终 runtime token audit、real-backend integration、freeze 或采集授权；未加载真实模型、未运行 smoke/formal collection。

*此文档及时实时更新*
