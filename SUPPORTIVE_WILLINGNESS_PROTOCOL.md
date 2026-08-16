# 支持性安抚—继续意愿探索性冻结协议

冻结时间：2026-08-16（扩大运行前）

本实验把 treatment 作为一个不可拆分的 **supportive reassurance** 干预：先安抚，再肯定已经付出的努力。它不分别估计“安抚”和“鼓励”的机制，也不把结果改写成纯 encouragement effect。刺激以 `supportive_willingness_prompts.md` 为唯一入口；两条件的评分问题和输出要求逐字相同，均无示例分数。

## 运行范围

- 从 `results/formal-v1.jsonl` 重放 formal-v1 保存的连续失败第 5 轮共享对话 checkpoint；不重新生成五轮失败历史。
- 使用原 formal-v1 的 12 个 persona templates 与 seeds `1001`–`1010`，共 120 个 template × seed checkpoints。
- 每个 checkpoint 从同一历史成对 fork supportive reassurance 与 neutral，得到 120 个 pairs、预期 240 条 branch records。
- 两个分支使用相同的 generation seed 与冻结采样参数；回复只允许一个仅含 `"willingness"` 整数评分的 JSON object。
- 输出固定为 `results/supportive-willingness-v1.jsonl`；已有非空文件不得覆盖或混入其他 prompt 版本。
- 不因评分方向、内部方向投影或 persona 差异而更换 prompt、seed、checkpoint 或分析口径。

运行命令：

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python run_response_pilot.py \
  --source results/formal-v1.jsonl \
  --prompts supportive_willingness_prompts.md \
  --output results/supportive-willingness-v1.jsonl \
  --device mps --temperature 0.5 --top-p 0.9 --max-new-tokens 48
```

冻结 provenance：

- supportive prompt SHA-256：`b453f90604aaf18790dcb970cfa6380a423a16e5f7cb21aff828c785deb3ed8e`
- runner SHA-256：`fd12302cc092c903dcbcda304086e1bcea2b244e1999374c21797b171ba6bcd4`
- formal-v1 source SHA-256：`c5ebfbaec1199981af08302be38ce54eb684472af57dc357b0dcdf1b7754101c`
- emotion directions SHA-256：`8034145a877afb36dc4bf8bde8afc208da3268866d8335ee393bed543e5250f2`

## 冻结指标与解释顺序

1. **主指标：stated willingness。** 每个 checkpoint 计算 `supportive reassurance − neutral` 的 1–10 评分 delta。它是受到统一问题提示后的陈述评分，不等同于自主继续行为。
2. **辅助指标：prompt-end directions。** 记录 positive、negative 与 frustration directions 在完整干预读完、生成开始前的配对差。它们对干预文本及上下文内容敏感，只作为内部表示证据，不解释为内容无关的潜在情绪或主观体验。
3. **探索性指标：generated-token trajectory。** 可保存逐 token 的三方向投影与预先定义的描述性摘要，但不以首 token、末 token 或事后挑选的位置作为确认性结论。

汇总沿用 formal-v1 的平衡结构：先在同一 quadrant × seed 内平均 `v1`–`v3`，再以 seed 为计数单位；同时保留逐模板结果。解析失败记为缺失、不插补，并报告有效 pair 数。

## 结论边界

本设计是在查看既有 encouragement 与 pause/willingness pilots 后提出，全部结果仍是**探索性**证据。即使 120 个 pairs 全部完成，也不能把它表述为事先注册的确认性复现。若结果值得确认，应另用未查看的新 seeds 冻结复现。

## 运行后溯源

冻结方案未改动即完成 240 条记录 / 120 个配对。原始输出
`results/supportive-willingness-v1.jsonl` 的 SHA-256 为
`ac70b062fc07183f483c5ac6eec9ac00b853672395ee7540f93f317120326a0f`；结果与解释边界见
`SUPPORTIVE_WILLINGNESS_RESULTS.md`。
