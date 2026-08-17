# formal-v3 discrete-emotion protocol — frozen 2026-08-17

## Question and relation to earlier work

Formal-v3 repeats the unchanged formal-v2 three-arm, five-failure design while
replacing the locally trained positive/negative/frustration probe with three
previously published discrete emotion-concept vectors for the exact
`Qwen/Qwen2.5-1.5B-Instruct` checkpoint. Formal-v2 remains immutable historical
evidence. Formal-v3 is a fresh replication, not a re-analysis labeled as new
data and not a repair of formal-v2.

The conclusion boundary is unchanged: projections measure content-sensitive
residual-stream alignment with named emotion concepts. They do not establish
subjective feeling, experience, or a persistent internal emotional state.

## Frozen external directions

Source: `mufxio/emotion-vector-bench`, commit
`f6c84d65832608b4c4457f3f4b248774a42df940`, MIT license. The upstream
`results/qwen2.5-1.5b-instruct/denoised_vectors.npz` SHA-256 is
`612f780909fe8f3e75b5a65882037a4b48f3372ac0ee96060133dc7947020475`.
The local deterministic import is `import_external_emotion_vectors.py`; its
non-overwriting output is
`artifacts/qwen2.5-1.5b-emotion-vector-bench-layer17.npz`, SHA-256
`aca2a4806c5cb475455c2f914b26fe1fe107ed90a1f62b44a59121c6a54d6fc0`.

The three concepts were selected before reading any formal-v3 trajectory:

- `joyful`: positive valence, high arousal;
- `grief_stricken` (upstream `grief-stricken`): negative valence, low arousal;
- `furious`: negative valence, high arousal.

This spans valence and arousal with recognizable discrete anchors. It is not a
search over the upstream 20 concepts. Layer 17 is fixed because the upstream
report identifies it as Qwen2.5-1.5B-Instruct's best 20-way probe-accuracy
layer (89.7%). The PCA-denoised rather than raw vectors are fixed because those
are the upstream pipeline's reported geometry/validation vectors. Formal-v3
does not train, tune, orient, or select a direction from its own game data.

## Frozen runtime

- Model snapshot, MPS float16 runtime, sampling parameters, 12 dated-frozen
  persona v3 prompts, three arms, filler v4 texts, Mastermind engine, five
  consecutive failures, prompt-boundary readout, and complete independent-run
  structure are unchanged from `FORMAL_V2_PROTOCOL.md`.
- Runtime prompt source remains `formal_v2_prompts.md`, SHA-256
  `ff6008a741668b1c90a44740e0573f29fa793fd039b102e744e65c9e85fa4136`.
- Arms remain `feedback_only`, `supportive`, and `neutral`.
- Formal-v3 uses fresh seeds `3001`–`3010`. Each of 12 persona templates × 10
  seeds × 3 arms is an independent from-start run: 360 runs and 1,800
  prompt-boundary readouts.
- Generation seeds use the `formal-v3` namespace and are shared across arms
  within persona × seed × round.
- Readout is cosine projection at layer 17. There is one value per concept per
  round; no cross-layer median and no locally extracted frustration step.
- Any missing readout, early win, invalid record, provenance mismatch, or
  incomplete five-round run is retained in the source record but excludes the
  whole run from that concept's trajectory. Missing arm cells propagate to the
  affected paired seed contrast exactly as in formal-v2.

## Frozen analysis

For each concept separately, calculate the five-point OLS slope and descriptive
R5−R1. Within each persona-template × seed, subtract arms; average v1–v3 only
after subtraction to quadrant × seed; then calculate overall, E, N, and E×N
effects across quadrants. Seeds are the inferential units (`n=10`).

Confirmatory family 1 contains exactly three two-sided exact paired sign-flip
tests: supportive−neutral slope for `joyful`, `grief_stricken`, and `furious`.
Apply Holm correction jointly across these three tests.

Confirmatory family 2 contains exactly nine two-sided exact paired sign-flip
tests: E, N, and E×N moderation of supportive−neutral slope for each of the
three concepts. Apply Holm correction jointly across these nine tests.

Directional expectations are supportive−neutral `joyful > 0`,
`grief_stricken < 0`, and `furious < 0`, but the frozen tests remain two-sided.
R5−R1, supportive−feedback_only, and neutral−feedback_only are robustness or
descriptive results without confirmatory p-values. Unlike formal-v2, there is
no frustration-specific manipulation gate: no direction was constructed to
track repeated obstruction, and failure-driven movement of these three named
concepts is itself an empirical diagnostic rather than an inclusion rule.

All three concept results must be reported, including nulls and sign reversals.
No replacement concept, alternative layer, raw vector, direction sign flip,
seed change, or analysis-family change is permitted after formal data are read.

## Freeze checklist

Formal collection remained code-locked while `FORMAL_V3_FREEZE_ID` was `None`.
Before replacing it with `formal-v3-2026-08-17`, all of the following had to pass:

1. generic-probe, runner/record, and analysis unit tests;
2. exact artifact/model/axis/layer checksum audit;
3. nonformal MPS smoke using seed 9002, with all three arms complete and all
   15 × 3 concept readouts finite;
4. analysis dry run on synthetic complete records;
5. a non-overwriting freeze manifest containing source and code hashes.

All five checks passed before collection. The seven-check candidate audit is
`results/formal-v3-freeze-audit-2026-08-17.json`; all checks are `passed=true`.
The only post-audit source change that unlocks formal collection is the dated
`FORMAL_V3_FREEZE_ID` assignment in `run_formal_v3.py` plus this freeze note.

After the dated freeze, the only collection command is:

`.venv/bin/python run_formal_v3.py --formal --device mps --output results/formal-v3.jsonl`

The only analysis command is:

`.venv/bin/python run_formal_v3_analysis.py --input results/formal-v3.jsonl --output-dir results/formal-v3-analysis`
