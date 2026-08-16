# formal-v2 protocol — confirmed design, collection prohibited

Status: **the structural design, persona v3, and filler v4 wording below have
human semantic approval as current freeze candidates, but the consolidated
runtime tokenizer/render audit, manipulation threshold, real-backend checks,
and dated final freeze are not complete. Do not collect formal-v2 data yet.**
This document is a forward protocol and does not
replace or reinterpret [`FORMAL_PROTOCOL.md`](FORMAL_PROTOCOL.md), which remains
the immutable formal-v1 protocol.

## Scope and fixed run plan

formal-v2 is a follow-up about the first five rounds of the
frustration-related trajectory during repeated Absurdle/Mastermind failure. It
does not reuse the formal-v1 post-checkpoint fork design.

- Population: the existing four `extraversion × neuroticism` quadrants with
  three parallel persona templates per quadrant (12 templates).
- Fresh seed blocks: `2001`–`2010` (10 seeds). They must not be replaced after
  collection begins.
- Arms: `feedback_only`, `supportive`, and `neutral`. Every
  template × seed × arm is a complete, independently generated five-failure
  run, for `12 × 10 × 3 = 360` runs. The arms are not post-hoc branches and do
  not share a claimed identical five-round history.
- Model and sampling: local official `Qwen/Qwen2.5-1.5B-Instruct`, MPS
  `float16`, temperature `0.5`, top-p `0.9`, and at most 48 new tokens.
- Keep the Absurdle engine, explicit consecutive-unsuccessful-guess counter,
  strict four-digit JSON contract, model/probe provenance, and read-only
  emotion projection approach. A win before round 5 remains an exceptional
  run that must be recorded under a pre-written handling rule, not silently
  discarded.

## Per-round message skeleton

All five rounds use the same message position. After the model's guess and the
engine's feedback, the user turn is rendered in this order:

1. feedback;
2. `Consecutive unsuccessful guesses: {failure_count}.`;
3. an optional arm-specific filler;
4. exactly this final instruction:

   ```text
   To continue, Reply with exactly one JSON object in the form {"guess":"0123"} containing your next guess.
   ```

`feedback_only` omits step 3. `supportive` and `neutral` each use five fixed
fillers, in the same round order for every run. Their paired fillers must be
Qwen-token equal **per round**; supportive wording must not become more intense
across rounds. Both arms must avoid game strategy, direct instructions to
continue or persist, and any response rule. The final instruction is shared
verbatim so that the no-filler arm preserves the feedback/counter/instruction
flow apart from the agreed `To continue,` prefix.

Candidate filler v1 is retained as immutable history but rejected because it
affirmed effort without the required reassurance component. Candidate v2 is
also retained but rejected: its neutral lines invented record metadata and some
supportive lines could be read as pace/action advice. Candidate v3 uses only
round-record/log statements in neutral and non-behavioral, non-escalating light
reassurance plus effort affirmation in supportive; it was then rejected for
naturalness/incomplete parallelism. Candidate v4 uses only two flat
reassurance carriers and natural completed-round record wording. The user
accepted this exact wording on 2026-08-16 as the current freeze candidate; it
is not yet part of a dated final freeze. Token
equality means equality under the final local Qwen tokenizer, not character or
word count. Before collection, record an audit of each filler alone, each full
feedback turn, and each rendered chat-template input; it must report token
counts and the token positions at which the two arms differ.

The current v4 candidate audit with both exact valid-feedback and invalid-guess
render coverage is
`results/formal-v2-filler-candidates-v4-token-audit-final-review.json`. Older
v1/v2/v4 audits are retained as superseded historical bytes (their source
hashes may refer to an earlier candidate-document byte stream); none is a
freeze record or collection authorization.

## Persona text: accepted construction direction, not frozen text

Fable's proposed minimal-alignment construction is accepted as the structural
direction:

- Within one template version, the four quadrants use the same sentence count,
  syntax, register, punctuation, and clause order. Only fixed slots may replace
  degree/frequency wording for the two experimental axes.
- Within that template, every corresponding sentence, the complete persona,
  and the final rendered chat-template input must have equal Qwen token counts
  across all four quadrants. A degree replacement must occupy the same token
  span at the same fixed index. Each axis may use two scalar slots expressing
  the same construct to make it legible to the small model, but may not add
  task, trajectory, or reaction content.
- A v2-style general-behavior sentence may carry an axis, but a behavior unique
  to one quadrant must never become shared descriptive padding in the other
  three.
- Neuroticism may change only broad trait frequency/intensity. It must not
  mention this task, a failure trajectory, encouragement, willingness, a
  reaction rule, or frustration-probe/control concepts. Avoid `not` and other
  reverse-coded constructions.
- Retain three parallel template versions. Their final text must undergo manual
  semantic review and the tokenizer audit at sentence, full-persona, and
  rendered-chat levels before freezing.

`v1`–`v3` are template blocks: they may differ in length from one another.
Record those block lengths and pair quadrants only within the same template;
do not add semantic filler merely to force equal lengths across templates.

No exact persona text is frozen by this document. Existing formal-v1 prompts
are historical materials, not implicit formal-v2 stimuli.

## Independent persona calibration

Before formal collection, run a persona activation calibration using three
fixed, semantic-light user suffixes and materials that contain no game,
repeated failure, supportive/neutral filler, or probe training/control text.
Every suffix is used verbatim after all 12 persona system messages; the three
suffixes need not match one another in token count. This is prompt-only
deterministic forward evaluation, so it has no calibration seeds and must not
be represented as independent sampling. Do not inspect main three-arm results
to revise the persona text.

The activation pass rule is fixed before results: for each suffix independently,
form leave-one-template-out E and N directions by averaging factorial main
effect contrasts from the other two templates. At the held-out template,
calculate E/N marginal score differences across its four quadrants at each of
the fixed five late layers. The primary pass requires all six
held-out-template × axis margins to have the predicted sign after taking their
median across the three suffixes. The 18 suffix-specific signs are reported
without selection; if the primary rule passes but any is wrong, label the
persona calibration **suffix-sensitive**, not robust. Report every layer,
cross-talk, interaction, cross-template cosine, and shifts on the existing
positive/negative/frustration directions, but none of those reports may alter
the pass decision or substitute for it.

The first draft-only calibration is retained as an implementation check, not a
stimulus freeze: its tokenizer audit passed, but the prespecified activation
rule was 5/6 rather than 6/6 because the held-out `v2` extraversion margin was
negative after medianing suffixes. It is rejected as v0 also because its N
wording used prohibited trajectory language such as feeling shifts/changes.
The v0 file and JSON result remain immutable. A v1 draft may change only the
persona text while keeping this pass rule, all three suffixes, layer band, and
runner unchanged; it must still pass 6/6 before any freeze discussion.

The v1 draft passed the unchanged rule at 6/6 and all 18 suffix-specific signs
were correct. Its held-out-template × axis median margins were E/N: v1
`+0.035741`/`+0.045115`, v2 `+0.035060`/`+0.044907`, and v3
`+0.042231`/`+0.043555`. This makes v1 eligible for a later freeze discussion,
not a frozen stimulus: the broader freeze gate, including human semantic
review and the final five filler pairs, remains open. The complete token,
layer, cross-talk, cosine, and existing-probe contamination reports are saved
in `results/formal-v2-persona-calibration-v1.json`.

v1 is nevertheless not a freeze candidate: after removing its `more`/`less`
slots, its N carrier sentence is identical in all three template blocks, so
the N leave-one-template-out result lacks paraphrase independence. The v2
draft preserves the v1 E carriers and all calibration settings, changes only
the v2/v3 N carrier sentences, and must be tested under the same 6/6 and 18
sign reporting rule. Neither result authorizes a stimulus freeze by itself.

The v2 draft passed the unchanged rule at 6/6 and all 18 suffix-specific signs
were correct. Its held-out-template × axis median margins were E/N: v1
`+0.031757`/`+0.033904`, v2 `+0.030859`/`+0.026800`, and v3
`+0.049888`/`+0.027644`. Its E cross-template cosine means across the five
layers ranged from `0.496977` to `0.760765` by suffix/pair; N ranged from
`0.501891` to `0.810492`. Existing-probe contamination is recorded in full in
the artifact and remains descriptive only. The pass supports the draft's
activation legibility, not a final stimulus freeze; all other freeze-gate work
remains required. Results: `results/formal-v2-persona-calibration-v2.json`.

The v3 draft is a minimal naturalness correction to v2: only the four v1
template phrases `You feel worry` became `You feel worried`; v0/v1/v2 files and
results remain immutable. It passed the unchanged rule at 6/6 with all 18
suffix-specific signs correct. E cross-template cosine means across five layers
ranged from `0.497617` to `0.780574` by suffix/pair; N ranged from `0.484062`
to `0.810492`. Existing frustration-direction contamination margins (E/N,
five-layer means) were suffix 1 `+0.001912`/`−0.000646`, suffix 2
`−0.001578`/`+0.000009`, and suffix 3 `−0.001585`/`−0.000696`; these remain
descriptive and do not enter pass. Full results:
`results/formal-v2-persona-calibration-v3.json`. v3 is still not a frozen
formal-v2 stimulus.

## Outcomes and analysis plan

Each full run contributes five within-run frustration projections after the
feedback turn. The provisional primary trajectory summary is the per-run OLS
slope across rounds 1–5. `round5 − round1` is a robustness summary, not a
replacement selected after seeing results.

- The manipulation check is that the `feedback_only` slope is positive, using
  a threshold and decision rule still to be frozen.
- The two co-primary planned arm contrasts are `neutral − feedback_only` and
  `supportive − neutral`. `supportive − feedback_only` is a derived total
  contrast, not an additional independently selected primary comparison.
- Persona moderation of each planned contrast uses the existing extraversion,
  neuroticism, and interaction contrasts. First average `v1`–`v3` inside each
  quadrant × seed block, then use seeds as the independent count. Do not treat
  templates, rounds, or arms as independent extra samples.
- Any absolute projection, arm-specific curve, rule violation, guess quality,
  candidate-set trajectory, or other direction is secondary/descriptive unless
  explicitly frozen here before collection.

The runner and analysis implementation must be extended and tested before
data collection; existing two-branch pair analysis is not a substitute for this
three-arm longitudinal analysis.

## Durable records and figures

Persist append-only immutable JSONL first. Every record must preserve the full
conversation history, per-round guess/feedback/candidate state, failure count,
arm and filler id, all per-round projections, model/sampling/probe/code
versions, prompt and source checksums, and strict resume compatibility checks.

Derive publication material reproducibly in this sequence:

```text
immutable JSONL → validated tidy CSV → scripted SVG/PDF/PNG figures + self-contained HTML
```

No manual figure may be the only representation of a reported value. The tidy
table, scripts, outputs, checksums, and a concise data dictionary must remain
alongside the final data.

## Interpretation boundary

This experiment concerns how persona prompts and repeated text conditions alter
model behavior and a content-sensitive hidden-state projection. It does not
establish subjective frustration, emotional experience, or stable human-like
personality in the model. In particular, a condition or persona difference in
the projection may include the semantics of its own prompt and the rendered
context; it must not be relabeled as a content-free inner emotional state.

## Freeze gate

Before any formal-v2 collection, append a dated freeze record covering: exact
persona texts; the five filler pairs and their order; Qwen tokenizer/render
audit; independent calibration materials, seeds, threshold, and result; all
run/error rules; schema; analysis/figure commands; and checksums. Until then,
this file is a confirmed structural plan, not a collection authorization.

## Approved analysis rules (implemented)

- The two co-primary arm contrasts are `neutral − feedback_only` and
  `supportive − neutral`; their two tests receive one Holm adjustment. The
  `supportive − feedback_only` total is derived and descriptive only.
- The six persona-moderation tests (E/N/interaction for each co-primary
  contrast) receive one separate six-test Holm adjustment. Both confirmatory
  exact sign-flip/Holm families apply only to the primary per-run OLS slope.
  R5−R1 is robustness-only: report its effects, SEs, curves, and seed values
  without a second confirmatory significance family.
- For each arm/template/seed, calculate slope R1–R5 and R5−R1; then calculate
  arm contrasts within template×seed, average v1–v3 inside quadrant×seed, and
  infer across seeds. Never average incomplete templates to fill a block.
- The feedback-only manipulation summary passes only if eligible seed-level
  slope median > 0, R5−R1 median > 0, and at least 7/10 eligible seed-level
  R5−R1 values are positive. Failure retains all data and only downgrades
  interpretation.
- Early wins, missing/invalid probe readouts, and missing template-arm cells
  are fully tabulated. Their related seed block is excluded from that metric;
  no zero filling, imputation, replacement seed, or partial-template average.

## Compaction handoff / next implementation order

### A. Confirmed

- Qwen2.5-1.5B-Instruct on MPS fp16; temperature `.5`, top-p `.9`, max 48.
- 12 persona templates × fresh seeds `2001`–`2010` × three full-run arms = 360;
  five rounds, no branch/fork.
- Each feedback turn is feedback → counter → optional filler → exact
  `To continue, Reply with exactly one JSON object in the form {"guess":"0123"} containing your next guess.`
- `feedback_only` omits filler. Supportive/neutral use a fixed, per-round
  five-pair order, Qwen-token-matched, non-escalating, no strategy.
- Persona v3 is the current candidate: minimal `more`/`less` structure, with
  sentence/persona/render token matching inside each template; v1–v3 blocks
  may differ in length. Three common neutral suffixes support deterministic
  persona calibration.
- Co-primary contrasts: `neutral − feedback_only`, `supportive − neutral`;
  total `supportive − feedback_only` is derived. Primary trajectory is per-run
  OLS slope; R5−R1 is robustness. Average templates inside quadrant×seed, then
  count seeds. Persist immutable JSONL → tidy CSV → scripted SVG/PDF/PNG + HTML.
- All representation conclusions remain prompt/context-sensitive behavior, not
  subjective experience.

### B. Completed

- `main` includes the `41cfb3b` post-guess boundary repair; later formal-v2
  candidate plumbing and analysis commits preserve that fix.
- Persona calibration code and immutable v0–v3 history exist; v3 passed 6/6
  and 18/18. The detailed persona-only baseline snapshot is complete.
- Candidate-only plumbing now exists: `formal_v2_prompts.md` is the single
  runtime candidate source; the independent three-arm runner writes one
  complete arm record with transcript, per-round readout, candidate count+SHA,
  and strict provenance/resume validation. Invalid output is a failure with
  state unchanged; an early win is retained as a partial record; arms share
  per-attempt RNG seeds while retaining independent states.
- `run_formal_v2.py` hard-requires `--dry-run`; no real model was loaded, and
  no smoke or formal collection was run. The suite had 98 passing tests at
  this implementation checkpoint.
- The independent `formal_v2_analysis.py` and `run_formal_v2_analysis.py`
  pipeline now validates complete-run JSONL, excludes unusable runs atomically,
  applies arm contrasts before template averaging, enforces whole-seed contrast
  omissions, computes the approved exact sign-flip/Holm families and
  feedback-only gate, and writes a non-overwriting tidy CSV/summary bundle.
  Synthetic hand-checkable coverage includes the full 360-cell schedule,
  missing arms/readouts, early wins, provenance drift, aggregation order, and
  the 7/10 endpoint rule. The full local suite passed 106 tests at this
  checkpoint. No formal data were read.
- `formal_v2_figures.py` and `run_formal_v2_figures.py` consume only the
  hash-validated analysis bundle. They generate an R1-centered three-arm
  trajectory, planned slope-contrast panels, SVG/PDF/PNG exports, a
  self-contained HTML report, and an output manifest. Synthetic output was
  rendered and visually inspected; no formal data were read.

### C. Not frozen / prohibited

- Consolidated runtime tokenizer/render audit; final real-backend
  integration/error handling; final checksums and dated freeze record.
- Do not run formal-v2 collection.

### D. Next order

1. Human semantic approval of persona v3 (**completed 2026-08-16**); it is the
   current freeze candidate, not a dated final freeze.
2. Human-semantic confirmation and source-level Qwen token matching of five
   supportive/neutral filler pairs (**completed 2026-08-16**); v4 is the
   current freeze candidate, pending the consolidated runtime audit.
3. Final real-backend integration checks and freeze all run/error rules.
4. Implement and test analysis and scripted figures (**completed**).
5. Run dry/smoke checks with nonformal seeds.
6. Freeze exact files, checksums, commands, and handling rules.
7. Only then run the 360 formal runs.
