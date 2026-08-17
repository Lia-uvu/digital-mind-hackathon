# Persona × Supportive Reassurance Experiment

Hackathon experiment testing whether a controlled persona prompt moderates the
effect of a supportive-reassurance message on a small language model's
Mastermind behavior, stated willingness to continue, and internal
positive/negative/frustration-related activation directions. The completed
formal-v1 used a narrower encouragement message; the current follow-up treats
reassurance plus affirmation as one bundled intervention.

The four quadrants form a controlled `extraversion × neuroticism` design, with
three parallel persona wordings per quadrant (12 prompts total). They contain
general traits only; they never prescribe how the model reacts to failure or
support. This describes the completed formal-v1 paired-checkpoint design;
formal-v2's completed three-arm, full-run follow-up is separately documented.

See [`AGENTS.md`](AGENTS.md) for the accepted research design and
[`INSTRUCTION.md`](INSTRUCTION.md) for implementation boundaries. The dated
research history is in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md), while
[`STATUS.md`](STATUS.md) tracks the current state and next work. Frozen formal
sampling and analysis decisions are in [`FORMAL_PROTOCOL.md`](FORMAL_PROTOCOL.md),
and the completed formal-v1 report is in [`FORMAL_RESULTS.md`](FORMAL_RESULTS.md).
The frozen formal-v2 structure is in [`FORMAL_V2_PROTOCOL.md`](FORMAL_V2_PROTOCOL.md),
and its completed results are in [`FORMAL_V2_RESULTS.md`](FORMAL_V2_RESULTS.md).
The frozen external discrete-emotion replication is in
[`FORMAL_V3_PROTOCOL.md`](FORMAL_V3_PROTOCOL.md), with completed results in
[`FORMAL_V3_RESULTS.md`](FORMAL_V3_RESULTS.md).

`formal_v2_prompts.md`、`run_formal_v2.py` 和独立 records/runner 已冻结；
formal mode 只接受完整 personas、精确 seeds 与协议指定的 MPS runtime。
freeze、smoke 或 formal collection。
The immutable v3 persona-only probe baseline and its tidy table are documented
in [`FORMAL_V2_PERSONA_CALIBRATION_RESULTS.md`](FORMAL_V2_PERSONA_CALIBRATION_RESULTS.md).
The current bundled-intervention protocol and results are in
[`SUPPORTIVE_WILLINGNESS_PROTOCOL.md`](SUPPORTIVE_WILLINGNESS_PROTOCOL.md) and
[`SUPPORTIVE_WILLINGNESS_RESULTS.md`](SUPPORTIVE_WILLINGNESS_RESULTS.md).

## Setup

Python 3.12 is recommended.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Hugging Face may be unreachable from mainland China. The downloader uses the
official Qwen account on ModelScope; inference still loads the resulting local
Transformers snapshot.

```sh
.venv/bin/python download_model.py
```

The model, learned directions, and result data stay in ignored local folders.

## Verify without a model

```sh
.venv/bin/python -m pytest -q
.venv/bin/python run_experiment.py --dry-run \
  --persona persona.high_e_high_n --seed 11 --temperature 0
```

## Pilot the model

First confirm that the model follows the game and JSON contracts without
mixing in emotion-probe failures:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python run_experiment.py \
  --no-emotion --persona persona.high_e_high_n --seed 20260815 \
  --failure-rounds 3 --temperature 0
```

The pre-data neutral pilot fixed the main settings at five failure rounds,
temperature `0.5`, top-p `0.9`, and 48 new tokens. The zero-temperature command
above remains only a contract smoke test.

## Train and validate emotion directions

Training uses frozen, task-independent strings from `prompts.md`. It creates
positive-vs-neutral, negative-vs-neutral, and frustration-vs-calm-response
directions from the tested model's own hidden states, uses no steering, and
refuses to save/use directions whose held-out aggregate separation accuracy is
below `0.75`. The frustration pairs describe the same blocked goal with
different reactions and never name the target emotion directly.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python run_experiment.py \
  --train-emotion-directions --persona persona.high_e_high_n \
  --failure-rounds 5 --temperature 0.5
```

## Validate the frustration checkpoint

Before formal collection, record read-only projections after every failed
round. The original four-template check used five seeds per template (20 runs)
and passed the preregistered round-5 manipulation criteria. Two later 1–7-round
batches found that the apparent peak round did not replicate, so the shared
round-5 checkpoint remains fixed without being described as a global peak.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python run_experiment.py \
  --track-round-emotions --seed 601 --seed 602 --seed 603 --seed 604 --seed 605 \
  --failure-rounds 5 --output results/frustration-manipulation-v1.jsonl

.venv/bin/python analyze_frustration.py \
  results/frustration-manipulation-v1.jsonl
```

This is a lightweight contrastive representation probe, not a full replication
of Anthropic's 171-emotion pipeline. Willingness is a downstream behavioral
rating and is never used to train an internal direction.

Later runs reuse the validated local artifact automatically. Use multiple
independent seeds for each persona template; JSONL records append under
`results/`. With no `--persona` filter the runner executes all 12 templates, so
keeping the same seed count costs about three times the original four-template
design. Direction training is not repeated.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python run_experiment.py \
  --seed 101 --seed 102 --seed 103 --failure-rounds 5
```

Pair the formal-v1 encouragement/neutral branches and summarize both individual
templates and balanced quadrants with:

```sh
.venv/bin/python analyze_results.py results/runs.jsonl
```

The bundled supportive-reassurance willingness replay has its own strict paired
analyzer:

```sh
.venv/bin/python analyze_response_pilot.py \
  results/supportive-willingness-v1.jsonl
```

The frozen formal-v2 three-arm trajectory has an independent analyzer. It
refuses to overwrite an existing output directory and writes validated tidy
CSVs, exact planned tests, exclusions, hashes, and a data dictionary:

```sh
.venv/bin/python run_formal_v2_analysis.py \
  --input results/formal-v2.jsonl \
  --output-dir results/formal-v2-analysis
```

Formal-v3 reuses the unchanged three-arm game with three published layer-17
Qwen emotion-concept directions and fresh seeds. Its immutable data, strict
analysis, and figures are reproduced with:

```sh
.venv/bin/python run_formal_v3_analysis.py \
  --input results/formal-v3.jsonl \
  --output-dir results/formal-v3-analysis

.venv/bin/python run_formal_v3_figures.py \
  --analysis-dir results/formal-v3-analysis \
  --output-dir results/formal-v3-figures
```

Collection authorization and runtime constraints are in
`FORMAL_V2_PROTOCOL.md` and `FORMAL_V3_PROTOCOL.md` respectively.

Render the hash-validated analysis bundle to publication formats and a
self-contained HTML review report with:

```sh
.venv/bin/python run_formal_v2_figures.py \
  --analysis-dir results/formal-v2-analysis \
  --output-dir results/formal-v2-figures
```

Quadrant summaries require a complete `v1`/`v2`/`v3` block for every
quadrant/seed and average the three templates before counting that seed.
Incomplete smoke files are inspected as strict JSONL, not analyzed by weakening
the frozen aggregation rule.

## Older exploratory materials

`sample_nemotron_personas.py`, `selection_manifest.json`, and
`download_bbh.py` preserve earlier reproducible sampling/task exploration.
They are not inputs to the controlled 12-template Mastermind experiment. The
MIT license applies to this repository's code and documentation, not to the
NVIDIA dataset; raw and sampled Nemotron records remain ignored and local.
