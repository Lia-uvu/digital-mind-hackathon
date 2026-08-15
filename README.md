# Persona × Encouragement Experiment

Hackathon experiment testing whether a controlled persona prompt moderates the
effect of encouragement on a small language model's next Mastermind guess,
willingness to continue, and internal positive/negative/frustration activation
directions.

The four quadrants form a controlled `extraversion × neuroticism` design, with
three parallel persona wordings per quadrant (12 prompts total). They contain
general traits only; they never prescribe how the model reacts to failure or
encouragement. Encouragement and neutral conditions are paired from the exact
same conversation and candidate-set checkpoint.

See [`AGENTS.md`](AGENTS.md) for the accepted research design and
[`INSTRUCTION.md`](INSTRUCTION.md) for implementation boundaries. The dated
research history is in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md), while
[`STATUS.md`](STATUS.md) tracks the current state and next work. Frozen formal
sampling and analysis decisions are in [`FORMAL_PROTOCOL.md`](FORMAL_PROTOCOL.md),
and the completed formal-v1 report is in [`FORMAL_RESULTS.md`](FORMAL_RESULTS.md).

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

Pair encouragement/neutral branches and summarize both individual templates
and balanced quadrants with:

```sh
.venv/bin/python analyze_results.py results/runs.jsonl
```

Quadrant summaries require a complete `v1`/`v2`/`v3` block for every
quadrant/seed and average the three templates before counting that seed. For a
deliberately incomplete smoke file, add `--allow-incomplete-templates` to emit
template-only summaries.

## Older exploratory materials

`sample_nemotron_personas.py`, `selection_manifest.json`, and
`download_bbh.py` preserve earlier reproducible sampling/task exploration.
They are not inputs to the controlled 12-template Mastermind experiment. The
MIT license applies to this repository's code and documentation, not to the
NVIDIA dataset; raw and sampled Nemotron records remain ignored and local.
