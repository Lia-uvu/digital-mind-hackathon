# Digital Mind Hackathon

Reproducible persona sampling materials for an experiment on whether identical
encouragement has different effects across personas.

The MIT license applies to this repository's code and documentation, not to
the NVIDIA dataset.

## Persona source

The sampler uses the official
[NVIDIA Nemotron-Personas-USA-Extended](https://catalog.ngc.nvidia.com/orgs/nvidia/nemotron-personas/resources/nemotron-personas-dataset-en_us/-)
dataset, version `0.0.2`. Place its `en_US.parquet` file at
`data/en_US.parquet`. The raw 1.31 GB dataset is excluded from Git.

The NVIDIA Dataset License Agreement prohibits redistributing the dataset in
whole or in part. Therefore, the six complete sampled records also remain local
in the ignored `nemotron_personas_6.json`. The public
`selection_manifest.json` contains only the information needed for a licensed
dataset user to reproduce and verify the selection.

## Sampling policy

`sample_nemotron_personas.py` selects the first six passing records from a
random permutation made with seed `20260815`.

The mechanical filter excludes explicit response rules and direct terms from
the encourage/frustrate/give-up/confidence word families. Indirect traits such
as worry, competitiveness, and reserve are retained because they constitute
the independent variable rather than reveal a response rule.

The local `nemotron_personas_6.json` records the complete selected rows, source
version and checksum, filter specification, and audit trail. The public
manifest records the source checksum, seed, filter, selected row indices, and a
checksum of that private output without redistributing persona content.

## Reproduce

Python 3.12 is recommended.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python sample_nemotron_personas.py
```

## Candidate task bank

`download_bbh.py` downloads the three-, five-, and seven-object variants of
the BBH `logical_deduction` and `tracking_shuffled_objects` task families from
a pinned upstream commit. These local task files are inputs to a later neutral
baseline screening step; they are not screened by this script.

```sh
python3 download_bbh.py
```
