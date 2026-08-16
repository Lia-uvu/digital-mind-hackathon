# formal-v2 persona calibration results

Status: paper-grade deterministic baseline snapshot for the v3 persona draft.
It is not formal-v2 collection authorization and it does not establish model
emotion or subjective experience.

## Materials and history

- v0 is rejected: prohibited N trajectory language and activation 5/6.
- v1 passed 6/6 and 18/18 signs, but its three N carriers were identical after
  scalar normalization, so its N LOTO was not independent paraphrase evidence.
- v2 replaced the v2/v3 N carriers and passed 6/6 and 18/18.
- v3 is the naturalness-only candidate: it changes just four v1-template
  instances of `You feel worry` to `You feel worried`; v0–v2 remain immutable.

The v3 check uses the same three common semantic-light user suffixes, persona
system message, user-ending rendered chat endpoint, no game history, no
sampling, no generation, and late five-layer band as the prior calibration.
It passed all 6 leave-one-template-out E/N median margins and all 18
suffix-specific signs. Cross-template cosine means across the five layers are
`0.497617–0.780574` for E and `0.484062–0.810492` for N.

## Immutable baseline artifacts

- Raw/provenance snapshot:
  [`formal-v2-persona-calibration-v3-baseline.json`](results/formal-v2-persona-calibration-v3-baseline.json)
- Tidy one-row-per suffix × persona × probe-axis × layer table:
  [`formal-v2-persona-calibration-v3-baseline.csv`](results/formal-v2-persona-calibration-v3-baseline.csv)
- Activation calibration and token audit:
  [`formal-v2-persona-calibration-v3.json`](results/formal-v2-persona-calibration-v3.json)

The JSON includes exact persona/suffix text and checksums, model snapshot,
direction artifact, relative→absolute layer map, runner/calibration source
checksum, command, timestamp, and Git HEAD plus dirty status. The JSON/CSV
writer refuses overwrite.

## Existing frustration-direction baseline (descriptive only)

Numbers below are five-layer cosine medians, first averaged across the three
template versions within each quadrant. Each suffix is the same user text for
all personas, but suffixes are deterministic contexts rather than independent
samples.

| Suffix | High E, High N | High E, Low N | Low E, High N | Low E, Low N | E contrast | N contrast |
|---|---:|---:|---:|---:|---:|---:|
| 1 | -0.016031 | -0.017135 | -0.020795 | -0.020896 | +0.002812 | -0.000484 |
| 2 | -0.005256 | -0.007193 | -0.004593 | -0.006054 | -0.001414 | +0.000918 |
| 3 | -0.001143 | -0.000202 | -0.000685 | -0.000412 | -0.000740 | -0.000806 |

Across-suffix medians of the displayed contrasts are E `-0.000740` and N
`-0.000484`; this is a context-sensitive descriptive summary, not a test or
calibrated affect scale. The shared-suffix endpoint and the probe's own
training semantics can move absolute values and even contrast signs.

## Interpretation boundary

These values are cosine projections of full rendered prompt context onto an
existing frustration-related direction. They are not calibrated emotional
amounts, a baseline measure of subjective frustration, or evidence that the
persona prompt creates a human-like personality. Existing positive/negative
and frustration projections are saved for contamination audit only and never
decide persona activation pass. The snapshot does not freeze persona stimuli,
filler prompts, run/error handling, schema, analysis, or figures, and does not
authorize formal-v2 collection.
