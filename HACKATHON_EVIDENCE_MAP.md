# Hackathon evidence map — working notes, not submission prose

Purpose: keep claims, evidence strength, and interpretation boundaries separate
before filling the hackathon template. This file is not a paper draft.

## 1. Research question components

| Component | Operationalization | Current answer |
|---|---|---|
| Persona prompt is internally legible | Independent v3 prompt-only activation calibration | Yes: frozen LOTO rule passed 6/6; 18/18 suffix-specific signs |
| Repeated failure changes the frustration-related projection | Frozen feedback-only five-round manipulation gate | Yes: gate passed; median slope `+0.00176164`, median R5−R1 `+0.0106883`, 9/10 endpoints positive |
| Narrow encouragement changes next-guess quality | formal-v1 paired encouragement−neutral normalized information efficiency | No reliable evidence: `−0.00996`, exact p `0.750` |
| Narrow encouragement changes stated willingness | formal-v1 paired willingness rating | Descriptive `+0.217/10`, exact p `0.250`; not reliable evidence |
| Supportive reassurance changes the five-round internal trajectory | formal-v2 supportive−neutral frustration-direction slope | Yes: `+0.00086080`, Holm p `0.003906`; 10/10 seeds and all three templates directionally consistent |
| Persona moderates the behavioral effect | formal-v1 E/N/interaction contrasts | No reliable evidence |
| Persona moderates the formal-v2 trajectory effect | Six frozen E/N/interaction contrasts | No reliable evidence; all six Holm p `1.0` |
| Published discrete emotion directions replicate a supportive−neutral trajectory difference | formal-v3 joyful/grief-stricken/furious slopes | Yes: all three Holm p `0.005859`; grief-stricken sign opposed the frozen expectation |
| Persona moderates the formal-v3 discrete-concept effects | Nine frozen E/N/interaction contrasts | Six of nine survive Holm; interpretation remains prompt/concept alignment, not subjective response |
| Projection change identifies lower subjective frustration | Not operationally identifiable from these probes | No; prompt semantics and readout boundary affect the projection |

## 2. Evidence tiers

### A. Frozen confirmatory results

1. **formal-v1 behavior**
   - 120 paired runs; encouragement and neutral forked from the same round-5
     checkpoint.
   - Primary normalized-information-efficiency effect: `−0.00996`, SE
     `0.01031`, exact p `0.750`.
   - No E, N, or interaction evidence; template directions were inconsistent.

2. **formal-v2 manipulation**
   - 360/360 independent full-arm runs; 1,800 round readouts; no missing runs,
     early wins, or analysis exclusions.
   - Feedback-only manipulation gate passed all frozen criteria.

3. **formal-v2 co-primary trajectory contrasts**
   - Neutral−feedback-only: `+0.00006791`, Holm p `0.791016`.
   - Supportive−neutral: `+0.00086080`, Holm p `0.003906`.
   - Supportive−feedback-only: `+0.00092870`, descriptive derived total only.

4. **formal-v2 moderation**
   - No evidence for E, N, or interaction moderation of either co-primary
   contrast; all six Holm-adjusted p-values were `1.0`.

5. **formal-v3 external discrete concepts**
   - 360 complete from-start arm runs, 1,800 rounds, no exclusions.
   - Supportive−neutral slope: joyful `+0.00194528`, grief-stricken
     `+0.00086388`, furious `−0.00106509`; all Holm p `0.005859`.
   - Six of nine frozen persona moderation tests survive Holm.
   - Grief-stricken reversed its directional expectation, and feedback-only
     decomposition shows all three contrasts depend on exact text semantics.
   - Only 17 behavior paths occurred; inference uses 10 seed blocks.

### B. Prespecified robustness/descriptive results

- formal-v2 supportive−neutral R5−R1: `+0.00301323`, SE `0.00054518`;
  robustness-only, no confirmatory p-value family.
- Template-specific supportive−neutral slopes:
  - v1 `+0.00078849` (35/40 cells positive)
  - v2 `+0.00089096` (38/40 positive)
  - v3 `+0.00090295` (38/40 positive)
- Fully invalid formal-v2 attempts: feedback-only 12, supportive 8, neutral
  12. Descriptive only.

### C. Auxiliary internal-representation results

- formal-v1 immediately after the encouragement message:
  - positive `+0.034769`
  - negative `−0.022279`
  - frustration `−0.006397`
  - high-N personas showed a somewhat larger immediate frustration-direction
    decrease (N contrast `−0.000483`, within-metric Holm p `0.0117`).
- Corrected post-guess boundary:
  - positive `−0.003834`
  - negative `+0.010702`
  - frustration `+0.002307`
  - no persona contrast evidence.
- These are position- and content-sensitive context representations, not
  calibrated emotional quantities.

### D. Exploratory pilots — context only

- Direct continuation-cue willingness pilot: `+0.917/10`; exploratory and
  contaminated by explicit continuation language.
- Supportive reassurance willingness v1: `+0.200/10`, exact p `0.250`; did not
  reproduce the earlier large willingness effect.
- Pause/no-history prompt-state experiments showed that the projection's sign
  can reverse when history/wrapper/context changes. These support the content-
  sensitivity warning, not an emotional-state claim.

## 3. Safe claim cards

| Claim | Strength | Required qualifier |
|---|---|---|
| The persona prompts encoded the intended E/N distinctions in hidden states. | Supported by deterministic calibration | Say “prompt distinctions were linearly legible,” not “the model acquired personalities.” |
| Repeated adversarial failure produced an increasing frustration-related trajectory. | Frozen manipulation passed | Call it a cosine projection, not subjective frustration. |
| Supportive reassurance altered the frustration-related trajectory relative to matched neutral text. | Frozen co-primary evidence | The slope difference was positive; do not call it frustration reduction. |
| Persona did not reliably moderate the supportive trajectory effect. | Frozen null result | Say “no reliable evidence detected,” not “all personas respond identically.” |
| Persona did not reliably improve or worsen next-guess quality after encouragement. | formal-v1 primary null | Limited to this model, prompts, task, seeds, and power. |
| The probe is content- and boundary-sensitive. | Supported across audits/pilots | This limits latent-emotion interpretation; it does not make every projection meaningless. |

## 4. Claims not supported

- “Encouragement lowered the model's frustration.”
- “Support made the model more frustrated.”
- “The four personas felt the same.”
- “Persona never affects responses to encouragement.”
- “The high-neuroticism effect replicated across designs.”
- “The internal projection measures subjective emotion.”
- “The exploratory willingness increase is a confirmed treatment effect.”

## 5. Honest high-level story options (headlines only)

These are possible organizing frames, not chosen titles or submission text.

1. **Legible traits, weak modulation**
   - Persona traits were internally legible, but did not reliably moderate
     behavioral or trajectory responses.

2. **Support changes representation, not necessarily affect**
   - Supportive text produced a robust hidden-state trajectory difference, yet
     content sensitivity prevented interpreting it as reduced frustration.

3. **A measurement lesson from a null moderation result**
   - A controlled null result exposed the distinction between prompt-semantic
     projection and latent emotional-state inference.

## 6. Artifact index

- Frozen v1 protocol/results: `FORMAL_PROTOCOL.md`, `FORMAL_RESULTS.md`
- Frozen v2 protocol/results: `FORMAL_V2_PROTOCOL.md`, `FORMAL_V2_RESULTS.md`
- Persona calibration: `FORMAL_V2_PERSONA_CALIBRATION_RESULTS.md`
- Raw formal-v2 data: `results/formal-v2.jsonl`
- Formal-v2 analysis summary: `results/formal-v2-analysis/summary.json`
- Formal-v2 visual report: `results/formal-v2-figures/formal-v2-report.html`
- Frozen v3 protocol/results: `FORMAL_V3_PROTOCOL.md`, `FORMAL_V3_RESULTS.md`
- Raw/analysis/visual v3 artifacts: `results/formal-v3.jsonl`,
  `results/formal-v3-analysis/`, `results/formal-v3-figures/formal-v3-report.html`

## 7. Items to decide together after receiving the template

- Which study is the main result: formal-v1 behavior, formal-v2 trajectory, or
  the combined measurement story.
- Whether exploratory willingness belongs in the main text, a limitations
  box, or is omitted.
- Desired tone: technical negative result, measurement caution, or personal
  hackathon learning story.
- Word/figure limits and required submission sections.
