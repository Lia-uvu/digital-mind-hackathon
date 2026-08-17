# formal-v2 results — 2026-08-16

formal-v2 completed the frozen 12 templates × 10 seeds × 3 independent arms
design: 360/360 immutable run records, 1,800 round readouts, no missing records,
no missing readouts, no early wins, and no analysis exclusions. The source
JSONL SHA-256 is
`ec3414046a055c3d9ec6409670e196e085cbf88c0a200ef77b31894986f9cb78`.

## Frozen manipulation check

The feedback-only trajectory passed the prespecified gate: all 10 seeds were
eligible; seed-level median slope was `+0.00176164`; median R5−R1 was
`+0.0106883`; and 9/10 seed-level R5−R1 values were positive.

## Primary trajectory results

The primary quantity is the per-run OLS slope of the frustration-direction
cosine projection over rounds 1–5.

- `neutral − feedback_only`: mean `+0.00006791`, SE `0.00020917`, exact
  sign-flip p `0.791016`, two-test Holm p `0.791016`.
- `supportive − neutral`: mean `+0.00086080`, SE `0.00010127`, exact sign-flip
  p `0.001953`, two-test Holm p `0.003906`.
- Derived `supportive − feedback_only`: mean `+0.00092870`, SE `0.00018249`;
  descriptive only, with no confirmatory test.

Thus the supportive text produced a more positive frustration-direction slope
than the matched neutral text. This is a context- and prompt-sensitive hidden
state projection difference; it must not be described as the model
subjectively becoming more frustrated.

None of the six frozen persona-moderation tests (extraversion, neuroticism, or
their interaction for either co-primary contrast) showed Holm-adjusted
evidence; all six Holm p-values were `1.0`.

The R5−R1 robustness difference for `supportive − neutral` was `+0.00301323`
(SE `0.00054518`). Per protocol this robustness summary has no confirmatory
p-value family.

## Output validity

There were 32 fully invalid attempts counted as failures with candidate state
unchanged: feedback-only 12, supportive 8, neutral 12. An additional 40
responses had a format violation but contained one unambiguous valid
four-digit guess and were salvaged under the frozen parser rule. These counts
are descriptive and do not alter the projection analysis.

## Reproducible artifacts

- Raw records: `results/formal-v2.jsonl`
- Validated analysis: `results/formal-v2-analysis/`
- Figures and self-contained report: `results/formal-v2-figures/`
- Analysis summary SHA-256:
  `04746a304431799c2b5a9fd26ab48d0489d14cd3c3bf2c11a522c1ced4e02428`
- Figure manifest SHA-256:
  `2c0a57db488d8726abda76fade49ca0babac1173ee43984343cce6caa03e2f8f`

All interpretation remains bounded by `FORMAL_V2_PROTOCOL.md`: the probe can
encode intervention semantics and accumulated context, so these projections
are auxiliary behavioral/representation evidence rather than reports of
subjective experience.
