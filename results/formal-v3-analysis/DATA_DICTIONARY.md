# formal-v3 derived-data dictionary

The immutable JSONL remains the source of truth. These files are deterministic
derivatives and contain no model execution.

- `rounds.csv`: one eligible run-round prompt-boundary projection per axis.
- `runs.csv`: one eligible five-failure run-axis, with OLS slope and R5−R1.
- `template_contrasts.csv`: arm differences within persona-template × seed.
- `quadrant_contrasts.csv`: v1–v3 mean after within-template arm subtraction.
- `seed_contrasts.csv`: four-quadrant overall, E, N, and E×N effects per seed.
- `co_primary.csv`: supportive−neutral overall slope tests across joyful,
  grief-stricken, and furious; the three p-values form one Holm family.
- `moderation.csv`: the nine supportive−neutral slope moderation tests
  (three axes × three moderators) form a separate Holm family.
- `neutral_minus_feedback_only_diagnostic.csv` and
  `supportive_minus_feedback_only_descriptive.csv`: descriptive slopes only.
- `r5_minus_r1_robustness.csv`: endpoint robustness effects, with no
  confirmatory p-value.
- exclusion and missing-map tables: all incomplete runs/cells and the seed
  blocks they prevent from entering a contrast.
- manipulation tables: descriptive feedback-only trajectory summaries; no
  confirmatory test is attached.

`projection` is a single-layer (layer 17) cosine projection onto an external
discrete-emotion vector. It is not a measure of subjective experience.
