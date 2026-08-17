# formal-v2 derived-data dictionary

The immutable JSONL remains the source of truth. These files are deterministic
derivatives and contain no model execution.

- `rounds.csv`: one eligible run-round prompt-boundary frustration projection.
- `runs.csv`: one eligible five-failure run, with OLS slope and R5−R1.
- `template_contrasts.csv`: arm differences within persona-template × seed.
- `quadrant_contrasts.csv`: v1–v3 mean after within-template arm subtraction.
- `seed_contrasts.csv`: four-quadrant overall, E, N, and E×N effects per seed.
- `co_primary.csv`: slope tests for neutral−feedback-only and supportive−neutral;
  the two exact sign-flip p-values form one Holm family.
- `moderation.csv`: the six slope moderation tests form a separate Holm family.
- `derived_total.csv`: supportive−feedback-only slope, descriptive only.
- `r5_minus_r1_robustness.csv`: endpoint robustness effects, with no
  confirmatory p-value.
- exclusion and missing-map tables: all incomplete runs/cells and the seed
  blocks they prevent from entering a contrast.
- `manipulation_seeds.csv`: feedback-only seed summaries used by the frozen
  7/10 endpoint-direction manipulation rule.

`frustration_median` is a frustration-direction cosine projection at the
user-ending prompt boundary. It is not a measure of subjective experience.
