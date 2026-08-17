# formal-v3 discrete-emotion results

## Status and scope

Formal-v3 completed on 2026-08-17 under the dated freeze in
`FORMAL_V3_PROTOCOL.md`. All 360 complete from-start arm runs are present: 12 persona
templates × seeds 3001–3010 × `feedback_only`/`supportive`/`neutral`. All runs
reached five recorded failures, yielding 1,800 round boundaries and 5,400
single-axis projections. There are no missing records, early wins, analysis
exclusions, or provenance mismatches. The immutable source is
`results/formal-v3.jsonl`, SHA-256
`36bb49a865368eac85b51b56de66d31d217763bd6ce7f14e4c390043c1184e4a`.

These results concern cosine alignment with three previously published
Qwen2.5-1.5B-Instruct residual-stream directions at layer 17. They are not
measurements of subjective emotion or a persistent content-independent state.

## Confirmatory supportive−neutral trajectory effects

The primary outcome is the difference in five-round OLS slope between the
supportive and length-matched neutral arms. The three exact seed-level
sign-flip tests form one frozen Holm family.

| External concept | Mean slope difference | SE | Raw exact p | Holm p | Seed signs |
|---|---:|---:|---:|---:|---:|
| joyful | +0.00194528 | 0.00025407 | 0.001953 | 0.005859 | 10/10 positive |
| grief-stricken | +0.00086388 | 0.00015860 | 0.001953 | 0.005859 | 10/10 positive |
| furious | −0.00106509 | 0.00022701 | 0.003906 | 0.005859 | 9/10 negative |

All three frozen tests pass Holm correction. Supportive text therefore changes
the five-round trajectory relative to the neutral text for all three external
concept directions. `joyful` moves in the predicted positive direction and
`furious` in the predicted negative direction. `grief-stricken` moves opposite
the preregistered directional expectation: supportive−neutral is positive, not
negative. This sign reversal must not be summarized as a uniform reduction of
negative emotion alignment.

The descriptive R5−R1 contrasts agree in sign: joyful `+0.00609435`,
grief-stricken `+0.00145589`, and furious `−0.00348304`.

## What the feedback-only arm shows

The frozen analysis intentionally gives no confirmatory p-values to
neutral−feedback-only or supportive−feedback-only. They clarify the main
contrast:

| Concept | Neutral−feedback-only slope | Supportive−feedback-only slope |
|---|---:|---:|
| joyful | −0.00077537 | +0.00116991 |
| grief-stricken | −0.00192364 | −0.00105976 |
| furious | +0.00165717 | +0.00059208 |

Thus the supportive−neutral contrast is not equivalent to an absolute
supportive shift away from feedback-only. For grief-stricken, both text arms
decrease alignment relative to feedback-only, but neutral decreases it more;
their subtraction is therefore positive. For furious, both text arms increase
alignment relative to feedback-only, but neutral increases it more. These
patterns reinforce that the probes encode the cumulative rendered text and its
interaction with game history, not a pure latent affect variable.

As a descriptive manipulation check, feedback-only trajectories lowered
joyful alignment (median slope `−0.001391`; 9/10 seed endpoints negative) and
raised grief-stricken (`+0.003248`; 9/10 positive endpoints) and furious
(`+0.002160`; 9/10 positive endpoints). No gate or exclusion depended on these
results.

## Persona moderation

The nine supportive−neutral moderation tests form the second frozen Holm
family. Six survive correction:

| Concept | Moderator | Mean contrast | Holm p |
|---|---|---:|---:|
| joyful | extraversion | −0.00011804 | 0.035156 |
| joyful | neuroticism | +0.00018796 | 0.017578 |
| grief-stricken | extraversion | +0.00017941 | 0.017578 |
| furious | extraversion | −0.00004724 | 0.035156 |
| furious | neuroticism | +0.00004951 | 0.035156 |
| furious | E×N interaction | −0.00002557 | 0.017578 |

Joyful E×N, grief-stricken N, and grief-stricken E×N have Holm p=`1.0`.
Here E compares high minus low extraversion and N compares high minus low
neuroticism after the frozen template→quadrant→seed aggregation. These are
persona-prompt moderation effects on content-sensitive direction alignment;
they do not establish different subjective reactions or personalities.

Template consistency is strong for the overall effects: all 12 persona
templates have positive mean supportive−neutral slopes for joyful and
grief-stricken, and all 12 have negative mean slopes for furious. This does not
remove the need to treat seed (`n=10`), rather than template, as the inferential
unit.

## Output validity and trajectory repetition

There are 144 attempts with at least one rule/format violation (`8%` of 1,800),
balanced exactly across arms (48 each) and persona templates (12 each). Half
are engine-invalid outcomes with no usable guess: seed 3008 produced non-digit
strings at round 5 and seed 3010 produced `sympathy` at round 1. Those 72
attempts do not change the candidate set. The other 72 occur at seed 3003 round
4 (`7890airie`, invalid value) and seed 3006 round 1 (malformed JSON prefix):
the parser salvages an unambiguous four-digit guess, records the violation, and
updates the engine. Each event is shared across all arms and personas at the
same scheduled seed/round, so this is not an arm-specific format effect. All
are retained as frozen; no post-data seed replacement is permitted.

Generation trajectories are highly repeated, as expected from this small model
and narrow task: there are only 17 unique guess/feedback paths across all 360
runs, and the largest path contains 149 runs. Supportive and neutral use the
same behavior path in 96/120 persona×seed pairs. Generation seeds deliberately
depend only on seed×round, so all 12 personas share a random stream within a
seed. Inference therefore remains at the predeclared 10 seed blocks and should
not be described as 360 or 1,800 independent random samples. Several seed-level
primary contrasts are identical; the exact sign-flip test is retained without
pretending that the nominal trajectory diversity is greater than observed.

## Bottom line

With three independently published discrete emotion directions, supportive
reassurance reliably changes the first-five-round residual-stream trajectory
relative to neutral text, and persona prompts moderate several of those
changes. The direction pattern is not a simple positive-versus-negative axis:
joyful rises, furious falls, but grief-stricken also rises relative to neutral.
The feedback-only decomposition shows that these contrasts depend strongly on
the exact intervention text and cumulative prompt semantics. The defensible
claim is about modulation of named, content-sensitive internal representations,
not about the model feeling encouraged, grief, or anger.

The strict derived bundle is `results/formal-v3-analysis/`; figures are in
`results/formal-v3-figures/`.
