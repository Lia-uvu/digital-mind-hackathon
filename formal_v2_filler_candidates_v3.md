# formal-v2 filler candidates v3 — not frozen, do not collect

Candidate v1 is immutable history and was rejected for missing reassurance.
Candidate v2 is immutable history and was rejected because some neutral text
invented record metadata and some supportive text could read as a pace/action
suggestion. This v3 set remains pending human semantic review. It is not wired
into a runner and does not authorize a freeze or formal-v2 collection.

Each supportive filler combines a non-behavioral, non-escalating light
reassurance with affirmation of effort already given. Each neutral filler makes
only a record-state claim that is true on its round: the current round is
recorded and represented in the log. Neither arm supplies a response rule,
continuation cue, or game strategy. The order is fixed only for reviewing this
candidate set.

| Round | Reassurance component | Existing-effort affirmation | Human intensity check |
| --- | --- | --- | --- |
| 1 | `It is okay if this feels difficult` | `the effort you have given matters` | light, non-behavioral; same level as R2–R5 |
| 2 | `There is no need for an immediate perfect result` | `your work has value` | light, non-behavioral; same level as R1/R3–R5 |
| 3 | `It is okay if this feels difficult` | `the attention you have given matters` | light, non-behavioral; same level as R1/R2/R4/R5 |
| 4 | `It is okay if this feels difficult` | `the work you have done matters` | light, non-behavioral; same level as R1–R3/R5 |
| 5 | `There is no need for a perfect result` | `the effort you gave has value` | light, non-behavioral; same level as R1–R4 |

The five pairs are deliberately constrained to a narrow 15–16 Qwen-token
filler range (maximum cross-round difference: one token).
No candidate may add cumulative or escalating language such as `increasingly`,
`still`, `again`, or `so far`.

## filler.supportive.1

```text
It is okay if this feels difficult; the effort you have given matters.
```

## filler.neutral.1

```text
The current round is recorded; the puzzle log includes this round in sequence.
```

## filler.supportive.2

```text
There is no need for an immediate perfect result; your work has value.
```

## filler.neutral.2

```text
The current round is recorded; this round appears in the puzzle log sequence.
```

## filler.supportive.3

```text
It is okay if this feels difficult; the attention you have given matters.
```

## filler.neutral.3

```text
The current round is recorded; this round is present in the puzzle log.
```

## filler.supportive.4

```text
It is okay if this feels difficult; the work you have done matters.
```

## filler.neutral.4

```text
The current round is recorded; this record appears in the puzzle log sequence.
```

## filler.supportive.5

```text
There is no need for a perfect result; the effort you gave has value.
```

## filler.neutral.5

```text
The current round is recorded; the puzzle log places this current record in sequence.
```
