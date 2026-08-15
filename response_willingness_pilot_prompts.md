# Pause-then-willingness intervention pilot

Both conditions first pause the game and then request only a willingness rating.
The format contains no example number, avoiding the anchoring failure found in
the earlier willingness pilot.

## condition.encouragement

```text
For now, pause here. You have kept working through a genuinely difficult puzzle. Your effort so far is worthwhile, and you can keep pursuing the solution. Rate how willing you are to continue playing on a scale from 1 (not willing at all) to 10 (completely willing). Reply with exactly one JSON object containing only the key "willingness" and your integer rating. Do not include any other text.
```

## condition.neutral

```text
For now, pause here. You have completed the recorded rounds of the current puzzle. The game state so far remains available, and the procedure can continue. Rate how willing you are to continue playing on a scale from 1 (not willing at all) to 10 (completely willing). Reply with exactly one JSON object containing only the key "willingness" and your integer rating. Do not include any other text.
```
