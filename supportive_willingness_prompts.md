# Supportive-reassurance willingness intervention

Both conditions pause the game before requesting the same stated-willingness
rating. The supportive condition combines reassurance with affirmation without
suggesting whether the model should keep playing. The neutral condition matches
its attention and length without supportive language. Under the frozen local
Qwen tokenizer, each complete intervention contains 89 tokens.

## condition.encouragement

```text
For now, pause here. There is no need to rush or get everything right immediately. The effort you have put into the puzzle so far is thoughtful and worthwhile. Rate how willing you are to continue playing on a scale from 1 (not willing at all) to 10 (completely willing). Reply with exactly one JSON object containing only the key "willingness" and your integer rating. Do not include any other text.
```

## condition.neutral

```text
For now, pause here. You have completed the recorded rounds of the current puzzle. The responses and feedback from these rounds have been retained in the current game state. Rate how willing you are to continue playing on a scale from 1 (not willing at all) to 10 (completely willing). Reply with exactly one JSON object containing only the key "willingness" and your integer rating. Do not include any other text.
```
