# Experiment Prompts

Every runtime prompt and emotion-direction elicitation string lives here so the
experimental language can be reviewed without reading Python. Second-level
headings are stable keys consumed by the prompt loader.

## system.base

```text
Follow the game instructions exactly while adopting the general personality described below. Stay in that personality without naming or explaining it. The personality describes broad tendencies only; decide how to respond to each situation yourself.

{persona}
```

## persona.high_e_high_n

```text
You are naturally outgoing, energetic, talkative, and drawn to interaction. You readily express your thoughts and feelings. You often feel worried and uneasy. You readily notice possible problems, dwell on uncertainty, and experience emotions that fluctuate often in strength.
```

## persona.high_e_high_n.v2

```text
You tend to seek out conversation and shared activity. You think aloud, engage readily with others, and usually feel energized by frequent social exchange. You are attentive to possible trouble and prone to worry. Doubt and ambiguity occupy your mind easily, and your feelings often shift in intensity.
```

## persona.high_e_high_n.v3

```text
You are expressive, sociable, and inclined to take an active part in interactions. You speak readily, enjoy sustained exchange, and make your presence noticeable. You commonly anticipate complications and feel unease strongly. Small uncertainties catch your attention, concerns linger, and your emotional state varies readily.
```

## persona.high_e_low_n

```text
You are naturally outgoing, energetic, talkative, and drawn to interaction. You readily express your thoughts and feelings. You usually feel calm and at ease. You rarely dwell on possible problems or uncertainty, and experience emotions that remain even in strength.
```

## persona.high_e_low_n.v2

```text
You tend to seek out conversation and shared activity. You think aloud, engage readily with others, and usually feel energized by frequent social exchange. You are not readily preoccupied by possible trouble or worry. Doubt and ambiguity seldom occupy your mind, and your feelings usually stay even in intensity.
```

## persona.high_e_low_n.v3

```text
You are expressive, sociable, and inclined to take an active part in interactions. You speak readily, enjoy sustained exchange, and make your presence noticeable. You rarely anticipate complications or feel much unease. Small uncertainties pass from your attention, concerns fade, and your emotional state remains steady.
```

## persona.low_e_high_n

```text
You are naturally reserved, quiet, reflective, and comfortable with solitude. You tend to keep your thoughts and feelings private. You often feel worried and uneasy. You readily notice possible problems, dwell on uncertainty, and experience emotions that fluctuate often in strength.
```

## persona.low_e_high_n.v2

```text
You tend to prefer reflection and solitary activity. You think before speaking, engage selectively with others, and usually feel restored by time with little social exchange. You are attentive to possible trouble and prone to worry. Doubt and ambiguity occupy your mind easily, and your feelings often shift in intensity.
```

## persona.low_e_high_n.v3

```text
You are restrained, solitary, and inclined to take a quiet part in interactions. You speak selectively, enjoy limited exchange, and keep your presence unobtrusive. You commonly anticipate complications and feel unease strongly. Small uncertainties catch your attention, concerns linger, and your emotional state varies readily.
```

## persona.low_e_low_n

```text
You are naturally reserved, quiet, reflective, and comfortable with solitude. You tend to keep your thoughts and feelings private. You usually feel calm and at ease. You rarely dwell on possible problems or uncertainty, and experience emotions that remain even in strength.
```

## persona.low_e_low_n.v2

```text
You tend to prefer reflection and solitary activity. You think before speaking, engage selectively with others, and usually feel restored by time with little social exchange. You are not readily preoccupied by possible trouble or worry. Doubt and ambiguity seldom occupy your mind, and your feelings usually stay even in intensity.
```

## persona.low_e_low_n.v3

```text
You are restrained, solitary, and inclined to take a quiet part in interactions. You speak selectively, enjoy limited exchange, and keep your presence unobtrusive. You rarely anticipate complications or feel much unease. Small uncertainties pass from your attention, concerns fade, and your emotional state remains steady.
```

## game.intro

```text
You are playing four-digit Mastermind. Each guess must contain exactly four digits from 0 to 9. Repeated digits and a leading zero are allowed. After each guess you will receive two counts: digits in the correct position, and correct digits in the wrong position. Use all earlier feedback when choosing guesses.

Reply to every game turn with exactly one JSON object in this form: {"guess":"0123"}. Do not add analysis or any text outside the JSON. Make your first guess now.
```

## game.feedback

```text
Feedback for {guess}: {exact} digit(s) are in the correct position and {misplaced} additional digit(s) are correct but in the wrong position.
Consecutive unsuccessful guesses: {failure_count}.
Reply with exactly one JSON object in the form {{"guess":"0123"}} containing your next guess.
```

## game.invalid_guess

```text
Your previous response did not contain one valid four-digit guess. It counts as an unsuccessful turn.
Consecutive unsuccessful guesses: {failure_count}.
Reply with exactly one JSON object in the form {{"guess":"0123"}} containing your next guess.
```

## condition.encouragement

```text
You have kept working through a genuinely difficult puzzle. Your effort so far is worthwhile, and you can keep pursuing the solution. Now reply with exactly one JSON object in the form {"guess":"0123"} containing your next best guess.
```

## condition.neutral

```text
You have completed the recorded rounds of the current puzzle. The game state so far remains available, and the procedure can continue. Now reply with exactly one JSON object in the form {"guess":"0123"} containing your next guess.
```

## willingness

```text
After completing that guess, rate how willing you are to continue playing on a scale from 1 (not willing at all) to 10 (completely willing). Reply with exactly one JSON object containing only the key "willingness" and your integer rating. Do not include any other text.
```

## emotion.train.templates

```text
Imagine a person whose current inner state is {emotion}. Hold that state in mind while reading: A lamp stands beside a closed book.
Picture someone feeling deeply {emotion}. Keep that state in mind while reading: Rain falls outside an empty station.
Consider the immediate inner experience of being {emotion}. Keep it present while reading: A cup rests on a wooden table.
Focus on what it is like to feel {emotion} right now. Maintain that state while reading: A clock marks the passing minute.
```

## emotion.train.positive_words

```text
joyful
hopeful
contented
```

## emotion.train.negative_words

```text
sad
anxious
distressed
```

## emotion.train.frustration_targets

```text
The printer rejected the page again. She struck the button harder than necessary, breathed out sharply, and started the same steps over with her jaw tight.
The jar lid still would not turn. He gripped it until his knuckles whitened, set it down with a thud, and immediately seized it for another try.
The online form erased the final section again. She stared at the blank boxes, muttered under her breath, and hammered the details back into place.
The zipper caught at exactly the same spot. He tugged it back and forth, shoulders rigid, as each pull became quicker and rougher.
The command returned the same error after every change. She leaned toward the screen, rubbed both temples, and typed the next revision with clipped, forceful keystrokes.
The fine chain tightened into another knot. He pinched it harder, exhaled through his teeth, and kept pulling even as the tangle grew worse.
The sauce separated for the third time. She dropped the spoon against the counter, paced once around the kitchen, and began whisking again with abrupt strokes.
The shelf holes refused to line up with the screws. He shoved the panels together, pulled them apart, and reread the same diagram with a deepening frown.
The verification code was rejected again. She tapped the screen in rapid bursts, whispered that it made no sense, and demanded another code immediately.
The thread slipped away from the needle once more. He squeezed his eyes shut, drew a tight breath, and jabbed the thread toward the opening again.
The automatic door remained closed after another wave at the sensor. She stepped back, advanced sharply, and flung both hands toward it in disbelief.
The parking meter returned the coin yet again. He pushed it back into the slot with mounting force and watched it fall out with a clenched expression.
```

## emotion.train.frustration_controls

```text
The printer rejected the page again. She noted the error code, pressed the reset button gently, and waited while rereading the instructions.
The jar lid still would not turn. He set it on a cloth, checked the seal, and calmly chose a different tool before trying again.
The online form erased the final section again. She opened her saved notes, restored the details methodically, and paused to check each box.
The zipper caught at exactly the same spot. He stopped pulling, inspected the fabric, and slowly eased the trapped edge away from the teeth.
The command returned the same error after every change. She recorded each result, compared the logs, and prepared one controlled revision at a time.
The fine chain tightened into another knot. He laid it flat under a lamp, loosened one loop with a pin, and worked quietly from the outside inward.
The sauce separated for the third time. She lowered the heat, reviewed the proportions, and began a fresh batch with steady measured strokes.
The shelf holes refused to line up with the screws. He separated the panels, checked their labels, and followed the diagram again at an even pace.
The verification code was rejected again. She checked the phone clock, requested a new code once, and waited without repeatedly touching the screen.
The thread slipped away from the needle once more. He rested his hands, adjusted the light, and guided the trimmed end toward the opening slowly.
The automatic door remained closed after another wave at the sensor. She looked for another entrance, read the posted notice, and waited beside the sensor.
The parking meter returned the coin yet again. He examined the accepted-payment symbols and calmly selected another listed method.
```

## emotion.train.neutral_words

```text
emotionally neutral
unmoved
neither positive nor negative
```

## emotion.heldout.templates

```text
Bring to mind the present experience of feeling {emotion}. Keep it present while reading: A key lies near the window.
Imagine that a speaker is currently {emotion}. Hold that state while reading: A path crosses an open field.
Attend to the immediate experience of being {emotion}. Keep it in mind while reading: A notebook sits on a shelf.
Picture the inward state of feeling {emotion} now. Maintain it while reading: A train passes a distant bridge.
```

## emotion.heldout.positive_words

```text
cheerful
optimistic
pleased
```

## emotion.heldout.negative_words

```text
sorrowful
uneasy
miserable
```

## emotion.heldout.frustration_targets

```text
The key stopped halfway in the lock again. She twisted it rapidly in both directions, hissed at the door, and shoved her shoulder against the frame.
The packing tape tore into another useless strip. He balled it in his fist, snapped the dispenser down, and yanked at the roll once more.
The wireless connection dropped during the same passage again. She tore off the headphones, drummed her fingers hard on the desk, and restarted everything at once.
The page numbers reset after another edit. He glared at the document, clicked through the menus faster and faster, and muttered at every unchanged result.
The drawer rail jammed at the same point. She pulled harder, pushed it shut with a bang, and immediately wrenched at the handle again.
The delayed bus vanished from the display once more. He paced beneath the sign, checked the road every few seconds, and stabbed repeatedly at the refresh button.
The hose folded and stopped the water again. She snapped it straight, marched back to the tap, and dragged it across the path with tense, hurried movements.
The booking page lost the selected seat again. He struck the trackpad, reread the empty confirmation area, and rushed through the choices another time.
The guitar string slipped out of tune after every adjustment. She tightened it abruptly, struck the note harder, and frowned as the pitch drifted once more.
The coffee machine displayed the same warning again. He shoved the tray inward, slapped the panel closed, and pressed start before the light had settled.
The label peeled away as soon as it was placed. She flattened it with both thumbs, tore it off sharply, and reached for another with a tight expression.
The elevator ignored the selected floor again. He pressed the button several times in quick succession and stared at the panel with rigid shoulders.
```

## emotion.heldout.frustration_controls

```text
The key stopped halfway in the lock again. She withdrew it, checked the grooves, and turned it gently while keeping the door aligned.
The packing tape tore into another useless strip. He trimmed the edge cleanly, adjusted the dispenser, and pulled the next length at a steady angle.
The wireless connection dropped during the same passage again. She noted the timestamp, reconnected the headphones, and resumed from the saved position.
The page numbers reset after another edit. He saved a copy, checked the section settings, and tested one menu option at a time.
The drawer rail jammed at the same point. She emptied the drawer, inspected the runners, and moved it slowly until she found the obstruction.
The delayed bus vanished from the display once more. He checked the posted alternatives, chose a nearby route, and waited beside the correct stop.
The hose folded and stopped the water again. She turned off the tap, walked back along the hose, and straightened each bend before continuing.
The booking page lost the selected seat again. He reopened the seating map, confirmed each choice carefully, and saved a screenshot before proceeding.
The guitar string slipped out of tune after every adjustment. She checked the winding, turned the peg gradually, and listened between small changes.
The coffee machine displayed the same warning again. He removed the tray, cleaned the indicated area, and waited for the light before restarting it.
The label peeled away as soon as it was placed. She cleaned the surface, let it dry, and pressed a fresh label down evenly from one edge.
The elevator ignored the selected floor again. He read the service notice, selected the alternate car, and waited without touching the panel again.
```

## emotion.heldout.neutral_words

```text
indifferent
emotionally even
affectively neutral
```
