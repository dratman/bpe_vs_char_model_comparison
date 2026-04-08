# Diary Entry 010: Layer 5 MLP — Not Just a Space Detector
## n391 is a space detector, but it also bleeds into nearby characters

### n391 fires for spaces — but not only spaces

At space positions: mean=4.22, range 2.35-6.79
At letter positions: mean=0.19, range -0.17 to 2.30

So n391 is clearly a space detector — spaces average 4.22, letters
average 0.19. That's a 20x ratio. But it's not binary. It also fires
at the FIRST CHARACTER after a space, with diminishing strength:

In "he said to him":
  ' '→7.2  's'→3.3  'a'→1.5  'i'→-0.1  'd'→-0.2
  ' '→4.5  't'→2.3  'o'→-0.1
  ' '→2.4  'h'→-0.2

The pattern: n391 fires strongly at the space, then moderately at the
next character, then weakly at the character after that, then goes dark.
It's not just marking "there's a space here" — it's marking "we're at
or near a word boundary, and here's how far past it we are."

### n391 is context-sensitive

The activation at spaces varies from 2.35 ("to ") to 6.79 ("the ").
This means n391 fires MORE strongly for some word boundaries than others.

Ranking:
  after "the":  6.79
  after "that": 5.24
  after "was":  5.04
  after "not":  4.59
  after "man":  4.48
  after "her":  4.37
  after "said": 4.17
  after "it":   3.91
  after "old":  3.50
  after "easy": 3.03
  after "to":   2.35-3.20

"the " triggers the strongest response. This might be because "the" is
the most common word in English, and the space after it is the most
important word boundary — it's where the model must switch from predicting
within a function word to predicting the start of a content word.

"to" triggers the weakest response. Perhaps because "to" often begins
infinitives (to go, to see) where the prediction is relatively constrained
— a verb is very likely to follow.

### There's a whole TEAM of space neurons

Layer 5 has at least 10 neurons that fire preferentially at spaces:
  n391: diff=+4.03 (the dominant one)
  n201: diff=+2.58
  n126: diff=+2.41
  n275: diff=+2.19
  n427: diff=+2.15

Five strong space neurons, plus five weaker ones. Together they form a
"space committee" in the final layer. Why so many? Because the prediction
after a space depends on context: after "the " you need noun-start
predictions, after "was " you need verb/adjective predictions. Different
space neurons might encode different TYPES of word boundaries.

### Comparison with Layer 0's space detector

Layer 0 had n357 as its space detector (firing at 2.58 for space, 0.01
for letters). Layer 5's n391 fires much more strongly (up to 7.2) and
is context-sensitive. Layer 0's n357 was a pure character detector.
Layer 5's n391 is a word-boundary analyzer that knows what kind of word
boundary this is.

The model has TWO space detection systems:
1. Layer 0: "Is this character a space?" (yes/no, simple)
2. Layer 5: "What kind of word boundary is this?" (graduated, context-aware)

### The residual stream at spaces

In Entry 005, I noted that the space at position 2 in "he said" had
residual norm 23.1 after Layer 5 — almost 6x larger than other positions.
Now I can see why: five space neurons each contribute activations of
2-7, which get projected back to 128 dimensions and added to the
residual stream. The cumulative effect of this space committee is a
massive amplification of the space position's representation.

This amplification directly affects the prediction: when the final
layer norm and logit projection see a vector with norm 23, the logits
will be larger and the softmax will be more peaked. The model is
CONFIDENT about its prediction after a space — but confident about
what? About the distribution over possible next characters, which has
been shaped by all the preceding context flowing through attention.

### What I don't yet understand

I've traced the space detection mechanism, the quotation mark circuit,
the character detectors, and the internal conflict between boosters
and suppressors. But I haven't looked at what happens BETWEEN the MLP
output and the final logits — the final layer norm and the lm_head
projection. That's where all these signals get converted into actual
character predictions. That's next.
