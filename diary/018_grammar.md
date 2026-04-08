# Diary Entry 018: Does the Model Know Grammar?
## Testing whether predictions change by syntactic context

### The distributions are genuinely different by category

Averaging first-letter predictions across multiple examples of each
grammatical context:

**After a pronoun** ("he ", "she ", "they ", "i "):
Top letters: h=0.12, w=0.12, s=0.10, c=0.09, a=0.08
These are mostly VERB-starting letters: had, was, went, could, said, are

**After a verb** ("he saw ", "she took ", "they found ", "he made "):
Top letters: t=0.23, a=0.12, h=0.11, o=0.09, i=0.07
These are OBJECT-starting letters: the, a, his, one, it — articles,
pronouns, and nouns that serve as objects

**After an article** ("the ", "a "):
Top letters: s=0.11, c=0.10, p=0.07, m=0.07, f=0.06
These are NOUN-starting letters: same, country, place, man, first

**After prep+article** ("in the ", "of the ", "from the "):
Looks almost identical to after_article. The preposition doesn't
change what noun follows "the".

### The critical comparison: after pronoun vs after verb

The biggest differences between "he " (pronoun context) and "he saw "
(verb context):

| Letter | After pronoun | After verb | Likely reason |
|--------|-------------|-----------|--------------|
| t | 0.05 | 0.23 | Verbs produce "the" as object |
| w | 0.12 | 0.03 | Pronouns produce "was/went/would" |
| c | 0.09 | 0.02 | Pronouns produce "could/came" |
| o | 0.02 | 0.09 | Verbs produce "of/on/one" |
| i | 0.02 | 0.07 | Verbs produce "it/in" |
| s | 0.10 | 0.05 | Pronouns produce "said/saw" |

After a pronoun, the model expects verbs (w for was/went, c for could,
s for said). After a verb, it expects objects (t for the, a for a,
o for of/one, i for it). This is genuine SYNTACTIC knowledge.

### "she " is different from "he "

After "she ": w=0.17, h=0.14 (was, had — past tense verbs)
After "he ":  s=0.10, c=0.09 (said, could — more varied)

The model predicts "she was" and "she had" as the most likely
continuations for "she", while "he" is more distributed. This
reflects the corpus statistics: in Gutenberg fiction, female
characters are more often described by their state ("she was") while
male characters take more varied actions.

### "he saw " strongly predicts "the"

After "he saw ", the probability of 't' is 0.23, and specifically
'th' is 0.21. The model knows that "saw" as a transitive verb is
almost always followed by an article+noun phrase ("saw the man",
"saw the light"). Compare with "he " where 't' is only 0.06.

This is a 4x increase in 't' probability, driven entirely by the
preceding verb. The model has learned verb subcategorization — that
transitive verbs like "saw" take direct objects.

### But the noun predictions after articles are generic

After "the " and "a ", the distributions are nearly identical:
s=0.11, c=0.09-0.10, p=0.07. The model doesn't distinguish which
nouns are more likely after "the" vs "a". And after "from the " the
distribution barely changes. The model knows THAT a noun is coming,
but not WHICH noun — that would require knowing the broader sentence
meaning, which is beyond this model's capacity.

### What this means for grammatical knowledge

The model has learned:
1. **Word class prediction**: After pronouns, predict verbs. After
   verbs, predict articles/objects. After articles, predict nouns.
2. **Verb subcategorization**: Transitive verbs strongly predict "the"
   as the next word (the start of a noun phrase object).
3. **Gender-linked distributions**: "she" and "he" produce different
   verb distributions, reflecting corpus statistics.

The model has NOT learned:
1. **Specific noun selection**: After "the", it can't predict which
   noun based on sentence meaning.
2. **Long-range agreement**: It can't enforce subject-verb agreement
   or maintain coreference across sentences.
3. **Semantic constraints**: "he saw the" predicts nouns generically,
   not specifically ("the man" vs "the wall" vs "the truth").

### This is proto-grammar

The model has internalized the statistical regularities of English
syntax at the character level. It knows that certain character
sequences (pronoun patterns) predict other sequences (verb-starting
patterns). But it implements this through character-level predictions
(first letter after space), not through explicit grammatical rules.

This is arguably how grammar SHOULD look in a character-level model:
not as explicit rules, but as statistical dependencies between word
patterns. The model doesn't know "pronouns precede verbs" as a rule.
It knows "after the characters h-e-space, the characters w and h are
more likely" — which is the character-level manifestation of that
grammatical pattern.
