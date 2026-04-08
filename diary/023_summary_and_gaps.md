# Diary Entry 023: Summary of Understanding and Remaining Gaps
## After 22 entries of being the model

### What I now understand clearly

**The complete forward pass pipeline:**
1. Character embedding (nearly orthogonal, rare chars have large norms)
2. Position embedding (position 0 is unique and large; positions 1-200
   are smooth and small; end positions grow slightly)
3. Layer 0: Character detection. MLP is 90% dark; each character has a
   signature neuron. Attention is diffuse (never specializes during
   training).
4. Layer 1: Global context distribution. All 4 heads read position 0
   and broadcast its content. MLP nearly dead.
5. Layer 2: Transition. Self+prev attention begins developing.
6. Layer 3: Short-range word processing. Reads 2 chars back. Makes
   tentative predictions that are context-dependent (not pure bigrams).
7. Layer 4: Medium-range word recognition. H2 reads word-start letter
   at 0.80-0.99. H1 reads prev char at 0.73-0.95. H0 reads mid-word.
   H3 reads the space. MLP interprets the gathered evidence and writes
   word-level predictions.
8. Layer 5: Long-range recognition + context. Reads 4-5 chars back for
   long words ("through"). Also reads long-range context (quotation
   marks). MLP has the strongest character-specific votes (+0.40).
9. Final: layer norm + dot product with (tied) embeddings → logits.

**The attention→MLP division:**
Attention GATHERS information from other positions.
MLP PROCESSES that information into predictions.
This holds universally across all layers and all test cases.

**The internal economy:**
- Boosters (L2H0, L2H1) push toward likely next characters
- Suppressors (L3H3, L1 heads) calibrate confidence downward
- The final prediction is the sum of competing votes

**Word recognition:**
- Works by reading previous characters of the current word
- Confidence ramps up as more characters are seen
- Distinctive prefixes ("sai", "throu") lead to high confidence
- Ambiguous prefixes ("di", "wi") remain uncertain
- Each layer extends the recognition window by ~1 character

**Context effects:**
- Quotation marks boost "said" prediction via Layer 5 attention
- Preceding words change next-word-start distribution (proto-grammar)
- "from " → t (0.42), "he " → flat distribution (0.06-0.10)

**Training dynamics:**
- Middle layers (3-4) specialize FIRST (step 400-1000)
- Layer 1 FST develops LATE (step 3800+)
- Layer 0 never specializes (attention stays diffuse)
- Local before global: word processing before context distribution

**Weight tying effects:**
- Rare characters have larger embedding norms (-0.77 correlation)
- Compensates for model's inability to strongly align toward rare chars
- Creates synergy: large norms help both input detection and output prediction

### What I'm NOT confident about

1. **Why Layer 0 attention never specializes.** It stays diffuse through
   all 30k training steps while every other layer develops structure.
   Is this because the MLP does all the work at Layer 0? Or because
   the attention heads are redundant at this layer?

2. **The exact role of Layer 2.** I characterized it as a "transition
   layer" but haven't traced its specific contribution to predictions.
   It has one head (H3) that stays diffuse while H0-H2 develop
   self+prev. What is H3 doing?

3. **How the model handles words it hasn't seen enough times.** The
   model succeeds on common words but fails on less common ones.
   Is there a threshold of corpus frequency below which a word can't
   be recognized? This would tell us something about the capacity
   limits of 512 MLP neurons per layer.

4. **The between-word similarity increase in larger models.** In Entry
   10.6 of findings.md, the dim=256 model showed higher between-word
   similarity than dim=128. I hypothesized syntactic clustering but
   wasn't confident. This remains unexplained.

5. **What the GELU nonlinearity actually does.** I know neurons fire
   or don't (sparse activation), but I haven't looked at neurons in
   their partially-active regime. Are some neurons operating in the
   gradual transition zone of GELU, or is it effectively binary?

### What I want to investigate next

Most promising directions, in order of expected insight:

1. **The GELU regime question** — quick to test, would tell me whether
   the MLP is a lookup table (binary activations) or a smooth
   computation (graded activations).

2. **Layer 2 H3** — the one head that doesn't specialize. Is it
   carrying information that the other heads don't?

3. **Word frequency vs recognition accuracy** — systematic test of
   which words the model can and can't recognize, correlated with
   corpus frequency. Would reveal the capacity boundary.
