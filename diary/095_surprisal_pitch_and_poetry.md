# Diary 095 — Surprisal and pitch: speech, music, and the reading of poems

Date: 2026-06-02

## The starting idea

Ralph recorded a voice memo (transcribed by the iPhone Voice Memos app
on iOS 18; transcription had several errors — "Texas Beach products"
→ text-to-speech, "tomes" → tones, "caves" → waves) about two
intertwined ideas:

1. **Visualize per-word uncertainty in prose.** Take a passage, run
   it through a language model, and color each word by its surprisal
   so the high-entropy moments are visually conspicuous. He used the
   word *perplexity*, which is the correct overall measure (exp of
   mean surprisal across a sequence) but for per-word visualization
   the right quantity is **surprisal = −log p(token | context)** —
   the information-theoretic surprise of seeing this particular
   token next.

2. **Connect that uncertainty to changes in speech pitch and to
   moments of change in music.** His intuition: high-surprisal text
   tokens may correspond to moments where pitch shifts, where the
   speaker changes register, or where a melody changes key. He asked
   whether text-to-speech (TTS) systems predict pitch in a way that
   would make this observable, and whether the literature has made
   this connection.

The conversation that followed worked through both ideas and then
took a literary turn by way of Wallace Stevens's *Peter Quince at the
Clavier* and a discussion of how poems are actually read aloud. This
diary captures what I want to remember from it.

## How modern TTS handles pitch

State-of-the-art TTS splits along two main architectural paradigms,
both of which already encode something like the surprisal–pitch
correlation Ralph was reaching for. The distinction matters for how
you would probe it experimentally.

**Paradigm 1 — explicit pitch prediction.** Examples: FastSpeech 2,
StyleTTS 2, NaturalSpeech 3. A pitch estimator (pYIN, CREPE, DIO)
extracts an F0 contour from every training audio file; F0 = the
fundamental frequency, i.e., pitch in Hz, with voiced/unvoiced flags
on consonants and silence. The model has a dedicated "variance
adaptor" that predicts F0 per phoneme or per frame from the encoded
text, trained against the ground-truth contour with L1/L2 loss. The
predicted F0 is then injected back into the encoder hidden state
(often via a learned embedding of quantized pitch values), and the
spectrogram decoder uses that enriched representation. Pitch is a
named, controllable, editable output.

**Paradigm 2 — token-emergent pitch.** Examples: Bark, AudioLM,
VALL-E, Suno, ElevenLabs (closed but consistent with this paradigm).
A neural codec (EnCodec, SoundStream) tokenizes audio into discrete
tokens at ~50–75 tokens/second, and a GPT-style transformer is
trained to predict those audio tokens autoregressively, conditioned
on text. Pitch isn't separately modeled; it lives in whichever
tokens the model emits, because the codec's decoder must reconstruct
it. Usually there is a hierarchy — "semantic" tokens that carry
content, "acoustic" tokens that carry timbre and prosody.

**The 2024–2026 frontier blurs the two.** NaturalSpeech 3 uses a
*factorized* codec — separate codebooks for content, prosody, and
timbre — so pitch is partly explicit again but at codebook level
rather than as a continuous contour. Flow-matching and diffusion
systems (F5-TTS, Voicebox) generate continuous mel-spectrograms and
let pitch emerge from the diffusion trajectory. GPT-4o-audio puts
text and audio tokens into one autoregressive multimodal transformer
so that pitch prediction is just another mode of next-token
prediction.

For Ralph's research interest, paradigm 1 is the more tractable one
because pitch is a named output of a named submodule that can be
probed directly.

## The literature has been on this for twenty years

The hypothesis that high-surprisal content carries prosodic
prominence (longer duration, wider pitch range, greater intensity)
is well-established, but most of it lives in psycholinguistics and
speech-prosody research rather than in the language-model literature
proper.

**Foundations.**

- **Aylett & Turk (2004, 2006) — Smooth Signal Redundancy
  Hypothesis.** Speakers redundantly amplify low-predictability
  content with prosodic prominence so the listener receives
  information at a smooth rate. High-surprisal words get bigger
  pitch excursions and longer durations; low-surprisal words get
  reduced. This is precisely the pattern Ralph described.

- **Jaeger (2010) — Uniform Information Density (UID).** Speakers
  make choices at every level (word order, *that*-mentioning,
  contraction, prosodic prominence) to keep per-word surprisal close
  to uniform across the utterance.

- **Bell, Brenier, Gregory, Girand, Jurafsky (2009).** Heavily
  cited: word duration in conversational speech is well-predicted
  by bigram and trigram probabilities. Predictable words are
  physically shorter.

- **Calhoun (2010); Wagner & Watson (2010); Watson, Arnold,
  Tanenhaus (2008).** Pitch accents fall preferentially on
  informationally focused (typically low-predictability) words.

- **Pate & Goldwater (2015).** n-gram surprisal predicts pitch
  accent placement.

**Neural-LM surprisal applied to prosody.** Once contextual neural
models existed, several groups switched from n-gram surprisal to
surprisal computed from BERT, GPT-2, etc. Representative threads:

- **Talman, Suni, Aalto, Vainio (2019, Helsinki).** "Predicting
  prosodic prominence from text with pre-trained contextualized
  word representations" — beat older predictability baselines.
- **Stehwien, Vu** and collaborators — neural pitch-accent
  detection.
- **Lai, Wang, Sun** and others (2020–2023) — neural-LM surprisal
  as a feature for pitch and duration prediction in TTS front-ends.

The general finding: neural-LM surprisal correlates with prosodic
prominence measures, often more strongly than older predictability
features, and adding surprisal as a conditioning signal improves
prosody prediction.

**The music parallel is also formalized.** Pearce and Wiggins'
**IDyOM** (Information Dynamics of Music, 2006 onward) computes
melodic surprisal from a variable-order Markov model trained on
melodies. Listener expectation, surprise ratings, and emotional
responses correlate with IDyOM surprisal — the music analogue of
word surprisal driving prosodic prominence. Huron's *Sweet
Anticipation* (2006) is the book-length cognitive-science synthesis.
Ralph's intuition that "a high-surprisal text token is analogous to
a key change in music" is precisely the parallel these researchers
made formal.

## Joint text-and-audio language models

Ralph asked whether anyone has trained on voice and corresponding
text *simultaneously*, letting pitch context aid in next-text-token
prediction during training. Yes, extensively. The field calls these
**speech language models (SLMs)** or **audio language models**, and
the design Ralph described is the recipe most systems converged on:

1. Tokenize speech with a learned codec or self-supervised model
   (HuBERT, wav2vec 2.0 for semantic tokens; EnCodec/SoundStream
   for acoustic tokens).
2. Concatenate or interleave speech tokens with text tokens into
   one stream.
3. Train a standard next-token-prediction transformer on the joint
   stream.

When the model predicts the next text token, recent speech tokens
are part of its context, and pitch lives inside those speech tokens.

**Examples worth knowing:**

- **AudioPaLM (Rubenstein et al., Google, 2023).** PaLM-2 extended
  to cover Universal Speech Model tokens. One transformer, next-
  token prediction across modalities.

- **Spirit-LM (Nguyen et al., Meta, 2024).** The closest direct
  match to Ralph's design. Explicitly *interleaves* speech and text
  tokens. The **Expressive** variant adds discrete pitch tokens
  (quantized F0) and style tokens to the speech representation —
  pitch is in the vocabulary. This is the paper to read for the
  specific research question we were discussing.

- **Moshi (Kyutai, 2024).** Full-duplex spoken-dialogue model,
  interleaved text+speech tokens, open-source.

- **VioLA (Wang et al., Microsoft, 2023); Qwen-Audio, Qwen2-Audio
  (Alibaba, 2023–2024); GPT-4o (OpenAI, 2024).**

**Measured effects:** joint training improves both directions.
Pitch information helps text-side tasks especially around
punctuation (rising terminal → question mark), sentence boundaries,
sentiment, and focus structure. Spirit-LM and Moshi ablations show
measurable drops when pitch tokens are stripped. Conversely, text
context sharpens speech-side prediction by pinning content.

**Caveat:** these systems require large (text, audio) aligned
corpora — LibriSpeech (1 kh), MLS (50 kh), GigaSpeech (10 kh),
Libri-Light (60 kh), and bigger internal corpora for commercial
models. The text-only LM literature suggests prosody-aware joint
training starts paying off only once both data and model scale past
text-only saturation. Small models (~100M params, ~1 kh audio) tend
to learn the modality alignment but not gain much over text-only
baselines.

## Wallace Stevens, *Peter Quince at the Clavier*

Ralph offered the poem in the middle of the conversation. It maps
onto the surprisal-pitch discussion almost too neatly.

**The opening claim — "Music is feeling, then, not sound" — is
Stevens collapsing the distinction between the physical pitch
contour and the listener's internal response.** In information-
theoretic terms: the salient quantity is the surprisal in the
listener, not the F0 in the air. Pitch is the carrier; music is the
entropy-shaped feeling it produces.

**Section II is the most prosodically alive part, and it enacts
what surprisal-driven prominence looks like in verse.** Short lines,
irregular rhythm, hesitations that rise and land. Then the strongest
moment in the poem:

> She turned—
> A cymbal crashed,
> And roaring horns.

The dash is a duration cue (pause); the cymbal and horns are pitch
and timbre dramatically reset. A high-surprisal token rendered as
drama. Stevens does the same thing one level up — a free-verse
section in delicate registers suddenly punctured by percussion.

**The four sections distribute information across forms the way a
speaker distributes surprisal across an utterance.** Section I is
iambic-leaning tercets, measured. Section II is free verse with
rapid dynamic shifts. Section III is rhymed couplets, almost
incantatory. Section IV returns to discursive grandeur. Smoothing,
then spiking.

**The closing argument and IDyOM.** "Now, in its immortality, it
plays / On the clear viol of her memory, / And makes a constant
sacrament of praise." Stevens claims beauty achieves immortality
through repeated internal performance of memory — which is what an
autoregressive predictive model with a remembered context does. The
viol plays in memory; memory keeps the contour intact. Pearce &
Wiggins would say: the listener's recall of a melody is itself a
generative act with its own surprisal profile.

**The position Stevens stakes out that the technical literature does
not.** Speech-language-model research treats pitch and text as two
coupled modalities that a joint predictor can model together.
Stevens treats them as one thing — *feeling* — that happens to
manifest as sound or as language depending on the carrier. UID and
its descendants keep the carriers analytically separate. Whether
Stevens or the modelers are closer to right is a question small
models in this lab are positioned to chip away at, because a single
representation can be used for whatever output the architecture
asks of it.

## How TTS actually does on plausible-pitch-from-text

For neutral English prose, the top systems (ElevenLabs,
OpenAI gpt-4o-audio voices, Google Studio/Journey, Azure Neural)
produce plausible prosody — sentence-final declination, yes-no
question rises, default nuclear accent on the last content word of
a phrase, pitch reset at major punctuation, pause and downstep at
clause boundaries. On a paragraph of straightforward prose, you
would have to listen carefully to know the audio was synthetic.

**Where they still fail, in ways relevant to the surprisal-pitch
question:**

- *Contrastive focus.* "I didn't say *he* stole the money" has
  six or seven different meanings by stress placement. From text
  alone, ambiguous. TTS picks the default focus.
- *Wh- and embedded questions.* Often given a default falling
  contour even when the speaker meant a rise.
- *Sarcasm, irony, marked registers.* The system reads the
  literal text; the intended curve isn't in the text.
- *Long-range discourse prosody.* Knowing a sentence is the
  *answer* to a previous question exceeds most TTS text
  encoders' context.
- *Poetry.* Line breaks confuse most systems; metrical patterns
  go unrecognized; Stevens-style cymbal-crash effects get
  flattened toward default reading.
- *Heteronyms and proper nouns.* "Bow," "lead," "tear," and any
  unfamiliar name — sometimes mis-stressed.

Strong open-source options (StyleTTS 2, F5-TTS, Kokoro, Bark/Suno)
are a step below on consistency and long-form coherence but still
produce good prosody on neutral prose.

## The narrow pitch range of poetry reading

Ralph's observation: when poetry is read aloud, the pitch is often
kept within a narrow range. This complicates the surprisal-pitch
story. Some patterns:

**Poets reading their own work tend toward the flat end.** T. S.
Eliot's Caedmon recordings of *Four Quartets* and *The Waste Land*
are nearly monotone — a deliberate refusal of dramatic effect.
Wallace Stevens read in a slow, careful, restrained voice. Auden
was similar. Robert Frost was more inflected but still restrained.
Yeats is the famous exception in the other direction — he read in
a half-chanting cadence that he believed restored the bardic
register; most listeners now find it eccentric. Sylvia Plath's
"Daddy" reading is another exception: dramatic, varied, almost
incantatory.

**The flat-reading aesthetic has a defensible argument.** If the
poem has done its prosodic work in the words and the meter, dramatic
vocal interpretation actively gets in the way. The reader's job is
to deliver the poem cleanly so the listener can hear the patterns
the poet built in. This tracks Eliot's argument in *The Three Voices
of Poetry* and a lot of subsequent criticism.

**Actors negotiate the same question differently.** Some restrain
themselves toward the poet-style flat delivery (Alec Guinness
reading Eliot, John Gielgud reading Shakespeare's sonnets). Others,
like Richard Burton, bring the full theatrical apparatus. Burton's
Hardy recordings — and his *Under Milk Wood* with Dylan Thomas —
use pitch range much closer to dramatic speech than to recited
verse. That works especially well for Hardy because Hardy's poems
sit closer to the Victorian recitation tradition than to high
modernism; "The Darkling Thrush" or "Channel Firing" can take
theatrical handling without sounding overdone.

**For the surprisal-pitch experiment, this gives several reading-
styles to test against the same text.** Computing surprisal on a
Hardy poem and measuring F0 range in (a) the poet's own reading
where one exists, (b) Burton's theatrical reading, (c) a current
actor's restrained reading, (d) a SOTA TTS reading would yield four
very different F0 traces over the same text. The dramatic actor's
reading is the one expected to track surprisal most closely; the
poet's the least. The TTS reading is the wild card because it
applies neutral-prose prosody to a text written for compressed-range
delivery, and that mismatch is probably audible in a specific way.

**The aesthetic question hiding inside all this.** Is the correct
reading of a poem the one that marks the surprisal moments
prosodically, or the one that leaves them in the text for the
listener to find? Both are defensible. Burton would have answered
one way; Stevens, reading his own work, the other. A surprisal-pitch
correlation study could measure the gap between the two answers but
cannot adjudicate between them.

## What I would actually do next

In order from least to most ambitious:

1. **Per-word surprisal visualizer on the existing char and BPE
   models.** Cheapest. Take prose, feed it through each model,
   capture log-probabilities, render with a color ramp keyed to
   per-token surprisal. Lets the high-entropy moments be seen
   directly. Char-vs-BPE disagreements on the same text are
   themselves diary material — they atomize the language
   differently (diary 093).

2. **TTS pitch correlation on neutral prose.** Take 50–100 short
   neutral passages, synthesize with ElevenLabs and OpenAI TTS-HD,
   extract F0 with pYIN or CREPE, peak-detect pitch accents,
   correlate against per-word surprisal from the char and BPE
   models. Hypothesis: positive correlation on neutral prose;
   stronger for whichever model assigns surprisal in the units the
   speaker prosodically marks (probably whole-word boundaries,
   favoring BPE).

3. **TTS vs. human readings of poetry.** Same pipeline applied to
   poems with multiple known readings. Hardy with Burton's
   theatrical delivery, plus a TTS rendering, plus surprisal from
   the char and BPE models. Document the gap between TTS prosody
   (which assumes neutral prose) and Burton's prosody (which
   marks the rhetorical structure).

4. **The Stevens / modernist case.** Same again on Stevens reading
   his own work versus a TTS rendering versus the surprisal
   contour. Hypothesis: Stevens's reading flattens *everything*,
   the TTS picks a default contour, and only the text-surprisal
   contour has the spikes — which is what Stevens may have
   intended us to find for ourselves.

5. **Spirit-LM-Expressive-style joint training at the lab's
   small-model scale.** Requires LibriSpeech-100h plus modifying
   `py/train.py` to accept interleaved text and speech tokens.
   Unlikely to beat text-only on text tasks at this scale
   (per the literature) but interpretable: which attention heads
   in a small joint model attend to pitch context when predicting
   content-bearing text tokens? And do they concentrate on the
   high-surprisal text positions?

(1) is the obvious starting point. (5) is the most aligned with
this lab's research program on transformer internals.

## References worth pulling later

- Aylett & Turk (2004, 2006) — Smooth Signal Redundancy.
- Jaeger (2010) — Uniform Information Density.
- Bell, Brenier, Gregory, Girand, Jurafsky (2009) — word duration
  and predictability.
- Calhoun (2010); Wagner & Watson (2010); Watson, Arnold, Tanenhaus
  (2008) — pitch accents and information structure.
- Pate & Goldwater (2015) — n-gram surprisal predicting pitch
  accents.
- Talman, Suni, Aalto, Vainio (2019) — pre-trained contextualized
  representations for prosodic prominence.
- Pearce & Wiggins, IDyOM (2006 onward); Huron, *Sweet Anticipation*
  (2006) — surprisal in music.
- Eliot, *The Three Voices of Poetry*.
- Rubenstein et al. (2023) — AudioPaLM.
- Nguyen et al. (2024) — Spirit-LM and Spirit-LM Expressive.
- Kyutai (2024) — Moshi.
- Aalto FastSpeech 2 line; StyleTTS 2; NaturalSpeech 3 — for
  paradigm-1 TTS architectures.
- EnCodec, SoundStream; AudioLM, VALL-E, Bark — for paradigm-2
  TTS architectures.
