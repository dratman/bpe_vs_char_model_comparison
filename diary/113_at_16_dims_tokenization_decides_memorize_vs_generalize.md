# Diary 113 — At 16 dimensions, tokenization decides: BPE memorizes The Raven, char can't — and the memorize↔generalize split falls out

Date: 2026-07-08

Figure: `doc/figures/113_d16_char_vs_bpe_capacity.png`

## The question (Ralph's reductionist line)

Working toward the popular account — "how and why are language models built the
way they are" — we shrank the machine and watched where literal completion
fails. Cut the embedding width from 128 to **16** (which, via the fixed 4×
hidden, also shrinks the MLP "switch bank" from 512 to 64), train a 1-layer
model on The Raven (6,226 chars), char vs BPE, and measure success at literal
completion.

## Rigor: trained to a definitive plateau

Earlier snapshots were undertrained and lied (a 3,000-step d16 char model looked
like a total collapse). So both 16-wide models were trained to **60,000 steps
with the loss logged every 500**, and the tails are dead flat (last five char
readings 1.098/1.105/1.099/1.098/1.104; BPE 0.081/0.084/0.086/0.081/0.084). The
deceleration set in while the learning rate was still healthy, so these are
genuine **capacity floors**, not the rate winding down.

## Result: same 16 dims, opposite outcomes — and only the tokenizer differs

| 16-wide model | params | train floor | knows it? | val (unseen) |
|---|---|---|---|---|
| char | 8,320 | **1.59 bits/char** | NO — hard wall | 3.9 bits/char |
| BPE | 12,864 | **0.04 bits/char** | YES — memorized | 8.3 bits/char |

(Random guessing among The Raven's 57 characters ≈ 5.83 bits/char. At 128-wide,
both memorize completely — diary 109 — so the tie there was about *cost*; here,
at the floor, it is about *capability*.)

- **char cannot memorize The Raven at 16 dims.** It bottoms out at 1.59 bits/char
  — far better than random (it learned plenty) but far from memorized (~0). A
  real wall.
- **BPE memorizes it completely** (~0.04 bits/char) — with ~13,000 parameters.

**Why:** BPE compresses The Raven from 6,226 characters into ~2,130 chunks
(~2.9 chars/token). A 16-dim model has room to hold ~2,000 chunks but not ~6,000
characters. Compression is the difference between "can't do the task at all" and
"does it perfectly." (It also explains the earlier depth/width intuition: the
char model must chain ~3× as many fine decisions, and a starved machine cannot.)

## The payoff: memorize↔generalize is decided by whether the text fits

The two panels of the figure make it undeniable. The *same* 16-dim size produces
two opposite kinds of machine, purely from tokenization:

- **BPE** *can* fit the poem → becomes a **pure memorizer** → its loss on unseen
  text **explodes above random** (8.3 bits/char — worse than a coin flip;
  confident confabulation).
- **char** *cannot* fit the poem → is forced to spend its 16 dimensions on the
  text's **statistics** (landing at ~1.6 bits/char, roughly a real char LM) →
  and it **generalizes** (unseen-text loss stays *below* random, ~3.9).

So, from first principles at the smallest scale: **whether a fixed tiny model
memorizes or generalizes is decided by whether tokenization lets the text fit
under its capacity.** Give it room → it memorizes and overfits. Starve it → it
is forced to generalize. And note the corollary (diary 112 was circling it):
**overfitting requires capacity** — the char model never overfits because it
*can't*; it has nothing to overfit with.

## Honest corrections folded in

- The 3,000-step d16 numbers (char "31% knows-next", loss 2.28) were
  **undertrained** and overstated char's failure; the true floor is 1.59
  bits/char, reached only after training to plateau.
- This also cleanly settles, for one case, the capacity-vs-training-time
  ambiguity that dogged the Alice work: at 16 dims the char floor is a genuine
  capacity wall (flat while LR still healthy), not undertraining.

## Setup (reproducibility)

1 layer, 4 heads, n_embd 16 (head_dim 4), MLP hidden 64; char block 256 / BPE
block 85 (vocab 512); batch 32; lr 1e-3 cosine; val_split 0.08; seed 1337;
60,000 steps; corpus `txt_local/Poe_The_Raven.txt`. Checkpoints
`pt/raven_{char,bpe}_L1_d16_plateau*`. Plot script produced
`doc/figures/113_d16_char_vs_bpe_capacity.png`.

## Status / for the account

This is the cleanest, most self-contained result of the whole line of work, and
the first that would make a *reader* sit up: one tiny knob (tokenization),
watched all the way to convergence, flips a machine between memorizing (and
overfitting) and generalizing. A strong candidate "scene" for the popular
account.

Related: 107–112 (the arc), 109 (BPE-vs-char tie at full width), 112
(consolidated results).
