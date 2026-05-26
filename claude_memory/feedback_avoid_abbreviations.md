---
name: Avoid uncommon or ambiguous abbreviations
description: Common unambiguous abbreviations are fine; uncommon or ambiguous ones get spelled out
type: feedback
originSessionId: 09c3c73f-03d3-45af-a2eb-906928c0fb3c
---
Avoid abbreviations whose meaning is not immediately obvious to a
reader. Common, unambiguous technical abbreviations are fine — the
test is clarity, not brevity.

  Spell out (uncommon or ambiguous):
    "arch"       → "architecture"  (could mean archive, archery, archaic)
    "hyperparams"→ "hyperparameters"
    "ckpt"       → "checkpoint"

  Fine to keep (common, unambiguous in software/ML context):
    "config"
    "param" / "params"
    "repo"
    "dir"
    "info"

Why: clarity over brevity. The reader should not have to pause to
disambiguate, but ordinary English shouldn't be twisted into longer
forms just to avoid every abbreviation. (Stated 2026-05-10, refined
the same day after I overcorrected on the first version.)

How to apply: when in doubt, spell it out. Inside code or shell
commands, abbreviations chosen by the language/tool stay as written.
The constraint applies to my prose, not to identifiers.
