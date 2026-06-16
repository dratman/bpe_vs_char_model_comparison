#!/usr/bin/env python3
"""
make_reversed_corpus.py — build the REVERSED corpus for coupler-queue item 0002
(reversed-text learnability pilot).

Methodological point (Ralph endorsed): we must NOT reverse the whole file and let
--val_split re-cut it, because the reversed model would then be held out on a
different *region* of the source than the forward model, confounding direction
with content. Instead we reverse the characters WITHIN each split, independently,
so the reversed model is held out on exactly the same source text read backwards.

train.py's continuous-mode split (verified by reading the code) is positional and,
for the char tokenizer, operates 1:1 on characters:
    text   = open(input).read()           # no transformation
    tokens = CharTokenizer.encode(text)    # one token per character, in order
    n      = len(tokens)                    # == len(text) for char tokenizer
    n_val  = int(n * val_split)
    n_train= n - n_val
    train  = tokens[:n_train]               # first 90% of characters
    val    = tokens[n_train:]               # last  10% of characters

Therefore, writing reversed_text = reverse(train_part) ++ reverse(val_part) and
re-running train.py with the SAME --val_split reproduces exactly reverse(train)
and reverse(val) as the reversed model's train/val sets.

This script writes the reversed corpus and then VERIFIES the reconstructed
boundary before declaring success.
"""
import sys

VAL_SPLIT = 0.1
SRC = "txt_local/corpus_high_quality_uppercase_2026_05_08.txt"
DST = "txt_local/corpus_high_quality_uppercase_2026_05_08_REVERSED_within_splits.txt"


def main():
    print(f"Reading forward corpus: {SRC}")
    with open(SRC, "r", encoding="utf-8") as f:
        text = f.read()
    n = len(text)
    n_val = int(n * VAL_SPLIT)
    n_train = n - n_val
    print(f"  characters: {n:,}")
    print(f"  n_train   : {n_train:,}  (first {100*(1-VAL_SPLIT):.0f}%)")
    print(f"  n_val     : {n_val:,}  (last {100*VAL_SPLIT:.0f}%)")

    train_part = text[:n_train]
    val_part = text[n_train:]
    assert len(train_part) == n_train and len(val_part) == n_val

    rev_train = train_part[::-1]
    rev_val = val_part[::-1]
    reversed_text = rev_train + rev_val
    assert len(reversed_text) == n, "reversed length must equal source length"

    print(f"Writing reversed corpus: {DST}")
    with open(DST, "w", encoding="utf-8") as f:
        f.write(reversed_text)

    # --- VERIFY: re-read the written file and re-apply train.py's split ---
    print("Verifying reconstructed split boundary...")
    with open(DST, "r", encoding="utf-8") as f:
        rt = f.read()
    rn = len(rt)
    rn_val = int(rn * VAL_SPLIT)
    rn_train = rn - rn_val
    re_train = rt[:rn_train]
    re_val = rt[rn_train:]

    assert rn == n, f"length mismatch: {rn} != {n}"
    assert rn_train == n_train and rn_val == n_val, "split boundary moved!"
    # The reversed model's train set must be exactly the forward train read backwards
    assert re_train == train_part[::-1], "reversed train != reverse(forward train)"
    assert re_val == val_part[::-1], "reversed val != reverse(forward val)"
    # And character sets must match so the char vocab/meta is identical
    assert set(rt) == set(text), "character set changed under reversal"

    print("VERIFIED:")
    print(f"  reversed train == reverse(forward train)  [{n_train:,} chars]")
    print(f"  reversed val   == reverse(forward val)    [{n_val:,} chars]")
    print(f"  identical character set -> identical char vocab/meta -> comparable val CE")
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
