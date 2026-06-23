#!/usr/bin/env python
"""
category_geometry_probe.py - Does a character-level model place same-category
words in a shared region of its residual stream?

THE QUESTION
------------
A "category" or "bucket" (animals: cat, dog, fox, ...) corresponds, in a
language model, to a shared direction in its internal vector space: members
point the same way. In a word/BPE model that direction can live in the static
embedding table -- "cat" has its own row. In a CHARACTER model there is no
"cat" row. The embedding table holds only single letters. So the category
"animal" cannot be looked up; it has to be ASSEMBLED in the residual stream,
across layers, after the network has recognized c-a-t as a unit. This probe
measures whether that assembled region exists, and at what depth it appears.

THE SURFACE-FORM CONTROL (the part that makes this mean something)
-----------------------------------------------------------------
Short animal words might cluster simply because they share letters, not because
they are animals. To separate spelling from meaning we use MINIMAL PAIRS: every
animal is matched with a non-animal object word it nearly rhymes with / shares
letters with --

    cat/hat  dog/log  fox/box  owl/bowl  hare/hair
    goat/coat  mouse/mouth  wolf/wool  horse/house

If "fox" sits with cat/dog/owl and AWAY from "box" despite the shared spelling,
that is semantics beating surface form -- the cleanest single thing this can
show. Word lengths are matched across the two groups so length is not the
hidden variable.

METHOD
------
- Read each word inside 4 carrier sentences whose slot sits at varied positions
  (so the positional component cancels when we average across frames).
- Capture the residual stream at every layer, at two readout positions:
    (a) the word's FINAL LETTER  (word just completed)
    (b) the position just AFTER it (the model's running summary of the word
        before it predicts onward).
- Average each word's vector across the 4 frames.
- Per layer: MEAN-CENTER the 18 word vectors (removes the dominant shared
  residual direction that otherwise makes every cosine ~0.9), L2-normalize,
  and compute cosine similarities.

METRICS (per layer, per readout)
- separation = mean within-category cosine  -  mean between-category cosine.
- permutation null: shuffle the animal/object labels many times, recompute
  separation, report where the real value lands (z-score + p) so the number
  has a scale.
- minimal-pair test: for each animal, is it closer to the centroid of the
  OTHER animals than to its surface-twin object? (e.g. "7 of 9"). The crispest
  statement of semantics-over-spelling.

Read-only inference on the best char checkpoint. Defaults to CPU so it does not
contend with a live MPS training run.

Usage:
    python py/category_geometry_probe.py \
        --model pt/char_uppercase_16L_1280.pt \
        --out_tsv terminal_logs/category_geometry_<date>.tsv
"""

import os
import sys
import argparse
import pickle

import numpy as np
import torch

# Imports resolve from py/ the same way sample.py / train.py do.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPTConfig, GPT
from tokenizer import load_tokenizer


# ----------------------------------------------------------------------------
# Word sets: animals vs surface-matched objects, paired index-for-index.
# Each animal[i] is a near-minimal-pair with object[i].
ANIMALS = ["cat", "dog", "fox", "owl", "hare", "goat", "mouse", "wolf", "horse"]
OBJECTS = ["hat", "log", "box", "bowl", "hair", "coat", "mouth", "wool", "house"]
#           cat    dog    fox    owl    hare    goat    mouse    wolf    horse

# Carrier sentences. "WORD" is the slot. Its position varies (early / late /
# mid / mid-late) so the positional signal cancels under frame-averaging.
# WORD is always followed by a space so the "after" readout is consistent.
FRAMES = [
    "\nThe WORD was very old.\n",
    "\nNear the river they found a WORD lying there.\n",
    "\nI saw the WORD again today.\n",
    "\nThey had kept that WORD for many years.\n",
]


def derive_meta_path(model_path):
    """Same convention as sample.py: strip _iter{N} / _final / _rolling."""
    base = model_path.replace(".pt", "")
    if "_iter" in base:
        base = base.rsplit("_iter", 1)[0]
    elif base.endswith("_final"):
        base = base[:-6]
    elif base.endswith("_rolling"):
        base = base[:-8]
    return base + "_meta.pkl"


def load_model_and_tokenizer(model_path, device):
    meta_path = derive_meta_path(model_path)
    if not os.path.exists(meta_path):
        sys.exit(f"Error: metadata file '{meta_path}' not found")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    gptconf = GPTConfig(**checkpoint["model_args"])
    _stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    model = GPT(gptconf)
    sys.stdout = _stdout
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    tokenizer = load_tokenizer(meta_path)
    return model, tokenizer, checkpoint


def register_residual_hooks(model):
    """Capture the residual stream at the input embedding and after every block.
    Returns (store, handles, layer_names). store[name] holds (1,T,C) on each
    forward pass."""
    store = {}
    handles = []
    names = []

    def make_hook(name):
        def hook(_module, _inp, out):
            # Block.forward returns the residual tensor; drop returns the
            # post-embedding tensor. Both are (1, T, C).
            store[name] = out.detach()
        return hook

    handles.append(model.transformer.drop.register_forward_hook(make_hook("embed")))
    names.append("embed")
    for i, block in enumerate(model.transformer.h):
        nm = f"L{i:02d}"
        handles.append(block.register_forward_hook(make_hook(nm)))
        names.append(nm)
    return store, handles, names


@torch.no_grad()
def collect_word_vectors(model, tokenizer, words, device, layer_names, store):
    """For each word, run it through every frame, capture residuals at the
    word's final-letter position and the position right after, average across
    frames. Returns two dicts: vectors[readout][layer] -> (n_words, C) array."""
    n_words = len(words)
    n_layers = len(layer_names)
    C = model.config.n_embd

    # accumulators: sum over frames, then divide
    acc = {
        "final": {nm: np.zeros((n_words, C), dtype=np.float64) for nm in layer_names},
        "after": {nm: np.zeros((n_words, C), dtype=np.float64) for nm in layer_names},
    }

    for wi, word in enumerate(words):
        for frame in FRAMES:
            assert "WORD" in frame
            text = frame.replace("WORD", word)
            # locate the slot: where 'word' was substituted
            slot_start = frame.index("WORD")
            final_letter_pos = slot_start + len(word) - 1
            after_pos = slot_start + len(word)  # the space after the word

            # sanity: every char must be in the char vocabulary
            for c in text:
                if c not in tokenizer.stoi:
                    sys.exit(f"Char {c!r} not in vocab (word={word!r})")
            ids = tokenizer.encode(text)
            assert len(ids) == len(text), "char tokenizer must be 1:1"
            x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]

            store.clear()
            model(x)  # populates store via hooks

            for nm in layer_names:
                resid = store[nm][0]  # (T, C)
                acc["final"][nm][wi] += resid[final_letter_pos].float().cpu().numpy()
                acc["after"][nm][wi] += resid[after_pos].float().cpu().numpy()

    nf = len(FRAMES)
    for readout in acc:
        for nm in layer_names:
            acc[readout][nm] /= nf
    return acc


def cosine_matrix(V):
    """Mean-center rows, L2-normalize, return cosine similarity matrix."""
    Vc = V - V.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Vc, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Vn = Vc / norms
    return Vn @ Vn.T, Vn


def separation_from_labels(cos, labels):
    """labels: 0/1 array. Return mean within-category cos minus mean between."""
    n = len(labels)
    iu = np.triu_indices(n, k=1)
    same = labels[iu[0]] == labels[iu[1]]
    within = cos[iu][same].mean()
    between = cos[iu][~same].mean()
    return within - between, within, between


def permutation_null(cos, n_per_group, n_perm, rng):
    """Null distribution of separation under random relabeling into two
    equal groups."""
    n = 2 * n_per_group
    null = np.empty(n_perm)
    base_labels = np.array([0] * n_per_group + [1] * n_per_group)
    for k in range(n_perm):
        perm = rng.permutation(n)
        lab = base_labels[perm]
        null[k], _, _ = separation_from_labels(cos, lab)
    return null


def minimal_pair_test(Vn, n_animals):
    """For each animal i, is it closer (cosine) to the centroid of the OTHER
    animals than to its paired object (index n_animals+i)? Count successes.
    Vn rows are unit vectors ordered [animals..., objects...]."""
    wins = 0
    details = []
    for i in range(n_animals):
        others = [j for j in range(n_animals) if j != i]
        animal_centroid = Vn[others].mean(axis=0)
        animal_centroid /= (np.linalg.norm(animal_centroid) or 1.0)
        sim_own = float(Vn[i] @ animal_centroid)
        sim_twin = float(Vn[i] @ Vn[n_animals + i])
        ok = sim_own > sim_twin
        wins += ok
        details.append((ANIMALS[i], OBJECTS[i], sim_own, sim_twin, ok))
    return wins, details


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="pt/char_uppercase_16L_1280.pt")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default, safe alongside MPS training), mps, or cuda")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_tsv", default=None)
    ap.add_argument("--heatmap", default=None,
                    help="optional PNG path for the cosine heatmap at the best layer")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    words = ANIMALS + OBJECTS
    n_animals = len(ANIMALS)
    labels = np.array([0] * len(ANIMALS) + [1] * len(OBJECTS))

    if not os.path.exists(args.model):
        sys.exit(f"Error: model file '{args.model}' not found")

    print(f"Loading {args.model} on {args.device} ...", flush=True)
    model, tokenizer, ckpt = load_model_and_tokenizer(args.model, args.device)
    if tokenizer.tokenizer_type != "char":
        sys.exit(f"This probe is for char models; got {tokenizer.tokenizer_type}")
    iter_num = ckpt.get("iter_num", "?")
    best_val = ckpt.get("best_val_loss", None)
    print(f"  {model.config.n_layer} layers, n_embd={model.config.n_embd}, "
          f"iter {iter_num}, val {best_val:.4f}" if best_val else "")

    store, handles, layer_names = register_residual_hooks(model)
    print(f"Probing {len(words)} words x {len(FRAMES)} frames "
          f"x {len(layer_names)} layers ...", flush=True)
    vectors = collect_word_vectors(model, tokenizer, words, args.device,
                                   layer_names, store)
    for h in handles:
        h.remove()

    # ---- per-layer, per-readout metrics --------------------------------
    rows = []
    for readout in ("final", "after"):
        for nm in layer_names:
            cos, Vn = cosine_matrix(vectors[readout][nm])
            sep, within, between = separation_from_labels(cos, labels)
            null = permutation_null(cos, n_animals, args.n_perm, rng)
            z = (sep - null.mean()) / (null.std() or 1.0)
            p = float((null >= sep).mean())
            wins, _ = minimal_pair_test(Vn, n_animals)
            rows.append(dict(readout=readout, layer=nm, within=within,
                             between=between, separation=sep, z=z, p=p,
                             minpair=wins, n_animals=n_animals))

    # ---- report --------------------------------------------------------
    print("\n" + "=" * 78)
    print("CATEGORY GEOMETRY: animals vs surface-matched objects")
    print("separation = within-category cosine minus between-category cosine")
    print("minpair = # of 9 animals nearer to other-animals centroid than to "
          "look-alike object")
    print("=" * 78)
    for readout in ("final", "after"):
        label = ("final letter of word" if readout == "final"
                 else "position after word")
        print(f"\n[readout: {label}]")
        print(f"{'layer':>6} {'within':>8} {'between':>8} {'separ':>8} "
              f"{'z':>7} {'p':>7} {'minpair':>8}")
        sub = [r for r in rows if r["readout"] == readout]
        for r in sub:
            star = " *" if r["p"] < 0.05 else ""
            print(f"{r['layer']:>6} {r['within']:>8.3f} {r['between']:>8.3f} "
                  f"{r['separation']:>8.3f} {r['z']:>7.1f} {r['p']:>7.3f} "
                  f"{r['minpair']:>5}/9{star}")
        best = max(sub, key=lambda r: r["separation"])
        print(f"  peak separation at {best['layer']}: {best['separation']:.3f} "
              f"(z={best['z']:.1f}, p={best['p']:.3f}, "
              f"minpair {best['minpair']}/9)")

    # minimal-pair detail at the overall best 'after' layer
    after_rows = [r for r in rows if r["readout"] == "after"]
    best = max(after_rows, key=lambda r: r["separation"])
    cos, Vn = cosine_matrix(vectors["after"][best["layer"]])
    wins, details = minimal_pair_test(Vn, n_animals)
    print(f"\nMinimal-pair detail at {best['layer']} (position-after readout):")
    print(f"  {'animal':>7} {'object':>7} {'sim_animals':>12} {'sim_twin':>9}  verdict")
    for a, o, so, st, ok in details:
        print(f"  {a:>7} {o:>7} {so:>12.3f} {st:>9.3f}  "
              f"{'animal-like' if ok else 'object-like'}")

    # ---- optional outputs ---------------------------------------------
    if args.out_tsv:
        with open(args.out_tsv, "w") as f:
            f.write("readout\tlayer\twithin\tbetween\tseparation\tz\tp\tminpair\n")
            for r in rows:
                f.write(f"{r['readout']}\t{r['layer']}\t{r['within']:.4f}\t"
                        f"{r['between']:.4f}\t{r['separation']:.4f}\t{r['z']:.3f}\t"
                        f"{r['p']:.4f}\t{r['minpair']}\n")
        print(f"\nWrote {args.out_tsv}")

    if args.heatmap:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            cos, _ = cosine_matrix(vectors["after"][best["layer"]])
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(words)))
            ax.set_yticks(range(len(words)))
            ax.set_xticklabels(words, rotation=90, fontsize=8)
            ax.set_yticklabels(words, fontsize=8)
            ax.axhline(n_animals - 0.5, color="k", lw=1)
            ax.axvline(n_animals - 0.5, color="k", lw=1)
            ax.set_title(f"Mean-centered cosine, {best['layer']} (after-word)\n"
                         f"{args.model}")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(args.heatmap, dpi=120)
            print(f"Wrote {args.heatmap}")
        except Exception as e:
            print(f"(heatmap skipped: {e})")


if __name__ == "__main__":
    main()
