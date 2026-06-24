#!/usr/bin/env python
"""
category_family_resemblance.py - Does a Wittgensteinian no-common-attribute
category ("game") cohere as one region in the model, or fragment?

Diary 105 showed "animal" — a category whose members share many correlated
attributes — is a single coherent, causal direction. Wittgenstein's actual
subject was the opposite: "game" (board games, ball games, card games, word
puzzles) has NO single common attribute, only overlapping family resemblances.
This asks whether the model nonetheless binds them into a region.

Design (with the confounds a naive version has, fixed — see the diary):

- HEADLINE = coherence-above-parts with a PERMUTATION NULL. For each category,
  pool its 12 words + 12 neutral contrast words, and ask whether the true
  grouping's (within-category minus to-contrast) cosine beats a null that
  randomly relabels 12 of the 24 as "category". This asks: is "game" a
  PRIVILEGED grouping in the geometry, or just 24 words? "animal" calibrates
  the scale (it should crush its null).

- The FREQUENCY CONFOUND is asymmetric and load-bearing. Game words are 10-300x
  rarer than the animals (dominoes 176, croquet 263 vs horse 55k), and rare
  words have noisier representations that INFLATE scatter — mimicking
  fragmentation. So: a "game fails to cohere" result would be UNINTERPRETABLE
  (could be rare-word noise); a "game coheres ANYWAY" result is ROBUST (the
  noise fights against it). That asymmetry is why coherence, not fragmentation,
  is the headline.

- INTRINSIC DIMENSIONALITY (PC1 variance fraction; participation ratio of the
  mean-centered category words) operationalizes "one attribute = one dimension"
  directly. Prediction: animal concentrates near ~1 direction; game spreads
  across more. Descriptive, not inferential (n=12 in 1280-d).

- NO classification / linear separability (24 points in 1280-d are always
  separable -> ~100% for any grouping, proves nothing). Cosine + permutation.
  The 2D projection is ILLUSTRATION (36 points to 2-d can look like anything);
  the nulls carry the claim.

- Fragmentation (within-subtype minus within-category) is reported but DEMOTED:
  the game subtypes were chosen disparate, so it partly re-measures the picks.

Read-only; CPU by default. Reuses the open-context after-word residual readout
from category_geometry_causal (same as the diary-105 causal test).

Usage:
    python py/category_family_resemblance.py --model pt/char_uppercase_16L_1280.pt \
        --out_tsv terminal_logs/family_resemblance_<date>.tsv \
        --out_plot plots/family_resemblance_<date>.png
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import category_geometry_probe as cg
import category_geometry_causal as cc

# Unified category (control): members share many correlated attributes.
ANIMAL_SUB = {"wild": ["lion", "tiger", "wolf", "fox"],
              "farm": ["horse", "cow", "sheep", "goat"],
              "bird": ["owl", "hen", "goose", "hawk"]}
# Family-resemblance category (test): no single common attribute.
GAME_SUB = {"board": ["chess", "draughts", "billiards", "dominoes"],
            "field": ["football", "tennis", "golf", "croquet"],
            "parlour": ["cards", "whist", "riddle", "puzzle"]}
# Neutral baseline (the "rest of the space").
CONTRAST = ["river", "mountain", "letter", "window", "candle", "church",
            "valley", "silver", "kitchen", "garden", "bridge", "cottage"]


def flatten(sub):
    words, subs = [], []
    for s, ws in sub.items():
        words += ws; subs += [s] * len(ws)
    return words, subs


def centered_unit(M):
    Mc = M - M.mean(0)
    return Mc / (np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-9)


def coherence_and_null(cat_M, ctr_M, n_perm, rng):
    """within(category) - between(category,contrast) on the pooled, mean-centered
    24 words, with a label-permutation null. Returns (coherence, z, p)."""
    n = len(cat_M)
    U = centered_unit(np.vstack([cat_M, ctr_M]))   # 24 x C
    cos = U @ U.T
    idx = np.arange(len(U))

    def stat(cat_idx):
        ci = np.zeros(len(U), bool); ci[cat_idx] = True
        within = cos[np.ix_(ci, ci)][np.triu_indices(ci.sum(), 1)].mean()
        between = cos[np.ix_(ci, ~ci)].mean()
        return within - between

    obs = stat(idx[:n])
    null = np.array([stat(rng.permutation(len(U))[:n]) for _ in range(n_perm)])
    z = (obs - null.mean()) / (null.std() or 1.0)
    p = float((null >= obs).mean())
    return obs, z, p


def within_breakdown(cat_M, ctr_M, subs):
    """within-subtype, within-category(cross-subtype), to-contrast (cosine on the
    pooled mean-centered space). Illustrative fragmentation numbers."""
    U = centered_unit(np.vstack([cat_M, ctr_M]))
    n = len(cat_M)
    cos = U @ U.T
    subs = np.array(subs)
    same = [cos[i, j] for i in range(n) for j in range(i + 1, n) if subs[i] == subs[j]]
    cross = [cos[i, j] for i in range(n) for j in range(i + 1, n) if subs[i] != subs[j]]
    to_ctr = cos[:n, n:].mean()
    return np.mean(same), np.mean(cross), to_ctr


def dimensionality(M):
    """PC1 variance fraction and participation ratio of the mean-centered words."""
    Mc = M - M.mean(0)
    s = np.linalg.svd(Mc, compute_uv=False)
    lam = s ** 2
    if lam.sum() == 0:
        return 0.0, 0.0
    pc1 = lam[0] / lam.sum()
    pr = (lam.sum() ** 2) / (lam ** 2).sum()      # participation ratio = effective dims
    return float(pc1), float(pr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="pt/char_uppercase_16L_1280.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_tsv", default=None)
    ap.add_argument("--out_plot", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    animals, an_subs = flatten(ANIMAL_SUB)
    games, gm_subs = flatten(GAME_SUB)
    all_words = animals + games + CONTRAST

    print(f"Loading {args.model} ...", flush=True)
    model, tok, _ = cg.load_model_and_tokenizer(args.model, args.device)
    model.eval()
    iv = cc.Intervenor(model)
    names = iv.names
    print(f"Probing {len(all_words)} words (12 animal / 12 game / 12 contrast) "
          f"x {len(cc.OPEN)} open frames ...", flush=True)
    resid, _ = cc.residuals_and_logits(model, tok, all_words, iv)
    iv.remove()

    ia = slice(0, len(animals))
    ig = slice(len(animals), len(animals) + len(games))
    ic = slice(len(animals) + len(games), len(all_words))

    rows = []
    for li, nm in enumerate(names):
        M = resid[nm]
        a_coh, a_z, a_p = coherence_and_null(M[ia], M[ic], args.n_perm, rng)
        g_coh, g_z, g_p = coherence_and_null(M[ig], M[ic], args.n_perm, rng)
        a_pc1, a_pr = dimensionality(M[ia])
        g_pc1, g_pr = dimensionality(M[ig])
        c_pc1, c_pr = dimensionality(M[ic])
        rows.append(dict(layer=nm, li=li,
                         a_coh=a_coh, a_z=a_z, a_p=a_p, g_coh=g_coh, g_z=g_z, g_p=g_p,
                         a_pr=a_pr, g_pr=g_pr, c_pr=c_pr,
                         a_pc1=a_pc1, g_pc1=g_pc1, c_pc1=c_pc1))

    print("\n" + "=" * 78)
    print("COHERENCE-ABOVE-PARTS (within-category minus to-contrast), with "
          "permutation null")
    print("  animal = unified category (calibration); game = family resemblance")
    print("=" * 78)
    print(f"{'layer':>6} | {'animal coh':>10} {'z':>6} {'p':>6} | "
          f"{'game coh':>9} {'z':>6} {'p':>6}")
    for r in rows:
        s = " *" if r["g_p"] < 0.05 else "  "
        print(f"{r['layer']:>6} | {r['a_coh']:>10.3f} {r['a_z']:>6.1f} {r['a_p']:>6.3f} | "
              f"{r['g_coh']:>9.3f} {r['g_z']:>6.1f} {r['g_p']:>6.3f}{s}")

    print("\n" + "=" * 78)
    print("INTRINSIC DIMENSIONALITY of the 12 category words (mean-centered)")
    print("  PR = participation ratio (effective # of dimensions, max 11); "
          "PC1 = variance frac in top direction")
    print("=" * 78)
    print(f"{'layer':>6} | {'animal PR':>9} {'game PR':>8} {'contrast PR':>11} | "
          f"{'an PC1':>7} {'gm PC1':>7} {'ct PC1':>7}")
    for r in rows:
        print(f"{r['layer']:>6} | {r['a_pr']:>9.1f} {r['g_pr']:>8.1f} {r['c_pr']:>11.1f} | "
              f"{r['a_pc1']:>7.2f} {r['g_pc1']:>7.2f} {r['c_pc1']:>7.2f}")

    # representative late layer
    late = [r for r in rows if r["li"] >= 10]
    rl = max(late, key=lambda r: r["a_coh"])
    nm = rl["layer"]
    print(f"\n--- representative late layer {nm} ---")
    a_same, a_cross, a_toc = within_breakdown(resid[nm][ia], resid[nm][ic], an_subs)
    g_same, g_cross, g_toc = within_breakdown(resid[nm][ig], resid[nm][ic], gm_subs)
    print("  cosine breakdown (DEMOTED — subtypes chosen disparate for game):")
    print(f"    animal: within-subtype {a_same:+.3f}  cross-subtype {a_cross:+.3f}  "
          f"to-contrast {a_toc:+.3f}")
    print(f"    game:   within-subtype {g_same:+.3f}  cross-subtype {g_cross:+.3f}  "
          f"to-contrast {g_toc:+.3f}")
    print(f"  --> the load-bearing question is cross-subtype vs to-contrast:")
    print(f"      animal binds across subtypes by {a_cross - a_toc:+.3f}; "
          f"game by {g_cross - g_toc:+.3f}")
    print(f"  coherence z at {nm}: animal {rl['a_z']:.1f}, game {rl['g_z']:.1f}  "
          f"(game sits at {100*rl['g_coh']/rl['a_coh']:.0f}% of animal's coherence)")

    if args.out_tsv:
        with open(args.out_tsv, "w") as f:
            f.write("layer\tanimal_coh\tanimal_z\tanimal_p\tgame_coh\tgame_z\tgame_p\t"
                    "animal_PR\tgame_PR\tcontrast_PR\tanimal_PC1\tgame_PC1\tcontrast_PC1\n")
            for r in rows:
                f.write(f"{r['layer']}\t{r['a_coh']:.4f}\t{r['a_z']:.3f}\t{r['a_p']:.4f}\t"
                        f"{r['g_coh']:.4f}\t{r['g_z']:.3f}\t{r['g_p']:.4f}\t"
                        f"{r['a_pr']:.3f}\t{r['g_pr']:.3f}\t{r['c_pr']:.3f}\t"
                        f"{r['a_pc1']:.4f}\t{r['g_pc1']:.4f}\t{r['c_pc1']:.4f}\n")
        print(f"\nWrote {args.out_tsv}")

    if args.out_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            li = [r["li"] for r in rows]; xt = [r["layer"] for r in rows]
            fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))

            ax[0].plot(li, [r["a_coh"] for r in rows], "o-", color="C0", label="animal")
            ax[0].plot(li, [r["g_coh"] for r in rows], "s-", color="C3", label="game")
            ax[0].axhline(0, color="grey", lw=.8, ls="--")
            ax[0].set_title("Coherence above parts\n(within-category minus to-contrast cosine)")
            ax[0].set_xlabel("layer"); ax[0].set_xticks(li); ax[0].set_xticklabels(xt, rotation=90, fontsize=7)
            ax[0].legend()

            ax[1].plot(li, [r["a_pr"] for r in rows], "o-", color="C0", label="animal")
            ax[1].plot(li, [r["g_pr"] for r in rows], "s-", color="C3", label="game")
            ax[1].plot(li, [r["c_pr"] for r in rows], "^-", color="grey", label="contrast")
            ax[1].set_title("Intrinsic dimensionality\n(participation ratio, effective # dims)")
            ax[1].set_xlabel("layer"); ax[1].set_xticks(li); ax[1].set_xticklabels(xt, rotation=90, fontsize=7)
            ax[1].legend()

            # ILLUSTRATION: 2D PCA of animal+game (mean-centered together) at late layer
            M = np.vstack([resid[nm][ia], resid[nm][ig]])
            Mc = M - M.mean(0)
            U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
            P = Mc @ Vt[:2].T
            cmap = {"wild": "#1f77b4", "farm": "#2ca02c", "bird": "#17becf",
                    "board": "#d62728", "field": "#ff7f0e", "parlour": "#9467bd"}
            labs = an_subs + gm_subs; wds = animals + games
            for i, (x, y) in enumerate(P):
                ax[2].scatter(x, y, color=cmap[labs[i]], s=40,
                              marker=("o" if i < len(animals) else "s"))
                ax[2].annotate(wds[i], (x, y), fontsize=7, alpha=.8)
            ax[2].set_title(f"ILLUSTRATION only — 2D PCA at {nm}\n"
                            "circles=animal subtypes, squares=game subtypes")
            ax[2].set_xlabel("PC1"); ax[2].set_ylabel("PC2")

            fig.suptitle("Does 'game' (no common attribute) cohere like 'animal'? — "
                         "char_uppercase_16L_1280", fontsize=12)
            fig.tight_layout()
            fig.savefig(args.out_plot, dpi=120)
            print(f"Wrote {args.out_plot}")
        except Exception as e:
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
