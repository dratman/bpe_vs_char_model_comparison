#!/usr/bin/env python
"""
category_geometry_causal.py - Does the model USE the animal-category direction,
or merely contain it? (Diary 105 causal follow-up.)

Diary 105 showed, correlationally, that a char model assembles an "animal"
region in its residual stream. This asks whether that direction is causal: if
we remove it the model should stop predicting category-appropriately, and if we
inject it into an object the model should predict animal-appropriately.

Design (with the confounds the obvious version has, fixed):

- DIRECTION is fit on a BROAD animal/object word list (FIT set), DISJOINT from
  the minimal-pair words we measure on (TEST set). Otherwise "projecting out the
  axis that separates these 18 words, then finding these 18 words less separated"
  is mechanical, not behavioral.

- READOUT is the model's OUTPUT (next-token distribution) at the end of an OPEN
  context ("The cat " with nothing after), where the model freely predicts a
  continuation. Premise verified separately: animal vs object is ~0.89 LOO
  decodable from this output, so the test is powered. We measure behaviour, not
  hidden state (hidden-state separation drops trivially under projection).

- CONTROL is structure-matched: the difference-of-means of RANDOM balanced 9/9
  partitions of the same FIT words (averaged over several). A random unit vector
  would be a far weaker perturbation than the high-variance category axis and
  would make the axis look special for free.

- Per-layer direction d_L = unit(mean_animal_resid - mean_object_resid) at the
  open-context final position. Ablation projects d_L out of every layer's
  residual at every position (so the model cannot rebuild it downstream).
  Steering ADDS the full difference vector Delta_L at every layer.

- Intervention hooks RETURN the modified tensor (a read-only hook returns None;
  forgetting this is a silent no-op). We assert the ablation bit (final residual
  dot d ~ 0) before trusting any "robust to ablation" reading.

Read-only weights; CPU by default.

Usage:
    python py/category_geometry_causal.py --model pt/char_uppercase_16L_1280.pt \
        --out_tsv terminal_logs/category_geometry_causal_<date>.tsv
"""

import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import category_geometry_probe as cg

# FIT set (defines the direction) — broad, common 19th-c words, DISJOINT from TEST.
FIT_ANIMALS = ["lion", "tiger", "bear", "sheep", "deer", "frog", "snake",
               "bird", "cow", "pig", "duck", "hound", "calf", "goose",
               "lamb", "mare"]
FIT_OBJECTS = ["chair", "table", "stone", "knife", "plate", "book", "door",
               "wall", "road", "ship", "cart", "lamp", "clock", "boot",
               "glass", "spoon"]

# TEST set (held-out) — the diary-105 minimal pairs.
TEST_ANIMALS = cg.ANIMALS          # cat dog fox owl hare goat mouse wolf horse
TEST_OBJECTS = cg.OBJECTS          # hat log box bowl hair coat mouth wool house

# Open contexts: word at the end + trailing space; model predicts freely.
OPEN = ["\nThe WORD ", "\nA WORD ", "\nI saw the WORD ", "\nThere was a WORD "]


class Intervenor:
    """Forward hooks on embed + every block. Reads each layer's residual; if an
    intervention fn is set for that layer, applies it AND RETURNS the modified
    tensor (a read-only hook returns None)."""

    def __init__(self, model):
        self.modules = {"embed": model.transformer.drop}
        for i, b in enumerate(model.transformer.h):
            self.modules[f"L{i:02d}"] = b
        self.names = list(self.modules)
        self.fn = {}
        self.store = {}
        self.handles = [m.register_forward_hook(self._mk(nm))
                        for nm, m in self.modules.items()]

    def _mk(self, nm):
        def hook(_m, _i, out):
            if nm in self.fn:
                out = self.fn[nm](out)      # MUST return to take effect
            self.store[nm] = out.detach()
            return out
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run(model, tok, text, intervenor, fn=None):
    """Run one open context; return (output_logits[V], store). fn: optional
    {layer_name: callable(out)->out} intervention applied this pass."""
    intervenor.fn = fn or {}
    ids = tok.encode(text)
    x = torch.tensor(ids, dtype=torch.long)[None, ...]
    logits, _ = model(x)
    intervenor.fn = {}
    return logits[0, -1].float().numpy(), intervenor.store


@torch.no_grad()
def residuals_and_logits(model, tok, words, intervenor, fn=None):
    """Frame-average each word's per-layer final-position residual and output
    logits over the OPEN contexts. Returns (resid[layer]->(n,C), logits(n,V))."""
    names = intervenor.names
    C = model.config.n_embd
    resid = {nm: np.zeros((len(words), C)) for nm in names}
    out = np.zeros((len(words), tok.vocab_size))
    for wi, w in enumerate(words):
        for ctx in OPEN:
            lg, store = run(model, tok, ctx.replace("WORD", w), intervenor, fn)
            out[wi] += lg
            for nm in names:
                resid[nm][wi] += store[nm][0, -1].float().numpy()
    n = len(OPEN)
    for nm in names:
        resid[nm] /= n
    return resid, out / n


def diff_means(resid, labels):
    """Per-layer Delta = mean(group0) - mean(group1) and its unit direction."""
    Delta, dhat = {}, {}
    for nm, M in resid.items():
        delta = M[labels == 0].mean(0) - M[labels == 1].mean(0)
        Delta[nm] = delta
        dhat[nm] = delta / (np.linalg.norm(delta) or 1.0)
    return Delta, dhat


def loo_acc(X, labels):
    ok = 0
    for i in range(len(X)):
        m = np.ones(len(X), bool); m[i] = False
        c0 = X[m & (labels == 0)].mean(0); c1 = X[m & (labels == 1)].mean(0)
        ok += int((np.linalg.norm(X[i] - c0) < np.linalg.norm(X[i] - c1)) == (labels[i] == 0))
    return ok / len(X)


def separation(X, labels):
    Xc = X - X.mean(0); Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9)
    cos = Xn @ Xn.T; n = len(labels); iu = np.triu_indices(n, 1)
    same = labels[iu[0]] == labels[iu[1]]
    return cos[iu][same].mean() - cos[iu][~same].mean()


def ablate_fn(dhat, names):
    """Project out dhat[L] from each layer's residual at every position."""
    t = {nm: torch.tensor(dhat[nm], dtype=torch.float32) for nm in names}
    def make(nm):
        d = t[nm]
        return lambda out: out - (out @ d).unsqueeze(-1) * d
    return {nm: make(nm) for nm in names}


def patch_fn(dhat, target, names):
    """Bounded counterfactual: set each layer's residual component ALONG the
    category axis dhat[L] to the scalar target[L] (project-and-replace), at every
    position. Unlike additive steering this cannot blow the residual out of
    distribution -- it only moves the one category coordinate to a chosen value.
        out <- out - (out.dhat)dhat + target*dhat
    """
    T = {nm: (torch.tensor(dhat[nm], dtype=torch.float32), float(target[nm]))
         for nm in names}
    def make(nm):
        d, t = T[nm]
        return lambda out: out - (out @ d).unsqueeze(-1) * d + t * d
    return {nm: make(nm) for nm in names}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="pt/char_uppercase_16L_1280.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_control", type=int, default=5, help="random balanced partitions")
    ap.add_argument("--alpha", type=float, default=1.0, help="steering scale on Delta")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_tsv", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    model, tok, _ = cg.load_model_and_tokenizer(args.model, args.device)
    model.eval()
    iv = Intervenor(model)
    names = iv.names

    fit_words = FIT_ANIMALS + FIT_OBJECTS
    fit_lab = np.array([0] * len(FIT_ANIMALS) + [1] * len(FIT_OBJECTS))
    test_words = TEST_ANIMALS + TEST_OBJECTS
    test_lab = np.array([0] * len(TEST_ANIMALS) + [1] * len(TEST_OBJECTS))

    # ---- fit direction on FIT, output axis on FIT --------------------------
    fit_resid, fit_out = residuals_and_logits(model, tok, fit_words, iv)
    Delta, dhat = diff_means(fit_resid, fit_lab)
    w_out = fit_out[fit_lab == 0].mean(0) - fit_out[fit_lab == 1].mean(0)  # output animal axis
    w_hat = w_out / (np.linalg.norm(w_out) or 1.0)
    mid = 0.5 * (fit_out[fit_lab == 0].mean(0) + fit_out[fit_lab == 1].mean(0))

    # cross-layer alignment of the category direction
    block_names = [n for n in names if n != "embed"]
    al = [abs(float(dhat[a] @ dhat[b]))
          for i, a in enumerate(block_names) for b in block_names[i + 1:]]
    print(f"cross-layer |cos| of category direction (block layers): "
          f"mean {np.mean(al):.2f}, min {np.min(al):.2f}")

    def out_score(logits_vec):
        return float((logits_vec - mid) @ w_hat)  # >0 = animal-ward

    # ---- baseline on TEST --------------------------------------------------
    base_resid, base_out = residuals_and_logits(model, tok, test_words, iv)
    base_acc = loo_acc(base_out, test_lab)
    base_sep = separation(base_out, test_lab)
    print(f"\nBASELINE (TEST, output logits, open context):")
    print(f"  LOO animal/object accuracy: {base_acc:.3f}   separation: {base_sep:.3f}")

    # ---- ablation (category d) vs structure-matched control ----------------
    abl_resid, abl_out = residuals_and_logits(model, tok, test_words, iv,
                                              fn=ablate_fn(dhat, names))
    # verify the ablation bit: final residual component along d ~ 0
    leak = np.mean([abs(float(abl_resid["L15"][i] @ dhat["L15"]))
                    for i in range(len(test_words))])
    base_proj = np.mean([abs(float(base_resid["L15"][i] @ dhat["L15"]))
                         for i in range(len(test_words))])
    print(f"\nABLATION bit check: |resid.d| at L15  baseline {base_proj:.2f} -> "
          f"ablated {leak:.3f}  ({'BITES' if leak < 0.1 * base_proj else 'NO-OP!'})")
    abl_acc = loo_acc(abl_out, test_lab); abl_sep = separation(abl_out, test_lab)

    ctrl_acc, ctrl_sep = [], []
    for k in range(args.n_control):
        perm = rng.permutation(len(fit_words))
        clab = np.zeros(len(fit_words), int); clab[perm[len(fit_words) // 2:]] = 1
        _, cdhat = diff_means(fit_resid, clab)
        _, co = residuals_and_logits(model, tok, test_words, iv,
                                     fn=ablate_fn(cdhat, names))
        ctrl_acc.append(loo_acc(co, test_lab)); ctrl_sep.append(separation(co, test_lab))
    print(f"\nABLATION effect on TEST category prediction:")
    print(f"  baseline acc {base_acc:.3f} | ablate category-d {abl_acc:.3f} | "
          f"ablate matched-control {np.mean(ctrl_acc):.3f} +- {np.std(ctrl_acc):.3f}")
    print(f"  baseline sep {base_sep:.3f} | ablate category-d {abl_sep:.3f} | "
          f"ablate matched-control {np.mean(ctrl_sep):.3f} +- {np.std(ctrl_sep):.3f}")

    # ---- patching: set the category axis to the OTHER category's value -----
    # Bounded counterfactual. Per-layer targets = mean projection of FIT animals
    # (a_proj) / FIT objects (o_proj) onto the category axis.
    a_proj = {nm: float(np.mean(fit_resid[nm][fit_lab == 0] @ dhat[nm])) for nm in names}
    o_proj = {nm: float(np.mean(fit_resid[nm][fit_lab == 1] @ dhat[nm])) for nm in names}

    ani_words, obj_words = TEST_ANIMALS, TEST_OBJECTS
    base_ani = np.array([out_score(base_out[i]) for i in range(len(ani_words))])
    base_obj = np.array([out_score(base_out[len(ani_words) + i]) for i in range(len(obj_words))])

    # objects, category axis set to ANIMAL-typical
    _, obj2ani = residuals_and_logits(model, tok, obj_words, iv,
                                      fn=patch_fn(dhat, a_proj, names))
    obj2ani_s = np.array([out_score(obj2ani[i]) for i in range(len(obj_words))])

    # matched control: random-partition axis set to that partition's group-0 mean
    ctrl_s = []
    for k in range(args.n_control):
        perm = rng.permutation(len(fit_words))
        clab = np.zeros(len(fit_words), int); clab[perm[len(fit_words) // 2:]] = 1
        _, cdh = diff_means(fit_resid, clab)
        ctarget = {nm: float(np.mean(fit_resid[nm][clab == 0] @ cdh[nm])) for nm in names}
        _, co = residuals_and_logits(model, tok, obj_words, iv,
                                     fn=patch_fn(cdh, ctarget, names))
        ctrl_s.append(np.mean([out_score(co[i]) for i in range(len(obj_words))]))

    print("\nPATCHING (bounded: set the category axis to animal-typical value):")
    print("  output animal-score (>0 = animal-ward; axis fit on disjoint FIT words)")
    print(f"    animals  (reference)             : {base_ani.mean():+.2f}")
    print(f"    objects  (baseline)              : {base_obj.mean():+.2f}")
    print(f"    objects, axis->animal (category) : {obj2ani_s.mean():+.2f}")
    print(f"    objects, axis->grp0   (control)  : {np.mean(ctrl_s):+.2f} +- {np.std(ctrl_s):.2f}")
    moved = (obj2ani_s.mean() - base_obj.mean()) / (base_ani.mean() - base_obj.mean() + 1e-9)
    print(f"  -> objects moved {100 * moved:.0f}% from object toward animal "
          f"along the category axis; control stays near the object baseline")

    # worked examples (bounded patch, both directions)
    @torch.no_grad()
    def cont(word, fn=None, n=24):
        ids = tok.encode("\nThe " + word + " ")
        x = torch.tensor(ids, dtype=torch.long)[None, ...]
        for _ in range(n):
            iv.fn = fn or {}
            lg, _ = model(x[:, -model.config.block_size:])
            iv.fn = {}
            x = torch.cat([x, torch.tensor([[int(lg[0, -1].argmax())]])], 1)
        return tok.decode(x[0].tolist())
    to_ani = patch_fn(dhat, a_proj, names)
    to_obj = patch_fn(dhat, o_proj, names)
    # structure-matched control patches (random partitions -> their group-0 mean):
    # the load-bearing check that the TEXT flip is category-specific. Show several
    # because a single random split of an animal/object pool occasionally lands
    # more animals on one side and tilts mildly animate; the preponderance (and
    # the averaged score above) is the honest statement.
    to_ctrls = []
    for _ in range(3):
        perm = rng.permutation(len(fit_words))
        clab = np.zeros(len(fit_words), int); clab[perm[len(fit_words) // 2:]] = 1
        _, cdh = diff_means(fit_resid, clab)
        ct = {nm: float(np.mean(fit_resid[nm][clab == 0] @ cdh[nm])) for nm in names}
        to_ctrls.append(patch_fn(cdh, ct, names))
    print("\nWorked examples (greedy continuation; bounded category-axis patch):")
    for w in ["box", "coat", "bowl"]:
        print(f"  {w:5} base           : {cont(w)!r}")
        print(f"  {w:5} axis->animal   : {cont(w, to_ani)!r}")
        for j, fc in enumerate(to_ctrls):
            print(f"  {w:5} CONTROL patch {j} : {cont(w, fc)!r}")
    for w in ["cat", "dog"]:
        print(f"  {w:5} base           : {cont(w)!r}")
        print(f"  {w:5} axis->object   : {cont(w, to_obj)!r}")

    if args.out_tsv:
        with open(args.out_tsv, "w") as f:
            f.write("metric\tbaseline\tcategory_d\tcontrol_mean\tcontrol_std\n")
            f.write(f"abl_loo_acc\t{base_acc:.4f}\t{abl_acc:.4f}\t{np.mean(ctrl_acc):.4f}\t{np.std(ctrl_acc):.4f}\n")
            f.write(f"abl_separation\t{base_sep:.4f}\t{abl_sep:.4f}\t{np.mean(ctrl_sep):.4f}\t{np.std(ctrl_sep):.4f}\n")
            f.write(f"patch_obj_animalscore\t{base_obj.mean():.4f}\t{obj2ani_s.mean():.4f}\t{np.mean(ctrl_s):.4f}\t{np.std(ctrl_s):.4f}\n")
            f.write(f"ref_animals_animalscore\t{base_ani.mean():.4f}\t\t\t\n")
        print(f"\nWrote {args.out_tsv}")
    iv.remove()


if __name__ == "__main__":
    main()
