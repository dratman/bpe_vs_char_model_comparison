"""
Shared data helpers for the char-vs-BPE training-dynamics study (coupler-queue 0012).

The corpus is cut by CHARACTER offset: text[:cut] is training text, text[cut:] is
the held-out region (cut = int(len(text) * (1 - heldout_frac))). Both tokenizers
see exactly the same split because the split happens before tokenization.
Token streams are cached on disk as small integer arrays (uint8 for the 78-char
vocabulary, uint16 for BPE 8192), keyed by the tokenizer file's sha256.
"""
import hashlib
import json
import os

import numpy as np
import torch


def sha256_file(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tokenizer_fingerprint(tokenizer, tokenizer_path=None):
    """sha256 of the saved tokenizer (BPE: its .json; char: sorted vocabulary)."""
    if tokenizer.tokenizer_type == 'bpe':
        if tokenizer_path is not None:
            jp = tokenizer_path.replace('.pkl', '.json')
            if os.path.exists(jp):
                return sha256_file(jp)
        return hashlib.sha256(tokenizer.tokenizer.to_str().encode()).hexdigest()
    if tokenizer.tokenizer_type == 'char':
        return hashlib.sha256(''.join(sorted(tokenizer.stoi)).encode()).hexdigest()
    raise ValueError(f"unsupported tokenizer {tokenizer.tokenizer_type}")


def heldout_cut(n_chars, heldout_frac):
    return int(n_chars * (1.0 - heldout_frac))


def encode_chars_fast(tokenizer, text):
    """Vectorised CharTokenizer.encode (unknown chars -> 0, as the class does)."""
    cps = np.frombuffer(text.encode('utf-32-le'), dtype=np.uint32)
    lut = np.zeros(int(cps.max()) + 1, dtype=np.uint16)
    for ch, i in tokenizer.stoi.items():
        if ord(ch) < len(lut):
            lut[ord(ch)] = i
    return lut[cps]


def encode_array(tokenizer, text):
    if tokenizer.tokenizer_type == 'char':
        arr = encode_chars_fast(tokenizer, text)
    else:
        arr = np.asarray(tokenizer.encode(text), dtype=np.int64)
    dtype = np.uint8 if tokenizer.vocab_size <= 256 else np.uint16
    assert arr.max() < np.iinfo(dtype).max
    return arr.astype(dtype)


def load_or_build_tokens(text, corpus_path, tokenizer, tokenizer_path, cache_dir,
                         heldout_frac):
    """Return (train_tokens, heldout_tokens, info) as torch tensors (uint8/int16),
    building and caching the token streams if the cache is absent or stale.
    `text` may be None if the cache is valid."""
    os.makedirs(cache_dir, exist_ok=True)
    fp = tokenizer_fingerprint(tokenizer, tokenizer_path)
    st = os.stat(corpus_path)
    key = dict(tokenizer=tokenizer.tokenizer_type, fingerprint=fp,
               corpus=os.path.abspath(corpus_path), corpus_size=st.st_size,
               heldout_frac=heldout_frac)
    stem = os.path.join(cache_dir, f"tokens_{tokenizer.tokenizer_type}_{fp[:12]}")
    meta_path, train_path, held_path = stem + '.json', stem + '_train.npy', stem + '_heldout.npy'
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if all(meta.get(k) == v for k, v in key.items()):
            print(f"Loading token cache {stem}_*.npy")
            tr = np.load(train_path)
            ho = np.load(held_path)
            return torch.from_numpy(tr), torch.from_numpy(ho), meta
    assert text is not None, "token cache missing and no text supplied"
    cut = heldout_cut(len(text), heldout_frac)
    print(f"Tokenizing train text ({cut:,} chars) and held-out text ({len(text) - cut:,} chars)...")
    tr = encode_array(tokenizer, text[:cut])
    ho = encode_array(tokenizer, text[cut:])
    np.save(train_path, tr)
    np.save(held_path, ho)
    meta = dict(key, cut=cut, n_chars=len(text), train_chars=cut,
                heldout_chars=len(text) - cut, train_tokens=int(len(tr)),
                heldout_tokens=int(len(ho)), chars_per_token_train=cut / len(tr),
                dtype=str(tr.dtype))
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=1)
    return torch.from_numpy(tr), torch.from_numpy(ho), meta
