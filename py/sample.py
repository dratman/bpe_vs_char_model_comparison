#!/usr/bin/env python
"""
sample.py - Sampling from models trained with train.py
Works with both standard softmax attention and linear attention models.
Supports both character-level and BPE tokenization.

Features:
- Lowercases prompts automatically (for lowercase-only vocabularies)
- Float16 support via --float16 flag
- Batched generation via --batch flag (faster for multiple samples)
- Residual-stream similarity probe via --probe build-index / --probe match

Probe usage:
  # Build an index over a directory of .txt extracts:
  python sample.py --model model.pt --probe build-index \
      --probe_corpus corpus/probe_demo --probe_index my.idx

  # Match a query against the index:
  python sample.py --model model.pt --probe match \
      --probe_index my.idx --probe_query "The cook melted fat in a skillet" \
      --probe_output match_summary.md

  Probe conventions:
  - "Activation" = residual stream, captured after each transformer block,
    PRE-final-layernorm. Layer 0 = post-embedding output (pre-block-0).
  - Cosine similarity with per-layer anisotropy correction: the global mean
    vector (over all reference tokens at each layer) is subtracted from every
    vector before comparison. Required for transformer residual spaces.
  - Similarity metric: cosine similarity.
  - All flags: --probe_corpus, --probe_index, --probe_layers (default: every
    other layer incl. first and last), --probe_threshold (default 0.5),
    --probe_topk (default 3), --probe_query, --probe_query_file, --probe_output.
"""

import io
import json
import os
import re
import sys
import zipfile
import argparse
import pickle
import torch
import numpy as np
from contextlib import nullcontext
from model import GPTConfig, GPT
from tokenizer import load_tokenizer


def capitalize_sentences(text):
    """Capitalize first letter, first letter after sentence-ending punctuation
    (even through a quote mark), and standalone 'i' before a space.

    Intended for output from lowercase-only models. Skip this for models
    trained with case preserved — see detect_case_preservation()."""
    if not text:
        return text
    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    text = re.sub(r'([.?!])\s*("?)\s*([a-z])',
                  lambda m: m.group(1) + ' ' + m.group(2) + m.group(3).upper(), text)
    text = re.sub(r'\bi ', 'I ', text)
    return text


def detect_case_preservation(tokenizer):
    """Return True if the tokenizer's vocabulary contains any uppercase letter.

    Char tokenizers expose itos directly. BPE tokenizers expose their
    vocab via the underlying HF tokenizer. If detection fails for any
    reason, return False (the safe default — apply lowercase-pipeline
    behavior).
    """
    try:
        if tokenizer.tokenizer_type == 'char':
            return any(c.isupper() for c in tokenizer.itos.values())
        elif tokenizer.tokenizer_type in ('bpe', 'wordpiece'):
            vocab = tokenizer.tokenizer.get_vocab()
            return any(c.isupper() for token in vocab.keys() for c in token)
    except Exception:
        pass
    return False


@torch.no_grad()
def generate_local(model, x_init, max_new_tokens, temperature=1.0, top_k=None, rep_penalty=1.0, device='cpu', stop_token_id=None):
    """
    Generate text from an initial prompt (single sequence), with optional
    repetition penalty and early stop.

    Thin wrapper around GPT.generate() — the single canonical generation loop
    now lives in model.py. Kept for backward compatibility with callers such as
    real_word_fraction.py and memorization_probe.py.

    x_init: (1, T) prompt indices.
    stop_token_id: Optional token ID to stop generation (e.g., newline).
    device: unused (x_init is already on the model's device); kept for API compat.
    Returns: (1, T+max_new_tokens) or shorter if stop_token_id fires.
    """
    return model.generate(
        x_init, max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        rep_penalty=rep_penalty,
        stop_token_id=stop_token_id,
    )


@torch.no_grad()
def generate_batched(model, x_init, num_samples, max_new_tokens, temperature=1.0, top_k=None, device='cpu'):
    """
    Generate multiple samples in parallel (batched).

    Thin wrapper around GPT.generate(): repeats the prompt across num_samples
    rows and runs them together. No repetition penalty or early stopping in
    batched mode (stop tokens are truncated after generation by the caller).

    x_init: (1, T) prompt indices.
    Returns: (num_samples, T+max_new_tokens).
    """
    x = x_init.repeat(num_samples, 1)
    return model.generate(
        x, max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )


def truncate_at_stop_token(tokens, stop_token_id, prompt_length):
    """
    Truncate a token sequence at the first stop token after the prompt.
    Returns the truncated list of tokens.
    """
    if stop_token_id is None:
        return tokens

    # Look for stop token only in generated portion (after prompt)
    for i in range(prompt_length, len(tokens)):
        if tokens[i] == stop_token_id:
            return tokens[:i]  # Exclude the stop token itself

    return tokens


# ── Probe helpers ─────────────────────────────────────────────────────────────

@torch.no_grad()
def _probe_hidden_states(model, token_ids, device):
    """Single forward pass; returns residual-stream hidden states as numpy arrays.

    Returns a list of (T, n_embd) float32 arrays, one per layer:
      index 0  = post-embedding output, pre-block-0   (layer 0)
      index k  = post-block-(k-1) output              (layers 1..n_layer)
    All captured pre-final-layernorm, as documented in the index convention.
    """
    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    _, _, hs = model.forward(x, return_hidden_states=True)
    return [h[0].cpu().float().numpy() for h in hs]


def _probe_default_layers(n_layer):
    """Every other layer from 0 to n_layer, inclusive of both endpoints."""
    return sorted(set(range(0, n_layer + 1, 2)) | {n_layer})


def _probe_parse_layers(spec, n_layer):
    if spec is None:
        return _probe_default_layers(n_layer)
    try:
        layers = sorted(set(int(x.strip()) for x in spec.split(',')))
    except ValueError:
        raise SystemExit(f"--probe_layers: expected comma-separated integers, got {spec!r}")
    for l in layers:
        if not 0 <= l <= n_layer:
            raise SystemExit(f"--probe_layers: layer {l} out of range [0, {n_layer}]")
    return layers


def _probe_save_index(path, arrays):
    """Write index arrays as a .npz file with a fixed modification timestamp.

    Identical input data always produces byte-identical output (the standard
    np.savez_compressed embeds the current timestamp, breaking this property).
    Compatible with np.load(path, allow_pickle=False).
    """
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:
        for name in sorted(arrays.keys()):
            buf = io.BytesIO()
            np.save(buf, np.asarray(arrays[name]))
            info = zipfile.ZipInfo(name + '.npy')
            info.date_time = (2000, 1, 1, 0, 0, 0)  # fixed: byte-identical output
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, buf.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


def probe_build_index(model, tokenizer, device, args):
    """Build a residual-stream similarity index from a directory of .txt extracts."""
    from pathlib import Path

    corpus_dir = Path(args.probe_corpus)
    if not corpus_dir.is_dir():
        raise SystemExit(f"--probe_corpus: not a directory: {corpus_dir}")
    files = sorted(corpus_dir.glob('*.txt'))
    if not files:
        raise SystemExit(f"No .txt files found in {corpus_dir}")

    n_layer = model.config.n_layer
    layers = _probe_parse_layers(args.probe_layers, n_layer)
    block_size = model.config.block_size

    print(f"Building index: {len(files)} files, layers {layers}, model {os.path.basename(args.model)}")

    extract_ids = []
    extract_token_lists = []
    per_layer_tok = {l: [] for l in layers}  # l -> list of (T_i, C) arrays

    for f in files:
        text = f.read_text(encoding='utf-8').strip()
        if not text:
            print(f"  Warning: empty extract {f.name}, skipping")
            continue
        token_ids = tokenizer.encode(text)
        if not token_ids:
            continue
        if len(token_ids) > block_size:
            token_ids = token_ids[:block_size]
        token_strs = [tokenizer.decode([t]) for t in token_ids]
        hs = _probe_hidden_states(model, token_ids, device)
        extract_ids.append(f.name)
        extract_token_lists.append(token_strs)
        for l in layers:
            per_layer_tok[l].append(hs[l])
        print(f"  {f.name}: {len(token_ids)} tokens")

    n = len(extract_ids)
    if n == 0:
        raise SystemExit("No valid extracts found.")
    print(f"Processed {n} extracts.")

    # Per-layer centering means: true global mean over ALL token positions of ALL extracts.
    centering_means = {
        l: np.concatenate(per_layer_tok[l], axis=0).mean(axis=0)
        for l in layers
    }

    arrays = {}
    for i in range(n):
        arrays[f'tokens_{i}'] = np.array(extract_token_lists[i])
        for l in layers:
            vecs = per_layer_tok[l][i]
            arrays[f'tok_{l}_{i}'] = vecs
            arrays[f'sum_{l}_{i}'] = vecs.mean(axis=0)  # per-extract summary = mean over positions
    for l in layers:
        arrays[f'mean_{l}'] = centering_means[l]

    meta = {
        'extract_ids': extract_ids,
        'layers': layers,
        'checkpoint_name': os.path.basename(args.model),
        'tokenizer_type': tokenizer.tokenizer_type,
        'n_embd': int(model.config.n_embd),
        'n_layer': n_layer,
        'convention': (
            'residual stream after each block, pre-final-layernorm; '
            'layer 0 = post-embedding output (pre-block-0)'
        ),
        'centering': (
            'per-layer mean over all token positions of all extracts; '
            'subtract from every vector before cosine comparison'
        ),
    }
    arrays['_meta_json'] = np.array(json.dumps(meta, sort_keys=True))
    arrays['extract_ids'] = np.array(extract_ids)

    _probe_save_index(args.probe_index, arrays)
    print(f"Index saved: {args.probe_index}  ({n} extracts × {len(layers)} layers)")


def probe_match(model, tokenizer, device, args):
    """Match a query text against a pre-built index; report per-token and extract-level matches."""
    if not os.path.exists(args.probe_index):
        raise SystemExit(f"Index not found: {args.probe_index}")

    data = np.load(args.probe_index, allow_pickle=False)
    meta = json.loads(str(data['_meta_json']))

    # Validate checkpoint / tokenizer match.
    ckpt_name = os.path.basename(args.model)
    if meta['checkpoint_name'] != ckpt_name:
        raise SystemExit(
            f"Index/checkpoint mismatch:\n"
            f"  index built with: {meta['checkpoint_name']}\n"
            f"  current checkpoint: {ckpt_name}"
        )
    if meta['tokenizer_type'] != tokenizer.tokenizer_type:
        raise SystemExit(
            f"Index/tokenizer mismatch:\n"
            f"  index built with: {meta['tokenizer_type']}\n"
            f"  current tokenizer: {tokenizer.tokenizer_type}"
        )

    layers = meta['layers']
    extract_ids = [str(x) for x in data['extract_ids'].tolist()]
    n_extracts = len(extract_ids)
    block_size = model.config.block_size

    centering_means = {l: data[f'mean_{l}'] for l in layers}

    # Per-extract summary vectors (mean over token positions) for extract-level matching.
    summary_vecs = {
        l: np.stack([data[f'sum_{l}_{i}'] for i in range(n_extracts)])
        for l in layers
    }

    # Per-token reference vectors and token strings.
    tok_vecs = {
        l: [data[f'tok_{l}_{i}'] for i in range(n_extracts)]
        for l in layers
    }
    ref_token_lists = [[str(t) for t in data[f'tokens_{i}'].tolist()] for i in range(n_extracts)]

    # Build flat lookup: (ref_vector_row_index) -> (extract_idx, token_pos).
    # Computed once — positions are the same at every layer.
    ref_lookup = []
    for ext_i in range(n_extracts):
        for tok_pos in range(len(ref_token_lists[ext_i])):
            ref_lookup.append((ext_i, tok_pos))
    ref_lookup = np.array(ref_lookup, dtype=np.int32)  # (R, 2)

    # Query text.
    if args.probe_query_file:
        with open(args.probe_query_file, 'r', encoding='utf-8') as f:
            query_text = f.read().strip()
    elif args.probe_query:
        query_text = args.probe_query
    elif args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            query_text = f.read().strip()
    else:
        query_text = args.prompt
    if not query_text or query_text == '\n':
        raise SystemExit("Provide query text via --probe_query, --probe_query_file, or --prompt")

    query_ids = tokenizer.encode(query_text)
    if len(query_ids) > block_size:
        query_ids = query_ids[:block_size]
        print(f"Query truncated to {block_size} tokens")
    query_token_strs = [tokenizer.decode([t]) for t in query_ids]
    T_q = len(query_ids)

    q_hs = _probe_hidden_states(model, query_ids, device)  # list of (T_q, C)

    threshold = args.probe_threshold
    topk = args.probe_topk

    print(f"\n=== Probe match: {n_extracts} extracts, layers {layers} ===")
    print(f"Model:    {ckpt_name}")
    print(f"Query:    '{query_text[:100]}{'...' if len(query_text) > 100 else ''}'")
    print(f"Q tokens: {T_q} | threshold: {threshold} | top-k: {topk}")
    print()

    # best_per_token_layer[q_pos][layer_index] = (cos, ext_i, tok_pos, extract_id, ref_tok_str)
    best_per_token_layer = [[None] * len(layers) for _ in range(T_q)]
    console_lines = []

    for li, l in enumerate(layers):
        mean_l = centering_means[l]

        # Centered, unit-normalised reference matrix (R, C).
        all_ref = np.concatenate([tok_vecs[l][i] - mean_l for i in range(n_extracts)], axis=0)
        norms = np.linalg.norm(all_ref, axis=1, keepdims=True).clip(1e-10)
        all_ref_unit = all_ref / norms

        # Centered, unit-normalised query matrix (T_q, C).
        q_c = q_hs[l] - mean_l
        q_norms = np.linalg.norm(q_c, axis=1, keepdims=True).clip(1e-10)
        q_unit = q_c / q_norms

        cos_matrix = q_unit @ all_ref_unit.T  # (T_q, R)

        R = len(ref_lookup)
        actual_k = min(topk, R)

        for q_pos in range(T_q):
            cos_row = cos_matrix[q_pos]
            top_idx = np.argpartition(cos_row, -actual_k)[-actual_k:]
            top_idx = top_idx[np.argsort(cos_row[top_idx])[::-1]]

            best_r = top_idx[0]
            best_cos = float(cos_row[best_r])
            best_ext_i, best_tok_pos = int(ref_lookup[best_r, 0]), int(ref_lookup[best_r, 1])
            best_per_token_layer[q_pos][li] = (
                best_cos, best_ext_i, best_tok_pos,
                extract_ids[best_ext_i],
                ref_token_lists[best_ext_i][best_tok_pos],
            )

            for r in top_idx:
                cos = float(cos_row[r])
                if cos >= threshold:
                    ext_i, tok_pos = int(ref_lookup[r, 0]), int(ref_lookup[r, 1])
                    console_lines.append(
                        f"token {q_pos:3d} ({query_token_strs[q_pos]!r:8}) | "
                        f"layer {l:2d} | cos {cos:.3f} | "
                        f"extract {extract_ids[ext_i]}, "
                        f"token {tok_pos:3d} ({ref_token_lists[ext_i][tok_pos]!r})"
                    )

    for line in console_lines:
        print(line)
    if not console_lines:
        print(f"(No token-level matches above threshold {threshold})")

    # ── Extract-level summary ──────────────────────────────────────────────
    print("\n=== Extract-level top matches per layer ===")
    extract_level = {}
    for l in layers:
        mean_l = centering_means[l]
        q_mean = q_hs[l].mean(axis=0) - mean_l
        q_mean_norm = np.linalg.norm(q_mean).clip(1e-10)
        q_mean_unit = q_mean / q_mean_norm

        sims = []
        for ext_i in range(n_extracts):
            sv = summary_vecs[l][ext_i] - mean_l
            sv_norm = np.linalg.norm(sv).clip(1e-10)
            cos = float(np.dot(q_mean_unit, sv / sv_norm))
            sims.append((cos, extract_ids[ext_i]))
        sims.sort(key=lambda x: -x[0])
        extract_level[l] = sims
        top_str = '  '.join(f"{eid}({c:.3f})" for c, eid in sims[:topk])
        print(f"  layer {l:2d}: {top_str}")

    # ── Layer-divergence table ─────────────────────────────────────────────
    print("\n=== Per-token best-match identity across layers ===")
    header = f"{'tok':>4} {'str':>8}  " + "  ".join(f"L{l:02d}({best_per_token_layer[0][li][3][:6] if best_per_token_layer[0][li] else '?':6})" for li, l in enumerate(layers))
    # Simpler header:
    print(f"{'tok':>4} {'str':>8}  " + "  ".join(f"L{l:02d}  " for l in layers))
    print("-" * (16 + 7 * len(layers)))
    for q_pos in range(min(T_q, 60)):
        q_str = query_token_strs[q_pos]
        row = f"{q_pos:>4} {q_str!r:>8}  "
        prev_ext = None
        for li, l in enumerate(layers):
            m = best_per_token_layer[q_pos][li]
            if m is None:
                row += "  --   "
            else:
                cos, ext_i, _, _, _ = m
                marker = "*" if (prev_ext is not None and ext_i != prev_ext) else " "
                row += f"{marker}{ext_i:02d}/{cos:.2f}  "
                prev_ext = ext_i
        print(row)
    if T_q > 60:
        print(f"  ... ({T_q - 60} more tokens not shown)")

    # ── Summary file ──────────────────────────────────────────────────────
    if args.probe_output:
        _write_probe_summary(
            args.probe_output, query_text, query_token_strs, layers,
            extract_ids, extract_level, best_per_token_layer, topk, meta,
        )
        print(f"\nSummary written: {args.probe_output}")


def _write_probe_summary(path, query_text, query_token_strs, layers,
                          extract_ids, extract_level, best_per_token_layer,
                          topk, meta):
    lines = [
        "# Probe Match Summary",
        "",
        f"**Model:** `{meta['checkpoint_name']}`  ",
        f"**Tokenizer:** {meta['tokenizer_type']}  ",
        f"**Layers probed:** {layers}  ",
        f"**Convention:** {meta['convention']}  ",
        "",
        "## Query",
        "",
        "```",
        query_text,
        "```",
        "",
        "## Extract-level top matches per layer",
        "",
    ]
    for l in layers:
        top = extract_level[l][:topk]
        top_str = ", ".join(f"`{eid}` ({c:.3f})" for c, eid in top)
        lines.append(f"- **Layer {l}:** {top_str}")

    lines += [
        "",
        "## Per-token best-match identity across layers",
        "",
        "Each cell: extract index / cosine. `*` = best-match extract changed from the previous layer.",
        "",
        "| tok | str | " + " | ".join(f"L{l}" for l in layers) + " |",
        "|-----|-----|" + "|".join("---" for _ in layers) + "|",
    ]
    T_q = len(query_token_strs)
    for q_pos in range(min(T_q, 80)):
        parts = [str(q_pos), f"`{query_token_strs[q_pos]}`"]
        prev_ext = None
        for li, l in enumerate(layers):
            m = best_per_token_layer[q_pos][li]
            if m is None:
                parts.append("--")
            else:
                cos, ext_i, _, eid, _ = m
                mark = "★" if (prev_ext is not None and ext_i != prev_ext) else ""
                parts.append(f"{mark}{ext_i}({cos:.2f})")
                prev_ext = ext_i
        lines.append("| " + " | ".join(parts) + " |")
    if T_q > 80:
        lines.append(f"| ... | ({T_q-80} more) |" + "|".join("" for _ in layers) + "|")

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Sample from character-level GPT (supports linear attention and BPE)')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint (.pt file)')
    parser.add_argument('--prompt', type=str, default="\n", help='Starting prompt text')
    parser.add_argument('--prompt_file', type=str, help='File containing prompt text')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to generate')
    parser.add_argument('--max_tokens', type=int, default=300, help='Maximum new tokens per sample')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature (0=greedy)')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k filtering (0=disabled)')
    parser.add_argument('--rep_penalty', type=float, default=0.0,
                        help='Repetition penalty 0.0=off, 1.15=gentle, 1.3=aggressive')
    parser.add_argument('--stop_on_newline', action='store_true',
                        help='Stop generation at newline (default: generate past newlines)')
    parser.add_argument('--corpus', type=str, default=None,
                        help='Path to corpus file (one word per line) for validation marking')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: None=random each run)')

    # Sampling options
    parser.add_argument('--no_lowercase', action='store_true',
                        help='Do NOT lowercase the prompt (default: lowercase prompts)')
    parser.add_argument('--no_compile', action='store_true',
                        help='Do NOT use torch.compile() (default: try to compile)')
    parser.add_argument('--float16', action='store_true',
                        help='Use float16 precision (may not work on all devices)')
    parser.add_argument('--batch', action='store_true',
                        help='Use batched generation (faster for multiple samples, but no rep_penalty)')

    # Probe options
    parser.add_argument('--probe', choices=['build-index', 'match'], default=None,
                        help='Probe mode: build-index (index a corpus) or match (query an index)')
    parser.add_argument('--probe_corpus', type=str, default=None,
                        help='[build-index] Directory of .txt extract files')
    parser.add_argument('--probe_index', type=str, default='probe.idx',
                        help='Path to save (build-index) or load (match) the index file')
    parser.add_argument('--probe_layers', type=str, default=None,
                        help='Comma-separated layer indices to capture (default: every other layer incl. 0 and n_layer)')
    parser.add_argument('--probe_threshold', type=float, default=0.5,
                        help='[match] Cosine similarity threshold for reporting (default: 0.5)')
    parser.add_argument('--probe_topk', type=int, default=3,
                        help='[match] Top-k matches to keep per token per layer (default: 3)')
    parser.add_argument('--probe_query', type=str, default=None,
                        help='[match] Query text (alternative: --probe_query_file or --prompt)')
    parser.add_argument('--probe_query_file', type=str, default=None,
                        help='[match] File containing query text')
    parser.add_argument('--probe_output', type=str, default=None,
                        help='[match] Path to write markdown summary file')

    args = parser.parse_args()

    # Check model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        sys.exit(1)

    # Determine metadata file path
    # Handle regular, _iter{N}, and _final checkpoint names
    model_base = args.model.replace('.pt', '')
    if '_iter' in model_base:
        model_base = model_base.rsplit('_iter', 1)[0]
    elif model_base.endswith('_final'):
        model_base = model_base[:-6]
    meta_path = model_base + '_meta.pkl'
    if not os.path.exists(meta_path):
        print(f"Error: Metadata file '{meta_path}' not found")
        print("Make sure this model was trained with train.py")
        sys.exit(1)

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    else:
        import time
        seed = int(time.time() * 1000) % (2**32)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Device selection
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    # Determine dtype for float16 option
    if args.float16:
        if device == 'cpu':
            dtype = torch.float32
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32

    # Load model (suppress model.py's __init__ print)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    _stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    model = GPT(gptconf)
    sys.stdout = _stdout
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    if dtype == torch.float16:
        model = model.half()

    model.eval()

    # Skip torch.compile in probe mode: the dynamic return value (hidden states)
    # would require a recompile and may not work reliably with all backends.
    if not args.probe and not args.no_compile and device != 'mps':
        try:
            model = torch.compile(model)
        except Exception:
            pass

    # Load tokenizer
    tokenizer = load_tokenizer(meta_path)
    vocab_size = tokenizer.vocab_size
    tokenizer_type = tokenizer.tokenizer_type

    # Route to probe modes (no text generation).
    if args.probe == 'build-index':
        if not args.probe_corpus:
            raise SystemExit("--probe_corpus is required for --probe build-index")
        probe_build_index(model, tokenizer, device, args)
        return

    if args.probe == 'match':
        probe_match(model, tokenizer, device, args)
        return

    # ── Normal sampling ───────────────────────────────────────────────────

    case_preserved = detect_case_preservation(tokenizer)

    corpus_words = None
    if args.corpus:
        if os.path.exists(args.corpus):
            with open(args.corpus, 'r', encoding='utf-8') as f:
                corpus_words = set(word.strip() for word in f.read().strip().split('\n') if word.strip())

    stop_token_id = None
    if args.stop_on_newline:
        newline_ids = tokenizer.encode('\n')
        if newline_ids:
            stop_token_id = newline_ids[0]

    if args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
    else:
        prompt_text = args.prompt

    if not args.no_lowercase and not case_preserved:
        prompt_text = prompt_text.lower()
    prompt_text = prompt_text.rstrip(' ')

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    iter_num = checkpoint.get('iter_num', '?')
    best_val = checkpoint.get('best_val_loss')
    val_str = f", val loss: {best_val:.4f}" if best_val else ""
    attn_type = "linear" if model_args.get('use_linear_attention', False) else "softmax"
    case_str = ", case-preserved" if case_preserved else ""
    print(f"params: {n_params:.0f}M, attention: {attn_type}, tokenizer: {tokenizer_type} (vocab: {vocab_size}{case_str}), iter: {iter_num}{val_str}")
    settings = f"temp: {args.temperature}, top_k: {args.top_k}"
    if args.rep_penalty > 0:
        settings += f", rep_penalty: {args.rep_penalty}"
    print(f"prompt: '{prompt_text[:60]}{'...' if len(prompt_text) > 60 else ''}' | {settings}")
    if args.batch and args.rep_penalty > 0:
        print("Note: rep_penalty ignored in batched mode")
    print()

    prompt_ids = tokenizer.encode(prompt_text)
    prompt_length = len(prompt_ids)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]

    if args.batch:
        y_batch = generate_batched(model, x, args.num_samples, args.max_tokens,
                                   temperature=args.temperature,
                                   top_k=args.top_k if args.top_k > 0 else None,
                                   device=device)
        for i in range(args.num_samples):
            tokens = y_batch[i].tolist()
            tokens = truncate_at_stop_token(tokens, stop_token_id, prompt_length)
            generated_text = tokenizer.decode(tokens)
            generated_text = generated_text.replace('\n', ' ')
            if not case_preserved:
                generated_text = capitalize_sentences(generated_text)
            if corpus_words is not None and args.stop_on_newline:
                word = generated_text.strip()
                generated_text = word + ' *' if word in corpus_words else word
            print(f"  [{i+1}] {generated_text}\n")
    else:
        for i in range(args.num_samples):
            y = generate_local(model, x, args.max_tokens,
                              temperature=args.temperature,
                              top_k=args.top_k if args.top_k > 0 else None,
                              rep_penalty=args.rep_penalty,
                              device=device,
                              stop_token_id=stop_token_id)
            generated_text = tokenizer.decode(y[0].tolist())
            generated_text = generated_text.replace('\n', ' ')
            if not case_preserved:
                generated_text = capitalize_sentences(generated_text)
            if corpus_words is not None and args.stop_on_newline:
                word = generated_text.strip()
                generated_text = word + ' *' if word in corpus_words else word
            print(f"  [{i+1}] {generated_text}\n")


if __name__ == '__main__':
    main()
