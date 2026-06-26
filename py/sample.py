#!/usr/bin/env python
"""
sample.py - Sampling from models trained with train.py
Works with both standard softmax attention and linear attention models.
Supports both character-level and BPE tokenization.

Features:
- Lowercases prompts automatically (for lowercase-only vocabularies)
- Float16 support via --float16 flag
- Batched generation via --batch flag (faster for multiple samples)

Usage: python sample.py --model model.pt --prompt "The Roman" --batch
"""

import os
import sys
import re
import argparse
import pickle
import torch
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

    # New options
    parser.add_argument('--no_lowercase', action='store_true',
                        help='Do NOT lowercase the prompt (default: lowercase prompts)')
    parser.add_argument('--no_compile', action='store_true',
                        help='Do NOT use torch.compile() (default: try to compile)')
    parser.add_argument('--float16', action='store_true',
                        help='Use float16 precision (may not work on all devices)')
    parser.add_argument('--batch', action='store_true',
                        help='Use batched generation (faster for multiple samples, but no rep_penalty)')

    args = parser.parse_args()

    # Check model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        sys.exit(1)

    # Determine metadata file path
    # Handle regular, _iter{N}, and _final checkpoint names
    model_base = args.model.replace('.pt', '')
    if '_iter' in model_base:
        # Strip _iter{N} suffix to get base name
        model_base = model_base.rsplit('_iter', 1)[0]
    elif model_base.endswith('_final'):
        # Strip _final suffix
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

    # Try torch.compile() for speedup (skip on MPS - not supported)
    if not args.no_compile and device != 'mps':
        try:
            model = torch.compile(model)
        except Exception:
            pass

    # Load tokenizer
    tokenizer = load_tokenizer(meta_path)
    vocab_size = tokenizer.vocab_size
    tokenizer_type = tokenizer.tokenizer_type

    # Auto-detect whether the model was trained on case-preserved data.
    # If so, do NOT lowercase the prompt, and do NOT apply post-hoc
    # capitalize_sentences (the model's own output already has correct
    # case). Either step would degrade output for a case-preserved model.
    case_preserved = detect_case_preservation(tokenizer)

    # Load corpus for validation if provided
    corpus_words = None
    if args.corpus:
        if os.path.exists(args.corpus):
            with open(args.corpus, 'r', encoding='utf-8') as f:
                corpus_words = set(word.strip() for word in f.read().strip().split('\n') if word.strip())

    # Determine newline token ID for stopping (default: don't stop at newline)
    stop_token_id = None
    if args.stop_on_newline:
        newline_ids = tokenizer.encode('\n')
        if newline_ids:
            stop_token_id = newline_ids[0]

    # Get prompt
    if args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
    else:
        prompt_text = args.prompt

    # Lowercase prompt for lowercase-only vocabularies, unless either
    # (a) the user passed --no_lowercase, or
    # (b) auto-detected the tokenizer is case-preserving.
    if not args.no_lowercase and not case_preserved:
        prompt_text = prompt_text.lower()

    # Strip trailing spaces — BPE attaches spaces to the front of the
    # next word, so a trailing space creates an unnatural token sequence
    # that produces garbled output.
    prompt_text = prompt_text.rstrip(' ')

    # Print compact header
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

    # Encode prompt
    prompt_ids = tokenizer.encode(prompt_text)
    prompt_length = len(prompt_ids)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]

    if args.batch:
        # Batched generation - all samples at once
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
        # Sequential generation - one sample at a time
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
