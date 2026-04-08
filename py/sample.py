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
import argparse
import pickle
import torch
from contextlib import nullcontext
from model import GPTConfig, GPT
from tokenizer import load_tokenizer


def apply_repetition_penalty(logits, generated_ids, penalty=1.2, window=50):
    """
    Reduce repetition by penalizing recently used tokens.

    Args:
        logits: Raw model outputs [vocab_size]
        generated_ids: List of token IDs generated so far
        penalty: Multiplicative penalty (>1.0 suppresses repetition)
        window: How many recent tokens to consider
    """
    # Look at recent tokens (or all if sequence is short)
    lookback = min(len(generated_ids), window)
    if lookback == 0:
        return logits

    recent_tokens = generated_ids[-lookback:]

    # Penalize each recent token proportional to its frequency
    for token_id in set(recent_tokens):
        count = recent_tokens.count(token_id)
        # Apply graduated penalty based on frequency
        logits[token_id] = logits[token_id] / (penalty ** count)

    return logits


@torch.no_grad()
def generate_local(model, x_init, max_new_tokens, temperature=1.0, top_k=None, rep_penalty=1.0, device='cpu', stop_token_id=None):
    """
    Generate text from initial prompt with repetition penalty
    x_init: (1, T) prompt indices
    stop_token_id: Optional token ID to stop generation (e.g., newline)
    Returns: (1, T+max_new_tokens)
    """
    x = x_init
    block_size = model.config.block_size
    generated_ids = []  # Track generated tokens for repetition penalty

    for _ in range(max_new_tokens):
        # Crop context if needed
        if x.size(1) > block_size:
            idx_cond = x[:, -block_size:]
        else:
            idx_cond = x

        # Get model prediction
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (1, vocab_size)

        # Apply repetition penalty if enabled
        if rep_penalty > 1.0 and len(generated_ids) > 0:
            logits = apply_repetition_penalty(
                logits.squeeze(0),
                generated_ids,
                penalty=rep_penalty,
                window=500
            ).unsqueeze(0)

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float('inf')

        # Apply temperature and sample
        if temperature <= 0.0:
            # Greedy decoding
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # Sample with temperature
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        # Track generated token for repetition penalty
        generated_ids.append(idx_next.item())

        # Check for stop token
        if stop_token_id is not None and idx_next.item() == stop_token_id:
            break

        x = torch.cat((x, idx_next), dim=1)

    return x


@torch.no_grad()
def generate_batched(model, x_init, num_samples, max_new_tokens, temperature=1.0, top_k=None, device='cpu'):
    """
    Generate multiple samples in parallel (batched).

    x_init: (1, T) prompt indices
    num_samples: number of samples to generate in parallel
    max_new_tokens: maximum tokens to generate per sample

    Returns: (num_samples, T+max_new_tokens)

    Note: Does not support early stopping or repetition penalty in batched mode.
    Stop tokens are handled after generation by truncating output.
    """
    # Repeat prompt for all samples: (1, T) -> (num_samples, T)
    x = x_init.repeat(num_samples, 1)
    block_size = model.config.block_size

    for _ in range(max_new_tokens):
        # Crop context if needed
        if x.size(1) > block_size:
            idx_cond = x[:, -block_size:]
        else:
            idx_cond = x

        # Get model prediction for all samples at once
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (num_samples, vocab_size)

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float('inf')

        # Apply temperature and sample
        if temperature <= 0.0:
            # Greedy decoding
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # Sample with temperature
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        x = torch.cat((x, idx_next), dim=1)

    return x


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
    parser.add_argument('--no_stop_on_newline', action='store_true',
                        help='Continue past newline instead of stopping (default: stop at newline)')
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
        print(f"Using seed: {args.seed}")
    else:
        # Use random seed (from system time/entropy)
        import time
        seed = int(time.time() * 1000) % (2**32)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        print(f"Using random seed: {seed}")

    # Device selection
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Determine dtype for float16 option
    if args.float16:
        if device == 'cuda':
            dtype = torch.float16
            print("Using float16 precision (CUDA)")
        elif device == 'mps':
            # MPS has limited float16 support - try it but warn
            dtype = torch.float16
            print("Using float16 precision (MPS - experimental, may not work)")
        else:
            print("Warning: float16 not supported on CPU, using float32")
            dtype = torch.float32
    else:
        dtype = torch.float32
        print("Using float32 precision")

    # Load model
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    # Initialize model
    model_args = checkpoint['model_args']

    # Check attention type
    use_linear = model_args.get('use_linear_attention', False)
    attn_type = "linear" if use_linear else "softmax"
    print(f"Attention type: {attn_type}")

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # Convert to float16 if requested
    if dtype == torch.float16:
        model = model.half()
        print("Model converted to float16")

    model.eval()

    # Try torch.compile() for speedup (skip on MPS - not supported)
    if not args.no_compile and device != 'mps':
        try:
            print("Attempting torch.compile()...")
            model = torch.compile(model)
            print("torch.compile() succeeded")
        except Exception as e:
            print(f"torch.compile() failed (this is OK, continuing without): {e}")
    elif device == 'mps' and not args.no_compile:
        print("Skipping torch.compile() (not supported on MPS)")

    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Load tokenizer
    print(f"Loading tokenizer from {meta_path}")
    tokenizer = load_tokenizer(meta_path)

    vocab_size = tokenizer.vocab_size
    tokenizer_type = tokenizer.tokenizer_type
    print(f"Tokenizer: {tokenizer_type}")
    print(f"Vocabulary size: {vocab_size} tokens")

    # Load corpus for validation if provided
    corpus_words = None
    if args.corpus:
        if not os.path.exists(args.corpus):
            print(f"Warning: Corpus file '{args.corpus}' not found, skipping validation")
        else:
            with open(args.corpus, 'r', encoding='utf-8') as f:
                corpus_words = set(word.strip() for word in f.read().strip().split('\n') if word.strip())
            print(f"Loaded corpus: {len(corpus_words)} unique words")

    # Determine newline token ID for stopping (default: stop at newline)
    stop_token_id = None
    if not args.no_stop_on_newline:
        newline_ids = tokenizer.encode('\n')
        if newline_ids:
            stop_token_id = newline_ids[0]
            print(f"Stop token ID (newline): {stop_token_id}")
        else:
            print("Warning: Could not find newline token in vocabulary")

    # Get prompt
    if args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
    else:
        prompt_text = args.prompt

    # Lowercase prompt for lowercase-only vocabularies (unless disabled)
    if not args.no_lowercase:
        original_prompt = prompt_text
        prompt_text = prompt_text.lower()
        if original_prompt != prompt_text:
            print(f"Prompt lowercased: '{original_prompt}' -> '{prompt_text}'")

    print(f"\nPrompt: '{prompt_text[:50]}{'...' if len(prompt_text) > 50 else ''}'")
    print(f"Generating {args.num_samples} samples...")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Rep-penalty: {args.rep_penalty}")
    if args.batch:
        print("Mode: BATCHED (all samples generated in parallel)")
        if args.rep_penalty > 0:
            print("Warning: Repetition penalty ignored in batched mode")
    else:
        print("Mode: SEQUENTIAL (one sample at a time)")
    print("=" * 70)

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

        # Decode and print each sample
        for i in range(args.num_samples):
            tokens = y_batch[i].tolist()

            # Truncate at stop token if needed
            tokens = truncate_at_stop_token(tokens, stop_token_id, prompt_length)

            generated_text = tokenizer.decode(tokens)

            # Capitalize first letter for readability
            if generated_text:
                generated_text = generated_text[0].upper() + generated_text[1:] if len(generated_text) > 1 else generated_text.upper()

            # If corpus validation is enabled and we're generating single words
            if corpus_words is not None and not args.no_stop_on_newline:
                word = generated_text.strip()
                if word in corpus_words:
                    generated_text = word + ' *'
                else:
                    generated_text = word

            print(generated_text)
    else:
        # Sequential generation - one sample at a time
        for i in range(args.num_samples):
            # Generate with repetition penalty
            y = generate_local(model, x, args.max_tokens,
                              temperature=args.temperature,
                              top_k=args.top_k if args.top_k > 0 else None,
                              rep_penalty=args.rep_penalty,
                              device=device,
                              stop_token_id=stop_token_id)

            # Decode and print (one line per sample)
            generated_text = tokenizer.decode(y[0].tolist())

            # Capitalize first letter for readability
            if generated_text:
                generated_text = generated_text[0].upper() + generated_text[1:] if len(generated_text) > 1 else generated_text.upper()

            # If corpus validation is enabled and we're generating single words
            if corpus_words is not None and not args.no_stop_on_newline:
                word = generated_text.strip()
                if word in corpus_words:
                    generated_text = word + ' *'
                else:
                    generated_text = word

            print(generated_text)

    print("\n" + "=" * 70)

    if checkpoint.get('iter_num'):
        print(f"Model trained for {checkpoint['iter_num']} iterations")
    if checkpoint.get('best_val_loss'):
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")


if __name__ == '__main__':
    main()
