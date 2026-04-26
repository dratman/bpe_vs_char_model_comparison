#!/usr/bin/env python3
"""
sample_imitator.py

Evaluate and sample from a trained imitator model.

Three modes of operation:

1. COMPARE (default): Take real text, run it through the frozen model's
   first N layers, then compare:
   - What the frozen model predicts (using the REAL layer-N vectors)
   - What the imitator predicts (using the imitator's PREDICTED vectors
     decoded through the frozen model's back half)
   This shows how faithfully the imitator reproduces the frozen model's
   behavior.

2. ROLLOUT: Start with a real text prefix, get the layer-N vectors for
   that prefix, then let the imitator run autoregressively — feeding
   its own predicted vectors back as input, one step at a time. At each
   step, decode the predicted vector through the frozen model's back
   half to get a token. This generates text from the imitator's learned
   model of mid-layer dynamics.

3. STATS: Run many batches and report aggregate statistics on how well
   the imitator's decoded tokens match the frozen model's decoded tokens.

Usage:
    # Compare mode (default) — show real vs imitator predictions side by side
    python py/sample_imitator.py \\
        --imitator pt/imitator_L8.pt \\
        --small_model pt/bpe_16L16H_iter170000.pt \\
        --input txt_local/corpus_books_shuffled_2026_04_18.txt \\
        --mode compare --num_samples 5

    # Rollout mode — let the imitator generate text autoregressively
    python py/sample_imitator.py \\
        --imitator pt/imitator_L8.pt \\
        --small_model pt/bpe_16L16H_iter170000.pt \\
        --input txt_local/corpus_books_shuffled_2026_04_18.txt \\
        --mode rollout --rollout_tokens 200

    # Stats mode — aggregate accuracy over many batches
    python py/sample_imitator.py \\
        --imitator pt/imitator_L8.pt \\
        --small_model pt/bpe_16L16H_iter170000.pt \\
        --input txt_local/corpus_books_shuffled_2026_04_18.txt \\
        --mode stats --num_batches 50
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

from imitator_model import Imitator, ImitatorConfig
from small_model_split import load_small_model, forward_to_layer, forward_from_layer
from tokenizer import load_tokenizer


def load_imitator(imitator_path, device):
    """Load a trained imitator from checkpoint."""
    print(f"Loading imitator from {imitator_path}")
    ckpt = torch.load(imitator_path, map_location=device, weights_only=False)
    config = ImitatorConfig(**ckpt['imitator_config'])
    imitator = Imitator(config).to(device)
    imitator.load_state_dict(ckpt['imitator'])
    imitator.eval()
    split_layer = ckpt['split_layer']
    print(f"Imitator: d_model={config.d_model}, n_layer={config.n_layer}, "
          f"split_layer={split_layer}, iter={ckpt.get('iter_num', '?')}, "
          f"best_val_loss={ckpt.get('best_val_loss', '?')}")
    return imitator, config, split_layer


def load_corpus_tokens(input_path, tokenizer, token_cache=None):
    """Load or tokenize corpus, return 1-D long tensor."""
    if token_cache and os.path.exists(token_cache):
        print(f"Loading cached tokens from {token_cache}")
        return torch.load(token_cache, map_location='cpu', weights_only=True)
    print(f"Tokenizing {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    ids = tokenizer.encode(text)
    tokens = torch.tensor(ids, dtype=torch.long)
    if token_cache:
        torch.save(tokens, token_cache)
    return tokens


def get_random_window(tokens, window_size, device):
    """Sample a single random window of token IDs."""
    start = torch.randint(len(tokens) - window_size, (1,)).item()
    return tokens[start:start + window_size].unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# COMPARE mode
# ---------------------------------------------------------------------------

@torch.no_grad()
def compare_predictions(imitator, small_model, tokenizer, tokens,
                        split_layer, block_size, device, num_samples=5):
    """
    For random text windows, show side-by-side:
    - The actual input text
    - The frozen model's next-token predictions (from real layer-N vectors)
    - The imitator's next-token predictions (imitator vectors decoded
      through the frozen model's back half)
    - Per-position cosine similarity between real and predicted vectors
    """
    imitator_dtype = next(imitator.parameters()).dtype

    for sample_idx in range(num_samples):
        # Sample a window of tokens
        idx = get_random_window(tokens, block_size, device)  # (1, T)

        # Get real layer-N activations
        real_acts = forward_to_layer(small_model, idx, split_layer)  # (1, T, D)

        # Imitator predicts next vectors from positions [0..T-2]
        inp = real_acts[:, :-1, :].to(imitator_dtype)  # (1, T-1, D)
        pred_acts = imitator(inp)  # (1, T-1, D)

        # Real target vectors at positions [1..T-1]
        target_acts = real_acts[:, 1:, :]  # (1, T-1, D)

        # Per-position cosine similarity
        cos_sim = F.cosine_similarity(
            pred_acts.float(), target_acts.float(), dim=-1
        )[0]  # (T-1,)

        # Decode FULL real activation sequence through the back half.
        # Using all T positions preserves correct causal context — each
        # position can attend to all positions before it, including position 0.
        real_logits_full = forward_from_layer(
            small_model, real_acts, split_layer
        )  # (1, T, vocab)
        # Positions 1..T-1 predict tokens 2..T, so slice accordingly
        real_logits = real_logits_full[:, 1:, :]  # (1, T-1, vocab)
        real_tokens = real_logits[0].argmax(dim=-1)  # (T-1,)

        # Decode imitator vectors through the back half.
        # Prepend the real activation at position 0 so the back half has
        # correct causal context starting from the beginning. The result
        # is a mixed sequence: [real_pos_0, imitator_pred_for_pos_1,
        # imitator_pred_for_pos_2, ...].
        mixed = torch.cat(
            [real_acts[:, 0:1, :], pred_acts.to(real_acts.dtype)], dim=1
        )  # (1, T, D)
        pred_logits_full = forward_from_layer(
            small_model, mixed, split_layer
        )  # (1, T, vocab)
        pred_logits = pred_logits_full[:, 1:, :]  # (1, T-1, vocab)
        pred_tokens = pred_logits[0].argmax(dim=-1)  # (T-1,)

        # The actual tokens at positions 2..T (what positions 1..T-1 predict)
        actual_tokens = idx[0, 2:]  # (T-2,)

        # Token-level match rates
        match = (pred_tokens == real_tokens).float().mean().item()
        # For actual-token comparison, align lengths (T-2 positions)
        match_actual = (pred_tokens[:-1] == actual_tokens).float().mean().item()
        # Also compute the frozen model's own accuracy against actual tokens
        frozen_actual = (real_tokens[:-1] == actual_tokens).float().mean().item()

        # Decode to text for display
        input_text = tokenizer.decode(idx[0].tolist())
        real_decoded = tokenizer.decode(real_tokens.tolist())
        pred_decoded = tokenizer.decode(pred_tokens.tolist())

        # Show first N characters
        show_chars = 300
        print(f"\n{'='*70}")
        print(f"Sample {sample_idx + 1}")
        print(f"{'='*70}")
        print(f"\nActual input text (first {show_chars} chars):")
        print(f"  {input_text[:show_chars]}")
        print(f"\nFrozen model's predictions (decoded from real layer-{split_layer} vectors):")
        print(f"  {real_decoded[:show_chars]}")
        print(f"\nImitator's predictions (decoded from predicted layer-{split_layer} vectors):")
        print(f"  {pred_decoded[:show_chars]}")
        print(f"\nStatistics:")
        print(f"  Mean cosine similarity (real vs predicted vectors): {cos_sim.mean().item():.4f}")
        print(f"  Min cosine similarity:  {cos_sim.min().item():.4f}")
        print(f"  Max cosine similarity:  {cos_sim.max().item():.4f}")
        print(f"  Frozen model top-1 matches actual next token: {frozen_actual*100:.1f}%")
        print(f"  Imitator top-1 matches frozen model top-1:   {match*100:.1f}%")
        print(f"  Imitator top-1 matches actual next token:    {match_actual*100:.1f}%")

        # Show some positions where they disagree
        disagree = (pred_tokens != real_tokens).nonzero(as_tuple=True)[0]
        if len(disagree) > 0:
            print(f"\n  First few disagreements (position: frozen->imitator):")
            for j in disagree[:10]:
                j = j.item()
                context = tokenizer.decode(idx[0, max(0,j-3):j+1].tolist())
                real_tok = tokenizer.decode([real_tokens[j].item()])
                pred_tok = tokenizer.decode([pred_tokens[j].item()])
                print(f"    pos {j:3d}: cos={cos_sim[j]:.3f}  "
                      f"frozen='{real_tok}'  imitator='{pred_tok}'  "
                      f"after '...{context}'")


@torch.no_grad()
def compare_prompt(imitator, small_model, tokenizer, prompt_text,
                   split_layer, block_size, device):
    """
    Run the compare analysis on a specific text passage rather than
    random corpus windows.
    """
    imitator_dtype = next(imitator.parameters()).dtype

    # Tokenize the prompt
    token_ids = tokenizer.encode(prompt_text)
    if len(token_ids) > block_size:
        print(f"Prompt has {len(token_ids)} tokens, truncating to {block_size}")
        token_ids = token_ids[:block_size]
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)  # (1, T)
    T = idx.size(1)

    # Get real layer-N activations
    real_acts = forward_to_layer(small_model, idx, split_layer)  # (1, T, D)

    # Imitator predicts next vectors from positions [0..T-2]
    inp = real_acts[:, :-1, :].to(imitator_dtype)  # (1, T-1, D)
    pred_acts = imitator(inp)  # (1, T-1, D)

    # Real target vectors at positions [1..T-1]
    target_acts = real_acts[:, 1:, :]  # (1, T-1, D)

    # Per-position cosine similarity
    cos_sim = F.cosine_similarity(
        pred_acts.float(), target_acts.float(), dim=-1
    )[0]  # (T-1,)

    # Decode FULL real activation sequence through back half
    real_logits_full = forward_from_layer(
        small_model, real_acts, split_layer
    )  # (1, T, vocab)
    real_logits = real_logits_full[:, 1:, :]  # (1, T-1, vocab)
    real_tokens = real_logits[0].argmax(dim=-1)  # (T-1,)

    # Decode imitator vectors with real position 0 prepended
    mixed = torch.cat(
        [real_acts[:, 0:1, :], pred_acts.to(real_acts.dtype)], dim=1
    )  # (1, T, D)
    pred_logits_full = forward_from_layer(
        small_model, mixed, split_layer
    )  # (1, T, vocab)
    pred_logits = pred_logits_full[:, 1:, :]  # (1, T-1, vocab)
    pred_tokens = pred_logits[0].argmax(dim=-1)  # (T-1,)

    # Actual tokens at positions 2..T (what positions 1..T-1 predict)
    actual_tokens = idx[0, 2:]  # (T-2,)

    # Match rates
    match = (pred_tokens == real_tokens).float().mean().item()
    match_actual = (pred_tokens[:-1] == actual_tokens).float().mean().item()
    frozen_actual = (real_tokens[:-1] == actual_tokens).float().mean().item()

    # Decode to text
    input_text = tokenizer.decode(idx[0].tolist())
    real_decoded = tokenizer.decode(real_tokens.tolist())
    pred_decoded = tokenizer.decode(pred_tokens.tolist())

    show_chars = 500
    print(f"\n{'='*70}")
    print(f"Prompt comparison ({T} tokens)")
    print(f"{'='*70}")
    print(f"\nActual input text (first {show_chars} chars):")
    print(f"  {input_text[:show_chars]}")
    print(f"\nFrozen model's predictions (decoded from real layer-{split_layer} vectors):")
    print(f"  {real_decoded[:show_chars]}")
    print(f"\nImitator's predictions (decoded from predicted layer-{split_layer} vectors):")
    print(f"  {pred_decoded[:show_chars]}")
    print(f"\nStatistics:")
    print(f"  Mean cosine similarity (real vs predicted vectors): {cos_sim.mean().item():.4f}")
    print(f"  Min cosine similarity:  {cos_sim.min().item():.4f}")
    print(f"  Max cosine similarity:  {cos_sim.max().item():.4f}")
    print(f"  Frozen model top-1 matches actual next token: {frozen_actual*100:.1f}%")
    print(f"  Imitator top-1 matches frozen model top-1:   {match*100:.1f}%")
    print(f"  Imitator top-1 matches actual next token:    {match_actual*100:.1f}%")

    # Show disagreements
    disagree = (pred_tokens != real_tokens).nonzero(as_tuple=True)[0]
    if len(disagree) > 0:
        print(f"\n  First few disagreements (position: frozen->imitator):")
        for j in disagree[:10]:
            j = j.item()
            context = tokenizer.decode(idx[0, max(0,j-3):j+1].tolist())
            real_tok = tokenizer.decode([real_tokens[j].item()])
            pred_tok = tokenizer.decode([pred_tokens[j].item()])
            print(f"    pos {j:3d}: cos={cos_sim[j]:.3f}  "
                  f"frozen='{real_tok}'  imitator='{pred_tok}'  "
                  f"after '...{context}'")


# ---------------------------------------------------------------------------
# ROLLOUT mode
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout(imitator, small_model, tokenizer, tokens,
            split_layer, block_size, device,
            prefix_tokens=64, rollout_tokens=200, num_samples=3):
    """
    Start with real text, get layer-N vectors for a prefix, then let the
    imitator generate autoregressively in vector space. Decode each
    predicted vector through the frozen model's back half to get tokens.
    """
    imitator_dtype = next(imitator.parameters()).dtype

    for sample_idx in range(num_samples):
        # Sample a window long enough for the prefix
        idx = get_random_window(tokens, prefix_tokens + 1, device)  # (1, prefix+1)

        # Get real layer-N activations for the prefix
        real_acts = forward_to_layer(small_model, idx, split_layer)  # (1, prefix+1, D)

        # Start the imitator with real vectors for the prefix
        # We use positions [0..prefix-1] as context
        vec_sequence = real_acts[:, :prefix_tokens, :].to(imitator_dtype)  # (1, prefix, D)

        # Decode the prefix to show what text we're starting from
        prefix_text = tokenizer.decode(idx[0, :prefix_tokens].tolist())

        generated_tokens = []

        for step in range(rollout_tokens):
            # Use at most block_size vectors as context for the imitator
            if vec_sequence.size(1) > block_size:
                context = vec_sequence[:, -block_size:, :]
            else:
                context = vec_sequence

            # Imitator predicts the next vector
            pred_all = imitator(context)          # (1, T, D)
            pred_vec = pred_all[:, -1:, :]        # (1, 1, D)

            # Decode this vector through the frozen model's back half
            logits = forward_from_layer(
                small_model, pred_vec.to(real_acts.dtype), split_layer
            )  # (1, 1, vocab)
            token_id = logits[0, 0].argmax().item()
            generated_tokens.append(token_id)

            # Append the predicted vector to the sequence for the next step
            vec_sequence = torch.cat([vec_sequence, pred_vec], dim=1)

        generated_text = tokenizer.decode(generated_tokens)

        print(f"\n{'='*70}")
        print(f"Rollout sample {sample_idx + 1}")
        print(f"{'='*70}")
        print(f"\nPrefix (real text, {prefix_tokens} tokens):")
        print(f"  {prefix_text[:300]}")
        print(f"\nImitator rollout ({rollout_tokens} tokens, autoregressive in vector space):")
        print(f"  {generated_text[:500]}")


# ---------------------------------------------------------------------------
# STATS mode
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_stats(imitator, small_model, tokenizer, tokens,
                  split_layer, block_size, batch_size, device, num_batches=50):
    """
    Run many batches and report aggregate statistics on how well the
    imitator's decoded tokens match the frozen model's decoded tokens.
    """
    imitator_dtype = next(imitator.parameters()).dtype

    all_cos_sims = []
    all_top1_match = []
    all_top5_match = []
    all_actual_match = []
    all_frozen_actual = []

    for batch_idx in range(num_batches):
        # Sample a batch
        ix = torch.randint(len(tokens) - block_size, (batch_size,))
        idx = torch.stack([tokens[i:i + block_size] for i in ix]).to(device)

        # Real activations
        real_acts = forward_to_layer(small_model, idx, split_layer)

        # Imitator predictions
        inp = real_acts[:, :-1, :].to(imitator_dtype)
        pred_acts = imitator(inp)
        target_acts = real_acts[:, 1:, :]

        # Cosine similarity
        cos_sim = F.cosine_similarity(
            pred_acts.float(), target_acts.float(), dim=-1
        )
        all_cos_sims.append(cos_sim.mean().item())

        # Decode FULL real activations through back half (correct causal context)
        real_logits_full = forward_from_layer(small_model, real_acts, split_layer)
        real_logits = real_logits_full[:, 1:, :]  # positions 1..T-1

        # Decode imitator predictions with real position 0 prepended
        mixed = torch.cat(
            [real_acts[:, 0:1, :], pred_acts.to(real_acts.dtype)], dim=1
        )
        pred_logits_full = forward_from_layer(small_model, mixed, split_layer)
        pred_logits = pred_logits_full[:, 1:, :]  # positions 1..T-1

        real_top1 = real_logits.argmax(dim=-1)     # (B, T-1)
        pred_top1 = pred_logits.argmax(dim=-1)     # (B, T-1)
        actual_next = idx[:, 2:]                    # (B, T-2) — tokens at positions 2..T-1

        # Top-1 match: imitator vs frozen model
        all_top1_match.append((pred_top1 == real_top1).float().mean().item())

        # Top-1 match: imitator vs actual corpus tokens (aligned to T-2)
        all_actual_match.append(
            (pred_top1[:, :-1] == actual_next).float().mean().item()
        )

        # Frozen model's own accuracy vs actual corpus tokens
        all_frozen_actual.append(
            (real_top1[:, :-1] == actual_next).float().mean().item()
        )

        # Top-5 match: is the frozen model's top-1 in the imitator's top 5?
        pred_top5 = pred_logits.topk(5, dim=-1).indices  # (B, T-1, 5)
        real_top1_expanded = real_top1.unsqueeze(-1)       # (B, T-1, 1)
        top5_hit = (pred_top5 == real_top1_expanded).any(dim=-1).float()
        all_top5_match.append(top5_hit.mean().item())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches}")

    print(f"\n{'='*70}")
    print(f"Aggregate statistics over {num_batches} batches "
          f"(batch_size={batch_size}, block_size={block_size})")
    print(f"{'='*70}")
    print(f"  Mean cosine similarity (real vs predicted vectors): "
          f"{np.mean(all_cos_sims):.4f} +/- {np.std(all_cos_sims):.4f}")
    print(f"  Frozen model top-1 matches actual corpus token: "
          f"{np.mean(all_frozen_actual)*100:.1f}%")
    print(f"  Imitator top-1 matches frozen model top-1:     "
          f"{np.mean(all_top1_match)*100:.1f}%")
    print(f"  Frozen model top-1 in imitator's top-5:        "
          f"{np.mean(all_top5_match)*100:.1f}%")
    print(f"  Imitator top-1 matches actual corpus token:    "
          f"{np.mean(all_actual_match)*100:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Evaluate and sample from a trained imitator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument('--imitator', type=str, required=True,
                   help='Path to trained imitator checkpoint')
    p.add_argument('--small_model', type=str, required=True,
                   help='Path to frozen small GPT checkpoint')
    p.add_argument('--input', type=str, required=True,
                   help='Corpus text file')
    p.add_argument('--token_cache', type=str, default=None,
                   help='Token cache file (from training)')
    p.add_argument('--mode', type=str, default='compare',
                   choices=['compare', 'rollout', 'stats'],
                   help='Evaluation mode')

    # Compare mode options
    p.add_argument('--num_samples', type=int, default=5,
                   help='Number of samples to show (compare/rollout modes)')
    p.add_argument('--prompt', type=str, default=None,
                   help='Specific text to analyze (compare mode; skips random sampling)')
    p.add_argument('--prompt_file', type=str, default=None,
                   help='File containing text to analyze (compare mode)')

    # Rollout mode options
    p.add_argument('--prefix_tokens', type=int, default=64,
                   help='Number of real tokens to use as prefix before rollout')
    p.add_argument('--rollout_tokens', type=int, default=200,
                   help='Number of tokens to generate in rollout')

    # Stats mode options
    p.add_argument('--num_batches', type=int, default=50,
                   help='Number of batches for stats mode')
    p.add_argument('--batch_size', type=int, default=8,
                   help='Batch size for stats mode')

    p.add_argument('--seed', type=int, default=None,
                   help='Random seed (default: random)')

    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = ('mps' if torch.backends.mps.is_available()
              else 'cuda' if torch.cuda.is_available()
              else 'cpu')
    print(f"Device: {device}")

    # Load frozen small model in bfloat16
    small_model, small_args = load_small_model(
        args.small_model, device=device, dtype=torch.bfloat16,
    )
    split_layer_from_model = small_args['n_layer'] // 2  # fallback

    # Load imitator
    imitator, imit_config, split_layer = load_imitator(args.imitator, device)

    # Load tokenizer
    model_base = args.small_model.replace('.pt', '')
    if '_iter' in model_base:
        model_base = model_base.rsplit('_iter', 1)[0]
    elif model_base.endswith('_final'):
        model_base = model_base[:-6]
    meta_path = model_base + '_meta.pkl'
    tokenizer = load_tokenizer(meta_path)
    print(f"Tokenizer: {tokenizer.tokenizer_type}, vocab={tokenizer.vocab_size}")

    # Load corpus tokens
    all_tokens = load_corpus_tokens(args.input, tokenizer, args.token_cache)
    # Use validation split (last 10%)
    n_val = int(len(all_tokens) * 0.1)
    val_tokens = all_tokens[-n_val:]
    print(f"Using {len(val_tokens):,} validation tokens")

    block_size = imit_config.block_size

    if args.mode == 'compare':
        # Check for --prompt or --prompt_file
        prompt_text = None
        if args.prompt:
            prompt_text = args.prompt
        elif args.prompt_file:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read()

        if prompt_text is not None:
            compare_prompt(
                imitator, small_model, tokenizer, prompt_text,
                split_layer, block_size, device,
            )
        else:
            compare_predictions(
                imitator, small_model, tokenizer, val_tokens,
                split_layer, block_size, device,
                num_samples=args.num_samples,
            )
    elif args.mode == 'rollout':
        rollout(
            imitator, small_model, tokenizer, val_tokens,
            split_layer, block_size, device,
            prefix_tokens=args.prefix_tokens,
            rollout_tokens=args.rollout_tokens,
            num_samples=args.num_samples,
        )
    elif args.mode == 'stats':
        compute_stats(
            imitator, small_model, tokenizer, val_tokens,
            split_layer, block_size, args.batch_size, device,
            num_batches=args.num_batches,
        )


if __name__ == '__main__':
    main()
