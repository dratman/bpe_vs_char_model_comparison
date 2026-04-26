#!/usr/bin/env python3
"""
train_imitator.py

Train a vector-to-vector "imitator" transformer that predicts the next
residual-stream vector at a fixed middle layer of a frozen small GPT.

Pipeline per training step:
    1. Sample a batch of token sequences from the corpus (continuous mode).
    2. Run the frozen small model through its first N blocks to get the
       residual-stream sequence at the layer-N boundary.
       Shape: (B, T, d_residual)
    3. The imitator's input is positions [0..T-2], target is [1..T-1]
       (next-vector prediction, like next-token but in vector space).
    4. Imitator forward produces predicted vectors (B, T-1, d_residual).
    5. Loss = (1 - cosine_similarity).mean() + lambda_mse * MSE.
    6. Backprop through the imitator only; the small model is frozen.

Usage:
    python py/train_imitator.py \\
        --small_model pt/bpe_16L16H_iter170000.pt \\
        --input txt_local/corpus_books_shuffled_2026_04_18.txt \\
        --output pt/imitator_L8.pt \\
        --split_layer 8 \\
        --d_model 512 --n_layer 6 --n_head 8 \\
        --block_size 512 --batch_size 8 \\
        --max_iters 5000 --learning_rate 3e-4 \\
        --sanity_check
"""

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from imitator_model import Imitator, ImitatorConfig
from small_model_split import load_small_model, forward_to_layer, sanity_check_split
from tokenizer import load_tokenizer


# ---------------------------------------------------------------------------
# Utilities (matching train.py conventions)
# ---------------------------------------------------------------------------

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_elapsed(start_time):
    elapsed = time.time() - start_time
    days = int(elapsed // 86400)
    hours = int((elapsed % 86400) // 3600)
    minutes = int((elapsed % 3600) // 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def get_lr(iter_num, learning_rate, warmup_iters, max_iters):
    """Linear warmup + cosine decay to learning_rate / 10."""
    min_lr = learning_rate / 10
    if iter_num < warmup_iters:
        return learning_rate * iter_num / max(1, warmup_iters)
    if iter_num > max_iters:
        return min_lr
    decay_ratio = (iter_num - warmup_iters) / max(1, max_iters - warmup_iters)
    decay_ratio = max(0.0, min(1.0, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_or_tokenize_corpus(input_path, tokenizer, cache_path=None):
    """
    Tokenize corpus text and return a 1-D long tensor of token IDs.
    If cache_path is given and exists, load from cache instead.
    """
    if cache_path is not None and os.path.exists(cache_path):
        print(f"[{get_timestamp()}] Loading cached tokens from {cache_path}")
        tokens = torch.load(cache_path, map_location='cpu', weights_only=True)
        print(f"[{get_timestamp()}] Loaded {len(tokens):,} tokens")
        return tokens

    print(f"[{get_timestamp()}] Reading corpus from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"[{get_timestamp()}] Read {len(text):,} characters; tokenizing...")
    ids = tokenizer.encode(text)
    tokens = torch.tensor(ids, dtype=torch.long)
    print(f"[{get_timestamp()}] Tokenized to {len(tokens):,} tokens")

    if cache_path is not None:
        print(f"[{get_timestamp()}] Caching tokens to {cache_path}")
        torch.save(tokens, cache_path)

    return tokens


def get_batch_tokens(data, batch_size, block_size, device):
    """
    Sample random windows of length `block_size` from a flat token tensor.
    Returns (B, block_size) long tensor on `device`.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    return x.to(device, non_blocking=True)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def vector_regression_loss(pred, target, lambda_mse=0.1):
    """
    pred, target: (B, T, D)

    Returns:
        total_loss  = (1 - cos_sim).mean() + lambda_mse * mse
        cos_loss    (detached, for logging)
        mse_loss    (detached, for logging)
    """
    cos = F.cosine_similarity(pred, target, dim=-1)   # (B, T)
    cos_loss = (1.0 - cos).mean()
    mse_loss = F.mse_loss(pred, target)
    total = cos_loss + lambda_mse * mse_loss
    return total, cos_loss.detach(), mse_loss.detach()


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(imitator, small_model, train_data, val_data,
                  split_layer, batch_size, block_size, eval_iters,
                  device, ctx, imitator_dtype):
    """Mean loss on a few random batches from train and val splits."""
    out = {}
    imitator.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        cos_losses = torch.zeros(eval_iters)
        mse_losses = torch.zeros(eval_iters)
        total_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            idx = get_batch_tokens(data, batch_size, block_size, device)
            with ctx:
                acts = forward_to_layer(small_model, idx, split_layer)
                acts = acts.to(imitator_dtype)
                inp = acts[:, :-1, :].contiguous()
                tgt = acts[:, 1:, :].contiguous()
                pred = imitator(inp)
                total, cos_l, mse_l = vector_regression_loss(pred, tgt)
            cos_losses[k] = cos_l.item()
            mse_losses[k] = mse_l.item()
            total_losses[k] = total.item()
        out[f'{split}_loss'] = total_losses.mean().item()
        out[f'{split}_cos'] = cos_losses.mean().item()
        out[f'{split}_mse'] = mse_losses.mean().item()
        out[f'{split}_cos_sim'] = 1.0 - cos_losses.mean().item()
    imitator.train()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train an imitator on a frozen model's mid-layer residual stream",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required paths
    p.add_argument('--small_model', type=str, required=True,
                   help='Path to frozen small GPT checkpoint (.pt)')
    p.add_argument('--input', type=str, required=True,
                   help='Corpus text file')
    p.add_argument('--output', type=str, required=True,
                   help='Output checkpoint path for the imitator')

    # Split point
    p.add_argument('--split_layer', type=int, default=8,
                   help='Number of small-model blocks in the front half')

    # Imitator architecture
    p.add_argument('--d_model', type=int, default=512,
                   help='Imitator internal hidden dimension')
    p.add_argument('--n_layer', type=int, default=6,
                   help='Imitator transformer blocks')
    p.add_argument('--n_head', type=int, default=8,
                   help='Imitator attention heads')
    p.add_argument('--block_size', type=int, default=512,
                   help='Max sequence length for the imitator (vectors, not tokens)')
    p.add_argument('--dropout', type=float, default=0.0)

    # Training
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_iters', type=int, default=30000)
    p.add_argument('--learning_rate', type=float, default=3e-4)
    p.add_argument('--warmup_iters', type=int, default=500)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--lambda_mse', type=float, default=0.1,
                   help='Weight on MSE term (cosine similarity is the primary signal)')

    # Eval / logging
    p.add_argument('--eval_interval', type=int, default=500)
    p.add_argument('--eval_iters', type=int, default=20)
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--save_interval', type=int, default=2000)
    p.add_argument('--val_split', type=float, default=0.1)

    # Precision
    p.add_argument('--precision', type=str, default='float32',
                   choices=['float16', 'float32', 'bfloat16'],
                   help='Imitator training precision')
    p.add_argument('--small_model_precision', type=str, default='bfloat16',
                   choices=['float16', 'float32', 'bfloat16'],
                   help='Precision for the frozen small model (default matches its training)')

    # Misc
    p.add_argument('--token_cache', type=str, default=None,
                   help='Cache tokenized corpus to this .pt file (speeds up restart)')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--sanity_check', action='store_true',
                   help='Verify split forward matches full forward before training')
    p.add_argument('--resume', type=str, default=None,
                   help='Resume imitator training from this checkpoint')

    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = ('mps' if torch.backends.mps.is_available()
              else 'cuda' if torch.cuda.is_available()
              else 'cpu')
    print(f"[{get_timestamp()}] Device: {device}")

    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    imitator_dtype = dtype_map[args.precision]
    small_dtype = dtype_map[args.small_model_precision]

    # Mixed-precision autocast context for the imitator
    ctx = nullcontext()
    scaler = None
    if imitator_dtype != torch.float32:
        if device == 'cuda':
            ctx = torch.amp.autocast(device_type='cuda', dtype=imitator_dtype)
            if imitator_dtype == torch.float16:
                scaler = torch.amp.GradScaler('cuda')
        elif device == 'mps':
            ctx = torch.amp.autocast(device_type='mps', dtype=imitator_dtype)

    # ------------------------------------------------------------------
    # Load frozen small model
    # ------------------------------------------------------------------
    print(f"[{get_timestamp()}] Loading small model from {args.small_model}")
    small_model, small_args = load_small_model(
        args.small_model, device=device, dtype=small_dtype,
    )
    n_layer_small = small_args['n_layer']
    d_residual = small_args['n_embd']
    small_block_size = small_args['block_size']

    if not (0 < args.split_layer < n_layer_small):
        raise ValueError(
            f"split_layer={args.split_layer} must be in (0, {n_layer_small})"
        )
    if args.block_size > small_block_size:
        raise ValueError(
            f"Imitator block_size {args.block_size} > small model block_size "
            f"{small_block_size}"
        )

    print(f"[{get_timestamp()}] Split: front half = blocks [0, {args.split_layer}), "
          f"residual dim = {d_residual}")

    # Optional sanity check
    if args.sanity_check:
        print(f"[{get_timestamp()}] Running split sanity check...")
        idx_demo = torch.randint(
            0, small_args['vocab_size'],
            (2, min(64, small_block_size)),
            device=device,
        )
        sanity_check_split(small_model, idx_demo, args.split_layer)

    # ------------------------------------------------------------------
    # Load tokenizer + corpus
    # ------------------------------------------------------------------
    # Derive meta path the same way sample.py does
    model_base = args.small_model.replace('.pt', '')
    if '_iter' in model_base:
        model_base = model_base.rsplit('_iter', 1)[0]
    elif model_base.endswith('_final'):
        model_base = model_base[:-6]
    meta_path = model_base + '_meta.pkl'
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Tokenizer metadata not found: {meta_path}")
    print(f"[{get_timestamp()}] Loading tokenizer from {meta_path}")
    tokenizer = load_tokenizer(meta_path)
    print(f"[{get_timestamp()}] Tokenizer: {tokenizer.tokenizer_type}, "
          f"vocab={tokenizer.vocab_size}")

    all_tokens = load_or_tokenize_corpus(args.input, tokenizer, args.token_cache)
    n = len(all_tokens)
    n_val = int(n * args.val_split)
    n_train = n - n_val
    train_data = all_tokens[:n_train]
    val_data = all_tokens[n_train:]
    print(f"[{get_timestamp()}] Train: {n_train:,} tokens | Val: {n_val:,} tokens")

    # ------------------------------------------------------------------
    # Build or resume imitator
    # ------------------------------------------------------------------
    config = ImitatorConfig(
        d_residual=d_residual,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        block_size=args.block_size,
        dropout=args.dropout,
        bias=True,
    )
    imitator = Imitator(config).to(device)
    # Keep imitator params in float32 for training stability
    # (autocast handles forward-pass precision if needed)

    optimizer = imitator.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type=device,
    )

    iter_num = 0
    best_val_loss = float('inf')
    if args.resume is not None:
        print(f"[{get_timestamp()}] Resuming imitator from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        imitator.load_state_dict(ckpt['imitator'])
        optimizer.load_state_dict(ckpt['optimizer'])
        iter_num = ckpt.get('iter_num', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"[{get_timestamp()}] Resumed at iter {iter_num}, "
              f"best_val_loss={best_val_loss:.4f}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    imitator_param_dtype = next(imitator.parameters()).dtype

    print(f"\n[{get_timestamp()}] Starting imitator training")
    print(f"  Imitator: d_model={args.d_model}, n_layer={args.n_layer}, "
          f"n_head={args.n_head}, block_size={args.block_size}")
    print(f"  Imitator precision: {args.precision} (params in {imitator_param_dtype})")
    print(f"  Small model precision: {args.small_model_precision}")
    print(f"  Split layer: {args.split_layer} of {n_layer_small}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max iters: {args.max_iters}")
    print(f"  LR: {args.learning_rate:.2e}, warmup: {args.warmup_iters}")
    print(f"  Loss: cosine + {args.lambda_mse} * MSE")
    print(f"  Output: {args.output}")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    training_start = time.time()
    t0 = time.time()
    speed_t0 = time.time()

    while iter_num < args.max_iters:
        # LR schedule
        lr = get_lr(iter_num, args.learning_rate, args.warmup_iters, args.max_iters)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Eval
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(
                imitator, small_model,
                train_data, val_data,
                split_layer=args.split_layer,
                batch_size=args.batch_size,
                block_size=args.block_size,
                eval_iters=args.eval_iters,
                device=device,
                ctx=ctx,
                imitator_dtype=imitator_param_dtype,
            )
            elapsed = format_elapsed(training_start)
            print(f"[{get_timestamp()}] [{elapsed}] iter {iter_num:6d} | "
                  f"train: loss={losses['train_loss']:.4f} "
                  f"cos_sim={losses['train_cos_sim']:.4f} "
                  f"mse={losses['train_mse']:.4f} | "
                  f"val: loss={losses['val_loss']:.4f} "
                  f"cos_sim={losses['val_cos_sim']:.4f} "
                  f"mse={losses['val_mse']:.4f} | lr={lr:.2e}")

            if math.isnan(losses['train_loss']) or math.isnan(losses['val_loss']):
                print("!" * 60)
                print("NaN detected; stopping.")
                print("!" * 60)
                break

            # Save best
            if losses['val_loss'] < best_val_loss and iter_num > 0:
                best_val_loss = losses['val_loss']
                ckpt = {
                    'imitator': imitator.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'imitator_config': vars(config),
                    'small_model_path': os.path.abspath(args.small_model),
                    'small_model_args': small_args,
                    'split_layer': args.split_layer,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'val_metrics': losses,
                    'config': vars(args),
                }
                torch.save(ckpt, args.output)
                print(f"  Saved best to {args.output} "
                      f"(val_loss={best_val_loss:.4f}, "
                      f"cos_sim={losses['val_cos_sim']:.4f})")

        # Periodic save
        if iter_num > 0 and iter_num % args.save_interval == 0:
            periodic_path = os.path.splitext(args.output)[0] + f'_iter{iter_num}.pt'
            ckpt = {
                'imitator': imitator.state_dict(),
                'optimizer': optimizer.state_dict(),
                'imitator_config': vars(config),
                'small_model_path': os.path.abspath(args.small_model),
                'small_model_args': small_args,
                'split_layer': args.split_layer,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': vars(args),
            }
            torch.save(ckpt, periodic_path)
            print(f"[{get_timestamp()}] [{format_elapsed(training_start)}] "
                  f"Saved periodic checkpoint to {periodic_path}")

        # ---- Training step ----

        # Sample tokens, run through frozen front half
        idx = get_batch_tokens(train_data, args.batch_size, args.block_size, device)
        with torch.no_grad():
            acts = forward_to_layer(small_model, idx, args.split_layer)  # (B, T, D)

        # Cast to imitator dtype
        acts = acts.to(imitator_param_dtype)

        # Next-vector prediction: input = [0..T-2], target = [1..T-1]
        inp = acts[:, :-1, :].contiguous()
        tgt = acts[:, 1:, :].contiguous()

        optimizer.zero_grad(set_to_none=True)
        with ctx:
            pred = imitator(inp)
            loss, cos_l, mse_l = vector_regression_loss(
                pred, tgt, lambda_mse=args.lambda_mse,
            )

        if torch.isnan(loss):
            print(f"!!! NaN at iter {iter_num}, stopping.")
            break

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(imitator.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(imitator.parameters(), args.grad_clip)
            optimizer.step()

        # Log
        if iter_num % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"iter {iter_num:6d}: loss={loss.item():.4f} "
                  f"cos={cos_l.item():.4f} mse={mse_l.item():.4f} "
                  f"time={dt*1000:.1f}ms")

        if iter_num > 0 and iter_num % 100 == 0:
            speed_dt = time.time() - speed_t0
            print(f"[{get_timestamp()}] [{format_elapsed(training_start)}] "
                  f"speed: {100/speed_dt:.2f} iter/s")
            speed_t0 = time.time()

        iter_num += 1

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    final_path = os.path.splitext(args.output)[0] + '_final.pt'
    ckpt = {
        'imitator': imitator.state_dict(),
        'optimizer': optimizer.state_dict(),
        'imitator_config': vars(config),
        'small_model_path': os.path.abspath(args.small_model),
        'small_model_args': small_args,
        'split_layer': args.split_layer,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': vars(args),
    }
    torch.save(ckpt, final_path)
    elapsed = format_elapsed(training_start)
    print(f"\n[{get_timestamp()}] [{elapsed}] Training complete.")
    print(f"  Final iteration: {iter_num}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Saved final to {final_path}")
    print(f"  Best checkpoint at {args.output}")


if __name__ == '__main__':
    main()
