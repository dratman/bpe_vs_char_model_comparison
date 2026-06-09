#!/usr/bin/env python
"""
benchmark_throughput.py — batch-size throughput sweep for the char model.

Times a FAITHFUL training step (zero_grad -> autocast forward -> backward ->
clip_grad_norm_(1.0) -> optimizer.step()) on synthetic on-GPU batches, replicating
train.py's inner loop exactly so the numbers are directly comparable to a real run.

It builds the model ONCE (batch size is not a model parameter) and reuses it
across batch sizes, catching CUDA OOM to discover the memory ceiling.

Defaults match the no-GELU matched-LR run (16L/8H/1280, block 4096, bf16, char
vocab 78, bias on, tied weights, dropout 0). Validate against the known real
number: batch 4 should reproduce ~0.535 s/iter and ~29% (A100-relative) MFU.

Usage:
  python py/benchmark_throughput.py --batches 4,8,16,32 --warmup 20 --iters 40
"""
import argparse
import time
import torch
from model import GPT, GPTConfig

A100_BF16_PEAK = 312e12   # what model.estimate_mfu() normalizes against
A6000_BF16_PEAK = 155e12  # RTX A6000 dense bf16 tensor-core peak (FP32 accumulate)


def run_step(model, optimizer, ctx, x, y):
    optimizer.zero_grad(set_to_none=True)
    with ctx:
        _, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', type=str, default='4,8,16,32')
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--iters', type=int, default=40)
    ap.add_argument('--block_size', type=int, default=4096)
    ap.add_argument('--n_layer', type=int, default=16)
    ap.add_argument('--n_head', type=int, default=8)
    ap.add_argument('--n_embd', type=int, default=1280)
    ap.add_argument('--vocab_size', type=int, default=78)
    ap.add_argument('--learning_rate', type=float, default=1.5e-4)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    device = 'cuda'
    name = torch.cuda.get_device_name(0)
    batches = [int(b) for b in args.batches.split(',')]

    cfg = GPTConfig(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        block_size=args.block_size, vocab_size=args.vocab_size,
        dropout=0.0, bias=True, tie_weights=True, no_gelu=True,
    )
    model = GPT(cfg).to(device)
    model.train()
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=args.learning_rate,
        betas=(0.9, 0.95), device_type=device)
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    nparams = model.get_num_params()
    tok_per_iter_factor = args.block_size  # tokens/iter = batch * block_size

    print(f"GPU: {name}")
    print(f"Model: {args.n_layer}L/{args.n_head}H/{args.n_embd}, block {args.block_size}, "
          f"vocab {args.vocab_size}, no_gelu=True, bf16")
    print(f"Non-embedding params: {nparams/1e6:.1f}M")
    print(f"Warmup {args.warmup} iters, timed {args.iters} iters per batch\n")
    hdr = (f"{'batch':>5} | {'tokens/iter':>11} | {'ms/iter':>9} | {'tokens/sec':>11} | "
           f"{'MFU(A100)':>9} | {'MFU(A6000)':>10} | {'peakGB':>7} | status")
    print(hdr)
    print("-" * len(hdr))

    for B in batches:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            x = torch.randint(0, args.vocab_size, (B, args.block_size), device=device, dtype=torch.long)
            y = torch.randint(0, args.vocab_size, (B, args.block_size), device=device, dtype=torch.long)

            for _ in range(args.warmup):
                run_step(model, optimizer, ctx, x, y)
            torch.cuda.synchronize()

            t0 = time.time()
            for _ in range(args.iters):
                run_step(model, optimizer, ctx, x, y)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / args.iters

            tokens_iter = B * tok_per_iter_factor
            tokens_sec = tokens_iter / dt
            mfu_a100 = model.estimate_mfu(B, dt)            # matches train.py's logged MFU
            flops_achieved = mfu_a100 * A100_BF16_PEAK
            mfu_a6000 = flops_achieved / A6000_BF16_PEAK
            peak_gb = torch.cuda.max_memory_allocated() / 1e9

            print(f"{B:>5} | {tokens_iter:>11,} | {dt*1000:>9.1f} | {tokens_sec:>11,.0f} | "
                  f"{mfu_a100*100:>8.1f}% | {mfu_a6000*100:>9.1f}% | {peak_gb:>7.1f} | ok")
            del x, y
        except torch.cuda.OutOfMemoryError:
            print(f"{B:>5} | {'-':>11} | {'-':>9} | {'-':>11} | {'-':>9} | {'-':>10} | {'-':>7} | OOM")
            torch.cuda.empty_cache()
            continue

    print(f"\nTotal GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


if __name__ == '__main__':
    main()
