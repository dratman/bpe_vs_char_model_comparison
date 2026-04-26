"""
small_model_split.py

Load a frozen small GPT model and split its forward pass at a given layer
boundary.  Two functions are exposed:

    forward_to_layer(model, idx, N)
        Embed token IDs and run through blocks [0, N).
        Returns the residual stream tensor (B, T, n_embd).

    forward_from_layer(model, x, N)
        Take a residual stream tensor at the layer-N boundary and run
        through blocks [N, n_layer) + ln_f + lm_head.
        Returns logits (B, T, vocab_size) or (B, 1, vocab_size).

    sanity_check_split(model, idx, N)
        Verify that forward_to_layer + forward_from_layer reproduces the
        full forward pass within numerical tolerance.

    load_small_model(path, device, dtype)
        Load a checkpoint produced by train.py, freeze all parameters,
        and return (model, model_args_dict).
"""

import sys
import os
import torch
from model import GPT, GPTConfig


def load_small_model(checkpoint_path, device='cpu', dtype=torch.float32):
    """
    Load a GPT checkpoint (train.py format), freeze all parameters,
    and move to the given device/dtype.

    Returns:
        model     -- frozen GPT on `device` in `dtype`
        model_args -- dict of model hyperparameters from checkpoint
    """
    print(f"Loading small model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint['model_args']

    # Suppress the init print from GPT.__init__
    _stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    sys.stdout = _stdout

    model.load_state_dict(checkpoint['model'])
    model.to(device=device, dtype=dtype)
    model.eval()

    # Freeze every parameter
    for p in model.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded {n_params:.1f}M-param model: n_layer={model_args['n_layer']}, "
          f"n_head={model_args['n_head']}, n_embd={model_args['n_embd']}, "
          f"block_size={model_args['block_size']}, vocab_size={model_args['vocab_size']}")
    return model, model_args


def forward_to_layer(model, idx, N):
    """
    Run the embedding + first N transformer blocks.

    Args:
        model: a GPT instance (frozen or not)
        idx:   (B, T) long tensor of token IDs
        N:     number of blocks to run (0-indexed: blocks 0..N-1)

    Returns:
        x: (B, T, n_embd) residual stream after block N-1
    """
    device = idx.device
    b, t = idx.size()
    assert t <= model.config.block_size, (
        f"Sequence length {t} exceeds block_size {model.config.block_size}"
    )
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    tok_emb = model.transformer.wte(idx)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)

    for block in model.transformer.h[:N]:
        x = block(x)

    return x


def forward_from_layer(model, x, N, last_token_only=False):
    """
    Run blocks [N, n_layer) + ln_f + lm_head.

    Args:
        model:           a GPT instance
        x:               (B, T, n_embd) residual stream at layer-N boundary
        N:               layer index where x was extracted
        last_token_only: if True, only compute logits for the final position

    Returns:
        logits: (B, T, vocab_size) or (B, 1, vocab_size)
    """
    for block in model.transformer.h[N:]:
        x = block(x)
    x = model.transformer.ln_f(x)

    if last_token_only:
        x = x[:, [-1], :]

    logits = model.lm_head(x)
    return logits


def sanity_check_split(model, idx, N, atol=1e-4):
    """
    Verify that forward_to_layer(N) piped into forward_from_layer(N)
    matches the full model forward pass.

    Raises AssertionError if the max absolute difference exceeds `atol`.
    """
    with torch.no_grad():
        # Full forward
        logits_full, _ = model(idx)

        # Split forward
        mid = forward_to_layer(model, idx, N)
        # Full forward only computes last-token logits when targets=None,
        # so we need to compare the same.
        logits_split = forward_from_layer(model, mid, N, last_token_only=True)

        diff = (logits_full - logits_split).abs().max().item()

    if diff > atol:
        raise AssertionError(
            f"Sanity check FAILED: max |full - split| = {diff:.6f} > atol={atol}"
        )
    print(f"Sanity check passed: max |full - split| = {diff:.2e} (atol={atol})")
