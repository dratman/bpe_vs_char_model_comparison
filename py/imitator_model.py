"""
imitator_model.py

A vector-to-vector transformer that predicts the next residual-stream
vector at a fixed middle layer of a frozen GPT model.

Architecture:
    Linear(d_residual -> d_model)    # project down from residual dim
    N x Block                         # standard transformer blocks (from model.py)
    Linear(d_model -> d_residual)    # project back up

NO token embeddings, NO positional embeddings.  The residual stream
vectors already carry positional information baked in by the small
model's first N layers.

The Block class is imported from model.py so that the imitator uses
the same attention + MLP internals as the small model (minus weight
sharing -- the imitator trains its own weights from scratch).
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn

from model import Block, GPTConfig


@dataclass
class ImitatorConfig:
    d_residual: int = 2048   # dimensionality of the small model's residual stream
    d_model: int = 512       # imitator's internal hidden dim
    n_layer: int = 6         # number of transformer blocks
    n_head: int = 8          # attention heads
    block_size: int = 512    # max sequence length (in vectors, not tokens)
    dropout: float = 0.0
    bias: bool = True


class Imitator(nn.Module):
    """
    Vector-to-vector transformer.

    Takes (B, T, d_residual) in, produces (B, T, d_residual) out.
    No embeddings; purely operates on continuous activation vectors.
    """

    def __init__(self, config: ImitatorConfig):
        super().__init__()
        self.config = config

        # Input/output projections: only needed when d_model != d_residual
        self.needs_projection = (config.d_model != config.d_residual)
        if self.needs_projection:
            self.proj_in = nn.Linear(config.d_residual, config.d_model, bias=config.bias)
            self.proj_out = nn.Linear(config.d_model, config.d_residual, bias=config.bias)

        # Transformer blocks -- reuse model.py's Block via a GPTConfig
        # with the imitator's dimensions
        block_config = GPTConfig(
            block_size=config.block_size,
            vocab_size=1,  # unused, but GPTConfig requires it
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.d_model,
            dropout=config.dropout,
            bias=config.bias,
        )
        self.blocks = nn.ModuleList([Block(block_config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model, elementwise_affine=config.bias)

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections (c_proj) per GPT-2 convention
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"Imitator: {n_params:.1f}M parameters "
              f"(d_model={config.d_model}, n_layer={config.n_layer}, "
              f"n_head={config.n_head}, block_size={config.block_size})")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: (B, T, d_residual) -- sequence of residual-stream vectors

        Returns:
            (B, T, d_residual) -- predicted next-vector at each position
        """
        B, T, D = x.size()
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds imitator block_size {self.config.block_size}"
        )

        if self.needs_projection:
            x = self.proj_in(x)            # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)                  # (B, T, d_model)
        if self.needs_projection:
            x = self.proj_out(x)           # (B, T, d_residual)
        return x

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Separate parameters into weight-decay and no-decay groups.
        Mirrors GPT.configure_optimizers from model.py.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"Imitator optimizer: {len(decay_params)} decayed tensors ({num_decay:,} params), "
              f"{len(nodecay_params)} non-decayed ({num_nodecay:,} params)")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer
