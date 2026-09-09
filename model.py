"""
model.py — the denoising network, from scratch.

The network's only job: given a noisy point x_t (shape (B, 2)) and the timestep
t (shape (B,)), predict the noise eps that was added. Output shape == input
shape (B, 2).

TWO IDEAS WORTH UNDERSTANDING
    1. The SAME network handles every noise level. Step t=5 (barely noisy) and
       step t=190 (almost pure noise) need very different denoising behavior, so
       the network must know *which* step it is on. We tell it by feeding t in.

    2. We don't feed the raw integer t — a bare "37.0" is a poor input for a
       neural net. Instead we use a SINUSOIDAL EMBEDDING (the same idea as
       positional encodings in the Transformer you built in 01): map the scalar
       t to a smooth, high-dimensional vector of sines and cosines at many
       frequencies. Nearby steps get similar embeddings; the network can read
       "how noisy am I" at multiple scales.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Map an integer timestep t to a (dim,)-vector of sines and cosines.

    For embedding dimension `dim`, we use dim/2 frequencies spaced
    geometrically. Component i uses frequency  1 / 10000^(2i/dim):

        emb(t) = [ sin(t*f_0), sin(t*f_1), ..., cos(t*f_0), cos(t*f_1), ... ]

    Low frequencies vary slowly across t (coarse "how far along"); high
    frequencies vary fast (fine distinctions). This is identical in spirit to
    the Transformer's positional encoding — here the "position" is the diffusion
    step instead of the token index.
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "time embedding dim must be even"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        # frequencies: 10000^(-i/half) for i = 0 .. half-1
        exponents = torch.arange(half, device=device, dtype=torch.float32) / half
        freqs = torch.exp(-math.log(10000.0) * exponents)   # (half,)
        args = t.float()[:, None] * freqs[None, :]          # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class Block(nn.Module):
    """A small residual block: LayerNorm -> Linear -> SiLU, plus a skip.

    Residual connections let gradients flow and make deeper MLPs trainable —
    the same reason they help in ResNets and Transformers.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.lin(self.norm(x)))


class DenoiseMLP(nn.Module):
    """Predict the noise eps given (x_t, t).

    Architecture:
        - embed t with sinusoids, then a tiny MLP -> a hidden-size time vector
        - project the 2D point up to hidden size
        - ADD the time vector (this is how the network is "told" the noise level)
        - a stack of residual blocks
        - project back down to 2D (the predicted noise)
    """

    def __init__(self, data_dim: int = 2, hidden: int = 128,
                 n_blocks: int = 4, time_dim: int = 64):
        super().__init__()
        # Timestep pathway: sinusoids -> MLP -> hidden-sized conditioning vector
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # Data pathway
        self.in_proj = nn.Linear(data_dim, hidden)
        self.blocks = nn.ModuleList([Block(hidden) for _ in range(n_blocks)])
        self.out_proj = nn.Linear(hidden, data_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, data_dim)   t: (B,)
        h = self.in_proj(x) + self.time_embed(t)   # inject timestep conditioning
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h)                     # predicted noise, (B, data_dim)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Shape sanity check.
    net = DenoiseMLP()
    x = torch.randn(8, 2)
    t = torch.randint(0, 200, (8,))
    out = net(x, t)
    print("output shape:", tuple(out.shape), "(expect (8, 2))")
    print("parameters:", net.num_params())
