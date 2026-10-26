"""Adapters turning encoder frames into LLM input embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn

from subasr.config import AdapterConfig


class _QFormerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_pad: torch.Tensor) -> torch.Tensor:
        h = self.n1(q)
        q = q + self.self_attn(h, h, h, need_weights=False)[0]
        q = q + self.cross_attn(self.n2(q), kv, kv, key_padding_mask=kv_pad, need_weights=False)[0]
        return q + self.ff(self.n3(q))


class WindowQFormer(nn.Module):
    """Window-level Q-Former (as in Granite Speech / SALMONN).

    Frames are cut into windows of ``window_frames``; ``num_queries`` learned queries
    attend to each window. With 25 frames/s, ``window_frames=25, num_queries=12``
    gives 12 LLM tokens per second of audio. Several encoder layers can be
    concatenated as input (multi-layer features help with singing and accents).
    """

    def __init__(self, in_dims: list[int], out_dim: int, cfg: AdapterConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(sum(in_dims), cfg.dim)
        self.frame_pos = nn.Parameter(torch.randn(cfg.window_frames, cfg.dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(cfg.num_queries, cfg.dim) * 0.02)
        self.layers = nn.ModuleList(_QFormerLayer(cfg.dim, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers))
        self.out_norm = nn.LayerNorm(cfg.dim)
        self.out_proj = nn.Linear(cfg.dim, out_dim)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        W, Q = self.cfg.window_frames, self.cfg.num_queries
        return (lengths + W - 1) // W * Q

    def forward(self, layers: list[torch.Tensor], hidden: torch.Tensor, lengths: torch.Tensor):
        W, Q = self.cfg.window_frames, self.cfg.num_queries
        x = self.in_proj(torch.cat(layers, dim=-1))
        B, T, D = x.shape
        N = (T + W - 1) // W
        x = nn.functional.pad(x, (0, 0, 0, N * W - T)).view(B * N, W, D) + self.frame_pos
        frame_idx = torch.arange(N * W, device=x.device).view(N, W)
        pad = (frame_idx[None] >= lengths[:, None, None]).view(B * N, W)
        pad[:, 0] = False  # fully padded windows: avoid NaN, their outputs are masked anyway
        q = self.queries.expand(B * N, Q, D)
        for layer in self.layers:
            q = layer(q, x, pad)
        out = self.out_proj(self.out_norm(q)).view(B, N * Q, -1)
        return out, self.output_lengths(lengths)


class GemmaNativeAdapter(nn.Module):
    """Gemma 4's own pretrained audio projector (25 tokens/s, already aligned with the LLM)."""

    def __init__(self, gemma):
        super().__init__()
        self.embedder = gemma.model.embed_audio

    def forward(self, layers: list[torch.Tensor], hidden: torch.Tensor, lengths: torch.Tensor):
        return self.embedder(inputs_embeds=hidden), lengths
