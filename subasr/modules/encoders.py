"""Audio encoders.

Both encoders take raw 16 kHz waveforms and return frame-level hidden states at
25 frames per second (40 ms per frame), plus the hidden states of the layers the
adapter reads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from subasr.audio.features import LogMel, spec_augment
from subasr.config import EncoderConfig


@dataclass
class EncoderOutput:
    hidden: torch.Tensor  # (B, T, D) input of the CTC / TDT / event heads
    lengths: torch.Tensor  # (B,)
    layers: list[torch.Tensor]  # hidden states selected by ``adapter_layers``

    @property
    def mask(self) -> torch.Tensor:
        return torch.arange(self.hidden.shape[1], device=self.hidden.device)[None] < self.lengths[:, None]


def _rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding on the last dim of ``x`` (B, H, T, Dh).

    ``positions``: (T,) shared by the batch, or (B, T) per sequence.
    """
    dh = x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh))
    ang = positions.float()[..., None] * inv  # (T, Dh/2) or (B, T, Dh/2)
    if ang.dim() == 3:
        ang = ang[:, None]  # broadcast over heads
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class RotaryAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        pos = torch.arange(T, device=x.device)
        q, k = _rope(q, pos), _rope(k, pos)
        attn_mask = key_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * mult), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    def __init__(self, dim: int, kernel: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, 2 * dim)
        self.dw = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.pw2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = F.glu(self.pw1(self.norm(x)), dim=-1) * mask[..., None]
        y = self.dw(y.transpose(1, 2)).transpose(1, 2)
        y = self.pw2(F.silu(self.norm2(y)))
        return self.dropout(y)


class ConformerBlock(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        d = cfg.d_model
        self.ff1 = FeedForward(d, cfg.ff_mult, cfg.dropout)
        self.attn_norm = nn.LayerNorm(d)
        self.attn = RotaryAttention(d, cfg.n_heads, cfg.dropout)
        self.conv = ConvModule(d, cfg.conv_kernel, cfg.dropout)
        self.ff2 = FeedForward(d, cfg.ff_mult, cfg.dropout)
        self.out_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.dropout(self.ff1(x))
        x = x + self.dropout(self.attn(self.attn_norm(x), mask))
        x = x + self.conv(x, mask)
        x = x + 0.5 * self.dropout(self.ff2(x))
        return self.out_norm(x)


class ConformerEncoder(nn.Module):
    """Log-mel -> 4x conv subsampling (25 frames/s) -> Conformer blocks with rotary attention."""

    frame_rate = 25.0

    def __init__(self, cfg: EncoderConfig, sample_rate: int = 16000):
        super().__init__()
        self.cfg = cfg
        self.frontend = LogMel(sample_rate, cfg.n_mels)
        c = 64
        self.subsample = nn.Sequential(
            nn.Conv2d(1, c, 3, stride=2, padding=1), nn.SiLU(), nn.Conv2d(c, c, 3, stride=2, padding=1), nn.SiLU()
        )
        self.proj = nn.Linear(c * math.ceil(math.ceil(cfg.n_mels / 2) / 2), cfg.d_model)
        self.blocks = nn.ModuleList(ConformerBlock(cfg) for _ in range(cfg.n_layers))
        self.dim = cfg.d_model
        self.layer_dims = [cfg.d_model] * len(cfg.adapter_layers)

    @staticmethod
    def _sub_len(n: torch.Tensor) -> torch.Tensor:
        n = (n - 1) // 2 + 1
        return (n - 1) // 2 + 1

    def forward(self, wave: torch.Tensor, wave_lens: torch.Tensor) -> EncoderOutput:
        feats, flen = self.frontend(wave, wave_lens)
        if self.training:
            feats = spec_augment(feats, flen, self.cfg.spec_augment)
        x = self.subsample(feats[:, None])  # (B, C, T', F')
        B, C, T, Fq = x.shape
        x = self.proj(x.permute(0, 2, 1, 3).reshape(B, T, C * Fq))
        lengths = self._sub_len(flen).clamp(max=T)
        mask = torch.arange(T, device=x.device)[None] < lengths[:, None]
        states = []
        for block in self.blocks:
            if self.cfg.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)
            states.append(x)
        layers = [states[i] for i in self.cfg.adapter_layers]
        return EncoderOutput(hidden=x, lengths=lengths, layers=layers)


class Gemma4AudioEncoder(nn.Module):
    """Gemma 4 audio tower (USM-style Conformer, 40 ms frames).

    ``hidden`` is the tower's projected output (``output_proj``), the same tensor
    Gemma's own audio projector reads. ``layers`` are raw Conformer layer outputs.
    """

    frame_rate = 25.0

    def __init__(self, cfg: EncoderConfig, gemma, feature_extractor):
        super().__init__()
        self.cfg = cfg
        self.tower = gemma.model.audio_tower
        self.fe = feature_extractor
        acfg = self.tower.config
        self.dim = acfg.output_proj_dims
        self.layer_dims = [acfg.hidden_size] * len(cfg.adapter_layers)
        if cfg.gradient_checkpointing:
            self.tower.gradient_checkpointing_enable()

    def forward(self, wave: torch.Tensor, wave_lens: torch.Tensor) -> EncoderOutput:
        arrays = [wave[i, : int(wave_lens[i])].float().cpu().numpy() for i in range(wave.shape[0])]
        feats = self.fe(
            arrays,
            sampling_rate=self.fe.sampling_rate,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            max_length=None,
        )
        dtype = next(self.tower.parameters()).dtype
        x = feats["input_features"].to(wave.device, dtype)
        fmask = feats["input_features_mask"].to(wave.device)
        if self.training:
            x = spec_augment(x, fmask.sum(1), self.cfg.spec_augment) * fmask[..., None]
        # Raw Conformer layer outputs are captured with hooks: the tower's own
        # ``hidden_states`` replaces the last layer with the projected output.
        captured: list[torch.Tensor] = []
        hooks = [layer.register_forward_hook(lambda m, i, o: captured.append(o)) for layer in self.tower.layers]
        try:
            out = self.tower(x, fmask, return_dict=True)
        finally:
            for h in hooks:
                h.remove()
        lengths = out.attention_mask.sum(1).long()
        layers = [captured[i] for i in self.cfg.adapter_layers]
        return EncoderOutput(hidden=out.last_hidden_state, lengths=lengths, layers=layers)
