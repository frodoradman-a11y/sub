"""Log-mel features (for the Conformer encoder) and SpecAugment."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from subasr.config import SpecAugmentConfig


def _hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)


def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> torch.Tensor:
    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0, sr / 2, n_freqs)
    mels = torch.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2)
    hz = 700.0 * (10 ** (mels / 2595.0) - 1.0)
    lower, center, upper = hz[:-2, None], hz[1:-1, None], hz[2:, None]
    up = (freqs[None] - lower) / (center - lower)
    down = (upper - freqs[None]) / (upper - center)
    return torch.clamp(torch.minimum(up, down), min=0.0)  # (n_mels, n_freqs)


class LogMel(nn.Module):
    """25 ms window, 10 ms hop log-mel spectrogram."""

    def __init__(
        self, sample_rate: int = 16000, n_mels: int = 80, win_ms: float = 25.0, hop_ms: float = 10.0, dither: float = 1e-4
    ):
        super().__init__()
        # Dither keeps digital silence (exact zeros, common in films) from mapping to
        # an extreme log value that never occurs in recorded training audio.
        self.dither = dither
        self.win = int(sample_rate * win_ms / 1000)
        self.hop = int(sample_rate * hop_ms / 1000)
        self.n_fft = 1 << (self.win - 1).bit_length()
        self.register_buffer("window", torch.hann_window(self.win), persistent=False)
        self.register_buffer("fbank", _mel_filterbank(n_mels, self.n_fft, sample_rate), persistent=False)

    def num_frames(self, num_samples: torch.Tensor) -> torch.Tensor:
        return num_samples // self.hop + 1

    @torch.no_grad()
    def forward(self, wave: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        wave = wave.float()
        if self.dither > 0:
            gen = torch.Generator(device=wave.device).manual_seed(0)
            wave = wave + self.dither * torch.randn(wave.shape, generator=gen, device=wave.device)
        spec = torch.stft(
            wave,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = spec.abs().pow(2)  # (B, F, T)
        mel = torch.matmul(self.fbank, power).clamp_min(1e-10).log10()
        mel = (mel.clamp_min(mel.amax(dim=(1, 2), keepdim=True) - 8.0) + 4.0) / 4.0
        feats = mel.transpose(1, 2)  # (B, T, n_mels)
        flen = self.num_frames(lengths).clamp(max=feats.shape[1])
        mask = torch.arange(feats.shape[1], device=feats.device)[None] < flen[:, None]
        return feats * mask[..., None], flen


def spec_augment(feats: torch.Tensor, lengths: torch.Tensor, cfg: SpecAugmentConfig) -> torch.Tensor:
    """Frequency and time masking (in place on a copy). ``feats``: (B, T, F)."""
    if not cfg.enabled:
        return feats
    feats = feats.clone()
    B, T, Fdim = feats.shape
    for b in range(B):
        for _ in range(cfg.freq_masks):
            w = int(torch.randint(0, cfg.freq_width + 1, ()).item())
            if w and w < Fdim:
                f0 = int(torch.randint(0, Fdim - w, ()).item())
                feats[b, :, f0 : f0 + w] = 0.0
        length = int(lengths[b].item())
        max_w = max(1, int(length * cfg.time_width_ratio))
        for _ in range(cfg.time_masks):
            w = int(torch.randint(0, max_w + 1, ()).item())
            if w and w < length:
                t0 = int(torch.randint(0, length - w, ()).item())
                feats[b, t0 : t0 + w, :] = 0.0
    return feats
