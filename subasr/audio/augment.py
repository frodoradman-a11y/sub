"""Waveform augmentation: music / noise mixing, speed perturbation, gain.

Music mixing is the main tool for robustness on soundtracks and songs: speech is
mixed with music at a random SNR (default -5 .. 20 dB) and the "music" event label
is switched on, so the audio-event head learns to detect it as well.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from subasr.audio.io import load_audio, resample
from subasr.config import AugmentConfig


def _read_paths(manifest: str | None) -> list[str]:
    if not manifest:
        return []
    paths = []
    base = Path(manifest).parent
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                p = Path(json.loads(line)["audio"])
                paths.append(str(p if p.is_absolute() else base / p))
    return paths


def mix_at_snr(signal: torch.Tensor, other: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add ``other`` (looped / cropped to length) to ``signal`` at the given SNR."""
    if other.numel() == 0:
        return signal
    if other.numel() < signal.numel():
        other = other.repeat(signal.numel() // other.numel() + 1)
    start = random.randint(0, other.numel() - signal.numel())
    other = other[start : start + signal.numel()]
    p_sig = signal.pow(2).mean().clamp_min(1e-8)
    p_oth = other.pow(2).mean().clamp_min(1e-8)
    scale = torch.sqrt(p_sig / (p_oth * 10 ** (snr_db / 10)))
    return signal + scale * other


class Augmenter:
    def __init__(self, cfg: AugmentConfig, sample_rate: int = 16000):
        self.cfg = cfg
        self.sr = sample_rate
        self.music = _read_paths(cfg.music_manifest)
        self.noise = _read_paths(cfg.noise_manifest)

    def _random_clip(self, paths: list[str], seconds: float) -> torch.Tensor:
        path = random.choice(paths)
        try:
            return load_audio(path, self.sr, duration=seconds + 1.0)
        except Exception:
            return torch.zeros(0)

    def __call__(self, wave: torch.Tensor, labels: set[str]) -> tuple[torch.Tensor, set[str]]:
        cfg = self.cfg
        if not cfg.enabled or wave.numel() == 0:
            return wave, labels
        labels = set(labels)
        if cfg.speed_prob > 0 and random.random() < cfg.speed_prob:
            speed = random.choice(cfg.speeds)
            wave = resample(wave, int(round(self.sr * speed)), self.sr)
        seconds = wave.numel() / self.sr
        if self.music and random.random() < cfg.music_prob:
            wave = mix_at_snr(wave, self._random_clip(self.music, seconds), random.uniform(*cfg.music_snr_db))
            labels.add("music")
        if self.noise and random.random() < cfg.noise_prob:
            wave = mix_at_snr(wave, self._random_clip(self.noise, seconds), random.uniform(*cfg.noise_snr_db))
            labels.add("noise")
        gain = 10 ** (random.uniform(*cfg.gain_db) / 20)
        wave = wave * gain
        peak = wave.abs().max()
        if peak > 1.0:
            wave = wave / peak
        return wave, labels
