"""Audio loading and resampling.

WAV/FLAC/OGG are read with soundfile; everything else (mp3, mp4, mkv, ...) is decoded
with ffmpeg, so video files can be subtitled directly.
"""

from __future__ import annotations

import functools
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_SOUNDFILE_EXT = {".wav", ".flac", ".ogg", ".oga", ".aiff", ".aif"}


@functools.lru_cache(maxsize=32)
def _sinc_kernel(orig: int, new: int, width_zeros: int = 6, rolloff: float = 0.99):
    base = min(orig, new) * rolloff
    width = math.ceil(width_zeros * orig / base)
    idx = torch.arange(-width, width + orig, dtype=torch.float64)[None, None] / orig
    t = torch.arange(0, -new, -1, dtype=torch.float64)[:, None, None] / new + idx
    t = (t * base).clamp(-width_zeros, width_zeros)
    window = torch.cos(t * math.pi / width_zeros / 2) ** 2
    t = t * math.pi
    kernel = torch.where(t == 0, torch.ones_like(t), torch.sin(t) / t)
    kernel = kernel * window * (base / orig)
    return kernel.float(), width


def resample(wave: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    """Band-limited (Hann-windowed sinc) resampling of a 1-D or (C, T) tensor."""
    if orig_sr == new_sr:
        return wave
    g = math.gcd(int(orig_sr), int(new_sr))
    orig, new = int(orig_sr) // g, int(new_sr) // g
    kernel, width = _sinc_kernel(orig, new)
    squeeze = wave.dim() == 1
    x = wave[None] if squeeze else wave
    length = x.shape[-1]
    x = F.pad(x.float(), (width, width + orig))
    y = F.conv1d(x[:, None], kernel.to(x.device), stride=orig)
    y = y.transpose(1, 2).reshape(x.shape[0], -1)[:, : math.ceil(new * length / orig)]
    return y[0] if squeeze else y


def _ffmpeg_decode(path: str, sr: int, offset: float, duration: float | None) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(f"ffmpeg is required to decode {path}")
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if offset:
        cmd += ["-ss", f"{offset:.3f}"]
    cmd += ["-i", path]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def load_audio(path: str | Path, sr: int = 16000, offset: float = 0.0, duration: float | None = None) -> torch.Tensor:
    """Mono float32 waveform at ``sr`` Hz."""
    path = str(path)
    if Path(path).suffix.lower() in _SOUNDFILE_EXT:
        import soundfile as sf

        info = sf.info(path)
        start = int(round(offset * info.samplerate))
        frames = -1 if duration is None else int(round(duration * info.samplerate))
        data, file_sr = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
        wave = torch.from_numpy(data.mean(axis=1))
        return resample(wave, file_sr, sr)
    return torch.from_numpy(_ffmpeg_decode(path, sr, offset, duration))


def audio_duration(path: str | Path) -> float:
    path = str(path)
    if Path(path).suffix.lower() in _SOUNDFILE_EXT:
        import soundfile as sf

        return sf.info(path).duration
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return float(out.strip())
