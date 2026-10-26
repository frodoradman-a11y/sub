"""Splitting long recordings (whole films) into encoder-sized segments.

1. Frames louder than a threshold (relative to the loudest part) are "active";
   short gaps are bridged and each active region gets a small margin.
2. Neighbouring regions are packed into one segment while the result stays under
   ``max_seconds`` and the gap between them is short.
3. Regions longer than ``max_seconds`` are cut at the quietest point near the
   limit, so words are rarely cut in half.

Long silences are dropped entirely. Quiet speech is not: the threshold is 50 dB
below the loudest frame (and at least ``silence_db``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

HOP_S = 0.01


@dataclass
class Segment:
    start: int  # sample index
    end: int

    def seconds(self, sr: int) -> tuple[float, float]:
        return self.start / sr, self.end / sr


def frame_energy_db(wave: torch.Tensor, sr: int, hop_s: float = HOP_S, win_s: float = 0.025) -> torch.Tensor:
    hop, win = int(sr * hop_s), int(sr * win_s)
    if wave.numel() < win:
        wave = torch.nn.functional.pad(wave, (0, win - wave.numel()))
    frames = wave.unfold(0, win, hop)
    return 10 * torch.log10(frames.pow(2).mean(dim=1).clamp_min(1e-10))


def _active_regions(active: list[bool], bridge: int, margin: int, n: int) -> list[tuple[int, int]]:
    regions: list[list[int]] = []
    for i, a in enumerate(active):
        if not a:
            continue
        if regions and i - regions[-1][1] <= bridge:
            regions[-1][1] = i + 1
        else:
            regions.append([i, i + 1])
    return [(max(0, s - margin), min(n, e + margin)) for s, e in regions]


def _cut_long(energy: torch.Tensor, s: int, e: int, max_f: int, search_f: int) -> list[tuple[int, int]]:
    out = []
    while e - s > max_f:
        lo, hi = s + max_f - search_f, s + max_f
        window = energy[lo:hi]
        if window.numel() > 20:
            window = torch.nn.functional.avg_pool1d(window[None, None], 21, 1, 10, count_include_pad=False)[0, 0]
        cut = max(lo + int(torch.argmin(window)), s + 1)
        out.append((s, cut))
        s = cut
    out.append((s, e))
    return out


def split_audio(
    wave: torch.Tensor,
    sr: int = 16000,
    max_seconds: float = 30.0,
    search_seconds: float = 6.0,
    silence_db: float = -45.0,
    max_pack_gap: float = 2.0,
    bridge_seconds: float = 0.5,
    margin_seconds: float = 0.2,
    min_seconds: float = 0.3,
) -> list[Segment]:
    total = wave.numel()
    if total == 0:
        return []
    hop = int(sr * HOP_S)
    energy = frame_energy_db(wave, sr)
    n = energy.numel()
    threshold = max(silence_db, float(energy.max()) - 50.0)
    regions = _active_regions(
        (energy > threshold).tolist(), int(bridge_seconds / HOP_S), int(margin_seconds / HOP_S), n
    )
    max_f = int(max_seconds / HOP_S)
    packed: list[tuple[int, int]] = []
    for s, e in regions:
        if packed and s - packed[-1][1] <= max_pack_gap / HOP_S and e - packed[-1][0] <= max_f:
            packed[-1] = (packed[-1][0], e)
        else:
            packed.append((s, e))
    segments = []
    for s, e in packed:
        for a, b in _cut_long(energy, s, e, max_f, int(search_seconds / HOP_S)):
            start, end = a * hop, min(total, b * hop) if b < n else total
            if end - start >= min_seconds * sr:
                segments.append(Segment(start, end))
    return segments
