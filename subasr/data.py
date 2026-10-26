"""Manifest dataset, duration-bucketed batching and collation.

Manifest format: one JSON object per line.

    {"audio": "clips/0001.wav", "text": "سلام دنیا", "duration": 1.8,
     "language": "fa", "offset": 0.0, "labels": ["speech", "music"],
     "context": "previous subtitle line", "glossary": ["تهران"]}

Only ``audio`` and ``text`` are required. Relative paths are resolved against the
manifest's directory. ``text`` may be empty for clips without speech (music,
noise): they teach the model not to invent text. ``labels`` lists the audio events
present; when missing, "speech" is assumed for non-empty text.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from subasr.audio.augment import Augmenter
from subasr.audio.io import audio_duration, load_audio
from subasr.config import AugmentConfig
from subasr.model import Batch


def read_manifest(path: str | Path) -> list[dict]:
    base = Path(path).parent
    items = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "audio" not in item:
                raise ValueError(f"{path}:{n}: missing 'audio'")
            p = Path(item["audio"])
            item["audio"] = str(p if p.is_absolute() else base / p)
            item.setdefault("text", "")
            items.append(item)
    return items


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        event_labels: list[str],
        sample_rate: int = 16000,
        augment: AugmentConfig | None = None,
        min_duration: float = 0.0,
        max_duration: float = float("inf"),
        default_language: str = "fa",
    ):
        self.sr = sample_rate
        self.event_labels = list(event_labels)
        self.default_language = default_language
        self.augmenter = Augmenter(augment, sample_rate) if augment is not None and augment.enabled else None
        items = read_manifest(manifest)
        for item in items:
            if "duration" not in item:
                item["duration"] = audio_duration(item["audio"])
        self.items = [it for it in items if min_duration <= float(it["duration"]) <= max_duration]

    def __len__(self) -> int:
        return len(self.items)

    def durations(self) -> list[float]:
        return [float(it["duration"]) for it in self.items]

    def __getitem__(self, i: int) -> dict:
        item = self.items[i]
        wave = load_audio(item["audio"], self.sr, float(item.get("offset", 0.0)), item.get("duration"))
        text = item["text"]
        labels = set(item.get("labels") or (["speech"] if text.strip() else []))
        if self.augmenter is not None:
            wave, labels = self.augmenter(wave, labels)
        events = torch.tensor([1.0 if lab in labels else 0.0 for lab in self.event_labels])
        return {
            "wave": wave,
            "text": text,
            "language": item.get("language", self.default_language),
            "context": item.get("context"),
            "glossary": item.get("glossary"),
            "events": events,
        }


def collate(items: list[dict]) -> Batch:
    lens = torch.tensor([it["wave"].numel() for it in items], dtype=torch.long)
    wave = torch.zeros(len(items), int(lens.max()))
    for i, it in enumerate(items):
        wave[i, : it["wave"].numel()] = it["wave"]
    return Batch(
        wave=wave,
        wave_lens=lens,
        texts=[it["text"] for it in items],
        languages=[it["language"] for it in items],
        contexts=[it["context"] for it in items],
        glossaries=[it["glossary"] for it in items],
        event_targets=torch.stack([it["events"] for it in items]),
    )


class DurationBatchSampler(Sampler[list[int]]):
    """Batches of similar-length clips whose padded size stays under ``batch_seconds``."""

    def __init__(self, durations: list[float], batch_seconds: float, max_batch_size: int, shuffle: bool = True, seed: int = 0):
        self.durations = durations
        self.batch_seconds = batch_seconds
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        order = sorted(range(len(self.durations)), key=lambda i: self.durations[i])
        batches, cur, longest = [], [], 0.0
        for i in order:
            d = self.durations[i]
            if cur and (max(longest, d) * (len(cur) + 1) > self.batch_seconds or len(cur) >= self.max_batch_size):
                batches.append(cur)
                cur, longest = [], 0.0
            cur.append(i)
            longest = max(longest, d)
        if cur:
            batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._batches()
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self._batches())
