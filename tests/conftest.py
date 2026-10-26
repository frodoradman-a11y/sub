"""Shared fixtures.

Synthetic "tone language": every Persian letter is a 110 ms tone with its own
frequency and a space is 70 ms of silence. A small model can learn it in a few
hundred CPU steps, which lets the tests check real training, decoding and
timestamps without downloading any data.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from subasr.config import config_from_dict
from subasr.text.tokenizer import LLM_SPECIALS, CharTokenizer

SR = 16000
LETTERS = "ابتدرزسلمنوهی"
WORDS = ["سلام", "دنیا", "بازی", "روز", "نان", "باران", "دست", "سرد", "مهر", "یاد"]
CHAR_SEC = 0.11
SPACE_SEC = 0.07


def tone_for(ch: str) -> float:
    return 300.0 + 120.0 * LETTERS.index(ch)


def synth(text: str, lead: float = 0.15, tail: float = 0.15, seed: int = 0, pause_prob: float = 0.0) -> np.ndarray:
    """``pause_prob``: chance that a space becomes a 0.3-1.6 s pause (as between phrases)."""
    rng = np.random.default_rng(seed)
    parts = [np.zeros(int(lead * SR))]
    for ch in text:
        if ch == " ":
            pause = rng.uniform(0.3, 1.6) if rng.random() < pause_prob else SPACE_SEC
            parts.append(np.zeros(int(pause * SR)))
            continue
        n = int(CHAR_SEC * SR)
        t = np.arange(n) / SR
        env = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)
        parts.append(0.5 * env * np.sin(2 * math.pi * tone_for(ch) * t))
    parts.append(np.zeros(int(tail * SR)))
    wave = np.concatenate(parts)
    return (wave + 0.003 * rng.standard_normal(wave.shape)).astype(np.float32)


def random_sentence(rng: random.Random, n_min: int = 1, n_max: int = 3) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(n_min, n_max)))


def write_dataset(root: Path, name: str, n: int, seed: int) -> Path:
    rng = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        text = random_sentence(rng, 1, 4)
        wave = synth(text, lead=rng.uniform(0.0, 0.6), tail=rng.uniform(0.0, 0.6), seed=seed * 1000 + i, pause_prob=0.3)
        path = root / f"{name}_{i:04d}.wav"
        sf.write(path, wave, SR)
        lines.append({"audio": path.name, "text": text, "duration": len(wave) / SR, "language": "fa"})
    manifest = root / f"{name}.jsonl"
    manifest.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")
    return manifest


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("data")
    write_dataset(root, "train", 96, seed=1)
    write_dataset(root, "dev", 8, seed=2)
    chars = sorted(set(LETTERS) | {" "})
    CharTokenizer(chars).save(root / "vocab.txt")
    CharTokenizer(chars, specials=LLM_SPECIALS).save(root / "llm_vocab.txt")
    return root


def tiny_config_dict(data_dir: Path, **train) -> dict:
    return {
        "model": {
            "ctc_tokenizer": {"type": "char", "path": str(data_dir / "vocab.txt")},
            "encoder": {
                "type": "conformer",
                "n_mels": 64,
                "d_model": 96,
                "n_layers": 3,
                "n_heads": 4,
                "conv_kernel": 15,
                "dropout": 0.05,
                "adapter_layers": [1, -1],
                "spec_augment": {"enabled": False},
            },
            "heads": {"pred_dim": 64, "joint_dim": 64},
            "adapter": {"type": "window_qformer", "window_frames": 25, "num_queries": 12, "dim": 64, "n_layers": 1, "n_heads": 4},
            "llm": {
                "type": "tiny",
                "tokenizer": str(data_dir / "llm_vocab.txt"),
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "freeze": False,
                "lora": {"enabled": False},
                "prompt_template": "<bos>{audio}",
                "max_new_tokens": 40,
            },
        },
        "train": {
            "train_manifest": str(data_dir / "train.jsonl"),
            "val_manifest": str(data_dir / "dev.jsonl"),
            "num_workers": 0,
            "precision": "fp32",
            "batch_seconds": 30,
            "max_batch_size": 8,
            "log_every": 50,
            "eval_every": 10**9,
            "save_every": 10**9,
            "augment": {"enabled": False},
            **train,
        },
    }


@pytest.fixture
def tiny_cfg(data_dir, tmp_path):
    return config_from_dict(tiny_config_dict(data_dir, output_dir=str(tmp_path / "run")))


@pytest.fixture(autouse=True)
def _seed():
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
