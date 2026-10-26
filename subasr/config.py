"""Typed configuration loaded from YAML.

Every field has a default so a YAML file only needs the values it changes.
Command-line overrides use dotted keys: ``train.lr=1e-4 model.llm.lora.r=32``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SpecAugmentConfig:
    enabled: bool = True
    freq_masks: int = 2
    freq_width: int = 20
    time_masks: int = 10
    time_width_ratio: float = 0.05


@dataclass
class EncoderConfig:
    # "conformer": small Conformer trained from scratch (tests, experiments).
    # "gemma4": the Gemma 4 audio tower (USM-style Conformer, 25 frames/s).
    type: str = "conformer"
    # HF model id or local directory for type=gemma4 (shared with llm.pretrained).
    pretrained: str | None = None
    # Conformer settings (type=conformer).
    n_mels: int = 80
    d_model: int = 144
    n_layers: int = 4
    n_heads: int = 4
    ff_mult: int = 4
    conv_kernel: int = 15
    dropout: float = 0.1
    # Encoder layers whose hidden states are concatenated for the adapter (python indexing).
    adapter_layers: list[int] = field(default_factory=lambda: [-1])
    freeze: bool = False
    gradient_checkpointing: bool = False
    spec_augment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)


@dataclass
class HeadsConfig:
    ctc: bool = True
    tdt: bool = True
    events: bool = True
    tdt_durations: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    pred_dim: int = 320
    joint_dim: int = 320
    max_symbols_per_frame: int = 8
    event_labels: list[str] = field(default_factory=lambda: ["speech", "singing", "music", "noise", "laughter"])


@dataclass
class AdapterConfig:
    # "window_qformer": learned queries per window (controls LLM tokens per second).
    # "gemma_native": Gemma 4's pretrained audio projector on the last encoder layer (25 tokens/s).
    type: str = "window_qformer"
    window_frames: int = 25
    num_queries: int = 12
    dim: int = 256
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1


@dataclass
class LoraConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    targets: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class LLMConfig:
    # "none": encoder-only model (fast mode only).
    # "tiny": small randomly initialised decoder (tests).
    # "gemma4": Gemma 4 text model.
    type: str = "none"
    pretrained: str | None = None
    tokenizer: str | None = None  # HF tokenizer path (defaults to pretrained); a vocab file for type=tiny
    dtype: str = "float32"
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    freeze: bool = True  # base weights; LoRA adapters stay trainable
    gradient_checkpointing: bool = False
    lora: LoraConfig = field(default_factory=LoraConfig)
    # Prompt. "chat" uses the tokenizer's chat template; otherwise a format string with {instruction}.
    prompt_template: str = "chat"
    instruction: str = "Transcribe the speech in this audio verbatim. Language: {language}."
    context_instruction: str = "Previous subtitle: {context}"
    glossary_instruction: str = "Names and terms that may appear: {glossary}"
    max_new_tokens: int = 200
    # Edit (NLE) mode: number of empty slots inserted after every hypothesis token.
    nle_slots_per_token: int = 1
    nle_extra_slots: int = 4


@dataclass
class TokenizerConfig:
    type: str = "char"  # "char" | "spm"
    path: str | None = None  # vocab.txt for char, model file for spm


@dataclass
class ModelConfig:
    sample_rate: int = 16000
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ctc_tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)


@dataclass
class AugmentConfig:
    enabled: bool = True
    music_manifest: str | None = None  # JSONL with {"audio": path} lines (music without vocals or with)
    noise_manifest: str | None = None
    music_prob: float = 0.3
    music_snr_db: list[float] = field(default_factory=lambda: [-5.0, 20.0])
    noise_prob: float = 0.2
    noise_snr_db: list[float] = field(default_factory=lambda: [0.0, 30.0])
    speed_prob: float = 0.3
    speeds: list[float] = field(default_factory=lambda: [0.9, 1.1])
    gain_db: list[float] = field(default_factory=lambda: [-6.0, 6.0])


@dataclass
class LossWeights:
    ctc: float = 0.3
    tdt: float = 1.0
    events: float = 0.1
    ar: float = 1.0
    nle: float = 0.5


@dataclass
class TrainConfig:
    # "encoder": encoder + heads (no LLM loaded)
    # "align":   adapter only, encoder and LLM frozen
    # "joint":   encoder (low LR) + heads + adapter + LoRA
    stage: str = "encoder"
    train_manifest: str = ""
    val_manifest: str | None = None
    output_dir: str = "runs/exp"
    init_from: str | None = None
    max_steps: int = 100000
    batch_seconds: float = 400.0  # total audio seconds per batch
    max_batch_size: int = 64
    min_duration: float = 0.3
    max_duration: float = 30.0
    grad_accum: int = 1
    lr: float = 1e-3  # heads / adapter / LoRA
    encoder_lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    precision: str = "bf16"  # "bf16" | "fp32" (bf16 only used on CUDA)
    num_workers: int = 4
    seed: int = 1234
    log_every: int = 50
    eval_every: int = 2000
    save_every: int = 2000
    eval_max_utts: int = 500
    nle_corrupt_prob: float = 0.5  # chance to build the edit hypothesis from a corrupted reference
    default_language: str = "fa"
    loss: LossWeights = field(default_factory=LossWeights)
    augment: AugmentConfig = field(default_factory=AugmentConfig)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _build(cls, data: dict[str, Any] | None):
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")
    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in data.items():
        if key not in fields:
            raise KeyError(f"unknown config key '{key}' for {cls.__name__}")
        ftype = fields[key].type
        sub = _DATACLASSES.get(ftype if isinstance(ftype, str) else getattr(ftype, "__name__", ""))
        kwargs[key] = _build(sub, value) if sub is not None and isinstance(value, dict) else value
    return cls(**kwargs)


_DATACLASSES = {
    c.__name__: c
    for c in (
        SpecAugmentConfig,
        EncoderConfig,
        HeadsConfig,
        AdapterConfig,
        LoraConfig,
        LLMConfig,
        TokenizerConfig,
        ModelConfig,
        AugmentConfig,
        LossWeights,
        TrainConfig,
        Config,
    )
}


def config_from_dict(data: dict[str, Any]) -> Config:
    return _build(Config, data)


def _apply_override(data: dict[str, Any], dotted: str) -> None:
    key, _, raw = dotted.partition("=")
    if not _:
        raise ValueError(f"override must look like key=value, got '{dotted}'")
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            value = float(value)  # YAML 1.1 reads "3e-4" as a string
        except ValueError:
            pass
    node = data
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> Config:
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    for item in overrides or []:
        _apply_override(data, item)
    return config_from_dict(data)


def config_to_dict(cfg: Config) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
