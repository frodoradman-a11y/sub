"""Loading Gemma 4 once and sharing its parts.

The encoder (audio tower), the native audio projector and the language model all
come from the same checkpoint. ``load_gemma4`` loads it a single time per build and
drops the vision tower, which this project does not use.
"""

from __future__ import annotations

import torch

_CACHE: dict[tuple[str, str], torch.nn.Module] = {}

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def load_gemma4(path: str, dtype: str = "float32"):
    key = (path, dtype)
    if key not in _CACHE:
        from transformers import Gemma4ForConditionalGeneration

        model = Gemma4ForConditionalGeneration.from_pretrained(path, dtype=DTYPES[dtype])
        model.model.vision_tower = None
        model.model.embed_vision = None
        _CACHE[key] = model
    return _CACHE[key]


def clear_cache() -> None:
    _CACHE.clear()


def load_gemma4_feature_extractor(path: str):
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor

    try:
        return Gemma4AudioFeatureExtractor.from_pretrained(path)
    except (OSError, ValueError):
        return Gemma4AudioFeatureExtractor()
