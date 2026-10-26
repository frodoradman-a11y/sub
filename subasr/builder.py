"""Building models from a config and saving / loading checkpoints.

A checkpoint stores the config, the character vocabularies (so it is
self-contained) and every weight except the frozen pretrained LLM. The frozen
Gemma weights are re-loaded from ``model.llm.pretrained`` when the checkpoint is
opened.
"""

from __future__ import annotations

from pathlib import Path

import torch

from subasr.config import Config, config_from_dict, config_to_dict
from subasr.model import SubASRModel
from subasr.text.tokenizer import (
    LLM_SPECIALS,
    CharTokenizer,
    build_ctc_tokenizer,
    build_llm_tokenizer,
)


def build_tokenizers(cfg: Config, ctc_vocab: list[str] | None = None, llm_vocab: list[str] | None = None):
    m = cfg.model
    if ctc_vocab is not None:
        ctc_tok = CharTokenizer(ctc_vocab[2:])
    else:
        ctc_tok = build_ctc_tokenizer(m.ctc_tokenizer.type, m.ctc_tokenizer.path)
    llm_tok = None
    if m.llm.type != "none":
        if llm_vocab is not None:
            llm_tok = CharTokenizer(llm_vocab[len(LLM_SPECIALS) :], specials=LLM_SPECIALS)
        else:
            llm_tok = build_llm_tokenizer(m.llm.type, m.llm.tokenizer or m.llm.pretrained)
    return ctc_tok, llm_tok


def build_model(cfg: Config, build_llm: bool = True, ctc_vocab=None, llm_vocab=None) -> SubASRModel:
    ctc_tok, llm_tok = build_tokenizers(cfg, ctc_vocab, llm_vocab)
    return SubASRModel(cfg.model, ctc_tok, llm_tok, build_llm=build_llm)


def _saved_keys(model: SubASRModel) -> set[str]:
    keys = set()
    llm_frozen = model.cfg.llm.freeze
    for name, _ in model.named_parameters():
        if name.startswith("llm.") and llm_frozen and "lora_" not in name:
            continue
        keys.add(name)
    for name, _ in model.named_buffers():
        if not name.startswith("llm."):
            keys.add(name)
    return keys


def save_checkpoint(path: str | Path, model: SubASRModel, cfg: Config, step: int, extra: dict | None = None) -> None:
    keys = _saved_keys(model)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keys}
    payload = {
        "config": config_to_dict(cfg),
        "state": state,
        "step": step,
        "ctc_vocab": model.ctc_tok.vocab if isinstance(model.ctc_tok, CharTokenizer) else None,
        "llm_vocab": model.llm_tok.vocab if isinstance(model.llm_tok, CharTokenizer) else None,
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_weights(model: SubASRModel, state: dict[str, torch.Tensor]) -> list[str]:
    """Load the matching weights; returns the keys of ``state`` that were not used."""
    own = model.state_dict()
    usable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(usable, strict=False)
    return sorted(set(state) - set(usable))


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu", build_llm: bool = True, config_overrides=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    data = payload["config"]
    if config_overrides:
        from subasr.config import _apply_override

        for item in config_overrides:
            _apply_override(data, item)
    cfg = config_from_dict(data)
    model = build_model(cfg, build_llm=build_llm, ctc_vocab=payload.get("ctc_vocab"), llm_vocab=payload.get("llm_vocab"))
    unused = [k for k in load_weights(model, payload["state"]) if build_llm or not k.startswith(("llm.", "adapter."))]
    if unused:
        raise RuntimeError(f"checkpoint weights not used by the model: {unused[:10]}")
    return model.to(device).eval(), cfg, payload.get("step", 0)
