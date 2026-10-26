"""Minimal LoRA for ``nn.Linear`` layers (no external dependency)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from subasr.config import LoraConfig


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        self.lora_a = nn.Linear(base.in_features, r, bias=False, device=base.weight.device)
        self.lora_b = nn.Linear(r, base.out_features, bias=False, device=base.weight.device)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.dropout = nn.Dropout(dropout)
        self.merged = False

    @property
    def weight(self):
        return self.base.weight

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.merged:
            return y
        lx = self.dropout(x).to(self.lora_a.weight.dtype)
        return y + (self.lora_b(self.lora_a(lx)) * self.scale).to(y.dtype)

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            delta = (self.lora_b.weight @ self.lora_a.weight) * self.scale
            self.base.weight += delta.to(self.base.weight.dtype)
            self.merged = True


def inject_lora(root: nn.Module, cfg: LoraConfig) -> list[str]:
    """Wrap every ``nn.Linear`` under ``root`` whose attribute name is in ``cfg.targets``."""
    replaced = []
    for name, module in list(root.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in cfg.targets:
                setattr(module, child_name, LoRALinear(child, cfg.r, cfg.alpha, cfg.dropout))
                replaced.append(f"{name}.{child_name}" if name else child_name)
    return replaced


def lora_parameters(root: nn.Module):
    for module in root.modules():
        if isinstance(module, LoRALinear):
            yield from (module.lora_a.weight, module.lora_b.weight)


def merge_lora(root: nn.Module) -> None:
    for module in root.modules():
        if isinstance(module, LoRALinear):
            module.merge()
