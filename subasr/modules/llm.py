"""Language-model backends.

Both backends expose the same small interface:

* ``embed(ids)``                       token embeddings
* ``hidden(input_ids, inputs_embeds, attention_mask, bidirectional)``
                                       final hidden states. ``input_ids`` holds the
                                       real ids at text positions and ``pad_id`` at
                                       audio positions (Gemma's per-layer embeddings
                                       are looked up from ids).
* ``logits(hidden)``                   LM head (applied only where needed: Gemma's
                                       vocabulary has 262k entries)
* ``generate(...)``                    greedy decoding with a KV cache

``bidirectional=True`` drops the causal mask; the edit (NLE) mode uses it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from subasr.config import LLMConfig
from subasr.modules.encoders import _rope


def positions_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


def _repeating(tokens: list[int], n: int = 8) -> bool:
    return len(tokens) >= 2 * n and tokens[-n:] == tokens[-2 * n : -n]


class _TinyBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.n1 = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.n2 = nn.RMSNorm(d)
        self.gate_proj = nn.Linear(d, 2 * d, bias=False)
        self.up_proj = nn.Linear(d, 2 * d, bias=False)
        self.down_proj = nn.Linear(2 * d, d, bias=False)

    def forward(self, x, pos, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k = _rope(q, pos), _rope(k, pos)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.o_proj(y.transpose(1, 2).reshape(B, L, D))
        h = self.n2(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class TinyLM(nn.Module):
    """Small decoder-only LM used for tests and quick experiments."""

    def __init__(self, cfg: LLMConfig, vocab_size: int):
        super().__init__()
        self.hidden_size = cfg.d_model
        self.vocab_size = vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(_TinyBlock(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(ids)

    def hidden(self, input_ids, inputs_embeds, attention_mask, bidirectional: bool = False) -> torch.Tensor:
        B, L, _ = inputs_embeds.shape
        keep = attention_mask.bool()[:, None, None, :]
        if not bidirectional:
            keep = keep & torch.ones(L, L, dtype=torch.bool, device=keep.device).tril()[None, None]
        keep = keep | torch.eye(L, dtype=torch.bool, device=keep.device)[None, None]  # padded rows stay finite
        pos = positions_from_mask(attention_mask)
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x, pos, keep)
        return self.norm(x)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden).float()

    @torch.no_grad()
    def generate(self, input_ids, inputs_embeds, attention_mask, max_new_tokens: int, eos_ids: list[int]):
        B = inputs_embeds.shape[0]
        out: list[list[int]] = [[] for _ in range(B)]
        done = [False] * B
        ids, emb, mask = input_ids, inputs_embeds, attention_mask
        for _ in range(max_new_tokens):
            h = self.hidden(ids, emb, mask)
            nxt = self.logits(h[:, -1]).argmax(-1)
            for b in range(B):
                if not done[b]:
                    tok = int(nxt[b])
                    if tok in eos_ids:
                        done[b] = True
                    else:
                        out[b].append(tok)
                        if _repeating(out[b]):
                            done[b] = True
            if all(done):
                break
            ids = torch.cat([ids, nxt[:, None]], 1)
            emb = torch.cat([emb, self.embed(nxt[:, None])], 1)
            mask = torch.cat([mask, mask.new_ones(B, 1)], 1)
        return out


class Gemma4LM(nn.Module):
    """Gemma 4 text model + LM head, fed with mixed text / audio embeddings."""

    def __init__(self, gemma):
        super().__init__()
        self.lm = gemma.model.language_model
        self.lm_head = gemma.lm_head
        self.text_config = gemma.config.get_text_config()
        self.hidden_size = self.text_config.hidden_size
        self.vocab_size = self.text_config.vocab_size

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm.embed_tokens(ids)

    def _per_layer(self, input_ids):
        if not self.text_config.hidden_size_per_layer_input:
            return None
        return self.lm.get_per_layer_inputs(input_ids, None)

    def _masks(self, inputs_embeds, attention_mask, bidirectional):
        if not bidirectional:
            return attention_mask
        from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask

        # The skip must be disabled: a ``None`` mask makes SDPA fall back to is_causal=True.
        kwargs = dict(
            config=self.text_config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            allow_is_bidirectional_skip=False,
        )
        return {
            "full_attention": create_bidirectional_mask(**kwargs),
            "sliding_attention": create_bidirectional_sliding_window_mask(**kwargs),
        }

    def hidden(self, input_ids, inputs_embeds, attention_mask, bidirectional: bool = False) -> torch.Tensor:
        dtype = self.lm.embed_tokens.weight.dtype
        inputs_embeds = inputs_embeds.to(dtype)
        out = self.lm(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=self._per_layer(input_ids),
            attention_mask=self._masks(inputs_embeds, attention_mask, bidirectional),
            position_ids=positions_from_mask(attention_mask),
            use_cache=False,
        )
        return out.last_hidden_state

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden).float()
        cap = self.text_config.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits

    @torch.no_grad()
    def generate(self, input_ids, inputs_embeds, attention_mask, max_new_tokens: int, eos_ids: list[int]):
        """Greedy decoding; prompts must be left-padded."""
        B = inputs_embeds.shape[0]
        dtype = self.lm.embed_tokens.weight.dtype
        mask = attention_mask
        pos = positions_from_mask(mask)
        out = self.lm(
            inputs_embeds=inputs_embeds.to(dtype),
            per_layer_inputs=self._per_layer(input_ids),
            attention_mask=mask,
            position_ids=pos,
            use_cache=True,
        )
        cache = out.past_key_values
        last_pos = pos[:, -1]
        result: list[list[int]] = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=mask.device)
        eos = torch.tensor(eos_ids, device=mask.device)
        logits = self.logits(out.last_hidden_state[:, -1])
        for _ in range(max_new_tokens):
            nxt = logits.argmax(-1)
            finished_now = torch.isin(nxt, eos) & ~done
            for b in range(B):
                if not done[b] and not finished_now[b]:
                    result[b].append(int(nxt[b]))
                    if _repeating(result[b]):
                        finished_now[b] = True
            done |= finished_now
            if bool(done.all()):
                break
            ids = nxt[:, None]
            mask = torch.cat([mask, mask.new_ones(B, 1)], 1)
            last_pos = last_pos + 1
            out = self.lm(
                inputs_embeds=self.embed(ids),
                per_layer_inputs=self._per_layer(ids),
                attention_mask=mask,
                position_ids=last_pos[:, None],
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            logits = self.logits(out.last_hidden_state[:, -1])
        return result
