"""Greedy TDT decoding with token timestamps.

The duration output lets the decoder jump over several frames at once, which is
where TDT gets its speed over plain RNN-T, and it gives every token a start frame
and a length.
"""

from __future__ import annotations

import torch

from subasr.decoding.ctc import Hypothesis
from subasr.modules.heads import TDTHead


@torch.no_grad()
def tdt_greedy(head: TDTHead, hidden: torch.Tensor, lengths: torch.Tensor, max_symbols: int = 8) -> list[Hypothesis]:
    enc_all = head.enc_proj(hidden)  # (B, T, J)
    hyps = []
    for b in range(hidden.shape[0]):
        T = int(lengths[b])
        enc = enc_all[b]
        tokens, frames, ends, probs = [], [], [], []
        last = torch.tensor([[head.blank]], device=hidden.device)
        pred, state = head.predictor(last)
        pred_p = head.pred_proj(pred[0, 0])
        t, emitted_here = 0, 0
        while t < T:
            tok_lp, dur_lp = head.joint(enc[t], pred_p)
            k = int(tok_lp.argmax())
            d = head.durations[int(dur_lp.argmax())]
            if k == head.blank:
                t += max(d, 1)
                emitted_here = 0
                continue
            tokens.append(k)
            frames.append(t)
            ends.append(min(t + max(d, 1), T))
            probs.append(float(tok_lp[k].exp()))
            pred, state = head.predictor(torch.tensor([[k]], device=hidden.device), state)
            pred_p = head.pred_proj(pred[0, 0])
            if d == 0:
                emitted_here += 1
                if emitted_here >= max_symbols:
                    t += 1
                    emitted_here = 0
            else:
                t += d
                emitted_here = 0
        conf = sum(probs) / len(probs) if probs else 1.0
        hyps.append(Hypothesis(tokens, frames, ends, conf))
    return hyps
