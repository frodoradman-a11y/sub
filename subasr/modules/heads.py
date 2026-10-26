"""Encoder heads: CTC, TDT (token-and-duration transducer) and audio events."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from subasr.losses.tdt import tdt_loss


class CTCHead(nn.Module):
    def __init__(self, dim: int, vocab: int):
        super().__init__()
        self.proj = nn.Linear(dim, vocab)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden).float().log_softmax(-1)

    @staticmethod
    def loss(log_probs, lengths, targets, target_lens, blank: int = 0) -> torch.Tensor:
        per_utt = F.ctc_loss(
            log_probs.transpose(0, 1),
            targets,
            lengths,
            target_lens,
            blank=blank,
            reduction="none",
            zero_infinity=True,
        )
        return per_utt


class TDTHead(nn.Module):
    def __init__(self, dim: int, vocab: int, durations: list[int], pred_dim: int, joint_dim: int, blank: int = 0):
        super().__init__()
        if 1 not in durations:
            raise ValueError("TDT durations must contain 1")
        self.vocab = vocab
        self.durations = list(durations)
        self.blank = blank
        self.embed = nn.Embedding(vocab, pred_dim)
        self.lstm = nn.LSTM(pred_dim, pred_dim, batch_first=True)
        self.enc_proj = nn.Linear(dim, joint_dim)
        self.pred_proj = nn.Linear(pred_dim, joint_dim)
        self.out = nn.Linear(joint_dim, vocab + len(durations))

    def predictor(self, tokens: torch.Tensor, state=None):
        out, state = self.lstm(self.embed(tokens), state)
        return out, state

    def joint(self, enc: torch.Tensor, pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``enc``: (..., J) projected frames, ``pred``: (..., J) projected predictor states (broadcast)."""
        logits = self.out(torch.tanh(enc + pred)).float()
        return logits[..., : self.vocab].log_softmax(-1), logits[..., self.vocab :].log_softmax(-1)

    def loss(self, hidden, lengths, targets, target_lens) -> torch.Tensor:
        B = targets.shape[0]
        start = targets.new_full((B, 1), self.blank)
        pred, _ = self.predictor(torch.cat([start, targets], dim=1))  # (B, U+1, P)
        enc = self.enc_proj(hidden)[:, :, None]  # (B, T, 1, J)
        tok, dur = self.joint(enc, self.pred_proj(pred)[:, None])  # (B, T, U+1, *)
        return tdt_loss(tok, dur, targets, lengths, target_lens, self.durations, self.blank)


class EventHead(nn.Module):
    """Frame-level multi-label audio events (speech, singing, music, noise, laughter)."""

    def __init__(self, dim: int, labels: list[str]):
        super().__init__()
        self.labels = list(labels)
        self.proj = nn.Linear(dim, len(labels))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden).float()

    def loss(self, logits: torch.Tensor, lengths: torch.Tensor, clip_targets: torch.Tensor) -> torch.Tensor:
        """Multiple-instance loss: utterance labels supervise the max over frames.

        A label present in the clip must fire somewhere; an absent label must fire nowhere.
        """
        mask = torch.arange(logits.shape[1], device=logits.device)[None] < lengths[:, None]
        pooled = logits.masked_fill(~mask[..., None], -1e4).amax(dim=1)
        return F.binary_cross_entropy_with_logits(pooled, clip_targets.float(), reduction="none").mean(-1)
