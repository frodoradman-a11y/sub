"""CTC decoding and forced alignment."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Hypothesis:
    tokens: list[int]
    frames: list[int]  # start frame of every token
    ends: list[int] = field(default_factory=list)  # end frame (exclusive) of every token
    confidence: float = 1.0


def ctc_greedy(log_probs: torch.Tensor, lengths: torch.Tensor, blank: int = 0) -> list[Hypothesis]:
    """Best-path decoding. Confidence = mean max-probability of the emitted frames."""
    best_lp, best = log_probs.max(-1)
    hyps = []
    for b in range(log_probs.shape[0]):
        T = int(lengths[b])
        ids = best[b, :T].tolist()
        lps = best_lp[b, :T]
        tokens, frames, ends, probs = [], [], [], []
        prev = blank
        for t, k in enumerate(ids):
            if k != blank and k != prev:
                tokens.append(k)
                frames.append(t)
                ends.append(t + 1)
                probs.append(float(lps[t].exp()))
            elif k != blank and k == prev:
                ends[-1] = t + 1
            prev = k
        conf = sum(probs) / len(probs) if probs else float(lps.exp().mean()) if T else 1.0
        hyps.append(Hypothesis(tokens, frames, ends, conf))
    return hyps


def ctc_collapse(ids: list[int], blank: int) -> list[int]:
    out, prev = [], None
    for k in ids:
        if k != blank and k != prev:
            out.append(k)
        prev = k
    return out


def forced_align(log_probs: torch.Tensor, targets: list[int], blank: int = 0) -> list[tuple[int, int]] | None:
    """Viterbi alignment of ``targets`` to ``log_probs`` (T, V).

    Returns ``(start, end)`` frames (end exclusive) for every target token, or None
    when the sequence is too long for the number of frames.
    """
    T = log_probs.shape[0]
    U = len(targets)
    if U == 0:
        return []
    S = 2 * U + 1
    labels = torch.full((S,), blank, dtype=torch.long)
    labels[1::2] = torch.tensor(targets, dtype=torch.long)
    repeats = sum(1 for i in range(1, U) if targets[i] == targets[i - 1])
    if T < U + repeats:
        return None
    lp = log_probs.float().cpu()[:, labels]  # (T, S)
    neg = torch.tensor(-1e30)
    # s-2 transition allowed for label states whose label differs from two states back.
    skip_ok = torch.zeros(S, dtype=torch.bool)
    skip_ok[3::2] = labels[3::2] != labels[1:-2:2]
    alpha = torch.full((S,), -1e30)
    alpha[0] = lp[0, 0]
    alpha[1] = lp[0, 1]
    back = torch.zeros(T, S, dtype=torch.long)
    for t in range(1, T):
        stay = alpha
        step = torch.cat([neg[None], alpha[:-1]])
        skip = torch.where(skip_ok, torch.cat([neg[None].repeat(2), alpha[:-2]]), neg)
        cand = torch.stack([stay, step, skip])
        best, arg = cand.max(0)
        alpha = best + lp[t]
        back[t] = arg
    s = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    path = [0] * T
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
    spans: list[list[int]] = [[-1, -1] for _ in range(U)]
    for t, s in enumerate(path):
        if s % 2 == 1:
            u = s // 2
            if spans[u][0] < 0:
                spans[u][0] = t
            spans[u][1] = t + 1
    return [(a, b) for a, b in spans]
