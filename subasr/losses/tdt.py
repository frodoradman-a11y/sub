"""Token-and-Duration Transducer loss (Xu et al., 2023), pure PyTorch.

Lattice node (t, u): ``t`` frames consumed, ``u`` labels emitted. From (t, u):
* blank with duration d > 0          -> (t + d, u)
* label y_{u+1} with duration d >= 0 -> (t + d, u + 1)
Each arc is scored ``log P_token + log P_duration``. A sequence ends with a blank
that lands exactly on frame T after all labels are emitted.

The loop runs over time only. Zero-duration label arcs inside one frame form a
linear recurrence along u, solved in closed form with ``logcumsumexp``. A fused
CUDA kernel (e.g. NeMo's) is faster for very large vocabularies; this version is
exact and has no extra dependencies.
"""

from __future__ import annotations

import torch

NEG = -1e9


def tdt_loss(
    token_logp: torch.Tensor,
    dur_logp: torch.Tensor,
    targets: torch.Tensor,
    frame_lens: torch.Tensor,
    target_lens: torch.Tensor,
    durations: list[int],
    blank: int = 0,
) -> torch.Tensor:
    """Negative log-likelihood per utterance.

    token_logp: (B, T, U+1, V) log-softmax over tokens.
    dur_logp:   (B, T, U+1, D) log-softmax over ``durations``.
    targets:    (B, U) label ids (padding ignored via ``target_lens``).
    """
    token_logp = token_logp.float()
    dur_logp = dur_logp.float()
    B, T, U1, _ = token_logp.shape
    U = U1 - 1
    lp_blank = token_logp[..., blank]  # (B, T, U+1)
    if U > 0:
        lp_y = token_logp[:, :, :U, :].gather(-1, targets[:, None, :, None].expand(B, T, U, 1)).squeeze(-1)
    else:
        lp_y = token_logp.new_zeros(B, T, 0)

    zero_idx = durations.index(0) if 0 in durations else None
    alphas: list[torch.Tensor] = []
    for t in range(T):
        if t == 0:
            base = token_logp.new_full((B, U1), NEG)
            base[:, 0] = 0.0
        else:
            base = token_logp.new_full((B, U1), NEG)
            for i, d in enumerate(durations):
                if d == 0 or t - d < 0:
                    continue
                prev = alphas[t - d]
                base = torch.logaddexp(base, prev + lp_blank[:, t - d] + dur_logp[:, t - d, :, i])
                if U > 0:
                    emit = prev[:, :U] + lp_y[:, t - d] + dur_logp[:, t - d, :U, i]
                    base = torch.cat([base[:, :1], torch.logaddexp(base[:, 1:], emit)], dim=1)
        if zero_idx is not None and U > 0:
            w = lp_y[:, t] + dur_logp[:, t, :U, zero_idx]  # weight of arc u -> u+1 inside frame t
            c = torch.cat([w.new_zeros(B, 1), w.cumsum(dim=1)], dim=1)
            alpha_t = c + torch.logcumsumexp(base - c, dim=1)
        else:
            alpha_t = base
        alphas.append(alpha_t)
    alpha = torch.stack(alphas, dim=1)  # (B, T, U+1)

    bidx = torch.arange(B, device=alpha.device)
    final = token_logp.new_full((B,), NEG)
    for i, d in enumerate(durations):
        if d == 0:
            continue
        t_idx = frame_lens - d
        ok = t_idx >= 0
        t_safe = t_idx.clamp(min=0)
        term = (
            alpha[bidx, t_safe, target_lens]
            + lp_blank[bidx, t_safe, target_lens]
            + dur_logp[bidx, t_safe, target_lens, i]
        )
        final = torch.where(ok, torch.logaddexp(final, term), final)
    nll = -final
    # Impossible alignments (too few frames): no loss, no gradient.
    return torch.where(final > NEG / 2, nll, torch.zeros_like(nll))
