import itertools
import math

import torch

from subasr.decoding.ctc import ctc_collapse, ctc_greedy, forced_align
from subasr.decoding.tdt import tdt_greedy
from subasr.losses.tdt import tdt_loss
from subasr.modules.heads import TDTHead


def brute_force_tdt(tok_lp, dur_lp, target, T, durations, blank=0):
    """Sum over all TDT paths by recursion (tiny sizes only)."""
    U = len(target)

    def rec(t, u):
        if t == T and u == U:
            return 0.0
        if t >= T:
            return -math.inf
        total = -math.inf
        for i, d in enumerate(durations):
            # Blank arcs (d > 0) may land on T only to finish; label arcs must stay inside.
            if d > 0 and (t + d < T or (t + d == T and u == U)):
                total = _lae(total, float(tok_lp[t, u, blank] + dur_lp[t, u, i]) + rec(t + d, u))
            if u < U and t + d < T:
                total = _lae(total, float(tok_lp[t, u, target[u]] + dur_lp[t, u, i]) + rec(t + d, u + 1))
        return total

    return rec(0, 0)


def _lae(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def test_tdt_loss_matches_brute_force():
    torch.manual_seed(0)
    durations = [0, 1, 2, 3]
    V = 4
    for T, target in [(4, [1, 2]), (5, [3]), (3, [1, 1, 2]), (4, [])]:
        U = len(target)
        tok = torch.randn(1, T, U + 1, V).log_softmax(-1)
        dur = torch.randn(1, T, U + 1, len(durations)).log_softmax(-1)
        tgt = torch.tensor([target if target else [0]])
        nll = tdt_loss(tok, dur, tgt, torch.tensor([T]), torch.tensor([U]), durations)
        ref = -brute_force_tdt(tok[0].double(), dur[0].double(), target, T, durations)
        assert abs(float(nll) - ref) < 1e-4, (T, target, float(nll), ref)


def test_tdt_loss_batched_equals_single_and_has_gradients():
    torch.manual_seed(1)
    durations = [0, 1, 2]
    V, T, U = 5, 6, 3
    tok = torch.randn(2, T, U + 1, V, requires_grad=True)
    dur = torch.randn(2, T, U + 1, 3, requires_grad=True)
    tgt = torch.tensor([[1, 2, 3], [4, 1, 0]])
    tl = torch.tensor([3, 2])
    fl = torch.tensor([6, 4])
    nll = tdt_loss(tok.log_softmax(-1), dur.log_softmax(-1), tgt, fl, tl, durations)
    single = tdt_loss(
        tok[1:, :4, :3].log_softmax(-1), dur[1:, :4, :3].log_softmax(-1), tgt[1:, :2], torch.tensor([4]), torch.tensor([2]), durations
    )
    assert abs(float(nll[1].detach()) - float(single[0].detach())) < 1e-4
    nll.sum().backward()
    assert torch.isfinite(tok.grad).all() and torch.isfinite(dur.grad).all()
    assert tok.grad.abs().sum() > 0


def test_tdt_loss_impossible_alignment_is_zero():
    durations = [1]  # no zero-duration arcs: needs T >= U + 1 frames
    tok = torch.randn(1, 2, 4, 5).log_softmax(-1)
    dur = torch.zeros(1, 2, 4, 1)
    nll = tdt_loss(tok, dur, torch.tensor([[1, 2, 3]]), torch.tensor([2]), torch.tensor([3]), durations)
    assert float(nll) == 0.0


def test_ctc_greedy_and_collapse():
    lp = torch.full((1, 6, 4), -10.0)
    for t, k in enumerate([0, 1, 1, 0, 2, 2]):
        lp[0, t, k] = 0.0
    hyp = ctc_greedy(lp, torch.tensor([6]))[0]
    assert hyp.tokens == [1, 2]
    assert hyp.frames == [1, 4] and hyp.ends == [3, 6]
    assert ctc_collapse([0, 3, 3, 0, 3, 1], 0) == [3, 3, 1]


def brute_force_align(lp, targets, blank=0):
    T, _ = lp.shape
    best, best_path = -math.inf, None
    for path in itertools.product(range(lp.shape[1]), repeat=T):
        if ctc_collapse(list(path), blank) == targets:
            s = sum(float(lp[t, k]) for t, k in enumerate(path))
            if s > best:
                best, best_path = s, path
    return best_path


def test_forced_align_matches_brute_force():
    torch.manual_seed(3)
    for targets in ([1, 2], [1, 1], [2, 1, 2]):
        lp = torch.randn(6, 3).log_softmax(-1)
        spans = forced_align(lp, targets)
        path = brute_force_align(lp, targets)
        # Rebuild the Viterbi path from the spans (other frames are blank) and compare
        # its score with the exhaustive optimum (ties may pick a different path).
        rebuilt = [0] * 6
        for (s, e), k in zip(spans, targets):
            assert 0 <= s < e <= 6
            for t in range(s, e):
                rebuilt[t] = k
        assert ctc_collapse(rebuilt, 0) == targets
        score = sum(float(lp[t, rebuilt[t]]) for t in range(6))
        best = sum(float(lp[t, k]) for t, k in enumerate(path))
        assert score >= best - 1e-5


def test_forced_align_too_short_returns_none():
    assert forced_align(torch.zeros(2, 3), [1, 1]) is None
    assert forced_align(torch.zeros(5, 3), []) == []


def test_tdt_greedy_timestamps_are_monotonic():
    torch.manual_seed(0)
    head = TDTHead(16, 6, [0, 1, 2, 3], 8, 8)
    hyps = tdt_greedy(head, torch.randn(2, 30, 16), torch.tensor([30, 12]), max_symbols=3)
    for h, T in zip(hyps, (30, 12)):
        assert h.frames == sorted(h.frames)
        assert all(0 <= f < T for f in h.frames)
        assert all(e > s for s, e in zip(h.frames, h.ends))
        assert len(h.tokens) <= T * 3
