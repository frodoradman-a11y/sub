"""Word / character error rates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from subasr.text.normalize import normalize_for_scoring


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Levenshtein distance (substitution, insertion, deletion all cost 1)."""
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


@dataclass
class ErrorRate:
    errors: int = 0
    total: int = 0

    @property
    def rate(self) -> float:
        return self.errors / max(self.total, 1)

    def add(self, errors: int, total: int) -> None:
        self.errors += errors
        self.total += total


@dataclass
class Scores:
    wer: ErrorRate
    cer: ErrorRate
    utterances: int

    def as_dict(self) -> dict[str, float]:
        return {"wer": self.wer.rate, "cer": self.cer.rate, "words": self.wer.total, "utterances": self.utterances}


def score(refs: Sequence[str], hyps: Sequence[str], zwnj: str = "remove") -> Scores:
    """Corpus-level WER and CER (CER ignores spaces) after ``normalize_for_scoring``."""
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must have the same length")
    wer, cer = ErrorRate(), ErrorRate()
    for ref, hyp in zip(refs, hyps):
        r = normalize_for_scoring(ref, zwnj)
        h = normalize_for_scoring(hyp, zwnj)
        wer.add(edit_distance(r.split(), h.split()), len(r.split()))
        rc, hc = r.replace(" ", ""), h.replace(" ", "")
        cer.add(edit_distance(rc, hc), len(rc))
    return Scores(wer, cer, len(refs))


def cer_between(a: str, b: str) -> float:
    """Character error rate of ``b`` against ``a`` (used by the hallucination guard)."""
    ra = normalize_for_scoring(a).replace(" ", "")
    rb = normalize_for_scoring(b).replace(" ", "")
    if not ra:
        return 0.0 if not rb else 1.0
    return edit_distance(ra, rb) / len(ra)
