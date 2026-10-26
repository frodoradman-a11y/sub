"""Building readable subtitle cues from timed words and writing SRT / WebVTT.

Rules (configurable): at most ``max_lines`` lines of ``max_chars`` characters,
cues between ``min_duration`` and ``max_duration`` seconds, a new cue after a pause
longer than ``max_gap`` and, when possible, at the end of a sentence.
Stretches of music without words become a "♪ [موسیقی] ♪" cue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from subasr.model import Word

SENTENCE_END = (".", "!", "?", "؟", "…")
CLAUSE_END = (",", "،", "؛", ";", ":")
RLE, PDF = "‫", "‬"


@dataclass
class CueStyle:
    max_chars: int = 42
    max_lines: int = 2
    max_duration: float = 7.0
    min_duration: float = 1.0
    max_gap: float = 1.0
    music_label: str = "♪ [موسیقی] ♪"
    rtl_embedding: bool = False  # wrap lines in RLE/PDF for players with weak bidi support


@dataclass
class Cue:
    start: float
    end: float
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _wrap(words: list[str], max_chars: int, max_lines: int) -> list[str] | None:
    """Balanced line breaking; None when the words do not fit."""
    text = " ".join(words)
    if len(text) <= max_chars:
        return [text]
    if max_lines < 2:
        return None
    best = None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        if len(a) <= max_chars and len(b) <= max_chars:
            diff = abs(len(a) - len(b))
            if best is None or diff < best[0]:
                best = (diff, [a, b])
    return best[1] if best else None


def build_cues(words: list[Word], music: list[tuple[float, float]] | None = None, style: CueStyle | None = None) -> list[Cue]:
    style = style or CueStyle()
    cues: list[Cue] = []
    group: list[Word] = []

    def flush():
        if group:
            lines = _wrap([w.text for w in group], style.max_chars, style.max_lines) or [" ".join(w.text for w in group)]
            cues.append(Cue(group[0].start, group[-1].end, lines))
            group.clear()

    for w in words:
        if group:
            gap = w.start - group[-1].end
            too_long = w.end - group[0].start > style.max_duration
            no_fit = _wrap([g.text for g in group] + [w.text], style.max_chars, style.max_lines) is None
            if gap > style.max_gap or too_long or no_fit:
                # Prefer to break after the last sentence / clause end inside the group.
                cut = None
                if no_fit or too_long:
                    for i in range(len(group) - 1, 0, -1):
                        if group[i - 1].text.endswith(SENTENCE_END + CLAUSE_END):
                            cut = i
                            break
                if cut is not None and cut < len(group):
                    rest = group[cut:]
                    del group[cut:]
                    flush()
                    group.extend(rest)
                else:
                    flush()
        group.append(w)
        if w.text.endswith(SENTENCE_END) and w.end - group[0].start >= style.min_duration * 2:
            flush()
    flush()

    for s, e in music or []:
        cues.append(Cue(s, e, [style.music_label]))
    cues.sort(key=lambda c: c.start)
    return _fix_timing(cues, style)


def _fix_timing(cues: list[Cue], style: CueStyle) -> list[Cue]:
    for i, c in enumerate(cues):
        if c.end - c.start < style.min_duration:
            limit = cues[i + 1].start - 0.05 if i + 1 < len(cues) else c.start + style.min_duration
            c.end = max(c.end, min(c.start + style.min_duration, limit))
        if i + 1 < len(cues) and c.end > cues[i + 1].start:
            c.end = max(c.start + 0.1, cues[i + 1].start - 0.05)
    return cues


def music_regions(
    frame_probs, frame_rate: float, labels: list[str], words: list[Word], offset: float = 0.0, threshold: float = 0.5, min_len: float = 1.5
) -> list[tuple[float, float]]:
    """Music/singing stretches without recognised words (seconds, absolute)."""
    if frame_probs is None:
        return []
    idx = [labels.index(n) for n in ("music",) if n in labels]
    speech = [labels.index(n) for n in ("speech", "singing") if n in labels]
    if not idx:
        return []
    is_music = (frame_probs[:, idx].max(-1).values > threshold)
    if speech:
        is_music &= frame_probs[:, speech].max(-1).values < threshold
    regions, start = [], None
    for t, m in enumerate(is_music.tolist() + [False]):
        if m and start is None:
            start = t
        elif not m and start is not None:
            regions.append((offset + start / frame_rate, offset + t / frame_rate))
            start = None
    out = []
    for s, e in regions:
        if e - s < min_len:
            continue
        if any(w.start < e and w.end > s for w in words):
            continue
        out.append((s, e))
    return out


def _ts(seconds: float, sep: str) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _line(text: str, style: CueStyle | None) -> str:
    return f"{RLE}{text}{PDF}" if style is not None and style.rtl_embedding else text


def to_srt(cues: list[Cue], style: CueStyle | None = None) -> str:
    blocks = []
    for i, c in enumerate(cues, 1):
        body = "\n".join(_line(l, style) for l in c.lines)
        blocks.append(f"{i}\n{_ts(c.start, ',')} --> {_ts(c.end, ',')}\n{body}\n")
    return "\n".join(blocks)


def to_vtt(cues: list[Cue], style: CueStyle | None = None) -> str:
    blocks = ["WEBVTT\n"]
    for c in cues:
        body = "\n".join(_line(l, style) for l in c.lines)
        blocks.append(f"{_ts(c.start, '.')} --> {_ts(c.end, '.')}\n{body}\n")
    return "\n".join(blocks)


def write_subtitles(cues: list[Cue], path: str | Path, style: CueStyle | None = None) -> None:
    path = Path(path)
    text = to_vtt(cues, style) if path.suffix.lower() == ".vtt" else to_srt(cues, style)
    path.write_text(text, encoding="utf-8")
