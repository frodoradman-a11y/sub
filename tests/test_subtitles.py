import torch

from subasr.model import Word
from subasr.subtitles import CueStyle, build_cues, music_regions, to_srt, to_vtt, write_subtitles


def _words(text: str, start: float = 0.0, step: float = 0.4, gap: float = 0.0):
    out, t = [], start
    for w in text.split():
        out.append(Word(w, t, t + step - 0.05))
        t += step + gap
    return out


def test_cues_respect_line_length_and_duration():
    words = _words(" ".join(["کلمه‌ای"] * 40))
    style = CueStyle(max_chars=20, max_lines=2, max_duration=4.0)
    cues = build_cues(words, style=style)
    assert sum(len(" ".join(c.lines).split()) for c in cues) == 40
    for c in cues:
        assert len(c.lines) <= 2
        assert all(len(line) <= 20 for line in c.lines)
        assert c.end - c.start <= 4.0 + 1e-6
    assert all(a.end <= b.start for a, b in zip(cues, cues[1:]))


def test_pause_and_sentence_end_split_cues():
    words = _words("سلام دنیا.") + _words("روز خوب", start=5.0)
    cues = build_cues(words)
    assert [c.text for c in cues] == ["سلام دنیا.", "روز خوب"]


def test_short_cues_are_extended():
    cues = build_cues([Word("سلام", 1.0, 1.2)])
    assert abs(cues[0].end - cues[0].start - 1.0) < 1e-6


def test_srt_and_vtt_format(tmp_path):
    cues = build_cues(_words("سلام دنیا"), music=[(3.0, 6.0)])
    srt = to_srt(cues)
    assert srt.startswith("1\n00:00:00,000 --> ")
    assert "2\n00:00:03,000 --> 00:00:06,000\n♪ [موسیقی] ♪" in srt
    vtt = to_vtt(cues, CueStyle(rtl_embedding=True))
    assert vtt.startswith("WEBVTT") and "00:00:03.000 --> 00:00:06.000" in vtt and "‫" in vtt
    write_subtitles(cues, tmp_path / "x.srt")
    assert (tmp_path / "x.srt").read_text(encoding="utf-8") == srt


def test_music_regions_skip_words_and_short_regions():
    labels = ["speech", "singing", "music"]
    probs = torch.zeros(250, 3)
    probs[0:100, 2] = 0.9  # 4 s music
    probs[150:160, 2] = 0.9  # 0.4 s: too short
    probs[200:250, 2] = 0.9  # music under words
    words = [Word("x", 18.2, 19.5)]  # absolute times
    regions = music_regions(probs, 25.0, labels, words, offset=10.0)
    assert regions == [(10.0, 14.0)]
