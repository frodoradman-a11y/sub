"""Subtitle a video or audio file.

    python -m subasr.infer --ckpt runs/joint/last.pt --input film.mp4 --output film.fa.srt --mode auto

Pipeline: decode audio (ffmpeg) -> split at quiet points (<= 30 s) -> batched
recognition in the chosen mode -> word timestamps -> music regions -> cues -> SRT/VTT.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from subasr.audio.io import load_audio
from subasr.audio.segment import split_audio
from subasr.builder import load_checkpoint
from subasr.model import SubASRModel, Word
from subasr.subtitles import CueStyle, build_cues, music_regions, write_subtitles


@dataclass
class SegmentResult:
    start: float
    end: float
    text: str
    mode: str
    confidence: float


def transcribe_file(
    model: SubASRModel,
    path: str,
    mode: str = "auto",
    language: str = "fa",
    glossary: list[str] | None = None,
    batch_size: int = 8,
    max_segment: float = 30.0,
    use_context: bool = False,
    nonspeech_threshold: float | None = 0.2,
    auto_escalate: str = "edit",
    style: CueStyle | None = None,
):
    """Returns (cues, segment results, all words)."""
    sr = model.cfg.sample_rate
    wave = load_audio(path, sr)
    segments = split_audio(wave, sr, max_seconds=max_segment)
    device = model.device
    if mode in ("edit", "quality", "auto") and not model.has_llm:
        mode = "fast"
    words: list[Word] = []
    seg_results: list[SegmentResult] = []
    music: list[tuple[float, float]] = []
    labels = model.cfg.heads.event_labels
    # With context every segment depends on the previous text, so batches have size 1.
    step = 1 if use_context and mode != "fast" else batch_size
    previous = None
    for i in range(0, len(segments), step):
        chunk = segments[i : i + step]
        lens = torch.tensor([s.end - s.start for s in chunk])
        batch = torch.zeros(len(chunk), int(lens.max()))
        for j, s in enumerate(chunk):
            batch[j, : s.end - s.start] = wave[s.start : s.end]
        out = model.transcribe(
            batch.to(device),
            lens.to(device),
            mode=mode,
            languages=[language] * len(chunk),
            contexts=[previous] * len(chunk) if use_context else None,
            glossaries=[glossary] * len(chunk) if glossary else None,
            nonspeech_threshold=nonspeech_threshold,
            auto_escalate=auto_escalate,
        )
        for s, r in zip(chunk, out):
            offset = s.start / sr
            seg_words = [Word(w.text, w.start + offset, w.end + offset) for w in r.words]
            words += seg_words
            music += music_regions(r.event_probs, r.frame_rate, labels, seg_words, offset)
            seg_results.append(SegmentResult(offset, s.end / sr, r.text, r.mode, r.confidence))
            if r.text:
                previous = r.text
    cues = build_cues(words, music, style)
    return cues, seg_results, words


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate subtitles with SubASR")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="audio or video file")
    ap.add_argument("--output", help="output .srt or .vtt (default: input name + .srt)")
    ap.add_argument("--mode", default="auto", choices=["fast", "edit", "quality", "auto"])
    ap.add_argument("--auto-escalate", default="edit", choices=["edit", "quality"])
    ap.add_argument("--language", default="fa")
    ap.add_argument("--glossary", help="text file with one name / term per line")
    ap.add_argument("--context", action="store_true", help="feed the previous subtitle to the LLM (slower)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-segment", type=float, default=30.0)
    ap.add_argument("--nonspeech-threshold", type=float, default=0.2, help="negative to disable")
    ap.add_argument("--max-chars", type=int, default=42)
    ap.add_argument("--rtl-embedding", action="store_true")
    ap.add_argument("--json", help="also write segment-level results as JSON")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_checkpoint(args.ckpt, device)
    glossary = None
    if args.glossary:
        glossary = [l.strip() for l in Path(args.glossary).read_text(encoding="utf-8").splitlines() if l.strip()]
    style = CueStyle(max_chars=args.max_chars, rtl_embedding=args.rtl_embedding)
    cues, segs, _ = transcribe_file(
        model,
        args.input,
        mode=args.mode,
        language=args.language,
        glossary=glossary,
        batch_size=args.batch_size,
        max_segment=args.max_segment,
        use_context=args.context,
        nonspeech_threshold=None if args.nonspeech_threshold < 0 else args.nonspeech_threshold,
        auto_escalate=args.auto_escalate,
        style=style,
    )
    output = args.output or str(Path(args.input).with_suffix(".srt"))
    write_subtitles(cues, output, style)
    if args.json:
        Path(args.json).write_text(
            json.dumps([s.__dict__ for s in segs], ensure_ascii=False, indent=1), encoding="utf-8"
        )
    modes: dict[str, int] = {}
    for s in segs:
        modes[s.mode] = modes.get(s.mode, 0) + 1
    print(f"wrote {output}: {len(cues)} cues from {len(segs)} segments, modes={modes}")


if __name__ == "__main__":
    main()
