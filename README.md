# SubASR

Subtitle-oriented speech recognition built on **Gemma 4**: the Gemma 4 audio
encoder with CTC / TDT / audio-event heads for fast transcription and word
timestamps, plus the Gemma 4 text model (LoRA) that edits or rewrites the
transcript for higher quality. Designed for Persian first, works for any language.

```
audio ─► Gemma 4 audio encoder (25 frames/s)
            ├─► TDT head ─────────────► fast mode (no LLM)
            ├─► CTC head ─► hypothesis ─┐
            ├─► audio-event head        │   (speech / singing / music / noise / laughter)
            └─► adapter ─► Gemma 4 LLM ◄┘─► edit mode (one bidirectional pass)
                                        └─► quality mode (autoregressive, with context + glossary)
final text ─► CTC forced alignment ─► word timestamps ─► subtitle cues ─► SRT / VTT
```

## Quick start

```bash
pip install -e ".[dev]"
python -m pytest -q                      # 42 tests, ~3 min on CPU (includes real training)

# data/train.jsonl: {"audio": "clip.wav", "text": "..."} per line
python -m subasr.tools.build_vocab --manifest data/train.jsonl --output data/vocab.txt
python -m subasr.train --config configs/gemma4_e2b.yaml train.stage=encoder train.output_dir=runs/enc
python -m subasr.train --config configs/gemma4_e2b.yaml train.stage=align train.init_from=runs/enc/last.pt train.output_dir=runs/align
python -m subasr.train --config configs/gemma4_e2b.yaml train.stage=joint train.init_from=runs/align/last.pt train.output_dir=runs/joint

python -m subasr.infer --ckpt runs/joint/last.pt --input film.mp4 --output film.fa.srt --mode auto
python -m subasr.evaluate --ckpt runs/joint/last.pt --manifest data/test.jsonl --modes fast edit quality
```

## Documentation

* [docs/guide.md](docs/guide.md) – full guide (Persian): data, configuration, training stages, inference, code map, tests, limitations.
* [docs/architecture-proposal.md](docs/architecture-proposal.md) – the design and the research behind it (Persian).
