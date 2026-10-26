"""Evaluate a checkpoint on a manifest (WER / CER / speed).

    python -m subasr.evaluate --ckpt runs/joint/last.pt --manifest test.jsonl --modes fast edit quality
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from subasr.builder import load_checkpoint
from subasr.data import DurationBatchSampler, ManifestDataset, collate
from subasr.metrics import score


def run(ckpt: str, manifest: str, modes: list[str], device: str, batch_seconds: float = 300.0, out: str | None = None):
    model, cfg, _ = load_checkpoint(ckpt, device)
    ds = ManifestDataset(manifest, cfg.model.heads.event_labels, cfg.model.sample_rate, None)
    loader = DataLoader(
        ds,
        batch_sampler=DurationBatchSampler(ds.durations(), batch_seconds, 64, shuffle=False),
        collate_fn=collate,
    )
    audio_seconds = sum(ds.durations())
    report = {}
    rows = []
    for mode in modes:
        if mode != "fast" and not model.has_llm:
            continue
        refs, hyps = [], []
        t0 = time.time()
        for batch in loader:
            batch.to(device)
            res = model.transcribe(batch.wave, batch.wave_lens, mode=mode, languages=batch.languages)
            refs += batch.texts
            hyps += [r.text for r in res]
            rows += [{"mode": mode, "ref": r, "hyp": h.text, "used": h.mode} for r, h in zip(batch.texts, res)]
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        s = score(refs, hyps)
        report[mode] = {**s.as_dict(), "rtfx": audio_seconds / max(elapsed, 1e-9)}
    if out:
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate SubASR")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--modes", nargs="+", default=["fast", "edit", "quality"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", help="per-utterance JSONL")
    args = ap.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    report = run(args.ckpt, args.manifest, args.modes, device, out=args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
