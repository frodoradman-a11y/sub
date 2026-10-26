"""Training entry point.

    python -m subasr.train --config configs/gemma4_e2b.yaml train.stage=encoder
    python -m subasr.train --config configs/gemma4_e2b.yaml train.stage=align train.init_from=runs/enc/last.pt
    python -m subasr.train --config configs/gemma4_e2b.yaml train.stage=joint train.init_from=runs/align/last.pt

Stages
------
encoder  encoder + CTC/TDT/event heads. The LLM is not loaded.
align    only the adapter is trained (encoder and LLM frozen): teaches the LLM to
         read the audio embeddings.
joint    encoder (encoder_lr) + heads + adapter + LoRA, all losses.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from subasr.builder import build_model, load_weights, save_checkpoint
from subasr.config import Config, load_config
from subasr.data import DurationBatchSampler, ManifestDataset, collate
from subasr.metrics import score
from subasr.model import SubASRModel

STAGES = ("encoder", "align", "joint")


def set_trainable(model: SubASRModel, stage: str) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    groups = []
    if stage in ("encoder", "joint"):
        if not model.cfg.encoder.freeze:
            groups.append(model.encoder)
        groups += [m for m in (model.ctc, model.tdt, model.events) if m is not None]
    if stage in ("align", "joint") and model.adapter is not None:
        groups.append(model.adapter)
    for module in groups:
        for p in module.parameters():
            p.requires_grad_(True)
    if stage == "joint" and model.llm is not None:
        for name, p in model.llm.named_parameters():
            if "lora_" in name or not model.cfg.llm.freeze:
                p.requires_grad_(True)


def make_optimizer(model: SubASRModel, cfg: Config):
    enc_params, other = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (enc_params if name.startswith("encoder.") else other).append(p)
    groups = []
    if enc_params:
        groups.append({"params": enc_params, "lr": cfg.train.encoder_lr})
    if other:
        groups.append({"params": other, "lr": cfg.train.lr})
    if not groups:
        raise RuntimeError("nothing to train in this stage")
    opt = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay, betas=(0.9, 0.98))
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    return opt


def lr_factor(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model: SubASRModel, loader, device, max_utts: int, modes: list[str]) -> dict[str, float]:
    model.eval()
    refs: list[str] = []
    hyps: dict[str, list[str]] = {m: [] for m in modes}
    for batch in loader:
        batch.to(device)
        for m in modes:
            out = model.transcribe(batch.wave, batch.wave_lens, mode=m, languages=batch.languages)
            hyps[m] += [o.text for o in out]
        refs += batch.texts
        if len(refs) >= max_utts:
            break
    model.train()
    result = {}
    for m in modes:
        s = score(refs, hyps[m])
        result[f"{m}_wer"], result[f"{m}_cer"] = s.wer.rate, s.cer.rate
    return result


def train(cfg: Config, device: str | None = None) -> Path:
    t = cfg.train
    if t.stage not in STAGES:
        raise ValueError(f"train.stage must be one of {STAGES}")
    random.seed(t.seed)
    torch.manual_seed(t.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(t.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg, build_llm=t.stage != "encoder")
    if t.init_from:
        payload = torch.load(t.init_from, map_location="cpu", weights_only=False)
        load_weights(model, payload["state"])
    model.to(device)
    set_trainable(model, t.stage)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"stage={t.stage} trainable={n_train / 1e6:.2f}M / total={n_total / 1e6:.2f}M device={device}", flush=True)

    labels = cfg.model.heads.event_labels
    train_ds = ManifestDataset(
        t.train_manifest, labels, cfg.model.sample_rate, t.augment, t.min_duration, t.max_duration, t.default_language
    )
    sampler = DurationBatchSampler(train_ds.durations(), t.batch_seconds, t.max_batch_size, seed=t.seed)
    loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=t.num_workers)
    val_loader = None
    if t.val_manifest:
        val_ds = ManifestDataset(t.val_manifest, labels, cfg.model.sample_rate, None, 0.0, t.max_duration, t.default_language)
        val_loader = DataLoader(
            val_ds,
            batch_sampler=DurationBatchSampler(val_ds.durations(), t.batch_seconds, t.max_batch_size, shuffle=False),
            collate_fn=collate,
            num_workers=t.num_workers,
        )
    eval_modes = ["fast"] + (["edit", "quality"] if model.has_llm and t.stage != "encoder" else [])

    opt = make_optimizer(model, cfg)
    use_bf16 = t.precision == "bf16" and device.startswith("cuda")
    log_path = out_dir / "log.jsonl"
    step, micro = 0, 0
    model.train()
    start = time.time()
    running: dict[str, float] = {}
    while step < t.max_steps:
        for batch in loader:
            batch.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss, logs = model(batch, t.stage, t.loss, t.nle_corrupt_prob)
            (loss / t.grad_accum).backward()
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v
            running["loss"] = running.get("loss", 0.0) + float(loss.detach())
            micro += 1
            if micro % t.grad_accum:
                continue
            params = [p for g in opt.param_groups for p in g["params"]]
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, t.grad_clip))
            f = lr_factor(step, t.warmup_steps, t.max_steps)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % t.log_every == 0 or step == 1:
                n = micro if step == 1 else t.log_every * t.grad_accum
                rec = {k: v / n for k, v in running.items()}
                rec.update(step=step, lr=opt.param_groups[-1]["lr"], grad_norm=grad_norm, elapsed=time.time() - start)
                print(json.dumps(rec), flush=True)
                with open(log_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                running = {}
            if val_loader is not None and step % t.eval_every == 0:
                res = evaluate(model, val_loader, device, t.eval_max_utts, eval_modes)
                res["step"] = step
                print(json.dumps({"eval": res}), flush=True)
                with open(log_path, "a") as fh:
                    fh.write(json.dumps({"eval": res}) + "\n")
            if step % t.save_every == 0:
                save_checkpoint(out_dir / "last.pt", model, cfg, step)
            if step >= t.max_steps:
                break
    save_checkpoint(out_dir / "last.pt", model, cfg, step)
    return out_dir / "last.pt"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train SubASR")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.stage=joint train.lr=5e-4")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    path = train(cfg, args.device)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
