"""Train the tiny model on the synthetic tone language and use the CLIs.

Checks that the whole pipeline learns: after the encoder stage the fast mode must
transcribe held-out clips almost perfectly, the joint stage must train the LLM
modes, and the subtitle CLI must produce a timed SRT for a long file.
"""

import json

import numpy as np
import pytest
import soundfile as sf
import torch

from subasr import evaluate as evaluate_cli
from subasr import infer as infer_cli
from subasr.builder import load_checkpoint
from subasr.config import config_from_dict
from subasr.infer import transcribe_file
from subasr.train import train

from conftest import SR, synth, tiny_config_dict


@pytest.mark.slow
def test_train_encoder_then_joint_then_subtitle(data_dir, tmp_path):
    torch.set_num_threads(4)
    enc_cfg = config_from_dict(
        tiny_config_dict(data_dir, output_dir=str(tmp_path / "enc"), stage="encoder", max_steps=700, lr=2e-3,
                         encoder_lr=2e-3, warmup_steps=30)
    )
    enc_ckpt = train(enc_cfg, "cpu")
    report = evaluate_cli.run(str(enc_ckpt), str(data_dir / "dev.jsonl"), ["fast"], "cpu")
    print("encoder stage:", json.dumps(report))
    assert report["fast"]["cer"] < 0.1, report

    joint_cfg = config_from_dict(
        tiny_config_dict(data_dir, output_dir=str(tmp_path / "joint"), stage="joint", max_steps=250, lr=2e-3,
                         encoder_lr=1e-4, warmup_steps=20, init_from=str(enc_ckpt))
    )
    joint_ckpt = train(joint_cfg, "cpu")
    report = evaluate_cli.run(str(joint_ckpt), str(data_dir / "dev.jsonl"), ["fast", "edit", "quality"], "cpu")
    print("joint stage:", json.dumps(report))
    assert report["fast"]["cer"] < 0.1, report
    assert report["edit"]["cer"] < 0.3, report
    assert report["quality"]["cer"] < 0.5, report
    for mode in report.values():
        assert mode["rtfx"] > 0

    # A 25 s "film": three sentences separated by pauses, one long silence.
    sentences = ["سلام دنیا", "باران سرد", "روز نان دست"]
    rng = np.random.default_rng(7)
    parts, starts = [], []
    t = 0.0
    for i, s in enumerate(sentences):
        # Same recording conditions as the training clips: a faint noise floor.
        gap = (0.003 * rng.standard_normal(int((6.0 if i == 1 else 1.5) * SR))).astype(np.float32)
        parts.append(gap)
        t += len(gap) / SR
        starts.append(t)
        clip = synth(s, lead=0.0, tail=0.0, seed=100 + i)
        parts.append(clip)
        t += len(clip) / SR
    film = tmp_path / "film.wav"
    sf.write(film, np.concatenate(parts + [(0.003 * rng.standard_normal(SR)).astype(np.float32)]), SR)

    model, _, _ = load_checkpoint(joint_ckpt)
    cues, segs, words = transcribe_file(model, str(film), mode="fast", nonspeech_threshold=None)
    text = " ".join(w.text for w in words)
    assert text == " ".join(sentences), text
    # Each sentence's first word starts within 0.25 s of the truth.
    firsts = [w for w in words if w.text in ("سلام", "باران", "روز")]
    for w, s in zip(firsts, starts):
        assert abs(w.start - s) < 0.25, (w, s)

    out = tmp_path / "film.srt"
    infer_cli.main(["--ckpt", str(joint_ckpt), "--input", str(film), "--output", str(out), "--mode", "auto",
                    "--nonspeech-threshold", "-1", "--json", str(tmp_path / "segs.json")])
    srt = out.read_text(encoding="utf-8")
    assert srt.startswith("1\n00:00:0") and "-->" in srt
    assert json.loads((tmp_path / "segs.json").read_text(encoding="utf-8"))
