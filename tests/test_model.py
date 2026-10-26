import torch

from subasr.builder import build_model, load_checkpoint, save_checkpoint
from subasr.config import LossWeights
from subasr.data import DurationBatchSampler, ManifestDataset, collate
from subasr.model import _corrupt
from subasr.train import make_optimizer, set_trainable

from conftest import synth


def _batch(cfg):
    ds = ManifestDataset(cfg.train.train_manifest, cfg.model.heads.event_labels)
    return collate([ds[i] for i in range(3)])


def test_dataset_and_sampler(tiny_cfg):
    ds = ManifestDataset(tiny_cfg.train.train_manifest, tiny_cfg.model.heads.event_labels)
    assert len(ds) == 96
    item = ds[0]
    assert item["wave"].dtype == torch.float32 and item["language"] == "fa"
    assert item["events"].tolist()[0] == 1.0  # speech
    sampler = DurationBatchSampler(ds.durations(), batch_seconds=10, max_batch_size=4)
    batches = list(sampler)
    assert sorted(i for b in batches for i in b) == list(range(96))
    for b in batches:
        assert len(b) <= 4
        assert max(ds.durations()[i] for i in b) * len(b) <= 10 or len(b) == 1


def test_all_stages_forward_backward(tiny_cfg):
    model = build_model(tiny_cfg)
    batch = _batch(tiny_cfg)
    for stage in ("encoder", "align", "joint"):
        set_trainable(model, stage)
        opt = make_optimizer(model, tiny_cfg)
        loss, logs = model(batch, stage, LossWeights(), nle_corrupt_prob=0.5)
        assert torch.isfinite(loss)
        expected = {"encoder": {"ctc", "tdt", "events"}, "align": {"ar", "nle"}, "joint": {"ctc", "tdt", "events", "ar", "nle"}}
        assert set(logs) == expected[stage]
        loss.backward()
        trained = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
        if stage == "align":
            assert trained and all(n.startswith("adapter.") for n in trained)
        opt.zero_grad(set_to_none=True)


def test_encoder_only_model_has_no_llm(tiny_cfg):
    model = build_model(tiny_cfg, build_llm=False)
    assert not model.has_llm and model.adapter is None
    batch = _batch(tiny_cfg)
    out = model.transcribe(batch.wave, batch.wave_lens, mode="auto")
    assert len(out) == 3 and all(o.mode == "fast" for o in out)


def test_transcribe_all_modes_and_timestamps(tiny_cfg):
    model = build_model(tiny_cfg).eval()
    wave = torch.from_numpy(synth("سلام دنیا"))[None]
    lens = torch.tensor([wave.shape[1]])
    for mode in ("fast", "edit", "quality", "auto"):
        res = model.transcribe(wave, lens, mode=mode)[0]
        assert isinstance(res.text, str)
        assert res.event_probs.shape[1] == len(tiny_cfg.model.heads.event_labels)
        dur = wave.shape[1] / 16000
        for w in res.words:
            assert 0 <= w.start <= w.end <= dur + 0.05


def test_alignment_of_known_text_follows_ctc_peaks(tiny_cfg):
    model = build_model(tiny_cfg, build_llm=False)
    tok = model.ctc_tok
    # Hand-made CTC posteriors: "سلام" in frames 2-9, "دنیا" in frames 14-21.
    T = 25
    lp = torch.full((T, tok.vocab_size), -20.0)
    lp[:, tok.blank_id] = 0.0
    for word, start in (("سلام", 2), ("دنیا", 14)):
        for i, ch in enumerate(word):
            lp[start + 2 * i, :] = -20.0
            lp[start + 2 * i, tok.token_id(ch)] = 0.0
    words = model.align_words("سلام دنیا", lp.log_softmax(-1), T)
    assert [w.text for w in words] == ["سلام", "دنیا"]
    assert abs(words[0].start - 2 / 25) < 1e-6 and abs(words[1].start - 14 / 25) < 1e-6
    assert words[0].end <= words[1].start


def test_checkpoint_roundtrip(tiny_cfg, tmp_path):
    model = build_model(tiny_cfg).eval()
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, tiny_cfg, step=7)
    loaded, cfg, step = load_checkpoint(path)
    assert step == 7 and cfg.model.encoder.d_model == tiny_cfg.model.encoder.d_model
    wave = torch.from_numpy(synth("روز"))[None]
    lens = torch.tensor([wave.shape[1]])
    a = model.transcribe(wave, lens, mode="edit")[0]
    b = loaded.transcribe(wave, lens, mode="edit")[0]
    assert a.text == b.text and a.fast_text == b.fast_text


def test_corrupt_changes_text_but_keeps_alphabet():
    import random

    random.seed(1)
    text = "سلام دنیا باران"
    out = _corrupt(text, rate=0.5)
    assert out != text
    assert set(out) <= set(text)
