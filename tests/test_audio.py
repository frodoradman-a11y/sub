import math
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf
import torch

from subasr.audio.augment import Augmenter, mix_at_snr
from subasr.audio.features import LogMel, spec_augment
from subasr.audio.io import audio_duration, load_audio, resample
from subasr.audio.segment import split_audio
from subasr.config import AugmentConfig, SpecAugmentConfig


def _peak_freq(wave: torch.Tensor, sr: int) -> float:
    spec = torch.fft.rfft(wave * torch.hann_window(wave.numel()))
    return float(spec.abs().argmax()) * sr / wave.numel()


def test_resample_preserves_frequency_and_length():
    sr = 44100
    t = torch.arange(sr) / sr
    wave = torch.sin(2 * math.pi * 1000 * t)
    out = resample(wave, sr, 16000)
    assert out.numel() == 16000
    assert abs(_peak_freq(out, 16000) - 1000) < 3
    # Content above the new Nyquist frequency is removed.
    high = resample(torch.sin(2 * math.pi * 12000 * t), sr, 16000)
    assert high[200:-200].abs().max() < 0.05  # edges see the abrupt onset


def test_load_audio_wav_and_ffmpeg(tmp_path):
    sr = 22050
    wave = (0.3 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    stereo = np.stack([wave, wave], axis=1)
    sf.write(tmp_path / "a.wav", stereo, sr)
    out = load_audio(tmp_path / "a.wav", 16000)
    assert out.numel() == 16000 and abs(_peak_freq(out, 16000) - 440) < 3
    part = load_audio(tmp_path / "a.wav", 16000, offset=0.5, duration=0.25)
    assert part.numel() == 4000
    assert abs(audio_duration(tmp_path / "a.wav") - 1.0) < 1e-3
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg missing")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(tmp_path / "a.wav"), str(tmp_path / "a.mp3")], check=True)
    mp3 = load_audio(tmp_path / "a.mp3", 16000)
    assert abs(mp3.numel() - 16000) < 2000 and abs(_peak_freq(mp3, 16000) - 440) < 5


def test_mix_at_snr():
    torch.manual_seed(0)
    sig = torch.randn(16000)
    noise = torch.randn(4000)
    mixed = mix_at_snr(sig, noise, 10.0)
    added = mixed - sig
    snr = 10 * torch.log10(sig.pow(2).mean() / added.pow(2).mean())
    assert abs(float(snr) - 10.0) < 0.01


def test_augmenter_adds_music_label(tmp_path):
    sf.write(tmp_path / "m.wav", np.random.randn(8000).astype(np.float32) * 0.1, 16000)
    (tmp_path / "music.jsonl").write_text('{"audio": "m.wav"}\n')
    cfg = AugmentConfig(music_manifest=str(tmp_path / "music.jsonl"), music_prob=1.0, speed_prob=0.0, noise_prob=0.0)
    aug = Augmenter(cfg)
    wave, labels = aug(torch.randn(16000) * 0.1, {"speech"})
    assert labels == {"speech", "music"}
    assert wave.numel() == 16000 and wave.abs().max() <= 1.0
    cfg.speed_prob, cfg.music_prob = 1.0, 0.0
    cfg.speeds = [1.1]
    wave, _ = Augmenter(cfg)(torch.randn(16000), set())
    assert abs(wave.numel() - 16000 / 1.1) < 2


def test_split_audio_cuts_in_silence_and_drops_silent_parts():
    sr = 16000
    tone = lambda s: 0.5 * torch.sin(2 * math.pi * 440 * torch.arange(int(s * sr)) / sr)
    silence = lambda s: torch.zeros(int(s * sr))
    wave = torch.cat([tone(20), silence(1.0), tone(15), silence(40), tone(5)])
    segs = split_audio(wave, sr, max_seconds=30, search_seconds=12)
    assert all((s.end - s.start) <= 30 * sr for s in segs)
    first_cut = segs[0].end / sr
    assert 20.0 <= first_cut <= 21.0  # inside the 1 s pause
    covered = sum(s.end - s.start for s in segs) / sr
    assert covered < 81 - 35  # most of the 40 s silence was dropped
    assert segs[-1].end == wave.numel()
    assert split_audio(torch.zeros(sr * 5), sr) == []


def test_logmel_and_specaugment_shapes():
    fe = LogMel(16000, 64)
    wave = torch.randn(2, 16000)
    feats, flen = fe(wave, torch.tensor([16000, 8000]))
    assert feats.shape == (2, 101, 64)
    assert flen.tolist() == [101, 51]
    assert float(feats[1, 60:].abs().sum()) == 0.0
    masked = spec_augment(feats, flen, SpecAugmentConfig(freq_masks=2, freq_width=10, time_masks=2))
    assert masked.shape == feats.shape and not torch.equal(masked, feats)
