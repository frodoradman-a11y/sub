import torch
import torch.nn as nn

from subasr.config import AdapterConfig, EncoderConfig, LLMConfig, LoraConfig
from subasr.modules.adapter import WindowQFormer
from subasr.modules.encoders import ConformerEncoder
from subasr.modules.heads import EventHead
from subasr.modules.llm import TinyLM
from subasr.modules.lora import LoRALinear, inject_lora, lora_parameters, merge_lora


def test_conformer_encoder_frame_rate_and_padding_invariance():
    cfg = EncoderConfig(n_mels=64, d_model=32, n_layers=2, n_heads=4, adapter_layers=[0, -1], dropout=0.0)
    enc = ConformerEncoder(cfg).eval()
    wave = torch.randn(2, 32000)
    out = enc(wave, torch.tensor([32000, 16000]))
    assert out.hidden.shape[:2] == (2, 51)  # 2 s at 25 frames/s
    assert out.lengths.tolist() == [51, 26]
    assert len(out.layers) == 2
    alone = enc(wave[1:, :16000], torch.tensor([16000]))
    # Only the STFT frames that straddle the end of the signal differ (reflection vs zero
    # padding); attention spreads that tiny difference, so compare loosely.
    diff = (alone.hidden[0, :26] - out.hidden[1, :26]).detach().abs()
    assert float(diff.mean()) < 0.02


def test_window_qformer_lengths_and_padding():
    cfg = AdapterConfig(window_frames=5, num_queries=2, dim=16, n_layers=1, n_heads=2, dropout=0.0)
    ad = WindowQFormer([8, 8], 12, cfg).eval()
    layers = [torch.randn(2, 12, 8), torch.randn(2, 12, 8)]
    out, lens = ad(layers, None, torch.tensor([12, 7]))
    assert out.shape == (2, 6, 12)
    assert lens.tolist() == [6, 4]
    # Frames beyond the length do not change the valid outputs.
    layers2 = [l.clone() for l in layers]
    for l in layers2:
        l[1, 7:] = 100.0
    out2, _ = ad(layers2, None, torch.tensor([12, 7]))
    assert torch.allclose(out[1, :4], out2[1, :4], atol=1e-5)
    assert torch.isfinite(out2).all()


def test_tiny_lm_causal_vs_bidirectional():
    lm = TinyLM(LLMConfig(d_model=32, n_layers=2, n_heads=4), vocab_size=20).eval()
    ids = torch.randint(0, 20, (1, 6))
    mask = torch.ones(1, 6, dtype=torch.long)
    ids2 = ids.clone()
    ids2[0, -1] = (ids[0, -1] + 1) % 20
    causal = lm.hidden(ids, lm.embed(ids), mask)
    causal2 = lm.hidden(ids2, lm.embed(ids2), mask)
    assert torch.allclose(causal[0, :5], causal2[0, :5], atol=1e-5)  # future token invisible
    bi = lm.hidden(ids, lm.embed(ids), mask, bidirectional=True)
    bi2 = lm.hidden(ids2, lm.embed(ids2), mask, bidirectional=True)
    assert not torch.allclose(bi[0, 0], bi2[0, 0], atol=1e-4)  # future token visible


def test_tiny_lm_left_padding_matches_unpadded_generation():
    torch.manual_seed(0)
    lm = TinyLM(LLMConfig(d_model=32, n_layers=2, n_heads=4), vocab_size=20).eval()
    ids = torch.randint(4, 20, (1, 5))
    single = lm.generate(ids, lm.embed(ids), torch.ones(1, 5, dtype=torch.long), 6, eos_ids=[3])
    padded_ids = torch.cat([torch.zeros(1, 3, dtype=torch.long), ids], 1)
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    padded = lm.generate(padded_ids, lm.embed(padded_ids), mask, 6, eos_ids=[3])
    assert single == padded


def test_lora_injection_merge_and_trainable_params():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8))
    model.q_proj = nn.Linear(8, 8)
    x = torch.randn(3, 8)
    before = model.q_proj(x)
    names = inject_lora(model, LoraConfig(r=2, alpha=4, dropout=0.0, targets=["q_proj"]))
    assert names == ["q_proj"] and isinstance(model.q_proj, LoRALinear)
    assert torch.allclose(model.q_proj(x), before)  # B initialised to zero
    with torch.no_grad():
        model.q_proj.lora_b.weight.normal_()
    tuned = model.q_proj(x)
    merge_lora(model)
    assert torch.allclose(model.q_proj(x), tuned, atol=1e-5)
    assert len(list(lora_parameters(model))) == 2


def test_event_head_multiple_instance_loss():
    head = EventHead(4, ["speech", "music"])
    logits = torch.tensor([[[5.0, -5.0], [-5.0, -5.0], [-5.0, -5.0]]])
    loss_ok = head.loss(logits, torch.tensor([3]), torch.tensor([[1.0, 0.0]]))
    loss_bad = head.loss(logits, torch.tensor([3]), torch.tensor([[0.0, 1.0]]))
    assert float(loss_ok) < 0.01 < float(loss_bad)
    # Frames past the length are ignored.
    logits[0, 2, 1] = 10.0
    assert float(head.loss(logits, torch.tensor([2]), torch.tensor([[1.0, 0.0]]))) < 0.01
