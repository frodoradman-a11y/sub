"""Integration with the real Gemma 4 classes from transformers.

A tiny randomly initialised Gemma 4 (audio tower + text model) and a character
tokenizer with a Gemma-style chat template are saved to disk and loaded through
the same code path as ``google/gemma-4-E2B-it``.
"""


import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Gemma4Config, Gemma4ForConditionalGeneration, PreTrainedTokenizerFast  # noqa: E402
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor  # noqa: E402

from subasr.builder import build_model, load_checkpoint, save_checkpoint  # noqa: E402
from subasr.config import LossWeights, config_from_dict  # noqa: E402
from subasr.data import ManifestDataset, collate  # noqa: E402
from subasr.modules.llm import Gemma4LM  # noqa: E402
from subasr.train import make_optimizer, set_trainable  # noqa: E402

from conftest import LETTERS, synth, tiny_config_dict  # noqa: E402

CHAT = (
    "{{ bos_token }}{% for m in messages %}<start_of_turn>{{ m['role'] }}\n{{ m['content'] }}<end_of_turn>\n"
    "{% endfor %}<start_of_turn>model\n"
)


def _tokenizer(path):
    from tokenizers import Tokenizer, models

    specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
    chars = sorted(set(LETTERS + " \nabcdefghijklmnopqrstuvwxyzTPLN.:,'"))
    vocab = {t: i for i, t in enumerate(specials + chars)}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>", unk_token="<unk>"
    )
    fast.add_special_tokens({"additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"]})
    fast.chat_template = CHAT
    fast.save_pretrained(path)
    return len(vocab)


@pytest.fixture(scope="module")
def gemma_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("gemma4")
    vocab = _tokenizer(path)
    text = dict(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, global_head_dim=32, vocab_size_per_layer_input=128,
        hidden_size_per_layer_input=16, sliding_window=16, num_kv_shared_layers=2,
        pad_token_id=0, eos_token_id=1, bos_token_id=2,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    )
    assert vocab <= 128
    audio = dict(hidden_size=64, num_hidden_layers=3, num_attention_heads=2, subsampling_conv_channels=[128, 8], output_proj_dims=48)
    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration(Gemma4Config(text_config=text, audio_config=audio, vision_config=None))
    model.save_pretrained(path)
    Gemma4AudioFeatureExtractor().save_pretrained(path)
    return path


def _cfg(data_dir, gemma_dir, tmp_path, adapter="window_qformer"):
    d = tiny_config_dict(data_dir, output_dir=str(tmp_path / "run"))
    d["model"]["encoder"] = {"type": "gemma4", "pretrained": str(gemma_dir), "adapter_layers": [1, -1],
                             "spec_augment": {"enabled": False}}
    d["model"]["adapter"]["type"] = adapter
    d["model"]["llm"] = {
        "type": "gemma4", "pretrained": str(gemma_dir), "dtype": "float32",
        "lora": {"enabled": True, "r": 4, "alpha": 8, "dropout": 0.0},
        "prompt_template": "chat", "max_new_tokens": 12,
    }
    return config_from_dict(d)


@pytest.mark.parametrize("adapter", ["window_qformer", "gemma_native"])
def test_gemma4_training_step(data_dir, gemma_dir, tmp_path, adapter):
    cfg = _cfg(data_dir, gemma_dir, tmp_path, adapter)
    model = build_model(cfg)
    assert model.encoder.dim == 48 and model.llm.hidden_size == 64
    ds = ManifestDataset(cfg.train.train_manifest, cfg.model.heads.event_labels)
    batch = collate([ds[i] for i in range(2)])
    set_trainable(model, "joint")
    opt = make_optimizer(model, cfg)
    loss, logs = model(batch, "joint", LossWeights(), 0.5)
    assert set(logs) == {"ctc", "tdt", "events", "ar", "nle"} and torch.isfinite(loss)
    loss.backward()
    lora_grads = [p.grad for n, p in model.llm.named_parameters() if "lora_b" in n]
    assert lora_grads and all(g is not None for g in lora_grads)
    frozen = [p for n, p in model.llm.named_parameters() if "lora_" not in n]
    assert all(not p.requires_grad for p in frozen)
    opt.step()


def test_gemma4_bidirectional_mask_and_cached_generation(data_dir, gemma_dir, tmp_path):
    cfg = _cfg(data_dir, gemma_dir, tmp_path)
    model = build_model(cfg).eval()
    lm: Gemma4LM = model.llm
    ids = torch.randint(10, 100, (1, 7))
    mask = torch.ones(1, 7, dtype=torch.long)
    ids2 = ids.clone()
    ids2[0, -1] = ids[0, -1] + 1
    with torch.no_grad():
        c1, c2 = lm.hidden(ids, lm.embed(ids), mask), lm.hidden(ids2, lm.embed(ids2), mask)
        b1 = lm.hidden(ids, lm.embed(ids), mask, bidirectional=True)
        b2 = lm.hidden(ids2, lm.embed(ids2), mask, bidirectional=True)
    assert torch.allclose(c1[0, :6], c2[0, :6], atol=1e-5)
    assert not torch.allclose(b1[0, 0], b2[0, 0], atol=1e-5)

    # Greedy decoding with the KV cache must equal step-by-step recomputation.
    torch.manual_seed(1)
    prompt = torch.randint(10, 100, (1, 6))
    cached = lm.generate(prompt, lm.embed(prompt), torch.ones(1, 6, dtype=torch.long), 8, eos_ids=[1])[0]
    seq = prompt
    ref = []
    with torch.no_grad():
        for _ in range(8):
            h = lm.hidden(seq, lm.embed(seq), torch.ones_like(seq))
            nxt = int(lm.logits(h[:, -1]).argmax())
            if nxt == 1:
                break
            ref.append(nxt)
            seq = torch.cat([seq, torch.tensor([[nxt]])], 1)
    assert cached[: len(ref)] == ref[: len(cached)] and len(cached) <= 8
    # Left padding does not change the result.
    padded = torch.cat([torch.zeros(1, 3, dtype=torch.long), prompt], 1)
    pmask = torch.tensor([[0, 0, 0] + [1] * 6])
    assert lm.generate(padded, lm.embed(padded), pmask, 8, eos_ids=[1])[0] == cached


def test_gemma4_transcribe_and_checkpoint(data_dir, gemma_dir, tmp_path):
    cfg = _cfg(data_dir, gemma_dir, tmp_path)
    model = build_model(cfg).eval()
    wave = torch.from_numpy(synth("سلام دنیا"))[None]
    lens = torch.tensor([wave.shape[1]])
    results = {m: model.transcribe(wave, lens, mode=m, glossaries=[["تهران"]], contexts=["روز"])[0] for m in ("fast", "edit", "quality")}
    assert results["quality"].mode in ("quality", "quality->edit")
    path = tmp_path / "g.pt"
    save_checkpoint(path, model, cfg, 3)
    saved = torch.load(path, weights_only=False)["state"]
    assert not any(k.startswith("llm.") and "lora_" not in k for k in saved)  # frozen LLM is not copied
    loaded, _, _ = load_checkpoint(path)
    again = loaded.transcribe(wave, lens, mode="edit", glossaries=[["تهران"]], contexts=["روز"])[0]
    assert again.text == results["edit"].text
