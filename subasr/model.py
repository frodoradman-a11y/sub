"""The SubASR model: encoder + CTC/TDT/event heads + adapter + LLM.

Inference modes
---------------
fast     encoder -> TDT (or CTC) greedy. No LLM.
edit     CTC hypothesis -> LLM with bidirectional attention edits it in one pass
         (non-autoregressive, NLE-style): every hypothesis token gets empty slots
         after it and the LLM output over all slots is CTC-collapsed.
quality  LLM generates the transcript autoregressively from audio + prompt
         (context, glossary). A guard compares it with the CTC hypothesis and falls
         back to the edit result when the two disagree wildly (hallucination).
auto     fast everywhere; segments with low confidence or music go to edit.

Timestamps for every mode come from CTC forced alignment of the final text.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from subasr.config import ModelConfig
from subasr.decoding.ctc import Hypothesis, ctc_collapse, ctc_greedy, forced_align
from subasr.decoding.tdt import tdt_greedy
from subasr.metrics import cer_between
from subasr.modules.adapter import GemmaNativeAdapter, WindowQFormer
from subasr.modules.encoders import ConformerEncoder, EncoderOutput, Gemma4AudioEncoder
from subasr.modules.heads import CTCHead, EventHead, TDTHead
from subasr.modules.llm import Gemma4LM, TinyLM
from subasr.modules.lora import inject_lora
from subasr.prompts import PromptBuilder
from subasr.text.normalize import normalize_text

MODES = ("fast", "edit", "quality", "auto")


@dataclass
class Word:
    text: str
    start: float  # seconds, relative to the segment
    end: float


@dataclass
class Transcript:
    text: str
    words: list[Word]
    mode: str
    confidence: float
    event_probs: torch.Tensor | None = None  # (T, n_events) at encoder frame rate
    frame_rate: float = 25.0
    fast_text: str = ""


@dataclass
class Batch:
    wave: torch.Tensor  # (B, S)
    wave_lens: torch.Tensor  # (B,)
    texts: list[str]
    languages: list[str]
    contexts: list[str | None] = field(default_factory=list)
    glossaries: list[list[str] | None] = field(default_factory=list)
    event_targets: torch.Tensor | None = None  # (B, n_events)

    def to(self, device) -> Batch:
        self.wave = self.wave.to(device)
        self.wave_lens = self.wave_lens.to(device)
        if self.event_targets is not None:
            self.event_targets = self.event_targets.to(device)
        return self


def _corrupt(text: str, rate: float = 0.12) -> str:
    """Random character edits, used to train the edit mode on realistic errors."""
    chars = list(text)
    pool = [c for c in chars if not c.isspace()] or ["a"]
    out = []
    for c in chars:
        r = random.random()
        if r < rate / 3:
            continue  # deletion
        if r < 2 * rate / 3:
            out.append(random.choice(pool))  # substitution
            continue
        out.append(c)
        if r < rate:
            out.append(random.choice(pool))  # insertion
    return "".join(out)


class SubASRModel(nn.Module):
    def __init__(self, cfg: ModelConfig, ctc_tokenizer, llm_tokenizer=None, build_llm: bool = True):
        super().__init__()
        self.cfg = cfg
        self.ctc_tok = ctc_tokenizer
        self.llm_tok = llm_tokenizer
        gemma = None
        needs_gemma = cfg.encoder.type == "gemma4" or (build_llm and cfg.llm.type == "gemma4")
        if needs_gemma:
            from subasr.modules.gemma import load_gemma4, load_gemma4_feature_extractor

            path = cfg.encoder.pretrained or cfg.llm.pretrained
            gemma = load_gemma4(path, cfg.llm.dtype)

        # Encoder
        if cfg.encoder.type == "conformer":
            self.encoder = ConformerEncoder(cfg.encoder, cfg.sample_rate)
        elif cfg.encoder.type == "gemma4":
            fe = load_gemma4_feature_extractor(cfg.encoder.pretrained or cfg.llm.pretrained)
            self.encoder = Gemma4AudioEncoder(cfg.encoder, gemma, fe)
            self.encoder.float()  # encoder is fine-tuned: keep fp32 master weights
        else:
            raise ValueError(f"unknown encoder type {cfg.encoder.type}")
        dim = self.encoder.dim
        vocab = ctc_tokenizer.vocab_size

        # Heads
        h = cfg.heads
        self.ctc = CTCHead(dim, vocab) if h.ctc else None
        self.tdt = TDTHead(dim, vocab, h.tdt_durations, h.pred_dim, h.joint_dim) if h.tdt else None
        self.events = EventHead(dim, h.event_labels) if h.events else None
        if self.ctc is None and self.tdt is None:
            raise ValueError("enable at least one of heads.ctc / heads.tdt")

        # LLM + adapter
        self.llm = None
        self.adapter = None
        if build_llm and cfg.llm.type != "none":
            if llm_tokenizer is None:
                raise ValueError("an LLM tokenizer is required when llm.type != none")
            if cfg.llm.type == "tiny":
                self.llm = TinyLM(cfg.llm, llm_tokenizer.vocab_size)
            elif cfg.llm.type == "gemma4":
                self.llm = Gemma4LM(gemma)
                if cfg.llm.gradient_checkpointing:
                    self.llm.lm.gradient_checkpointing_enable()
            else:
                raise ValueError(f"unknown llm type {cfg.llm.type}")
            if cfg.llm.freeze:
                for p in self.llm.parameters():
                    p.requires_grad_(False)
            if cfg.llm.lora.enabled:
                inject_lora(self.llm, cfg.llm.lora)
            if cfg.adapter.type == "window_qformer":
                self.adapter = WindowQFormer(self.encoder.layer_dims, self.llm.hidden_size, cfg.adapter)
            elif cfg.adapter.type == "gemma_native":
                if gemma is None:
                    raise ValueError("adapter.type=gemma_native needs the Gemma 4 encoder")
                self.adapter = GemmaNativeAdapter(gemma)
                self.adapter.float()
            else:
                raise ValueError(f"unknown adapter type {cfg.adapter.type}")
            self.prompts = PromptBuilder(cfg.llm, llm_tokenizer)
        if gemma is not None:
            from subasr.modules.gemma import clear_cache

            clear_cache()

    # ------------------------------------------------------------------ utils
    @property
    def has_llm(self) -> bool:
        return self.llm is not None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def frame_rate(self) -> float:
        return self.encoder.frame_rate

    def encode(self, wave: torch.Tensor, wave_lens: torch.Tensor) -> EncoderOutput:
        return self.encoder(wave, wave_lens)

    def ctc_targets(self, texts: list[str]) -> list[list[int]]:
        return [self.ctc_tok.encode(normalize_text(t)) for t in texts]

    def llm_targets(self, texts: list[str]) -> list[list[int]]:
        return [self.llm_tok.encode(normalize_text(t)) for t in texts]

    @staticmethod
    def _pad(seqs: list[list[int]], device, value: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        lens = torch.tensor([len(s) for s in seqs], dtype=torch.long, device=device)
        out = torch.full((len(seqs), max(1, int(lens.max()) if seqs else 1)), value, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            if s:
                out[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        return out, lens

    # ----------------------------------------------------------- encoder losses
    def encoder_losses(self, enc: EncoderOutput, texts: list[str], event_targets=None) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        targets, tlens = self._pad(self.ctc_targets(texts), enc.hidden.device)
        if self.ctc is not None:
            lp = self.ctc(enc.hidden)
            per = CTCHead.loss(lp, enc.lengths, targets, tlens, self.ctc_tok.blank_id)
            losses["ctc"] = (per / tlens.clamp(min=1)).mean()
        if self.tdt is not None:
            per = self.tdt.loss(enc.hidden, enc.lengths, targets, tlens)
            losses["tdt"] = (per / (tlens + 1)).mean()
        if self.events is not None and event_targets is not None:
            losses["events"] = self.events.loss(self.events(enc.hidden), enc.lengths, event_targets).mean()
        return losses

    # ------------------------------------------------------------- LLM inputs
    def audio_embeddings(self, enc: EncoderOutput) -> tuple[torch.Tensor, torch.Tensor]:
        return self.adapter(enc.layers, enc.hidden, enc.lengths)

    def _assemble(self, pieces: list[list[tuple[str, torch.Tensor]]], left_pad: bool = False):
        """Concatenate per-sample pieces of ("ids", LongTensor) / ("emb", FloatTensor).

        Returns input_ids (audio positions hold pad_id), inputs_embeds, attention_mask
        and the (start, end) position of every piece for every sample.
        """
        device = self.device
        pad_id = self.llm_tok.pad_id
        seq_ids, seq_emb, spans = [], [], []
        for parts in pieces:
            ids_list, emb_list, sp, pos = [], [], [], 0
            for kind, value in parts:
                if kind == "ids":
                    ids = value.to(device)
                    emb = self.llm.embed(ids[None])[0]
                else:
                    emb = value
                    ids = torch.full((value.shape[0],), pad_id, dtype=torch.long, device=device)
                ids_list.append(ids)
                emb_list.append(emb.to(self.llm_dtype))
                sp.append((pos, pos + ids.shape[0]))
                pos += ids.shape[0]
            seq_ids.append(torch.cat(ids_list))
            seq_emb.append(torch.cat(emb_list))
            spans.append(sp)
        L = max(s.shape[0] for s in seq_ids)
        B, H = len(seq_ids), seq_emb[0].shape[-1]
        input_ids = torch.full((B, L), pad_id, dtype=torch.long, device=device)
        embeds = torch.zeros(B, L, H, dtype=seq_emb[0].dtype, device=device)
        mask = torch.zeros(B, L, dtype=torch.long, device=device)
        for b in range(B):
            n = seq_ids[b].shape[0]
            off = L - n if left_pad else 0
            input_ids[b, off : off + n] = seq_ids[b]
            embeds[b, off : off + n] = seq_emb[b]
            mask[b, off : off + n] = 1
            spans[b] = [(s + off, e + off) for s, e in spans[b]]
        return input_ids, embeds, mask, spans

    @property
    def llm_dtype(self):
        return self.llm.embed_tokens.weight.dtype if isinstance(self.llm, TinyLM) else self.llm.lm.embed_tokens.weight.dtype

    def _prompt_pieces(self, audio, alens, languages, contexts, glossaries):
        pieces = []
        for b in range(audio.shape[0]):
            ctx = contexts[b] if contexts else None
            glo = glossaries[b] if glossaries else None
            before, after = self.prompts.build(languages[b], ctx, glo)
            parts = []
            if self.llm_tok.bos_id is not None and (not before or before[0] != self.llm_tok.bos_id):
                before = [self.llm_tok.bos_id] + before
            parts.append(("ids", torch.tensor(before, dtype=torch.long)))
            parts.append(("emb", audio[b, : int(alens[b])]))
            if after:
                parts.append(("ids", torch.tensor(after, dtype=torch.long)))
            pieces.append(parts)
        return pieces

    def _slots(self, hyp: list[int]) -> list[int]:
        k, pad = self.cfg.llm.nle_slots_per_token, self.llm_tok.pad_id
        slots = [pad] * k
        for t in hyp:
            slots += [t] + [pad] * k
        return slots + [pad] * self.cfg.llm.nle_extra_slots

    # --------------------------------------------------------------- LLM losses
    def ar_loss(self, audio, alens, batch: Batch) -> torch.Tensor:
        targets = self.llm_targets(batch.texts)
        eos = self.llm_tok.eos_ids[0]
        pieces = self._prompt_pieces(audio, alens, batch.languages, batch.contexts, batch.glossaries)
        for parts, tgt in zip(pieces, targets):
            parts.append(("ids", torch.tensor(tgt + [eos], dtype=torch.long)))
        ids, emb, mask, spans = self._assemble(pieces)
        hidden = self.llm.hidden(ids, emb, mask)
        sel_h, sel_y = [], []
        for b, (parts_span, tgt) in enumerate(zip(spans, targets)):
            start, end = parts_span[-1]
            sel_h.append(hidden[b, start - 1 : end - 1])
            sel_y.append(ids[b, start:end])
        logits = self.llm.logits(torch.cat(sel_h))
        return F.cross_entropy(logits, torch.cat(sel_y))

    def nle_loss(self, audio, alens, batch: Batch, hyps: list[str]) -> torch.Tensor:
        targets = self.llm_targets(batch.texts)
        pieces = self._prompt_pieces(audio, alens, batch.languages, batch.contexts, batch.glossaries)
        for parts, hyp in zip(pieces, hyps):
            parts.append(("ids", torch.tensor(self._slots(self.llm_tok.encode(normalize_text(hyp))), dtype=torch.long)))
        ids, emb, mask, spans = self._assemble(pieces)
        hidden = self.llm.hidden(ids, emb, mask, bidirectional=True)
        slot_h = [hidden[b, s:e] for b, ((s, e)) in enumerate(sp[-1] for sp in spans)]
        n = max(x.shape[0] for x in slot_h)
        lp = torch.stack([F.pad(self.llm.logits(x), (0, 0, 0, n - x.shape[0])) for x in slot_h]).log_softmax(-1)
        in_lens = torch.tensor([x.shape[0] for x in slot_h], device=lp.device)
        tgt, tlens = self._pad(targets, lp.device)
        per = F.ctc_loss(
            lp.transpose(0, 1), tgt, in_lens, tlens, blank=self.llm_tok.pad_id, reduction="none", zero_infinity=True
        )
        return (per / tlens.clamp(min=1)).mean()

    # ------------------------------------------------------------- training step
    def forward(self, batch: Batch, stage: str, weights, nle_corrupt_prob: float = 0.5):
        enc = self.encode(batch.wave, batch.wave_lens)
        losses: dict[str, torch.Tensor] = {}
        if stage in ("encoder", "joint"):
            losses.update(self.encoder_losses(enc, batch.texts, batch.event_targets))
        if self.has_llm and stage in ("align", "joint"):
            audio, alens = self.audio_embeddings(enc)
            if weights.ar > 0:
                losses["ar"] = self.ar_loss(audio, alens, batch)
            if weights.nle > 0:
                hyps = self._training_hypotheses(enc, batch.texts, nle_corrupt_prob)
                losses["nle"] = self.nle_loss(audio, alens, batch, hyps)
        total = sum(getattr(weights, k) * v for k, v in losses.items())
        return total, {k: float(v.detach()) for k, v in losses.items()}

    @torch.no_grad()
    def _training_hypotheses(self, enc: EncoderOutput, texts: list[str], corrupt_prob: float) -> list[str]:
        model_hyps = None
        if self.ctc is not None and corrupt_prob < 1.0:
            model_hyps = ctc_greedy(self.ctc(enc.hidden.detach()), enc.lengths, self.ctc_tok.blank_id)
        out = []
        for i, text in enumerate(texts):
            if model_hyps is None or random.random() < corrupt_prob:
                out.append(_corrupt(normalize_text(text)))
            else:
                out.append(self.ctc_tok.decode(model_hyps[i].tokens))
        return out

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def transcribe(
        self,
        wave: torch.Tensor,
        wave_lens: torch.Tensor,
        mode: str = "fast",
        languages: list[str] | None = None,
        contexts: list[str | None] | None = None,
        glossaries: list[list[str] | None] | None = None,
        auto_threshold: float = 0.85,
        auto_escalate: str = "edit",
        nonspeech_threshold: float | None = None,
        guard_cer: float = 0.6,
    ) -> list[Transcript]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode in ("edit", "quality") and not self.has_llm:
            raise ValueError(f"mode '{mode}' needs a model with an LLM")
        self.eval()
        B = wave.shape[0]
        languages = languages or ["fa"] * B
        contexts = contexts or [None] * B
        glossaries = glossaries or [None] * B
        enc = self.encode(wave, wave_lens)
        ctc_lp = self.ctc(enc.hidden) if self.ctc is not None else None
        ctc_hyps = ctc_greedy(ctc_lp, enc.lengths, self.ctc_tok.blank_id) if ctc_lp is not None else None
        fast_hyps: list[Hypothesis] = (
            tdt_greedy(self.tdt, enc.hidden, enc.lengths, self.cfg.heads.max_symbols_per_frame)
            if self.tdt is not None
            else ctc_hyps
        )
        fast_text = [self.ctc_tok.decode(h.tokens) for h in fast_hyps]
        ctc_text = [self.ctc_tok.decode(h.tokens) for h in ctc_hyps] if ctc_hyps else fast_text
        event_probs = torch.sigmoid(self.events(enc.hidden)) if self.events is not None else None

        texts = list(fast_text)
        modes = ["fast"] * B
        confidences = [h.confidence for h in fast_hyps]

        todo_edit: list[int] = []
        todo_quality: list[int] = []
        if mode == "edit":
            todo_edit = list(range(B))
        elif mode == "quality":
            todo_quality = list(range(B))
        elif mode == "auto" and self.has_llm:
            for b in range(B):
                music = self._music_ratio(event_probs, enc.lengths, b) if event_probs is not None else 0.0
                if confidences[b] < auto_threshold or music > 0.3:
                    (todo_quality if auto_escalate == "quality" else todo_edit).append(b)

        if todo_edit or todo_quality:
            audio, alens = self.audio_embeddings(enc)
            # The guard of the quality mode needs the edit result as a fallback.
            edit_idx = sorted(set(todo_edit) | set(todo_quality))
            edit_out = self._edit(audio, alens, edit_idx, ctc_text, languages, contexts, glossaries)
            for b, t in zip(edit_idx, edit_out):
                if b in todo_edit:
                    texts[b], modes[b] = t, "edit"
            if todo_quality:
                gen = self._generate(audio, alens, todo_quality, languages, contexts, glossaries)
                fallback = dict(zip(edit_idx, edit_out))
                for b, t in zip(todo_quality, gen):
                    ref = ctc_text[b]
                    if ref and cer_between(ref, t) > guard_cer:
                        texts[b], modes[b] = fallback[b], "quality->edit"
                    else:
                        texts[b], modes[b] = t, "quality"

        results = []
        for b in range(B):
            text = normalize_text(texts[b])
            if nonspeech_threshold is not None and event_probs is not None:
                if self._speech_prob(event_probs, enc.lengths, b) < nonspeech_threshold:
                    text = ""
            T = int(enc.lengths[b])
            words = self.align_words(text, ctc_lp[b, :T] if ctc_lp is not None else None, T)
            results.append(
                Transcript(
                    text=text,
                    words=words,
                    mode=modes[b],
                    confidence=confidences[b],
                    event_probs=event_probs[b, :T].cpu() if event_probs is not None else None,
                    frame_rate=self.frame_rate,
                    fast_text=fast_text[b],
                )
            )
        return results

    def _label_index(self, name: str) -> int | None:
        labels = self.cfg.heads.event_labels
        return labels.index(name) if name in labels else None

    def _music_ratio(self, probs, lengths, b) -> float:
        i = self._label_index("music")
        if i is None:
            return 0.0
        T = int(lengths[b])
        return float((probs[b, :T, i] > 0.5).float().mean()) if T else 0.0

    def _speech_prob(self, probs, lengths, b) -> float:
        idx = [i for i in (self._label_index("speech"), self._label_index("singing")) if i is not None]
        T = int(lengths[b])
        if not idx or T == 0:
            return 1.0
        return float(probs[b, :T][:, idx].max())

    def _edit(self, audio, alens, idx, hyps_text, languages, contexts, glossaries) -> list[str]:
        sub = [audio[b] for b in idx]
        pieces = self._prompt_pieces(
            torch.stack(sub),
            alens[idx],
            [languages[b] for b in idx],
            [contexts[b] for b in idx],
            [glossaries[b] for b in idx],
        )
        for parts, b in zip(pieces, idx):
            slots = self._slots(self.llm_tok.encode(normalize_text(hyps_text[b])))
            parts.append(("ids", torch.tensor(slots, dtype=torch.long)))
        ids, emb, mask, spans = self._assemble(pieces)
        hidden = self.llm.hidden(ids, emb, mask, bidirectional=True)
        out = []
        for i in range(len(idx)):
            s, e = spans[i][-1]
            best = self.llm.logits(hidden[i, s:e]).argmax(-1).tolist()
            out.append(self.llm_tok.decode(ctc_collapse(best, self.llm_tok.pad_id)))
        return out

    def _generate(self, audio, alens, idx, languages, contexts, glossaries) -> list[str]:
        pieces = self._prompt_pieces(
            torch.stack([audio[b] for b in idx]),
            alens[idx],
            [languages[b] for b in idx],
            [contexts[b] for b in idx],
            [glossaries[b] for b in idx],
        )
        ids, emb, mask, _ = self._assemble(pieces, left_pad=True)
        out = self.llm.generate(ids, emb, mask, self.cfg.llm.max_new_tokens, self.llm_tok.eos_ids)
        return [self.llm_tok.decode(o) for o in out]

    def align_words(self, text: str, ctc_lp: torch.Tensor | None, num_frames: int) -> list[Word]:
        """Word timestamps by CTC forced alignment; uniform spacing as a fallback."""
        words = text.split()
        if not words:
            return []
        fr = self.frame_rate
        spans = None
        if ctc_lp is not None:
            ids, owner = self.ctc_tok.encode_words_for_alignment(words)
            tok_spans = forced_align(ctc_lp, ids, self.ctc_tok.blank_id) if ids else None
            if tok_spans is not None:
                spans = [[None, None] for _ in words]
                for (s, e), w in zip(tok_spans, owner):
                    if w < 0 or s < 0:
                        continue
                    spans[w][0] = s if spans[w][0] is None else min(spans[w][0], s)
                    spans[w][1] = e if spans[w][1] is None else max(spans[w][1], e)
        if spans is None:
            step = num_frames / len(words)
            spans = [[i * step, (i + 1) * step] for i in range(len(words))]
        # Words with no alignable characters take the gap between their neighbours.
        for i, sp in enumerate(spans):
            if sp[0] is None:
                prev_end = next((spans[j][1] for j in range(i - 1, -1, -1) if spans[j][1] is not None), 0)
                next_start = next((spans[j][0] for j in range(i + 1, len(spans)) if spans[j][0] is not None), num_frames)
                sp[0], sp[1] = prev_end, max(prev_end, next_start)
        return [Word(w, s / fr, e / fr) for w, (s, e) in zip(words, spans)]
