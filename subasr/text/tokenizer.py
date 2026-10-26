"""Tokenizers.

* ``CharTokenizer`` - character vocabulary. Used for the CTC/TDT heads (blank id 0)
  and, with extra special tokens, as the tokenizer of the tiny test LLM.
* ``SentencePieceTokenizer`` - BPE units for the CTC/TDT heads (ids shifted by one so
  that 0 stays the blank).
* ``HFTokenizer`` - wrapper giving a Hugging Face tokenizer the small interface the
  model needs (``encode``, ``decode``, ``bos_id``, ``eos_ids``, ``pad_id``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from subasr.text.normalize import normalize_text

BLANK = "<blank>"
UNK = "<unk>"
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"


class CharTokenizer:
    """One token per character. ``specials`` occupy the first ids."""

    def __init__(self, chars: Iterable[str], specials: Iterable[str] = (BLANK, UNK)):
        self.specials = list(specials)
        self.vocab: list[str] = list(self.specials)
        for ch in chars:
            if ch not in self.vocab:
                self.vocab.append(ch)
        self.index = {tok: i for i, tok in enumerate(self.vocab)}
        self.unk_id = self.index.get(UNK, None)

    # --- construction -------------------------------------------------
    @classmethod
    def build(cls, texts: Iterable[str], min_count: int = 1, specials: Iterable[str] = (BLANK, UNK)):
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(normalize_text(text))
        chars = sorted(ch for ch, c in counts.items() if c >= min_count)
        return cls(chars, specials)

    @classmethod
    def load(cls, path: str | Path, specials: Iterable[str] = (BLANK, UNK)):
        specials = list(specials)
        chars = []
        for line in Path(path).read_text(encoding="utf-8").split("\n"):
            if line == "":
                continue
            tok = " " if line == "<space>" else line
            if tok not in specials:
                chars.append(tok)
        return cls(chars, specials)

    def save(self, path: str | Path) -> None:
        lines = ["<space>" if t == " " else t for t in self.vocab]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- interface ----------------------------------------------------
    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def blank_id(self) -> int:
        return self.index[BLANK]

    @property
    def pad_id(self) -> int:
        return self.index.get(PAD, self.index.get(BLANK, 0))

    @property
    def bos_id(self) -> int | None:
        return self.index.get(BOS)

    @property
    def eos_ids(self) -> list[int]:
        return [self.index[EOS]] if EOS in self.index else []

    def token_id(self, token: str) -> int | None:
        return self.index.get(token)

    def encode(self, text: str) -> list[int]:
        ids = []
        i = 0
        while i < len(text):
            # Special tokens written inline in prompt templates, e.g. "<bos>".
            if text[i] == "<":
                end = text.find(">", i)
                if end != -1 and text[i : end + 1] in self.index and text[i : end + 1] in self.specials:
                    ids.append(self.index[text[i : end + 1]])
                    i = end + 1
                    continue
            idx = self.index.get(text[i], self.unk_id)
            if idx is not None:
                ids.append(idx)
            i += 1
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        n_special = len(self.specials)
        return "".join(self.vocab[i] for i in ids if n_special <= i < len(self.vocab))

    def encode_words_for_alignment(self, words: list[str]) -> tuple[list[int], list[int]]:
        """Token ids for a word sequence plus, for every token, the index of its word (-1 = separator)."""
        ids: list[int] = []
        owner: list[int] = []
        space = self.index.get(" ")
        for w, word in enumerate(words):
            if w > 0 and space is not None:
                ids.append(space)
                owner.append(-1)
            for ch in word:
                idx = self.index.get(ch)
                if idx is not None and idx not in (self.blank_id, self.unk_id):
                    ids.append(idx)
                    owner.append(w)
        return ids, owner


class SentencePieceTokenizer:
    """SentencePiece model with ids shifted by +1; id 0 is the CTC blank."""

    def __init__(self, model_path: str | Path):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.unk_id = self.sp.unk_id() + 1

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size() + 1

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def blank_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        return [i + 1 for i in self.sp.encode(text)]

    def decode(self, ids: Iterable[int]) -> str:
        return self.sp.decode([i - 1 for i in ids if i > 0])

    def encode_words_for_alignment(self, words: list[str]) -> tuple[list[int], list[int]]:
        ids: list[int] = []
        owner: list[int] = []
        for w, word in enumerate(words):
            for piece in self.sp.encode(word):
                if piece + 1 != self.unk_id:
                    ids.append(piece + 1)
                    owner.append(w)
        return ids, owner


class HFTokenizer:
    """Adapter around a Hugging Face tokenizer (e.g. Gemma 4)."""

    def __init__(self, name_or_path: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name_or_path)

    @property
    def vocab_size(self) -> int:
        return len(self.tok)

    @property
    def pad_id(self) -> int:
        return self.tok.pad_token_id if self.tok.pad_token_id is not None else 0

    @property
    def bos_id(self) -> int | None:
        return self.tok.bos_token_id

    @property
    def eos_ids(self) -> list[int]:
        ids = {self.tok.eos_token_id} if self.tok.eos_token_id is not None else set()
        for name in ("<end_of_turn>", "<turn|>", "<|turn_end|>"):
            idx = self.tok.convert_tokens_to_ids(name)
            if isinstance(idx, int) and idx != self.tok.unk_token_id and idx is not None:
                ids.add(idx)
        return sorted(ids)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: Iterable[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=True)

    def apply_chat_template(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_ctc_tokenizer(kind: str, path: str | None):
    if kind == "char":
        if path is None:
            raise ValueError("model.ctc_tokenizer.path must point to a vocab file (see subasr-build-vocab)")
        return CharTokenizer.load(path)
    if kind == "spm":
        if path is None:
            raise ValueError("model.ctc_tokenizer.path must point to a SentencePiece model")
        return SentencePieceTokenizer(path)
    raise ValueError(f"unknown CTC tokenizer type '{kind}'")


LLM_SPECIALS = (PAD, UNK, BOS, EOS)


def build_llm_tokenizer(llm_type: str, path: str | None):
    if llm_type == "tiny":
        if path is None:
            raise ValueError("model.llm.tokenizer must point to a vocab file for the tiny LLM")
        return CharTokenizer.load(path, specials=LLM_SPECIALS)
    if llm_type == "gemma4":
        if path is None:
            raise ValueError("model.llm.tokenizer (or model.llm.pretrained) is required")
        return HFTokenizer(path)
    raise ValueError(f"unknown LLM type '{llm_type}'")
