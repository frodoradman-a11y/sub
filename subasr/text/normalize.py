"""Persian-aware text normalisation.

Two levels:
* ``normalize_text``  - canonical spelling used for training targets and output
  (Arabic letter variants unified, diacritics and tatweel removed, ZWNJ cleaned).
* ``normalize_for_scoring`` - additionally strips punctuation and case and
  unifies digits so WER/CER compare words, not formatting.
"""

from __future__ import annotations

import re
import unicodedata

ZWNJ = "‌"

_CHAR_MAP = {
    "ي": "ی",  # Arabic yeh -> Persian yeh
    "ى": "ی",  # alef maksura -> Persian yeh
    "ك": "ک",  # Arabic kaf -> Persian keheh
    "ة": "ه",  # teh marbuta -> heh
    "ۀ": "هٔ",  # heh with yeh above (decomposed form)
    "ـ": "",  # tatweel
    "‍": "",  # zero width joiner
    "​": "",  # zero width space
    "‎": "",  # LRM
    "‏": "",  # RLM
    "﻿": "",  # BOM
    " ": " ",  # nbsp
}

# Harakat, tanwin, shadda, sukun, superscript alef.
_DIACRITICS = re.compile("[ً-ْٰٕٓ]")

_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_TO_ASCII_DIGITS = str.maketrans(_PERSIAN_DIGITS + _ARABIC_DIGITS, "0123456789" * 2)
_TO_PERSIAN_DIGITS = str.maketrans("0123456789" + _ARABIC_DIGITS, _PERSIAN_DIGITS * 2)

_TABLE = str.maketrans({k: v for k, v in _CHAR_MAP.items()})
_SPACES = re.compile(r"\s+")
_ZWNJ_RUNS = re.compile(f"{ZWNJ}+")
_ZWNJ_NEAR_SPACE = re.compile(f" ?{ZWNJ} ?")


def normalize_text(text: str, digits: str = "keep") -> str:
    """Canonical spelling. ``digits``: "keep", "ascii" or "persian"."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_TABLE)
    text = _DIACRITICS.sub("", text)
    # NFC turns "ه" + hamza-above back into "ۀ"; keep a single canonical form.
    text = text.replace("هٔ", "ۀ")
    if digits == "ascii":
        text = text.translate(_TO_ASCII_DIGITS)
    elif digits == "persian":
        text = text.translate(_TO_PERSIAN_DIGITS)
    text = _ZWNJ_RUNS.sub(ZWNJ, text)
    text = _SPACES.sub(" ", text).strip()
    # ZWNJ next to a space or at the edges is meaningless.
    text = _ZWNJ_NEAR_SPACE.sub(lambda m: " " if " " in m.group(0) else ZWNJ, text)
    return text.strip(ZWNJ + " ")


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch)[0] in ("P", "S") and ch not in "%"


def normalize_for_scoring(text: str, zwnj: str = "remove") -> str:
    """Normalisation used before computing WER / CER.

    ``zwnj``: "remove" joins the parts ("می‌روم" -> "میروم"), "space" splits them,
    "keep" leaves the character.
    """
    text = normalize_text(text, digits="ascii").lower()
    text = "".join(" " if _is_punct(ch) else ch for ch in text)
    if zwnj == "remove":
        text = text.replace(ZWNJ, "")
    elif zwnj == "space":
        text = text.replace(ZWNJ, " ")
    return _SPACES.sub(" ", text).strip()
