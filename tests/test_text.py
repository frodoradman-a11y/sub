from subasr.metrics import cer_between, edit_distance, score
from subasr.text.normalize import ZWNJ, normalize_for_scoring, normalize_text
from subasr.text.tokenizer import LLM_SPECIALS, CharTokenizer


def test_normalize_arabic_variants_and_diacritics():
    assert normalize_text("كتاب علي") == "کتاب علی"
    assert normalize_text("كِتابٌ") == "کتاب"
    assert normalize_text("سـلـام") == "سلام"
    assert normalize_text("  سلام‍   دنیا \n") == "سلام دنیا"


def test_normalize_zwnj_and_digits():
    assert normalize_text(f"می{ZWNJ}{ZWNJ}روم") == f"می{ZWNJ}روم"
    assert normalize_text(f"می {ZWNJ}روم") == "می روم"
    assert normalize_text(f"{ZWNJ}سلام{ZWNJ}") == "سلام"
    assert normalize_text("سال ۱۴۰۳", digits="ascii") == "سال 1403"
    assert normalize_text("سال 1403", digits="persian") == "سال ۱۴۰۳"


def test_normalize_for_scoring():
    assert normalize_for_scoring("«سلام»، دنیا!") == "سلام دنیا"
    assert normalize_for_scoring(f"می{ZWNJ}روم") == "میروم"
    assert normalize_for_scoring(f"می{ZWNJ}روم", zwnj="space") == "می روم"
    assert normalize_for_scoring("Hello, World") == "hello world"
    assert normalize_for_scoring("۵۰%") == "50%"


def test_edit_distance_and_scores():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance([], ["a"]) == 1
    s = score(["سلام دنیا", "روز خوب"], ["سلام دنیا", "روز"])
    assert s.wer.errors == 1 and s.wer.total == 4
    assert abs(s.wer.rate - 0.25) < 1e-9
    assert s.cer.total == len("سلامدنیا") + len("روزخوب")
    assert cer_between("سلام", "سلام") == 0.0
    assert cer_between("", "") == 0.0
    assert cer_between("", "x") == 1.0


def test_char_tokenizer_roundtrip(tmp_path):
    tok = CharTokenizer.build(["سلام دنیا", "باران"])
    ids = tok.encode("سلام دنیا")
    assert tok.blank_id == 0 and 0 not in ids
    assert tok.decode(ids) == "سلام دنیا"
    assert tok.encode("x") == [tok.unk_id]
    tok.save(tmp_path / "v.txt")
    again = CharTokenizer.load(tmp_path / "v.txt")
    assert again.vocab == tok.vocab
    assert " " in again.vocab


def test_llm_tokenizer_specials_and_alignment_ids():
    tok = CharTokenizer("سلام ", specials=LLM_SPECIALS)
    ids = tok.encode("<bos>سلام<eos>")
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_ids[0]
    assert tok.decode(ids) == "سلام"
    ctc = CharTokenizer("سلام ")
    ids, owner = ctc.encode_words_for_alignment(["سلام", "x", "سلام"])
    space = ctc.token_id(" ")
    assert ids.count(space) == 2
    assert owner == [0, 0, 0, 0, -1, -1, 2, 2, 2, 2]
