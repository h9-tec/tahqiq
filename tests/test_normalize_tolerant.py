from pathlib import Path

from harfiyyan.normalize.tolerant import normalize_tolerant as n


def test_diacritized_equals_undiacritized():
    assert n("اَلسَّلَامُ عَلَيْكُم!") == n("السلام عليكم")


def test_alef_variants_unified():
    forms = ["اسم", "أسم", "إسم", "آسم", "ٱسم"]
    out = {n(s) for s in forms}
    assert out == {"اسم"}


def test_ya_alef_maksura_unified():
    assert n("على") == n("علي")


def test_ta_marbuta_to_ha():
    assert n("مدرسة") == n("مدرسه")


def test_digits():
    assert n("٢٠٢٦") == "2026"
    assert n("۱۲۳ aa") == "123 aa"


def test_punctuation_becomes_space():
    assert n("مرحبا،كيف الحال؟") == "مرحبا كيف الحال"


def test_tatweel_removed():
    assert n("الـسـلام") == "السلام"


def test_hamza_folded():
    assert n("سؤال") == n("سوال")
    assert n("قائل") == n("قايل")
    assert "ء" not in n("شيء")


def test_repeats_not_collapsed():
    out = n("ما في شيء ما في شيء").split()
    assert out.count("ما") == 2 and out.count("شي") == 2


def test_empty():
    assert n("") == "" and n("   ") == ""


def test_idempotent():
    s = "قال: «أحمد» ذهب إلى المدرسة، ٢٠٢٦!"
    assert n(n(s)) == n(s)


def test_variants_table(tmp_path: Path):
    p = tmp_path / "variants.yaml"
    p.write_text("variants:\n  ليش: [ليه, لويش]\n  شلون: [شلونك_no]\n", encoding="utf-8")
    assert n("ليه رحت", variants_path=str(p)) == "ليش رحت"
    # Untouched words pass through.
    assert n("شلون رحت", variants_path=str(p)) == "شلون رحت"
