import unicodedata

from a2t.data import diacritic_stats, strip_diacritics


def test_removes_harakat_shadda_sukun_tanwin():
    vocalised = "كَتَبَ الطَّالِبُ دَرْسًا"
    assert strip_diacritics(vocalised) == "كتب الطالب درسا"


def test_removes_every_code_point_in_spec():
    marks = [chr(c) for c in range(0x064B, 0x0653)] + ["ٰ", "ـ"]
    assert strip_diacritics("ب" + "".join(marks) + "ت") == "بت"


def test_dagger_alif_and_tatweel():
    assert strip_diacritics("هٰذا") == "هذا"          # U+0670
    assert strip_diacritics("جمـــيل") == "جميل"      # U+0640


def test_keeps_hamza_and_madda_letters():
    # Precomposed hamza/madda letters and combining hamza (U+0653-U+0655) are outside the spec.
    text = "أإآؤئء"
    assert strip_diacritics(text) == text
    assert strip_diacritics("أ") == "أ"


def test_leaves_other_scripts_and_digits_alone():
    text = "Tamil தமிழ் 123 ١٢٣ ؟،"
    assert strip_diacritics(text) == text


def test_idempotent_and_nfc_safe():
    s = unicodedata.normalize("NFC", "مُحَمَّدٌ")
    once = strip_diacritics(s)
    assert strip_diacritics(once) == once == "محمد"


def test_stats():
    st = diacritic_stats(["كَتَبَ", "كتب"])
    assert st["sentences_changed"] == 1 and st["chars_removed"] == 3 and st["sentences"] == 2
