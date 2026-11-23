from a2t.synth import FloresOverlap, normalize_for_match, script_ratio, split_sentences

FLORES_AR = ["قال الباحثون إن الفئران التي كانت مصابة بمرض السكري أصبحت الآن غير مصابة به."]


def test_script_ratio():
    assert script_ratio("مرحبا بالعالم", "arb_Arab") == 1.0
    assert script_ratio("வணக்கம் உலகம்", "tam_Taml") == 1.0
    assert script_ratio("hello مرحبا", "arb_Arab") < 0.6


def test_split_sentences_arabic_question_mark():
    assert split_sentences("جملة أولى. جملة ثانية؟ ثالثة!\nفقرة جديدة.", "arb_Arab") == [
        "جملة أولى.", "جملة ثانية؟", "ثالثة!", "فقرة جديدة."]


def test_overlap_exact_after_normalisation():
    ov = FloresOverlap(FLORES_AR, "arb_Arab", 8)
    vocalised = "قالَ الباحثونَ إنّ الفئران التي كانت مصابة بمرض السكري أصبحت الآن غير مصابة به!"
    assert ov.hits(vocalised)  # diacritics and punctuation ignored


def test_overlap_shared_8gram():
    ov = FloresOverlap(FLORES_AR, "arb_Arab", 8)
    embedded = "وفي تقرير آخر الفئران التي كانت مصابة بمرض السكري أصبحت الآن غير مصابة به حسب الدراسة"
    assert ov.hits(embedded)
    assert not ov.hits("ذهب الطالب إلى المكتبة لقراءة كتاب عن تاريخ العلوم في القرن الماضي.")


def test_normalize_for_match_lowercase_and_punct():
    assert normalize_for_match("Hello, World!", "eng_Latn") == "hello world"


def test_normalize_keeps_tamil_words_whole():
    # Tamil vowel signs are combining marks; they must not be treated as separators.
    assert normalize_for_match("மாணவர் நூலகத்திற்குச் சென்றார்.", "tam_Taml") == "மாணவர் நூலகத்திற்குச் சென்றார்"


def test_tamil_overlap_no_false_positive():
    flores_ta = ["ஆராய்ச்சியாளர்கள் நீரிழிவு நோயால் பாதிக்கப்பட்ட எலிகள் இப்போது குணமடைந்துள்ளன என்று கூறினர்."]
    ov = FloresOverlap(flores_ta, "tam_Taml", 8)
    assert not ov.hits("தமிழ்நாடு இந்தியாவின் தெற்குப் பகுதியில் அமைந்துள்ள ஒரு மாநிலம் ஆகும், இதன் தலைநகரம் சென்னை.")
    assert ov.hits("ஆராய்ச்சியாளர்கள் நீரிழிவு நோயால் பாதிக்கப்பட்ட எலிகள் இப்போது குணமடைந்துள்ளன என்று கூறினர்!")
