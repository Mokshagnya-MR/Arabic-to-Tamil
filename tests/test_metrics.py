import numpy as np
import pytest
import sacrebleu

from a2t.evaluate import (bootstrap_indices, bootstrap_scores, corpus_scores, make_metrics, paired_delta,
                          score_from_stats, sentence_stats, verdict)

REFS = ["நான் பள்ளிக்குச் சென்றேன்.", "இன்று மழை பெய்கிறது.", "அவர் ஒரு புத்தகம் படித்தார்.",
        "தமிழ் ஒரு பழமையான மொழி.", "நாங்கள் நாளை சந்திப்போம்."]
GOOD = ["நான் பள்ளிக்கு சென்றேன்.", "இன்று மழை பெய்கிறது.", "அவர் புத்தகம் படித்தார்.",
        "தமிழ் பழமையான மொழி.", "நாங்கள் நாளை சந்திப்போம்."]
BAD = ["நான் சென்றேன்.", "மழை.", "அவர் படித்தார்.", "மொழி.", "நாளை."]


@pytest.fixture(scope="module")
def ms():
    # 13a keeps this test offline; flores200 (the real setting) is checked in test_flores200_tokenizer
    return make_metrics(2, "13a")


def test_chrfpp_matches_sacrebleu_directly(ms):
    ours = corpus_scores(ms, GOOD, REFS)["chrf++"]
    ref = sacrebleu.corpus_chrf(GOOD, [REFS], word_order=2).score
    assert ours == pytest.approx(ref)
    assert ms["chrf++"].get_signature().format().find("nw:2") >= 0


def test_identical_is_100(ms):
    s = corpus_scores(ms, REFS, REFS)
    assert s["chrf++"] == pytest.approx(100.0) and s["bleu"] == pytest.approx(100.0)


def test_stats_sum_reproduces_corpus_score(ms):
    for name in ("chrf++", "bleu"):
        st = sentence_stats(ms[name], GOOD, REFS)
        assert st.shape[0] == len(REFS)
        assert score_from_stats(ms[name], st.sum(0)) == pytest.approx(corpus_scores(ms, GOOD, REFS)[name])


def test_length_mismatch_raises(ms):
    with pytest.raises(ValueError):
        corpus_scores(ms, GOOD[:-1], REFS)


def test_paired_bootstrap_identical_systems_zero_delta(ms):
    idx = bootstrap_indices(len(REFS), 200, seed=1)
    a = bootstrap_scores(ms["chrf++"], GOOD, REFS, idx)
    d = paired_delta(1.0, 1.0, a, a.copy(), 0.95)
    assert d["delta"] == 0 and d["ci_low"] == 0 and d["ci_high"] == 0


def test_paired_bootstrap_detects_better_system(ms):
    idx = bootstrap_indices(len(REFS), 300, seed=1)
    g = bootstrap_scores(ms["chrf++"], GOOD, REFS, idx)
    b = bootstrap_scores(ms["chrf++"], BAD, REFS, idx)
    pg, pb = corpus_scores(ms, GOOD, REFS)["chrf++"], corpus_scores(ms, BAD, REFS)["chrf++"]
    d = paired_delta(pg, pb, g, b, 0.95)
    assert d["delta"] > 0 and d["ci_low"] > 0 and d["p_value"] < 0.05


def test_bootstrap_is_seeded():
    assert np.array_equal(bootstrap_indices(10, 5, 3), bootstrap_indices(10, 5, 3))


def test_verdict_rule():
    assert verdict(4.0, 3.2, 3.0) == "PASS"
    assert verdict(3.5, 2.1, 3.0) == "INCONCLUSIVE"
    assert verdict(2.9, 1.0, 3.0) == "FAIL"


@pytest.mark.online
def test_flores200_tokenizer():
    m = make_metrics(2, "flores200")
    assert corpus_scores(m, REFS, REFS)["bleu"] == pytest.approx(100.0)
    assert "flores200" in str(m["bleu"].get_signature())
