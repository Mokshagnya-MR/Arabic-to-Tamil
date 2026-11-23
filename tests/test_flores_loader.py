import json

import pytest

from a2t.data import FloresError, load_parallel, load_source, parse_flores_jsonl
from a2t.utils import load_config


def _write(dirpath, split, lang, rows):
    p = dirpath / split / f"{lang}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            # same field layout as openlanguagedata/flores_plus
            f.write(json.dumps({"id": r[0], "iso_639_3": lang[:3], "iso_15924": lang[4:], "glottocode": "x",
                                "text": r[1], "url": "u", "domain": "d", "topic": "t",
                                "has_image": False, "has_hyperlink": False, "last_updated": "2"},
                               ensure_ascii=False) + "\n")


@pytest.fixture
def cfg():
    c = load_config()
    c["flores"]["expected_sizes"] = {"dev": 3, "devtest": 3}
    return c


@pytest.fixture
def flores_dir(tmp_path):
    _write(tmp_path, "devtest", "arb_Arab", [(2, "مرحبا بالعالم"), (1, "كَتَبَ الطالبُ"), (3, "جملة ثالثة")])
    _write(tmp_path, "devtest", "tam_Taml", [(1, "ஒன்று"), (2, "இரண்டு"), (3, "மூன்று")])
    _write(tmp_path, "devtest", "eng_Latn", [(1, "one"), (2, "two"), (3, "three")])
    return tmp_path


def test_parse_sorts_by_id_and_keeps_text(flores_dir):
    rows = parse_flores_jsonl(flores_dir / "devtest" / "arb_Arab.jsonl")
    assert [r["id"] for r in rows] == [1, 2, 3]
    assert rows[1]["text"] == "مرحبا بالعالم"


def test_parallel_alignment(cfg, flores_dir):
    data = load_parallel(cfg, ["arb_Arab", "tam_Taml", "eng_Latn"], "devtest", local_dir=flores_dir)
    assert [r["text"] for r in data["tam_Taml"]] == ["ஒன்று", "இரண்டு", "மூன்று"]


def test_misaligned_ids_raise(cfg, flores_dir):
    _write(flores_dir, "devtest", "mal_Mlym", [(1, "a"), (2, "b"), (4, "c")])
    with pytest.raises(FloresError):
        load_parallel(cfg, ["arb_Arab", "mal_Mlym"], "devtest", local_dir=flores_dir)


def test_wrong_size_raises(cfg, flores_dir):
    cfg["flores"]["expected_sizes"]["devtest"] = 1012
    with pytest.raises(FloresError, match="expected 1012"):
        load_parallel(cfg, ["arb_Arab"], "devtest", local_dir=flores_dir)


def test_duplicate_and_empty_rejected(tmp_path):
    _write(tmp_path, "dev", "x_Y", [(1, "a"), (1, "b")])
    with pytest.raises(FloresError, match="duplicate"):
        parse_flores_jsonl(tmp_path / "dev" / "x_Y.jsonl")
    _write(tmp_path, "dev", "z_Y", [(1, "  ")])
    with pytest.raises(FloresError, match="empty"):
        parse_flores_jsonl(tmp_path / "dev" / "z_Y.jsonl")


def test_source_variant_strips_diacritics(cfg, flores_dir):
    rows = load_source(cfg, "arb_nodiac", "devtest", local_dir=flores_dir)
    assert rows[0]["text"] == "كتب الطالب"
    assert load_source(cfg, "arb", "devtest", local_dir=flores_dir)[0]["text"] == "كَتَبَ الطالبُ"


@pytest.mark.online
def test_real_flores_devtest_sizes():
    c = load_config()
    data = load_parallel(c, ["arb_Arab", "eng_Latn", "tam_Taml", "mal_Mlym"], "devtest")
    assert all(len(v) == 1012 for v in data.values())
