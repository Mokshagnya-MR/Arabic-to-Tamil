"""FLORES+ loading, Arabic diacritic stripping, and source variants."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

# Short vowels / tanwin / shadda / sukun (U+064B-U+0652), dagger alif (U+0670), tatweel (U+0640).
ARABIC_DIACRITICS_RE = re.compile("[ً-ْٰـ]")


def strip_diacritics(text: str) -> str:
    """Remove Arabic short vowels, dagger alif and tatweel. Nothing else is touched."""
    return ARABIC_DIACRITICS_RE.sub("", text)


def diacritic_stats(texts: list[str]) -> dict:
    """How much `strip_diacritics` actually changes a corpus (FLORES Arabic is mostly unvocalised)."""
    removed = [len(ARABIC_DIACRITICS_RE.findall(t)) for t in texts]
    total_chars = sum(len(t) for t in texts)
    return {
        "sentences": len(texts),
        "sentences_changed": sum(1 for r in removed if r),
        "chars_removed": sum(removed),
        "chars_total": total_chars,
        "pct_chars_removed": round(100 * sum(removed) / max(total_chars, 1), 4),
        "pct_sentences_changed": round(100 * sum(1 for r in removed if r) / max(len(texts), 1), 2),
    }


TRANSFORMS = {None: lambda s: s, "strip_diacritics": strip_diacritics}


# ---------------------------------------------------------------- FLORES+
class FloresError(RuntimeError):
    pass


def parse_flores_jsonl(path: str | Path) -> list[dict]:
    """Parse one FLORES+ file (`<split>/<lang>.jsonl`) into [{id, text}], sorted by id."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "id" not in rec or "text" not in rec:
                raise FloresError(f"{path}:{n}: missing 'id' or 'text'")
            text = unicodedata.normalize("NFC", rec["text"]).strip()
            if not text:
                raise FloresError(f"{path}:{n}: empty text")
            rows.append({"id": int(rec["id"]), "text": text})
    rows.sort(key=lambda r: r["id"])
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise FloresError(f"{path}: duplicate ids")
    return rows


def flores_file(cfg: dict, lang: str, split: str, local_dir: str | Path | None = None) -> Path:
    """Path to a FLORES+ file: from `local_dir` (tests / offline) or downloaded from the gated HF repo."""
    rel = f"{split}/{lang}.jsonl"
    if local_dir is not None:
        p = Path(local_dir) / rel
        if not p.exists():
            raise FloresError(f"missing {p}")
        return p
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

    fc = cfg["flores"]
    try:
        return Path(hf_hub_download(fc["repo_id"], rel, repo_type="dataset", revision=fc.get("revision")))
    except GatedRepoError as e:
        raise FloresError(
            f"{fc['repo_id']} is gated. Accept its terms at https://huggingface.co/datasets/{fc['repo_id']} "
            "and run `hf auth login` in this venv."
        ) from e
    except HfHubHTTPError as e:
        if getattr(e.response, "status_code", None) in (401, 403):
            raise FloresError(f"Not authorised for {fc['repo_id']}: run `hf auth login`.") from e
        raise


def load_flores(cfg: dict, lang: str, split: str, local_dir: str | Path | None = None) -> list[dict]:
    rows = parse_flores_jsonl(flores_file(cfg, lang, split, local_dir))
    expected = cfg.get("flores", {}).get("expected_sizes", {}).get(split)
    if expected is not None and len(rows) != expected:
        raise FloresError(f"{lang}/{split}: expected {expected} sentences, got {len(rows)}")
    return rows


def load_parallel(cfg: dict, langs: list[str], split: str, local_dir=None) -> dict[str, list[dict]]:
    """Load several languages and check they are aligned sentence-by-sentence (same ids, same order)."""
    data = {lang: load_flores(cfg, lang, split, local_dir) for lang in langs}
    ref_ids = None
    for lang, rows in data.items():
        ids = [r["id"] for r in rows]
        if ref_ids is None:
            ref_ids = ids
        elif ids != ref_ids:
            raise FloresError(f"{lang}/{split} is not aligned with {langs[0]}")
    return data


def load_source(cfg: dict, source: str, split: str, local_dir=None) -> list[dict]:
    """A named source variant from cfg['sources'] (e.g. arb_nodiac) as [{id, text}]."""
    spec = cfg["sources"][source]
    fn = TRANSFORMS[spec.get("transform")]
    return [{"id": r["id"], "text": fn(r["text"])} for r in load_flores(cfg, spec["lang"], split, local_dir)]
