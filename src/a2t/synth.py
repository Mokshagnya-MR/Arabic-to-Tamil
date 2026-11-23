"""Step 3: synthetic arb<->tam corpus from monolingual Wikipedia via the best pivot route.

Pre-translation  : NFC, sentence split, script check, 3-200 tokens, dedupe, FLORES dev/devtest overlap removal,
                   seeded shuffle, take N.
Translation      : Arabic mono through the forward route, Tamil mono through the matching reverse route,
                   in fixed 5k chunks (so a pilot run is exactly the first chunk of the full run).
Post-translation : drop empty / wrong-script / out-of-range outputs, length ratio > max_len_ratio,
                   FLORES overlap on the synthetic side, duplicate pairs.
"""
from __future__ import annotations

import logging
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

from .data import load_flores, strip_diacritics
from .pivot import selected_route
from .translate import ModelPool, cached_translate
from .utils import out_dir, write_json, write_jsonl

log = logging.getLogger(__name__)
CHUNK = 5000

SCRIPT_RANGES = {"arb_Arab": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
                 "tam_Taml": [(0x0B80, 0x0BFF)]}
SENT_FINAL = (".", "!", "?", "؟")
SENT_END = {"arb_Arab": re.compile(r"(?<=[.!?؟])\s+"), "tam_Taml": re.compile(r"(?<=[.!?])\s+")}


# ---------------------------------------------------------------- text utilities
def script_ratio(text: str, lang: str) -> float:
    letters = [c for c in text if unicodedata.category(c)[0] in "LM"]
    if not letters:
        return 0.0
    ranges = SCRIPT_RANGES[lang]
    ok = sum(1 for c in letters if any(a <= ord(c) <= b for a, b in ranges))
    return ok / len(letters)


def normalize_for_match(text: str, lang: str) -> str:
    """Lowercase, NFC, punctuation/symbols -> space. Combining marks are KEPT: Tamil vowel signs (Mc/Mn)
    are not `\\w` in Python's re, so a `\\W` regex would shatter every Tamil word into fragments."""
    t = unicodedata.normalize("NFC", text).lower()
    if lang == "arb_Arab":
        t = strip_diacritics(t)
    t = "".join(" " if unicodedata.category(c)[0] in "PSZC" or c == "_" else c for c in t)
    return " ".join(t.split())


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def n_tokens(text: str) -> int:
    return len(text.split())


class FloresOverlap:
    """Exact (normalised) match or any shared word n-gram with FLORES dev+devtest in one language."""

    def __init__(self, sentences: list[str], lang: str, n: int):
        self.lang, self.n = lang, n
        norm = [normalize_for_match(s, lang) for s in sentences]
        self.exact = set(norm)
        self.grams: set = set()
        for s in norm:
            self.grams |= ngrams(s.split(), n)

    def hits(self, text: str) -> bool:
        t = normalize_for_match(text, self.lang)
        return t in self.exact or bool(ngrams(t.split(), self.n) & self.grams)


def build_overlap(cfg: dict, lang: str) -> FloresOverlap:
    fc, f = cfg["flores"], cfg["step3"]["filter"]
    sents = [r["text"] for sp in (fc["dev_split"], fc["test_split"]) for r in load_flores(cfg, lang, sp)]
    return FloresOverlap(sents, lang, f["overlap_ngram"])


def split_sentences(doc: str, lang: str) -> list[str]:
    out = []
    for para in doc.split("\n"):
        para = para.strip()
        if not para:
            continue
        out += [s.strip() for s in SENT_END[lang].split(para) if s.strip()]
    return out


# ---------------------------------------------------------------- monolingual data
def mono_files(cfg: dict, lang: str) -> list[Path]:
    from huggingface_hub import HfApi, hf_hub_download

    m = cfg["step3"]["mono"]
    spec = m[lang]
    files = sorted(f for f in HfApi().list_repo_files(m["repo_id"], repo_type="dataset")
                   if f.startswith(spec["subset"] + "/") and f.endswith(".parquet"))[: spec["max_files"]]
    return [Path(hf_hub_download(m["repo_id"], f, repo_type="dataset")) for f in files]


def candidate_sentences(cfg: dict, lang: str, overlap: FloresOverlap, stats: Counter) -> list[str]:
    import pyarrow.parquet as pq

    f, m = cfg["step3"]["filter"], cfg["step3"]["mono"]
    per_doc = m.get("max_sents_per_doc", 5)
    seen, out, docs = set(), [], 0
    for path in mono_files(cfg, lang):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["text"], batch_size=1000):
            for doc in batch.column(0).to_pylist():
                docs += 1
                if docs > m["max_docs"]:
                    break
                kept = 0
                for s in split_sentences(unicodedata.normalize("NFC", doc), lang):
                    if kept >= per_doc:  # article cap reached: skip the rest of this article
                        stats["skip_per_doc_cap"] += 1; continue
                    stats["seen"] += 1
                    s = " ".join(s.split())
                    if not s.endswith(SENT_FINAL):  # headings, list items, captions
                        stats["drop_no_final_punct"] += 1; continue
                    if not f["min_tokens"] <= n_tokens(s) <= f["max_tokens"]:
                        stats["drop_length"] += 1; continue
                    if script_ratio(s, lang) < f["min_script_ratio"]:
                        stats["drop_script"] += 1; continue
                    key = normalize_for_match(s, lang)
                    if key in seen:
                        stats["drop_duplicate"] += 1; continue
                    seen.add(key)
                    if overlap.hits(s):
                        stats["drop_flores_overlap"] += 1; continue
                    out.append(s); kept += 1
            if docs > m["max_docs"]:
                break
        if docs > m["max_docs"]:
            break
    stats["docs"] = min(docs, m["max_docs"])
    stats["candidates"] = len(out)
    return out


# ---------------------------------------------------------------- route over chunks
def translate_chunks(pool, cfg, items, hops, tag):
    chunks = [items[i:i + CHUNK] for i in range(0, len(items), CHUNK)]
    for h, hop in enumerate(hops, 1):  # hop-major: each model loads once
        for c, chunk in enumerate(chunks):
            rows = cached_translate(pool, cfg, hop["model"], chunk, hop["src"], hop["tgt"], tag=f"{tag}.c{c}.hop{h}")
            chunks[c] = [{"id": r["id"], "text": r["hyp"]} for r in rows]
    return {it["id"]: it["text"] for ch in chunks for it in ch}


# ---------------------------------------------------------------- step 3
def step3(cfg: dict, n: int | None = None) -> Path:
    s3, f = cfg["step3"], cfg["step3"]["filter"]
    n = n or s3["n_per_direction"]
    route = selected_route(cfg)
    fwd, rev = cfg["step2"]["routes"][route], s3["reverse_routes"][route]
    sd = out_dir(cfg, "synth")
    rng = random.Random(cfg["seed"])

    overlap = {lang: build_overlap(cfg, lang) for lang in f["flores_overlap_langs"]}
    stats = {}
    mono = {}
    for lang, prefix in (("arb_Arab", "ar"), ("tam_Taml", "ta")):
        st = Counter()
        cands = candidate_sentences(cfg, lang, overlap[lang], st)
        rng.shuffle(cands)
        if len(cands) < n:
            log.warning("%s: only %d candidates for n=%d", lang, len(cands), n)
        mono[lang] = [{"id": f"{prefix}{i:07d}", "text": s} for i, s in enumerate(cands[:n])]
        stats[lang] = dict(st, sampled=len(mono[lang]))
        log.info("%s mono: %s", lang, stats[lang])

    pool = ModelPool(cfg)
    try:
        syn_tam = translate_chunks(pool, cfg, mono["arb_Arab"], fwd, f"synth.{route}.fwd")
        syn_arb = translate_chunks(pool, cfg, mono["tam_Taml"], rev, f"synth.{route}.rev")
    finally:
        pool.close()

    pairs = [{"id": it["id"], "arb": it["text"], "tam": syn_tam[it["id"]], "origin": "arb_mono"} for it in mono["arb_Arab"]]
    pairs += [{"id": it["id"], "arb": syn_arb[it["id"]], "tam": it["text"], "origin": "tam_mono"} for it in mono["tam_Taml"]]

    post = Counter()
    kept, seen = [], set()
    for p in pairs:
        syn_side = "tam" if p["origin"] == "arb_mono" else "arb"
        syn_lang = "tam_Taml" if syn_side == "tam" else "arb_Arab"
        syn = p[syn_side]
        la, lt = n_tokens(p["arb"]), n_tokens(p["tam"])
        if not syn.strip():
            post["drop_empty"] += 1; continue
        if not f["min_tokens"] <= n_tokens(syn) <= f["max_tokens"]:
            post["drop_length"] += 1; continue
        if script_ratio(syn, syn_lang) < f["min_script_ratio"]:
            post["drop_script"] += 1; continue
        if max(la, lt) / max(1, min(la, lt)) > f["max_len_ratio"]:
            post["drop_ratio"] += 1; continue
        if syn_lang in overlap and overlap[syn_lang].hits(syn):
            post["drop_flores_overlap"] += 1; continue
        k = (normalize_for_match(p["arb"], "arb_Arab"), normalize_for_match(p["tam"], "tam_Taml"))
        if k in seen or k[0] in seen or k[1] in seen:
            post["drop_duplicate"] += 1; continue
        seen.update([k, k[0], k[1]])
        kept.append(p)
    post["kept"] = len(kept)
    post["kept_arb_mono"] = sum(p["origin"] == "arb_mono" for p in kept)
    post["kept_tam_mono"] = sum(p["origin"] == "tam_mono" for p in kept)

    corpus = sd / f"corpus.{route}.n{n}.jsonl"
    write_jsonl(corpus, kept)
    write_json(sd / f"stats.{route}.n{n}.json", {"route": route, "n_per_direction": n, "pre": stats, "post": dict(post)})
    write_json(sd / "latest.json", {"corpus": corpus.name, "route": route, "n_per_direction": n})
    log.info("synthetic corpus: %s -> %s", dict(post), corpus)
    return corpus
