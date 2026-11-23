"""Model wrappers (NLLB-200, M2M100, IndicTrans2) and cached batch translation.

Cache: outputs/translations/<model>/<tag>.<key>.jsonl, one {"id", "src", "hyp"} per line,
appended after every batch. The key hashes model id + revision, language pair, generation
settings and the exact input texts, so a rerun resumes from the first missing id and any
change to inputs or settings starts a new file instead of silently mixing outputs.
"""
from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

from .utils import dtype_kwarg, out_dir, read_jsonl, resolve_device, resolve_dtype, sha

log = logging.getLogger(__name__)


def limit_vram(cfg: dict) -> None:
    """Cap PyTorch's allocator below the card's memory. On Windows (WDDM) exceeding VRAM does not raise OOM:
    the driver silently pages to system RAM and generation slows ~25x. With a cap we get a real
    OutOfMemoryError, which `translate` handles by halving the batch."""
    import torch

    frac = cfg.get("max_vram_fraction")
    if frac:
        torch.cuda.set_per_process_memory_fraction(float(frac), 0)


class Translator:
    """Loads one model and translates lists of sentences between FLORES language codes."""

    def __init__(self, cfg: dict, model_key: str, adapter_path: str | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.cfg = cfg
        self.key = model_key
        self.spec = cfg["models"][model_key]
        self.family = self.spec["family"]
        self.device = resolve_device(cfg.get("device", "auto"))
        self.dtype = resolve_dtype(cfg.get("dtype", "auto"), self.device)
        self.adapter_path = adapter_path
        hf_id, rev = self.spec["hf_id"], self.spec.get("revision")
        remote = self.family == "indictrans2"

        if self.device == "cuda":
            limit_vram(cfg)
        self.oom_events = 0
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=rev, trust_remote_code=remote)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            hf_id, revision=rev, trust_remote_code=remote, **dtype_kwarg(self.dtype)
        )
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.revision = rev or self._resolved_revision(hf_id)
        if self.family == "indictrans2":
            from IndicTransToolkit.processor import IndicProcessor

            self.ip = IndicProcessor(inference=True)
        log.info("loaded %s (%s) on %s/%s in %.1fs", model_key, hf_id, self.device, self.dtype, time.time() - t0)
        self._torch = torch

    @staticmethod
    def _resolved_revision(hf_id: str) -> str | None:
        try:
            from huggingface_hub import model_info

            return model_info(hf_id).sha
        except Exception:
            return None

    # ------------------------------------------------------------ language codes
    def code(self, flores_lang: str) -> str:
        return self.cfg.get("lang_codes", {}).get(self.family, {}).get(flores_lang, flores_lang)

    # ------------------------------------------------------------ one batch
    def _translate_batch(self, texts: list[str], src: str, tgt: str) -> list[str]:
        torch = self._torch
        g = self.cfg["generation"]
        tok = self.tokenizer
        gen_kw = dict(num_beams=g["beam_size"], max_new_tokens=g["max_new_tokens"], num_return_sequences=1)

        if self.family == "indictrans2":
            texts = self.ip.preprocess_batch(texts, src_lang=src, tgt_lang=tgt)
        elif self.family in ("nllb", "m2m100"):
            tok.src_lang = self.code(src)

        enc = tok(texts, return_tensors="pt", padding="longest", truncation=True,
                  max_length=g["max_input_tokens"]).to(self.device)

        if self.family == "nllb":
            gen_kw["forced_bos_token_id"] = tok.convert_tokens_to_ids(self.code(tgt))
        elif self.family == "m2m100":
            gen_kw["forced_bos_token_id"] = tok.get_lang_id(self.code(tgt))
        elif self.family == "indictrans2":
            gen_kw["use_cache"] = True
            gen_kw["min_length"] = 0

        with torch.inference_mode():
            out = self.model.generate(**enc, **gen_kw)
        hyps = tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if self.family == "indictrans2":
            hyps = self.ip.postprocess_batch(hyps, lang=tgt)
        return [h.strip() for h in hyps]

    # ------------------------------------------------------------ many sentences
    def translate(self, texts: list[str], src: str, tgt: str, batch_size: int | None = None,
                  on_batch=None) -> list[str]:
        """Translate in length-sorted batches (longest first, so OOM shows up early); keeps input order.
        `on_batch(indices, hyps)` is called after each batch (used for incremental caching)."""
        torch = self._torch
        bs = batch_size or self.spec.get("batch_size") or self.cfg["generation"]["batch_size"]
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        result: list[str | None] = [None] * len(texts)
        i = 0
        while i < len(order):
            idx = order[i:i + bs]
            try:
                hyps = self._translate_batch([texts[j] for j in idx], src, tgt)
            except torch.cuda.OutOfMemoryError:
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                self.oom_events += 1
                log.warning("CUDA OOM: batch size -> %d", bs)
                torch.cuda.empty_cache()
                continue
            for j, h in zip(idx, hyps):
                result[j] = h
            if on_batch:
                on_batch(idx, hyps)
            i += len(idx)
        return result  # type: ignore[return-value]

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device == "cuda":
            self._torch.cuda.empty_cache()


# ---------------------------------------------------------------- model pool (one model on GPU at a time)
class ModelPool:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.current: Translator | None = None

    def get(self, model_key: str, adapter_path: str | None = None) -> Translator:
        if self.current is None or self.current.key != model_key or self.current.adapter_path != adapter_path:
            if self.current is not None:
                self.current.close()
            self.current = Translator(self.cfg, model_key, adapter_path)
        return self.current

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None


# ---------------------------------------------------------------- cached translation
def _adapter_hash(adapter_path: str | None) -> str | None:
    """Content hash of the adapter weights, so a retrained adapter at the same path never reuses old outputs."""
    if not adapter_path:
        return None
    import hashlib

    h = hashlib.sha256()
    for f in sorted(Path(adapter_path).glob("adapter_model.*")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def configured_batch_size(cfg: dict, model_key: str, batch_size: int | None = None) -> int:
    return batch_size or cfg["models"][model_key].get("batch_size") or cfg["generation"]["batch_size"]


def cache_key(cfg: dict, model_key: str, src: str, tgt: str, items: list[dict],
              adapter_path: str | None = None, batch_size: int | None = None) -> str:
    spec = cfg["models"][model_key]
    g = cfg["generation"]
    return sha({
        "hf_id": spec["hf_id"], "revision": spec.get("revision"), "adapter": _adapter_hash(adapter_path),
        # fp16 padding effects make outputs depend (very slightly) on batch composition, so batch size is keyed
        "bs": configured_batch_size(cfg, model_key, batch_size),
        "src": src, "tgt": tgt, "beam": g["beam_size"], "max_new": g["max_new_tokens"],
        "max_in": g["max_input_tokens"], "dtype": cfg.get("dtype", "auto"),
        "inputs": sha([(it["id"], it["text"]) for it in items], 16),
    })


def cached_translate(pool: ModelPool, cfg: dict, model_key: str, items: list[dict], src: str, tgt: str,
                     tag: str, adapter_path: str | None = None, batch_size: int | None = None) -> list[dict]:
    """Translate [{id, text}] -> [{id, src, hyp}] in input order, resuming from the jsonl cache."""
    key = cache_key(cfg, model_key, src, tgt, items, adapter_path, batch_size)
    # The filename has no tag: identical work requested under two tags shares one cache file.
    path = out_dir(cfg, "translations", model_key) / f"{src}-{tgt}.n{len(items)}.{key}.jsonl"
    done = {r["id"]: r for r in read_jsonl(path)}
    todo = [it for it in items if it["id"] not in done]
    if todo:
        log.info("%s: %d/%d to translate -> %s", tag, len(todo), len(items), path.name)
        tr = pool.get(model_key, adapter_path)
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            meta_path.write_text(json.dumps({
                "first_tag": tag, "model": model_key, "hf_id": tr.spec["hf_id"], "revision": tr.revision, "adapter": adapter_path,
                "src": src, "tgt": tgt, "generation": cfg["generation"], "device": tr.device,
                "dtype": str(tr.dtype), "n": len(items),
            }, indent=2), encoding="utf-8")
        t0 = time.time()
        with open(path, "a", encoding="utf-8") as f:
            def on_batch(idx, hyps):
                for j, h in zip(idx, hyps):
                    it = todo[j]
                    rec = {"id": it["id"], "src": it["text"], "hyp": h}
                    done[it["id"]] = rec
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

            tr.translate([it["text"] for it in todo], src, tgt, batch_size=batch_size, on_batch=on_batch)
        dt = time.time() - t0
        peak = tr._torch.cuda.max_memory_allocated() / 2**30 if tr.device == "cuda" else 0.0
        log.info("%s: %d sentences in %.1fs (%.2f sent/s, peak %.2f GB, OOM halvings %d)",
                 tag, len(todo), dt, len(todo) / max(dt, 1e-9), peak, tr.oom_events)
        _log_speed(cfg, model_key, src, tgt, len(todo), dt, peak, tr.oom_events)
    else:
        log.info("%s: cached (%d) %s", tag, len(items), path.name)
    return [done[it["id"]] for it in items]


def _log_speed(cfg, model_key, src, tgt, n, seconds, peak_gb=0.0, oom=0):
    p = out_dir(cfg) / "throughput.jsonl"
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"model": model_key, "src": src, "tgt": tgt, "n": n,
                            "batch_size": configured_batch_size(cfg, model_key),
                            "seconds": round(seconds, 2), "sent_per_s": round(n / max(seconds, 1e-9), 3),
                            "peak_vram_gb": round(peak_gb, 2), "oom_halvings": oom}) + "\n")


# ---------------------------------------------------------------- system outputs (what evaluate.py scores)
def save_system(cfg: dict, name: str, split: str, rows: list[dict], meta: dict) -> Path:
    """Write final hypotheses of a named system plus its metadata into outputs/systems/."""
    from .utils import read_json, write_json, write_jsonl

    d = out_dir(cfg, "systems")
    path = d / f"{name}.{split}.jsonl"
    write_jsonl(path, [{"id": r["id"], "hyp": r["hyp"]} for r in rows])
    manifest_path = d / "manifest.json"
    manifest = read_json(manifest_path, {})
    manifest.setdefault(name, {}).update(meta)
    manifest[name].setdefault("splits", {})[split] = path.name
    write_json(manifest_path, manifest)
    return path
