"""sacrebleu wrapper (chrF++ primary, spBLEU/flores200), paired bootstrap, hypothesis verdicts, reports.

Bootstrap: one index matrix (seeded) of shape [B, N] is shared by every system that is scored
against the same references, so all comparisons are paired. Scores per resample are recomputed
from summed sentence-level sufficient statistics, exactly as sacrebleu computes corpus scores.
"""
from __future__ import annotations

import csv
import logging
from functools import lru_cache

import numpy as np

from .data import load_flores
from .utils import out_dir, read_json, read_jsonl, write_json

log = logging.getLogger(__name__)

METRICS = ("chrf++", "bleu")


# ---------------------------------------------------------------- metric wrapper
def make_metrics(chrf_word_order: int = 2, bleu_tokenize: str = "flores200") -> dict:
    from sacrebleu.metrics import BLEU, CHRF

    return {"chrf++": CHRF(word_order=chrf_word_order), "bleu": BLEU(tokenize=bleu_tokenize)}


@lru_cache(maxsize=4)
def _metrics_cached(word_order: int, tokenize: str) -> dict:
    return make_metrics(word_order, tokenize)


def metrics_for(cfg: dict) -> dict:
    e = cfg["evaluation"]
    return _metrics_cached(e["chrf_word_order"], e["bleu_tokenize"])


def corpus_scores(cfg_or_metrics, hyps: list[str], refs: list[str]) -> dict[str, float]:
    ms = cfg_or_metrics if "chrf++" in cfg_or_metrics else metrics_for(cfg_or_metrics)
    _check(hyps, refs)
    return {name: float(m.corpus_score(hyps, [refs]).score) for name, m in ms.items()}


def sentence_stats(metric, hyps: list[str], refs: list[str]) -> np.ndarray:
    """[N, k] sufficient statistics; their column sum reproduces the corpus score."""
    _check(hyps, refs)
    return np.asarray(metric._extract_corpus_statistics(hyps, [refs]), dtype=np.float64)


def score_from_stats(metric, summed: np.ndarray) -> float:
    vec = [int(x) if float(x).is_integer() else float(x) for x in summed]
    return float(metric._compute_score_from_stats(vec).score)


def _check(hyps, refs):
    if len(hyps) != len(refs):
        raise ValueError(f"{len(hyps)} hypotheses vs {len(refs)} references")
    if any(h is None for h in hyps):
        raise ValueError("missing hypotheses (None)")


def bootstrap_indices(n: int, samples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(samples, n))


def bootstrap_scores(metric, hyps: list[str], refs: list[str], idx: np.ndarray) -> np.ndarray:
    stats = sentence_stats(metric, hyps, refs)
    return np.array([score_from_stats(metric, stats[row].sum(axis=0)) for row in idx])


def percentile_ci(values: np.ndarray, ci: float) -> tuple[float, float]:
    a = (1 - ci) / 2
    return float(np.quantile(values, a)), float(np.quantile(values, 1 - a))


def paired_delta(point_a: float, point_b: float, boot_a: np.ndarray, boot_b: np.ndarray, ci: float) -> dict:
    d = boot_a - boot_b
    lo, hi = percentile_ci(d, ci)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))  # two-sided, from the bootstrap distribution
    return {"delta": point_a - point_b, "ci_low": lo, "ci_high": hi, "p_value": p}


def verdict(delta: float, ci_low: float, threshold: float) -> str:
    if ci_low >= threshold:
        return "PASS"
    if delta >= threshold:
        return "INCONCLUSIVE"
    return "FAIL"


# ---------------------------------------------------------------- loading systems / refs
def load_refs(cfg: dict, lang: str, split: str) -> list[str]:
    return [r["text"] for r in load_flores(cfg, lang, split)]


def _ref_ids(cfg, lang, split):
    return [r["id"] for r in load_flores(cfg, lang, split)]


def load_system(cfg: dict, name: str, split: str) -> list[str]:
    manifest = read_json(out_dir(cfg, "systems") / "manifest.json", {})
    if name not in manifest or split not in manifest[name].get("splits", {}):
        raise KeyError(f"system {name!r} has no {split} output")
    rows = read_jsonl(out_dir(cfg, "systems") / manifest[name]["splits"][split])
    ids = _ref_ids(cfg, manifest[name]["tgt_lang"], split)
    if [r["id"] for r in rows] != ids:
        raise ValueError(f"{name}/{split}: ids not aligned with FLORES references")
    return [r["hyp"] for r in rows]


def resolve_name(cfg: dict, name: str) -> str:
    if "@selected" in name:
        sel = read_json(out_dir(cfg) / "step2_selection.json", {})
        return name.replace("@selected", sel.get("selected", "B1"))
    return name


# ---------------------------------------------------------------- main report
def evaluate(cfg: dict, split: str | None = None) -> dict:
    e = cfg["evaluation"]
    split = split or cfg["flores"]["test_split"]
    manifest = read_json(out_dir(cfg, "systems") / "manifest.json", {})
    names = [n for n, m in manifest.items() if split in m.get("splits", {})]
    if not names:
        raise RuntimeError("no system outputs yet; run step1/step2 first")
    ms = metrics_for(cfg)

    by_tgt: dict[str, list[str]] = {}
    for n in names:
        by_tgt.setdefault(manifest[n]["tgt_lang"], []).append(n)

    point, boot = {}, {}
    for tgt, group in by_tgt.items():
        refs = load_refs(cfg, tgt, split)
        idx = bootstrap_indices(len(refs), e["bootstrap_samples"], cfg["seed"])
        for n in group:
            hyps = load_system(cfg, n, split)
            point[n] = corpus_scores(ms, hyps, refs)
            boot[n] = {m: bootstrap_scores(ms[m], hyps, refs, idx) for m in METRICS}
            log.info("%-28s chrF++ %.2f  BLEU %.2f", n, point[n]["chrf++"], point[n]["bleu"])

    rows = []
    for n in names:
        meta = manifest[n]
        for m in METRICS:
            lo, hi = percentile_ci(boot[n][m], e["ci"])
            rows.append({"system": n, "kind": meta.get("kind"), "model": meta.get("model"),
                         "source": meta.get("source"), "route": meta.get("route"),
                         "src_lang": meta.get("src_lang"), "tgt_lang": meta.get("tgt_lang"),
                         "split": split, "metric": m, "score": round(point[n][m], 2),
                         "ci_low": round(lo, 2), "ci_high": round(hi, 2), "n_boot": e["bootstrap_samples"]})

    comps = []
    def compare(label, a, b, metric, threshold=None):
        if a not in point or b not in point:
            log.info("skip %s: %s or %s not available", label, a, b)
            return
        if manifest[a]["tgt_lang"] != manifest[b]["tgt_lang"]:
            raise ValueError(f"{label}: {a} and {b} use different references")
        d = paired_delta(point[a][metric], point[b][metric], boot[a][metric], boot[b][metric], e["ci"])
        rec = {"comparison": label, "metric": metric, "a": a, "b": b,
               "score_a": round(point[a][metric], 2), "score_b": round(point[b][metric], 2),
               "delta": round(d["delta"], 2), "ci_low": round(d["ci_low"], 2), "ci_high": round(d["ci_high"], 2),
               "p_value": round(d["p_value"], 4), "threshold": threshold,
               "verdict": verdict(d["delta"], d["ci_low"], threshold) if threshold is not None else ""}
        comps.append(rec)

    for h, spec in e["hypotheses"].items():
        for m in METRICS:
            compare(h, resolve_name(cfg, spec["a"]), resolve_name(cfg, spec["b"]), m,
                    spec["threshold"] if m == spec["metric"] else None)
    for a, b in e["comparisons"]:
        a, b = resolve_name(cfg, a), resolve_name(cfg, b)
        for m in METRICS:
            compare(f"{a} vs {b}", a, b, m)

    od = out_dir(cfg)
    _write_csv(od / "results.csv", rows)
    _write_csv(od / "comparisons.csv", comps)
    signatures = {m: str(ms[m].get_signature()) for m in METRICS}
    write_json(od / "metric_signatures.json", signatures)
    (od / "results.md").write_text(_markdown(cfg, split, rows, comps, signatures), encoding="utf-8")
    log.info("wrote %s, %s, %s", od / "results.csv", od / "comparisons.csv", od / "results.md")
    return {"rows": rows, "comparisons": comps}


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _markdown(cfg, split, rows, comps, signatures) -> str:
    ci = int(cfg["evaluation"]["ci"] * 100)
    L = [f"# Results — FLORES+ {split} (rev {cfg['flores']['revision'][:8]})", ""]
    hnames = cfg["evaluation"]["hypotheses"]
    hyp = [c for c in comps if c["comparison"] in hnames]
    if hyp:
        L += ["## Hypotheses", "",
              f"| | metric | a | b | a | b | Δ (a−b) | {ci}% CI of Δ | threshold | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c in hyp:
            L.append(f"| {c['comparison']} | {c['metric']} | `{c['a']}` | `{c['b']}` | {c['score_a']:.2f} | "
                     f"{c['score_b']:.2f} | {c['delta']:+.2f} | [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}] | "
                     + (f"≥ {c['threshold']} | **{c['verdict']}** |" if c["verdict"] else "— | (secondary) |"))
        L += ["", "Verdict rule: PASS if the CI lower bound ≥ threshold; INCONCLUSIVE if the point estimate ≥ "
              "threshold but the CI lower bound is below it; FAIL if the point estimate < threshold.", ""]

    L += ["## Systems", "", f"| system | route | source | target | chrF++ [{ci}% CI] | BLEU [{ci}% CI] |",
          "|---|---|---|---|---|---|"]
    by = {}
    for r in rows:
        by.setdefault(r["system"], {})[r["metric"]] = r
    for s in sorted(by, key=lambda s: (by[s]["chrf++"]["tgt_lang"], s)):
        c, b = by[s]["chrf++"], by[s]["bleu"]
        L.append(f"| `{s}` | {c['route']} | {c['source']} | {c['tgt_lang']} | "
                 f"{c['score']:.2f} [{c['ci_low']:.2f}, {c['ci_high']:.2f}] | "
                 f"{b['score']:.2f} [{b['ci_low']:.2f}, {b['ci_high']:.2f}] |")

    other = [c for c in comps if c["comparison"] not in hnames]
    if other:
        L += ["", "## Paired comparisons", "", f"| a | b | metric | a | b | Δ | {ci}% CI | p |",
              "|---|---|---|---|---|---|---|---|"]
        for c in other:
            L.append(f"| `{c['a']}` | `{c['b']}` | {c['metric']} | {c['score_a']:.2f} | {c['score_b']:.2f} | "
                     f"{c['delta']:+.2f} | [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}] | {c['p_value']:.3f} |")

    ds = read_json(out_dir(cfg) / f"diacritic_stats.{split}.json")
    if ds:
        L += ["", "## Diacritic stripping (source b)", "",
              f"{ds['sentences_changed']}/{ds['sentences']} Arabic sentences changed "
              f"({ds['pct_sentences_changed']}%), {ds['chars_removed']} characters removed "
              f"({ds['pct_chars_removed']}% of characters)."]
    sel = read_json(out_dir(cfg) / "step2_selection.json")
    if sel:
        L += ["", f"Step-3 route: **{sel['selected']}** (chrF++ on {sel['selected_on']}: "
              + ", ".join(f"{k} {v:.2f}" for k, v in sel["chrf++"].items()) + ")."]
    L += ["", "## sacrebleu signatures", ""] + [f"- {k}: `{v}`" for k, v in signatures.items()]
    L += ["", f"Bootstrap: {cfg['evaluation']['bootstrap_samples']} paired resamples, seed {cfg['seed']}, "
          "percentile intervals."]
    return "\n".join(L) + "\n"
