"""Step 1 (direct baselines) and Step 2 (pivot routes + controls)."""
from __future__ import annotations

import logging

from .data import diacritic_stats, load_flores, load_source
from .translate import ModelPool, cached_translate, save_system
from .utils import out_dir, read_json, write_json

log = logging.getLogger(__name__)


def run_route(pool: ModelPool, cfg: dict, items: list[dict], hops: list[dict], tag: str,
              adapter_path: str | None = None) -> tuple[list[dict], list[list[dict]]]:
    """Run hop 1 on `items`, feed each hop's output to the next. Returns (final rows, per-hop rows)."""
    per_hop = []
    cur = items
    for h, hop in enumerate(hops, 1):
        rows = cached_translate(pool, cfg, hop["model"], cur, hop["src"], hop["tgt"],
                                tag=f"{tag}.hop{h}", adapter_path=adapter_path if h == 1 else None)
        per_hop.append(rows)
        cur = [{"id": r["id"], "text": r["hyp"]} for r in rows]
    return per_hop[-1], per_hop


def step1(cfg: dict, splits: tuple[str, ...] | None = None) -> None:
    s1 = cfg["step1"]
    splits = splits or (cfg["flores"]["test_split"],)
    pool = ModelPool(cfg)
    try:
        for split in splits:
            arb = [r["text"] for r in load_flores(cfg, "arb_Arab", split)]
            write_json(out_dir(cfg) / f"diacritic_stats.{split}.json", diacritic_stats(arb))
            # group by model so each model is loaded once
            for m in s1["models"]:
                for source in s1["sources"]:
                    items = load_source(cfg, source, split)
                    src_lang = cfg["sources"][source]["lang"]
                    rows = cached_translate(pool, cfg, m, items, src_lang, s1["target"], tag=f"direct.{source}.{split}")
                    save_system(cfg, f"direct.{m}.{source}", split, rows, {
                        "kind": "direct", "model": m, "source": source, "src_lang": src_lang,
                        "tgt_lang": s1["target"], "route": f"{src_lang}->{s1['target']}",
                    })
    finally:
        pool.close()


def step2(cfg: dict) -> dict:
    s2 = cfg["step2"]
    test, sel = cfg["flores"]["test_split"], s2.get("select_on", cfg["flores"]["dev_split"])
    splits = [sel] + ([test] if test != sel else [])
    systems = [("pivot", name, s2["source"], hops) for name, hops in s2["routes"].items()]
    systems += [("control", name, c["source"], c["hops"]) for name, c in s2.get("controls", {}).items()]

    # Run hop-by-hop across all systems and splits, grouped by model, so each model loads once per hop level.
    pool = ModelPool(cfg)
    try:
        cur = {(kind, name, split): load_source(cfg, source, split)
               for split in splits for kind, name, source, _ in systems}
        hops_of = {(kind, name): hops for kind, name, _, hops in systems}
        final: dict = {}
        for h in range(max(len(x[3]) for x in systems)):
            work = [(k, hops_of[k[:2]][h]) for k in cur if h < len(hops_of[k[:2]])]
            for (kind, name, split), hop in sorted(work, key=lambda w: w[1]["model"]):
                rows = cached_translate(pool, cfg, hop["model"], cur[(kind, name, split)], hop["src"], hop["tgt"],
                                        tag=f"{kind}.{name}.{split}.hop{h + 1}")
                final[(kind, name, split)] = rows
                cur[(kind, name, split)] = [{"id": r["id"], "text": r["hyp"]} for r in rows]
        for split in splits:
            for kind, name, source, hops in systems:
                save_system(cfg, f"{kind}.{name}", split, final[(kind, name, split)], {
                    "kind": kind, "source": source, "src_lang": hops[0]["src"], "tgt_lang": hops[-1]["tgt"],
                    "route": " -> ".join([hops[0]["src"]] + [f"{h['tgt']} ({h['model']})" for h in hops]),
                    "model": "+".join(h["model"] for h in hops),
                })
    finally:
        pool.close()
    return select_route(cfg)


def select_route(cfg: dict) -> dict:
    """Pick the Step-3 route: highest chrF++ on `step2.select_on` (dev by default) among step2.routes."""
    from .evaluate import corpus_scores, load_refs, load_system

    s2 = cfg["step2"]
    split = s2.get("select_on", cfg["flores"]["dev_split"])
    scores = {}
    for name in s2["routes"]:
        hyps = load_system(cfg, f"pivot.{name}", split)
        refs = load_refs(cfg, s2["routes"][name][-1]["tgt"], split)
        scores[name] = corpus_scores(cfg, hyps, refs)["chrf++"]
    best = max(scores, key=scores.get)
    sel = {"selected": best, "selected_on": split, "chrf++": scores}
    write_json(out_dir(cfg) / "step2_selection.json", sel)
    log.info("Step-3 route: %s (chrF++ on %s: %s)", best, split, scores)
    return sel


def selected_route(cfg: dict) -> str:
    r = cfg["step3"].get("route", "auto")
    if r != "auto":
        return r
    sel = read_json(out_dir(cfg) / "step2_selection.json")
    if not sel:
        raise RuntimeError("No Step-2 selection found; run `a2t step2` first or set step3.route.")
    return sel["selected"]
