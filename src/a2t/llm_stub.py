"""Step 5 (optional, gap G7): prompted-LLM baseline. Stub, disabled by default.

Same contract as Translator.translate: list of source strings in, list of target strings out, same order.
To enable: implement `_complete`, set step5.enabled/provider/model in the config, then run `a2t step5`.
Outputs go through save_system() so evaluate.py scores them like every other system.
"""
from __future__ import annotations

LANG_NAMES = {"arb_Arab": "Modern Standard Arabic", "tam_Taml": "Tamil", "eng_Latn": "English", "mal_Mlym": "Malayalam"}

PROMPT = (
    "Translate the following {src_name} sentence into {tgt_name}. "
    "Output only the translation, with no explanation.\n\n{text}"
)


class PromptedLLMTranslator:
    def __init__(self, cfg: dict):
        s5 = cfg.get("step5", {})
        if not s5.get("enabled"):
            raise RuntimeError("Step 5 is disabled (step5.enabled: false in the config).")
        self.provider, self.model = s5.get("provider"), s5.get("model")

    def prompt(self, text: str, src: str, tgt: str) -> str:
        return PROMPT.format(src_name=LANG_NAMES[src], tgt_name=LANG_NAMES[tgt], text=text)

    def translate(self, texts: list[str], src: str, tgt: str, **_) -> list[str]:
        return [self._complete(self.prompt(t, src, tgt)).strip() for t in texts]

    def _complete(self, prompt: str) -> str:  # pragma: no cover - provider-specific
        raise NotImplementedError("Implement the API call for your provider here (zero temperature, fixed seed).")


def step5(cfg: dict) -> None:
    from .data import load_source
    from .translate import save_system

    tr = PromptedLLMTranslator(cfg)
    split = cfg["flores"]["test_split"]
    items = load_source(cfg, "arb", split)
    hyps = tr.translate([it["text"] for it in items], "arb_Arab", "tam_Taml")
    save_system(cfg, "llm.arb", split, [{"id": it["id"], "hyp": h} for it, h in zip(items, hyps)],
                {"kind": "llm", "model": f"{tr.provider}:{tr.model}", "source": "arb",
                 "src_lang": "arb_Arab", "tgt_lang": "tam_Taml", "route": "arb_Arab->tam_Taml (prompted LLM)"})
