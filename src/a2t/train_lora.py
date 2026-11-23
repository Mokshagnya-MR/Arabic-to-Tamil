"""Step 4: LoRA adapters (peft) on frozen NLLB-200, trained on the synthetic arb<->tam corpus.

Tuning and checkpoint selection use FLORES dev only (chrF++ arb->tam). Devtest is decoded once,
with the selected adapter, at the very end.
"""
from __future__ import annotations

import csv
import itertools
import logging
import random
from pathlib import Path

from .data import load_source
from .evaluate import corpus_scores, load_refs
from .translate import ModelPool, cached_translate, save_system
from .utils import dtype_kwarg, out_dir, read_json, read_jsonl, set_seed, write_json

log = logging.getLogger(__name__)
LANG_FIELD = {"arb_Arab": "arb", "tam_Taml": "tam"}
SOURCE_OF = {"arb_Arab": "arb", "tam_Taml": "tam"}


def latest_corpus(cfg: dict) -> Path:
    latest = read_json(out_dir(cfg, "synth") / "latest.json")
    if not latest:
        raise RuntimeError("no synthetic corpus; run `a2t step3` first")
    return out_dir(cfg, "synth") / latest["corpus"]


def build_examples(cfg: dict, pairs: list[dict], tokenizer, max_length: int) -> list[dict]:
    """One example per pair per direction; tokenised with the right NLLB language tokens."""
    examples = []
    for src, tgt in cfg["step4"]["directions"]:
        tokenizer.src_lang, tokenizer.tgt_lang = src, tgt
        srcs = [p[LANG_FIELD[src]] for p in pairs]
        tgts = [p[LANG_FIELD[tgt]] for p in pairs]
        for i in range(0, len(srcs), 2048):
            enc = tokenizer(srcs[i:i + 2048], text_target=tgts[i:i + 2048], max_length=max_length, truncation=True)
            for j in range(len(enc["input_ids"])):
                examples.append({"input_ids": enc["input_ids"][j], "attention_mask": enc["attention_mask"][j],
                                 "labels": enc["labels"][j]})
    random.Random(cfg["seed"]).shuffle(examples)
    return examples


EPOCH_FILE = "epoch_checkpoints.json"


def checkpoints(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))


def epoch_steps(run_dir: Path) -> set[int]:
    return set(read_json(run_dir / EPOCH_FILE) or [])


def epoch_checkpoints(run_dir: Path) -> list[Path]:
    """Checkpoints saved at epoch ends (the only ones scored on dev). Runs from before mid-epoch
    saving existed have no EPOCH_FILE and only epoch checkpoints, so all of them count."""
    if not (run_dir / EPOCH_FILE).exists():
        return checkpoints(run_dir)
    steps = epoch_steps(run_dir)
    return [c for c in checkpoints(run_dir) if int(c.name.split("-")[1]) in steps]


def prune_checkpoints(run_dir: Path, keep_step: int | None = None) -> None:
    """Delete mid-epoch checkpoints except `keep_step` (the newest, needed to resume)."""
    import shutil

    keep = epoch_steps(run_dir) | ({keep_step} if keep_step is not None else set())
    for c in checkpoints(run_dir):
        if int(c.name.split("-")[1]) not in keep:
            shutil.rmtree(c, ignore_errors=True)


def make_checkpoint_callback(run_dir: Path):
    """Force a save at every epoch end (recorded in EPOCH_FILE) and keep only the newest mid-epoch save."""
    from transformers import TrainerCallback

    class CheckpointCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            write_json(run_dir / EPOCH_FILE, sorted(epoch_steps(run_dir) | {state.global_step}))
            control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            prune_checkpoints(run_dir, keep_step=state.global_step)
            return control

    return CheckpointCallback()


def train_adapter(cfg: dict, examples: list[dict], run_dir: Path, lr: float, r: int, epochs: float,
                  max_steps: int = -1) -> list[Path]:
    """Train one adapter; returns the per-epoch checkpoint directories.

    Checkpoints are saved every `step4.train.save_steps` optimizer steps (only the newest is kept) and
    at every epoch end (all kept). A run counts as finished only once log_history.json is written. An
    unfinished run resumes from its newest checkpoint (weights, optimizer, scheduler, RNG, data position),
    so a stopped job loses at most `save_steps` steps. `max_steps > 0` is only for timing probes
    (no checkpoints are saved)."""
    if max_steps <= 0 and (run_dir / "log_history.json").exists():
        log.info("%s: finished earlier, reusing %d checkpoint(s)", run_dir.name, len(epoch_checkpoints(run_dir)))
        return epoch_checkpoints(run_dir)
    resume = checkpoints(run_dir)[-1] if max_steps <= 0 and checkpoints(run_dir) else None
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)

    from .translate import limit_vram

    if torch.cuda.is_available():
        limit_vram(cfg)  # get a real OOM instead of silent paging to system RAM on Windows
    s4 = cfg["step4"]
    t, lc = s4["train"], s4["lora"]
    spec = cfg["models"][s4["base_model"]]
    set_seed(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec.get("revision"))
    use_bf16 = t["bf16"] and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForSeq2SeqLM.from_pretrained(spec["hf_id"], revision=spec.get("revision"),
                                                  **dtype_kwarg(torch.bfloat16 if use_bf16 else torch.float32))
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=r, lora_alpha=lc["alpha"] * r // lc["r"],
                                             lora_dropout=lc["dropout"], target_modules=lc["target_modules"]))
    model.print_trainable_parameters()

    args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=t["per_device_batch_size"], gradient_accumulation_steps=t["grad_accum"],
        warmup_ratio=t["warmup_ratio"], weight_decay=t["weight_decay"], lr_scheduler_type="linear",
        bf16=use_bf16, gradient_checkpointing=t["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=max_steps, logging_steps=50 if max_steps <= 0 else 10,
        save_strategy="steps" if max_steps <= 0 else "no", save_steps=t.get("save_steps", 500),
        report_to=[], seed=cfg["seed"], data_seed=cfg["seed"],
        dataloader_num_workers=0, remove_unused_columns=False, group_by_length=True,
    )
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=examples,
                             data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100),
                             callbacks=[make_checkpoint_callback(run_dir)] if max_steps <= 0 else None)
    if resume:
        log.info("%s: resuming from %s", run_dir.name, resume.name)
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    if max_steps <= 0:
        prune_checkpoints(run_dir)  # training done: only epoch checkpoints are needed now
        write_json(run_dir / "log_history.json", trainer.state.log_history)
    del trainer, model
    torch.cuda.empty_cache()
    return epoch_checkpoints(run_dir)


def dev_score(cfg: dict, pool: ModelPool, adapter: Path, src: str = "arb_Arab", tgt: str = "tam_Taml") -> float:
    split = cfg["flores"]["dev_split"]
    items = load_source(cfg, SOURCE_OF[src], split)
    rows = cached_translate(pool, cfg, cfg["step4"]["base_model"], items, src, tgt, tag=f"lora.dev.{adapter.name}",
                            adapter_path=str(adapter))
    return corpus_scores(cfg, [r["hyp"] for r in rows], load_refs(cfg, tgt, split))["chrf++"]


def step4(cfg: dict, corpus: str | None = None, tune: bool = True) -> dict:
    from transformers import AutoTokenizer

    s4 = cfg["step4"]
    ld = out_dir(cfg, "lora")
    corpus_path = Path(corpus) if corpus else latest_corpus(cfg)
    pairs = read_jsonl(corpus_path)
    spec = cfg["models"][s4["base_model"]]
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec.get("revision"))
    examples = build_examples(cfg, pairs, tok, s4["train"]["max_length"])
    log.info("corpus %s: %d pairs -> %d directional examples", corpus_path.name, len(pairs), len(examples))
    pool = ModelPool(cfg)

    lr, r = s4["train"]["learning_rate"], s4["lora"]["r"]
    tune_rows = []
    if tune:
        tu = s4["tune"]
        sub = examples[: tu["max_examples"]]
        keys = list(tu["grid"])
        for combo in itertools.product(*(tu["grid"][k] for k in keys)):
            hp = dict(zip(keys, combo))
            run = ld / "tune" / f"lr{hp['learning_rate']}_r{hp['r']}"
            ckpts = train_adapter(cfg, sub, run, hp["learning_rate"], hp["r"], tu["epochs"])
            pool.close()
            score = dev_score(cfg, pool, ckpts[-1])
            tune_rows.append({**hp, "dev_chrf++": round(score, 2), "checkpoint": str(ckpts[-1])})
            log.info("tune %s -> dev chrF++ %.2f", hp, score)
        with open(ld / "tuning.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(tune_rows[0]))
            w.writeheader(); w.writerows(tune_rows)
        best = max(tune_rows, key=lambda x: x["dev_chrf++"])
        lr, r = best["learning_rate"], best["r"]

    run = ld / f"final_lr{lr}_r{r}"
    ckpts = train_adapter(cfg, examples, run, lr, r, s4["train"]["epochs"])
    pool.close()
    ck_scores = {str(c): dev_score(cfg, pool, c) for c in ckpts}
    best_ckpt = Path(max(ck_scores, key=ck_scores.get))
    log.info("checkpoint dev chrF++: %s -> %s", ck_scores, best_ckpt.name)

    test = cfg["flores"]["test_split"]
    for src, tgt in s4["directions"]:
        name = "arb" if src == "arb_Arab" else "tam2arb"
        items = load_source(cfg, SOURCE_OF[src], test)
        rows = cached_translate(pool, cfg, s4["base_model"], items, src, tgt, tag=f"lora.{test}", adapter_path=str(best_ckpt))
        save_system(cfg, f"lora.{name}", test, rows, {"kind": "lora", "model": f"{s4['base_model']}+lora",
                    "source": SOURCE_OF[src], "src_lang": src, "tgt_lang": tgt, "route": f"{src}->{tgt} (LoRA)",
                    "adapter": str(best_ckpt)})
    # Direct NLLB baseline for the reverse direction (tam->arb), for the LoRA comparison.
    items = load_source(cfg, "tam", test)
    rows = cached_translate(pool, cfg, s4["base_model"], items, "tam_Taml", "arb_Arab", tag=f"direct.tam.{test}")
    save_system(cfg, f"direct.{s4['base_model']}.tam2arb", test, rows, {"kind": "direct", "model": s4["base_model"],
                "source": "tam", "src_lang": "tam_Taml", "tgt_lang": "arb_Arab", "route": "tam_Taml->arb_Arab"})
    pool.close()
    summary = {"corpus": corpus_path.name, "learning_rate": lr, "r": r, "tuning": tune_rows,
               "checkpoint_dev_chrf++": ck_scores, "selected_checkpoint": str(best_ckpt)}
    write_json(ld / "summary.json", summary)
    return summary
