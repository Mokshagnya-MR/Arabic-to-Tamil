"""Command line: one command per step.

    a2t check             environment, GPU, HF login and gated-repo access
    a2t download          fetch FLORES+ and all models; print commit SHAs to pin in the config
    a2t step1             direct baselines (G1, G2, G4)                     ~15 min
    a2t step2             pivot routes B1/B2 + controls, route selection (G3) ~15 min
    a2t evaluate          scores, CIs, paired bootstrap, H1/H2 verdicts
    a2t step3 [--pilot]   synthetic corpus                                   hours
    a2t step4 [--no-tune] LoRA on NLLB-200 (G6)                              hours
    a2t step5             prompted-LLM baseline stub (G7), disabled by default
    a2t all               step1 + step2 + evaluate (never starts step3/4)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from .utils import disable_power_throttling, env_info, load_config, log_env, out_dir, set_seed


def _setup_logging(cfg: dict, step: str) -> None:
    logs = out_dir(cfg, "logs")
    path = logs / f"{time.strftime('%Y%m%d-%H%M%S')}-{step}.log"
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")])
    for noisy in ("httpx", "httpx2", "httpcore2", "urllib3", "filelock", "huggingface_hub.file_download"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _gated_repos(cfg):
    yield cfg["flores"]["repo_id"], "dataset"
    for m in cfg["models"].values():
        yield m["hf_id"], "model"


def cmd_check(cfg, args) -> int:
    from huggingface_hub import HfApi

    print(json.dumps(env_info(), indent=2))
    api = HfApi()
    try:
        who = api.whoami()
        print(f"HF login: OK ({who.get('name')})")
    except Exception:
        print("HF login: NOT LOGGED IN  ->  run `hf auth login` (read token) in this venv")
    ok = True
    for repo, kind in _gated_repos(cfg):
        try:
            api.auth_check(repo, repo_type=kind)
            print(f"  access OK   {kind:7s} {repo}")
        except Exception as e:
            ok = False
            print(f"  NO ACCESS   {kind:7s} {repo}  ({type(e).__name__}) -> accept terms at "
                  f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}")
    return 0 if ok else 1


def cmd_download(cfg, args) -> int:
    from huggingface_hub import model_info, snapshot_download

    from .data import load_flores

    fc = cfg["flores"]
    for split in (fc["dev_split"], fc["test_split"]):
        for lang in ("arb_Arab", "eng_Latn", "tam_Taml", "mal_Mlym"):
            n = len(load_flores(cfg, lang, split))
            print(f"FLORES+ {split}/{lang}: {n}")
    pins = {}
    for key, m in cfg["models"].items():
        rev = m.get("revision") or model_info(m["hf_id"]).sha
        # weights in one format only (skip rust/onnx/duplicate .bin when safetensors exists)
        files = [s.rfilename for s in model_info(m["hf_id"], revision=rev).siblings]
        ignore = ["*.ot", "*.onnx", "*.msgpack", "*.h5", "*.tflite"]
        if any(f.endswith(".safetensors") for f in files):
            ignore.append("pytorch_model*.bin")
        snapshot_download(m["hf_id"], revision=rev, ignore_patterns=ignore)
        pins[key] = rev
        print(f"{key:16s} {m['hf_id']}  @ {rev}")
    print("\nPin these in configs/default.yaml (models.<key>.revision):")
    print(json.dumps(pins, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="a2t", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("download")
    sub.add_parser("step1")
    sub.add_parser("step2")
    s3 = sub.add_parser("step3")
    s3.add_argument("--pilot", action="store_true", help="use step3.pilot_n sentences per direction")
    s3.add_argument("--n", type=int, default=None)
    s4 = sub.add_parser("step4")
    s4.add_argument("--no-tune", action="store_true")
    s4.add_argument("--corpus", default=None)
    sub.add_parser("step5")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--split", default=None)
    sub.add_parser("all")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    _setup_logging(cfg, args.cmd)
    if disable_power_throttling():
        logging.getLogger("a2t").info("Windows power throttling (EcoQoS) disabled for this process")
    set_seed(cfg["seed"])
    if args.cmd == "check":
        return cmd_check(cfg, args)
    log_env(cfg, args.cmd)
    if args.cmd == "download":
        return cmd_download(cfg, args)

    from . import evaluate, pivot

    if args.cmd in ("step1", "all"):
        pivot.step1(cfg)
    if args.cmd in ("step2", "all"):
        pivot.step2(cfg)
    if args.cmd in ("evaluate", "all"):
        evaluate.evaluate(cfg, args.split if args.cmd == "evaluate" else None)
    if args.cmd == "step3":
        from .synth import step3

        step3(cfg, n=cfg["step3"]["pilot_n"] if args.pilot else args.n)
    if args.cmd == "step4":
        from .train_lora import step4

        step4(cfg, corpus=args.corpus, tune=not args.no_tune)
    if args.cmd == "step5":
        from .llm_stub import step5

        step5(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
