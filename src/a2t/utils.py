"""Shared helpers: config, seeds, device, environment logging, jsonl I/O, hashing."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


# ---------------------------------------------------------------- config
def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    out = Path(cfg.get("outputs", "outputs"))
    cfg["outputs"] = str(out if out.is_absolute() else ROOT / out)
    return cfg


def out_dir(cfg: dict, *parts: str) -> Path:
    p = Path(cfg["outputs"]).joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- determinism
def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def disable_power_throttling() -> bool:
    """Windows only: opt this process out of execution-speed power throttling (EcoQoS), which otherwise can
    park a windowless background job on efficiency cores and starve the GPU (~1.5-3x slower). Affects this
    process only and ends with it; no system setting changes. Returns True if applied."""
    if sys.platform != "win32":
        return False
    import ctypes

    class State(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

    st = State(1, 1, 0)  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED controlled, and off
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    return bool(k32.SetProcessInformation(ctypes.c_void_p(k32.GetCurrentProcess()), 4,  # ProcessPowerThrottling
                                          ctypes.byref(st), ctypes.sizeof(st)))


def resolve_device(pref: str = "auto") -> str:
    import torch

    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(pref: str, device: str):
    import torch

    if pref == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[pref]


def dtype_kwarg(dtype) -> dict:
    """`from_pretrained` dtype argument: `torch_dtype` before transformers 4.56, `dtype` after.
    (An unknown kwarg would be silently ignored and the model would load in fp32.)"""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


# ---------------------------------------------------------------- environment
_PKGS = [
    "torch", "transformers", "tokenizers", "sentencepiece", "sacrebleu", "peft",
    "accelerate", "huggingface_hub", "IndicTransToolkit", "numpy", "pyarrow", "PyYAML",
]


def env_info() -> dict:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {},
    }
    for p in _PKGS:
        try:
            info["packages"][p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            info["packages"][p] = None
    try:
        import torch

        info["cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    except ImportError:
        pass
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip() or None
    except Exception:
        info["git_commit"] = None
    return info


def log_env(cfg: dict, step: str, extra: dict | None = None) -> Path:
    """Append one record per run to outputs/env.jsonl."""
    rec = {"step": step, **env_info(), "config": strip_private(cfg)}
    if extra:
        rec.update(extra)
    path = out_dir(cfg) / "env.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def strip_private(cfg: dict) -> dict:
    return {k: v for k, v in copy.deepcopy(cfg).items() if not k.startswith("_")}


# ---------------------------------------------------------------- io / hashing
def sha(obj: Any, n: int = 10) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # a run killed mid-write can leave one torn final line; skip it
                    continue
    return rows


def iter_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path: str | os.PathLike, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)
