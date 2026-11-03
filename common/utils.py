"""Small shared helpers: config loading, seeding, JSON I/O, timing and GPU-memory bookkeeping."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

SEED = 6304


def load_cfg(path: str | os.PathLike) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(p: str | os.PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | os.PathLike, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)
    os.replace(tmp, path)  # atomic: a disconnect never leaves a half-written file


def load_json(path: str | os.PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def append_jsonl(row: dict, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=_json_default) + "\n")


def write_jsonl(rows: Iterable[dict], path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


@contextmanager
def timer(store: dict, key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + time.perf_counter() - t0


def reset_peak_mem() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_mem_gb() -> float:
    import torch

    if not torch.cuda.is_available():
        return float("nan")
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**3


def dir_size_mb(path: str | os.PathLike) -> float:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1024**2
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2


def rng_state() -> dict:
    import torch

    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    import torch

    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])
