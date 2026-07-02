from __future__ import annotations

import csv
import json
import logging
import os
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_override(raw: str) -> tuple[list[str], Any]:
    key, value = raw.split("=", 1)
    return key.split("."), yaml.safe_load(value)


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    cfg = deepcopy(config)
    for raw in overrides:
        keys, value = parse_override(raw)
        cursor = cfg
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = value
    return cfg


def resolve_path(raw_path: str, roots: Iterable[Path]) -> Path:
    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return path
    for root in roots:
        candidate = root / path
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve path '{raw_path}'")


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def make_run_dir(output_root: str, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(output_root) / f"{run_name}_{stamp}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"pnot_heat.{log_path.parent.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


class CSVLogger:
    def __init__(self, path: Path, fieldnames: List[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self._initialized = False

    def log(self, row: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._initialized and self.path.stat().st_size == 0:
                writer.writeheader()
                self._initialized = True
            writer.writerow(row)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"
