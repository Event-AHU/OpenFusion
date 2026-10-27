"""NPY dataset for XCS spectrum -> ne + Ti + Te regression.

Supported layouts under ``data_path``:
1. Pre-split NPY files:
    train_xcs.npy, train_y.npy, train_shotnum.npy
    val_xcs.npy,   val_y.npy,   val_shotnum.npy
    test_xcs.npy,  test_y.npy,  test_shotnum.npy
2. Full cleaned Ti/Te export:
    X_spec_full.npy, X_macro.npy, Y_data.npy, shotnums.npy, metadata.json

``*_y.npy`` is already target-side preprocessed and must have shape [N, 66]:
    y[:, 0:1]    ne1_val
    y[:, 1:2]    ne2_val
    y[:, 2:34]   Ti_profile, 32 points
    y[:, 34:66]  Te_profile, 32 points

This dataset only reads the pre-smoothed/interpolated Ti/Te targets; it does
not run any Ti/Te smoothing or interpolation during training.
"""

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.datasets import SpectralTFTransform

NE_SLICE = slice(0, 2)
TI_SLICE = slice(2, 34)
TE_SLICE = slice(34, 66)
TARGET_DIM = 66
TARGET_PROFILE_POINTS = 32
TARGET_LAYOUT = {
    "ne1_val": (0, 1),
    "ne2_val": (1, 2),
    "Ti_profile": (2, 34),
    "Te_profile": (34, 66),
}
TARGET_COLUMNS = [
    "ne1_val",
    "ne2_val",
    *[f"Ti_{index:02d}" for index in range(32)],
    *[f"Te_{index:02d}" for index in range(32)],
]
FULL_XCS_FILE = "X_spec_full.npy"
FULL_MACRO_FILE = "X_macro.npy"
FULL_TITE_FILE = "Y_data.npy"
FULL_SHOT_FILE = "shotnums.npy"
DEFAULT_SPLIT_RATIOS = (0.7, 0.2, 0.1)


def _as_pair(value, default: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return int(value), int(value)
    if isinstance(value, str):
        parts = value.replace(",", " ").replace("x", " ").split()
        if len(parts) == 1:
            return int(parts[0]), int(parts[0])
        return int(parts[0]), int(parts[1])
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return int(value[0]), int(value[0])
        return int(value[0]), int(value[1])
    return default


def _as_ratios(value, default: Tuple[float, float, float] = DEFAULT_SPLIT_RATIOS) -> Tuple[float, float, float]:
    if value is None:
        return default
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
        ratios = tuple(float(part) for part in parts)
    elif isinstance(value, Sequence):
        ratios = tuple(float(part) for part in value)
    else:
        return default
    if len(ratios) != 3:
        raise ValueError(f"split_ratios must contain train/val/test values, got {ratios}")
    total = sum(ratios)
    if total <= 0:
        raise ValueError(f"split_ratios must sum to a positive value, got {ratios}")
    return tuple(ratio / total for ratio in ratios)


def split_721_indices(total_size: int, seed: int = 0, ratios: Tuple[float, float, float] = DEFAULT_SPLIT_RATIOS):
    """Return deterministic train/val/test indices for a full unsplit NPY export."""
    total_size = int(total_size)
    if total_size <= 0:
        raise ValueError(f"Cannot split an empty dataset: total_size={total_size}")

    train_ratio, val_ratio, _ = ratios
    train_size = int(round(train_ratio * total_size))
    val_size = int(round(val_ratio * total_size))
    test_size = total_size - train_size - val_size
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError(
            f"Dataset too small for split ratios {ratios}: total={total_size}, "
            f"train={train_size}, val={val_size}, test={test_size}"
        )

    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(total_size).astype(np.int64)
    return {
        "train": indices[:train_size],
        "val": indices[train_size:train_size + val_size],
        "test": indices[train_size + val_size:],
    }


class XCSNeDataset(Dataset):
    """Read one split of the preprocessed XCS -> ne/Ti/Te NPY dataset."""

    SPLITS = ("train", "val", "test")

    def __init__(self, data_path: str, args, split: str = "train"):
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got {split!r}")

        self.data_path = Path(data_path)
        self.args = args
        self.split = split
        self.transform = SpectralTFTransform(args)
        self.metadata = self.read_metadata(self.data_path)

        self.xcs_input_size = _as_pair(getattr(args, "xcs_input_size", (35, 488)), (35, 488))
        self.xcs_original_size = _as_pair(getattr(args, "xcs_original_size", (35, 487)), (35, 487))
        self.npy_mmap = bool(getattr(args, "npy_mmap", True))
        self.dataset_format = "split"
        self.source_n_samples = 0
        self.sample_indices = None

        xcs_path = self.data_path / f"{split}_xcs.npy"
        y_path = self.data_path / f"{split}_y.npy"
        shot_path = self.data_path / f"{split}_shotnum.npy"

        if xcs_path.is_file() or y_path.is_file():
            self._load_presplit_files(xcs_path, y_path, shot_path)
        else:
            xcs_path, y_path, shot_path = self._load_full_export(split)

        self._validate_shapes(xcs_path, y_path, shot_path if shot_path.is_file() else None)

        self.y_mean = torch.zeros(TARGET_DIM, dtype=torch.float32)
        self.y_std = torch.ones(TARGET_DIM, dtype=torch.float32)
        self.normalize_y = False

        print(
            f"XCSNeDataset[{split}] loaded from {self.data_path}: "
            f"format={self.dataset_format}, X_spec={tuple(self.X_spec.shape)}, "
            f"Y={tuple(self.Y_data.shape)}, shots={tuple(self.shotnums.shape)}, "
            f"mmap={self.npy_mmap}, indices={self._indices_summary()}, "
            f"target_layout={self._target_layout_summary()}"
        )

    def _load_presplit_files(self, xcs_path: Path, y_path: Path, shot_path: Path):
        if not xcs_path.is_file():
            raise FileNotFoundError(f"Missing XCS NPY file for split={self.split}: {xcs_path}")
        if not y_path.is_file():
            raise FileNotFoundError(f"Missing target NPY file for split={self.split}: {y_path}")

        mmap_mode = "r" if self.npy_mmap else None
        self.X_spec = np.load(xcs_path, mmap_mode=mmap_mode)
        self.Y_data = torch.from_numpy(np.asarray(np.load(y_path))).float()
        self.source_n_samples = int(self.X_spec.shape[0])

        if shot_path.is_file():
            self.shotnums = torch.from_numpy(np.asarray(np.load(shot_path))).long()
        else:
            self.shotnums = torch.arange(self.Y_data.shape[0], dtype=torch.long)

    def _load_full_export(self, split: str):
        xcs_path = self.data_path / FULL_XCS_FILE
        macro_path = self.data_path / FULL_MACRO_FILE
        y_path = self.data_path / FULL_TITE_FILE
        shot_path = self.data_path / FULL_SHOT_FILE

        missing = [path for path in (xcs_path, macro_path, y_path) if not path.is_file()]
        if missing:
            expected_split = self.data_path / f"{split}_xcs.npy"
            raise FileNotFoundError(
                "Could not find either the pre-split XCS-ne/Ti/Te files or the full cleaned export. "
                f"Missing split file starts at {expected_split}; missing full-export files: {missing}"
            )

        self.dataset_format = "full_export"
        mmap_mode = "r" if self.npy_mmap else None
        self.X_spec = np.load(xcs_path, mmap_mode=mmap_mode)
        self.source_n_samples = int(self.X_spec.shape[0])
        split_seed = getattr(self.args, "split_seed", None)
        if split_seed is None:
            split_seed = getattr(self.args, "seed", 0)
        split_seed = int(split_seed)
        ratios = _as_ratios(getattr(self.args, "split_ratios", None))
        self.sample_indices = split_721_indices(self.source_n_samples, seed=split_seed, ratios=ratios)[split]

        full_y = self._build_full_export_targets(macro_path, y_path)
        self.Y_data = torch.from_numpy(full_y[self.sample_indices]).float()

        if shot_path.is_file():
            shotnums = np.asarray(np.load(shot_path))
            self.shotnums = torch.from_numpy(shotnums[self.sample_indices]).long()
        else:
            self.shotnums = torch.from_numpy(self.sample_indices.copy()).long()

        return xcs_path, y_path, shot_path

    def _build_full_export_targets(self, macro_path: Path, y_path: Path) -> np.ndarray:
        macro = np.asarray(np.load(macro_path), dtype=np.float32)
        tite = np.asarray(np.load(y_path), dtype=np.float32)
        if macro.ndim != 2:
            raise ValueError(f"{macro_path} must have shape [N, F], got {macro.shape}")
        if tite.ndim != 2 or tite.shape[1] != TARGET_DIM - 2:
            raise ValueError(
                f"{y_path} must have shape [N, {TARGET_DIM - 2}] containing Ti(32)+Te(32), "
                f"got {tite.shape}"
            )
        if macro.shape[0] != self.source_n_samples or tite.shape[0] != self.source_n_samples:
            raise ValueError(
                f"Full-export sample count mismatch: {FULL_XCS_FILE} has {self.source_n_samples}, "
                f"{macro_path.name} has {macro.shape[0]}, {y_path.name} has {tite.shape[0]}"
            )

        feature_names = (self.metadata or {}).get("feature_names") or []
        try:
            ne_indices = [feature_names.index("ne1_val"), feature_names.index("ne2_val")]
        except ValueError as exc:
            raise ValueError(
                f"{macro_path} metadata.json must declare feature_names with ne1_val and ne2_val "
                "to build the [ne(2), Ti(32), Te(32)] target."
            ) from exc
        if max(ne_indices) >= macro.shape[1]:
            raise ValueError(
                f"ne feature indices {ne_indices} exceed {macro_path} feature dimension {macro.shape[1]}"
            )

        ne = macro[:, ne_indices].astype(np.float32, copy=False)
        full_y = np.concatenate([ne, tite], axis=1).astype(np.float32, copy=False)
        if not np.isfinite(full_y).all():
            raise ValueError(f"Combined targets from {macro_path.name}+{y_path.name} contain NaN or inf values.")
        return full_y

    def _validate_shapes(self, xcs_path: Path, y_path: Path, shot_path: Optional[Path]):
        if self.X_spec.ndim not in (3, 4):
            raise ValueError(
                f"{xcs_path} must have shape [N, 35, W] or [N, 1, 35, W], "
                f"got {self.X_spec.shape}"
            )

        n_samples = int(self.Y_data.shape[0])
        xcs_samples = int(self.X_spec.shape[0])
        if self.Y_data.ndim != 2 or self.Y_data.shape[1] != TARGET_DIM:
            raise ValueError(
                f"{y_path} must have shape [N, {TARGET_DIM}] ordered as "
                f"[ne1, ne2, Ti(32), Te(32)], got {tuple(self.Y_data.shape)}. "
                "Provide target arrays with the expected dimensions and ordering."
            )
        if self.dataset_format == "split" and int(self.Y_data.shape[0]) != xcs_samples:
            raise ValueError(
                f"XCS/Y sample count mismatch: {xcs_path} has {xcs_samples}, "
                f"{y_path} has {self.Y_data.shape[0]}"
            )
        if self.dataset_format == "full_export":
            if self.sample_indices is None:
                raise ValueError("Internal error: full_export dataset has no sample_indices.")
            if int(self.sample_indices.shape[0]) != n_samples:
                raise ValueError(
                    f"Internal split index count mismatch: indices={self.sample_indices.shape[0]}, "
                    f"Y={n_samples}"
                )
        if int(self.shotnums.shape[0]) != n_samples:
            raise ValueError(
                f"Y/shotnum sample count mismatch: {y_path} has {n_samples}, "
                f"{shot_path or 'generated_shotnum'} has {self.shotnums.shape[0]}"
            )

        if self.X_spec.ndim == 3:
            if int(self.X_spec.shape[1]) != 35:
                raise ValueError(f"{xcs_path} expected 35 detector rows, got {self.X_spec.shape}")
        else:
            channels = int(self.X_spec.shape[1])
            height = int(self.X_spec.shape[2])
            if channels != 1 or height != 35:
                raise ValueError(f"{xcs_path} expected [N, 1, 35, W], got {self.X_spec.shape}")

        if not torch.isfinite(self.Y_data).all():
            raise ValueError(f"{y_path} contains NaN or inf values.")

        self._validate_target_metadata(y_path)

    def _validate_target_metadata(self, y_path: Path):
        if not self.metadata:
            return

        target_flat_dim = self.metadata.get("target_flat_dim")
        if target_flat_dim is not None and int(target_flat_dim) != TARGET_DIM:
            raise ValueError(
                f"{y_path} uses target_dim={TARGET_DIM}, but metadata.json declares "
                f"target_flat_dim={target_flat_dim}."
            )

        target_slices = self.metadata.get("target_slices") or {}
        for name, expected in TARGET_LAYOUT.items():
            if name not in target_slices:
                continue
            actual = tuple(int(value) for value in target_slices[name])
            if actual != expected:
                raise ValueError(
                    f"{y_path} target layout mismatch for {name}: "
                    f"metadata has {actual}, expected {expected}."
                )

        target_shapes = self.metadata.get("target_shapes") or {}
        for name in ("Ti_profile", "Te_profile"):
            if name not in target_shapes:
                continue
            actual_shape = tuple(int(value) for value in target_shapes[name])
            if actual_shape != (TARGET_PROFILE_POINTS,):
                raise ValueError(
                    f"{y_path} metadata target shape for {name} is {actual_shape}; "
                    f"expected ({TARGET_PROFILE_POINTS},)."
                )

        profile_grid = self.metadata.get("profile_grid")
        if profile_grid is not None and len(profile_grid) != TARGET_PROFILE_POINTS:
            raise ValueError(
                f"{y_path} metadata profile_grid has {len(profile_grid)} points; "
                f"expected {TARGET_PROFILE_POINTS}."
            )

    def _target_layout_summary(self) -> str:
        smoothing = "pre-smoothed"
        if self.metadata:
            smoothing = self.metadata.get("smoothing_method", smoothing)
        return (
            "ne1=[0:1], ne2=[1:2], Ti=[2:34] 32pt, "
            f"Te=[34:66] 32pt, {smoothing}"
        )

    def _indices_summary(self) -> str:
        if self.sample_indices is None:
            return "contiguous split files"
        return (
            f"{self.split}={len(self.sample_indices)}/{self.source_n_samples}, "
            f"first={self.sample_indices[:5].tolist()}"
        )

    @staticmethod
    def load_train_target_stats(
        data_path: str,
        target_stats_path: str = "",
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        stats_path = Path(target_stats_path) if target_stats_path else Path(data_path) / "target_stats_train.npz"
        if not stats_path.is_file():
            return None

        stats = np.load(stats_path)
        if "y_mean" not in stats or "y_std" not in stats:
            raise KeyError(f"{stats_path} must contain y_mean and y_std arrays.")
        y_mean = torch.from_numpy(np.asarray(stats["y_mean"], dtype=np.float32)).float()
        y_std = torch.from_numpy(np.asarray(stats["y_std"], dtype=np.float32)).float()
        if y_mean.numel() != TARGET_DIM or y_std.numel() != TARGET_DIM:
            raise ValueError(
                f"{stats_path} must contain {TARGET_DIM}-element y_mean/y_std arrays, "
                f"got {y_mean.numel()} and {y_std.numel()}."
            )
        y_std[y_std == 0] = 1.0
        print(
            f"Loaded target stats from {stats_path}: "
            f"ne_mean={y_mean[:2].tolist()}, "
            f"Ti_mean_range=({y_mean[TI_SLICE].min().item():.3f}, {y_mean[TI_SLICE].max().item():.3f}), "
            f"Te_mean_range=({y_mean[TE_SLICE].min().item():.3f}, {y_mean[TE_SLICE].max().item():.3f})"
        )
        return y_mean, y_std

    @staticmethod
    def read_metadata(data_path: str) -> Optional[dict]:
        metadata_path = Path(data_path) / "metadata.json"
        if not metadata_path.is_file():
            return None
        with open(metadata_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def set_target_normalization(self, y_mean: torch.Tensor, y_std: torch.Tensor):
        if y_mean.numel() != TARGET_DIM or y_std.numel() != TARGET_DIM:
            raise ValueError(
                f"Target normalization expects {TARGET_DIM} values, "
                f"got y_mean={y_mean.numel()}, y_std={y_std.numel()}."
            )
        self.y_mean = y_mean.detach().float().clone().cpu()
        self.y_std = y_std.detach().float().clone().cpu()
        self.y_std[self.y_std == 0] = 1.0
        self.normalize_y = True
        print(
            f"Target normalization enabled for split={self.split}: "
            f"ne mean={self.y_mean[:2].tolist()}, target_dim={self.y_mean.numel()}"
        )

    def get_original_y(self, y_normalized: torch.Tensor) -> torch.Tensor:
        if not self.normalize_y:
            return y_normalized.cpu() if y_normalized.is_cuda else y_normalized
        y_norm_cpu = y_normalized.detach().cpu()
        return y_norm_cpu * self.y_std + self.y_mean

    @staticmethod
    def split_targets(y: torch.Tensor):
        return y[..., NE_SLICE], y[..., TI_SLICE], y[..., TE_SLICE]

    def __len__(self):
        return int(self.Y_data.shape[0])

    def _get_xcs_matrix(self, idx: int) -> np.ndarray:
        source_idx = int(self.sample_indices[idx]) if self.sample_indices is not None else int(idx)
        xcs = np.asarray(self.X_spec[source_idx], dtype=np.float32).copy()
        if xcs.ndim == 3:
            if xcs.shape[0] == 1:
                xcs = xcs[0]
            elif xcs.shape[-1] == 1:
                xcs = xcs[..., 0]
            else:
                raise ValueError(f"Expected one channel for XCS sample, got shape {xcs.shape}")
        if xcs.ndim != 2:
            raise ValueError(f"Expected XCS sample shape [35, W], got {xcs.shape}")
        return xcs

    def __getitem__(self, idx):
        xcs_matrix = self._get_xcs_matrix(idx)
        xcs = self.transform({"xcs": xcs_matrix})["xcs"]

        target = self.Y_data[idx]
        if self.normalize_y:
            target = (target - self.y_mean) / self.y_std
        return xcs.float(), target.float(), self.shotnums[idx]
