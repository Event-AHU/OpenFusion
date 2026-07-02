from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DataStats:
    branch_mean: np.ndarray
    branch_std: np.ndarray
    target_mean: float
    target_std: float
    pos_min: np.ndarray
    pos_max: np.ndarray
    time_min: float
    time_max: float


def build_branch_feature(record: np.void, include_base_samples: bool) -> np.ndarray:
    sample_type = int(record["sample_type"])
    one_hot = np.zeros(3, dtype=np.float32)
    one_hot[sample_type] = 1.0
    kheat = np.asarray([record["kheat"]], dtype=np.float32)
    yi = np.asarray([0.0 if np.isnan(record["yi"]) else record["yi"]], dtype=np.float32)
    hflux = np.asarray(record["hflux"], dtype=np.float32)
    if sample_type == 0:
        if not include_base_samples:
            raise ValueError("Base sample included unexpectedly.")
        hflux = np.zeros_like(hflux)
    return np.concatenate([one_hot, kheat, yi, hflux], axis=0).astype(np.float32)


def build_splits(
    dataset_path: str,
    include_base_samples: bool,
    train_ratio: float,
    val_ratio: float,
    holdout_kheat_for_test: Sequence[float],
    seed: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
) -> Dict[str, np.ndarray]:
    data = np.load(dataset_path, mmap_mode="r")
    indices = np.arange(len(data))
    if not include_base_samples:
        indices = indices[data["sample_type"][indices] != 0]

    rng = np.random.default_rng(seed)
    holdout_kheat_for_test = np.asarray(holdout_kheat_for_test, dtype=np.float32)
    if holdout_kheat_for_test.size > 0:
        kheat_values = data["kheat"][indices].astype(np.float32)
        holdout_mask = np.isin(kheat_values, holdout_kheat_for_test)
        candidate_train_val = indices[~holdout_mask]
        test_idx = rng.permutation(indices[holdout_mask])
        train_val = rng.permutation(candidate_train_val)
        n_train = int(len(train_val) * train_ratio)
        train_idx = train_val[:n_train]
        val_idx = train_val[n_train:]
    else:
        perm = rng.permutation(indices)
        n_train = int(len(perm) * train_ratio)
        n_val = int(len(perm) * val_ratio)
        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]

    if max_train_samples is not None:
        train_idx = train_idx[:max_train_samples]
    if max_val_samples is not None:
        val_idx = val_idx[:max_val_samples]
    if max_test_samples is not None:
        test_idx = test_idx[:max_test_samples]

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def estimate_stats(
    dataset_path: str,
    pos_path: str,
    train_indices: Sequence[int],
    include_base_samples: bool,
    samples_per_field: int,
    seed: int,
) -> DataStats:
    data = np.load(dataset_path, mmap_mode="r")
    pos = np.load(pos_path).astype(np.float32)

    branch_features = np.stack(
        [build_branch_feature(data[idx], include_base_samples) for idx in train_indices],
        axis=0,
    )
    branch_mean = branch_features.mean(axis=0)
    branch_std = branch_features.std(axis=0)
    branch_std[branch_std < 1.0e-6] = 1.0

    rng = np.random.default_rng(seed + 1)
    sampled_values = []
    num_points = pos.shape[0]
    num_times = data["u1"].shape[-1]
    for idx in train_indices:
        point_idx = rng.integers(0, num_points, size=samples_per_field)
        time_idx = rng.integers(0, num_times, size=samples_per_field)
        sampled_values.append(np.asarray(data[idx]["u1"][point_idx, time_idx], dtype=np.float32))
    sampled_values_arr = np.concatenate(sampled_values, axis=0)
    target_mean = float(sampled_values_arr.mean())
    target_std = float(sampled_values_arr.std())
    if target_std < 1.0e-6:
        target_std = 1.0

    pos_min = pos.min(axis=0)
    pos_max = pos.max(axis=0)
    return DataStats(
        branch_mean=branch_mean.astype(np.float32),
        branch_std=branch_std.astype(np.float32),
        target_mean=target_mean,
        target_std=target_std,
        pos_min=pos_min.astype(np.float32),
        pos_max=pos_max.astype(np.float32),
        time_min=0.0,
        time_max=1.0,
    )


class HeatPNOTDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        pos_path: str,
        indices: Sequence[int],
        stats: DataStats,
        include_base_samples: bool,
        node_subsample: int | None,
        time_indices: Sequence[int] | None,
        seed: int,
    ) -> None:
        self.dataset = np.load(dataset_path, mmap_mode="r")
        self.pos = np.load(pos_path).astype(np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.stats = stats
        self.include_base_samples = include_base_samples
        self.node_subsample = node_subsample
        self.time_indices = list(time_indices) if time_indices is not None else None
        self.rng = np.random.default_rng(seed)
        self.num_points = self.pos.shape[0]
        self.num_times = self.dataset["u1"].shape[-1]

        pos_den = np.maximum(stats.pos_max - stats.pos_min, 1.0e-8)
        self.pos_norm = ((self.pos - stats.pos_min) / pos_den).astype(np.float32)
        self.time_grid = np.linspace(stats.time_min, stats.time_max, self.num_times, dtype=np.float32)

    def __len__(self) -> int:
        if self.time_indices is None:
            return len(self.indices)
        return len(self.indices) * len(self.time_indices)

    def _sample_index_time(self, item: int) -> tuple[int, int]:
        if self.time_indices is None:
            return int(self.indices[item]), int(self.rng.integers(0, self.num_times))
        sample_offset = item // len(self.time_indices)
        time_offset = item % len(self.time_indices)
        return int(self.indices[sample_offset]), int(self.time_indices[time_offset])

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        sample_index, time_index = self._sample_index_time(item)
        record = self.dataset[sample_index]
        if self.node_subsample is not None and self.node_subsample < self.num_points:
            node_ids = self.rng.choice(self.num_points, size=self.node_subsample, replace=False)
            node_ids.sort()
        else:
            node_ids = np.arange(self.num_points, dtype=np.int64)

        branch = build_branch_feature(record, self.include_base_samples)
        branch = ((branch - self.stats.branch_mean) / self.stats.branch_std).astype(np.float32)
        branch_rep = np.broadcast_to(branch[None, :], (len(node_ids), branch.shape[0])).copy()
        coords = self.pos_norm[node_ids]
        time_col = np.full((len(node_ids), 1), self.time_grid[time_index], dtype=np.float32)
        model_input = np.concatenate([coords, time_col, branch_rep], axis=1).astype(np.float32)
        target = np.asarray(record["u1"][node_ids, time_index], dtype=np.float32)
        target_norm = (target - self.stats.target_mean) / self.stats.target_std

        return {
            "input": torch.from_numpy(model_input),
            "coords": torch.from_numpy(coords),
            "target": torch.from_numpy(target_norm.astype(np.float32)),
            "target_denorm": torch.from_numpy(target),
            "node_ids": torch.from_numpy(node_ids.astype(np.int64)),
            "sample_index": torch.tensor(sample_index, dtype=torch.long),
            "time_index": torch.tensor(time_index, dtype=torch.long),
        }


def collate_heat_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in batch[0]}


def make_dataloaders(config: Dict, seed: int) -> tuple[Dict[str, DataLoader], Dict[str, np.ndarray], DataStats]:
    splits = build_splits(
        dataset_path=config["paths"]["dataset"],
        include_base_samples=config["data"]["include_base_samples"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        holdout_kheat_for_test=config["data"].get("holdout_kheat_for_test", []),
        seed=seed,
        max_train_samples=config["data"]["max_train_samples"],
        max_val_samples=config["data"]["max_val_samples"],
        max_test_samples=config["data"]["max_test_samples"],
    )
    stats = estimate_stats(
        dataset_path=config["paths"]["dataset"],
        pos_path=config["paths"]["positions"],
        train_indices=splits["train"],
        include_base_samples=config["data"]["include_base_samples"],
        samples_per_field=config["data"]["norm_value_samples_per_field"],
        seed=seed,
    )
    return make_dataloaders_from_state(config, seed, splits, stats)


def make_dataloaders_from_state(
    config: Dict,
    seed: int,
    splits: Dict[str, np.ndarray],
    stats: DataStats,
) -> tuple[Dict[str, DataLoader], Dict[str, np.ndarray], DataStats]:

    loaders: Dict[str, DataLoader] = {}
    for offset, split_name in enumerate(["train", "val", "test"]):
        is_train = split_name == "train"
        dataset = HeatPNOTDataset(
            dataset_path=config["paths"]["dataset"],
            pos_path=config["paths"]["positions"],
            indices=splits[split_name],
            stats=stats,
            include_base_samples=config["data"]["include_base_samples"],
            node_subsample=config["data"]["train_node_subsample"] if is_train else config["data"]["eval_node_subsample"],
            time_indices=None if is_train else config["evaluation"]["eval_time_indices"],
            seed=seed + offset,
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=is_train,
            num_workers=config["training"]["num_workers"],
            collate_fn=collate_heat_batch,
        )
    return loaders, splits, stats


def stats_from_checkpoint(payload: Dict) -> DataStats:
    raw = payload["stats"]
    return DataStats(
        branch_mean=np.asarray(raw["branch_mean"], dtype=np.float32),
        branch_std=np.asarray(raw["branch_std"], dtype=np.float32),
        target_mean=float(raw["target_mean"]),
        target_std=float(raw["target_std"]),
        pos_min=np.asarray(raw["pos_min"], dtype=np.float32),
        pos_max=np.asarray(raw["pos_max"], dtype=np.float32),
        time_min=float(raw["time_min"]),
        time_max=float(raw["time_max"]),
    )
