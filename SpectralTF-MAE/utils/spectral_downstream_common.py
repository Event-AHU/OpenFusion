"""Shared helpers for bidirectional CXRS/XCS spectral fine-tuning and evaluation.

Both directions preserve the pretrained tensor geometry:
    XCS:  [B, 1, 35, 490]
    CXRS: [B, 1, 12, 1024]

Each selected input uses the full unmasked spectrum. Loss and metrics are
calculated on the output modality's fixed peak interval.
"""

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from utils.spectral_pretraining_common import (
    FIXED_R2_RANGES,
    crop_to_original,
    finalize_r2,
    init_r2_accumulator,
    make_fixed_r2_region_mask,
    safe_mean,
    update_r2_accumulator,
)


def load_model_weights_only(model, ckpt_path, device="cpu", prefix="checkpoint"):
    """Strictly load a fine-tuned spectral checkpoint for testing."""
    if not ckpt_path:
        raise ValueError(f"{prefix} path required")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"{prefix} not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "Test checkpoint does not match the complete fine-tuning architecture; "
            "use matching test YAML geometry and model settings."
        ) from exc
    print(f"Loaded full {prefix} state from {ckpt_path} (epoch {checkpoint.get('epoch', 'unknown')})")
    return checkpoint


def build_full_spectrum_inputs(tasks_dict, args, domain_specs):
    """Build complete-spectrum downstream inputs; no masked patches are used."""
    del domain_specs
    input_dict = {
        task: tasks_dict[task]
        for task in args.in_domains
        if task in tasks_dict
    }
    return input_dict, None


def init_error_accumulator():
    return {"sum_abs": 0.0, "sum_sq": 0.0, "count": 0}


def update_error_accumulator(accum, pred, target, valid_mask=None):
    pred = pred.detach().float()
    target = target.detach().float()

    if valid_mask is not None:
        valid_mask = valid_mask.detach().bool()
        if valid_mask.dim() == 3:
            valid_mask = valid_mask[:, None, :, :]
        valid_mask = valid_mask.expand_as(target)
        pred = pred[valid_mask]
        target = target[valid_mask]
    else:
        pred = pred.reshape(-1)
        target = target.reshape(-1)

    finite_mask = torch.isfinite(pred) & torch.isfinite(target)
    pred = pred[finite_mask]
    target = target[finite_mask]

    if target.numel() == 0:
        return

    diff = (pred - target).double()
    accum["sum_abs"] += diff.abs().sum().item()
    accum["sum_sq"] += (diff ** 2).sum().item()
    accum["count"] += int(target.numel())


def finalize_error_metrics(accum):
    count = accum["count"]
    if count == 0:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan")}

    mse = accum["sum_sq"] / count
    mae = accum["sum_abs"] / count
    rmse = float(np.sqrt(mse))
    return {"mse": float(mse), "rmse": rmse, "mae": float(mae)}


def init_downstream_metric_accumulators(out_domains: Iterable[str]):
    return {
        task: {
            "r2": init_r2_accumulator(),
            "err": init_error_accumulator(),
        }
        for task in out_domains
    }


def update_downstream_peak_metrics(accumulators, preds, tasks_dict, args, domain_specs):
    for task in args.out_domains:
        if task not in preds or task not in tasks_dict or task not in FIXED_R2_RANGES:
            continue

        original_size = domain_specs[task]["original_size"]
        pred = crop_to_original(preds[task].float(), original_size)
        target_full = tasks_dict[task].float()
        target = crop_to_original(target_full, original_size)

        peak_mask = make_fixed_r2_region_mask(
            target=target,
            task=task,
            valid_mask=None,
        )

        update_r2_accumulator(
            accumulators[task]["r2"],
            pred=pred,
            target=target,
            valid_mask=peak_mask,
        )
        update_error_accumulator(
            accumulators[task]["err"],
            pred=pred,
            target=target,
            valid_mask=peak_mask,
        )


def finalize_downstream_peak_metrics(accumulators, out_domains: Iterable[str]):
    stats = {}
    peak_r2_values = []

    for task in out_domains:
        if task not in accumulators:
            continue

        r2 = finalize_r2(accumulators[task]["r2"])
        err = finalize_error_metrics(accumulators[task]["err"])

        stats[f"{task}_peak_r2"] = r2
        stats[f"{task}_peak_mse"] = err["mse"]
        stats[f"{task}_peak_rmse"] = err["rmse"]
        stats[f"{task}_peak_mae"] = err["mae"]
        peak_r2_values.append(r2)

    stats["mean_peak_r2"] = safe_mean(peak_r2_values)
    return stats


def save_metrics_json(path, stats):
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
