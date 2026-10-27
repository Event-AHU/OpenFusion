"""SpectralTF domain specifications, reconstruction metrics, plots and checkpoints."""

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

import SpectralTF  # noqa: F401 - registers SpectralTF model constructors
from SpectralTF.input_adapters import PatchedInputAdapter
from SpectralTF.output_adapters import SpatialOutputAdapter
from utils.model_builder import create_model

R2_EPS = 1e-12

FIXED_R2_RANGES = {
    "cxrs": (550, 650),
    "xcs": (235, 335),
}

def get_fixed_peak_bounds(task: str, width: int) -> Tuple[int, int]:
    """Return fixed peak bounds in full-spectrum tensor coordinates.

    Full-spectrum tensors use the physical global ranges:
        cxrs: 550:650
        xcs:  235:335
    """
    if task not in FIXED_R2_RANGES:
        raise ValueError(f"No fixed peak range defined for task: {task}")

    width = int(width)
    if width <= 0:
        return 0, 0

    full_start, full_end = FIXED_R2_RANGES[task]
    full_start = int(full_start)
    full_end = int(full_end)
    if full_start >= width:
        return 0, 0

    return max(0, full_start), min(width, full_end)



DEFAULT_SPECTRAL_SPECS = {
    "cxrs": {
        "input_size": (12, 1024),
        "original_size": (12, 1024),
        "patch_size": (3, 16),
    },
    "xcs": {
        "input_size": (35, 490),
        "original_size": (35, 487),
        "patch_size": (5, 14),
    },
}


def as_pair(value) -> Tuple[int, int]:
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
    raise ValueError(f"Cannot parse pair from {value!r}")


def get_domain_specs(
    args, domains: Iterable[str]
) -> Dict[str, Dict[str, Tuple[int, int]]]:
    specs = {}
    for domain in domains:
        default = DEFAULT_SPECTRAL_SPECS.get(
            domain,
            {
                "input_size": as_pair(args.input_size),
                "original_size": as_pair(args.input_size),
                "patch_size": as_pair(args.patch_size),
            },
        )
        input_size = as_pair(
            getattr(args, f"{domain}_input_size", default["input_size"])
        )
        # Keep CLI --pad_xcs_width_to consistent with the padded array
        # produced by SpectralTFTransform, including on reconstruction tasks.
        pad_width = getattr(args, "pad_xcs_width_to", None) if domain == "xcs" else None
        if pad_width is not None and int(pad_width) > 0:
            input_size = (input_size[0], int(pad_width))
            args.xcs_input_size = list(input_size)
        original_size = as_pair(
            getattr(args, f"{domain}_original_size", default["original_size"])
        )
        patch_size = as_pair(
            getattr(args, f"{domain}_patch_size", default["patch_size"])
        )

        if input_size[0] % patch_size[0] != 0 or input_size[1] % patch_size[1] != 0:
            raise ValueError(
                f"{domain}: input_size={input_size} must be divisible by patch_size={patch_size}. "
                "Pad/crop in the yaml or change the patch size."
            )
        if original_size[0] > input_size[0] or original_size[1] > input_size[1]:
            raise ValueError(
                f"{domain}: original_size={original_size} cannot exceed input_size={input_size}."
            )

        specs[domain] = {
            "input_size": input_size,
            "original_size": original_size,
            "patch_size": patch_size,
        }
    return specs


def resolve_device(device_name: str) -> torch.device:
    """Select an available accelerator without assuming a hard-coded GPU index."""
    requested = torch.device(device_name)
    if requested.type != "cuda":
        return requested
    if not torch.cuda.is_available():
        print(f"CUDA device {device_name} is unavailable; using CPU.")
        return torch.device("cpu")
    if requested.index is not None and requested.index >= torch.cuda.device_count():
        print(f"CUDA index {requested.index} is unavailable; using cuda:0.")
        return torch.device("cuda:0")
    return requested


def cosine_lr(base_lr, min_lr, epoch_float, epochs, warmup_epochs=0, warmup_lr=0.0):
    if warmup_epochs > 0 and epoch_float < warmup_epochs:
        alpha = epoch_float / max(1e-12, warmup_epochs)
        return warmup_lr + alpha * (base_lr - warmup_lr)
    progress = (epoch_float - warmup_epochs) / max(1, epochs - warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)


def move_batch_to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        task: tensor.to(device, non_blocking=True) for task, tensor in batch.items()
    }


def make_input_dict(
    tasks_dict: Dict[str, torch.Tensor], in_domains: List[str]
) -> Dict[str, torch.Tensor]:
    return {task: tasks_dict[task] for task in in_domains if task in tasks_dict}


def mask_to_pixel_mask(
    mask: torch.Tensor, spatial_size: Tuple[int, int], patch_size: Tuple[int, int]
) -> torch.Tensor:
    H, W = spatial_size
    p_h, p_w = patch_size
    n_h, n_w = H // p_h, W // p_w
    mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=n_h, nw=n_w)
    mask = F.interpolate(
        mask.unsqueeze(1).float(), size=(H, W), mode="nearest"
    ).squeeze(1)
    return mask.bool()


def crop_to_original(
    tensor: torch.Tensor, original_size: Tuple[int, int]
) -> torch.Tensor:
    h, w = original_size
    return tensor[..., :h, :w]


def init_r2_accumulator():
    return {
        "sum_y": 0.0,
        "sum_y2": 0.0,
        "sum_res2": 0.0,
        "count": 0,
    }


def update_r2_accumulator(accum, pred, target, valid_mask=None):
    """
    pred / target: [B, C, H, W]
    valid_mask:   [B, H, W] or [B, C, H, W]
    """
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

    pred = pred.double()
    target = target.double()

    accum["sum_y"] += target.sum().item()
    accum["sum_y2"] += (target**2).sum().item()
    accum["sum_res2"] += ((target - pred) ** 2).sum().item()
    accum["count"] += int(target.numel())


def finalize_r2(accum):
    count = accum["count"]

    if count < 2:
        return float("nan")

    ss_tot = accum["sum_y2"] - (accum["sum_y"] ** 2) / count
    ss_res = accum["sum_res2"]

    if abs(ss_tot) < R2_EPS:
        return float("nan")

    return 1.0 - ss_res / ss_tot


def safe_mean(values):
    values = [v for v in values if np.isfinite(v)]
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def get_task_peak_loss_weight(args, task):
    """
    Return the extra fixed-peak loss weight used by full-spectrum pretraining.
    """
    task_specific_weight = getattr(args, f"{task}_peak_loss_weight", None)

    if task_specific_weight is None:
        weight = getattr(args, "peak_loss_weight", 0.5)
    else:
        weight = task_specific_weight

    weight = float(weight)
    weight = min(max(weight, 0.0), 1.0)

    return weight


def make_fixed_r2_region_mask(target, task, valid_mask=None):
    """
    target: [B, C, H, W]
    task:   'cxrs' or 'xcs'

    Return:
        region_mask: [B, H, W]

    Meaning:
        cxrs only selects wavelength 550:650
        xcs  only selects wavelength 235:335

    If valid_mask is provided, the final region is fixed-peak-region AND valid_mask.
    """
    if task not in FIXED_R2_RANGES:
        raise ValueError(f"No fixed R2 range defined for task: {task}")

    B, C, H, W = target.shape

    start, end = get_fixed_peak_bounds(task, W)

    region_mask = torch.zeros((B, H, W), dtype=torch.bool, device=target.device)
    if end > start:
        region_mask[:, :, start:end] = True

    if valid_mask is not None:
        valid_mask = valid_mask.detach().bool()

        if valid_mask.dim() == 4:
            valid_mask = valid_mask[:, 0]

        region_mask = region_mask & valid_mask

    return region_mask


def compute_fixed_peak_mse_loss(
    pred_full: torch.Tensor,
    target_full: torch.Tensor,
    mask: torch.Tensor,
    task: str,
    args,
    domain_specs,
) -> torch.Tensor:
    """
    Fixed peak-region MSE loss.

    pred_full / target_full:
        [B, 1, H_input, W_input]

    mask:
        [B, N_tokens]
        0 = visible
        1 = masked

    If args.loss_on_unmasked is False, the loss is computed on fixed-peak-region
    AND masked reconstruction region.  Otherwise, it is computed on the full
    fixed peak region.
    """
    original_size = domain_specs[task]["original_size"]
    patch_size = domain_specs[task]["patch_size"]

    pred = crop_to_original(pred_full.float(), original_size)
    target = crop_to_original(target_full.float(), original_size)

    valid_mask = None

    if not args.loss_on_unmasked and mask is not None:
        pixel_mask = mask_to_pixel_mask(
            mask,
            pred_full.shape[-2:],
            patch_size,
        )
        valid_mask = crop_to_original(pixel_mask, original_size)

    peak_mask = make_fixed_r2_region_mask(
        target=target,
        task=task,
        valid_mask=valid_mask,
    )

    if peak_mask.dim() == 3:
        peak_mask = peak_mask[:, None, :, :]

    peak_mask = peak_mask.expand_as(target)

    if peak_mask.sum().item() == 0:
        return pred.new_zeros(())

    loss = (pred - target) ** 2
    loss = loss[peak_mask].mean()

    return loss


def _sampling_grid(num_sampling_points: int) -> Tuple[int, int]:
    """Return near-square subplot layouts; 12 -> 3x4, 35 -> 5x7."""
    if num_sampling_points == 12:
        return 3, 4
    if num_sampling_points == 35:
        return 5, 7

    rows = int(math.floor(math.sqrt(num_sampling_points)))
    rows = max(rows, 1)
    while rows > 1 and num_sampling_points % rows != 0:
        rows -= 1
    cols = int(math.ceil(num_sampling_points / rows))
    return rows, cols


def _ensure_axes_2d(axes, rows: int, cols: int):
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        axes = axes.reshape(rows, cols)
    return axes


def _set_square_box(ax):
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)


def _save_full_reconstruction_heatmap(target, pred, task, epoch, sample_idx, save_path):
    import matplotlib.pyplot as plt

    vmin = min(float(np.nanmin(target)), float(np.nanmin(pred)))
    vmax = max(float(np.nanmax(target)), float(np.nanmax(pred)))

    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6

    num_sampling_points = target.shape[0]
    fig_height = max(3.0, min(6.0, num_sampling_points * 0.18))

    fig, axes = plt.subplots(1, 2, figsize=(12, fig_height), constrained_layout=True)

    im0 = axes[0].imshow(target, aspect="auto", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{task} Original")
    axes[0].set_xlabel("Wavelength index")
    axes[0].set_ylabel("Sampling point")

    im1 = axes[1].imshow(pred, aspect="auto", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{task} Full Reconstruction")
    axes[1].set_xlabel("Wavelength index")
    axes[1].set_ylabel("Sampling point")

    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85)

    fig.suptitle(
        f"Epoch {epoch} | {task} | sample {sample_idx} | Full reconstruction heatmap",
        y=1.02,
    )

    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_full_spectral_lines(target, pred, task, epoch, sample_idx, save_path):
    import matplotlib.pyplot as plt

    num_sampling_points, num_wavelengths = target.shape
    wavelength_indices = np.arange(num_wavelengths)

    fig_height = max(4.0, min(80.0, 2.2 * num_sampling_points))

    fig, axes = plt.subplots(
        num_sampling_points,
        1,
        figsize=(14, fig_height),
        sharex=True,
    )

    if num_sampling_points == 1:
        axes = [axes]

    for point_idx in range(num_sampling_points):
        ax = axes[point_idx]

        target_spectrum = target[point_idx, :]
        pred_spectrum = pred[point_idx, :]

        ax.plot(
            wavelength_indices,
            target_spectrum,
            linewidth=1.3,
            label="Raw target",
            alpha=0.85,
        )

        ax.plot(
            wavelength_indices,
            pred_spectrum,
            linestyle="--",
            linewidth=1.3,
            label="Prediction",
            alpha=0.85,
        )

        ax.set_title(f"Sampling point {point_idx}")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)

        if point_idx == 0:
            ax.legend(loc="best")

    axes[-1].set_xlabel("Wavelength Index")

    fig.suptitle(
        f"Epoch {epoch} | {task} | sample {sample_idx} | Full spectral line comparison",
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _save_masked_spectral_lines(
    target, pred, mask_np, task, epoch, sample_idx, save_path
):
    import matplotlib.pyplot as plt

    num_sampling_points, num_wavelengths = target.shape
    wavelength_indices = np.arange(num_wavelengths)

    fig_height = max(4.0, min(80.0, 2.2 * num_sampling_points))

    masked_pred = pred.copy()
    masked_pred[~mask_np] = np.nan

    fig, axes = plt.subplots(
        num_sampling_points,
        1,
        figsize=(14, fig_height),
        sharex=True,
    )

    if num_sampling_points == 1:
        axes = [axes]

    for point_idx in range(num_sampling_points):
        ax = axes[point_idx]

        target_spectrum = target[point_idx, :]
        masked_pred_spectrum = masked_pred[point_idx, :]
        mask_line = mask_np[point_idx, :]

        ax.plot(
            wavelength_indices,
            target_spectrum,
            linewidth=1.3,
            label="Raw target",
            alpha=0.85,
        )

        ax.plot(
            wavelength_indices,
            masked_pred_spectrum,
            linestyle="--",
            linewidth=1.5,
            label="Prediction on masked region",
            alpha=0.95,
        )

        masked_ratio_line = float(mask_line.mean())

        ax.set_title(
            f"Sampling point {point_idx} | masked ratio={masked_ratio_line:.3f}"
        )

        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)

        if point_idx == 0:
            ax.legend(loc="best")

    axes[-1].set_xlabel("Wavelength Index")

    fig.suptitle(
        f"Epoch {epoch} | {task} | sample {sample_idx} | Raw target + masked-region prediction",
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _save_peak_reconstruction_heatmap(target, pred, task, epoch, sample_idx, save_path):
    import matplotlib.pyplot as plt

    if task not in FIXED_R2_RANGES:
        return

    start, end = get_fixed_peak_bounds(task, target.shape[1])
    if end <= start:
        return

    target_peak = target[:, start:end]
    pred_peak = pred[:, start:end]

    vmin = min(float(np.nanmin(target_peak)), float(np.nanmin(pred_peak)))
    vmax = max(float(np.nanmax(target_peak)), float(np.nanmax(pred_peak)))

    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6

    num_sampling_points = target_peak.shape[0]
    fig_height = max(3.0, min(6.0, num_sampling_points * 0.18))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, fig_height),
        constrained_layout=True,
    )

    im0 = axes[0].imshow(target_peak, aspect="auto", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{task} Peak Raw Target")
    axes[0].set_xlabel(f"Wavelength index {start}:{end}")
    axes[0].set_ylabel("Sampling point")

    im1 = axes[1].imshow(pred_peak, aspect="auto", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{task} Peak Prediction")
    axes[1].set_xlabel(f"Wavelength index {start}:{end}")
    axes[1].set_ylabel("Sampling point")

    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85)

    fig.suptitle(
        f"Epoch {epoch} | {task} | sample {sample_idx} | Fixed peak-region heatmap",
        y=1.02,
    )

    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_peak_spectral_lines(target, pred, task, epoch, sample_idx, save_path):
    """Plot raw normalized target and model prediction in each fixed peak window.

    Curves show the original normalized target and the direct model output.
    """
    import matplotlib.pyplot as plt

    if task not in FIXED_R2_RANGES:
        return
    start, end = get_fixed_peak_bounds(task, target.shape[1])
    if end <= start:
        return
    target_peak = target[:, start:end]
    pred_peak = pred[:, start:end]
    rows, cols = _sampling_grid(target_peak.shape[0])
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.25))
    axes = _ensure_axes_2d(axes, rows, cols)
    wavelength_indices = np.arange(start, end)
    for point_idx, ax in enumerate(axes.flat):
        _set_square_box(ax)
        if point_idx >= target_peak.shape[0]:
            ax.axis('off')
            continue
        ax.plot(wavelength_indices, target_peak[point_idx], linewidth=1.2, label='Raw target')
        ax.plot(wavelength_indices, pred_peak[point_idx], '--', linewidth=1.2, label='Prediction')
        ax.set_title(f'Point {point_idx}', fontsize=8.5)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis='both', labelsize=8)
        if point_idx % cols == 0:
            ax.set_ylabel('Intensity', fontsize=8)
        if point_idx // cols == rows - 1:
            ax.set_xlabel('Wavelength index', fontsize=8)
        if point_idx == 0:
            ax.legend(loc='best', fontsize=8)
    fig.suptitle(f'Epoch {epoch} | {task} | sample {sample_idx} | Peak spectra ({start}:{end})')
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.055, top=0.89, wspace=0.28, hspace=0.45)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def get_model(args, domain_specs):
    print(f"Creating spectral model {args.model}")

    decoder_context_tasks = getattr(args, "decoder_context_tasks", args.in_domains)
    if isinstance(decoder_context_tasks, str):
        decoder_context_tasks = [x for x in decoder_context_tasks.split("-") if x]
    else:
        decoder_context_tasks = list(decoder_context_tasks)

    print("Domain specs:")
    for domain, spec in domain_specs.items():
        h, w = spec["input_size"]
        p_h, p_w = spec["patch_size"]
        print(
            f"  {domain}: input_size={spec['input_size']}, original_size={spec['original_size']}, "
            f"patch_size={spec['patch_size']}, tokens={(h // p_h) * (w // p_w)}"
        )

    input_adapters = {
        domain: PatchedInputAdapter(
            num_channels=1,
            stride_level=1,
            patch_size_full=domain_specs[domain]["patch_size"],
            image_size=domain_specs[domain]["input_size"],
        )
        for domain in args.in_domains
    }

    output_adapters = {
        domain: SpatialOutputAdapter(
            num_channels=1,
            stride_level=1,
            patch_size_full=domain_specs[domain]["patch_size"],
            image_size=domain_specs[domain]["input_size"],
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            use_task_queries=args.decoder_use_task_queries,
            task=domain,
            context_tasks=decoder_context_tasks,
            use_xattn=args.decoder_use_xattn,
        )
        for domain in args.out_domains
    }

    model = create_model(
        args.model,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path,
    )
    return model


def save_checkpoint(
    path, model, optimizer, scaler, epoch, args, stats, best_metric_value
):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
        "stats": stats,
        "best_metric_name": args.best_metric,
        "best_metric_value": best_metric_value,
        "best_r2": best_metric_value,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state, strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    elif optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint
