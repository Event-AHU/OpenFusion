# Copyright (c) EPFL VILAB.
# All rights reserved.
#
# Joint CXRS/XCS masked reconstruction with peak-aware token selection.

import argparse
import datetime
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from einops import rearrange

from SpectralTF.criterion import MaskedMSELoss
from utils.datasets import build_spectral_pretraining_dataset
from utils.shared_checkpoint import validate_shared_geometry
from utils.optim_factory import create_optimizer
from utils.spectral_pretraining_common import (
    _save_full_reconstruction_heatmap,
    _save_full_spectral_lines,
    _save_masked_spectral_lines,
    _save_peak_reconstruction_heatmap,
    _save_peak_spectral_lines,
    compute_fixed_peak_mse_loss,
    cosine_lr,
    crop_to_original,
    finalize_r2,
    get_domain_specs,
    get_model,
    get_task_peak_loss_weight,
    init_r2_accumulator,
    load_checkpoint,
    make_fixed_r2_region_mask,
    make_input_dict,
    mask_to_pixel_mask,
    move_batch_to_device,
    safe_mean,
    save_checkpoint,
    resolve_device,
    set_optimizer_lr,
    update_r2_accumulator,
)


def get_args():
    config_parser = argparse.ArgumentParser(
        description="Training Config", add_help=False
    )
    config_parser.add_argument(
        "-c",
        "--config",
        default="",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser("Spectral SpectralTF pretraining", add_help=True)

    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--in_domains", default="cxrs-xcs", type=str)
    parser.add_argument("--out_domains", default="cxrs-xcs", type=str)
    parser.add_argument("--allow_missing_modalities", action="store_true")
    parser.add_argument(
        "--spectral_normalize",
        default="minmax",
        choices=[
            "peak_ref_minmax",
            "per_wavelength_robust",
            "per_wavelength",
            "minmax",
            "zscore",
            "none",
        ],
        help=(
            "peak_ref_minmax uses fixed-peak min/max as the reference and "
            "applies that scale to the complete spectrum without clipping."
        ),
    )
    parser.add_argument("--spectral_normalize_eps", default=1e-6, type=float)
    parser.add_argument(
        "--spectral_masking",
        default="peak_aware",
        choices=["peak_aware"],
        help="Masking strategy for spectral pretraining.",
    )
    parser.add_argument(
        "--peak_mask_fraction",
        default=0.7,
        type=float,
        help="When using peak_aware masking, fraction of masked tokens sampled from peak regions.",
    )
    parser.add_argument(
        "--peak_token_quantile",
        default=0.8,
        type=float,
        help="Per-sample quantile for defining peak tokens. 0.8 means top 20 percent tokens are candidates.",
    )
    parser.add_argument(
        "--peak_score_mode",
        default="max",
        choices=["max", "mean", "absmax"],
        help="How to compute patch-level peak score.",
    )
    parser.add_argument(
        "--best_metric",
        default="mean_peak",
        choices=["mean", "mean_peak", "cxrs", "xcs", "cxrs_peak", "xcs_peak"],
        help="Metric used to save best checkpoint.",
    )

    parser.add_argument(
        "--peak_loss_weight",
        default=0.7,
        type=float,
        help=(
            "Weight for fixed peak-region MSE loss. "
            "Final loss = (1 - w) * normal_masked_mse + w * peak_region_masked_mse."
        ),
    )
    parser.add_argument(
        "--peak_pool_mask_cap",
        default=0.7,
        type=float,
        help=(
            "Maximum fraction of detected peak candidate tokens that can be masked. "
            "This prevents all peak tokens from being masked."
        ),
    )

    parser.add_argument("--cxrs_input_size", default=[12, 1024], type=int, nargs=2)
    parser.add_argument("--cxrs_original_size", default=[12, 1024], type=int, nargs=2)
    parser.add_argument("--cxrs_patch_size", default=[3, 16], type=int, nargs=2)
    parser.add_argument("--xcs_input_size", default=[35, 490], type=int, nargs=2)
    parser.add_argument("--xcs_original_size", default=[35, 487], type=int, nargs=2)
    parser.add_argument("--xcs_patch_size", default=[5, 14], type=int, nargs=2)
    parser.add_argument("--pad_xcs_width_to", default=490, type=int)

    parser.add_argument("--model", default="pretrain_SpectralTF_base", type=str)
    parser.add_argument("--num_encoded_tokens", default=100, type=int)
    parser.add_argument("--num_global_tokens", default=1, type=int)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--decoder_use_task_queries", default=True, action="store_true")
    parser.add_argument("--decoder_use_xattn", default=True, action="store_true")
    parser.add_argument("--decoder_dim", default=256, type=int)
    parser.add_argument("--decoder_depth", default=2, type=int)
    parser.add_argument("--decoder_num_heads", default=8, type=int)
    parser.add_argument("--drop_path", type=float, default=0.0)
    parser.add_argument("--loss_on_unmasked", default=False, action="store_true")
    parser.add_argument(
        "--no_loss_on_unmasked", action="store_false", dest="loss_on_unmasked"
    )
    parser.set_defaults(loss_on_unmasked=False)
    parser.add_argument("--fp32_output_adapters", type=str, default="")

    parser.add_argument("--opt", default="adamw", type=str)
    parser.add_argument("--opt_eps", default=1e-8, type=float)
    parser.add_argument("--opt_betas", default=[0.9, 0.95], type=float, nargs="+")
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--decoder_decay", type=float, default=None)
    parser.add_argument("--blr", type=float, default=1e-4)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=40)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--task_balancer", type=str, default="none")
    parser.add_argument("--balancer_lr_scale", type=float, default=1.0)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--save_ckpt_freq", default=10, type=int)
    parser.add_argument("--eval_freq", default=1, type=int)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--max_val_batches", default=-1, type=int)

    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--vis_freq", default=10, type=int)
    parser.add_argument("--vis_max_samples", default=2, type=int)
    parser.add_argument(
        "--timestamp_output", dest="timestamp_output", action="store_true"
    )
    parser.add_argument(
        "--no_timestamp_output", dest="timestamp_output", action="store_false"
    )
    parser.set_defaults(timestamp_output=True)

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    if not isinstance(args.data_path, str) or not args.data_path.strip():
        parser.error("Dataset path is empty. Set --data_path or data_path in the YAML config.")
    return args

def patchify_spectral_for_scores(
    x: torch.Tensor, patch_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Convert spectral image to patch vectors for peak scoring.

    Args:
        x: [B, 1, H, W]
        patch_size: (p_h, p_w)

    Returns:
        patches: [B, N, p_h * p_w]
    """
    B, C, H, W = x.shape
    p_h, p_w = patch_size

    if C != 1:
        raise ValueError(f"Expected single-channel spectral input, got C={C}")
    if H % p_h != 0 or W % p_w != 0:
        raise ValueError(
            f"Input size {(H, W)} must be divisible by patch_size={patch_size}"
        )

    n_h = H // p_h
    n_w = W // p_w

    patches = rearrange(
        x[:, 0],
        "b (nh ph) (nw pw) -> b (nh nw) (ph pw)",
        nh=n_h,
        nw=n_w,
        ph=p_h,
        pw=p_w,
    )
    return patches

def compute_patch_peak_scores(
    x: torch.Tensor,
    patch_size: Tuple[int, int],
    score_mode: str = "max",
) -> torch.Tensor:
    """
    Compute one peak score for each patch.

    Args:
        x: [B, 1, H, W]
        patch_size: (p_h, p_w)
        score_mode:
            max: patch maximum, recommended for positive emission peaks
            mean: patch mean intensity
            absmax: maximum absolute value, useful if negative peaks matter

    Returns:
        scores: [B, N]
    """
    patches = patchify_spectral_for_scores(x, patch_size)

    if score_mode == "max":
        return patches.amax(dim=-1)
    if score_mode == "mean":
        return patches.mean(dim=-1)
    if score_mode == "absmax":
        return patches.abs().amax(dim=-1)

    raise ValueError(f"Unsupported peak_score_mode: {score_mode}")

def build_peak_aware_task_masks(
    tasks_dict: Dict[str, torch.Tensor],
    args,
    domain_specs,
) -> Dict[str, torch.Tensor]:
    """
    Peak-aware masking without forced peak masking.

    Strategy:
        1. Detect peak candidate tokens by per-sample quantile.
        2. Try to sample peak_mask_fraction of masked tokens from peak candidates.
        3. But mask at most peak_pool_mask_cap of all peak candidates.
        4. Fill the remaining mask budget from background tokens.

    This guarantees that some peak tokens remain visible.
    """
    if getattr(args, "spectral_masking", None) != "peak_aware":
        raise ValueError("Joint pretraining requires spectral_masking=peak_aware.")

    input_domains = [task for task in args.in_domains if task in tasks_dict]

    if len(input_domains) == 0:
        return None

    score_dict = {}
    token_counts = {}
    batch_size = None
    device = None

    for task in input_domains:
        x = tasks_dict[task]

        if batch_size is None:
            batch_size = x.shape[0]
            device = x.device

        scores = compute_patch_peak_scores(
            x=x,
            patch_size=domain_specs[task]["patch_size"],
            score_mode=getattr(args, "peak_score_mode", "max"),
        )

        score_dict[task] = scores
        token_counts[task] = scores.shape[1]

    total_tokens = sum(token_counts.values())
    num_keep = int(args.num_encoded_tokens)
    if not 0 < num_keep < total_tokens:
        raise ValueError(
            f"Peak-aware pretraining must mask at least one token and keep at least one token: "
            f"num_encoded_tokens={num_keep}, total_tokens={total_tokens}"
        )
    num_mask = total_tokens - num_keep

    if num_mask <= 0:
        return {
            task: torch.zeros(
                (batch_size, token_counts[task]),
                dtype=torch.long,
                device=device,
            )
            for task in input_domains
        }

    all_scores = torch.cat(
        [score_dict[task] for task in input_domains],
        dim=1,
    )  # [B, total_tokens]

    mask_all = torch.zeros(
        (batch_size, total_tokens),
        dtype=torch.long,
        device=device,
    )

    peak_fraction = float(getattr(args, "peak_mask_fraction", 0.7))
    peak_fraction = min(max(peak_fraction, 0.0), 1.0)

    peak_quantile = float(getattr(args, "peak_token_quantile", 0.8))
    peak_quantile = min(max(peak_quantile, 0.0), 1.0)

    peak_pool_mask_cap = float(getattr(args, "peak_pool_mask_cap", 0.7))
    peak_pool_mask_cap = min(max(peak_pool_mask_cap, 0.0), 1.0)

    all_indices = torch.arange(total_tokens, device=device)

    peak_mask_counts = []
    peak_visible_counts = []

    for b in range(batch_size):
        scores_b = all_scores[b].float()

        threshold = torch.quantile(scores_b, peak_quantile)

        peak_idx = all_indices[scores_b > threshold]
        bg_idx = all_indices[scores_b <= threshold]

        if peak_idx.numel() == 0:
            perm = torch.randperm(total_tokens, device=device)
            selected = perm[:num_mask]
            mask_all[b, selected] = 1
            peak_mask_counts.append(0)
            peak_visible_counts.append(0)
            continue

        target_peak_mask = int(round(num_mask * peak_fraction))

        # Leave a fraction of peak candidates visible for spectral context.
        max_peak_mask = int(math.floor(peak_idx.numel() * peak_pool_mask_cap))

        n_peak_mask = min(
            target_peak_mask,
            max_peak_mask,
            int(peak_idx.numel()),
            num_mask,
        )

        n_bg_mask = num_mask - n_peak_mask
        n_bg_mask = min(n_bg_mask, int(bg_idx.numel()))

        sampled_parts = []

        if n_peak_mask > 0:
            peak_perm = torch.randperm(peak_idx.numel(), device=device)[:n_peak_mask]
            sampled_parts.append(peak_idx[peak_perm])

        if n_bg_mask > 0:
            bg_perm = torch.randperm(bg_idx.numel(), device=device)[:n_bg_mask]
            sampled_parts.append(bg_idx[bg_perm])

        if len(sampled_parts) > 0:
            selected = torch.cat(sampled_parts, dim=0)
        else:
            selected = torch.empty(0, dtype=torch.long, device=device)

        # Fill a mask shortfall from remaining unselected tokens.
        if selected.numel() < num_mask:
            already = torch.zeros(total_tokens, dtype=torch.bool, device=device)
            already[selected] = True
            rest_idx = all_indices[~already]

            need = min(num_mask - selected.numel(), rest_idx.numel())

            if need > 0:
                rest_perm = torch.randperm(rest_idx.numel(), device=device)[:need]
                selected = torch.cat([selected, rest_idx[rest_perm]], dim=0)

        if selected.numel() > num_mask:
            perm = torch.randperm(selected.numel(), device=device)[:num_mask]
            selected = selected[perm]

        mask_all[b, selected] = 1

        peak_bool = torch.zeros(total_tokens, dtype=torch.bool, device=device)
        peak_bool[peak_idx] = True
        peak_selected = int(peak_bool[selected].sum().item())

        peak_mask_counts.append(peak_selected)
        peak_visible_counts.append(int(peak_idx.numel() - peak_selected))

    task_masks = {}
    start = 0

    for task in input_domains:
        n = token_counts[task]
        task_masks[task] = mask_all[:, start:start + n]
        start += n

    if not hasattr(args, "_printed_peak_aware_info"):
        print(
            f"Peak-aware masking enabled. "
            f"total_tokens={total_tokens}, num_mask={num_mask}, num_keep={num_keep}, "
            f"peak_fraction={peak_fraction}, peak_pool_mask_cap={peak_pool_mask_cap}, "
            f"avg_peak_masked={np.mean(peak_mask_counts):.2f}, "
            f"avg_peak_visible={np.mean(peak_visible_counts):.2f}"
        )
        args._printed_peak_aware_info = True

    return task_masks

def build_pretraining_inputs_and_masks(
    tasks_dict,
    args,
    domain_specs,
):
    """
    Full-spectrum pretraining:
        complete spectral input
        random / peak-aware masking
    """
    if set(args.in_domains) != {"cxrs", "xcs"} or set(args.out_domains) != {"cxrs", "xcs"}:
        raise ValueError("Joint peak-aware pretraining requires both cxrs and xcs inputs/outputs")
    if not all(task in tasks_dict for task in args.in_domains):
        raise ValueError("Joint pretraining batch must contain paired CXRS and XCS spectra")
    input_dict = make_input_dict(tasks_dict, args.in_domains)

    task_masks = build_peak_aware_task_masks(
        tasks_dict=tasks_dict,
        args=args,
        domain_specs=domain_specs,
    )

    return input_dict, task_masks

def train_one_epoch(
    model,
    data_loader,
    tasks_loss_fn,
    optimizer,
    scaler,
    device,
    epoch,
    args,
    domain_specs,
):
    model.train()
    amp_enabled = device.type == "cuda"
    total_loss = 0.0
    total_steps = 0
    start = time.time()

    for step, (x, _) in enumerate(data_loader):
        epoch_float = epoch + step / max(1, len(data_loader))
        lr = cosine_lr(
            args.lr,
            args.min_lr,
            epoch_float,
            args.epochs,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
        )
        set_optimizer_lr(optimizer, lr)

        tasks_dict = move_batch_to_device(x, device)
        input_dict, task_masks = build_pretraining_inputs_and_masks(
            tasks_dict=tasks_dict,
            args=args,
            domain_specs=domain_specs,
        )

        optimizer.zero_grad(set_to_none=True)
        autocast_ctx = torch.cuda.amp.autocast if amp_enabled else nullcontext
        with autocast_ctx():
            preds, masks = model(
                input_dict,
                mask_inputs=True,
                task_masks=task_masks,
                num_encoded_tokens=args.num_encoded_tokens,
                fp32_output_adapters=args.fp32_output_adapters,
            )

            task_losses = {}
            task_loss_logs = {}

            for task in args.out_domains:
                pred = preds[task].float()
                target = tasks_dict[task].float()

                # Loss is computed on the raw normalized spectrum.
        
                loss_mask = None if args.loss_on_unmasked else masks.get(task, None)

                normal_loss = tasks_loss_fn[task](
                    pred,
                    target,
                    mask=loss_mask,
                )

                peak_loss = compute_fixed_peak_mse_loss(
                    pred_full=pred,
                    target_full=target,
                    mask=loss_mask,
                    task=task,
                    args=args,
                    domain_specs=domain_specs,
                )

                peak_weight = get_task_peak_loss_weight(args, task)

                task_loss = normal_loss + peak_weight * peak_loss

                task_losses[task] = task_loss

                task_loss_logs[f"{task}_loss"] = task_loss.detach().item()
                task_loss_logs[f"{task}_normal_loss"] = normal_loss.detach().item()
                task_loss_logs[f"{task}_peak_loss"] = peak_loss.detach().item()
                task_loss_logs[f"{task}_peak_weight"] = peak_weight

            loss = sum(task_losses.values())

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch={epoch}, step={step}: {loss.item()}"
            )

        if amp_enabled:
            scaler.scale(loss).backward()
            if args.clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        total_loss += loss.item()
        total_steps += 1

        if step % args.print_freq == 0:
            loss_parts = " ".join(
                [
                    f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in task_loss_logs.items()
                ]
            )

            print(
                f"Epoch [{epoch}/{args.epochs}] step [{step}/{len(data_loader)}] lr={lr:.6e} loss={loss.item():.6f} {loss_parts}"
            )

    elapsed = time.time() - start
    avg_loss = total_loss / max(total_steps, 1)
    print(
        f"Epoch {epoch} training: loss={avg_loss:.6f}, time={str(datetime.timedelta(seconds=int(elapsed)))}"
    )
    return {"train_loss": avg_loss}

@torch.no_grad()
def evaluate_r2(
    model,
    data_loader,
    device,
    args,
    domain_specs,
    print_prefix="Validation R2",
):
    model.eval()

    r2_accum = {
        task: init_r2_accumulator()
        for task in args.out_domains
    }

    peak_r2_accum = {
        task: init_r2_accumulator()
        for task in args.out_domains
    }


    for batch_idx, batch in enumerate(data_loader):
        if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
            break

        if isinstance(batch, (list, tuple)):
            sample_batch = batch[0]
        else:
            sample_batch = batch

        tasks_dict = move_batch_to_device(sample_batch, device)

        input_dict, task_masks = build_pretraining_inputs_and_masks(
            tasks_dict=tasks_dict,
            args=args,
            domain_specs=domain_specs,
        )

        preds, masks = model(
            input_dict,
            mask_inputs=True,
            task_masks=task_masks,
            num_encoded_tokens=args.num_encoded_tokens,
            fp32_output_adapters=args.fp32_output_adapters,
        )

        for task in args.out_domains:
            if task not in preds or task not in tasks_dict:
                continue

            original_size = domain_specs[task]["original_size"]
            patch_size = domain_specs[task]["patch_size"]

            pred_full = preds[task].float()
            target_full = tasks_dict[task].float()

            # R2/checkpoint metrics use the raw normalized target.

            pred = crop_to_original(pred_full, original_size)
            target = crop_to_original(target_full, original_size)

            valid_mask = None

            if not args.loss_on_unmasked and task in masks:
                pixel_mask = mask_to_pixel_mask(
                    masks[task],
                    pred_full.shape[-2:],
                    patch_size,
                )
                valid_mask = crop_to_original(pixel_mask, original_size)

            update_r2_accumulator(
                r2_accum[task],
                pred=pred,
                target=target,
                valid_mask=valid_mask,
            )

            fixed_peak_mask = make_fixed_r2_region_mask(
                target=target,
                task=task,
                valid_mask=valid_mask,
            )

            update_r2_accumulator(
                peak_r2_accum[task],
                pred=pred,
                target=target,
                valid_mask=fixed_peak_mask,
            )


    stats = {}

    for task in args.out_domains:
        stats[task] = finalize_r2(r2_accum[task])
        stats[f"{task}_peak"] = finalize_r2(peak_r2_accum[task])

    stats["mean"] = safe_mean([
        stats[task]
        for task in args.out_domains
    ])

    stats["mean_peak"] = safe_mean([
        stats[f"{task}_peak"]
        for task in args.out_domains
    ])


    print(
        print_prefix + ": "
        + " ".join([
            f"{k}={v:.6f}" if np.isfinite(v) else f"{k}=nan"
            for k, v in stats.items()
        ])
    )

    return stats

@torch.no_grad()
def visualize_epoch_reconstruction_results(
    model, data_loader, device, epoch, output_dir, args, domain_specs
):
    model.eval()

    try:
        sample_batch, _ = next(iter(data_loader))
    except StopIteration:
        return

    tasks_dict = move_batch_to_device(sample_batch, device)
    input_dict, task_masks = build_pretraining_inputs_and_masks(
        tasks_dict=tasks_dict,
        args=args,
        domain_specs=domain_specs,
    )

    # Reuse one forward pass for every visualization of each batch.
    preds, masks = model(
        input_dict,
        mask_inputs=True,
        task_masks=task_masks,
        num_encoded_tokens=args.num_encoded_tokens,
        fp32_output_adapters=args.fp32_output_adapters,
    )

    vis_root = Path(output_dir) / "visualizations" / f"epoch_{epoch:04d}"
    vis_root.mkdir(parents=True, exist_ok=True)

    batch_size = next(iter(tasks_dict.values())).shape[0]

    # Visualize the same sample indices for both spectral modalities.
    num_samples = min(getattr(args, "vis_max_samples", 2), batch_size)
    sample_indices = list(range(num_samples))

    for task in args.out_domains:
        if task not in tasks_dict or task not in preds:
            continue

        original_size = domain_specs[task]["original_size"]
        patch_size = domain_specs[task]["patch_size"]

        # patch-level mask -> pixel-level mask
        pixel_mask = None
        if task in masks:
            pixel_mask = mask_to_pixel_mask(
                masks[task], preds[task].shape[-2:], patch_size
            )
            pixel_mask = crop_to_original(pixel_mask, original_size)

        for sample_idx in sample_indices:
            sample_dir = vis_root / task / f"sample_{sample_idx:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            target = crop_to_original(
                tasks_dict[task][sample_idx, 0].detach().cpu(), original_size
            ).numpy()

            pred = crop_to_original(
                preds[task][sample_idx, 0].detach().cpu(), original_size
            ).numpy()

            mask_np = None
            if pixel_mask is not None:
                mask_np = pixel_mask[sample_idx].detach().cpu().numpy().astype(bool)

            # 1. Full-spectrum reconstruction visualization.
            if bool(getattr(args, "visualize_full_spectrum", True)):
                _save_full_reconstruction_heatmap(
                    target=target,
                    pred=pred,
                    task=task,
                    epoch=epoch,
                    sample_idx=sample_idx,
                    save_path=sample_dir / "01_full_reconstruction_heatmap.png",
                )

                _save_full_spectral_lines(
                    target=target,
                    pred=pred,
                    task=task,
                    epoch=epoch,
                    sample_idx=sample_idx,
                    save_path=sample_dir / "02_full_spectral_lines.png",
                )

            # 2. Masked-region reconstruction visualization.
            if bool(getattr(args, "visualize_masked_spectrum", True)) and mask_np is not None:
                _save_masked_spectral_lines(
                    target=target,
                    pred=pred,
                    mask_np=mask_np,
                    task=task,
                    epoch=epoch,
                    sample_idx=sample_idx,
                    save_path=sample_dir / "03_masked_spectral_lines.png",
                )

            # Visualize the fixed spectral peak region.
            if bool(getattr(args, "visualize_fixed_peak", True)):
                _save_peak_reconstruction_heatmap(
                    target=target,
                    pred=pred,
                    task=task,
                    epoch=epoch,
                    sample_idx=sample_idx,
                    save_path=sample_dir / "04_fixed_peak_heatmap.png",
                )

                _save_peak_spectral_lines(
                    target=target, pred=pred, task=task, epoch=epoch,
                    sample_idx=sample_idx,
                    save_path=sample_dir / "05_fixed_peak_spectral_lines.png",
                )

def main():
    args = get_args()
    args.in_domains = (
        args.in_domains.split("-")
        if isinstance(args.in_domains, str)
        else args.in_domains
    )
    args.out_domains = (
        args.out_domains.split("-")
        if isinstance(args.out_domains, str)
        else args.out_domains
    )
    args.all_domains = list(dict.fromkeys(args.in_domains + args.out_domains))
    args.fp32_output_adapters = [x for x in args.fp32_output_adapters.split("-") if x]

    total_batch_size = args.batch_size
    args.lr = args.blr * total_batch_size / 256

    base_output_dir = Path(args.output_dir)

    if args.timestamp_output:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_output_dir.parent / f"{base_output_dir.name}_{timestamp}"
    else:
        output_dir = base_output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)

    print(f"Output directory: {output_dir}")

    device = resolve_device(args.device)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(
            f"Using CUDA device: {device}, current_device={torch.cuda.current_device()}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    domain_specs = get_domain_specs(args, args.all_domains)
    validate_shared_geometry(domain_specs)
    model = get_model(args, domain_specs).to(device)
    with open(output_dir / "model.txt", "w") as f:
        f.write(str(model))

    tasks_loss_fn = {
        domain: MaskedMSELoss(patch_size=domain_specs[domain]["patch_size"], stride=1)
        for domain in args.out_domains
    }

    dataset_train = build_spectral_pretraining_dataset(args, split="train")
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    dataset_val = None
    data_loader_val = None
    val_path = os.path.join(args.data_path, "val")
    if os.path.isdir(val_path):
        dataset_val = build_spectral_pretraining_dataset(args, split="val")
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    optimizer = create_optimizer(args, model)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_metric_value = -float("inf")

    if args.resume:
        checkpoint = load_checkpoint(
            args.resume, model, optimizer=optimizer, scaler=scaler, device=device
        )
        args.start_epoch = int(checkpoint.get("epoch", args.start_epoch - 1)) + 1

        # Restore the saved selection metric from supported checkpoint fields.
        best_metric_value = float(
            checkpoint.get(
                "best_metric_value", checkpoint.get("best_r2", best_metric_value)
            )
        )

        previous_best_metric_name = checkpoint.get("best_metric_name", "best_r2")

        print(
            f"Resumed from {args.resume} at epoch {args.start_epoch}, "
            f"previous_best_metric={previous_best_metric_name}, "
            f"best_metric_value={best_metric_value:.6f}, "
            f"current_best_metric={args.best_metric}"
        )

    print(f"Start spectral SpectralTF pretraining for {args.epochs} epochs on {device}.")
    log_path = output_dir / "log_spectral_pretraining.jsonl"

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            tasks_loss_fn,
            optimizer,
            scaler,
            device,
            epoch,
            args,
            domain_specs,
        )

        stats = {"epoch": epoch, **train_stats}
        current_best_metric = None

        if data_loader_val is not None and (
            epoch % args.eval_freq == 0 or epoch == args.epochs - 1
        ):
            r2_stats = evaluate_r2(
                model=model,
                data_loader=data_loader_val,
                device=device,
                args=args,
                domain_specs=domain_specs,
            )

            stats.update({
                f"val_{k}_r2": v
                for k, v in r2_stats.items()
            })

            current_best_metric = r2_stats.get(args.best_metric)

            if current_best_metric is None:
                raise KeyError(
                    f"best_metric={args.best_metric} was not found in r2_stats. "
                    f"Available metrics: {list(r2_stats.keys())}"
                )

            if epoch % args.vis_freq == 0 or epoch == args.epochs - 1:
                visualize_epoch_reconstruction_results(
                    model=model,
                    data_loader=data_loader_val,
                    device=device,
                    epoch=epoch,
                    output_dir=args.output_dir,
                    args=args,
                    domain_specs=domain_specs,
                )

            if (
                current_best_metric is not None
                and np.isfinite(current_best_metric)
                and current_best_metric > best_metric_value
            ):
                best_metric_value = float(current_best_metric)

                save_checkpoint(
                    output_dir / "checkpoint_best.pth",
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    args,
                    stats,
                    best_metric_value,
                )

                print(
                    f"New best {args.best_metric} R2: "
                    f"{best_metric_value:.6f} at epoch {epoch}"
                )

        with open(log_path, "a") as f:
            f.write(json.dumps(stats) + "\n")

    print("Training finished.")

if __name__ == "__main__":
    main()
