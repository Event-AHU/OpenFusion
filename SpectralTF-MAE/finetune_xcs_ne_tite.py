# Copyright (c) EPFL VILAB.
# All rights reserved.
#
# Fine-tune the shared encoder on full XCS spectra to predict ne(2), Ti(32),
# and Te(32). The XCS width is padded from 487 to 490 without wavelength
# cropping. Pretrained spectral decoders are excluded; prediction heads are
# trained for the 66 standardized regression targets.

import argparse
import datetime
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import yaml

from utils.optim_factory import create_optimizer
from utils.spectral_pretraining_common import cosine_lr, get_domain_specs, load_checkpoint, resolve_device, save_checkpoint, set_optimizer_lr
from utils.xcs_ne_dataset import XCSNeDataset
from utils.xcs_ne_model import build_xcs_ne_model, freeze_backbone_parts, load_pretrained_backbone_only
from utils.shared_checkpoint import validate_shared_geometry




def get_args():
    config_parser = argparse.ArgumentParser(description="XCS -> ne/Ti/Te Fine-tuning Config", add_help=False)
    config_parser.add_argument("-c", "--config", default="", type=str, metavar="FILE")

    parser = argparse.ArgumentParser("Spectral downstream one-shot fine-tuning: xcs -> ne + Ti + Te", add_help=True)

    parser.add_argument(
        "--data_path",
        default="",
        type=str,
        help="Directory containing train/val/test XCS-ne-Ti-Te NPY files.",
    )
    parser.add_argument("--target_stats_path", default="", type=str, help="Optional target_stats_train.npz path. Defaults to data_path/target_stats_train.npz.")
    parser.add_argument("--npy_mmap", dest="npy_mmap", action="store_true", help="Memory-map XCS npy files to reduce RAM usage.")
    parser.add_argument("--no_npy_mmap", dest="npy_mmap", action="store_false")
    parser.set_defaults(npy_mmap=True)
    parser.add_argument("--split_seed", default=None, type=int, help="Seed for internal 7:2:1 split when data_path contains full X_spec_full/Y_data files.")
    parser.add_argument("--split_ratios", default=[0.7, 0.2, 0.1], type=float, nargs=3, help="Train/val/test ratios for full-export data_path.")
    parser.add_argument("--in_domains", default="xcs", type=str)
    parser.add_argument("--out_domains", default="ne-ti-te", type=str)
    parser.add_argument("--allow_missing_modalities", action="store_true")
    parser.add_argument(
        "--spectral_normalize",
        default="minmax",
        choices=[
            "peak_ref_minmax", "per_wavelength_robust", "per_wavelength",
            "minmax", "zscore", "none",
        ],
    )
    parser.add_argument("--spectral_normalize_eps", default=1e-6, type=float)

    # Shared encoder geometry: [35, 490] with XCS 5x14 patches (245 full tokens).
    parser.add_argument("--xcs_input_size", default=[35, 490], type=int, nargs=2)
    parser.add_argument("--xcs_original_size", default=[35, 487], type=int, nargs=2)
    parser.add_argument("--xcs_patch_size", default=[5, 14], type=int, nargs=2)
    parser.add_argument("--pad_xcs_width_to", default=490, type=int)

    parser.add_argument("--model", default="pretrain_SpectralTF_base", type=str)
    parser.add_argument("--num_global_tokens", default=1, type=int)
    parser.add_argument("--drop_path", type=float, default=0.0)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--input_size", default=224, type=int)

    parser.add_argument("--pretrained_ckpt", default="", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--test_only", default=False, action="store_true", help="Load --resume and run the test split only, without further fine-tuning.")
    parser.add_argument("--no_test_only", action="store_false", dest="test_only")
    parser.add_argument("--freeze_backbone", default=False, action="store_true")
    parser.add_argument("--freeze_encoder", default=False, action="store_true")
    parser.add_argument("--freeze_input_adapters", default=False, action="store_true")

    parser.add_argument("--opt", default="adamw", type=str)
    parser.add_argument("--opt_eps", default=1e-8, type=float)
    parser.add_argument("--opt_betas", default=[0.9, 0.95], type=float, nargs="+")
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--decoder_decay", type=float, default=None)
    parser.add_argument("--blr", type=float, default=5e-5)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--task_balancer", type=str, default="none")
    parser.add_argument("--balancer_lr_scale", type=float, default=1.0)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--save_ckpt_freq", default=10, type=int)
    parser.add_argument("--eval_freq", default=1, type=int)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--max_val_batches", default=-1, type=int)
    parser.add_argument("--max_test_batches", default=-1, type=int)
    parser.add_argument(
        "--best_metric",
        default="mean_r2",
        choices=["mean_r2", "ne1_r2", "ne2_r2", "ti_r2", "te_r2", "val_loss"],
    )

    parser.add_argument("--output_dir", default="./output/finetune/xcs_to_ne_tite_once")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--save_predictions", default=True, action="store_true")
    parser.add_argument("--no_save_predictions", action="store_false", dest="save_predictions")
    parser.add_argument("--visualize_tite", default=True, action="store_true", help="Save Ti/Te True-vs-Pred and relative-error figures on the test split.")
    parser.add_argument("--no_visualize_tite", action="store_false", dest="visualize_tite")
    parser.add_argument("--num_visual_samples", default=16, type=int)
    parser.add_argument("--timestamp_output", dest="timestamp_output", action="store_true")
    parser.add_argument("--no_timestamp_output", dest="timestamp_output", action="store_false")
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonicalize_args(args):
    args.in_domains = args.in_domains.split("-") if isinstance(args.in_domains, str) else args.in_domains
    args.out_domains = args.out_domains.split("-") if isinstance(args.out_domains, str) else args.out_domains
    args.all_domains = ["xcs"]

    if args.pad_xcs_width_to and int(args.pad_xcs_width_to) > 0:
        args.xcs_input_size = [int(args.xcs_original_size[0]), int(args.pad_xcs_width_to)]

    if args.in_domains != ["xcs"] or args.out_domains != ["ne", "ti", "te"]:
        print(
            "Warning: this script is designed for downstream xcs -> ne + Ti + Te. "
            f"Current in_domains={args.in_domains}, out_domains={args.out_domains}."
        )

    input_h, input_w = [int(x) for x in args.xcs_input_size]
    patch_h, patch_w = [int(x) for x in args.xcs_patch_size]
    if input_h % patch_h != 0 or input_w % patch_w != 0:
        raise ValueError(
            "xcs_input_size must be divisible by xcs_patch_size for PatchedInputAdapter. "
            f"Got xcs_input_size={args.xcs_input_size}, xcs_patch_size={args.xcs_patch_size}. "
            "For the shared 5x14 checkpoint, use pad_xcs_width_to: 490."
        )

    args.lr = args.blr * args.batch_size / 256
    return args


def collate_xcs_ne(batch):
    xcs, y, shots = zip(*batch)
    return torch.stack(xcs, dim=0), torch.stack(y, dim=0), torch.as_tensor(shots, dtype=torch.long)


def move_batch_to_device(batch, device):
    xcs, targets, shots = batch
    return xcs.to(device, non_blocking=True), targets.to(device, non_blocking=True), shots.to(device, non_blocking=True)


def init_ne_metric_accumulators():
    # ne1/ne2 are evaluated independently; Ti and Te are each flattened over
    # all samples and all 32 radial points to produce one R2 per profile type.
    return {
        name: {"sum_y": 0.0, "sum_y2": 0.0, "sum_res2": 0.0, "count": 0.0}
        for name in ("ne1", "ne2", "ti", "te")
    }


# Pooled Ti/Te metrics use normalized values at positive physical targets.

def init_9692_tite_accumulator():
    return {
        "mse_sum": 0.0,
        "abs_err_sum": 0.0,
        "target_sum": 0.0,
        "target_sq_sum": 0.0,
        "valid_count": 0.0,
    }


def update_9692_tite_metric(accum, pred_norm, target_norm, target_original):
    """Accumulate normalized Ti/Te errors at physically positive targets.

    Detached CPU copies keep these evaluation metrics independent of autograd
    and the model's execution device.
    """
    if pred_norm.ndim != 2 or target_norm.ndim != 2 or target_original.ndim != 2:
        raise ValueError(
            "9692 metric expects 2D tensors [B, 66], got "
            f"pred={tuple(pred_norm.shape)}, target_norm={tuple(target_norm.shape)}, "
            f"target_original={tuple(target_original.shape)}"
        )
    if pred_norm.shape[1] != 66 or target_norm.shape[1] != 66 or target_original.shape[1] != 66:
        raise ValueError(
            "9692 metric expects 66 dimensions [ne(2), Ti(32), Te(32)], got "
            f"pred={tuple(pred_norm.shape)}, target_norm={tuple(target_norm.shape)}, "
            f"target_original={tuple(target_original.shape)}"
        )

    # CPU copies are used only to calculate metrics.
    pred_tite_norm = pred_norm[:, 2:66].detach().float().cpu()
    target_tite_norm = target_norm[:, 2:66].detach().float().cpu()
    target_tite_original = target_original[:, 2:66].detach().float().cpu()

    # Ignore targets that are nonpositive in physical units.
    mask = (target_tite_original > 0.0).to(target_tite_norm.dtype)

    # Accumulate errors and moments in normalized target space.
    squared_err = (pred_tite_norm - target_tite_norm) ** 2
    masked_squared_err = squared_err * mask
    masked_targets = target_tite_norm * mask

    accum["mse_sum"] += float(masked_squared_err.sum().item())
    accum["abs_err_sum"] += float((torch.abs(pred_tite_norm - target_tite_norm) * mask).sum().item())
    accum["target_sum"] += float(masked_targets.sum().item())
    accum["target_sq_sum"] += float((masked_targets ** 2).sum().item())
    accum["valid_count"] += float(mask.sum().item())

def finalize_9692_tite_metric(accum):
    count = float(accum["valid_count"])

    if count <= 0:
        return {
            "tite_9692_r2": float("nan"),
            "tite_9692_mse": float("nan"),
            "tite_9692_mae": float("nan"),
            "tite_9692_rmse": float("nan"),
            "tite_9692_target_var": float("nan"),
        }

    avg_mse = accum["mse_sum"] / count
    target_mean = accum["target_sum"] / count
    target_var = (accum["target_sq_sum"] / count) - (target_mean ** 2)

    if target_var > 1e-8:
        r2 = 1.0 - (avg_mse / target_var)
    else:
        r2 = 0.0

    return {
        "tite_9692_r2": float(r2),
        "tite_9692_mse": float(avg_mse),
        "tite_9692_mae": float(accum["abs_err_sum"] / count),
        "tite_9692_rmse": float(np.sqrt(avg_mse)),
        "tite_9692_target_var": float(target_var),
    }


def _update_one_metric(metric, pred_values, target_values):
    pred_values = pred_values.detach().double().cpu().reshape(-1)
    target_values = target_values.detach().double().cpu().reshape(-1)
    finite = torch.isfinite(pred_values) & torch.isfinite(target_values)
    if not torch.any(finite):
        return
    target_values = target_values[finite]
    residual = target_values - pred_values[finite]
    metric["sum_y"] += float(target_values.sum().item())
    metric["sum_y2"] += float((target_values ** 2).sum().item())
    metric["sum_res2"] += float((residual ** 2).sum().item())
    metric["sum_abs"] = metric.get("sum_abs", 0.0) + float(residual.abs().sum().item())
    metric["count"] += float(target_values.numel())


def update_ne_metrics(accum, pred, target):
    if pred.ndim != 2 or target.ndim != 2 or pred.shape[1] != 66 or target.shape[1] != 66:
        raise ValueError(
            f"Metrics expect [B, 66] predictions/targets, got {tuple(pred.shape)} and {tuple(target.shape)}"
        )
    _update_one_metric(accum["ne1"], pred[:, 0], target[:, 0])
    _update_one_metric(accum["ne2"], pred[:, 1], target[:, 1])
    _update_one_metric(accum["ti"], pred[:, 2:34], target[:, 2:34])
    _update_one_metric(accum["te"], pred[:, 34:66], target[:, 34:66])


def _finalize_one_metric(metric):
    count = float(metric["count"])
    if count <= 0:
        return float("nan"), float("nan"), float("nan")
    ss_tot = metric["sum_y2"] - (metric["sum_y"] ** 2) / count
    ss_res = metric["sum_res2"]
    r2 = float("nan") if abs(ss_tot) < 1e-12 else float(1.0 - ss_res / ss_tot)
    rmse = float(np.sqrt(ss_res / count))
    mae = float(metric.get("sum_abs", 0.0) / count)
    return r2, rmse, mae


def finalize_ne_metrics(accum):
    stats = {}
    r2_values = []
    for name in ("ne1", "ne2", "ti", "te"):
        r2, rmse, mae = _finalize_one_metric(accum[name])
        stats[f"{name}_r2"] = r2
        # Report MAE/RMSE for all target groups, including the Ti and Te
        # profiles.  These are computed in the original physical-value space.
        stats[f"{name}_rmse"] = rmse
        stats[f"{name}_mae"] = mae
        if np.isfinite(r2):
            r2_values.append(r2)
    stats["mean_r2"] = float(np.mean(r2_values)) if r2_values else float("nan")
    return stats


def is_higher_better(metric_name):
    return metric_name.endswith("_r2")


def train_one_epoch(model, data_loader, optimizer, scaler, device, epoch, args):
    model.train()
    total_loss = 0.0
    total_steps = 0
    start = time.time()
    amp_enabled = device.type == "cuda"
    autocast_ctx = torch.cuda.amp.autocast if amp_enabled else nullcontext

    for step, batch in enumerate(data_loader):
        epoch_float = epoch + step / max(len(data_loader), 1)
        lr = cosine_lr(args.lr, args.min_lr, epoch_float, args.epochs, args.warmup_epochs, args.warmup_lr)
        set_optimizer_lr(optimizer, lr)

        xcs, targets_norm, _ = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx():
            pred_norm = model(xcs)
            loss = F.mse_loss(pred_norm, targets_norm)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}: {loss.item()}")

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

        total_loss += float(loss.item())
        total_steps += 1

        if step % args.print_freq == 0:
            print(
                f"Epoch [{epoch}/{args.epochs}] step [{step}/{len(data_loader)}] "
                f"lr={lr:.6e} loss={loss.item():.6f}"
            )

    avg_loss = total_loss / max(total_steps, 1)
    elapsed = time.time() - start
    print(
        f"Epoch {epoch} training: loss={avg_loss:.6f}, "
        f"time={str(datetime.timedelta(seconds=int(elapsed)))}"
    )
    return {"train_loss": avg_loss}


@torch.no_grad()
def evaluate_ne(model, data_loader, dataset, device, args, print_prefix="Validation"):
    model.eval()
    accum = init_ne_metric_accumulators()

    accum_9692 = init_9692_tite_accumulator()

    total_loss = 0.0
    total_steps = 0

    for batch_idx, batch in enumerate(data_loader):
        if args.max_val_batches > 0 and batch_idx >= args.max_val_batches and print_prefix.lower().startswith("validation"):
            break
        if args.max_test_batches > 0 and batch_idx >= args.max_test_batches and print_prefix.lower().startswith("test"):
            break

        xcs, targets_norm, _ = move_batch_to_device(batch, device)
        pred_norm = model(xcs)
        loss = F.mse_loss(pred_norm, targets_norm)

        pred = dataset.get_original_y(pred_norm)
        target = dataset.get_original_y(targets_norm)
        update_ne_metrics(accum, pred, target)

        update_9692_tite_metric(
            accum_9692,
            pred_norm=pred_norm,
            target_norm=targets_norm,
            target_original=target,
        )

        total_loss += float(loss.item())
        total_steps += 1

    stats = finalize_ne_metrics(accum)

    stats.update(finalize_9692_tite_metric(accum_9692))

    stats["loss"] = total_loss / max(total_steps, 1)
    print(
        print_prefix + ": "
        + " ".join([
            f"{k}={v:.6f}" if np.isfinite(v) else f"{k}=nan"
            for k, v in stats.items()
        ])
    )
    return stats


@torch.no_grad()
def collect_predictions(model, data_loader, dataset, device, args):
    model.eval()
    pred_all, target_all, shot_all = [], [], []
    for batch_idx, batch in enumerate(data_loader):
        if args.max_test_batches > 0 and batch_idx >= args.max_test_batches:
            break
        xcs, targets_norm, shots = move_batch_to_device(batch, device)
        pred_norm = model(xcs)
        pred = dataset.get_original_y(pred_norm).detach().cpu().numpy()
        target = dataset.get_original_y(targets_norm).detach().cpu().numpy()
        pred_all.append(pred)
        target_all.append(target)
        shot_all.append(shots.detach().cpu().numpy())
    if not pred_all:
        return None
    pred = np.concatenate(pred_all, axis=0)
    target = np.concatenate(target_all, axis=0)
    return {
        "pred": pred,
        "target": target,
        "pred_ne": pred[:, :2],
        "target_ne": target[:, :2],
        "pred_ti": pred[:, 2:34],
        "target_ti": target[:, 2:34],
        "pred_te": pred[:, 34:66],
        "target_te": target[:, 34:66],
        "shotnum": np.concatenate(shot_all, axis=0),
    }


def visualize_tite_results(predictions, save_dir, num_samples=16):
    """Match 96.92.py test visualization for Ti/Te profiles and relative errors."""
    if predictions is None:
        print("No test predictions available for Ti/Te visualization.")
        return

    target_ti = predictions["target_ti"]
    pred_ti = predictions["pred_ti"]
    target_te = predictions["target_te"]
    pred_te = predictions["pred_te"]
    shotnums = predictions["shotnum"]

    total_samples = int(target_ti.shape[0])
    num_samples = min(max(int(num_samples), 1), total_samples)
    if num_samples <= 0:
        print("No test samples available for Ti/Te visualization.")
        return

    target_ti = target_ti[:num_samples]
    pred_ti = pred_ti[:num_samples]
    target_te = target_te[:num_samples]
    pred_te = pred_te[:num_samples]
    shotnums = shotnums[:num_samples]

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    grid_rho = np.linspace(0.0, 1.0, 32)
    cols = 4
    rows = (num_samples + cols - 1) // cols

    fig_ti, axes_ti = plt.subplots(rows, cols, figsize=(20, 5 * rows), squeeze=False)
    axes_ti = axes_ti.flatten()
    for i in range(num_samples):
        mse = np.mean((target_ti[i] - pred_ti[i]) ** 2)
        ax = axes_ti[i]
        ax.plot(grid_rho, target_ti[i], "b-", label="True")
        ax.plot(grid_rho, pred_ti[i], "r--", label="Pred")
        ax.set_title(f"Shot {int(shotnums[i])} (MSE: {mse:.1f})")
        ax.set_xlabel("rho")
        ax.set_ylabel("Ti (eV)")
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

    for j in range(num_samples, len(axes_ti)):
        axes_ti[j].axis("off")

    plt.tight_layout()
    ti_path = save_dir / "test_predictions_Ti_smooth.png"
    plt.savefig(ti_path)
    plt.close(fig_ti)
    print(f"Ti visualization saved to: {ti_path}")

    fig_te, axes_te = plt.subplots(rows, cols, figsize=(20, 5 * rows), squeeze=False)
    axes_te = axes_te.flatten()
    for i in range(num_samples):
        mse = np.mean((target_te[i] - pred_te[i]) ** 2)
        ax = axes_te[i]
        ax.plot(grid_rho, target_te[i], "b-", label="True")
        ax.plot(grid_rho, pred_te[i], "r--", label="Pred")
        ax.set_title(f"Shot {int(shotnums[i])} (MSE: {mse:.1f})")
        ax.set_xlabel("rho")
        ax.set_ylabel("Te (eV)")
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

    for j in range(num_samples, len(axes_te)):
        axes_te[j].axis("off")

    plt.tight_layout()
    te_path = save_dir / "test_predictions_Te_smooth.png"
    plt.savefig(te_path)
    plt.close(fig_te)
    print(f"Te visualization saved to: {te_path}")

    epsilon = 1e-6
    ti_rel_error = (target_ti - pred_ti) / (target_ti + epsilon)
    te_rel_error = (target_te - pred_te) / (target_te + epsilon)

    fig_err, axes_err = plt.subplots(2, 2, figsize=(15, 12))

    for i in range(num_samples):
        axes_err[0, 0].plot(grid_rho, ti_rel_error[i], alpha=0.5, linewidth=1)
    axes_err[0, 0].axhline(0, color="k", linestyle="--", linewidth=1)
    axes_err[0, 0].set_title("Ti Relative Error Profiles ((True-Pred)/True)")
    axes_err[0, 0].set_xlabel("rho")
    axes_err[0, 0].set_ylabel("Relative Error")
    axes_err[0, 0].set_ylim(-1.0, 1.0)
    axes_err[0, 0].grid(True, alpha=0.3)

    for i in range(num_samples):
        axes_err[0, 1].plot(grid_rho, te_rel_error[i], alpha=0.5, linewidth=1)
    axes_err[0, 1].axhline(0, color="k", linestyle="--", linewidth=1)
    axes_err[0, 1].set_title("Te Relative Error Profiles ((True-Pred)/True)")
    axes_err[0, 1].set_xlabel("rho")
    axes_err[0, 1].set_ylabel("Relative Error")
    axes_err[0, 1].set_ylim(-1.0, 1.0)
    axes_err[0, 1].grid(True, alpha=0.3)

    axes_err[1, 0].hist(
        ti_rel_error.flatten(), bins=50, range=(-1, 1), alpha=0.7, edgecolor="black"
    )
    axes_err[1, 0].set_title("Ti Relative Error Distribution")
    axes_err[1, 0].set_xlabel("Relative Error")
    axes_err[1, 0].set_ylabel("Count")
    axes_err[1, 0].grid(True, alpha=0.3)

    axes_err[1, 1].hist(
        te_rel_error.flatten(), bins=50, range=(-1, 1), alpha=0.7, edgecolor="black"
    )
    axes_err[1, 1].set_title("Te Relative Error Distribution")
    axes_err[1, 1].set_xlabel("Relative Error")
    axes_err[1, 1].set_ylabel("Count")
    axes_err[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    err_path = save_dir / "test_predictions_RelError.png"
    plt.savefig(err_path)
    plt.close(fig_err)
    print(f"Relative-error visualization saved to: {err_path}")


def save_metrics_json(path, stats):
    serializable = {k: (None if not np.isfinite(v) else float(v)) for k, v in stats.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def save_ne_checkpoint(path, model, optimizer, scaler, epoch, args, stats, best_metric_value, dataset):
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
        "target_y_mean": dataset.y_mean.cpu(),
        "target_y_std": dataset.y_std.cpu(),
        "target_layout": {"ne": [0, 2], "ti": [2, 34], "te": [34, 66]},
        "target_dim": 66,
    }
    torch.save(checkpoint, path)


def run_test(model, test_loader, test_dataset, device, args, output_dir):
    test_stats = evaluate_ne(
        model,
        test_loader,
        test_dataset,
        device,
        args,
        print_prefix="Test ne/Ti/Te metrics",
    )
    metrics_path = output_dir / "metrics_test_ne_tite.json"
    save_metrics_json(metrics_path, test_stats)
    print(f"Saved test metrics to: {metrics_path}")

    predictions = None
    if args.save_predictions or args.visualize_tite:
        predictions = collect_predictions(model, test_loader, test_dataset, device, args)

    if predictions is not None and args.save_predictions:
        pred_path = output_dir / "predictions_test_ne_tite.npz"
        np.savez_compressed(pred_path, **predictions)
        print(f"Saved test predictions to: {pred_path}")

    if predictions is not None and args.visualize_tite:
        visualize_tite_results(
            predictions,
            output_dir,
            num_samples=args.num_visual_samples,
        )

    return test_stats


def main():
    args = canonicalize_args(get_args())
    if args.test_only and not args.resume:
        raise ValueError(
            "--test_only requires a fine-tuned checkpoint via --resume "
            "(or resume: in YAML)."
        )

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
        print(f"Using CUDA device: {device}, current_device={torch.cuda.current_device()}")

    set_seed(args.seed)
    cudnn.benchmark = True

    train_dataset = XCSNeDataset(args.data_path, args, split="train")
    val_dataset = XCSNeDataset(args.data_path, args, split="val")
    test_dataset = XCSNeDataset(args.data_path, args, split="test")

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError(
            "Empty split found in NPY dataset. Expected non-empty train/val/test files under "
            f"{args.data_path}."
        )

    print(
        "Data Split from preprocessed NPY - "
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    stats = XCSNeDataset.load_train_target_stats(args.data_path, args.target_stats_path)
    if stats is None:
        print("target_stats_train.npz not found; computing y_mean/y_std from train_y.npy.")
        train_y = train_dataset.Y_data
        y_mean = train_y.mean(dim=0)
        y_std = train_y.std(dim=0, unbiased=False)
        y_std[y_std == 0] = 1.0
    else:
        y_mean, y_std = stats

    for split_dataset in (train_dataset, val_dataset, test_dataset):
        split_dataset.set_target_normalization(y_mean, y_std)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_xcs_ne,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_xcs_ne,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_xcs_ne,
    )

    domain_specs = get_domain_specs(args, ["xcs"])
    validate_shared_geometry(domain_specs)
    model = build_xcs_ne_model(args, domain_specs).to(device)

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer=None, scaler=None, device=device)
        if "target_y_mean" in checkpoint and "target_y_std" in checkpoint:
            train_dataset.set_target_normalization(checkpoint["target_y_mean"], checkpoint["target_y_std"])
            val_dataset.set_target_normalization(checkpoint["target_y_mean"], checkpoint["target_y_std"])
            test_dataset.set_target_normalization(checkpoint["target_y_mean"], checkpoint["target_y_std"])
        args.start_epoch = int(checkpoint.get("epoch", args.start_epoch - 1)) + 1
    else:
        if not args.pretrained_ckpt:
            raise ValueError("Fine-tuning requires --pretrained_ckpt, usually the pretraining checkpoint_best.pth.")
        load_pretrained_backbone_only(model, args.pretrained_ckpt, device=device, prefix="pretrained_ckpt", expected_model_name=args.model)

    freeze_backbone_parts(
        model,
        freeze_backbone=args.freeze_backbone,
        freeze_input_adapters=args.freeze_input_adapters,
        freeze_encoder=args.freeze_encoder,
    )

    with open(output_dir / "model.txt", "w") as f:
        f.write(str(model))
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    dataset_metadata = XCSNeDataset.read_metadata(args.data_path)
    dataset_info = {
        "data_path": args.data_path,
        "split_counts": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "target_y_mean": y_mean.tolist(),
        "target_y_std": y_std.tolist(),
        "metadata": dataset_metadata,
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    if args.test_only:
        print(f"Test-only mode: evaluating resumed checkpoint {args.resume}")
        run_test(model, test_loader, test_dataset, device, args, output_dir)
        print("XCS -> ne + Ti + Te test-only evaluation finished.")
        return

    optimizer = create_optimizer(args, model)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    higher_better = is_higher_better(args.best_metric)
    best_metric_value = -float("inf") if higher_better else float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        best_metric_value = float(checkpoint.get("best_metric_value", best_metric_value))

    print(f"Start XCS -> ne + Ti + Te fine-tuning for {args.epochs} epochs on {device}.")
    log_path = output_dir / "log_spectral_finetuning_xcs_to_ne_tite.jsonl"

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args)
        stats = {"epoch": epoch, **train_stats}

        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_stats = evaluate_ne(model, val_loader, val_dataset, device, args, print_prefix="Validation ne/Ti/Te metrics")
            stats.update({f"val_{k}": v for k, v in val_stats.items()})

            current_metric = val_stats["loss"] if args.best_metric == "val_loss" else val_stats.get(args.best_metric)
            if current_metric is None:
                raise KeyError(f"best_metric={args.best_metric} not found in validation stats: {list(val_stats.keys())}")
            improved = (
                np.isfinite(current_metric)
                and ((higher_better and current_metric > best_metric_value)
                     or ((not higher_better) and current_metric < best_metric_value))
            )
            if improved:
                best_metric_value = float(current_metric)
                save_ne_checkpoint(
                    output_dir / "checkpoint_best.pth",
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    args,
                    stats,
                    best_metric_value,
                    train_dataset,
                )
                direction = "higher" if higher_better else "lower"
                print(
                    f"New best {args.best_metric} ({direction} is better): "
                    f"{best_metric_value:.6f} at epoch {epoch}"
                )

        if epoch % args.save_ckpt_freq == 0 or epoch == args.epochs - 1:
            save_ne_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch:04d}.pth",
                model,
                optimizer,
                scaler,
                epoch,
                args,
                stats,
                best_metric_value,
                train_dataset,
            )

        with open(log_path, "a") as f:
            f.write(json.dumps(stats) + "\n")

    best_ckpt = output_dir / "checkpoint_best.pth"
    if best_ckpt.is_file():
        print(f"Loading best checkpoint for automatic testing: {best_ckpt}")
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        if "target_y_mean" in checkpoint and "target_y_std" in checkpoint:
            train_dataset.set_target_normalization(checkpoint["target_y_mean"], checkpoint["target_y_std"])
            val_dataset.set_target_normalization(checkpoint["target_y_mean"], checkpoint["target_y_std"])
            test_dataset.set_target_normalization(checkpoint["target_y_mean"], checkpoint["target_y_std"])
    else:
        print("Warning: no checkpoint_best.pth found; testing current final model.")

    run_test(model, test_loader, test_dataset, device, args, output_dir)

    print("XCS -> ne + Ti + Te one-shot fine-tuning and testing finished.")


if __name__ == "__main__":
    main()
