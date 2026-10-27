# Copyright (c) EPFL VILAB.
# All rights reserved.
#
# Full-spectrum XCS -> CXRS fine-tuning with fixed-peak supervision.
# Input shape: [B, 1, 35, 490]; output shape: [B, 1, 12, 1024].
# Inference uses every input patch without masking.

import argparse
import datetime
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml

from utils.datasets import build_spectral_pretraining_dataset
from utils.shared_checkpoint import load_shared_pretraining_weights, validate_shared_geometry
from utils.optim_factory import create_optimizer
from utils.spectral_downstream_common import (
    build_full_spectrum_inputs,
    finalize_downstream_peak_metrics,
    init_downstream_metric_accumulators,
    update_downstream_peak_metrics,
)
from utils.spectral_pretraining_common import (
    FIXED_R2_RANGES,
    _save_peak_reconstruction_heatmap,
    _save_peak_spectral_lines,
    compute_fixed_peak_mse_loss,
    cosine_lr,
    crop_to_original,
    get_domain_specs,
    get_model,
    load_checkpoint,
    move_batch_to_device,
    save_checkpoint,
    resolve_device,
    set_optimizer_lr,
)


def get_args():
    config_parser = argparse.ArgumentParser(description="Fine-tuning Config", add_help=False)
    config_parser.add_argument(
        "-c", "--config", default="", type=str, metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser(
        "Spectral downstream fine-tuning: complete xcs -> cxrs peak supervision",
        add_help=True,
    )

    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--in_domains", default="xcs", type=str)
    parser.add_argument("--out_domains", default="cxrs", type=str)
    parser.add_argument("--allow_missing_modalities", action="store_true")
    parser.add_argument(
        "--spectral_normalize",
        default="minmax",
        choices=[
            "peak_ref_minmax",
            "per_wavelength_robust", "per_wavelength", "minmax", "zscore", "none",
        ],
        help=(
            "peak_ref_minmax uses fixed-peak min/max to scale the complete "
            "spectrum without clipping non-peak values."
        ),
    )
    parser.add_argument("--spectral_normalize_eps", default=1e-6, type=float)
    parser.add_argument(
        "--pretrained_ckpt", default="", type=str,
        help="Pretraining checkpoint used to initialize downstream fine-tuning.",
    )
    parser.add_argument(
        "--best_metric", default="cxrs_peak_r2",
        choices=["cxrs_peak_r2", "xcs_peak_r2", "mean_peak_r2", "cxrs_peak_mse", "cxrs_peak_rmse", "cxrs_peak_mae", "xcs_peak_mse", "xcs_peak_rmse", "xcs_peak_mae"],
        help="Metric used to save checkpoint_best.pth. R2 is maximized; error metrics are minimized.",
    )



    # Spectral shapes must match upstream checkpoints.
    parser.add_argument("--cxrs_input_size", default=[12, 1024], type=int, nargs=2)
    parser.add_argument("--cxrs_original_size", default=[12, 1024], type=int, nargs=2)
    parser.add_argument("--cxrs_patch_size", default=[3, 16], type=int, nargs=2)
    parser.add_argument("--xcs_input_size", default=[35, 490], type=int, nargs=2)
    parser.add_argument("--xcs_original_size", default=[35, 487], type=int, nargs=2)
    parser.add_argument("--xcs_patch_size", default=[5, 14], type=int, nargs=2)
    parser.add_argument("--pad_xcs_width_to", default=490, type=int)

    parser.add_argument("--model", default="pretrain_SpectralTF_base", type=str)
    parser.add_argument("--num_global_tokens", default=1, type=int)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--decoder_use_task_queries", default=True, action="store_true")
    parser.add_argument("--decoder_use_xattn", default=True, action="store_true")
    parser.add_argument(
        "--decoder_context_tasks",
        default="xcs-cxrs",
        type=str,
        help=(
            "Task embeddings used by output adapters. Keep xcs-cxrs for better "
            "reuse of pretraining checkpoints, while actual input remains xcs only."
        ),
    )
    parser.add_argument("--decoder_dim", default=256, type=int)
    parser.add_argument("--decoder_depth", default=2, type=int)
    parser.add_argument("--decoder_num_heads", default=8, type=int)
    parser.add_argument("--drop_path", type=float, default=0.0)
    parser.add_argument("--loss_on_unmasked", default=True, action="store_true")
    parser.add_argument("--fp32_output_adapters", type=str, default="")

    parser.add_argument("--freeze_encoder", default=False, action="store_true")
    parser.add_argument("--freeze_input_adapters", default=False, action="store_true")
    parser.add_argument("--freeze_output_adapters", default=False, action="store_true")

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

    parser.add_argument("--output_dir", default="./output/finetune/xcs_to_cxrs")
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


def is_higher_better(metric_name):
    return metric_name.endswith("_r2")


def maybe_freeze_model_parts(model, args):
    if args.freeze_encoder and hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("Frozen encoder parameters.")

    if args.freeze_input_adapters and hasattr(model, "input_adapters"):
        for p in model.input_adapters.parameters():
            p.requires_grad = False
        print("Frozen input adapter parameters.")

    if args.freeze_output_adapters and hasattr(model, "output_adapters"):
        for p in model.output_adapters.parameters():
            p.requires_grad = False
        print("Frozen output adapter parameters.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")


def train_one_epoch(model, data_loader, optimizer, scaler, device, epoch, args, domain_specs):
    model.train()
    amp_enabled = device.type == "cuda"
    total_loss = 0.0
    total_steps = 0
    start = time.time()

    for step, (x, _) in enumerate(data_loader):
        epoch_float = epoch + step / max(1, len(data_loader))
        lr = cosine_lr(
            args.lr, args.min_lr, epoch_float, args.epochs,
            warmup_epochs=args.warmup_epochs, warmup_lr=args.warmup_lr,
        )
        set_optimizer_lr(optimizer, lr)

        tasks_dict = move_batch_to_device(x, device)
        input_dict, task_masks = build_full_spectrum_inputs(tasks_dict, args, domain_specs)

        optimizer.zero_grad(set_to_none=True)
        autocast_ctx = torch.cuda.amp.autocast if amp_enabled else nullcontext
        with autocast_ctx():
            preds, masks = model(
                input_dict,
                mask_inputs=False,
                task_masks=task_masks,
                fp32_output_adapters=args.fp32_output_adapters,
            )

            losses = []
            loss_logs = {}
            for task in args.out_domains:
                if task not in preds:
                    continue

                target = tasks_dict[task].float()
                # Loss is computed on the raw normalized spectrum.
                peak_loss = compute_fixed_peak_mse_loss(
                    pred_full=preds[task].float(),
                    target_full=target,
                    mask=None,
                    task=task,
                    args=args,
                    domain_specs=domain_specs,
                )
                losses.append(peak_loss)
                loss_logs[f"{task}_peak_loss"] = peak_loss.detach().item()

            loss = sum(losses)

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

        total_loss += loss.item()
        total_steps += 1

        if step % args.print_freq == 0:
            parts = " ".join([f"{k}={v:.6f}" for k, v in loss_logs.items()])
            print(
                f"Epoch [{epoch}/{args.epochs}] step [{step}/{len(data_loader)}] "
                f"lr={lr:.6e} loss={loss.item():.6f} {parts}"
            )

    avg_loss = total_loss / max(total_steps, 1)
    elapsed = time.time() - start
    print(
        f"Epoch {epoch} training: loss={avg_loss:.6f}, "
        f"time={str(datetime.timedelta(seconds=int(elapsed)))}"
    )
    return {"train_loss": avg_loss}


@torch.no_grad()
def evaluate_downstream(model, data_loader, device, args, domain_specs, print_prefix="Validation"):
    model.eval()
    accumulators = init_downstream_metric_accumulators(args.out_domains)

    for batch_idx, batch in enumerate(data_loader):
        if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
            break

        sample_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
        tasks_dict = move_batch_to_device(sample_batch, device)
        input_dict, task_masks = build_full_spectrum_inputs(tasks_dict, args, domain_specs)

        preds, masks = model(
            input_dict,
            mask_inputs=False,
            task_masks=task_masks,
            fp32_output_adapters=args.fp32_output_adapters,
        )
        update_downstream_peak_metrics(accumulators, preds, tasks_dict, args, domain_specs)

    stats = finalize_downstream_peak_metrics(accumulators, args.out_domains)
    print(
        print_prefix + ": "
        + " ".join([
            f"{k}={v:.6f}" if np.isfinite(v) else f"{k}=nan"
            for k, v in stats.items()
        ])
    )
    return stats


@torch.no_grad()
def visualize_downstream_results(model, data_loader, device, epoch, output_dir, args, domain_specs):
    model.eval()

    try:
        sample_batch, _ = next(iter(data_loader))
    except StopIteration:
        return

    tasks_dict = move_batch_to_device(sample_batch, device)
    input_dict, task_masks = build_full_spectrum_inputs(tasks_dict, args, domain_specs)

    preds, masks = model(
        input_dict,
        mask_inputs=False,
        task_masks=task_masks,
        fp32_output_adapters=args.fp32_output_adapters,
    )

    vis_root = Path(output_dir) / "visualizations" / f"epoch_{epoch:04d}"
    vis_root.mkdir(parents=True, exist_ok=True)

    batch_size = next(iter(tasks_dict.values())).shape[0]
    num_samples = min(getattr(args, "vis_max_samples", 2), batch_size)

    for task in args.out_domains:
        if task not in tasks_dict or task not in preds or task not in FIXED_R2_RANGES:
            continue

        original_size = domain_specs[task]["original_size"]
        raw_target_full = tasks_dict[task].float()
        for sample_idx in range(num_samples):
            sample_dir = vis_root / task / f"sample_{sample_idx:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            target = crop_to_original(raw_target_full[sample_idx, 0].detach().cpu(), original_size).numpy()
            pred = crop_to_original(preds[task][sample_idx, 0].detach().cpu(), original_size).numpy()

            _save_peak_reconstruction_heatmap(
                target=target,
                pred=pred,
                task=task,
                epoch=epoch,
                sample_idx=sample_idx,
                save_path=sample_dir / "01_fixed_peak_heatmap.png",
            )
            _save_peak_spectral_lines(
                target=target, pred=pred, task=task, epoch=epoch,
                sample_idx=sample_idx,
                save_path=sample_dir / "02_fixed_peak_spectral_lines.png",
            )


def main():
    args = get_args()
    args.in_domains = args.in_domains.split("-") if isinstance(args.in_domains, str) else args.in_domains
    args.out_domains = args.out_domains.split("-") if isinstance(args.out_domains, str) else args.out_domains
    args.all_domains = list(dict.fromkeys(args.in_domains + args.out_domains))
    args.fp32_output_adapters = [x for x in args.fp32_output_adapters.split("-") if x]

    if (args.in_domains, args.out_domains) not in [(["xcs"], ["cxrs"]), (["cxrs"], ["xcs"])]:
        raise ValueError(f"Supported spectral tasks are xcs->cxrs and cxrs->xcs, got {args.in_domains}->{args.out_domains}")

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
        print(f"Using CUDA device: {device}, current_device={torch.cuda.current_device()}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    domain_specs = get_domain_specs(args, args.all_domains)
    validate_shared_geometry(domain_specs)
    model = get_model(args, domain_specs).to(device)

    if not args.resume:
        if not args.pretrained_ckpt:
            raise ValueError(
                "Fine-tuning requires --pretrained_ckpt, usually the pretraining checkpoint_best.pth."
            )
        load_shared_pretraining_weights(
            model, args.pretrained_ckpt, device=device, expected_model_name=args.model
        )

    maybe_freeze_model_parts(model, args)

    with open(output_dir / "model.txt", "w") as f:
        f.write(str(model))

    optimizer = create_optimizer(args, model)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    higher_better = is_higher_better(args.best_metric)
    best_metric_value = -float("inf") if higher_better else float("inf")

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer=optimizer, scaler=scaler, device=device)
        args.start_epoch = int(checkpoint.get("epoch", args.start_epoch - 1)) + 1
        best_metric_value = float(
            checkpoint.get("best_metric_value", checkpoint.get("best_r2", best_metric_value))
        )
        print(
            f"Resumed from {args.resume} at epoch {args.start_epoch}, "
            f"best_metric_value={best_metric_value:.6f}, current_best_metric={args.best_metric}"
        )

    dataset_train = build_spectral_pretraining_dataset(args, split="train")
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

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

    print(f"Start downstream fine-tuning for {args.epochs} epochs on {device}.")
    log_path = output_dir / f"log_spectral_finetuning_{args.in_domains[0]}_to_{args.out_domains[0]}.jsonl"

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, scaler, device, epoch, args, domain_specs
        )
        stats = {"epoch": epoch, **train_stats}

        if data_loader_val is not None and (epoch % args.eval_freq == 0 or epoch == args.epochs - 1):
            val_stats = evaluate_downstream(
                model=model,
                data_loader=data_loader_val,
                device=device,
                args=args,
                domain_specs=domain_specs,
                print_prefix="Validation fixed-peak metrics",
            )
            stats.update({f"val_{k}": v for k, v in val_stats.items()})

            current_metric = val_stats.get(args.best_metric)
            if current_metric is None:
                raise KeyError(
                    f"best_metric={args.best_metric} not found. Available metrics: {list(val_stats.keys())}"
                )

            improved = (
                np.isfinite(current_metric)
                and ((higher_better and current_metric > best_metric_value)
                     or ((not higher_better) and current_metric < best_metric_value))
            )
            if improved:
                best_metric_value = float(current_metric)
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
                direction = "higher" if higher_better else "lower"
                print(
                    f"New best {args.best_metric} ({direction} is better): "
                    f"{best_metric_value:.6f} at epoch {epoch}"
                )

            if epoch % args.vis_freq == 0 or epoch == args.epochs - 1:
                visualize_downstream_results(
                    model=model,
                    data_loader=data_loader_val,
                    device=device,
                    epoch=epoch,
                    output_dir=args.output_dir,
                    args=args,
                    domain_specs=domain_specs,
                )

        if epoch % args.save_ckpt_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch:04d}.pth",
                model,
                optimizer,
                scaler,
                epoch,
                args,
                stats,
                best_metric_value,
            )

        with open(log_path, "a") as f:
            f.write(json.dumps(stats) + "\n")

    print("Fine-tuning finished.")


if __name__ == "__main__":
    main()
