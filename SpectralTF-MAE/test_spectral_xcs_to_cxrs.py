# Copyright (c) EPFL VILAB.
# All rights reserved.
#
# Downstream testing: complete xcs spectrum -> cxrs peak-region evaluation.
#
# This script uses the complete xcs spectrum without masking and reports loss,
# errors, R2, and visualizations only on the cxrs fixed peak interval. It evaluates
# finetune_spectral_xcs_to_cxrs.py.

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml

from utils.datasets import build_spectral_pretraining_dataset
from utils.shared_checkpoint import validate_shared_geometry
from utils.spectral_downstream_common import (
    build_full_spectrum_inputs,
    finalize_downstream_peak_metrics,
    init_downstream_metric_accumulators,
    load_model_weights_only,
    save_metrics_json,
    update_downstream_peak_metrics,
)
from utils.spectral_pretraining_common import (
    FIXED_R2_RANGES,
    _save_peak_reconstruction_heatmap,
    _save_peak_spectral_lines,
    crop_to_original,
    get_domain_specs,
    get_model,
    move_batch_to_device,
    resolve_device,
)


def get_args():
    config_parser = argparse.ArgumentParser(description="Testing Config", add_help=False)
    config_parser.add_argument(
        "-c", "--config", default="", type=str, metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser(
        "Spectral downstream testing: complete xcs -> cxrs peak evaluation",
        add_help=True,
    )

    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], type=str)
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
    )
    parser.add_argument("--spectral_normalize_eps", default=1e-6, type=float)

    parser.add_argument(
        "--checkpoint", default="", type=str,
        help="Path to a user-provided fine-tuned checkpoint.",
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
    parser.add_argument("--fp32_output_adapters", type=str, default="")

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_test_batches", default=-1, type=int)
    parser.add_argument("--output_dir", default="./output/test/xcs_to_cxrs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--save_visualizations", default=True, action="store_true")
    parser.add_argument("--no_save_visualizations", action="store_false", dest="save_visualizations")
    parser.add_argument("--vis_max_samples", default=8, type=int)
    parser.add_argument("--save_predictions", default=False, action="store_true")

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    if not isinstance(args.data_path, str) or not args.data_path.strip():
        parser.error("Dataset path is empty. Set --data_path or data_path in the YAML config.")
    return args


@torch.no_grad()
def test_downstream(model, data_loader, device, args, domain_specs):
    model.eval()
    accumulators = init_downstream_metric_accumulators(args.out_domains)

    saved_prediction_batches = []

    for batch_idx, batch in enumerate(data_loader):
        if args.max_test_batches > 0 and batch_idx >= args.max_test_batches:
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

        if args.save_predictions:
            batch_save = {}
            for task in args.out_domains:
                if task not in preds or task not in tasks_dict:
                    continue
                original_size = domain_specs[task]["original_size"]
                batch_save[f"{task}_pred"] = crop_to_original(
                    preds[task].detach().cpu(), original_size
                ).numpy()
                batch_save[f"{task}_target"] = crop_to_original(
                    tasks_dict[task].detach().cpu(), original_size
                ).numpy()
            saved_prediction_batches.append(batch_save)

    stats = finalize_downstream_peak_metrics(accumulators, args.out_domains)
    return stats, saved_prediction_batches


@torch.no_grad()
def visualize_test_results(model, data_loader, device, output_dir, args, domain_specs):
    model.eval()
    vis_root = Path(output_dir) / "visualizations" / args.split
    vis_root.mkdir(parents=True, exist_ok=True)

    saved = 0
    for batch in data_loader:
        if saved >= args.vis_max_samples:
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

        batch_size = next(iter(tasks_dict.values())).shape[0]

        for local_idx in range(batch_size):
            if saved >= args.vis_max_samples:
                break

            for task in args.out_domains:
                if task not in tasks_dict or task not in preds or task not in FIXED_R2_RANGES:
                    continue

                original_size = domain_specs[task]["original_size"]
                raw_target_full = tasks_dict[task].float()

                sample_dir = vis_root / task / f"sample_{saved:04d}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                target = crop_to_original(
                    raw_target_full[local_idx, 0].detach().cpu(), original_size
                ).numpy()
                pred = crop_to_original(
                    preds[task][local_idx, 0].detach().cpu(), original_size
                ).numpy()

                _save_peak_reconstruction_heatmap(
                    target=target,
                    pred=pred,
                    task=task,
                    epoch=0,
                    sample_idx=saved,
                    save_path=sample_dir / "01_fixed_peak_heatmap.png",
                )
                _save_peak_spectral_lines(
                    target=target,
                    pred=pred,
                    task=task,
                    epoch=0,
                    sample_idx=saved,
                    save_path=sample_dir / "02_fixed_peak_spectral_lines.png",
                )

            saved += 1


def save_prediction_npz(output_dir, prediction_batches):
    if len(prediction_batches) == 0:
        return

    merged = {}
    keys = prediction_batches[0].keys()
    for key in keys:
        merged[key] = np.concatenate([batch[key] for batch in prediction_batches if key in batch], axis=0)

    path = Path(output_dir) / "predictions_fixed_peak.npz"
    np.savez_compressed(path, **merged)
    print(f"Saved predictions to: {path}")


def main():
    args = get_args()
    args.in_domains = args.in_domains.split("-") if isinstance(args.in_domains, str) else args.in_domains
    args.out_domains = args.out_domains.split("-") if isinstance(args.out_domains, str) else args.out_domains
    args.all_domains = list(dict.fromkeys(args.in_domains + args.out_domains))
    args.fp32_output_adapters = [x for x in args.fp32_output_adapters.split("-") if x]
    args.max_val_batches = args.max_test_batches

    if (args.in_domains, args.out_domains) not in [(["xcs"], ["cxrs"]), (["cxrs"], ["xcs"])]:
        raise ValueError(f"Supported spectral tasks are xcs->cxrs and cxrs->xcs, got {args.in_domains}->{args.out_domains}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    if not args.checkpoint:
        raise ValueError("Testing requires --checkpoint, usually downstream checkpoint_best.pth.")
    load_model_weights_only(model, args.checkpoint, device=device, prefix="test checkpoint")

    dataset_test = build_spectral_pretraining_dataset(args, split=args.split)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    stats, prediction_batches = test_downstream(model, data_loader_test, device, args, domain_specs)

    print(
        f"Test fixed-peak metrics on split={args.split}: "
        + " ".join([
            f"{k}={v:.6f}" if np.isfinite(v) else f"{k}=nan"
            for k, v in stats.items()
        ])
    )

    metrics_path = output_dir / f"metrics_{args.split}_fixed_peak.json"
    save_metrics_json(metrics_path, stats)
    print(f"Saved metrics to: {metrics_path}")

    if args.save_predictions:
        save_prediction_npz(output_dir, prediction_batches)

    if args.save_visualizations:
        visualize_test_results(model, data_loader_test, device, output_dir, args, domain_specs)

    with open(output_dir / "test_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Testing finished.")


if __name__ == "__main__":
    main()
