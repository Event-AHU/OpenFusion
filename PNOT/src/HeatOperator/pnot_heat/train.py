from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .data import build_branch_feature, make_dataloaders, make_dataloaders_from_state, stats_from_checkpoint
from .model import HeatPNOT
from .utils import (
    CSVLogger,
    apply_overrides,
    count_parameters,
    ensure_dir,
    format_seconds,
    load_config,
    make_run_dir,
    resolve_path,
    save_json,
    save_yaml,
    set_seed,
    setup_logger,
)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def denormalize_target(tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return tensor * std + mean


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict):
    scheduler_cfg = config["training"]["scheduler"]
    if scheduler_cfg["name"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"],
            eta_min=scheduler_cfg["min_lr"],
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_cfg['name']}")


def align_cosine_lr(optimizer: torch.optim.Optimizer, config: Dict, completed_epochs: int) -> None:
    scheduler_cfg = config["training"]["scheduler"]
    if scheduler_cfg["name"] != "cosine" or completed_epochs <= 0:
        return
    total_epochs = max(int(config["training"]["epochs"]), 1)
    base_lr = float(config["training"]["lr"])
    min_lr = float(scheduler_cfg["min_lr"])
    progress = min(float(completed_epochs) / float(total_epochs), 1.0)
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr


def load_resume_payload(config: Dict, roots: List[Path], device: torch.device):
    resume_cfg = config.get("resume")
    if not resume_cfg or not resume_cfg.get("path"):
        return None, None

    ckpt_path = resolve_path(str(resume_cfg["path"]), roots)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return ckpt_path, ckpt


def load_resume_checkpoint(model: nn.Module, config: Dict, ckpt_path: Path | None, ckpt, logger) -> tuple[int, float]:
    if ckpt is None:
        return 0, float("inf")

    resume_cfg = config.get("resume", {})
    state = ckpt.get("model_state", ckpt)
    strict = bool(resume_cfg.get("strict", True))
    model.load_state_dict(state, strict=strict)

    start_epoch = int(ckpt.get("epoch", 0)) if bool(resume_cfg.get("continue_epoch", True)) else 0
    best_val_loss = float(ckpt.get("best_val_loss", float("inf"))) if start_epoch > 0 else float("inf")
    logger.info(
        "Loaded resume checkpoint: %s | start_epoch=%d | best_val_loss=%.8g | strict=%s",
        ckpt_path,
        start_epoch,
        best_val_loss,
        strict,
    )
    return start_epoch, best_val_loss


def graph_sobolev_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    coords: torch.Tensor,
    k: int,
    eps: float,
) -> torch.Tensor:
    if k <= 0 or pred.shape[1] <= 1:
        return pred.new_zeros(())

    num_nodes = pred.shape[1]
    k_eff = min(k, num_nodes - 1)
    dist = torch.cdist(coords, coords).clamp_min(eps)
    knn = dist.topk(k_eff + 1, dim=-1, largest=False).indices[:, :, 1:]
    gathered_dist = dist.gather(2, knn)

    pred_nei = pred.gather(1, knn.reshape(pred.shape[0], -1)).reshape(pred.shape[0], num_nodes, k_eff)
    target_nei = target.gather(1, knn.reshape(target.shape[0], -1)).reshape(target.shape[0], num_nodes, k_eff)
    pred_grad = (pred_nei - pred.unsqueeze(-1)) / gathered_dist
    target_grad = (target_nei - target.unsqueeze(-1)) / gathered_dist
    return torch.mean((pred_grad - target_grad) ** 2)


def get_loss_config(config: Dict) -> Dict:
    return config.get(
        "loss",
        {
            "sobolev_weight": 0.0,
            "sobolev_k": 4,
            "sobolev_eps": 1.0e-6,
        },
    )


def train_one_epoch(model: nn.Module, loader, optimizer, device: torch.device, grad_clip: float | None, loss_config: Dict) -> Dict[str, float]:
    model.train()
    criterion = nn.MSELoss()
    losses: List[float] = []
    mse_losses: List[float] = []
    sobolev_losses: List[float] = []
    sobolev_weight = float(loss_config.get("sobolev_weight", 0.0))
    sobolev_k = int(loss_config.get("sobolev_k", 4))
    sobolev_eps = float(loss_config.get("sobolev_eps", 1.0e-6))
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch["input"])
        mse_loss = criterion(pred, batch["target"])
        sobolev_loss = graph_sobolev_loss(pred, batch["target"], batch["coords"], sobolev_k, sobolev_eps)
        loss = mse_loss + sobolev_weight * sobolev_loss
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
        mse_losses.append(float(mse_loss.item()))
        sobolev_losses.append(float(sobolev_loss.item()))
    return {
        "loss": float(np.mean(losses)),
        "mse": float(np.mean(mse_losses)),
        "sobolev": float(np.mean(sobolev_losses)),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, target_mean: float, target_std: float) -> Dict[str, float]:
    model.eval()
    criterion = nn.MSELoss()
    losses: List[float] = []
    rel_l2_values: List[float] = []
    abs_err_sum = 0.0
    abs_target_sum = 0.0
    sq_err_sum = 0.0
    sq_target_sum = 0.0
    count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pred = model(batch["input"])
        loss = criterion(pred, batch["target"])
        pred_denorm = denormalize_target(pred, target_mean, target_std)
        target_denorm = batch["target_denorm"]
        err = pred_denorm - target_denorm

        losses.append(float(loss.item()))
        for sample_idx in range(err.shape[0]):
            rel_l2 = torch.linalg.norm(err[sample_idx].reshape(-1)) / (
                torch.linalg.norm(target_denorm[sample_idx].reshape(-1)) + 1.0e-8
            )
            rel_l2_values.append(float(rel_l2.item()))
        abs_err_sum += float(err.abs().sum().item())
        abs_target_sum += float(target_denorm.abs().sum().item())
        sq_err_sum += float((err ** 2).sum().item())
        sq_target_sum += float((target_denorm ** 2).sum().item())
        count += int(target_denorm.numel())

    return {
        "loss": float(np.mean(losses)),
        "rel_l2": float(np.mean(rel_l2_values)),
        "mae": abs_err_sum / max(count, 1),
        "rmae": abs_err_sum / (abs_target_sum + 1.0e-12),
        "rrmse": float(np.sqrt(sq_err_sum / (sq_target_sum + 1.0e-12))),
        "num_values": count,
    }


@torch.no_grad()
def predict_field_for_time(
    model: nn.Module,
    record: np.void,
    stats,
    positions: np.ndarray,
    time_index: int,
    device: torch.device,
    include_base_samples: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    pos_den = np.maximum(stats.pos_max - stats.pos_min, 1.0e-8)
    coords = ((positions - stats.pos_min) / pos_den).astype(np.float32)
    num_times = record["u1"].shape[-1]
    time_value = np.linspace(stats.time_min, stats.time_max, num_times, dtype=np.float32)[time_index]
    branch = build_branch_feature(record, include_base_samples)
    branch = ((branch - stats.branch_mean) / stats.branch_std).astype(np.float32)
    branch_rep = np.broadcast_to(branch[None, :], (positions.shape[0], branch.shape[0])).copy()
    model_input = np.concatenate(
        [coords, np.full((positions.shape[0], 1), time_value, dtype=np.float32), branch_rep],
        axis=1,
    )
    pred_norm = model(torch.from_numpy(model_input).float().to(device).unsqueeze(0)).squeeze(0)
    pred = (pred_norm.cpu().numpy() * stats.target_std + stats.target_mean).astype(np.float32)
    true = np.asarray(record["u1"][:, time_index], dtype=np.float32)
    rel_l2 = float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1.0e-8))
    return true, pred, rel_l2


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_val_loss: float, config: Dict, stats, splits):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
            "stats": {
                "branch_mean": stats.branch_mean.tolist(),
                "branch_std": stats.branch_std.tolist(),
                "target_mean": stats.target_mean,
                "target_std": stats.target_std,
                "pos_min": stats.pos_min.tolist(),
                "pos_max": stats.pos_max.tolist(),
                "time_min": stats.time_min,
                "time_max": stats.time_max,
            },
            "splits": {key: value.tolist() for key, value in splits.items()},
        },
        path,
    )


def plot_loss_curve(metrics_csv: Path, output_path: Path) -> None:
    data = np.genfromtxt(metrics_csv, delimiter=",", names=True)
    if data.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(data["epoch"], data["train_loss"], label="train loss")
    ax.plot(data["epoch"], data["val_loss"], label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("normalized MSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_field_triptych(positions: np.ndarray, true: np.ndarray, pred: np.ndarray, output_path: Path, title: str) -> None:
    err = pred - true
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, values, name in zip(axes, [true, pred, np.abs(err)], ["true", "pred", "abs error"]):
        sc = ax.scatter(positions[:, 0], positions[:, 1], c=values, s=6, cmap="viridis")
        ax.set_title(name)
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(path: Path, config: Dict, dataset_counts: Dict[str, int], metrics: Dict[str, float], run_dir: Path, best_epoch: int, parameter_count: int) -> None:
    report = f"""# HeatOperator PNOT 训练报告

## 1. 任务摘要

本次实现使用 PNOT 学习热边界条件到温度场 `u1(x, y, t)` 的映射，数据集与 DeepONet HeatOperator 版本一致。

## 2. 数据组织

- 数据文件: `{config["paths"]["dataset"]}`
- 空间坐标: `{config["paths"]["positions"]}`
- 训练样本数: {dataset_counts["train"]}
- 验证样本数: {dataset_counts["val"]}
- 测试样本数: {dataset_counts["test"]}
- 是否包含基础场样本: {config["data"]["include_base_samples"]}

## 3. 模型配置

- 模型: PNOT graph neural operator transformer
- trunk 输入维度: {config["model"]["trunk_size"]}
- branch 输入维度: {config["model"]["branch_size"]}
- hidden dim: {config["model"]["n_hidden"]}
- layers: {config["model"]["n_layers"]}
- heads: {config["model"]["n_head"]}
- experts: {config["model"]["n_experts"]}
- 参数量: {parameter_count}
- 训练 epoch: {config["training"]["epochs"]}
- batch size: {config["training"]["batch_size"]}
- Sobolev loss weight: {config.get("loss", {}).get("sobolev_weight", 0.0)}
- Sobolev kNN: {config.get("loss", {}).get("sobolev_k", 0)}

## 4. 最终指标

- best epoch: {best_epoch}
- test normalized MSE: {metrics["test_loss"]:.6f}
- test MAE: {metrics["test_mae"]:.6f}
- test rMAE: {metrics["test_rmae"]:.6f}
- test rRMSE: {metrics["test_rrmse"]:.6f}
- test relative L2: {metrics["test_rel_l2"]:.6f}
- val relative L2(best checkpoint): {metrics["val_rel_l2"]:.6f}

## 5. 产物位置

- 日志: `{run_dir / "logs" / "train.log"}`
- 指标表: `{run_dir / "metrics" / "epoch_metrics.csv"}`
- checkpoint: `{run_dir / "checkpoints"}`
- 图片: `{run_dir / "figures"}`
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = apply_overrides(load_config(str(config_path)), args.override)
    module_root = Path(__file__).resolve().parents[1]
    pnot_root = Path(__file__).resolve().parents[2]
    baseline_root = Path(__file__).resolve().parents[3]
    operator_root = Path(__file__).resolve().parents[4]
    roots = [Path.cwd(), module_root, pnot_root, baseline_root, operator_root]
    config["paths"]["dataset"] = str(resolve_path(config["paths"]["dataset"], roots))
    config["paths"]["positions"] = str(resolve_path(config["paths"]["positions"], roots))

    output_root = Path(config["paths"]["output_root"])
    if not output_root.is_absolute():
        output_root = pnot_root / output_root
    config["paths"]["output_root"] = str(output_root)

    set_seed(config["seed"])
    run_dir = make_run_dir(config["paths"]["output_root"], config["run_name"])
    logs_dir = ensure_dir(run_dir / "logs")
    metrics_dir = ensure_dir(run_dir / "metrics")
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    figures_dir = ensure_dir(run_dir / "figures")
    logger = setup_logger(logs_dir / "train.log")
    save_yaml(run_dir / "config_resolved.yaml", config)

    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.info("CUDA requested but unavailable; falling back to CPU")
        device = torch.device("cpu")

    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s", device)
    start_time = time.time()

    resume_path, resume_ckpt = load_resume_payload(config, roots, device)
    if resume_ckpt is not None and bool(config.get("resume", {}).get("use_checkpoint_data_state", True)):
        stats = stats_from_checkpoint(resume_ckpt)
        splits = {key: np.asarray(value, dtype=np.int64) for key, value in resume_ckpt["splits"].items()}
        loaders, splits, stats = make_dataloaders_from_state(config, config["seed"], splits, stats)
        logger.info("Using data stats and splits from resume checkpoint: %s", resume_path)
    else:
        loaders, splits, stats = make_dataloaders(config, seed=config["seed"])
    dataset_counts = {key: len(value) for key, value in splits.items()}
    model = HeatPNOT(**config["model"]).to(device)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
    scheduler = create_scheduler(optimizer, config)
    start_epoch, best_val_loss = load_resume_checkpoint(model, config, resume_path, resume_ckpt, logger)
    align_cosine_lr(optimizer, config, start_epoch)
    logger.info("Model parameters: %d", parameter_count)
    logger.info("Dataset split sizes | train=%d val=%d test=%d", dataset_counts["train"], dataset_counts["val"], dataset_counts["test"])
    loss_config = get_loss_config(config)
    logger.info(
        "Loss config | sobolev_weight=%.6g sobolev_k=%d sobolev_eps=%.3g",
        float(loss_config.get("sobolev_weight", 0.0)),
        int(loss_config.get("sobolev_k", 4)),
        float(loss_config.get("sobolev_eps", 1.0e-6)),
    )

    csv_logger = CSVLogger(
        metrics_dir / "epoch_metrics.csv",
        [
            "epoch",
            "lr",
            "train_loss",
            "train_mse_loss",
            "train_sobolev_loss",
            "val_loss",
            "val_rel_l2",
            "val_mae",
            "val_rmae",
            "val_rrmse",
        ],
    )

    best_epoch = start_epoch
    if start_epoch > 0:
        save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, scheduler, start_epoch, best_val_loss, config, stats, splits)
        save_checkpoint(checkpoints_dir / "last.pt", model, optimizer, scheduler, start_epoch, best_val_loss, config, stats, splits)
    for epoch in range(start_epoch + 1, config["training"]["epochs"] + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, device, config["training"]["grad_clip"], loss_config)
        val_metrics = evaluate(model, loaders["val"], device, stats.target_mean, stats.target_std)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_mse_loss": train_metrics["mse"],
            "train_sobolev_loss": train_metrics["sobolev"],
            "val_loss": val_metrics["loss"],
            "val_rel_l2": val_metrics["rel_l2"],
            "val_mae": val_metrics["mae"],
            "val_rmae": val_metrics["rmae"],
            "val_rrmse": val_metrics["rrmse"],
        }
        csv_logger.log(row)
        logger.info(
            "Epoch %03d | train_loss=%.6f | train_mse=%.6f | train_sobolev=%.6f | val_loss=%.6f | val_rel_l2=%.6f | val_mae=%.6f | time=%s",
            epoch,
            train_metrics["loss"],
            train_metrics["mse"],
            train_metrics["sobolev"],
            val_metrics["loss"],
            val_metrics["rel_l2"],
            val_metrics["mae"],
            format_seconds(time.time() - epoch_start),
        )

        save_checkpoint(checkpoints_dir / "last.pt", model, optimizer, scheduler, epoch, best_val_loss, config, stats, splits)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, scheduler, epoch, best_val_loss, config, stats, splits)

    plot_loss_curve(metrics_dir / "epoch_metrics.csv", figures_dir / "loss_curve.png")
    best_ckpt = torch.load(checkpoints_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    val_metrics = evaluate(model, loaders["val"], device, stats.target_mean, stats.target_std)
    test_metrics = evaluate(model, loaders["test"], device, stats.target_mean, stats.target_std)
    summary_metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val_loss": val_metrics["loss"],
        "val_rel_l2": val_metrics["rel_l2"],
        "val_mae": val_metrics["mae"],
        "val_rmae": val_metrics["rmae"],
        "val_rrmse": val_metrics["rrmse"],
        "test_loss": test_metrics["loss"],
        "test_rel_l2": test_metrics["rel_l2"],
        "test_mae": test_metrics["mae"],
        "test_rmae": test_metrics["rmae"],
        "test_rrmse": test_metrics["rrmse"],
        "test_num_values": test_metrics["num_values"],
        "elapsed_seconds": time.time() - start_time,
    }
    save_json(metrics_dir / "summary.json", summary_metrics)

    dataset = np.load(config["paths"]["dataset"], mmap_mode="r")
    positions = np.load(config["paths"]["positions"]).astype(np.float32)
    viz_count = min(config["data"]["num_visualization_samples"], len(splits["test"]))
    for i in range(viz_count):
        sample_index = int(splits["test"][i])
        record = dataset[sample_index]
        for time_index in config["data"]["viz_time_indices"]:
            true, pred, rel_l2 = predict_field_for_time(
                model=model,
                record=record,
                stats=stats,
                positions=positions,
                time_index=int(time_index),
                device=device,
                include_base_samples=config["data"]["include_base_samples"],
            )
            sample_name = f"sample_{sample_index}_k{float(record['kheat']):.0f}_yi_{0.0 if np.isnan(record['yi']) else float(record['yi']):.0f}"
            plot_field_triptych(
                positions=positions,
                true=true,
                pred=pred,
                output_path=figures_dir / f"{sample_name}_t{time_index}.png",
                title=f"{sample_name} t={time_index} rel_l2={rel_l2:.4f}",
            )

    write_report(run_dir / "report.md", config, dataset_counts, summary_metrics, run_dir, best_epoch, parameter_count)
    logger.info("Training complete. Best epoch: %d", best_epoch)
    logger.info(
        "Final metrics | val_rel_l2=%.6f | test_rel_l2=%.6f | test_mae=%.6f | elapsed=%s",
        summary_metrics["val_rel_l2"],
        summary_metrics["test_rel_l2"],
        summary_metrics["test_mae"],
        format_seconds(summary_metrics["elapsed_seconds"]),
    )


if __name__ == "__main__":
    main()
