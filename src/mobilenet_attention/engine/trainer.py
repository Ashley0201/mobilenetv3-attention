"""训练与验证循环。"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = False,
    grad_clip: float | None = None,
) -> dict[str, float]:
    """训练一个 epoch，返回平均损失与训练集准确率。"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = labels.numel()
        running_loss += float(loss.detach()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum())
        total += batch_size

    return {"loss": running_loss / max(total, 1), "accuracy": correct / max(total, 1)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    """在给定划分上评估，返回损失、指标与混淆矩阵。"""
    model.eval()
    total_loss = 0.0
    total = 0
    targets: list[int] = []
    predictions: list[int] = []

    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(criterion(logits, labels)) * labels.numel()
        total += labels.numel()
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = compute_metrics(targets, predictions, class_names)
    metrics["loss"] = total_loss / max(total, 1)
    return metrics


class JsonlLogger:
    """按行追加 JSON 日志，保留全部真实训练记录。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """保存可恢复训练的完整检查点（含模型、优化器、调度器与指标）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "metrics": metrics,
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
        },
        target,
    )
    return target


def load_checkpoint(path: str | Path, model: nn.Module, device: torch.device) -> dict[str, Any]:
    """加载权重用于推理；返回检查点中的元信息。"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def build_scheduler(
    optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int = 0, eta_min: float = 1e-6
) -> torch.optim.lr_scheduler.LRScheduler:
    """余弦退火调度；warmup 阶段线性升高学习率，末尾保留 eta_min 避免学习率为 0。"""
    if warmup_epochs > 0:
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=eta_min
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, main], milestones=[warmup_epochs]
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=eta_min
    )


def measure_latency(
    model: nn.Module,
    device: torch.device,
    input_size: int = 224,
    batch_size: int = 1,
    warmup: int = 10,
    repeats: int = 50,
) -> dict[str, float]:
    """测量模型前向延迟：预热 + CUDA 同步，返回毫秒与吞吐。"""
    model.eval()
    dummy = torch.randn(batch_size, 3, input_size, input_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    per_batch_ms = elapsed / repeats * 1000
    return {
        "batch_size": batch_size,
        "input_size": input_size,
        "warmup": warmup,
        "repeats": repeats,
        "latency_ms_per_batch": per_batch_ms,
        "latency_ms_per_image": per_batch_ms / batch_size,
        "throughput_images_per_s": batch_size * repeats / elapsed,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """按参数与缓冲区实际字节数估算，不包含 Python 对象开销。"""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_bytes / (1024**2)


def expected_steps_per_epoch(num_samples: int, batch_size: int) -> int:
    return math.ceil(num_samples / batch_size)
