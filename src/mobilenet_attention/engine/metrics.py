"""分类评估指标与推理统计。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader


def compute_metrics(
    targets: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """计算 Accuracy / Precision / Recall / F1（macro）与混淆矩阵。"""
    y_true = np.asarray(targets)
    y_pred = np.asarray(predictions)
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "num_samples": int(y_true.size),
    }


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], float]:
    """返回真实标签、预测标签与平均交叉熵损失；推理不更新梯度。"""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    targets: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(criterion(logits, labels))
        total_samples += labels.numel()
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())

    return targets, predictions, total_loss / max(total_samples, 1)
