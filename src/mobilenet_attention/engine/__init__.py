"""训练与评估子包。"""

from .metrics import collect_predictions, compute_metrics
from .trainer import (
    JsonlLogger,
    build_scheduler,
    count_parameters,
    evaluate,
    load_checkpoint,
    measure_latency,
    model_size_mb,
    save_checkpoint,
    train_one_epoch,
)

__all__ = [
    "JsonlLogger",
    "build_scheduler",
    "collect_predictions",
    "compute_metrics",
    "count_parameters",
    "evaluate",
    "load_checkpoint",
    "measure_latency",
    "model_size_mb",
    "save_checkpoint",
    "train_one_epoch",
]
