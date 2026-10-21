"""MobileNetV3-Large 基线训练脚本。

用法：
    python scripts/train.py --config configs/baseline_mobilenetv3_large.yaml
    python scripts/train.py --config configs/smoke.yaml --set training.epochs=1

输出目录（每个实验 ID 唯一，不覆盖已有结果）：
    outputs/<experiment_id>/
        config_snapshot.yaml   实际生效配置
        environment.json       软硬件与 Git 提交
        train_log.jsonl        逐 epoch 真实日志
        best.pt                验证集最优检查点
        last.pt                最后一轮检查点（支持恢复）
        val_metrics.json       最优轮次的验证指标
        test_metrics.json      独立测试集指标（仅在训练结束后评估一次）
        latency.json           模型前向延迟
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mobilenet_attention.data import (  # noqa: E402
    build_cifar10_datasets,
    build_dataloaders,
    dataset_summary,
)
from mobilenet_attention.engine import (  # noqa: E402
    JsonlLogger,
    build_scheduler,
    count_parameters,
    evaluate,
    measure_latency,
    model_size_mb,
    save_checkpoint,
    train_one_epoch,
)
from mobilenet_attention.models import build_mobilenet_v3_large  # noqa: E402
from mobilenet_attention.utils.config import (  # noqa: E402
    apply_overrides,
    load_config,
    save_config_snapshot,
)
from mobilenet_attention.utils.env import environment_snapshot  # noqa: E402
from mobilenet_attention.utils.seed import resolve_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 MobileNetV3-Large 基线")
    parser.add_argument("--config", required=True, help="训练配置 YAML 路径")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖配置项，例如 --set training.epochs=3",
    )
    parser.add_argument("--resume", default=None, help="从 last.pt 恢复训练")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许写入已存在的实验目录（默认拒绝，避免覆盖旧结果）",
    )
    return parser.parse_args()


def parse_overrides(pairs: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set 参数格式应为 KEY=VALUE，收到：{pair}")
        key, raw = pair.split("=", 1)
        overrides[key.strip()] = raw.strip()
    return overrides


def prepare_output_dir(root: Path, experiment_id: str, overwrite: bool) -> Path:
    """实验目录必须唯一；已存在时默认拒绝，防止覆盖历史证据。"""
    target = root / experiment_id
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise SystemExit(
            f"实验目录已存在且非空：{target}\n"
            "请修改 experiment.id，或显式使用 --overwrite（会覆盖旧结果）。"
        )
    target.mkdir(parents=True, exist_ok=True)
    return target


class _LimitedLoader:
    """仅用于冒烟测试：截断 DataLoader 的迭代批次数，不改变数据划分。"""

    def __init__(self, loader, limit: int) -> None:
        self.loader = loader
        self.limit = limit
        self.dataset = getattr(loader, "dataset", None)
        self.batch_size = getattr(loader, "batch_size", None)

    def __len__(self) -> int:
        return min(len(self.loader), self.limit)

    def __iter__(self):
        for index, batch in enumerate(self.loader):
            if index >= self.limit:
                break
            yield batch


def resolve_data_config(training_config: dict) -> dict:
    """数据集配置与训练配置分离，路径相对项目根目录。"""
    data_path = Path(training_config["data"]["config"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    dataset_config = load_config(data_path)
    dataset_section = dataset_config["dataset"]
    if not Path(dataset_section["root"]).is_absolute():
        dataset_section["root"] = str(PROJECT_ROOT / dataset_section["root"])
    return dataset_section


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = load_config(config_path)
    config = apply_overrides(config, parse_overrides(args.set))
    dataset_config = resolve_data_config(config)
    config["dataset"] = {k: v for k, v in dataset_config.items() if not k.startswith("_")}

    exp = config["experiment"]
    seed = int(exp["seed"])
    set_seed(seed, deterministic=bool(exp.get("deterministic", True)))
    device = resolve_device(config["training"].get("device", "auto"))
    print(f"设备：{device}")

    output_root = PROJECT_ROOT / exp["output_root"]
    out_dir = prepare_output_dir(output_root, exp["id"], args.overwrite)

    started_at = datetime.now(UTC).isoformat()
    env = environment_snapshot(PROJECT_ROOT)
    save_config_snapshot(config, out_dir / "config_snapshot.yaml")
    (out_dir / "environment.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger = JsonlLogger(out_dir / "train_log.jsonl")
    logger.log({"event": "start", "time": started_at, "environment": env, "device": str(device)})

    datasets = build_cifar10_datasets(
        root=dataset_config["root"],
        input_size=int(dataset_config["input_size"]),
        val_size=int(dataset_config["split"]["val_size"]),
        split_seed=int(dataset_config["split"]["seed"]),
        download=bool(dataset_config.get("download", False)),
        augmentation=dataset_config.get("augmentation", {}),
    )
    summary = dataset_summary(datasets)
    logger.log({"event": "dataset", **summary})
    print(
        f"数据划分：train={summary['train_size']} val={summary['val_size']} "
        f"test={summary['test_size']}"
    )

    loaders = build_dataloaders(
        datasets,
        batch_size=int(config["data"]["batch_size"]),
        eval_batch_size=int(config["data"]["eval_batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"]["pin_memory"]),
    )
    # 冒烟测试可限制批次数；正式配置不设置该字段，即使用完整数据
    limit_train = config["training"].get("limit_train_batches")
    limit_val = config["evaluation"].get("limit_val_batches")
    if limit_train:
        loaders["train"] = _LimitedLoader(loaders["train"], int(limit_train))
        logger.log({"event": "limit", "train_batches": int(limit_train)})
    if limit_val:
        loaders["val"] = _LimitedLoader(loaders["val"], int(limit_val))

    model = build_mobilenet_v3_large(
        num_classes=int(config["model"]["num_classes"]),
        pretrained=config["model"]["pretrained"],
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    print(f"模型参数：{count_parameters(model):,}，预训练：{config['model']['pretrained']}")
    logger.log(
        {
            "event": "model",
            "parameters": count_parameters(model),
            "size_mb": model_size_mb(model),
            "pretrained": config["model"]["pretrained"],
        }
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["training"]["label_smoothing"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = build_scheduler(
        optimizer,
        epochs=int(config["training"]["epochs"]),
        warmup_epochs=int(config["training"]["warmup_epochs"]),
    )
    amp = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch = 0
    best_val_accuracy = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_accuracy = float(checkpoint.get("metrics", {}).get("val_accuracy", -1.0))
        logger.log({"event": "resume", "from": str(args.resume), "start_epoch": start_epoch})

    epochs = int(config["training"]["epochs"])
    grad_clip = config["training"].get("grad_clip")
    class_names = datasets["class_names"]

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            scaler=scaler,
            amp=amp,
            grad_clip=float(grad_clip) if grad_clip is not None else None,
        )
        val_metrics = evaluate(model, loaders["val"], criterion, device, class_names)
        # 记录本 epoch 实际使用的学习率（调度器在记录后步进）
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        duration = time.perf_counter() - epoch_start

        record = {
            "event": "epoch",
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "learning_rate": current_lr,
            "duration_s": duration,
        }
        logger.log(record)
        print(
            f"[{epoch + 1}/{epochs}] train_loss={record['train_loss']:.4f} "
            f"train_acc={record['train_accuracy']:.4f} "
            f"val_loss={record['val_loss']:.4f} val_acc={record['val_accuracy']:.4f} "
            f"({duration:.1f}s)"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, epoch, record, config)
            (out_dir / "val_metrics.json").write_text(
                json.dumps(
                    {"epoch": epoch, **val_metrics, "class_names": class_names},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.log({"event": "best", "epoch": epoch, "val_accuracy": best_val_accuracy})

        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, epoch, record, config)

    # 用验证集最优权重做独立测试集评估，测试集不参与任何模型选择
    best_path = out_dir / "best.pt"
    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model_state"])

    latency = measure_latency(
        model,
        device,
        input_size=int(dataset_config["input_size"]),
        batch_size=int(config["evaluation"]["latency"]["batch_size"]),
        warmup=int(config["evaluation"]["latency"]["warmup"]),
        repeats=int(config["evaluation"]["latency"]["repeats"]),
    )
    (out_dir / "latency.json").write_text(
        json.dumps(latency, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.log({"event": "latency", **latency})

    if config["evaluation"].get("test_after_training", True):
        test_metrics = evaluate(model, loaders["test"], criterion, device, class_names)
        (out_dir / "test_metrics.json").write_text(
            json.dumps({"class_names": class_names, **test_metrics}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.log({"event": "test", **test_metrics})
        print(
            f"独立测试集：accuracy={test_metrics['accuracy']:.4f} "
            f"f1_macro={test_metrics['f1_macro']:.4f}"
        )

    logger.log(
        {
            "event": "finish",
            "best_val_accuracy": best_val_accuracy,
            "finished_at": datetime.now(UTC).isoformat(),
        }
    )
    print(f"完成，结果目录：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
