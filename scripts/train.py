"""MobileNetV3-Large 基线 / ECA 训练脚本。

用法：
    python scripts/train.py --config configs/baseline_mobilenetv3_large.yaml
    python scripts/train.py --config configs/eca_non_se.yaml --set experiment.seed=42

输出目录（每个实验 ID 唯一，不覆盖已有结果）：
    outputs/<experiment_id>/
        config_snapshot.yaml   实际生效配置（含 scheduler / transforms / attention）
        environment.json       软硬件与 Git 提交
        split_indices.json     训练/验证索引与 SHA256
        run_meta.json          命令行、起止时间、耗时、退出码
        train_log.jsonl        逐 epoch 真实日志
        best.pt                验证集最优检查点
        last.pt                最后一轮检查点（支持恢复）
        val_metrics.json       最优轮次的验证指标
        test_metrics.json      独立测试集指标（仅在训练结束后评估一次）
        latency.json           模型前向延迟
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mobilenet_attention.data import (  # noqa: E402
    build_cifar10_datasets,
    build_dataloaders,
    dataset_summary,
    transform_spec,
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
from mobilenet_attention.models import build_model, describe_model, validate_model_config  # noqa: E402
from mobilenet_attention.utils.config import (  # noqa: E402
    apply_overrides,
    load_config,
    save_config_snapshot,
)
from mobilenet_attention.utils.env import environment_snapshot  # noqa: E402
from mobilenet_attention.utils.seed import resolve_device, set_seed  # noqa: E402

SUPPORTED_MONITORS = ("val_accuracy", "val_loss")
SUPPORTED_METRICS = (
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "confusion_matrix",
    "loss",
    "num_samples",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 MobileNetV3-Large 基线 / ECA")
    parser.add_argument("--config", required=True, help="训练配置 YAML 路径")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖配置项，例如 --set training.epochs=3 --set training.amp=false",
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


def prepare_output_dir(
    root: Path,
    experiment_id: str,
    overwrite: bool,
    resume_path: Path | None = None,
) -> tuple[Path, str]:
    """解析输出目录，区分「新建」「恢复」「显式覆盖」三种语义。

    - 新建：目录不存在或为空。
    - 恢复：目录必须已存在，追加日志，不清空历史文件；禁止与 --overwrite 同用。
    - 覆盖：必须显式 --overwrite，且不能用于恢复训练。
    """
    target = root / experiment_id
    exists_nonempty = target.exists() and any(target.iterdir())

    if resume_path is not None:
        if overwrite:
            raise SystemExit("--resume 与 --overwrite 不能同时使用：恢复训练不会清空目录")
        if not target.exists():
            raise SystemExit(f"恢复训练要求输出目录已存在：{target}")
        return target, "resume"

    if exists_nonempty and not overwrite:
        raise SystemExit(
            f"实验目录已存在且非空：{target}\n"
            "请修改 experiment.id；确需覆盖请显式加 --overwrite（会与旧结果混在一起）。"
        )
    target.mkdir(parents=True, exist_ok=True)
    return target, "overwrite" if exists_nonempty else "new"


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


def resolve_scheduler_config(training: dict[str, Any]) -> dict[str, Any]:
    """调度器参数必须有默认值并写入快照，保证可复现。"""
    raw = training.get("scheduler") or {}
    name = str(raw.get("name", "cosine")).lower()
    if name != "cosine":
        raise ValueError(f"暂只支持 cosine 调度器，收到 {name}")
    return {
        "name": "cosine",
        "warmup_epochs": int(training.get("warmup_epochs", 0)),
        "warmup_start_factor": float(raw.get("warmup_start_factor", 0.1)),
        "eta_min": float(raw.get("eta_min", 1e-6)),
    }


def validate_evaluation_config(evaluation: dict[str, Any]) -> tuple[str, list[str]]:
    """校验 monitor 与 report_metrics，避免配置写了却不生效。"""
    monitor = str(evaluation.get("monitor", "val_accuracy"))
    if monitor not in SUPPORTED_MONITORS:
        raise ValueError(f"不支持的 monitor：{monitor}，可选 {SUPPORTED_MONITORS}")

    requested = list(evaluation.get("report_metrics") or SUPPORTED_METRICS)
    unknown = [m for m in requested if m not in SUPPORTED_METRICS]
    if unknown:
        raise ValueError(f"不支持的指标：{unknown}，可选 {SUPPORTED_METRICS}")
    return monitor, requested


def select_metrics(metrics: dict[str, Any], requested: list[str]) -> dict[str, Any]:
    return {k: metrics[k] for k in requested if k in metrics}


def is_improvement(value: float, best: float, monitor: str) -> bool:
    """val_accuracy 越大越好，val_loss 越小越好。"""
    if monitor == "val_loss":
        return value < best
    return value > best


def indices_sha256(indices: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()


def save_split_indices(datasets: dict[str, Any], path: Path) -> dict[str, Any]:
    train_idx = list(datasets["train_indices"])
    val_idx = list(datasets["val_indices"])
    payload = {
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "train_indices_sha256": indices_sha256(train_idx),
        "val_indices_sha256": indices_sha256(val_idx),
        "train_indices": train_idx,
        "val_indices": val_idx,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {
        "train_size": payload["train_size"],
        "val_size": payload["val_size"],
        "train_indices_sha256": payload["train_indices_sha256"],
        "val_indices_sha256": payload["val_indices_sha256"],
    }


def write_run_meta(
    path: Path,
    *,
    command: str,
    started_at: str,
    finished_at: str | None,
    duration_s: float | None,
    exit_code: int | None,
    config_path: Path,
    output_mode: str = "new",
    resume_from: str | None = None,
) -> None:
    payload = {
        "command": command,
        "config_path": str(config_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "exit_code": exit_code,
        "output_mode": output_mode,
        "resume_from": resume_from,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    config["dataset"]["transforms"] = transform_spec(
        int(dataset_config["input_size"]), dataset_config.get("augmentation", {})
    )
    config["training"]["scheduler"] = resolve_scheduler_config(config["training"])

    monitor, report_metrics = validate_evaluation_config(config["evaluation"])
    # 模型配置在创建任何目录之前校验，避免未知名称留下半成品实验目录
    validate_model_config(str(config["model"]["name"]), config["model"].get("attention"))

    exp = config["experiment"]
    seed = int(exp["seed"])
    set_seed(seed, deterministic=bool(exp.get("deterministic", True)))
    device = resolve_device(config["training"].get("device", "auto"))
    print(f"设备：{device}")

    output_root = PROJECT_ROOT / exp["output_root"]
    resume_path = Path(args.resume) if args.resume else None
    if resume_path is not None and not resume_path.is_absolute():
        resume_path = (PROJECT_ROOT / resume_path).resolve()
    out_dir, output_mode = prepare_output_dir(output_root, exp["id"], args.overwrite, resume_path)
    print(f"输出模式：{output_mode}（{out_dir}）")

    started_at = datetime.now(UTC).isoformat()
    start_time = time.perf_counter()
    run_meta_path = out_dir / "run_meta.json"
    command = " ".join([sys.executable, *sys.argv])
    write_run_meta(
        run_meta_path,
        command=command,
        started_at=started_at,
        finished_at=None,
        duration_s=None,
        exit_code=None,
        config_path=config_path,
        output_mode=output_mode,
        resume_from=str(resume_path) if resume_path else None,
    )

    env = environment_snapshot(PROJECT_ROOT)
    save_config_snapshot(config, out_dir / "config_snapshot.yaml")
    (out_dir / "environment.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger = JsonlLogger(out_dir / "train_log.jsonl")
    logger.log(
        {
            "event": "start",
            "time": started_at,
            "environment": env,
            "device": str(device),
            "command": command,
            "monitor": monitor,
            "report_metrics": report_metrics,
        }
    )

    datasets = build_cifar10_datasets(
        root=dataset_config["root"],
        input_size=int(dataset_config["input_size"]),
        val_size=int(dataset_config["split"]["val_size"]),
        split_seed=int(dataset_config["split"]["seed"]),
        download=bool(dataset_config.get("download", False)),
        augmentation=dataset_config.get("augmentation", {}),
    )
    summary = dataset_summary(datasets)
    split_info = save_split_indices(datasets, out_dir / "split_indices.json")
    logger.log({"event": "dataset", **summary, **split_info})
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
        logger.log({"event": "limit", "val_batches": int(limit_val)})

    model = build_model(
        name=str(config["model"]["name"]),
        num_classes=int(config["model"]["num_classes"]),
        pretrained=config["model"]["pretrained"],
        dropout=float(config["model"]["dropout"]),
        attention=config["model"].get("attention"),
    ).to(device)
    identity = describe_model(model)
    print(
        f"模型：{identity['model_name']}，参数：{count_parameters(model):,}，"
        f"预训练：{config['model']['pretrained']}"
    )
    logger.log(
        {
            "event": "model",
            **identity,
            "parameters": count_parameters(model),
            "size_mb": model_size_mb(model),
            "configured_name": config["model"]["name"],
            "attention_config": config["model"].get("attention"),
        }
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["training"]["label_smoothing"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler_config = config["training"]["scheduler"]
    scheduler = build_scheduler(
        optimizer,
        epochs=int(config["training"]["epochs"]),
        warmup_epochs=int(scheduler_config["warmup_epochs"]),
        eta_min=float(scheduler_config["eta_min"]),
        warmup_start_factor=float(scheduler_config["warmup_start_factor"]),
    )
    amp = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    logger.log(
        {
            "event": "optim",
            "optimizer": "AdamW",
            "learning_rate": float(config["training"]["learning_rate"]),
            "weight_decay": float(config["training"]["weight_decay"]),
            "label_smoothing": float(config["training"]["label_smoothing"]),
            "grad_clip": config["training"].get("grad_clip"),
            "amp": amp,
            "scheduler": scheduler_config,
        }
    )

    start_epoch = 0
    best_monitor_value = float("inf") if monitor == "val_loss" else -1.0
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_monitor_value = float(checkpoint.get("metrics", {}).get(monitor, best_monitor_value))
        logger.log({"event": "resume", "from": str(resume_path), "start_epoch": start_epoch})

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

        monitored = float(val_metrics["accuracy" if monitor == "val_accuracy" else "loss"])
        if is_improvement(monitored, best_monitor_value, monitor):
            best_monitor_value = monitored
            save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, epoch, record, config)
            (out_dir / "val_metrics.json").write_text(
                json.dumps(
                    {"epoch": epoch, "class_names": class_names, **val_metrics},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.log(
                {"event": "best", "epoch": epoch, "monitor": monitor, "value": best_monitor_value}
            )

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
        # 完整指标始终保存；report_metrics 仅声明对外汇报的子集，避免丢失诊断信息
        (out_dir / "test_metrics.json").write_text(
            json.dumps(
                {
                    "class_names": class_names,
                    **test_metrics,
                    "reported": select_metrics(test_metrics, report_metrics),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.log({"event": "test", **test_metrics})
        print(
            f"独立测试集：accuracy={test_metrics['accuracy']:.4f} "
            f"f1_macro={test_metrics['f1_macro']:.4f}"
        )

    finished_at = datetime.now(UTC).isoformat()
    duration_s = time.perf_counter() - start_time
    logger.log(
        {
            "event": "finish",
            "monitor": monitor,
            "best_monitor_value": best_monitor_value,
            "finished_at": finished_at,
            "duration_s": duration_s,
        }
    )
    write_run_meta(
        run_meta_path,
        command=command,
        started_at=started_at,
        finished_at=finished_at,
        duration_s=duration_s,
        exit_code=0,
        config_path=config_path,
        output_mode=output_mode,
        resume_from=str(resume_path) if resume_path else None,
    )
    print(f"完成，结果目录：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
