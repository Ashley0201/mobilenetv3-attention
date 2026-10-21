"""独立评估脚本：加载检查点，在指定划分上复算指标。

用法：
    python scripts/evaluate.py --checkpoint outputs/<exp>/best.pt --split test
    python scripts/evaluate.py --checkpoint outputs/<exp>/best.pt --split val

评估只读取权重，不更新任何参数，也不参与模型选择。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mobilenet_attention.data import build_cifar10_datasets, build_dataloaders  # noqa: E402
from mobilenet_attention.engine import evaluate, measure_latency  # noqa: E402
from mobilenet_attention.models import build_mobilenet_v3_large  # noqa: E402
from mobilenet_attention.utils.config import load_config  # noqa: E402
from mobilenet_attention.utils.seed import resolve_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估已训练检查点")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-config", default="configs/dataset_cifar10.yaml")
    parser.add_argument("--output", default=None, help="指标输出路径（默认打印到终端）")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise SystemExit(f"检查点不存在：{checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    seed = int(config.get("experiment", {}).get("seed", 42))
    set_seed(seed)

    data_config_path = Path(args.data_config)
    if not data_config_path.is_absolute():
        data_config_path = PROJECT_ROOT / data_config_path
    dataset_config = load_config(data_config_path)["dataset"]
    root = Path(dataset_config["root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    device = resolve_device(config.get("training", {}).get("device", "auto"))
    model = build_mobilenet_v3_large(
        num_classes=int(dataset_config["num_classes"]),
        pretrained="none",  # 权重来自检查点，不重复下载预训练
        dropout=float(config.get("model", {}).get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    datasets = build_cifar10_datasets(
        root=root,
        input_size=int(dataset_config["input_size"]),
        val_size=int(dataset_config["split"]["val_size"]),
        split_seed=int(dataset_config["split"]["seed"]),
        download=False,
        augmentation=dataset_config.get("augmentation", {}),
    )
    loaders = build_dataloaders(datasets, batch_size=256, num_workers=4)

    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=float(config.get("training", {}).get("label_smoothing", 0.0))
    )
    metrics = evaluate(model, loaders[args.split], criterion, device, datasets["class_names"])
    latency = measure_latency(
        model,
        device,
        input_size=int(dataset_config["input_size"]),
        batch_size=1,
        warmup=10,
        repeats=50,
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        **metrics,
        "latency": latency,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
