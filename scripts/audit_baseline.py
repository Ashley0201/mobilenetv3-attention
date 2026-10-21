"""基线只读审计：不改动任何已有实验产物，只新增 outputs/stage2_audit/ 下的证据文件。

检查项：
- 检查点元信息（epoch、seed、pretrained、类别数、优化器/调度器状态）
- 预训练权重与官方缓存的一致性
- 数据划分的可复现性（索引哈希）
- 理论 FLOPs（torch.utils.flop_counter，batch=1, 224×224）
- 推理延迟重复测量（5 轮 × 50 次）
- ECA 插入位置的理论参数/计算开销（仅解析计算，未实现模块）

用法：
    python scripts/audit_baseline.py
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mobilenet_attention.data import build_cifar10_datasets  # noqa: E402
from mobilenet_attention.engine import measure_latency, model_size_mb  # noqa: E402
from mobilenet_attention.models import build_mobilenet_v3_large  # noqa: E402
from mobilenet_attention.utils.seed import resolve_device  # noqa: E402

RUN_DIR = PROJECT_ROOT / "outputs" / "baseline_mobilenetv3_large_cifar10"
OUT_DIR = PROJECT_ROOT / "outputs" / "stage2_audit"
WEIGHTS_CACHE = Path(torch.hub.get_dir()) / "checkpoints" / "mobilenet_v3_large-8738ca79.pth"


def sha256_of_indices(indices: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()


def audit_checkpoint() -> dict:
    best = torch.load(RUN_DIR / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(RUN_DIR / "last.pt", map_location="cpu", weights_only=False)
    cfg = best["config"]
    state = best["model_state"]

    result = {
        "best_epoch": best["epoch"],
        "best_metrics": {
            k: best["metrics"][k]
            for k in ("val_accuracy", "val_loss", "val_f1_macro", "train_accuracy")
        },
        "last_epoch": last["epoch"],
        "has_optimizer_state": bool(best.get("optimizer_state")),
        "has_scheduler_state": best.get("scheduler_state") is not None,
        "state_dict_tensors": len(state),
        "classifier_shape": list(state["classifier.3.weight"].shape),
        "seed": cfg["experiment"]["seed"],
        "deterministic": cfg["experiment"]["deterministic"],
        "pretrained": cfg["model"]["pretrained"],
        "num_classes": cfg["model"]["num_classes"],
        "config_sections": sorted(cfg.keys()),
        "scheduler_in_config": "scheduler" in cfg["training"],
        "checkpoint_size_mib": (RUN_DIR / "best.pt").stat().st_size / 1024**2,
        "model_size_mib": model_size_mb(
            build_mobilenet_v3_large(num_classes=10, pretrained="none")
        ),
    }

    if WEIGHTS_CACHE.is_file():
        official = torch.load(WEIGHTS_CACHE, map_location="cpu", weights_only=True)
        rel = {}
        for key in ("features.0.0.weight", "features.4.block.0.0.weight", "features.16.0.weight"):
            a, b = state[key].float(), official[key].float()
            rel[key] = float((a - b).norm() / b.norm())
        result["relative_l2_to_official_weights"] = rel
        result["official_weights_sha256"] = hashlib.sha256(WEIGHTS_CACHE.read_bytes()).hexdigest()
    return result


def audit_split() -> dict:
    ds = build_cifar10_datasets(
        root=PROJECT_ROOT / "data", input_size=224, val_size=5000, split_seed=42
    )
    ds_again = build_cifar10_datasets(
        root=PROJECT_ROOT / "data", input_size=224, val_size=5000, split_seed=42
    )
    train_idx, val_idx = ds["train_indices"], ds["val_indices"]
    return {
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "overlap": len(set(train_idx) & set(val_idx)),
        "train_indices_sha256": sha256_of_indices(train_idx),
        "val_indices_sha256": sha256_of_indices(val_idx),
        "recompute_identical": ds_again["train_indices"] == train_idx
        and ds_again["val_indices"] == val_idx,
        "val_class_counts": [ds["val"].targets.count(i) for i in range(10)],
    }


def audit_flops() -> dict:
    model = build_mobilenet_v3_large(num_classes=10, pretrained="none").eval()
    dummy = torch.randn(1, 3, 224, 224)
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(dummy)
    total_flops = counter.get_total_flops()
    breakdown = counter.get_flop_counts()["Global"]
    return {
        "batch_size": 1,
        "input_size": 224,
        "total_flops": total_flops,
        "total_macs": total_flops // 2,
        "breakdown": {str(k): int(v) for k, v in breakdown.items()},
        "convention": "torch.utils.flop_counter 按 2×MACs 计；shape-based 理论值，非硬件实测",
    }


def audit_latency() -> dict:
    device = resolve_device("auto")
    model = build_mobilenet_v3_large(num_classes=10, pretrained="imagenet1k_v1").to(device)
    runs = [
        measure_latency(model, device, input_size=224, batch_size=1, warmup=10, repeats=50)[
            "latency_ms_per_batch"
        ]
        for _ in range(5)
    ]
    return {
        "runs_ms": runs,
        "mean_ms": statistics.mean(runs),
        "stdev_ms": statistics.stdev(runs),
        "min_ms": min(runs),
        "max_ms": max(runs),
        "median_ms": statistics.median(runs),
        "protocol": "batch=1, 224×224, FP32, eval(), no_grad, warmup=10, repeats=50, CUDA sync",
    }


def audit_eca_overhead(baseline_macs: int) -> dict:
    """按 ECA 论文公式解析计算插入到非 SE 块（投影卷积前）的开销。

    仅做形状统计与公式计算，未实现 ECA 模块，因此不是实测。
    """
    from torchvision.models.mobilenetv3 import InvertedResidual

    model = build_mobilenet_v3_large(num_classes=10, pretrained="none").eval()
    shapes: dict[int, tuple[int, int, int]] = {}

    def hook(index: int):
        def fn(_module, _inputs, output):
            shapes[index] = tuple(output.shape[1:])

        return fn

    targets = [
        i
        for i, layer in enumerate(model.features)
        if isinstance(layer, InvertedResidual)
        and not any(type(c).__name__ == "SqueezeExcitation" for c in layer.block)
    ]
    handles = [model.features[i].block[1].register_forward_hook(hook(i)) for i in targets]
    with torch.no_grad():
        model(torch.randn(1, 3, 224, 224))
    for handle in handles:
        handle.remove()

    def kernel_size(channels: int, gamma: float = 2.0, b: float = 1.0) -> int:
        value = int(abs(torch.log2(torch.tensor(float(channels))) / gamma + b / gamma))
        return value + 1 if value % 2 == 0 else value

    rows = []
    for index in targets:
        channels, height, width = shapes[index]
        k = kernel_size(channels)
        # 口径：全局平均池化每个元素 1 次加法；1D 卷积每个输出 1 次乘加（k×C）
        macs = channels * height * width + k * channels
        rows.append(
            {
                "block_index": index,
                "channels": channels,
                "spatial": [height, width],
                "kernel": k,
                "params": k,
                "macs": macs,
                "flops_2x_macs": 2 * macs,
            }
        )
    return {
        "insertion": "after depthwise conv, before projection conv, only blocks without SE",
        "block_indices": targets,
        "blocks": rows,
        "total_params": sum(r["params"] for r in rows),
        "total_macs": sum(r["macs"] for r in rows),
        "total_flops_2x_macs": sum(r["flops_2x_macs"] for r in rows),
        "baseline_macs_from_counter": baseline_macs,
        "overhead_ratio": (sum(r["macs"] for r in rows) / baseline_macs) if baseline_macs else None,
        "note": (
            "解析计算，未实现模块；ECA 模块本身参数仅 1D conv 权重，无 bias。"
            "池化按 1 次加法/元素计，1D 卷积按 1 次乘加/输出计。"
        ),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "checkpoint": audit_checkpoint(),
        "split": audit_split(),
        "flops": audit_flops(),
        "latency": audit_latency(),
    }
    report["eca_overhead_estimate"] = audit_eca_overhead(report["flops"]["total_macs"])
    target = OUT_DIR / "baseline_audit.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n写入 {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
