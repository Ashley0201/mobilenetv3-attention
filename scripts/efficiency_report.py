"""完整效率指标：参数量、可训练参数量、模型大小、FLOPs/MACs、推理显存、训练峰值显存（多次）。

关键设计：每个模型在独立子进程中测量，避免上一个模型的缓存块污染显存读数。

- 参数量 / 可训练参数量：模块参数统计，实测；
- 模型大小：参数与缓冲区字节数，实测（不含 Python 对象开销）；
- FLOPs/MACs：torch.utils.flop_counter，理论计数，明确标注统计范围；
- 推理显存：torch.cuda.max_memory_allocated 峰值，实测；
- 训练峰值显存：batch=128 + AMP，重复 3 次，实测。

用法：
    python scripts/efficiency_report.py
输出：
    outputs/stage3a/efficiency_full.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mobilenet_attention.models import ECALayer, build_model, describe_model  # noqa: E402
from mobilenet_attention.utils.seed import resolve_device, set_seed  # noqa: E402

OUTPUT = PROJECT_ROOT / "outputs" / "stage3a" / "efficiency_full.json"

PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {"name": "mobilenet_v3_large", "attention": None},
    "eca_non_se": {
        "name": "mobilenet_v3_large_eca",
        "attention": {"type": "eca", "placement": "non_se_blocks"},
    },
}


def param_report(model: nn.Module) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    attention = sum(
        p.numel() for m in model.modules() if isinstance(m, ECALayer) for p in m.parameters()
    )
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "attention": attention,
        "parameter_bytes": param_bytes,
        "buffer_bytes": buffer_bytes,
        "model_size_mib": (param_bytes + buffer_bytes) / (1024**2),
        "is_measured": True,
    }


def flop_report(model: nn.Module, device: torch.device, input_size: int = 224) -> dict[str, Any]:
    """计数器口径：只统计注册了 FLOP 公式的算子（卷积、全连接）。"""
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(dummy)
    total = int(counter.get_total_flops())
    breakdown = {str(k): int(v) for k, v in counter.get_flop_counts()["Global"].items()}
    return {
        "tool": "torch.utils.flop_counter.FlopCounterMode",
        "scope": "仅卷积与全连接；不含 GAP、Sigmoid、逐元素乘法",
        "batch_size": 1,
        "input_size": input_size,
        "total_flops": total,
        "total_macs": total // 2,
        "breakdown": breakdown,
        "is_measured": False,
        "note": "shape-based 理论计数，不是硬件实测吞吐",
    }


def analytic_eca(model: nn.Module, device: torch.device, input_size: int = 224) -> dict | None:
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, ECALayer)]
    if not layers:
        return None
    shapes: dict[str, tuple[int, int, int]] = {}

    def hook(name: str):
        def fn(_m, inputs, _o):
            shapes[name] = tuple(inputs[0].shape[1:])

        return fn

    handles = [m.register_forward_hook(hook(n)) for n, m in layers]
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, input_size, input_size, device=device))
    for h in handles:
        h.remove()

    rows = []
    for name, module in layers:
        c, h, w = shapes[name]
        k = module.kernel_size
        rows.append(
            {
                "layer": name,
                "channels": c,
                "spatial": [h, w],
                "kernel": k,
                "gap_macs": h * w * c,
                "conv1d_macs": k * c,
                "mul_macs": h * w * c,
                "total_macs": 2 * h * w * c + k * c,
            }
        )
    return {
        "layers": rows,
        "total_macs": sum(r["total_macs"] for r in rows),
        "is_measured": False,
        "note": "由真实前向钩子取形状后解析计算；每元素一次乘或加记 1 MAC",
    }


def inference_memory(model: nn.Module, device: torch.device, batch_size: int = 1) -> dict | None:
    """前向显存：weights 为构建后常驻量，peak 为本次前向峰值，delta 为激活增量。"""
    if device.type != "cuda":
        return None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    model.eval()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(torch.randn(batch_size, 3, 224, 224, device=device))
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    result = {
        "batch_size": batch_size,
        "weights_and_buffers_mib": base / 1024**2,
        "peak_total_mib": peak / 1024**2,
        "activation_delta_mib": (peak - base) / 1024**2,
        "is_measured": True,
    }
    torch.cuda.empty_cache()
    return result


def training_peak(
    model: nn.Module, device: torch.device, batch_size: int = 128, repeats: int = 3
) -> dict[str, Any] | None:
    if device.type != "cuda":
        return None
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    samples = []
    for _ in range(repeats):
        model.train()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        images = torch.randn(batch_size, 3, 224, 224, device=device)
        labels = torch.randint(0, 10, (batch_size,), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        samples.append(torch.cuda.max_memory_allocated() / 1024**2)
        del images, labels, loss
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "batch_size": batch_size,
        "amp": True,
        "repeats": repeats,
        "samples_mib": samples,
        "mean_mib": statistics.mean(samples),
        "stdev_mib": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "max_mib": max(samples),
        "is_measured": True,
    }


def measure_one(key: str) -> dict[str, Any]:
    """在干净进程中测量单个模型。"""
    preset = PRESETS[key]
    set_seed(42)
    device = resolve_device("auto")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    model = build_model(
        name=preset["name"],
        num_classes=10,
        pretrained="imagenet1k_v1",
        attention=preset["attention"],
    ).to(device)

    entry: dict[str, Any] = {
        "identity": describe_model(model),
        "parameters": param_report(model),
        "inference_memory_bs1": inference_memory(model, device, 1),
        "flops": flop_report(model, device),
    }
    eca = analytic_eca(model, device)
    if eca:
        entry["eca_analytic"] = eca
    entry["training_peak"] = training_peak(model, device)
    return entry


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="完整效率指标（每模型独立进程）")
    parser.add_argument("--single", choices=sorted(PRESETS), default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single:
        print(json.dumps(measure_one(args.single), ensure_ascii=False))
        return 0

    report: dict[str, Any] = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "measurement_isolation": "每个模型在独立子进程中测量，避免显存读数互相污染",
        "models": {},
        "measured_at": datetime.now(UTC).isoformat(),
    }

    for key in PRESETS:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--single", key],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=1800,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(f"{key} 测量失败：\n{result.stderr[-2000:]}")
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        report["models"][key] = payload
        entry = payload
        print(
            f"{key}: params={entry['parameters']['total']:,} "
            f"trainable={entry['parameters']['trainable']:,} "
            f"size={entry['parameters']['model_size_mib']:.4f} MiB "
            f"infer_peak={entry['inference_memory_bs1']['peak_total_mib']:.1f} MiB "
            f"train_peak_mean={entry['training_peak']['mean_mib']:.1f} MiB"
        )

    base = report["models"]["baseline"]
    eca = report["models"]["eca_non_se"]
    report["delta"] = {
        "parameters": eca["parameters"]["total"] - base["parameters"]["total"],
        "trainable_parameters": eca["parameters"]["trainable"] - base["parameters"]["trainable"],
        "model_size_mib": eca["parameters"]["model_size_mib"]
        - base["parameters"]["model_size_mib"],
        "flops": eca["flops"]["total_flops"] - base["flops"]["total_flops"],
        "macs_counter": eca["flops"]["total_macs"] - base["flops"]["total_macs"],
        "eca_analytic_macs": eca["eca_analytic"]["total_macs"],
        "eca_analytic_macs_percent": eca["eca_analytic"]["total_macs"]
        / base["flops"]["total_macs"]
        * 100,
        "inference_peak_total_mib_bs1": (
            eca["inference_memory_bs1"]["peak_total_mib"]
            - base["inference_memory_bs1"]["peak_total_mib"]
        ),
        "training_peak_mean_mib": (
            eca["training_peak"]["mean_mib"] - base["training_peak"]["mean_mib"]
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["delta"], ensure_ascii=False, indent=1))
    print(f"写入 {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
