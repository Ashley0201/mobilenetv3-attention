"""效率测量：参数量、模型体积、理论 FLOPs/MACs、推理延迟与训练显存峰值。

基线与 ECA 使用同一脚本、同一输入、同一设备，结果写入独立 JSON。
不覆盖阶段 2 的 outputs/stage2_benchmark.json。

用法：
    python scripts/measure_efficiency.py --model baseline
    python scripts/measure_efficiency.py --model eca_non_se
    python scripts/measure_efficiency.py --model baseline --output outputs/stage3a/baseline_efficiency.json
"""

from __future__ import annotations

import argparse
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

from mobilenet_attention.engine import measure_latency, model_size_mb  # noqa: E402
from mobilenet_attention.models import (  # noqa: E402
    ECALayer,
    build_model,
    describe_model,
)
from mobilenet_attention.utils.seed import resolve_device, set_seed  # noqa: E402

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {
        "name": "mobilenet_v3_large",
        "pretrained": "imagenet1k_v1",
        "attention": None,
    },
    "eca_non_se": {
        "name": "mobilenet_v3_large_eca",
        "pretrained": "imagenet1k_v1",
        "attention": {"type": "eca", "placement": "non_se_blocks", "gamma": 2.0, "b": 1.0},
    },
}


def attention_parameters(model: nn.Module) -> int:
    """按模块实例统计 ECA 参数（Sequential 索引使参数名不含类名）。"""
    return sum(
        p.numel() for m in model.modules() if isinstance(m, ECALayer) for p in m.parameters()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测量模型效率指标")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_PRESETS))
    parser.add_argument("--output", default=None, help="输出 JSON 路径")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--latency-runs", type=int, default=5, help="延迟重复轮数")
    parser.add_argument("--latency-repeats", type=int, default=50, help="每轮计时次数")
    parser.add_argument("--train-batch-size", type=int, default=128)
    return parser.parse_args()


def gpu_state() -> dict[str, Any] | None:
    """记录 GPU 温度、SM 时钟与占用，用于解释延迟波动。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,clocks.sm,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    line = result.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return None
    return {
        "temperature_c": int(parts[0]),
        "sm_clock_mhz": int(parts[1]),
        "utilization_percent": int(parts[2]),
        "memory_used_mib": int(parts[3]),
    }


def measure_flops(model: nn.Module, input_size: int, device: torch.device) -> dict[str, Any]:
    """shape-based 理论 FLOPs；计数口径 2×MACs，非硬件实测吞吐。

    注意：FlopCounterMode 只统计注册了公式的算子。本模型中它覆盖卷积与全连接，
    不覆盖 AdaptiveAvgPool2d、Sigmoid 和逐元素乘法，因此 ECA 的池化与重标定
    开销不在此计数内，需与 analytic_attention_overhead 一起阅读。
    """
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(dummy)
    total = counter.get_total_flops()
    breakdown = counter.get_flop_counts()["Global"]
    return {
        "batch_size": 1,
        "input_size": input_size,
        "total_flops": int(total),
        "total_macs": int(total // 2),
        "breakdown": {str(k): int(v) for k, v in breakdown.items()},
        "convention": "torch.utils.flop_counter；理论值，非硬件实测",
    }


def measure_train_peak(
    model: nn.Module, device: torch.device, input_size: int, batch_size: int
) -> float | None:
    """训练一步（AMP + 反向 + 优化器）后的显存峰值，单位 MiB。"""
    if device.type != "cuda":
        return None
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    images = torch.randn(batch_size, 3, input_size, input_size, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)

    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=True):
        loss = criterion(model(images), labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**2
    del images, labels, loss, optimizer
    torch.cuda.empty_cache()
    return peak


def analytic_attention_overhead(
    model: nn.Module, input_size: int, device: torch.device
) -> dict[str, Any] | None:
    """用真实前向钩子取每个 ECA 的输入形状，解析计算 GAP/1D 卷积/重标定的 MACs。

    计数口径（每元素一次乘或加记 1 MAC）：
    - GAP：H*W 次加法
    - 1D 卷积：k*C 次乘加
    - 逐通道相乘：H*W*C 次乘法
    """
    layers = [(name, m) for name, m in model.named_modules() if isinstance(m, ECALayer)]
    if not layers:
        return None

    shapes: dict[str, tuple[int, int, int]] = {}

    def hook(name: str):
        def fn(_module, inputs, _output):
            shapes[name] = tuple(inputs[0].shape[1:])

        return fn

    handles = [m.register_forward_hook(hook(name)) for name, m in layers]
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, input_size, input_size, device=device))
    for handle in handles:
        handle.remove()

    rows = []
    for name, module in layers:
        channels, height, width = shapes[name]
        k = module.kernel_size
        gap_macs = height * width * channels
        conv_macs = k * channels
        mul_macs = height * width * channels
        rows.append(
            {
                "layer": name,
                "channels": channels,
                "spatial": [height, width],
                "kernel": k,
                "params": sum(p.numel() for p in module.parameters()),
                "gap_macs": gap_macs,
                "conv1d_macs": conv_macs,
                "mul_macs": mul_macs,
                "total_macs": gap_macs + conv_macs + mul_macs,
            }
        )
    return {
        "layers": rows,
        "total_params": sum(r["params"] for r in rows),
        "total_macs": sum(r["total_macs"] for r in rows),
        "conv1d_macs_only": sum(r["conv1d_macs"] for r in rows),
        "convention": "每元素一次乘或加记 1 MAC；形状来自真实前向钩子",
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    preset = MODEL_PRESETS[args.model]
    set_seed(42)
    device = resolve_device("auto")

    model = build_model(
        name=preset["name"],
        num_classes=10,
        pretrained=preset["pretrained"],
        attention=preset["attention"],
    ).to(device)
    model.eval()
    identity = describe_model(model)

    state_before = gpu_state()
    latency_runs = [
        measure_latency(
            model,
            device,
            input_size=args.input_size,
            batch_size=1,
            warmup=10,
            repeats=args.latency_repeats,
        )["latency_ms_per_batch"]
        for _ in range(args.latency_runs)
    ]
    state_after = gpu_state()

    report: dict[str, Any] = {
        "model": args.model,
        "identity": identity,
        "attention": getattr(model, "attention", None),
        "pretrained": preset["pretrained"],
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameters": {
            "total": sum(p.numel() for p in model.parameters()),
            "backbone": sum(p.numel() for p in model.features.parameters()),
            "classifier": sum(p.numel() for p in model.classifier.parameters()),
            "attention": attention_parameters(model),
        },
        "model_size_mib": model_size_mb(model),
        "flops": measure_flops(model, args.input_size, device),
        "analytic_attention_overhead": analytic_attention_overhead(model, args.input_size, device),
        "latency": {
            "batch_size": 1,
            "input_size": args.input_size,
            "warmup": 10,
            "repeats_per_run": args.latency_repeats,
            "runs": latency_runs,
            "mean_ms": statistics.mean(latency_runs),
            "stdev_ms": statistics.stdev(latency_runs) if len(latency_runs) > 1 else 0.0,
            "min_ms": min(latency_runs),
            "max_ms": max(latency_runs),
            "median_ms": statistics.median(latency_runs),
        },
        "gpu_state_before": state_before,
        "gpu_state_after": state_after,
        "train_peak_memory_mib": measure_train_peak(
            model, device, args.input_size, args.train_batch_size
        ),
        "measured_at": datetime.now(UTC).isoformat(),
    }

    output = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "outputs" / "stage3a" / f"{args.model}_efficiency.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"model={args.model}")
    print(
        f"params total={report['parameters']['total']:,} attention={report['parameters']['attention']}"
    )
    print(f"flops={report['flops']['total_flops']:,} macs={report['flops']['total_macs']:,}")
    print(
        f"latency mean={report['latency']['mean_ms']:.3f} ms stdev={report['latency']['stdev_ms']:.3f}"
    )
    print(f"train peak={report['train_peak_memory_mib']}")
    print(f"写入 {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
