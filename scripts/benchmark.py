"""基线资源基准：参数量、模型体积、推理延迟与训练显存峰值。

只测量标准 MobileNetV3-Large（CIFAR-10 分类头），不含注意力模块。
结果写入 outputs/stage2_benchmark.json，作为阶段 2 的可追溯证据。

用法：
    python scripts/benchmark.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mobilenet_attention.engine import measure_latency, model_size_mb  # noqa: E402
from mobilenet_attention.models import build_mobilenet_v3_large  # noqa: E402
from mobilenet_attention.utils.seed import resolve_device, set_seed  # noqa: E402

OUTPUT = PROJECT_ROOT / "outputs" / "stage2_benchmark.json"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    set_seed(42)
    device = resolve_device("auto")
    model = build_mobilenet_v3_large(num_classes=10, pretrained="imagenet1k_v1").to(device)

    report: dict = {
        "model": "mobilenet_v3_large",
        "num_classes": 10,
        "pretrained": "imagenet1k_v1",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "model_size_mib": model_size_mb(model),
        "latency": {},
    }
    for batch_size in (1, 16, 128):
        report["latency"][f"bs{batch_size}"] = measure_latency(
            model, device, input_size=224, batch_size=batch_size, warmup=10, repeats=30
        )
        print(
            f"bs={batch_size}: {report['latency'][f'bs{batch_size}']['latency_ms_per_batch']:.2f} ms"
        )

    # 训练态显存峰值：FP16 自动混合精度 + 反向传播，batch 128
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    images = torch.randn(128, 3, 224, 224, device=device)
    labels = torch.randint(0, 10, (128,), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        loss = criterion(model(images), labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
        report["train_peak_memory_mib_bs128_amp"] = torch.cuda.max_memory_allocated() / 1024**2
        print(f"训练显存峰值：{report['train_peak_memory_mib_bs128_amp']:.0f} MiB")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
