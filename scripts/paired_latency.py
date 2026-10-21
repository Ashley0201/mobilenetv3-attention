"""交替测量基线与 ECA 的推理延迟，降低热漂移与 GPU 状态差异带来的偏差。

同一进程内按 B→E→E→B 交替顺序重复多轮，每轮内部仍是 warmup + CUDA 同步 + 多次计时。
结果写入 outputs/stage3a/paired_latency.json，不覆盖任何已有文件。

用法：
    python scripts/paired_latency.py
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mobilenet_attention.engine import measure_latency  # noqa: E402
from mobilenet_attention.models import build_mobilenet_v3_large  # noqa: E402
from mobilenet_attention.utils.seed import resolve_device, set_seed  # noqa: E402

OUTPUT = PROJECT_ROOT / "outputs" / "stage3a" / "paired_latency.json"


def gpu_state() -> dict | None:
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
    parts = [p.strip() for p in result.stdout.strip().splitlines()[0].split(",")]
    if len(parts) < 4:
        return None
    return {
        "temperature_c": int(parts[0]),
        "sm_clock_mhz": int(parts[1]),
        "utilization_percent": int(parts[2]),
        "memory_used_mib": int(parts[3]),
    }


def build(attention: dict | None, device: torch.device):
    return build_mobilenet_v3_large(
        num_classes=10, pretrained="imagenet1k_v1", attention=attention
    ).to(device)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    set_seed(42)
    device = resolve_device("auto")
    baseline = build(None, device)
    eca = build({"type": "eca", "placement": "non_se_blocks"}, device)

    rounds = 6
    repeats = 100
    samples: dict[str, list[float]] = {"baseline": [], "eca_non_se": []}
    order: list[str] = []

    for round_index in range(rounds):
        # 交替顺序，抵消固定先后顺序带来的热漂移
        sequence = [("baseline", baseline), ("eca_non_se", eca)]
        if round_index % 2 == 1:
            sequence.reverse()
        for name, model in sequence:
            value = measure_latency(
                model, device, input_size=224, batch_size=1, warmup=20, repeats=repeats
            )["latency_ms_per_batch"]
            samples[name].append(value)
            order.append(name)
            print(f"round {round_index + 1} {name}: {value:.3f} ms", flush=True)

    report = {
        "protocol": {
            "batch_size": 1,
            "input_size": 224,
            "warmup": 20,
            "repeats_per_measurement": repeats,
            "rounds": rounds,
            "order": order,
            "gpu_state": gpu_state(),
        },
        "samples_ms": samples,
        "summary": {
            name: {
                "mean_ms": statistics.mean(values),
                "stdev_ms": statistics.stdev(values),
                "median_ms": statistics.median(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "n": len(values),
            }
            for name, values in samples.items()
        },
        "measured_at": datetime.now(UTC).isoformat(),
    }
    report["delta_mean_ms"] = (
        report["summary"]["eca_non_se"]["mean_ms"] - report["summary"]["baseline"]["mean_ms"]
    )
    report["delta_percent"] = (
        report["delta_mean_ms"] / report["summary"]["baseline"]["mean_ms"] * 100
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"delta mean = {report['delta_mean_ms']:.3f} ms ({report['delta_percent']:.1f}%)")
    print(f"写入 {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
