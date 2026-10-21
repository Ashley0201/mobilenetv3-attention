"""基线与 ECA 的交替 profiler 对照（3 轮，B/E 顺序交替）。

目的：GPU 时钟在两次 profile 之间会变化，单轮对比可能被频率差异污染。
本脚本在同一进程内交替采集多轮，报告每轮 kernel 数与耗时，并以中位数作为结论依据。
kernel 数不受时钟影响，可直接对比。

输出（不覆盖已有文件）：
    outputs/stage3a/profiler/paired/comparison.json
    outputs/stage3a/profiler/paired/*_trace_round{N}.json

用法：
    python scripts/profile_paired.py
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mobilenet_attention.models import build_model  # noqa: E402
from mobilenet_attention.utils.seed import resolve_device, set_seed  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3a" / "profiler" / "paired"

PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {"name": "mobilenet_v3_large", "attention": None},
    "eca_non_se": {
        "name": "mobilenet_v3_large_eca",
        "attention": {"type": "eca", "placement": "non_se_blocks"},
    },
}

ROUNDS = 3
ITERATIONS = 50


def gpu_state() -> dict | None:
    import subprocess

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


def burn(model: torch.nn.Module, device: torch.device, iterations: int = 200) -> None:
    dummy = torch.randn(1, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(iterations):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()


def profile_once(
    name: str, model: torch.nn.Module, device: torch.device, round_index: int
) -> dict[str, Any]:
    burn(model, device, 200)
    dummy = torch.randn(1, 3, 224, 224, device=device)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    before = gpu_state()
    with profile(activities=activities, record_shapes=True, profile_memory=False) as prof:
        with torch.no_grad():
            for _ in range(ITERATIONS):
                model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
    after = gpu_state()

    trace_path = OUTPUT_DIR / f"{name}_trace_round{round_index}.json"
    prof.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text(encoding="utf-8")).get("traceEvents", [])
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    total_us = sum(float(e.get("dur", 0)) for e in kernels)

    key_averages = prof.key_averages()
    related = {}
    for keyword in ("adaptive_avg_pool", "avg_pool", "conv1d", "sigmoid", "mul"):
        matches = [e for e in key_averages if keyword in str(e.key).lower()]
        if matches:
            related[keyword] = {
                "count_total": sum(e.count for e in matches),
                "count_per_iteration": sum(e.count for e in matches) / ITERATIONS,
                "device_us_total": sum(getattr(e, "self_device_time_total", 0) for e in matches),
                "device_us_per_iteration": sum(
                    getattr(e, "self_device_time_total", 0) for e in matches
                )
                / ITERATIONS,
            }

    return {
        "round": round_index,
        "model": name,
        "gpu_state_before": before,
        "gpu_state_after": after,
        "kernel_launches": len(kernels),
        "kernel_launches_per_iteration": len(kernels) / ITERATIONS,
        "kernel_us_per_iteration": total_us / ITERATIONS,
        "related_ops": related,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    set_seed(42)
    device = resolve_device("auto")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        name: build_model(
            name=preset["name"],
            num_classes=10,
            pretrained="imagenet1k_v1",
            attention=preset["attention"],
        ).to(device)
        for name, preset in PRESETS.items()
    }

    samples: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    order: list[str] = []
    for round_index in range(ROUNDS):
        sequence = list(models.items())
        if round_index % 2 == 1:
            sequence.reverse()
        for name, model in sequence:
            result = profile_once(name, model, device, round_index)
            samples[name].append(result)
            order.append(name)
            print(
                f"round {round_index + 1} {name}: "
                f"kernels/iter={result['kernel_launches_per_iteration']:.1f} "
                f"us/iter={result['kernel_us_per_iteration']:.1f} "
                f"clock={result['gpu_state_before']['sm_clock_mhz']}MHz"
            )

    summary = {}
    for name, results in samples.items():
        launches = [r["kernel_launches_per_iteration"] for r in results]
        times = [r["kernel_us_per_iteration"] for r in results]
        summary[name] = {
            "kernel_launches_per_iteration_samples": launches,
            "kernel_launches_per_iteration_median": statistics.median(launches),
            "kernel_us_per_iteration_samples": times,
            "kernel_us_per_iteration_median": statistics.median(times),
            "clocks_mhz": [r["gpu_state_before"]["sm_clock_mhz"] for r in results],
        }

    delta_launches = (
        summary["eca_non_se"]["kernel_launches_per_iteration_median"]
        - summary["baseline"]["kernel_launches_per_iteration_median"]
    )
    delta_us = (
        summary["eca_non_se"]["kernel_us_per_iteration_median"]
        - summary["baseline"]["kernel_us_per_iteration_median"]
    )

    report = {
        "protocol": {
            "batch_size": 1,
            "input_size": 224,
            "iterations_per_profile": ITERATIONS,
            "rounds": ROUNDS,
            "order": order,
            "warmup_per_profile": 200,
        },
        "per_round": samples,
        "summary": summary,
        "delta_kernel_launches_per_iteration": delta_launches,
        "delta_kernel_us_per_iteration": delta_us,
        "related_ops_eca_median": {
            key: statistics.median(
                r["related_ops"][key]["device_us_per_iteration"]
                for r in samples["eca_non_se"]
                if key in r["related_ops"]
            )
            for key in ("adaptive_avg_pool", "avg_pool", "conv1d", "sigmoid", "mul")
            if any(key in r["related_ops"] for r in samples["eca_non_se"])
        },
        "caveat": (
            "kernel 数不受时钟影响，可直接比较；kernel 耗时受 GPU 频率影响，"
            "表中记录每轮时钟，跨轮比较需谨慎。"
        ),
        "measured_at": datetime.now(UTC).isoformat(),
    }
    (OUTPUT_DIR / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"delta kernels/iter = {delta_launches:+.1f}")
    print(f"delta kernel us/iter = {delta_us:+.1f}")
    print(f"写入 {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
