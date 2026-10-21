"""基线与 ECA 的 torch.profiler 对照分析。

协议（两个模型完全一致）：
1. 全局时钟预热：连续前向直到 GPU 时钟稳定，避免首个模型在低频下被测量；
2. 交替测量未启用 profiler 的延迟（多轮，取中位数）；
3. 分别用 profiler 采集（warmup 20 + 采集 50 次），记录每次迭代的 kernel 数与耗时；
4. profiler 结束后再交替测一次延迟，用于判断 profiler 是否造成持续影响。

不预设瓶颈原因，只输出可核查的算子耗时、kernel 数量与关键算子明细。

用法：
    python scripts/profile_models.py
输出（不覆盖已有文件）：
    outputs/stage3a/profiler/baseline_profile.json
    outputs/stage3a/profiler/eca_profile.json
    outputs/stage3a/profiler/comparison.json
    outputs/stage3a/profiler/*_trace.json
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mobilenet_attention.models import build_model  # noqa: E402
from mobilenet_attention.utils.seed import resolve_device, set_seed  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3a" / "profiler"

PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {"name": "mobilenet_v3_large", "attention": None},
    "eca_non_se": {
        "name": "mobilenet_v3_large_eca",
        "attention": {"type": "eca", "placement": "non_se_blocks"},
    },
}

ECA_KEYWORDS = (
    "adaptive_avg_pool",
    "avg_pool",
    "conv1d",
    "sigmoid",
    "mul",
    "fill",
)


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


def build_models(device: torch.device) -> dict[str, torch.nn.Module]:
    return {
        name: build_model(
            name=preset["name"],
            num_classes=10,
            pretrained="imagenet1k_v1",
            attention=preset["attention"],
        ).to(device)
        for name, preset in PRESETS.items()
    }


def warm_clock(
    models: dict[str, torch.nn.Module],
    device: torch.device,
    iterations: int = 400,
    target_clock_mhz: int = 2000,
    max_rounds: int = 20,
) -> dict[str, Any]:
    """交替前向直到 GPU 时钟达到工作频率，避免把低频状态写进结果。

    profiler 前后的时钟差异会直接污染算子耗时对比，因此显式等待并记录时钟。
    """
    dummy = torch.randn(1, 3, 224, 224, device=device)
    for model in models.values():
        model.eval()

    history: list[dict | None] = []
    rounds = 0
    for round_index in range(max_rounds):
        rounds = round_index + 1
        with torch.no_grad():
            for _ in range(max(iterations // len(models), 1)):
                for model in models.values():
                    model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        state = gpu_state()
        history.append(state)
        clock = (state or {}).get("sm_clock_mhz", 0)
        if clock >= target_clock_mhz:
            break
    return {"rounds": rounds, "clock_history": history, "final": history[-1]}


def time_once(
    model: torch.nn.Module, device: torch.device, warmup: int = 20, repeats: int = 100
) -> float:
    dummy = torch.randn(1, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed / repeats * 1000


def interleaved_latency(
    models: dict[str, torch.nn.Module], device: torch.device, rounds: int = 6
) -> dict[str, Any]:
    """交替测量，抵消固定顺序带来的热漂移。"""
    samples: dict[str, list[float]] = {name: [] for name in models}
    for round_index in range(rounds):
        order = list(models.items())
        if round_index % 2 == 1:
            order.reverse()
        for name, model in order:
            samples[name].append(time_once(model, device))
    return {
        "samples_ms": samples,
        "summary": {
            name: {
                "mean_ms": statistics.mean(values),
                "median_ms": statistics.median(values),
                "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min_ms": min(values),
                "max_ms": max(values),
                "n": len(values),
            }
            for name, values in samples.items()
        },
    }


def summarize_events(events: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "key": getattr(event, "key", None),
            "count": getattr(event, "count", None),
            "self_cpu_time_total_us": getattr(event, "self_cpu_time_total", None),
            "cpu_time_total_us": getattr(event, "cpu_time_total", None),
            "self_device_time_total_us": getattr(event, "self_device_time_total", None),
            "device_time_total_us": getattr(event, "device_time_total", None),
        }
        for event in events
    ]


def collect_profile(
    name: str, model: torch.nn.Module, device: torch.device, repeats: int = 50
) -> dict[str, Any]:
    dummy = torch.randn(1, 3, 224, 224, device=device)
    model.eval()

    # 每个模型单独 profile 前都重新拉高时钟，减少两次测量之间的频率差异
    per_model_clock = warm_clock({name: model}, device, iterations=600)

    with torch.no_grad():
        for _ in range(20):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    state_before = gpu_state()
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(repeats):
                model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
    state_after = gpu_state()

    key_averages = prof.key_averages()
    by_cuda = sorted(
        key_averages, key=lambda e: getattr(e, "self_device_time_total", 0), reverse=True
    )
    by_cpu = sorted(key_averages, key=lambda e: e.self_cpu_time_total, reverse=True)

    related = {}
    for keyword in ECA_KEYWORDS:
        matches = [e for e in key_averages if keyword in str(getattr(e, "key", "")).lower()]
        if matches:
            related[keyword] = summarize_events(
                sorted(
                    matches,
                    key=lambda e: getattr(e, "self_device_time_total", 0),
                    reverse=True,
                )[:5]
            )

    trace_path = OUTPUT_DIR / f"{name}_trace.json"
    prof.export_chrome_trace(str(trace_path))

    return {
        "model": name,
        "device": str(device),
        "batch_size": 1,
        "input_size": 224,
        "warmup": 20,
        "profiled_iterations": repeats,
        "activities": [a.name for a in activities],
        "per_model_clock_warmup": per_model_clock,
        "gpu_state_before": state_before,
        "gpu_state_after": state_after,
        "top_by_cuda_self_time": summarize_events(by_cuda[:15]),
        "top_by_cpu_self_time": summarize_events(by_cpu[:15]),
        "related_ops": related,
        "total_events": len(key_averages),
        "trace_file": str(trace_path.relative_to(PROJECT_ROOT)),
    }


def trace_kernel_stats(trace_path: Path, iterations: int) -> dict[str, Any]:
    """从 Chrome trace 统计 CUDA kernel 数与耗时，并按迭代归一。"""
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    events = data.get("traceEvents", [])
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    total_us = sum(float(e.get("dur", 0)) for e in kernels)
    by_name: dict[str, dict[str, float]] = {}
    for event in kernels:
        key = str(event.get("name", ""))[:120]
        slot = by_name.setdefault(key, {"count": 0.0, "total_us": 0.0})
        slot["count"] += 1
        slot["total_us"] += float(event.get("dur", 0))
    top = sorted(by_name.items(), key=lambda kv: kv[1]["total_us"], reverse=True)[:15]
    return {
        "kernel_launches": len(kernels),
        "kernel_launches_per_iteration": len(kernels) / iterations,
        "kernel_total_us": total_us,
        "kernel_us_per_iteration": total_us / iterations,
        "top_kernels": [
            {
                "name": name,
                "count": int(value["count"]),
                "count_per_iteration": value["count"] / iterations,
                "total_us": value["total_us"],
            }
            for name, value in top
        ],
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    set_seed(42)
    device = resolve_device("auto")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = build_models(device)
    clock_warmup = warm_clock(models, device, iterations=600)

    before = interleaved_latency(models, device, rounds=6)
    print("profiler 前延迟：")
    for name, summary in before["summary"].items():
        print(f"  {name}: median {summary['median_ms']:.3f} ms (n={summary['n']})")

    results: dict[str, Any] = {}
    for name, model in models.items():
        result = collect_profile(name, model, device, repeats=50)
        result["latency_before_profiler_ms"] = before["summary"][name]["median_ms"]
        result["clock_warmup"] = clock_warmup
        result["kernel_stats"] = trace_kernel_stats(
            OUTPUT_DIR / f"{name}_trace.json", iterations=50
        )
        results[name] = result
        (OUTPUT_DIR / f"{name}_profile.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"{name}: kernels/iter={result['kernel_stats']['kernel_launches_per_iteration']:.1f} "
            f"kernel_us/iter={result['kernel_stats']['kernel_us_per_iteration']:.1f}"
        )

    after = interleaved_latency(models, device, rounds=6)
    print("profiler 后延迟：")
    for name, summary in after["summary"].items():
        print(f"  {name}: median {summary['median_ms']:.3f} ms (n={summary['n']})")

    base = results["baseline"]
    eca = results["eca_non_se"]
    comparison = {
        "protocol": {
            "batch_size": 1,
            "input_size": 224,
            "warmup_before_profile": 20,
            "profiled_iterations": 50,
            "latency_rounds": 6,
            "clock_warmup_iterations": 400,
        },
        "clock_warmup_before": clock_warmup,
        "latency_before_profiler": before["summary"],
        "latency_after_profiler": after["summary"],
        "kernel_stats": {
            "baseline": base["kernel_stats"],
            "eca_non_se": eca["kernel_stats"],
        },
        "delta_kernel_launches_per_iteration": (
            eca["kernel_stats"]["kernel_launches_per_iteration"]
            - base["kernel_stats"]["kernel_launches_per_iteration"]
        ),
        "delta_kernel_us_per_iteration": (
            eca["kernel_stats"]["kernel_us_per_iteration"]
            - base["kernel_stats"]["kernel_us_per_iteration"]
        ),
        "profiler_impact": {
            name: {
                "before_ms": before["summary"][name]["median_ms"],
                "after_ms": after["summary"][name]["median_ms"],
                "drift_percent": (
                    after["summary"][name]["median_ms"] - before["summary"][name]["median_ms"]
                )
                / before["summary"][name]["median_ms"]
                * 100,
            }
            for name in models
        },
        "note": (
            "profiler 会引入额外 CPU/GPU 开销，单次墙钟不作为延迟结论；kernel 数按 50 次迭代归一。"
        ),
        "measured_at": datetime.now(UTC).isoformat(),
    }
    (OUTPUT_DIR / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"写入 {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
