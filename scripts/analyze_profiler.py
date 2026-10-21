"""从已采集的 profiler trace 中提取可核查的统计，不重新运行模型。

统计内容：
- CUDA kernel 数量（按迭代归一）与按名称聚合的耗时；
- CUDA runtime API 调用（launch / memcpy / synchronize）次数；
- CPU 侧 aten 算子调用次数与 self CPU 时间；
- 是否存在额外内存拷贝或同步。

用法：
    python scripts/analyze_profiler.py
输出：
    outputs/stage3a/profiler/paired/analysis.json
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = PROJECT_ROOT / "outputs" / "stage3a" / "profiler" / "paired"
OUTPUT = TRACE_DIR / "analysis.json"

MODELS = ("baseline", "eca_non_se")
ROUNDS = 3
ITERATIONS = 50


def load_trace(name: str, round_index: int) -> list[dict[str, Any]]:
    path = TRACE_DIR / f"{name}_trace_round{round_index}.json"
    return json.loads(path.read_text(encoding="utf-8"))["traceEvents"]


def summarize(name: str) -> dict[str, Any]:
    kernel_counts: list[int] = []
    kernel_us: list[float] = []
    runtime_calls: dict[str, list[int]] = defaultdict(list)
    cpu_calls: dict[str, list[int]] = defaultdict(list)
    cpu_us: dict[str, list[float]] = defaultdict(list)
    kernel_by_name: dict[str, list[float]] = defaultdict(list)
    kernel_count_by_name: dict[str, list[float]] = defaultdict(list)
    memcpy_us: list[float] = []

    for round_index in range(ROUNDS):
        events = load_trace(name, round_index)
        kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
        kernel_counts.append(len(kernels) / ITERATIONS)
        kernel_us.append(sum(float(e.get("dur", 0)) for e in kernels) / ITERATIONS)

        per_name_us: dict[str, float] = defaultdict(float)
        per_name_count: dict[str, int] = defaultdict(int)
        for event in kernels:
            key = str(event.get("name", ""))
            per_name_us[key] += float(event.get("dur", 0))
            per_name_count[key] += 1
        for key, value in per_name_us.items():
            kernel_by_name[key].append(value / ITERATIONS)
        for key, value in per_name_count.items():
            kernel_count_by_name[key].append(value / ITERATIONS)

        runtime = [e for e in events if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"]
        runtime_counter = Counter(str(e.get("name", "")) for e in runtime)
        for key in (
            "cudaLaunchKernel",
            "cudaMemcpyAsync",
            "cudaMemcpy",
            "cudaStreamSynchronize",
            "cudaDeviceSynchronize",
            "cudaMemsetAsync",
        ):
            runtime_calls[key].append(runtime_counter.get(key, 0) / ITERATIONS)

        copies = [
            e
            for e in events
            if e.get("ph") == "X" and e.get("cat") in ("Memcpy", "memcpy", "gpu_memcpy")
        ]
        memcpy_us.append(sum(float(e.get("dur", 0)) for e in copies) / ITERATIONS)

        cpu_ops = [e for e in events if e.get("ph") == "X" and e.get("cat") == "cpu_op"]
        counter = Counter(str(e.get("name", "")) for e in cpu_ops)
        # Chrome trace 的 dur 是含子调用的总时长，不是 self time；此处按总时长统计并标注
        total_time: dict[str, float] = defaultdict(float)
        for event in cpu_ops:
            total_time[str(event.get("name", ""))] += float(event.get("dur", 0))
        for key, value in counter.items():
            cpu_calls[key].append(value / ITERATIONS)
        for key, value in total_time.items():
            cpu_us[key].append(value / ITERATIONS)

    def median(values: list[float]) -> float:
        return statistics.median(values) if values else 0.0

    top_kernels = sorted(kernel_by_name.items(), key=lambda kv: median(kv[1]), reverse=True)[:20]
    top_cpu = sorted(cpu_us.items(), key=lambda kv: median(kv[1]), reverse=True)[:20]

    return {
        "model": name,
        "kernel_launches_per_iteration": {
            "samples": kernel_counts,
            "median": median(kernel_counts),
        },
        "kernel_us_per_iteration": {"samples": kernel_us, "median": median(kernel_us)},
        "kernel_by_name_us_per_iteration": [
            {
                "name": key,
                "median_us": median(values),
                "median_count": median(kernel_count_by_name.get(key, [0.0])),
            }
            for key, values in top_kernels
        ],
        "cuda_runtime_calls_per_iteration": {
            key: {"samples": values, "median": median(values)}
            for key, values in runtime_calls.items()
        },
        "memcpy_us_per_iteration": {"samples": memcpy_us, "median": median(memcpy_us)},
        "top_cpu_ops_by_total_time": [
            {"name": key, "median_us_per_iteration": median(values)} for key, values in top_cpu
        ],
        "cpu_calls_per_iteration": {
            key: median(values)
            for key, values in sorted(cpu_calls.items(), key=lambda kv: -median(kv[1]))[:20]
        },
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    result = {
        "source": str(TRACE_DIR.relative_to(PROJECT_ROOT)),
        "rounds": ROUNDS,
        "iterations_per_trace": ITERATIONS,
        "models": {name: summarize(name) for name in MODELS},
        "measured_at": datetime.now(UTC).isoformat(),
    }

    base = result["models"]["baseline"]
    eca = result["models"]["eca_non_se"]
    result["delta"] = {
        "kernel_launches_per_iteration": (
            eca["kernel_launches_per_iteration"]["median"]
            - base["kernel_launches_per_iteration"]["median"]
        ),
        "kernel_us_per_iteration": (
            eca["kernel_us_per_iteration"]["median"] - base["kernel_us_per_iteration"]["median"]
        ),
        "cudaLaunchKernel_per_iteration": (
            eca["cuda_runtime_calls_per_iteration"]["cudaLaunchKernel"]["median"]
            - base["cuda_runtime_calls_per_iteration"]["cudaLaunchKernel"]["median"]
        ),
        "memcpy_us_per_iteration": (
            eca["memcpy_us_per_iteration"]["median"] - base["memcpy_us_per_iteration"]["median"]
        ),
    }

    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["delta"], ensure_ascii=False, indent=1))
    print(f"写入 {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
