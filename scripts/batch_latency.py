"""不同 batch size 下的基线与 ECA 延迟对照，结果写入 JSON 以便追溯。

用法：
    python scripts/batch_latency.py
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mobilenet_attention.engine import measure_latency  # noqa: E402
from mobilenet_attention.models import build_mobilenet_v3_large  # noqa: E402
from mobilenet_attention.utils.seed import resolve_device, set_seed  # noqa: E402

OUTPUT = PROJECT_ROOT / "outputs" / "stage3a" / "batch_latency.json"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    set_seed(42)
    device = resolve_device("auto")
    baseline = build_mobilenet_v3_large(num_classes=10, pretrained="imagenet1k_v1").to(device)
    eca = build_mobilenet_v3_large(
        num_classes=10,
        pretrained="imagenet1k_v1",
        attention={"type": "eca", "placement": "non_se_blocks"},
    ).to(device)

    report: dict = {
        "protocol": {"input_size": 224, "warmup": 10, "repeats": 30, "rounds": 4},
        "batch_sizes": {},
        "measured_at": datetime.now(UTC).isoformat(),
    }

    for batch_size in (1, 16, 128):
        samples: dict[str, list[float]] = {"baseline": [], "eca_non_se": []}
        for round_index in range(4):
            sequence = [("baseline", baseline), ("eca_non_se", eca)]
            if round_index % 2 == 1:
                sequence.reverse()
            for name, model in sequence:
                samples[name].append(
                    measure_latency(
                        model,
                        device,
                        input_size=224,
                        batch_size=batch_size,
                        warmup=10,
                        repeats=30,
                    )["latency_ms_per_batch"]
                )
        base_mean = statistics.mean(samples["baseline"])
        eca_mean = statistics.mean(samples["eca_non_se"])
        report["batch_sizes"][str(batch_size)] = {
            "samples_ms": samples,
            "baseline_mean_ms": base_mean,
            "eca_mean_ms": eca_mean,
            "delta_ms": eca_mean - base_mean,
            "delta_percent": (eca_mean - base_mean) / base_mean * 100,
        }
        print(
            f"bs={batch_size}: baseline {base_mean:.2f} ms, eca {eca_mean:.2f} ms, "
            f"delta {eca_mean - base_mean:+.2f} ms ({(eca_mean - base_mean) / base_mean * 100:+.1f}%)"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
