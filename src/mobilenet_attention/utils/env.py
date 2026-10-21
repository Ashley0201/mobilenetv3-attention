"""运行环境快照，用于实验结果追溯。"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def _run_git(project_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_commit(project_root: Path) -> str | None:
    return _run_git(project_root, "rev-parse", "HEAD")


def _git_dirty(project_root: Path) -> bool | None:
    status = _run_git(project_root, "status", "--porcelain")
    if status is None:
        return None
    return bool(status)


def _nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def environment_snapshot(project_root: Path | None = None) -> dict[str, Any]:
    """采集真实环境信息；未安装 torchvision 时如实记录为 None。"""
    import torchvision

    snapshot: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "nvidia_driver": _nvidia_driver_version(),
        "deterministic": {
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        snapshot["gpu"] = {
            "name": props.name,
            "total_memory_mib": props.total_memory // (1024**2),
            "capability": f"{props.major}.{props.minor}",
        }
    if project_root is not None:
        root = Path(project_root)
        snapshot["git_commit"] = _git_commit(root)
        snapshot["git_dirty"] = _git_dirty(root)
    return snapshot
