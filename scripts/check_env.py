"""采集并输出真实环境信息，用于阶段 1 环境审计。

用法：
    python scripts/check_env.py            # 打印报告
    python scripts/check_env.py --json     # 输出 JSON，便于机器读取
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], timeout: int = 15) -> str | None:
    """执行外部命令并返回 stdout，失败或不存在时返回 None。"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_system_info() -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "in_virtualenv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "project_root": str(PROJECT_ROOT),
    }


def get_git_info() -> dict[str, Any]:
    version = _run(["git", "--version"])
    inside = _run(["git", "rev-parse", "--is-inside-work-tree"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run(["git", "rev-parse", "--short", "HEAD"])
    return {
        "git_version": version,
        "is_repo": inside == "true",
        "branch": branch,
        "commit": commit,
    }


def get_gpu_info() -> list[dict[str, Any]]:
    """通过 nvidia-smi 查询 GPU 名称、显存与驱动版本，不猜测。"""
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        timeout=20,
    )
    if not output:
        return []
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "name": parts[0],
                "driver_version": parts[1],
                "memory_total_mib": int(parts[2]),
                "memory_used_mib": int(parts[3]),
                "compute_capability": parts[4],
            }
        )
    return gpus


def get_torch_info() -> dict[str, Any]:
    """采集 PyTorch/Torchvision 与 CUDA 状态；未安装时如实报告。"""
    info: dict[str, Any] = {
        "torch_installed": False,
        "torchvision_installed": False,
        "cuda_available": False,
    }
    try:
        import torch
    except ImportError:
        info["error"] = "torch 未安装"
        return info

    info["torch_installed"] = True
    info["torch_version"] = torch.__version__
    info["torch_cuda_version"] = torch.version.cuda
    info["cudnn_version"] = torch.backends.cudnn.version()
    info["cuda_available"] = torch.cuda.is_available()
    info["device_count"] = torch.cuda.device_count()

    if info["cuda_available"]:
        devices = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_mib": props.total_memory // (1024**2),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
        info["cuda_devices"] = devices
    else:
        info["cuda_reason"] = _cuda_unavailable_reason(torch)

    try:
        import torchvision

        info["torchvision_installed"] = True
        info["torchvision_version"] = torchvision.__version__
    except ImportError:
        pass
    return info


def _cuda_unavailable_reason(torch: Any) -> str:
    if torch.version.cuda is None:
        return "当前 PyTorch 为 CPU 版本"
    return "PyTorch 为 CUDA 版本，但未检测到可用 CUDA 设备"


def get_disk_info() -> dict[str, Any]:
    usage = shutil.disk_usage(PROJECT_ROOT)
    return {
        "total_gib": round(usage.total / 1024**3, 1),
        "free_gib": round(usage.free / 1024**3, 1),
    }


def collect() -> dict[str, Any]:
    return {
        "system": get_system_info(),
        "git": get_git_info(),
        "gpu": get_gpu_info(),
        "torch": get_torch_info(),
        "disk": get_disk_info(),
    }


def format_report(info: dict[str, Any]) -> str:
    system = info["system"]
    git = info["git"]
    torch_info = info["torch"]
    disk = info["disk"]

    lines = [
        "=" * 64,
        "环境检查报告",
        "=" * 64,
        f"项目根目录   : {system['project_root']}",
        f"操作系统     : {system['os']}",
        f"处理器       : {system['processor']}",
        f"Python       : {system['python_version']} ({system['python_executable']})",
        f"虚拟环境     : {'是' if system['in_virtualenv'] else '否'}",
        f"Git          : {git['git_version'] or '未找到'}",
        f"Git 仓库     : {'是' if git['is_repo'] else '否'}"
        f" (分支 {git['branch'] or 'N/A'}, 提交 {git['commit'] or 'N/A'})",
        f"磁盘剩余     : {disk['free_gib']} GiB / {disk['total_gib']} GiB",
        "",
        "GPU (nvidia-smi):",
    ]
    if info["gpu"]:
        for gpu in info["gpu"]:
            lines.append(
                f"  - {gpu['name']} | 驱动 {gpu['driver_version']} | "
                f"显存 {gpu['memory_used_mib']}/{gpu['memory_total_mib']} MiB | "
                f"算力 {gpu['compute_capability']}"
            )
    else:
        lines.append("  - 未检测到 NVIDIA GPU 或 nvidia-smi 不可用")

    lines.append("")
    lines.append("PyTorch:")
    if torch_info.get("torch_installed"):
        lines.append(f"  torch         : {torch_info['torch_version']}")
        lines.append(f"  torchvision   : {torch_info.get('torchvision_version', '未安装')}")
        lines.append(f"  CUDA 编译版本 : {torch_info.get('torch_cuda_version') or 'CPU 版'}")
        lines.append(f"  cuDNN         : {torch_info.get('cudnn_version') or 'N/A'}")
        lines.append(f"  CUDA 可用     : {'是' if torch_info['cuda_available'] else '否'}")
        if torch_info["cuda_available"]:
            for dev in torch_info["cuda_devices"]:
                lines.append(
                    f"  设备 {dev['index']}      : {dev['name']} | "
                    f"{dev['total_memory_mib']} MiB | 算力 {dev['capability']}"
                )
        elif "cuda_reason" in torch_info:
            lines.append(f"  原因          : {torch_info['cuda_reason']}")
    else:
        lines.append(f"  {torch_info.get('error', 'torch 未安装')}")

    lines.append("=" * 64)
    return "\n".join(lines)


def main() -> int:
    # Windows 管道/重定向默认使用 GBK，统一为 UTF-8 避免中文乱码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="检查项目运行环境")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    args = parser.parse_args()

    info = collect()
    print(json.dumps(info, ensure_ascii=False, indent=2) if args.json else format_report(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
