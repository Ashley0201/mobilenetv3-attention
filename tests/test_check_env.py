"""环境检查脚本的基础测试。

仅验证采集逻辑的结构与真实性约束，不断言具体硬件型号。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "check_env.py"


@pytest.fixture(scope="module")
def check_env_module():
    spec = importlib.util.spec_from_file_location("check_env", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_exists():
    assert SCRIPT_PATH.is_file()


def test_system_info_keys(check_env_module):
    info = check_env_module.get_system_info()
    for key in ("os", "python_version", "python_executable", "in_virtualenv", "project_root"):
        assert key in info
    assert Path(info["project_root"]) == PROJECT_ROOT
    assert info["python_version"].startswith("3.")


def test_gpu_info_structure(check_env_module):
    gpus = check_env_module.get_gpu_info()
    assert isinstance(gpus, list)
    for gpu in gpus:
        assert gpu["name"]
        assert isinstance(gpu["memory_total_mib"], int)
        assert gpu["memory_total_mib"] > 0


def test_torch_info_reports_truthfully(check_env_module):
    info = check_env_module.get_torch_info()
    assert isinstance(info["cuda_available"], bool)
    if not info["torch_installed"]:
        assert info["cuda_available"] is False
        pytest.skip("torch 未安装")
    assert info["torch_version"]
    if info["cuda_available"]:
        assert info["cuda_devices"]
        assert all(d["total_memory_mib"] > 0 for d in info["cuda_devices"])
    else:
        assert "cuda_reason" in info


def test_format_report_contains_sections(check_env_module):
    report = check_env_module.format_report(check_env_module.collect())
    for section in ("环境检查报告", "GPU", "PyTorch", "Python"):
        assert section in report


def test_cli_json_output():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert set(payload) == {"system", "git", "gpu", "torch", "disk"}
