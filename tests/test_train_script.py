"""训练脚本辅助函数测试（不启动正式训练）。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train.py"


@pytest.fixture(scope="module")
def train_script():
    spec = importlib.util.spec_from_file_location("train_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_overrides(train_script):
    result = train_script.parse_overrides(["training.epochs=3", "model.pretrained=none"])
    assert result == {"training.epochs": "3", "model.pretrained": "none"}


def test_parse_overrides_rejects_invalid(train_script):
    with pytest.raises(ValueError):
        train_script.parse_overrides(["training.epochs"])


def test_prepare_output_dir_refuses_existing(train_script, tmp_path: Path):
    target = tmp_path / "exp"
    target.mkdir()
    (target / "best.pt").write_text("old", encoding="utf-8")
    with pytest.raises(SystemExit):
        train_script.prepare_output_dir(tmp_path, "exp", overwrite=False)


def test_prepare_output_dir_allows_overwrite(train_script, tmp_path: Path):
    target = tmp_path / "exp"
    target.mkdir()
    (target / "best.pt").write_text("old", encoding="utf-8")
    result = train_script.prepare_output_dir(tmp_path, "exp", overwrite=True)
    assert result == target


def test_resolve_data_config_reads_dataset_section(train_script):
    config = {"data": {"config": "configs/dataset_cifar10.yaml"}}
    dataset_config = train_script.resolve_data_config(config)
    assert dataset_config["name"] == "cifar10"
    assert Path(dataset_config["root"]).is_absolute()


def test_limited_loader_truncates(train_script):
    loader = [(i, i) for i in range(10)]
    limited = train_script._LimitedLoader(loader, 3)
    assert len(limited) == 3
    assert list(limited) == [(0, 0), (1, 1), (2, 2)]
