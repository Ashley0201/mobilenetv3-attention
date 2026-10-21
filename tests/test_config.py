"""配置工具测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mobilenet_attention.utils.config import (
    apply_overrides,
    load_config,
    save_config_snapshot,
    set_by_dotted_key,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_dataset_config():
    config = load_config(PROJECT_ROOT / "configs" / "dataset_cifar10.yaml")
    assert config["dataset"]["num_classes"] == 10
    assert config["dataset"]["split"]["val_size"] == 5000
    assert config["_config_path"].endswith("dataset_cifar10.yaml")


def test_apply_overrides_does_not_mutate_source():
    config = {"training": {"epochs": 10}}
    merged = apply_overrides(config, {"training.epochs": "3"})
    assert merged["training"]["epochs"] == "3"
    assert config["training"]["epochs"] == 10


def test_set_by_dotted_key_creates_nested():
    config: dict = {}
    set_by_dotted_key(config, "a.b.c", 1)
    assert config == {"a": {"b": {"c": 1}}}


def test_snapshot_excludes_internal_keys(tmp_path: Path):
    config = {"experiment": {"id": "x"}, "_config_path": "secret"}
    path = save_config_snapshot(config, tmp_path / "snap.yaml")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded == {"experiment": {"id": "x"}}


def test_invalid_config_file(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)
