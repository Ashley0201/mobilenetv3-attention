"""配置工具测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mobilenet_attention.utils.config import (
    apply_overrides,
    load_config,
    parse_scalar,
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
    assert merged["training"]["epochs"] == 3
    assert config["training"]["epochs"] == 10


def test_parse_scalar_types():
    assert parse_scalar("false") is False
    assert parse_scalar("true") is True
    assert parse_scalar("null") is None
    assert parse_scalar("0.0003") == 0.0003
    assert parse_scalar("42") == 42
    assert parse_scalar("eca") == "eca"


def test_override_bool_false_is_real_false():
    """回归测试：旧实现 bool("false") 为 True，会把 AMP 误开。"""
    config = {"training": {"amp": True}}
    merged = apply_overrides(config, {"training.amp": "false"})
    assert merged["training"]["amp"] is False
    assert bool(merged["training"]["amp"]) is False


def test_override_null_becomes_none():
    """回归测试：旧实现 grad_clip=null 会变成字符串 "null"。"""
    config = {"training": {"grad_clip": 1.0}}
    merged = apply_overrides(config, {"training.grad_clip": "null"})
    assert merged["training"]["grad_clip"] is None
    assert not isinstance(merged["training"]["grad_clip"], str)


def test_override_integer_and_float():
    config = {"training": {"epochs": 15, "learning_rate": 3e-4}}
    merged = apply_overrides(config, {"training.epochs": "3", "training.learning_rate": "0.001"})
    assert merged["training"]["epochs"] == 3
    assert isinstance(merged["training"]["epochs"], int)
    assert merged["training"]["learning_rate"] == 0.001
    assert isinstance(merged["training"]["learning_rate"], float)


def test_override_string_value_stays_string():
    config = {"model": {"pretrained": "imagenet1k_v1"}}
    merged = apply_overrides(config, {"model.pretrained": "none"})
    assert merged["model"]["pretrained"] == "none"


def test_override_new_nested_key():
    config = {"model": {"num_classes": 10}}
    merged = apply_overrides(config, {"model.attention.type": "eca"})
    assert merged["model"]["attention"] == {"type": "eca"}


def test_set_by_dotted_key_creates_nested():
    config: dict = {}
    set_by_dotted_key(config, "a.b.c", 1)
    assert config == {"a": {"b": {"c": 1}}}


def test_snapshot_excludes_internal_keys(tmp_path: Path):
    config = {"experiment": {"id": "x"}, "_config_path": "secret"}
    path = save_config_snapshot(config, tmp_path / "snap.yaml")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded == {"experiment": {"id": "x"}}


def test_snapshot_preserves_types(tmp_path: Path):
    config = {"training": {"amp": False, "grad_clip": None, "epochs": 3}}
    path = save_config_snapshot(config, tmp_path / "snap.yaml")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["training"]["amp"] is False
    assert loaded["training"]["grad_clip"] is None
    assert loaded["training"]["epochs"] == 3


def test_invalid_config_file(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)
