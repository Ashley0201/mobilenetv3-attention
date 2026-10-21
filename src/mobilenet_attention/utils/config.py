"""配置加载与快照工具。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置，并记录来源路径（以下划线开头，不写入快照）。"""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件格式错误：{config_path}")
    config["_config_path"] = str(config_path)
    return config


def set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """按 a.b.c 形式写入嵌套配置。"""
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """返回应用覆盖项后的新配置，不修改输入。"""
    merged = deepcopy(config)
    for key, value in overrides.items():
        if value is not None:
            set_by_dotted_key(merged, key, value)
    return merged


def save_config_snapshot(config: dict[str, Any], path: str | Path) -> Path:
    """保存去除内部字段后的配置快照，用于实验追溯。"""
    snapshot = {k: v for k, v in config.items() if not k.startswith("_")}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, allow_unicode=True, sort_keys=False)
    return target
