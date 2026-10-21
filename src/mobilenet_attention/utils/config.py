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


def parse_scalar(raw: str) -> Any:
    """把命令行字符串解析为 YAML 标量，修复布尔与 null 的类型错误。

    yaml.safe_load("false") -> False，yaml.safe_load("null") -> None。
    未加引号的普通字符串仍保持字符串，例如 "eca" -> "eca"。
    """
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        # 非法 YAML 标量（例如含冒号未加引号）按原始字符串处理
        return raw


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """返回应用覆盖项后的新配置，不修改输入。

    覆盖值若为字符串，先按 YAML 标量解析，避免 "--set training.amp=false"
    被 bool("false") 判为 True，或 "grad_clip=null" 变成字符串 "null"。
    键一旦出现即视为显式覆盖，因此 null 会真正写入 None。
    """
    merged = deepcopy(config)
    for key, value in overrides.items():
        resolved = parse_scalar(value) if isinstance(value, str) else value
        set_by_dotted_key(merged, key, resolved)
    return merged


def save_config_snapshot(config: dict[str, Any], path: str | Path) -> Path:
    """保存去除内部字段后的配置快照，用于实验追溯。"""
    snapshot = {k: v for k, v in config.items() if not k.startswith("_")}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, allow_unicode=True, sort_keys=False)
    return target
