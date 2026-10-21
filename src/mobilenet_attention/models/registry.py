"""模型注册表：配置中的 model.name 决定实际构建的模型。

支持名称：
- mobilenet_v3_large      标准 MobileNetV3-Large（不含额外注意力）
- mobilenet_v3_large_eca  MobileNetV3-Large + ECA（插入位置由 attention 配置决定）

未知名称直接报错，不会静默退回默认模型。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from .baseline import build_mobilenet_v3_large

SUPPORTED_MODELS = ("mobilenet_v3_large", "mobilenet_v3_large_eca")


def _build_plain(
    num_classes: int,
    pretrained: str | None,
    dropout: float,
    attention: dict[str, Any] | None,
) -> nn.Module:
    if attention:
        raise ValueError(
            "model.name=mobilenet_v3_large 不接受 attention 配置；"
            "请改用 model.name=mobilenet_v3_large_eca"
        )
    return build_mobilenet_v3_large(
        num_classes=num_classes, pretrained=pretrained, dropout=dropout, attention=None
    )


def _build_eca(
    num_classes: int,
    pretrained: str | None,
    dropout: float,
    attention: dict[str, Any] | None,
) -> nn.Module:
    config = dict(attention or {})
    config.setdefault("type", "eca")
    config.setdefault("placement", "non_se_blocks")
    if str(config["type"]).lower() != "eca":
        raise ValueError(f"mobilenet_v3_large_eca 只接受 type=eca，收到 {config['type']}")
    return build_mobilenet_v3_large(
        num_classes=num_classes, pretrained=pretrained, dropout=dropout, attention=config
    )


MODEL_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "mobilenet_v3_large": _build_plain,
    "mobilenet_v3_large_eca": _build_eca,
}


def validate_model_config(name: str, attention: dict[str, Any] | None = None) -> str:
    """在创建输出目录、加载数据之前校验模型配置，避免留下半成品实验目录。"""
    key = str(name or "").strip().lower()
    if key not in MODEL_BUILDERS:
        raise ValueError(f"未知模型名称：{name}，可选 {SUPPORTED_MODELS}")
    if key == "mobilenet_v3_large" and attention:
        raise ValueError(
            "model.name=mobilenet_v3_large 不接受 attention 配置；"
            "请改用 model.name=mobilenet_v3_large_eca"
        )
    if (
        key == "mobilenet_v3_large_eca"
        and attention
        and str(attention.get("type", "eca")).lower() != "eca"
    ):
        raise ValueError(f"mobilenet_v3_large_eca 只接受 type=eca，收到 {attention.get('type')}")
    return key


def build_model(
    name: str,
    num_classes: int,
    pretrained: str | None = None,
    dropout: float = 0.2,
    attention: dict[str, Any] | None = None,
) -> nn.Module:
    """按名称构建模型；名称未知或配置不匹配时抛出 ValueError。"""
    key = str(name or "").strip().lower()
    if key not in MODEL_BUILDERS:
        raise ValueError(f"未知模型名称：{name}，可选 {SUPPORTED_MODELS}")
    model = MODEL_BUILDERS[key](
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        attention=attention,
    )
    model.model_name = key
    return model


def describe_model(model: nn.Module) -> dict[str, Any]:
    """运行时模型身份，用于日志与测试比对。"""
    attention = getattr(model, "attention", None)
    return {
        "model_name": getattr(model, "model_name", None),
        "class": type(model).__name__,
        "num_classes": getattr(model, "num_classes", None),
        "pretrained": getattr(model, "pretrained", None),
        "attention_type": attention.get("type") if attention else None,
        "attention_placement": attention.get("placement") if attention else None,
        "attention_blocks": len(attention["inserted_blocks"]) if attention else 0,
        "attention_params": attention.get("total_params", 0) if attention else 0,
    }
