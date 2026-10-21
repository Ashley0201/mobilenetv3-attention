"""模型构建：标准 Torchvision MobileNetV3-Large 基线（可选 ECA）。

预训练权重由配置显式选择：
- "imagenet1k_v1" / "imagenet1k_v2"：真实 ImageNet 预训练权重；
- null / "none"：随机初始化（报告中必须如实标注，不能称为预训练）。

注意力模块默认不插入；只有传入 attention 配置时才在预训练权重加载完成后插入，
避免改变 state_dict 键名导致权重加载失败。
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)

from .attention import insert_eca

WEIGHT_CHOICES: dict[str, MobileNet_V3_Large_Weights | None] = {
    "imagenet1k_v1": MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    "imagenet1k_v2": MobileNet_V3_Large_Weights.IMAGENET1K_V2,
    "none": None,
}


def resolve_weights(name: str | None) -> MobileNet_V3_Large_Weights | None:
    """把配置字符串映射为 Torchvision 权重枚举，未知取值直接报错。"""
    key = (name or "none").lower()
    if key not in WEIGHT_CHOICES:
        raise ValueError(f"未知预训练权重选项：{name}，可选 {sorted(WEIGHT_CHOICES)}")
    return WEIGHT_CHOICES[key]


def build_mobilenet_v3_large(
    num_classes: int,
    pretrained: str | None = None,
    dropout: float = 0.2,
    attention: dict[str, Any] | None = None,
) -> nn.Module:
    """构建 MobileNetV3-Large，替换分类头，输出原始 logits [B, num_classes]。

    attention 为 None 时保持标准基线结构；给出时按配置插入 ECA。
    """
    weights = resolve_weights(pretrained)
    model = mobilenet_v3_large(weights=weights)

    in_features = model.classifier[3].in_features
    # 只替换最后一层 Linear，保留原分类头中的 Dropout 与隐藏层结构
    model.classifier[3] = nn.Linear(in_features, num_classes)

    if dropout != model.classifier[2].p:
        model.classifier[2] = nn.Dropout(p=dropout)

    model.num_classes = num_classes
    model.pretrained = pretrained if weights is not None else "none"

    if attention:
        attention_type = str(attention.get("type", "")).lower()
        if attention_type != "eca":
            raise ValueError(f"不支持的注意力类型：{attention.get('type')}")
        model.attention = insert_eca(
            model,
            placement=str(attention.get("placement", "non_se_blocks")),
            gamma=float(attention.get("gamma", 2.0)),
            b=float(attention.get("b", 1.0)),
            kernel_size=attention.get("kernel_size"),
        )
    return model


def model_summary(model: nn.Module) -> dict[str, Any]:
    """参数统计，用于实验记录。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "pretrained": getattr(model, "pretrained", "unknown"),
        "num_classes": getattr(model, "num_classes", None),
        "attention": getattr(model, "attention", None),
    }
