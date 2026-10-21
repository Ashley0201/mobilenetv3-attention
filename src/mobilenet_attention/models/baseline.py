"""模型构建：标准 Torchvision MobileNetV3-Large 基线。

本阶段不插入任何注意力模块。预训练权重由配置显式选择：
- "imagenet1k_v1" / "imagenet1k_v2"：真实 ImageNet 预训练权重；
- null：随机初始化（报告中必须如实标注，不能称为预训练）。
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)

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
) -> nn.Module:
    """构建 MobileNetV3-Large，替换分类头，输出原始 logits [B, num_classes]。"""
    weights = resolve_weights(pretrained)
    model = mobilenet_v3_large(weights=weights)

    in_features = model.classifier[3].in_features
    # 只替换最后一层 Linear，保留原分类头中的 Dropout 与隐藏层结构
    model.classifier[3] = nn.Linear(in_features, num_classes)

    if dropout != model.classifier[2].p:
        model.classifier[2] = nn.Dropout(p=dropout)

    model.num_classes = num_classes
    model.pretrained = pretrained if weights is not None else "none"
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
    }
