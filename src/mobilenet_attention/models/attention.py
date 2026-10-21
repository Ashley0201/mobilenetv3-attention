"""注意力模块插入逻辑。

当前仅支持 ECA，且默认不改变基线行为：只有显式给出 attention 配置时才插入。
插入位置与 Torchvision 的 SqueezeExcitation 对齐——深度卷积之后、投影卷积之前。
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision.models.mobilenetv3 import InvertedResidual
from torchvision.ops.misc import SqueezeExcitation

from .eca import ECALayer, eca_kernel_size

PLACEMENTS = ("non_se_blocks", "se_blocks_only", "all_blocks")


def _is_se_block(layer: InvertedResidual) -> bool:
    return any(isinstance(child, SqueezeExcitation) for child in layer.block)


def _first_conv(module: nn.Module) -> nn.Conv2d | None:
    """取 Conv2dNormActivation 中的首个 Conv2d（该容器是 Sequential，无 .conv 属性）。"""
    if isinstance(module, nn.Conv2d):
        return module
    if isinstance(module, nn.Sequential):
        first = module[0] if len(module) > 0 else None
        if isinstance(first, nn.Conv2d):
            return first
    return None


def _depthwise_index(layer: InvertedResidual) -> int:
    """定位深度卷积：按 groups>1 的 Conv2d 判定，兼容有无 expand 卷积的块。"""
    for index, child in enumerate(layer.block):
        conv = _first_conv(child)
        if conv is not None and conv.groups > 1:
            return index
    raise RuntimeError("未找到深度卷积层，无法确定 ECA 插入位置")


def insert_eca(
    model: nn.Module,
    placement: str = "non_se_blocks",
    gamma: float = 2.0,
    b: float = 1.0,
    kernel_size: int | None = None,
) -> dict[str, Any]:
    """在 MobileNetV3 的 InvertedResidual 中插入 ECA，返回插入信息。

    必须在加载预训练权重之后调用，否则会改变 state_dict 的键顺序。
    """
    if placement not in PLACEMENTS:
        raise ValueError(f"未知插入位置：{placement}，可选 {PLACEMENTS}")

    inserted: list[dict[str, Any]] = []
    for index, layer in enumerate(model.features):
        if not isinstance(layer, InvertedResidual):
            continue
        has_se = _is_se_block(layer)
        if placement == "non_se_blocks" and has_se:
            continue
        if placement == "se_blocks_only" and not has_se:
            continue

        position = _depthwise_index(layer) + 1
        # 深度卷积输出通道数即 ECA 的通道数
        depthwise = _first_conv(layer.block[position - 1])
        assert depthwise is not None
        channels = depthwise.out_channels
        k = kernel_size if kernel_size is not None else eca_kernel_size(channels, gamma=gamma, b=b)

        eca = ECALayer(channels, gamma=gamma, b=b, kernel_size=k)
        children = list(layer.block.children())
        children.insert(position, eca)
        layer.block = nn.Sequential(*children)
        inserted.append(
            {"feature_index": index, "position": position, "channels": channels, "kernel_size": k}
        )

    if not inserted:
        raise RuntimeError(f"未插入任何 ECA 模块，placement={placement}")

    model.attention = {
        "type": "eca",
        "placement": placement,
        "gamma": gamma,
        "b": b,
        "inserted_blocks": inserted,
        "total_params": sum(
            p.numel() for m in model.modules() if isinstance(m, ECALayer) for p in m.parameters()
        ),
    }
    return model.attention
