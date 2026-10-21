"""ECA（Efficient Channel Attention）模块。

参考：Wang et al., "ECA-Net: Efficient Channel Attention for Deep Convolutional
Neural Networks", CVPR 2020, https://arxiv.org/abs/1910.03151

核心操作（输入 X ∈ R^{C×H×W}）：
1. 全局平均池化得到 z ∈ R^C；
2. 用 1D 卷积实现局部跨通道交互：s = sigmoid(Conv1d_k(z))；
3. 通道重标定：Y = X ⊙ s。

与 SE 的区别：不做通道降维，参数量仅为 1D 卷积的 k 个权重（无 bias）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def eca_kernel_size(channels: int, gamma: float = 2.0, b: float = 1.0) -> int:
    """按论文公式自适应选择 1D 卷积核大小，并保证为奇数。

    k = |log2(C)/gamma + b/gamma|_odd
    """
    if channels < 2:
        raise ValueError(f"ECA 需要通道数 >= 2，收到 {channels}")
    t = int(abs(math.log2(channels) / gamma + b / gamma))
    return t + 1 if t % 2 == 0 else t


class ECALayer(nn.Module):
    """通道注意力：GAP -> 1D 卷积 -> Sigmoid -> 逐通道相乘。"""

    def __init__(
        self,
        channels: int,
        gamma: float = 2.0,
        b: float = 1.0,
        kernel_size: int | None = None,
    ) -> None:
        super().__init__()
        if kernel_size is None:
            kernel_size = eca_kernel_size(channels, gamma=gamma, b=b)
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"ECA 卷积核必须为正奇数，收到 {kernel_size}")

        self.channels = channels
        self.kernel_size = kernel_size
        self.gamma = gamma
        self.b = b

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D 卷积覆盖相邻 k 个通道，padding 保证长度不变；无 bias
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = self.activation(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, kernel_size={self.kernel_size}"
