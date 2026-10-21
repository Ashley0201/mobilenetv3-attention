"""随机种子与设备选择。"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """固定 Python / NumPy / PyTorch 随机种子。

    deterministic=True 时启用 cuDNN 确定性算法。部分 CUDA 算子没有确定性实现，
    因此使用 warn_only，并在实验报告中披露该限制。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str = "auto") -> torch.device:
    """解析设备；显式要求 cuda 但不可用时直接报错，避免静默退回 CPU。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但 torch.cuda.is_available() 为 False")
    return torch.device(requested)
