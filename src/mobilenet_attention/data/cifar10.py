"""CIFAR-10 数据集构建。

划分策略（固定、可复现）：
- 官方 train 50000 张 -> 分层抽取 5000 张作为验证集，其余 45000 张训练；
- 官方 test 10000 张仅用于最终独立评估，不参与任何调参。

图像从 32x32 上采样到 ImageNet 预训练权重期望的输入尺寸（默认 224），
归一化使用 ImageNet 均值/标准差，与 Torchvision 预训练协议一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(
    input_size: int = 224, scale: tuple[float, float] = (0.6, 1.0), horizontal_flip: bool = True
) -> transforms.Compose:
    """训练增强：随机裁剪缩放 + 水平翻转 + ImageNet 归一化。"""
    ops: list[Any] = [
        transforms.RandomResizedCrop(input_size, scale=scale, antialias=True),
    ]
    if horizontal_flip:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(ops)


def build_eval_transform(input_size: int = 224) -> transforms.Compose:
    """评估：整图缩放到输入尺寸，不做随机增强。"""
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def stratified_split(targets: list[int], val_size: int, seed: int) -> tuple[list[int], list[int]]:
    """按类别分层划分训练/验证索引，保证验证集类别分布与训练集一致。"""
    indices = list(range(len(targets)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=targets,
        shuffle=True,
    )
    return sorted(train_idx), sorted(val_idx)


class TransformSubset(Dataset):
    """索引子集 + 独立 transform。

    torchvision 的 Subset 不携带 transform，直接修改底层 CIFAR10 的 transform 会让
    train/val 共享同一份设置。这里显式保存 transform，保证两者互不影响。
    """

    def __init__(self, dataset: Dataset, indices: list[int], transform: Any) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.targets = [dataset.targets[i] for i in indices]  # type: ignore[attr-defined]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, target = self.dataset[self.indices[index]]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_cifar10_datasets(
    root: str | Path,
    input_size: int = 224,
    val_size: int = 5000,
    split_seed: int = 42,
    download: bool = False,
    augmentation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建 train / val / test 三个数据集，返回数据集与划分信息。"""
    root = Path(root)
    augmentation = augmentation or {}
    scale = tuple(augmentation.get("random_resized_crop_scale", (0.6, 1.0)))
    flip = bool(augmentation.get("horizontal_flip", True))

    train_transform = build_train_transform(input_size, scale=scale, horizontal_flip=flip)
    eval_transform = build_eval_transform(input_size)

    # 同一份数据按需切换 transform，避免重复读取磁盘
    train_full = CIFAR10(root=root, train=True, download=download, transform=None)
    test_set = CIFAR10(root=root, train=False, download=download, transform=eval_transform)

    train_idx, val_idx = stratified_split(train_full.targets, val_size, split_seed)

    train_set = TransformSubset(train_full, train_idx, train_transform)
    val_set = TransformSubset(train_full, val_idx, eval_transform)

    return {
        "train": train_set,
        "val": val_set,
        "test": test_set,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "class_names": list(train_full.classes),
        "class_to_idx": dict(train_full.class_to_idx),
    }


def build_dataloaders(
    datasets: dict[str, Any],
    batch_size: int = 128,
    eval_batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """构建 DataLoader；Windows 下需由脚本入口的 __main__ 保护。"""
    common = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    return {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True, drop_last=False, **common
        ),
        "val": DataLoader(datasets["val"], batch_size=eval_batch_size, shuffle=False, **common),
        "test": DataLoader(datasets["test"], batch_size=eval_batch_size, shuffle=False, **common),
    }


def dataset_summary(datasets: dict[str, Any]) -> dict[str, Any]:
    """统计划分规模与类别分布，用于实验记录。"""
    train_targets = list(datasets["train"].targets)
    val_targets = list(datasets["val"].targets)
    return {
        "train_size": len(datasets["train"]),
        "val_size": len(datasets["val"]),
        "test_size": len(datasets["test"]),
        "class_names": datasets["class_names"],
        "class_to_idx": datasets["class_to_idx"],
        "train_class_counts": _counts(train_targets, len(datasets["class_names"])),
        "val_class_counts": _counts(val_targets, len(datasets["class_names"])),
    }


def _counts(targets: list[int], num_classes: int) -> list[int]:
    counts = [0] * num_classes
    for t in targets:
        counts[t] += 1
    return counts


def targets_tensor(dataset: TransformSubset) -> torch.Tensor:
    """返回子集对应的标签张量，便于测试检查。"""
    return torch.tensor(dataset.targets)
