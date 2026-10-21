"""数据集与划分测试（依赖本地已下载的 CIFAR-10）。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mobilenet_attention.data import (
    CIFAR10_CLASSES,
    build_cifar10_datasets,
    build_dataloaders,
    build_eval_transform,
    build_train_transform,
    dataset_summary,
    stratified_split,
    targets_tensor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


@pytest.fixture(scope="module")
def datasets():
    if not (DATA_ROOT / "cifar-10-batches-py").is_dir():
        pytest.skip("本地未找到 CIFAR-10，先运行 scripts/download_cifar10.py")
    return build_cifar10_datasets(
        root=DATA_ROOT, input_size=64, val_size=1000, split_seed=42, download=False
    )


def test_class_mapping_matches_official(datasets):
    assert datasets["class_names"] == CIFAR10_CLASSES
    assert datasets["class_to_idx"]["airplane"] == 0
    assert datasets["class_to_idx"]["truck"] == 9


def test_split_sizes_and_disjoint(datasets):
    summary = dataset_summary(datasets)
    assert summary["train_size"] == 49000
    assert summary["val_size"] == 1000
    assert summary["test_size"] == 10000

    train_idx = set(datasets["train_indices"])
    val_idx = set(datasets["val_indices"])
    assert not (train_idx & val_idx)
    assert train_idx | val_idx == set(range(50000))


def test_stratified_split_is_deterministic():
    targets = [i % 10 for i in range(5000)]
    first = stratified_split(targets, val_size=500, seed=42)
    second = stratified_split(targets, val_size=500, seed=42)
    assert first == second


def test_transforms_output_shape_and_range():
    train_transform = build_train_transform(64)
    eval_transform = build_eval_transform(64)
    from PIL import Image

    image = Image.new("RGB", (32, 32), color=(128, 128, 128))
    for transform in (train_transform, eval_transform):
        tensor = transform(image)
        assert tensor.shape == (3, 64, 64)
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()


def test_dataset_items_are_tensors(datasets):
    image, label = datasets["train"][0]
    assert image.shape == (3, 64, 64)
    assert isinstance(label, int)
    assert 0 <= label <= 9


def test_train_and_val_use_different_transforms(datasets):
    assert datasets["train"].transform is not datasets["val"].transform


def test_class_balance_preserved(datasets):
    train_counts = targets_tensor(datasets["train"]).bincount()
    val_counts = targets_tensor(datasets["val"]).bincount()
    assert len(train_counts) == 10
    assert len(val_counts) == 10
    assert train_counts.sum() == 49000
    assert val_counts.sum() == 1000
    # 分层抽样后验证集每类数量应接近 100
    assert val_counts.min() >= 95
    assert val_counts.max() <= 105


def test_dataloader_batches(datasets):
    loaders = build_dataloaders(
        datasets, batch_size=8, eval_batch_size=16, num_workers=0, pin_memory=False
    )
    images, labels = next(iter(loaders["train"]))
    assert images.shape == (8, 3, 64, 64)
    assert labels.shape == (8,)
    assert labels.dtype == torch.int64
