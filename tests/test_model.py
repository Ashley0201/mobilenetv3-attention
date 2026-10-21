"""基线模型测试：结构、形状、梯度与预训练选择。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mobilenet_attention.models import (
    build_mobilenet_v3_large,
    model_summary,
    resolve_weights,
)


def test_forward_shape_and_logits():
    model = build_mobilenet_v3_large(num_classes=10, pretrained="none")
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()
    # 输出必须是原始 logits：不满足概率和为 1
    assert not torch.allclose(logits.sum(dim=1), torch.ones(2), atol=1e-3)


def test_classifier_head_replaced():
    model = build_mobilenet_v3_large(num_classes=7, pretrained="none")
    head = model.classifier[3]
    assert isinstance(head, nn.Linear)
    assert head.out_features == 7


def test_parameter_count_is_reported():
    model = build_mobilenet_v3_large(num_classes=10, pretrained="none")
    summary = model_summary(model)
    assert summary["total_parameters"] == sum(p.numel() for p in model.parameters())
    assert summary["trainable_parameters"] == summary["total_parameters"]
    assert summary["pretrained"] == "none"


def test_unknown_pretrained_name_raises():
    with pytest.raises(ValueError):
        resolve_weights("imagenet-fake")


def test_pretrained_none_means_random_init():
    torch.manual_seed(0)
    first = build_mobilenet_v3_large(num_classes=10, pretrained="none")
    torch.manual_seed(1)
    second = build_mobilenet_v3_large(num_classes=10, pretrained="none")
    with torch.no_grad():
        diff = (first.features[0][0].weight - second.features[0][0].weight).abs().max()
    assert diff > 0


def test_backward_produces_finite_gradients():
    model = build_mobilenet_v3_large(num_classes=10, pretrained="none")
    criterion = nn.CrossEntropyLoss()
    logits = model(torch.randn(2, 3, 224, 224))
    loss = criterion(logits, torch.tensor([1, 9]))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_pretrained_weights_are_really_loaded():
    """确认 imagenet1k_v1 权重来自官方缓存文件，而非随机初始化。"""
    cache = Path(torch.hub.get_dir()) / "checkpoints" / "mobilenet_v3_large-8738ca79.pth"
    if not cache.is_file():
        pytest.skip("官方预训练权重未缓存，先运行一次 imagenet1k_v1 构建")

    reference = torch.load(cache, map_location="cpu", weights_only=True)
    model = build_mobilenet_v3_large(num_classes=10, pretrained="imagenet1k_v1")
    first_conv = model.features[0][0].weight.detach().cpu()
    assert torch.equal(first_conv, reference["features.0.0.weight"])

    random_model = build_mobilenet_v3_large(num_classes=10, pretrained="none")
    random_conv = random_model.features[0][0].weight.detach().cpu()
    assert not torch.allclose(first_conv, random_conv)
