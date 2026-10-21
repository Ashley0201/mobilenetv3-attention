"""ECA 模块与插入逻辑测试。"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torchvision.models.mobilenetv3 import InvertedResidual

from mobilenet_attention.models import (
    ECALayer,
    build_mobilenet_v3_large,
    eca_kernel_size,
    insert_eca,
)

EXPECTED_NON_SE_BLOCKS = [1, 2, 3, 7, 8, 9, 10]
EXPECTED_SE_BLOCKS = [4, 5, 6, 11, 12, 13, 14, 15]


def test_kernel_size_matches_paper_formula():
    # k = |log2(C)/2 + 1/2|_odd
    expected = {16: 3, 24: 3, 64: 3, 72: 3, 80: 3, 184: 5, 200: 5, 240: 5, 960: 5}
    for channels, k in expected.items():
        assert eca_kernel_size(channels) == k


def test_kernel_size_is_odd_and_positive():
    for channels in range(2, 1025, 7):
        k = eca_kernel_size(channels)
        assert k >= 1
        assert k % 2 == 1


def test_kernel_size_rejects_invalid_channels():
    with pytest.raises(ValueError):
        eca_kernel_size(1)


def test_layer_preserves_shape():
    layer = ECALayer(64)
    x = torch.randn(2, 64, 28, 28)
    y = layer(x)
    assert y.shape == x.shape


def test_layer_output_is_scaled_input():
    """ECA 输出应为输入逐通道缩放，且缩放系数在 (0, 1)。"""
    layer = ECALayer(32)
    x = torch.randn(1, 32, 8, 8)
    y = layer(x)
    ratio = (y / x).detach()
    assert torch.allclose(ratio, ratio[:, :, :1, :1].expand_as(ratio), atol=1e-6)
    assert torch.all(ratio > 0)
    assert torch.all(ratio < 1)


def test_layer_parameter_count_equals_kernel_size():
    for channels in (16, 64, 240):
        layer = ECALayer(channels)
        assert sum(p.numel() for p in layer.parameters()) == layer.kernel_size


def test_layer_has_no_bias():
    layer = ECALayer(64)
    assert layer.conv.bias is None


def test_layer_gradients_flow():
    layer = ECALayer(32)
    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.conv.weight.grad is not None
    assert torch.isfinite(layer.conv.weight.grad).all()


def test_layer_zero_weight_gives_half_scale():
    """1D 卷积权重为 0 时 sigmoid(0)=0.5，输出应严格等于 0.5×输入。"""
    layer = ECALayer(32)
    with torch.no_grad():
        layer.conv.weight.zero_()
    x = torch.randn(2, 32, 8, 8)
    y = layer(x)
    assert torch.allclose(y, 0.5 * x, atol=1e-6)


def test_layer_handles_spatial_size_one():
    layer = ECALayer(16)
    x = torch.randn(1, 16, 1, 1)
    assert layer(x).shape == x.shape


def test_layer_rejects_even_kernel():
    with pytest.raises(ValueError):
        ECALayer(32, kernel_size=4)


def test_layer_fixed_kernel_overrides_formula():
    layer = ECALayer(240, kernel_size=3)
    assert layer.kernel_size == 3
    assert sum(p.numel() for p in layer.parameters()) == 3


def test_insert_eca_only_touches_non_se_blocks():
    model = build_mobilenet_v3_large(num_classes=10, pretrained="none")
    info = insert_eca(model, placement="non_se_blocks")
    assert [d["feature_index"] for d in info["inserted_blocks"]] == EXPECTED_NON_SE_BLOCKS
    assert info["total_params"] == 29

    for index, layer in enumerate(model.features):
        if not isinstance(layer, InvertedResidual):
            continue
        types = [type(c).__name__ for c in layer.block]
        eca_count = types.count("ECALayer")
        if index in EXPECTED_NON_SE_BLOCKS:
            assert eca_count == 1
            assert "SqueezeExcitation" not in types
        else:
            assert eca_count == 0
            assert "SqueezeExcitation" in types


def test_insert_eca_position_is_between_depthwise_and_projection():
    model = build_mobilenet_v3_large(num_classes=10, pretrained="none")
    info = insert_eca(model, placement="non_se_blocks")
    for entry in info["inserted_blocks"]:
        layer = model.features[entry["feature_index"]]
        types = [type(c).__name__ for c in layer.block]
        eca_position = types.index("ECALayer")
        # 插入位置之前必须是深度卷积（groups > 1）
        depthwise = layer.block[eca_position - 1][0]
        assert isinstance(depthwise, nn.Conv2d)
        assert depthwise.groups > 1
        # 之后必须是投影卷积（groups == 1）
        projection = layer.block[eca_position + 1][0]
        assert isinstance(projection, nn.Conv2d)
        assert projection.groups == 1
        assert entry["position"] == eca_position


def test_insert_eca_all_blocks():
    model = build_mobilenet_v3_large(num_classes=10, pretrained="none")
    info = insert_eca(model, placement="all_blocks")
    assert len(info["inserted_blocks"]) == 15


def test_insert_eca_se_blocks_only():
    model = build_mobilenet_v3_large(num_classes=10, pretrained="none")
    info = insert_eca(model, placement="se_blocks_only")
    assert [d["feature_index"] for d in info["inserted_blocks"]] == EXPECTED_SE_BLOCKS


def test_insert_eca_rejects_unknown_placement():
    model = build_mobilenet_v3_large(num_classes=10, pretrained="none")
    with pytest.raises(ValueError):
        insert_eca(model, placement="everywhere")


def test_parameter_delta_matches_analytic_estimate():
    """阶段 2 审计的解析值为 +29 参数、+790,064 MACs（此处只校验参数）。"""
    baseline = build_mobilenet_v3_large(num_classes=10, pretrained="none")
    eca_model = build_mobilenet_v3_large(
        num_classes=10, pretrained="none", attention={"type": "eca", "placement": "non_se_blocks"}
    )
    delta = sum(p.numel() for p in eca_model.parameters()) - sum(
        p.numel() for p in baseline.parameters()
    )
    assert delta == 29


def test_forward_shape_with_eca():
    model = build_mobilenet_v3_large(
        num_classes=10, pretrained="none", attention={"type": "eca", "placement": "non_se_blocks"}
    )
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()


def test_backward_with_eca():
    model = build_mobilenet_v3_large(
        num_classes=10, pretrained="none", attention={"type": "eca", "placement": "non_se_blocks"}
    )
    criterion = nn.CrossEntropyLoss()
    loss = criterion(model(torch.randn(2, 3, 224, 224)), torch.tensor([1, 9]))
    loss.backward()
    eca_grads = [p.grad for m in model.modules() if isinstance(m, ECALayer) for p in m.parameters()]
    assert len(eca_grads) == 7
    assert all(g is not None and torch.isfinite(g).all() for g in eca_grads)


def test_attention_metadata_recorded():
    model = build_mobilenet_v3_large(
        num_classes=10, pretrained="none", attention={"type": "eca", "placement": "non_se_blocks"}
    )
    assert model.attention["type"] == "eca"
    assert model.attention["placement"] == "non_se_blocks"
    assert model.attention["gamma"] == 2.0
    assert model.attention["b"] == 1.0
    assert model.attention["total_params"] == 29


def test_unknown_attention_type_raises():
    with pytest.raises(ValueError):
        build_mobilenet_v3_large(num_classes=10, pretrained="none", attention={"type": "cbam"})


def test_eca_checkpoint_roundtrip(tmp_path):
    from mobilenet_attention.engine import save_checkpoint

    model = build_mobilenet_v3_large(
        num_classes=10, pretrained="none", attention={"type": "eca", "placement": "non_se_blocks"}
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(
        tmp_path / "eca.pt",
        model,
        optimizer,
        None,
        epoch=0,
        metrics={"val_accuracy": 0.5},
        config={"model": {"attention": {"type": "eca", "placement": "non_se_blocks"}}},
    )

    restored = build_mobilenet_v3_large(
        num_classes=10, pretrained="none", attention={"type": "eca", "placement": "non_se_blocks"}
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    restored.load_state_dict(checkpoint["model_state"])
    for (name_a, param_a), (name_b, param_b) in zip(
        model.state_dict().items(), restored.state_dict().items(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)
    assert checkpoint["config"]["model"]["attention"]["type"] == "eca"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 不可用")
def test_eca_cuda_forward_and_backward():
    model = build_mobilenet_v3_large(
        num_classes=10, pretrained="none", attention={"type": "eca", "placement": "non_se_blocks"}
    ).cuda()
    logits = model(torch.randn(2, 3, 224, 224, device="cuda"))
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1], device="cuda"))
    loss.backward()
    assert torch.isfinite(logits).all()
    assert all(
        torch.isfinite(p.grad).all()
        for m in model.modules()
        if isinstance(m, ECALayer)
        for p in m.parameters()
        if p.grad is not None
    )
