"""模型注册表测试：model.name 必须决定实际构建的模型。"""

from __future__ import annotations

import pytest
import torch

from mobilenet_attention.models import (
    SUPPORTED_MODELS,
    build_model,
    describe_model,
)


def test_supported_names_are_declared():
    assert set(SUPPORTED_MODELS) == {"mobilenet_v3_large", "mobilenet_v3_large_eca"}


def test_baseline_name_builds_plain_model():
    model = build_model("mobilenet_v3_large", num_classes=10, pretrained="none")
    identity = describe_model(model)
    assert identity["model_name"] == "mobilenet_v3_large"
    assert identity["attention_type"] is None
    assert identity["attention_blocks"] == 0
    assert not hasattr(model, "attention")


def test_eca_name_builds_model_with_eca():
    model = build_model(
        "mobilenet_v3_large_eca",
        num_classes=10,
        pretrained="none",
        attention={"type": "eca", "placement": "non_se_blocks"},
    )
    identity = describe_model(model)
    assert identity["model_name"] == "mobilenet_v3_large_eca"
    assert identity["attention_type"] == "eca"
    assert identity["attention_placement"] == "non_se_blocks"
    assert identity["attention_blocks"] == 7
    assert identity["attention_params"] == 29


def test_eca_name_has_sensible_defaults():
    """未显式给 attention 时，ECA 变体应使用默认的 ECA 配置，而不是静默退化为基线。"""
    model = build_model("mobilenet_v3_large_eca", num_classes=10, pretrained="none")
    identity = describe_model(model)
    assert identity["attention_type"] == "eca"
    assert identity["attention_placement"] == "non_se_blocks"
    assert identity["attention_blocks"] == 7


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="未知模型名称"):
        build_model("resnet50", num_classes=10, pretrained="none")


def test_empty_name_raises():
    with pytest.raises(ValueError, match="未知模型名称"):
        build_model("", num_classes=10, pretrained="none")


def test_baseline_name_rejects_attention_config():
    """防止配置写 model.name=baseline 却期望插入注意力。"""
    with pytest.raises(ValueError, match="不接受 attention"):
        build_model(
            "mobilenet_v3_large",
            num_classes=10,
            pretrained="none",
            attention={"type": "eca"},
        )


def test_eca_name_rejects_non_eca_type():
    with pytest.raises(ValueError, match="只接受 type=eca"):
        build_model(
            "mobilenet_v3_large_eca",
            num_classes=10,
            pretrained="none",
            attention={"type": "cbam"},
        )


def test_name_is_case_insensitive():
    model = build_model("MobileNet_V3_Large", num_classes=10, pretrained="none")
    assert describe_model(model)["model_name"] == "mobilenet_v3_large"


def test_baseline_parameter_count_unchanged():
    """基线结构必须与阶段 2 完全一致。"""
    model = build_model("mobilenet_v3_large", num_classes=10, pretrained="none")
    assert sum(p.numel() for p in model.parameters()) == 4_214_842


def test_eca_parameter_count():
    model = build_model(
        "mobilenet_v3_large_eca",
        num_classes=10,
        pretrained="none",
        attention={"type": "eca", "placement": "non_se_blocks"},
    )
    assert sum(p.numel() for p in model.parameters()) == 4_214_871


def test_both_variants_forward():
    for name in SUPPORTED_MODELS:
        model = build_model(name, num_classes=10, pretrained="none").eval()
        with torch.no_grad():
            logits = model(torch.randn(2, 3, 224, 224))
        assert logits.shape == (2, 10)
        assert torch.isfinite(logits).all()


def test_baseline_weights_load_into_baseline_only():
    baseline = build_model("mobilenet_v3_large", num_classes=10, pretrained="none")
    eca = build_model("mobilenet_v3_large_eca", num_classes=10, pretrained="none")
    with pytest.raises(RuntimeError):
        eca.load_state_dict(baseline.state_dict())
    # 反向：ECA 权重也不能加载进基线
    with pytest.raises(RuntimeError):
        baseline.load_state_dict(eca.state_dict())


def test_describe_model_matches_runtime_state():
    model = build_model(
        "mobilenet_v3_large_eca",
        num_classes=10,
        pretrained="none",
        attention={"type": "eca", "placement": "se_blocks_only"},
    )
    identity = describe_model(model)
    assert identity["attention_blocks"] == len(model.attention["inserted_blocks"]) == 8
    assert identity["num_classes"] == model.num_classes == 10
