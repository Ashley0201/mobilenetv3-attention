"""训练引擎测试：单批次训练、指标、检查点保存与加载。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mobilenet_attention.engine import (
    JsonlLogger,
    build_scheduler,
    compute_metrics,
    evaluate,
    load_checkpoint,
    measure_latency,
    save_checkpoint,
    train_one_epoch,
)


@pytest.fixture()
def tiny_setup():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10))
    images = torch.randn(16, 3, 8, 8)
    labels = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)
    return model, loader, labels


def test_single_batch_training_decreases_loss(tiny_setup):
    model, loader, _ = tiny_setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = evaluate(model, loader, criterion, torch.device("cpu"), [str(i) for i in range(10)])
    stats = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
    after = evaluate(model, loader, criterion, torch.device("cpu"), [str(i) for i in range(10)])
    assert stats["loss"] > 0
    assert after["loss"] < before["loss"]


def test_metrics_are_correct_for_perfect_predictions():
    metrics = compute_metrics([0, 1, 2, 3], [0, 1, 2, 3], ["a", "b", "c", "d"])
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["confusion_matrix"] == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def test_metrics_handle_wrong_predictions():
    metrics = compute_metrics([0, 1, 2, 3], [0, 0, 2, 2], ["a", "b", "c", "d"])
    assert metrics["accuracy"] == 0.5
    assert metrics["num_samples"] == 4


def test_checkpoint_roundtrip(tiny_setup, tmp_path: Path):
    model, loader, _ = tiny_setup
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = build_scheduler(optimizer, epochs=2)
    path = save_checkpoint(
        tmp_path / "best.pt",
        model,
        optimizer,
        scheduler,
        epoch=0,
        metrics={"val_accuracy": 0.5},
        config={"experiment": {"id": "test"}},
    )
    assert path.is_file()

    restored = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10))
    meta = load_checkpoint(path, restored, torch.device("cpu"))
    assert meta["epoch"] == 0
    for (name_a, param_a), (name_b, param_b) in zip(
        model.state_dict().items(), restored.state_dict().items(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)


def test_scheduler_steps(tiny_setup):
    model, _, _ = tiny_setup
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = build_scheduler(optimizer, epochs=3, warmup_epochs=1)
    lrs = []
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    assert len(lrs) == 3
    assert all(lr > 0 for lr in lrs)


def test_scheduler_warmup_curve_matches_baseline(tiny_setup):
    """warmup=1 时首轮 LR = 0.1×base，第二轮回到 base，之后余弦下降。"""
    model, _, _ = tiny_setup
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = build_scheduler(
        optimizer, epochs=4, warmup_epochs=1, eta_min=1e-6, warmup_start_factor=0.1
    )
    lrs = []
    for _ in range(4):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert abs(lrs[0] - 0.01) < 1e-9
    assert abs(lrs[1] - 0.1) < 1e-9
    assert lrs[2] < lrs[1]
    assert lrs[3] < lrs[2]


def test_scheduler_eta_min_is_respected(tiny_setup):
    model, _, _ = tiny_setup
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = build_scheduler(optimizer, epochs=2, warmup_epochs=0, eta_min=0.005)
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] >= 0.005 - 1e-12


def test_grad_clip_limits_parameter_update(tiny_setup):
    """同一批数据下，梯度裁剪应让参数更新幅度更小。"""
    model, loader, _ = tiny_setup
    criterion = nn.CrossEntropyLoss()

    unclipped = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10))
    clipped = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10))
    clipped.load_state_dict(unclipped.state_dict())

    opt_a = torch.optim.SGD(unclipped.parameters(), lr=0.5)
    opt_b = torch.optim.SGD(clipped.parameters(), lr=0.5)
    train_one_epoch(unclipped, loader, criterion, opt_a, torch.device("cpu"))
    train_one_epoch(clipped, loader, criterion, opt_b, torch.device("cpu"), grad_clip=1e-4)

    delta_unclipped = (unclipped[1].weight - model[1].weight).abs().sum()
    delta_clipped = (clipped[1].weight - model[1].weight).abs().sum()
    assert delta_clipped < delta_unclipped


def test_label_smoothing_increases_loss_for_confident_correct(tiny_setup):
    """配置的 label_smoothing 通过 CrossEntropyLoss 生效，而非被忽略。"""
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    targets = torch.tensor([0, 1])
    plain = nn.CrossEntropyLoss()(logits, targets)
    smoothed = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, targets)
    assert smoothed > plain


def test_jsonl_logger_appends(tmp_path: Path):
    logger = JsonlLogger(tmp_path / "log.jsonl")
    logger.log({"epoch": 0, "val_accuracy": 0.5})
    logger.log({"epoch": 1, "val_accuracy": 0.6})
    lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert '"epoch": 1' in lines[1]


def test_measure_latency_cpu():
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    result = measure_latency(
        model, torch.device("cpu"), input_size=32, batch_size=1, warmup=1, repeats=3
    )
    assert result["latency_ms_per_batch"] > 0
    assert result["throughput_images_per_s"] > 0
