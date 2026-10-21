"""阶段 2 冒烟验证：GPU 前向、反向传播、单批次训练与小样本过拟合。

本脚本只验证工程流程，输出不是正式实验结果。

用法：
    python scripts/sanity_check.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mobilenet_attention.data import build_cifar10_datasets  # noqa: E402
from mobilenet_attention.models import build_mobilenet_v3_large  # noqa: E402
from mobilenet_attention.utils.seed import resolve_device, set_seed  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "outputs" / "stage2_sanity_check.json"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    set_seed(42)
    device = resolve_device("auto")
    report: dict = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    print(f"设备：{device}")

    model = build_mobilenet_v3_large(num_classes=10, pretrained="imagenet1k_v1").to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # 1) 前向 + 反向（单个真实批次尺寸）
    images = torch.randn(16, 3, 224, 224, device=device)
    labels = torch.randint(0, 10, (16,), device=device)
    logits = model(images)
    loss = criterion(logits, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(
        torch.sqrt(
            sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None)
        )
    )
    report["forward_backward"] = {
        "logits_shape": list(logits.shape),
        "loss": float(loss),
        "grad_norm": grad_norm,
        "finite_logits": bool(torch.isfinite(logits).all()),
        "finite_grads": bool(
            all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        ),
    }
    print(
        f"前向/反向：logits={tuple(logits.shape)} loss={float(loss):.4f} grad_norm={grad_norm:.4f}"
    )

    # 2) 真实数据单批次训练：损失应下降
    datasets = build_cifar10_datasets(root=PROJECT_ROOT / "data", input_size=224, val_size=500)
    batch_images, batch_labels = [], []
    for index in range(16):
        image, label = datasets["train"][index]
        batch_images.append(image)
        batch_labels.append(label)
    images = torch.stack(batch_images).to(device)
    labels = torch.tensor(batch_labels, device=device)

    losses = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    report["single_batch"] = {"losses": losses, "first": losses[0], "last": losses[-1]}
    print(f"单批次 5 步：loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "单批次训练损失未下降"

    # 3) 小样本过拟合：200 张图分小批次累积，避免单 batch 占用超过 8 GB 显存
    overfit_images = torch.stack([datasets["train"][i][0] for i in range(200)]).to(device)
    overfit_labels = torch.tensor([datasets["train"][i][1] for i in range(200)], device=device)
    overfit_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    overfit_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    overfit_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(overfit_images, overfit_labels),
        batch_size=32,
        shuffle=True,
    )
    start = time.perf_counter()
    final_accuracy = 0.0
    for step in range(60):
        for batch_images, batch_labels in overfit_loader:
            overfit_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(batch_images)
                loss = criterion(logits, batch_labels)
            overfit_scaler.scale(loss).backward()
            overfit_scaler.step(overfit_optimizer)
            overfit_scaler.update()
        with torch.no_grad():
            predictions = model(overfit_images).argmax(dim=1)
        final_accuracy = float((predictions == overfit_labels).float().mean())
        if step % 20 == 0:
            print(f"  step {step}: loss={float(loss.detach()):.4f} acc={final_accuracy:.3f}")
    elapsed = time.perf_counter() - start
    report["overfit"] = {
        "samples": 200,
        "steps": 60,
        "final_train_accuracy": final_accuracy,
        "final_loss": float(loss.detach()),
        "duration_s": elapsed,
    }
    print(
        f"小样本过拟合：acc={final_accuracy:.3f} loss={float(loss.detach()):.4f} 用时 {elapsed:.1f}s"
    )

    report["peak_memory_mib"] = (
        torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告写入：{REPORT_PATH}")
    return 0 if final_accuracy >= 0.95 else 1


if __name__ == "__main__":
    raise SystemExit(main())
