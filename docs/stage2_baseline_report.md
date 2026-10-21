# 阶段 2 报告：基线复现与数据集选型

- 执行日期：2026-10-20
- 实验 ID：`baseline_mobilenetv3_large_cifar10`
- 状态：基线已跑通，结果可追溯；尚未插入注意力模块

## 1. 环境兼容性核实（真实输出）

```text
Python 3.12.3  (C:\Users\Sylvia\Desktop\mobilenetv3-attention\.venv\Scripts\python.exe)
torch          2.14.1+cu130
torchvision    0.29.1+cu130
numpy          2.5.2
CUDA compiled  13.0
cuda available True
GPU            NVIDIA GeForce RTX 4060 Laptop GPU
capability     (8, 9)
cuDNN          92400
torchvision 要求 torch >= 2.14.0（METADATA 实测）
pip check      No broken requirements found.
```

- 安装来源：官方索引 `https://download.pytorch.org/whl/cu130`
- wheel：`torch-2.14.1+cu130-cp312-cp312-win_amd64`、`torchvision-0.29.1+cu130-cp312-cp312-win_amd64`
- 解释器隔离：`sys.prefix` 指向项目 `.venv`，`numpy/torch/torchvision` 均从 `.venv\Lib\site-packages` 加载
- 全局 Python 的 `~umpy` 残留未影响本项目（实测 `pip check` 通过）

## 2. 数据集

主数据集：CIFAR-10（官方来源，MD5 已校验）。

| 项目 | 实测 |
| --- | --- |
| 压缩包 | `data\cifar-10-python.tar.gz`，170,498,071 字节（162.6 MiB） |
| MD5 | `C58F30108F718F92721AF3B95E74349A`（与官方页面一致） |
| 训练集 | 45,000（从官方 50,000 分层抽取） |
| 验证集 | 5,000 |
| 测试集 | 10,000（官方 test，仅最终评估） |
| 类别 | 10 类，映射与 Torchvision 一致 |
| 磁盘占用 | `data` 合计约 340 MiB |

详细比较与选择理由见 `docs/dataset_selection.md`。

## 3. 模型配置

| 项目 | 值 |
| --- | --- |
| 架构 | `torchvision.models.mobilenet_v3_large` |
| 预训练 | `MobileNet_V3_Large_Weights.IMAGENET1K_V1`（真实 ImageNet 权重） |
| 权重 URL | `https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth` |
| 权重 SHA256 | `8738CA797C879B547D18BBD15DA5736FF2557B2036A9AF72225393CA61759A04` |
| 分类头 | 仅替换 `classifier[3]` 为 `Linear(1280, 10)`，保留 Dropout(0.2) |
| 参数量 | 4,214,842 |
| 模型体积 | 16.17 MiB |
| 输入 | [B, 3, 224, 224] |
| 输出 | [B, 10] 原始 logits |

`tests/test_model.py::test_pretrained_weights_are_really_loaded` 直接从官方缓存文件比对第一层卷积权重，确认加载的不是随机初始化。

## 4. 训练配置

| 项目 | 值 |
| --- | --- |
| Epochs | 15 |
| Batch size | 128（评估 256） |
| 优化器 | AdamW，lr=3e-4，weight_decay=0.01 |
| 调度 | 1 epoch 线性 warmup + 余弦退火至 1e-6 |
| 损失 | CrossEntropyLoss(label_smoothing=0.1) |
| AMP | 开启（CUDA） |
| 梯度裁剪 | 1.0 |
| DataLoader workers | 4 |
| 随机种子 | 42 |
| 设备 | cuda |

## 5. 验证流程（训练前）

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| GPU 前向 | logits (16, 10)，全部有限 | `outputs/stage2_sanity_check.json` |
| 反向传播 | grad_norm 4.6627，梯度全部有限 | 同上 |
| 单批次训练 5 步 | loss 2.4053 → 0.2410 | 同上 |
| 小样本过拟合（200 张 / 60 步，batch=32） | 训练准确率 1.000，loss 0.0097，峰值显存 1,668 MiB | 同上 |
| 数据加载 | 45,000/5,000/10,000，类别均衡 | `train_log.jsonl` 的 dataset 事件 |
| 单批次训练 | smoke 1 epoch 10 批次，正常保存权重 | `outputs/smoke_baseline/` |

说明：过拟合测试最初把 200 张 224×224 图像放在单个 batch 中，耗时 235 秒，当时 `torch.cuda.max_memory_allocated()` 显示 8,669.9 MiB，超过 8,188 MiB 专用显存，判断为 Windows WDDM 共享内存回退（该次运行的 JSON 已被后续重跑覆盖，数值仅为当时终端观察）。已改为 batch=32 分小批次累积，重跑结果如上：峰值 1,668 MiB、28 秒。正式训练 batch=128 的峰值为 2,899 MiB，始终在专用显存范围内。

## 6. 基线训练结果（真实）

训练时间：2026-10-20 14:22:47 → 14:41:59 UTC，15 epochs 合计约 1,108.7 秒。

| Epoch | train_loss | train_acc | val_loss | val_acc |
| --- | --- | --- | --- | --- |
| 1 | 1.2942 | 0.6844 | 0.8419 | 0.8740 |
| 2 | 0.7395 | 0.9086 | 0.6872 | 0.9282 |
| 3 | 0.6395 | 0.9489 | 0.6496 | 0.9402 |
| 4 | 0.5960 | 0.9658 | 0.6160 | 0.9538 |
| 5 | 0.5733 | 0.9746 | 0.6098 | 0.9564 |
| 6 | 0.5554 | 0.9827 | 0.6015 | 0.9592 |
| 7 | 0.5431 | 0.9872 | 0.5907 | 0.9630 |
| 8 | 0.5338 | 0.9908 | 0.5921 | 0.9600 |
| 9 | 0.5271 | 0.9930 | 0.5817 | 0.9660 |
| 10 | 0.5223 | 0.9950 | 0.5835 | 0.9658 |
| 11 | 0.5183 | 0.9961 | 0.5812 | 0.9654 |
| 12 | 0.5158 | 0.9971 | 0.5774 | 0.9676 |
| 13 | 0.5146 | 0.9970 | 0.5727 | 0.9694 |
| 14 | 0.5135 | 0.9977 | 0.5714 | 0.9694 |
| 15 | 0.5123 | 0.9983 | 0.5712 | 0.9694 |

最佳轮次：epoch 12（0-based，即第 13 轮），val_accuracy = 0.9694。

### 独立测试集（仅评估一次）

| 指标 | 值 |
| --- | --- |
| Accuracy | 0.9679 |
| Precision (macro) | 0.9679 |
| Recall (macro) | 0.9679 |
| F1 (macro) | 0.9679 |
| Loss（label_smoothing=0.1） | 0.5837 |
| 样本数 | 10,000 |

用 `scripts/evaluate.py` 复算，accuracy 与 F1 完全一致；loss 与训练脚本一致（均含 label smoothing）。

混淆矩阵显示主要混淆对：cat↔dog（43/32）、truck↔automobile（19/12）。

### 资源指标

| 指标 | 值 | 条件 |
| --- | --- | --- |
| 推理延迟 | 5.48 ms/张 | batch=1, 224×224, warmup=10, repeats=30 |
| 推理延迟 | 6.76 ms/batch（0.42 ms/张） | batch=16 |
| 推理延迟 | 81.21 ms/batch（0.63 ms/张） | batch=128 |
| 训练显存峰值 | 2,899 MiB | batch=128, AMP |
| 单轮训练时间 | 60–117 s | 实测，含验证 |

## 7. 产出文件

```text
outputs/baseline_mobilenetv3_large_cifar10/
  config_snapshot.yaml                       生效配置
  environment.json                           软硬件快照
  train_log.jsonl                            逐 epoch 日志 + 数据/模型/延迟事件
  best.pt                                    epoch 12 最优检查点（50.9 MB）
  last.pt                                    最后一轮检查点
  val_metrics.json                           验证集指标 + 混淆矩阵
  test_metrics.json                          独立测试集指标
  test_metrics_recheck.json                  复算结果（一致性检查）
  test_metrics_recheck_label_smoothing.json  修正 loss 口径后的复算
  latency.json                               推理延迟
outputs/smoke_baseline/                      流程冒烟结果（非正式实验）
outputs/stage2_sanity_check.json             前向/反向/过拟合验证
outputs/stage2_benchmark.json                参数量/体积/延迟/显存基准
outputs/logs/baseline_stdout.log             训练标准输出
outputs/logs/baseline_stderr.log             训练 stderr（tqdm 进度）
```

## 8. 测试与代码质量（真实输出）

```text
.\.venv\Scripts\python.exe -m pytest tests/ -q
42 passed in 14.04s

.\.venv\Scripts\python.exe -m ruff check src tests scripts
All checks passed!

.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
21 files already formatted
```

测试覆盖：环境采集、配置加载/覆盖/快照、数据划分/类别均衡/transform、模型形状/梯度/预训练真实性、训练循环/指标/检查点往返/日志/延迟、训练脚本辅助函数。

## 9. 已执行命令（摘要）

```powershell
.\.venv\Scripts\python.exe -V
.\.venv\Scripts\python.exe -m pip show torch torchvision numpy
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch, torchvision; ..."
.\.venv\Scripts\python.exe scripts/download_cifar10.py
.\.venv\Scripts\python.exe scripts/sanity_check.py
.\.venv\Scripts\python.exe scripts/benchmark.py
.\.venv\Scripts\python.exe scripts/train.py --config configs/smoke.yaml
.\.venv\Scripts\python.exe scripts/train.py --config configs/baseline_mobilenetv3_large.yaml
.\.venv\Scripts\python.exe scripts/evaluate.py --checkpoint outputs/baseline_mobilenetv3_large_cifar10/best.pt --split test
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
```

## 10. 剩余风险与限制

1. **单次运行**：当前仅 1 个随机种子（42），尚无均值/标准差；论文对比需至少 3 个种子。
2. **测试集已使用一次**：本阶段用于最终评估，后续调参不得再参考该结果；注意力改进的最终评估应使用同一测试集但只在方法冻结后评估。
3. **无 Git 首次提交**：`environment.json` 的 `git_commit` 为 null；训练结果暂时无法对应到代码提交。
4. **`nvidia-smi` 显存占用波动**：训练期间桌面应用占用显存，未影响本次训练（峰值 2.9 GiB）。
5. **32×32 输入上采样到 224**：计算量大于原生 32×32 训练，属于为匹配预训练权重的有意选择，需在论文中说明。
6. **确定性**：`warn_only=True` 下个别 CUDA 算子可能非确定性，跨机器逐位复现不保证。
7. **未测 FLOPs**：AGENTS.md 要求在有合适方法时报告；当前未引入测量工具，阶段 4 补充。
8. **预训练权重缓存**：位于 `C:\Users\Sylvia\.cache\torch\hub\checkpoints`，未纳入项目；重装需联网。
9. **冒烟测试已修正**：`scripts/sanity_check.py` 原先 200 张单 batch 过拟合触发 WDDM 共享内存回退，已改为 batch=32，重跑峰值 1,668 MiB；正式训练不受影响。

## 11. 进入阶段 3（注意力改进）前必须完成

1. 创建 Git 首次提交，冻结基线代码与结果记录（权重不入库）。
2. 确认基线数字作为对比基准：val_acc 0.9694 / test_acc 0.9679（单种子）。
3. 确定注意力机制候选（SE / ECA / CBAM），给出文献依据、插入位置与参数开销。
4. 设计消融实验矩阵与统一训练协议（相同划分、epochs、增强、种子）。
5. 规划多种子重复（至少 3 个），避免单次结果下结论。
6. 明确测试集使用策略：方法冻结后只评估一次。
