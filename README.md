# 基于注意力机制改进的 MobileNetV3 轻量化图像分类算法研究与智能识别系统设计

毕业设计项目：在 MobileNetV3 主干网络上引入注意力机制，完成可复现的基线训练、消融实验对比，并构建 Gradio 图像分类演示系统。

## 项目状态

当前处于 **阶段 2 收尾：基线冻结与可复现性审计（已完成，等待提交授权）**。

- 主数据集：CIFAR-10（官方来源，MD5 已校验）
- 基线：MobileNetV3-Large + ImageNet1K_V1 预训练
- 基线测试集准确率：0.9679（单种子，详见阶段 2 报告）
- 尚未插入注意力模块
- 基线冻结报告：[docs/stage2_freeze_audit.md](docs/stage2_freeze_audit.md)
- 阶段 3 ECA 协议（未实施）：[docs/stage3_eca_protocol.md](docs/stage3_eca_protocol.md)

## 环境要求

- Windows 11 / Python 3.12（已验证）
- NVIDIA GPU（已验证：RTX 4060 Laptop GPU, 8 GB）
- PyTorch 2.14.1+cu130、Torchvision 0.29.1+cu130

详细环境信息见 [docs/environment_report.md](docs/environment_report.md)。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

## 数据准备

```powershell
# 断点续传下载 CIFAR-10 并校验官方 MD5
.\.venv\Scripts\python.exe scripts/download_cifar10.py
```

## 训练基线

```powershell
# 完整基线（15 epochs，约 19 分钟）
.\.venv\Scripts\python.exe scripts/train.py --config configs/baseline_mobilenetv3_large.yaml

# 流程冒烟（10 个批次，不构成实验结果）
.\.venv\Scripts\python.exe scripts/train.py --config configs/smoke.yaml
```

结果写入 `outputs/<experiment_id>/`：配置快照、环境快照、逐 epoch 日志、最佳/最后检查点、验证与测试指标、延迟。

## 评估与验证

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --checkpoint outputs/baseline_mobilenetv3_large_cifar10/best.pt --split test
.\.venv\Scripts\python.exe scripts/sanity_check.py
.\.venv\Scripts\python.exe scripts/benchmark.py
```

## 环境自检

```powershell
.\.venv\Scripts\python.exe scripts/check_env.py
```

## 测试与代码质量

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
```

## 目录结构

```
configs/            数据集与实验配置（YAML）
docs/               环境报告、数据集选型、阶段报告
scripts/            下载、训练、评估、基准与检查脚本
src/mobilenet_attention/
  data/             CIFAR-10 加载、划分与 DataLoader
  models/           MobileNetV3-Large 基线构建
  engine/           训练循环、指标、检查点、延迟测量
  utils/            配置、随机种子、环境快照
tests/              单元测试
data/               数据集（不提交）
outputs/            实验结果（权重不提交，日志/指标提交）
```

## 实验记录

- [阶段 2 基线报告](docs/stage2_baseline_report.md)
- [数据集选型](docs/dataset_selection.md)

## 研究目标

1. 复现 MobileNetV3-Large 基线。
2. 实现并验证注意力模块改进。
3. 在统一数据划分与评测协议下做公平对比与消融实验。
4. 构建可运行的图像分类应用。
5. 以真实实验证据支撑论文写作。

## 学术规范

所有指标须可追溯到结果文件或日志；不虚构实验数据、显卡信息或参考文献。
