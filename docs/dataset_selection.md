# 数据集选型报告（阶段 2）

- 决策日期：2026-10-20
- 主数据集：**CIFAR-10**
- 用途：MobileNetV3-Large 基线、注意力改进对比与消融实验

## 1. 候选数据集比较

| 维度 | CIFAR-10 | Intel Image Classification |
| --- | --- | --- |
| 原始来源 | 多伦多大学 Alex Krizhevsky 官方页面 | 最初由 Intel 在 Analytics Vidhya DataHack 竞赛发布 |
| 本项目获取渠道 | 官方 URL + MD5 校验 | Kaggle 第三方上传（puneet6060），需账号/API |
| 官方固定划分 | 是：50,000 train / 10,000 test | 上传者划分：约 14k train / 3k test / 7k prediction，非官方统一划分 |
| 类别数 | 10（互斥） | 6（buildings/forest/glacier/mountain/sea/street） |
| 图像尺寸 | 32×32 | 150×150 |
| 压缩包体积 | 162.6 MiB（实测 170,498,071 字节） | 约 351.8 MiB（Kaggle API 元数据 totalBytes = 368,859,597） |
| 磁盘需求（解压后） | 约 340 MiB（含压缩包） | 约 400–500 MiB（估算，未下载验证） |
| 许可证/使用条件 | 官方公开发布用于研究，要求引用 Krizhevsky 2009 技术报告 | Kaggle 标注 "Data files © Original Authors"，无明确开源许可证 |
| 可复现性 | 高：固定 URL、固定 MD5、固定划分 | 中：依赖 Kaggle 账号、上传者版本、无校验和 |
| 与本项目算力匹配 | 好：50k 张 32×32，224 输入下单轮约 1 分钟（实测） | 好：数据量相近 |
| 演示系统直观性 | 中：10 类物体，32×32 分辨率低，上传真实照片存在域偏移 | 好：6 类自然场景，150×150，演示效果更直观 |
| 论文对比实验适配 | 高：学术界标准基准，结果可与其他工作比较 | 中：竞赛数据集，缺乏统一评测协议与文献基线 |

## 2. 选择理由

选择 CIFAR-10 作为主数据集：

1. **官方来源与固定划分**：官方页面提供固定 URL 与 MD5，train/test 划分由数据集作者定义，避免自造划分带来的可复现性问题。
2. **可验证的完整性**：实测下载文件 MD5 为 `c58f30108f718f92721af3b95e74349a`，与官方页面一致。
3. **学术界可比性**：CIFAR-10 是轻量化网络与注意力机制文献中的常用基准，便于在论文中对照已有结果。
4. **规模与算力匹配**：45,000 训练样本在 RTX 4060 Laptop（8 GB）上单轮约 60–120 秒（实测），支持多组消融实验与多种子重复。
5. **类别互斥、标签干净**：10 类互斥，适合计算混淆矩阵与分类别错误分析。

## 3. 已知局限与应对

| 局限 | 影响 | 应对 |
| --- | --- | --- |
| 32×32 分辨率低 | 上传手机照片演示时存在分辨率/域偏移 | 在应用层明确说明训练域；必要时补充“数据集样例测试”模式 |
| 与 ImageNet 域差异 | 预训练特征迁移效果受限 | 使用 ImageNet 预训练并做完整微调；报告微调策略 |
| 10 类语义较粗 | 演示系统类别有限 | 在论文与 UI 中如实说明适用范围 |
| 非自然图像分布 | 真实场景泛化有限 | 在阶段 5 增加失败案例分析 |

Intel Image Classification 未采用的原因：获取依赖 Kaggle 账号与第三方上传版本，缺少官方固定划分与校验和，许可证表述不明确，不利于论文复现与合规说明。

## 4. 数据划分策略

```text
官方 train（50,000）
├── 训练集 45,000（分层抽样）
└── 验证集  5,000（分层抽样，seed=42）
官方 test（10,000）—— 仅最终独立评估，不参与调参
```

- 划分方式：`sklearn.model_selection.train_test_split(test_size=5000, stratify=targets, random_state=42)`
- 分层后每类：训练 4,500 / 验证 500（实测确认）
- 验证集用于选择最佳检查点与（后续）超参数选择
- 测试集只在训练结束后评估一次，脚本默认行为即如此

## 5. 类别映射

| 索引 | 类别 | 索引 | 类别 |
| --- | --- | --- | --- |
| 0 | airplane | 5 | dog |
| 1 | automobile | 6 | frog |
| 2 | bird | 7 | horse |
| 3 | cat | 8 | ship |
| 4 | deer | 9 | truck |

与 `torchvision.datasets.CIFAR10.class_to_idx` 实测一致。

## 6. 预处理与增强

| 阶段 | 操作 |
| --- | --- |
| 训练 | RandomResizedCrop(224, scale=(0.6, 1.0)) → RandomHorizontalFlip → ToTensor → Normalize(ImageNet mean/std) |
| 验证/测试 | Resize((224, 224)) → ToTensor → Normalize(ImageNet mean/std) |

- 输入尺寸 224：与 ImageNet 预训练权重的训练分辨率一致
- 归一化使用 ImageNet 统计量：与预训练权重匹配，不是 CIFAR-10 自身统计量
- 未使用 RandAugment、MixUp、CutMix 等强增强，保持基线简洁可复现

## 7. 随机种子与可复现性

| 项目 | 取值 |
| --- | --- |
| 全局随机种子 | 42 |
| 数据划分种子 | 42 |
| DataLoader shuffle | 由全局种子控制 |
| 确定性设置 | `torch.use_deterministic_algorithms(True, warn_only=True)`、cuDNN deterministic |
| 已知限制 | 部分 CUDA 算子无确定性实现，`warn_only` 下可能引入微小非确定性 |

## 8. 评估指标

| 指标 | 说明 |
| --- | --- |
| Accuracy | 总体正确率 |
| Precision (macro) | 各类精确率算术平均 |
| Recall (macro) | 各类召回率算术平均 |
| F1 (macro) | 各类 F1 算术平均 |
| 混淆矩阵 | 10×10，用于分类别错误分析 |
| 参数量 | 4,214,842（num_classes=10，实测） |
| 模型体积 | 16.17 MiB（按参数与缓冲区字节数） |
| 推理延迟 | batch=1/16/128，224×224，warmup=10，CUDA 同步 |
| 训练显存峰值 | batch=128 + AMP，实测 2,899 MiB |

## 9. 数据获取与校验命令

```powershell
# 断点续传下载 + MD5 校验（官方 URL）
.\.venv\Scripts\python.exe scripts/download_cifar10.py

# 实测 MD5
Get-FileHash data\cifar-10-python.tar.gz -Algorithm MD5
# C58F30108F718F92721AF3B95E74349A（与官方一致）
```

预训练权重来源（Torchvision 官方 URL，实测可下载）：

```text
MobileNet_V3_Large_Weights.IMAGENET1K_V1
https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth
SHA256 8738CA797C879B547D18BBD15DA5736FF2557B2036A9AF72225393CA61759A04
```

## 10. 引用

Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. University of Toronto.
