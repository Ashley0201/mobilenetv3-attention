# 阶段 3 实验协议：ECA 注意力改进（待批准，未实施）

- 编写日期：2026-10-20
- 状态：**协议草案。本阶段未修改模型、未新增模块、未启动训练。**
- 基线冻结版本：`baseline_mobilenetv3_large_cifar10`（val 0.9694 / test 0.9679，seed 42）

## 1. 文献依据

| 项目 | 内容 |
| --- | --- |
| 论文 | Q. Wang, B. Wu, P. Zhu, P. Li, W. Zuo, Q. Hu. *ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks*. CVPR 2020. |
| arXiv | https://arxiv.org/abs/1910.03151（v4，2020-04-07） |
| DOI | https://doi.org/10.48550/arXiv.1910.03151 |
| 官方实现 | https://github.com/BangguWu/ECANet（MIT License） |
| 核心主张 | 通道注意力中避免降维（no dimensionality reduction）对学习通道注意力重要；用 1D 卷积实现局部跨通道交互，可在显著降低复杂度的同时保持性能 |
| 论文报告 | 以 ResNet50 为骨干，ECA 模块参数 80、计算 4.7e-4 GFLOPs，对比 SE 的 24.37M 参数、3.86 GFLOPs；Top-1 提升超过 2% |

**本项目不声称提出新机制。** 贡献定位为：在轻量化骨干上系统评估 ECA 的插入策略，并给出参数量、计算量、延迟与精度的完整对照。

## 2. ECA 数学定义

给定某卷积层输出特征图 \(X \in \mathbb{R}^{C \times H \times W}\)：

```text
1) 全局平均池化：  z_c = (1/(H·W)) · Σ_{i,j} X_{c,i,j}          z ∈ R^C
2) 局部跨通道交互： s = σ( Conv1D_k(z) )                          s ∈ R^C
3) 通道重标定：     Y = X ⊙ s（按通道广播）                        Y ∈ R^{C×H×W}
```

其中 `Conv1D_k` 的核大小按通道数自适应：

```text
k = | (log2(C) / γ) + (b / γ) |_odd ,  γ = 2, b = 1
|·|_odd 表示取最近的奇数
```

- 无全连接降维，因此参数量仅为 `k`（1D 卷积权重，无 bias）。
- 本项目的通道数对应核大小（由实际特征图形状计算）：

| 块索引 | 通道 C | k | 参数 |
| --- | --- | --- | --- |
| 1 | 16 | 3 | 3 |
| 2 | 64 | 3 | 3 |
| 3 | 72 | 3 | 3 |
| 7 | 240 | 5 | 5 |
| 8 | 200 | 5 | 5 |
| 9 | 184 | 5 | 5 |
| 10 | 184 | 5 | 5 |
| 合计 | — | — | **29** |

> 以上为解析计算（`outputs/stage2_audit/baseline_audit.json` 的 `eca_overhead_estimate`），不是模块实测。

## 3. 插入位置与理论依据

### 主方案（P1）：仅插入无 SE 的 InvertedResidual 块

MobileNetV3-Large 的 15 个 InvertedResidual 块中，8 个已含 SE（块 4,5,6,11,12,13,14,15），7 个不含（块 1,2,3,7,8,9,10）。

插入点：**深度卷积输出之后、投影卷积（1×1）之前**，即 `block[depthwise] → ECA → block[projection]`。

```text
输入  [B, C_exp, H, W]  (C_exp 为扩展通道数)
  ↓ GAP
     [B, C_exp, 1, 1]
  ↓ Conv1d(k, groups=1) + Sigmoid
     [B, C_exp, 1, 1]
  ↓ 广播相乘
输出  [B, C_exp, H, W]   （形状不变）
```

依据：

1. **与 SE 的位置一致**：Torchvision 的 `SqueezeExcitation` 同样位于 depthwise 之后、projection 之前。把 ECA 放在同一位置，可让"有注意力"与"无注意力"的块在结构上对称，避免把位置差异混入机制差异。
2. **不重复叠加通道注意力**：SE 块已有通道重标定，再叠加 ECA 会使该块同时受两种通道门控影响，混淆消融结论。主方案保持 SE 原样。
3. **作用在扩展通道上**：ECA 的核大小随通道数自适应，扩展通道（16–240）比投影后的输出通道（16–80）提供更丰富的通道交互空间；同时该位置梯度可直接回传到 depthwise 卷积。
4. **计算位置代价低**：块 1 的空间尺寸最大（112×112），但通道仅 16；块 7–10 通道较多但空间仅 14×14。总增量约 0.79M MACs，占基线 215.3M MACs 的 **0.367%**。

### 备选方案（仅作消融，不作为主结果）

| 编号 | 方案 | 参数变化 | 说明 |
| --- | --- | --- | --- |
| A1 | ECA 插入全部 15 个块 | +34 | 与 SE 叠加，检验"注意力越多是否越好" |
| A2 | 用 ECA 替换全部 8 个 SE | −1,512,246 | 参数量大幅下降，精度变化会与"参数减少"混淆，必须单独标注，不能当作纯注意力收益 |
| A3 | 固定 k=3 而非自适应 | +21 | 检验自适应核大小的贡献 |

## 4. 开销测量方法

所有指标对基线与改进模型使用**同一脚本、同一输入、同一设备**测量，结果写入各自实验目录的 `efficiency.json`。

| 指标 | 方法 | 口径 |
| --- | --- | --- |
| 参数量 | `sum(p.numel() for p in model.parameters())` | 含分类头；分别报告 backbone / classifier |
| 可训练参数量 | 同上，过滤 `requires_grad` | 本实验全参数微调，两者相等 |
| 模型体积 | `Σ(param.numel() × element_size) + Σ(buffer...)` | MiB，不含 Python 对象开销 |
| 理论 FLOPs / MACs | `torch.utils.flop_counter.FlopCounterMode` | batch=1、224×224、`eval()`、`no_grad`；报告 FLOPs 与 MACs=FLOPs/2；标注为 shape-based 理论值 |
| 推理延迟 | `measure_latency`：warmup=10 → `cuda.synchronize()` → 计时 50 次 → `cuda.synchronize()` | batch=1 与 batch=16；**重复 5 轮**，报告均值±标准差、最小/最大 |
| 训练显存峰值 | `torch.cuda.max_memory_allocated()` | batch=128 + AMP，训练一步后读取 |
| 单轮训练耗时 | 训练循环墙钟时间 | 报告 15 轮的中位数与总时长 |
| GPU 状态 | `nvidia-smi --query-gpu=temperature.gpu,clocks.sm,utilization.gpu` | 每次延迟测量前后各记录一次 |

新增脚本（阶段 3 实施时编写）：`scripts/measure_efficiency.py`，输出：

```json
{
  "model": "baseline | eca_non_se",
  "parameters": {"total": 0, "backbone": 0, "classifier": 0},
  "model_size_mib": 0.0,
  "flops": {"total": 0, "macs": 0, "convention": "..."},
  "latency": {"bs1": {"mean_ms": 0, "stdev_ms": 0, "runs": []}, "bs16": {}},
  "train_peak_memory_mib": 0,
  "gpu_state_before": {}, "gpu_state_after": {}
}
```

## 5. 对照实验的一致性约束

基线与 ECA 模型必须完全一致，唯一变量是注意力模块：

| 项目 | 固定值 | 校验方式 |
| --- | --- | --- |
| 数据集 | CIFAR-10，官方 MD5 `c58f30108f718f92721af3b95e74349a` | 启动时校验 MD5 |
| 划分 | 45,000 / 5,000 / 10,000，seed 42，分层 | 训练开始保存 `split_indices.json`，哈希必须为 `439d2886...` / `6781efc5...` |
| 输入尺寸 | 224×224 | 配置快照 |
| 训练变换 | RandomResizedCrop(224, scale=(0.6,1.0)) → RandomHorizontalFlip → ToTensor → Normalize(ImageNet) | 配置中显式写出变换链 |
| 评估变换 | Resize((224,224)) → ToTensor → Normalize(ImageNet) | 同上 |
| 优化器 | AdamW，lr=3e-4，weight_decay=0.01 | 配置快照 |
| 调度器 | 1 epoch 线性 warmup（start_factor=0.1）+ CosineAnnealingLR(eta_min=1e-6) | 配置快照新增 `scheduler` 段 |
| 损失 | CrossEntropyLoss(label_smoothing=0.1) | 配置快照 |
| Epochs | 15 | 配置快照 |
| Batch size | 128（评估 256） | 配置快照 |
| AMP | 开启 | 配置快照 |
| 梯度裁剪 | 1.0 | 配置快照 |
| 预训练 | ImageNet1K_V1，SHA256 `8738ca79...` | 启动时比对缓存哈希 |
| 模型选择 | 验证集 accuracy 最高的 epoch | 训练脚本固定 |
| 评估指标 | Accuracy、macro P/R/F1、混淆矩阵、loss | 同一 `compute_metrics` |
| 测试集 | 仅在全部训练与选择完成后评估一次 | 训练脚本 `test_after_training` 受控 |

**验证集**用于选择 checkpoint；**测试集**不参与任何选择。阶段 3 的所有模型共享同一测试集，只在方法冻结后各评估一次。

## 6. 随机种子设计

至少 3 个种子，覆盖训练初始化、数据打乱与增强随机性：

| 种子 | 用途 |
| --- | --- |
| 42 | 与阶段 2 基线一致，可对齐 |
| 2024 | 独立重复 |
| 7 | 独立重复 |

- 每个种子下，基线与 ECA 模型使用**相同种子**，构成配对比较。
- 数据划分种子固定为 42，不随训练种子变化（保证所有实验用同一验证集，验证指标可比）。
- 实验矩阵（主实验）：

| 编号 | 模型 | 种子 | 运行数 |
| --- | --- | --- | --- |
| B42 / B2024 / B7 | baseline | 42 / 2024 / 7 | 3 |
| E42 / E2024 / E7 | ECA（非 SE 块） | 42 / 2024 / 7 | 3 |

- 可选消融（时间允许时，每种 1 个种子 42）：A1 全部块、A3 固定 k=3。A2 替换 SE 需单独说明参数减少。
- 预计耗时：单次 15 epochs 约 19 分钟（阶段 2 实测 1,108.7 秒），6 次约 2 小时；加消融约 3 小时。8 GB 显存足够（基线峰值 2.9 GiB）。
- 统计汇报：每个模型报告 3 个种子的均值 ± 样本标准差；同时给出配对差值（同种子 E − B）及其均值，避免只比较总体均值而忽略配对结构。3 个种子不足以做显著性检验，论文中只描述趋势并明确样本量限制。

## 7. 目录与追溯规范

```text
outputs/
  baseline_mobilenetv3_large_cifar10/      # 阶段 2，已冻结，不覆盖
  eca_non_se_cifar10_seed42/
    config_snapshot.yaml                   生效配置（含 scheduler/transforms/attention）
    environment.json                       软硬件 + git commit（提交后非 null）
    split_indices.json                     训练/验证索引与 SHA256
    train_log.jsonl                        逐 epoch 日志
    best.pt / last.pt                      权重（不入库）
    val_metrics.json                       最优轮验证指标
    test_metrics.json                      测试集指标（方法冻结后一次）
    efficiency.json                        参数/FLOPs/延迟/显存
    run_meta.json                          命令行、开始/结束时间、耗时、退出码
  eca_non_se_cifar10_seed2024/
  eca_non_se_cifar10_seed7/
  baseline_cifar10_seed2024/               # 基线补种子
  baseline_cifar10_seed7/
  summary/
    stage3_comparison.csv                  每行一次运行的关键指标
    stage3_comparison.md                   均值±标准差与配对差值
    stage3_efficiency.csv                  参数量/FLOPs/延迟对照
```

规则：

1. 每个实验 ID 唯一；目录已存在且非空时训练脚本默认拒绝写入（现有 `prepare_output_dir` 行为）。
2. 每次运行在 `run_meta.json` 记录完整命令行与 Git 提交哈希。
3. 汇总脚本只读取各实验目录，不重写单次结果。
4. `outputs/**/*.pt` 保持被 `.gitignore` 忽略；JSON/JSONL/YAML 可入库。

## 8. 实施步骤（批准后执行）

1. **修复 P1 缺陷**：`apply_overrides` 用 `yaml.safe_load` 解析值；配置中显式记录 scheduler 与 transforms；训练开始时保存 `split_indices.json`。
2. **实现 ECA 模块**（`src/mobilenet_attention/models/eca.py`）：
   - `ECALayer(channels, gamma=2, b=1, kernel_size=None)`；
   - `forward`：`gap → conv1d → sigmoid → x * s`；
   - 单元测试：形状保持、梯度有限、参数量等于 k、`k` 为奇数、`k` 自适应公式与论文一致。
3. **实现插入逻辑**（`models/attention.py` 或 `baseline.py` 的构建选项）：
   - 配置项 `model.attention: {type: eca, placement: non_se_blocks, gamma: 2, b: 1}`；
   - `pretrained: none` 时随机初始化；`pretrained: imagenet1k_v1` 时先加载官方权重，再随机初始化 ECA（ECA 无预训练权重，必须在报告中说明）。
4. **验证**：形状、梯度、参数量、FLOPs 与解析值一致（允许 ±1 的取整差异）；单批次训练；200 张小样本过拟合。
5. **冒烟训练**：`configs/smoke_eca.yaml`，10 个批次，确认保存/加载/日志。
6. **正式训练**：6 次主实验，逐次检查 `train_log.jsonl`。
7. **测试集评估**：全部训练完成后，用 `scripts/evaluate.py` 对每个 `best.pt` 各评估一次。
8. **汇总与误差分析**：生成 `summary/`，比较混淆矩阵差异与分类别 F1。
9. **测试与质量门禁**：`pytest`、`ruff check`、`ruff format --check` 全通过。

## 9. 预期结果与风险

| 风险 | 说明 | 应对 |
| --- | --- | --- |
| ECA 无提升 | CIFAR-10 仅 15 epochs，基线已达 96.8%，提升空间小 | 如实报告；补充"不提升"的分析，不调测试集 |
| 提升小于种子波动 | 3 个种子的标准差可能大于 ECA 增益 | 报告配对差值；必要时增加种子至 5 个 |
| 与 SE 叠加导致冗余 | 主方案已避开 SE 块 | 消融 A1 可量化叠加效果 |
| ECA 随机初始化影响收敛 | 预训练骨干 + 随机 ECA | 使用 warmup；报告首轮验证指标 |
| 延迟测量噪声 | 桌面程序占用 GPU | 测量前记录 GPU 状态；5 轮取均值±标准差 |
| 参数量对比误导 | A2 替换 SE 会减少 1.51M 参数 | 单独表格，明确标注"参数减少"与"注意力替换"两个因素 |

## 10. 推荐执行命令（批准后）

```powershell
# 0) 先完成基线冻结提交
git add -A
git commit -m "chore: freeze MobileNetV3-Large CIFAR-10 baseline and audit evidence"

# 1) 实现 ECA 后运行单元测试
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts

# 2) 冒烟
.\.venv\Scripts\python.exe scripts/train.py --config configs/smoke_eca.yaml

# 3) 主实验（6 次，逐个执行并检查日志）
.\.venv\Scripts\python.exe scripts/train.py --config configs/eca_non_se.yaml --set experiment.seed=42
.\.venv\Scripts\python.exe scripts/train.py --config configs/eca_non_se.yaml --set experiment.seed=2024
.\.venv\Scripts\python.exe scripts/train.py --config configs/eca_non_se.yaml --set experiment.seed=7
.\.venv\Scripts\python.exe scripts/train.py --config configs/baseline_mobilenetv3_large.yaml --set experiment.seed=2024 --set experiment.id=baseline_cifar10_seed2024
.\.venv\Scripts\python.exe scripts/train.py --config configs/baseline_mobilenetv3_large.yaml --set experiment.seed=7 --set experiment.id=baseline_cifar10_seed7

# 4) 效率测量
.\.venv\Scripts\python.exe scripts/measure_efficiency.py --config configs/eca_non_se.yaml

# 5) 汇总
.\.venv\Scripts\python.exe scripts/summarize_stage3.py
```

> 注意：`--set` 的布尔解析缺陷需在步骤 1 修复后，才能安全使用 `--set experiment.seed=...` 这类数值覆盖（数值本身不受影响，但同一机制下的布尔项会出错）。

## 11. 本阶段明确未执行的操作

1. 未创建 `eca.py` 或任何注意力模块。
2. 未修改 `build_mobilenet_v3_large`。
3. 未新增 `configs/eca_non_se.yaml` 或 `configs/smoke_eca.yaml`。
4. 未启动任何 ECA 训练或测试集评估。
5. 未安装新依赖。
