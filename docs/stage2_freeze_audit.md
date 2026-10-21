# 基线冻结与可复现性审计报告（阶段 2 收尾）

- 审计日期：2026-10-20
- 审计对象：`outputs/baseline_mobilenetv3_large_cifar10/`
- 审计方式：只读检查 + 独立只读复算；**未重新训练、未重跑测试集评估、未修改基线算法**
- 证据目录：`outputs/stage2_audit/`

## 1. 审计结论摘要

| 结论 | 说明 |
| --- | --- |
| 基线结果可信 | 权重、配置、日志、指标四者可相互印证；测试集指标两次独立复算完全一致 |
| 可复现性部分成立 | 同机同环境可复现；跨机逐位复现不保证（确定性设置 + `warn_only`） |
| 不可提交 | Git 仓库尚无任何提交；`environment.json` 的 `git_commit` 为 null |
| 已知缺口 | 配置快照未记录调度器参数；`--set` 覆盖存在布尔解析缺陷；FLOPs 为本次审计新增，原实验未记录 |

## 2. 逐项审计

### 2.1 Git 状态

```text
分支: main
提交: 无（fatal: your current branch 'main' does not have any commits yet）
已跟踪文件: 0
工作区: 全部为未跟踪（??）
```

- 无提交意味着当前代码状态无法与实验结果建立版本对应。
- `git config user.name = daiyuchen`，`user.email = 3041605069@qq.com`，无 remote。
- 未执行 `git commit`（未获授权）。

### 2.2 `.gitignore` 有效性

`git check-ignore -v` 实测：

| 路径 | 结果 |
| --- | --- |
| `.venv` | 被 `.gitignore:10` 忽略 |
| `data` | 被 `.gitignore:22` 忽略 |
| `outputs/.../best.pt` | 被 `.gitignore:38` 忽略 |
| `outputs/.../last.pt` | 被 `.gitignore:38` 忽略 |
| `outputs/logs/baseline_stdout.log` | 被 `.gitignore:50` 忽略 |
| `.pytest_cache` / `.ruff_cache` | 被忽略 |

`git add -A --dry-run` 共 59 项（含本次审计新增的 7 个文件），总大小约 0.13 MB，**不含**权重、数据集、虚拟环境、缓存、日志。通过。

> 注：`.gitignore` 中文注释在 PowerShell 5.1 默认编码下显示为乱码，但文件本身为 UTF-8 且无替换字符（实测 `replacement chars = 0`）。不影响 Git 行为。

### 2.3 基线身份核验

| 项目 | 要求 | 实际 | 结论 |
| --- | --- | --- | --- |
| 架构 | MobileNetV3-Large | `mobilenet_v3_large`，312 个 state_dict 张量，`classifier.3.weight` 为 (10, 1280) | 通过 |
| 预训练 | ImageNet V1 | `pretrained: imagenet1k_v1`；缓存文件 SHA256 `8738CA79...9A04` | 通过 |
| 预训练真实性 | 非随机初始化 | 与官方权重相对 L2：`features.0.0` 1.19%、`features.4.block.0.0` 3.34%、`features.16.0` 13.69%；与随机初始化相对 L2 均 > 2.4 | 通过 |
| 数据集 | CIFAR-10 | 官方 MD5 `C58F30108F718F92721AF3B95E74349A`；Torchvision `_check_integrity() = True` | 通过 |
| 随机种子 | 42 | 配置、checkpoint 内配置、日志均一致 | 通过 |
| 类别顺序 | 10 类固定顺序 | `airplane...truck`，配置快照与 checkpoint 内均保存 | 通过 |
| 参数量 | — | 4,214,842 | 通过 |

### 2.4 数据划分

| 项目 | 实际 |
| --- | --- |
| 训练集 | 45,000，每类 4,500 |
| 验证集 | 5,000，每类 500 |
| 测试集 | 10,000（官方 test） |
| 训练/验证交集 | 0 |
| 训练索引 SHA256 | `439d2886be7d934cfcf5add00aa6e368bb431dd849d6a130530d1dc7cee35480` |
| 验证索引 SHA256 | `6781efc5c7c0ad15f37adeac39dfdc1ba7cb6ad7a65e07d775bd3aec67050a12` |
| 重新计算一致性 | 完全一致（`recompute_identical: true`） |
| 分层策略 | `train_test_split(test_size=5000, stratify=targets, random_state=42)` |

类别映射与 `torchvision.datasets.CIFAR10.class_to_idx` 一致。

### 2.5 测试集使用时机

`train_log.jsonl` 事件顺序（共 31 条）：

```text
索引 0    start
索引 1    dataset
索引 2    model
索引 3-27 epoch（15 条）+ best（10 条）
索引 28   latency
索引 29   test
索引 30   finish
```

- 测试评估发生在全部 15 个 epoch 与延迟测量**之后**，且使用 `best.pt`（epoch 12）权重。
- 训练循环内只对 `loaders["val"]` 评估；`loaders["test"]` 仅在 `test_after_training` 分支使用一次。
- 复核：`scripts/evaluate.py` 分别在 22:44 与 22:45 对测试集复算两次，结果完全相同（acc 0.9679、F1 0.9678665670080754），说明评估是确定性的；这两次复算属于阶段 2 已发生的记录，**本次审计未再运行测试集**。

结论：测试集仅用于训练结束后的最终评估，未参与模型选择。通过。

> 说明：审计中额外执行了 `--split val` 的复算（新增 `outputs/stage2_audit/val_recheck.json`），验证集不属于测试集，不消耗测试集配额。

### 2.6 配置与元信息完整性

| 项目 | 是否保存 | 位置 |
| --- | --- | --- |
| 类别顺序 | 是 | `config_snapshot.yaml` → `dataset.class_names`；checkpoint `config.dataset` |
| 训练变换 | 部分 | 仅记录 `augmentation.random_resized_crop_scale`、`horizontal_flip`；具体变换链未以字符串形式保存 |
| 归一化参数 | 是 | `dataset.normalization.mean/std`（ImageNet） |
| 输入尺寸 | 是 | `dataset.input_size = 224` |
| 优化器 | 部分 | 记录 `optimizer: adamw`、`lr`、`weight_decay`；`train.py` 实际硬编码 `torch.optim.AdamW` |
| 学习率调度 | **否** | 仅记录 `warmup_epochs: 1`、`epochs: 15`；余弦退火类型与 `eta_min=1e-6` 只存在于代码中 |
| 随机种子 | 是 | `experiment.seed = 42`、`dataset.split.seed = 42` |
| 最佳 epoch | 是 | `best.pt` 的 `epoch = 12`；`val_metrics.json` 的 `epoch = 12` |
| 逐 epoch 学习率 | 是 | `train_log.jsonl` 每条 epoch 记录 |
| 环境快照 | 是 | `environment.json`（但 `git_commit` 为 null） |
| 数据划分索引 | **否** | 仅记录规模与策略，未保存索引列表或哈希；本次审计补算并写入 `baseline_audit.json` |

### 2.7 延迟测量严谨性

代码实现（`src/mobilenet_attention/engine/trainer.py:158`）：

```text
model.eval() → torch.no_grad()
→ warmup=10 次前向
→ torch.cuda.synchronize()
→ time.perf_counter() 计时 repeats=50 次
→ torch.cuda.synchronize()
→ 平均
```

| 检查项 | 结论 |
| --- | --- |
| 预热 | 有（10 次） |
| CUDA 同步 | 有（计时前后各一次） |
| 重复计时 | 有（50 次取平均） |
| 输入固定 | 是（224×224，随机张量） |
| batch size 记录 | 是（batch=1） |
| 只测前向 | 是（不含预处理、不含后处理） |
| 多次独立重复 | **原实验只有 1 轮**，未报告方差 |
| 计时器 | `perf_counter`，含 Python 循环开销 |
| GPU 状态 | 未记录时钟频率、温度、占用，无法排除降频影响 |

原实验记录：`latency_ms_per_batch = 6.760`（`latency.json`）。
本次审计独立运行 5 轮（每轮 50 次）：`[3.931, 4.283, 4.451, 4.264, 4.062]`，均值 4.198 ms，标准差 0.203 ms。
审计过程中另一次手工运行结果为 5 轮均值 4.30 ms、标准差 0.04 ms，同样明显低于 6.76 ms。

结论：**方法学正确，但单轮结果不足以作为论文数据**。原始 6.76 ms 与审计时 4.0–4.5 ms 差异约 38%，最可能原因是审计时 GPU 空闲（桌面程序显存占用从约 1.7 GB 降至 0.75 GB）与时钟状态不同，而非代码差异。建议阶段 3 起统一报告 5 轮 × 50 次的均值±标准差，并记录 GPU 温度/时钟。原始 `latency.json` **不覆盖**。

### 2.8 FLOPs / MACs

- 原实验（阶段 2）**未测量**，`stage2_benchmark.json` 无 FLOPs 字段。
- 本次审计用 `torch.utils.flop_counter.FlopCounterMode` 实测（batch=1，224×224，eval，no_grad）：

```text
总 FLOPs   430,645,120
总 MACs    215,322,560（= FLOPs / 2）
卷积       428,161,920
全连接      2,483,200
```

- 计数口径已用 3×4×8×8 卷积小例子校核：计数 = 2×MACs，与 `2·M·N·K` 定义一致。
- 这是 shape-based 理论值，不是硬件实测吞吐；已在该 JSON 的 `convention` 字段标注。

### 2.9 全新终端可运行性

在 `C:\Users\Sylvia\AppData\Local\Temp\opencode`（项目外目录）用绝对路径调用 `.venv` 解释器：

```text
cwd            C:\Users\Sylvia\AppData\Local\Temp\opencode
sys.prefix     C:\Users\Sylvia\Desktop\mobilenetv3-attention\.venv
mobilenet_attention  C:\Users\Sylvia\Desktop\mobilenetv3-attention\src\mobilenet_attention\__init__.py
torch.cuda.is_available()  True
```

`scripts/check_env.py --json` 与 `scripts/train.py --help` 均能在项目外目录正常执行。可编辑安装（`pip install -e .`）使包路径与工作目录解耦。通过。

依赖一致性：`pip check` → `No broken requirements found.`；`requirements.txt` 中 `torch==2.14.1+cu130` 必须配合 `--index-url https://download.pytorch.org/whl/cu130`，否则默认 PyPI 找不到该本地版本号（实测报错）。README 已说明。

### 2.10 测试与代码质量（本次执行）

```text
.\.venv\Scripts\python.exe -m pytest tests/ -q
42 passed in 13.10s

.\.venv\Scripts\python.exe -m ruff check src tests scripts
All checks passed!

.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
25 files already formatted
```

## 3. 审计通过项与失败项

### 通过

1. 模型确为 MobileNetV3-Large，分类头 (10, 1280)，参数量 4,214,842。
2. 预训练权重确为 ImageNet1K_V1，与官方缓存相对误差在微调后的合理范围内，非随机初始化。
3. 数据集为官方 CIFAR-10，MD5 与 Torchvision 内部校验均通过。
4. seed=42 在配置、checkpoint、日志三处一致；数据划分可重算且完全一致。
5. 训练/验证/测试划分互斥，分层均衡。
6. 测试集仅在训练结束后评估一次，未参与调优。
7. 最佳 epoch（12）与 `best.pt`、`val_metrics.json` 一致。
8. 逐 epoch 损失、准确率、学习率、耗时完整记录。
9. 检查点含模型、优化器、调度器状态与配置，可恢复训练。
10. 延迟测量包含预热、CUDA 同步、重复计时。
11. `.gitignore` 有效，权重/数据/环境/缓存均未进入暂存区。
12. 项目可在任意工作目录通过 `.venv` 绝对路径运行。
13. `pytest` 42 项通过，Ruff 检查与格式检查通过。

### 失败 / 缺陷

1. **无 Git 提交**：代码版本与实验结果无法对应。
2. **调度器配置未入快照**：`CosineAnnealingLR`、`eta_min=1e-6`、`SequentialLR` 结构只存在于代码。
3. **数据划分索引未保存**：仅保存规模与策略，无法从产物独立验证划分（本次审计补算了哈希）。
4. **`--set` 覆盖布尔值失效**：`apply_overrides` 不做类型转换，`training.amp=false` 会被 `bool("false")` 判为 `True`；`grad_clip=null` 会变成字符串 `"null"` 导致 `float()` 报错。
5. **`evaluation.monitor`、`report_metrics`、`model.name` 未生效**：配置项被读取但训练脚本未使用（监控指标硬编码为 `val_accuracy`）。
6. **延迟单轮、无方差**：原始 6.76 ms 与审计 4.2–4.3 ms 差异显著，缺少统计口径。
7. **原实验未测 FLOPs**：本次审计补测，但原始产物中缺失。
8. **训练变换未完整记录**：只记录增强参数，未记录变换链与 `Resize` 插值方式（`antialias=True`）。
9. **`environment.json` 缺 `git_commit`**、缺 GPU 驱动版本与 CUDA 运行时（仅记录编译版本）。
10. **无多种子重复**：单种子结果无法给出置信区间。

## 4. 当前基线结果可信度说明

| 维度 | 评价 |
| --- | --- |
| 数值真实性 | 高。日志、`val_metrics.json`、`test_metrics.json`、`best.pt` 内记录四者一致；两次独立复算完全一致 |
| 协议合规性 | 高。测试集仅最终使用一次；验证集用于选 checkpoint |
| 可追溯性 | 中。配置、环境、日志齐全；但缺 Git 提交、缺划分索引、缺调度器参数 |
| 统计稳健性 | 低。单种子、单次运行，无法区分改进效果与随机波动 |
| 与文献可比性 | 中。CIFAR-10 上 96.79% 属于合理区间，但训练协议（224 上采样、15 epochs、AdamW）与常见报告不同，不能直接与论文表格并列 |

结论：**作为阶段 3 的内部对照基准可用；作为论文最终数字需补齐多种子与配置记录**。

## 5. 修复优先级

| 优先级 | 问题 | 建议修复 | 是否影响现有结果 |
| --- | --- | --- | --- |
| P0 | 无 Git 提交 | 按第 6 节命令提交 | 否 |
| P0 | 调度器/变换链未入快照 | 在配置中显式写 `scheduler`、`transforms` 段，并在训练脚本中读取 | 否（仅影响后续实验） |
| P1 | `--set` 布尔解析缺陷 | 用 `yaml.safe_load` 解析覆盖值 | 否 |
| P1 | 划分索引未保存 | 训练开始时保存 `split_indices.json` 或哈希 | 否 |
| P1 | 延迟单轮无方差 | 报告 5 轮 × 50 次均值±标准差，记录 GPU 状态 | 否（新增测量，不覆盖旧文件） |
| P2 | `monitor` / `report_metrics` / `model.name` 未生效 | 要么实现，要么从配置删除以免误导 | 否 |
| P2 | 环境快照缺驱动/CUDA runtime | 扩展 `environment_snapshot` | 否 |
| P2 | 单种子 | 阶段 3 起至少 3 个种子 | 否 |
| P3 | 变换链细节 | 在配置中写全 | 否 |

## 6. 建议的 Git 提交方案（未执行）

当前无任何提交，建议拆成两次提交，先冻结代码与文档，再单独提交实验记录。

### 提交 1：代码与文档

```powershell
git add .gitignore AGENTS.md README.md pyproject.toml requirements.txt
git add configs scripts src tests
git add docs
git commit -m "chore: initialize MobileNetV3 baseline project with reproducible training pipeline"
```

建议纳入版本控制的文件：

```text
.gitignore
AGENTS.md
README.md
pyproject.toml
requirements.txt
configs/baseline_mobilenetv3_large.yaml
configs/dataset_cifar10.yaml
configs/smoke.yaml
docs/dataset_selection.md
docs/environment_report.md
docs/stage2_baseline_report.md
docs/stage2_freeze_audit.md
scripts/audit_baseline.py
scripts/benchmark.py
scripts/check_env.py
scripts/download_cifar10.py
scripts/evaluate.py
scripts/sanity_check.py
scripts/train.py
src/mobilenet_attention/**/*.py
tests/*.py
outputs/.gitkeep
```

### 提交 2：基线实验记录（不含权重）

```powershell
git add outputs/baseline_mobilenetv3_large_cifar10/*.json
git add outputs/baseline_mobilenetv3_large_cifar10/*.jsonl
git add outputs/baseline_mobilenetv3_large_cifar10/*.yaml
git add outputs/smoke_baseline outputs/stage2_benchmark.json outputs/stage2_sanity_check.json
git add outputs/stage2_audit
git commit -m "docs: record frozen MobileNetV3-Large CIFAR-10 baseline results"
```

明确**不纳入**：`data/`、`.venv/`、`outputs/**/*.pt`、`outputs/logs/`、`.pytest_cache/`、`.ruff_cache/`、`__pycache__/`。

### 提交后建议打标签

```powershell
git tag -a baseline-v1 -m "Frozen MobileNetV3-Large CIFAR-10 baseline (val 0.9694 / test 0.9679, seed 42)"
```

## 7. 审计产物清单

```text
outputs/stage2_audit/
  baseline_audit.json       检查点/划分/FLOPs/延迟/ECA 开销的只读审计结果
  artifact_manifest.sha256  基线 10 个产物的 SHA256 清单
  code_manifest.sha256      38 个源码/配置/文档的 SHA256 清单
  val_recheck.json          验证集复算（不使用测试集）
```

原始基线产物未做任何修改，哈希与审计前一致（见 `artifact_manifest.sha256`）。

## 8. 未执行的操作（明确声明）

1. 未执行 `git commit`、`git add`、`git tag`。
2. 未重新训练基线。
3. 未重新运行测试集评估。
4. 未修改模型算法、配置默认值或既有实验产物。
5. 未安装任何新依赖（FLOPs 使用 PyTorch 内置 `torch.utils.flop_counter`）。
6. 未删除任何文件。
