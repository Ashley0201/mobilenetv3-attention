# 阶段 3A 报告：配置修复、ECA 实现与冒烟验证

- 执行日期：2026-10-21
- 起点提交：`9523cda`（`main` 与 `origin/main` 同步）
- 状态：**未提交、未推送**；未启动正式多种子训练；未重新评估基线测试集
- 原始基线产物 `outputs/baseline_mobilenetv3_large_cifar10/` 未改动

## 1. 配置与训练入口审计结论

| 编号 | 问题 | 影响 | 处理 |
| --- | --- | --- | --- |
| C1 | `apply_overrides` 不解析类型 | `--set training.amp=false` 经 `bool("false")` 变 True；`grad_clip=null` 变字符串导致 `float()` 崩溃 | 新增 `parse_scalar`，用 `yaml.safe_load` 解析；覆盖键显式写入，`null` 真正变 `None` |
| C2 | 调度器参数只在代码里 | 快照无法复现 `eta_min`、warmup 起点比例 | 配置新增 `training.scheduler` 段，`build_scheduler` 接收 `warmup_start_factor` |
| C3 | `evaluation.monitor` 未被读取 | 配置写 `val_loss` 也不生效 | 实现 monitor 校验与方向判断（accuracy 越大越好、loss 越小越好） |
| C4 | `report_metrics` 未被使用 | 配置项形同虚设 | 校验取值；`test_metrics.json` 保存完整指标并额外给出 `reported` 子集 |
| C5 | 变换链未记录 | 只记录增强参数，无法还原具体算子 | 新增 `transform_spec()`，写入 `dataset.transforms` |
| C6 | 划分索引未保存 | 无法独立验证划分 | 新增 `split_indices.json`（索引 + SHA256） |
| C7 | 无运行元数据 | 无法追溯命令与耗时 | 新增 `run_meta.json`（命令、起止时间、耗时、退出码） |
| C8 | 环境快照缺驱动/脏标记 | 无法判断结果对应哪次代码状态 | 增加 `nvidia_driver`、`git_dirty`、确定性开关状态 |
| C9 | `model.name` 未使用 | 配置可写任意值而不报错 | 未新增多模型支持，保持现状；在报告中标注为已知限制 |

## 2. ECA 实现

新增 `src/mobilenet_attention/models/eca.py`：

```text
k = |log2(C)/gamma + b/gamma|_odd，gamma=2，b=1
GAP → Conv1d(1,1,k,padding=k//2,bias=False) → Sigmoid → 逐通道相乘
```

- 输入输出形状一致 `[B, C, H, W]`
- 参数量等于 `k`（无 bias、无降维）
- 实测核大小：C=16/64/72 → k=3；C=184/200/240 → k=5

新增 `src/mobilenet_attention/models/attention.py`：

- 通过 `groups > 1` 定位深度卷积（兼容块 1 无 expand 卷积的结构）
- 插入位置 = 深度卷积之后、投影卷积之前
- 支持 `non_se_blocks`（默认）、`se_blocks_only`、`all_blocks`
- `build_mobilenet_v3_large` 新增 `attention` 参数；**默认 None，基线行为不变**
- 顺序保证：先加载 ImageNet 预训练，再插入 ECA，避免 state_dict 键错位

## 3. 测试与静态检查（实际执行）

```text
.\.venv\Scripts\python.exe -m pytest tests/ -q
90 passed in 12.47s

.\.venv\Scripts\python.exe -m ruff check src tests scripts
All checks passed!

.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
32 files already formatted
```

新增测试覆盖：

| 文件 | 内容 |
| --- | --- |
| `tests/test_eca.py`（22 项） | 核大小公式、奇数约束、形状保持、缩放语义、参数数、无 bias、梯度、1×1 输入、固定 k、三种插入位置、插入位置正确性、参数增量、前向/反向、元数据、未知类型报错、checkpoint 往返、CUDA 前后向 |
| `tests/test_config.py`（新增 7 项） | `false`/`true`/`null`/整数/浮点/字符串解析、布尔回归、null 回归、快照类型保真 |
| `tests/test_train_script.py`（新增 8 项） | scheduler 默认与显式、非法 scheduler、monitor 校验、指标校验、monitor 方向、划分哈希、指标筛选、run_meta |
| `tests/test_engine.py`（新增 4 项） | warmup 曲线、eta_min 下限、梯度裁剪确实减小更新、label smoothing 确实增大损失 |
| `tests/test_data.py`（新增 3 项） | 变换链记录、关闭翻转、记录与实际 Compose 顺序一致 |

## 4. 配置修复的实际验证

用 `--set` 覆盖布尔与 null，读取 `train_log.jsonl` 的 `optim` 事件与快照：

```text
amp False  grad_clip None  learning_rate 0.0005
snapshot amp False  grad_clip None  lr 0.0005
```

对比：修复前 `bool("false")` 会得到 True，`float("null")` 会直接抛错。

调度器一致性：用修复后的 `build_scheduler` 重算 15 轮学习率，与阶段 2 基线日志逐轮比对：

```text
max abs diff = 0.0
identical within 1e-12: True
```

说明配置化后的调度器与阶段 2 行为完全一致，基线结果仍可比。

## 5. 结构核对（真实 torchvision 结构）

`torchvision.models.mobilenet_v3_large` 共 17 个 `features` 项：

- 索引 0、16：`Conv2dNormActivation`
- 索引 1–15：`InvertedResidual`
- 含 SE 的块：4, 5, 6, 11, 12, 13, 14, 15
- 不含 SE 的块：1, 2, 3, 7, 8, 9, 10

插入结果（实测 `model.attention["inserted_blocks"]`）：

| 块索引 | 位置 | 通道 | k |
| --- | --- | --- | --- |
| 1 | 1 | 16 | 3 |
| 2 | 2 | 64 | 3 |
| 3 | 2 | 72 | 3 |
| 7 | 2 | 240 | 5 |
| 8 | 2 | 200 | 5 |
| 9 | 2 | 184 | 5 |
| 10 | 2 | 184 | 5 |

块 1 的位置为 1，其余为 2，因为块 1 没有 expand 卷积，深度卷积本身位于索引 0。

## 6. 参数量与计算开销（实测）

### 参数量

| 模型 | 总参数 | backbone | classifier | ECA |
| --- | --- | --- | --- | --- |
| 基线 | 4,214,842 | 2,971,952 | 1,242,890 | 0 |
| ECA | 4,214,871 | 2,971,981 | 1,242,890 | 29 |

- 增量 **+29**（+0.000688%）
- 模型体积 16.1718 → 16.1719 MiB（+0.0001 MiB）
- 与阶段 2 解析预估的 29 完全一致

### FLOPs / MACs

`torch.utils.flop_counter` 实测（batch=1，224×224）：

| 模型 | FLOPs | MACs |
| --- | --- | --- |
| 基线 | 430,645,120 | 215,322,560 |
| ECA | 430,654,112 | 215,327,056 |
| 差值 | +8,992 | +4,496 |

- 计数器只统计卷积与全连接，因此差值恰好等于 7 个 1D 卷积的 `k·C` 之和（4,496），**不含** GAP、Sigmoid、逐元素乘法
- 用真实前向钩子取形状后解析计算完整 ECA 开销：

| 项目 | MACs |
| --- | --- |
| GAP | 785,568 |
| 1D 卷积 | 4,496 |
| 逐通道相乘 | 785,568 |
| 合计 | 1,575,632（占基线 0.7318%） |

结论：**必须同时报告两个口径**，只报计数器会低估约 350 倍（4,496 vs 1,575,632）。

口径差异说明：阶段 2 审计的 790,064 MACs 只计 GAP 与 1D 卷积（785,568 + 4,496），本次实测把逐通道相乘也计入（+785,568），因此合计 1,575,632。两者不矛盾，是口径不同；论文中采用完整口径 1,575,632（+0.7318%）。

### 延迟（实测，含 GPU 状态记录）

单独测量（`outputs/stage3a/*_efficiency.json`，5 轮 × 50 次）：

| 模型 | 均值 ms | 标准差 |
| --- | --- | --- |
| 基线 | 4.933 | 0.163 |
| ECA | 5.592 | 0.201 |

交替配对测量（`paired_latency.json`，6 轮 × 100 次，B/E 顺序交替）：

| 模型 | 均值 ms | 标准差 | 中位数 |
| --- | --- | --- | --- |
| 基线 | 5.046 | 0.087 | 5.027 |
| ECA | 5.781 | 0.129 | 5.771 |

- 配对差值 **+0.735 ms（+14.6%）**
- 多 batch size 对照（`batch_latency.json`，4 轮 × 30 次）：

| batch | 基线 ms | ECA ms | 差值 | 相对 |
| --- | --- | --- | --- | --- |
| 1 | 5.04 | 5.83 | +0.79 | +15.6% |
| 16 | 6.74 | 7.06 | +0.32 | +4.8% |
| 128 | 81.00 | 85.81 | +4.81 | +5.9% |

- batch=1 的相对开销最大，符合 ECA 引入多个小算子（GAP、1D 卷积、Sigmoid、广播乘）导致 kernel 启动开销占比升高的解释；该解释为推断，未做 profiler 分解
- GPU 状态：测量前温度 48–49 °C、SM 时钟 210 MHz（空闲），测量后 52–53 °C、2250 MHz。已记录在 JSON 中

### 训练显存峰值（batch=128，AMP）

| 模型 | 峰值 MiB |
| --- | --- |
| 基线 | 2,912.3 |
| ECA | 3,108.0 |
| 差值 | +195.7（+6.7%） |

## 7. 冒烟与流程验证

| 检查 | 结果 |
| --- | --- |
| `smoke_eca`（10 批次，1 epoch） | 通过，val_acc 0.1172（随机初始化，无意义，仅验证流程） |
| `smoke_baseline_stage3a` | 通过；与阶段 2 `smoke_baseline` 的 loss/acc 完全一致（2.311295 / 0.128125 / 2.304878 / 0.1171875），说明基线路径未被改动 |
| `--set` 布尔/null 覆盖 | amp=False、grad_clip=None 已写入日志与快照 |
| `evaluation.monitor=val_loss` | best 事件记录 `monitor=val_loss`，选最小 loss |
| 非法 monitor | 启动即报 `ValueError`，不产生半成品目录 |
| ECA checkpoint 往返 | 7 个 `conv.weight` 形状 `(1,1,3)`×3、`(1,1,5)`×4，加载后参数逐位一致 |
| 恢复训练 | 2 epochs 后从 `last.pt` 续跑至 3 epochs，事件含 `resume start_epoch=2` |
| 预训练兼容 | 插入 ECA 后 `features.0.0.weight` 与官方权重逐位相等；块 4 的 SE 参数保留 |
| 划分一致性 | `split_indices.json` 哈希 `439d2886…` / `6781efc5…`，与阶段 2 审计一致 |
| 独立评估 | `scripts/evaluate.py` 能加载 ECA 检查点并输出 `attention` 元数据 |

说明：`smoke_eca` 的准确率 0.1172、`smoke_eca_val_recheck.json` 的 0.1 均为随机初始化单轮结果，**不是实验结果**。

## 8. 修改文件清单

新增：

```text
src/mobilenet_attention/models/eca.py
src/mobilenet_attention/models/attention.py
scripts/measure_efficiency.py
scripts/paired_latency.py
scripts/batch_latency.py
configs/eca_non_se.yaml
configs/smoke_eca.yaml
tests/test_eca.py
```

修改：

```text
src/mobilenet_attention/utils/config.py      # parse_scalar + 覆盖语义
src/mobilenet_attention/utils/env.py         # 驱动、脏标记、确定性
src/mobilenet_attention/data/cifar10.py      # transform_spec
src/mobilenet_attention/data/__init__.py
src/mobilenet_attention/engine/trainer.py    # scheduler 参数化
src/mobilenet_attention/models/baseline.py   # attention 参数
src/mobilenet_attention/models/__init__.py
scripts/train.py                             # 配置校验、快照、划分、run_meta、monitor
scripts/evaluate.py                          # 支持 ECA 检查点
configs/baseline_mobilenetv3_large.yaml      # 显式 scheduler（行为不变）
configs/smoke.yaml                           # 显式 scheduler
tests/test_config.py
tests/test_data.py
tests/test_engine.py
tests/test_train_script.py
```

## 9. 实际执行的命令

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
.\.venv\Scripts\python.exe scripts/measure_efficiency.py --model baseline
.\.venv\Scripts\python.exe scripts/measure_efficiency.py --model eca_non_se
.\.venv\Scripts\python.exe scripts/paired_latency.py
.\.venv\Scripts\python.exe scripts/batch_latency.py
.\.venv\Scripts\python.exe scripts/train.py --config configs/smoke_eca.yaml
.\.venv\Scripts\python.exe scripts/train.py --config configs/smoke.yaml --set experiment.id=smoke_baseline_stage3a
.\.venv\Scripts\python.exe scripts/train.py --config configs/smoke_eca.yaml --set experiment.id=smoke_eca_override --set training.amp=false --set training.grad_clip=null --set training.learning_rate=0.0005
.\.venv\Scripts\python.exe scripts/train.py --config configs/smoke_eca.yaml --set experiment.id=smoke_eca_monitor_loss --set evaluation.monitor=val_loss
.\.venv\Scripts\python.exe scripts/train.py --config configs/smoke_eca.yaml --set evaluation.monitor=test_accuracy
.\.venv\Scripts\python.exe scripts/train.py --config configs/smoke.yaml --set experiment.id=smoke_amp_off --set training.amp=false --set training.grad_clip=null --set training.warmup_epochs=0
.\.venv\Scripts\python.exe scripts/train.py --config configs/smoke_eca.yaml --set experiment.id=smoke_eca_resume3 --set training.epochs=2 --set training.limit_train_batches=3 --set evaluation.limit_val_batches=1
.\.venv\Scripts\python.exe scripts/train.py --config configs/smoke_eca.yaml --set experiment.id=smoke_eca_resume3 --set training.epochs=3 --set training.limit_train_batches=3 --set evaluation.limit_val_batches=1 --resume outputs/smoke_eca_resume3/last.pt --overwrite
.\.venv\Scripts\python.exe scripts/evaluate.py --checkpoint outputs/smoke_eca/best.pt --split val --output outputs/stage3a/smoke_eca_val_recheck.json
```

## 10. 新增产物（均在独立目录）

```text
outputs/stage3a/
  baseline_efficiency.json      基线参数/FLOPs/延迟/显存
  eca_non_se_efficiency.json    ECA 参数/FLOPs/延迟/显存/解析开销
  paired_latency.json           交替配对延迟（6 轮 × 100 次）
  batch_latency.json            bs=1/16/128 对照
  smoke_eca_val_recheck.json    ECA 检查点验证集复算（未用测试集）
outputs/smoke_eca*/             冒烟实验目录（各含快照、日志、run_meta、split_indices）
outputs/smoke_baseline_stage3a/
outputs/smoke_amp_off/
```

## 11. 未解决风险与限制

1. **延迟开销无法用 MACs 解释**：完整 ECA MACs 仅 +0.73%，但 bs=1 延迟 +15.6%。可能由小算子启动开销与 Python 层额外调用导致，**未用 profiler 验证**，论文中不得把延迟差值归因为计算量。
2. **单次测量噪声**：配对测量标准差 0.09–0.13 ms，差值 0.735 ms 约为标准差的 6–8 倍，方向上稳定；但跨机器、跨驱动不可比。
3. **训练显存 +6.7%** 只测了一次训练步，未取多次峰值统计。
4. **`model.name` 仍未生效**：配置可写任意名称，代码固定构建 MobileNetV3-Large。
5. **ECA 权重为随机初始化**：ImageNet 预训练权重不含 ECA，训练时必须从随机 ECA 开始；需在论文中说明。
6. **仅验证流程，未验证精度**：ECA 是否能提升 CIFAR-10 准确率尚不可知，需正式多种子实验。
7. **`git_dirty=true`**：当前所有快照都记录了脏工作区，正式实验前必须先提交代码。
8. **恢复训练时 `--overwrite` 会重写目录**：现有实现允许覆盖，正式实验应改用新实验 ID 而非覆盖。
9. **`report_metrics` 语义**：`test_metrics.json` 现在同时保存完整指标与 `reported` 子集，字段比阶段 2 多一层嵌套，下游脚本需注意。
10. **未测训练吞吐**：只测了推理延迟与显存，未测每轮实际训练时间差异。

## 12. Git 工作区状态

- 分支 `main`，HEAD `9523cda`，与 `origin/main` 同步
- 15 个已跟踪文件被修改，11 项未跟踪（3 个脚本、2 个模型文件、2 个配置、1 个测试、若干 outputs 目录）
- 未执行 `git add`、`git commit`、`git push`、`git tag`
- `outputs/**/*.pt` 仍被 `.gitignore` 忽略；JSON/JSONL/YAML 可入库

## 13. 进入正式实验前必须完成

1. 审阅本报告与代码差异，确认 ECA 插入位置与开销口径。
2. 提交阶段 3A 代码，使后续 `environment.json` 的 `git_commit` 对应干净工作区。
3. 明确正式实验矩阵：基线 + ECA × 种子 {42, 2024, 7}，共 6 次。
4. 决定是否补做 profiler 分解，解释延迟差值来源。
5. 正式实验前删除或隔离 `outputs/smoke_*`，避免与正式结果混淆。
