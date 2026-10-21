# 阶段 3A 收尾审计报告

- 审计日期：2026-10-21
- 起点提交：`9523cda`（`main` 与 `origin/main` 同步）
- 状态：**未提交、未推送、未启动正式训练**
- 原始产物 `outputs/baseline_mobilenetv3_large_cifar10/` 未改动

## 1. 工作区盘点

### 1.1 Git 状态

```text
## main...origin/main
HEAD 9523cda chore: freeze MobileNetV3 baseline
```

15 个已跟踪文件修改，新增 29 项未跟踪（源码 3、测试 2、配置 2、脚本 7、文档 1、实验目录 14）。无删除、无分支变更、无 commit/tag/push。

### 1.2 变更分类

| 类别 | 文件 |
| --- | --- |
| 源码 | `models/eca.py`、`models/attention.py`、`models/registry.py`（新增）；`models/baseline.py`、`models/__init__.py`、`utils/config.py`、`utils/env.py`、`data/cifar10.py`、`data/__init__.py`、`engine/trainer.py`（修改） |
| 测试 | `tests/test_eca.py`、`tests/test_model_registry.py`（新增）；`test_config.py`、`test_data.py`、`test_engine.py`、`test_train_script.py`（修改） |
| 配置 | `configs/eca_non_se.yaml`、`configs/smoke_eca.yaml`（新增）；`baseline_mobilenetv3_large.yaml`、`smoke.yaml`（加 scheduler 段，行为不变） |
| 脚本 | `measure_efficiency.py`、`paired_latency.py`、`batch_latency.py`、`profile_models.py`、`profile_paired.py`、`analyze_profiler.py`、`efficiency_report.py` |
| 文档 | `docs/stage3a_report.md` |
| 实验指标 | `outputs/stage3a/*.json`（efficiency、latency、profiler summary、identity recheck） |
| 临时 smoke | `outputs/smoke_*`、`outputs/stage3a_audit_*`、`outputs/stage3a_final_*`、`outputs/stage3a_bad_name/` |

### 1.3 大文件与敏感内容

- 权重：所有 `*.pt` 被 `.gitignore` 忽略，`git add --dry-run` 未包含任何 `.pt`
- 数据集：`/data/` 被忽略
- 缓存：`.venv`、`.pytest_cache`、`.ruff_cache`、`__pycache__`、`.egg-info` 均被忽略
- 凭据：全仓扫描 `.env`/`*.key`/`*.pem`/`kaggle.json`/`credentials.json`/`secrets.yaml`，**未发现**
- profiler trace：6 个文件共约 310 MB，已新增规则 `outputs/**/*_trace*.json` 排除
- 暂存预览：134 个文件、约 4.59 MB，其中最大的是 `split_indices.json`（331 KB/个）

## 2. 任务 2：model.name 修复

### 2.1 实现

新增 `src/mobilenet_attention/models/registry.py`：

| 名称 | 行为 |
| --- | --- |
| `mobilenet_v3_large` | 标准 MobileNetV3-Large；传 attention 配置直接报错 |
| `mobilenet_v3_large_eca` | 强制 ECA；未给 attention 时使用默认 `{type: eca, placement: non_se_blocks}` |
| 其他 | `ValueError: 未知模型名称：...，可选 (...)` |

- `build_model()` 统一入口，训练、评估、效率脚本全部改用它
- `describe_model()` 输出运行时身份（名称、类别、注意力类型/位置/块数/参数）
- `validate_model_config()` 在创建输出目录**之前**校验，避免未知名称留下半成品目录

### 2.2 验证结果（实测）

| 检查 | 结果 |
| --- | --- |
| `model.name=mobilenet_v3_large` | 日志 `model_name=mobilenet_v3_large`、`attention_type=None`、参数 4,214,842 |
| `model.name=mobilenet_v3_large_eca` | 日志 `model_name=mobilenet_v3_large_eca`、`attention_type=eca`、块数 7、参数 4,214,871 |
| `--set model.name=resnet50` | `ValueError`，且 `outputs/stage3a_bad_name3/` 与 `stage3a_bad_name4/` **均未创建** |
| baseline 检查点评估 | `model_name=mobilenet_v3_large`、`attention_blocks=0`、val_acc 0.9694（与阶段 2 一致） |
| ECA 检查点评估 | `model_name=mobilenet_v3_large_eca`、`attention_blocks=7`、`attention_params=29` |
| 交叉加载 | baseline 权重载入 ECA 模型、反向均抛 `RuntimeError`（键不匹配） |
| 基线结构 | 参数量仍为 4,214,842，未变 |

修复前 `stage3a_bad_name/` 目录残留（旧版本在建模前未校验，留下 5 个文件、无权重）。该目录保留未删，作为审计证据。

## 3. 任务 3：输出目录覆盖风险

### 3.1 语义划分

| 场景 | 行为 | `run_meta.output_mode` |
| --- | --- | --- |
| 新实验 ID | 创建目录 | `new` |
| 已存在且非空、未授权 | `SystemExit`，不写入任何文件 | — |
| 已存在 + `--overwrite` | 允许写入（旧文件可能残留） | `overwrite` |
| `--resume` | 要求目录已存在，追加日志，**不清空** | `resume` |
| `--resume` + `--overwrite` | `SystemExit`（互斥） | — |

### 3.2 实测

| 命令 | 结果 |
| --- | --- |
| 重复 `stage3a_audit_baseline_smoke` | exit=1，提示「实验目录已存在且非空」，旧文件未变 |
| `--resume` 已存在目录 | exit=0，`output_mode=resume`，`train_log.jsonl` 保留旧事件并追加 |
| `--resume` + `--overwrite` | exit=1，提示互斥 |
| `--resume` 不存在目录 | 单测覆盖，抛 `SystemExit` |

单测新增 6 项：拒绝覆盖、允许覆盖、新目录、空目录不算覆盖、恢复不清空、恢复缺失目录报错、互斥。

## 4. 任务 4：Profiler 分析

### 4.1 协议

- 设备：RTX 4060 Laptop，batch=1，224×224，eval + no_grad
- 每次 profile 前预热 200 次；采集 50 次迭代
- 交替 3 轮（B→E→E→B→B→E），每轮记录 GPU 时钟
- 所有 6 次采集时钟均为 **2250 MHz**，消除了前几轮低频污染

### 4.2 kernel 数量（不受时钟影响）

| 指标 | baseline | ECA | 差值 |
| --- | --- | --- | --- |
| kernel 数/迭代 | 282 | 310 | **+28** |
| `cudaLaunchKernel`/迭代 | 254 | 282 | **+28** |

新增 28 个 kernel，正好等于 7 个 ECA × 4 个 kernel（GAP reduce、Conv1d、Sigmoid、逐通道乘）。

### 4.3 ECA 相关 kernel 耗时（3 轮中位数）

| 算子 | baseline us/it | ECA us/it | 差值 |
| --- | --- | --- | --- |
| GAP（reduce_mean） | 26.5 | 61.4 | +34.9 |
| Conv1d（conv2d_grouped_direct） | 0 | 14.1 | +14.1 |
| Sigmoid | 0 | 12.5 | +12.5 |
| 逐通道乘法 | 17.4 | 43.2 | +25.8 |
| **合计** | — | — | **+87.3** |

注：baseline 的 GAP 与乘法来自 MobileNetV3 自带的 8 个 SE 块，不是 ECA。

### 4.4 kernel 总耗时

| 轮次 | baseline us/it | ECA us/it |
| --- | --- | --- |
| 1 | 757.9 | 1186.5 |
| 2 | 752.6 | 855.6 |
| 3 | 757.1 | 1016.7 |
| 中位数 | 757.1 | 1016.7 |

ECA 三轮波动很大（855–1187），而 baseline 极稳定（752–758）。ECA 的 ECA kernel 合计仅 +87.3 us/it，无法解释 +259.6 us/it 的中位数差值。**说明剩余差值主要来自 profiler 干扰下的测量噪声，而不是 ECA 算子本身。**

### 4.5 内存拷贝与同步

| 指标 | baseline | ECA |
| --- | --- | --- |
| `cudaMemcpyAsync`/it | 0 | 0 |
| `cudaMemcpy`/it | 0 | 0 |
| `cudaStreamSynchronize`/it | 0 | 0 |
| `cudaDeviceSynchronize`/it | 0 | 0 |
| Memcpy 类事件耗时 | 0 us | 0 us |

**未发现额外张量拷贝或显式同步。**

### 4.6 布局转换

CPU 侧 `aten::transpose` 与 `aten::as_strided` 仅在 ECA 模型出现：

| 算子 | baseline 调用/迭代 | ECA 调用/迭代 |
| --- | --- | --- |
| `aten::adaptive_avg_pool2d` | 9 | 16 |
| `aten::transpose` | 0 | 16 |
| `aten::as_strided` | 0 | 58 |
| `aten::unsqueeze` | 0 | 21 |

这些是 `squeeze(-1).transpose(-1,-2)` 与 `transpose(...).unsqueeze(-1)` 产生的**视图操作**，`aten::transpose` 本身不复制数据，但会增加 CPU 侧调用与 kernel 参数准备开销。证据强度：**中等**（调用次数确定，但没有单独的耗时归因）。

### 4.7 profiler 自身影响

| 指标 | baseline | ECA |
| --- | --- | --- |
| profile 前中位延迟 | 4.829 ms | 5.578 ms |
| profile 后中位延迟 | 6.032 ms | 7.273 ms |
| 漂移 | **+24.9%** | **+30.4%** |

profiler 结束后两个模型都变慢约 25–30%，说明 profiler 会留下持续影响（可能是 kineto 缓冲、CUDA context 状态或 GPU 降频）。**因此 profiler 的墙钟数据不能作为延迟结论，只能用于算子计数与相对结构分析。**

### 4.8 瓶颈结论（按证据强度）

| 结论 | 证据 | 强度 |
| --- | --- | --- |
| ECA 每迭代新增 28 个 kernel | 3 轮完全一致（282→310） | 强 |
| ECA 自身 kernel 合计约 +87 us/it | 按 kernel 名聚合，3 轮稳定 | 强 |
| 无额外 memcpy / 显式同步 | runtime API 计数全为 0 | 强 |
| 存在视图/布局操作增加 CPU 调用 | `transpose` 16、`as_strided` 58 | 中 |
| kernel launch 数量本身是瓶颈 | 28 个 launch × 典型 3–5 us ≈ 84–140 us，与 +87 us 同量级，但未做单独消融 | **弱**（相关而非因果） |
| profiler 下的 +259 us/it 是 ECA 计算导致 | 与 kernel 明细矛盾 | 已被否定 |

未做「去掉 profiler 后单独计时 ECA 子模块」的消融，因此**不能断言 kernel launch 是唯一原因**。

## 5. 任务 5：效率指标（实测）

每个模型在**独立子进程**中测量，避免显存读数互相污染。

| 指标 | baseline | ECA | 差值 |
| --- | --- | --- | --- |
| 总参数量 | 4,214,842 | 4,214,871 | +29 |
| 可训练参数量 | 4,214,842 | 4,214,871 | +29 |
| 模型大小 | 16.1718 MiB | 16.1719 MiB | +0.0001 MiB |
| FLOPs（计数器） | 430,645,120 | 430,654,112 | +8,992 |
| MACs（计数器） | 215,322,560 | 215,327,056 | +4,496 |
| ECA 完整 MACs（解析） | — | 1,575,632 | +0.7318% |
| 推理峰值显存（bs=1） | 48.83 MiB | 48.84 MiB | +0.003 MiB |
| 推理激活增量（bs=1） | 32.58 MiB | 32.58 MiB | 0 |
| 训练峰值显存（bs=128+AMP，3 次） | 2912.3 / 2945.3 / 2945.3，均值 2934.3 | 3108.0 / 3141.5 / 3141.5，均值 3130.4 | +196.1 MiB（+6.7%） |

### 口径说明

| 指标 | 工具 | 范围 | 是否实测 |
| --- | --- | --- | --- |
| 参数量 | `sum(p.numel())` | 全部参数 | 实测 |
| 模型大小 | 参数+缓冲区字节数 | 不含 Python 对象开销 | 实测 |
| FLOPs/MACs | `torch.utils.flop_counter` | 仅卷积+全连接 | **理论计数** |
| ECA 完整 MACs | 前向钩子取形状 + 公式 | GAP+Conv1d+乘法 | **解析计算** |
| 推理显存 | `torch.cuda.max_memory_allocated` | bs=1 前向 | 实测 |
| 训练峰值显存 | 同上 | bs=128 + AMP，3 次 | 实测 |

**不可混用**：计数器 MACs +4,496（仅 Conv1d）与解析 MACs +1,575,632（含 GAP 与乘法）相差 350 倍，是两个不同口径。

### 延迟（非 profiler，交替配对）

| batch | baseline | ECA | 差值 |
| --- | --- | --- | --- |
| 1 | 5.046 ms | 5.781 ms | +0.735 ms（+14.6%） |
| 16 | 6.74 ms | 7.06 ms | +0.32 ms（+4.8%） |
| 128 | 81.00 ms | 85.81 ms | +4.81 ms（+5.9%） |

## 6. 验证（实际执行）

```text
.\.venv\Scripts\python.exe -m pytest tests/ -q
109 passed in 34.00s

.\.venv\Scripts\python.exe -m ruff check src tests scripts
All checks passed!

.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
38 files already formatted
```

短程 smoke（均 1 epoch、有限批次，**非实验结果**）：

| 配置 | 结果 | 模型身份 |
| --- | --- | --- |
| `stage3a_final_baseline` | train_loss 2.3113 / val_acc 0.1172 | `mobilenet_v3_large`，0 注意力块，4,214,842 |
| `stage3a_final_eca` | train_loss 2.4212 / val_acc 0.1172 | `mobilenet_v3_large_eca`，7 块，4,214,871 |

未启动 15 epoch 训练，未启动多种子实验，未重新评估正式测试集。

## 7. 新增产物

```text
outputs/stage3a/
  efficiency_full.json          完整效率指标（独立进程测量）
  baseline_efficiency.json      阶段 3A 早期测量（保留）
  eca_non_se_efficiency.json    阶段 3A 早期测量（保留）
  paired_latency.json           交替配对延迟
  batch_latency.json            bs=1/16/128
  baseline_checkpoint_identity.json
  eca_smoke_identity_recheck.json
  final_eca_identity.json
  smoke_eca_val_recheck.json
  profiler/
    baseline_profile.json / eca_non_se_profile.json
    comparison.json
    paired/comparison.json
    paired/analysis.json
    *_trace*.json（6 个，共约 310 MB，已被 .gitignore 排除）
outputs/stage3a_final_baseline/  冒烟
outputs/stage3a_final_eca/       冒烟
outputs/stage3a_audit_*/         审计用冒烟
outputs/stage3a_bad_name/        修复前残留（无权重）
```

## 8. 修改文件

新增：

```text
src/mobilenet_attention/models/registry.py
scripts/profile_models.py
scripts/profile_paired.py
scripts/analyze_profiler.py
scripts/efficiency_report.py
tests/test_model_registry.py
```

修改：

```text
scripts/train.py          # build_model、validate_model_config、输出模式、run_meta
scripts/evaluate.py       # build_model、身份输出
scripts/measure_efficiency.py  # build_model、identity 字段
src/mobilenet_attention/models/__init__.py  # 导出 registry
.gitignore                # 排除 profiler trace
```

其余文件为阶段 3A 早期已存在的改动（见第 1.2 节）。

## 9. 剩余风险

1. **ECA 延迟开销来源未完全归因**：kernel 数 +28 确定，ECA 算子合计 +87 us/it 确定，但非 profiler 延迟 +735 us（bs=1）远大于此。未做「仅 ECA 子模块」或「去掉 profiler」的隔离实验，不能断言 kernel launch 是唯一原因。
2. **profiler 污染**：profile 后延迟上升 25–30%，且 ECA 的 kernel 总耗时三轮波动 855–1187 us，说明 profiler 数据不适合做绝对耗时结论。
3. **训练峰值显存只测 3 次**，且 ECA 两次读数相同（3141.5），可能受分配器缓存影响；+196 MiB 应视为量级估计。
4. **`stage3a_bad_name/` 残留**：修复前的半成品目录，未删除（按规则不删历史产物），建议审核后手动清理。
5. **`split_indices.json` 每个 331 KB**，14 个实验目录合计约 4.6 MB；若后续实验增多，应考虑只保留哈希、不保留完整索引。
6. **`model.name` 与 `attention` 的双重来源**：ECA 变体允许省略 attention 用默认值，但配置里同时写 name 和 attention 时，两者必须一致（已加校验），仍需在文档中强调。
7. **未做正式多种子训练**，ECA 的精度收益仍未知。
8. **`git_dirty=true`**：所有冒烟快照都记录了脏工作区。

## 10. Git 状态

```text
分支 main，HEAD 9523cda，与 origin/main 同步
15 个已跟踪文件修改
29 项未跟踪（含 14 个实验目录）
未执行 git add / commit / push / tag / branch
git add -A --dry-run：134 个文件、约 4.59 MB，无 .pt、无 trace、无 data、无 .venv
```

## 11. 建议的下一步

1. 审核本报告与 `models/registry.py`、`scripts/train.py` 的输出模式逻辑。
2. 决定是否清理 `outputs/stage3a_bad_name/` 与早期 smoke 目录。
3. 提交阶段 3A 代码（需授权），使正式实验的 `git_commit` 对应干净工作区。
4. 若需论文级延迟结论，补做「profiler 外单独计时 ECA 子模块」的隔离实验。
5. 正式实验前确认 6 次运行矩阵与测试集使用策略。
