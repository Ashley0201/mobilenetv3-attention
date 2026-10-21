# 阶段 3A 提交前 Git 审核（最终版）

- 审核日期：2026-10-21
- 基线提交：`9523cda75724211912c43099c3f19b48fa14d3af`
- 审核方式：只读检查（`git status`、`git diff`、`git check-ignore`、`Get-FileHash`、文件读取）
- 本文件为阶段 3A 审核**新增**文件，随本次提交纳入版本库
- 归档操作：未执行正式训练，未重新评估测试集，未 amend/rebase，未强制推送

## 1. 实际数量

| 项目 | 数量 | 说明 |
| --- | --- | --- |
| 已跟踪文件修改 | **16** | 含 `.gitignore` |
| 未跟踪文件（应用新忽略规则后） | **29** | 见第 3 节 |
| 候选提交文件 | **44** | 16 修改 + 28 新增 |
| 候选提交总体积 | **301,387 B（0.287 MB）** | 逐文件求和（含本文件） |
| 被忽略的模型权重 | 28 个、1,359.4 MB | `*.pt` / `outputs/**/*.pt` |
| 被忽略的 profiler trace | 8 个、414.6 MB | `outputs/**/*_trace*.json` |
| 被忽略的 smoke/审计运行目录 | 13 个 | 第 5 节 |

## 2. 已跟踪文件修改（16 个）

| 文件 | 变更 | 分类 | 建议 |
| --- | --- | --- | --- |
| `.gitignore` | +17 | 配置 | 提交 |
| `configs/baseline_mobilenetv3_large.yaml` | +5 | 配置 | 提交（显式 scheduler，行为与阶段 2 一致） |
| `configs/smoke.yaml` | +4 | 配置 | 提交 |
| `scripts/evaluate.py` | +11/-2 | 源码 | 提交 |
| `scripts/train.py` | +271/-30 | 源码 | 提交 |
| `src/mobilenet_attention/data/__init__.py` | +2 | 源码 | 提交 |
| `src/mobilenet_attention/data/cifar10.py` | +32 | 源码 | 提交 |
| `src/mobilenet_attention/engine/trainer.py` | +13/-1 | 源码 | 提交 |
| `src/mobilenet_attention/models/__init__.py` | +16/-5 | 源码 | 提交 |
| `src/mobilenet_attention/models/baseline.py` | +30/-3 | 源码 | 提交 |
| `src/mobilenet_attention/utils/config.py` | +24/-2 | 源码 | 提交 |
| `src/mobilenet_attention/utils/env.py` | +42/-9 | 源码 | 提交 |
| `tests/test_config.py` | +58/-1 | 测试 | 提交 |
| `tests/test_data.py` | +28 | 测试 | 提交 |
| `tests/test_engine.py` | +56 | 测试 | 提交 |
| `tests/test_train_script.py` | +141/-1 | 测试 | 提交 |

`git diff --stat`：16 files changed, 686 insertions(+), 49 deletions(-)。
`git diff --check`：exit 0，无空白错误。

## 3. 未跟踪文件（29 个，应用新忽略规则后）

### 3.1 源码（3）

| 文件 | 字节 | SHA-256 前 16 位 |
| --- | --- | --- |
| `src/mobilenet_attention/models/eca.py` | 2,369 | `6288dd8ceebd81b3` |
| `src/mobilenet_attention/models/attention.py` | 3,427 | `7f39f454ca5a9cac` |
| `src/mobilenet_attention/models/registry.py` | 3,903 | `504c1ea20118d87e` |

### 3.2 测试（2）

| 文件 | 字节 | SHA-256 前 16 位 |
| --- | --- | --- |
| `tests/test_eca.py` | 8,898 | `2f2a8a9455f7e19c` |
| `tests/test_model_registry.py` | 4,553 | `11593f2817d94079` |

### 3.3 配置（2）

| 文件 | 字节 | SHA-256 前 16 位 |
| --- | --- | --- |
| `configs/eca_non_se.yaml` | 1,237 | `c84bb5aa8e58e750` |
| `configs/smoke_eca.yaml` | 1,022 | `00a992935e7fb699` |

### 3.4 文档（3）

| 文件 | 字节 | SHA-256 前 16 位 |
| --- | --- | --- |
| `docs/stage3a_report.md` | 15,024 | `7a0aca2674acac26` |
| `docs/stage3a_audit_report.md` | 14,758 | `ce30087baf71b294` |
| `docs/stage3a_git_review.md` | 本文件 | 本次新增 |

### 3.5 脚本（7）

| 文件 | 字节 |
| --- | --- |
| `scripts/measure_efficiency.py` | 10,620 |
| `scripts/paired_latency.py` | 4,335 |
| `scripts/batch_latency.py` | 2,868 |
| `scripts/profile_models.py` | 13,671 |
| `scripts/profile_paired.py` | 8,093 |
| `scripts/analyze_profiler.py` | 6,757 |
| `scripts/efficiency_report.py` | 10,902 |

### 3.6 正式实验汇总指标（11 个 JSON，102,216 B）

| 文件 | 字节 | 来源与口径 |
| --- | --- | --- |
| `outputs/stage3a/baseline_efficiency.json` | 1,440 | `measure_efficiency.py --model baseline`，实测参数/FLOPs/延迟/显存 |
| `outputs/stage3a/eca_non_se_efficiency.json` | 4,853 | 同上，ECA 变体；含解析 ECA MACs |
| `outputs/stage3a/efficiency_full.json` | 6,718 | `efficiency_report.py`，每模型独立子进程，最终口径 |
| `outputs/stage3a/paired_latency.json` | 1,563 | 交替配对延迟，6 轮 × 100 次 |
| `outputs/stage3a/batch_latency.json` | 1,774 | bs=1/16/128 对照 |
| `outputs/stage3a/baseline_checkpoint_identity.json` | 1,989 | 阶段 2 基线 `best.pt` 的验证集复算，用于身份核对 |
| `outputs/stage3a/profiler/baseline_profile.json` | 20,232 | 单轮 profiler 汇总（kernel 数、算子耗时） |
| `outputs/stage3a/profiler/eca_non_se_profile.json` | 21,117 | 同上，ECA 变体 |
| `outputs/stage3a/profiler/comparison.json` | 10,231 | 单轮 profiler 对照 + profiler 影响 |
| `outputs/stage3a/profiler/paired/analysis.json` | 21,326 | 3 轮交替 trace 的 kernel/runtime 统计 |
| `outputs/stage3a/profiler/paired/comparison.json` | 10,973 | 3 轮交替 profiler 对照 |

以上均不依赖未提交权重即可解释；profiler 汇总已标注「不用于延迟结论」。

### 3.7 保留本地、不提交（3 个 JSON，来自 smoke 检查点）

| 文件 | 字节 | 不提交理由 |
| --- | --- | --- |
| `outputs/stage3a/eca_smoke_identity_recheck.json` | 1,957 | 检查点来自 `stage3a_audit_eca_smoke`（1 epoch、有限批次） |
| `outputs/stage3a/final_eca_identity.json` | 1,950 | 检查点来自 `stage3a_final_eca`（1 epoch smoke） |
| `outputs/stage3a/smoke_eca_val_recheck.json` | 1,814 | 检查点来自 `smoke_eca`（1 epoch smoke） |

已通过 `.gitignore` 显式排除，本地文件保留。

## 4. 重复 `split_indices.json` 的 SHA-256（重新计算）

本次重新计算全部 13 份，结果与旧报告一致：

| SHA-256 | 大小 | 数量 |
| --- | --- | --- |
| `671eb97eca1c3378c86974b8420369e4481871e0afa0e8b70d5710a2e28110ee` | 339,147 B | 13 |

出现位置（全部位于阶段 3A smoke/审计目录）：

```text
outputs/smoke_amp_off/
outputs/smoke_baseline_stage3a/
outputs/smoke_eca/
outputs/smoke_eca_monitor_loss/
outputs/smoke_eca_override/
outputs/smoke_eca_resume/
outputs/smoke_eca_resume3/
outputs/smoke_eca_v2/
outputs/stage3a_audit_baseline_smoke/
outputs/stage3a_audit_eca_smoke/
outputs/stage3a_bad_name/
outputs/stage3a_final_baseline/
outputs/stage3a_final_eca/
```

结论：13 份完全相同，无不同划分；本地文件全部保留，未删除。
划分规则记录在此，不重复提交索引文件：

```text
训练/验证来自官方 CIFAR-10 train（50,000）分层抽取：
  train 45,000，val 5,000，stratify=targets，random_state=42
官方 test 10,000 仅用于最终评估
索引哈希：
  train 439d2886be7d934cfcf5add00aa6e368bb431dd849d6a130530d1dc7cee35480
  val   6781efc5c7c0ad15f37adeac39dfdc1ba7cb6ad7a65e07d775bd3aec67050a12
```

## 5. `.gitignore` 调整与验证

### 5.1 本次新增规则（最小改动）

```text
# 阶段 3A smoke/审计运行目录：保留本地，不入库
outputs/smoke_amp_off/
outputs/smoke_baseline_stage3a/
outputs/smoke_eca/
outputs/smoke_eca_monitor_loss/
outputs/smoke_eca_override/
outputs/smoke_eca_resume/
outputs/smoke_eca_resume3/
outputs/smoke_eca_v2/
outputs/stage3a_audit_baseline_smoke/
outputs/stage3a_audit_eca_smoke/
outputs/stage3a_bad_name/
outputs/stage3a_final_baseline/
outputs/stage3a_final_eca/

# 阶段 3A 中由 smoke 检查点产生的复算结果：非正式实验结果，保留本地
outputs/stage3a/eca_smoke_identity_recheck.json
outputs/stage3a/final_eca_identity.json
outputs/stage3a/smoke_eca_val_recheck.json
```

未删除、未修改既有规则。

### 5.2 实测验证

| 路径 | `git check-ignore -v` 结果 |
| --- | --- |
| `outputs/smoke_eca/best.pt` | 命中 `outputs/smoke_eca/` |
| `outputs/smoke_eca/split_indices.json` | 命中 `outputs/smoke_eca/` |
| `outputs/stage3a_final_eca/best.pt` | 命中 `outputs/stage3a_final_eca/` |
| `outputs/stage3a/eca_smoke_identity_recheck.json` | 命中显式文件规则 |
| `outputs/stage3a/final_eca_identity.json` | 命中显式文件规则 |
| `outputs/stage3a/smoke_eca_val_recheck.json` | 命中显式文件规则 |
| `outputs/stage3a/efficiency_full.json` | **未忽略**（`!outputs/**/*.json`） |
| `outputs/stage3a/profiler/paired/analysis.json` | **未忽略** |
| `outputs/stage3a/profiler/baseline_trace.json` | 命中 `outputs/**/*_trace*.json` |

覆盖检查：

| 类别 | 规则 | 状态 |
| --- | --- | --- |
| 虚拟环境 | `.venv/` | 已覆盖 |
| 数据集 | `/data/` | 已覆盖 |
| 模型权重 | `*.pt`、`*.pth`、`*.onnx`、`outputs/**/*.pt` | 已覆盖 |
| 日志/临时 | `*.log`、`*.tmp`、`/logs/` | 已覆盖 |
| profiler trace | `outputs/**/*_trace*.json` | 已覆盖 |
| 轻量实验报告 | `!outputs/**/*.json`、`*.jsonl`、`*.yaml`、`*.md` | 保留可提交 |

## 6. 排除清单（实测数量与体积）

| 类型 | 数量 | 体积 | 处理 |
| --- | --- | --- | --- |
| `*.pt` 模型权重 | 28 | 1,359.4 MB | 忽略，本地保留 |
| profiler trace | 8 | 414.6 MB | 忽略，本地保留 |
| smoke/审计运行目录 | 13 | 约 1,173 MB（含权重） | 忽略，本地保留 |
| `split_indices.json` | 13 | 4.4 MB | 随 smoke 目录一并忽略 |
| `outputs/logs/*.log` | 2 | 0.33 MB | 忽略 |
| `.venv`、缓存、`__pycache__`、`.egg-info` | — | — | 忽略 |

注：阶段 2 的 `outputs/smoke_baseline/` 是已跟踪目录，本次未修改、未重复提交。

## 7. `outputs/stage3a_bad_name/` 判定

5 个文件、0.33 MB，已随新规则忽略，本地保留。

| 文件 | 字节 | SHA-256 |
| --- | --- | --- |
| `config_snapshot.yaml` | 2,593 | `a9e45d141642c1ac5419e96830ba53d560ea236c6734bf881a06d96f5ac308e8` |
| `environment.json` | 699 | `f7971ca46d76f45cc73bae6eca55e9777326fcab39d75d5d292b7c32c6e0d943` |
| `run_meta.json` | 477 | `c392b717aa6817f734fdd7ba144ee0a8f05e5dfc85b53d22260a2e2a5e257c40` |
| `split_indices.json` | 339,147 | `671eb97eca1c3378c86974b8420369e4481871e0afa0e8b70d5710a2e28110ee` |
| `train_log.jsonl` | 1,786 | `fa341d4df6b15858fcd07fb94d3d621f3877e5aa641f98cb7565ca7b57afb72e` |

- 测试目的：验证未知 `model.name` 不会静默退回默认模型
- 实际结果：修复前版本创建目录并写入 5 个文件，`train_log.jsonl` 只有 `start → dataset → limit → limit`，无 `model` 事件；`run_meta.exit_code = null`
- 修复后：`validate_model_config()` 在创建目录前校验，`stage3a_bad_name3/`、`stage3a_bad_name4/` 均未创建
- 覆盖位置：`tests/test_model_registry.py::test_unknown_name_raises`、`test_empty_name_raises`
- 结论：**非正式实验，不提交**。错误行为已由单测固化，无需保留 331 KB 索引

## 8. 预期暂存结果

| 项目 | 数值 |
| --- | --- |
| 暂存文件数 | **44** |
| 总体积 | **301,387 B（0.287 MB）** |
| 扩展名分布 | `.py` 25、`.json` 11、`.yaml` 4、`.md` 3、`.gitignore` 1 |
| 权重 / trace / 数据集 / 缓存 / 密钥 | 0 |

## 9. 风险与说明

1. **profiler 汇总的 `trace_file` 指向本地路径**：`baseline_profile.json` 与 `eca_non_se_profile.json` 内嵌 `outputs/stage3a/profiler/*_trace.json`，该文件未入库。在其他机器上路径无效；需要完整复核时应重新运行 `scripts/profile_models.py` / `scripts/profile_paired.py` 生成 trace。
2. **延迟数据跨机器不可比**：所有延迟测量绑定 RTX 4060 Laptop + 当前驱动，且记录到 GPU 时钟。
3. **ECA 精度收益未知**：阶段 3A 只做流程验证与开销测量，未跑正式训练。
4. **工作区将保持脏状态**：13 个 smoke 目录、8 个 trace、28 个权重仍留在本地但被忽略；`git status` 在提交后应显示干净（被忽略项不显示）。
5. **`split_indices.json` 不再入库**：复现依赖 `dataset_cifar10.yaml` 中的划分参数（seed=42、val_size=5000）与第 4 节记录的哈希；训练脚本每次运行都会重新生成并可核对。

## 10. 建议的暂存与提交命令（本次执行）

```powershell
git add .gitignore configs docs scripts src tests
git add outputs/stage3a/baseline_efficiency.json
git add outputs/stage3a/eca_non_se_efficiency.json
git add outputs/stage3a/efficiency_full.json
git add outputs/stage3a/paired_latency.json
git add outputs/stage3a/batch_latency.json
git add outputs/stage3a/baseline_checkpoint_identity.json
git add outputs/stage3a/profiler/baseline_profile.json
git add outputs/stage3a/profiler/eca_non_se_profile.json
git add outputs/stage3a/profiler/comparison.json
git add outputs/stage3a/profiler/paired/analysis.json
git add outputs/stage3a/profiler/paired/comparison.json

git status --short
git diff --cached --name-status
git diff --cached --stat
git diff --cached --check
git commit -m "feat: add ECA attention and experiment auditing"
```
