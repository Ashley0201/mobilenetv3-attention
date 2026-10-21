# 环境审计报告（阶段 1）

- 审计日期：2026-10-20
- 执行方式：本地 Windows 原生环境，项目内 `.venv`
- 结论：环境可用，CUDA 可用，阶段 1 验收通过；尚未开始任何训练。

## 1. 系统与硬件（实测）

| 项目 | 实测值 | 来源 |
| --- | --- | --- |
| 工作目录 | `C:\Users\Sylvia\Desktop\mobilenetv3-attention` | `Get-Location` |
| 操作系统 | Windows 11 家庭中文版，版本 10.0.26200，64 位 | `Get-CimInstance Win32_OperatingSystem` |
| CPU | 13th Gen Intel Core i9-13900HX，24 核 / 32 线程 | `Get-CimInstance Win32_Processor` |
| 内存 | 约 31.7 GiB（34,065,997,824 字节） | `Get-CimInstance Win32_ComputerSystem` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU | `nvidia-smi`、`torch.cuda` |
| 显存 | 8188 MiB 总计；审计时已用约 1741 MiB（桌面应用占用） | `nvidia-smi` |
| 显卡驱动 | 581.80（驱动报告 CUDA 13.0） | `nvidia-smi` |
| 计算能力 | 8.9 | `nvidia-smi`、`torch.cuda.get_device_capability` |
| 磁盘 | C: 剩余 53.9 GiB / 总计 475.7 GiB | `shutil.disk_usage` |

说明：`Win32_VideoController.AdapterRAM` 对 RTX 4060 Laptop 报告约 4 GiB，与真实显存不符（WMI 32 位字段限制），因此本报告显存以 `nvidia-smi` 与 PyTorch 的 8188 MiB 为准。

## 2. 软件环境

| 项目 | 版本 |
| --- | --- |
| Python（venv 内） | 3.12.3 |
| 系统 Python | `C:\Python312\python.exe`（3.12.3）；另有 3.14 与 Anaconda，未使用 |
| pip | 26.2.1（venv 内） |
| Git | 2.45.1.windows.1 |
| PyTorch | 2.14.1+cu130 |
| Torchvision | 0.29.1+cu130 |
| CUDA 运行时（PyTorch 编译） | 13.0 |
| cuDNN | 92400 |
| NumPy | 2.5.2 |
| Pillow | 12.3.0 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| pandas | 3.0.6 |
| PyYAML | 6.0.3 |
| tqdm | 4.70.1 |
| Gradio | 6.30.0 |
| pytest | 9.1.1 |
| Ruff | 0.17.1 |

PyTorch 安装来源：官方索引 `https://download.pytorch.org/whl/cu130`，wheel 为 `torch-2.14.1+cu130-cp312-cp312-win_amd64`。选择 cu130 是因为本机驱动 581.80 报告 CUDA 13.0，且该索引提供与 Python 3.12 匹配的 Windows wheel。

## 3. 环境审计结果

审计前状态：

- 工作目录仅含空的 `AGENTS.md`，不是 Git 仓库。
- 系统 Python 3.12 未安装 torch、torchvision、gradio、pytest。
- 系统 site-packages 存在 `WARNING: Ignoring invalid distribution ~umpy`，属于全局环境污染，与本项目 `.venv` 无关。
- 无项目虚拟环境。

审计后状态：

- 已创建项目本地 `.venv`（Python 3.12.3），未改动其他 Python 环境。
- 已 `git init -b main`，尚未创建首次提交。
- 已安装全部核心与开发依赖，并以 editable 方式安装本项目包。
- `torch.cuda.is_available()` 为 `True`，检测到 1 块 GPU。

## 4. CUDA 功能验证（实测）

| 检查项 | 结果 |
| --- | --- |
| `torch.cuda.is_available()` | True |
| `torch.cuda.device_count()` | 1 |
| 设备名称 | NVIDIA GeForce RTX 4060 Laptop GPU |
| 显存（PyTorch 报告） | 8187 MiB |
| 计算能力 | 8.9 |
| cuDNN 版本 | 92400 |
| GPU 矩阵乘法冒烟测试 | 20 次 1024×1024 FP32 矩阵乘，总计 0.048 s，显存占用 12.1 MiB |

## 5. 模型可用性冒烟验证

```text
torchvision.models.mobilenet_v3_large(weights=None, num_classes=10)
输入 [2, 3, 224, 224] -> 输出 [2, 10]（原始 logits）
参数量 4,214,842（num_classes=10）
```

此结果仅证明 Torchvision 模型可构建、可前向，不代表任何分类精度。

## 6. 已执行命令与结果

```powershell
# 审计
Get-Location
Get-ChildItem -Force
git --version
git status
python --version
python -m pip --version
where.exe python
nvidia-smi
python -c "import torch, torchvision; ..."        # 审计时 ModuleNotFoundError: torch
python -m pip list

# 环境创建与依赖
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install scikit-learn matplotlib pandas PyYAML tqdm gradio pytest ruff
.\.venv\Scripts\python.exe -m pip install -e . --no-deps

# 验证
.\.venv\Scripts\python.exe scripts/check_env.py
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
```

测试结果（真实输出）：

```text
pytest:            9 passed in 8.11s
ruff check:        All checks passed!
ruff format check: 5 files already formatted
```

## 7. 已知限制与风险

1. 显存仅 8 GB，训练时需控制 batch size；MobileNetV3-Large 在 224×224 下可行，但更大分辨率需验证。
2. 审计时 GPU 已被桌面程序占用约 1.7 GB，实际可用显存约 6.4 GB。
3. C: 剩余约 54 GiB，数据集与实验输出建议先确认体积，必要时放到其他磁盘。
4. 系统全局 Python 存在损坏的 `~umpy` 残留，不影响 `.venv`，但后续命令必须显式使用 `.venv` 解释器。
5. 当前 Git 仓库无提交，分支名虽为 `main`，但尚无 commit，`check_env.py` 中显示 `N/A` 属预期。
6. `nvidia-smi` 与 PyTorch 对显存的报告一致（8188/8187 MiB），但 `Win32_VideoController` 不可信。

## 8. 进入阶段 2 前需确认的事项

1. 选择数据集（官方来源、许可证、固定划分），并确认下载体积与磁盘余量。
2. 决定是否使用 ImageNet 预训练权重；首次下载需联网并占用额外缓存空间。
3. 建立 Git 首次提交，作为后续算法改动的可回退基线。
4. 确认训练 batch size 与输入分辨率（受 8 GB 显存约束）。
5. 明确评测指标与推理延迟测量规范（warm-up、同步、batch size）。

## 9. 阶段 2 复验补充（2026-10-20）

阶段 2 开始时重新核实，结论与阶段 1 一致：

```text
torch          2.14.1+cu130
torchvision    0.29.1+cu130（METADATA: Requires-Dist: torch (>=2.14.0)）
numpy          2.5.2
CUDA compiled  13.0
cuda available True
GPU            NVIDIA GeForce RTX 4060 Laptop GPU, 8187 MiB, capability 8.9
cuDNN          92400
pip check      No broken requirements found.
sys.prefix     C:\Users\Sylvia\Desktop\mobilenetv3-attention\.venv
```

补充事实：

- PyTorch/Torchvision/NumPy 均从项目 `.venv\Lib\site-packages` 加载，未混用全局包。
- 磁盘：C: 剩余 54.1 GiB（阶段 2 开始时）；`data` 目录占用约 340 MiB。
- 实测训练资源：batch=128 + AMP 峰值显存 2,899 MiB，单轮 60–117 秒。
- 预训练权重：`mobilenet_v3_large-8738ca79.pth`（IMAGENET1K_V1），
  SHA256 `8738CA797C879B547D18BBD15DA5736FF2557B2036A9AF72225393CA61759A04`。
- Git 首次提交仍未创建，`environment.json` 中 `git_commit` 为 null。
