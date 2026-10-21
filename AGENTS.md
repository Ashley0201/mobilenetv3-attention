<!-- magpie:begin · written by magpie from its Library -->
# AGENTS.md — MobileNetV3 Attention Graduation Project

## A. Project identity

Project title:
基于注意力机制改进的 MobileNetV3 轻量化图像分类算法研究与智能识别系统设计

Environment:
- OS: Windows native
- Language: Python
- Deep learning: PyTorch + Torchvision
- Interface: Gradio
- Testing: pytest
- Code quality: Ruff
- Version control: Git
- User preference: concise, readable code with necessary Chinese comments
- Work mode: high autonomy, evidence-based execution

Primary objectives:
1. Implement a reproducible MobileNetV3 baseline.
2. Implement and evaluate a justified attention-based improvement.
3. Run fair comparison and ablation experiments.
4. Build a working image classification application.
5. Produce a thesis supported by actual code and experimental evidence.

Priority:
Correctness > reproducibility > scientific validity > maintainability > performance > visual polish.

## B. Autonomous execution policy

Operate autonomously for routine, reversible development tasks.

You may:
- Inspect the repository and environment.
- Create or edit project files.
- Install dependencies inside the project virtual environment when needed.
- Run tests, linters and diagnostic commands.
- Run small smoke tests.
- Fix ordinary implementation errors and rerun tests.
- Update documentation and experiment records.
- Propose and execute low-risk refactoring when necessary.

Before destructive or externally consequential actions, ask for approval.

Never:
- Delete or overwrite existing experiment evidence without authorization.
- Silently change a dataset split or the evaluation protocol.
- Replace the baseline implementation to manufacture a performance improvement.
- Commit secrets, credentials, private data or large datasets.
- Claim a command succeeded unless its actual result was observed.
- Fabricate logs, metrics, citations, papers or screenshots.

When execution is unavailable, provide the exact command and clearly mark it as not executed.

## C. Initial environment inspection

Before installing or modifying dependencies:
1. Inspect the current repository and existing Git status.
2. Check Python and pip versions.
3. Run nvidia-smi if available.
4. Inspect GPU name, VRAM and driver information when available.
5. Check PyTorch, Torchvision and CUDA availability.
6. Determine whether the current environment is a clean project virtual environment.
7. Save a concise environment report in docs/environment_report.md.

Use Windows-compatible commands by default. Do not assume PowerShell and CMD have identical syntax.

Do not install a different PyTorch build until the current environment has been inspected. Use the official PyTorch installation selector to select the compatible package build.

If CUDA is unavailable, diagnose the reason. Do not silently describe CPU execution as GPU execution.

## D. Project implementation rules

Use Python 3.11 as the preferred starting point, subject to verified package compatibility.

Core stack:
- torch
- torchvision
- numpy
- Pillow
- scikit-learn
- matplotlib
- pandas
- PyYAML
- tqdm
- gradio
- pytest
- ruff

Use a project-local .venv.

Keep dependencies minimal. Add new packages only when justified and record the reason.

Use pathlib for file paths. Avoid hardcoded absolute paths.

Use YAML configuration for dataset locations, model variants, attention settings, training hyperparameters, random seeds and output directories.

Separate:
- Model definition
- Dataset loading and transforms
- Training and validation
- Evaluation and metrics
- Inference
- Web application
- Testing and benchmarking

Keep code concise. Add Chinese comments for non-obvious algorithms, tensor dimensions and important implementation decisions. Do not comment every trivial line.

## E. Model and algorithm requirements

Implement a standard Torchvision MobileNetV3 baseline first.

Default candidate:
- MobileNetV3-Large
- ImageNet pretrained weights, when download and loading are available
- Replace the classification head for the selected dataset

Consider MobileNetV3-Small only when resource constraints or the research objective justify it.

Keep the baseline unchanged and reproducible.

Investigate attention mechanisms such as SE, ECA and CBAM. Choose the primary mechanism based on original literature, implementation complexity, computational cost and research motivation.

Implement attention as a separate module. Make the insertion position configurable where practical.

For each modification, explain:
1. Research motivation
2. Mathematical operation
3. Insertion position
4. Input and output shapes
5. Parameter and computation overhead
6. Expected benefits and potential drawbacks

Model contract:
- Input: [B, C, H, W]
- Output: [B, num_classes]
- Output raw logits, not probabilities
- Apply Softmax only in the inference or presentation layer when probabilities are needed

Do not claim novelty merely because an existing module was inserted into a network. Describe the contribution accurately and support it with experiments.

## F. Dataset policy

Do not commit to a dataset before checking accessibility, license, class definitions, size and split availability.

Evaluate candidate datasets using:
- Reproducible download
- Clear usage terms
- Suitable classification difficulty
- Enough samples for training and validation
- Manageable image size and storage requirements
- Suitability for the demonstration application

Prefer an official dataset source and fixed split when available.

Keep training, validation and test data separate. Do not tune hyperparameters on the test set.

Record dataset source, version or download date, split strategy, sample counts and class mapping.

Do not silently download very large files or use an unverified third-party mirror.

## G. Training and evaluation

Training must support:
- Reproducible random seeds
- CPU/GPU device selection
- Training and validation loops
- Learning-rate scheduling where appropriate
- Best-checkpoint saving
- Resume from checkpoint
- Structured logging
- Independent test evaluation

Every experiment must preserve:
- Unique experiment ID
- Configuration snapshot
- Random seed
- Software and hardware details
- Training log
- Best checkpoint
- Evaluation metrics
- Benchmark settings

Report at minimum:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Parameter count
- Model file size
- Inference latency
- FLOPs/MACs when a suitable, documented measurement method is available

Use the same dataset split, preprocessing and evaluation protocol for fair model comparison.

Repeat important experiments with multiple seeds when time permits. Report mean and standard deviation. Explicitly disclose any limitations.

Measure inference latency with warm-up, synchronization where required, fixed input size and documented batch size. Separate model-only latency from end-to-end application latency.

## H. Testing and code quality

After changing a module, run the relevant tests.

Required test coverage:
- Model forward shape and finite outputs
- Attention module shape preservation and gradient flow
- Dataset and label mapping
- Training loss and backward propagation
- Checkpoint loading
- Inference output and invalid input handling

Run:
python -m pytest tests/ -q
ruff check src tests scripts
ruff format --check src tests scripts

Do not suppress failing tests without investigating the reason.

A test passing proves only the behavior it checks; it does not prove classification accuracy or research validity.

## I. Windows command policy

Prefer commands that work in PowerShell.

Use:
python -m pip ...
python -m pytest ...
python -m ...
rather than relying on shell-specific aliases.

Activate the virtual environment using:
.\.venv\Scripts\Activate.ps1

If PowerShell activation is blocked, use the environment's interpreter directly:
.\.venv\Scripts\python.exe -m pytest tests/ -q

Never assume the working directory. Confirm the project root before running commands that create or modify files.

## J. Application requirements

Build a Gradio image classification interface.

Required features:
- Image upload and preview
- Predicted class
- Top-K predictions and scores
- Loading a trained checkpoint
- Consistent preprocessing
- Clear errors for invalid images or checkpoints

Use the actual dataset class mapping.

Do not manipulate predictions in the UI to improve apparent accuracy. Explain that model confidence is not necessarily calibrated probability.

## K. Thesis and references

The thesis must match the implemented model and real experiments.

Every reported number must trace back to a result file or log.

Verify references using original papers, publisher pages, official repositories or other reliable sources.

Record paper title, authors, publication venue, year, DOI when available, official link and the specific claim supported.

Do not invent citations, publication details, metrics, experiments or implementation status.

Clearly distinguish:
- Established methods from literature
- Modifications implemented in this project
- Hypotheses
- Observed results
- Limitations

Follow the institution's academic integrity and AI assistance policies.

## L. Change management and evidence

Before significant changes:
1. Inspect affected files and tests.
2. Identify risks and expected behavior.
3. Preserve the existing baseline and experiment outputs.

After changes:
1. List modified files and reasons.
2. Run relevant tests.
3. Record actual commands and outcomes.
4. Report remaining failures and limitations.
5. Update documentation if interfaces or workflows changed.

Do not overwrite a checkpoint or result file from a different experiment. Use unique output directories and experiment identifiers.

Use Git checkpoints before major algorithm modifications when Git is available.

## M. Stage gates

Stage 1: Environment and project skeleton.
Stage 2: Dataset loader and baseline.
Stage 3: Attention module and improved model.
Stage 4: Comparison, ablation and error analysis.
Stage 5: Gradio application and inference testing.
Stage 6: Thesis, reproducibility audit and final acceptance.

Do not advance past a failed critical stage gate without reporting the failure and resolving it or documenting an explicit workaround.

At the end of each stage, report:
- Completed deliverables
- Files changed
- Commands executed
- Actual test results
- Evidence locations
- Remaining risks
- Recommended next task

The final deliverable is a runnable, testable, reproducible research project, not merely a collection of generated source files.
<!-- magpie:end -->
