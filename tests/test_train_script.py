"""训练脚本辅助函数测试（不启动正式训练）。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train.py"


@pytest.fixture(scope="module")
def train_script():
    spec = importlib.util.spec_from_file_location("train_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_overrides(train_script):
    result = train_script.parse_overrides(["training.epochs=3", "model.pretrained=none"])
    assert result == {"training.epochs": "3", "model.pretrained": "none"}


def test_parse_overrides_rejects_invalid(train_script):
    with pytest.raises(ValueError):
        train_script.parse_overrides(["training.epochs"])


def test_prepare_output_dir_refuses_existing(train_script, tmp_path: Path):
    target = tmp_path / "exp"
    target.mkdir()
    (target / "best.pt").write_text("old", encoding="utf-8")
    with pytest.raises(SystemExit):
        train_script.prepare_output_dir(tmp_path, "exp", overwrite=False)
    assert (target / "best.pt").read_text(encoding="utf-8") == "old"


def test_prepare_output_dir_allows_overwrite(train_script, tmp_path: Path):
    target = tmp_path / "exp"
    target.mkdir()
    (target / "best.pt").write_text("old", encoding="utf-8")
    result, mode = train_script.prepare_output_dir(tmp_path, "exp", overwrite=True)
    assert result == target
    assert mode == "overwrite"


def test_prepare_output_dir_new_experiment(train_script, tmp_path: Path):
    result, mode = train_script.prepare_output_dir(tmp_path, "brand_new", overwrite=False)
    assert result == tmp_path / "brand_new"
    assert result.is_dir()
    assert mode == "new"


def test_prepare_output_dir_empty_dir_is_not_overwrite(train_script, tmp_path: Path):
    target = tmp_path / "empty"
    target.mkdir()
    result, mode = train_script.prepare_output_dir(tmp_path, "empty", overwrite=False)
    assert result == target
    assert mode == "new"


def test_resume_does_not_clear_existing_dir(train_script, tmp_path: Path):
    target = tmp_path / "exp"
    target.mkdir()
    marker = target / "train_log.jsonl"
    marker.write_text("history\n", encoding="utf-8")
    result, mode = train_script.prepare_output_dir(
        tmp_path, "exp", overwrite=False, resume_path=target / "last.pt"
    )
    assert result == target
    assert mode == "resume"
    assert marker.read_text(encoding="utf-8") == "history\n"


def test_resume_rejects_missing_dir(train_script, tmp_path: Path):
    with pytest.raises(SystemExit):
        train_script.prepare_output_dir(
            tmp_path, "missing", overwrite=False, resume_path=tmp_path / "missing" / "last.pt"
        )


def test_resume_and_overwrite_are_mutually_exclusive(train_script, tmp_path: Path):
    target = tmp_path / "exp"
    target.mkdir()
    (target / "last.pt").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        train_script.prepare_output_dir(
            tmp_path, "exp", overwrite=True, resume_path=target / "last.pt"
        )


def test_resolve_data_config_reads_dataset_section(train_script):
    config = {"data": {"config": "configs/dataset_cifar10.yaml"}}
    dataset_config = train_script.resolve_data_config(config)
    assert dataset_config["name"] == "cifar10"
    assert Path(dataset_config["root"]).is_absolute()


def test_limited_loader_truncates(train_script):
    loader = [(i, i) for i in range(10)]
    limited = train_script._LimitedLoader(loader, 3)
    assert len(limited) == 3
    assert list(limited) == [(0, 0), (1, 1), (2, 2)]


def test_resolve_scheduler_config_defaults(train_script):
    result = train_script.resolve_scheduler_config({"warmup_epochs": 1})
    assert result == {
        "name": "cosine",
        "warmup_epochs": 1,
        "warmup_start_factor": 0.1,
        "eta_min": 1e-6,
    }


def test_resolve_scheduler_config_explicit(train_script):
    result = train_script.resolve_scheduler_config(
        {
            "warmup_epochs": 2,
            "scheduler": {"name": "cosine", "warmup_start_factor": 0.05, "eta_min": 1e-5},
        }
    )
    assert result["warmup_epochs"] == 2
    assert result["warmup_start_factor"] == 0.05
    assert result["eta_min"] == 1e-5


def test_resolve_scheduler_rejects_unknown(train_script):
    with pytest.raises(ValueError):
        train_script.resolve_scheduler_config({"scheduler": {"name": "step"}})


def test_validate_evaluation_config_accepts_defaults(train_script):
    monitor, metrics = train_script.validate_evaluation_config({})
    assert monitor == "val_accuracy"
    assert "accuracy" in metrics


def test_validate_evaluation_config_rejects_unknown_monitor(train_script):
    with pytest.raises(ValueError):
        train_script.validate_evaluation_config({"monitor": "test_accuracy"})


def test_validate_evaluation_config_rejects_unknown_metric(train_script):
    with pytest.raises(ValueError):
        train_script.validate_evaluation_config({"report_metrics": ["auc"]})


def test_is_improvement_direction(train_script):
    assert train_script.is_improvement(0.9, 0.8, "val_accuracy")
    assert not train_script.is_improvement(0.7, 0.8, "val_accuracy")
    assert train_script.is_improvement(0.5, 0.6, "val_loss")
    assert not train_script.is_improvement(0.7, 0.6, "val_loss")


def test_save_split_indices_writes_hashes(train_script, tmp_path: Path):
    datasets = {
        "train_indices": [0, 2, 4],
        "val_indices": [1, 3],
    }
    path = tmp_path / "split_indices.json"
    info = train_script.save_split_indices(datasets, path)
    assert info["train_size"] == 3
    assert info["val_size"] == 2
    assert len(info["train_indices_sha256"]) == 64
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["train_indices"] == [0, 2, 4]
    assert payload["val_indices_sha256"] == info["val_indices_sha256"]


def test_select_metrics_filters(train_script):
    metrics = {"accuracy": 0.9, "f1_macro": 0.88, "confusion_matrix": [[1]]}
    selected = train_script.select_metrics(metrics, ["accuracy", "f1_macro"])
    assert selected == {"accuracy": 0.9, "f1_macro": 0.88}


def test_write_run_meta(train_script, tmp_path: Path):
    path = tmp_path / "run_meta.json"
    train_script.write_run_meta(
        path,
        command="python train.py",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:10:00+00:00",
        duration_s=600.0,
        exit_code=0,
        config_path=PROJECT_ROOT / "configs" / "smoke.yaml",
        output_mode="new",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["exit_code"] == 0
    assert payload["duration_s"] == 600.0
    assert payload["command"] == "python train.py"
    assert payload["output_mode"] == "new"
    assert payload["resume_from"] is None
