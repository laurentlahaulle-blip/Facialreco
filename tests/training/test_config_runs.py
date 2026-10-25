from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from maisonaccueil.contracts import Metric
from maisonaccueil.pipeline import ResourceBusyError, ResourceLock
from maisonaccueil.training.config import ExperimentConfig
from maisonaccueil.training.data import AlignedTrainingSet
from maisonaccueil.training.manifest import RAW_PREPROCESS, Manifest, ManifestError
from maisonaccueil.training.runs import RunDir, build_model_ref, device_guard
from maisonaccueil.training.synthetic import write_synthetic_dataset


def test_config_hash_and_roundtrip(tmp_path: Path) -> None:
    cfg = ExperimentConfig(name="exp1", seed=3, stage_channels=(8, 16), stage_blocks=(1, 1))
    assert cfg.config_hash == ExperimentConfig(
        name="exp1", seed=3, stage_channels=(8, 16), stage_blocks=(1, 1)
    ).config_hash
    assert cfg.config_hash != ExperimentConfig(name="exp1", seed=4).config_hash
    cfg.save(tmp_path / "c.json")
    assert ExperimentConfig.load(tmp_path / "c.json") == cfg


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "a b"},
        {"name": "x", "loss": "softmax"},
        {"name": "x", "stage_channels": (8,), "stage_blocks": (1, 1)},
        {"name": "x", "preprocess": "other-v1"},
        {"name": "x", "data_kind": "web"},
    ],
)
def test_config_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)  # type: ignore[arg-type]


def test_model_ref_origin_is_honest() -> None:
    sha = "ab" * 32
    ref = build_model_ref(ExperimentConfig(name="exp"), sha)
    assert ref.origin == "scratch"
    assert ref.name == "maisonaccueil-scratch"
    assert ref.version == f"exp-{sha[:12]}"
    assert ref.metric is Metric.COSINE and ref.preprocess == "align112-arcface5-v1"
    tuned = build_model_ref(ExperimentConfig(name="exp", init_weights="third_party.pt"), sha)
    assert tuned.origin == "finetuned"
    smoke = build_model_ref(ExperimentConfig(name="exp", data_kind="synthetic"), sha)
    assert smoke.name.endswith("-smoke") and smoke.key != ref.key


def test_gpu_lock_is_reserved_for_cuda(tmp_path: Path) -> None:
    lock_dir = tmp_path / "locks"
    cfg = ExperimentConfig(name="exp", device="cuda")
    with device_guard(cfg, lock_dir):
        assert (lock_dir / "gpu-0.lock").exists()
        with pytest.raises(ResourceBusyError), device_guard(cfg, lock_dir):
            pass
    assert not (lock_dir / "gpu-0.lock").exists()


def test_busy_gpu_blocks_training(tmp_path: Path) -> None:
    lock_dir = tmp_path / "locks"
    with ResourceLock("gpu-0", owner="autre-session", lock_dir=lock_dir):
        with pytest.raises(ResourceBusyError), device_guard(
            ExperimentConfig(name="exp", device="cuda:0"), lock_dir
        ):
            pass


def test_cpu_does_not_reserve_gpu(tmp_path: Path) -> None:
    lock_dir = tmp_path / "locks"
    with device_guard(ExperimentConfig(name="exp"), lock_dir):
        assert not (lock_dir / "gpu-0.lock").exists()


def test_run_dir_layout(tmp_path: Path) -> None:
    cfg = ExperimentConfig(name="exp")
    run = RunDir.create(cfg, tmp_path / "runs")
    other = RunDir.create(cfg, tmp_path / "runs")
    assert run.path != other.path
    assert (run.path / "config.json").exists()
    run.append_metrics({"step": 1, "loss": 2.0})
    run.append_metrics({"step": 2, "loss": 1.5})
    assert [m["step"] for m in run.read_metrics()] == [1, 2]
    ref = build_model_ref(cfg, "cd" * 32)
    run.write_model_ref(ref, "cd" * 32, cfg)
    assert run.read_model_ref() == ref


def test_synthetic_dataset_reader(tmp_path: Path) -> None:
    manifest = write_synthetic_dataset(
        tmp_path / "d", 3, 2, 2, seed=0, split_of_session={1: "test"}
    )
    ds = AlignedTrainingSet.from_manifest(manifest, tmp_path / "d")
    assert len(ds) == 6 and ds.num_classes == 3
    x, y = ds[0]
    assert x.shape == (3, 112, 112) and x.dtype == np.float32 and y == 0
    assert all(p.split("/")[1] == "s0" for p in ds.paths)  # séance 1 = test, exclue


def test_raw_manifest_is_refused_for_training(tmp_path: Path) -> None:
    m = write_synthetic_dataset(tmp_path / "d", 2, 1, 1, seed=0)
    raw = Manifest(RAW_PREPROCESS, m.sessions, m.samples)
    with pytest.raises(ManifestError):
        AlignedTrainingSet.from_manifest(raw, tmp_path / "d")


def test_package_imports_without_torch() -> None:
    code = (
        "import sys, maisonaccueil.training as t, maisonaccueil.training.__main__;"
        "assert 'torch' not in sys.modules, 'torch importé au chargement'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
