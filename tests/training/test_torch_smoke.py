"""Tests de fumée PyTorch sur données SYNTHÉTIQUES, CPU, quelques pas.

Ils vérifient que le code s'exécute (formes, perte finie, journal, empreintes,
déterminisme, export). Ils ne démontrent RIEN sur la qualité du modèle.
Ignorés proprement si torch (extra ``training``) n'est pas installé.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from maisonaccueil.training.config import ExperimentConfig
from maisonaccueil.training.data import AlignedTrainingSet
from maisonaccueil.training.synthetic import write_synthetic_dataset

from .conftest import landmarks_at, make_face, make_frame

torch = pytest.importorskip("torch", reason="PyTorch (extra 'training') non installé")

from maisonaccueil.training.embedder import ScratchFaceEmbedder  # noqa: E402
from maisonaccueil.training.net import EmbeddingNet, MarginHead  # noqa: E402
from maisonaccueil.training.trainer import TrainResult, load_embedding_net, train  # noqa: E402


def _config(**kw: object) -> ExperimentConfig:
    base: dict[str, object] = dict(
        name="smoke",
        data_kind="synthetic",
        stage_channels=(8, 8, 16, 16),
        stage_blocks=(1, 1, 1, 1),
        embedding_dim=16,
        batch_size=8,
        epochs=1,
        max_steps=3,
    )
    base.update(kw)
    return ExperimentConfig(**base)  # type: ignore[arg-type]


def _run(tmp_path: Path, name: str, **kw: object) -> TrainResult:
    manifest = write_synthetic_dataset(tmp_path / "data", 4, 2, 4, seed=0)
    ds = AlignedTrainingSet.from_manifest(manifest, tmp_path / "data")
    return train(_config(**kw), ds, runs_root=tmp_path / name, lock_dir=tmp_path / "locks")


def test_network_shapes() -> None:
    cfg = _config()
    net = EmbeddingNet(cfg).eval()
    out = net(torch.zeros(2, 3, 112, 112))
    assert tuple(out.shape) == (2, 16)


@pytest.mark.parametrize("loss", ["cosface", "arcface"])
def test_margin_head_loss_is_finite_and_differentiable(loss: str) -> None:
    torch.manual_seed(0)
    head = MarginHead(16, 4, loss, 30.0, 0.35)
    emb = torch.randn(6, 16, requires_grad=True)
    value = head(emb, torch.tensor([0, 1, 2, 3, 0, 1]))
    assert torch.isfinite(value)
    value.backward()
    assert emb.grad is not None and torch.isfinite(emb.grad).all()


def test_smoke_training_writes_run(tmp_path: Path) -> None:
    result = _run(tmp_path, "runs")
    assert result.steps == 3 and np.isfinite(result.last_loss)
    files = {p.name for p in result.run.path.iterdir()}
    assert {"config.json", "metrics.jsonl", "weights.pt", "model_ref.json", "summary.json"} <= files
    assert len(result.run.read_metrics()) == 3
    ref = result.run.read_model_ref()
    assert ref == result.model_ref and ref.origin == "scratch" and ref.name.endswith("-smoke")
    summary = json.loads((result.run.path / "summary.json").read_text("utf-8"))
    assert "SYNTHÉTIQUES" in summary["quality_note"]
    net, _, _ = load_embedding_net(result.run)  # vérifie l'empreinte des poids
    assert not (tmp_path / "locks" / "gpu-0.lock").exists()  # CPU : pas de réservation


def test_same_seed_same_weights(tmp_path: Path) -> None:
    a = _run(tmp_path, "a")
    b = _run(tmp_path, "b")
    assert a.weights_sha256 == b.weights_sha256
    c = _run(tmp_path, "c", seed=1)
    assert c.weights_sha256 != a.weights_sha256


def test_torch_backend_embedder(tmp_path: Path) -> None:
    result = _run(tmp_path, "runs")
    emb = ScratchFaceEmbedder.from_run(result.run.path, backend="torch")
    e = emb.embed(make_frame(), make_face(landmarks_at(2.0, 150, 80)))
    assert e is not None and e.model == result.model_ref
    assert abs(float(np.linalg.norm(e.vector)) - 1.0) < 1e-5


@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None or importlib.util.find_spec("onnxruntime") is None,
    reason="onnx/onnxruntime non installés (voir docs/contract-changes/E-onnx-export.md)",
)
def test_onnx_export_matches_torch(tmp_path: Path) -> None:
    from maisonaccueil.training.export import export_onnx

    result = _run(tmp_path, "runs")
    export_onnx(result.run)
    onnx_emb = ScratchFaceEmbedder.from_run(result.run.path, backend="onnx")
    torch_emb = ScratchFaceEmbedder.from_run(result.run.path, backend="torch")
    frame, face = make_frame(), make_face(landmarks_at(2.0, 150, 80))
    a, b = onnx_emb.embed(frame, face), torch_emb.embed(frame, face)
    assert a is not None and b is not None
    assert float(np.dot(a.vector, b.vector)) > 0.999


@pytest.mark.gpu
def test_cuda_training_reserves_gpu(tmp_path: Path) -> None:  # pragma: no cover - matériel
    if not torch.cuda.is_available():
        pytest.skip("CUDA indisponible")
    result = _run(tmp_path, "runs", device="cuda")
    assert result.steps == 3
