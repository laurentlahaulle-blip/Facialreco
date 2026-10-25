from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from maisonaccueil.contracts import Metric, ModelRef
from maisonaccueil.contracts.interfaces import FaceEmbedder
from maisonaccueil.training.config import ExperimentConfig
from maisonaccueil.training.embedder import ScratchFaceEmbedder
from maisonaccueil.training.runs import build_model_ref

from .conftest import landmarks_at, make_face, make_frame


class FakeBackend:
    """Projection aléatoire fixe : double de test, PAS un modèle."""

    def __init__(self, dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.proj = rng.standard_normal((3 * 112 * 112, dim)).astype(np.float32)
        self.calls = 0

    def __call__(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        self.calls += 1
        assert batch.shape[1:] == (3, 112, 112) and batch.dtype == np.float32
        return (batch.reshape(batch.shape[0], -1) @ self.proj).astype(np.float32)


@pytest.fixture
def ref() -> ModelRef:
    return build_model_ref(ExperimentConfig(name="fake", embedding_dim=16), "ef" * 32)


def test_implements_face_embedder(ref: ModelRef) -> None:
    emb = ScratchFaceEmbedder(ref, FakeBackend(16))
    assert isinstance(emb, FaceEmbedder)
    assert emb.model.origin == "scratch" and emb.model.metric is Metric.COSINE


def test_embed_returns_normalized_vector_with_model_ref(ref: ModelRef) -> None:
    emb = ScratchFaceEmbedder(ref, FakeBackend(16))
    e = emb.embed(make_frame(), make_face(landmarks_at(2.0, 150, 80)))
    assert e is not None
    assert e.model == ref and e.vector.dtype == np.float32
    assert abs(float(np.linalg.norm(e.vector)) - 1.0) < 1e-5


def test_unusable_faces_return_none(ref: ModelRef) -> None:
    backend = FakeBackend(16)
    emb = ScratchFaceEmbedder(ref, backend)
    assert emb.embed(make_frame(), make_face(None)) is None
    assert emb.embed(make_frame(), make_face(landmarks_at(0.2, 5, 5))) is None
    assert backend.calls == 0


def test_invalid_backend_output_returns_none(ref: ModelRef) -> None:
    wrong_dim = ScratchFaceEmbedder(ref, FakeBackend(8))
    assert wrong_dim.embed(make_frame(), make_face(landmarks_at(2.0, 150, 80))) is None
    zeros = ScratchFaceEmbedder(ref, lambda b: np.zeros((b.shape[0], 16), np.float32))
    assert zeros.embed(make_frame(), make_face(landmarks_at(2.0, 150, 80))) is None


def test_preprocess_mismatch_is_refused() -> None:
    other = ModelRef("x", "v", 16, Metric.COSINE, "other-v1", "scratch")
    with pytest.raises(ValueError):
        ScratchFaceEmbedder(other, FakeBackend(16))
