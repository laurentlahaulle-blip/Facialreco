"""Doubles de test ``Fake*`` du lot D (représentation, stockage). Jamais présentés comme
des résultats réels ; ``ModelRef.origin = "test-double"``."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from maisonaccueil.contracts import (
    Embedding,
    Frame,
    Metric,
    ModelRef,
    ProfileId,
    TrackedFace,
)

FAKE_COLOR_MODEL_REF = ModelRef(
    name="fake-color-histogram",
    version="v1",
    embedding_dim=24,
    metric=Metric.COSINE,
    preprocess="bbox-crop-v1",
    origin="test-double",
)


class FakeColorEmbedder:
    """Histogramme de couleur (8 classes × 3 canaux) de la boîte, normalisé.

    Deux rectangles synthétiques de même couleur donnent des vecteurs proches : de quoi
    tester galerie et chaîne sans réseau de neurones.
    """

    def __init__(self, model: ModelRef = FAKE_COLOR_MODEL_REF, min_face_px: float = 0.0) -> None:
        if model.embedding_dim != 24:
            raise ValueError("FakeColorEmbedder produit 24 dimensions")
        self._model = model
        self.min_face_px = min_face_px
        self.calls = 0

    @property
    def model(self) -> ModelRef:
        return self._model

    def embed(self, frame: Frame, face: TrackedFace) -> Embedding | None:
        self.calls += 1
        b = face.detection.bbox
        if min(b.width, b.height) < self.min_face_px:
            return None
        x0, y0 = max(0, int(b.x)), max(0, int(b.y))
        x1 = min(frame.width, int(b.x + b.width))
        y1 = min(frame.height, int(b.y + b.height))
        crop = frame.image[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        hist = np.concatenate(
            [cv2.calcHist([crop], [c], None, [8], [0, 256]).reshape(-1) for c in range(3)]
        ).astype(np.float32)
        n = float(np.linalg.norm(hist))
        if n <= 0:
            return None
        return Embedding(self._model, (hist / n).astype(np.float32))


class FakeRandomEmbedder:
    """Vecteurs aléatoires normalisés, reproductibles (graine), sans regarder l'image."""

    def __init__(self, model: ModelRef, seed: int = 0) -> None:
        self._model = model
        self._rng = np.random.default_rng(seed)

    @property
    def model(self) -> ModelRef:
        return self._model

    def random_embedding(self) -> Embedding:
        v = self._rng.standard_normal(self._model.embedding_dim).astype(np.float32)
        return Embedding(self._model, (v / np.linalg.norm(v)).astype(np.float32))

    def embed(self, frame: Frame, face: TrackedFace) -> Embedding | None:
        return self.random_embedding()


class FakeEmbeddingStore:
    """``EmbeddingStore`` en mémoire (le vrai stockage relève du lot C)."""

    def __init__(self) -> None:
        self._rows: list[tuple[int, ProfileId, int | None, Embedding]] = []
        self._next_id = 1

    def load_embeddings(self, model: ModelRef) -> Iterable[tuple[ProfileId, Embedding]]:
        return [(pid, e) for _i, pid, _r, e in self._rows if e.model.key == model.key]

    def add_embedding(
        self, profile_id: ProfileId, reference_id: int | None, embedding: Embedding
    ) -> int:
        rid = self._next_id
        self._next_id += 1
        self._rows.append((rid, profile_id, reference_id, embedding))
        return rid

    def delete_embeddings_for_model(self, model: ModelRef) -> int:
        before = len(self._rows)
        self._rows = [r for r in self._rows if r[3].model.key != model.key]
        return before - len(self._rows)


class FakeLeakyEmbeddingStore(FakeEmbeddingStore):
    """Stockage défaillant qui ignore le filtre par modèle (test de robustesse)."""

    def load_embeddings(self, model: ModelRef) -> Iterable[tuple[ProfileId, Embedding]]:
        return [(pid, e) for _i, pid, _r, e in self._rows]


__all__ = [
    "FAKE_COLOR_MODEL_REF",
    "FakeColorEmbedder",
    "FakeEmbeddingStore",
    "FakeLeakyEmbeddingStore",
    "FakeRandomEmbedder",
]
