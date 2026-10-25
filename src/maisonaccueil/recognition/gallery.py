"""Galerie en mémoire, adossée à un ``EmbeddingStore`` (lot C), filtrée par ``ModelRef``.

Recherche top-k **par profil** : le score d'un profil est le meilleur score parmi ses
références (``references_compared`` = nombre de références comparées). Les scores sont
des ``RawScore`` techniques (cosinus ou L2), jamais des probabilités.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from maisonaccueil.contracts import (
    Candidate,
    Embedding,
    FrameId,
    Metric,
    ModelRef,
    ProfileId,
    RawScore,
    RecognitionResult,
    TrackId,
    ensure_compatible,
)
from maisonaccueil.contracts.interfaces import EmbeddingStore


def _unit(v: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


class InMemoryGallery:
    """Implémente ``Gallery``. Thread-safe (verrou interne) pour lecture/ajout concurrents.

    ``search`` renvoie un ``RecognitionResult`` dont ``frame_id``/``track_id`` sont
    neutres (0, ""), car la galerie ne connaît pas l'image : utiliser ``rank`` pour
    obtenir directement les candidats, ou ``Recognizer`` pour un résultat complet.
    """

    def __init__(self, model: ModelRef, store: EmbeddingStore | None = None) -> None:
        self._model = model
        self._store = store
        self._lock = threading.Lock()
        self._profiles: list[ProfileId] = []
        self._matrix: npt.NDArray[np.float32] = np.zeros((0, model.embedding_dim), np.float32)
        if store is not None:
            self.reload()

    @property
    def model(self) -> ModelRef:
        return self._model

    def __len__(self) -> int:
        with self._lock:
            return len(self._profiles)

    @property
    def profile_ids(self) -> frozenset[ProfileId]:
        with self._lock:
            return frozenset(self._profiles)

    def _prepare(self, emb: Embedding) -> npt.NDArray[np.float32]:
        ensure_compatible(self._model, emb.model)
        v = emb.vector.astype(np.float32, copy=True)
        return _unit(v) if self._model.metric == Metric.COSINE else v

    def reload(self) -> int:
        """Recharge depuis le stockage (seulement les représentations de ``model``)."""
        if self._store is None:
            return len(self)
        rows = list(self._store.load_embeddings(self._model))
        self._replace(rows)
        return len(rows)

    def _replace(self, rows: Iterable[tuple[ProfileId, Embedding]]) -> None:
        profiles: list[ProfileId] = []
        vecs: list[npt.NDArray[np.float32]] = []
        for pid, emb in rows:
            vecs.append(self._prepare(emb))  # lève si le stockage renvoie un autre modèle
            profiles.append(pid)
        matrix = np.stack(vecs) if vecs else np.zeros((0, self._model.embedding_dim), np.float32)
        with self._lock:
            self._profiles, self._matrix = profiles, matrix

    def add(
        self,
        profile_id: ProfileId,
        embedding: Embedding,
        persist: bool = False,
        reference_id: int | None = None,
    ) -> None:
        """Ajoute une référence en mémoire (et dans le stockage si ``persist``)."""
        v = self._prepare(embedding)
        if persist:
            if self._store is None:
                raise RuntimeError("Aucun EmbeddingStore configuré")
            self._store.add_embedding(profile_id, reference_id, embedding)
        with self._lock:
            self._profiles = [*self._profiles, profile_id]
            self._matrix = np.vstack([self._matrix, v[None, :]])

    def remove_profile(self, profile_id: ProfileId) -> int:
        """Retire un profil de la mémoire (la suppression en base relève du lot C)."""
        with self._lock:
            keep = [i for i, p in enumerate(self._profiles) if p != profile_id]
            removed = len(self._profiles) - len(keep)
            self._profiles = [self._profiles[i] for i in keep]
            self._matrix = self._matrix[keep]
        return removed

    def rank(self, query: Embedding, top_k: int = 3) -> tuple[Candidate, ...]:
        if top_k < 1:
            raise ValueError("top_k doit être >= 1")
        q = self._prepare(query)
        with self._lock:
            profiles, matrix = self._profiles, self._matrix
        if not profiles:
            return ()
        metric = self._model.metric
        if metric == Metric.COSINE:
            scores = matrix @ q
        else:
            scores = np.linalg.norm(matrix - q[None, :], axis=1)
        best: dict[ProfileId, float] = {}
        counts: dict[ProfileId, int] = {}
        for pid, s in zip(profiles, scores.tolist(), strict=True):
            counts[pid] = counts.get(pid, 0) + 1
            cur = best.get(pid)
            if cur is None or (s > cur if metric == Metric.COSINE else s < cur):
                best[pid] = s
        order = sorted(
            best.items(), key=lambda kv: (-kv[1] if metric == Metric.COSINE else kv[1], kv[0])
        )
        return tuple(
            Candidate(pid, RawScore(float(s), metric), counts[pid]) for pid, s in order[:top_k]
        )

    def search(self, query: Embedding, top_k: int = 3) -> RecognitionResult | None:
        candidates = self.rank(query, top_k)
        return RecognitionResult(
            frame_id=FrameId(0),
            track_id=TrackId(""),
            monotonic_ns=0,
            model=self._model,
            candidates=candidates,
        )
