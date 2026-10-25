"""Représentations faciales liées à la version du modèle qui les a produites.

Règle absolue : deux représentations ne sont comparables que si leurs ``ModelRef``
sont identiques (même nom, même version, même dimension, même métrique, même
prétraitement). Utiliser ``ensure_compatible`` avant toute comparaison.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


class Metric(enum.StrEnum):
    COSINE = "cosine"  # similarité cosinus dans [-1, 1], plus haut = plus proche
    L2 = "l2"  # distance euclidienne, plus bas = plus proche


@dataclass(frozen=True, slots=True)
class ModelRef:
    """Identité exacte d'un moteur de représentation.

    ``name``        : ex. "reference-xyz" ou "maisonaccueil-scratch".
    ``version``     : version des POIDS (hash court ou étiquette d'expérience), pas du code.
    ``preprocess``  : identifiant du pipeline d'alignement/normalisation (ex. "align112-v1").
    ``origin``      : "pretrained" (poids tiers) ou "scratch" (entraîné de zéro par nous)
                      ou "finetuned" (poids tiers adaptés). Ne jamais déclarer "scratch"
                      un modèle partiellement préentraîné.
    """

    name: str
    version: str
    embedding_dim: int
    metric: Metric
    preprocess: str
    origin: str

    def __post_init__(self) -> None:
        if self.origin not in {"pretrained", "scratch", "finetuned", "test-double"}:
            raise ValueError(f"ModelRef.origin invalide: {self.origin!r}")
        if self.embedding_dim <= 0:
            raise ValueError("ModelRef.embedding_dim doit être > 0")

    @property
    def key(self) -> str:
        """Clé stable pour le stockage (colonne ``model_key``)."""
        return f"{self.name}@{self.version}#{self.preprocess}/{self.metric}/{self.embedding_dim}"


class IncompatibleEmbeddingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Embedding:
    model: ModelRef
    vector: npt.NDArray[np.float32] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.vector.ndim != 1 or self.vector.shape[0] != self.model.embedding_dim:
            raise ValueError(
                f"Embedding: dimension {self.vector.shape} incompatible avec "
                f"{self.model.embedding_dim}"
            )
        if self.vector.dtype != np.float32:
            raise ValueError("Embedding.vector doit être float32")


def ensure_compatible(a: ModelRef, b: ModelRef) -> None:
    """Lève ``IncompatibleEmbeddingError`` si les deux références diffèrent."""
    if a != b:
        raise IncompatibleEmbeddingError(f"Représentations incomparables: {a.key} vs {b.key}")
