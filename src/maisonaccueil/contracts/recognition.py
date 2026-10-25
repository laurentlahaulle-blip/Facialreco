"""Résultat technique de reconnaissance (avant décision).

Un ``RawScore`` est un score TECHNIQUE du modèle (ex. similarité cosinus). Ce n'est
PAS une probabilité. Il ne doit jamais être affiché comme « % de fiabilité ».
Une éventuelle probabilité calibrée ne pourra venir que du lot G, après mesure,
et portera ``calibrated=True`` avec la référence de la calibration.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .embeddings import Metric, ModelRef
from .ids import FrameId, ProfileId, TrackId


@dataclass(frozen=True, slots=True)
class RawScore:
    value: float
    metric: Metric

    def is_better_than(self, other: RawScore) -> bool:
        if self.metric != other.metric:
            raise ValueError("Comparaison de scores de métriques différentes")
        if self.metric == Metric.COSINE:
            return self.value > other.value
        return self.value < other.value


@dataclass(frozen=True, slots=True)
class Candidate:
    profile_id: ProfileId
    score: RawScore
    references_compared: int  # nombre de références du profil effectivement comparées


class QualityFlag(enum.StrEnum):
    FACE_TOO_SMALL = "face_too_small"
    BLURRY = "blurry"
    NOT_FRONTAL = "not_frontal"
    OCCLUDED = "occluded"
    LOW_LIGHT = "low_light"


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    """Résultat pour UN visage suivi dans UNE image.

    ``candidates`` est trié du meilleur au moins bon (peut être vide si la galerie
    ne contient aucune référence compatible avec ``model``).
    """

    frame_id: FrameId
    track_id: TrackId
    monotonic_ns: int
    model: ModelRef
    candidates: tuple[Candidate, ...]
    quality_flags: frozenset[QualityFlag] = frozenset()

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None
