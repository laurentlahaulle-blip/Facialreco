"""Décision finale par passage : confirmé / inconnu / incertain.

La décision est distincte du score : elle résulte d'une politique (seuils, nombre
d'observations concordantes, fenêtre temporelle) appliquée à plusieurs
``RecognitionResult`` d'un même ``TrackId``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .embeddings import ModelRef
from .ids import ProfileId, TrackId
from .recognition import RawScore


class DecisionStatus(enum.StrEnum):
    CONFIRMED = "confirmed"  # identité confirmée selon la politique en vigueur
    UNKNOWN = "unknown"  # rejet : visage suffisamment observé, aucun profil accepté
    UNCERTAIN = "uncertain"  # observations insuffisantes ou contradictoires


@dataclass(frozen=True, slots=True)
class Decision:
    track_id: TrackId
    status: DecisionStatus
    monotonic_ns: int
    model: ModelRef
    policy: str  # identifiant/version de la politique de décision (ex. "temporal-v1")
    observations: int  # nombre de résultats pris en compte
    window_ms: int  # durée couverte par ces observations
    profile_id: ProfileId | None = None  # renseigné SEULEMENT si CONFIRMED
    best_score: RawScore | None = None  # information technique, pas une probabilité
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status == DecisionStatus.CONFIRMED and self.profile_id is None:
            raise ValueError("Decision CONFIRMED sans profile_id")
        if self.status != DecisionStatus.CONFIRMED and self.profile_id is not None:
            raise ValueError("profile_id interdit hors CONFIRMED")
