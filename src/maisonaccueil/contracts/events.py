"""Événements d'arrivée et demandes de lecture audio."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime

from .ids import EventId, ProfileId, TrackId


class ArrivalKind(enum.StrEnum):
    KNOWN = "known"  # personne enregistrée, décision CONFIRMED
    UNKNOWN = "unknown"  # visage rejeté (inconnu) ; aucun nom n'est jamais prononcé


@dataclass(frozen=True, slots=True)
class ArrivalEvent:
    """Émis UNE fois par présence. Une personne restée présente (même avec pertes
    de suivi temporaires) ne génère pas de nouvel événement."""

    event_id: EventId
    kind: ArrivalKind
    track_id: TrackId
    monotonic_ns: int
    wall_time: datetime
    profile_id: ProfileId | None = None

    def __post_init__(self) -> None:
        if (self.kind == ArrivalKind.KNOWN) != (self.profile_id is not None):
            raise ValueError("profile_id requis pour KNOWN et interdit pour UNKNOWN")


class AudioPriority(enum.IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2


@dataclass(frozen=True, slots=True)
class AudioRequest:
    """Demande de lecture d'un message vocal LOCAL.

    Exactement l'un de ``audio_path`` (fichier local préenregistré) ou ``text``
    (synthèse vocale locale) doit être fourni. ``dedupe_key`` permet au lecteur
    d'ignorer une demande identique déjà en file (arrivées simultanées).
    ``expires_after_ms`` : au-delà, la demande est abandonnée (accueil périmé).
    """

    request_id: EventId
    arrival_event_id: EventId
    monotonic_ns: int
    dedupe_key: str
    priority: AudioPriority = AudioPriority.NORMAL
    audio_path: str | None = None
    text: str | None = None
    profile_id: ProfileId | None = None
    expires_after_ms: int = 10_000

    def __post_init__(self) -> None:
        if (self.audio_path is None) == (self.text is None):
            raise ValueError("AudioRequest: fournir exactement audio_path OU text")
