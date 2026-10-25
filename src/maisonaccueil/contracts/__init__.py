"""Contrats partagés de MaisonAccueil — PROPRIÉTÉ DU COORDINATEUR.

Toute modification passe par une proposition écrite dans docs/contract-changes/.
Les lots importent depuis ce paquet, jamais depuis l'implémentation d'un autre lot.
"""

from importlib.resources import files

from .decision import Decision, DecisionStatus
from .embeddings import (
    Embedding,
    IncompatibleEmbeddingError,
    Metric,
    ModelRef,
    ensure_compatible,
)
from .events import ArrivalEvent, ArrivalKind, AudioPriority, AudioRequest
from .faces import BoundingBox, Detection, Landmarks5, TrackedFace
from .frames import CaptureState, CaptureStatus, Frame, SourceConfig, SourceKind
from .ids import (
    CameraId,
    EventId,
    FrameId,
    ProfileId,
    SessionId,
    TrackId,
    new_event_id,
    new_profile_id,
    new_session_id,
    new_track_id,
)
from .recognition import Candidate, QualityFlag, RawScore, RecognitionResult

CONTRACTS_VERSION = "1.0.0"
SCHEMA_VERSION = 1


def schema_sql() -> str:
    """Texte du schéma SQLite partagé (contracts/schema.sql)."""
    return files(__package__).joinpath("schema.sql").read_text(encoding="utf-8")


__all__ = [
    "CONTRACTS_VERSION",
    "SCHEMA_VERSION",
    "ArrivalEvent",
    "ArrivalKind",
    "AudioPriority",
    "AudioRequest",
    "BoundingBox",
    "CameraId",
    "Candidate",
    "CaptureState",
    "CaptureStatus",
    "Decision",
    "DecisionStatus",
    "Detection",
    "Embedding",
    "EventId",
    "Frame",
    "FrameId",
    "IncompatibleEmbeddingError",
    "Landmarks5",
    "Metric",
    "ModelRef",
    "ProfileId",
    "QualityFlag",
    "RawScore",
    "RecognitionResult",
    "SessionId",
    "SourceConfig",
    "SourceKind",
    "TrackId",
    "TrackedFace",
    "ensure_compatible",
    "new_event_id",
    "new_profile_id",
    "new_session_id",
    "new_track_id",
    "schema_sql",
]
