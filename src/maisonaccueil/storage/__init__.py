"""Lot C — Stockage SQLite (schéma ``contracts/schema.sql``).

Point d'entrée typique ::

    db = Database("var/c/maisonaccueil.sqlite")
    profiles = ProfileRepository(db)
    store = SqliteEmbeddingStore(db)   # implémente contracts.interfaces.EmbeddingStore
"""

from .arrivals import ArrivalLogEntry, ArrivalLogRepository
from .database import Database, SchemaVersionError
from .embeddings import CorruptEmbeddingError, SqliteEmbeddingStore
from .profiles import DeletionReport, Profile, ProfileNotFoundError, ProfileRepository
from .references import (
    SPLITS,
    CollectionSession,
    ReferenceImage,
    ReferenceNotFoundError,
    ReferenceRepository,
    SessionNotFoundError,
)
from .settings import SettingsRepository

__all__ = [
    "SPLITS",
    "ArrivalLogEntry",
    "ArrivalLogRepository",
    "CollectionSession",
    "CorruptEmbeddingError",
    "Database",
    "DeletionReport",
    "Profile",
    "ProfileNotFoundError",
    "ProfileRepository",
    "ReferenceImage",
    "ReferenceNotFoundError",
    "ReferenceRepository",
    "SchemaVersionError",
    "SessionNotFoundError",
    "SettingsRepository",
    "SqliteEmbeddingStore",
]
