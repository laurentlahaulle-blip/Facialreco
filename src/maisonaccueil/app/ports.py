"""Interfaces des dépôts injectés dans l'interface.

Ce sont des sous-ensembles structurels de l'API publique du stockage (lot C,
``docs/lots/C-storage.md``) : ``ProfileRepository``, ``ReferenceRepository``,
``SettingsRepository`` et ``ArrivalLogRepository`` les satisfont sans adaptateur.
L'interface ne dépend que de ces Protocols ; les doubles ``Fake*`` (``app.fakes``)
servent au développement et aux tests.
"""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol

from maisonaccueil.contracts import ArrivalKind, CameraId, EventId, ProfileId, SessionId


class ProfileRecord(Protocol):
    @property
    def profile_id(self) -> ProfileId: ...
    @property
    def display_name(self) -> str: ...
    @property
    def spoken_name(self) -> str | None: ...
    @property
    def welcome_text(self) -> str | None: ...
    @property
    def consent_at(self) -> datetime: ...
    @property
    def active(self) -> bool: ...


class DeletionSummary(Protocol):
    @property
    def references_deleted(self) -> int: ...
    @property
    def embeddings_deleted(self) -> int: ...
    @property
    def sessions_deleted(self) -> int: ...
    @property
    def arrivals_deleted(self) -> int: ...
    @property
    def files_deleted(self) -> int: ...
    @property
    def files_skipped(self) -> tuple[str, ...]: ...


class ProfileStore(Protocol):
    def list(self, *, include_inactive: bool = False) -> builtins.list[Any]: ...

    def find(self, profile_id: ProfileId) -> Any: ...

    def create(
        self,
        display_name: str,
        *,
        consent_at: datetime,
        consent_note: str | None = None,
        spoken_name: str | None = None,
        welcome_text: str | None = None,
    ) -> Any: ...

    def update(self, profile_id: ProfileId, **fields: str | None) -> Any: ...

    def delete(self, profile_id: ProfileId, *, purge_arrivals: bool = True) -> Any:
        """Suppression COMPLÈTE (références, représentations, séances, fichiers, journal)."""
        ...


class CollectionSessionRecord(Protocol):
    @property
    def session_id(self) -> SessionId: ...


class StoredReference(Protocol):
    @property
    def reference_id(self) -> int: ...


class ReferenceSink(Protocol):
    """Enregistrement des images de la collecte guidée (séance ``split='enroll'``)."""

    def create_session(
        self,
        profile_id: ProfileId | None,
        *,
        started_at: datetime,
        split: str | None = None,
        camera_id: CameraId | None = None,
        conditions: str | None = None,
    ) -> Any: ...

    def add_image_bytes(
        self,
        profile_id: ProfileId,
        data: bytes,
        *,
        extension: str = ".png",
        captured_at: datetime,
        session_id: SessionId | None = None,
        width: int | None = None,
        height: int | None = None,
        quality_note: str | None = None,
    ) -> Any: ...

    def delete(self, reference_id: int) -> bool: ...


class SettingsStore(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...

    def set(self, key: str, value: Any) -> None: ...


class ArrivalRecord(Protocol):
    @property
    def event_id(self) -> EventId: ...
    @property
    def kind(self) -> ArrivalKind: ...
    @property
    def occurred_at(self) -> datetime: ...
    @property
    def profile_id(self) -> ProfileId | None: ...


class ArrivalHistory(Protocol):
    def list_recent(self, limit: int = 100) -> Sequence[Any]: ...
