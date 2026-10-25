"""Doubles de développement ``Fake*`` — JAMAIS présentés comme des résultats réels.

- ``FakeFrameSource`` : images SYNTHÉTIQUES (dégradé + barre mobile), aucun visage, aucune
  donnée biométrique. Sert à voir l'aperçu bouger sans caméra (lot A non intégré).
- ``FakeProfileStore``, ``FakeReferenceSink``, ``FakeSettingsStore``,
  ``FakeArrivalHistory`` : dépôts en mémoire respectant ``app.ports``.

Aucun faux moteur de reconnaissance n'est fourni : sans moteur, l'interface affiche
« moteur non disponible ».
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

import numpy as np

from maisonaccueil.contracts import (
    ArrivalKind,
    CameraId,
    CaptureState,
    CaptureStatus,
    EventId,
    Frame,
    FrameId,
    ProfileId,
    SessionId,
    SourceConfig,
    SourceKind,
    new_profile_id,
    new_session_id,
)

FAKE_CAMERA_ID = CameraId("fake-synthetic")


class FakeFrameSource:
    """Source synthétique conforme à ``contracts.interfaces.FrameSource``."""

    def __init__(
        self,
        camera_id: CameraId = FAKE_CAMERA_ID,
        *,
        width: int = 640,
        height: int = 480,
        fps: float = 15.0,
        max_frames: int | None = None,
    ) -> None:
        self._camera_id = camera_id
        self._width = width
        self._height = height
        self._fps = fps
        self._max_frames = max_frames
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._captured = 0
        self._state = CaptureState(camera_id, CaptureStatus.STOPPED, time.monotonic_ns())
        self._on_state: Callable[[CaptureState], None] | None = None

    @property
    def state(self) -> CaptureState:
        with self._lock:
            return self._state

    def _set_state(self, status: CaptureStatus, detail: str = "") -> None:
        with self._lock:
            self._state = CaptureState(
                self._camera_id,
                status,
                time.monotonic_ns(),
                detail=detail,
                frames_captured=self._captured,
            )
            state = self._state
        if self._on_state is not None:
            self._on_state(state)

    def make_frame(self, index: int) -> Frame:
        h, w = self._height, self._width
        img = np.empty((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(40, 200, w, dtype=np.uint8)[None, :]
        img[:, :, 1] = np.linspace(60, 160, h, dtype=np.uint8)[:, None]
        img[:, :, 2] = 90
        x = (index * 8) % max(1, w - 20)
        img[:, x : x + 20, :] = 230
        return Frame(
            frame_id=FrameId(index),
            camera_id=self._camera_id,
            monotonic_ns=time.monotonic_ns(),
            wall_time=datetime.now(UTC),
            image=img,
        )

    def start(
        self,
        on_frame: Callable[[Frame], None],
        on_state: Callable[[CaptureState], None],
    ) -> None:
        if self._thread is not None:
            raise RuntimeError("FakeFrameSource déjà démarrée")
        self._on_state = on_state
        self._stop.clear()
        self._set_state(CaptureStatus.CONNECTING, "source synthétique")
        self._thread = threading.Thread(
            target=self._run, args=(on_frame,), name="fake-frame-source", daemon=True
        )
        self._thread.start()

    def _run(self, on_frame: Callable[[Frame], None]) -> None:
        self._set_state(CaptureStatus.RUNNING, "source synthétique (aucun visage)")
        period = 1.0 / self._fps if self._fps > 0 else 0.0
        index = 0
        while not self._stop.is_set():
            if self._max_frames is not None and index >= self._max_frames:
                break
            on_frame(self.make_frame(index))
            index += 1
            with self._lock:
                self._captured = index
            if period:
                self._stop.wait(period)

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            self._thread = None
            self._set_state(CaptureStatus.STOPPED, "arrêt demandé")


def fake_source_factory(config: SourceConfig) -> FakeFrameSource:
    """Fabrique de niveau module (transmissible à un processus ``spawn``)."""
    if config.kind != SourceKind.SYNTHETIC:
        raise ValueError("FakeFrameSource ne sert que les sources SYNTHETIC")
    return FakeFrameSource(
        config.camera_id,
        width=config.width or 640,
        height=config.height or 480,
        fps=config.fps or 15.0,
    )


# -- dépôts en mémoire --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FakeProfile:
    profile_id: ProfileId
    display_name: str
    consent_at: datetime
    active: bool = True
    spoken_name: str | None = None
    welcome_text: str | None = None
    consent_note: str | None = None


@dataclass(frozen=True, slots=True)
class FakeDeletionReport:
    profile_id: ProfileId
    references_deleted: int
    embeddings_deleted: int
    sessions_deleted: int
    arrivals_deleted: int
    files_deleted: int
    files_skipped: tuple[str, ...] = ()


class FakeProfileStore:
    _UPDATABLE = frozenset({"display_name", "spoken_name", "welcome_text", "consent_note"})

    def __init__(self) -> None:
        self.profiles: dict[ProfileId, FakeProfile] = {}
        self.on_delete: Callable[[ProfileId], tuple[int, int, int]] | None = None

    def list(self, *, include_inactive: bool = False) -> list[FakeProfile]:
        rows = [p for p in self.profiles.values() if include_inactive or p.active]
        return sorted(rows, key=lambda p: (p.display_name.casefold(), p.profile_id))

    def find(self, profile_id: ProfileId) -> FakeProfile | None:
        return self.profiles.get(profile_id)

    def create(
        self,
        display_name: str,
        *,
        consent_at: datetime,
        consent_note: str | None = None,
        spoken_name: str | None = None,
        welcome_text: str | None = None,
    ) -> FakeProfile:
        if not isinstance(consent_at, datetime) or consent_at.tzinfo is None:
            raise ValueError("Consentement requis (datetime timezone-aware)")
        name = display_name.strip()
        if not name:
            raise ValueError("display_name vide")
        p = FakeProfile(
            new_profile_id(), name, consent_at, True, spoken_name, welcome_text, consent_note
        )
        self.profiles[p.profile_id] = p
        return p

    def update(self, profile_id: ProfileId, **fields: str | None) -> FakeProfile:
        unknown = set(fields) - self._UPDATABLE
        if unknown:
            raise ValueError(f"Champs non modifiables : {sorted(unknown)}")
        p = self.profiles[profile_id]
        p = replace(p, **fields)  # type: ignore[arg-type]
        self.profiles[profile_id] = p
        return p

    def delete(self, profile_id: ProfileId, *, purge_arrivals: bool = True) -> FakeDeletionReport:
        if profile_id not in self.profiles:
            raise KeyError(profile_id)
        del self.profiles[profile_id]
        refs, sessions, arrivals = self.on_delete(profile_id) if self.on_delete else (0, 0, 0)
        return FakeDeletionReport(profile_id, refs, 0, sessions, arrivals, refs)


@dataclass(frozen=True, slots=True)
class FakeSession:
    session_id: SessionId
    profile_id: ProfileId | None
    started_at: datetime
    split: str | None
    camera_id: CameraId | None
    conditions: str | None


@dataclass(frozen=True, slots=True)
class FakeReference:
    reference_id: int
    profile_id: ProfileId
    session_id: SessionId | None
    size: int
    width: int | None
    height: int | None
    quality_note: str | None


class FakeReferenceSink:
    """Garde en mémoire la TAILLE des images reçues (pas leur contenu)."""

    def __init__(self) -> None:
        self.sessions: dict[SessionId, FakeSession] = {}
        self.references: dict[int, FakeReference] = {}
        self._next = 1

    def create_session(
        self,
        profile_id: ProfileId | None,
        *,
        started_at: datetime,
        split: str | None = None,
        camera_id: CameraId | None = None,
        conditions: str | None = None,
    ) -> FakeSession:
        s = FakeSession(new_session_id(), profile_id, started_at, split, camera_id, conditions)
        self.sessions[s.session_id] = s
        return s

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
    ) -> FakeReference:
        if not data:
            raise ValueError("Image vide")
        ref = FakeReference(
            self._next, profile_id, session_id, len(data), width, height, quality_note
        )
        self.references[ref.reference_id] = ref
        self._next += 1
        return ref

    def delete(self, reference_id: int) -> bool:
        return self.references.pop(reference_id, None) is not None

    def purge_profile(self, profile_id: ProfileId) -> tuple[int, int, int]:
        refs = [k for k, r in self.references.items() if r.profile_id == profile_id]
        for k in refs:
            del self.references[k]
        ses = [sid for sid, s in self.sessions.items() if s.profile_id == profile_id]
        for sid in ses:
            del self.sessions[sid]
        return len(refs), len(ses), 0


class FakeSettingsStore:
    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self.values: dict[str, Any] = dict(initial or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if not key:
            raise ValueError("Clé de réglage vide")
        self.values[key] = value


@dataclass(frozen=True, slots=True)
class FakeArrivalEntry:
    event_id: EventId
    kind: ArrivalKind
    occurred_at: datetime
    profile_id: ProfileId | None = None


class FakeArrivalHistory:
    def __init__(self, entries: list[FakeArrivalEntry] | None = None) -> None:
        self.entries = list(entries or [])

    def list_recent(self, limit: int = 100) -> list[FakeArrivalEntry]:
        rows = sorted(self.entries, key=lambda e: e.occurred_at, reverse=True)
        return rows[:limit]
