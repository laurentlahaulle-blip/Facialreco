"""Interfaces (Protocols) entre modules. Toute implémentation réelle ou de test
doit les respecter ; l'intégration ne dépend que de ces interfaces.

Les doubles de test (fakes) doivent se nommer ``Fake*`` ou vivre sous ``tests/`` et
ne jamais être présentés comme des résultats réels.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Protocol, runtime_checkable

from .decision import Decision
from .embeddings import Embedding, ModelRef
from .events import ArrivalEvent, AudioRequest
from .faces import Detection, TrackedFace
from .frames import CaptureState, Frame
from .ids import ProfileId, TrackId
from .recognition import RecognitionResult


@runtime_checkable
class FrameSource(Protocol):
    """Lot A. ``start`` ne bloque pas. Les images sont poussées via ``on_frame`` depuis
    le thread/processus de capture ; le consommateur doit être rapide (déposer dans une
    ``pipeline.LatestQueue``). ``stop`` est idempotent et libère la caméra."""

    def start(
        self,
        on_frame: Callable[[Frame], None],
        on_state: Callable[[CaptureState], None],
    ) -> None: ...

    def stop(self, timeout_s: float = 5.0) -> None: ...

    @property
    def state(self) -> CaptureState: ...


@runtime_checkable
class FaceDetector(Protocol):
    """Lot D."""

    name: str

    def detect(self, frame: Frame) -> Sequence[Detection]: ...


@runtime_checkable
class FaceTracker(Protocol):
    """Lot D. Associe les détections successives à des ``TrackId``.
    ``lost_tracks`` renvoie les passages terminés depuis le dernier appel."""

    def update(self, frame: Frame, detections: Sequence[Detection]) -> Sequence[TrackedFace]: ...

    def lost_tracks(self) -> Sequence[TrackId]: ...


@runtime_checkable
class FaceEmbedder(Protocol):
    """Lots D (référence) et E (notre modèle). Moteur interchangeable."""

    @property
    def model(self) -> ModelRef: ...

    def embed(self, frame: Frame, face: TrackedFace) -> Embedding | None:
        """Retourne None si le visage est inutilisable (trop petit, flou…)."""
        ...


@runtime_checkable
class Gallery(Protocol):
    """Lot D (recherche) adossé au lot C (stockage). Ne compare que des représentations
    du même ``ModelRef`` ; lève ``IncompatibleEmbeddingError`` sinon."""

    @property
    def model(self) -> ModelRef: ...

    def search(self, query: Embedding, top_k: int = 3) -> RecognitionResult | None: ...


@runtime_checkable
class Recognizer(Protocol):
    """Chaîne vision complète d'une image : détection + suivi + représentation + galerie."""

    def process(self, frame: Frame) -> Sequence[RecognitionResult]: ...

    def lost_tracks(self) -> Sequence[TrackId]: ...


@runtime_checkable
class DecisionEngine(Protocol):
    """Lot F. Agrège les résultats par passage et émet des décisions."""

    def observe(self, result: RecognitionResult) -> Decision | None: ...

    def track_lost(self, track_id: TrackId, monotonic_ns: int) -> None: ...


@runtime_checkable
class ArrivalTracker(Protocol):
    """Lot F. Transforme les décisions en arrivées, en supprimant les répétitions
    tant que la personne reste présente."""

    def on_decision(self, decision: Decision) -> ArrivalEvent | None: ...

    def tick(self, monotonic_ns: int) -> None:
        """Appel périodique : expiration des présences."""
        ...

    def note_track_activity(self, track_id: TrackId, monotonic_ns: int) -> None:
        """Signe de vie : appelé pour chaque RecognitionResult, même sans décision
        (contrats 1.1, proposition F-arrival-track-activity)."""
        ...

    def track_lost(self, track_id: TrackId, monotonic_ns: int) -> None:
        """Appelé pour chaque TrackId renvoyé par ``Recognizer.lost_tracks()``."""
        ...


@runtime_checkable
class WelcomeComposer(Protocol):
    """Lot F. Construit la demande audio d'un événement (None = rien à dire)."""

    def compose(self, event: ArrivalEvent) -> AudioRequest | None: ...


@runtime_checkable
class AudioPlayer(Protocol):
    """Lot F. Lecture locale, non bloquante, avec file et déduplication."""

    def submit(self, request: AudioRequest) -> bool: ...

    def close(self) -> None: ...


@runtime_checkable
class EmbeddingStore(Protocol):
    """Lot C. Accès aux représentations stockées, filtrées par modèle."""

    def load_embeddings(self, model: ModelRef) -> Iterable[tuple[ProfileId, Embedding]]: ...

    def add_embedding(
        self, profile_id: ProfileId, reference_id: int | None, embedding: Embedding
    ) -> int: ...

    def delete_embeddings_for_model(self, model: ModelRef) -> int: ...
