"""Chaîne de reconnaissance d'une image : détection → suivi → qualité → représentation →
galerie. Implémente ``Recognizer``.

Un ``RecognitionResult`` est émis pour CHAQUE visage suivi, y compris quand la
représentation est impossible (candidats vides, ``quality_flags`` renseignés) : la
décision (lot F) peut ainsi compter les observations insuffisantes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from maisonaccueil.contracts import (
    Candidate,
    Frame,
    QualityFlag,
    RecognitionResult,
    TrackId,
    ensure_compatible,
)
from maisonaccueil.contracts.interfaces import (
    EmbeddingStore,
    FaceDetector,
    FaceEmbedder,
    FaceTracker,
)
from maisonaccueil.vision.quality import QualityConfig, assess_quality
from maisonaccueil.vision.tracking import IouTracker

from .gallery import InMemoryGallery


@dataclass(frozen=True, slots=True)
class RecognizerConfig:
    top_k: int = 3
    quality: QualityConfig = QualityConfig()
    # Indicateurs qui empêchent de calculer une représentation (économie de calcul).
    skip_embedding_on: frozenset[QualityFlag] = frozenset({QualityFlag.FACE_TOO_SMALL})


class ReferenceRecognizer:
    def __init__(
        self,
        detector: FaceDetector,
        tracker: FaceTracker,
        embedder: FaceEmbedder,
        gallery: InMemoryGallery,
        config: RecognizerConfig | None = None,
    ) -> None:
        ensure_compatible(embedder.model, gallery.model)
        self.detector = detector
        self.tracker = tracker
        self.embedder = embedder
        self.gallery = gallery
        self.config = config or RecognizerConfig()

    def process(self, frame: Frame) -> Sequence[RecognitionResult]:
        detections = self.detector.detect(frame)
        tracked = self.tracker.update(frame, detections)
        results: list[RecognitionResult] = []
        for face in tracked:
            flags = assess_quality(frame.image, face.detection, self.config.quality)
            candidates: tuple[Candidate, ...] = ()
            if not (flags & self.config.skip_embedding_on):
                emb = self.embedder.embed(frame, face)
                if emb is not None:
                    candidates = self.gallery.rank(emb, self.config.top_k)
            results.append(
                RecognitionResult(
                    frame_id=frame.frame_id,
                    track_id=face.track_id,
                    monotonic_ns=frame.monotonic_ns,
                    model=self.embedder.model,
                    candidates=candidates,
                    quality_flags=flags,
                )
            )
        return results

    def lost_tracks(self) -> Sequence[TrackId]:
        return self.tracker.lost_tracks()


def build_reference_recognizer(
    store: EmbeddingStore | None = None,
    models_dir: Path | None = None,
    config: RecognizerConfig | None = None,
) -> ReferenceRecognizer:
    """Assemble YuNet + IouTracker + SFace + galerie. Lève si les poids sont absents ou
    altérés (``ModelFileMissingError`` / ``ModelIntegrityError``)."""
    from maisonaccueil.vision.detection import YuNetDetector
    from maisonaccueil.vision.model_files import SFACE_2021DEC, YUNET_2023MAR, resolve_verified

    from .embedders import SFaceEmbedder

    detector = YuNetDetector(resolve_verified(YUNET_2023MAR, models_dir), verify=False)
    embedder = SFaceEmbedder(resolve_verified(SFACE_2021DEC, models_dir), verify=False)
    gallery = InMemoryGallery(embedder.model, store)
    return ReferenceRecognizer(detector, IouTracker(), embedder, gallery, config)
