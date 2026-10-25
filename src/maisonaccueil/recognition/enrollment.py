"""Calcul des représentations d'enrôlement à partir des images de référence.

L'appelant fournit les références (profil, ``reference_id``, chemin du fichier local) et
un ``EmbeddingStore`` ; ce module ne lit jamais SQLite directement. Pour chaque image :
lecture → détection → choix du visage principal (le plus grand) → contrôle qualité →
représentation (``FaceEmbedder`` quelconque, donc aussi le moteur du lot E) → stockage
lié au ``ModelRef`` exact via ``add_embedding(profile_id, reference_id, emb)``.

Règles de sélection (prudentes : mieux vaut ignorer une image que d'enrôler la mauvaise
personne) :
- aucun visage → ignorée (``no_face``) ;
- visage principal de qualité insuffisante → ignorée (``low_quality``), sans repli sur un
  visage plus petit, qui pourrait être une autre personne ;
- deux visages de tailles comparables (second ≥ ``ambiguous_area_ratio`` × le plus grand)
  → ignorée (``ambiguous_faces``).

Idempotence : une référence déjà représentée pour ce modèle n'est pas recalculée. L'interface
``EmbeddingStore`` (contrats 1.2) ne permet pas de le savoir ; en attendant la proposition
``docs/contract-changes/D-represented-references.md`` :
1. si le stockage expose ``represented_reference_ids(model)``, elle est utilisée ;
2. sinon l'appelant peut passer ``already_represented`` ;
3. dans tous les cas, l'enrôleur retient ce qu'il a ajouté (même instance, même modèle).
"""

from __future__ import annotations

import enum
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from maisonaccueil.contracts import (
    CameraId,
    Detection,
    Embedding,
    Frame,
    FrameId,
    ModelRef,
    ProfileId,
    QualityFlag,
    TrackedFace,
    TrackId,
    ensure_compatible,
)
from maisonaccueil.contracts.interfaces import EmbeddingStore, FaceDetector, FaceEmbedder
from maisonaccueil.vision.quality import QualityConfig, assess_quality

ENROLLMENT_CAMERA_ID = CameraId("enrollment")


class SkipReason(enum.StrEnum):
    ALREADY_REPRESENTED = "already_represented"
    DUPLICATE_IN_BATCH = "duplicate_in_batch"
    FILE_MISSING = "file_missing"
    UNREADABLE = "unreadable"
    NO_FACE = "no_face"
    AMBIGUOUS_FACES = "ambiguous_faces"
    LOW_QUALITY = "low_quality"
    EMBEDDING_FAILED = "embedding_failed"
    STORE_ERROR = "store_error"


@dataclass(frozen=True, slots=True)
class ReferenceItem:
    profile_id: ProfileId
    reference_id: int
    image_path: Path


@dataclass(frozen=True, slots=True)
class AddedReference:
    profile_id: ProfileId
    reference_id: int
    embedding_id: int
    quality_flags: frozenset[QualityFlag]  # indicateurs tolérés (non bloquants)


@dataclass(frozen=True, slots=True)
class SkippedReference:
    profile_id: ProfileId
    reference_id: int
    reason: SkipReason
    detail: str = ""


@dataclass(slots=True)
class EnrollmentReport:
    model: ModelRef
    processed: int = 0
    added: list[AddedReference] = field(default_factory=list)
    skipped: list[SkippedReference] = field(default_factory=list)

    @property
    def added_count(self) -> int:
        return len(self.added)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    def skipped_by_reason(self) -> dict[SkipReason, int]:
        out: dict[SkipReason, int] = {}
        for s in self.skipped:
            out[s.reason] = out.get(s.reason, 0) + 1
        return out

    def summary(self) -> str:
        """Bilan lisible (français) ; aucun score n'y figure."""
        parts = [f"{self.processed} référence(s) traitée(s)", f"{self.added_count} ajoutée(s)"]
        reasons = ", ".join(f"{r.value}: {n}" for r, n in sorted(self.skipped_by_reason().items()))
        parts.append(f"{self.skipped_count} ignorée(s)" + (f" ({reasons})" if reasons else ""))
        return f"Enrôlement [{self.model.key}] : " + ", ".join(parts)


@dataclass(frozen=True, slots=True)
class EnrollmentConfig:
    quality: QualityConfig = QualityConfig()
    # Indicateurs qui font rejeter une image d'enrôlement (seuils NON mesurés, cf. quality.py).
    reject_flags: frozenset[QualityFlag] = frozenset(
        {
            QualityFlag.FACE_TOO_SMALL,
            QualityFlag.BLURRY,
            QualityFlag.LOW_LIGHT,
            QualityFlag.NOT_FRONTAL,
            QualityFlag.OCCLUDED,
        }
    )
    ambiguous_area_ratio: float = 0.5
    max_image_side: int = 4096  # au-delà : image refusée (protection mémoire)


def read_image(path: Path) -> npt.NDArray[np.uint8] | None:
    """Lit une image en BGR uint8 ; gère les chemins non ASCII (Windows). None si illisible."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        return None
    return np.ascontiguousarray(img, dtype=np.uint8)


def _as_item(ref: ReferenceItem | tuple[ProfileId | str, int, Path | str]) -> ReferenceItem:
    if isinstance(ref, ReferenceItem):
        return ref
    profile_id, reference_id, path = ref
    return ReferenceItem(ProfileId(str(profile_id)), int(reference_id), Path(path))


class ReferenceEnroller:
    """Transforme des images de référence en représentations stockées pour ``embedder.model``."""

    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        config: EnrollmentConfig | None = None,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.config = config or EnrollmentConfig()
        self._represented: set[int] = set()

    @property
    def model(self) -> ModelRef:
        return self.embedder.model

    # -- étapes unitaires (réutilisables par l'interface pour un retour immédiat) --

    def select_face(
        self, detections: Sequence[Detection]
    ) -> tuple[Detection | None, SkipReason | None]:
        if not detections:
            return None, SkipReason.NO_FACE
        ordered = sorted(detections, key=lambda d: d.bbox.area, reverse=True)
        main = ordered[0]
        if len(ordered) > 1 and ordered[1].bbox.area >= (
            self.config.ambiguous_area_ratio * main.bbox.area
        ):
            return None, SkipReason.AMBIGUOUS_FACES
        return main, None

    def embed_image(
        self, image: npt.NDArray[np.uint8], reference_id: int = 0
    ) -> tuple[Embedding | None, frozenset[QualityFlag], SkipReason | None, str]:
        """Représentation du visage principal d'une image (sans stockage)."""
        h, w = image.shape[:2]
        if max(h, w) > self.config.max_image_side:
            return None, frozenset(), SkipReason.UNREADABLE, f"image trop grande ({w}×{h})"
        frame = Frame(
            frame_id=FrameId(reference_id),
            camera_id=ENROLLMENT_CAMERA_ID,
            monotonic_ns=time.monotonic_ns(),
            wall_time=datetime.now(UTC),
            image=image,
        )
        face, reason = self.select_face(self.detector.detect(frame))
        if face is None:
            return None, frozenset(), reason, ""
        flags = assess_quality(image, face, self.config.quality)
        blocking = flags & self.config.reject_flags
        if blocking:
            detail = ",".join(sorted(f.value for f in blocking))
            return None, flags, SkipReason.LOW_QUALITY, detail
        tracked = TrackedFace(
            track_id=TrackId(f"enroll-{reference_id}"),
            detection=face,
            monotonic_ns=frame.monotonic_ns,
            track_age_frames=0,
        )
        emb = self.embedder.embed(frame, tracked)
        if emb is None:
            return None, flags, SkipReason.EMBEDDING_FAILED, "représentation refusée"
        ensure_compatible(self.model, emb.model)
        return emb, flags, None, ""

    # -- traitement par lot --

    def _known_represented(
        self, store: EmbeddingStore, already_represented: Iterable[int] | None
    ) -> set[int]:
        known = set(self._represented)
        if already_represented is not None:
            known.update(int(r) for r in already_represented)
        probe = getattr(store, "represented_reference_ids", None)
        if callable(probe):
            known.update(int(r) for r in probe(self.model))
        return known

    def enroll(
        self,
        references: Iterable[ReferenceItem | tuple[ProfileId | str, int, Path | str]],
        store: EmbeddingStore,
        already_represented: Iterable[int] | None = None,
    ) -> EnrollmentReport:
        report = EnrollmentReport(model=self.model)
        known = self._known_represented(store, already_represented)
        seen: set[int] = set()
        for raw in references:
            item = _as_item(raw)
            report.processed += 1

            def skip(reason: SkipReason, detail: str = "", it: ReferenceItem = item) -> None:
                report.skipped.append(
                    SkippedReference(it.profile_id, it.reference_id, reason, detail)
                )

            if item.reference_id in seen:
                skip(SkipReason.DUPLICATE_IN_BATCH)
                continue
            seen.add(item.reference_id)
            if item.reference_id in known:
                skip(SkipReason.ALREADY_REPRESENTED)
                continue
            if not item.image_path.is_file():
                skip(SkipReason.FILE_MISSING, str(item.image_path))
                continue
            image = read_image(item.image_path)
            if image is None:
                skip(SkipReason.UNREADABLE, str(item.image_path))
                continue
            emb, flags, reason, detail = self.embed_image(image, item.reference_id)
            if emb is None:
                skip(reason or SkipReason.EMBEDDING_FAILED, detail)
                continue
            try:
                embedding_id = store.add_embedding(item.profile_id, item.reference_id, emb)
            except (KeyError, ValueError, RuntimeError) as exc:  # erreurs signalées par C
                skip(SkipReason.STORE_ERROR, f"{type(exc).__name__}: {exc}")
                continue
            self._represented.add(item.reference_id)
            report.added.append(
                AddedReference(item.profile_id, item.reference_id, embedding_id, flags)
            )
        return report


def build_reference_enroller(
    models_dir: Path | None = None, config: EnrollmentConfig | None = None
) -> ReferenceEnroller:
    """YuNet + SFace (poids vérifiés par sha256). Lève ``ModelFileMissingError`` /
    ``ModelIntegrityError`` si les poids sont absents ou altérés."""
    from maisonaccueil.vision.detection import YuNetDetector
    from maisonaccueil.vision.model_files import SFACE_2021DEC, YUNET_2023MAR, resolve_verified

    from .embedders import SFaceEmbedder

    detector = YuNetDetector(resolve_verified(YUNET_2023MAR, models_dir), verify=False)
    embedder = SFaceEmbedder(resolve_verified(SFACE_2021DEC, models_dir), verify=False)
    return ReferenceEnroller(detector, embedder, config)
