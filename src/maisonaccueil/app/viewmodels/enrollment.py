"""Collecte guidée (``docs/DATA_POLICY.md``, parcours par personne, étape 2).

Séance d'enrôlement ``split='enroll'`` : ~20 s face caméra à 1–3 m, légers mouvements de
tête. La sélection automatique d'images nettes et variées est faite CÔTÉ VISION et arrive
sous forme d'``EnrollmentSample`` ; ce modèle de vue affiche consignes et progression et
enregistre les images via ``ReferenceSink`` (dépôt de références du lot C). Il ne calcule
aucune représentation : elles seront produites par le moteur, une fois disponible.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from maisonaccueil.contracts import CameraId, ProfileId, SessionId

from ..messages import EnrollmentSample
from ..ports import ProfileStore, ReferenceSink
from .base import Observable, ValidationError
from .formatting import CAMERA_UNAVAILABLE, ENGINE_UNAVAILABLE


class EnrollmentPhase(enum.StrEnum):
    IDLE = "idle"  # aucun profil choisi
    READY = "ready"  # profil choisi, consignes affichées
    CAPTURING = "capturing"
    DONE = "done"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class GuidanceStep:
    title: str
    instruction: str
    duration_s: float


GUIDANCE_STEPS: tuple[GuidanceStep, ...] = (
    GuidanceStep("Face à la caméra", "Regardez la caméra, à 1 à 3 mètres, visage dégagé.", 5),
    GuidanceStep("Gauche / droite", "Tournez lentement la tête un peu à gauche, puis à droite.", 5),
    GuidanceStep("Haut / bas", "Levez légèrement le menton, puis baissez-le.", 5),
    GuidanceStep("Distance", "Avancez puis reculez d'un pas en restant face à la caméra.", 5),
)

PREPARATION = (
    "Avant de commencer : la personne a donné son consentement ; éclairage normal de la "
    "pièce ; lunettes habituelles portées. Si possible, refaire une séance sous un second "
    "éclairage (lampe / jour). Les séances d'évaluation se font d'autres jours."
)


class EnrollmentViewModel(Observable):
    def __init__(
        self,
        profiles: ProfileStore,
        sink: ReferenceSink,
        *,
        target_images: int = 20,
        max_duration_s: float = 60.0,
        camera_id: CameraId | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
        now_utc: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        super().__init__()
        if target_images < 1:
            raise ValueError("target_images doit être >= 1")
        self._profiles = profiles
        self._sink = sink
        self._target = target_images
        self._max_ns = int(max_duration_s * 1e9)
        self._camera_id = camera_id
        self._clock = clock
        self._now = now_utc
        self.phase = EnrollmentPhase.IDLE
        self._profile_id: ProfileId | None = None
        self._profile_name = ""
        self._session_id: SessionId | None = None
        self._started_ns: int | None = None
        self._elapsed_ns = 0
        self._reference_ids: list[int] = []
        self.camera_available = False
        self.engine_available = False
        self.message = "Choisissez un profil pour commencer."
        self.rejected_samples = 0

    # -- environnement --------------------------------------------------------------
    def set_availability(self, *, camera: bool, engine: bool) -> None:
        if (camera, engine) != (self.camera_available, self.engine_available):
            self.camera_available = camera
            self.engine_available = engine
            self._notify()

    # -- parcours -------------------------------------------------------------------
    def select_profile(self, profile_id: ProfileId) -> None:
        if self.phase == EnrollmentPhase.CAPTURING:
            raise ValidationError("Collecte en cours : terminez ou annulez d'abord.")
        p = self._profiles.find(profile_id)
        if p is None:
            raise ValidationError("Profil introuvable.")
        if getattr(p, "consent_at", None) is None:
            raise ValidationError("Profil sans consentement : collecte impossible.")
        if not getattr(p, "active", True):
            raise ValidationError("Profil désactivé : réactivez-le avant la collecte.")
        self._profile_id = profile_id
        self._profile_name = p.display_name
        self._session_id = None
        self._reference_ids = []
        self._started_ns = None
        self._elapsed_ns = 0
        self.phase = EnrollmentPhase.READY
        self.message = PREPARATION
        self._notify()

    def start(self, conditions: str = "") -> SessionId:
        if self.phase != EnrollmentPhase.READY or self._profile_id is None:
            raise ValidationError("Choisissez d'abord un profil.")
        if not self.camera_available:
            raise ValidationError(f"Collecte impossible : {CAMERA_UNAVAILABLE}.")
        session = self._sink.create_session(
            self._profile_id,
            started_at=self._now(),
            split="enroll",
            camera_id=self._camera_id,
            conditions=conditions.strip() or None,
        )
        self._session_id = SessionId(str(session.session_id))
        self._started_ns = self._clock()
        self._elapsed_ns = 0
        self.phase = EnrollmentPhase.CAPTURING
        self.message = self._capture_message()
        self._notify()
        return self._session_id

    def add_sample(self, sample: EnrollmentSample) -> bool:
        """Enregistre une image retenue par la vision. Ignorée hors collecte."""
        if self.phase != EnrollmentPhase.CAPTURING or self._profile_id is None:
            self.rejected_samples += 1
            return False
        ref = self._sink.add_image_bytes(
            self._profile_id,
            sample.data,
            extension=sample.extension,
            captured_at=self._now(),
            session_id=self._session_id,
            width=sample.width,
            height=sample.height,
            quality_note=sample.quality_note,
        )
        self._reference_ids.append(int(ref.reference_id))
        if len(self._reference_ids) >= self._target:
            self._finish("Collecte terminée : nombre d'images visé atteint.")
        else:
            self.message = self._capture_message()
            self._notify()
        return True

    def tick(self) -> None:
        if self.phase != EnrollmentPhase.CAPTURING or self._started_ns is None:
            return
        self._elapsed_ns = self._clock() - self._started_ns
        if self._elapsed_ns >= self._max_ns:
            self._finish(
                f"Durée maximale atteinte avec {len(self._reference_ids)} image(s) sur "
                f"{self._target}. Vous pouvez refaire une séance."
            )
            return
        self.message = self._capture_message()
        self._notify()

    def finish(self) -> None:
        if self.phase != EnrollmentPhase.CAPTURING:
            raise ValidationError("Aucune collecte en cours.")
        self._finish(f"Collecte arrêtée avec {len(self._reference_ids)} image(s).")

    def cancel(self) -> int:
        """Annule la séance et supprime les images déjà enregistrées pendant celle-ci."""
        removed = 0
        for rid in self._reference_ids:
            if self._sink.delete(rid):
                removed += 1
        self._reference_ids = []
        self.phase = EnrollmentPhase.CANCELLED
        self.message = f"Collecte annulée : {removed} image(s) supprimée(s)."
        self._notify()
        return removed

    # -- affichage ------------------------------------------------------------------
    @property
    def profile_name(self) -> str:
        return self._profile_name

    @property
    def session_id(self) -> SessionId | None:
        return self._session_id

    @property
    def collected(self) -> int:
        return len(self._reference_ids)

    @property
    def target(self) -> int:
        return self._target

    @property
    def progress(self) -> float:
        """Fraction 0..1 des images visées (progression, pas une mesure de qualité)."""
        return min(1.0, len(self._reference_ids) / self._target)

    @property
    def elapsed_s(self) -> float:
        return self._elapsed_ns / 1e9

    @property
    def current_step_index(self) -> int:
        t = self.elapsed_s
        acc = 0.0
        for i, step in enumerate(GUIDANCE_STEPS):
            acc += step.duration_s
            if t < acc:
                return i
        return len(GUIDANCE_STEPS) - 1

    @property
    def current_step(self) -> GuidanceStep:
        return GUIDANCE_STEPS[self.current_step_index]

    @property
    def availability_text(self) -> str:
        parts = []
        if not self.camera_available:
            parts.append(CAMERA_UNAVAILABLE)
        if not self.engine_available:
            parts.append(
                f"{ENGINE_UNAVAILABLE} : aucune sélection automatique d'images ; les "
                "représentations seront calculées quand un moteur sera installé"
            )
        return " ; ".join(parts)

    def _capture_message(self) -> str:
        step = self.current_step
        return (
            f"Étape {self.current_step_index + 1}/{len(GUIDANCE_STEPS)} — {step.title} : "
            f"{step.instruction} ({self.collected}/{self._target} images retenues)"
        )

    def _finish(self, message: str) -> None:
        self.phase = EnrollmentPhase.DONE
        self.message = message
        self._notify()
