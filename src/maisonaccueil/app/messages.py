"""Messages du processus vision vers l'interface (en plus de ``VisionOutput`` et
``CaptureState``). Tous sont des dataclasses figées, sérialisables par ``pickle``
(``multiprocessing.Queue``).

Ces types sont propres au lot B en attendant une décision du coordinateur
(voir ``docs/contract-changes/B-ui-messages.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from maisonaccueil.contracts import ModelRef


@dataclass(frozen=True, slots=True)
class PreviewImage:
    """Aperçu RÉDUIT d'une image, calculé côté vision. Pixels BGR888 contigus
    (``height`` lignes de ``width * 3`` octets). Jamais écrit sur disque."""

    frame_id: int
    camera_id: str
    monotonic_ns: int
    width: int
    height: int
    source_width: int
    source_height: int
    data: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("PreviewImage : dimensions invalides")
        if len(self.data) != self.width * self.height * 3:
            raise ValueError("PreviewImage : taille des données incohérente (BGR888 attendu)")


@dataclass(frozen=True, slots=True)
class EngineStatus:
    """État du moteur de reconnaissance tel que connu du processus vision.

    ``available=False`` : l'interface affiche « moteur non disponible » et ignore tout
    résultat de reconnaissance (aucun résultat inventé)."""

    available: bool
    model: ModelRef | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.available and self.model is None:
            raise ValueError("EngineStatus disponible sans ModelRef")
        if not self.available and self.model is not None:
            raise ValueError("EngineStatus indisponible avec un ModelRef")


ENGINE_STATUS_UNAVAILABLE = EngineStatus(available=False, detail="aucun moteur configuré")


@dataclass(frozen=True, slots=True)
class EnrollmentSample:
    """Image de visage retenue par la sélection automatique (côté vision) pendant une
    collecte guidée. ``data`` est une image encodée (PNG/JPEG). L'interface ne fait que
    l'enregistrer via le dépôt de références."""

    monotonic_ns: int
    data: bytes = field(repr=False)
    extension: str = ".png"
    width: int | None = None
    height: int | None = None
    quality_note: str | None = None

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("EnrollmentSample vide")
