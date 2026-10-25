"""Détection de visage et identifiant de suivi d'un passage."""

from __future__ import annotations

from dataclasses import dataclass

from .ids import FrameId, TrackId


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Boîte en pixels dans le repère de l'image source (origine en haut à gauche)."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("BoundingBox: largeur et hauteur doivent être > 0")

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: BoundingBox) -> float:
        x1, y1 = max(self.x, other.x), max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Landmarks5:
    """Cinq points (œil gauche, œil droit, nez, coin gauche, coin droit de la bouche)."""

    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.points) != 5:
            raise ValueError("Landmarks5 attend exactement 5 points")


@dataclass(frozen=True, slots=True)
class Detection:
    """Visage détecté dans une image, sans identité.

    ``detector_score`` est le score technique du détecteur (confiance de "c'est un visage"),
    sans rapport avec l'identité de la personne.
    """

    frame_id: FrameId
    bbox: BoundingBox
    detector_score: float
    detector_name: str
    landmarks: Landmarks5 | None = None


@dataclass(frozen=True, slots=True)
class TrackedFace:
    """Détection associée à un passage. Le même ``track_id`` désigne la même silhouette
    suivie d'image en image ; il ne dit rien de l'identité."""

    track_id: TrackId
    detection: Detection
    monotonic_ns: int
    track_age_frames: int  # nombre d'images depuis la création du passage
