"""Image capturée, horodatage et état de la source vidéo."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import numpy.typing as npt

from .ids import CameraId, FrameId


@dataclass(frozen=True, slots=True)
class Frame:
    """Une image capturée.

    ``monotonic_ns`` sert à TOUS les calculs de durée (time.monotonic_ns()).
    ``wall_time`` (UTC, timezone-aware) sert uniquement à l'affichage et aux journaux.
    ``image`` est BGR uint8, forme (hauteur, largeur, 3). Elle ne doit pas être modifiée
    en place par les consommateurs (copier si nécessaire).
    Aucune image n'est écrite sur disque par défaut.
    """

    frame_id: FrameId
    camera_id: CameraId
    monotonic_ns: int
    wall_time: datetime
    image: npt.NDArray[np.uint8] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.wall_time.tzinfo is None:
            raise ValueError("Frame.wall_time doit être timezone-aware (UTC)")
        if self.image.ndim != 3 or self.image.shape[2] != 3 or self.image.dtype != np.uint8:
            raise ValueError("Frame.image doit être un tableau BGR uint8 (H, W, 3)")

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


class SourceKind(enum.StrEnum):
    USB = "usb"  # index de webcam (ex. "0")
    NETWORK = "network"  # URL RTSP/HTTP, réseau local uniquement
    FILE = "file"  # fichier vidéo de test
    SYNTHETIC = "synthetic"  # générateur de test, jamais utilisé comme donnée biométrique


@dataclass(frozen=True, slots=True)
class SourceConfig:
    camera_id: CameraId
    kind: SourceKind
    location: str  # index USB, URL ou chemin selon ``kind``
    width: int | None = None
    height: int | None = None
    fps: float | None = None


class CaptureStatus(enum.StrEnum):
    STOPPED = "stopped"
    CONNECTING = "connecting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"  # coupure détectée, reprise automatique en cours
    FAILED = "failed"  # abandon après épuisement des tentatives


@dataclass(frozen=True, slots=True)
class CaptureState:
    camera_id: CameraId
    status: CaptureStatus
    monotonic_ns: int
    detail: str = ""
    frames_captured: int = 0
    frames_dropped: int = 0  # images abandonnées par la file bornée
