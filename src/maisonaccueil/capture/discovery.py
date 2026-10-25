"""Énumération des webcams USB.

Chaque index est sondé SOUS réservation ``ResourceLock("camera-<index>")`` prise sans
attente : un index déjà réservé par une autre session (ou par cette application) est
signalé ``busy`` et n'est PAS ouvert. Aucune capture exclusive n'est donc ouverte en
parallèle d'une autre session. La capture d'essai est relâchée immédiatement ; aucune
image n'est conservée.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from pathlib import Path

from maisonaccueil.pipeline import ResourceBusyError, ResourceLock

from .opencv_source import (
    CaptureFactory,
    VideoCaptureLike,
    default_capture_factory,
    default_usb_backend,
    resolve_backend,
)

logger = logging.getLogger(__name__)


class UsbProbeStatus(enum.StrEnum):
    AVAILABLE = "available"  # ouverture réussie puis relâchée
    BUSY = "busy"  # réservée par une autre session : non sondée


@dataclass(frozen=True, slots=True)
class UsbCameraInfo:
    index: int
    status: UsbProbeStatus
    detail: str = ""

    @property
    def location(self) -> str:
        """Valeur à placer dans ``SourceConfig.location``."""
        return str(self.index)


def list_usb_cameras(
    max_index: int = 5,
    *,
    backend: str | int | None = None,
    capture_factory: CaptureFactory | None = None,
    lock_owner: str = "maisonaccueil-capture-probe",
    lock_dir: Path | None = None,
    include_busy: bool = True,
) -> list[UsbCameraInfo]:
    """Retourne les index USB ``0..max_index-1`` présents (disponibles ou réservés).

    Les index absents (ouverture impossible) ne figurent pas dans le résultat.
    """
    factory = capture_factory or default_capture_factory
    api = resolve_backend(backend if backend is not None else default_usb_backend())
    found: list[UsbCameraInfo] = []
    for index in range(max_index):
        lock = ResourceLock(f"camera-{index}", owner=lock_owner, lock_dir=lock_dir)
        try:
            lock.acquire()
        except ResourceBusyError:
            if include_busy:
                info = lock.holder() or {}
                owner = info.get("owner", "?")
                found.append(UsbCameraInfo(index, UsbProbeStatus.BUSY, f"réservée par {owner}"))
            continue
        try:
            cap: VideoCaptureLike | None = None
            try:
                cap = factory(index, api, [])
                if cap.isOpened():
                    found.append(UsbCameraInfo(index, UsbProbeStatus.AVAILABLE))
            except Exception:
                logger.debug("Sondage de l'index %d en erreur", index, exc_info=True)
            finally:
                if cap is not None:
                    cap.release()
        finally:
            lock.release()
    return found
