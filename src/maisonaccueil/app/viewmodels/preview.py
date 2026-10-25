"""Aperçu : dernière image réduite reçue du processus vision (aucun calcul ici)."""

from __future__ import annotations

import time
from collections.abc import Callable

from ..messages import PreviewImage
from .base import Observable


class PreviewViewModel(Observable):
    def __init__(
        self, *, stale_after_ms: int = 2_000, clock: Callable[[], int] = time.monotonic_ns
    ) -> None:
        super().__init__()
        self._stale_ns = stale_after_ms * 1_000_000
        self._clock = clock
        self._image: PreviewImage | None = None
        self._received_at: int | None = None
        self._stale = True
        self.received = 0

    def apply_preview(self, image: PreviewImage) -> None:
        self._image = image
        self._received_at = self._clock()
        self._stale = False
        self.received += 1
        self._notify()

    def tick(self) -> None:
        stale = self._received_at is None or self._clock() - self._received_at > self._stale_ns
        if stale != self._stale:
            self._stale = stale
            self._notify()

    @property
    def image(self) -> PreviewImage | None:
        """Dernière image, ou None si aucune ou trop ancienne (flux interrompu)."""
        return None if self._stale else self._image

    @property
    def caption(self) -> str:
        if self._image is None:
            return "aucune image reçue"
        if self._stale:
            return "flux interrompu (aucune image récente)"
        img = self._image
        return (
            f"{img.camera_id} · image {img.frame_id} · source {img.source_width}×"
            f"{img.source_height} (aperçu {img.width}×{img.height})"
        )
