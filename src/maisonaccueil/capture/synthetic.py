"""Source synthétique : motifs générés et compteur, pour les tests logiciels.

Aucune image de visage, aucune donnée biométrique : ces images ne servent jamais à juger
la qualité de la reconnaissance.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import numpy.typing as npt

from maisonaccueil.contracts import CameraId

from .base import DropCounter, EndOfStream, ReconnectPolicy, ThreadedFrameSource

PATTERNS = ("bars", "checker", "gradient")


class SyntheticSource(ThreadedFrameSource):
    """Générateur d'images de test.

    ``fps`` : cadence cible (``None`` = aussi vite que possible, pour les tests de charge).
    ``max_frames`` : nombre d'images avant fin de flux (``None`` = infini).
    Le numéro de l'image est incrusté en texte et codé dans le pixel (0, 0) :
    canal B = n % 256, canal G = (n // 256) % 256.
    """

    def __init__(
        self,
        camera_id: CameraId,
        *,
        width: int = 320,
        height: int = 240,
        fps: float | None = 30.0,
        pattern: str = "bars",
        max_frames: int | None = None,
        policy: ReconnectPolicy | None = None,
        drop_counter: DropCounter | None = None,
        stats_interval_s: float = 1.0,
    ) -> None:
        if width < 16 or height < 16:
            raise ValueError("Dimensions synthétiques trop petites (minimum 16x16)")
        if fps is not None and fps <= 0:
            raise ValueError("fps doit être > 0 ou None")
        if pattern not in PATTERNS:
            raise ValueError(f"Motif inconnu {pattern!r} ; choix : {', '.join(PATTERNS)}")
        super().__init__(
            camera_id, policy=policy, drop_counter=drop_counter, stats_interval_s=stats_interval_s
        )
        self.width = width
        self.height = height
        self.fps = fps
        self.pattern = pattern
        self.max_frames = max_frames
        self._generated = 0
        self._next_due_ns = 0
        self._base = self._make_base()

    def _describe(self) -> str:
        return f"synthétique:{self.pattern} {self.width}x{self.height} ({self.camera_id})"

    def _make_base(self) -> npt.NDArray[np.uint8]:
        h, w = self.height, self.width
        img = np.zeros((h, w, 3), np.uint8)
        if self.pattern == "bars":
            colors = [
                (255, 255, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0),
                (255, 0, 255), (0, 0, 255), (255, 0, 0), (0, 0, 0),
            ]
            step = max(1, w // len(colors))
            for i, c in enumerate(colors):
                img[:, i * step : (i + 1) * step] = c
        elif self.pattern == "checker":
            cell = max(4, min(h, w) // 8)
            yy, xx = np.indices((h, w))
            mask = ((yy // cell) + (xx // cell)) % 2 == 0
            img[mask] = (200, 200, 200)
            img[~mask] = (40, 40, 40)
        else:  # gradient
            img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
            img[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
            img[:, :, 2] = 128
        return img

    def render(self, n: int) -> npt.NDArray[np.uint8]:
        """Image numéro ``n`` (déterministe)."""
        img = np.roll(self._base, shift=(n * 2) % self.width, axis=1)
        side = max(4, min(self.height, self.width) // 6)
        x = (n * 3) % max(1, self.width - side)
        y = self.height // 2 - side // 2
        cv2.rectangle(img, (x, y), (x + side, y + side), (0, 128, 255), thickness=-1)
        cv2.putText(
            img, f"#{n}", (4, self.height - 6), cv2.FONT_HERSHEY_SIMPLEX,
            max(0.3, self.height / 480), (0, 0, 0), 2, cv2.LINE_AA,
        )
        img[0, 0] = (n % 256, (n // 256) % 256, 0)
        return img

    @staticmethod
    def decode_counter(image: npt.NDArray[np.uint8]) -> int:
        """Relit le compteur codé dans le pixel (0, 0) (modulo 65536)."""
        return int(image[0, 0, 0]) + 256 * int(image[0, 0, 1])

    def _open_device(self) -> None:
        self._next_due_ns = time.monotonic_ns()

    def _close_device(self) -> None:
        pass

    def _read_image(self) -> npt.NDArray[np.uint8] | None:
        if self.max_frames is not None and self._generated >= self.max_frames:
            raise EndOfStream
        if self.fps is not None:
            period_ns = int(1e9 / self.fps)
            wait_ns = self._next_due_ns - time.monotonic_ns()
            if wait_ns > 0 and self._wait(wait_ns / 1e9):
                return None
            self._next_due_ns = max(self._next_due_ns + period_ns, time.monotonic_ns())
        self._generated += 1
        return self.render(self._generated)
