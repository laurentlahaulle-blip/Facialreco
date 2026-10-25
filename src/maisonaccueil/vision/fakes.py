"""Doubles de test ``Fake*`` du lot D (vision). Jamais présentés comme résultats réels.

``FakeBlobDetector`` « détecte » des rectangles de couleur unie sur fond sombre dans des
images SYNTHÉTIQUES ; il sert à tester la chaîne sans poids ni photo de personne.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np
import numpy.typing as npt

from maisonaccueil.contracts import BoundingBox, Detection, Frame

from .alignment import landmarks_from_bbox


class FakeScriptedDetector:
    """Renvoie des boîtes fournies par une fonction ``frame -> boîtes`` (scénarios)."""

    name = "fake-scripted"

    def __init__(self, script: Callable[[Frame], Sequence[BoundingBox]]) -> None:
        self._script = script

    def detect(self, frame: Frame) -> Sequence[Detection]:
        return [
            Detection(
                frame_id=frame.frame_id,
                bbox=b,
                detector_score=1.0,
                detector_name=self.name,
                landmarks=landmarks_from_bbox(b.x, b.y, b.width, b.height),
            )
            for b in self._script(frame)
        ]


class FakeBlobDetector:
    """Détecte les zones claires (luminance > seuil) d'au moins ``min_area`` pixels."""

    name = "fake-blob"

    def __init__(self, threshold: int = 40, min_area: int = 64) -> None:
        self.threshold = threshold
        self.min_area = min_area

    def detect(self, frame: Frame) -> Sequence[Detection]:
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        _t, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        contours, _h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: list[Detection] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < self.min_area:
                continue
            out.append(
                Detection(
                    frame_id=frame.frame_id,
                    bbox=BoundingBox(float(x), float(y), float(w), float(h)),
                    detector_score=1.0,
                    detector_name=self.name,
                    landmarks=landmarks_from_bbox(x, y, w, h),
                )
            )
        out.sort(key=lambda d: (d.bbox.x, d.bbox.y))
        return out


def synthetic_blob_image(
    height: int,
    width: int,
    blobs: Sequence[tuple[int, int, int, int, tuple[int, int, int]]],
    texture_seed: int | None = 0,
) -> npt.NDArray[np.uint8]:
    """Image BGR de fond noir avec rectangles colorés (x, y, w, h, couleur BGR).

    Une texture déterministe (damier) est ajoutée dans chaque rectangle pour qu'il ne
    soit pas jugé flou ; ``texture_seed=None`` la désactive.
    """
    img = np.zeros((height, width, 3), np.uint8)
    for x, y, w, h, color in blobs:
        img[y : y + h, x : x + w] = color
        if texture_seed is not None:
            yy, xx = np.mgrid[0:h, 0:w]
            checker = (((yy // 4) + (xx // 4)) % 2).astype(np.int16) * 30 - 15
            patch = img[y : y + h, x : x + w].astype(np.int16) + checker[..., None]
            img[y : y + h, x : x + w] = np.clip(patch, 0, 255).astype(np.uint8)
    return img
