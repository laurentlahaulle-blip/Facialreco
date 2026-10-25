"""Détecteur de référence : YuNet via ``cv2.FaceDetectorYN`` (poids MIT, OpenCV Zoo).

Exécution CPU (backend OpenCV DNN). Une instance n'est pas thread-safe : une par flux.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from maisonaccueil.contracts import BoundingBox, Detection, Frame, Landmarks5

from .model_files import YUNET_2023MAR, resolve_verified


class YuNetDetector:
    """Implémente ``FaceDetector``. Points renvoyés dans l'ordre image (voir alignment.py)."""

    def __init__(
        self,
        model_path: Path | None = None,
        score_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        top_k: int = 50,
        min_face_px: float = 0.0,
        verify: bool = True,
    ) -> None:
        if model_path is None:
            model_path = resolve_verified(YUNET_2023MAR)
        elif verify:
            resolve_verified(YUNET_2023MAR, model_path.parent)
        self.name = f"yunet-2023mar-{YUNET_2023MAR.short_hash}"
        self.score_threshold = score_threshold
        self.min_face_px = min_face_px
        self._input_size = (320, 320)
        self._impl = cv2.FaceDetectorYN.create(
            str(model_path), "", self._input_size, score_threshold, nms_threshold, top_k
        )

    def detect(self, frame: Frame) -> Sequence[Detection]:
        size = (frame.width, frame.height)
        if size != self._input_size:
            self._impl.setInputSize(size)
            self._input_size = size
        _ret, faces = self._impl.detect(frame.image)
        if faces is None:
            return []
        out: list[Detection] = []
        for row in np.asarray(faces, dtype=np.float64).reshape(-1, 15):
            x, y, w, h = (float(v) for v in row[:4])
            # Rognage à l'image : YuNet peut renvoyer des boîtes débordantes.
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(frame.width), x + w), min(float(frame.height), y + h)
            if x1 - x0 <= 1 or y1 - y0 <= 1:
                continue
            if min(x1 - x0, y1 - y0) < self.min_face_px:
                continue
            pts = tuple((float(row[4 + 2 * i]), float(row[5 + 2 * i])) for i in range(5))
            out.append(
                Detection(
                    frame_id=frame.frame_id,
                    bbox=BoundingBox(x0, y0, x1 - x0, y1 - y0),
                    detector_score=float(row[14]),
                    detector_name=self.name,
                    landmarks=Landmarks5(pts),
                )
            )
        return out
