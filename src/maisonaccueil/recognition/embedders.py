"""Représentation de référence : SFace via ``cv2.FaceRecognizerSF`` (poids Apache-2.0).

Le visage est aligné par ``maisonaccueil.vision.align_face`` (prétraitement versionné
``PREPROCESS_ID``), puis passé au réseau. Le vecteur est normalisé (norme L2 = 1) ; la
métrique est la similarité cosinus. Ce score n'est pas une probabilité.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from maisonaccueil.contracts import Embedding, Frame, Metric, ModelRef, TrackedFace
from maisonaccueil.vision.alignment import (
    PREPROCESS_ID,
    AlignmentError,
    align_face,
    landmarks_from_bbox,
)
from maisonaccueil.vision.model_files import SFACE_2021DEC, resolve_verified

SFACE_MODEL_REF = ModelRef(
    name="opencv-zoo-sface",
    version=f"2021dec-{SFACE_2021DEC.short_hash}",
    embedding_dim=128,
    metric=Metric.COSINE,
    preprocess=PREPROCESS_ID,
    origin="pretrained",
)


def l2_normalize(vec: npt.NDArray[np.floating]) -> npt.NDArray[np.float32] | None:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 1e-12:
        return None
    return (v / n).astype(np.float32)


class SFaceEmbedder:
    """Implémente ``FaceEmbedder``. Une instance par thread (OpenCV DNN non thread-safe).

    ``min_face_px`` : en dessous, ``embed`` renvoie None (visage inutilisable).
    """

    def __init__(
        self, model_path: Path | None = None, min_face_px: float = 32.0, verify: bool = True
    ) -> None:
        if model_path is None:
            model_path = resolve_verified(SFACE_2021DEC)
        elif verify:
            resolve_verified(SFACE_2021DEC, model_path.parent)
        self.min_face_px = min_face_px
        self._impl = cv2.FaceRecognizerSF.create(str(model_path), "")

    @property
    def model(self) -> ModelRef:
        return SFACE_MODEL_REF

    def embed_aligned(self, aligned: npt.NDArray[np.uint8]) -> Embedding | None:
        """Représentation d'un visage DÉJÀ aligné (112×112 BGR, ``PREPROCESS_ID``)."""
        if aligned.shape != (112, 112, 3) or aligned.dtype != np.uint8:
            raise ValueError("Visage aligné attendu : uint8 (112, 112, 3)")
        feat = self._impl.feature(aligned)
        vec = l2_normalize(feat)
        return None if vec is None else Embedding(SFACE_MODEL_REF, vec)

    def embed(self, frame: Frame, face: TrackedFace) -> Embedding | None:
        det = face.detection
        if min(det.bbox.width, det.bbox.height) < self.min_face_px:
            return None
        lm = det.landmarks or landmarks_from_bbox(
            det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height
        )
        try:
            aligned = align_face(frame.image, lm)
        except AlignmentError:
            return None
        return self.embed_aligned(aligned)
