"""Alignement 5 points vers un visage normalisé 112×112.

Convention d'ordre des points (``Landmarks5``) adoptée par le lot D : ordre en
**coordonnées image**, c'est-à-dire œil situé à gauche dans l'image, œil situé à droite,
nez, coin de bouche à gauche dans l'image, coin de bouche à droite. C'est l'ordre natif
de YuNet et du gabarit ArcFace ci-dessous (voir docs/contract-changes/D-landmarks-order.md).

Toute modification du gabarit, de la taille ou de la méthode d'estimation doit changer
``PREPROCESS_ID`` : des représentations calculées avec des prétraitements différents ne
sont pas comparables.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from maisonaccueil.contracts import Landmarks5

PREPROCESS_ID = "align112-arcface5-v1"
ALIGNED_SIZE = 112

# Gabarit 5 points « ArcFace » 112×112 (même gabarit que cv2.FaceRecognizerSF.alignCrop).
ARCFACE_TEMPLATE_112: npt.NDArray[np.float32] = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class AlignmentError(ValueError):
    """Points dégénérés : aucune similitude ne peut être estimée."""


def similarity_matrix(landmarks: Landmarks5) -> npt.NDArray[np.float64]:
    """Matrice 2×3 (rotation + échelle uniforme + translation) image → gabarit."""
    src = np.asarray(landmarks.points, dtype=np.float32).reshape(5, 2)
    if not np.all(np.isfinite(src)):
        raise AlignmentError("Points non finis")
    matrix, _inliers = cv2.estimateAffinePartial2D(src, ARCFACE_TEMPLATE_112, method=cv2.LMEDS)
    if matrix is None or not np.all(np.isfinite(matrix)):
        raise AlignmentError("Estimation de la similitude impossible")
    return np.asarray(matrix, dtype=np.float64)


def align_face(image: npt.NDArray[np.uint8], landmarks: Landmarks5) -> npt.NDArray[np.uint8]:
    """Retourne le visage aligné BGR uint8 (112, 112, 3). N'altère pas ``image``."""
    matrix = similarity_matrix(landmarks)
    aligned = cv2.warpAffine(
        image,
        matrix,
        (ALIGNED_SIZE, ALIGNED_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return np.ascontiguousarray(aligned, dtype=np.uint8)


def landmarks_from_bbox(x: float, y: float, w: float, h: float) -> Landmarks5:
    """Points approximatifs déduits d'une boîte (gabarit ramené à la boîte).

    Repli pour un détecteur sans points : l'alignement est alors grossier.
    """
    rel = ARCFACE_TEMPLATE_112 / ALIGNED_SIZE
    pts = tuple((float(x + px * w), float(y + py * h)) for px, py in rel)
    return Landmarks5(pts)
