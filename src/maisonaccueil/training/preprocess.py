"""Prétraitement des visages pour NOTRE modèle (lot E).

Ce module n'est PAS un détecteur : il suppose que la détection et les 5 points
(``Landmarks5``) sont fournis en amont (lot D, préentraîné, hors périmètre « de zéro »).
Il ne fait qu'une transformation géométrique déterministe (similitude 5 points) et une
normalisation des pixels, sans aucun paramètre appris.

Alignement (identifiant ``PREPROCESS_ID = "align112-arcface5-v1"``, PARTAGÉ avec le
prétraitement de référence du lot D, ``maisonaccueil.vision.alignment``) :

- ``Landmarks5`` en **coordonnées image** : œil à gauche dans l'image, œil à droite,
  nez, coin de bouche gauche-image, coin de bouche droite-image (ordre YuNet/ArcFace) ;
- similitude (rotation + échelle uniforme + translation) estimée par
  ``cv2.estimateAffinePartial2D(..., method=cv2.LMEDS)`` vers le gabarit ArcFace 5 points ;
- ``warpAffine`` 112 × 112, interpolation bilinéaire, bord constant noir → BGR uint8.

La règle « importer uniquement via les contrats » interdit d'importer ``vision`` : la
même géométrie est donc recopiée ici, et un test vérifie l'égalité avec D quand ce
module est disponible. Toute divergence impose un identifiant propre au lot E.

Normalisation des pixels (interne au modèle, fixée par sa configuration et ses poids,
donc couverte par ``ModelRef.version``) : BGR → RGB, ``float32``,
``(x - 127.5) / 127.5`` ∈ [-1, 1], disposition ``(3, 112, 112)``.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from maisonaccueil.contracts import Landmarks5

PREPROCESS_ID = "align112-arcface5-v1"  # identique à vision.alignment (lot D)
INPUT_SIZE = 112

# Gabarit 5 points « ArcFace » 112×112 (coordonnées géométriques uniquement, aucun poids),
# identique à vision.alignment.ARCFACE_TEMPLATE_112. Ordre en coordonnées image.
TEMPLATE_112: npt.NDArray[np.float32] = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

DEFAULT_MIN_EYE_DISTANCE_PX = 16.0


def eye_distance(landmarks: Landmarks5) -> float:
    (lx, ly), (rx, ry) = landmarks.points[0], landmarks.points[1]
    return float(np.hypot(rx - lx, ry - ly))


def align_face(
    image_bgr: npt.NDArray[np.uint8],
    landmarks: Landmarks5,
    min_eye_distance_px: float = DEFAULT_MIN_EYE_DISTANCE_PX,
) -> npt.NDArray[np.uint8] | None:
    """Renvoie le visage aligné ``(112, 112, 3)`` BGR uint8, ou None s'il est inutilisable
    (visage trop petit, points dégénérés, transformation non estimable)."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("align_face attend une image BGR uint8 (H, W, 3)")
    if eye_distance(landmarks) < min_eye_distance_px:
        return None
    src = np.asarray(landmarks.points, dtype=np.float32)
    if not np.all(np.isfinite(src)):
        return None
    matrix, _ = cv2.estimateAffinePartial2D(src, TEMPLATE_112, method=cv2.LMEDS)
    if matrix is None or not np.all(np.isfinite(matrix)):
        return None
    aligned = cv2.warpAffine(
        image_bgr,
        matrix,
        (INPUT_SIZE, INPUT_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return np.ascontiguousarray(aligned, dtype=np.uint8)


def to_input_array(aligned_bgr: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Visage aligné BGR uint8 ``(112, 112, 3)`` → entrée réseau float32 ``(3, 112, 112)``."""
    if aligned_bgr.shape != (INPUT_SIZE, INPUT_SIZE, 3) or aligned_bgr.dtype != np.uint8:
        raise ValueError(
            f"to_input_array attend un visage aligné uint8 {(INPUT_SIZE, INPUT_SIZE, 3)}, "
            f"reçu {aligned_bgr.shape} {aligned_bgr.dtype}"
        )
    rgb = aligned_bgr[:, :, ::-1].astype(np.float32)
    normalized = (rgb - 127.5) / 127.5
    return np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)
