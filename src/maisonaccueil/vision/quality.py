"""Indicateurs de qualité d'un visage détecté (``QualityFlag``).

Heuristiques simples et rapides. Les seuils par défaut sont des valeurs de départ
NON MESURÉES : ils devront être réglés par le lot G sur l'ensemble ``tune``.
``OCCLUDED`` n'est pas évalué par cette implémentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from maisonaccueil.contracts import Detection, QualityFlag

QUALITY_POLICY_ID = "quality-heuristic-v1"


@dataclass(frozen=True, slots=True)
class QualityConfig:
    min_face_px: float = 48.0  # côté minimal de la boîte (pixels)
    blur_min_laplacian_var: float = 25.0  # variance du laplacien sur le visage 112×112
    low_light_max_mean: float = 45.0  # luminance moyenne (0-255) sous laquelle = sombre
    max_yaw_ratio: float = 0.35  # |décalage nez / milieu des yeux| / distance inter-yeux
    min_detector_score: float = 0.0  # 0 = pas de filtre supplémentaire


def _face_crop(image: npt.NDArray[np.uint8], det: Detection) -> npt.NDArray[np.uint8] | None:
    h, w = image.shape[:2]
    x0 = max(0, int(np.floor(det.bbox.x)))
    y0 = max(0, int(np.floor(det.bbox.y)))
    x1 = min(w, int(np.ceil(det.bbox.x + det.bbox.width)))
    y1 = min(h, int(np.ceil(det.bbox.y + det.bbox.height)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return image[y0:y1, x0:x1]


def yaw_ratio(det: Detection) -> float | None:
    """Décalage horizontal du nez par rapport au milieu des yeux, normalisé."""
    if det.landmarks is None:
        return None
    (lx, ly), (rx, ry), (nx, _ny) = det.landmarks.points[:3]
    eye_dist = float(np.hypot(rx - lx, ry - ly))
    if eye_dist <= 1e-6:
        return None
    return abs(nx - (lx + rx) / 2.0) / eye_dist


def assess_quality(
    image: npt.NDArray[np.uint8],
    det: Detection,
    config: QualityConfig | None = None,
    aligned: npt.NDArray[np.uint8] | None = None,
) -> frozenset[QualityFlag]:
    """Évalue un visage. ``aligned`` (112×112) est préféré pour le flou s'il est fourni."""
    cfg = config or QualityConfig()
    flags: set[QualityFlag] = set()

    if min(det.bbox.width, det.bbox.height) < cfg.min_face_px:
        flags.add(QualityFlag.FACE_TOO_SMALL)

    patch = aligned if aligned is not None else _face_crop(image, det)
    if patch is None:
        flags.add(QualityFlag.FACE_TOO_SMALL)
    else:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if aligned is None:
            gray = cv2.resize(gray, (112, 112), interpolation=cv2.INTER_AREA)
        if float(gray.mean()) < cfg.low_light_max_mean:
            flags.add(QualityFlag.LOW_LIGHT)
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < cfg.blur_min_laplacian_var:
            flags.add(QualityFlag.BLURRY)

    ratio = yaw_ratio(det)
    if ratio is not None and ratio > cfg.max_yaw_ratio:
        flags.add(QualityFlag.NOT_FRONTAL)

    return frozenset(flags)
