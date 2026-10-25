"""Outils de test du lot D : images SYNTHÉTIQUES uniquement (aucune photo de personne)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from maisonaccueil.contracts import CameraId, Frame, FrameId
from maisonaccueil.vision.model_files import ModelFile, locate

MS = 1_000_000


def make_frame(
    index: int,
    image: npt.NDArray[np.uint8] | None = None,
    t_ms: int | None = None,
    shape: tuple[int, int] = (240, 320),
) -> Frame:
    img = image if image is not None else np.zeros((*shape, 3), np.uint8)
    t = (index * 40 if t_ms is None else t_ms) * MS
    return Frame(FrameId(index), CameraId("synthetic"), t, datetime.now(UTC), img)


def weights_path(model: ModelFile) -> Path | None:
    return locate(model)
