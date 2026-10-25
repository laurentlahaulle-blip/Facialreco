from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from maisonaccueil.contracts import (
    BoundingBox,
    CameraId,
    Detection,
    Frame,
    FrameId,
    Landmarks5,
    TrackedFace,
    TrackId,
)
from maisonaccueil.training.preprocess import TEMPLATE_112


@pytest.fixture(autouse=True)
def isolated_lock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Les verrous de test ne touchent jamais le dossier commun de la machine."""
    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("MAISONACCUEIL_LOCK_DIR", str(lock_dir))
    return lock_dir


def make_frame(seed: int = 0, h: int = 480, w: int = 640) -> Frame:
    """Image de BRUIT (aucun visage) : sert uniquement à exercer le code."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return Frame(FrameId(1), CameraId("synthetic"), 0, datetime.now(UTC), image)


def landmarks_at(scale: float, tx: float, ty: float) -> Landmarks5:
    """Gabarit 112 agrandi d'un facteur ``scale`` puis translaté (similitude connue)."""
    pts = tuple((float(x * scale + tx), float(y * scale + ty)) for x, y in TEMPLATE_112)
    return Landmarks5(pts)


def make_face(landmarks: Landmarks5 | None) -> TrackedFace:
    det = Detection(FrameId(1), BoundingBox(100, 100, 224, 224), 0.9, "fake", landmarks)
    return TrackedFace(TrackId("trk-test"), det, 0, 1)
