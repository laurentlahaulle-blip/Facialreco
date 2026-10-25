"""Détecteurs. Les tests YuNet sont ignorés si les poids sont absents de models/d.

Images synthétiques uniquement : zéro visage détecté est un résultat acceptable ;
on vérifie le fonctionnement (types, formes), jamais une précision.
"""

from pathlib import Path

import numpy as np
import pytest

from maisonaccueil.contracts import BoundingBox, Detection
from maisonaccueil.contracts.interfaces import FaceDetector
from maisonaccueil.vision import (
    YUNET_2023MAR,
    FakeBlobDetector,
    FakeScriptedDetector,
    ModelFileMissingError,
    ModelIntegrityError,
    resolve_verified,
    synthetic_blob_image,
)
from tests.vision.helpers import make_frame, weights_path

YUNET = weights_path(YUNET_2023MAR)
needs_yunet = pytest.mark.skipif(YUNET is None, reason="poids YuNet absents (models/d)")


def test_fake_blob_detector_finds_synthetic_rectangles():
    blobs = [(20, 30, 60, 60, (0, 0, 200)), (200, 100, 50, 70, (0, 200, 0))]
    img = synthetic_blob_image(240, 320, blobs)
    d = FakeBlobDetector()
    assert isinstance(d, FaceDetector)
    out = d.detect(make_frame(3, img))
    assert len(out) == 2
    assert all(isinstance(x, Detection) and x.landmarks is not None for x in out)
    assert out[0].bbox == BoundingBox(20, 30, 60, 60)
    assert out[0].frame_id == 3


def test_fake_scripted_detector():
    d = FakeScriptedDetector(lambda f: [BoundingBox(f.frame_id, 0, 10, 10)])
    out = d.detect(make_frame(5))
    assert out[0].bbox.x == 5


def test_missing_and_corrupted_weights_are_rejected(tmp_path: Path):
    with pytest.raises(ModelFileMissingError):
        resolve_verified(YUNET_2023MAR, tmp_path)
    (tmp_path / YUNET_2023MAR.filename).write_bytes(b"x" * YUNET_2023MAR.size)
    with pytest.raises(ModelIntegrityError):
        resolve_verified(YUNET_2023MAR, tmp_path)


@needs_yunet
def test_yunet_smoke_on_synthetic_images():
    from maisonaccueil.vision import YuNetDetector

    det = YuNetDetector()
    assert isinstance(det, FaceDetector)
    rng = np.random.default_rng(0)
    for shape in [(240, 320), (480, 640), (240, 320)]:  # changement de taille d'entrée
        img = rng.integers(0, 255, (*shape, 3), dtype=np.uint8)
        out = det.detect(make_frame(1, img))
        assert isinstance(out, list)
        for d in out:  # zéro visage attendu, mais si détection : contrat respecté
            assert d.landmarks is not None and d.detector_name == det.name
            assert d.bbox.x >= 0 and d.bbox.x + d.bbox.width <= shape[1]
    blank = det.detect(make_frame(2, np.zeros((240, 320, 3), np.uint8)))
    assert blank == []
