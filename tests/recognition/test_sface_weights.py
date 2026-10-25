"""Tests de fumée avec les poids réels (ignorés si absents de models/d).

Entrées synthétiques uniquement : on vérifie formes, types, déterminisme et câblage,
jamais une capacité à reconnaître des personnes.
"""

import cv2
import numpy as np
import pytest

from maisonaccueil.contracts import Landmarks5
from maisonaccueil.contracts.interfaces import FaceEmbedder, Recognizer
from maisonaccueil.recognition import (
    SFACE_MODEL_REF,
    FakeEmbeddingStore,
    build_reference_recognizer,
)
from maisonaccueil.vision import ARCFACE_TEMPLATE_112, SFACE_2021DEC, YUNET_2023MAR, align_face
from tests.vision.helpers import make_frame, weights_path

SFACE = weights_path(SFACE_2021DEC)
needs_sface = pytest.mark.skipif(SFACE is None, reason="poids SFace absents (models/d)")
needs_both = pytest.mark.skipif(
    SFACE is None or weights_path(YUNET_2023MAR) is None, reason="poids absents (models/d)"
)


def _synthetic_aligned(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (112, 112, 3), dtype=np.uint8)


def test_model_ref_is_pretrained_and_versioned():
    assert SFACE_MODEL_REF.origin == "pretrained"
    assert SFACE_MODEL_REF.embedding_dim == 128
    assert SFACE_2021DEC.short_hash in SFACE_MODEL_REF.version


@needs_sface
def test_sface_embedding_shape_norm_determinism():
    from maisonaccueil.recognition import SFaceEmbedder

    emb = SFaceEmbedder()
    assert isinstance(emb, FaceEmbedder)
    a1 = emb.embed_aligned(_synthetic_aligned(0))
    a2 = emb.embed_aligned(_synthetic_aligned(0))
    assert a1 is not None and a2 is not None
    assert a1.vector.shape == (128,) and a1.vector.dtype == np.float32
    assert np.isclose(np.linalg.norm(a1.vector), 1.0, atol=1e-4)
    assert np.allclose(a1.vector, a2.vector)
    assert a1.model == SFACE_MODEL_REF
    with pytest.raises(ValueError):
        emb.embed_aligned(np.zeros((100, 100, 3), np.uint8))


@needs_sface
def test_our_alignment_matches_opencv_aligncrop():
    rec = cv2.FaceRecognizerSF.create(str(SFACE), "")
    rng = np.random.default_rng(3)
    img = cv2.GaussianBlur(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8), (7, 7), 0)
    pts = ARCFACE_TEMPLATE_112 * 1.8 + np.array([120.0, 60.0], np.float32)
    lm = Landmarks5(tuple((float(x), float(y)) for x, y in pts))
    ours = align_face(img, lm).astype(np.int16)
    # face_info au format YuNet : x, y, w, h, 5 points, score
    info = np.array([[100, 50, 220, 220, *pts.reshape(-1), 0.99]], np.float32)
    theirs = rec.alignCrop(img, info).astype(np.int16)
    assert theirs.shape == ours.shape
    assert float(np.abs(ours - theirs).mean()) < 3.0


@needs_both
def test_reference_recognizer_smoke_on_synthetic_frames():
    rec = build_reference_recognizer(store=FakeEmbeddingStore())
    assert isinstance(rec, Recognizer)
    rng = np.random.default_rng(0)
    for i in range(3):
        img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        results = rec.process(make_frame(i, img))
        assert isinstance(results, list)  # zéro visage attendu sur du bruit
        for r in results:
            assert r.model == SFACE_MODEL_REF and r.candidates == ()
    assert isinstance(rec.lost_tracks(), list)
