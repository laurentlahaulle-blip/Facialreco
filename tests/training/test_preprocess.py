from __future__ import annotations

import importlib.util

import cv2
import numpy as np
import pytest

from maisonaccueil.contracts import Landmarks5
from maisonaccueil.training.preprocess import (
    INPUT_SIZE,
    PREPROCESS_ID,
    TEMPLATE_112,
    align_face,
    to_input_array,
)

from .conftest import landmarks_at, make_frame


def test_preprocess_id_is_shared_with_reference_pipeline() -> None:
    assert PREPROCESS_ID == "align112-arcface5-v1"


def test_alignment_inverts_known_similarity() -> None:
    frame = make_frame()
    scale, tx, ty = 2.0, 150.0, 80.0
    aligned = align_face(frame.image, landmarks_at(scale, tx, ty))
    assert aligned is not None
    assert aligned.shape == (INPUT_SIZE, INPUT_SIZE, 3) and aligned.dtype == np.uint8
    # Référence : même similitude appliquée explicitement.
    matrix = np.array([[1 / scale, 0, -tx / scale], [0, 1 / scale, -ty / scale]])
    expected = cv2.warpAffine(frame.image, matrix, (INPUT_SIZE, INPUT_SIZE))
    diff = np.abs(aligned.astype(int) - expected.astype(int))
    assert float(diff.mean()) < 2.0


def test_alignment_does_not_modify_frame() -> None:
    frame = make_frame()
    before = frame.image.copy()
    align_face(frame.image, landmarks_at(2.0, 150, 80))
    assert np.array_equal(before, frame.image)


def test_too_small_face_is_rejected() -> None:
    frame = make_frame()
    assert align_face(frame.image, landmarks_at(0.2, 10, 10)) is None


def test_non_finite_landmarks_are_rejected() -> None:
    frame = make_frame()
    pts = ((0.0, 0.0), (100.0, 0.0), (float("nan"), 50.0), (10.0, 90.0), (90.0, 90.0))
    assert align_face(frame.image, Landmarks5(pts)) is None


def test_to_input_array_layout_and_range() -> None:
    aligned = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    aligned[..., 0] = 255  # canal bleu (BGR)
    x = to_input_array(aligned)
    assert x.shape == (3, INPUT_SIZE, INPUT_SIZE) and x.dtype == np.float32
    assert np.allclose(x[2], 1.0)  # bleu en dernier après BGR → RGB
    assert np.allclose(x[0], -1.0)
    with pytest.raises(ValueError):
        to_input_array(np.zeros((100, 100, 3), dtype=np.uint8))


@pytest.mark.skipif(
    importlib.util.find_spec("maisonaccueil.vision.alignment") is None,
    reason="vision.alignment (lot D) absent de cette branche",
)
def test_same_geometry_as_reference_alignment() -> None:
    """Garantit que l'identifiant partagé recouvre bien la même géométrie que D.
    (Import direct uniquement dans ce test de cohérence, pas dans le code du lot.)"""
    from maisonaccueil.vision import alignment  # type: ignore[import-not-found,unused-ignore]

    assert alignment.PREPROCESS_ID == PREPROCESS_ID
    assert np.array_equal(alignment.ARCFACE_TEMPLATE_112, TEMPLATE_112)
    frame = make_frame(3)
    lm = landmarks_at(1.7, 200, 120)
    ours = align_face(frame.image, lm)
    assert ours is not None
    assert np.array_equal(alignment.align_face(frame.image, lm), ours)
